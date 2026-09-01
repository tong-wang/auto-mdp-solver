"""PPO / MaskablePPO training for the game2048 domain (spec §8).

Solve levels (spec §8.6), selected with ``--level``:

* **L0** — faithful defaults: the library's own settings plus only what the
  problem forces (gamma = beta, the algorithm class the IR's action mode
  forces, the env as built). No VecNormalize, one env, constant LR, MLP.
  The control run and the ruler for what the configuration layer is worth.
  Reporting only — never a gate.
* **L1** — the derived config (default). Every knob below is a forced move
  read off the IR, printed with its rationale at startup:

  | knob | value | why |
  |---|---|---|
  | gamma | 1.0 | = objective.discount_factor (beta); a game, no time value |
  | gae_lambda | 0.95 | merges pay immediately; no long-delayed credit |
  | n_envs x n_steps | 4 x 512 = 2048 | >= 2048 transitions and >= 10 episodes (T~48) |
  | batch_size | 128 | rollout/16, inside the rollout/32..rollout/8 band |
  | lr | 3e-4 -> 3e-5 | per-step reward is deterministic given (board, move) — the spawn perturbs the STATE, not the reward — so this is the dense near-deterministic-reward case, not the exogenous-noise one |
  | clip | 0.2 -> 0.05 | clip_final = clip_init/4 |
  | ent_coef | 0.005 | default; not a bandit-like exploration problem |
  | n_epochs / target_kl | 10 / 0.02 | fixed; target_kl is a safety valve |
  | norm_obs | off | rl.obs_normalization=false: tile magnitudes drift upward all episode, so running stats chase a moving target |
  | norm_reward | on (gamma passed) | spec §8.3 |
  | net_arch | (64,64) | obs dim 9 (vec). Structured obs (grid/onehot) is an ARCH question, not a width one — predicted escalation: arch |
  | budget | 2M steps | ~40k episodes x T~48, and ~976 updates (>= 300) |

Policy family follows the observation shape (spec §8.5): ndim >= 3 (the
``grid`` and ``onehot`` modes) -> CnnPolicy with ``SmallBoardCnn``; flat
``vec`` -> MlpPolicy. SB3's NatureCNN is never used — its 8x8 stride-4 kernels
cannot run on a 3x3 board. ``--extractor`` swaps the CNN extractor for one of
the lever-matrix arms (``embed``/``conv3d``/``rowcol``/``axial``/``attn-*``);
``small`` is
the incumbent. Each arm's trigger, prediction, and licence live in
ESCALATION.md #E14; the executable gate is ``game2048_extractor_gate.py``.

Usage:
    python game2048_ppo_train.py -s 3x3_20 -o onehot
    python game2048_ppo_train.py -s 3x3_20 -o vec --level L0
    python game2048_ppo_train.py -s 3x3_20 -o grid -a free -r score_penalty
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from game2048_board import MOVES, apply_move
from game2048_gym import ACTION_MODES, OBSERVATION_MODES, REWARD_MODES, Game2048Env
from game2048_scenarios import SCENARIOS

# selection seeds for model selection, disjoint from the reporting protocol's
# 0..n_seeds-1 (spec §8.6: the terminal checkpoint is never the deliverable)
SELECTION_SEED_BASE = 1_000_000


# ---------------------------------------------------------------------------
# Feature extractor (spec §8.5)
# ---------------------------------------------------------------------------


def _same_conv2d(prev: int, c_out: int, k: int) -> list[nn.Module]:
    """Stride-1 area-preserving conv for odd AND even kernels.

    Conv2d's own ``padding`` is symmetric, which is 'same' only for odd k; an
    even kernel with padding k//2 GROWS the map each layer (3x3 -> 4x4 -> 5x5
    at k=2), silently turning a kernel-geometry lever into a capacity lever
    (caught by the extractor gate's trainable counts). Even k gets
    asymmetric zero-padding instead, so every arm convolves the same board.
    """
    if k % 2:
        return [nn.Conv2d(prev, c_out, kernel_size=k, stride=1, padding=k // 2)]
    return [nn.ZeroPad2d((k // 2, k // 2 - 1, k // 2, k // 2 - 1)),
            nn.Conv2d(prev, c_out, kernel_size=k, stride=1, padding=0)]


class SmallBoardCnn(BaseFeaturesExtractor):
    """Stride-1, same-padding conv stack for small boards.

    No pooling: absolute position matters on a 2048 board (a corner is not a
    centre), so the usual translation-invariance machinery is exactly wrong
    here. Kernel is clamped to the board so a 2x2 grid still works.
    """

    def __init__(self, observation_space, features_dim: int = 128,
                 channels: tuple[int, ...] = (32, 64), kernel_size: int = 3):
        super().__init__(observation_space, features_dim)
        c_in, h, w = observation_space.shape
        k = max(1, min(kernel_size, h, w))
        layers: list[nn.Module] = []
        prev = c_in
        for c_out in channels:
            layers += _same_conv2d(prev, c_out, k) + [nn.ReLU()]
            prev = c_out
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class EmbedBoardCnn(BaseFeaturesExtractor):
    """SmallBoardCnn with the first layer factorized through a tile embedding.

    A bias-free 1x1 conv on one-hot planes IS an embedding table: each cell's
    one-hot picks one column of the (n_planes x embed_dim) weight matrix, with
    no activation in between — so this is exactly a rank-``embed_dim``
    factorization of the incumbent's first kernel, and the vocabulary size K
    stops setting the first conv's width (the vocab trigger, K~17 at 4x4).
    Everything after the embedding is SmallBoardCnn unchanged.
    """

    def __init__(self, observation_space, features_dim: int = 128,
                 channels: tuple[int, ...] = (32, 64), kernel_size: int = 3,
                 embed_dim: int = 8):
        super().__init__(observation_space, features_dim)
        c_in, h, w = observation_space.shape
        k = max(1, min(kernel_size, h, w))
        layers: list[nn.Module] = [
            nn.Conv2d(c_in, embed_dim, kernel_size=1, bias=False)]
        prev = embed_dim
        for c_out in channels:
            layers += _same_conv2d(prev, c_out, k) + [nn.ReLU()]
            prev = c_out
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class Conv3dExponentCnn(BaseFeaturesExtractor):
    """Conv3d stack sharing weights along the tile-exponent axis.

    The doubling law (domain test 38: the dynamics are equivariant to an
    exponent shift) licenses weight sharing along the exponent axis the same
    way board translation licenses Conv2d's spatial sharing: the one-hot tile
    planes become an axis of a 3D volume and every kernel spans (3, k, k) —
    the merge rule couples plane j to j+1, so 3 is the coupling's support.
    Plane 0 (empty cells) is NOT on the exponent ladder: it rides as a second
    input channel broadcast constant along the axis, so the shared kernels see
    the empty mask at every depth and the shift licence stays exact (a
    constant-along-axis channel commutes with the shift by construction). No
    pooling anywhere: absolute position and absolute exponent both matter to
    the head, so the trunk is equivariant and the flatten+linear head is what
    breaks the symmetry. (8, 8) channels are roughly SmallBoardCnn-matched at
    3x3; the head grows with K at 4x4 because the exponent axis is kept, not
    collapsed — record the trainable count, don't hide it.
    """

    def __init__(self, observation_space, features_dim: int = 128,
                 channels: tuple[int, ...] = (8, 8), kernel_size: int = 3):
        super().__init__(observation_space, features_dim)
        n_planes, h, w = observation_space.shape
        k = max(1, min(kernel_size, h, w))
        layers: list[nn.Module] = []
        prev = 2                                    # (ladder, empties) channels
        for c_out in channels:
            layers += [nn.Conv3d(prev, c_out, kernel_size=(3, k, k),
                                 stride=1, padding=(1, k // 2, k // 2)),
                       nn.ReLU()]
            prev = c_out
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.cnn(self._volume(
                torch.as_tensor(observation_space.sample()[None]).float()
            )).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def _volume(self, observations: torch.Tensor) -> torch.Tensor:
        """(B, K, n, n) planes -> (B, 2, K-1, n, n): channel 0 the exponent
        ladder (planes 1..K-1), channel 1 the empty mask at every depth."""
        tiles = observations[:, 1:].unsqueeze(1)
        empty = observations[:, :1].unsqueeze(1).expand(
            -1, -1, tiles.shape[2], -1, -1)
        return torch.cat([tiles, empty], dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(self._volume(observations)))


class RowColCnn(BaseFeaturesExtractor):
    """Two axis-aligned branches: full-row 1xn kernels and full-column nx1.

    A slide moves whole rows or whole columns at once, so the geometry the
    action reads is axis-aligned, not square-local (pre-registered #E14:
    predicted moot at 3x3, where the incumbent's same-padded 3x3 kernel
    already spans the board). Each branch's kernel covers its entire axis in
    one application — depth adds no receptive field — so the trunk is one
    conv per branch and the linear head is what mixes the two. Channels per
    branch = ``channels[0]`` (128 is roughly SmallBoardCnn-matched).
    """

    def __init__(self, observation_space, features_dim: int = 128,
                 channels: tuple[int, ...] = (32, 64)):
        super().__init__(observation_space, features_dim)
        c_in, h, w = observation_space.shape
        c = channels[0]
        self.rows = nn.Sequential(
            nn.Conv2d(c_in, c, kernel_size=(1, w)), nn.ReLU(), nn.Flatten())
        self.cols = nn.Sequential(
            nn.Conv2d(c_in, c, kernel_size=(h, 1)), nn.ReLU(), nn.Flatten())
        self.linear = nn.Sequential(
            nn.Linear(c * h + c * w, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat(
            [self.rows(observations), self.cols(observations)], dim=1))


class AxialCnn(BaseFeaturesExtractor):
    """Full-axis comparators composed across axes by broadcast-mix rounds.

    the AX arm (#E24). RowColCnn's row and column features meet
    exactly once, in the flatten head; here each round broadcasts the
    per-row and per-column summaries back onto the grid and re-mixes them
    per cell, so after round 1 a cell has seen its whole row AND column,
    and after round 2 the whole board — composition through repetition,
    one axis-global step per round (the same shape as what a slide does).
    ``axial - rowcol`` therefore isolates the composition rounds;
    ``rowcol - control`` isolates kernel geometry. Same axis-aligned
    licence as RowColCnn (#E14). Head stays position-specific flatten.

    channels = (c, ca): per-cell state width and per-axis summary width.
    Defaults (72, 71) land within 0.03% of the 4x4 control's 261,984
    trainables (the gate prints the count).
    """

    def __init__(self, observation_space, features_dim: int = 128,
                 channels: tuple[int, ...] = (72, 71), n_rounds: int = 2):
        super().__init__(observation_space, features_dim)
        c_in, h, w = observation_space.shape
        c, ca = channels[0], channels[1]
        self.embed = nn.Conv2d(c_in, c, kernel_size=1)
        self.row_convs = nn.ModuleList(
            [nn.Conv2d(c, ca, kernel_size=(1, w)) for _ in range(n_rounds)])
        self.col_convs = nn.ModuleList(
            [nn.Conv2d(c, ca, kernel_size=(h, 1)) for _ in range(n_rounds)])
        self.mixes = nn.ModuleList(
            [nn.Conv2d(c + 2 * ca, c, kernel_size=1) for _ in range(n_rounds)])
        self.linear = nn.Sequential(
            nn.Flatten(), nn.Linear(c * h * w, features_dim), nn.ReLU())

    def trunk(self, observations: torch.Tensor) -> torch.Tensor:
        """The (B, c, h, w) grid after the mixing rounds — split out so the
        gate can assert the information-reach pattern (cross after round 1,
        whole board after round 2) on the grid itself."""
        x = torch.relu(self.embed(observations))
        h, w = x.shape[-2], x.shape[-1]
        for row_conv, col_conv, mix in zip(self.row_convs, self.col_convs,
                                           self.mixes):
            rows = torch.relu(row_conv(x)).expand(-1, -1, -1, w)  # (B,ca,h,1)->
            cols = torch.relu(col_conv(x)).expand(-1, -1, h, -1)  # (B,ca,1,w)->
            x = x + torch.relu(mix(torch.cat([x, rows, cols], dim=1)))
        return x

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.trunk(observations))


class _AttnBlock(nn.Module):
    """Pre-LN transformer encoder block taking an additive attention bias.

    Written out rather than using nn.TransformerEncoderLayer so the positional
    bias tensor is reachable by the gate: the ``rel``/``axis`` arms ARE that
    tensor's sharing structure, and a lever whose structure cannot be asserted
    is not a lever.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.ReLU(),
            nn.Linear(ffn_mult * d_model, d_model))

    def forward(self, x: torch.Tensor,
                bias: torch.Tensor | None) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=bias, need_weights=False)
        x = x + a
        return x + self.ffn(self.ln2(x))


class BoardAttnExtractor(BaseFeaturesExtractor):
    """Cells as tokens: one-hot -> bias-free embed -> pre-LN transformer.

    The registered ATT node (#E7) is the null-structure ablation of the CNN:
    no locality prior, no weight sharing over space except what the positional
    scheme puts back. Tokens are the n_cells cells, so the trunk is
    permutation-EQUIVARIANT and the board is a multiset until the positional
    term breaks that degeneracy — which is why the positional scheme, not the
    attention, is this pool's axis (subclasses fix ``POS``).

    No pooling, and the head flattens tokens in a fixed row-major order: as in
    SmallBoardCnn, absolute position matters on a 2048 board and the head is
    where symmetry is meant to break. D4-orbit-shared positions are excluded
    by #E13 (policy equivariance refuted), not overlooked.
    """

    POS = "abs"

    def __init__(self, observation_space, features_dim: int = 128,
                 d_model: int = 96, n_layers: int = 2, n_heads: int = 4,
                 ffn_mult: int = 2):
        super().__init__(observation_space, features_dim)
        n_planes, h, w = observation_space.shape
        assert d_model % n_heads == 0, (d_model, n_heads)
        self.n_cells, self.n_heads, self.d_model = h * w, n_heads, d_model
        self.embed = nn.Linear(n_planes, d_model, bias=False)
        self.blocks = nn.ModuleList(
            [_AttnBlock(d_model, n_heads, ffn_mult) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.linear = nn.Sequential(
            nn.Linear(self.n_cells * d_model, features_dim), nn.ReLU())

        rows = torch.arange(self.n_cells) // w
        cols = torch.arange(self.n_cells) % w
        if self.POS == "abs":
            self.pos_embed = nn.Parameter(
                torch.empty(self.n_cells, d_model).normal_(std=0.02))
        elif self.POS == "rel":
            # one weight per (drow, dcol) offset, shared across all position
            # pairs — attention's form of the translation licence conv uses
            span = (2 * h - 1) * (2 * w - 1)
            dr, dc = rows[:, None] - rows[None, :], cols[:, None] - cols[None, :]
            self.register_buffer(
                "rel_index", ((dr + h - 1) * (2 * w - 1) + (dc + w - 1)).long())
            self.rel_bias = nn.Parameter(
                torch.empty(n_heads, span).normal_(std=0.02))
        elif self.POS == "axis":
            # the slide geometry: a move acts along whole rows / whole columns,
            # so the only structure injected is "shares my row" / "shares my
            # column" — two learned scalars per head, everything else pinned 0
            self.register_buffer("row_mask",
                                 (rows[:, None] == rows[None, :]).float())
            self.register_buffer("col_mask",
                                 (cols[:, None] == cols[None, :]).float())
            self.axis_bias = nn.Parameter(
                torch.empty(n_heads, 2).normal_(std=0.02))
        else:
            raise ValueError(f"unknown positional scheme {self.POS!r}")

    def attn_bias(self) -> torch.Tensor | None:
        """(n_heads, n_cells, n_cells) additive score bias, or None for abs."""
        if self.POS == "abs":
            return None
        if self.POS == "rel":
            return self.rel_bias[:, self.rel_index]
        return (self.axis_bias[:, 0, None, None] * self.row_mask
                + self.axis_bias[:, 1, None, None] * self.col_mask)

    def tokens(self, observations: torch.Tensor) -> torch.Tensor:
        """(B, planes, h, w) -> (B, n_cells, d_model), positional term applied."""
        x = self.embed(observations.flatten(2).transpose(1, 2))
        return x + self.pos_embed if self.POS == "abs" else x

    def trunk(self, observations: torch.Tensor,
              positional: bool = True) -> torch.Tensor:
        """Token states after the blocks. ``positional=False`` zeroes the
        positional path, leaving a permutation-equivariant trunk (gate 3)."""
        x = self.embed(observations.flatten(2).transpose(1, 2))
        if positional and self.POS == "abs":
            x = x + self.pos_embed
        bias = self.attn_bias() if positional else None
        if bias is not None:
            b = observations.shape[0]
            bias = bias.unsqueeze(0).expand(b, -1, -1, -1).reshape(
                b * self.n_heads, self.n_cells, self.n_cells)
        for blk in self.blocks:
            x = blk(x, bias)
        return self.ln_f(x)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.trunk(observations).flatten(1))


class AttnAbsExtractor(BoardAttnExtractor):
    """Learned absolute position per cell — the null-structure arm."""
    POS = "abs"


class AttnRelExtractor(BoardAttnExtractor):
    """Relative (drow, dcol) attention bias — translation sharing."""
    POS = "rel"


class AttnAxisExtractor(BoardAttnExtractor):
    """Same-row / same-column bias — the slide geometry (KG trigger)."""
    POS = "axis"


EXTRACTORS = {"small": SmallBoardCnn, "embed": EmbedBoardCnn,
              "conv3d": Conv3dExponentCnn, "rowcol": RowColCnn,
              "axial": AxialCnn,
              "attn-abs": AttnAbsExtractor, "attn-rel": AttnRelExtractor,
              "attn-axis": AttnAxisExtractor}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--scenario_name", default="3x3_20",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", default="vec",
                   choices=list(OBSERVATION_MODES))
    p.add_argument("-a", "--action_mode", default="masked",
                   choices=list(ACTION_MODES))
    p.add_argument("-r", "--reward_mode", default="score",
                   choices=list(REWARD_MODES))
    p.add_argument("--level", default="L1", choices=["L0", "L1"],
                   help="solve level (spec §8.6); L0 is the faithful-defaults control")
    p.add_argument("--policy", default="auto",
                   choices=["auto", "mlp", "cnn"],
                   help="auto picks CnnPolicy for a 3-D observation and "
                        "MlpPolicy otherwise; mlp/cnn force one")

    # L1-derived defaults — the tuner warm-starts trial 0 from exactly these
    p.add_argument("--total_timesteps", type=int, default=2_000_000)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    # §8.2 schedule pair. This campaign expresses the floor as a FRACTION
    # (--lr_floor_frac, so the floor follows a retuned lr); --lr_final is
    # the spec's absolute spelling, and wins when both are given.
    p.add_argument("--lr_final", type=float, default=None,
                   help="absolute LR floor (spec §8.2 schedule pair). "
                        "Overrides --lr_floor_frac when set")
    # --- tier-1 CLI surface (spec §8.2): names other domains join on ---
    p.add_argument("--tag", type=str, default=None,
                   help="free-text suffix appended to the run dir name")
    p.add_argument("--vf_coef", type=float, default=0.5,
                   help="value-loss coefficient (SB3 default 0.5)")
    p.add_argument("--gym_log", type=str, default=None,
                   help="per-episode/step gym log filename (spec §11); "
                        "None disables gym-side logging")
    p.add_argument("--norm_obs", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="VecNormalize obs normalization (spec §8.3). OFF by "
                        "default here: tile magnitudes drift upward all "
                        "episode, so running stats chase a moving target")
    p.add_argument("--norm_reward", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="VecNormalize reward normalization (spec §8.3)")
    p.add_argument("--vecnorm_clip_obs", type=float, default=10.0,
                   help="VecNormalize clip_obs (spec §8.3)")
    p.add_argument("--checkpoint_every_frac", type=float, default=None,
                   help="checkpoint every this FRACTION of the budget — "
                        "keeps a checkpoint count true when the budget "
                        "moves. Overrides --checkpoint-freq when set")
    p.add_argument("--n-envs", dest="n_envs", type=int, default=4)
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.005)
    p.add_argument("--target_kl", type=float, default=0.02)
    p.add_argument("--net_arch", type=int, nargs="+", default=[64, 64])
    p.add_argument("--features_dim", type=int, default=128)
    p.add_argument("--channels", type=int, nargs="+", default=[32, 64])
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--extractor", default="small",
                   choices=list(EXTRACTORS),
                   help="CnnPolicy feature extractor: "
                        "small = the SmallBoardCnn incumbent; embed = first "
                        "layer factorized through a tile embedding; conv3d = "
                        "exponent-axis weight sharing (doubling-law licence); "
                        "rowcol = axis-aligned full-length kernels; "
                        "axial = per-axis row+column streams; "
                        "attn-{abs,rel,axis} = cells-as-tokens transformer "
                        "(the ATT pool, #E21/#E22), the positional scheme being the "
                        "pool's axis")
    p.add_argument("--embed-dim", dest="embed_dim", type=int, default=8,
                   help="tile-embedding width for --extractor embed")
    p.add_argument("--d-model", dest="d_model", type=int, default=96,
                   help="attn arms: token width (3x3 trainable-matched value)")
    p.add_argument("--attn-layers", dest="attn_layers", type=int, default=2,
                   help="attn arms: number of pre-LN encoder blocks")
    p.add_argument("--attn-heads", dest="attn_heads", type=int, default=4,
                   help="attn arms: attention heads (must divide --d-model)")
    p.add_argument("--ffn-mult", dest="ffn_mult", type=int, default=2,
                   help="attn arms: FFN hidden = ffn_mult x d_model")
    p.add_argument("--resume-from", dest="resume_from", type=str, default=None,
                   help="path to a parent run's final {scenario}_{algo}.zip; "
                        "continues it toward --total_timesteps (read as the "
                        "FINAL total, not the increment) with the anneal "
                        "picked up mid-schedule, as if the parent had been "
                        "launched at that total all along")
    p.add_argument("--anneal-steps", dest="anneal_steps", type=int,
                   default=None,
                   help="lr/clip reach their floor at this step and HOLD "
                        "there; total_timesteps becomes a pure cap. Default "
                        "= total_timesteps (the historical behavior, where "
                        "the anneal spans the whole run)")
    p.add_argument("--lr-floor-frac", dest="lr_floor_frac", type=float,
                   default=0.1,
                   help="anneal floor as a fraction of learning_rate "
                        "(1.0 = constant lr; historical default 0.1)")
    p.add_argument("--clip-floor-frac", dest="clip_floor_frac", type=float,
                   default=0.25,
                   help="anneal floor as a fraction of clip_range "
                        "(1.0 = constant clip; historical default 0.25)")
    p.add_argument("--resume-fresh-schedule", dest="resume_fresh_schedule",
                   action="store_true",
                   help="with --resume-from: rebuild lr/clip schedules, "
                        "ent_coef and target_kl from THIS invocation's args "
                        "instead of the parent's pickled ones (the HP-probe "
                        "mechanism; without it a resumed leg keeps the "
                        "parent's treatment exactly)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--eval_freq", type=int, default=25_000)
    p.add_argument("--checkpoint-freq", dest="checkpoint_freq", type=int,
                   default=1_000_000,
                   help="save ckpt_{steps}.zip every this many env steps "
                        "(0 = off). best_model.zip is a running MAX that is "
                        "OVERWRITTEN, so without these the weights behind "
                        "every non-final peak are unrecoverable — and 4x4 "
                        "curves are humped (SA peaked at 6.45M then lost 11%%; "
                        "#E27). Keeping them is what would let a mab-style "
                        "two-block funnel rescore the top-k later; saving is "
                        "cheap (~5MB/ckpt) and commits to no protocol change")
    p.add_argument("--n_eval_episodes", type=int, default=200)
    p.add_argument("--progress_bar", action="store_true")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    return _build_arg_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_schedule(initial: float, final: float):
    """SB3 passes progress_remaining: 1 -> 0 over training."""
    def f(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)
    return f


def _capped_linear_schedule(initial: float, final: float,
                            total_steps: int, anneal_steps: int):
    """Linear anneal that reaches ``final`` at ``anneal_steps`` and HOLDS.

    With anneal_steps == total_steps this is exactly ``_linear_schedule``;
    with anneal_steps < total_steps the tail past the floor trains at a
    constant floor, so "converged" keeps meaning "flat at the floor" even
    when total_steps is a generous cap the run does not reach.
    """
    def f(progress_remaining: float) -> float:
        t = (1.0 - progress_remaining) * total_steps
        a = min(t / anneal_steps, 1.0)
        return initial + a * (final - initial)
    return f


def _resolve_policy(policy: str, obs_shape: tuple[int, ...]) -> str:
    if policy == "mlp":
        return "MlpPolicy"
    if policy == "cnn":
        return "CnnPolicy"
    return "CnnPolicy" if len(obs_shape) >= 3 else "MlpPolicy"


def _make_env(args, seed_offset: int = 0, reward_mode: str | None = None):
    def _init():
        env = Game2048Env(
            SCENARIOS[args.scenario_name],
            observation_mode=args.observation_mode,
            action_mode=args.action_mode,
            reward_mode=reward_mode or args.reward_mode,
            logger_filename=getattr(args, "gym_log", None),
        )
        env = Monitor(env)
        if args.action_mode == "masked":
            env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
        return env
    return _init


def _ir_mdp_fingerprint() -> str:
    """The IR's mdp-layer fingerprint (spec §8.4 provenance).

    Read from the schema beside the domain at run time, so it can never drift
    from the IR the domain was generated from. Best-effort: a missing schema
    or an mdp_ir version without the helper must not stop a training run.
    """
    try:
        from mdp_ir import load_ir
        ir_path = Path(__file__).resolve().parent / "game2048_schema.json"
        return str(load_ir(ir_path).mdp_fingerprint())
    except Exception as exc:                     # never block a launch
        return f"unavailable({type(exc).__name__})"


def _run_name(args, algo_tag: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"{algo_tag}_{stamp}_obs{args.observation_mode}"
            f"_act{args.action_mode}_rew{args.reward_mode}_{args.level}")
    defaults = _build_arg_parser().parse_args([])
    for key, short in (("learning_rate", "lr"), ("n_steps", "ns"),
                       ("vf_coef", "vf"),
                       ("batch_size", "bs"), ("ent_coef", "ent"),
                       ("gamma", "g"), ("gae_lambda", "lam"),
                       ("features_dim", "fdim"),
                       ("kernel_size", "k"), ("embed_dim", "ed"),
                       # attn arms' architecture dials, same rule as channels
                       # below: a width/depth change must be in the dir name
                       ("d_model", "dm"), ("attn_layers", "nl"),
                       ("attn_heads", "nh"), ("ffn_mult", "fm"),
                       ("total_timesteps", "ts"), ("anneal_steps", "as"),
                       ("lr_floor_frac", "lrf"),
                       ("clip_floor_frac", "clf"), ("target_kl", "tkl"),
                       # a second seed of the SAME config is a different run,
                       # not the same one: without this, seed-paired arms
                       # launched in one job land in one dir and race for
                       # best_model.zip (2026-08-01, 4x4 seeds 42/43)
                       ("seed", "sd")):
        val = getattr(args, key)
        # a level preset may switch a knob OFF (L0 sets target_kl=None); the
        # level tag already carries that, and None has no ":g" format
        if val is not None and val != getattr(defaults, key):
            name += f"_{short}{val:g}"
    # non-default policy family / extractor / widths change the architecture —
    # they must be visible in the run dir, not only in args.txt (the #E14 pool
    # launches six architecture arms in one job; a dir name that hides the
    # arm is the B3 collision all over again)
    if args.policy != defaults.policy:
        name += f"_pol{args.policy}"
    if args.extractor != defaults.extractor:
        name += f"_ex{args.extractor}"
    if list(args.channels) != list(defaults.channels):
        name += "_ch" + "x".join(str(c) for c in args.channels)
    if list(args.net_arch) != list(defaults.net_arch):
        name += "_na" + "x".join(str(w) for w in args.net_arch)
    if getattr(args, "tag", None):
        name += f"_{args.tag}"
    # a warm-started run is NOT interchangeable with a fresh one of the same
    # total — the first leg annealed on a different schedule; say so in the dir
    if args.resume_from:
        name += "_resumed"
    # a fresh-schedule resume is a TREATMENT change, not a continuation
    if getattr(args, "resume_fresh_schedule", False):
        name += "_freshsched"
    return name


class _Tee:
    """Mirror stdout/stderr into the run's own train.log (spec §8.4)."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.file = path.open("a", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)
    if getattr(args, "checkpoint_every_frac", None) is not None:
        if not (0.0 < args.checkpoint_every_frac <= 1.0):
            raise SystemExit("--checkpoint_every_frac must be in (0, 1], got "
                             f"{args.checkpoint_every_frac}")
        # resolve onto the absolute knob so ONE value reaches the callback,
        # the run name and the args log (no second source of truth)
        args.checkpoint_freq = max(
            1, int(args.checkpoint_every_frac * args.total_timesteps))
    is_masked = args.action_mode == "masked"
    algo_cls = MaskablePPO if is_masked else PPO
    algo_tag = "mppo" if is_masked else "ppo"
    if args.level == "L0":
        # faithful defaults: only what the problem forces (spec §8.6)
        args.n_envs = 1
        args.n_steps, args.batch_size, args.n_epochs = 2048, 64, 10
        args.learning_rate, args.clip_range, args.ent_coef = 3e-4, 0.2, 0.0
        args.gae_lambda, args.target_kl = 0.95, None
        args.net_arch = [64, 64]
        args.policy = "mlp"

    run_dir = (Path(__file__).resolve().parent / args.outdir
               / args.scenario_name / _run_name(args, algo_tag))
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.stdout, run_dir / "train.log")
    sys.stderr = _Tee(sys.stderr, run_dir / "train.log")
    print(f"outdir={run_dir}")

    train_env = DummyVecEnv([_make_env(args) for _ in range(args.n_envs)])
    train_env.seed(args.seed)
    # obs norm OFF per rl.obs_normalization (drifting tile magnitudes);
    # reward norm ON with gamma passed (spec §8.3)
    use_vecnorm = args.level != "L0"
    if use_vecnorm:
        if args.resume_from:
            # carry the parent's reward-normalization running stats across the
            # boundary; a fresh VecNormalize would re-estimate them from zero
            # and mis-scale the advantages for the first rollouts
            vn_path = Path(args.resume_from).parent / "vecnormalize.pkl"
            train_env = VecNormalize.load(str(vn_path), train_env)
        else:
            train_env = VecNormalize(train_env,
                                     norm_obs=args.norm_obs,
                                     clip_obs=args.vecnorm_clip_obs,
                                     norm_reward=args.norm_reward,
                                     gamma=args.gamma)

    # SELECTION IS ALWAYS ON THE CAMPAIGN OBJECTIVE. best_model.zip is a max
    # over eval/mean_reward, so an env built from args would select on the
    # TRAINING reward -- under -r empty it would keep the checkpoint best at
    # hoarding empty cells, then @8192 would score that model in merge
    # points. Pinning score here makes best_model comparable across reward
    # modes and matches the seal already on the @8192 path
    # (game2048_benchmark_common.evaluate). No-op for score-trained arms.
    eval_env = DummyVecEnv([_make_env(args, reward_mode="score")])
    eval_env.seed(SELECTION_SEED_BASE)
    if args.reward_mode != "score":
        print(f"SELECTION PINNED: trained on {args.reward_mode}, but "
              f"eval/mean_reward and best_model select on merge SCORE")
    if use_vecnorm:
        eval_env = VecNormalize(eval_env, norm_obs=args.norm_obs,
                                clip_obs=args.vecnorm_clip_obs,
                                norm_reward=False,
                                gamma=args.gamma, training=False)

    obs_shape = train_env.observation_space.shape
    policy = _resolve_policy(args.policy, obs_shape)
    policy_kwargs = {"net_arch": list(args.net_arch)}
    if policy == "CnnPolicy":
        # embed and conv3d read one-hot SEMANTICS (a lookup table / an
        # exponent axis), not just a 3D shape — refuse the grid mode loudly
        if args.extractor in ("embed", "conv3d") and args.observation_mode != "onehot":
            raise SystemExit(
                f"--extractor {args.extractor} reads the one-hot planes; "
                f"run it with -o onehot, not {args.observation_mode!r}")
        fe_kwargs: dict = {"features_dim": args.features_dim}
        if args.extractor.startswith("attn"):
            # attn arms have their own capacity dials; passing the CNN's
            # channels/kernel here would silently make them no-ops
            fe_kwargs.update(d_model=args.d_model, n_layers=args.attn_layers,
                             n_heads=args.attn_heads, ffn_mult=args.ffn_mult)
        else:
            fe_kwargs["channels"] = tuple(args.channels)
            # rowcol/axial kernels are fixed by the axis geometry (1xn/nx1)
            if args.extractor not in ("rowcol", "axial"):
                fe_kwargs["kernel_size"] = args.kernel_size
            if args.extractor == "embed":
                fe_kwargs["embed_dim"] = args.embed_dim
        policy_kwargs["features_extractor_class"] = EXTRACTORS[args.extractor]
        policy_kwargs["features_extractor_kwargs"] = fe_kwargs
    elif args.extractor != "small":
        raise SystemExit(f"--extractor {args.extractor} needs a CNN policy; "
                         f"this run resolved to {policy}")

    print(f"\nlevel={args.level}  algo={algo_tag}  policy={policy}  "
          f"extractor={args.extractor}  obs_shape={obs_shape}")
    if args.level == "L1":
        print("L1 derivation (spec §8.6):")
        rollout = args.n_steps * args.n_envs
        for line in (
            f"  gamma={args.gamma}: = objective.discount_factor (beta)",
            f"  gae_lambda={args.gae_lambda}: merges pay immediately",
            f"  rollout={args.n_steps}x{args.n_envs}={rollout}: >=2048 and >=10 episodes",
            f"  batch_size={args.batch_size}: rollout/16",
            f"  lr={args.learning_rate:g}->"
            f"{args.learning_rate * args.lr_floor_frac:g}: "
            "per-step reward is deterministic given (board, move)",
            f"  clip={args.clip_range}->"
            f"{args.clip_range * args.clip_floor_frac:g}: clip_init*frac",
            f"  ent_coef={args.ent_coef}: default; not an exploration problem",
            f"  norm_obs=False: tile magnitudes drift all episode",
            f"  norm_reward={use_vecnorm}: spec §8.3, gamma passed",
            f"  net_arch={args.net_arch}: obs dim {int(np.prod(obs_shape))}",
        ):
            print(line)
        if policy == "CnnPolicy":
            print("  NOTE predicted escalation: arch — structured board obs")

    kwargs: dict = dict(
        policy=policy, env=train_env, seed=args.seed, verbose=1,
        gamma=args.gamma, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir),
    )
    if args.level == "L0":
        kwargs.update(learning_rate=args.learning_rate,
                      clip_range=args.clip_range)
    else:
        anneal_steps = args.anneal_steps or args.total_timesteps
        lr_floor = (args.lr_final if args.lr_final is not None
                    else args.learning_rate * args.lr_floor_frac)
        lr_sched = _capped_linear_schedule(
            args.learning_rate, lr_floor,
            args.total_timesteps, anneal_steps)
        clip_sched = _capped_linear_schedule(
            args.clip_range, args.clip_range * args.clip_floor_frac,
            args.total_timesteps, anneal_steps)
        kwargs.update(
            learning_rate=lr_sched,
            clip_range=clip_sched,
            target_kl=args.target_kl,
        )
    if args.resume_from:
        # The parent's saved zip carries policy weights AND optimizer state, so
        # this is a true continuation, not a re-initialization from weights.
        # learn() below gets the REMAINING steps with reset_num_timesteps=False;
        # SB3 then sets _total_timesteps = remaining + num_timesteps, so
        # progress_remaining starts at 1 - done/total and the lr/clip anneal
        # picks up exactly where a fresh run of args.total_timesteps would be.
        # load() restores the PARENT's pickled schedules; only
        # --resume-fresh-schedule swaps in this invocation's treatment. The
        # override must target the saved DATA keys ("learning_rate",
        # "clip_range") — _setup_model() rebuilds lr_schedule from
        # data["learning_rate"] after load, so overriding "lr_schedule"
        # itself is silently undone.
        custom = ({"learning_rate": kwargs["learning_rate"],
                   "clip_range": kwargs["clip_range"]}
                  if args.resume_fresh_schedule else None)
        model = algo_cls.load(args.resume_from, env=train_env,
                              tensorboard_log=str(run_dir),
                              custom_objects=custom)
        done_steps = model.num_timesteps
        train_steps = args.total_timesteps - done_steps
        if train_steps <= 0:
            raise SystemExit(
                f"--total_timesteps {args.total_timesteps:,} is not beyond the "
                f"parent's {done_steps:,} already-trained steps")
        print(f"\nRESUMED from {args.resume_from}\n"
              f"  parent steps  = {done_steps:,}\n"
              f"  this leg adds = {train_steps:,}  -> {args.total_timesteps:,} total")
        if args.resume_fresh_schedule:
            model.ent_coef = args.ent_coef
            model.target_kl = args.target_kl
            print(f"  FRESH SCHEDULE (probe treatment, not the parent's):\n"
                  f"    lr {args.learning_rate:g} -> floor "
                  f"{args.learning_rate * args.lr_floor_frac:g}"
                  f" @ {args.anneal_steps or args.total_timesteps:,} steps\n"
                  f"    clip {args.clip_range:g} -> floor "
                  f"{args.clip_range * args.clip_floor_frac:g}\n"
                  f"    ent_coef={args.ent_coef:g}  target_kl={args.target_kl}")
        print(f"  lr resumes at   "
              f"{model.lr_schedule(1.0 - done_steps / args.total_timesteps):.3g}")
    else:
        model = algo_cls(**kwargs)
        train_steps = args.total_timesteps
    # §8.4 provenance: what CODE produced this run. algo_class pins the
    # trainer subclass, ir_mdp_fingerprint pins the model layer the domain
    # was generated from, so a run dir joins to its IR revision.
    (run_dir / f"{args.scenario_name}_{algo_tag}_args.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in sorted(vars(args).items()))
        + f"\npolicy={policy}\nobs_shape={obs_shape}\n"
        + f"sb3_version={__import__('stable_baselines3').__version__}\n"
        + f"algo_class={algo_cls.__name__}\n"
        + f"ir_mdp_fingerprint={_ir_mdp_fingerprint()}\n"
    )

    # No early stopping, ever: spec §8.6 — "A training run runs to its budget.
    # The training trajectory is too noisy to make any judgment from."
    # --patience-evals (SB3 StopTrainingOnNoModelImprovement) lived here behind
    # a None default long after the user call of 2026-08-01 dropped
    # patience-stopping, and it truncated the 2026-07-31 40M pair at 28.5% and
    # 32.4% of budget. Removed 2026-08-22: passing the flag now fails loudly at
    # argparse instead of silently shortening an arm.
    cb_cls = MaskableEvalCallback if is_masked else EvalCallback
    callback = cb_cls(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True, render=False,
    )
    callbacks: list = [callback]
    if args.checkpoint_freq:
        # save_freq is counted in CALLS, which happen once per env step per
        # env — so divide by n_envs to make --checkpoint-freq mean ENV STEPS,
        # the unit every other budget in this domain is quoted in (same
        # correction eval_freq needs above). Names are ckpt_{steps}_steps.zip
        # with steps = the true env-step count, so a checkpoint joins the
        # evaluations.npz timeline directly.
        ckpt_dir = run_dir / "checkpoints"
        callbacks.append(CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(ckpt_dir), name_prefix="ckpt", verbose=0))
        print(f"periodic checkpoints: every {args.checkpoint_freq:,} env "
              f"steps -> {ckpt_dir.name}/ (best_model.zip is a running max "
              "and overwrites; these do not)")

    model.learn(total_timesteps=train_steps, callback=callbacks,
                reset_num_timesteps=not args.resume_from,
                progress_bar=args.progress_bar)

    model_path = run_dir / f"{args.scenario_name}_{algo_tag}.zip"
    model.save(model_path)
    if use_vecnorm:
        train_env.save(str(run_dir / "vecnormalize.pkl"))

    evaluate = maskable_evaluate if is_masked else __import__(
        "stable_baselines3.common.evaluation", fromlist=["evaluate_policy"]
    ).evaluate_policy
    mean_r, std_r = evaluate(model, eval_env, n_eval_episodes=200,
                             deterministic=True)
    print(f"\nfinal (selection seeds): mean_reward={mean_r:.2f} +/- {std_r:.2f}")
    print(f"model      = {model_path}")
    print(f"best model = {run_dir / 'best_model.zip'}")


if __name__ == "__main__":
    main()

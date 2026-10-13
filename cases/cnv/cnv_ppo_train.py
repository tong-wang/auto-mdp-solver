"""SB3 PPO training script for the censored newsvendor (spec §8).

Trains PPO on CnvEnv, saves ~20 checkpoints for the post-hoc screen
(``cnv_select.py``), the terminal model and the VecNormalize stats, then runs
a short smoke eval that is never a result.

Example usage:
    python cnv_ppo_train.py -s base -o timing_belief
    python cnv_ppo_train.py -s base -o timing_raw            # frame stack = horizon
    python cnv_ppo_train.py -g paper510 -o timing_belief     # generalist on the grid sampler
    python cnv_ppo_train.py -s base --level L0 --learning_rate 3e-4 --lr-final 3e-4 \\
        --clip-init 0.2 --clip-final 0.2 --n-steps 2048 --batch-size 64 --ent-coef 0 \\
        --target-kl -1 --n-envs 1 --no-norm-obs --no-norm-reward --checkpoint-every-frac 0
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import stable_baselines3
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cnv_grids import GRIDS
from cnv_gym import ALL_ACTION_MODES, ALL_OBSERVATION_MODES, CnvEnv
from cnv_scenarios import SCENARIOS
from mdp_conformance.launch import assert_l1_current

HERE = Path(__file__).resolve().parent

# the raw modes carry only the last period and need a memory over the horizon
# (IR rl.requires_memory); the belief modes are Markov on their own
RAW_MODES = ("event_raw", "timing_raw", "full_raw")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on the censored newsvendor.")
    p.add_argument("-s", "--scenario_name", type=str, default="base",
                   choices=list(SCENARIOS) + list(GRIDS))
    p.add_argument("-g", "--grid_name", type=str, default=None, choices=list(GRIDS.keys()),
                   help="train a GENERALIST on this GRIDS entry via grid.as_sampler(); "
                        "overrides -s, and the run lands under results/{grid_name}/")
    p.add_argument("-o", "--observation_mode", type=str, default="timing_belief",
                   choices=list(ALL_OBSERVATION_MODES))
    p.add_argument("-a", "--action_mode", type=str, default="order_discrete",
                   choices=list(ALL_ACTION_MODES))
    p.add_argument("-r", "--reward_mode", type=str, default="profit", choices=["profit"])
    # spec §8.6 solve level — RECORDED, never inferred: the claim about how much
    # search produced this config, which only the caller knows
    p.add_argument("--level", type=str, default="L1",
                   help="solve level for the run name and args log (L0, L1, L2(hp), ...)")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--tag", type=str, default="",
                   help="the ESCALATION.md address this run was launched under (spec §8.2 tier 1)")
    p.add_argument("--checkpoint-every-frac", type=float, default=0.05,
                   help="checkpoint cadence as a fraction of the budget (0.05 = ~20 checkpoints; 0 disables)")
    p.add_argument("--report-every", type=int, default=50_000)
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument("--gym-log", type=int, default=0, choices=[0, 1],
                   help="write per-episode/per-step gym logs (spec §11); off by default")
    # tier 3: memory. None = derived at runtime from the observation mode
    # (horizon for the raw modes, 1 for the belief modes, IR rl.frame_stack)
    p.add_argument("--frame-stack", type=int, default=None,
                   help="FrameStackObservation depth; default: horizon for *_raw modes, 1 otherwise")
    # PPO hyperparameters (tier 2; --learning_rate dest is the mdp_tuning contract)
    p.add_argument("--learning_rate", "--lr-init", dest="learning_rate", type=float, default=1e-4)
    p.add_argument("--lr-final", type=float, default=1e-5)
    p.add_argument("--clip-init", type=float, default=0.2)
    p.add_argument("--clip-final", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=1.0)          # gamma = beta = 1 (undiscounted)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True)
    # negative disables the safety valve (SB3's own default is None) — the L0 form
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--net_arch", type=int, nargs="+", default=[64, 64])
    # arch layer: the action HEAD. "categorical" is SB3's flat softmax over the 403
    # stocking levels (the derivation); "dgauss" is the discretized Gaussian of
    # cnv_dgauss_head.py — two numbers (location, width) whose location gradient pools
    # every sample, the mechanism a flat categorical lacks on this heavy-tailed order
    # distribution (#E44). The action space is unchanged either way.
    p.add_argument("--policy-dist", dest="policy_dist", default="categorical",
                   choices=["categorical", "dgauss"],
                   help="action head: SB3's flat softmax, or the discretized Gaussian")
    # #E87: common random numbers in training. K > 1: the training envs share episode seeds in groups
    # of K and the advantage is each member's Monte-Carlo return-to-go minus the leave-one-out mean of
    # its group's (cnv_group_baseline); needs K | n_envs and horizon | n_steps. 0 = off (GAE + critic).
    p.add_argument("--group-baseline", dest="group_baseline", type=int, default=0,
                   help="group size K for the leave-one-out group baseline (0 = off)")
    # #E98: the group's shared worlds WITHOUT its advantage — plain PPO (GAE + critic) on training envs
    # that share episode seeds in groups of K (n_envs / K distinct worlds per rollout column).
    p.add_argument("--share-worlds", dest="share_worlds", type=int, default=0,
                   help="group size K for shared episode seeds under plain PPO (0 = off; not with --group-baseline)")
    # #E88: branch-and-grid (cnv_branch_ppo): each group-episode branches at t_b ~ truncated exponential;
    # before t_b the members copy one parent's actions, at t_b they take K different actions (a grid).
    p.add_argument("--branch", action="store_true", default=False,
                   help="branch-and-grid group PPO (needs --group-baseline K)")
    p.add_argument("--branch-tmax", dest="branch_tmax", type=int, default=20)
    p.add_argument("--branch-tau", dest="branch_tau", type=float, default=5.0)
    p.add_argument("--branch-grid", dest="branch_grid", default="topk", choices=["topk", "window"])
    # #E96: staged training — start from a trained checkpoint (its weights, extractor statistics and
    # VecNormalize stats), optionally with the features extractor frozen (the encoding fixed).
    p.add_argument("--init-from", dest="init_from", default=None,
                   help="warm start from this model .zip (and its VecNormalize stats)")
    p.add_argument("--belief-estimator", dest="belief_estimator", default=None,
                   help="raw modes (#E109; arch): a SUPERVISED belief estimator (cnv_belief_estimator.py, trained and frozen) renders "
                        "[ttg, ratio, log mean, CV] from the raw frames as the policy's features — the staged form; implies --freeze-extractor")
    p.add_argument("--freeze-extractor", dest="freeze_extractor", action="store_true", default=False,
                   help="freeze the features extractor's parameters (with --init-from: the encoding fixed)")
    p.add_argument("--branch-hybrid", dest="branch_hybrid", action="store_true", default=False,
                   help="#E97: group comparison at the branch step only, the critic's GAE elsewhere")
    p.add_argument("--branch-uniform", dest="branch_uniform", type=float, default=0.0,
                   help="#E91: weight of a uniform-over-the-episode part in the branch-time law (0 = early only)")
    # #E110: the vine (cnv_vine_ppo, owmr #E30's coordinate vine): the main rollout untouched; at each step, with
    # probability p, an env is forked on its own world into a parent copy and fresh-draw siblings played to the
    # episode's end on common continuation noise; a side PPO pass on sibling-vs-parent-copy contrasts.
    p.add_argument("--vine", action="store_true", default=False, help="the vine side pass (cnv_vine_ppo.VinePPO)")
    p.add_argument("--vine-p", dest="vine_p", type=float, default=0.2, help="per-step, per-env fork probability")
    p.add_argument("--vine-siblings", dest="vine_siblings", type=int, default=1, help="fresh-draw siblings per fork")
    p.add_argument("--vine-weight", dest="vine_weight", type=float, default=1.0,
                   help="side advantages' scale after rescaling to the main pass's units")
    p.add_argument("--vine-lr-scale", dest="vine_lr_scale", type=float, default=None,
                   help="a separate side optimizer at this × the main lr (None = share the policy's, owmr #E30's form)")
    p.add_argument("--vine-baseline", dest="vine_baseline", default="parent", choices=["parent", "loo"],
                   help="siblings against the parent copy (owmr) or every copy against the others' mean (1-D RLOO)")
    p.add_argument("--mu-param", dest="mu_param", default="log", choices=["log", "affine", "sqrt"],
                   help="dgauss location: multiplicative exp() for constant relative resolution "
                        "(log, the default here) or affine across the support (clark_scarf's form)")
    p.add_argument("--tau-min", dest="tau_min", type=float, default=0.5,
                   help="dgauss width floor, in order units")
    p.add_argument("--tau-scale", dest="tau_scale", type=float, default=290.0,
                   help="dgauss softplus scale for the width")
    p.add_argument("--zero-atom", dest="zero_atom", action="store_true",
                   help="dgauss only: a third head output owns P(order 0) (#E57; arch)")
    p.add_argument("--raw-compress", dest="raw_compress", action="store_true",
                   help="raw modes: learned compression of the frame stack, then asinh + running standardisation (cnv_raw_extractor; #E68; arch). Run with --no-norm-obs")
    p.add_argument("--raw-mix", dest="raw_mix", action="store_true",
                   help="with --raw-compress: a linear mixing layer after the asinh, so log-ratios of the compressed sums are one linear step away (#E69; arch)")
    p.add_argument("--raw-no-asinh", dest="raw_asinh", action="store_false", default=True,
                   help="with --raw-compress: no asinh between the compression and the standardisation (#E70; arch)")
    p.add_argument("--raw-depth", dest="raw_depth", type=int, default=1,
                   help="with --raw-compress: number of GELU layers in the compression (#E70; arch)")
    p.add_argument("--raw-frame", dest="raw_frame", action="store_true",
                   help="raw modes: the per-frame log-likelihood extractor (cnv_raw_extractor.FrameSumExtractor; #E72; arch) instead of the MLP compression")
    p.add_argument("--raw-frame-v2", dest="raw_frame_v2", action="store_true",
                   help="raw modes: the CORRECTED per-frame extractor (#E74; arch): reset frame excluded from the pool, phi on the observation features only, present state passed directly; implies --raw-frame")
    p.add_argument("--raw-phi-time", dest="raw_phi_time", action="store_true",
                   help="with --raw-frame-v2: keep the frame's time-to-go in phi (v2b, #E74b) — the sum then carries nonlinear functions of the count")
    p.add_argument("--raw-stats-off", dest="raw_stats_off", action="store_true",
                   help="with --raw-frame-v2: NO normalisation after the pool (#E76; arch) — identity, the body learns the scale")
    p.add_argument("--raw-no-units", dest="raw_no_units", action="store_true",
                   help="with --raw-stats-off: pure identity after the pool, no fixed units either (#E76 arm A)")
    p.add_argument("--raw-stats-freeze", dest="raw_stats_freeze", action="store_true",
                   help="with --raw-stats-off: the pooled channels standardised by the FIRST rollout's mean/sd, frozen (#E77); count/state in fixed units")
    p.add_argument("--raw-stats-freeze-at", dest="raw_stats_freeze_at", type=int, default=0,
                   help="with --raw-stats-off: pooled-channel statistics run through the first K rollouts, then frozen (#E78); implies --raw-stats-freeze")
    p.add_argument("--raw-pool", dest="raw_pool", default="sum", choices=["sum", "mean", "kernel"],
                   help="with --raw-frame: how the per-frame channels are pooled over the stack (#E72; arch)")
    p.add_argument("--popart", dest="popart", action="store_true",
                   help="PopArt-normalised value head (cnv_popart.PopArtPPO; #E57; hp)")
    p.add_argument("--n-envs", type=int, default=4)             # structural, never tuned
    # VecNormalize (spec §8.3): both flags reachable so L0 can switch the wrapper off
    p.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    p.add_argument("--no-norm-reward", action="store_false", dest="norm_reward", default=True)
    p.add_argument("--no-norm-obs", action="store_false", dest="norm_obs", default=True)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# The L1 derivation as data (spec §8.4 / §8.6). The parser defaults above ARE
# these values; this dict is what the launch check verifies a run against and
# what the run name diffs. Each row's basis, from the §8.6 table at T = 4
# (the `base` instance and every N = 4 cell), beta = 1.0:
#   gamma 1.0              — gamma = beta; the objective is undiscounted.
#   gae_lambda 0.95        — credit horizon 1/(1-lambda) = 20 > T = 4: an
#                            order's consequence (what a stock-out teaches)
#                            realizes over the whole remaining horizon, so the
#                            credit must span the episode; at T = 4 the
#                            variance cost of doing so is negligible.
#   n_envs 4, n_steps 512  — rollout 2048 transitions = 512 episodes >=
#                            max(2048, 10 x 4); the floor is met by n_steps
#                            alone at this short T, so n_envs stays at the
#                            structural default.
#   batch_size 128         — rollout/16, power of two, in [rollout/32, /8].
#   learning_rate 1e-4 -> 1e-5, clip 0.2 -> 0.05 — noisy-reward branch of the
#                            LR row: per-step reward variance is dominated by
#                            the Poisson draw and the hidden rate; finals =
#                            init/10 and init/4.
#   ent_coef 0.01          — the premature-determinism hazard row: the
#                            over-ordering the paper proves optimal IS
#                            exploration (a stock-out buys information), a
#                            bandit-like hazard.
#   n_epochs 10, target_kl 0.02 — fixed by the row; the valve is never tuned.
#   net_arch [64, 64]      — obs dim <= 6, x4 under the frame stack = 24 <= 32.
#   norm_obs True          — §8.3 prior: posterior shape (10..10^2), rate (1..5),
#                            critical ratio (0..1) are heterogeneous stationary
#                            magnitudes; checked at the breadth tier, not asserted.
#   norm_reward True, normalize_advantage True — §8.3/§8.6 prior rows.
# NOT here: vf_coef, max_grad_norm (SB3's own, no row), vecnorm_clip_obs,
# frame_stack (tier 3, derived per mode at runtime), total_timesteps (the
# budget row derives a band: 20k-50k episodes x 4 = 80k-200k steps AND >= 300
# updates x 2048 = 614k; 1M is the update floor rounded to the checkpoint grid,
# 244 episodes x 1000 updates), checkpoint_every_frac, seed.
_L1_DERIVED: dict[str, object] = {
    "learning_rate": 1e-4, "lr_final": 1e-5,
    "clip_init": 0.2, "clip_final": 0.05,
    "n_envs": 4, "n_steps": 512, "batch_size": 128,
    "n_epochs": 10, "target_kl": 0.02,
    "gamma": 1.0, "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "net_arch": [64, 64],
    "norm_obs": True, "norm_reward": True, "normalize_advantage": True,
}
_L1_BASIS: dict[str, object] = {"scenario_name": "base", "episode_len": 4, "beta": 1.0}

_GYM_KNOBS = frozenset({"norm_obs", "norm_reward", "vecnorm_clip_obs", "frame_stack"})
# the action HEAD is an arch-layer deviation (spec §8.6): moving it off the categorical
# derivation tags the run L2+(arch), which is what a head swap is.
_ARCH_KNOBS: frozenset[str] = frozenset({"policy_dist", "mu_param", "tau_min", "tau_scale", "zero_atom", "belief_estimator", "raw_compress", "raw_mix", "raw_asinh", "raw_depth", "raw_frame", "raw_frame_v2", "raw_phi_time", "raw_pool", "raw_stats_off", "raw_no_units", "raw_stats_freeze", "raw_stats_freeze_at"})
_LAYER_ORDER = ("hp", "gym", "arch")
_NOT_CONFIG = frozenset({"seed", "total_timesteps"})

_SHORT_KEYS: dict[str, str] = {
    "total_timesteps": "steps", "seed": "seed",
    "learning_rate": "lr", "lr_final": "lrf", "clip_init": "clip", "clip_final": "clipf",
    "n_steps": "nsteps", "batch_size": "bs", "n_epochs": "ep", "gamma": "gamma",
    "gae_lambda": "gae", "ent_coef": "ent", "vf_coef": "vf", "max_grad_norm": "grad",
    "target_kl": "kl", "net_arch": "arch", "n_envs": "nenvs", "frame_stack": "fs",
    "vecnorm_clip_obs": "clipobs", "norm_reward": "normrew", "norm_obs": "normobs",
    "normalize_advantage": "normadv", "group_baseline": "grp", "share_worlds": "shw",
    "branch": "br", "branch_tmax": "brT", "branch_tau": "brtau", "branch_grid": "brgrid", "branch_uniform": "brU",
    "freeze_extractor": "frzx", "branch_hybrid": "brH",
    "vine": "vine", "vine_p": "vp", "vine_siblings": "vsib", "vine_weight": "vw", "vine_lr_scale": "vls", "vine_baseline": "vbl",
}
_SKIP_KEYS = {"outdir", "scenario_name", "grid_name", "progress_bar", "checkpoint_every_frac",
              "report_every", "gym_log", "tag", "level", "init_from", "belief_estimator",
              "observation_mode", "action_mode", "reward_mode"}


_RUN_NAME_MAX = 240   # the filesystem's 255-byte component limit, with room for the tag


def build_run_name(args: argparse.Namespace) -> str:
    defaults = vars(_build_arg_parser().parse_args([]))
    defaults.update(_L1_DERIVED)          # the derivation wins where it speaks
    defaults["frame_stack"] = derived_frame_stack(args.observation_mode, args)
    parts = [args.level, f"obs{args.observation_mode}", f"act{args.action_mode}", f"rew{args.reward_mode}"]
    if getattr(args, "init_from", None):
        parts.append("warm")   # #E96: the source checkpoint is in the args log, not the name
    if getattr(args, "belief_estimator", None):
        parts.append("bestim")   # #E109: the estimator's path is in the args log, not the name
    for key, default_val in defaults.items():
        if key in _SKIP_KEYS:
            continue
        val = getattr(args, key)
        if val != default_val:
            label = _SHORT_KEYS.get(key, key)
            if isinstance(val, (list, tuple)):
                formatted = "-".join(str(v) for v in val)
            elif isinstance(val, float):
                formatted = f"{val:.3g}"
            else:
                formatted = str(val)
            parts.append(f"{label}{formatted}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if getattr(args, "tag", "") else ""
    name = f"PPO_{timestamp}_{'_'.join(parts)}{suffix}"
    if len(name.encode()) > _RUN_NAME_MAX:
        # ENAMETOOLONG (a wide search can move every knob): keep the head, replace the tail with a digest
        # of the full knob string; the complete configuration is in the run dir's *_args.txt regardless
        digest = hashlib.sha1("_".join(parts).encode()).hexdigest()[:8]
        head, tail = f"PPO_{timestamp}_", f"_x{digest}{suffix}"
        name = head + "_".join(parts)[: _RUN_NAME_MAX - len(head) - len(tail)].rstrip("_") + tail
    return name


def derived_frame_stack(observation_mode: str, args: argparse.Namespace) -> int:
    """The IR's rl.frame_stack, applied per mode: the horizon for the raw modes,
    1 for the belief modes. A grid target reads the family horizon."""
    if observation_mode not in RAW_MODES:
        return 1
    source = GRIDS[args.grid_name] if args.grid_name else (
        GRIDS[args.scenario_name] if args.scenario_name in GRIDS else SCENARIOS[args.scenario_name])
    return int(source.horizon)


def derive_deviation_tags(args: argparse.Namespace) -> str:
    """Which layers this configuration moved off the derivation ("" = at it)."""
    if str(args.level).lower() == "l0":
        return ""
    baseline = vars(_build_arg_parser().parse_args([]))
    baseline.update(_L1_DERIVED)
    baseline["frame_stack"] = derived_frame_stack(args.observation_mode, args)
    moved: set[str] = set()
    for dest, want in baseline.items():
        if dest in _SKIP_KEYS or dest in _NOT_CONFIG:
            continue
        got = getattr(args, dest, want)
        got = tuple(got) if isinstance(got, list) else got
        want = tuple(want) if isinstance(want, list) else want
        if got != want:
            moved.add("gym" if dest in _GYM_KNOBS else "arch" if dest in _ARCH_KNOBS else "hp")
    return "+".join(layer for layer in _LAYER_ORDER if layer in moved)


def resolve_target(args: argparse.Namespace):
    """Spec §9.6 target dispatch: a GRIDS entry trains a generalist on the
    grid's derived sampler and lands under results/{grid_name}/."""
    if args.grid_name:
        args.scenario_name = args.grid_name
        return GRIDS[args.grid_name].as_sampler()
    if args.scenario_name in GRIDS:
        return GRIDS[args.scenario_name].as_sampler()
    return SCENARIOS[args.scenario_name]


def resolve_paths(outdir_arg: str, scenario_name: str, run_name: str) -> tuple[Path, Path]:
    outdir = (HERE / outdir_arg / scenario_name / run_name).resolve()
    ckpt_dir = outdir / "checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return outdir, ckpt_dir


class _Tee:
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, data: str) -> int:
        self._stream.write(data); self._fh.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush(); self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def tee_console(outdir: Path, name: str = "train.log") -> None:
    """Mirror stdout/stderr into outdir/train.log (spec §8.4): the run dir is
    only known once the timestamped name exists, so the script captures its
    own console."""
    fh = open(outdir / name, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def _ir_fingerprint() -> str:
    try:
        from mdp_ir.schema import load_ir
        return load_ir(HERE / "cnv_schema.json").mdp_fingerprint()
    except Exception:
        return "?"


def record_algo_class(args: argparse.Namespace, outdir: Path, model) -> None:
    """Spec §8.4: `algo_class` is the class actually constructed — PPO or the domain's own subclass (VinePPO,
    BranchGroupPPO, PopArtPPO) — so the args file is corrected from the built model, not from the flags."""
    path = outdir / f"{args.scenario_name}_ppo_args.txt"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(f"algo_class: {type(model).__name__}" if ln.startswith("algo_class: ") else ln
                              for ln in lines) + "\n")


def write_args(args: argparse.Namespace, outdir: Path, deviation_tags: str) -> None:
    with open(outdir / f"{args.scenario_name}_ppo_args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        # spec §8.4 provenance set — literal keys a checker reads without domain knowledge
        f.write("algo_class: PPO\n")
        f.write(f"sb3_version: {stable_baselines3.__version__}\n")
        f.write(f"ir_mdp_fingerprint: {_ir_fingerprint()}\n")
        f.write(f"solve_level: {args.level}\n")
        if deviation_tags:
            f.write(f"deviation_tags: {deviation_tags}\n")


# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------

class NewsvendorMetricsCallback(BaseCallback):
    """Log mean order, demand, stock-out rate and profit periodically."""

    def __init__(self, report_every_steps: int, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.report_every_steps = report_every_steps
        self._reset()

    def _reset(self) -> None:
        self._order = self._demand = self._stockout = self._profit = 0.0
        self._n = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict) or "profit" not in info:
                continue
            self._order += float(info.get("order", 0.0))
            self._demand += float(info.get("demand", 0.0))
            self._stockout += float(info.get("stockout_event", 0.0))
            self._profit += float(info["profit"].get("total", 0.0))
            self._n += 1
        if self.num_timesteps > 0 and self.num_timesteps % self.report_every_steps == 0 and self._n > 0:
            print(f"step={self.num_timesteps}  order={self._order / self._n:.2f}  "
                  f"demand={self._demand / self._n:.2f}  stockout={self._stockout / self._n:.3f}  "
                  f"profit/period={self._profit / self._n:.3f}")
            self._reset()
        return True


# ---------------------------------------------------------------------------
# Build env / model / callbacks
# ---------------------------------------------------------------------------

def make_env_fn(args: argparse.Namespace, scenario, outdir: Path | None, rank: int,
                frame_stack: int, group_size: int = 0) -> Callable[[], object]:
    def _make():
        env = CnvEnv(scenario=scenario, observation_mode=args.observation_mode,
                     action_mode=args.action_mode, reward_mode=args.reward_mode,
                     logger_filename=(str(outdir / f"train_log_{rank}") if (outdir and args.gym_log) else None))
        if frame_stack > 1:
            env = FrameStackObservation(env, stack_size=frame_stack, padding_type="zero")
        env = Monitor(env, filename=str(outdir / f"monitor_{rank}") if outdir else None)
        if group_size > 1:   # #E87: the group's members share every episode seed
            from cnv_group_baseline import GroupSeedWrapper
            env = GroupSeedWrapper(env, rank, group_size)
        return env
    return _make


def build_training_env(args: argparse.Namespace, outdir: Path, scenario, frame_stack: int) -> VecNormalize:
    k = int(getattr(args, "group_baseline", 0) or 0) or int(getattr(args, "share_worlds", 0) or 0)
    env = DummyVecEnv([make_env_fn(args, scenario, outdir, i, frame_stack, k) for i in range(args.n_envs)])
    return VecNormalize(env, norm_obs=args.norm_obs, norm_reward=args.norm_reward,
                        clip_obs=args.vecnorm_clip_obs, gamma=args.gamma)


def build_schedule(start: float, end: float):
    if start == end:
        return float(start)
    return lambda progress_remaining: end + (start - end) * float(progress_remaining)


def build_model(args: argparse.Namespace, env, outdir: Path) -> PPO:
    policy_cls: type | str = "MlpPolicy"
    pkw: dict = {"net_arch": list(args.net_arch)}
    if getattr(args, "policy_dist", "categorical") == "dgauss":
        from cnv_dgauss_head import DGaussActorCriticPolicy   # saved models resolve it by module path
        policy_cls = DGaussActorCriticPolicy
        import math
        # the only problem information in the head: the scenario's own prior mean, which the
        # agent holds before it acts (and which the _ctx modes render). NOT a measured optimum
        # — the init is flat regardless (see cnv_dgauss_head).
        src = GRIDS[args.grid_name] if args.grid_name else (
            GRIDS[args.scenario_name] if args.scenario_name in GRIDS else SCENARIOS[args.scenario_name])
        probe = src(0) if callable(src) else src
        pkw.update(tau_min=float(args.tau_min), tau_scale=float(args.tau_scale),
                   mu_param=str(args.mu_param), log_center=math.log(max(float(probe.prior_mean), 1e-6)),
                   zero_atom=bool(args.zero_atom))
    else:
        assert not args.zero_atom, "--zero-atom needs --policy-dist dgauss"
    if args.raw_compress:
        assert args.observation_mode in RAW_MODES, "--raw-compress is for the *_raw modes"
        from cnv_raw_extractor import RawCompressExtractor
        pkw.update(features_extractor_class=RawCompressExtractor,
                   features_extractor_kwargs={"mix": bool(args.raw_mix), "asinh": bool(args.raw_asinh), "depth": int(args.raw_depth)})
    if args.raw_frame_v2:
        args.raw_frame = True
    if getattr(args, "raw_stats_freeze_at", 0):
        args.raw_stats_freeze = True
    if getattr(args, "belief_estimator", None):   # #E109: the staged form — a frozen supervised estimator is the whole extractor
        assert args.observation_mode in RAW_MODES and not args.raw_compress and not args.raw_frame, "--belief-estimator is for the *_raw modes, without --raw-frame / --raw-compress"
        from cnv_belief_estimator import FrozenBeliefExtractor
        pkw.update(features_extractor_class=FrozenBeliefExtractor, features_extractor_kwargs={"estimator_path": str(Path(args.belief_estimator).resolve())})
        args.freeze_extractor = True
    if args.raw_frame:
        assert args.observation_mode in RAW_MODES and not args.raw_compress, "--raw-frame is for the *_raw modes, instead of --raw-compress"
        from cnv_raw_extractor import FrameSumExtractor
        pkw.update(features_extractor_class=FrameSumExtractor, features_extractor_kwargs={"pool": str(args.raw_pool), "v2": bool(args.raw_frame_v2), "phi_time": bool(args.raw_phi_time), "standardize": not bool(args.raw_stats_off), "units": not bool(args.raw_no_units), "stats_freeze": bool(args.raw_stats_freeze), "freeze_at": max(1, int(getattr(args, "raw_stats_freeze_at", 0) or 1))})
    algo = PPO
    extra: dict = {}
    if args.popart:
        from cnv_popart import PopArtPPO
        algo = PopArtPPO
    if getattr(args, "branch", False):   # #E88
        assert not args.popart and int(args.group_baseline or 0) >= 2, "--branch needs --group-baseline K >= 2 (and no --popart)"
        from cnv_branch_ppo import BranchGroupPPO
        algo = BranchGroupPPO
        extra = dict(group_size=int(args.group_baseline), branch_tmax=int(args.branch_tmax),
                     branch_tau=float(args.branch_tau), branch_grid=str(args.branch_grid),
                     branch_uniform=float(args.branch_uniform), branch_hybrid=bool(args.branch_hybrid))
    if getattr(args, "vine", False):   # #E110
        assert algo is PPO and not args.group_baseline and not getattr(args, "share_worlds", 0), \
            "--vine keeps the main rollout untouched: not with --popart / --branch / --group-baseline / --share-worlds"
        from cnv_vine_ppo import VinePPO
        algo = VinePPO
        extra = dict(vine_p=float(args.vine_p), vine_siblings=int(args.vine_siblings), vine_weight=float(args.vine_weight),
                     vine_lr_scale=args.vine_lr_scale, vine_baseline=str(args.vine_baseline))
    return algo(
        policy=policy_cls,
        env=env,
        learning_rate=build_schedule(args.learning_rate, args.lr_final),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=build_schedule(args.clip_init, args.clip_final),
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl if args.target_kl >= 0 else None,
        normalize_advantage=args.normalize_advantage,
        policy_kwargs=pkw,
        stats_window_size=500,
        verbose=1,
        tensorboard_log=str(outdir),
        seed=args.seed,
        device="cpu",
        **extra,
    )


def warm_start(model, env: VecNormalize, src: Path) -> None:
    """#E96: load a checkpoint's policy (weights and buffers — the frame extractor's frozen statistics
    included) and its VecNormalize statistics into a freshly built model and training env."""
    from cnv_ppo_eval import _vecnorm_for
    model.set_parameters(str(src), exact_match=True, device="cpu")
    vn_path = _vecnorm_for(src, None)
    if vn_path is not None and Path(vn_path).exists():
        vn = VecNormalize.load(str(vn_path), env.venv)
        for attr in ("obs_rms", "ret_rms"):   # obs_rms exists only when observations were normalised
            if attr in vn.__dict__:
                setattr(env, attr, vn.__dict__[attr])
    print(f"warm start from {src} (vecnormalize {vn_path})")


def build_callbacks(args: argparse.Namespace, ckpt_dir: Path, horizon: int | None = None) -> CallbackList:
    """Checkpoints and metrics only — selection is post-hoc (spec §9.7)."""
    callbacks = [NewsvendorMetricsCallback(report_every_steps=max(1, args.report_every))]
    if args.checkpoint_every_frac > 0:
        save_freq = max(1, int(args.total_timesteps * args.checkpoint_every_frac / max(1, args.n_envs)))
        callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=str(ckpt_dir),
                                            name_prefix="ppo_cnv", save_replay_buffer=False,
                                            save_vecnormalize=True))
    if getattr(args, "raw_compress", False) or getattr(args, "raw_frame", False):
        from cnv_raw_extractor import RawStatsCallback
        callbacks.append(RawStatsCallback())   # the extractor's running statistics: refreshed between an update and the next rollout (#E71 + review)
    if int(getattr(args, "group_baseline", 0) or 0) > 1 and not getattr(args, "branch", False):   # --branch computes its own
        from cnv_group_baseline import GroupBaselineCallback
        callbacks.append(GroupBaselineCallback(args.group_baseline, int(horizon)))   # #E87
    return CallbackList(callbacks)


# ---------------------------------------------------------------------------
# Smoke eval (never a result, spec §9.7)
# ---------------------------------------------------------------------------

def smoke_eval(model: PPO, args: argparse.Namespace, scenario, vecnorm_path: Path,
               frame_stack: int, n_episodes: int = 50) -> float:
    eval_env = DummyVecEnv([make_env_fn(args, scenario, None, 0, frame_stack)])
    if vecnorm_path.exists():
        eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    totals = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = np.array([False])
        total = 0.0
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = eval_env.step(action)
            total += float(reward[0])
        totals.append(total)
    eval_env.close()
    return float(np.mean(totals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    scenario = resolve_target(args)
    frame_stack = args.frame_stack if args.frame_stack is not None else derived_frame_stack(args.observation_mode, args)
    args.frame_stack = frame_stack

    run_name = build_run_name(args)
    outdir, ckpt_dir = resolve_paths(args.outdir, args.scenario_name, run_name)

    # §8.6: the launch check runs AFTER every knob is resolved and BEFORE any artifact exists
    assert_l1_current(args, derived=_L1_DERIVED, basis=_L1_BASIS, episode_len=scenario.horizon)
    deviation_tags = derive_deviation_tags(args)
    print(f"LEVEL: {args.level} (authored: what was searched on this target)"
          + (f"   deviations from the L1 derivation: {deviation_tags}" if deviation_tags
             else "   sitting at the L1 derivation"))

    tee_console(outdir)
    write_args(args, outdir, deviation_tags)
    print(scenario)
    print(f"observation_mode={args.observation_mode}  frame_stack={frame_stack}  outdir={outdir}")

    env = build_training_env(args, outdir, scenario, frame_stack)
    model = build_model(args, env, outdir)
    record_algo_class(args, outdir, model)
    if getattr(args, "init_from", None):   # #E96: staged training
        warm_start(model, env, Path(args.init_from))
    if getattr(args, "belief_estimator", None):   # #E109: SB3's orthogonal init ran over the loaded weights — restore them
        model.policy.features_extractor.load_estimator(str(Path(args.belief_estimator).resolve()))
        print(f"belief estimator reloaded after the policy build: {args.belief_estimator}")
    if getattr(args, "freeze_extractor", False):
        for prm in model.policy.features_extractor.parameters():
            prm.requires_grad_(False)
        print(f"features extractor frozen: {sum(p.numel() for p in model.policy.features_extractor.parameters())} parameters")
    if args.group_baseline:
        assert args.group_baseline >= 2 and args.n_envs % args.group_baseline == 0, "--group-baseline K: K >= 2 must divide n_envs"
        assert args.n_steps % int(scenario.horizon) == 0, "--group-baseline: n_steps must be a multiple of the horizon"
    if args.share_worlds:   # #E98
        assert not args.group_baseline and not args.branch, "--share-worlds is the group's worlds without its advantage"
        assert args.share_worlds >= 2 and args.n_envs % args.share_worlds == 0, "--share-worlds K: K >= 2 must divide n_envs"
    callbacks =build_callbacks(args, ckpt_dir, horizon=int(scenario.horizon))

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=args.progress_bar)

    model_path = outdir / f"{args.scenario_name}_ppo.zip"
    vecnorm_path = outdir / "vecnormalize.pkl"
    model.save(str(model_path))
    env.save(str(vecnorm_path))
    env.close()

    mean_profit = smoke_eval(model, args, scenario, vecnorm_path, frame_stack)
    print(f"saved model        -> {model_path}")
    print(f"saved vecnormalize -> {vecnorm_path}")
    print(f"smoke eval (50 episodes, deterministic): mean profit {mean_profit:.3f}")
    print("NOTE: the smoke eval is not a result (§9.7) and the terminal checkpoint is not the "
          "deliverable (§8.6). Screen, then confirm:")
    print(f"  python cnv_select.py {outdir}")


if __name__ == "__main__":
    main()

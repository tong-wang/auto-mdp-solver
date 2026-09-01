"""Deployable game2048 policy (spec §12).

Wraps a trained model behind ``act(board) -> move`` so a caller needs no SB3 or
Gym knowledge — just a board.

**Observation contract.** ``act`` takes the board as a flat, row-major list of
tile values, length ``grid_size**2``, with 0 for an empty cell: cell (r, c) is
index ``r * grid_size + c``. Actual tile values, not exponents — the log2
scaling is applied here, and applying it twice would silently mis-feed the
network. The encoding then follows the ``observation_mode`` (all eight of
``_MODES``: vec, grid, onehot, thermo, onehot_s, and the ``_la``
lookahead composition of the last three) the model was
trained with:

    vec     (n_cells,)          log2 of each tile, 0 stays 0
    grid    (1, n, n)           the same values in board shape
    onehot  (n_cells+1, n, n)   plane k marks tiles of value 2**k, plane 0 empty

**This file is deliberately not standalone.** The board rules and the encoding
come from ``game2048_gym`` / ``game2048_board``, because a masked policy's
action mask is part of its I/O contract (spec §7.1) and duplicating that logic
is how the deployed policy and the trained one drift apart.

``action_mode``, ``observation_mode`` and the scenario must match training —
the run directory's ``{scenario}_{algo}_args.txt`` records all three, and the
constructor takes them explicitly rather than guessing.

VecNormalize obs stats are loaded when present, but this domain trains with
``norm_obs=False`` (tile magnitudes drift all episode), so in practice only the
reward scaler is stored and inference needs no obs transform. The code path is
kept because it is part of the contract, not because this domain exercises it.

Usage:
    from game2048_policy import Game2048Policy
    policy = Game2048Policy("results/3x3_20/<run>/best_model.zip",
                            observation_mode="onehot", grid_size=3)
    move = policy.act([0, 0, 2, 0, 4, 0, 0, 0, 2])
"""

from __future__ import annotations

from math import log2
from pathlib import Path

import numpy as np

from game2048_board import MOVES, apply_move, valid_moves

_MODES = ("vec", "grid", "onehot", "thermo", "onehot_s", "onehot_la",
          "thermo_la", "onehot_s_la")

class Game2048Policy:
    """A trained game2048 policy behind a board-in, move-out interface."""

    def __init__(
        self,
        model_path: str | Path,
        vecnorm_path: str | Path | None = None,
        observation_mode: str = "vec",
        action_mode: str = "masked",
        grid_size: int = 3,
        deterministic: bool = True,
    ) -> None:
        assert observation_mode in _MODES, \
            f"observation_mode must be one of {_MODES}, got {observation_mode!r}"
        assert action_mode in ("masked", "free"), \
            f"action_mode must be 'masked' or 'free', got {action_mode!r}"

        self.model_path = Path(model_path).resolve()
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.grid_size = grid_size
        self.n_cells = grid_size ** 2
        self.deterministic = deterministic

        if action_mode == "masked":
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(str(self.model_path), device="cpu")
        else:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(self.model_path), device="cpu")

        # obs stats (§8.3/§9.5). Default to the file beside the model; this
        # domain trains with norm_obs=False, so `_obs_rms` is normally None.
        self._obs_rms = None
        self._clip_obs = 10.0
        if vecnorm_path is None:
            candidate = self.model_path.parent / "vecnormalize.pkl"
            vecnorm_path = candidate if candidate.exists() else None
        if vecnorm_path is not None:
            import pickle
            with open(vecnorm_path, "rb") as fh:
                vecnorm = pickle.load(fh)
            if getattr(vecnorm, "norm_obs", False):
                self._obs_rms = vecnorm.obs_rms
                self._clip_obs = vecnorm.clip_obs

    # -- observation encoding (must mirror game2048_gym._get_obs) ----------

    def _encode(self, board: list[int]) -> np.ndarray:
        assert len(board) == self.n_cells, (
            f"board must be a flat list of {self.n_cells} tile values "
            f"({self.grid_size}x{self.grid_size}), got {len(board)}"
        )
        n = self.grid_size
        if self.observation_mode.endswith("_la"):
            # base encoding of the board + its four afterstates in MOVES
            # order — mirrors game2048_gym._encode_board's lookahead branch
            base = self.observation_mode.removesuffix("_la")
            blocks = [self._encode_base(board, base)]
            blocks += [self._encode_base(apply_move(board, m)[0], base)
                       for m in MOVES]
            return np.concatenate(blocks, axis=0)
        if self.observation_mode in ("onehot", "thermo", "onehot_s"):
            return self._encode_base(board, self.observation_mode)
        flat = np.array([0.0 if v == 0 else log2(int(v)) for v in board],
                        dtype=np.float32)
        return flat if self.observation_mode == "vec" else flat.reshape(1, n, n)

    def _encode_base(self, board: list[int], base: str) -> np.ndarray:
        """One block under a BASE mode — mirrors game2048_gym._encode_base."""
        if base == "onehot_s":
            n = self.grid_size
            scalar = np.array(
                [0.0 if v == 0 else log2(int(v)) / self.n_cells for v in board],
                dtype=np.float32).reshape(1, n, n)
            return np.concatenate(
                [self._encode_planes(board, "onehot"), scalar], axis=0)
        return self._encode_planes(board, base)


    def _encode_planes(self, board: list[int], base: str) -> np.ndarray:
        """One (n_cells+1, n, n) block — mirrors game2048_gym._encode_planes."""
        n = self.grid_size
        planes = np.zeros((self.n_cells + 1, n, n), dtype=np.float32)
        for i, v in enumerate(board):
            k = 0 if v == 0 else min(int(log2(int(v))), self.n_cells)
            r, c = i // n, i % n
            if base == "thermo" and k > 0:
                planes[1:k + 1, r, c] = 1.0
            else:
                planes[k, r, c] = 1.0
        return planes

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_rms is None:
            return obs
        norm = (obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + 1e-8)
        return np.clip(norm, -self._clip_obs, self._clip_obs).astype(np.float32)

    # -- the interface ------------------------------------------------------

    def act(self, board: list[int]) -> int:
        """The slide to play: 0=Up, 1=Down, 2=Right, 3=Left.

        Under ``action_mode="masked"`` the mask comes from the same
        ``valid_moves`` the MDP layer uses, so the deployed policy is offered
        exactly the actions it was trained on.
        """
        obs = self._normalize(self._encode(board))[None]
        if self.action_mode == "masked":
            mask = np.array(valid_moves(board), dtype=bool)
            if not mask.any():                 # game over: nothing to choose
                return 0
            action, _ = self.model.predict(
                obs, action_masks=mask[None], deterministic=self.deterministic
            )
        else:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(np.asarray(action).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Smoke test — replays the policy against the RAW _mdp loop, no gym stack, to
# prove the wrapper's encoding and masking reproduce eval-time behaviour
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from game2048_board import render
    from game2048_mdp import advance, init_state, open_episode
    from game2048_scenarios import SCENARIOS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("-s", "--scenario_name", default="3x3_20",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", default="vec", choices=list(_MODES))
    p.add_argument("-a", "--action_mode", default="masked",
                   choices=["masked", "free"])
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    scenario = SCENARIOS[args.scenario_name]
    policy = Game2048Policy(
        args.model_path,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        grid_size=scenario.grid_size,
    )

    print(f"{'seed':>5}  {'moves':>6}  {'score':>8}  {'max_tile':>9}")
    for seed in range(args.episodes):
        state, _ = init_state(scenario, seed)
        state = open_episode(scenario, state)
        score = 0.0
        while not state.terminated and state.period < scenario.T_cap:
            move = policy.act(list(state.board))
            state, info = advance(scenario, state, move)
            score += info["gained"]
        print(f"{seed:>5}  {state.period:>6}  {score:>8.0f}  "
              f"{max(state.board):>9}")
    print(f"\nfinal board (seed {args.episodes - 1}):")
    print(render(state.board))

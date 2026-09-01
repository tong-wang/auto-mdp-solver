"""game2048 Gymnasium wrapper.

Owns only Gym-facing concerns — spaces, observation encoding, reward
assembly, termination flags. The dynamics live in ``game2048_mdp.py`` and the
board rules in ``game2048_board.py``.

observation_mode (all encode the SAME `board` state variable):
  - "vec"    : (n_cells,) float32, log2-scaled tile values (0 stays 0) -> MlpPolicy
  - "grid"   : (1, n, n) float32, the same values in board shape      -> CnnPolicy
  - "onehot" : (n_cells+1, n, n) float32 binary planes; plane k marks tiles of
               value 2^k, plane 0 marks empty cells                   -> CnnPolicy
  - "thermo" : (n_cells+1, n, n) float32 binary planes, thermometer-coded:
               plane 0 marks empty cells (as in onehot), plane k>=1 marks
               tiles of value >= 2^k, so ordering between two cells is a
               LINEAR functional of the encoding (the thermometer arm, #E24;
               same shape as onehot — a pure encoding swap)          -> CnnPolicy
  - "onehot_s": (n_cells+2, n, n) float32 — the onehot planes plus ONE
               scalar plane carrying log2(tile)/n_cells in [0, 1]
               (the scalar-plane arms, #E25/#E26). The augment-not-
               substitute answer to thermometer's failure: the identity
               basis (equality = the merge predicate) keeps its orthogonal
               planes, and ordering between cells becomes a difference on
               a single well-conditioned channel. Normalized by n_cells so
               the plane cannot dominate input energy as boards grow
               (norm_obs=False in this domain)                       -> CnnPolicy
  - "onehot_la" / "thermo_la" / "onehot_s_la" : (5*W, n, n) float32, where
               W is the width of the named base encoding (n_cells+1 for
               onehot/thermo, n_cells+2 for onehot_s) — the base encoding of
               the board followed by the base encoding of the
               four post-merge pre-spawn AFTERSTATES, one per slide in MOVES
               order (the lookahead arms, #E24/#E25). The suffix composes
               with any base mode above it, so "onehot_s_la" is the scalar
               plane and the lookahead together. The afterstate is the
               deterministic half of the transition (the same apply_move the
               action mask already uses), so this integrates one step of the
               game's own dynamics into the observation; an illegal slide
               contributes the unchanged board, consistent with the mask.
               The blocks are bit-exact against ``afterstate_candidates``,
               which is what ``game2048_enclook_gate.py`` asserts — one
               engine, one encoder, no drift                         -> CnnPolicy

action_mode (spec §7.1 — the mdp layer accepts all four slides either way):
  - "masked" : illegal slides are removed from the policy's support via
               ``action_masks()``; use MaskablePPO
  - "free"   : all four slides always offered; an illegal one is a no-op that
               still consumes a move. ``action_masks()`` reports all-True so
               the env stays usable under either algorithm.

reward_mode:
  - "score"         : merge points earned this move — the objective itself
  - "score_penalty" : minus ``invalid_penalty`` per illegal slide. Shaping for
                      the free action mode, where the cost of stalling
                      otherwise arrives only through the T_cap truncation.
                      Evaluation always scores pure merge points regardless.
  - "log2_score"    : ``log2(1 + gained)`` — the move's haul compressed to
                      roughly [0, 15]. Motivation (user, 2026-08-15): raw
                      score spans orders of magnitude and arrives in rare
                      huge spikes, which is heavy-tailed fuel for the
                      advantage normalizer. NOTE this is
                      log of the move TOTAL, not the sum of per-merge log2
                      payments — ``info`` carries only the aggregate
                      ``gained``, and decomposing it would be an mdp/IR
                      change, not a gym-side training lever. The two differ
                      only on multi-merge moves, where log-of-sum tracks the
                      LARGEST merge (log2 of a sum is within +1 of log2 of
                      its max).
  - "empty"         : the number of empty cells AFTER the transition
                      (post-spawn), in [0, n_cells-1]; 0 at game over. The
                      densest, most stationary signal available here — it
                      neither grows over the episode nor spikes. Its return
                      decomposes as (N - o_0)*T - T(T+1)/2 + sum_t M_t with
                      M_t the cumulative merges, i.e. survive long AND merge
                      early (a merge at time s is paid once per remaining
                      step). Locally it is merge-size blind, but 2048's
                      conservation law (a slide conserves the tile sum; only
                      spawns add) means no junk-farming policy exists —
                      holding any occupancy level forces climbing the whole
                      tile hierarchy, so the compact "few big tiles + empty
                      space" structure emerges as the optimum rather than
                      being hand-coded. Under "free" an invalid move scores
                      0 rather than its unchanged empty count, which would
                      otherwise pay the agent to stall; intended for masked.

  Evaluation is UNAFFECTED by all of these: ``game2048_benchmark_common.
  evaluate`` builds its own env with reward_mode="score" and reports
  ``env.total_score``, so a training-time reward cannot reach a leaderboard
  number. ``total_score`` (true merge points) and ``total_reward`` (whatever
  this mode paid) are tracked separately on the env for the same reason.

Termination follows the IR: ``terminated`` on game over (the problem's own
absorbing state), ``truncated`` at T_cap (an external move cap).
"""

from __future__ import annotations

import logging
from math import log2
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from game2048_board import MOVES, apply_move, side
from game2048_mdp import Game2048State, advance, init_state, open_episode, valid_actions
from game2048_scenarios import Game2048Scenario

ScenarioSource = Game2048Scenario | Callable[[int], Game2048Scenario]

OBSERVATION_MODES = ("vec", "grid", "onehot", "thermo", "onehot_s",
                     "onehot_la", "thermo_la", "onehot_s_la")

ACTION_MODES = ("masked", "free")
REWARD_MODES = ("score", "score_penalty", "log2_score", "empty")


def _log2_tiles(board) -> np.ndarray:
    """Tile values as exponents: 0 -> 0, 2 -> 1, 4 -> 2, ... Keeps the feature
    range tiny and linear in the quantity that actually matters."""
    return np.array(
        [0.0 if v == 0 else log2(int(v)) for v in board], dtype=np.float32
    )


class Game2048Env(gym.Env):
    """Gymnasium wrapper over the game2048 MDP."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioSource,
        observation_mode: str = "vec",
        action_mode: str = "masked",
        reward_mode: str = "score",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()

        assert observation_mode in OBSERVATION_MODES, \
            f"observation_mode must be one of {OBSERVATION_MODES}, got {observation_mode!r}."
        assert action_mode in ACTION_MODES, \
            f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}."
        assert reward_mode in REWARD_MODES, \
            f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}."

        self.scenario = scenario
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.reward_mode = reward_mode

        # spaces are sized from the template scenario; a grid sampler must not
        # vary grid_size (the observation shape would change under the policy)
        template = scenario(0) if callable(scenario) else scenario
        self.n = template.grid_size
        self.n_cells = template.n_cells
        self.n_planes = self.n_cells + 1

        self._current_scenario: Game2048Scenario
        self._state: Game2048State
        self._info: dict
        self._episode_seed: int
        self._step: int
        self.total_reward: float
        self.total_score: float
        self.action_space = gym.spaces.Discrete(len(MOVES))
        self.observation_space = self._build_observation_space()

        self.logger_e = logging.getLogger("episode_logger")
        self.logger_s = logging.getLogger("step_logger")
        if logger_filename is not None:
            self._init_logger(logger_filename)

    # -- spaces / observations ---------------------------------------------

    def _build_observation_space(self) -> gym.spaces.Box:
        if self.observation_mode == "vec":
            # log2 exponent; 2**n_cells is the largest tile an n x n board can hold
            return gym.spaces.Box(
                low=0.0, high=float(self.n_cells),
                shape=(self.n_cells,), dtype=np.float32,
            )
        if self.observation_mode == "grid":
            return gym.spaces.Box(
                low=0.0, high=float(self.n_cells),
                shape=(1, self.n, self.n), dtype=np.float32,
            )
        base = self.observation_mode.removesuffix("_la")
        n_blocks = 1 + len(MOVES) if self.observation_mode.endswith("_la") else 1
        block = self.n_planes + 1 if base == "onehot_s" else self.n_planes
        return gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(n_blocks * block, self.n, self.n), dtype=np.float32,
        )

    def _encode_board(self, board) -> np.ndarray:
        """Encode an arbitrary board under this env's observation_mode.

        Split out of ``_get_obs`` so the afterstate candidates go through
        the SAME encoder as the observation itself —
        one implementation, no drift.
        """
        if self.observation_mode == "vec":
            return _log2_tiles(board)
        if self.observation_mode == "grid":
            return _log2_tiles(board).reshape(1, self.n, self.n)
        if self.observation_mode.endswith("_la"):
            # base encoding of the board, then of its four afterstates in
            # MOVES order — the same apply_move the action mask relies on.
            # General over any board (an afterstate's own encoding under a
            # _la mode is that board plus ITS afterstates), so
            # afterstate_candidates keeps the one-encoder-no-drift property.
            base = self.observation_mode.removesuffix("_la")
            blocks = [self._encode_base(board, base)]
            blocks += [self._encode_base(apply_move(board, m)[0], base)
                       for m in MOVES]
            return np.concatenate(blocks, axis=0)
        return self._encode_base(board, self.observation_mode)

    def _encode_base(self, board, base: str) -> np.ndarray:
        """One block under a BASE (non-_la) plane mode, incl. onehot_s."""
        if base == "onehot_s":
            scalar = (_log2_tiles(board) / self.n_cells).reshape(
                1, self.n, self.n).astype(np.float32)
            return np.concatenate(
                [self._encode_planes(board, "onehot"), scalar], axis=0)
        return self._encode_planes(board, base)

    def _encode_planes(self, board, base: str) -> np.ndarray:
        """One (n_planes, n, n) block under a plane encoding.

        "onehot": plane k marks tiles of value 2^k, plane 0 empty cells.
        "thermo": plane 0 empty cells; plane k>=1 marks tiles >= 2^k, so the
        planes of a tile are nested and exp_a - exp_b is a sum of plane
        differences — ordering made linear (#E24).
        """
        planes = np.zeros((self.n_planes, self.n, self.n), dtype=np.float32)
        for i, v in enumerate(board):
            k = 0 if v == 0 else min(int(log2(int(v))), self.n_planes - 1)
            r, c = i // self.n, i % self.n
            if base == "thermo" and k > 0:
                planes[1:k + 1, r, c] = 1.0
            else:
                planes[k, r, c] = 1.0
        return planes

    def _get_obs(self) -> np.ndarray:
        return self._encode_board(self._state.board)

    # -- afterstate candidates: the one-encoder source the _la modes are
    #    gate-checked against (game2048_enclook_gate.py)

    def afterstate_candidates(self) -> tuple[np.ndarray, np.ndarray]:
        """Per action: (encoded afterstate, raw merge gain) for the CURRENT
        board — the board after ``M`` (slide+merge), before ``S`` (the spawn
        draw). Deterministic given (board, action); the same engine call the
        action mask already uses, so no new knowledge enters here.

        Returns ``(cand_obs, cand_gains)`` with shapes ``(4, *obs_shape)``
        and ``(4,)``. An illegal slide leaves the board unchanged (its
        "afterstate" is the board itself, gain 0); under masking the policy
        puts zero probability there, so those rows never reach the baseline.

        Gains are RAW merge points — the caller owns any reward
        normalization, exactly as it owns it for ``step()``'s reward.
        """
        outs = [apply_move(self._state.board, m) for m in MOVES]
        cand_obs = np.stack([self._encode_board(b) for b, _, _ in outs])
        cand_gains = np.array([g for _, g, _ in outs], dtype=np.float32)
        return cand_obs.astype(np.float32), cand_gains

    # -- action masking (spec §7.1) ----------------------------------------

    def action_masks(self) -> np.ndarray:
        """Slides MaskablePPO may sample.

        In "free" mode nothing is masked. In "masked" mode the mask comes from
        the mdp layer's ``valid_actions``; on a game-over board no slide is
        legal, so all-True is returned rather than an empty support — the
        episode has already ended and the value is never acted on.
        """
        if self.action_mode == "free":
            return np.ones(len(MOVES), dtype=bool)
        mask = np.array(valid_actions(self._state), dtype=bool)
        if not mask.any():
            return np.ones(len(MOVES), dtype=bool)
        return mask

    # -- episode -----------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        # per-env stream (spec §7): global np.random would replay identical
        # episode-seed sequences across SubprocVecEnv workers
        self._episode_seed = (
            seed if seed is not None
            else int(self.np_random.integers(0, 2_147_483_647))
        )
        self._current_scenario = (
            self.scenario(self._episode_seed) if callable(self.scenario)
            else self.scenario
        )
        assert self._current_scenario.grid_size == self.n, (
            "grid_size must not vary across a sampler: the observation shape "
            f"is fixed at {self.n}x{self.n}"
        )

        state, self._info = init_state(self._current_scenario, self._episode_seed)
        # place the opening tile now, so the first observation is the board the
        # first slide lands on and the action mask is meaningful (see
        # game2048_mdp.open_episode — the trajectory is unchanged either way)
        self._state = open_episode(self._current_scenario, state)
        self._step = 0
        self.total_reward = 0.0
        self.total_score = 0.0
        return self._get_obs(), self._info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        move = int(action.item() if isinstance(action, np.ndarray) else action)

        self._state, self._info = advance(
            self._current_scenario, self._state, move)

        gained = float(self._info["gained"])
        if self.reward_mode == "log2_score":
            reward = float(log2(1.0 + gained))
        elif self.reward_mode == "empty":
            # deterministic given (board, move) exactly like score: the spawn
            # position varies with the draw but the COUNT always falls by one
            reward = float(self._state.board.count(0)) \
                if self._info["move_valid"] else 0.0
        else:
            reward = gained
            if self.reward_mode == "score_penalty":
                reward -= self._current_scenario.invalid_penalty * (
                    1 - self._info["move_valid"]
                )

        self.total_score += gained
        self.total_reward += reward
        self._step += 1

        terminated = bool(self._state.terminated)
        truncated = (not terminated) and self._state.period >= self._current_scenario.T_cap

        self.logger_s.info(
            f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
            f"{move}\t{self._info['move_valid']}\t{gained:.1f}\t"
            f"{self._state.spawn_count}\t{max(self._state.board)}\t"
            f"{reward:.4f}\t{self.total_reward:.4f}"
        )
        if terminated or truncated:
            self.logger_e.info(
                f"{self._episode_seed}\t{self._step}\t"
                f"{self._current_scenario.grid_size}\t"
                f"{self._current_scenario.spawn_value.prob_4}\t"
                f"{self._state.spawn_count}\t{max(self._state.board)}\t"
                f"{sum(self._state.board)}\t{self.total_score:.1f}\t"
                f"{self.total_reward:.4f}\t{int(terminated)}"
            )

        return self._get_obs(), reward, terminated, truncated, self._info

    def _init_logger(self, filename: str) -> None:
        fmt = logging.Formatter("%(asctime)s\t%(message)s")
        for logger, suffix in ((self.logger_e, "episode"), (self.logger_s, "step")):
            handler = logging.FileHandler(f"{filename}_{suffix}.log")
            handler.setFormatter(fmt)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Module self-test — the Stage-2 gate: random episodes in every mode, with
# observation_space.contains(obs) asserted at every step
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from game2048_scenarios import SCENARIOS

    sc = SCENARIOS["3x3_20"]
    rng = np.random.default_rng(0)

    for obs_mode in OBSERVATION_MODES:
        for act_mode in ACTION_MODES:
            for rew_mode in REWARD_MODES:
                env = Game2048Env(sc, observation_mode=obs_mode,
                                  action_mode=act_mode, reward_mode=rew_mode)
                obs, _ = env.reset(seed=11)
                assert env.observation_space.contains(obs), \
                    f"reset obs outside space in {obs_mode}/{act_mode}"
                assert sum(env._state.board) > 0, "reset must show the opening tile"

                steps = 0
                while True:
                    mask = env.action_masks()
                    legal = [a for a, ok in enumerate(mask) if ok]
                    obs, reward, terminated, truncated, info = env.step(
                        int(rng.choice(legal))
                    )
                    steps += 1
                    assert env.observation_space.contains(obs), \
                        f"step obs outside space in {obs_mode}/{act_mode}"
                    assert np.isfinite(reward), "reward must be finite"
                    if terminated or truncated:
                        break
                print(f"{obs_mode:7s} {act_mode:7s} {rew_mode:14s} "
                      f"obs{tuple(env.observation_space.shape)} steps={steps:3d} "
                      f"score={env.total_score:6.0f} reward={env.total_reward:8.1f} "
                      f"max_tile={max(env._state.board):5d} "
                      f"{'terminated' if terminated else 'truncated'}")

    # the free mode must actually admit illegal slides (no-ops that cost a move)
    env = Game2048Env(sc, action_mode="free")
    env.reset(seed=3)
    noops = 0
    for _ in range(40):
        _, _, term, trunc, info = env.step(0)      # hammer Up
        noops += 1 - info["move_valid"]
        if term or trunc:
            break
    assert noops > 0, "free mode should let an illegal slide through as a no-op"
    print(f"\nfree mode: {noops} illegal slides absorbed as no-ops")
    print("game2048_gym: all modes ok")

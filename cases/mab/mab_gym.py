"""Gymnasium wrapper for the multi-armed bandit simulator."""

from __future__ import annotations

from typing import Any
import logging
import numpy as np
import gymnasium as gym

from mab_scenarios import MabScenario, ScenarioSource
from mab_mdp import MabState, init_state, advance
from mab_bayes import bayes_post_mean, bayes_post_sd


class MabEnv(gym.Env):
    """Standard Gymnasium wrapper around the domain-level bandit simulator.

    observation_mode (both are lossless belief-state parametrizations):
    - "stats": [pulls[0..K-1], payouts[0..K-1], time_to_go]
    - "bayes": [post_mean[0..K-1], post_sd[0..K-1], time_to_go]
               exact conjugate posterior of each arm's true mean under the
               branch prior (Beta(1,1) / N(0,1) with known sigma) — see
               mab_bayes.py, shared with the IR's expr_builtins.

    Observations are taken at the pre-pull state (spec: observation_point).
    The hidden arm means never appear in any mode.

    Action ("arm" mode): Discrete(K) — index of the arm to pull; every arm
    is always feasible (no masking needed).

    Reward ("payout" mode): the realized pull payout (sense=maximize).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioSource,
        observation_mode: str = "stats",
        action_mode: str = "arm",
        reward_mode: str = "payout",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()

        assert observation_mode in {"stats", "bayes", "bayes_h"}, \
            (f"observation_mode must be 'stats', 'bayes' or 'bayes_h', "
             f"got '{observation_mode}'.")
        assert action_mode in {"arm"}, \
            f"action_mode must be 'arm', got '{action_mode}'."
        assert reward_mode in {"payout"}, \
            f"reward_mode must be 'payout', got '{reward_mode}'."

        # scenario source: a fixed scenario, or a sampler resolved per episode
        # in reset(). Space building below reads only family-level attributes
        # (horizon, n_arms, payout bounds), which samplers expose under the
        # same names as a concrete scenario.
        self.scenario = scenario
        self._scenario_ep: MabScenario | None = None
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.reward_mode = reward_mode

        self.n_arms  = int(scenario.n_arms)
        self.horizon = int(scenario.horizon)
        # a source delegates `payout` to its template, so a source and a
        # concrete scenario read the same way (spec §5.2 family-level attrs)
        self._is_gauss = not scenario.payout.is_discrete

        self._state: MabState
        self._info: dict
        self._episode_seed: int
        self._step: int
        self.total_reward: float

        self.action_space = gym.spaces.Discrete(self.n_arms)
        self.observation_space = self._build_observation_space()

        self.logger_e = logging.getLogger("episode_logger")
        self.logger_s = logging.getLogger("step_logger")
        if logger_filename is not None:
            self._init_logger(logger_filename)

    def _init_logger(self, filename: str) -> None:
        fmt = logging.Formatter("%(asctime)s\t%(message)s")

        self.logger_e.handlers.clear()
        self.logger_e.setLevel(logging.DEBUG)
        fh_e = logging.FileHandler(f"{filename}_episode.log")
        fh_e.setFormatter(fmt)
        self.logger_e.addHandler(fh_e)

        self.logger_s.handlers.clear()
        self.logger_s.setLevel(logging.DEBUG)
        fh_s = logging.FileHandler(f"{filename}_step.log")
        fh_s.setFormatter(fmt)
        self.logger_s.addHandler(fh_s)

    def _build_observation_space(self) -> gym.spaces.Box:
        K, T = self.n_arms, self.horizon
        if self.observation_mode == "stats":
            if self._is_gauss:
                pay_lo, pay_hi = -np.inf, np.inf
            else:
                pay_lo, pay_hi = 0.0, float(T)
            low  = [0.0] * K + [pay_lo] * K + [0.0]
            high = [float(T)] * K + [pay_hi] * K + [float(T)]
        else:  # bayes / bayes_h
            if self._is_gauss:
                mean_lo, mean_hi, sd_hi = -np.inf, np.inf, 1.0
            else:
                mean_lo, mean_hi, sd_hi = 0.0, 1.0, 0.5
            low  = [mean_lo] * K + [0.0] * K + [0.0]
            high = [mean_hi] * K + [sd_hi] * K + [float(T)]
            if self.observation_mode == "bayes_h":
                # the episode's horizon, bounded by the family maximum (a grid
                # exposes its widest cell, spec §5.6 mirroring rule)
                low, high = low + [0.0], high + [float(T)]
        return gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        s = self._state
        # the EPISODE's horizon, not the construction-time one. Identical for a
        # fixed scenario; they diverge under a horizon-varying sampler (#E36's
        # generalist grid), where `self.horizon` is the family maximum used to
        # size the space and would make time_to_go wrong in every cell but the
        # longest.
        ep_T = float(getattr(self._scenario_ep, "horizon", self.horizon))
        time_to_go = ep_T - s.period
        if self.observation_mode == "stats":
            obs = (
                [float(n) for n in s.pulls]
                + [float(x) for x in s.payouts]
                + [time_to_go]
            )
        else:  # bayes / bayes_h
            obs = (
                bayes_post_mean(s.pulls, s.payouts, self._is_gauss)
                + bayes_post_sd(s.pulls, s.payouts, self._is_gauss)
                + [time_to_go]
            )
            if self.observation_mode == "bayes_h":
                obs = obs + [ep_T]
        return np.array(obs, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        if seed is not None:
            self._episode_seed = seed
        else:
            # per-env stream (spec §7): global np.random would replay
            # identical episode-seed sequences across SubprocVecEnv workers
            self._episode_seed = int(self.np_random.integers(0, 2_147_483_647))

        self._scenario_ep = (
            self.scenario(self._episode_seed)
            if callable(self.scenario)
            else self.scenario
        )
        self._state, self._info = init_state(
            scenario=self._scenario_ep,
            episode_seed=self._episode_seed,
        )
        self._step = 0
        self.total_reward = 0.0

        return self._get_obs(), self._info

    def step(
        self,
        action: np.ndarray | int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        arm = int(action)

        self._state, self._info = advance(self._scenario_ep, self._state, arm)

        obs = self._get_obs()
        reward = float(self._info["payoff"]["total"])
        terminated = self._state.terminated
        truncated = False

        self.total_reward += reward
        self._step += 1

        self.logger_s.info(
            f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
            f"{arm}\t{self._info['payout']:.4f}\t{reward:.4f}\t"
            f"{self.total_reward:.4f}"
        )

        if terminated:
            self.logger_e.info(
                f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
                f"{self.total_reward:.4f}\t{self._info['opt_mean']:.4f}"
            )

        return obs, reward, terminated, truncated, self._info


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from mab_scenarios import SCENARIOS

    for scenario_name, source in SCENARIOS.items():
        for mode in ("stats", "bayes"):
            env = MabEnv(scenario=source, observation_mode=mode)
            obs, info = env.reset(seed=7)
            assert env.observation_space.contains(obs), (scenario_name, mode, obs)
            rng = np.random.default_rng(0)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                action = int(rng.integers(env.n_arms))
                obs, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs), \
                    (scenario_name, mode, steps, obs)
                steps += 1
            print(
                f"{scenario_name:18s} obs={mode:5s}  steps={steps}  "
                f"total_reward={env.total_reward:9.2f}  "
                f"opt_total={env.horizon * info['opt_mean']:9.2f}"
            )
    print("all obs/action modes OK (observation_space.contains asserted each step)")

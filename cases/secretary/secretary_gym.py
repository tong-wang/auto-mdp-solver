"""Gymnasium wrapper for the classical secretary simulator."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

import secretary_mdp as mdp
from secretary_mdp import SecretaryState
from secretary_scenarios import SecretaryScenario


class SecretaryEnv(gym.Env):
    """Single-agent wrapper exposing only the classical relative-rank signal.

    observation_mode ``relative`` renders, in declared order,
    ``[time_to_go, relative_rank]``. Hidden absolute ranks never appear.
    Action mode ``accept`` is ``Discrete(2)``: reject 0 or accept 1.
    Reward mode ``success`` returns the terminal best-candidate indicator.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: SecretaryScenario,
        observation_mode: str = "relative",
        action_mode: str = "accept",
        reward_mode: str = "success",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()
        assert observation_mode == "relative"
        assert action_mode == "accept"
        assert reward_mode == "success"
        self.scenario = scenario
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.reward_mode = reward_mode
        self.n_candidates = int(scenario.n_candidates)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array(
                [float(self.n_candidates), float(self.n_candidates)],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self._state: SecretaryState
        self._info: dict
        self._episode_seed: int
        self.total_reward = 0.0
        self._logger_filename = logger_filename

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                float(self.scenario.n_candidates - self._state.period),
                float(self._state.relative_rank),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            self._episode_seed = int(self.np_random.integers(0, 2_147_483_647))
        self._state, self._info = mdp.init_state(self.scenario, self._episode_seed)
        self.total_reward = 0.0
        obs = self._get_obs()
        assert self.observation_space.contains(obs)
        return obs, self._info

    def step(
        self, action: np.ndarray | int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._state, self._info = mdp.advance(
            self.scenario, self._state, int(action)
        )
        reward = float(self._info["outcome"]["total"])
        self.total_reward += reward
        obs = self._get_obs()
        terminated = self._state.terminated
        truncated = False
        assert self.observation_space.contains(obs)
        return obs, reward, terminated, truncated, self._info


if __name__ == "__main__":
    from secretary_scenarios import SCENARIOS

    for scenario_name, scenario in SCENARIOS.items():
        env = SecretaryEnv(scenario)
        rng = np.random.default_rng(0)
        for episode_seed in range(5):
            obs, _ = env.reset(seed=episode_seed)
            terminated = truncated = False
            while not (terminated or truncated):
                obs, _, terminated, truncated, _ = env.step(
                    int(rng.integers(0, 2))
                )
                assert env.observation_space.contains(obs)
        print(f"{scenario_name}: all obs/action modes OK")

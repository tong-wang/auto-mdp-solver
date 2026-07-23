"""FNV Gymnasium wrapper."""

from __future__ import annotations

from typing import Any, Callable, Union
import logging
import numpy as np
import gymnasium as gym

from fnv_scenarios import FnvScenario
from fnv_mdp import FnvState, init_state, advance

# a scenario source: a concrete scenario, or a callable episode_seed -> scenario
# (e.g. a grid-derived sampler; spec §5.6). The gym only requires callability
# and the family-level attributes both forms expose.
ScenarioSource = Union[FnvScenario, Callable[[int], FnvScenario]]


class FnvEnv(gym.Env):
    """FNV Gymnasium wrapper.

    action_mode:
    - "box": continuous order quantity in [0, mu + 5*stdev]

    observation_mode:
    - "vec": [stdev, T, lamb, period, inventory, information]

    reward_mode:
    - "profit": -ordering_cost each step; +revenue at terminal
    - "regret":  same but subtracts profit_max at terminal

    Notes:
    - The core simulator is the source of truth for dynamics.
    - This wrapper handles only Gym-facing concerns: spaces, formatting, reset/step.
    - If scenario is callable (e.g. a grid-derived sampler), parameters are
      re-drawn each episode; spaces use the family-level bounds it exposes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioSource,
        action_mode: str = "box",
        observation_mode: str = "vec",
        reward_mode: str = "profit",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()

        assert observation_mode in {"vec"}, \
            f"observation_mode must be 'vec', got '{observation_mode}'."
        assert action_mode in {"box"}, \
            f"action_mode must be 'box', got '{action_mode}'."
        assert reward_mode in {"profit", "regret"}, \
            f"reward_mode must be 'profit' or 'regret', got '{reward_mode}'."

        self.scenario         = scenario
        self.observation_mode = observation_mode
        self.action_mode      = action_mode
        self.reward_mode      = reward_mode

        self._current_scenario: FnvScenario
        self._state:            FnvState
        self._info:             dict
        self._episode_seed:     int
        self._step:             int
        self.total_reward:      float

        self.action_space      = self._build_action_space()
        self.observation_space = self._build_observation_space()

        self.logger_e = logging.getLogger("episode_logger")
        self.logger_s = logging.getLogger("step_logger")
        if logger_filename is not None:
            self._init_logger(logger_filename)

    def _init_logger(self, filename: str) -> None:
        fmt = logging.Formatter("%(asctime)s\t%(message)s")

        self.logger_e.handlers.clear()
        self.logger_e.setLevel(logging.DEBUG)
        ch_e = logging.StreamHandler()
        ch_e.setFormatter(fmt)
        self.logger_e.addHandler(ch_e)
        fh_e = logging.FileHandler(f"{filename}_episode.log")
        fh_e.setFormatter(fmt)
        self.logger_e.addHandler(fh_e)

        self.logger_s.handlers.clear()
        self.logger_s.setLevel(logging.DEBUG)
        ch_s = logging.StreamHandler()
        ch_s.setFormatter(fmt)
        self.logger_s.addHandler(ch_s)
        fh_s = logging.FileHandler(f"{filename}_step.log")
        fh_s.setFormatter(fmt)
        self.logger_s.addHandler(fh_s)

    def _build_action_space(self) -> gym.spaces.Box:
        # family-level bounds: concrete scenarios expose their own stdev; a
        # grid-derived sampler exposes the widest cell's stdev (bound aggregate).
        stdev = self.scenario.stdev
        if self.scenario.mmfe_mode == "additive":
            high = self.scenario.mu + 5.0 * stdev
        else:
            high = float(np.exp(self.scenario.mu + 5.0 * stdev))
        return gym.spaces.Box(
            low=np.float32(0.0), high=np.float32(high), shape=(1,), dtype=np.float32,
        )

    def _build_observation_space(self) -> gym.spaces.Box:
        N = self.scenario.N
        low  = np.array([0.0, 0.0, 0.0, 1.0,     0.0,    -np.inf], dtype=np.float32)
        high = np.array([np.inf, 1.0, np.inf, float(N), np.inf, np.inf], dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        sc, s = self._current_scenario, self._state
        return np.array(
            [sc.stdev, sc.T, sc.lamb, float(s.period), s.inventory, s.information],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        self._episode_seed = seed if seed is not None else int(np.random.randint(0, 2_147_483_647))

        self._current_scenario = (
            self.scenario(self._episode_seed) if callable(self.scenario)
            else self.scenario
        )

        self._state, self._info = init_state(self._current_scenario, self._episode_seed)
        self._step        = 0
        self.total_reward = 0.0

        return self._get_obs(), self._info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        decision = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        decision = max(0.0, decision)

        self._state, self._info = advance(self._current_scenario, self._state, decision)
        terminated = self._state.terminated

        reward = -self._info["cost"]
        if terminated:
            reward += self._info["revenue"]
            if self.reward_mode == "regret":
                reward -= self._info["profit_max"]

        self.total_reward += reward
        self._step        += 1

        self.logger_s.info(
            f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
            f"{decision:.4f}\t{self._info['cost']:.4f}\t"
            f"{self._state.inventory:.4f}\t{self._state.information:.4f}\t"
            f"{reward:.4f}\t{self.total_reward:.4f}"
        )

        if terminated:
            self.logger_e.info(
                f"{self._episode_seed}\t{self._step}\t"
                f"{self._current_scenario.stdev}\t{self._current_scenario.T}\t"
                f"{self._current_scenario.lamb}\t"
                f"{self._info['demand']:.4f}\t{self._info['profit']:.4f}\t"
                f"{self._info['profit_max']:.4f}\t{self.total_reward:.4f}"
            )

        return self._get_obs(), reward, terminated, False, self._info


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fnv_scenarios import scenario_simple

    for sc in (scenario_simple,):
        label = sc.scenario_name if isinstance(sc, FnvScenario) else "sampler"
        print(f"\n=== {label} ===")
        env = FnvEnv(scenario=sc)
        print("action_space :", env.action_space)
        print("obs_space    :", env.observation_space)

        obs, info = env.reset(seed=42)
        print(f"reset obs: {obs}")

        terminated = False
        while not terminated:
            obs, reward, terminated, _, info = env.step(np.array([0.3], dtype=np.float32))
            print(f"  obs={obs}  reward={reward:.4f}")
        print(f"total_reward: {env.total_reward:.4f}")

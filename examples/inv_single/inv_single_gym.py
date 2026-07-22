"""Gymnasium wrapper for the single-echelon inventory simulator."""

from __future__ import annotations

from typing import Any
import logging
import warnings
import numpy as np
import gymnasium as gym

from inv_single_scenarios import InvSingleScenario, ScenarioSource
from inv_single_mdp import (
    InvSingleState,
    init_state,
    advance1,
    advance2,
)


class InvSingleEnv(gym.Env):
    """Standard Gymnasium wrapper around the domain-level inventory simulator.

    observation_mode:
    - "vec":      [period, inventory, pipeline[0], ..., pipeline[L]]
    - "vec_d":    same as "vec" plus demand from the most recent D event
    - "vec_d_ip": [period, inventory_position, demand]
                  IP = inventory + sum(pipeline); compact sufficient statistic
                  for base-stock policies under positive leadtime

    All observations are taken at the pre-order state (after advance1).
    pipeline length = scenario.leadtime.max() + 1.

    Action (float32 scalar in a Box of shape (1,)):
        order quantity q >= 0; rounded to nearest integer internally.

    Reward:
        -total_cost per period.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioSource,
        observation_mode: str = "vec",
        action_mode: str = "continuous",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()

        assert observation_mode in {
            "vec",
            "vec_d",
            "vec_d_ip",
        }, f"observation_mode must be 'vec', 'vec_d', or 'vec_d_ip', got '{observation_mode}'."

        assert action_mode in {
            "continuous",
            "discrete",
        }, f"action_mode must be 'continuous' or 'discrete', got '{action_mode}'."

        if action_mode == "discrete" and not scenario.demand.is_discrete:
            warnings.warn(
                f"action_mode='discrete' with continuous demand ({type(scenario.demand).__name__}): "
                "discrete actions may be too coarse.",
                stacklevel=2,
            )
        if action_mode == "continuous" and scenario.demand.is_discrete:
            warnings.warn(
                f"action_mode='continuous' with discrete demand ({type(scenario.demand).__name__}): "
                "consider action_mode='discrete' for a more natural action space.",
                stacklevel=2,
            )

        # scenario source: a fixed scenario, or a sampler resolved per episode
        # in reset(). Space building below reads only family-level attributes
        # (horizon, leadtime, allow_backlog, demand bounds), which samplers
        # expose under the same names as a concrete scenario.
        self.scenario = scenario
        self._scenario_ep: InvSingleScenario | None = None
        self.observation_mode = observation_mode
        self.action_mode = action_mode

        self._state: InvSingleState
        self._info: dict
        self._episode_seed: int
        self._step: int
        self.total_reward: float

        self.action_space = self._build_action_space()
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

    def _build_action_space(self) -> gym.spaces.Box | gym.spaces.Discrete:
        high = (
            2.0
            * self.scenario.demand.max()
            * max(1, self.scenario.leadtime.max())
        )
        if self.action_mode == "discrete":
            return gym.spaces.Discrete(int(high) + 1)
        return gym.spaces.Box(
            low=np.float32(0.0),
            high=np.float32(high),
            shape=(1,),
            dtype=np.float32,
        )

    def _build_observation_space(self) -> gym.spaces.Box:
        inv_low = 0.0 if not self.scenario.allow_backlog else -np.inf

        if self.observation_mode == "vec_d_ip":
            low  = [0.0, inv_low, 0.0]
            high = [float(self.scenario.horizon), np.inf, self.scenario.demand.max()]
        else:
            L = self.scenario.leadtime.max() + 1  # pipeline length = max + 1
            low  = [0.0, inv_low] + [0.0] * L
            high = [float(self.scenario.horizon), np.inf] + [np.inf] * L
            if self.observation_mode == "vec_d":
                low  += [0.0]
                high += [self.scenario.demand.max()]

        return gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        if self.observation_mode == "vec_d_ip":
            ip = float(self._state.inventory) + sum(float(x) for x in self._state.pipeline)
            obs = [float(self._state.period), ip, float(self._state.demand)]
        else:
            obs = [float(self._state.period), float(self._state.inventory)] + [
                float(x) for x in self._state.pipeline
            ]
            if self.observation_mode == "vec_d":
                obs += [float(self._state.demand)]
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
            self._episode_seed = int(np.random.randint(0, 2_147_483_647))

        self._scenario_ep = (
            self.scenario(self._episode_seed)
            if callable(self.scenario)
            else self.scenario
        )
        state, self._info = init_state(
            scenario=self._scenario_ep,
            episode_seed=self._episode_seed,
        )
        self._state = advance1(self._scenario_ep, state)
        self._step = 0
        self.total_reward = 0.0

        return self._get_obs(), self._info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.action_mode == "discrete":
            order = int(action)
        else:
            order = max(0.0, float(action[0]))

        new_state, self._info = advance2(self._scenario_ep, self._state, order)
        self._state = (
            advance1(self._scenario_ep, new_state)
            if not new_state.terminated
            else new_state
        )

        obs = self._get_obs()
        reward = -float(self._info["cost"]["total"])
        terminated = new_state.terminated
        truncated = False

        self.total_reward += reward
        self._step += 1

        # inventory/pipeline are physical state, read off the completed-period
        # state (new_state); demand is an info diagnostic. self._state has since
        # advanced to the next period's pre-order state, so use new_state here.
        self.logger_s.info(
            f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
            f"{order}\t{self._info['demand']}\t{new_state.inventory}\t"
            f"{list(new_state.pipeline)}\t{reward:.4f}\t{self.total_reward:.4f}"
            f"\t{obs}"
        )

        if terminated:
            self.logger_e.info(
                f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
                f"{self.total_reward:.4f}"
            )

        return obs, reward, terminated, truncated, self._info


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from inv_single_scenarios import scenario_simple

    print(f"Scenario: {scenario_simple}")

    for mode in ("vec", "vec_d"):
        print(f"\n=== observation_mode={mode} ===")
        env = InvSingleEnv(scenario=scenario_simple, observation_mode=mode)
        print("action_space :", env.action_space)
        print("obs_space    :", env.observation_space)

        obs, info = env.reset(seed=42)
        print(f"reset obs: {obs}")

        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"t={info['action_period']:2d}  order={info['order']:6.2f}  "
                f"d={info['demand']:2d}  reward={reward:.2f}  obs={obs}"
            )
        print(f"total_reward: {env.total_reward:.2f}")

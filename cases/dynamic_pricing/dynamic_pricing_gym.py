"""Dynamic pricing Gymnasium wrapper."""

from __future__ import annotations

from typing import Any
import logging

import numpy as np
import gymnasium as gym

import dynamic_pricing_mdp as mdp
from dynamic_pricing_mdp import DynamicPricingState
from dynamic_pricing_scenarios import DynamicPricingScenario

# floor on the intensity action before inverting lambda(p): intensity -> 0 is
# the null (shut-off) price, which is +inf; the floor caps it at a finite,
# demand-killing price (log(a/eps)/alpha).
_INTENSITY_EPS = 1e-6

# raw price-action upper bound: at p=8 (a=100, alpha=1) intensity ~ 0.03,
# effectively the null price (IR gym.action_modes bounds)
_PRICE_HIGH = 8.0


def intensity_to_price(scenario: DynamicPricingScenario, intensity: float) -> float:
    """Invert lambda(p) = a * exp(-alpha * p): the paper's native control.

    Used by the 'intensity' action mode and by the deployable policy wrapper —
    the reparametrization is part of the trained model's action contract.
    """
    lam = min(max(float(intensity), _INTENSITY_EPS), scenario.a)
    return float(np.log(scenario.a / lam) / scenario.alpha)


class DynamicPricingEnv(gym.Env):
    """Dynamic pricing Gymnasium wrapper.

    action_mode:
    - "price":     Box [0, 8]; the agent sets the price directly (clipped).
    - "intensity": Box [0, lambda*], lambda* = a/e; the paper's native control.
                   Reparametrized to price via p(lambda) = log(a/lambda)/alpha;
                   intensity -> 0 gives the null (shut-off) price.

    observation_mode:
    - "vec":   [inventory, time_to_go]
    - "vec_d": [inventory, time_to_go, units_sold (last period)]

    reward_mode:
    - "revenue": per-step revenue total = sales + salvage (maximize, no sign flip)

    Notes:
    - The core simulator is the source of truth for dynamics.
    - This wrapper handles only Gym-facing concerns: spaces, formatting, reset/step.
    - Episodes end `terminated` (not truncated) at the horizon — T is part of
      the problem — and early when inventory hits 0 (absorbing, no reorder).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: DynamicPricingScenario,
        action_mode: str = "price",
        observation_mode: str = "vec",
        reward_mode: str = "revenue",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()

        assert action_mode in {"price", "intensity"}, \
            f"action_mode must be 'price' or 'intensity', got '{action_mode}'."
        assert observation_mode in {"vec", "vec_d"}, \
            f"observation_mode must be 'vec' or 'vec_d', got '{observation_mode}'."
        assert reward_mode in {"revenue"}, \
            f"reward_mode must be 'revenue', got '{reward_mode}'."

        self.scenario: DynamicPricingScenario = scenario
        self.action_mode      = action_mode
        self.observation_mode = observation_mode
        self.reward_mode      = reward_mode

        self._state:        DynamicPricingState
        self._info:         dict
        self._episode_seed: int
        self._step:         int
        self.total_reward:  float

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
        fh_e = logging.FileHandler(f"{filename}_episode.log")
        fh_e.setFormatter(fmt)
        self.logger_e.addHandler(fh_e)

        self.logger_s.handlers.clear()
        self.logger_s.setLevel(logging.DEBUG)
        fh_s = logging.FileHandler(f"{filename}_step.log")
        fh_s.setFormatter(fmt)
        self.logger_s.addHandler(fh_s)

    def _build_action_space(self) -> gym.spaces.Box:
        if self.action_mode == "price":
            high = _PRICE_HIGH
        else:
            high = self.scenario.demand.max_intensity()
        return gym.spaces.Box(
            low=np.float32(0.0), high=np.float32(high), shape=(1,), dtype=np.float32,
        )

    def _build_observation_space(self) -> gym.spaces.Box:
        sc = self.scenario
        low  = [0.0, 0.0]
        high = [float(sc.n0), float(sc.horizon)]
        if self.observation_mode == "vec_d":
            low.append(0.0)
            high.append(float(sc.n0))
        return gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        sc, s = self.scenario, self._state
        feats = [float(s.inventory), float(sc.horizon - s.period)]
        if self.observation_mode == "vec_d":
            feats.append(float(self._info["units_sold"]))
        return np.array(feats, dtype=np.float32)

    def _to_price(self, action) -> float:
        raw = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        if self.action_mode == "price":
            return float(np.clip(raw, 0.0, _PRICE_HIGH))
        return intensity_to_price(self.scenario, raw)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        # per-env stream (spec §7): global np.random would replay identical
        # episode-seed sequences across SubprocVecEnv workers
        self._episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2_147_483_647))

        self._state, self._info = mdp.init_state(self.scenario, self._episode_seed)
        self._step        = 0
        self.total_reward = 0.0

        return self._get_obs(), self._info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        price = self._to_price(action)

        self._state, self._info = mdp.advance(self.scenario, self._state, price)
        terminated = self._state.terminated

        reward = self._info["revenue"]["total"]

        self.total_reward += reward
        self._step        += 1

        self.logger_s.info(
            f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
            f"{price:.4f}\t{self._info['arrivals']}\t{self._info['units_sold']}\t"
            f"{self._state.inventory}\t{reward:.4f}\t{self.total_reward:.4f}"
        )

        if terminated:
            self.logger_e.info(
                f"{self._episode_seed}\t{self._step}\t"
                f"{self.scenario.a}\t{self.scenario.alpha}\t{self.scenario.n0}\t"
                f"{self.scenario.n0 - self._state.inventory}\t{self.total_reward:.4f}"
            )

        return self._get_obs(), reward, terminated, False, self._info


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dynamic_pricing_scenarios import SCENARIOS

    for name, sc in SCENARIOS.items():
        for action_mode in ("price", "intensity"):
            env = DynamicPricingEnv(scenario=sc, action_mode=action_mode, observation_mode="vec_d")
            print(f"\n=== {name} / {action_mode} ===")
            print("action_space :", env.action_space)
            print("obs_space    :", env.observation_space)

            obs, info = env.reset(seed=42)
            terminated = False
            steps = 0
            while not terminated:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                assert env.observation_space.contains(obs), f"obs {obs} out of space"
            print(f"steps={steps}  total_reward={env.total_reward:.2f}  final obs={obs}")

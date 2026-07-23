"""Gymnasium wrapper for the ADI + flexible-delivery simulator."""

from __future__ import annotations

from typing import Any
import logging

import numpy as np
import gymnasium as gym

from adi_flex_scenarios import AdiFlexScenario
import adi_flex_mdp as mdp
from adi_flex_mdp import AdiFlexState


# upper bounds on the two decisions, confirmed at the Phase-A gate:
# order 0-60 covers every policy in the paper's Table 3 with headroom;
# hold-back 0-20 sits above Table 2's most conservative protection level (11)
# and above anything Poisson(6) realistically demands.
ORDER_MAX     = 60
HOLD_BACK_MAX = 20


class AdiFlexEnv(gym.Env):
    """Standard Gymnasium wrapper around the domain-level AdiFlex simulator.

    observation_mode:
    - "vec":     [period, inventory, due_now, due_next] — the raw system state
    - "vec_mip": [period, modified_inventory_position, due_next]
                 MIP = inventory - due_now - due_next, the paper's u_i. Its
                 propositions 1-2 show the optimal ordering policy is (s, S) in
                 exactly this statistic, so this mode hands the policy the
                 sufficient statistic instead of making it infer one.

    All observations are taken at the pre-order state (after advance1).

    Action (MultiDiscrete([ORDER_MAX + 1, HOLD_BACK_MAX + 1])):
        [order quantity, hold-back], both whole units, both committed before
        this period's demand is revealed.

        A discrete space is deliberate rather than incidental. Beyond matching
        the paper's integrality assumption, it sidesteps the Gaussian-policy
        failure mode of continuous action heads, whose initial mean sits at raw
        0 — for an order quantity that is a dead zone (order nothing, learn
        nothing). A categorical head starts uniform over the range instead.

    Reward:
        -total_cost per period (the objective's sense is minimize).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: AdiFlexScenario,
        observation_mode: str = "vec",
        action_mode: str = "joint",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()

        assert observation_mode in {"vec", "vec_mip"}, (
            f"observation_mode must be 'vec' or 'vec_mip', got '{observation_mode}'."
        )
        assert action_mode in {"joint"}, (
            f"action_mode must be 'joint', got '{action_mode}'."
        )

        self.scenario = scenario
        self.observation_mode = observation_mode
        self.action_mode = action_mode

        self._state: AdiFlexState
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
        """Attach file handlers for the per-episode and per-step logs.

        Deliberately file-only, with no console StreamHandler. Episodes here
        are 12 steps long, so a console handler emits a line every few hundred
        microseconds and dominates the run: measured end-to-end, it drops PPO
        from thousands of steps/s to ~220. The file handlers themselves cost
        about 10%. What you actually want on the console during training is
        SB3's own verbose=1 rollout table, which is unaffected by this.
        """
        fmt = logging.Formatter("%(asctime)s\t%(message)s")

        self.logger_e.handlers.clear()
        self.logger_e.setLevel(logging.DEBUG)
        self.logger_e.propagate = False
        fh_e = logging.FileHandler(f"{filename}_episode.log")
        fh_e.setFormatter(fmt)
        self.logger_e.addHandler(fh_e)

        self.logger_s.handlers.clear()
        self.logger_s.setLevel(logging.DEBUG)
        self.logger_s.propagate = False
        fh_s = logging.FileHandler(f"{filename}_step.log")
        fh_s.setFormatter(fmt)
        self.logger_s.addHandler(fh_s)

    def _build_action_space(self) -> gym.spaces.MultiDiscrete:
        # one component per canonical decision, in the IR's `encodes` order
        return gym.spaces.MultiDiscrete([ORDER_MAX + 1, HOLD_BACK_MAX + 1])

    def _build_observation_space(self) -> gym.spaces.Box:
        horizon = float(self.scenario.horizon)
        if self.observation_mode == "vec_mip":
            low  = [0.0, -np.inf, 0.0]
            high = [horizon, np.inf, np.inf]
        else:
            low  = [0.0, -np.inf, 0.0, 0.0]
            high = [horizon, np.inf, np.inf, np.inf]

        return gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        s = self._state
        if self.observation_mode == "vec_mip":
            mip = float(s.inventory - s.due_now - s.due_next)
            obs = [float(s.period), mip, float(s.due_next)]
        else:
            obs = [
                float(s.period),
                float(s.inventory),
                float(s.due_now),
                float(s.due_next),
            ]
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

        state, self._info = mdp.init_state(
            scenario=self.scenario,
            episode_seed=self._episode_seed,
        )
        self._state = mdp.advance1(self.scenario, state)
        self._step = 0
        self.total_reward = 0.0

        return self._get_obs(), self._info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        order_quantity = int(np.asarray(action).reshape(-1)[0])
        hold_back      = int(np.asarray(action).reshape(-1)[1])

        new_state, self._info = mdp.advance2(
            self.scenario, self._state,
            order_quantity=order_quantity, hold_back=hold_back,
        )
        self._state = (
            mdp.advance1(self.scenario, new_state)
            if not new_state.terminated
            else new_state
        )

        obs = self._get_obs()
        reward = -float(self._info["cost"]["total"])
        terminated = new_state.terminated
        truncated = False

        self.total_reward += reward
        self._step += 1

        # inventory / due_* are physical state, read off the completed-period
        # state (new_state); the demand draws are info diagnostics. self._state
        # has since advanced to the next period's pre-order state.
        self.logger_s.info(
            f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
            f"{order_quantity}\t{hold_back}\t{self._info['demand_now']}\t"
            f"{self._info['demand_next']}\t{self._info['demand_later']}\t"
            f"{self._info['region']}\t{new_state.inventory}\t{new_state.due_now}\t"
            f"{new_state.due_next}\t{reward:.4f}\t{self.total_reward:.4f}\t{obs}"
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
    from adi_flex_scenarios import SCENARIOS

    # every observation mode x every registered scenario, under random actions,
    # asserting the observation stays inside its declared space at every step
    checked = 0
    for mode in ("vec", "vec_mip"):
        for name, scenario in SCENARIOS.items():
            env = AdiFlexEnv(scenario=scenario, observation_mode=mode)
            obs, info = env.reset(seed=42)
            assert env.observation_space.contains(obs), (
                f"{mode}/{name}: reset obs {obs} outside {env.observation_space}"
            )
            env.action_space.seed(7)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                assert env.observation_space.contains(obs), (
                    f"{mode}/{name}: step {steps} obs {obs} outside space"
                )
                assert np.isfinite(reward), f"{mode}/{name}: non-finite reward"
                steps += 1
                checked += 1
            assert steps == scenario.horizon, (
                f"{mode}/{name}: ran {steps} steps, expected {scenario.horizon}"
            )
        print(f"observation_mode={mode:8s}  all {len(SCENARIOS)} scenarios OK")

    env = AdiFlexEnv(scenario=SCENARIOS["exp4"], observation_mode="vec")
    print(f"\naction_space : {env.action_space}")
    print(f"obs_space    : {env.observation_space}")
    print(f"\n{checked} steps checked, every observation inside its declared space")

"""Gymnasium wrapper for the single-echelon inventory simulator."""

from __future__ import annotations

from typing import Any
import logging
import numpy as np
import gymnasium as gym

from inv_single_scenarios import InvSingleScenario, ScenarioSource
from inv_single_mdp import (
    InvSingleState,
    init_state,
    advance1,
    advance2,
)


# The lead-time LAW feature the `_slt` modes render. The IR declares it as
# `{"derived": "leadtime_law", "expr": "leadtime.probs", "dim": 3}`: `probs`
# folds from the SELECTED candidate at load (v0.10.7, upstream #77), and `dim`
# is the categorical candidate's support size. The gym cannot read the folded
# literal — it renders from the episode's generator, which under a grid sampler
# changes per episode — so it mirrors the declaration here, and
# `inv_single_test.py::test_the_law_feature_reads_the_law_not_the_constant`
# holds the two together.
LEADTIME_LAW_DIM = 3


def leadtime_law(gen) -> list[float]:
    """The pmf of one lead-time draw, in the generator's own support order —
    `families.law()`'s rendering: a deterministic source is the one-point law,
    a discrete one its (normalised) probabilities. Defined under every
    candidate, so a mode carrying it is narrowed by declared WIDTH, never by
    family."""
    probs = getattr(gen, "probabilities", None)
    if probs is None:
        return [1.0]
    return [float(q) for q in probs]


class InvSingleEnv(gym.Env):
    """Standard Gymnasium wrapper around the domain-level inventory simulator.

    observation_mode:
    - "vec":      [period, inventory, pipeline[0], ..., pipeline[L]]
    - "vec_ip":   [period, inventory_position]
                  IP = inventory + sum(pipeline); compact sufficient statistic
                  for base-stock policies under deterministic leadtime +
                  backlog (and *insufficient* under stochastic leadtime, where
                  orders can cross, or under lost sales)
    - "vec_ip_ctx_slt": "vec_ip" plus the same context "vec_ctx_slt" carries —
                    the lead-time law and the cost context. The RESTRICTED-
                    INFORMATION control for a generalist: identical conditioning
                    to `vec_ctx_slt`, with the pipeline collapsed to its sum.
                    The context is not a courtesy — across cells the optimal
                    order-up-to level moves with both `b` and the law, so an IP
                    arm without it could not adapt at all and the contrast would
                    be unfair rather than restricted. Same width rule.
    - "vec_ctx_slt": same as "vec_ctx" plus the LEAD-TIME LAW in force — the
                    pmf over the categorical candidate's support {1,2,3}, which
                    the IR feature `leadtime.probs` folds from the SELECTED
                    candidate at load (upstream #77, v0.10.7) and the gym
                    renders from the episode's generator via `leadtime_law()`.
                    Under a deterministic candidate the law is the one-point
                    pmf (width 1) — defined, but not the 3-wide vector the IR
                    declares (`dim: 3`), so construction refuses by WIDTH, not
                    by family: `LEADTIME_LAW_DIM` mirrors the IR's `dim`. The
                    `lt_variance_*` grids sweep this vector, so a generalist
                    over them is otherwise being asked to adapt to a regime it
                    cannot see.
    - "vec_ctx":  same as "vec" plus the cost context (critical fractile
                  b/(b+h), fixed order cost K) — the grid16 generalist's mode

    All observations are taken at the pre-order state (after advance1).
    pipeline length = scenario.leadtime.max() + 1.

    Action — the order quantity q >= 0. Integrality is decided HERE, not in the
    MDP layer: `advance2` takes a float and stores whatever it is handed, so the
    action_mode is what commits.
    - "discrete" (default): Discrete(q_high + 1); q is integral, which is what
                    keeps the IR's `pipeline: int_vector` honest. It is the
                    default because the order flows into integer stock
                    (`inventory: int`, `pipeline: int_vector`), so a fractional
                    order would put non-integers into an integer state.
    - "continuous": Box[0, q_high]; q passes through unrounded
    - "hurdle":     MultiDiscrete([2, q_high]) — a two-part ("hurdle") encoding
                    that separates the two decisions the (s,S) structure
                    actually makes: head 0 is *whether* to order at all, head 1
                    is *how many* units in [1, q_high]. On a categorical head
                    this should be near-equivalent to "discrete" — the atom at
                    zero is already its own action there — and that is the
                    point: it isolates the quantity head so it can later be
                    swapped for an ordinal or zero-inflated parameterization
                    without also moving the order/do-not-order decision.

    Reward:
        -total_cost per period.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioSource,
        observation_mode: str = "vec",
        action_mode: str = "discrete",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()

        assert observation_mode in {
            "vec",
            "vec_ip",
            "vec_ip_ctx_slt",
            "vec_ctx",
            "vec_ctx_slt",
        }, (
            "observation_mode must be 'vec', 'vec_ip', 'vec_ip_ctx_slt', "
            f"'vec_ctx' or 'vec_ctx_slt', got '{observation_mode}'."
        )
        if observation_mode.endswith("_slt"):
            # The mode shows the lead-time LAW, and the IR declares that
            # feature `dim: 3` — the categorical candidate's support. The law
            # is defined under every candidate (a deterministic source is the
            # one-point pmf), so the refusal is by width: a generator whose
            # law is not 3 points wide cannot fill a 3-wide feature, and a
            # policy trained on one could not read the other. This replaces
            # the family assertion (`hasattr(..., "probabilities")`) that
            # v0.10.4 needed while the IR read a CONSTANT here: that check let
            # a 2-point padded law through and refused a 1-point one on type,
            # neither of which is the declared width.
            law = leadtime_law(scenario.leadtime)
            assert len(law) == LEADTIME_LAW_DIM, (
                f"observation_mode '{observation_mode}' shows the lead-time law "
                f"as a {LEADTIME_LAW_DIM}-wide feature (IR `dim`); scenario "
                f"'{scenario.scenario_name}' has "
                f"{type(scenario.leadtime).__name__} whose law has {len(law)} "
                f"support point(s): {law}. Use a non-_slt mode, or a scenario "
                f"whose lead-time law lives on the {LEADTIME_LAW_DIM}-point support.")

        assert action_mode in {
            "continuous",
            "discrete",
            "hurdle",
        }, (
            "action_mode must be 'continuous', 'discrete' or 'hurdle', "
            f"got '{action_mode}'."
        )

        # scenario source: a fixed scenario, or a sampler resolved per episode
        # in reset(). Space building below reads only family-level attributes
        # (horizon, leadtime, stockout_mode, demand bounds), which samplers
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
        if self.action_mode == "hurdle":
            # [order?, quantity-1] — the quantity head never has to represent
            # zero, so its support is exactly the conditional distribution
            # p(q | q > 0) that a later ordinal/gamma head would model.
            return gym.spaces.MultiDiscrete([2, int(high)])
        return gym.spaces.Box(
            low=np.float32(0.0),
            high=np.float32(high),
            shape=(1,),
            dtype=np.float32,
        )

    def decode_action(self, action) -> float:
        """The order quantity an action encodes (the IR's `transform`).

        Public because anything that reads a trained policy's output — the
        policy probe, a deployed wrapper — must decode it exactly as `step`
        does, and a second copy of this arithmetic is a drift waiting to happen.
        """
        arr = np.asarray(action).reshape(-1)
        if self.action_mode == "hurdle":
            return 0.0 if int(arr[0]) == 0 else float(int(arr[1]) + 1)
        a = arr[0]
        if self.action_mode == "discrete":
            return float(int(a))
        return max(0.0, float(a))

    def _build_observation_space(self) -> gym.spaces.Box:
        inv_low = 0.0 if self.scenario.stockout_mode == "lost_sales" else -np.inf

        if self.observation_mode in ("vec_ip", "vec_ip_ctx_slt"):
            low  = [0.0, inv_low]
            high = [float(self.scenario.horizon), np.inf]
            if self.observation_mode == "vec_ip_ctx_slt":
                n = len(leadtime_law(self.scenario.leadtime))
                low  += [0.0] * n + [0.0, 0.0]
                high += [1.0] * n + [1.0, np.inf]
        else:
            L = self.scenario.leadtime.max() + 1  # pipeline length = max + 1
            low  = [0.0, inv_low] + [0.0] * L
            high = [float(self.scenario.horizon), np.inf] + [np.inf] * L
            if self.observation_mode in ("vec_ctx", "vec_ctx_slt"):
                if self.observation_mode == "vec_ctx_slt":
                    # the lead-time LAW, one entry per point of its own support
                    n = len(leadtime_law(self.scenario.leadtime))
                    low  += [0.0] * n
                    high += [1.0] * n
                # the cost context a cross-regime generalist conditions on
                # (IR mode `vec_ctx`): the critical fractile b/(b+h) and the
                # fixed order cost K. Raw values — VecNormalize owns scaling.
                low  += [0.0, 0.0]
                high += [1.0, np.inf]

        return gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        # LEADING FEATURE IS TIME-TO-GO, not the period index (2026-08-28, F5).
        # Same information for a fixed horizon — the two are affinely related —
        # but it is the finite-horizon convention the other domains use
        # (`adi_flex`: `N - period`), and it counts DOWN to the terminal
        # boundary the policy has to notice. Bounds are unchanged: both live in
        # [0, horizon]. Read the EPISODE's horizon: a grid sampler could resolve
        # a different one per episode.
        sc_ep = self._scenario_ep if self._scenario_ep is not None else self.scenario
        ttg = float(sc_ep.horizon - self._state.period)
        if self.observation_mode in ("vec_ip", "vec_ip_ctx_slt"):
            ip = float(self._state.inventory) + sum(float(x) for x in self._state.pipeline)
            obs = [ttg, ip]
            if self.observation_mode == "vec_ip_ctx_slt":
                sc = self._scenario_ep
                obs += leadtime_law(sc.leadtime)
                b, h = float(sc.shortage_cost), float(sc.holding_cost)
                obs += [b / (b + h), float(sc.order_cost_fixed)]
        else:
            obs = [ttg, float(self._state.inventory)] + [
                float(x) for x in self._state.pipeline
            ]
            if self.observation_mode in ("vec_ctx", "vec_ctx_slt"):
                # read the EPISODE's resolved scenario: under a grid sampler
                # the cell (and its costs) changes per episode
                sc = self._scenario_ep
                if self.observation_mode == "vec_ctx_slt":
                    # the law the episode's source was configured with, NOT the
                    # realized draw — a realization would be a leak, and the
                    # policy needs the distribution to condition on the regime
                    obs += leadtime_law(sc.leadtime)
                b = float(sc.shortage_cost)
                h = float(sc.holding_cost)
                obs += [b / (b + h), float(sc.order_cost_fixed)]
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
        order = self.decode_action(action)

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

    for mode in ("vec", "vec_ip", "vec_ctx",
                 "vec_ctx_slt", "vec_ip_ctx_slt"):
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

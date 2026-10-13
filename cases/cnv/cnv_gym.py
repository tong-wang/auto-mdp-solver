"""Gymnasium wrapper for the censored-newsvendor simulator."""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

import cnv_mdp as mdp
from cnv_bayes import gamma_poisson_quantile
from cnv_event_posterior import EventGridPosterior
from cnv_mdp import CnvState
from cnv_scenarios import CnvScenario, ScenarioSource

# the IR's gym.observation_modes, in the declared render order (spec §7): what
# the wrapper accepts is exactly the menu the IR names, and each mode renders
# exactly that mode's features
OBSERVATION_MODES = ("timing_belief", "full_belief", "event_raw", "timing_raw", "full_raw",
                     "timing_belief_ctx", "full_belief_ctx", "full_belief_log", "full_belief_ms", "timing_belief_log")

# CLAIRVOYANT DIAGNOSTIC MODES — deliberately NOT in the IR menu above, and never
# a board arm. `LatentPoissonDemand` declares the rate hidden (spec §5.2: "no
# observation mode may expose it ... a benchmark that reads it is clairvoyant"),
# so a mode that reads it is not a deployable policy and cannot be crowned. It
# exists to cut the problem in half: with lambda handed over there is NOTHING to
# infer — no posterior, no censoring, no exploration — and because the transition
# carries no inventory between periods (`cnv_mdp.advance`: leftover is discarded,
# sales = min(demand, order)), an episode is 100 INDEPENDENT newsvendor draws
# whose optimal order is ONE CONSTANT, the Poisson(lambda) critical-ratio
# quantile. Its exact bar already exists as the `perfect` benchmark
# (`perfect_profit`). Whatever such an arm fails to reach is PPO's optimisation,
# not representation, inference or the horizon — which is what #E52 asks.
DIAGNOSTIC_OBSERVATION_MODES = ("oracle_rate", "oracle_log")   # oracle_log: the same rate, given as its log (#E60)

# ASSISTANCE OBSERVATION MODES — deliberately NOT in the IR menu either (#E84), and never a board arm:
# rungs of the assistance ladder, each a declared step back from the posterior-only modes. A _myo mode
# renders its _log mode plus log(1 + y_myo), y_myo the myopic order under the SAME posterior. Unlike
# the oracles they read nothing hidden. They stay out of the IR by the human's choice (2026-10-02):
# declaring them needs the posterior quantile as an mdp-block builtin, which moves the signed-off mdp
# fingerprint, and mdp_stage's rl.current then reads every existing run as stale (no carry rule for a
# declaration-only move). The value maps a mode to the information type whose posterior it renders,
# which is also the premium action's anchor.
MYO_INFO = {"timing_belief_log_myo": "timing", "full_belief_log_myo": "full"}
ASSIST_OBSERVATION_MODES = tuple(MYO_INFO)

# HISTORY-BELIEF MODES — feasible (they read nothing hidden; the same information set as the
# matching *_raw mode) but NOT in the IR menu, for an expressibility reason rather than an
# assistance one (#E107): the IR's gym block declares a mode as expressions over the interpreter's
# state namespace, and the EVENT posterior is a function of the whole observed history (the
# censored order LEVELS, one by one — the paper's §3), which no running total in `CnvState` carries.
# So the gym carries the belief itself, per episode, on `cnv_event_posterior.EventGridPosterior`,
# and renders it as the _log modes render theirs: (log posterior mean, posterior CV). The mode is
# a CONTROL on the decision layer (#E105: the event_raw crown already holds this statistic in its
# latent), gated by the domain test against the benchmark's own grid. Bringing it into the menu
# needs a history-derived observation builtin upstream — a proposal, not a local edit.
HISTORY_BELIEF_MODES = ("event_belief_log",)
ALL_OBSERVATION_MODES = OBSERVATION_MODES + DIAGNOSTIC_OBSERVATION_MODES + ASSIST_OBSERVATION_MODES + HISTORY_BELIEF_MODES
ACTION_MODES = ("order_discrete",)

# ASSISTANCE ACTION MODE — deliberately NOT in the IR's action menu (#E84). "premium_discrete" is the
# assistance ladder's top rung (R3): the agent emits a in {0, ..., PREMIUM_MAX} and stocks
# y = min(y_myo + a, arrivals_buffer), y_myo the myopic order under the posterior its _myo observation
# mode renders. Its transform reads the STATE (the posterior), and IR v1 lets an action-mode transform
# reference only the action, the decisions and constants, so the IR cannot declare it; the mdp layer
# still receives a plain order. It reads nothing hidden (y_myo is a function of the observed posterior),
# so it is a deployable policy class, labelled as a step back, never a board arm. a >= 0 only: the
# paper proves the optimal order under censoring is at least the myopic one (the human, 2026-10-02).
# PREMIUM_MAX = 20 is the human's design choice for the rung — a restriction of the policy class, not
# a derived envelope; whether it binds is reported (the share of periods at a = PREMIUM_MAX).
ASSIST_ACTION_MODES = ("premium_discrete",)
ALL_ACTION_MODES = ACTION_MODES + ASSIST_ACTION_MODES
PREMIUM_MAX = 20


def render_obs(observation_mode: str, sc, st, info: dict) -> np.ndarray:
    """The per-period observation vector of every mode that reads only the episode's scenario, the state
    and the last period's info — the env renders through it, and so does ``cnv_policy.py`` on the raw
    ``cnv_mdp`` loop (spec §12), so the two cannot drift. ``event_belief_log`` (a grid posterior the env
    carries) and the ``_myo`` assistance modes stay in ``CnvEnv._get_obs``."""
    ttg = float(sc.horizon - st.period)
    ratio = sc.critical_ratio
    if observation_mode == "oracle_rate":
        # the episode's REALIZED latent rate, read straight off the resolved
        # scenario's demand generator: no inference of any kind is required
        obs = [ttg, ratio, float(sc.demand.rate)]
    elif observation_mode == "oracle_log":
        # the same number as its log: does the oracle's floor come from PPO's noise, or from an
        # input on which rates 0.3 and 1.3 sit 0.08 apart once standardised? (#E60)
        obs = [ttg, ratio, float(np.log(max(float(sc.demand.rate), 1e-9)))]
    elif observation_mode == "timing_belief":
        obs = [ttg, ratio,
               sc.prior_shape + st.cum_sales,
               sc.prior_rate + st.cum_duration]
    elif observation_mode == "full_belief":
        obs = [ttg, ratio,
               sc.prior_shape + st.cum_demand,
               sc.prior_rate + st.period]
    elif observation_mode == "timing_belief_ctx":
        obs = [ttg, ratio, sc.prior_rate, sc.prior_mean,
               sc.prior_shape + st.cum_sales,
               sc.prior_rate + st.cum_duration]
    elif observation_mode == "full_belief_ctx":
        obs = [ttg, ratio, sc.prior_rate, sc.prior_mean,
               sc.prior_shape + st.cum_demand,
               sc.prior_rate + st.period]
    elif observation_mode == "full_belief_log":
        # the SAME sufficient statistic as full_belief, recoded: a bijection of (shape, rate)
        shape, rate = sc.prior_shape + st.cum_demand, sc.prior_rate + st.period
        obs = [ttg, ratio, float(np.log(shape / rate)), float(1.0 / np.sqrt(shape))]
    elif observation_mode == "timing_belief_log":
        # the TIMING posterior (sales and stock-out durations), recoded as full_belief_log is
        shape, rate = sc.prior_shape + st.cum_sales, sc.prior_rate + st.cum_duration
        obs = [ttg, ratio, float(np.log(shape / rate)), float(1.0 / np.sqrt(shape))]
    elif observation_mode == "full_belief_ms":
        shape, rate = sc.prior_shape + st.cum_demand, sc.prior_rate + st.period
        obs = [ttg, ratio, float(shape / rate), float(np.sqrt(shape) / rate)]
    elif observation_mode in ("event_raw", "timing_raw", "full_raw"):
        obs = [ttg, ratio, sc.prior_rate, sc.prior_mean]
        if observation_mode == "event_raw":
            obs += [float(info["sales"]), float(info["stockout_event"])]
        elif observation_mode == "timing_raw":
            obs += [float(info["sales"]), float(info["stockout_time"])]
        else:  # full_raw
            obs += [float(info["demand"])]
    else:
        raise ValueError(f"render_obs does not render {observation_mode!r}: the env carries its state")
    return np.array(obs, dtype=np.float32)


class CnvEnv(gym.Env):
    """Censored-newsvendor Gymnasium wrapper (Jain, Rudi & Wang 2015).

    observation_mode — the IR's menu; every vector leads with time-to-go
    (``horizon_N - period``, counting down: the horizon is the problem's own
    boundary) and the critical ratio (r - c) / r, the cost regime a
    generalist across the grid conditions on:
    - "timing_belief" (default): [time_to_go, critical_ratio, post_shape, post_rate]
                     the paper's timing model as a Markov state — the Gamma
                     posterior after conjugate updating on sales and stock-out
                     times: shape = alpha + cum_sales, rate = beta + cum_duration
    - "full_belief":  [time_to_go, critical_ratio, post_shape, post_rate]
                     the full-observation posterior: shape = alpha + cum_demand,
                     rate = beta + periods played
    - "timing_belief_ctx", "full_belief_ctx": the belief modes with the prior's
                     own parameters inserted — [time_to_go, critical_ratio,
                     prior_rate, prior_mean, post_shape, post_rate] — so a
                     generalist over the paper510 grid sees every grid axis
                     explicitly (unit_cost enters as the critical ratio)
    - "event_raw":    [time_to_go, critical_ratio, prior_rate, prior_mean,
                       last sales, last stock-out flag]
                     the paper's event observation O^E for the last period only;
                     the policy needs a memory (frame stack over the horizon)
    - "timing_raw":   [..., last sales, last stock-out time] — O^ST; memory
    - "full_raw":     [..., last demand] — O^F; memory

    The hidden rate lambda appears in no mode (spec §5.2). The *_raw modes
    render the last period's ``info`` — at reset, the initial info (no sales,
    no stock-out, a full period's duration).

    - "event_belief_log" (history belief, outside the IR menu — see
                     HISTORY_BELIEF_MODES): [time_to_go, critical_ratio,
                     log posterior mean, posterior CV] under the exact EVENT
                     posterior carried on a grid; the event analogue of
                     timing_belief_log (#E107)
    - "timing_belief_log_myo", "full_belief_log_myo" (assistance, outside the
                     IR — see ASSIST_OBSERVATION_MODES): the _log mode plus
                     log(1 + y_myo), y_myo the myopic order under the same
                     posterior (#E84)

    action_mode:
    - "order_discrete" (default): Discrete(arrivals_buffer + 1); the stocking
      level y in whole units, 0..arrivals_buffer — the simulator's own arrival
      cap, derived from the widest grid cell's demand law (IR decision bounds)
    - "premium_discrete" (assistance, outside the IR — see ASSIST_ACTION_MODES):
      Discrete(PREMIUM_MAX + 1); the units stocked ABOVE the myopic order,
      y = min(y_myo + a, arrivals_buffer); only with a _myo observation mode

    reward_mode:
    - "profit" (default): the per-period objective r * sales - c * y

    Notes:
    - The core simulator is the reference implementation of the schema's
      dynamics and objective; this wrapper adds none of its own.
    - This wrapper handles only Gym-facing concerns: spaces, formatting,
      reset/step. A callable scenario source is resolved per episode in
      ``reset()``; space building reads only family-level attributes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioSource,
        observation_mode: str = "timing_belief",
        action_mode: str = "order_discrete",
        reward_mode: str = "profit",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()
        assert observation_mode in ALL_OBSERVATION_MODES, (
            f"observation_mode must be one of {ALL_OBSERVATION_MODES}, got '{observation_mode}'."
        )
        assert action_mode in ALL_ACTION_MODES, (
            f"action_mode must be one of {ALL_ACTION_MODES}, got '{action_mode}'."
        )
        assert action_mode != "premium_discrete" or observation_mode in MYO_INFO, (
            f"premium_discrete is anchored on the posterior a _myo mode renders "
            f"({tuple(MYO_INFO)}), got observation_mode '{observation_mode}'."
        )
        assert reward_mode == "profit", f"reward_mode must be 'profit', got '{reward_mode}'."

        # scenario source: a fixed scenario, or a sampler resolved per episode
        # in reset(). Space building reads only family-level attributes
        # (horizon, arrivals_buffer, demand.max(), price), which sources and
        # grid samplers expose under the same names as a concrete scenario.
        self.scenario = scenario
        self._scenario_ep: CnvScenario | None = None
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.reward_mode = reward_mode

        self._state: CnvState
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
        for logger, suffix in ((self.logger_e, "_episode.log"), (self.logger_s, "_step.log")):
            logger.handlers.clear()
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(f"{filename}{suffix}")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    # -- spaces --------------------------------------------------------------

    def _build_action_space(self) -> gym.spaces.Discrete:
        if self.action_mode == "premium_discrete":
            return gym.spaces.Discrete(PREMIUM_MAX + 1)
        # the IR's decision bounds [0, arrivals_buffer]: a stocking level above
        # the number of arrivals the simulator can generate can never sell
        return gym.spaces.Discrete(int(self.scenario.arrivals_buffer) + 1)

    def myopic_order(self) -> int:
        """The myopic order under the posterior this env's _myo mode renders.

        One function for both uses — the observation's log(1 + y_myo) and the
        premium action's anchor — so the two can never disagree; it is the
        myopic_timing / myopic_full benchmark's own formula.
        """
        sc, st = self._scenario_ep, self._state
        if self.observation_mode == "event_belief_log":
            return self._event_post.myopic_order(sc.critical_ratio)
        if MYO_INFO[self.observation_mode] == "timing":
            shape, rate = sc.prior_shape + st.cum_sales, sc.prior_rate + st.cum_duration
        else:
            shape, rate = sc.prior_shape + st.cum_demand, sc.prior_rate + st.period
        return int(gamma_poisson_quantile(shape, rate, sc.critical_ratio))

    def decode_action(self, action) -> int:
        """The stocking level an action encodes at the current state.

        order_discrete: the IR's identity transform. premium_discrete: the
        myopic order plus the premium, capped at the arrival buffer.

        Public because anything that reads a trained policy's output — the
        policy probe, a deployed wrapper — must decode it exactly as `step`
        does. A policy hands `step` its ACTION, never a decoded order.
        """
        a = max(0, int(np.asarray(action).reshape(-1)[0]))
        if self.action_mode == "premium_discrete":
            assert a <= PREMIUM_MAX, f"premium {a} outside 0..{PREMIUM_MAX} (an order passed as an action?)"
            return min(self.myopic_order() + a, int(self._scenario_ep.arrivals_buffer))
        return a

    def _build_observation_space(self) -> gym.spaces.Box:
        sc = self.scenario
        n = float(sc.horizon)
        cap = float(sc.arrivals_buffer)
        # generous envelopes (spec §4.1): the model's own domain where it has
        # one; the posterior shape grows by at most the cap per period
        if self.observation_mode in ("oracle_rate", "oracle_log"):     # clairvoyant diagnostics
            low = [0.0, 0.0, 0.0 if self.observation_mode == "oracle_rate" else -np.inf]
            high = [n, 1.0, np.inf]
        elif self.observation_mode in ("timing_belief", "full_belief"):
            low = [0.0, 0.0, 0.0, 0.0]
            high = [n, 1.0, np.inf, np.inf]
        elif self.observation_mode in ("timing_belief_ctx", "full_belief_ctx"):
            low = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            high = [n, 1.0, np.inf, np.inf, np.inf, np.inf]
        elif self.observation_mode in ("full_belief_log", "timing_belief_log", "event_belief_log"):   # (log posterior mean, posterior CV) — #E56/#E59/#E65/#E107
            low = [0.0, 0.0, -np.inf, 0.0]
            high = [n, 1.0, np.inf, np.inf]
        elif self.observation_mode in MYO_INFO:                                    # + log(1 + y_myo) — #E84
            low = [0.0, 0.0, -np.inf, 0.0, 0.0]
            high = [n, 1.0, np.inf, np.inf, np.inf]
        elif self.observation_mode == "full_belief_ms":    # (posterior mean, posterior sd), no log — #E63
            low = [0.0, 0.0, 0.0, 0.0]
            high = [n, 1.0, np.inf, np.inf]
        elif self.observation_mode == "event_raw":
            low = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            high = [n, 1.0, np.inf, np.inf, cap, 1.0]
        elif self.observation_mode == "timing_raw":
            low = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            high = [n, 1.0, np.inf, np.inf, cap, 1.0]
        else:  # full_raw
            low = [0.0, 0.0, 0.0, 0.0, 0.0]
            high = [n, 1.0, np.inf, np.inf, cap]
        return gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    # -- observation ---------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # read the EPISODE's resolved scenario: under a grid sampler the cell
        # (its cost and prior) changes per episode
        sc = self._scenario_ep
        st = self._state
        if self.observation_mode == "event_belief_log":
            # the EVENT posterior (no conjugate form), carried on the episode's grid, rendered as the _log modes are
            m, cv = self._event_post.stats()
            obs = [float(sc.horizon - st.period), sc.critical_ratio, float(np.log(m)), float(cv)]
        elif self.observation_mode in MYO_INFO:
            # the _log rendering of the mode's posterior, plus log(1 + y_myo) under that same posterior
            if MYO_INFO[self.observation_mode] == "timing":
                shape, rate = sc.prior_shape + st.cum_sales, sc.prior_rate + st.cum_duration
            else:
                shape, rate = sc.prior_shape + st.cum_demand, sc.prior_rate + st.period
            obs = [float(sc.horizon - st.period), sc.critical_ratio, float(np.log(shape / rate)),
                   float(1.0 / np.sqrt(shape)), float(np.log1p(self.myopic_order()))]
        else:
            return render_obs(self.observation_mode, sc, st, self._info)
        return np.array(obs, dtype=np.float32)

    # -- gym protocol --------------------------------------------------------

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
        self._state, self._info = mdp.init_state(self._scenario_ep, self._episode_seed)
        if self.observation_mode in HISTORY_BELIEF_MODES:
            sc = self._scenario_ep
            self._event_post = EventGridPosterior(sc.prior_shape, sc.prior_rate, int(sc.arrivals_buffer))
        self._step = 0
        self.total_reward = 0.0
        return self._get_obs(), self._info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        order = self.decode_action(action)

        self._state, self._info = mdp.advance(self._scenario_ep, self._state, order)
        if self.observation_mode in HISTORY_BELIEF_MODES:
            self._event_post.update(int(self._info["sales"]), int(self._info["stockout_event"]), int(order))

        obs = self._get_obs()
        reward = float(self._info["profit"]["total"])
        terminated = self._state.terminated
        truncated = False

        self.total_reward += reward
        self._step += 1

        self.logger_s.info(
            f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
            f"{order}\t{self._info['demand']}\t{self._info['sales']}\t"
            f"{self._info['stockout_event']}\t{self._info['stockout_time']:.4f}\t"
            f"{reward:.4f}\t{self.total_reward:.4f}"
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
    from cnv_scenarios import source_base

    for mode in ALL_OBSERVATION_MODES:
        for amode in ACTION_MODES:
            print(f"\n=== observation_mode={mode} action_mode={amode} ===")
            env = CnvEnv(scenario=source_base, observation_mode=mode, action_mode=amode)
            print("action_space :", env.action_space)
            print("obs_space    :", env.observation_space)
            obs, info = env.reset(seed=42)
            assert env.observation_space.contains(obs), obs
            print(f"reset obs: {obs}")
            terminated = truncated = False
            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs), obs
                print(f"t={info['action_period']:2d}  y={info['order']:3d}  D={info['demand']:3d}  "
                      f"reward={reward:8.2f}  obs={obs}")
            print(f"total_reward: {env.total_reward:.2f}")

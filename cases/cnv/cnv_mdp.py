"""Cnv core MDP — the censored newsvendor with stock-out timing.

This module contains the domain-level simulator only. It does not depend on
Gym/Gymnasium, and it is the reference implementation of the dynamics that
``cnv_schema.json`` declares (the differential gate holds the two together
bit-for-bit).

Key components:
1. CnvState: the running simulation state — the clock and the three running
   totals the paper's belief models are built from.
2. State transition functions:
   a. init_state(scenario, episode_seed) -> (CnvState, info)
   b. advance(scenario, state, order) -> (CnvState, info)

The scenario configuration (CnvScenario) lives in cnv_scenarios.py and the
stochastic primitives (demand / epoch generators) in cnv_uncertainty.py.

Notations (paper: Jain, Rudi & Wang 2015):
    - y (order):        stocking level placed at the start of the period
    - D (demand):       arrivals in the period, Poisson(lambda), lambda hidden
    - s (sales):        min(D, y); the rest is lost, unobserved
    - e (stockout_event): 1[D >= y] — every stocked unit sold
    - t^s (stockout_time): arrival epoch of the y-th customer when D >= y >= 1;
                        0 when y = 0; 1 when no stock-out
    - S, T (cum_sales, cum_duration): the timing sufficient statistic —
                        posterior on lambda is Gamma(alpha + S, beta + T)
    - cum_demand:       the full-observation statistic's first coordinate

Event sequence within a period:
    A — arrivals: demand and epochs realized; sales, stock-out flag and time
    U — the running totals absorb this period's sales, duration and demand

The arrival process is simulated by the conditional-uniform method: the count
D is Poisson(lambda) and, given D, the arrival epochs are the first D entries
of a per-period vector of i.i.d. uniforms, whose order statistics are the
arrival times. Demand is capped at the epoch vector's length
(``scenario.arrivals_buffer``), a rendering envelope at the 1e-8 tail.

Period indexing is 0-based; ``state.period`` is the running counter and
``info["action_period"]`` the period the decision acted on (spec §6.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cnv_scenarios import CnvScenario


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CnvState:
    """Full internal state of the simulator.

    This captures everything the simulator needs to generate transitions — it is
    the MDP state from the *simulator's* perspective, not the agent's.  Whether
    a field is observable to the RL agent is NOT determined here; observation
    construction is delegated to the gym wrapper (cnv_gym.py) according to the
    ``observation_mode`` it is configured with.

    The three running totals are not read by any transition: the world needs
    only the clock and the seeds. They are carried so the belief observation
    modes can render the paper's sufficient statistics without a memory of
    the raw history (spec §6.4, "realized random outcomes that state stores
    only so the gym can observe them").
    """

    # current period (0-indexed; incremented after each advance())
    period: int

    # True once period == scenario.horizon
    terminated: bool

    # cumulative sales S — timing statistic, first coordinate
    cum_sales: int

    # cumulative sales duration T — timing statistic, second coordinate
    cum_duration: float

    # cumulative demand — the full-observation statistic's first coordinate
    cum_demand: int

    ## internal RNG state
    episode_seed: int
    seed_salt: int = field(repr=False)


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------


def init_state(scenario: CnvScenario, episode_seed: int) -> tuple[CnvState, dict]:
    """Initialize the state at the start of an episode: nothing sold yet."""
    state = CnvState(
        period=0,
        terminated=False,
        cum_sales=0,
        cum_duration=0.0,
        cum_demand=0,
        episode_seed=episode_seed,
        seed_salt=scenario.seed_salt,
    )
    # info carries the time index, this transition's random outcomes and event
    # diagnostics, and the economic quantities; the running totals live on the
    # state and are not duplicated here (spec §6.4)
    info: dict = {
        "action_period": state.period - 1,  # no action taken yet
        "order": 0,
        "demand": 0,
        "sales": 0,
        "stockout_event": 0,
        "stockout_time": 1.0,
        "profit": {"revenue": 0.0, "stocking": 0.0, "total": 0.0},
    }
    return state, info


def stockout_time_of(epochs: np.ndarray, demand: int, order: int) -> float:
    """The paper's t^s: the epoch of the y-th arrival when D >= y >= 1.

    ``epochs[:demand]`` are the period's arrival epochs (unordered); the y-th
    smallest is the moment the last stocked unit sold. y = 0 stocks nothing —
    an immediate stock-out at time 0, no sales duration observed; no stock-out
    means the newsvendor watched sales for the whole period, t^s = 1.
    """
    if order == 0:
        return 0.0
    if demand >= order:
        return float(np.sort(epochs[:demand])[order - 1])
    return 1.0


def advance(scenario: CnvScenario, state: CnvState, order: int) -> tuple[CnvState, dict]:
    """Stock ``order`` units, realize the period's arrivals, sell, and absorb
    the observation into the running totals.

    Returns (next_state, info). Reward is NOT computed here; the gym wrapper
    derives it from ``info["profit"]`` in step().
    """
    assert not state.terminated, (
        "Episode is already finished. Call init_state() to start a new one."
    )
    assert order >= 0, f"order must be non-negative, got {order}."

    # A — arrivals
    demand = min(scenario.demand.sample(state), scenario.arrivals_buffer)
    epochs = scenario.epochs.sample(state)
    sales = int(min(demand, order))
    stockout_event = 1 if demand >= order else 0
    stockout_time = stockout_time_of(epochs, demand, order)

    # objective components, evaluated at END_OF_PERIOD (spec §6.4)
    revenue = scenario.price * sales
    stocking = -scenario.unit_cost * order
    total = revenue + stocking

    # U — the running totals absorb this period's observation
    next_period = state.period + 1
    next_state = CnvState(
        period=next_period,
        terminated=next_period >= scenario.horizon,
        cum_sales=state.cum_sales + sales,
        cum_duration=state.cum_duration + stockout_time,
        cum_demand=state.cum_demand + demand,
        episode_seed=state.episode_seed,
        seed_salt=state.seed_salt,
    )
    info: dict = {
        "action_period": state.period,   # = the INPUT period
        "order": order,
        "demand": demand,
        "sales": sales,
        "stockout_event": stockout_event,
        "stockout_time": stockout_time,
        "profit": {"revenue": revenue, "stocking": stocking, "total": total},
    }
    return next_state, info


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------


def _run_smoke(source, order: int = 10, episode_seed: int = 3) -> None:
    scenario = source(episode_seed) if callable(source) else source
    print(f"\n=== {scenario.scenario_name}: {scenario.desc} ===")
    print(f"realized lambda = {scenario.demand.rate:.4f} (hidden from the agent)")
    state, _ = init_state(scenario, episode_seed)
    total = 0.0
    while not state.terminated:
        state, info = advance(scenario, state, order=order)
        total += info["profit"]["total"]
        print(
            f"t={info['action_period']:2d}  y={info['order']:3d}  D={info['demand']:3d}  "
            f"s={info['sales']:3d}  e={info['stockout_event']}  t^s={info['stockout_time']:.3f}  "
            f"S={state.cum_sales:3d}  T={state.cum_duration:.3f}  "
            f"profit={info['profit']['total']:7.2f}"
        )
    print(f"episode profit: {total:.2f}")


if __name__ == "__main__":
    from cnv_scenarios import source_base, source_fig3_c10

    _run_smoke(source_base, order=10, episode_seed=3)
    _run_smoke(source_fig3_c10, order=0, episode_seed=3)

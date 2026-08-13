"""FNV core MDP.

This module contains the domain-level simulator only.
It does not depend on Gym/Gymnasium.

State/info is represented from the *simulator's* full perspective.  What is
observable to the RL agent is NOT defined here — observations are constructed
in the gym wrapper (fnv_gym.py) based on the ``observation_mode`` it is
configured with.

Key components:
1. FnvState: running simulation state.
2. State transition functions:
    a. init_state(scenario, episode_seed) -> (FnvState, info)
    b. advance(scenario, state, decision) -> (FnvState, info)

The scenario configuration (FnvScenario) lives in fnv_scenarios.py and the
stochastic primitives (demand signal generators) in fnv_uncertainty.py. The
experiment-design grids (FnvScenarioGrid) live in fnv_grids.py and never reach
this layer (spec §5.6).

Notations:
    - N: number of ordering periods (default 3)
    - tau[n]: time epoch at which signal[n] is revealed
              tau = [0, linspace(0,T,N), 1]  (length N+2)
    - scenario.signal: NormalDemandSignal producing the increment revealed at
              each period n (transition tau[n] -> tau[n+1]); the increment at
              period 0 has stdev 0 because tau[0] = tau[1] = 0
    - I[n]: cumulative information = sum of signals[0..n-1] samples
    - D: realized demand = mu + I[N+1]
    - inventory: cumulative quantity ordered so far (= sum of past decisions)
    - information: I[period], demand signal available before the current order

NOTE: The information revealed AFTER placing the order at period n is I[n+1],
which becomes the observation for period n+1.  D is only realized after the
last order (period N) is placed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from fnv_scenarios import FnvScenario


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FnvState:
    """Full internal state of the simulator.

    This captures everything the simulator needs to generate transitions — it is
    the MDP state from the *simulator's* perspective, not the agent's.  Whether
    a field is observable to the RL agent is NOT determined here; observation
    construction is delegated to the gym wrapper (fnv_gym.py) according to
    the ``observation_mode`` it is configured with.
    """

    period:      int    # current ordering period (1-indexed; 1..N)
    terminated:  bool
    inventory:   float  # cumulative orders placed so far
    information: float  # I[period]: cumulative demand signal before this period's order
    total_cost:  float  # sum of ordering costs incurred so far

    episode_seed: int
    seed_salt:    int = field(repr=False)


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------

def init_state(scenario: FnvScenario, episode_seed: int) -> tuple[FnvState, dict]:
    """Initialize state at the start of an episode.

    information=0 because tau[1]=tau[0]=0 so signal[0].stdev=0 always.
    """
    state = FnvState(
        period=1,
        terminated=False,
        inventory=0.0,
        information=0.0,
        total_cost=0.0,
        episode_seed=episode_seed,
        seed_salt=scenario.seed_salt,
    )
    # info carries the time index, transition diagnostics, and economic
    # quantities (per-step cost + cumulative total_cost). The physical state
    # fields inventory/information live on the returned state and are not
    # duplicated here (spec §4.7).
    info: dict = {
        "action_period": state.period - 1,  # no action taken yet
        "decision":      0.0,
        "cost":          0.0,
        "total_cost":    0.0,
    }
    return state, info


def advance(
    scenario: FnvScenario,
    state:    FnvState,
    decision: float,
) -> tuple[FnvState, dict]:
    """Place order, reveal next demand signal, compute costs.

    At the final period (period == N) also realizes demand, revenue, and profit.
    Returns (next_state, info).
    """
    assert not state.terminated, "Episode already finished."
    assert decision >= 0, f"decision must be non-negative, got {decision}."

    n = state.period

    cost_n         = scenario.ordering_cost_rate(n) * decision
    new_inventory  = state.inventory + decision
    new_total_cost = state.total_cost + cost_n

    # reveal demand signal for the transition tau[n] -> tau[n+1]
    new_information = state.information + scenario.signal.sample(state)

    new_period = n + 1
    terminated = new_period > scenario.N

    next_state = FnvState(
        period=new_period,
        terminated=terminated,
        inventory=new_inventory,
        information=new_information,
        total_cost=new_total_cost,
        episode_seed=state.episode_seed,
        seed_salt=state.seed_salt,
    )

    info: dict = {
        "action_period": n,   # action_period = input state.period
        "decision":      decision,
        "cost":          cost_n,
        "total_cost":    new_total_cost,
    }

    if terminated:
        if scenario.mmfe_mode == "additive":
            demand = scenario.mu + new_information
        else:
            demand = float(np.exp(scenario.mu + new_information))
        sales      = min(demand, new_inventory)
        revenue    = scenario.r * sales
        profit     = revenue - new_total_cost
        profit_max = (scenario.r - scenario.c1) * max(demand, 0.0)
        info.update({
            "demand":     demand,
            "sales":      sales,
            "revenue":    revenue,
            "profit":     profit,
            "profit_max": profit_max,
        })

    return next_state, info


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

def _run_episode(scenario: FnvScenario, decision: float = 0.3, episode_seed: int = 42) -> None:
    print(f"\n=== {scenario.scenario_name} ===")
    print(f"  signal: {scenario.signal}")
    state, _ = init_state(scenario, episode_seed)
    while not state.terminated:
        state, info = advance(scenario, state, decision)
        print(
            f"  t={info['action_period']}  dec={info['decision']:.2f}  "
            f"cost={info['cost']:.4f}  inv={state.inventory:.4f}  "
            f"info={state.information:.4f}"
            + (f"  demand={info['demand']:.4f}  profit={info['profit']:.4f}" if state.terminated else "")
        )


if __name__ == "__main__":
    from fnv_scenarios import scenario_simple

    for ep in range(3):
        print(f"\n--- episode_seed={ep} ---")
        _run_episode(scenario_simple, episode_seed=ep)

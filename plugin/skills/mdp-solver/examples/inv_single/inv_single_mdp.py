"""InvSingle core MDP.

This module contains the domain-level simulator only.
It does not depend on Gym/Gymnasium.

Key components:
1. InvSingleState: mutable simulation state (period, net inventory, pipeline).
2. State transition functions:
   a. init_state(scenario, episode_seed) — initialize an empty state.
   b. advance1(scenario, state) — execute all events before 'O'.
   c. advance2(scenario, state, order) — execute 'O' and remaining events,
      compute costs, and advance the period counter.

The scenario configuration (InvSingleScenario) lives in inv_single_scenarios.py
and the stochastic primitives (demand / lead-time generators) in
inv_single_uncertainty.py.

Notations:
    - inventory: net inventory level (negative = backlog when stockout_mode='backlog')
    - pipeline: list where pipeline[k] = qty arriving at the k-th R event from now
    - period: 0-indexed current time step

Event sequence:
    R — receiving order (pipeline[0] flows into inventory as received)
    O — placing new order (appended to pipeline)
    D — demand arrival (demand subtracted from inventory)

Valid sequences (R must precede D):
    O-R-D (default), R-D-O, R-O-D

Pipeline convention:
    - pipeline[k] = qty arriving at the k-th R event from now (fixed length = leadtime.max() + 1)
    - R event: arrival = pipeline[0]; pipeline = pipeline[1:] + [0]  (shift left)
    - O event: pipeline[lt] += order  (direct index, no length change)
    Length is constant throughout; no append/pop needed.

    Consequence: lt=k means the order arrives exactly k R-events after O runs.
    In O-R-D, lt=0 yields same-period arrival (O then R in the same period).
    In R-O-D, lt=0 is semantically broken (R already ran) — disallowed by validation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from inv_single_scenarios import InvSingleScenario


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InvSingleState:
    """Full internal state of the simulator.

    This captures everything the simulator needs to generate transitions — it is
    the MDP state from the *simulator's* perspective, not the agent's.  Whether
    a field is observable to the RL agent is NOT determined here; observation
    construction is delegated to the gym wrapper (inv_single_gym.py) according to
    the ``observation_mode`` it is configured with.
    """

    # current time period (0-indexed; incremented after each advance2())
    period: int

    # True once period == scenario.horizon
    terminated: bool

    # net inventory: positive = on-hand stock, negative = backlog (if allowed)
    inventory: int

    # pipeline[k] = qty arriving at the k-th R event from now;
    # len(pipeline) == scenario.leadtime.max() + 1 (fixed throughout)
    pipeline: list[int]

    # within-period accumulators; reset at the start of each advance1()
    received:   int
    demand:     int
    lost_sales: int

    seed_salt:    int = field(repr=False)
    episode_seed: int


    def copy(self) -> InvSingleState:
        return InvSingleState(
            period=self.period,
            terminated=self.terminated,
            inventory=self.inventory,
            pipeline=list(self.pipeline),
            received=self.received,
            demand=self.demand,
            lost_sales=self.lost_sales,
            seed_salt=self.seed_salt,
            episode_seed=self.episode_seed,
        )


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------

def init_state(
    scenario: InvSingleScenario,
    episode_seed: int,
) -> tuple[InvSingleState, dict]:
    """Initialize an empty inventory state at the start of an episode."""
    state = InvSingleState(
        seed_salt=scenario.seed_salt,
        episode_seed=episode_seed,
        period=0,
        terminated=False,
        inventory=0,
        pipeline=[0] * (scenario.leadtime.max() + 1),
        received=0,
        demand=0,
        lost_sales=0,
    )
    # info carries the time index (period), transition random-outcome/event
    # diagnostics (order, leadtime, received, demand, lost_sales), and economic
    # quantities (cost). The physical state snapshot (inventory, pipeline) lives
    # on the returned state and is not duplicated here (spec §4.7).
    info = {
        "action_period": state.period - 1,  # no action taken yet
        "order":       0,
        "leadtime":    0,
        "received":    0,
        "demand":      0,
        "lost_sales":  0,
        "cost": {
            "holding":        0.0,
            "shortage":       0.0,
            "order_fixed":    0.0,
            "order_variable": 0.0,
            "total":          0.0,
        },
    }
    return state, info


def _apply_event_R(state: InvSingleState) -> None:
    """R event: receive pipeline[0] into inventory and shift left."""
    state.received = state.pipeline[0]
    state.pipeline = state.pipeline[1:] + [0]
    state.inventory += state.received


def _apply_event_D(
    scenario: InvSingleScenario,
    state: InvSingleState,
) -> None:
    """D event: realize demand; update state.demand and state.lost_sales."""
    state.demand = scenario.demand.sample(state)
    state.inventory -= state.demand
    if scenario.stockout_mode == "lost_sales":
        state.lost_sales = max(0, -state.inventory)
        state.inventory  = max(0, state.inventory)


def _apply_event_O(
    scenario: InvSingleScenario,
    state: InvSingleState,
    order: float,
) -> int:
    """O event: place order in pipeline[lt]; return sampled leadtime."""
    if order > 0:
        leadtime = scenario.leadtime.sample(state)
        state.pipeline[leadtime] += order
        return leadtime
    return 0


def advance1(
    scenario: InvSingleScenario,
    state: InvSingleState,
) -> InvSingleState:
    """Execute all events before 'O' in the scenario's event_sequence.

    Returns the pre-order state — the natural observation point for the agent.
    Within-period accumulators received and lost_sales are reset to 0 at the
    start. demand is NOT reset — it retains the most recent D-event value so
    that the pre-order observation always carries a meaningful demand signal.
    """
    assert not state.terminated, (
        "Episode is already finished. Call init_state() to start a new one."
    )
    state1            = state.copy()
    state1.received   = 0
    state1.lost_sales = 0

    for event in scenario.event_sequence:
        if event == 'O':
            break
        elif event == 'R':
            _apply_event_R(state1)
        elif event == 'D':
            _apply_event_D(scenario, state1)

    return state1


def advance2(
    scenario: InvSingleScenario,
    state:    InvSingleState,
    order:    float,
) -> tuple[InvSingleState, dict]:
    """Execute 'O' and all remaining events, compute costs, advance period.

    Must be called with the state returned by advance1().
    """
    assert order >= 0, f"order must be non-negative, got {order}."

    state2   = state.copy()
    leadtime = 0

    o_idx = scenario.event_sequence.index('O')
    for event in scenario.event_sequence[o_idx:]:
        if event == 'O':
            leadtime = _apply_event_O(scenario, state2, order)
        elif event == 'R':
            _apply_event_R(state2)
        elif event == 'D':
            _apply_event_D(scenario, state2)

    # costs assessed at end of period on final inventory
    fix_cost   = scenario.order_cost_fixed  if order > 0 else 0.0
    var_cost   = scenario.order_cost_linear * order
    ord_cost   = fix_cost + var_cost
    hold_cost  = scenario.holding_cost  * max(0,  state2.inventory)
    if scenario.stockout_mode == "backlog":
        short_cost = scenario.shortage_cost * max(0, -state2.inventory)
    else:
        short_cost = scenario.shortage_cost * state2.lost_sales
    total_cost = ord_cost + hold_cost + short_cost

    state2.period    += 1
    state2.terminated = state2.period >= scenario.horizon

    info2 = {
        # action_period = input state.period (state2.period has advanced)
        "action_period": state.period,
        "order":        order,
        "leadtime":     leadtime,
        "received":     state2.received,
        "demand":       state2.demand,
        "lost_sales":   state2.lost_sales,
        "cost": {
            "holding":        hold_cost,
            "shortage":       short_cost,
            "order_fixed":    fix_cost,
            "order_variable": var_cost,
            "total":          total_cost,
        },
    }
    return state2, info2


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

def _run_test(scenario: InvSingleScenario, order: int = 40, episode_seed: int = 42) -> None:
    print(f"\n=== {scenario.scenario_name} | seq={scenario.event_sequence} ===")
    state, _ = init_state(scenario=scenario, episode_seed=episode_seed)
    print("Initial state:", state)
    total_cost = 0.0
    while not state.terminated:
        state1    = advance1(scenario, state)
        state, info2 = advance2(scenario, state1, order=order)
        total_cost += info2["cost"]["total"]
        print(
            f"t={info2['action_period']:2d}  order={info2['order']:3d}  "
            f"lt={info2['leadtime']}  arr={info2['received']:3d}  "
            f"d={info2['demand']:2d}  inv={state.inventory:4d}  "
            f"pipeline={state.pipeline}  cost={info2['cost']['total']:.2f}"
        )
    print(f"Total cost: {total_cost:.2f}")


if __name__ == "__main__":
    from inv_single_scenarios import test_rdo, test_rod, test_ord

    _run_test(test_rdo)
    _run_test(test_rod)
    _run_test(test_ord)

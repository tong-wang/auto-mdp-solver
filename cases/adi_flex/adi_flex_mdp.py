"""AdiFlex core MDP.

This module contains the domain-level simulator only.
It does not depend on Gym/Gymnasium.

Key components:
1. AdiFlexState: mutable simulation state (period, inventory, advance-demand
   profile).
2. State transition functions:
   a. init_state(scenario, episode_seed) — initialize an empty state.
   b. advance1(scenario, state) — execute all events before 'O' (none here;
      'O' leads the sequence, so this is the pre-order observation point).
   c. advance2(scenario, state, order_quantity, hold_back) — execute 'O', 'D',
      'F', compute costs, and advance the period counter.

The scenario configuration (AdiFlexScenario) lives in adi_flex_scenarios.py and
the stochastic primitives (the three demand generators) in
adi_flex_uncertainty.py.

Notations:
    - inventory: net inventory (negative = overdue demand backlogged)
    - due_now:   v^i_i     — unsatisfied advance demand due by this period
    - due_next:  v^{i+1}_i — unsatisfied advance demand due by next period
    - period:    0-indexed current time step

Event sequence (O, D, F):
    O — place the replenishment order; it arrives immediately (L = 0)
    D — this period's orders arrive, one draw per due date
    F — fulfil, earliest due date first, reserving `hold_back` before the
        not-yet-due class

Fulfilment (the five regions of Wang & Toktay 2008, eqs. 18-19):
    Let avail = inventory + order, and let the three due-date buckets be
        need_now   = due_now  + d_now
        need_next  = due_next + d_next
        need_later = d_later
    Serving earliest-due-date first leaves the running surpluses
        r0 = avail - need_now
        r1 = r0    - need_next
        r2 = r1    - hold_back
    and the outcome is piecewise in which of those first goes negative:
        region 1  r0 < 0            — short on what is already due; backlog
        region 2  r1 < 0            — due-now covered, due-next partly covered
        region 3  r2 < 0            — surplus below the hold-back; nothing ships early
        region 4  r2 < need_later   — hold-back retained, remainder ships early
        region 5  otherwise         — everything observed is served

    The hold-back is deliberately applied *only* before the due-later bucket:
    orders due now or next period cannot be preempted by anything that has not
    yet arrived, since an order arriving next period is due next period at the
    earliest. So protecting them would cost holding with no shortage benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adi_flex_scenarios import AdiFlexScenario


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AdiFlexState:
    """Full internal state of the simulator.

    This captures everything the simulator needs to generate transitions — it
    is the MDP state from the *simulator's* perspective, not the agent's.
    Whether a field is observable to the RL agent is NOT determined here;
    observation construction is delegated to the gym wrapper
    (adi_flex_gym.py) according to the ``observation_mode`` it is configured
    with.
    """

    # current time period (0-indexed; incremented after each advance2())
    period: int

    # True once period == scenario.horizon
    terminated: bool

    # net inventory: positive = on-hand stock, negative = overdue backlog
    inventory: int

    # advance-demand profile: unsatisfied orders not yet overdue
    due_now:  int
    due_next: int

    # within-period accumulators; reset at the start of each advance1()
    demand_now:   int
    demand_next:  int
    demand_later: int
    region:       int

    seed_salt:    int = field(repr=False)
    episode_seed: int

    def copy(self) -> AdiFlexState:
        return AdiFlexState(
            period=self.period,
            terminated=self.terminated,
            inventory=self.inventory,
            due_now=self.due_now,
            due_next=self.due_next,
            demand_now=self.demand_now,
            demand_next=self.demand_next,
            demand_later=self.demand_later,
            region=self.region,
            seed_salt=self.seed_salt,
            episode_seed=self.episode_seed,
        )


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------


def init_state(
    scenario: AdiFlexScenario,
    episode_seed: int,
) -> tuple[AdiFlexState, dict]:
    """Initialize an empty state at the start of an episode.

    The paper starts every experiment from (x, v^i, v^{i+1}) = (0, 0, 0):
    no stock, no backlog, no outstanding advance orders.
    """
    state = AdiFlexState(
        seed_salt=scenario.seed_salt,
        episode_seed=episode_seed,
        period=0,
        terminated=False,
        inventory=0,
        due_now=0,
        due_next=0,
        demand_now=0,
        demand_next=0,
        demand_later=0,
        region=0,
    )
    # info carries the time index (action_period), transition random-outcome
    # diagnostics (the three demand draws, the fulfilment region), and economic
    # quantities (cost). The physical state snapshot (inventory, due_now,
    # due_next) lives on the returned state and is not duplicated here.
    info = {
        "action_period":  state.period - 1,   # no action taken yet
        "order_quantity": 0,
        "hold_back":      0,
        "demand_now":     0,
        "demand_next":    0,
        "demand_later":   0,
        "region":         0,
        "cost": {
            "order_fixed": 0.0,
            "holding":     0.0,
            "shortage":    0.0,
            "total":       0.0,
        },
    }
    return state, info


def _apply_event_D(
    scenario: AdiFlexScenario,
    state: AdiFlexState,
) -> None:
    """D event: this period's orders arrive, one independent draw per due date."""
    state.demand_now   = scenario.demand_now.sample(state)
    state.demand_next  = scenario.demand_next.sample(state)
    state.demand_later = scenario.demand_later.sample(state)


def _apply_event_F(
    state: AdiFlexState,
    avail: int,
    hold_back: int,
) -> None:
    """F event: fulfil earliest-due-date first, reserving `hold_back` before
    the not-yet-due class. See the module docstring for the five regions."""
    need_now   = state.due_now  + state.demand_now
    need_next  = state.due_next + state.demand_next
    need_later = state.demand_later

    r0 = avail - need_now
    r1 = r0 - need_next
    r2 = r1 - hold_back

    if r0 < 0:
        state.region    = 1
        state.inventory = r0            # unfilled overdue demand is backlogged
        state.due_now   = need_next
        state.due_next  = need_later
    elif r1 < 0:
        state.region    = 2
        state.inventory = 0
        state.due_now   = -r1
        state.due_next  = need_later
    elif r2 < 0:
        state.region    = 3
        state.inventory = r1            # surplus below the hold-back: none ships early
        state.due_now   = 0
        state.due_next  = need_later
    elif r2 < need_later:
        state.region    = 4
        state.inventory = hold_back     # exactly the reserve is retained
        state.due_now   = 0
        state.due_next  = need_later - r2
    else:
        state.region    = 5
        state.inventory = r1 - need_later
        state.due_now   = 0
        state.due_next  = 0


def advance1(
    scenario: AdiFlexScenario,
    state: AdiFlexState,
) -> AdiFlexState:
    """Execute all events before 'O' in the event sequence.

    'O' leads the sequence (O, D, F), so nothing runs here — the returned state
    is the pre-order observation point. The within-period demand accumulators
    are reset so the pre-order observation never carries a stale draw.
    """
    assert not state.terminated, (
        "Episode is already finished. Call init_state() to start a new one."
    )
    state1 = state.copy()
    state1.demand_now   = 0
    state1.demand_next  = 0
    state1.demand_later = 0
    state1.region       = 0
    return state1


def advance2(
    scenario:       AdiFlexScenario,
    state:          AdiFlexState,
    order_quantity: int,
    hold_back:      int,
) -> tuple[AdiFlexState, dict]:
    """Execute 'O', 'D', 'F', compute costs, advance the period.

    Must be called with the state returned by advance1(). Both decisions are
    committed here, before the D event reveals this period's demand — the same
    information set the paper's protection-level heuristics use.
    """
    assert order_quantity >= 0, \
        f"order_quantity must be non-negative, got {order_quantity}."
    assert hold_back >= 0, \
        f"hold_back must be non-negative, got {hold_back}."

    state2 = state.copy()

    # O: the order arrives immediately (L = 0)
    avail = state2.inventory + int(order_quantity)
    # D: realize this period's three demand streams
    _apply_event_D(scenario, state2)
    # F: fulfil, earliest due date first, behind the hold-back
    _apply_event_F(state2, avail=avail, hold_back=int(hold_back))

    # costs assessed at end of period on final inventory — L(x_{i+1}) in eq (5)
    fix_cost   = scenario.order_cost_fixed if order_quantity > 0 else 0.0
    hold_cost  = scenario.holding_cost   * max(0,  state2.inventory)
    short_cost = scenario.backorder_cost * max(0, -state2.inventory)
    total_cost = fix_cost + hold_cost + short_cost

    state2.period    += 1
    state2.terminated = state2.period >= scenario.horizon

    info2 = {
        # action_period = input state.period (state2.period has advanced)
        "action_period":  state.period,
        "order_quantity": order_quantity,
        "hold_back":      hold_back,
        "demand_now":     state2.demand_now,
        "demand_next":    state2.demand_next,
        "demand_later":   state2.demand_later,
        "region":         state2.region,
        "cost": {
            "order_fixed": fix_cost,
            "holding":     hold_cost,
            "shortage":    short_cost,
            "total":       total_cost,
        },
    }
    return state2, info2


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------


def _run_test(
    scenario: AdiFlexScenario,
    order_quantity: int = 5,
    hold_back: int = 3,
    episode_seed: int = 3,
) -> None:
    print(f"\n=== {scenario.scenario_name}: {scenario.desc} ===")
    state, _ = init_state(scenario=scenario, episode_seed=episode_seed)
    total_cost = 0.0
    ordered = demanded = 0
    print("  t  order  hold  d_now  d_next  d_later  reg   inv  due_now  due_next     cost")
    while not state.terminated:
        state1 = advance1(scenario, state)
        state, info = advance2(
            scenario, state1,
            order_quantity=order_quantity, hold_back=hold_back,
        )
        total_cost += info["cost"]["total"]
        ordered    += info["order_quantity"]
        demanded   += (info["demand_now"] + info["demand_next"]
                       + info["demand_later"])
        print(
            f"{info['action_period']:3d}  {info['order_quantity']:5d}  "
            f"{info['hold_back']:4d}  {info['demand_now']:5d}  "
            f"{info['demand_next']:6d}  {info['demand_later']:7d}  "
            f"{info['region']:3d}  {state.inventory:4d}  {state.due_now:7d}  "
            f"{state.due_next:8d}  {info['cost']['total']:7.2f}"
        )
    outstanding = max(0, -state.inventory) + state.due_now + state.due_next
    served      = demanded - outstanding
    print(f"Total cost: {total_cost:.2f}")
    print(f"Conservation: ordered={ordered}  demanded={demanded}  "
          f"outstanding={outstanding}  served={served}  "
          f"on_hand={max(0, state.inventory)}  "
          f"[served + on_hand == ordered: {served + max(0, state.inventory) == ordered}]")


if __name__ == "__main__":
    from adi_flex_scenarios import scenario_exp0, scenario_exp4, scenario_exp7

    # exp4 is the Stage-0 target; the two corners exercise the degenerate ends
    _run_test(scenario_exp4)
    _run_test(scenario_exp0)
    _run_test(scenario_exp7)

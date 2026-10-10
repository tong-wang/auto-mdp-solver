"""Owmr core MDP — one warehouse, N retailers, periodic review, central control
(Dogru, de Kok & van Houtum 2010).

This module contains the domain-level simulator only. It does not depend on
Gym/Gymnasium, and it computes no reward: cost quantities travel in the
``info`` dict and the gym wrapper turns them into a reward.

State/info is represented from the *simulator's* full perspective. What is
observable to the RL agent is NOT defined here — observations are constructed
in the gym wrapper (owmr_gym.py) based on the ``observation_mode`` it is
configured with.

Key components:
1. OwmrState: period, warehouse on-hand and its l0-slot pipeline, each
   retailer's net stock (negative = backlog) and its l_rt-slot pipeline, plus
   the within-period accumulators the info dict reports.
2. State transition functions:
   a. init_state(scenario, episode_seed) — mean-cover start.
   b. advance1(scenario, state) — the two receipt events (RW at the
      warehouse, RR at the retailers); returns the pre-decision state, which
      is the agent's observation point.
   c. advance2(scenario, state, order, ship) — the order (O), the allocation
      (S) and demand (D) events, costs, and the period counter.

Event sequence (fixed): RW -> O -> RR -> S -> D
    RW — the supplier delivery due now lands at the warehouse; the warehouse
         pipeline shifts one slot
    O  — the warehouse orders ``order`` units, landing after l0 periods
    RR — each retailer's shipment due now lands; its pipeline shifts
    S  — the warehouse ships ``ship[i]`` to retailer i; a request exceeding
         on-hand is scaled down proportionally (never refused, never negative)
    D  — demand is realized at every retailer; unmet demand backlogs

advance1 applies RW and RR, advance2 applies O, S and D. O and RR touch
disjoint variables, so this is the schema's order exactly. The paper lists
the order before the warehouse receipt and the allocation before the retailer
receipt; with every lead time >= 1 the trajectories are identical, and
receipt-before-dispatch lets every pipeline width NAME its lead time with no
dead slot.

Notations:
    - warehouse: wh; retailers: rt, index i = 0..N-1 (the paper's i = 1..N)
    - the paper's echelon inventory position IP_0 is derived by
      ``echelon_position``; retailer positions by ``inventory_position``
"""

from __future__ import annotations

from dataclasses import dataclass, field

from owmr_scenarios import OwmrScenario


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OwmrState:
    """Full internal state of the simulator.

    This captures everything the simulator needs to generate transitions — it
    is the MDP state from the *simulator's* perspective, not the agent's.
    Whether a field is observable to the RL agent is NOT determined here;
    observation construction is delegated to the gym wrapper (owmr_gym.py)
    according to the ``observation_mode`` it is configured with.
    """

    period: int
    terminated: bool

    # warehouse on-hand after all period events; never negative (shipments
    # are scaled to what is on hand)
    wh_stock: float
    # wh_pipe[k] = units ordered from the supplier landing at the warehouse
    # receipt event k+1 periods from now; exactly l0 slots
    wh_pipe: list[float]

    # rt_stock[i] = net stock at retailer i; negative = backlog
    rt_stock: list[float]
    # rt_pipe[i][k] = units shipped to retailer i landing at its receipt
    # event k+1 periods from now; exactly l_rt slots per retailer
    rt_pipe: list[list[float]]

    # within-period accumulators; refreshed by advance1 / advance2 and
    # reported through info — never read by a later transition
    wh_arrival: float
    rt_arrival: list[float]
    shipped: list[float]
    ship_scale: float
    demand: list[float]

    episode_seed: int
    seed_salt: int = field(repr=False)

    def copy(self) -> "OwmrState":
        return OwmrState(
            period=self.period,
            terminated=self.terminated,
            wh_stock=self.wh_stock,
            wh_pipe=list(self.wh_pipe),
            rt_stock=list(self.rt_stock),
            rt_pipe=[list(row) for row in self.rt_pipe],
            wh_arrival=self.wh_arrival,
            rt_arrival=list(self.rt_arrival),
            shipped=list(self.shipped),
            ship_scale=self.ship_scale,
            demand=list(self.demand),
            episode_seed=self.episode_seed,
            seed_salt=self.seed_salt,
        )


# ---------------------------------------------------------------------------
# Derived views (the paper's coordinates)
# ---------------------------------------------------------------------------


def inventory_position(state: OwmrState) -> list[float]:
    """Retailer i's inventory position: net stock plus everything in flight
    toward it — the paper's IP_i."""
    return [state.rt_stock[i] + sum(state.rt_pipe[i]) for i in range(len(state.rt_stock))]


def echelon_position(state: OwmrState) -> float:
    """The warehouse's echelon inventory position, the paper's IP_0: every
    unit in the system — on hand anywhere, in flight anywhere — net of
    retailer backlog."""
    return (
        state.wh_stock
        + sum(state.wh_pipe)
        + sum(state.rt_stock)
        + sum(sum(row) for row in state.rt_pipe)
    )


def system_onhand(state: OwmrState) -> float:
    """Physical stock on hand anywhere: warehouse plus positive retailer stock."""
    return state.wh_stock + sum(max(0.0, x) for x in state.rt_stock)


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------


def init_state(scenario: OwmrScenario, episode_seed: int) -> tuple[OwmrState, dict]:
    """Mean-cover start: every pipeline slot holds one period of mean demand,
    the warehouse holds sum(mu_rt) on hand, each retailer holds mu_i.

    Chosen at Phase A over an empty start: the system begins in flow, so the
    first periods look like steady state under any sensible policy and the
    number is comparable to the paper's per-period bound after a few periods.
    It is not the optimum and hands the agent nothing.
    """
    n = scenario.n_retailers
    mu = list(scenario.mu_rt)
    mu_sys = sum(mu)
    state = OwmrState(
        period=0,
        terminated=False,
        wh_stock=float(mu_sys),
        wh_pipe=[float(mu_sys) for _ in range(scenario.l0)],
        rt_stock=[float(mu[i]) for i in range(n)],
        rt_pipe=[[float(mu[i]) for _ in range(scenario.l_rt)] for i in range(n)],
        wh_arrival=0.0,
        rt_arrival=[0.0] * n,
        shipped=[0.0] * n,
        ship_scale=1.0,
        demand=[0.0] * n,
        episode_seed=episode_seed,
        seed_salt=scenario.seed_salt,
    )
    info = {
        "action_period": state.period - 1,   # no action taken yet
        "order": 0.0,
        "ship": [0.0] * n,
        "wh_arrival": 0.0,
        "rt_arrival": [0.0] * n,
        "shipped": [0.0] * n,
        "ship_scale": 1.0,
        "demand": [0.0] * n,
        "cost": {"wh_holding": 0.0, "rt_holding": 0.0, "rt_shortage": 0.0, "total": 0.0},
    }
    return state, info


def _apply_event_RW(state: OwmrState) -> None:
    """RW: the supplier delivery due now lands; the warehouse pipeline shifts."""
    state.wh_arrival = state.wh_pipe[0]
    state.wh_stock += state.wh_arrival
    state.wh_pipe = state.wh_pipe[1:] + [0.0]


def _apply_event_O(scenario: OwmrScenario, state: OwmrState, order: float) -> None:
    """O: the order enters the warehouse pipeline's last slot, landing after
    exactly l0 receipt events. The supplier has ample stock: no clip."""
    state.wh_pipe[scenario.l0 - 1] += order


def _apply_event_RR(state: OwmrState) -> None:
    """RR: each retailer's shipment due now lands; its pipeline shifts."""
    n = len(state.rt_stock)
    state.rt_arrival = [state.rt_pipe[i][0] for i in range(n)]
    state.rt_stock = [state.rt_stock[i] + state.rt_arrival[i] for i in range(n)]
    state.rt_pipe = [state.rt_pipe[i][1:] + [0.0] for i in range(n)]


def _apply_event_S(scenario: OwmrScenario, state: OwmrState, ship: list[float]) -> None:
    """S: the allocation. Requests are scaled down proportionally when their
    sum exceeds warehouse on-hand (spec 7.2 option 2), so the warehouse never
    goes negative and no allocation is ever negative — the balance
    assumption's fiction is not in the model. Dispatched units enter each
    retailer pipeline's LAST slot, landing after exactly l_rt receipt events.
    """
    n = scenario.n_retailers
    requested = sum(ship)
    scale = min(1.0, state.wh_stock / requested) if requested > 0 else 1.0
    shipped = [ship[i] * scale for i in range(n)]
    state.wh_stock -= sum(shipped)
    last = scenario.l_rt - 1
    state.rt_pipe = [
        state.rt_pipe[i][:last] + [state.rt_pipe[i][last] + shipped[i]]
        for i in range(n)
    ]
    state.shipped = shipped
    state.ship_scale = scale


def _apply_event_D(scenario: OwmrScenario, state: OwmrState) -> None:
    """D: demand is realized at every retailer; unmet demand backlogs."""
    n = scenario.n_retailers
    state.demand = scenario.demand.sample(state)
    state.rt_stock = [state.rt_stock[i] - state.demand[i] for i in range(n)]


def advance1(scenario: OwmrScenario, state: OwmrState) -> OwmrState:
    """Execute the two receipt events (RW, RR) and return the pre-decision
    state — the agent's observation point: deliveries have landed, nothing
    has been ordered or shipped, demand has not been realized."""
    assert not state.terminated, (
        "Episode is already finished. Call init_state() to start a new one."
    )
    state1 = state.copy()
    _apply_event_RW(state1)
    _apply_event_RR(state1)
    return state1


def advance2(
    scenario: OwmrScenario,
    state: OwmrState,
    order: float,
    ship: list[float],
) -> tuple[OwmrState, dict]:
    """Execute O, S and D, compute costs, advance the period.

    Must be called with the state returned by advance1(). ``order`` is one
    non-negative quantity; ``ship`` carries one non-negative request per
    retailer, exactly n_retailers long. Feasibility of the joint request is
    the S event's business (proportional scale-down), not the caller's.
    """
    n = scenario.n_retailers
    assert order >= 0, f"order must be non-negative, got {order}."
    assert len(ship) == n, f"ship must have {n} entries, got {len(ship)}."
    assert all(q >= 0 for q in ship), f"shipments must be non-negative, got {ship}."

    state2 = state.copy()
    _apply_event_O(scenario, state2, float(order))
    _apply_event_S(scenario, state2, [float(q) for q in ship])
    _apply_event_D(scenario, state2)

    # costs on end-of-period levels (paper section 3): h0 on warehouse
    # on-hand, h0 + h_i on retailer on-hand, p_i on retailer backlog; stock
    # in transit uncharged (a policy-independent constant, section 3.1)
    wh_holding = scenario.h0 * state2.wh_stock
    rt_holding = sum(
        [(scenario.h0 + scenario.h_rt[i]) * max(0, state2.rt_stock[i]) for i in range(n)]
    )
    rt_shortage = sum(
        [scenario.p_rt[i] * max(0, -state2.rt_stock[i]) for i in range(n)]
    )
    total = wh_holding + rt_holding + rt_shortage

    state2.period += 1
    state2.terminated = state2.period >= scenario.horizon

    info = {
        "action_period": state.period,   # state2.period has advanced
        "order": float(order),
        "ship": [float(q) for q in ship],
        "wh_arrival": state2.wh_arrival,
        "rt_arrival": list(state2.rt_arrival),
        "shipped": list(state2.shipped),
        "ship_scale": state2.ship_scale,
        "demand": list(state2.demand),
        "cost": {
            "wh_holding": wh_holding,
            "rt_holding": rt_holding,
            "rt_shortage": rt_shortage,
            "total": total,
        },
    }
    return state2, info


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------


def _run_test(scenario: OwmrScenario, episode_seed: int = 42) -> None:
    """Smoke episode under a fixed policy: order mean system demand, ship one
    period of mean demand to each retailer."""
    print(
        f"\n=== {scenario.scenario_name} | N={scenario.n_retailers} "
        f"(l0,l_rt)=({scenario.l0},{scenario.l_rt}) T={scenario.horizon} "
        f"p={scenario.p_rt[0]} cv={scenario.cv_rt[0]} ==="
    )
    state, _ = init_state(scenario, episode_seed)
    total = 0.0
    while not state.terminated:
        state1 = advance1(scenario, state)
        order = scenario.mu_sys
        ship = list(scenario.mu_rt)
        state, info = advance2(scenario, state1, order, ship)
        total += info["cost"]["total"]
        if info["action_period"] < 4:
            print(
                f"t={info['action_period']:2d}  d={[round(x, 2) for x in info['demand']]}  "
                f"shipped={[round(x, 2) for x in info['shipped']]}  "
                f"wh={state.wh_stock:.2f}  rt={[round(x, 2) for x in state.rt_stock]}  "
                f"cost={info['cost']['total']:.2f}"
            )
    print(f"undiscounted total cost: {total:.2f}  (per period {total / scenario.horizon:.2f})")


if __name__ == "__main__":
    from owmr_scenarios import SCENARIOS

    for name in SCENARIOS:
        _run_test(SCENARIOS[name])

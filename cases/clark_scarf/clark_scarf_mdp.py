"""ClarkScarf core MDP — serial multi-echelon inventory (Clark & Scarf 1960).

Domain-level simulator only; no Gym/Gymnasium dependency and no reward (cost
quantities travel in the ``info`` dict).

Key components:
1. ClarkScarfState: period, physical stock per installation, and one
   in-flight pipeline vector per installation, each `leadtime` slots long.
2. State transition functions:
   a. init_state(scenario, episode_seed) — one period of cover at every stage.
   b. advance1(scenario, state) — execute the A (arrival) event; returns the
      pre-decision state, which is the agent's observation point.
   c. advance2(scenario, state, ship) — execute S (ship) and D (demand),
      compute costs, advance the period counter.

Event sequence (fixed): A -> S -> D
    A — every in-flight batch due now lands; the pipeline register shifts one
        slot toward the destination
    S — all links dispatch simultaneously, each clipped to what its source
        installation holds; the top link draws from the outside supplier and
        is unclipped
    D — retailer demand is realized; unmet demand backlogs as negative stock[0]

Coordinates
-----------
State is carried in **raw installation** coordinates; echelon stock is derived
by ``echelon_stock()``, which offsets the in-flight terms one level per
Assumption 3 (stock in transit *to* a level belongs to the echelon above). The two
are related by an invertible linear map, so this is an implementation choice
with no bearing on the model; raw was chosen because the shipping constraint is
then a per-component box (``ship_k <= stock[k]`` at the source) rather than a
coupled differencing condition.

No padding
----------
Every vector is exactly ``scenario.n_echelons`` long, and the pipeline is
``n_echelons`` rows of ``leadtime`` slots. All three widths NAME a scenario
constant in the schema, so the rendering is exactly as wide as the instance
and there is no cap on chain length, lead time, or action width (F9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from clark_scarf_scenarios import ClarkScarfScenario


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClarkScarfState:
    """Full internal state of the simulator.

    This is the MDP state from the *simulator's* perspective, not the agent's.
    What the agent observes is decided by the gym wrapper's
    ``observation_mode`` (raw installation stock, or the echelon running sums).
    """

    period: int
    terminated: bool

    # stock[k] = physical on-hand at installation k+1.
    # stock[0] is the retailer and may go negative (backlog); upper levels
    # are non-negative because no installation ever ships more than it holds.
    stock: list[float]

    # pipeline[k][s] = units in flight into installation k+1, landing after
    # s+1 more arrival events. One vector per installation, each exactly
    # `leadtime` slots long -- the schema names the lead time at the width
    # site (`length: "leadtime"`), so there is NO lead-time cap: an instance
    # renders exactly the slots it selects. Each A event drops the head and
    # appends a zero; a dispatch enters at the last slot.
    pipeline: list[list[float]]

    # within-period accumulators; refreshed each advance1/advance2
    demand: int
    arrived: list[float]
    shipped: list[float]

    seed_salt: int = field(repr=False)
    episode_seed: int = 0

    def copy(self) -> "ClarkScarfState":
        return ClarkScarfState(
            period=self.period,
            terminated=self.terminated,
            stock=list(self.stock),
            pipeline=[list(row) for row in self.pipeline],
            demand=self.demand,
            arrived=list(self.arrived),
            shipped=list(self.shipped),
            seed_salt=self.seed_salt,
            episode_seed=self.episode_seed,
        )


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def in_flight(state: ClarkScarfState) -> list[float]:
    """Total units in flight into each installation, summed over every
    pipeline slot — the per-level transit aggregate that costs and echelon
    coordinates read. The per-slot timing detail stays in ``state.pipeline``.
    """
    return [sum(row) for row in state.pipeline]


def inventory_position(state: ClarkScarfState) -> list[float]:
    """Per-level inventory position: on-hand plus everything in flight toward it.

    NOT what holding is charged on — see ``echelon_stock`` for that. Kept
    because it is the natural per-level building block of ``echelon_position``.
    """
    flight = in_flight(state)
    return [
        state.stock[k] + flight[k]
        for k in range(len(state.stock))
    ]


def echelon_stock(scenario: ClarkScarfScenario, state: ClarkScarfState) -> list[float]:
    """Echelon stock ``x_j`` for the LIVE levels — the paper's coordinates.

    Assumption 3: a level's costs are a function of the stock at that level
    "plus all other stock in the system which is actually at a lower level **or
    in transit to a lower level**". *Lower* is the operative word — stock in
    transit **to** level j is heading to j, not below it, so it belongs to
    echelon j+1 and NOT to echelon j.

    Hence the in-flight terms are offset one level relative to the on-hand
    terms. In the paper's own two-echelon instantiation, ``x_2`` = on hand at 2
    + on hand at 1 + in transit 2->1, while ``x_1`` = on hand at 1 alone and
    ``w_1`` is carried as its own coordinate.

    This is echelon **stock**, which is what holding is charged on — not
    echelon **position** ``u = x_1 + w_1 + ...``, which is what the paper's
    ``f_n(u)`` optimizes over. Conflating them double-charges pipeline stock.
    """
    flight = in_flight(state)
    out, run = [], 0.0
    for j in range(scenario.n_echelons):
        run += state.stock[j]
        if j > 0:
            run += flight[j - 1]
        out.append(run)
    return out


def echelon_position(scenario: ClarkScarfScenario, state: ClarkScarfState) -> list[float]:
    """Echelon inventory POSITION ``u_j`` for the LIVE levels.

    ``u_j`` = echelon stock at level j+1 **plus everything on order into it** —
    in the paper's two-echelon notation, ``u = x_1 + w_1``. This is the argument
    of ``f_n(u)`` in Eq. (4): the quantity a decision actually controls, and the
    quantity an echelon base-stock rule is stated on.

    Distinct from ``echelon_stock``, which is what holding is CHARGED on. The
    two differ by exactly the in-flight terms, and conflating them is what
    double-charges pipeline stock. Stage 5's readback works on this one.
    """
    pos = inventory_position(state)
    out, run = [], 0.0
    for j in range(scenario.n_echelons):
        run += pos[j]
        out.append(run)
    return out


def ship_capacity(scenario: ClarkScarfScenario, state: ClarkScarfState) -> list[float]:
    """Per-link upper bound on this period's shipment, read at the decision point.

    Link k (0-indexed, feeding installation k+1) draws from installation k+2,
    so it is capped by ``stock[k+1]`` — a per-component box. The top link
    (k == n_echelons-1) draws from the outside supplier and is capped only by
    the action-scale cap. Inert links are capped at 0.
    """
    caps = []
    for k in range(scenario.n_echelons):
        if k == scenario.n_echelons - 1:
            caps.append(float(scenario.ship_max))  # outside supplier
        else:
            caps.append(max(0.0, state.stock[k + 1]))
    return caps


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------


def init_state(
    scenario: ClarkScarfScenario,
    episode_seed: int,
) -> tuple[ClarkScarfState, dict]:
    """Start the episode with one period of cover at every stage.

    ``stock[k] = demand_mean`` and one period of cover in every pipeline
    slot (each vector is exactly ``leadtime`` long). Stated in raw
    coordinates; in echelon terms this reads
    ``x_k = (k+1) * demand_mean``. It is not the optimum, so it hands the agent
    nothing — it only starts the chain in flow rather than after an N-period
    forced stockout.
    """
    n = scenario.n_echelons
    cover = float(scenario.demand.mean())
    state = ClarkScarfState(
        period=0,
        terminated=False,
        stock=[cover] * n,
        pipeline=[[cover] * scenario.leadtime for _ in range(n)],
        demand=0,
        arrived=[0.0] * n,
        shipped=[0.0] * n,
        seed_salt=scenario.seed_salt,
        episode_seed=episode_seed,
    )
    info = {
        "action_period": state.period - 1,   # no action taken yet
        "demand": 0,
        "arrived": [0.0] * n,
        "shipped": [0.0] * n,
        "cost": {"holding": 0.0, "shortage": 0.0, "total": 0.0},
    }
    return state, info


def _apply_event_A(state: ClarkScarfState) -> None:
    """A event: each pipeline's head lands; the vector shifts one step closer."""
    state.arrived = [row[0] for row in state.pipeline]
    for k in range(len(state.stock)):
        state.stock[k] += state.pipeline[k][0]
    state.pipeline = [row[1:] + [0.0] for row in state.pipeline]


def _apply_event_S(
    scenario: ClarkScarfScenario,
    state: ClarkScarfState,
    ship: list[float],
) -> None:
    """S event: every link dispatches simultaneously.

    Each shipment is clipped to what its SOURCE installation holds at the start
    of the event, so all clips read the same pre-shipment state — a link cannot
    forward stock that another link delivers in the same period. The top link
    draws from the outside supplier and is unclipped. Dispatched units enter
    each pipeline's LAST slot, so they land exactly ``leadtime`` arrival
    events later — general in the lead time, with no upper bound.
    """
    n = scenario.n_echelons
    a = [0.0] * n
    for k in range(n):
        if k == n - 1:
            a[k] = max(0.0, float(ship[k]))                       # outside supplier
        else:
            a[k] = min(max(0.0, float(ship[k])), state.stock[k + 1])

    # debit sources: installation k+2 gives up what link k took from it.
    # The top link's source is external, so nothing is debited for it.
    for k in range(n - 1):
        state.stock[k + 1] -= a[k]

    for k in range(n):
        state.pipeline[k][scenario.leadtime - 1] += a[k]

    state.shipped = a


def _apply_event_D(
    scenario: ClarkScarfScenario,
    state: ClarkScarfState,
) -> None:
    """D event: retailer demand is realized; unmet demand backlogs."""
    state.demand = scenario.demand.sample(state)
    state.stock[0] -= state.demand


def advance1(
    scenario: ClarkScarfScenario,
    state: ClarkScarfState,
) -> ClarkScarfState:
    """Execute the A event and return the pre-decision state.

    This is the agent's observation point: arrivals have landed, nothing has
    been dispatched, and demand has not yet been realized.
    """
    assert not state.terminated, (
        "Episode is already finished. Call init_state() to start a new one."
    )
    state1 = state.copy()
    _apply_event_A(state1)
    return state1


def advance2(
    scenario: ClarkScarfScenario,
    state: ClarkScarfState,
    ship: list[float],
) -> tuple[ClarkScarfState, dict]:
    """Execute S and D, compute costs, advance the period.

    Must be called with the state returned by advance1(). ``ship`` carries one
    quantity per link, exactly ``n_echelons`` long.
    """
    assert len(ship) == scenario.n_echelons, (
        f"ship must have {scenario.n_echelons} entries, got {len(ship)}."
    )
    assert all(q >= 0 for q in ship), f"shipments must be non-negative, got {ship}."

    state2 = state.copy()
    _apply_event_S(scenario, state2, ship)
    _apply_event_D(scenario, state2)

    # Costs assessed at end of period on ECHELON STOCK (Assumption 3): a level's
    # cost covers stock at that level plus all stock at a LOWER level or IN
    # TRANSIT TO A LOWER level. So stock in transit *to* installation k+1 sits in
    # echelon k+2, not k+1 — it is charged its SOURCE's rate (h_install[k+1]) and
    # only starts paying the destination's higher rate once it lands. Stock in
    # transit to the TOP level is in no echelon's stock at all (on order, not yet
    # in the system), which the in-flight loop states directly by stopping one
    # short of the top rather than relying on a zero rate above the chain.
    n = scenario.n_echelons
    hold_cost = scenario.h_install[0] * max(0.0, state2.stock[0])
    for k in range(1, n):
        hold_cost += scenario.h_install[k] * state2.stock[k]
    flight = in_flight(state2)
    for k in range(n - 1):
        hold_cost += scenario.h_install[k + 1] * flight[k]
    short_cost = scenario.p_short * max(0.0, -state2.stock[0])
    # linear shipping/ordering cost on executed shipments (model: c_k >= 0 per
    # link; every current scenario sets 0 by design -- rendered so nonzero
    # values are expressible without structural change)
    ship_cost = sum(
        scenario.c_ship[k] * state2.shipped[k] for k in range(n)
    )
    total_cost = hold_cost + short_cost + ship_cost

    state2.period += 1
    state2.terminated = state2.period >= scenario.horizon

    info2 = {
        "action_period": state.period,   # state2.period has advanced
        "demand": state2.demand,
        "arrived": list(state2.arrived),
        "shipped": list(state2.shipped),
        "cost": {
            "holding": hold_cost,
            "shortage": short_cost,
            "shipping": ship_cost,
            "total": total_cost,
        },
    }
    return state2, info2


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------


def _run_test(scenario: ClarkScarfScenario, episode_seed: int = 42) -> None:
    """Smoke episode under a fixed 'ship one period of mean demand' policy."""
    print(
        f"\n=== {scenario.scenario_name} | N={scenario.n_echelons} "
        f"L={scenario.leadtime} T={scenario.horizon} p={scenario.p_short} ==="
    )
    state, _ = init_state(scenario, episode_seed)
    q = float(scenario.demand.mean())
    total = 0.0
    while not state.terminated:
        state1 = advance1(scenario, state)
        caps = ship_capacity(scenario, state1)
        ship = [min(q, caps[k]) for k in range(scenario.n_echelons)]
        state, info = advance2(scenario, state1, ship)
        total += info["cost"]["total"]
        if info["action_period"] < 5:
            print(
                f"t={info['action_period']:2d}  d={info['demand']:3d}  "
                f"ship={[round(x, 1) for x in info['shipped']]}  "
                f"stock={[round(x, 1) for x in state.stock]}  "
                f"ech={[round(x, 1) for x in echelon_stock(scenario, state)]}  "
                f"cost={info['cost']['total']:.2f}"
            )
    print(f"undiscounted total cost: {total:.2f}")


if __name__ == "__main__":
    from clark_scarf_scenarios import SCENARIOS

    for name in ("n2_l1_p09", "n3_l2_p09", "n4_l2_p09", "verify_tiny"):
        _run_test(SCENARIOS[name])

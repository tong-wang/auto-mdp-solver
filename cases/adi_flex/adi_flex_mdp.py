"""ADI-flex core MDP.

This module contains the domain-level simulator only.
It does not depend on Gym/Gymnasium.

State/info is represented from the *simulator's* full perspective.  What is
observable to the RL agent is NOT defined here — observations are constructed
in the gym wrapper (adi_flex_gym.py) based on the ``observation_mode`` it is
configured with.

**Two-step domain (spec §1.1, since F7).** The period's two decisions sit at
different information sets — the order is placed before the period's demand is
observed, the allocation at fulfillment after it (paper §2 event sequence) —
so the transition is split:

1. ``advance1(scenario, state, order)`` — O, R, D: place the order, receive
   the pipeline, realize the demand. Returns the post-demand **state**.
2. ``advance2(scenario, state, allocate)`` — F: the allocation decision, the
   period's charge, and the period boundary. Returns ``(next_state, info)``.

**ONE state class** covers both points, and it holds the paper's `(x_i, W_i,
V_i)` and nothing else: `advance1` returns a state of the same type, read one
event later. The order, the units received and the demand draw are transition
outcomes, so they travel in `info`, not on the state — and no cost is ever
stored, which is what keeps this layer reward-agnostic (spec §6.4).

``merge_info(info1, info2)`` assembles the period's record: each half reports
the charges its own events incurred, so the total exists only once both are in.

``valid_allocation(scenario, state, index, taken)`` is the mask logic (spec
§7.1): the feasible range of ONE allocation component given what the earlier
ones spent, owned here and only *formatted* by the gym.

**Every width is the scenario's** (F10). `adv` is `T_dl` wide, `d` is
`T_dl + 1`, the allocation is `n_alloc = T_dl` — the paper's general-T
model (§4: "there are T + 1 segments, with demand lead times ranging from 0 to
T"), not the T = 2 case its numerics are limited to. Nothing here may index a
due-offset by literal.

Notations (Wang & Toktay 2008):
    - inv  = x_i: net inventory at the pre-order review point; negative = backlog
      of OVERDUE (due-now) demand only — unmet not-yet-due demand lives in adv
    - pipe = W_i = (w_i, ..., w_{i+L-1}): pipe[k] arrives at the receive event k
      periods hence. Its dimension is the supply lead time itself, so it is
      EMPTY at L=0 (the order arrives the same period). Eq. (1): the pipeline
      shifts one position forward and the new order is inserted last
    - adv  = V_i = (v_i^i, ..., v_i^{i+T_dl-1}): unsatisfied observed demand,
      one slot per due-offset 0..T_dl-1. Width T_dl, NOT T_dl+1: demand due
      T_dl periods out can only have been observed this period, so it never
      has a carried slot. After advance1 the same field holds V_i + D_i with
      the due-now class settled into `inv`, so it stays T_dl wide and its
      slot 0 is the next period's due-now class
    - demand vector d = D_i ~ ONE `independent` draw over lambda_seg: component
      tau is Poisson(lambda_seg[tau]), independent but not identically
      distributed across offsets 0..T_dl
    - fulfillment: the due-now class is forced (it can only be served from
      stock on hand now, so it nets into inv); every class still ahead —
      offsets 1..T_dl, nearest first — is filled early UP TO THE ALLOCATION,
      the decision. §4.2's "serve due-next first, then protect" is a heuristic
      POLICY over that decision, never a dynamic (F7, and F10 again)
    - per-period cost: K*1{order>0} + h*inv+ + p*inv-  (END_OF_PERIOD)

NOTE: the allocation is a live decision only when scenario.alloc_enabled;
homogeneous instances force the MAXIMAL early fill (fill-as-early-as-possible
FCFS — provably optimal for a homogeneous customer base, paper §3; the same
policy the pre-F7 encoding wrote as sigma=0). Cost quantities travel in the
``info`` dict (info["cost"]) — reward is derived in the gym wrapper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from adi_flex_scenarios import AdiFlexScenario

# No decision CAPS live here (F9). The IR's `order_max` / `alloc_max` are
# rendering constants — `mdp.model` puts capacity limits on the replenishment
# quantity explicitly out of scope ("supply is uncapacitated, so any upper
# bound on an order is a rendering choice and never a physical one"), and
# `alloc_max` only sizes a Discrete space. Both live in adi_flex_gym.py, which
# owns the action and observation spaces. What this layer enforces is what the
# model asserts: order >= 0, and the state-dependent feasible allocation
# (valid_allocation).


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AdiFlexState:
    """Full internal state of the simulator (the paper's (x_i, W_i, V_i)).

ONE state class serves both decision points — the period boundary and
    the within-period allocation set. The fields ARE the paper's state and
    nothing else: `(x_i, W_i, V_i)` plus the period counters. What changes
    between the two points is where in the event sequence they were read, not
    their type — after ``advance1`` the demand is realized, the due-now class
    is settled into ``inv``, and ``adv`` holds the dues still ahead.

    Realized outcomes — the order, the units received, the demand draw — are
    NOT here. They describe what a transition did and are never read back, so
    they travel in ``info`` (spec §6.4).

    This captures everything the simulator needs to generate transitions — it
    is the MDP state from the *simulator's* perspective, not the agent's.
    Whether a field is observable to the RL agent is NOT determined here;
    observation construction is delegated to the gym wrapper (adi_flex_gym.py)
    according to the ``observation_mode`` it is configured with.
    """

    # current time period (0-indexed; incremented by advance2())
    period: int

    # True once period == scenario.N
    terminated: bool

    # net inventory x_i. At a period boundary: negative = overdue backlog.
    # After advance1: the due-now class has been settled against it, so a
    # negative value is exactly this period's shortfall.
    inv: int

    # supply pipeline W_i, len L; pipe[k] arrives at the R event k periods
    # hence. EMPTY at L=0, where an order arrives the period it is placed.
    pipe: tuple[int, ...]

    # advance-demand profile V_i, len T_dl: unsatisfied demand by due date,
    # nearest first. At a period boundary adv[0] is due this period; after
    # advance1 that class is settled and adv[0] is due next period.
    adv: tuple[int, ...]

    ## internal RNG state
    episode_seed: int
    seed_salt:    int = field(repr=False)


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------

def init_state(scenario: AdiFlexScenario, episode_seed: int) -> tuple[AdiFlexState, dict]:
    """Initialize state at the start of an episode: empty system, period 0."""
    state = AdiFlexState(
        period=0,
        terminated=False,
        inv=0,
        pipe=(0,) * scenario.L,
        adv=(0,) * scenario.T_dl,
        episode_seed=episode_seed,
        seed_salt=scenario.seed_salt,
    )
    info: dict = {
        "action_period": state.period - 1,  # no action taken yet
        "order":         0,
        "allocate":      (0,) * scenario.n_alloc,
        "d":             (0,) * (scenario.T_dl + 1),
        "received":      0,
        "early_fill":    0,
        "surplus":       0,
        "outstanding":   0,
        "cost":          {"order_fixed": 0.0, "holding": 0.0, "backorder": 0.0, "total": 0.0},
    }
    return state, info


def advance1(
    scenario: AdiFlexScenario,
    state:    AdiFlexState,
    order:    int,
) -> tuple[AdiFlexState, dict]:
    """O, R, D events: place the order, receive the pipeline, realize demand.

    Returns ``(state, info)`` — the post-demand state, which is the
    allocation's information set, and what these events realized. Only one
    fulfillment is forced and it happens here: demand due this period can be
    served from nothing but stock on hand now, so it settles against ``inv``.
    Everything still ahead is ``advance2``'s decision.
    """
    assert not state.terminated, "Episode already finished."
    order = int(order)
    # the model's own bound on an order: below only, at 0 (supply is
    # uncapacitated — any ceiling is the action space's, not the dynamics')
    assert order >= 0, f"order must be non-negative, got {order}."

    # O + R: eq. (1) — the pipeline shifts one position forward and the order
    # placed this period is inserted in the LAST position, so the vector stays
    # L wide. At L=0 there is no pipeline at all and the order arrives at once.
    pipe = list(state.pipe)
    if scenario.L > 0:
        received = pipe[0]
        pipe = pipe[1:] + [order]
    else:
        received = order
    inv = state.inv + received

    # D: realize the demand vector — ONE draw of the whole vector (F8: the
    # IR's `independent` family, components in index order from one rng)
    d = scenario.demand.sample(state)

    # V_i + D_i, added ELEMENTWISE BY DUE DATE, with the due-now class settled
    # straight into `inv` (F10/F11): what remains is the dues still ahead,
    # offsets 1..T_dl. The newest class d[T_dl] has no carried counterpart —
    # a demand due T_dl out can only have been observed this period — which is
    # what keeps the width at T_dl rather than growing it.
    T = scenario.T_dl
    # at T_dl = 0 nothing is ever carried and nothing is ever ahead: the whole
    # draw is due on arrival and `adv` stays empty (the no-ADI reduction)
    carried = state.adv[0] if T else 0
    inv -= carried + d[0]
    adv = (tuple(state.adv[k] + d[k] for k in range(1, T)) + (d[T],)) if T else ()

    info: dict = {
        "action_period": state.period,   # both halves process this period
        "order":         order,
        "received":      received,
        "d":             d,
        # each half reports the charges ITS events incurred; the O event's is
        # this one's. merge_info() forms the period's total from both.
        "cost":          {"order_fixed": scenario.K if order > 0 else 0.0},
    }
    return replace(state, inv=inv, pipe=tuple(pipe), adv=adv), info


def valid_allocation(
    scenario: AdiFlexScenario,
    state:    AdiFlexState,
    index:    int = 0,
    taken:    int = 0,
) -> int:
    """Largest feasible allocation to class ``index`` of a post-advance1 state.

    The feasible set for one component is {0, ..., min(free stock not yet
    spent, that class's outstanding demand)} — the state-dependent bound the
    IR's `allocate` decision declares and the gym's action mask enforces (spec
    §7.1: validity logic lives in the mdp layer). ``taken`` is what earlier
    components of the same allocation already consumed, so the components share
    one budget: the gym walks them in order, passing the running total.

    When the allocation is not a live decision (homogeneous branch), the forced
    maximal fill IS this value at every component.
    """
    return min(max(state.inv, 0) - taken, state.adv[index])


def advance2(
    scenario: AdiFlexScenario,
    state:    AdiFlexState,
    allocate: int | Sequence[int] = 0,
) -> tuple[AdiFlexState, dict]:
    """F event: fill the classes ahead early, charge the period, close it.

    Must be called with the state returned by advance1(). ``allocate`` is one
    quantity per class still ahead (due-offsets 1..T_dl, ``scenario.n_alloc``
    of them, nearest first). A scalar broadcasts to every component — the
    convention the interpreter CLI uses for a vector decision, and what the
    homogeneous branch passes.

    Total transition: components are clipped in order against the shared free
    stock, so any integer input is safe (the mask makes infeasible proposals
    unlikely, the clip makes them harmless). When ``scenario.alloc_enabled`` is
    False the maximal fill is forced regardless of the input
    (fill-as-early-as-possible, provably optimal for a homogeneous base,
    paper §3) — ``allocate`` is then **inert**, and callers should pass nothing
    rather than a literal that looks causal. ``info`` reports both sides:
    ``allocate`` as driven, ``fills`` as realized per class.
    """
    n = scenario.n_alloc
    assert len(state.adv) == scenario.T_dl, \
        f"adv is {len(state.adv)} wide, want T_dl = {scenario.T_dl}."
    a_req = ((int(allocate),) * n if isinstance(allocate, (int, np.integer))
             else tuple(int(x) for x in allocate))
    assert len(a_req) == n, f"allocate must hold {n} components, got {len(a_req)}."

    # nearest-due class first, each clipped to what the shared free stock and
    # its own outstanding demand allow
    fills: list[int] = []
    taken = 0
    for k in range(n):
        cap  = valid_allocation(scenario, state, k, taken)
        want = a_req[k] if scenario.alloc_enabled else cap
        got  = min(max(want, 0), cap)
        fills.append(got)
        taken += got

    new_inv    = state.inv - taken
    new_adv    = tuple(state.adv[k] - fills[k] for k in range(n))
    new_period = state.period + 1

    holding   = scenario.h * max(0, new_inv)
    backorder = scenario.p * max(0, -new_inv)

    next_state = replace(
        state,
        period=new_period,
        terminated=new_period >= scenario.N,
        inv=new_inv,
        adv=new_adv,
    )

    info: dict = {
        "action_period": state.period,   # the period both halves processed
        # the REQUEST and the REALIZATION, kept apart on purpose (F20). `allocate`
        # is what was driven in — and it is INERT when alloc_enabled is False,
        # where the branch forces the maximal fill and never reads it, so a
        # homogeneous record shows the caller's argument beside a fill it did
        # not cause. `fills` is the per-class truth in both branches; their
        # difference is what the shared-surplus clip took.
        "allocate":      a_req,
        "fills":         tuple(fills),
        "early_fill":    sum(fills),
        # the feasible-set coordinates the allocation (and its mask) started
        # from: all the free stock, and the nearest class's outstanding demand
        "surplus":       max(state.inv, 0),
        "outstanding":   state.adv[0] if n else 0,
        # the F / END_OF_PERIOD charges; the O event's is advance1's to report
        "cost": {"holding": holding, "backorder": backorder},
    }
    return next_state, info


def merge_info(info1: dict, info2: dict) -> dict:
    """The period's single record, assembled from the two halves.

    Both carry the same ``action_period`` and each reports the charges its own
    events incurred — ``advance1`` the O event's fixed cost, ``advance2`` the
    end-of-period holding and backorder — so the period total can only be
    formed once both are in hand.
    """
    cost = {**info1["cost"], **info2["cost"]}
    cost["total"] = cost["order_fixed"] + cost["holding"] + cost["backorder"]
    return {**info1, **info2, "cost": cost}


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

def _run_episode(
    scenario: AdiFlexScenario, order: int = 6, allocate: int = 2, episode_seed: int = 3
) -> None:
    print(f"\n=== {scenario.scenario_name}: {scenario.desc} ===")
    state, _ = init_state(scenario, episode_seed)
    total = 0.0
    while not state.terminated:
        state1, info1 = advance1(scenario, state, order)
        for k in range(scenario.n_alloc):
            assert 0 <= valid_allocation(scenario, state1, k) <= max(state1.inv, 0)
        state, info2 = advance2(scenario, state1, allocate)
        info = merge_info(info1, info2)
        total += info["cost"]["total"]
        print(
            f"  t={info['action_period']:2d}  d={info['d']}  "
            f"recv={info['received']:2d}  early={info['early_fill']}  "
            f"inv={state.inv:3d}  adv={state.adv}  cost={info['cost']['total']:.0f}"
        )
    print(f"  periods={state.period}  total cost={total:.2f}")


if __name__ == "__main__":
    from adi_flex_scenarios import SCENARIOS

    _run_episode(SCENARIOS["het_exp4"])
    _run_episode(SCENARIOS["homog_L0_T2"], allocate=0)

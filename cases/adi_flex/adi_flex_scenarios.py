"""Scenario configuration and named registry for the ADI-flex domain.

Defines AdiFlexScenario, AdiFlexScenarioSampler, all named scenario instances,
and the SCENARIOS registry. Import SCENARIOS in training and evaluation
scripts to look up a scenario by name:

    from adi_flex_scenarios import SCENARIOS
    scenario = SCENARIOS["homog_L0_T2"]

The problem (Wang & Toktay 2008, Mgmt Sci 54(4)): periodic-review inventory
with advance demand information and flexible (early) delivery. Demand observed
in period i is due up to two periods out; on-hand stock serves overdue backlog
and due-now demand first, then next-period demand (both forced fills), and may
pre-fill the farthest class early UP TO THE ALLOCATION DECISION (F7: the
decision is the allocated quantity; the paper's protection level sigma is the
parameter of its PL heuristics for this choice). Two branches, reported
separately (never head-to-head):

- homogeneous (paper §3): one demand lead time T; encoded as a one-hot rate
  vector (6 on slot T) with alloc_enabled=False (FCFS fill-as-early-as-
  possible, provably optimal). Instances homog_L{l}_T{t} reproduce Fig 4's
  (L, T) grid at N=30.
- heterogeneous (paper §4): rate vector (lambda0, lambda1, lambda2) over the
  three due-offset segments; joint (order, allocation) decision. Instances
  het_exp1..6 reproduce Table 2's demand mixes at N=12; het_exp0/het_exp7 are
  validation endpoints (no ADI / fully homogeneous).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import math

import numpy as np

from adi_flex_uncertainty import DemandVector

# structural caps shared by state layout and the IR.
# NEITHER of the two widths is here (F10). The supply pipeline's width is the
# scenario's own L (paper sec 2, eq. 1 - W_i = (w_i, ..., w_{i+L-1})), and the
# demand lead time T_dl is the scenario's own too: the IR declares it on the
# `demand` axis, so an instance may widen it. Both are per-instance state, not
# global caps; the observation pads them (adi_flex_gym.pipe_obs_slots /
# adv_obs_slots).
HORIZON_CAP = 30   # global horizon cap (= IR mdp.horizon.T); scenario N <= this


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AdiFlexScenario:
    """ADI-flex scenario parameters.

    lambda_seg holds the Poisson rate per due-offset; the single vector-valued
    generator `demand` is derived from it in __post_init__ (F8 — one draw per
    period, not one generator per offset).
    """

    # demand-model constants (scenario axis: demand)
    # arrival rate per due-offset tau in 0..T_dl — ONE vector, not one field
    # per offset (the customer base is a mix over offsets, not three demands).
    # lambda0/1/2 remain as read-only views so every caller keeps working.
    lambda_seg: tuple[float, ...]

    # structural / sizing parameters (scenario axes: leadtime, sizing)
    L: int           # supply lead time in periods (paper sweeps 0..4)
    N: int           # planning horizon in periods (30 homog / 12 het)

    # cost parameters (scenario axis: cost)
    K: float         # fixed cost per order
    h: float         # holding cost per unit per period
    p: float         # backorder penalty per unit per period (overdue only)

    # discount factor alpha. Lives on the IR's objective as `discount_factor`
    # (its schema home) rather than as a scenario constant — it is a property of
    # the objective, not a swept design axis. 1 in all paper numerics.
    discount: float = 1.0

    # categorical scenario-mode flag (customer_base branch)
    alloc_enabled: bool = True   # False = homogeneous: maximal early fill forced (FCFS)

    # stochastic model instance (derived from the rates in __post_init__):
    # ONE generator drawing the whole demand vector per period, matching the
    # IR's `independent` family — not one generator per due-offset (F8).
    demand: DemandVector = field(init=False, repr=False)

    # reproducibility
    seed_salt: int = field(default=2008, repr=False)

    # identifier and description
    scenario_name: str | None = None
    desc: str = ""

    @property
    def T_dl(self) -> int:
        """Demand lead time: the largest due-offset a customer may order at
        (the paper's T). The rate vector IS the declaration of it — one rate
        per offset 0..T_dl — so this is derived, never a second source of
        truth. IR constant `T_dl`, on the `demand` axis.
        """
        return len(self.lambda_seg) - 1

    @property
    def n_alloc(self) -> int:
        """Width of the allocation decision — one component per class whose
        fulfillment is a CHOICE.

        Only the due-now class is forced: it must be served from whatever stock
        is on hand, so it nets straight into `inv`. Everything still ahead —
        due-offsets 1..T_dl, `T_dl` classes — may be filled early or held, and
        that is the allocation. IR constant `n_alloc`.

        Serving the due-next class early is weakly optimal (nothing arriving
        later can beat it to its due date, and shipping now saves a period of
        holding), which is why paper §4.2's *heuristic* fills it first — but
        that is a policy result, not a constraint the model imposes, so it is
        not forced here. Same distinction F7 drew for the protection level.
        """
        return self.T_dl

    @property
    def n_sigma(self) -> int:
        """Width of the protection-level action of the `order_protection` mode.

        `T_dl - 1`, not `T_dl`: sigma_0 would reserve stock ahead of the
        NEAREST class still ahead, holding it while adv[0] goes unserved with
        no more-urgent competitor for it and early shipment costless — the unit
        ships next period anyway, having paid holding for nothing. Weakly
        dominated, the same argument F10 makes for filling due-next early, so
        sigma_0 is identically 0 and is not part of the action.

        It is also the paper's own count: §4.2 after eq. 20 needs sigma^{i+1}
        AND sigma^{i+2} at T = 3, i.e. T - 1 levels.
        """
        return max(self.T_dl - 1, 0)

    @property
    def sigma_max(self) -> int:
        """Cap on one protection level. DERIVED, and mirrors the IR expression.

        sigma_t is held back instead of serving adv[t], so it can only ever be
        useful for demand that arrives LATER but comes due SOONER than adv[t].
        Demand arriving s periods out at due-offset j is due before adv[t] iff
        j <= t - s, so summing over s = 1..t the protected quantity is

            sum_{j < t} (t - j) * D[j]

        a WEIGHTED sum of independent Poissons — mean sum (t-j)*lambda_j,
        variance sum (t-j)^2*lambda_j — and NOT itself Poisson, which is why
        F14's aggregate-a-tail rule is applied with weights rather than by
        scaling a per-period bound by a count.

        FOUR sigma, the action-space multiple (F14): a discrete action space is
        explored and slack is paid in sample efficiency, where a loose
        observation Box costs nothing. Reserving beyond this holds stock no
        arriving demand could preempt adv[t] for, so it pays holding while
        adv[t] goes unserved — dominated up to the tail, the same standard
        `order_max` is held to.

        Validated against the paper independently: this gives 16/14/12/8 on
        het_exp0/1/2/4 against its own plmax = min{s : P(d0 > s) < 0.001} of
        15/13/11/8 — two routes to the same tail, agreeing within 1.

        UNIFORM across components, matching the IR: a multi-decision action
        mode carries one [lo, hi] per decision, not one per component, so the
        cap is the max over t. At T_dl = 3 that over-provisions sigma_1 (10 ->
        22); a cap that cannot bind is the right kind of slack.

        DEGENERATES TO 0, and that is now sayable. The derivation returns 0
        wherever no reserve can exist — T_dl <= 1, lambda_0 = 0, and every
        homogeneous instance, where the allocation is not a live decision at
        all — which is 16 of the 30 declared instances. It used to carry a
        `max(1, .)` floor because schema v1 refused a bound with lo >= hi, so a
        discrete action with exactly ONE legal value could not be declared as
        [0, 0]; the floor was a concession to the validator and not part of the
        argument, and the IR mirrored it so the two would not diverge.
        Upstream #69 (shipped v0.9.30, this campaign's proposal) made the
        predicate type-aware — `lo == hi` is a legal discrete bound — so the
        floor is gone from both and the derivation stands as itself.
        """
        lam = self.lambda_seg
        caps = [
            math.ceil(
                sum((t - j) * lam[j] for j in range(t))
                + 4.0 * math.sqrt(sum((t - j) ** 2 * lam[j] for j in range(t)))
            )
            for t in range(1, self.T_dl)
        ]
        return max(caps, default=0)

    @property
    def order_max(self) -> int:
        """Largest order the action space needs to offer.

        From the DYNAMICS, not from measurement: an order can only ever serve
        demand that still arrives, and at most `N` periods of it do, so a
        bound on the horizon's total demand cannot bind under any policy, any
        instance, or any scenario a future round adds.

        `demand.max_total(N)`, NOT `N * demand_max` — the aggregate's own
        bound, not a per-period one multiplied up, which would inflate the tail
        term by sqrt(N).

        At FOUR sigma, not the observation sites' six: a discrete action space
        is explored, so slack here is paid for in sample efficiency, while a
        loose observation Box costs nothing. Four is spec §4.1's own example
        and still clears the largest order any solved policy places (58) by
        ~2x. A cap justified by that 58 instead would be valid only for the
        scenarios already solved — hence the horizon-demand argument, at the
        tightest multiple that argument supports.
        """
        return int(self.demand.max_total(self.N, sigmas=4.0))

    @property
    def lambda0(self) -> float:
        return self.lambda_seg[0]

    @property
    def lambda1(self) -> float:
        return self.lambda_seg[1]

    @property
    def lambda2(self) -> float:
        return self.lambda_seg[2]

    def __post_init__(self) -> None:
        self.lambda_seg = tuple(float(x) for x in self.lambda_seg)
        # T_dl >= 0. The old bound was 1, justified as "the forced cascade
        # serves due-now and due-next" — the PRE-F11 model. F11 established
        # that only the due-now class is forced and every class ahead is the
        # allocation, which makes T_dl = 0 coherent: nothing is ahead, so
        # `adv` is empty and `n_alloc` is 0. That is the paper's T = 0 (§3.3
        # runs numerics at T in {0,1,2}) and it reduces the domain to a plain
        # periodic-review (s,S) system with no advance demand information.
        assert len(self.lambda_seg) >= 1, \
            "lambda_seg needs one rate per due-offset 0..T_dl, with T_dl >= 0."
        assert all(r >= 0 for r in self.lambda_seg), "demand rates must be non-negative."
        assert sum(self.lambda_seg) > 0, "at least one segment must have positive demand."
        assert self.L >= 0, f"L must be non-negative, got {self.L}."
        assert 1 <= self.N <= HORIZON_CAP, f"N must be in [1, {HORIZON_CAP}]."
        assert self.K >= 0 and self.h >= 0 and self.p >= 0, "costs must be non-negative."
        assert 0 < self.discount <= 1, "discount must be in (0, 1]."
        # one demand process, one draw: the vector's components are the
        # due-offsets, independent but not identically distributed (F8)
        self.demand = DemandVector(rates=self.lambda_seg)


# ---------------------------------------------------------------------------
# Scenario sampler (generalist training: one instance drawn per episode)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AdiFlexScenarioSampler:
    """Samples one member AdiFlexScenario uniformly at the start of each episode.

    Members must agree on the family-level attributes exposed here
    (alloc_enabled, N) so the gym wrapper can build spaces before sampling.
    """

    members: tuple[AdiFlexScenario, ...]

    # family-level attributes (validated equal across members)
    alloc_enabled: bool = True
    N: int = 12

    # reproducibility (episode-level instance draw, stream 0)
    seed_salt: int = 2008

    # identifier and description
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        assert len(self.members) >= 1, "sampler needs at least one member."
        assert all(m.alloc_enabled == self.alloc_enabled for m in self.members), \
            "members must share alloc_enabled (one branch per sampler)."
        assert all(m.N == self.N for m in self.members), "members must share N."
        # T_dl sets the number of allocation micro-steps a period takes, and the
        # gym fixes its action/observation spaces before the first draw — so a
        # sampler whose members disagreed on it would change episode shape
        # mid-family. L may differ (the observation pads); T_dl may not.
        assert all(m.T_dl == self.members[0].T_dl for m in self.members), \
            "members must share T_dl (it sets the period's agent-step count)."

    @property
    def T_dl(self) -> int:
        return self.members[0].T_dl

    @property
    def n_alloc(self) -> int:
        return self.members[0].n_alloc

    @property
    def n_sigma(self) -> int:
        return self.members[0].n_sigma

    @property
    def sigma_max(self) -> int:
        """Family bound: one action space has to offer whichever member's."""
        return max(m.sigma_max for m in self.members)

    @property
    def order_max(self) -> int:
        """Largest order the action space needs to offer.

        From the DYNAMICS, not from measurement: an order can only ever serve
        demand that still arrives, and at most `N` periods of it do, so a
        bound on the horizon's total demand cannot bind under any policy, any
        instance, or any scenario a future round adds.

        `demand.max_total(N)`, NOT `N * demand_max` — the aggregate's own
        bound, not a per-period one multiplied up, which would inflate the tail
        term by sqrt(N).

        At FOUR sigma, not the observation sites' six: a discrete action space
        is explored, so slack here is paid for in sample efficiency, while a
        loose observation Box costs nothing. Four is spec §4.1's own example
        and still clears the largest order any solved policy places (58) by
        ~2x. A cap justified by that 58 instead would be valid only for the
        scenarios already solved — hence the horizon-demand argument, at the
        tightest multiple that argument supports.
        """
        return int(self.demand.max_total(self.N, sigmas=4.0))

    @property
    def order_max(self) -> int:
        """Family bound: one action space has to offer whichever member's."""
        return max(m.order_max for m in self.members)

    def __call__(self, episode_seed: int) -> AdiFlexScenario:
        ss  = np.random.SeedSequence([0, episode_seed, self.seed_salt])
        rng = np.random.default_rng(ss)
        m = self.members[int(rng.integers(len(self.members)))]
        # fresh copy named after the sampler (registry-key consistency); the
        # drawn member stays identifiable via desc and its constants
        return replace(m, scenario_name=self.scenario_name,
                       desc=f"drawn member: {m.scenario_name}")


# ---------------------------------------------------------------------------
# Fixed scenarios
# ---------------------------------------------------------------------------

_HOMOG_COSTS = dict(K=100.0, h=1.0, p=9.0)
_HET_COSTS   = dict(K=100.0, h=1.0, p=9.0)


def _homog(l: int, t: int) -> AdiFlexScenario:
    rates = {0: (6.0, 0.0, 0.0), 1: (0.0, 6.0, 0.0), 2: (0.0, 0.0, 6.0)}[t]
    return AdiFlexScenario(
        lambda_seg=rates,
        L=l, N=30, alloc_enabled=False, **_HOMOG_COSTS,
        scenario_name=f"homog_L{l}_T{t}",
        desc=f"homogeneous branch, supply lead time L={l}, demand lead time T={t} "
             f"(Fig 4 grid cell; lambda=6, N=30)",
    )


def _het(n: int, rates: tuple[float, float, float], note: str) -> AdiFlexScenario:
    return AdiFlexScenario(
        lambda_seg=rates,
        L=0, N=12, alloc_enabled=True, **_HET_COSTS,
        scenario_name=f"het_exp{n}",
        desc=f"heterogeneous branch, demand mix {rates} (Table 2 Exp {n}); {note}",
    )


def _het3(n: int, rates: tuple[float, float, float, float], note: str) -> AdiFlexScenario:
    """T_dl = 3 heterogeneous instance — a THIRD leaderboard, not a het_* cell.

    Two axes move at once against het_exp*: the demand lead time (2 -> 3, so
    the allocation carries n_alloc = 2 components and a period is 3 agent
    steps) and the supply lead time (0 -> 1). Costs, horizon and total arrival
    rate are held at the het_* values so the mixes stay comparable *within*
    this board.

    Outside the paper's numerics on purpose: §3.3 limits those to T = 0,1,2
    because L = 0, T = 2 is the only two-dimensional cell. The model is stated
    for general T (§4), which is what this domain renders — but neither the
    DP/AP solvers nor the PL heuristics follow (§4.2 after eq. 20 needs a
    ladder of protection levels), so these instances have **no reference
    bracket** until those are re-derived.
    """
    return AdiFlexScenario(
        lambda_seg=rates,
        L=1, N=12, alloc_enabled=True, **_HET_COSTS,
        scenario_name=f"het3_exp{n}",
        desc=f"T_dl=3 heterogeneous branch, demand mix {rates} at L=1; {note}",
    )


_HOMOG_SCENARIOS = {
    s.scenario_name: s for s in (_homog(l, t) for l in range(5) for t in range(3))
}

_TDL1_SCENARIOS = {
    "homog_L0_Tdl0": AdiFlexScenario(
        lambda_seg=(6.0,),
        L=0, N=30, alloc_enabled=False, **_HOMOG_COSTS,
        scenario_name="homog_L0_Tdl0",
        desc="homogeneous, L=0 and NO advance demand at all (T_dl=0, lambda=6 "
             "all due on arrival; N=30). The paper's T=0 (§3.3 runs numerics at "
             "T in {0,1,2}), which reduces this domain to a plain "
             "periodic-review (s,S) system — `inv_single` with a fixed order "
             "cost. `adv` is EMPTY and `n_alloc` is 0, so there is no "
             "allocation decision and no profile: `mip = inv` exactly, which "
             "makes `vec` and `vec_mip` the SAME observation. That is the "
             "point — it is the NULL CONTROL for RQ4. Any difference between "
             "the two arms here is not information, because there is no "
             "difference in what they observe.",
    ),
    "homog_L0_Tdl1": AdiFlexScenario(
        lambda_seg=(0.0, 6.0),
        L=0, N=30, alloc_enabled=False, **_HOMOG_COSTS,
        scenario_name="homog_L0_Tdl1",
        desc="homogeneous, L=0 and a demand profile exactly ONE class wide "
             "(T_dl=1, lambda=6 all due next period; N=30). The paper's Case 1, "
             "T <= L+1, where Prop 1 collapses the state to the scalar MIP — so "
             "`vec_mip` renders (u, time_to_go) with an EMPTY V~ tail and `vec` "
             "renders (inv, adv0, time_to_go). Two free state numbers against "
             "one, with the paper claiming the one suffices: the sharpest form "
             "of RQ4 this domain admits. Same demand process as homog_L0_T1, "
             "which declares T_dl=2 and so carries a constant-zero slot in both "
             "modes; this instance carries none. A separate board, not a "
             "homog_grid cell — a sampler's members must agree on T_dl.",
    ),
}


_HET_SCENARIOS = {
    s.scenario_name: s
    for s in (
        _het(0, (6.0, 0.0, 0.0), "validation endpoint: no ADI, classic (s,S) model"),
        _het(1, (5.0, 1.0, 0.0), "no crossover; heuristics are optimal here"),
        _het(2, (4.0, 1.0, 1.0), "540-sweep member"),
        _het(3, (3.0, 1.0, 2.0), "540-sweep member"),
        _het(4, (2.0, 1.0, 3.0), "base instance: mid-ADI, heuristic gaps peak"),
        _het(5, (1.0, 1.0, 4.0), "540-sweep member"),
        _het(6, (0.0, 1.0, 5.0), "no crossover; heuristics are optimal here"),
        _het(7, (0.0, 0.0, 6.0), "validation endpoint: fully homogeneous (= homog L0 T2 at N=12)"),
    )
}

_HET3_SCENARIOS = {
    s.scenario_name: s
    for s in (
        _het3(1, (3.0, 1.0, 1.0, 1.0), "front-loaded: most demand due on arrival"),
        _het3(2, (2.0, 1.0, 1.0, 2.0), "balanced ends"),
        _het3(3, (1.0, 1.0, 1.0, 3.0), "back-loaded: the far class dominates"),
        _het3(4, (0.0, 1.0, 1.0, 4.0), "no due-now demand; all crossover-eligible or near"),
        _het3(5, (0.0, 0.0, 1.0, 5.0), "two classes, both crossover-eligible"),
        _het3(6, (0.0, 0.0, 0.0, 6.0), "validation endpoint: fully homogeneous at T=3"),
    )
}

sampler_homog_grid = AdiFlexScenarioSampler(
    members=tuple(_HOMOG_SCENARIOS.values()),
    alloc_enabled=False, N=30,
    scenario_name="homog_grid",
    desc="uniform draw over the 15 homogeneous (L,T)-grid instances",
)

sampler_het_mix = AdiFlexScenarioSampler(
    members=tuple(_HET_SCENARIOS[f"het_exp{n}"] for n in range(1, 7)),
    alloc_enabled=True, N=12,
    scenario_name="het_mix",
    desc="uniform draw over het_exp1..6 (Table 2 demand mixes; endpoints excluded)",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, AdiFlexScenario | AdiFlexScenarioSampler] = {
    **_HOMOG_SCENARIOS,
    **_TDL1_SCENARIOS,
    **_HET_SCENARIOS,
    **_HET3_SCENARIOS,
    "homog_grid": sampler_homog_grid,
    "het_mix":    sampler_het_mix,
}


if __name__ == "__main__":
    for name, s in SCENARIOS.items():
        print(f"{name:14s}  {s.desc}")

"""Named scenario registry for the single-echelon inventory (InvSingle) domain.

Defines InvSingleScenario and all named scenario instances. Import SCENARIOS in
training and evaluation scripts to look up a scenario by name:

    from inv_single_scenarios import SCENARIOS
    scenario = SCENARIOS["simple"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from inv_single_uncertainty import (
    DemandGenerator,
    LeadtimeGenerator,
    EpisodeDemand,
    PoissonDemand,
    DeterministicLeadtime,
    DiscreteLeadtime,
)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InvSingleScenario:
    """Single-echelon inventory scenario configuration."""

    # valid event sequences: permutations of R, O, D with R before D
    _VALID_SEQUENCES: ClassVar[set[tuple[str, str, str]]] = {
        ('R', 'D', 'O'),
        ('R', 'O', 'D'),
        ('O', 'R', 'D'),
    }

    # planning horizon T
    horizon: int

    # demand model
    demand: DemandGenerator

    # lead time model
    leadtime: LeadtimeGenerator

    # cost parameters (paper defaults: h=0.2, b=2.0, c=1.0, K=25)
    holding_cost:      float  # h: per unit held per period
    shortage_cost:     float  # b: per unit backordered (or lost) per period
    order_cost_linear: float  # c: variable cost per unit ordered
    order_cost_fixed:  float  # K: fixed cost per order placed (0 = no fixed cost)

    # True = unfulfilled demand accumulates as backlog;
    # False = lost-sales (excess demand dropped each period)
    allow_backlog: bool = True

    # event sequence: permutation of R, O, D with R before D
    # O-R-D (default), R-D-O, R-O-D
    event_sequence: tuple[str, str, str] = ('O', 'R', 'D')

    # scenario-level seed salt (mixed with episode seed in all RNG calls)
    seed_salt: int = field(default=4729, repr=False)

    # optional human-readable name
    scenario_name: str | None = None

    # human-readable description of the scenario
    desc: str = ""

    def __post_init__(self) -> None:
        assert self.horizon >= 1,            "horizon must be >= 1."
        assert self.holding_cost >= 0,       "holding_cost must be >= 0."
        assert self.shortage_cost >= 0,      "shortage_cost must be >= 0."
        assert self.order_cost_linear >= 0,  "order_cost_linear must be >= 0."
        assert self.order_cost_fixed >= 0,   "order_cost_fixed must be >= 0."
        assert isinstance(self.demand,   DemandGenerator),   \
            "demand must be a DemandGenerator."
        assert isinstance(self.leadtime, LeadtimeGenerator), \
            "leadtime must be a LeadtimeGenerator."
        assert self.event_sequence in self._VALID_SEQUENCES, (
            f"event_sequence must be one of {self._VALID_SEQUENCES}, "
            f"got {self.event_sequence}."
        )
        # R-O-D with lt=0 is broken: O runs after R, so a lt=0 order misses
        # this period's R and is not collected until the next period.
        if self.event_sequence == ('R', 'O', 'D'):
            assert self.leadtime.min() >= 1, (
                "R-O-D sequence requires leadtime.min() >= 1; "
                "lt=0 orders cannot be received in the same period when R precedes O."
            )

# ---------------------------------------------------------------------------
# Demand model shared by all paper-derived scenarios
# ---------------------------------------------------------------------------
# Paper (Gijsbrechts et al. 2021): Gamma=5 demand values drawn uniformly from
# {10, ..., 50} with random weights; fresh distribution each episode.
_demand_paper = EpisodeDemand(support_size=5, support_low=10, support_high=50)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

scenario_simple = InvSingleScenario(
    scenario_name="simple",
    desc="zero-leadtime Poisson(10) baseline; sanity-check / fast-training",
    horizon=30,
    demand=PoissonDemand(rate=10.0),
    leadtime=DeterministicLeadtime(value=0),
    holding_cost=1.0,
    shortage_cost=9.0,
    order_cost_linear=0.0,
    order_cost_fixed=0.0,
    allow_backlog=True,
    event_sequence=("O", "R", "D"),
)

scenario_simple_single = InvSingleScenario(
    scenario_name="simple-single",
    desc="single-period (horizon=1) version of simple",
    horizon=1,
    demand=PoissonDemand(rate=10.0),
    leadtime=DeterministicLeadtime(value=0),
    holding_cost=1.0,
    shortage_cost=9.0,
    order_cost_linear=0.0,
    order_cost_fixed=0.0,
    allow_backlog=True,
    event_sequence=("O", "R", "D"),
)

scenario_simple_slt = InvSingleScenario(
    scenario_name="simple-slt",
    desc="simple with stochastic lead time L~{1,2}",
    horizon=30,
    demand=PoissonDemand(rate=10.0),
    leadtime=DiscreteLeadtime(values=[1, 2], probabilities=[1, 1]),
    holding_cost=1.0,
    shortage_cost=9.0,
    order_cost_linear=0.0,
    order_cost_fixed=0.0,
    allow_backlog=True,
    event_sequence=("O", "R", "D"),
)

scenario_simple_sltc = InvSingleScenario(
    scenario_name="simple-sltc",
    desc="simple with stochastic lead time L~{1,2,3}",
    horizon=30,
    demand=PoissonDemand(rate=10.0),
    leadtime=DiscreteLeadtime(values=[1, 2, 3], probabilities=[1, 1, 1]),
    holding_cost=1.0,
    shortage_cost=9.0,
    order_cost_linear=0.0,
    order_cost_fixed=0.0,
    allow_backlog=True,
    event_sequence=("O", "R", "D"),
)

scenario_simple_k = InvSingleScenario(
    scenario_name="simple-k",
    desc="simple with fixed ordering cost K=20",
    horizon=30,
    demand=PoissonDemand(rate=10.0),
    leadtime=DeterministicLeadtime(value=0),
    holding_cost=1.0,
    shortage_cost=9.0,
    order_cost_linear=0.0,
    order_cost_fixed=20.0,
    allow_backlog=True,
    event_sequence=("O", "R", "D"),
)

scenario_simple_lt = InvSingleScenario(
    scenario_name="simple-lt",
    desc="simple with deterministic lead time L=2",
    horizon=30,
    demand=PoissonDemand(rate=10.0),
    leadtime=DeterministicLeadtime(value=2),
    holding_cost=1.0,
    shortage_cost=9.0,
    order_cost_linear=0.0,
    order_cost_fixed=0.0,
    allow_backlog=True,
    event_sequence=("O", "R", "D"),
)


scenario_paper_det = InvSingleScenario(
    scenario_name="paper_det",
    desc="paper baseline, deterministic lead time L=3 (h=0.2, b=2, c=1, K=25)",
    horizon=30,
    demand=_demand_paper,
    leadtime=DeterministicLeadtime(value=3),
    holding_cost=0.2,
    shortage_cost=2.0,
    order_cost_linear=1.0,
    order_cost_fixed=25.0,
    allow_backlog=True,
    event_sequence=("O", "R", "D"),
)

scenario_paper_stochastic = InvSingleScenario(
    scenario_name="paper_stochastic",
    desc="paper baseline, stochastic lead time L~{2,3,4,5} (mean ≈ 3.5)",
    horizon=30,
    demand=_demand_paper,
    leadtime=DiscreteLeadtime(
        values=[2, 3, 4, 5],
        probabilities=[1, 3, 3, 1],
    ),
    holding_cost=0.2,
    shortage_cost=2.0,
    order_cost_linear=1.0,
    order_cost_fixed=25.0,
    allow_backlog=True,
    event_sequence=("O", "R", "D"),
)

scenario_paper_lost_sales = InvSingleScenario(
    scenario_name="paper_lost_sales",
    desc="lost-sales variant: unmet demand dropped, not backordered",
    horizon=30,
    demand=_demand_paper,
    leadtime=DiscreteLeadtime(
        values=[2, 3, 4, 5],
        probabilities=[1, 3, 3, 1],
    ),
    holding_cost=0.2,
    shortage_cost=2.0,
    order_cost_linear=1.0,
    order_cost_fixed=25.0,
    allow_backlog=False,
    event_sequence=("O", "R", "D"),
)

scenario_high_penalty = InvSingleScenario(
    scenario_name="high_penalty",
    desc="high shortage penalty (b=20); tests whether agent carries more safety stock",
    horizon=30,
    demand=_demand_paper,
    leadtime=DiscreteLeadtime(
        values=[2, 3, 4, 5],
        probabilities=[1, 3, 3, 1],
    ),
    holding_cost=0.2,
    shortage_cost=20.0,
    order_cost_linear=1.0,
    order_cost_fixed=25.0,
    allow_backlog=True,
    event_sequence=("O", "R", "D"),
)


scenario_no_fixed_cost = InvSingleScenario(
    scenario_name="no_fixed_cost",
    desc="no fixed ordering cost (K=0); reduces the (s,S) incentive",
    horizon=30,
    demand=_demand_paper,
    leadtime=DiscreteLeadtime(
        values=[2, 3, 4, 5],
        probabilities=[1, 3, 3, 1],
    ),
    holding_cost=0.2,
    shortage_cost=2.0,
    order_cost_linear=1.0,
    order_cost_fixed=0.0,
    allow_backlog=True,
    event_sequence=("O", "R", "D"),
)

# ---------------------------------------------------------------------------
# Test scenarios
#
# Exercise the three event sequences (R-D-O, R-O-D, O-R-D) in the smoke-test
# driver in inv_single_mdp.py. Poisson demand; not part of SCENARIOS.
# ---------------------------------------------------------------------------

_test_demand = PoissonDemand(rate=30.0)
_test_lt_123 = DiscreteLeadtime(values=[1, 2, 3], probabilities=[1, 2, 1])
_test_lt_012 = DiscreteLeadtime(values=[0, 1, 2], probabilities=[1, 2, 1])
_test_costs = dict(
    holding_cost=0.2, shortage_cost=2.0, order_cost_linear=1.0, order_cost_fixed=25.0
)

test_rdo = InvSingleScenario(
    scenario_name="test_rdo",
    desc="R-D-O event sequence",
    horizon=10,
    demand=_test_demand,
    leadtime=_test_lt_123,
    event_sequence=("R", "D", "O"),
    **_test_costs,
)

test_rod = InvSingleScenario(
    scenario_name="test_rod",
    desc="R-O-D event sequence (requires leadtime >= 1)",
    horizon=10,
    demand=_test_demand,
    leadtime=_test_lt_123,
    event_sequence=("R", "O", "D"),
    **_test_costs,
)

test_ord = InvSingleScenario(
    scenario_name="test_ord",
    desc="O-R-D event sequence (default)",
    horizon=10,
    demand=_test_demand,
    leadtime=_test_lt_012,
    event_sequence=("O", "R", "D"),
    **_test_costs,
)


# Registry mapping name → scenario for convenient lookup in training scripts.
SCENARIOS: dict[str, InvSingleScenario] = {
    s.scenario_name: s
    for s in [
        scenario_simple,
        scenario_simple_single,
        scenario_simple_slt,
        scenario_simple_sltc,
        scenario_simple_k,
        scenario_simple_lt,
        scenario_paper_det,
        scenario_paper_stochastic,
        scenario_paper_lost_sales,
        scenario_high_penalty,
        scenario_no_fixed_cost,
    ]
}


if __name__ == "__main__":
    for name, s in SCENARIOS.items():
        print(
            f"{name:30s}  horizon={s.horizon}  "
            f"lt=[{s.leadtime.min()}..{s.leadtime.max()}]  "
            f"h={s.holding_cost}  b={s.shortage_cost}  "
            f"K={s.order_cost_fixed}  backlog={s.allow_backlog}"
        )

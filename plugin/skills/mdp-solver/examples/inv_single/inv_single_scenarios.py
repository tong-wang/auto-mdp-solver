"""Named scenario registry for the single-echelon inventory domain.

Defines InvSingleScenario, the world-latent InvSingleScenarioSampler,
the InvSingleMixtureSampler combinator, named instances, and the SCENARIOS
registry. Registry values are ScenarioSources: either a concrete scenario or
a sampler callable as `source(episode_seed) -> InvSingleScenario`.

Seed scheme v2 (spec §6.3): meta-level draws (branch 0) key
leaf-first on

    [substream_id, 0, episode_seed, seed_salt]

Every meta-level drawer instance in this module carries a distinct
substream_id (module-scoped uniqueness; see _check_meta_substreams below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Union

import numpy as np

from inv_single_uncertainty import (
    DemandGenerator,
    LeadtimeGenerator,
    FixedDistributionDemand,
    PoissonDemand,
    DeterministicLeadtime,
    DiscreteLeadtime,
)

SEED_SCHEME = "v2"

_META_BRANCH = 0


def meta_key(substream_id: int, episode_seed: int, seed_salt: int) -> list[int]:
    """v2 meta seed key: [substream_id, 0, episode_seed, seed_salt] (leaf-first)."""
    return [substream_id, _META_BRANCH, episode_seed, seed_salt]


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

    # scenario-level seed salt (universal reproducibility knob, >= 1)
    seed_salt: int = field(default=4729, repr=False)

    # optional human-readable name
    scenario_name: str | None = None

    # human-readable description of the scenario
    desc: str = ""

    def __post_init__(self) -> None:
        assert self.horizon >= 1,            "horizon must be >= 1."
        assert self.seed_salt >= 1,          "seed_salt must be >= 1 (v2 rule)."
        assert self.holding_cost >= 0,       "holding_cost must be >= 0."
        assert self.shortage_cost >= 0,      "shortage_cost must be >= 0."
        assert self.order_cost_linear >= 0,  "order_cost_linear must be >= 0."
        assert self.order_cost_fixed >= 0,   "order_cost_fixed must be >= 0."
        assert isinstance(self.demand,   DemandGenerator),   \
            "demand must be a DemandGenerator."
        assert isinstance(self.leadtime, LeadtimeGenerator), \
            "leadtime must be a LeadtimeGenerator."
        # v2: source instances composed into one scenario carry distinct ids
        assert self.demand.source_id != self.leadtime.source_id, (
            f"demand and leadtime must have distinct source_ids, both are "
            f"{self.demand.source_id}."
        )
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
# Family-level demand bounds
# ---------------------------------------------------------------------------


class FamilyDemandBounds:
    """Family-level demand attributes exposed by a sampler under the same
    names a concrete generator uses (max(), mean(), is_discrete), so the gym
    wrapper and eval scripts can build spaces and print headers without
    generating a scenario first (spec §5.2 family-level attributes)."""

    def __init__(self, max_value: float, mean_value: float, is_discrete: bool) -> None:
        self._max        = max_value
        self._mean       = mean_value
        self.is_discrete = is_discrete

    def max(self) -> float:
        return self._max

    def mean(self) -> float:
        return self._mean

    def __repr__(self) -> str:
        return f"FamilyDemandBounds(max={self._max}, mean={self._mean})"


# ---------------------------------------------------------------------------
# World-latent sampler
# ---------------------------------------------------------------------------


@dataclass
class InvSingleScenarioSampler:
    """World-latent sampler: draws a per-episode demand distribution.

    A support of `support_size` integer values is drawn uniformly (without
    replacement) from {support_low, ..., support_high} with random weights;
    the returned concrete scenario holds the realized distribution as a
    FixedDistributionDemand. Pure function `episode_seed -> scenario`
    (spec §6.3): stateless, deterministic per seed.

    Family-level attributes (fixed across all generated scenarios) mirror
    InvSingleScenario field names so gyms and scripts read them uniformly.
    """

    # family-level scenario attributes
    horizon:           int
    leadtime:          LeadtimeGenerator
    holding_cost:      float
    shortage_cost:     float
    order_cost_linear: float
    order_cost_fixed:  float
    allow_backlog:     bool = True
    event_sequence:    tuple[str, str, str] = ('O', 'R', 'D')

    # latent-draw configuration (paper: Gamma=5 values from {10..50})
    support_size: int = 5
    support_low:  int = 10
    support_high: int = 50

    # meta-level child id (§4.2); distinct per drawer in this module
    substream_id: int = 0

    seed_salt: int = field(default=4729, repr=False)
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        assert self.support_size >= 1, "support_size must be >= 1."
        assert 0 <= self.support_low <= self.support_high, "support bounds invalid."
        assert self.support_size <= (self.support_high - self.support_low + 1), (
            "support_size exceeds the number of integers in "
            "[support_low, support_high]."
        )
        assert self.seed_salt >= 1, "seed_salt must be >= 1 (v2 rule)."

    @property
    def demand(self) -> FamilyDemandBounds:
        """Family-level demand bounds (the latent part of the family)."""
        return FamilyDemandBounds(
            max_value=float(self.support_high),
            mean_value=(self.support_low + self.support_high) / 2.0,
            is_discrete=True,
        )

    def __call__(self, episode_seed: int) -> InvSingleScenario:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                meta_key(self.substream_id, episode_seed, self.seed_salt)
            )
        )
        vals = rng.choice(
            np.arange(self.support_low, self.support_high + 1),
            size=self.support_size,
            replace=False,
        ).astype(int)
        raw_probs = rng.uniform(1.0, 10.0, size=self.support_size)
        return InvSingleScenario(
            scenario_name=self.scenario_name,
            desc=self.desc,
            horizon=self.horizon,
            demand=FixedDistributionDemand(vals=vals, probs=raw_probs / raw_probs.sum()),
            leadtime=self.leadtime,
            holding_cost=self.holding_cost,
            shortage_cost=self.shortage_cost,
            order_cost_linear=self.order_cost_linear,
            order_cost_fixed=self.order_cost_fixed,
            allow_backlog=self.allow_backlog,
            event_sequence=self.event_sequence,
            seed_salt=self.seed_salt,
        )


# ---------------------------------------------------------------------------
# Mixture sampler (reference implementation of the §4.3 contract)
# ---------------------------------------------------------------------------


@dataclass
class InvSingleMixtureSampler:
    """Draws one component scenario source per seed.

    Components may be fixed scenarios or samplers, including other mixtures.
    The component index is drawn on the §4.2 meta key; episode_seed is then
    delegated VERBATIM to the chosen component, so a nested component behaves
    identically standalone and embedded (standalone equivalence).
    """

    # (weight, component)
    components: list[tuple[float, "ScenarioSource"]]

    # meta-level child id (§4.2); distinct per drawer in this module
    substream_id: int = 0

    seed_salt: int = field(default=4729, repr=False)
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        assert len(self.components) >= 2, "a mixture needs >= 2 components."
        weights = np.asarray([w for w, _ in self.components], dtype=float)
        assert np.all(weights > 0), "mixture weights must be positive."
        self._probs = weights / weights.sum()
        assert self.seed_salt >= 1, "seed_salt must be >= 1 (v2 rule)."
        # family agreement: components must pose comparable problems for one gym
        horizons = {self._family_of(c).horizon for _, c in self.components}
        backlogs = {self._family_of(c).allow_backlog for _, c in self.components}
        assert len(horizons) == 1, f"components disagree on horizon: {horizons}."
        assert len(backlogs) == 1, f"components disagree on allow_backlog: {backlogs}."

    @staticmethod
    def _family_of(component: "ScenarioSource"):
        return component  # scenario and sampler expose the same family attrs

    # -- family-level attributes (same names as InvSingleScenario) ----------
    @property
    def horizon(self) -> int:
        return self._family_of(self.components[0][1]).horizon

    @property
    def allow_backlog(self) -> bool:
        return self._family_of(self.components[0][1]).allow_backlog

    @property
    def leadtime(self) -> LeadtimeGenerator:
        # bound-relevant: the component with the longest pipeline dominates
        return max(
            (self._family_of(c).leadtime for _, c in self.components),
            key=lambda lt: lt.max(),
        )

    @property
    def demand(self) -> FamilyDemandBounds:
        fams = [self._family_of(c).demand for _, c in self.components]
        return FamilyDemandBounds(
            max_value=max(f.max() for f in fams),
            mean_value=float(np.dot(self._probs, [f.mean() for f in fams])),
            is_discrete=all(f.is_discrete for f in fams),
        )

    def __call__(self, episode_seed: int) -> InvSingleScenario:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                meta_key(self.substream_id, episode_seed, self.seed_salt)
            )
        )
        k = int(rng.choice(len(self.components), p=self._probs))
        component = self.components[k][1]
        return component(episode_seed) if callable(component) else component


ScenarioSource = Union[
    InvSingleScenario, InvSingleScenarioSampler, InvSingleMixtureSampler
]


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


# ---------------------------------------------------------------------------
# Paper-derived scenario family: per-episode demand distribution
# (Gijsbrechts et al. 2021: Gamma=5 values from {10..50}, fresh each episode)
# — world-latent samplers under v2; substream ids assigned sequentially.
# ---------------------------------------------------------------------------

_paper_costs = dict(
    holding_cost=0.2, shortage_cost=2.0, order_cost_linear=1.0,
)
_paper_slt = dict(values=[2, 3, 4, 5], probabilities=[1, 3, 3, 1])

sampler_paper_stochastic = InvSingleScenarioSampler(
    scenario_name="paper_stochastic",
    desc="paper baseline, stochastic lead time L~{2,3,4,5} (mean ≈ 3.5)",
    horizon=30,
    leadtime=DiscreteLeadtime(**_paper_slt),
    order_cost_fixed=25.0,
    substream_id=0,
    **_paper_costs,
)

sampler_paper_lost_sales = InvSingleScenarioSampler(
    scenario_name="paper_lost_sales",
    desc="lost-sales variant: unmet demand dropped, not backordered",
    horizon=30,
    leadtime=DiscreteLeadtime(**_paper_slt),
    order_cost_fixed=25.0,
    allow_backlog=False,
    substream_id=1,
    **_paper_costs,
)


# ---------------------------------------------------------------------------
# Test scenarios
#
# Exercise the three event sequences (R-D-O, R-O-D, O-R-D) in the smoke-test
# driver in inv_single_mdp.py. Poisson demand; not part of SCENARIOS.
# ---------------------------------------------------------------------------

_test_lt_123 = dict(values=[1, 2, 3], probabilities=[1, 2, 1])
_test_lt_012 = dict(values=[0, 1, 2], probabilities=[1, 2, 1])
_test_costs = dict(
    holding_cost=0.2, shortage_cost=2.0, order_cost_linear=1.0, order_cost_fixed=25.0
)

test_rdo = InvSingleScenario(
    scenario_name="test_rdo",
    desc="R-D-O event sequence",
    horizon=10,
    demand=PoissonDemand(rate=30.0),
    leadtime=DiscreteLeadtime(**_test_lt_123),
    event_sequence=("R", "D", "O"),
    **_test_costs,
)

test_rod = InvSingleScenario(
    scenario_name="test_rod",
    desc="R-O-D event sequence (requires leadtime >= 1)",
    horizon=10,
    demand=PoissonDemand(rate=30.0),
    leadtime=DiscreteLeadtime(**_test_lt_123),
    event_sequence=("R", "O", "D"),
    **_test_costs,
)

test_ord = InvSingleScenario(
    scenario_name="test_ord",
    desc="O-R-D event sequence (default)",
    horizon=10,
    demand=PoissonDemand(rate=30.0),
    leadtime=DiscreteLeadtime(**_test_lt_012),
    event_sequence=("O", "R", "D"),
    **_test_costs,
)


# Registry mapping name -> scenario source for lookup in training scripts.
SCENARIOS: dict[str, ScenarioSource] = {
    s.scenario_name: s
    for s in [
        scenario_simple,
        scenario_simple_k,
        sampler_paper_stochastic,
        sampler_paper_lost_sales,
    ]
}


def _check_meta_substreams() -> None:
    """v2 module-scoped uniqueness: meta-level drawers carry distinct substream ids."""
    drawers = [s for s in SCENARIOS.values() if callable(s)]
    ids = [s.substream_id for s in drawers]
    assert len(ids) == len(set(ids)), f"duplicate meta substream ids: {sorted(ids)}"


_check_meta_substreams()


if __name__ == "__main__":
    for name, s in SCENARIOS.items():
        kind = "sampler " if callable(s) else "scenario"
        print(
            f"{name:20s} [{kind}]  horizon={s.horizon}  "
            f"lt=[{s.leadtime.min()}..{s.leadtime.max()}]  "
            f"h={s.holding_cost}  b={s.shortage_cost}  "
            f"K={s.order_cost_fixed}  backlog={s.allow_backlog}"
        )

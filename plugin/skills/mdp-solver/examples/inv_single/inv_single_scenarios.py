"""Named scenario registry for the single-echelon inventory domain.

**Composition layer only** (catalog model, IR_LAYERING_PLAN §10): this module
composes generators from ``inv_single_uncertainty`` into named scenarios and
combines whole scenarios (mixtures). It owns **no sampling** — a source's
per-episode latents live on the generator itself (``LatentDiscreteDemand`` /
``LatentPoissonDemand``), realized via ``generator.realize(episode_seed,
seed_salt)`` on the meta branch at the generator's own ``source_id``.

Registry values are ScenarioSources: a concrete ``InvSingleScenario``, an
``InvSingleScenarioSource`` (callable ``source(episode_seed) ->
InvSingleScenario`` realizing a template's latent generators), or an
``InvSingleMixtureSampler``. "Fixed scenario vs sampler" is therefore derived
from content — whether any composed generator is latent — not a class split.

Seed scheme v2 (spec §6.3): a source's latents key leaf-first on

    [source_id, 0, episode_seed, seed_salt]

(see ``inv_single_uncertainty.meta_key``). The only composition-scoped meta
drawer is the mixture, whose ``substream_id`` allocates at or above
``N_SOURCE_IDS`` so it can never collide with a source's latent stream.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import ClassVar, Union

import numpy as np

from inv_single_uncertainty import (
    N_SOURCE_IDS,
    DemandGenerator,
    LeadtimeGenerator,
    DiscreteDemand,
    PoissonDemand,
    LatentDiscreteDemand,
    LatentPoissonDemand,
    DeterministicLeadtime,
    DiscreteLeadtime,
    meta_key,
)

SEED_SCHEME = "v2"


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InvSingleScenario:
    """Single-echelon inventory scenario configuration.

    May hold latent generators (then it is a *template*: resolve it through
    an ``InvSingleScenarioSource`` before simulation) or only concrete ones
    (then it is directly runnable).
    """

    # valid event sequences: permutations of R, O, D with R before D
    _VALID_SEQUENCES: ClassVar[set[tuple[str, str, str]]] = {
        ("R", "D", "O"),
        ("R", "O", "D"),
        ("O", "R", "D"),
    }

    # planning horizon T
    horizon: int

    # demand model
    demand: DemandGenerator

    # lead time model
    leadtime: LeadtimeGenerator

    # cost parameters (paper defaults: h=0.2, b=2.0, c=1.0, K=25)
    holding_cost: float  # h: per unit held per period
    shortage_cost: float  # b: per unit backordered (or lost) per period
    order_cost_linear: float  # c: variable cost per unit ordered
    order_cost_fixed: (
        float  # K: fixed cost per order placed (0 = no fixed cost)
    )

    # "backlog"    = unfulfilled demand accumulates as negative inventory;
    # "lost_sales" = excess demand is dropped each period
    stockout_mode: str = "backlog"

    # event sequence: permutation of R, O, D with R before D
    # O-R-D (default), R-D-O, R-O-D
    event_sequence: tuple[str, str, str] = ("O", "R", "D")

    # scenario-level seed salt (universal reproducibility knob, >= 1)
    seed_salt: int = field(default=4729, repr=False)

    # optional human-readable name
    scenario_name: str | None = None

    # human-readable description of the scenario
    desc: str = ""

    def __post_init__(self) -> None:
        assert self.horizon >= 1, "horizon must be >= 1."
        assert self.seed_salt >= 1, "seed_salt must be >= 1 (v2 rule)."
        assert self.holding_cost >= 0, "holding_cost must be >= 0."
        assert self.shortage_cost >= 0, "shortage_cost must be >= 0."
        assert self.order_cost_linear >= 0, "order_cost_linear must be >= 0."
        assert self.order_cost_fixed >= 0, "order_cost_fixed must be >= 0."
        assert self.stockout_mode in {"backlog", "lost_sales"}, (
            f"stockout_mode must be 'backlog' or 'lost_sales', got "
            f"'{self.stockout_mode}'."
        )
        assert isinstance(
            self.demand, DemandGenerator
        ), "demand must be a DemandGenerator."
        assert isinstance(
            self.leadtime, LeadtimeGenerator
        ), "leadtime must be a LeadtimeGenerator."
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
        if self.event_sequence == ("R", "O", "D"):
            assert self.leadtime.min() >= 1, (
                "R-O-D sequence requires leadtime.min() >= 1; "
                "lt=0 orders cannot be received in the same period when R precedes O."
            )

    @property
    def has_latents(self) -> bool:
        return self.demand.latent or self.leadtime.latent


# ---------------------------------------------------------------------------
# Scenario source: per-episode realization of a template's latent generators
# ---------------------------------------------------------------------------


class InvSingleScenarioSource:
    """Callable ScenarioSource over a template with latent generators.

    A pure function ``episode_seed -> InvSingleScenario``: each latent
    generator realizes itself on the meta branch at its own ``source_id``
    (``generator.realize``); concrete generators pass through. All other
    attribute access (family-level bounds ``demand.max()``, ``horizon``,
    ``stockout_mode``, ``seed_salt``, ``scenario_name``, ...) delegates to the
    template, so gym wrappers and eval scripts read a source and a concrete
    scenario uniformly (spec §5.2 family-level attributes).
    """

    def __init__(self, template: InvSingleScenario) -> None:
        assert template.has_latents, (
            "InvSingleScenarioSource wraps a template with latent generators; "
            "register a fully concrete scenario directly."
        )
        self._template = template

    def __call__(self, episode_seed: int) -> InvSingleScenario:
        t = self._template
        return dataclasses.replace(
            t,
            demand=t.demand.realize(episode_seed, t.seed_salt),
            leadtime=t.leadtime.realize(episode_seed, t.seed_salt),
        )

    def __getattr__(self, item: str):
        return getattr(self._template, item)

    def __repr__(self) -> str:
        return f"InvSingleScenarioSource({self._template!r})"


# ---------------------------------------------------------------------------
# Mixture sampler (reference implementation of the §4.3 contract)
# ---------------------------------------------------------------------------


@dataclass
class InvSingleMixtureSampler:
    """Draws one component scenario source per seed.

    Components may be fixed scenarios or sources, including other mixtures.
    The component index is drawn on the §4.2 meta key; episode_seed is then
    delegated VERBATIM to the chosen component, so a nested component behaves
    identically standalone and embedded (standalone equivalence).

    The mixture is composition-scoped, so its ``substream_id`` allocates at or
    above ``N_SOURCE_IDS`` — never colliding with a source's latent stream.
    """

    # (weight, component)
    components: list[tuple[float, "ScenarioSource"]]

    # meta-level child id; >= N_SOURCE_IDS (composition-scoped drawer)
    substream_id: int = N_SOURCE_IDS

    seed_salt: int = field(default=4729, repr=False)
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        assert len(self.components) >= 2, "a mixture needs >= 2 components."
        weights = np.asarray([w for w, _ in self.components], dtype=float)
        assert np.all(weights > 0), "mixture weights must be positive."
        self._probs = weights / weights.sum()
        assert self.seed_salt >= 1, "seed_salt must be >= 1 (v2 rule)."
        assert self.substream_id >= N_SOURCE_IDS, (
            f"mixture substream_id must be >= N_SOURCE_IDS ({N_SOURCE_IDS}); "
            f"ids below that belong to the sources' latent streams."
        )
        # family agreement: components must pose comparable problems for one gym
        horizons = {self._family_of(c).horizon for _, c in self.components}
        modes = {
            self._family_of(c).stockout_mode for _, c in self.components
        }
        assert (
            len(horizons) == 1
        ), f"components disagree on horizon: {horizons}."
        assert (
            len(modes) == 1
        ), f"components disagree on stockout_mode: {modes}."

    @staticmethod
    def _family_of(component: "ScenarioSource"):
        return component  # scenario and source expose the same family attrs

    # -- family-level attributes (same names as InvSingleScenario) ----------
    @property
    def horizon(self) -> int:
        return self._family_of(self.components[0][1]).horizon

    @property
    def stockout_mode(self) -> str:
        return self._family_of(self.components[0][1]).stockout_mode

    @property
    def leadtime(self) -> LeadtimeGenerator:
        # bound-relevant: the component with the longest pipeline dominates
        return max(
            (self._family_of(c).leadtime for _, c in self.components),
            key=lambda lt: lt.max(),
        )

    @property
    def demand(self) -> DemandGenerator:
        # bound-relevant envelope over components: family-level mean()/max()
        # are exposed by concrete and latent generators alike, so return the
        # widest component's demand generator for space sizing
        fams = [self._family_of(c).demand for _, c in self.components]
        return max(fams, key=lambda f: f.max())

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
    InvSingleScenario, InvSingleScenarioSource, InvSingleMixtureSampler
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
    stockout_mode="backlog",
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
    stockout_mode="backlog",
    event_sequence=("O", "R", "D"),
)


# Every named scenario is scenario_simple + overrides (one base, everything
# layered on top — the twin of the IR's base ⊕ instances). The simple base is
# a fixed Poisson(10) demand with deterministic lead time 0; variants swap in a
# lead time, an event order, a stockout mode, or a latent demand family.

# lead-time family: a positive deterministic lead time, then its two non-default
# event-order variants. lt=2 (>= 1) makes R-O-D well-posed (an order placed
# after R is received at the next R, next period).
scenario_lt = dataclasses.replace(
    scenario_simple,
    scenario_name="lt",
    desc="simple with deterministic lead time L=2 (O-R-D)",
    leadtime=DeterministicLeadtime(value=2),
)

scenario_rdo = dataclasses.replace(
    scenario_lt,
    scenario_name="rdo",
    desc="lt with R-D-O event sequence (receive, demand, then order)",
    event_sequence=("R", "D", "O"),
)

scenario_rod = dataclasses.replace(
    scenario_lt,
    scenario_name="rod",
    desc="lt with R-O-D event sequence (receive, then order, then demand)",
    event_sequence=("R", "O", "D"),
)

scenario_lost_sales = dataclasses.replace(
    scenario_simple,
    scenario_name="lost_sales",
    desc="simple with lost sales: unmet demand dropped, not backordered",
    stockout_mode="lost_sales",
)


# ---------------------------------------------------------------------------
# Demand-latent family: scenario_simple with a per-episode latent demand
# distribution and a stochastic lead time (generator-owned latents).
#   - discrete: a 3-point support drawn from {8..12} with random weights
#   - poisson : a Poisson rate drawn from a Gamma(alpha, beta) prior
# Same economics as simple; twins of the IR's `discrete`/`poisson` instances.
# The demand generators share the demand slot's stream (source_id 0): scenarios
# sharing a latent recipe see identical draws per episode_seed (CRN).
# ---------------------------------------------------------------------------

_discrete_demand = LatentDiscreteDemand(support_size=3, support_low=8, support_high=12)
_poisson_demand = LatentPoissonDemand(alpha=9.0, beta=0.3)
_slt = DiscreteLeadtime(values=[2, 3, 4, 5], probabilities=[1, 3, 3, 1])

source_discrete = InvSingleScenarioSource(dataclasses.replace(
    scenario_simple, scenario_name="discrete",
    desc="3-point discrete demand, stochastic lead time L~{2,3,4,5}",
    demand=_discrete_demand, leadtime=_slt,
))

source_poisson = InvSingleScenarioSource(dataclasses.replace(
    scenario_simple, scenario_name="poisson",
    desc="Poisson demand, hidden rate lambda ~ Gamma(9, 0.3) (mean 30); stochastic lead time",
    demand=_poisson_demand, leadtime=_slt,
))

source_discrete_lost_sales = InvSingleScenarioSource(dataclasses.replace(
    scenario_simple, scenario_name="discrete_lost_sales",
    desc="discrete demand, stochastic lead time, lost sales",
    demand=_discrete_demand, leadtime=_slt, stockout_mode="lost_sales",
))

source_poisson_lost_sales = InvSingleScenarioSource(dataclasses.replace(
    scenario_simple, scenario_name="poisson_lost_sales",
    desc="Poisson demand, stochastic lead time, lost sales",
    demand=_poisson_demand, leadtime=_slt, stockout_mode="lost_sales",
))

# cross-family mixture (spec §5.3): nature picks the demand regime per
# episode — the components differ in demand FAMILY (discrete vs poisson),
# which the generator-owned-latent design makes free: each component's
# generators realize on their own slot streams, the mixture only picks.
# Twin of the IR's `mix_demand` mixture (substream_id = N_SOURCE_IDS).
source_mix_demand = InvSingleMixtureSampler(
    scenario_name="mix_demand",
    desc="50/50 demand-regime pick per episode: discrete vs poisson",
    components=[(0.5, source_discrete), (0.5, source_poisson)],
    substream_id=N_SOURCE_IDS,
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
    holding_cost=0.2,
    shortage_cost=2.0,
    order_cost_linear=1.0,
    order_cost_fixed=25.0,
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
        scenario_lt,
        scenario_rdo,
        scenario_rod,
        scenario_lost_sales,
        source_discrete,
        source_poisson,
        source_discrete_lost_sales,
        source_poisson_lost_sales,
        source_mix_demand,
    ]
}


def _check_registry() -> None:
    """Every callable source resolves seed 0 to a fully concrete scenario;
    a registered plain scenario must already be concrete."""
    for name, s in SCENARIOS.items():
        resolved = s(0) if callable(s) else s
        assert isinstance(resolved, InvSingleScenario) and not resolved.has_latents, (
            f"registry entry {name!r} does not resolve to a concrete scenario"
        )


_check_registry()


if __name__ == "__main__":
    for name, s in SCENARIOS.items():
        kind = "source  " if callable(s) else "scenario"
        # cost attrs are per-component on a mixture — print them only where
        # the entry exposes them (scenario / source; mixture shows components)
        costs = (
            f"h={s.holding_cost}  b={s.shortage_cost}  K={s.order_cost_fixed}"
            if hasattr(s, "holding_cost")
            else f"components={len(s.components)}"
        )
        print(
            f"{name:20s} [{kind}]  horizon={s.horizon}  "
            f"lt=[{s.leadtime.min()}..{s.leadtime.max()}]  "
            f"{costs}  stockout={s.stockout_mode}"
        )

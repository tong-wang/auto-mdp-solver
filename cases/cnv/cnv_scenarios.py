"""Named scenario registry for the censored-newsvendor domain.

**Composition layer only** (catalog model): this module composes generators
from ``cnv_uncertainty`` into named scenarios. It owns **no sampling** — the
problem's one world latent (the arrival rate lambda) lives on the demand
generator (``LatentPoissonDemand``), realized via ``generator.realize(
episode_seed, seed_salt)`` on the meta branch at the generator's own
``source_id``.

Every registered entry is a ``CnvScenarioSource``: a callable
``source(episode_seed) -> CnvScenario`` realizing the template's latent rate.
There is no concrete (latent-free) instance in the paper's study — the
learning problem *is* the hidden rate — so "fixed scenario vs source" here is
always source, derived from content (``has_latents``) rather than declared.

Seed scheme v2 (spec §6.3): the rate draws leaf-first on

    [source_id, 0, episode_seed, seed_salt]

(see ``cnv_uncertainty.meta_key``). Instances sharing the prior share lambda
draws per seed — common random numbers across the design grid.

Paper: Jain, Rudi & Wang (2015), Operations Research 63(1), §5.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Union

from cnv_uncertainty import (
    DemandGenerator,
    EpochGenerator,
    LatentPoissonDemand,
    UniformEpochs,
)

SEED_SCHEME = "v2"

# the IR's arrivals_buffer: uniform epochs drawn per period, hence the demand
# cap — the 1e-8 upper quantile of the marginal per-period demand of the widest
# grid cell (1/beta = 16, E[lambda] = 50). A rendering envelope, never physics.
ARRIVALS_BUFFER = 402


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CnvScenario:
    """Censored-newsvendor scenario configuration.

    Holds the latent demand generator (then it is a *template*: resolve it
    through a ``CnvScenarioSource`` before simulation) or its realized
    concrete form (then it is directly runnable).
    """

    # structural / sizing parameters
    horizon: int              # N ordering periods

    # stochastic model instances
    demand: DemandGenerator   # arrivals per period; carries the hidden rate
    epochs: EpochGenerator    # where in the period the arrivals fall

    # economics (numerical -> grid axes)
    price: float              # r: revenue per unit sold
    unit_cost: float          # c: cost per unit stocked

    # prior on the arrival rate (numerical -> grid axes); shape = rate * mean
    prior_rate: float         # beta: Gamma rate of the prior on lambda
    prior_mean: float         # E[lambda] = alpha / beta

    # reproducibility (universal knob, >= 1 under v2)
    seed_salt: int = field(default=4729, repr=False)

    # identifier and description
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        assert self.horizon >= 1, "horizon must be >= 1."
        assert self.seed_salt >= 1, "seed_salt must be >= 1 (v2 rule)."
        assert self.price > 0, "price must be positive."
        assert 0 < self.unit_cost < self.price, "unit_cost must lie in (0, price)."
        assert self.prior_rate > 0, "prior_rate must be positive."
        assert self.prior_mean > 0, "prior_mean must be positive."
        assert isinstance(self.demand, DemandGenerator), "demand must be a DemandGenerator."
        assert isinstance(self.epochs, EpochGenerator), "epochs must be an EpochGenerator."
        assert self.demand.source_id != self.epochs.source_id, (
            f"demand and epochs must have distinct source_ids, both are "
            f"{self.demand.source_id}."
        )
        # the prior the scenario states and the prior the generator draws from
        # are one fact, kept in two places for the belief observation modes
        if isinstance(self.demand, LatentPoissonDemand):
            assert self.demand.alpha == self.prior_shape and self.demand.beta == self.prior_rate, (
                f"demand generator prior ({self.demand.alpha}, {self.demand.beta}) "
                f"disagrees with the scenario's ({self.prior_shape}, {self.prior_rate})."
            )

    # -- derived quantities --------------------------------------------------

    @property
    def prior_shape(self) -> float:
        """Gamma shape alpha = prior_rate * prior_mean (the IR's expression)."""
        return self.prior_rate * self.prior_mean

    @property
    def critical_ratio(self) -> float:
        """The newsvendor ratio (r - c) / r."""
        return (self.price - self.unit_cost) / self.price

    @property
    def arrivals_buffer(self) -> int:
        """Epochs drawn per period = the demand cap (the IR's arrivals_buffer)."""
        return self.epochs.size

    @property
    def has_latents(self) -> bool:
        return self.demand.latent or self.epochs.latent


# ---------------------------------------------------------------------------
# Scenario source: per-episode realization of the template's latent rate
# ---------------------------------------------------------------------------


class CnvScenarioSource:
    """Callable ScenarioSource over a template with a latent demand generator.

    A pure function ``episode_seed -> CnvScenario``: the latent generator
    realizes itself on the meta branch at its own ``source_id``; concrete
    generators pass through. All other attribute access (``horizon``,
    ``price``, ``prior_rate``, ``demand.max()``, ``scenario_name``, ...)
    delegates to the template, so gym wrappers and eval scripts read a source
    and a concrete scenario uniformly (spec §5.2 family-level attributes).
    """

    def __init__(self, template: CnvScenario) -> None:
        assert template.has_latents, (
            "CnvScenarioSource wraps a template with latent generators; "
            "register a fully concrete scenario directly."
        )
        self._template = template

    def __call__(self, episode_seed: int) -> CnvScenario:
        t = self._template
        return dataclasses.replace(
            t,
            demand=t.demand.realize(episode_seed, t.seed_salt),
            epochs=t.epochs.realize(episode_seed, t.seed_salt),
        )

    def __getattr__(self, item: str):
        return getattr(self._template, item)

    def __repr__(self) -> str:
        return f"CnvScenarioSource({self._template!r})"


ScenarioSource = Union[CnvScenario, CnvScenarioSource]


# ---------------------------------------------------------------------------
# Instances — the paper's §5 study (twins of the IR's base + instances)
# ---------------------------------------------------------------------------


def make_source(
    scenario_name: str,
    desc: str,
    *,
    horizon: int,
    unit_cost: float,
    prior_rate: float,
    prior_mean: float,
    price: float = 2.0,
    arrivals_buffer: int = ARRIVALS_BUFFER,
    seed_salt: int = 4729,
) -> CnvScenarioSource:
    """Compose one template from the paper's parameters and wrap it.

    The generators are built from the same expressions the IR uses
    (``prior_rate * prior_mean`` for the shape), so the differential adapter
    and the registry cannot disagree on the prior.
    """
    template = CnvScenario(
        scenario_name=scenario_name,
        desc=desc,
        horizon=horizon,
        demand=LatentPoissonDemand(alpha=prior_rate * prior_mean, beta=prior_rate, source_id=0),
        epochs=UniformEpochs(size=arrivals_buffer, source_id=1),
        price=price,
        unit_cost=unit_cost,
        prior_rate=prior_rate,
        prior_mean=prior_mean,
        seed_salt=seed_salt,
    )
    return CnvScenarioSource(template)


# base: the paper's Fig. 5(a) cell (alpha = 10, beta = 1, c = 1), N = 4
source_base = make_source(
    "base",
    "paper Fig. 5(a) cell: N=4, r=2, c=1, prior Gamma(10, 1) (E[lambda]=10, 1/beta=1)",
    horizon=4, unit_cost=1.0, prior_rate=1.0, prior_mean=10.0,
)

# the Fig. 3(a)/4(a) family: alpha = 10/16, beta = 1/16 (E[lambda] = 10, 1/beta = 16)
source_fig3_c10 = make_source(
    "fig3_c10",
    "paper Fig. 3(a)/4(a) family at newsvendor ratio 0.5: N=4, c=1.0, prior Gamma(10/16, 1/16)",
    horizon=4, unit_cost=1.0, prior_rate=0.0625, prior_mean=10.0,
)
source_fig3_c15 = make_source(
    "fig3_c15",
    "paper Fig. 3(a)/4(a) family at newsvendor ratio 0.25: N=4, c=1.5, prior Gamma(10/16, 1/16)",
    horizon=4, unit_cost=1.5, prior_rate=0.0625, prior_mean=10.0,
)

# the Fig. 6 long-horizon myopic simulations (separate leaderboards: N = 100)
source_fig6a = make_source(
    "fig6a",
    "paper Fig. 6(a): N=100, c=1.0, prior Gamma(10/16, 1/16); the myopic-policy simulation",
    horizon=100, unit_cost=1.0, prior_rate=0.0625, prior_mean=10.0,
)
source_fig6b = make_source(
    "fig6b",
    "paper Fig. 6(b): N=100, c=1.5, prior Gamma(10/16, 1/16); the myopic-policy simulation",
    horizon=100, unit_cost=1.5, prior_rate=0.0625, prior_mean=10.0,
)


# Registry mapping name -> scenario source for lookup in training scripts.
# DIAGNOSTIC scenarios (#E58; not in the IR, never a board target): fig6a's costs and
# horizon with the rate pinned by a degenerate prior, Gamma(1e6 * lam, 1e6) — sd(lam) =
# sqrt(lam / 1e6). Nothing is latent, nothing is inferred: 100 rounds of a standard
# newsvendor at a known rate, whose optimum is ONE constant order. If PPO cannot solve
# this, the estimator is the problem on its own; if it can, the trouble is in reading
# the rate (#E53-#E57 could not tell those apart).
_PINNED_RATE = 1.0e6
source_fixed_low = make_source(
    "fixed_low", "diagnostic: rate pinned at 0.3 (optimal order 0), fig6a costs, N=100",
    horizon=100, unit_cost=1.0, prior_rate=_PINNED_RATE, prior_mean=0.3)
source_fixed_high = make_source(
    "fixed_high", "diagnostic: rate pinned at 80 (optimal order 80), fig6a costs, N=100",
    horizon=100, unit_cost=1.0, prior_rate=_PINNED_RATE, prior_mean=80.0)
DIAGNOSTIC_SCENARIOS = ("fixed_low", "fixed_high")

SCENARIOS: dict[str, ScenarioSource] = {
    s.scenario_name: s
    for s in [
        source_base,
        source_fig3_c10,
        source_fig3_c15,
        source_fig6a,
        source_fig6b,
        source_fixed_low,
        source_fixed_high,
    ]
}


def _check_registry() -> None:
    """Every callable source resolves seed 0 to a fully concrete scenario;
    a registered plain scenario must already be concrete."""
    for name, s in SCENARIOS.items():
        resolved = s(0) if callable(s) else s
        assert isinstance(resolved, CnvScenario) and not resolved.has_latents, (
            f"registry entry {name!r} does not resolve to a concrete scenario"
        )


_check_registry()


if __name__ == "__main__":
    for name, s in SCENARIOS.items():
        kind = "source  " if callable(s) else "scenario"
        realized = s(0) if callable(s) else s
        print(
            f"{name:10s} [{kind}]  N={s.horizon}  r={s.price}  c={s.unit_cost}  "
            f"prior=Gamma({s.prior_shape:g}, {s.prior_rate:g})  E[lambda]={s.prior_mean:g}  "
            f"buffer={s.arrivals_buffer}  lambda(seed 0)={realized.demand.rate:.3f}"
        )

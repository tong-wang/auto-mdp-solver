"""Scenario configuration and named registry for the dynamic pricing domain.

Defines DynamicPricingScenario and all named scenario instances. Import
SCENARIOS in training and evaluation scripts to look up a scenario by name:

    from dynamic_pricing_scenarios import SCENARIOS
    scenario = SCENARIOS["simple"]

The problem (Gallego & van Ryzin 1994): sell a fixed initial stock n0 over a
finite selling horizon by setting a price each period; price-sensitive Poisson
demand, no reorder, no backlog; maximize expected revenue. The continuous-time
intensity-control problem is discretized into `horizon` equal periods of
length dt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dynamic_pricing_uncertainty import PoissonArrivals


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DynamicPricingScenario:
    """Dynamic pricing scenario parameters.

    a / alpha / dt define the demand model lambda(p) = a * exp(-alpha * p)
    with per-period arrivals ~ Poisson(lambda(p) * dt); the composed generator
    is derived into ``demand`` in __post_init__.
    """

    # structural / sizing parameters
    horizon: int   # T: number of decision periods (dt = horizon_length / T)
    n0:      int   # initial stock

    # demand-model constants (scenario axis: demand)
    a:     float   # demand scale
    alpha: float   # price sensitivity
    dt:    float   # period length; discretization granularity, not a scenario axis

    # salvage (scenario axis: salvage)
    q: float       # salvage value per unsold unit at the horizon (0 WLOG per the paper)

    # structural flags (v1 supports only the paper's no-backlog / no-reorder case)
    allow_backlog: bool = False
    allow_reorder: bool = False

    # stochastic model instance (derived from a / alpha / dt in __post_init__)
    demand: PoissonArrivals = field(init=False, repr=False)

    # reproducibility
    seed_salt: int = field(default=1994, repr=False)

    # identifier and description
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        assert self.horizon >= 1, "horizon must be >= 1."
        assert self.n0 >= 1,      "n0 must be >= 1."
        assert self.a > 0,        "a must be positive."
        assert self.alpha > 0,    "alpha must be positive."
        assert self.dt > 0,       "dt must be positive."
        assert self.q >= 0,       "q must be non-negative."
        assert not self.allow_backlog, "backlogging is not supported (demand at zero stock is lost)."
        assert not self.allow_reorder, "reordering is not supported (inventory is non-increasing)."
        self.demand = PoissonArrivals(a=self.a, alpha=self.alpha, dt=self.dt)


# ---------------------------------------------------------------------------
# Fixed scenarios
# ---------------------------------------------------------------------------

scenario_simple = DynamicPricingScenario(
    horizon=50, n0=25,
    a=100.0, alpha=1.0, dt=0.02, q=0.0,
    scenario_name="simple",
    desc="demand high relative to stock (a=100, n0=25, unit horizon); GvR Fig 1 bottom case",
)

scenario_ample_stock = DynamicPricingScenario(
    horizon=50, n0=25,
    a=20.0, alpha=1.0, dt=0.02, q=0.0,
    scenario_name="ample_stock",
    desc="demand low relative to stock (a=20): run-out is unlikely, myopic price ~ optimal",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, DynamicPricingScenario] = {
    "simple":      scenario_simple,
    "ample_stock": scenario_ample_stock,
}


if __name__ == "__main__":
    for name, s in SCENARIOS.items():
        print(f"{name:12s}  {s}")
        print(f"{'':12s}  demand={s.demand}  lambda*={s.demand.max_intensity():.3f}")

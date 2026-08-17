"""ClarkScarf scenario definitions and the SCENARIOS / GRIDS registries.

A ``ClarkScarfScenario`` is one fully specified problem instance: chain length,
lead time, horizon, the demand generator, and the cost structure. It composes
the stochastic primitives from ``clark_scarf_uncertainty`` and owns no sampling
of its own.

Dependency order: clark_scarf_uncertainty <- clark_scarf_scenarios <- clark_scarf_mdp

No padding
----------
Every state and action vector is exactly ``n_echelons`` long. The schema names
the chain length at all three width sites — ``stock`` is
``length: "n_echelons"``, ``pipe`` is ``length: ["n_echelons", "leadtime"]``,
and the ``ship`` decision is ``dim: "n_echelons"`` — so an instance renders
precisely the links it has. The former ``N_LEVELS_MAX = 4`` padding is gone
(F9): it existed only because ``Decision.dim`` could not name a constant, and
upstream #33/#36 lifted that.

``h_install`` and ``c_ship`` remain declared four long. Entries at index >=
``n_echelons`` are simply never read — that is a per-instance *data* tail, not
a rendering width, and it costs nothing because no expression indexes past the
chain length.

Echelon cost accounting
-----------------------
``h_install[k]`` is the **cumulative** echelon rate for position k:
``H_k = sum of h_j for j >= k+1``. Charging position k the cumulative rate on
its own stock is algebraically identical to charging each echelon its own rate
``h_k`` on echelon stock ``x_k = sum(stock[0:k+1])`` — the paper's Assumption 3.
Stock in transit to the TOP level is in no echelon's stock at all (on order,
not yet in the system), which the holding rule states directly by summing the
in-flight terms over ``range(n_echelons - 1)`` rather than relying on a zero
rate above the chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from clark_scarf_uncertainty import DemandGenerator, PoissonDemand



@dataclass(slots=True)
class ClarkScarfScenario:
    """One fully specified serial multi-echelon instance."""

    scenario_name: str
    horizon: int

    # structure
    n_echelons: int          # live installations; level 1 = retailer, level N = top
    leadtime: int            # transit periods on EVERY link

    # uncertainty
    demand: DemandGenerator = field(default_factory=lambda: PoissonDemand(10.0))

    # economics
    h_install: tuple[float, ...] = (2.0, 1.0, 0.5, 0.0)   # cumulative echelon rates
    # c_ship[k] = per-unit cost of shipping into installation k+1 (top link =
    # ordering from the outside supplier). Model domain: real >= 0 per link;
    # every current scenario sets 0 -- a DESIGN convenience, not model
    # structure (schema mdp.model.quantities.c_ship, and the `narrowed`
    # citation on each decision).
    c_ship: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    p_short: float = 9.0                                   # retailer backlog penalty
    ship_max: float = 200.0                                # action-scale cap

    seed_salt: int = 1

    def __post_init__(self) -> None:
        assert self.n_echelons >= 2, (
            f"n_echelons={self.n_echelons} must be >= 2 (the model's domain: a "
            f"chain needs a retailer and at least one level above it). There is "
            f"no upper bound -- `stock`, `pipe` and the `ship` decision all NAME "
            f"n_echelons at their width sites, so the rendering is exactly as "
            f"wide as the instance."
        )
        assert self.leadtime >= 1, (
            f"leadtime={self.leadtime} must be >= 1 (the model's domain). There "
            f"is no upper bound: each pipeline vector is `length: leadtime` in "
            f"the schema, so the rendering carries exactly the slots the "
            f"instance selects."
        )
        assert self.horizon >= 1, "horizon must be >= 1."
        assert len(self.h_install) >= self.n_echelons, (
            f"h_install must have at least n_echelons={self.n_echelons} entries, "
            f"got {len(self.h_install)}. Entries beyond that are never read."
        )
        assert all(
            self.h_install[k] >= self.h_install[k + 1]
            for k in range(self.n_echelons - 1)
        ), (
            "h_install must be non-increasing up the chain (cumulative echelon "
            f"rates), got {self.h_install}."
        )
        assert self.p_short >= 0, "p_short must be non-negative."
        assert len(self.c_ship) >= self.n_echelons, (
            f"c_ship must have at least n_echelons={self.n_echelons} entries, "
            f"got {len(self.c_ship)}. Entries beyond that are never read."
        )
        assert all(c >= 0 for c in self.c_ship), (
            f"unit shipping costs must be non-negative, got {self.c_ship}."
        )
        self.c_ship = tuple(float(x) for x in self.c_ship)
        self.h_install = tuple(float(x) for x in self.h_install)

    @property
    def has_latents(self) -> bool:
        """True if any composed generator draws a per-episode world latent.

        Always False for the shipped candidate — Clark & Scarf hold the demand
        distribution fixed across the horizon.
        """
        return bool(getattr(self.demand, "latent", False))


class ClarkScarfScenarioSource:
    """Callable that resolves a latent-bearing template to a concrete scenario.

    Deliberately a *separate* class: ``ClarkScarfScenario`` must NOT be
    callable, or "did this resolve to a concrete scenario?" can never be
    answered — a self-returning ``__call__`` leaves the result still callable
    and indistinguishable from an unresolved source.

    This domain ships no latents (Clark & Scarf hold the demand distribution
    fixed across the horizon), so nothing constructs this today. It exists so
    that appending a latent-bearing demand candidate later needs no change to
    the layering: ``SCENARIOS`` entries simply become sources, and every
    consumer already handles both via ``source(seed) if callable(source)``.
    """

    def __init__(self, template: ClarkScarfScenario) -> None:
        assert template.has_latents, (
            "ClarkScarfScenarioSource wraps a latent-bearing template; a "
            "concrete scenario should be used directly."
        )
        self.template = template

    # family-level attributes consumers read before resolving (space sizing)
    @property
    def scenario_name(self) -> str:
        return self.template.scenario_name

    @property
    def horizon(self) -> int:
        return self.template.horizon

    @property
    def n_echelons(self) -> int:
        return self.template.n_echelons

    @property
    def leadtime(self) -> int:
        return self.template.leadtime

    def __call__(self, episode_seed: int) -> ClarkScarfScenario:
        t = self.template
        resolved = t.demand.realize(episode_seed, t.seed_salt)
        return ClarkScarfScenario(
            scenario_name=t.scenario_name,
            horizon=t.horizon,
            n_echelons=t.n_echelons,
            leadtime=t.leadtime,
            demand=resolved,
            h_install=t.h_install,
            p_short=t.p_short,
            ship_max=t.ship_max,
            seed_salt=t.seed_salt,
        )


# ---------------------------------------------------------------------------
# The registered instances — mirrors mdp.scenario.instances in the IR
# ---------------------------------------------------------------------------

# cumulative echelon rates by chain length, from echelon rates
# h_retail = 1.0 at level 1 and h_echelon = 0.5 at every level above
_H = {
    2: (1.5, 0.5, 0.0, 0.0),
    3: (2.0, 1.0, 0.5, 0.0),
    4: (2.5, 1.5, 1.0, 0.5),
}

_DEMAND_MEAN = 10.0
_HORIZON = 50


def _make(name: str, n: int, lead: int, p: float, **over) -> ClarkScarfScenario:
    return ClarkScarfScenario(
        scenario_name=name,
        horizon=over.pop("horizon", _HORIZON),
        n_echelons=n,
        leadtime=lead,
        demand=PoissonDemand(over.pop("demand_mean", _DEMAND_MEAN)),
        h_install=_H[n],
        p_short=p,
        **over,
    )


SCENARIOS: dict[str, ClarkScarfScenario] = {
    # structural covering set: every (n_echelons, leadtime) combination
    "n2_l1_p09": _make("n2_l1_p09", 2, 1, 9.0),
    "n2_l2_p09": _make("n2_l2_p09", 2, 2, 9.0),
    "n3_l1_p09": _make("n3_l1_p09", 3, 1, 9.0),
    "n3_l2_p09": _make("n3_l2_p09", 3, 2, 9.0),   # Stage-0 target
    "n4_l1_p09": _make("n4_l1_p09", 4, 1, 9.0),
    "n4_l2_p09": _make("n4_l2_p09", 4, 2, 9.0),
    # cost variation at the canonical structure
    "n3_l2_p04": _make("n3_l2_p04", 3, 2, 4.0),
    "n3_l2_p19": _make("n3_l2_p19", 3, 2, 19.0),
    # test-only fixture: small enough to brute-force the full joint DP, so the
    # Clark-Scarf decomposition's optimality is CHECKED rather than assumed
    "verify_tiny": _make(
        "verify_tiny", 2, 1, 4.0, horizon=4, demand_mean=2.0, ship_max=40.0
    ),
    # test-only fixture: exercises the pipeline shift register beyond the
    # swept {1, 2} so the general-leadtime recursion is gated (laws +
    # differential), not merely claimed
    "verify_l3": _make(
        "verify_l3", 2, 3, 9.0, horizon=12, demand_mean=4.0, ship_max=80.0
    ),
}


if __name__ == "__main__":
    for nm, sc in SCENARIOS.items():
        live = sc.n_echelons
        print(
            f"{nm:12s} N={live} L={sc.leadtime} T={sc.horizon:2d} "
            f"p={sc.p_short:5.1f} h_install={sc.h_install} "
            f"demand={sc.demand!r}"
        )

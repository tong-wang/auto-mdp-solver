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

from dataclasses import dataclass, field, replace

from clark_scarf_uncertainty import DemandGenerator, FamilyDemand, PoissonDemand



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
    # The objective's discount, materializing the IR constant `beta` (F11,
    # upstream #68). It lives HERE, on the instance, and not as a module
    # constant in four scripts, because that is what naming it in the IR buys:
    # `objective.discount_factor` reads "beta", so an instance may render this
    # model at another discount and everything that scores must follow the
    # instance rather than a literal it was written beside. Every current
    # instance is 0.95 (`human_confirmed` at the Phase-A sign-off).
    beta: float = 0.95

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
        assert 0.0 < self.beta <= 1.0, (
            f"beta={self.beta} must be in (0, 1] -- the model declares the "
            f"quantity's domain as real in [0, 1] and a zero discount is not a "
            f"finite-horizon objective."
        )
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


_DEMAND_SCALE = 1.0   # gamma theta; see the IR constant `demand_scale`


def _gamma(mean: float, scale: float = _DEMAND_SCALE) -> FamilyDemand:
    """Continuous demand, moment-matched to `PoissonDemand(mean)` at scale 1.

    Built as `FamilyDemand` — the same class the differential adapter routes
    every non-`poisson` candidate to — rather than a bespoke `GammaDemand`.
    One class, one sampling path: a second implementation of the same
    distribution would be invisible to the differential, which exercises the
    adapter's path only, and would be free to drift from it.

    shape = mean/theta and scale = theta, so mean is exactly `mean` for any
    theta and var = mean*theta. theta = 1 gives var = mean, which is Poisson's
    own variance — the branches differ in the lattice and in nothing else.
    """
    return FamilyDemand("gamma", {"shape": mean / scale, "scale": scale},
                        source_id=_DEMAND_STREAM_ID)


_DEMAND_STREAM_ID = 0   # matches the IR slot's stream_id and PoissonDemand's default


def _make(name: str, n: int, lead: int, p: float, **over) -> ClarkScarfScenario:
    mean = over.pop("demand_mean", _DEMAND_MEAN)
    demand = over.pop("demand", None) or PoissonDemand(mean)
    return ClarkScarfScenario(
        scenario_name=name,
        horizon=over.pop("horizon", _HORIZON),
        n_echelons=n,
        leadtime=lead,
        demand=demand,
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

# -- the continuous branch ---------------------------------------------------
#
# The `g{theta}_` mirror of every cell above. A separate FRAME, not a competing arm:
# scores are never subtracted across the two (the lattice changes the problem,
# not just its solution), so each branch carries its own leaderboard and its
# own campaign log. Continuous is the IR's DEFAULT candidate -- it is the
# paper's own model (section 2 poses a density) -- while the Poisson cells keep
# their names because 1129 run artifacts and the whole discrete campaign log
# refer to them.
_GAMMA_SPECS: dict[str, dict] = {
    "n2_l1_p09": dict(n=2, lead=1, p=9.0),
    "n2_l2_p09": dict(n=2, lead=2, p=9.0),
    "n3_l1_p09": dict(n=3, lead=1, p=9.0),
    "n3_l2_p09": dict(n=3, lead=2, p=9.0),
    "n4_l1_p09": dict(n=4, lead=1, p=9.0),
    "n4_l2_p09": dict(n=4, lead=2, p=9.0),
    "n3_l2_p04": dict(n=3, lead=2, p=4.0),
    "n3_l2_p19": dict(n=3, lead=2, p=19.0),
    "verify_tiny": dict(n=2, lead=1, p=4.0, horizon=4, demand_mean=2.0, ship_max=40.0),
    "verify_l3": dict(n=2, lead=3, p=9.0, horizon=12, demand_mean=4.0, ship_max=80.0),
}
def _gamma_tag(scale: float = _DEMAND_SCALE) -> str:
    """`g{theta}` with the decimal point dropped — g1, g2, g05, g025.

    theta rides in the NAME because it is a design axis we expect to sweep:
    var = mean*theta, so it moves demand variability at fixed mean, which is
    the axis Poisson structurally cannot offer. Same terse style as the
    existing axes (n3 = 3 echelons, l2 = leadtime 2, p09 = shortage 9.0), so
    a cell reads left to right as rendering-then-structure. The mean needs no
    token: the base name already carries it (verify_tiny at 2, verify_l3 at 4).
    """
    txt = f"{scale:g}".replace("0.", "0").replace(".", "")
    return f"g{txt}"


for _base, _spec in _GAMMA_SPECS.items():
    _name = f"{_gamma_tag()}_{_base}"
    _kw = dict(_spec)
    _mean = _kw.pop("demand_mean", _DEMAND_MEAN)
    SCENARIOS[_name] = _make(
        _name, _kw.pop("n"), _kw.pop("lead"), _kw.pop("p"),
        demand=_gamma(_mean), **_kw,
    )


# -- the undiscounted twins (F12) --------------------------------------------
#
# `_b1` renders the SAME model at beta = 1. Possible at all only because the IR
# names the discount (F11): before that, beta was a literal in the objective and
# a second discount meant re-rendering the whole domain, which would have
# re-based every recorded number instead of opening a cell beside them.
#
# Each twin is its OWN FRAME. A different objective is a different problem, so
# no score crosses between a cell and its twin -- not even to say "worse".
# What they exist to ask is what beta = 0.95 was suppressing: beta^45 is 0.10,
# so the last fifth of the horizon carries almost no weight in the objective
# while carrying 45 of 83 undiscounted units of the RL-vs-DP gap, and the
# raw-echelon separation this campaign is FOR reads +1.57 discounted against
# +43.62 undiscounted on the same two policies.
#
# The DP stays exactly solvable: finite-horizon backward induction takes
# beta = 1 as the ordinary case, and `verify_tiny_b1` proves it rather than
# assuming it -- the brute force reads `scenario.beta` on both recursions, so
# the fixture checks the decomposition against the joint optimum at the SAME
# discount it was solved with.
for _twin, _src in (("n3_l2_p09_b1", "n3_l2_p09"),
                    ("verify_tiny_b1", "verify_tiny"),
                    (f"{_gamma_tag()}_n3_l2_p09_b1", f"{_gamma_tag()}_n3_l2_p09")):
    SCENARIOS[_twin] = replace(SCENARIOS[_src], scenario_name=_twin, beta=1.0)


if __name__ == "__main__":
    for nm, sc in SCENARIOS.items():
        live = sc.n_echelons
        print(
            f"{nm:12s} N={live} L={sc.leadtime} T={sc.horizon:2d} "
            f"p={sc.p_short:5.1f} h_install={sc.h_install} "
            f"demand={sc.demand!r}"
        )

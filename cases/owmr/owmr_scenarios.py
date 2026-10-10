"""Owmr scenario definitions and the SCENARIOS registry.

An ``OwmrScenario`` is one fully specified one-warehouse, N-retailer problem
instance: structure (N, lead times, horizon), the demand generator, and the
cost structure. It composes the stochastic primitives from
``owmr_uncertainty`` and owns no sampling of its own.

Dependency order: owmr_uncertainty  <-  owmr_scenarios  <-  owmr_mdp

Widths name constants
---------------------
The warehouse pipeline is exactly ``l0`` slots and every retailer pipeline
exactly ``l_rt`` slots; the shipment decision is exactly ``n_retailers`` wide.
All three widths NAME scenario constants in the schema, so an instance renders
precisely the shape it selects and nothing is padded or capped.

Costs (the paper's section 3, per period, on end-of-period levels)
--------------------------------------------------------------------
``h0`` on warehouse on-hand; ``h0 + h_rt[i]`` on retailer i's on-hand;
``p_rt[i]`` per unit of retailer i's backlog. Stock in transit is not charged:
it is a policy-independent constant in the paper's average-cost model
(section 3.1) and the paper's reported figures exclude it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from owmr_uncertainty import DemandGenerator, GammaDemand


@dataclass(slots=True)
class OwmrScenario:
    """One fully specified one-warehouse, N-retailer instance."""

    scenario_name: str
    horizon: int

    # structure
    n_retailers: int          # N
    l0: int                   # supplier -> warehouse lead time (periods)
    l_rt: int                 # warehouse -> retailer lead time, identical across retailers

    # uncertainty
    demand: DemandGenerator

    # economics (per unit-period); the vectors are n_retailers long
    h0: float
    h_rt: tuple[float, ...]
    p_rt: tuple[float, ...]

    # demand parameters as the model states them (the generator realizes them)
    mu_rt: tuple[float, ...]
    cv_rt: tuple[float, ...]

    # ACTION-SCALE multiplier: order and each shipment are capped at
    # order_scale_mult * (l0 + 1) * sum(mu_rt). Not a physical limit.
    order_scale_mult: float = 10.0

    seed_salt: int = field(default=1, repr=False)
    desc: str = ""

    def __post_init__(self) -> None:
        n = self.n_retailers
        assert n >= 1, f"n_retailers={n} must be >= 1 (the model's domain, N in Z+)."
        assert self.l0 >= 1, (
            f"l0={self.l0} must be >= 1 (the model's domain, l0 in Z+). No upper "
            f"bound: the warehouse pipeline is `length: l0` in the schema."
        )
        assert self.l_rt >= 1, (
            f"l_rt={self.l_rt} must be >= 1: l_i = 0 is out of scope under this "
            f"rendering's receipt-before-dispatch event order (schema out_of_scope)."
        )
        assert self.horizon >= 1, "horizon must be >= 1."
        assert self.h0 >= 0, f"h0 must be >= 0, got {self.h0}."
        assert self.seed_salt >= 1, "seed_salt must be >= 1 (v2 seed grammar)."
        assert self.order_scale_mult > 0, "order_scale_mult must be > 0."
        for name, vec in (("h_rt", self.h_rt), ("p_rt", self.p_rt),
                          ("mu_rt", self.mu_rt), ("cv_rt", self.cv_rt)):
            assert len(vec) == n, f"{name} must have length {n}, got {len(vec)}."
        assert all(h >= 0 for h in self.h_rt), f"h_rt must be >= 0, got {self.h_rt}."
        assert all(p > 0 for p in self.p_rt), f"p_rt must be > 0, got {self.p_rt}."
        assert all(m > 0 for m in self.mu_rt), f"mu_rt must be > 0, got {self.mu_rt}."
        assert all(c > 0 for c in self.cv_rt), f"cv_rt must be > 0, got {self.cv_rt}."
        assert isinstance(self.demand, DemandGenerator), (
            "demand must be a DemandGenerator instance."
        )
        assert self.demand.n_retailers == n, (
            f"demand generator serves {self.demand.n_retailers} retailers, "
            f"scenario has {n}."
        )
        # a hand-written generator carries the model's (mu, cv); they must be
        # the scenario's, or the constants and the draws disagree silently
        if hasattr(self.demand, "mu_rt"):
            assert list(self.demand.mu_rt) == [float(m) for m in self.mu_rt], (
                "demand generator means disagree with mu_rt"
            )
            assert list(self.demand.cv_rt) == [float(c) for c in self.cv_rt], (
                "demand generator cvs disagree with cv_rt"
            )
        self.h_rt = tuple(float(x) for x in self.h_rt)
        self.p_rt = tuple(float(x) for x in self.p_rt)
        self.mu_rt = tuple(float(x) for x in self.mu_rt)
        self.cv_rt = tuple(float(x) for x in self.cv_rt)

    # -- derived quantities every layer reads ----------------------------------

    @property
    def mu_sys(self) -> float:
        """Mean system demand per period, sum(mu_rt)."""
        return sum(self.mu_rt)

    @property
    def order_max(self) -> float:
        """The action-scale cap on the order and on each shipment — the
        schema's ``order_scale_mult * (l0 + 1) * sum(mu_rt)``, evaluated in
        the same operation order."""
        return self.order_scale_mult * (self.l0 + 1) * sum(self.mu_rt)

    @property
    def has_latents(self) -> bool:
        """True if any composed generator draws a per-episode world latent.
        Always False for the shipped candidate: the paper holds the demand
        distribution fixed."""
        return bool(getattr(self.demand, "latent", False))


class OwmrScenarioSource:
    """Callable that resolves a latent-bearing template to a concrete scenario.

    Deliberately a *separate* class: ``OwmrScenario`` must NOT be callable, or
    "did this resolve to a concrete scenario?" can never be answered. This
    domain ships no latents, so nothing constructs this today; it exists so
    that appending a latent-bearing demand candidate later needs no change to
    the layering — ``SCENARIOS`` entries simply become sources, and every
    consumer already handles both via ``source(seed) if callable(source)``.
    """

    def __init__(self, template: OwmrScenario) -> None:
        assert template.has_latents, (
            "OwmrScenarioSource wraps a latent-bearing template; a concrete "
            "scenario should be used directly."
        )
        self.template = template

    def __getattr__(self, item: str):
        # family-level attributes consumers read before resolving (space sizing)
        return getattr(self.template, item)

    def __call__(self, episode_seed: int) -> OwmrScenario:
        t = self.template
        return OwmrScenario(
            scenario_name=t.scenario_name,
            horizon=t.horizon,
            n_retailers=t.n_retailers,
            l0=t.l0,
            l_rt=t.l_rt,
            demand=t.demand.realize(episode_seed, t.seed_salt),
            h0=t.h0,
            h_rt=t.h_rt,
            p_rt=t.p_rt,
            mu_rt=t.mu_rt,
            cv_rt=t.cv_rt,
            order_scale_mult=t.order_scale_mult,
            seed_salt=t.seed_salt,
            desc=t.desc,
        )


# ---------------------------------------------------------------------------
# The registered instances — mirrors mdp.scenario.instances in the IR
# (Dogru et al. 2010 section 4.1 identical-retailer test bed, the four cells
# chosen at Phase A: p in {4, 19} x cv in {0.5, 2})
# ---------------------------------------------------------------------------

_N = 5
_HORIZON = 100
_L0, _L_RT = 3, 1
_H0, _H_RT = 0.5, 0.5
_MU = 1.0


def _make(name: str, p: float, cv: float, desc: str) -> OwmrScenario:
    mu_rt = (_MU,) * _N
    cv_rt = (cv,) * _N
    return OwmrScenario(
        scenario_name=name,
        desc=desc,
        horizon=_HORIZON,
        n_retailers=_N,
        l0=_L0,
        l_rt=_L_RT,
        demand=GammaDemand(mu_rt=mu_rt, cv_rt=cv_rt),
        h0=_H0,
        h_rt=(_H_RT,) * _N,
        p_rt=(p,) * _N,
        mu_rt=mu_rt,
        cv_rt=cv_rt,
    )


SCENARIOS: dict[str, OwmrScenario] = {
    "base": _make("base", 4.0, 0.5,
                  "low penalty, low variability: the balance-assumption bound "
                  "is known tight here (paper gap ~0.05%) — the readback's "
                  "validation cell"),
    "hicv": _make("hicv", 4.0, 2.0,
                  "low penalty, high variability: the bound is known loose "
                  "(paper average gap ~5%, up to ~17%) — a discover cell"),
    "hipen": _make("hipen", 19.0, 0.5,
                   "high penalty (95% service target), low variability: bound tight"),
    "hipen_hicv": _make("hipen_hicv", 19.0, 2.0,
                        "high penalty, high variability: bound loose — a discover cell"),
}


if __name__ == "__main__":
    for nm, sc in SCENARIOS.items():
        print(
            f"{nm:11s} N={sc.n_retailers} (l0,l_rt)=({sc.l0},{sc.l_rt}) T={sc.horizon} "
            f"h0={sc.h0} h_rt={sc.h_rt[0]} p={sc.p_rt[0]:5.1f} cv={sc.cv_rt[0]} "
            f"order_max={sc.order_max:.0f}  demand={sc.demand!r}"
        )

"""Dynamic pricing stochastic primitives.

Defines the SamplingContext protocol and the arrivals generator classes for
the price-sensitive demand process. These are the stochastic building blocks
that DynamicPricingScenario is composed from; they have no dependency on the
MDP dynamics.

Seed scheme v2 (spec §6.3): intrinsic draws key leaf-first on

    [(draw,) period, source_id, 1, episode_seed, seed_salt]

Every generator *instance* carries a `source_id` — the child id under
branch 1 of the seed tree; arrivals defaults to 0.

Dependency order: dynamic_pricing_uncertainty  ←  dynamic_pricing_scenarios  ←  dynamic_pricing_mdp
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

SEED_SCHEME = "v2"

_INTRINSIC_BRANCH = 1


# ---------------------------------------------------------------------------
# Sampling context protocol
# ---------------------------------------------------------------------------


class SamplingContext(Protocol):
    """Minimal interface a state must expose for generator.sample().

    DynamicPricingState satisfies this structurally — no import of the
    concrete type is needed here, avoiding circular dependencies.
    """

    period:       int
    episode_seed: int
    seed_salt:    int


def intrinsic_key(
    source_id: int, ctx: SamplingContext, draw: int | None = None
) -> list[int]:
    """v2 intrinsic seed key: [(draw,) period, source_id, 1, episode_seed, seed_salt].

    Leaf-first so the trailing word is always seed_salt (>= 1): SeedSequence
    zero-pads entropy lists shorter than its 4-word pool, so routinely-zero
    leaf ids (period 0, draw 0) must never sit in trailing position.
    """
    key = [ctx.period, source_id, _INTRINSIC_BRANCH, ctx.episode_seed, ctx.seed_salt]
    if draw is not None:
        key.insert(0, draw)
    return key


# ---------------------------------------------------------------------------
# Arrivals generators
# ---------------------------------------------------------------------------


class ArrivalsGenerator:
    """Abstract base for price-sensitive demand-arrival generators.

    The generator is a price-conditioned primitive: the rng seed depends only
    on (period, episode_seed, seed_salt), so the underlying randomness at
    period t is independent of every decision; the CURRENT price enters only
    through the distribution's rate. Realized arrivals are therefore
    decision-path independent — past prices cannot affect future draws.
    """

    is_discrete = True
    source_id: int  # child id under branch 1 of the seed tree; set per instance

    def intensity(self, price: float) -> float:
        """Instantaneous demand intensity lambda(price)."""
        raise NotImplementedError

    def sample(self, ctx: SamplingContext, price: float) -> int:
        """Arrivals in one period at the given price, reproducible from ctx."""
        raise NotImplementedError

    def mean(self, price: float) -> float:
        """Expected arrivals per period at the given price (for planners)."""
        raise NotImplementedError

    def max(self) -> float:
        """Practical upper bound on per-period arrivals (space construction)."""
        raise NotImplementedError


class PoissonArrivals(ArrivalsGenerator):
    """Poisson arrivals under the regular exponential demand family.

    lambda(p) = a * exp(-alpha * p); per-period arrivals ~ Poisson(lambda(p) * dt),
    where dt is the period length. The seed key is built by intrinsic_key()
    (v2 seed tree, spec §6.3), matching the IR interpreter's derived key so
    trajectories diff bit-exactly.
    """

    def __init__(self, a: float, alpha: float, dt: float, source_id: int = 0) -> None:
        assert a > 0,     "a must be positive."
        assert alpha > 0, "alpha must be positive."
        assert dt > 0,    "dt must be positive."
        self.a         = a
        self.alpha     = alpha
        self.dt        = dt
        self.source_id = source_id

    def intensity(self, price: float) -> float:
        return self.a * float(np.exp(-self.alpha * price))

    def rate(self, price: float) -> float:
        """Per-period Poisson mean: lambda(price) * dt."""
        return self.intensity(price) * self.dt

    def sample(self, ctx: SamplingContext, price: float) -> int:
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, ctx))
        )
        return int(rng.poisson(self.rate(price)))

    def mean(self, price: float) -> float:
        return self.rate(price)

    def max(self) -> float:
        """Upper bound at the highest possible rate (price 0): mean + 6 sigma."""
        rate0 = self.rate(0.0)
        return float(np.ceil(rate0 + 6.0 * np.sqrt(rate0) + 1.0))

    def max_intensity(self) -> float:
        """lambda* = a/e: the revenue-maximizing intensity for the exponential
        family (Gallego & van Ryzin Prop. 1) — the natural intensity-action bound."""
        return self.a / float(np.e)

    def __repr__(self) -> str:
        return f"PoissonArrivals(a={self.a}, alpha={self.alpha}, dt={self.dt})"

"""ADI-flex stochastic primitives.

Defines the SamplingContext protocol and the segment-demand generator for the
advance-demand-information model of Wang & Toktay (2008). These are the
stochastic building blocks that AdiFlexScenario is composed from; they have no
dependency on the MDP dynamics.

Dependency order: adi_flex_uncertainty  ←  adi_flex_scenarios  ←  adi_flex_mdp
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Sampling context protocol
# ---------------------------------------------------------------------------


class SamplingContext(Protocol):
    """Minimal interface a state must expose for generator.sample().

    AdiFlexState satisfies this structurally — no import of the concrete type
    is needed here, avoiding circular dependencies.
    """

    period:       int
    episode_seed: int
    seed_salt:    int


# ---------------------------------------------------------------------------
# Demand generators
# ---------------------------------------------------------------------------


class DemandGenerator:
    """Abstract base for per-segment demand generators."""

    is_discrete = True

    def sample(self, ctx: SamplingContext) -> int:
        """Draw one sample, fully reproducible given ctx.period / episode_seed."""
        raise NotImplementedError

    def mean(self) -> float:
        """Expected value, for deterministic planners (LP/DP baselines)."""
        raise NotImplementedError

    def max(self) -> float:
        """Practical upper bound on the sampled value (space construction)."""
        raise NotImplementedError


class DemandVector(DemandGenerator):
    """The demand vector D_i — all due-offsets in ONE draw.

    There is one demand process, indexed by its due-offset τ — the customer
    base is a *mix* over offsets (Wang & Toktay 2008 §4), not a set of separate
    demands. This draws the whole vector at once: component τ is Poisson at
    ``rates[τ]``, independent across components but **not identically
    distributed**, which is the IR's `independent` family (proposed from this
    campaign as auto-mdp-solver#49, usable at stage level from v0.9.10).

    The seed key is the plain period key
    ``[_STREAM_ID, period, episode_seed, seed_salt]``, and the components are
    drawn **in index order from that one rng** — matching
    ``mdp_ir.runtime.sample_family``'s `independent` branch, which is what keeps
    interpreter and simulator trajectories bit-identical so the differential is
    a real check rather than a coincidence.

    **The key has moved twice, and this is the last time.** Originally
    ``[stream_id(=1+τ), period, …]`` (one source per offset); then
    ``[τ, _STREAM_ID, period, …]`` at F3, when the three sources collapsed into
    one indexed source and τ entered through the stage's ``key_exprs``; now the
    period key alone, because a single vector draw needs no component index. F8
    therefore re-seeds — same distribution, different realization — which cost
    nothing only because the solution stage was reset and held for this pin.

    A rate of 0 is a degenerate always-zero component (the one-hot homogeneous
    instances); it still consumes its draw, so trajectories stay comparable
    across instances.
    """

    _STREAM_ID = 1

    def __init__(self, rates: tuple[float, ...]) -> None:
        assert len(rates) > 0, "at least one due-offset."
        assert all(r >= 0 for r in rates), "rates must be non-negative."
        self.rates = tuple(float(r) for r in rates)

    def sample(self, ctx: SamplingContext) -> tuple[int, ...]:
        ss = np.random.SeedSequence(
            [self._STREAM_ID, ctx.period, ctx.episode_seed, ctx.seed_salt]
        )
        rng = np.random.default_rng(ss)
        # in index order from the one rng — `independent`'s documented contract
        return tuple(int(rng.poisson(r)) for r in self.rates)

    def mean(self) -> tuple[float, ...]:
        """Per-component means. Deliberately not a scalar: an inid vector has
        no meaningful average, and the aggregate a bound wants is the sum —
        the same reasoning that makes `independent.mean` raise upstream."""
        return self.rates

    def max(self) -> float:
        """Mean + 6 sigma of one period's TOTAL, at least 1 — an observation
        bound, so generous (space construction, spec §4.1)."""
        return self.max_total(1)

    def max_total(self, periods: int, sigmas: float = 6.0) -> float:
        """Practical upper bound on the demand arriving over ``periods``
        periods — mean + ``sigmas`` sigma of the SUM.

        Not ``periods * max()``. A sum of independent Poissons is Poisson at
        the summed rate, so the aggregate's spread grows as sqrt(periods)
        while its mean grows linearly; multiplying a per-period bound by a
        count inflates the tail term by sqrt(periods) and runs 2.5x loose over
        a 30-period horizon. Exact for this family, not an approximation.

        ``sigmas`` is a call-site choice because the two consumers pay
        different prices for slack. An **observation** bound costs nothing when
        it is loose — the Box is an indication, and nothing at runtime reads it
        — so those sites take the generous default. An **action** bound is a
        discrete space the policy must explore, where slack is paid for in
        sample efficiency, so `AdiFlexScenario.order_max` asks for 4 (spec
        §4.1's own example): at the horizon total that still clears the largest
        order any solved policy places by ~2x, and it is drawn once per
        episode rather than once per period.
        """
        assert periods >= 1, "periods must be at least 1."
        total = sum(self.rates) * periods
        return float(np.ceil(total + sigmas * np.sqrt(total) + 1.0))

    def __repr__(self) -> str:
        return f"DemandVector(rates={self.rates})"

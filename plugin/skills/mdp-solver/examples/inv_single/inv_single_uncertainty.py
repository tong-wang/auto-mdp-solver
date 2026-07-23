"""InvSingle stochastic primitives (seed scheme v2).

Defines the SamplingContext protocol and all generator classes for demand
and lead time. These are the stochastic building blocks that InvSingleScenario
is composed from; they have no dependency on the MDP dynamics.

Seed scheme v2 (spec §6.3): one tree
seed_salt -> episode_seed -> branch -> ..., encoded leaf-first (root last).
Intrinsic draws (branch 1) key on

    [(draw,) period, source_id, 1, episode_seed, seed_salt]

Every generator *instance* carries a `source_id` — the child id under
branch 1; instances composed into one scenario must have distinct ids
(checked by InvSingleScenario). Demand defaults to 0, lead time to 1.

Dependency order: inv_single_uncertainty  <-  inv_single_scenarios  <-  inv_single_mdp
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

SEED_SCHEME = "v2"

_INTRINSIC_BRANCH = 1


# ---------------------------------------------------------------------------
# Sampling context protocol
# ---------------------------------------------------------------------------


class SamplingContext(Protocol):
    """Minimal interface a state must expose for generator.sample().

    InvSingleState satisfies this structurally — no import of the concrete
    type is needed here, avoiding circular dependencies.
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
# Demand generators
# ---------------------------------------------------------------------------


class DemandGenerator:
    """Abstract per-period demand model."""

    is_discrete: bool
    source_id:   int  # child id under branch 1; set per instance

    def sample(self, state: SamplingContext) -> int:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def max(self) -> float:
        raise NotImplementedError

    def phi(self, d: float) -> float:
        """PMF (discrete) or PDF (continuous) evaluated at d."""
        raise NotImplementedError


class FixedDistributionDemand(DemandGenerator):
    """Per-period demand from a fixed, fully realized discrete distribution.

    The (vals, probs) pair typically comes from the world-latent draw in
    InvSingleScenarioSampler; within a concrete scenario they are plain
    constants, so phi() is an exact PMF lookup — no integrating over the
    episode-level demand draw, which the sampler has already realized.
    """

    is_discrete = True

    def __init__(self, vals, probs, source_id: int = 0) -> None:
        vals  = np.asarray(vals, dtype=int)
        probs = np.asarray(probs, dtype=float)
        assert vals.ndim == 1 and vals.shape == probs.shape, \
            "vals and probs must be 1-D and match in length."
        assert np.all(vals >= 0), "demand values must be non-negative."
        assert np.all(probs >= 0) and probs.sum() > 0, \
            "probs must be non-negative and sum to > 0."
        self.vals      = vals
        self.probs     = probs / probs.sum()
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> int:
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, state))
        )
        return int(rng.choice(self.vals, p=self.probs))

    def mean(self) -> float:
        return float(np.dot(self.vals, self.probs))

    def max(self) -> float:
        return float(self.vals.max())

    def phi(self, d: float) -> float:
        k = int(d)
        hit = self.probs[self.vals == k]
        return float(hit[0]) if len(hit) else 0.0

    def __repr__(self) -> str:
        return (
            f"FixedDistributionDemand(vals={self.vals.tolist()}, "
            f"probs={np.round(self.probs, 4).tolist()})"
        )


class PoissonDemand(DemandGenerator):
    """Independent Poisson demand."""

    is_discrete = True

    def __init__(self, rate: float, source_id: int = 0) -> None:
        assert rate > 0, "rate must be positive."
        self.rate      = rate
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> int:
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, state))
        )
        return int(rng.poisson(lam=self.rate))

    def mean(self) -> float:
        return self.rate

    def max(self) -> float:
        return self.rate + 4.0 * self.rate ** 0.5

    def phi(self, d: float) -> float:
        k = int(d)
        if k < 0:
            return 0.0
        # log-form: the direct rate**k / k! overflows for large k
        return math.exp(k * math.log(self.rate) - self.rate - math.lgamma(k + 1))

    def __repr__(self) -> str:
        return f"PoissonDemand(rate={self.rate})"


# ---------------------------------------------------------------------------
# Lead time generators
# ---------------------------------------------------------------------------


class LeadtimeGenerator:
    """Abstract lead time model."""

    source_id: int  # child id under branch 1; set per instance

    def sample(self, state: SamplingContext) -> int:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def min(self) -> int:
        raise NotImplementedError

    def max(self) -> int:
        raise NotImplementedError


class DeterministicLeadtime(LeadtimeGenerator):
    """Fixed lead time."""

    def __init__(self, value: int, source_id: int = 1) -> None:
        assert value >= 0, "Lead time must be non-negative."
        self.value     = value
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> int:
        return self.value

    def mean(self) -> float:
        return float(self.value)

    def min(self) -> int:
        return self.value

    def max(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"DeterministicLeadtime(value={self.value})"


class DiscreteLeadtime(LeadtimeGenerator):
    """Stochastic lead time with a custom discrete distribution."""

    def __init__(
        self,
        values:        list[int],
        probabilities: list[float],
        source_id:     int = 1,
    ) -> None:
        assert len(values) == len(probabilities), \
            "values and probabilities must match in length."
        assert all(v >= 0 for v in values), \
            "All lead time values must be non-negative."
        assert all(p >= 0 for p in probabilities), \
            "Probabilities must be non-negative."

        self.values = values
        s = sum(probabilities)
        self.probabilities = [p / s for p in probabilities]
        self._mean = sum(v * p for v, p in zip(values, self.probabilities))
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> int:
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, state))
        )
        return int(rng.choice(a=self.values, p=self.probabilities))

    def mean(self) -> float:
        return self._mean

    def min(self) -> int:
        return min(self.values)

    def max(self) -> int:
        return max(self.values)

    def __repr__(self) -> str:
        return f"DiscreteLeadtime(values={self.values}, prob={self.probabilities})"

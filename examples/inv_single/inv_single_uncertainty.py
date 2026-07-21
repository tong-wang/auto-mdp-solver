"""InvSingle stochastic primitives.

Defines the SamplingContext protocol and all generator classes for demand
and lead time. These are the stochastic building blocks that InvSingleScenario
is composed from; they have no dependency on the MDP dynamics.

Dependency order: inv_single_uncertainty  ←  inv_single_scenarios  ←  inv_single_mdp
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np


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


# ---------------------------------------------------------------------------
# Demand generators
# ---------------------------------------------------------------------------


class DemandGenerator:
    """Abstract per-period demand model."""

    _STREAM = 1  # demand occupies RNG stream 1
    is_discrete: bool

    def sample(self, state: SamplingContext) -> int:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def max(self) -> float:
        raise NotImplementedError

    def phi(self, d: float) -> float:
        """PMF (discrete) or PDF (continuous) evaluated at d."""
        raise NotImplementedError


class EpisodeDemand(DemandGenerator):
    """Per-episode demand distribution.

    At the start of each episode a support of `support_size` integer values is
    drawn uniformly (without replacement) from {support_low, ..., support_high}
    with random weights.  Within the episode, each period's demand is sampled
    from that fixed distribution.

    Both the support and the per-step samples are derived deterministically from
    the episode seed, so demand is path-independent and fully reproducible.
    """

    is_discrete = True

    _SUB_SUPPORT = 0  # sub-stream for episode-level support sampling
    _SUB_SAMPLE  = 1  # sub-stream for per-step demand sampling

    def __init__(
        self,
        support_size: int = 5,
        support_low:  int = 10,
        support_high: int = 50,
    ) -> None:
        assert support_size >= 1, "support_size must be >= 1."
        assert 0 <= support_low <= support_high, "support bounds invalid."
        assert support_size <= (support_high - support_low + 1), (
            "support_size exceeds the number of integers in [support_low, support_high]."
        )
        self.support_size = support_size
        self.support_low  = support_low
        self.support_high = support_high

    def _episode_distribution(
        self, episode_seed: int, seed_salt: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministically derive this episode's demand vals and probs."""
        ss  = np.random.SeedSequence(
            [self._SUB_SUPPORT, self._STREAM, episode_seed, seed_salt]
        )
        rng = np.random.default_rng(ss)
        vals = rng.choice(
            np.arange(self.support_low, self.support_high + 1),
            size=self.support_size,
            replace=False,
        )
        raw_probs = rng.uniform(1.0, 10.0, size=self.support_size)
        return vals.astype(int), raw_probs / raw_probs.sum()

    def sample(self, state: SamplingContext) -> int:
        vals, probs = self._episode_distribution(
            state.episode_seed, state.seed_salt
        )
        ss  = np.random.SeedSequence(
            [self._SUB_SAMPLE, self._STREAM, state.period,
             state.episode_seed, state.seed_salt]
        )
        rng = np.random.default_rng(ss)
        return int(rng.choice(vals, p=probs))

    def mean(self) -> float:
        return (self.support_low + self.support_high) / 2.0

    def max(self) -> float:
        return float(self.support_high)

    def phi(self, d: float) -> float:
        raise NotImplementedError("EpisodeDemand has no episode-independent phi(); marginal PMF requires integrating over episode randomness.")

    def __repr__(self) -> str:
        return (
            f"EpisodeDemand(support_size={self.support_size}, "
            f"support_low={self.support_low}, support_high={self.support_high})"
        )


class NormalDemand(DemandGenerator):
    """Independent normal demand."""

    is_discrete = False

    def __init__(self, mu: float, sigma: float) -> None:
        assert sigma >= 0, "sigma must be non-negative."
        self.mu    = mu
        self.sigma = sigma

    def sample(self, state: SamplingContext) -> float:
        ss  = np.random.SeedSequence(
            [self._STREAM, state.period, state.episode_seed, state.seed_salt]
        )
        rng = np.random.default_rng(ss)
        return float(rng.normal(loc=self.mu, scale=self.sigma))

    def mean(self) -> float:
        return self.mu

    def max(self) -> float:
        return self.mu + 4.0 * self.sigma

    def phi(self, d: float) -> float:
        z = (d - self.mu) / self.sigma
        return math.exp(-0.5 * z * z) / (self.sigma * math.sqrt(2.0 * math.pi))

    def __repr__(self) -> str:
        return f"NormalDemand(mu={self.mu}, sigma={self.sigma})"


class PoissonDemand(DemandGenerator):
    """Independent Poisson demand."""

    is_discrete = True

    def __init__(self, rate: float) -> None:
        assert rate > 0, "rate must be positive."
        self.rate = rate

    def sample(self, state: SamplingContext) -> int:
        ss  = np.random.SeedSequence(
            [self._STREAM, state.period, state.episode_seed, state.seed_salt]
        )
        rng = np.random.default_rng(ss)
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

    _STREAM = 2  # lead time occupies RNG stream 2

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

    def __init__(self, value: int) -> None:
        assert value >= 0, "Lead time must be non-negative."
        self.value = value

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

    def sample(self, state: SamplingContext) -> int:
        ss  = np.random.SeedSequence(
            [self._STREAM, state.period, state.episode_seed, state.seed_salt]
        )
        rng = np.random.default_rng(ss)
        return int(rng.choice(a=self.values, p=self.probabilities))

    def mean(self) -> float:
        return self._mean

    def min(self) -> int:
        return min(self.values)

    def max(self) -> int:
        return max(self.values)

    def __repr__(self) -> str:
        return f"DiscreteLeadtime(values={self.values}, prob={self.probabilities})"

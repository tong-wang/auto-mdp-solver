"""AdiFlex stochastic primitives.

Defines the SamplingContext protocol and the generator classes for the three
demand streams — orders arriving due immediately, due next period, and due two
periods out. These are the stochastic building blocks that AdiFlexScenario is
composed from; they have no dependency on the MDP dynamics.

Seed scheme v2 (spec §6.3): intrinsic draws key leaf-first on

    [(draw,) period, source_id, 1, episode_seed, seed_salt]

Every generator *instance* carries a `source_id` — the child id under branch 1
of the seed tree. AdiFlex composes three demand generators into one scenario,
so their source ids are distinct (0, 1, 2), one per due date; they must draw
independently.

Dependency order: adi_flex_uncertainty  <-  adi_flex_scenarios  <-  adi_flex_mdp
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

    AdiFlexState satisfies this structurally — no import of the concrete type
    is needed here, avoiding circular dependencies.
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
    leaf ids (period 0, source 0, draw 0) must never sit in trailing position.
    """
    key = [ctx.period, source_id, _INTRINSIC_BRANCH, ctx.episode_seed, ctx.seed_salt]
    if draw is not None:
        key.insert(0, draw)
    return key


# ---------------------------------------------------------------------------
# Demand generators
# ---------------------------------------------------------------------------


class DemandGenerator:
    """Abstract per-period demand model for one due-date class.

    Unlike single-stream domains, AdiFlex composes *three* demand generators
    into one scenario — one per due date. They must draw independently, so the
    source id is an instance attribute set at construction (0, 1, 2) rather
    than a class constant — the child id under branch 1 of the v2 seed tree.
    """

    source_id: int
    is_discrete: bool

    def sample(self, state: SamplingContext) -> int:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def max(self) -> float:
        raise NotImplementedError

    def phi(self, d: float) -> float:
        """PMF evaluated at d — used by the dynamic-programming benchmarks."""
        raise NotImplementedError


class PoissonDemand(DemandGenerator):
    """Independent Poisson demand on its own branch-1 source.

    A rate of zero is legitimate and common here: it is how the scenario ladder
    switches a due-date class off entirely (e.g. exp7 = (0, 0, 6) puts all mass
    two periods out, recovering the homogeneous model of the paper's section 3).
    A zero-rate generator still draws — it must not be short-circuited, or the
    domain and the IR interpreter would disagree on stream consumption.
    """

    is_discrete = True

    def __init__(self, rate: float, source_id: int) -> None:
        assert rate >= 0, "rate must be non-negative (0 switches the stream off)."
        assert source_id >= 0, "source_id is a branch-1 child id (>= 0)."
        self.rate      = float(rate)
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> int:
        # v2 seed tree (spec §6.3): the key is built by intrinsic_key(), matching
        # the IR interpreter's derived key so trajectories diff bit-exactly.
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
        if self.rate == 0.0:
            return 1.0 if k == 0 else 0.0
        # log-form: the direct rate**k / k! overflows for large k
        return math.exp(k * math.log(self.rate) - self.rate - math.lgamma(k + 1))

    def support_max(self, tail: float = 1e-9) -> int:
        """Smallest k with P(X > k) <= tail — the truncation point the DP
        benchmarks enumerate over."""
        if self.rate == 0.0:
            return 0
        k, cum = 0, 0.0
        while cum < 1.0 - tail and k < 1000:
            cum += self.phi(k)
            k += 1
        return k

    def __repr__(self) -> str:
        return f"PoissonDemand(rate={self.rate}, source_id={self.source_id})"

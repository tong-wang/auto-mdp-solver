"""InvSingle stochastic primitives (seed scheme v2).

Defines the SamplingContext protocol and all generator classes for demand
and lead time. These are the stochastic building blocks that InvSingleScenario
is composed from; they have no dependency on the MDP dynamics.

**Layer rule (catalog model, IR_LAYERING_PLAN §10):** everything about ONE
source of randomness lives here — its concrete families, its per-episode
latents, and BOTH seed branches. A latent-bearing generator (e.g.
``LatentPoissonDemand``) draws its episode latent in ``realize(episode_seed,
seed_salt)`` on the meta branch and returns the realized concrete generator;
a concrete generator's ``realize`` returns itself. The scenario layer only
*composes* generators — it owns no sampling.

The per-slot ``Family{Source}`` bridges at the bottom expose the harness's
generic family runtime (``mdp_ir.runtime.FamilyGenerator``) under this
domain's generator interfaces: any registered distribution family — with its
latent recipe — becomes a demand/leadtime generator with **zero new domain
code**, keying the same v2 seed template as the hand-written classes. The
hand-written classes stay for the shipped candidates (exact ``phi()``,
domain-named settings); a freshly appended catalog candidate needs only the
bridge.

Seed scheme v2 (spec §6.3): one tree
seed_salt -> episode_seed -> branch -> ..., encoded leaf-first (root last).
Intrinsic draws (branch 1) key on

    [(draw,) period, source_id, 1, episode_seed, seed_salt]

and a source's episode latents key on the meta branch (branch 0) at the SAME
``source_id``:

    [source_id, 0, episode_seed, seed_salt]

— one stream identity per source of randomness, on both branches, so
instances sharing a latent recipe share latent draws (common random numbers).

Every generator *instance* carries a `source_id`; instances composed into one
scenario must have distinct ids (checked by InvSingleScenario). Demand
defaults to 0, lead time to 1; meta-level drawers that are composition-scoped
(mixtures) allocate at or above ``N_SOURCE_IDS``.

Dependency order: inv_single_uncertainty  <-  inv_single_scenarios  <-  inv_single_mdp
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from mdp_ir.runtime import FamilyGenerator

SEED_SCHEME = "v2"

_INTRINSIC_BRANCH = 1
_META_BRANCH = 0

# source_id range of this domain's uncertainty slots (demand=0, leadtime=1);
# composition-scoped meta drawers (mixtures) take substream_id >= this
N_SOURCE_IDS = 2


# ---------------------------------------------------------------------------
# Sampling context protocol
# ---------------------------------------------------------------------------


class SamplingContext(Protocol):
    """Minimal interface a state must expose for generator.sample().

    InvSingleState satisfies this structurally — no import of the concrete
    type is needed here, avoiding circular dependencies.
    """

    period: int
    episode_seed: int
    seed_salt: int


def intrinsic_key(
    source_id: int, ctx: SamplingContext, draw: int | None = None
) -> list[int]:
    """v2 intrinsic seed key: [(draw,) period, source_id, 1, episode_seed, seed_salt].

    Leaf-first so the trailing word is always seed_salt (>= 1): SeedSequence
    zero-pads entropy lists shorter than its 4-word pool, so routinely-zero
    leaf ids (period 0, draw 0) must never sit in trailing position.
    """
    key = [
        ctx.period,
        source_id,
        _INTRINSIC_BRANCH,
        ctx.episode_seed,
        ctx.seed_salt,
    ]
    if draw is not None:
        key.insert(0, draw)
    return key


def meta_key(substream_id: int, episode_seed: int, seed_salt: int) -> list[int]:
    """v2 meta seed key: [substream_id, 0, episode_seed, seed_salt] (leaf-first).

    For a source's episode latents, ``substream_id`` IS the source's
    ``source_id`` (one stream identity per source, both branches); mixtures —
    the only composition-scoped drawers — use ids >= ``N_SOURCE_IDS``.
    """
    return [substream_id, _META_BRANCH, episode_seed, seed_salt]


# ---------------------------------------------------------------------------
# Demand generators
# ---------------------------------------------------------------------------


class DemandGenerator:
    """Abstract per-period demand model.

    ``latent`` marks a generator whose parameters are drawn once per episode
    on the meta branch; ``realize()`` performs that draw and returns the
    realized concrete generator (a concrete generator returns itself), so
    every composition resolves uniformly via ``gen.realize(seed, salt)``.
    ``mean()``/``max()`` are family-level on a latent generator (envelope over
    the latent prior — what gym wrappers size spaces from) and exact on a
    concrete one.
    """

    is_discrete: bool
    latent: bool = False
    source_id: int  # child id under branches 1 (intrinsic) and 0 (latents)

    def realize(self, episode_seed: int, seed_salt: int) -> "DemandGenerator":
        return self

    def sample(self, state: SamplingContext) -> int:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def max(self) -> float:
        raise NotImplementedError

    def phi(self, d: float) -> float:
        """PMF (discrete) or PDF (continuous) evaluated at d."""
        raise NotImplementedError


class DiscreteDemand(DemandGenerator):
    """Per-period demand from a fixed, fully realized discrete distribution.

    The (vals, probs) pair typically comes from LatentDiscreteDemand.realize();
    within a concrete scenario they are plain constants, so phi() is an exact
    PMF lookup — no integrating over the episode-level demand draw, which the
    latent generator has already realized.
    """

    is_discrete = True

    def __init__(self, vals, probs, source_id: int = 0) -> None:
        vals = np.asarray(vals, dtype=int)
        probs = np.asarray(probs, dtype=float)
        assert (
            vals.ndim == 1 and vals.shape == probs.shape
        ), "vals and probs must be 1-D and match in length."
        assert np.all(vals >= 0), "demand values must be non-negative."
        assert (
            np.all(probs >= 0) and probs.sum() > 0
        ), "probs must be non-negative and sum to > 0."
        self.vals = vals
        self.probs = probs / probs.sum()
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
            f"DiscreteDemand(vals={self.vals.tolist()}, "
            f"probs={np.round(self.probs, 4).tolist()})"
        )


class PoissonDemand(DemandGenerator):
    """Independent Poisson demand."""

    is_discrete = True

    def __init__(self, rate: float, source_id: int = 0) -> None:
        assert rate > 0, "rate must be positive."
        self.rate = rate
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> int:
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, state))
        )
        return int(rng.poisson(lam=self.rate))

    def mean(self) -> float:
        return self.rate

    def max(self) -> float:
        return self.rate + 4.0 * self.rate**0.5

    def phi(self, d: float) -> float:
        k = int(d)
        if k < 0:
            return 0.0
        # log-form: the direct rate**k / k! overflows for large k
        return math.exp(
            k * math.log(self.rate) - self.rate - math.lgamma(k + 1)
        )

    def __repr__(self) -> str:
        return f"PoissonDemand(rate={self.rate})"


# ---------------------------------------------------------------------------
# Latent demand generators (per-episode world latents, spec §5.2)
#
# Each owns its slot's episode latent: realize() draws on the meta branch at
# THIS generator's source_id and returns the realized concrete generator.
# Family-level mean()/max() are envelopes over the latent prior, so gym
# wrappers and eval scripts size spaces without realizing first.
# ---------------------------------------------------------------------------


class LatentDiscreteDemand(DemandGenerator):
    """Per-episode discrete demand distribution (hidden world latent).

    Each episode draws a support of `support_size` integers uniformly without
    replacement from {support_low, ..., support_high} with normalized-uniform
    weights, realized as a DiscreteDemand.
    """

    is_discrete = True
    latent = True

    def __init__(
        self,
        support_size: int = 3,
        support_low: int = 8,
        support_high: int = 12,
        source_id: int = 0,
    ) -> None:
        assert support_size >= 1, "support_size must be >= 1."
        assert 0 <= support_low <= support_high, "support bounds invalid."
        assert support_size <= (support_high - support_low + 1), (
            "support_size exceeds the number of integers in "
            "[support_low, support_high]."
        )
        self.support_size = support_size
        self.support_low = support_low
        self.support_high = support_high
        self.source_id = source_id

    def realize(self, episode_seed: int, seed_salt: int) -> DiscreteDemand:
        rng = np.random.default_rng(
            np.random.SeedSequence(meta_key(self.source_id, episode_seed, seed_salt))
        )
        vals = rng.choice(
            np.arange(self.support_low, self.support_high + 1),
            size=self.support_size,
            replace=False,
        ).astype(int)
        raw_probs = rng.uniform(1.0, 10.0, size=self.support_size)
        return DiscreteDemand(
            vals=vals, probs=raw_probs / raw_probs.sum(), source_id=self.source_id
        )

    def sample(self, state: SamplingContext) -> int:
        raise RuntimeError(
            "unrealized latent demand — resolve the scenario source first: "
            "scenario = source(episode_seed)"
        )

    def mean(self) -> float:
        return (self.support_low + self.support_high) / 2.0

    def max(self) -> float:
        return float(self.support_high)

    def __repr__(self) -> str:
        return (
            f"LatentDiscreteDemand(size={self.support_size}, "
            f"support=[{self.support_low}..{self.support_high}])"
        )


class LatentPoissonDemand(DemandGenerator):
    """Poisson demand with a per-episode hidden rate (world latent).

    Each episode draws lambda ~ Gamma(alpha, beta) in the shape-rate
    parametrization (E[lambda] = alpha/beta — the conjugate prior for a
    Poisson rate), realized as a PoissonDemand. The rate is hidden (spec
    §5.2): not automatically observed, so a deployable policy must infer it
    and a baseline that reads it is clairvoyant.
    """

    is_discrete = True
    latent = True

    def __init__(self, alpha: float = 9.0, beta: float = 0.3, source_id: int = 0) -> None:
        assert alpha > 0, "alpha (Gamma shape) must be > 0."
        assert beta > 0, "beta (Gamma rate) must be > 0."
        self.alpha = alpha
        self.beta = beta
        self.source_id = source_id

    def realize(self, episode_seed: int, seed_salt: int) -> PoissonDemand:
        rng = np.random.default_rng(
            np.random.SeedSequence(meta_key(self.source_id, episode_seed, seed_salt))
        )
        lam = float(rng.gamma(shape=self.alpha, scale=1.0 / self.beta))
        return PoissonDemand(rate=lam, source_id=self.source_id)

    def sample(self, state: SamplingContext) -> int:
        raise RuntimeError(
            "unrealized latent demand — resolve the scenario source first: "
            "scenario = source(episode_seed)"
        )

    def mean(self) -> float:
        return self.alpha / self.beta

    def max(self) -> float:
        # envelope over the whole latent range: a high-quantile rate
        # (mean + 4 sd) then a Poisson 4-sigma tail, matching PoissonDemand.max()
        lam_hi = self.mean() + 4.0 * (self.alpha**0.5) / self.beta
        return float(lam_hi + 4.0 * lam_hi**0.5)

    def __repr__(self) -> str:
        return f"LatentPoissonDemand(alpha={self.alpha}, beta={self.beta})"


# ---------------------------------------------------------------------------
# Lead time generators
# ---------------------------------------------------------------------------


class LeadtimeGenerator:
    """Abstract lead time model. Same realize() contract as DemandGenerator:
    concrete generators return themselves; a latent variant would draw its
    episode latent on the meta branch at its own source_id."""

    latent: bool = False
    source_id: int  # child id under branches 1 (intrinsic) and 0 (latents)

    def realize(self, episode_seed: int, seed_salt: int) -> "LeadtimeGenerator":
        return self

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
        self.value = value
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
        values: list[int],
        probabilities: list[float],
        source_id: int = 1,
    ) -> None:
        assert len(values) == len(
            probabilities
        ), "values and probabilities must match in length."
        assert all(
            v >= 0 for v in values
        ), "All lead time values must be non-negative."
        assert all(
            p >= 0 for p in probabilities
        ), "Probabilities must be non-negative."

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
        return (
            f"DiscreteLeadtime(values={self.values}, prob={self.probabilities})"
        )


# ---------------------------------------------------------------------------
# Family-generic bridges (harness runtime; IR_LAYERING_PLAN §10f)
#
# One bridge per uncertainty slot: mdp_ir.runtime.FamilyGenerator supplies
# sample()/realize()/mean()/max()/min() over ANY registered distribution
# family (latent recipes included, same v2 seed template), and the domain
# base class supplies the interface identity the scenario asserts. A new
# catalog candidate — a family none of the classes above implement — is
# constructed as e.g. FamilyDemand.from_ir(ir, "demand") with no new code.
# ---------------------------------------------------------------------------


class FamilyDemand(FamilyGenerator, DemandGenerator):
    """Any registered distribution family as demand — the zero-code path.
    ``phi()`` stays NotImplemented (hand-written generators provide exact
    PMFs; DP benchmarks require one of those)."""


class FamilyLeadtime(FamilyGenerator, LeadtimeGenerator):
    """Any registered distribution family as lead time — the zero-code path."""

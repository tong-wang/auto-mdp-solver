"""Cnv stochastic primitives (seed scheme v2).

Defines the SamplingContext protocol and all generator classes for the two
sources of intrinsic randomness in the censored newsvendor: the per-period
number of customer arrivals (``demand``) and where in the period those
arrivals fall (``epochs``). These are the stochastic building blocks that
CnvScenario is composed from; they have no dependency on the MDP dynamics.

**Layer rule (catalog model):** everything about ONE source of randomness
lives here — its concrete families, its per-episode latents, and BOTH seed
branches. The demand slot carries the problem's world latent: the arrival
rate lambda, drawn once per episode from the conjugate Gamma prior and never
revealed. ``LatentPoissonDemand.realize(episode_seed, seed_salt)`` draws it on
the meta branch and returns the realized ``PoissonDemand``; a concrete
generator's ``realize`` returns itself. The scenario layer only *composes*
generators — it owns no sampling.

The arrival process is simulated by the conditional-uniform method: given the
Poisson count D, the arrival epochs are D independent uniforms on [0, 1], and
the stock-out time is the y-th smallest of them (``cnv_mdp``). ``UniformEpochs``
draws a fixed-length vector of ``size`` uniforms per period — the demand cap —
in one vectorised call, which numpy guarantees bit-identical to the
interpreter's ``size`` sequential scalar draws from the same stream.

Seed scheme v2 (spec §6.3): one tree
seed_salt -> episode_seed -> branch -> ..., encoded leaf-first (root last).
Intrinsic draws (branch 1) key on

    [(draw,) period, source_id, 1, episode_seed, seed_salt]

and a source's episode latents key on the meta branch (branch 0) at the SAME
``source_id``:

    [source_id, 0, episode_seed, seed_salt]

— one stream identity per source of randomness, on both branches, so
instances sharing the latent recipe share lambda draws per seed (common
random numbers across the grid).

Every generator *instance* carries a ``source_id``; instances composed into
one scenario must have distinct ids (checked by CnvScenario). Demand is 0,
epochs 1; composition-scoped meta drawers (a grid's training sampler)
allocate at or above ``N_SOURCE_IDS``.

Dependency order: cnv_uncertainty  <-  cnv_scenarios  <-  cnv_mdp
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from mdp_ir.runtime import FamilyGenerator

SEED_SCHEME = "v2"

_INTRINSIC_BRANCH = 1
_META_BRANCH = 0

# source_id range of this domain's uncertainty slots (demand=0, epochs=1);
# composition-scoped meta drawers (grid samplers) take substream_id >= this
N_SOURCE_IDS = 2


# ---------------------------------------------------------------------------
# Sampling context protocol
# ---------------------------------------------------------------------------


class SamplingContext(Protocol):
    """Minimal interface a state must expose for generator.sample().

    CnvState satisfies this structurally — no import of the concrete type is
    needed here, avoiding circular dependencies.
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
    ``source_id`` (one stream identity per source, both branches); the only
    composition-scoped drawer here is a grid's training sampler, which uses
    an id >= ``N_SOURCE_IDS``.
    """
    return [substream_id, _META_BRANCH, episode_seed, seed_salt]


# ---------------------------------------------------------------------------
# Demand generators — the number of arrivals in a period
# ---------------------------------------------------------------------------


class DemandGenerator:
    """Abstract per-period demand model.

    ``latent`` marks a generator whose parameters are drawn once per episode
    on the meta branch; ``realize()`` performs that draw and returns the
    realized concrete generator (a concrete generator returns itself), so
    every composition resolves uniformly via ``gen.realize(seed, salt)``.
    ``mean()``/``max()``/``sd()`` are family-level on a latent generator
    (envelopes over the latent prior — what gym wrappers size spaces from)
    and exact on a concrete one.
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

    def sd(self) -> float:
        raise NotImplementedError

    def max(self) -> float:
        raise NotImplementedError

    def phi(self, d: float) -> float:
        """PMF evaluated at d."""
        raise NotImplementedError


class PoissonDemand(DemandGenerator):
    """Poisson arrivals at a known rate — the realized form of the latent
    generator below, and the clairvoyant's view of the world."""

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

    def sd(self) -> float:
        return self.rate**0.5

    def max(self) -> float:
        return self.rate + 4.0 * self.rate**0.5

    def phi(self, d: float) -> float:
        k = int(d)
        if k < 0:
            return 0.0
        # log-form: the direct rate**k / k! overflows for large k
        return math.exp(k * math.log(self.rate) - self.rate - math.lgamma(k + 1))

    def __eq__(self, other) -> bool:
        return (type(other) is type(self) and other.rate == self.rate
                and other.source_id == self.source_id)

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.rate, self.source_id))

    def __repr__(self) -> str:
        return f"PoissonDemand(rate={self.rate})"


class LatentPoissonDemand(DemandGenerator):
    """Poisson arrivals with a hidden per-episode rate (the world latent).

    Each episode draws lambda ~ Gamma(alpha, beta) in the shape-rate
    parametrization — the conjugate prior of the paper (§3), E[lambda] =
    alpha / beta — realized as a PoissonDemand. The rate is hidden (spec
    §5.2): no observation mode may expose it, a deployable policy must infer
    it from censored sales, and a benchmark that reads it is clairvoyant.
    """

    is_discrete = True
    latent = True

    def __init__(self, alpha: float, beta: float, source_id: int = 0) -> None:
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

    def sd(self) -> float:
        # marginal (negative-binomial) spread: Var[D] = E[lambda] * (1 + 1/beta)
        return (self.mean() * (1.0 + 1.0 / self.beta)) ** 0.5

    def max(self) -> float:
        # envelope over the whole latent range: a high-quantile rate
        # (mean + 4 sd) then a Poisson 4-sigma tail, matching PoissonDemand.max()
        lam_hi = self.mean() + 4.0 * (self.alpha**0.5) / self.beta
        return float(lam_hi + 4.0 * lam_hi**0.5)

    def phi(self, d: float) -> float:
        """Marginal (prior-predictive) PMF: negative binomial with r = alpha
        and success probability p = 1 / (beta + 1)."""
        k = int(d)
        if k < 0:
            return 0.0
        p = 1.0 / (self.beta + 1.0)
        return math.exp(
            math.lgamma(k + self.alpha) - math.lgamma(self.alpha) - math.lgamma(k + 1)
            + k * math.log(p) + self.alpha * math.log(1.0 - p)
        )

    def __eq__(self, other) -> bool:
        return (type(other) is type(self) and other.alpha == self.alpha
                and other.beta == self.beta and other.source_id == self.source_id)

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.alpha, self.beta, self.source_id))

    def __repr__(self) -> str:
        return f"LatentPoissonDemand(alpha={self.alpha}, beta={self.beta})"


# ---------------------------------------------------------------------------
# Epoch generators — where in the period the arrivals fall
# ---------------------------------------------------------------------------


class EpochGenerator:
    """Abstract per-period arrival-epoch model: ``sample`` returns a vector of
    ``size`` epochs in [0, 1], of which the first ``demand`` are the period's
    arrivals (conditional-uniform representation of a Poisson process).
    ``size`` doubles as the simulator's demand cap."""

    is_discrete = False
    latent: bool = False
    source_id: int
    size: int

    def realize(self, episode_seed: int, seed_salt: int) -> "EpochGenerator":
        return self

    def sample(self, state: SamplingContext) -> np.ndarray:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def max(self) -> float:
        return 1.0

    def min(self) -> float:
        return 0.0


class UniformEpochs(EpochGenerator):
    """``size`` i.i.d. uniform epochs on [0, 1] per period.

    Drawn in one vectorised call; numpy's ``Generator.uniform(size=n)``
    consumes the stream exactly as ``n`` scalar calls do, so the vector is
    bit-identical to the interpreter's sequential ``iid`` draw
    (``cnv_test.py`` gates this).
    """

    def __init__(self, size: int, source_id: int = 1) -> None:
        assert size >= 1, "size must be >= 1."
        self.size = int(size)
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> np.ndarray:
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, state))
        )
        return rng.uniform(0.0, 1.0, size=self.size)

    def mean(self) -> float:
        return 0.5

    def __eq__(self, other) -> bool:
        return (type(other) is type(self) and other.size == self.size
                and other.source_id == self.source_id)

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.size, self.source_id))

    def __repr__(self) -> str:
        return f"UniformEpochs(size={self.size})"


# ---------------------------------------------------------------------------
# Family-generic bridges (harness runtime; catalog model)
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


class FamilyEpochs(FamilyGenerator, EpochGenerator):
    """Any registered list-valued family as the epoch vector — the zero-code
    path. ``size`` is read from the family settings."""

    @property
    def size(self) -> int:  # type: ignore[override]
        return int(self.settings["size"])

    def sample(self, ctx: SamplingContext) -> np.ndarray:
        return np.asarray(FamilyGenerator.sample(self, ctx), dtype=float)

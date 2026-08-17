"""ClarkScarf stochastic primitives (seed scheme v2).

Defines the SamplingContext protocol and the demand generators. These are the
stochastic building blocks that ClarkScarfScenario is composed from; they have
no dependency on the MDP dynamics.

This domain has exactly **one** source of randomness: retailer demand. Clark &
Scarf hold the demand distribution fixed across the horizon, so there is no
world latent anywhere in the model — every generator here is concrete and
``realize()`` returns ``self``. The latent machinery is still honoured in the
interface so an appended candidate with a per-episode latent needs no change to
the layering.

Seed scheme v2 (spec §6.3): one tree
seed_salt -> episode_seed -> branch -> ..., encoded leaf-first (root last).
Intrinsic draws (branch 1) key on

    [period, source_id, 1, episode_seed, seed_salt]

and a source's episode latents would key on the meta branch (branch 0) at the
SAME ``source_id``:

    [source_id, 0, episode_seed, seed_salt]

Dependency order: clark_scarf_uncertainty <- clark_scarf_scenarios <- clark_scarf_mdp
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from mdp_ir.runtime import FamilyGenerator

SEED_SCHEME = "v2"

_INTRINSIC_BRANCH = 1
_META_BRANCH = 0

# source_id range of this domain's uncertainty slots (demand=0); a
# composition-scoped meta drawer (mixture) would take substream_id >= this
N_SOURCE_IDS = 1


# ---------------------------------------------------------------------------
# Sampling context protocol
# ---------------------------------------------------------------------------


class SamplingContext(Protocol):
    """Minimal interface a state must expose for generator.sample().

    ClarkScarfState satisfies this structurally — no import of the concrete
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
    ``source_id`` (one stream identity per source, both branches).
    """
    return [substream_id, _META_BRANCH, episode_seed, seed_salt]


# ---------------------------------------------------------------------------
# Demand generators
# ---------------------------------------------------------------------------


class DemandGenerator:
    """Abstract per-period retailer demand model.

    ``realize()`` exists for interface parity with latent-bearing domains: a
    concrete generator returns itself, so every composition resolves uniformly
    via ``gen.realize(seed, salt)``. This domain ships only concrete
    generators.

    ``phi()`` is the exact PMF and is what the Clark-Scarf DP benchmark
    integrates against — a candidate without one cannot back the DP row.
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


class PoissonDemand(DemandGenerator):
    """Independent Poisson retailer demand with a fixed rate.

    The shipped candidate. Whole-unit demand keeps every stock level integral,
    which is what lets the DP reference enumerate exactly rather than on a grid.
    """

    is_discrete = True

    def __init__(self, rate: float, source_id: int = 0) -> None:
        assert rate > 0, "rate must be positive."
        self.rate = float(rate)
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
        return math.exp(k * math.log(self.rate) - self.rate - math.lgamma(k + 1))

    def __repr__(self) -> str:
        return f"PoissonDemand(rate={self.rate})"


# ---------------------------------------------------------------------------
# Family-generic bridge (harness runtime; IR_LAYERING_PLAN §10f)
#
# mdp_ir.runtime.FamilyGenerator supplies sample()/realize()/mean()/max() over
# ANY registered distribution family (latent recipes included, same v2 seed
# template), and the domain base class supplies the interface identity the
# scenario asserts. A catalog candidate appended later — a family PoissonDemand
# does not implement — is constructed as FamilyDemand.from_parts(...) with no
# new domain code. ``phi()`` stays NotImplemented there, so such a candidate
# is trainable and evaluable but cannot back the exact DP row.
# ---------------------------------------------------------------------------


class FamilyDemand(FamilyGenerator, DemandGenerator):
    """Any registered distribution family as retailer demand — the zero-code path."""

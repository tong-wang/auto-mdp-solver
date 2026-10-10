"""Owmr stochastic primitives (seed scheme v2).

Defines the SamplingContext protocol and the demand generators. These are the
stochastic building blocks that OwmrScenario is composed from; they have no
dependency on the MDP dynamics.

This domain has exactly **one** source of randomness: per-period customer
demand at the retailers, independent across retailers and periods and
stationary (Dogru, de Kok & van Houtum 2010, section 3). A generator here
therefore produces a whole **vector** — one draw per retailer — from ONE
intrinsic stream, the components drawn in retailer order. That mirrors the
IR's `independent` family exactly: one seed key per period for the slot, N
gamma draws from the rng it seeds. Lead times are fixed integers and are
scenario constants, not generators. There is no world latent anywhere in the
model, so every generator is concrete and ``realize()`` returns ``self``; the
latent machinery is honoured in the interface so an appended candidate with a
per-episode latent needs no change to the layering.

Seed scheme v2 (spec section 6.3): one tree
seed_salt -> episode_seed -> branch -> ..., encoded leaf-first (root last).
Intrinsic draws (branch 1) key on

    [period, source_id, 1, episode_seed, seed_salt]

and a source's episode latents would key on the meta branch (branch 0) at the
SAME ``source_id``:

    [source_id, 0, episode_seed, seed_salt]

Dependency order: owmr_uncertainty  <-  owmr_scenarios  <-  owmr_mdp
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from mdp_ir.runtime import FamilyGenerator

SEED_SCHEME = "v2"

_INTRINSIC_BRANCH = 1
_META_BRANCH = 0

# source_id range of this domain's uncertainty slots (demand = 0); a
# composition-scoped meta drawer (a mixture) would take substream_id >= this
N_SOURCE_IDS = 1


# ---------------------------------------------------------------------------
# Sampling context protocol
# ---------------------------------------------------------------------------


class SamplingContext(Protocol):
    """Minimal interface a state must expose for generator.sample().

    OwmrState satisfies this structurally — no import of the concrete type is
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
    ``source_id`` (one stream identity per source, both branches).
    """
    return [substream_id, _META_BRANCH, episode_seed, seed_salt]


# ---------------------------------------------------------------------------
# Demand generators — vector-valued, one component per retailer
# ---------------------------------------------------------------------------


class DemandGenerator:
    """Abstract per-period retailer-demand model, for all retailers at once.

    ``sample()`` returns a list of length ``n_retailers`` — realized demand at
    each retailer this period — drawn from one intrinsic stream in retailer
    order. ``mean()`` / ``sd()`` / ``max()`` return per-retailer lists (the
    expected value of a vector draw is a vector); ``system_mean()`` is their
    sum, the quantity the action scale and the initial state are built from.

    ``realize()`` exists for interface parity with latent-bearing domains: a
    concrete generator returns itself. This domain ships only concrete
    generators.
    """

    is_discrete: bool
    latent: bool = False
    source_id: int  # child id under branches 1 (intrinsic) and 0 (latents)
    n_retailers: int

    def realize(self, episode_seed: int, seed_salt: int) -> "DemandGenerator":
        return self

    def sample(self, state: SamplingContext) -> list[float]:
        raise NotImplementedError

    def mean(self) -> list[float]:
        raise NotImplementedError

    def sd(self) -> list[float]:
        raise NotImplementedError

    def max(self) -> list[float]:
        """Per-retailer practical upper bound, mean + 4 sd (spec 4.2)."""
        return [m + 4.0 * s for m, s in zip(self.mean(), self.sd())]

    def system_mean(self) -> float:
        return float(sum(self.mean()))


class GammaDemand(DemandGenerator):
    """Independent gamma demand per retailer at the paper's (mu_i, cv_i).

    The shipped candidate. shape = 1 / cv^2 and scale = mu * cv^2 give exactly
    the mean and coefficient of variation the paper's test bed states; the
    paper's own two-component Erlang mixture is a computational device with
    the same two moments. Continuous on (0, inf) with no mass at zero, as the
    model requires, and closed under convolution: demand over l periods at
    retailer i is gamma(l * shape_i, scale_i), which is what makes the
    balance-assumption lower bound exact rather than numerically convolved.

    Draw order is the IR's: one rng seeded by the slot's intrinsic key, then
    one ``rng.gamma(shape_i, scale_i)`` per retailer in index order — the
    same call sequence ``mdp_ir.runtime.sample_family`` makes for the
    ``independent`` family, so the differential is bit-exact by construction.
    """

    is_discrete = False

    def __init__(
        self,
        mu_rt: Sequence[float],
        cv_rt: Sequence[float],
        source_id: int = 0,
    ) -> None:
        assert len(mu_rt) == len(cv_rt) >= 1, (
            f"mu_rt and cv_rt must be equal-length, non-empty; got "
            f"{len(mu_rt)} and {len(cv_rt)}."
        )
        assert all(m > 0 for m in mu_rt), f"every mean must be > 0, got {mu_rt}."
        assert all(c > 0 for c in cv_rt), f"every cv must be > 0, got {cv_rt}."
        self.mu_rt = [float(m) for m in mu_rt]
        self.cv_rt = [float(c) for c in cv_rt]
        self.n_retailers = len(self.mu_rt)
        # the IR's settings expressions, evaluated in the same operation order
        # so the resolved shape/scale are the identical floats
        self.shape = [1.0 / (c * c) for c in self.cv_rt]
        self.scale = [
            self.mu_rt[i] * self.cv_rt[i] * self.cv_rt[i]
            for i in range(self.n_retailers)
        ]
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> list[float]:
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, state))
        )
        return [
            float(rng.gamma(shape=self.shape[i], scale=self.scale[i]))
            for i in range(self.n_retailers)
        ]

    def mean(self) -> list[float]:
        return list(self.mu_rt)

    def sd(self) -> list[float]:
        return [m * c for m, c in zip(self.mu_rt, self.cv_rt)]

    def leadtime_demand_shape_scale(self, i: int, periods: int) -> tuple[float, float]:
        """Gamma parameters of retailer i's demand summed over ``periods``
        periods — the convolution the lower-bound benchmark needs."""
        return periods * self.shape[i], self.scale[i]

    def __repr__(self) -> str:
        return f"GammaDemand(mu_rt={self.mu_rt}, cv_rt={self.cv_rt})"


# ---------------------------------------------------------------------------
# Family-generic bridge (harness runtime; IR_LAYERING_PLAN section 10f)
#
# mdp_ir.runtime.FamilyGenerator supplies sample()/realize() over ANY
# registered distribution family (the `independent` vector recipe included,
# same v2 seed template), and the domain base class supplies the interface
# identity the scenario asserts. A catalog candidate appended later — a family
# GammaDemand does not implement — is constructed as
# FamilyDemand.from_parts(...) with no new domain code. Its moments follow the
# harness's rules: `independent` has no scalar mean, so callers that need
# per-retailer moments use the hand-written class.
# ---------------------------------------------------------------------------


class FamilyDemand(FamilyGenerator, DemandGenerator):
    """Any registered distribution family as retailer demand — the zero-code path."""

    @property
    def n_retailers(self) -> int:  # type: ignore[override]
        size = self.settings.get("size", 1)
        return int(size)

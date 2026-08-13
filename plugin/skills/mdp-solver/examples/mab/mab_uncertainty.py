"""Mab stochastic primitives (seed scheme v2).

Defines the SamplingContext protocol, the per-pull payout generators for the
two branches (Bernoulli / Gaussian), and — per spec §5.2 — the
**latent-bearing** generators that own the arm means. A concrete payout
generator holds the seed's realized arm means and is decision-conditioned:
``sample(state, arm)`` takes the pulled arm as an extra argument (spec §6.3)
— the seed key contains only (period, episode_seed, seed_salt) slots, so
realized randomness is decision-path independent: the arm selects which
transform of the period's primitive draw is observed.

The arm means are a **world latent**: nature fixes them once per seed and the
policy never sees them. Per spec §5.2 the latent belongs to the source of
randomness it parameterizes, so it lives here — ``LatentBernoulliPayout`` /
``LatentGaussianPayout`` realize themselves into a concrete generator in
``realize(episode_seed, seed_salt)``, keyed on the **meta branch at the
generator's own source_id** (one stream identity per source, both branches).
``mab_scenarios.py`` only *composes* them; it draws nothing.

Because the arm means are i.i.d., the vector latent is K independent scalar
draws taken in arm order from one meta-keyed rng — component for component the
IR candidate's single ``iid`` draw (``{of, size, ...}`` draws its `size`
components in order from that same rng), so the differential is bit-exact.

Seed scheme v2 (spec §6.3), leaf-first:

    intrinsic (branch 1)  [(draw,) period, source_id, 1, episode_seed, seed_salt]
    meta      (branch 0)  [source_id, 0, episode_seed, seed_salt]

Dependency order: mab_uncertainty  <-  mab_scenarios  <-  mab_mdp
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

SEED_SCHEME = "v2"

_INTRINSIC_BRANCH = 1
_META_BRANCH = 0

# This domain has exactly ONE source of randomness: the payout of the arm pulled
# this round. Bernoulli and Gaussian are alternative recipes for it and never
# coexist, so they SHARE its id — unlike demand and leadtime, which are distinct
# coexisting sources and take an id each. Matches the single `payout` slot in
# mab_schema.json. The id is one stream identity used on BOTH seed branches:
# the intrinsic per-pull draw and the meta per-seed arm-mean latent.
# Composition-scoped drawers (mixtures) would allocate at or above N_SOURCE_IDS.
#
# Within that one source the per-pull draw is additionally keyed on the CHOSEN
# ARM, via the leading `draw` slot of intrinsic_key (mirrors `key_exprs:
# ["int(arm)"]` on the slot's `pull` stage, solver >= v0.5.9). That refines the
# period slot into K independent per-arm streams, of which a round reads
# exactly one. Without it a single primitive variate per period is shared by
# every arm, so the period's luck is fixed before the choice is made: Gaussian
# payouts would differ across arms only by their means (identical noise) and
# Bernoulli payouts would be comonotone (a win at p=0.1 forcing a win at every
# higher p). Keying on a decision is sound here because the arm selects *which*
# pre-determined exogenous stream is read, never what it contains — arm i's
# payout at period t is still a fixed function of the episode seed.
PAYOUT_SOURCE_ID = 0
N_SOURCE_IDS = 1


# ---------------------------------------------------------------------------
# Sampling context protocol
# ---------------------------------------------------------------------------


class SamplingContext(Protocol):
    """Minimal interface a state must expose for generator.sample().

    MabState satisfies this structurally — no import of the concrete type is
    needed here, avoiding circular dependencies.
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

    ``draw`` carries the stage's evaluated ``key_exprs`` values (spec §6.3);
    in this domain that is the pulled arm, giving one stream per (arm, period).
    """
    key = [ctx.period, source_id, _INTRINSIC_BRANCH, ctx.episode_seed, ctx.seed_salt]
    if draw is not None:
        key.insert(0, draw)
    return key


def meta_key(substream_id: int, episode_seed: int, seed_salt: int) -> list[int]:
    """v2 meta seed key: [substream_id, 0, episode_seed, seed_salt] (spec §6.3).

    For a source's per-seed latents ``substream_id`` **is** the generator's
    ``source_id`` — one stream identity per source across both branches. Only
    composition-scoped drawers (mixtures) would allocate their own id, at or
    above ``N_SOURCE_IDS``.
    """
    return [substream_id, _META_BRANCH, episode_seed, seed_salt]


# ---------------------------------------------------------------------------
# Payout generators
# ---------------------------------------------------------------------------


class PayoutGenerator:
    """Abstract per-pull payout model over a fixed vector of arm means."""

    is_discrete: bool
    source_id:   int  # child id under branches 1 (intrinsic) and 0 (latents)
    latent:      bool = False

    means: list[float]  # realized per-seed arm means (the world latent)

    @property
    def n_arms(self) -> int:
        return len(self.means)

    def realize(self, episode_seed: int, seed_salt: int) -> "PayoutGenerator":
        """A concrete generator is already realized and returns itself, so
        every composition resolves uniformly (spec §5.2)."""
        return self

    def sample(self, state: SamplingContext, arm: int) -> float:
        """Payout of pulling ``arm`` this period (decision-conditioned)."""
        raise NotImplementedError

    def mean(self, arm: int) -> float:
        return self.means[arm]

    def opt_mean(self) -> float:
        return max(self.means)

    def min_payout(self) -> float:
        raise NotImplementedError

    def max_payout(self) -> float:
        raise NotImplementedError


class BernoulliPayout(PayoutGenerator):
    """0/1 payout with per-arm success probability means[arm]."""

    is_discrete = True

    def __init__(
        self, means: Sequence[float], source_id: int = PAYOUT_SOURCE_ID
    ) -> None:
        means = [float(p) for p in means]
        assert len(means) >= 1, "means must be non-empty."
        assert all(0.0 <= p <= 1.0 for p in means), \
            "Bernoulli means must lie in [0, 1]."
        self.means     = means
        self.source_id = source_id

    def sample(self, state: SamplingContext, arm: int) -> int:
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, state, draw=arm))
        )
        return int(rng.random() < self.means[arm])

    def min_payout(self) -> float:
        return 0.0

    def max_payout(self) -> float:
        return 1.0

    def __repr__(self) -> str:
        return f"BernoulliPayout(means={np.round(self.means, 4).tolist()})"


class GaussianPayout(PayoutGenerator):
    """Normal payout N(means[arm], sigma^2) with known common sigma."""

    is_discrete = False

    def __init__(
        self,
        means: Sequence[float],
        sigma: float = 1.0,
        source_id: int = PAYOUT_SOURCE_ID,
    ) -> None:
        means = [float(m) for m in means]
        assert len(means) >= 1, "means must be non-empty."
        assert sigma > 0, "sigma must be positive."
        self.means     = means
        self.sigma     = float(sigma)
        self.source_id = source_id

    def sample(self, state: SamplingContext, arm: int) -> float:
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, state, draw=arm))
        )
        return float(rng.normal(self.means[arm], self.sigma))

    def min_payout(self) -> float:
        # 8-sigma envelope below the smallest mean (space bound, not a truncation)
        return min(self.means) - 8.0 * self.sigma

    def max_payout(self) -> float:
        return max(self.means) + 8.0 * self.sigma

    def __repr__(self) -> str:
        return (
            f"GaussianPayout(means={np.round(self.means, 4).tolist()}, "
            f"sigma={self.sigma})"
        )


# ---------------------------------------------------------------------------
# Latent payout generators (per-seed world latents, spec §5.2)
#
# Alternative recipes for the ONE payout source, so both carry
# PAYOUT_SOURCE_ID: realize() draws on the meta branch at that id and returns
# the realized concrete generator. The arm means are i.i.d., so the vector
# latent is K scalar draws taken in arm order from one rng — the same order as
# the IR sampler's declared draws, which is what keeps the differential exact.
# min_payout()/max_payout() are family-level ENVELOPES over the latent prior,
# so gym wrappers can size spaces before any seed is drawn.
# ---------------------------------------------------------------------------


def _unrealized(what: str) -> RuntimeError:
    return RuntimeError(
        f"unrealized latent payout generator — {what} is only defined once the "
        "arm means are drawn; resolve the scenario source first "
        "(MabScenarioSource(template)(episode_seed))"
    )


class LatentBernoulliPayout(PayoutGenerator):
    """Bernoulli arm probabilities drawn once per seed (hidden world latent).

    p_i ~ Uniform(low, high) i.i.d. over the K arms; realized as a concrete
    ``BernoulliPayout``.
    """

    is_discrete = True
    latent = True

    def __init__(
        self,
        n_arms: int = 10,
        low: float = 0.0,
        high: float = 1.0,
        source_id: int = PAYOUT_SOURCE_ID,
    ) -> None:
        assert n_arms >= 2, "n_arms must be >= 2."
        assert 0.0 <= low < high <= 1.0, \
            "Bernoulli means are probabilities: need 0 <= low < high <= 1."
        self._n_arms   = int(n_arms)
        self.low       = float(low)
        self.high      = float(high)
        self.source_id = int(source_id)

    @property
    def n_arms(self) -> int:
        return self._n_arms

    def realize(self, episode_seed: int, seed_salt: int) -> BernoulliPayout:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                meta_key(self.source_id, episode_seed, seed_salt)
            )
        )
        means = [
            float(rng.uniform(self.low, self.high)) for _ in range(self._n_arms)
        ]
        return BernoulliPayout(means, source_id=self.source_id)

    # -- family-level envelopes over the latent prior -----------------------
    def min_payout(self) -> float:
        return 0.0

    def max_payout(self) -> float:
        return 1.0

    # -- unrealized: fail loudly, never draw --------------------------------
    def sample(self, state: SamplingContext, arm: int) -> float:
        raise _unrealized("a payout")

    def mean(self, arm: int) -> float:
        raise _unrealized("an arm mean")

    def opt_mean(self) -> float:
        raise _unrealized("the best arm's mean")

    def __repr__(self) -> str:
        return (
            f"LatentBernoulliPayout(n_arms={self._n_arms}, "
            f"p~U({self.low}, {self.high}))"
        )


class LatentGaussianPayout(PayoutGenerator):
    """Gaussian arm means drawn once per seed (hidden world latent).

    mu_i ~ Normal(prior_mean, prior_sd) i.i.d. over the K arms, with a known
    common payout sd ``sigma``; realized as a concrete ``GaussianPayout``.
    """

    is_discrete = False
    latent = True

    def __init__(
        self,
        n_arms: int = 10,
        prior_mean: float = 0.0,
        prior_sd: float = 1.0,
        sigma: float = 1.0,
        source_id: int = PAYOUT_SOURCE_ID,
    ) -> None:
        assert n_arms >= 2,   "n_arms must be >= 2."
        assert prior_sd > 0,  "prior_sd must be positive."
        assert sigma > 0,     "sigma must be positive."
        self._n_arms    = int(n_arms)
        self.prior_mean = float(prior_mean)
        self.prior_sd   = float(prior_sd)
        self.sigma      = float(sigma)
        self.source_id  = int(source_id)

    @property
    def n_arms(self) -> int:
        return self._n_arms

    def realize(self, episode_seed: int, seed_salt: int) -> GaussianPayout:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                meta_key(self.source_id, episode_seed, seed_salt)
            )
        )
        means = [
            float(rng.normal(self.prior_mean, self.prior_sd))
            for _ in range(self._n_arms)
        ]
        return GaussianPayout(means, sigma=self.sigma, source_id=self.source_id)

    # -- family-level envelopes over the latent prior -----------------------
    # 6 prior sd covers the drawn means, then 8 payout sd around the extreme
    # mean — an envelope for space bounds, not a truncation.
    def min_payout(self) -> float:
        return self.prior_mean - 6.0 * self.prior_sd - 8.0 * self.sigma

    def max_payout(self) -> float:
        return self.prior_mean + 6.0 * self.prior_sd + 8.0 * self.sigma

    # -- unrealized: fail loudly, never draw --------------------------------
    def sample(self, state: SamplingContext, arm: int) -> float:
        raise _unrealized("a payout")

    def mean(self, arm: int) -> float:
        raise _unrealized("an arm mean")

    def opt_mean(self) -> float:
        raise _unrealized("the best arm's mean")

    def __repr__(self) -> str:
        return (
            f"LatentGaussianPayout(n_arms={self._n_arms}, "
            f"mu~N({self.prior_mean}, {self.prior_sd}^2), sigma={self.sigma})"
        )

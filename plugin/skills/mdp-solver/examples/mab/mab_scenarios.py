"""Named scenario registry for the multi-armed bandit domain.

Defines MabScenario (a bandit instance: horizon + a payout generator), the
generic MabScenarioSource wrapper, and the SCENARIOS registry. Registry values
are ScenarioSources: a concrete scenario, or a callable
``source(episode_seed) -> MabScenario``.

This module **only composes** (spec §5.2). It draws nothing: the arm means are
a world latent owned by the payout generator that parameterizes them, so they
are declared and drawn in ``mab_uncertainty.py`` (``LatentBernoulliPayout`` /
``LatentGaussianPayout``) and realized here by calling ``payout.realize(...)``.
There is deliberately no per-branch sampler subclass — swapping the payout
family or its latent recipe touches ``mab_uncertainty.py`` and the template
line below, nothing else.

A MabScenario holding a latent payout generator is a **template**: register it
wrapped in a MabScenarioSource, never simulate it directly (the latent's
``sample()`` raises).

Seed scheme v2 (spec §6.3): both seed branches key at the payout source's id,
via the helpers in ``mab_uncertainty`` — meta (per seed) through ``meta_key``,
intrinsic (per pull) through ``intrinsic_key``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from mab_uncertainty import (
    PayoutGenerator,
    LatentBernoulliPayout,
    LatentGaussianPayout,
)

SEED_SCHEME = "v2"


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MabScenario:
    """A bandit instance: horizon + the payout model over the K arms.

    Concrete when ``payout`` holds realized arm means; a **template** when
    ``payout`` is a latent generator (see ``has_latents``), in which case it
    must be registered wrapped in a MabScenarioSource.
    """

    # planning horizon T (number of pulls)
    horizon: int

    # payout model: concrete (realized arm means) or latent (a recipe for them)
    payout: PayoutGenerator

    # scenario-level seed salt (universal reproducibility knob, >= 1)
    seed_salt: int = field(default=4729, repr=False)

    # optional human-readable name
    scenario_name: str | None = None

    # human-readable description of the scenario
    desc: str = ""

    def __post_init__(self) -> None:
        assert self.horizon >= 1,   "horizon must be >= 1."
        assert self.seed_salt >= 1, "seed_salt must be >= 1 (v2 rule)."
        assert isinstance(self.payout, PayoutGenerator), \
            "payout must be a PayoutGenerator."

    @property
    def n_arms(self) -> int:
        return self.payout.n_arms

    @property
    def has_latents(self) -> bool:
        """True when any composed generator still owes a per-seed draw."""
        return bool(self.payout.latent)

    # family-level payout envelope (delegates to the generator: an envelope
    # over the latent prior on a template, exact on a concrete scenario)
    def min_payout(self) -> float:
        return self.payout.min_payout()

    def max_payout(self) -> float:
        return self.payout.max_payout()


# ---------------------------------------------------------------------------
# Scenario source — the one generic wrapper (spec §5.2)
# ---------------------------------------------------------------------------


class MabScenarioSource:
    """Callable ScenarioSource over a template with latent generators.

    A pure function ``episode_seed -> MabScenario``: the payout generator
    realizes itself on the meta branch at its own ``source_id``
    (``generator.realize``); a concrete generator would pass through. All other
    attribute access (``horizon``, ``n_arms``, ``min_payout()``, ``seed_salt``,
    ``scenario_name``, ...) delegates to the template, so gym wrappers and eval
    scripts read a source and a concrete scenario uniformly (spec §5.2
    family-level attributes).
    """

    def __init__(self, template: MabScenario) -> None:
        assert template.has_latents, (
            "MabScenarioSource wraps a template with latent generators; "
            "register a fully concrete scenario directly."
        )
        self._template = template

    def __call__(self, episode_seed: int) -> MabScenario:
        t = self._template
        return dataclasses.replace(
            t, payout=t.payout.realize(episode_seed, t.seed_salt)
        )

    def __getattr__(self, item: str):
        return getattr(self._template, item)

    def __repr__(self) -> str:
        return f"MabScenarioSource({self._template!r})"


ScenarioSource = Union[MabScenario, MabScenarioSource]


# ---------------------------------------------------------------------------
# Scenarios — one per branch (independent branches, separate leaderboards:
# the payout scales differ, so no head-to-head comparison)
# ---------------------------------------------------------------------------

_template_bern = MabScenario(
    scenario_name="bern_K10_T1000",
    desc="Bernoulli branch: 10 arms, p_i ~ U(0,1) fresh each seed, T=1000",
    horizon=1000,
    payout=LatentBernoulliPayout(n_arms=10, low=0.0, high=1.0),
)

_template_gauss = MabScenario(
    scenario_name="gauss_K10_T1000",
    desc="Gaussian branch: 10 arms, mu_i ~ N(0,1) fresh each seed, sigma=1, T=1000",
    horizon=1000,
    payout=LatentGaussianPayout(n_arms=10, prior_mean=0.0, prior_sd=1.0, sigma=1.0),
)

source_bern = MabScenarioSource(_template_bern)
source_gauss = MabScenarioSource(_template_gauss)


# ---------------------------------------------------------------------------
# #E32 robustness cells — the IR's `gauss_*` scenario instances
# ---------------------------------------------------------------------------
#
# K and T are axis-tagged scenario constants (`n_arms` / `arm_max` /
# `horizon_T`), so each cell is one `mdp.scenario.instances` entry overriding
# them; these are the same set, materialized. Specialists, one policy and one
# leaderboard per cell (spec §5.4) — deliberately NOT a `MabScenarioGrid`,
# which §5.6 reserves for a *generalist* target and whose family-agreement
# assertion would rightly reject cells that disagree on `n_arms`.
#
# Axes: K arms x n pulls-per-arm, with T = K * n, so the ladder is indexed by
# the information budget B = n / sigma^2 (the tuned cell K=10, T=1000 is
# B=100 — about 8x past the ~13 pulls/arm that resolve the top-two arm gap).
# The three sigma cells reach a given B by a second, independent route.

_K_LADDER = (5, 10, 20)
_N_LADDER = (2, 5, 10, 20, 50, 100,
             # #E35: past the point where #E34 measured the rule crossing
             # thompson (T ~ 14,000), so PPO can be trained in the regime
             # where a fixed-quantile index provably loses
             200, 500, 1000, 2000)           # pulls per arm; T = K * n
_SIGMA_CELLS = ((10, 200, 2.0), (10, 400, 2.0), (10, 320, 4.0))   # (K, T, sigma)


def _gauss_cell(n_arms: int, horizon: int, sigma: float = 1.0) -> MabScenarioSource:
    """One robustness cell as a scenario source (mirrors the IR instance)."""
    name = f"gauss_K{n_arms}_T{horizon}" + ("" if sigma == 1.0 else f"_s{sigma:g}")
    return MabScenarioSource(MabScenario(
        scenario_name=name,
        desc=(f"Gaussian branch: {n_arms} arms, mu_i ~ N(0,1) fresh each seed, "
              f"sigma={sigma:g}, T={horizon} "
              f"(n={horizon // n_arms} pulls/arm, B={horizon / n_arms / sigma**2:g})"),
        horizon=horizon,
        payout=LatentGaussianPayout(n_arms=n_arms, prior_mean=0.0,
                                    prior_sd=1.0, sigma=sigma),
    ))


_grid_cells = [
    _gauss_cell(k, k * n)
    for k in _K_LADDER for n in _N_LADDER
    if (k, k * n) != (10, 1000)              # the base gaussian scenario
] + [_gauss_cell(k, t, sg) for k, t, sg in _SIGMA_CELLS]


# Registry mapping name -> scenario source for lookup in training scripts.
SCENARIOS: dict[str, ScenarioSource] = {
    s.scenario_name: s
    for s in [
        source_bern,
        source_gauss,
        *_grid_cells,
    ]
}


def _check_registry() -> None:
    """Every callable source resolves seed 0 to a fully concrete scenario;
    a registered plain scenario must already be concrete."""
    for name, s in SCENARIOS.items():
        resolved = s(0) if callable(s) else s
        assert isinstance(resolved, MabScenario) and not resolved.has_latents, (
            f"registry entry {name!r} does not resolve to a concrete scenario"
        )


_check_registry()


if __name__ == "__main__":
    for name, s in SCENARIOS.items():
        kind = "source  " if callable(s) else "scenario"
        print(
            f"{name:18s} [{kind}]  horizon={s.horizon}  K={s.n_arms}  "
            f"payout=[{s.min_payout():.1f}, {s.max_payout():.1f}]  {s.payout!r}"
        )
        example = s(3)
        print(f"  seed 3 means: {np.round(example.payout.means, 3).tolist()}")

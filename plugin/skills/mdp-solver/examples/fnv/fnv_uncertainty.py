"""FNV stochastic primitives.

Defines the SamplingContext protocol and the demand-signal generator classes
that drive the MMFE information process. These are the stochastic building
blocks that FnvScenario is composed from; they have no dependency on the MDP
dynamics.

Seed scheme v2 (spec §6.3): intrinsic draws key leaf-first on

    [(draw,) period, source_id, 1, episode_seed, seed_salt]

Every generator *instance* carries a `source_id` — the child id under branch 1
of the seed tree; the demand signal defaults to 0.

Dependency order: fnv_uncertainty  ←  fnv_scenarios  ←  fnv_mdp
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

SEED_SCHEME = "v2"

_INTRINSIC_BRANCH = 1


# ---------------------------------------------------------------------------
# Sampling context protocol
# ---------------------------------------------------------------------------


class SamplingContext(Protocol):
    """Minimal interface a state must expose for generator.sample().

    FnvState satisfies this structurally — no import of the concrete type is
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
    """
    key = [ctx.period, source_id, _INTRINSIC_BRANCH, ctx.episode_seed, ctx.seed_salt]
    if draw is not None:
        key.insert(0, draw)
    return key


# ---------------------------------------------------------------------------
# Demand signal generators
# ---------------------------------------------------------------------------


class DemandSignalGenerator:
    """Abstract base for the MMFE demand-signal increment process.

    A single generator models the whole information process: ``sample(state)``
    returns the demand-signal increment revealed at ``state.period`` (the
    transition from tau[period] to tau[period+1]). Its rng is keyed by
    intrinsic_key() (v2 seed tree, spec §6.3) on (period, episode_seed,
    seed_salt) so draws are decision-path independent and fully reproducible.
    """

    is_discrete = False
    source_id: int  # child id under branch 1 of the seed tree; set per instance

    def sample(self, state: SamplingContext) -> float:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def max(self) -> float:
        raise NotImplementedError


class NormalDemandSignal(DemandSignalGenerator):
    """Independent mean-zero normal increments with a per-period stdev schedule.

    ``stdevs[n]`` is the std dev of the increment revealed at period ``n`` (the
    transition from tau[n] to tau[n+1]); ``stdevs[0] == 0`` because
    tau[0] == tau[1] == 0. The schedule is derived from the scenario's ``stdev``
    hyper-parameter and the epoch spacing (see FnvScenario). The seed key is
    built by intrinsic_key(), matching the IR interpreter's derived key so
    trajectories diff bit-exactly.
    """

    def __init__(self, stdevs: tuple[float, ...], source_id: int = 0) -> None:
        assert all(s >= 0 for s in stdevs), "stdevs must be non-negative."
        self.stdevs = tuple(float(s) for s in stdevs)
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> float:
        stdev = self.stdevs[state.period]
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, state))
        )
        return float(rng.normal(0.0, stdev))

    def mean(self) -> float:
        """Expected increment (the process is mean-zero)."""
        return 0.0

    def max(self) -> float:
        """Upper bound on the cumulative signal: 5 std devs of the total."""
        total_var = sum(s * s for s in self.stdevs)
        return 5.0 * float(np.sqrt(total_var))

    def __repr__(self) -> str:
        return f"NormalDemandSignal(stdevs={tuple(round(s, 4) for s in self.stdevs)})"

"""The secretary problem's hidden ranking uncertainty.

Each episode has a uniformly random permutation of the distinct absolute
ranks ``1..N``. This module owns that distribution, its v2 seed stream, and
the sampling operation. ``secretary_mdp.init_state`` invokes ``sample()`` once
per reset and stores the returned arrival order in episode state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Sequence

import numpy as np
from mdp_ir.runtime import meta_key


@dataclass(frozen=True, slots=True)
class RankingUncertainty:
    """Seeded uniform distribution over all complete absolute rankings."""

    n_candidates: int
    substream_id: int = 0

    def __post_init__(self) -> None:
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be positive")
        if self.substream_id < 0:
            raise ValueError("substream_id must be nonnegative")

    @property
    def support(self) -> tuple[int, ...]:
        """The distinct absolute ranks permuted by the reset draw."""
        return tuple(range(1, self.n_candidates + 1))

    @property
    def n_realizations(self) -> int:
        """Number of equally likely complete rankings."""
        return factorial(self.n_candidates)

    def seed_key(self, episode_seed: int, seed_salt: int) -> list[int]:
        """Return this source's canonical v2 reset-time seed key."""
        return meta_key(self.substream_id, episode_seed, seed_salt)

    def sample(self, episode_seed: int, seed_salt: int) -> tuple[int, ...]:
        """Draw one uniformly random complete ranking for an episode reset."""
        rng = np.random.default_rng(np.random.SeedSequence(
            meta_key(self.substream_id, episode_seed, seed_salt)
        ))
        return tuple(int(value) for value in rng.choice(
            self.support,
            size=self.n_candidates,
            replace=False,
        ))

    def contains(self, realization: Sequence[int]) -> bool:
        """Whether ``realization`` is one complete ranking in the support."""
        return len(realization) == self.n_candidates and set(realization) == set(
            self.support
        )

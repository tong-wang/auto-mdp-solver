"""Seed-key helpers for reset-time secretary uncertainty.

The realized candidate permutation belongs to episode state, not to the
settings-only scenario.  This module owns only the seed-scheme boundary; the
draw itself remains in ``secretary_mdp.init_state`` where the state is created.
"""

from __future__ import annotations


_META_BRANCH = 0


def meta_key(
    substream_id: int,
    episode_seed: int,
    seed_salt: int,
) -> list[int]:
    """Return the v2 meta key ``[substream, 0, episode_seed, seed_salt]``."""
    return [
        int(substream_id),
        _META_BRANCH,
        int(episode_seed),
        int(seed_salt),
    ]

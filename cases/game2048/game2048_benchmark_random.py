"""Random-valid baseline: uniform over the slides that change the board.

The floor every other method must clear. It is not a uniform draw over all
four directions — that would conflate "plays badly" with "wastes moves on
no-ops", and the free-vs-masked question is a separate axis.
"""

from __future__ import annotations

import numpy as np

from game2048_benchmark_common import POLICY_STREAM_RANDOM


class RandomValidPolicy:
    """Uniform over legal slides, on its own RNG stream (spec §6.3)."""

    def __init__(self, episode_seed: int, seed_salt: int) -> None:
        self.rng = np.random.default_rng(
            np.random.SeedSequence([POLICY_STREAM_RANDOM, episode_seed, seed_salt])
        )

    def act(self, board: list[int], mask) -> int:
        legal = [a for a, ok in enumerate(mask) if ok]
        return int(self.rng.choice(legal))


def make_policy(env):
    return RandomValidPolicy(env._episode_seed, env._current_scenario.seed_salt)

"""Independent fair-action baseline for the secretary problem."""

from __future__ import annotations

import numpy as np

from secretary_mdp import SecretaryState
from secretary_scenarios import SecretaryScenario

_POLICY_SUBSTREAM = 9996


def policy_key(episode_seed: int, seed_salt: int) -> list[int]:
    return [_POLICY_SUBSTREAM, 0, int(episode_seed), int(seed_salt)]


class RandomPolicy:
    def __init__(self, episode_seed: int, seed_salt: int) -> None:
        self._rng = np.random.default_rng(
            np.random.SeedSequence(policy_key(episode_seed, seed_salt))
        )

    def act(self, state: SecretaryState) -> int:
        return int(self._rng.integers(0, 2))


def make_policy(scenario: SecretaryScenario, episode_seed: int) -> RandomPolicy:
    return RandomPolicy(episode_seed, scenario.seed_salt)

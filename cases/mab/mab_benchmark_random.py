"""Random benchmark: pull a uniformly random arm each round.

The floor of the leaderboard — expected reward T * E[mean of a random arm].
"""

from __future__ import annotations

import numpy as np

from mab_benchmark_common import POLICY_STREAM_RANDOM


class RandomPolicy:
    """Uniform-random arm choice; deterministic per (episode_seed, seed_salt)."""

    def __init__(self, n_arms: int, episode_seed: int, seed_salt: int) -> None:
        self.n_arms = n_arms
        self._rng = np.random.default_rng(
            np.random.SeedSequence(
                [POLICY_STREAM_RANDOM, episode_seed, seed_salt]
            )
        )

    def act(self, pulls: list[int], payouts: list[float], period: int) -> int:
        return int(self._rng.integers(self.n_arms))


def make_policy(env) -> RandomPolicy:
    return RandomPolicy(env.n_arms, env._episode_seed, env._scenario_ep.seed_salt)


if __name__ == "__main__":
    from mab_scenarios import SCENARIOS
    from mab_gym import MabEnv

    for name in SCENARIOS:
        env = MabEnv(scenario=SCENARIOS[name])
        totals = []
        for seed in range(5):
            env.reset(seed=seed)
            pol = make_policy(env)
            done, total = False, 0.0
            while not done:
                _, r, done, _, _ = env.step(
                    pol.act(env._state.pulls, env._state.payouts, env._state.period)
                )
                total += r
            totals.append(total)
        print(f"{name:18s} random  5-episode mean reward: {np.mean(totals):9.2f}")

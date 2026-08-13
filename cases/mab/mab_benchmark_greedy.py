"""Greedy (myopic Bayes) benchmark: always pull the arm with the highest
conjugate posterior mean — pure exploitation, no deliberate exploration.

The classic under-explorer: it locks onto an early lucky arm and pays
linear regret when that arm is not the true best.
"""

from __future__ import annotations

import numpy as np

from mab_bayes import bayes_post_mean


class GreedyPolicy:
    """Argmax of the per-arm posterior mean (ties -> lowest index)."""

    def __init__(self, is_gauss: bool) -> None:
        self.is_gauss = is_gauss

    def act(self, pulls: list[int], payouts: list[float], period: int) -> int:
        return int(np.argmax(bayes_post_mean(pulls, payouts, self.is_gauss)))


def make_policy(env) -> GreedyPolicy:
    return GreedyPolicy(env._is_gauss)


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
        print(f"{name:18s} greedy  5-episode mean reward: {np.mean(totals):9.2f}")

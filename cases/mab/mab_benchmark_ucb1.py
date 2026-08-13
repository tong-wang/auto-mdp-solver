"""UCB1 benchmark: optimistic index policy (Auer, Cesa-Bianchi & Fischer 2002).

Pull each arm once, then pull the arm maximizing

    empirical_mean_i + scale * sqrt(2 * ln(t) / n_i)

with scale = 1 on the Bernoulli branch (rewards in [0,1], the textbook
constant) and scale = sigma on the Gaussian branch (known noise sd; the
sub-Gaussian generalization). Logarithmic regret on both branches.
"""

from __future__ import annotations

import math

import numpy as np


class Ucb1Policy:
    """UCB1 with a family-aware exploration scale."""

    def __init__(self, n_arms: int, scale: float) -> None:
        self.n_arms = n_arms
        self.scale  = scale

    def act(self, pulls: list[int], payouts: list[float], period: int) -> int:
        for i in range(self.n_arms):
            if pulls[i] == 0:
                return i
        t = period + 1  # 1-based round count; every arm pulled at least once
        idx = [
            payouts[i] / pulls[i]
            + self.scale * math.sqrt(2.0 * math.log(t) / pulls[i])
            for i in range(self.n_arms)
        ]
        return int(np.argmax(idx))


def make_policy(env) -> Ucb1Policy:
    scale = env._scenario_ep.payout.sigma if env._is_gauss else 1.0
    return Ucb1Policy(env.n_arms, scale)


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
        print(f"{name:18s} ucb1    5-episode mean reward: {np.mean(totals):9.2f}")

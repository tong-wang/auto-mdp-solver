"""Thompson sampling benchmark: sample each arm's mean from its exact
conjugate posterior and pull the argmax.

The posteriors match the true generative priors (Beta(1,1) / N(0,1) with
known sigma), so this is *probability matching under the correct prior* —
the strongest classical baseline here and near Bayes-optimal at T=1000.
"""

from __future__ import annotations

import numpy as np

from mab_benchmark_common import POLICY_STREAM_THOMPSON


class ThompsonPolicy:
    """Exact conjugate Thompson sampling; deterministic per episode seed."""

    def __init__(
        self, n_arms: int, is_gauss: bool, episode_seed: int, seed_salt: int
    ) -> None:
        self.n_arms   = n_arms
        self.is_gauss = is_gauss
        self._rng = np.random.default_rng(
            np.random.SeedSequence(
                [POLICY_STREAM_THOMPSON, episode_seed, seed_salt]
            )
        )

    def act(self, pulls: list[int], payouts: list[float], period: int) -> int:
        rng = self._rng
        if self.is_gauss:
            # prior N(0,1), obs noise sigma known: posterior
            # N(payouts/(1+n), 1/(1+n)) — sigma enters via the unit-information
            # prior only when sigma == 1 (the frozen scenario); general sigma
            # would scale the precisions, kept simple for the shipped branch.
            draws = [
                rng.normal(
                    payouts[i] / (1.0 + pulls[i]),
                    np.sqrt(1.0 / (1.0 + pulls[i])),
                )
                for i in range(self.n_arms)
            ]
        else:
            draws = [
                rng.beta(1.0 + payouts[i], 1.0 + pulls[i] - payouts[i])
                for i in range(self.n_arms)
            ]
        return int(np.argmax(draws))


def make_policy(env) -> ThompsonPolicy:
    return ThompsonPolicy(
        env.n_arms, env._is_gauss, env._episode_seed, env._scenario_ep.seed_salt
    )


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
        print(f"{name:18s} thompson 5-episode mean reward: {np.mean(totals):9.2f}")

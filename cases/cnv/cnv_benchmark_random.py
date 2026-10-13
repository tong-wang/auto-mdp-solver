"""Floor benchmarks — ``random``, role ``feasible`` (spec §8.6: the
did-it-learn-anything floors are a build diagnosis, never an escalation
trigger). Two policies (``--policy``):

- ``uniform`` — a stocking level drawn uniformly from the whole action space
  0..arrivals_buffer, on a policy stream disjoint from the model's streams.
- ``prior_mean`` — stock the prior mean E[lambda] every period, rounded: the
  no-learning newsvendor who never updates.
"""

from __future__ import annotations

import numpy as np

POLICIES = ("uniform", "prior_mean")

# policy stream, kept away from the model's streams (0, 1) and the grid sampler
POLICY_STREAM_RANDOM = 9996


def make_policy(policy: str):
    if policy == "prior_mean":
        return lambda env_probe: (lambda env, obs=None: int(round(env._scenario_ep.prior_mean)))
    if policy == "uniform":
        def factory(env_probe):
            holder = {"seed": None, "rng": None}

            def act(env, obs=None) -> int:
                if holder["seed"] != env._episode_seed:
                    holder["seed"] = env._episode_seed
                    holder["rng"] = np.random.default_rng(np.random.SeedSequence(
                        [POLICY_STREAM_RANDOM, env._episode_seed, env._scenario_ep.seed_salt]))
                return int(holder["rng"].integers(0, env._scenario_ep.arrivals_buffer + 1))
            return act
        return factory
    raise ValueError(f"unknown random policy {policy!r}; choose from {POLICIES}")


if __name__ == "__main__":
    from cnv_gym import CnvEnv
    from cnv_scenarios import SCENARIOS

    env = CnvEnv(scenario=SCENARIOS["base"])
    for policy in POLICIES:
        act = make_policy(policy)(env)
        totals = []
        for seed in range(50):
            env.reset(seed=seed)
            done, total = False, 0.0
            while not done:
                _, r, done, _, _ = env.step(act(env))
                total += r
            totals.append(total)
        print(f"base  {policy:10s} 50-episode mean profit: {np.mean(totals):8.2f}")

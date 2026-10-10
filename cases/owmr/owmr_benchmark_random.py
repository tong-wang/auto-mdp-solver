"""The random floor for owmr (spec 9, role `feasible`).

Uniform actions in the default `order_frac` box: order ~ U[0, order_max] and
independent U[0, 1] fractions of on-hand per retailer. It exists because spec
8.6 makes "L1 <= random" a build diagnosis rather than an escalation trigger.
The policy is the gym's own `action_space.sample()`, seeded per episode so the
floor is reproducible on the CRN block.
"""

from __future__ import annotations

import numpy as np


def random_act_fn(seed: int = 12345):
    """Batched action callback: one draw per env per step from a private rng
    (not the env's), so the floor does not consume the demand streams."""
    rng = np.random.default_rng(seed)

    def act(obs, envs):
        return np.stack([
            rng.uniform(e.action_space.low, e.action_space.high).astype(np.float32) for e in envs
        ])
    return act


if __name__ == "__main__":
    from owmr_eval_common import eval_seed_block, rollout_seeds
    from owmr_scenarios import SCENARIOS

    per = rollout_seeds(scenario=SCENARIOS["base"], observation_mode="raw", action_mode="order_frac",
                        seeds=eval_seed_block(64), act_fn=random_act_fn(), batch=64, progress_every=0)
    print(f"random floor, 64 seeds: mean cost {per['cost_total'].mean():.1f}")

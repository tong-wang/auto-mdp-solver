"""Group (leave-one-out) baseline for PPO on a seed-keyed simulator — common random numbers in training (#E87).

The cnv world is keyed by the episode seed: two envs reset with the same seed draw the same hidden rate
and the same per-period demand and arrival epochs, whatever they order. So K envs that share every
episode seed and sample their own actions are K paired runs of one world — the comparison #E85's
fixed-schedule sweep made after the fact, made inside training.

Advantage of member i at period t = its Monte-Carlo return-to-go minus the mean return-to-go of the
OTHER K - 1 members at the same period. The baseline is built from trajectories that share i's
exogenous future but not its actions, so the policy gradient stays unbiased; the world-to-world
variance (episode profit sd ~520 at fig6b, ~1130 at fig6a) cancels out of the comparison. At t = 0
every member is in the identical state, so the first decision gets an exact paired comparison.

Requirements, asserted: K >= 2 divides n_envs; every episode lasts exactly the horizon (cnv has no
early exit), n_steps is a multiple of it, so every rollout holds whole, aligned episodes and the
returns-to-go are exact — the critic is not used for the advantage (it still trains, unread).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GroupSeedWrapper(gym.Wrapper):
    """Maps the seed SB3 hands env `rank` (training seed + rank) to training seed + rank // K.

    The members of a group are then seeded identically, so their per-env streams draw the same
    episode seed at every later (unseeded) reset: cnv's env uses np_random for nothing else.
    """

    def __init__(self, env: gym.Env, rank: int, group_size: int) -> None:
        super().__init__(env)
        self.rank, self.group_size = int(rank), int(group_size)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            seed = seed - self.rank + self.rank // self.group_size
        return self.env.reset(seed=seed, options=options)


def returns_to_go(rewards: np.ndarray, episode_starts: np.ndarray, gamma: float) -> np.ndarray:
    """(n_steps, n_envs) discounted returns-to-go, cut at episode starts; the rollout must end at an
    episode boundary (asserted by the caller)."""
    n_steps = rewards.shape[0]
    out = np.zeros_like(rewards, dtype=np.float64)
    acc = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in range(n_steps - 1, -1, -1):
        nxt_start = episode_starts[t + 1] if t + 1 < n_steps else np.ones(rewards.shape[1])
        acc = rewards[t] + gamma * acc * (1.0 - nxt_start)
        out[t] = acc
    return out


def group_advantages(rewards: np.ndarray, episode_starts: np.ndarray, gamma: float, group_size: int) -> np.ndarray:
    """Leave-one-out group advantage: G_i - mean_{j in group(i), j != i} G_j, envs grouped as rank // K."""
    g = returns_to_go(rewards, episode_starts, gamma)
    n_steps, n_envs = g.shape
    k = int(group_size)
    grp = g.reshape(n_steps, n_envs // k, k)
    loo = (grp.sum(axis=2, keepdims=True) - grp) / (k - 1)
    return (grp - loo).reshape(n_steps, n_envs)


class GroupBaselineCallback(BaseCallback):
    """Overwrites the rollout buffer's advantages with the group version, after SB3 computed GAE."""

    def __init__(self, group_size: int, horizon: int) -> None:
        super().__init__()
        self.k, self.horizon = int(group_size), int(horizon)

    def _on_training_start(self) -> None:
        n_envs, n_steps = self.model.n_envs, self.model.n_steps
        assert self.k >= 2 and n_envs % self.k == 0, f"group size {self.k} must be >= 2 and divide n_envs {n_envs}"
        assert n_steps % self.horizon == 0, f"n_steps {n_steps} must be a multiple of the horizon {self.horizon}"

    def _on_rollout_start(self) -> None:
        # every env must be at the start of an episode: whole, aligned episodes per rollout
        venv = self.training_env.venv if hasattr(self.training_env, "venv") else self.training_env
        periods = [e.unwrapped._state.period for e in venv.envs]
        assert all(p == 0 for p in periods), f"episodes not aligned with the rollout: periods {periods}"

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        assert np.all(buf.episode_starts[0] == 1.0), "rollout does not start at an episode start"
        adv = group_advantages(buf.rewards, buf.episode_starts, self.model.gamma, self.k)
        g0 = returns_to_go(buf.rewards, buf.episode_starts, self.model.gamma)[buf.episode_starts == 1.0]
        a0 = adv[buf.episode_starts == 1.0]
        buf.advantages[:] = adv.astype(buf.advantages.dtype)
        # the instrument: how much of the start-of-episode return spread the group removed
        self.logger.record("group/ret0_std", float(g0.std()))
        self.logger.record("group/adv0_std", float(a0.std()))
        self.logger.record("group/var_ratio_t0", float(g0.var() / max(a0.var(), 1e-12)))

"""Branch-and-grid group PPO (#E88): exact same-state action comparisons at a random early decision.

Builds on the group baseline (#E87, `cnv_group_baseline`): the training envs come in groups of K that
share every world (same hidden rate, same per-period demand and arrival epochs). Each group-episode
draws a branch time t_b from a truncated discrete exponential, P(t_b = k) ∝ exp(-k / tau), k = 0..tmax:

  t <  t_b  the group's first member (the parent) samples from the policy; the others copy its action,
            so at t_b all K members stand in the IDENTICAL state, facing the identical future;
  t == t_b  the K members take K DIFFERENT actions — the policy's top-K by probability ("topk") or a
            window around its mode ("window") — so the group never collapses at the branch;
  t >  t_b  every member samples from the policy on its own.

Advantages are each member's Monte-Carlo return-to-go minus the leave-one-out mean of its group's
(`group_advantages`), set to ZERO before the branch (the members acted identically there: nothing to
compare). The branch step's executed grid actions are stored with their log-probabilities under the
current policy, so PPO's clipped objective performs an all-action update over the K evaluated actions,
within its usual trust region; after the branch it is the #E87 group update. The critic still trains on
SB3's GAE returns and is not read for the advantage.
"""
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from cnv_group_baseline import group_advantages

PRE, BRANCH, POST = 0, 1, 2


def truncated_exponential_probs(tmax: int, tau: float) -> np.ndarray:
    k = np.arange(int(tmax) + 1, dtype=float)
    p = np.exp(-k / float(tau))
    return p / p.sum()


def branch_time_probs(tmax: int, tau: float, uniform: float, horizon: int) -> np.ndarray:
    """(1 - w) x truncated exponential on 0..tmax + w x uniform on 0..horizon-1 (#E91); w = 0 is #E88's law."""
    pe = truncated_exponential_probs(tmax, tau)
    if uniform <= 0.0:
        return pe
    p = np.zeros(int(horizon))
    p[: len(pe)] += (1.0 - uniform) * pe
    p += uniform / int(horizon)
    return p / p.sum()


def grid_actions(probs: np.ndarray, k: int, mode: str) -> np.ndarray:
    """K distinct actions for the branch: the policy's top-K, or a window around its mode."""
    n = len(probs)
    assert n >= k, f"action space {n} smaller than the group {k}"
    if mode == "topk":
        return np.argsort(-probs, kind="stable")[:k]
    m = int(np.argmax(probs))
    lo = min(max(0, m - 2), n - k)                     # mode - 2 .. mode + K - 3, kept inside the space
    return np.arange(lo, lo + k)


def _base_envs(env):
    e = env
    while hasattr(e, "venv"):
        e = e.venv
    return e.envs


class BranchGroupPPO(PPO):
    def __init__(self, *args, group_size: int = 8, branch_tmax: int = 20, branch_tau: float = 5.0,
                 branch_grid: str = "topk", branch_uniform: float = 0.0, branch_hybrid: bool = False, **kwargs) -> None:
        self.group_size, self.branch_tmax, self.branch_tau = int(group_size), int(branch_tmax), float(branch_tau)
        self.branch_uniform = float(branch_uniform)
        # #E97 hybrid: the group comparison only at the branch step; the critic's GAE (SB3's own advantage) after
        # the branch and on the parent's pre-branch steps; the copies stay at zero. Each kind is scaled to unit
        # standard deviation within the rollout, so neither swamps the other under PPO's minibatch normalisation.
        self.branch_hybrid = bool(branch_hybrid)
        self.branch_grid = str(branch_grid)
        super().__init__(*args, **kwargs)
        self._branch_probs = truncated_exponential_probs(self.branch_tmax, self.branch_tau)
        self._branch_rng = np.random.default_rng(None if self.seed is None else int(self.seed) + 7_919)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_branch_rng", "_phases", "_tb", "_last_gae", "_last_group_adv"]

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        assert isinstance(self.action_space, spaces.Discrete), "branch-and-grid needs a discrete action space"
        assert self._last_obs is not None
        k, n_envs = self.group_size, env.num_envs
        assert k >= 2 and n_envs % k == 0
        n_groups = n_envs // k
        envs = _base_envs(env)
        horizon = int(envs[0].unwrapped.scenario.horizon)
        assert n_rollout_steps % horizon == 0, "n_steps must be a multiple of the horizon"
        if self.branch_uniform > 0.0 and len(self._branch_probs) != horizon:   # #E91: the mixture needs the horizon
            self._branch_probs = branch_time_probs(self.branch_tmax, self.branch_tau, self.branch_uniform, horizon)
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        phases = np.zeros((n_rollout_steps, n_envs), dtype=np.int8)
        if not hasattr(self, "_tb") or self._tb is None:
            self._tb = np.zeros(n_groups, dtype=int)
        branch_stats = []

        for step in range(n_rollout_steps):
            periods = np.array([e.unwrapped._state.period for e in envs])
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                dist = self.policy.get_distribution(obs_tensor)
                values = self.policy.predict_values(obs_tensor)
                sampled = dist.sample().cpu().numpy().astype(np.int64)
                probs = dist.distribution.probs.cpu().numpy()
            executed = sampled.copy()
            for g in range(n_groups):
                mem = np.arange(g * k, (g + 1) * k)
                p = periods[mem]
                assert (p == p[0]).all(), f"group {g} members out of step: {p}"
                if p[0] == 0:
                    self._tb[g] = int(self._branch_rng.choice(len(self._branch_probs), p=self._branch_probs))
                if p[0] < self._tb[g]:
                    executed[mem] = sampled[mem[0]]                  # the parent's action, copied
                    phases[step, mem] = PRE
                elif p[0] == self._tb[g]:
                    assert np.allclose(self._last_obs[mem], self._last_obs[mem[0]]), "members not in the same state at the branch"
                    grid = grid_actions(probs[mem[0]], k, self.branch_grid)
                    executed[mem] = grid
                    phases[step, mem] = BRANCH
                    branch_stats.append((int(p[0]), int(np.argmax(probs[mem[0]]))))
                else:
                    phases[step, mem] = POST
            with th.no_grad():
                log_probs = dist.log_prob(th.as_tensor(executed, device=self.device))

            new_obs, rewards, dones, infos = env.step(executed)
            self.num_timesteps += n_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            rollout_buffer.add(self._last_obs, executed.reshape(-1, 1), rewards, self._last_episode_starts,
                               values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)   # value targets (GAE)
        assert np.all(rollout_buffer.episode_starts[0] == 1.0), "rollout does not start at an episode start"
        gae = rollout_buffer.advantages.astype(np.float64).copy()      # SB3's GAE, critic-based
        adv = group_advantages(rollout_buffer.rewards, rollout_buffer.episode_starts, self.gamma, k)
        adv[phases == PRE] = 0.0
        self._last_gae, self._last_group_adv = gae, adv.copy()
        if self.branch_hybrid:
            br = phases == BRANCH
            parent = np.zeros_like(br); parent[:, ::k] = True               # each group's first member
            use_gae = (phases == POST) | ((phases == PRE) & parent)
            hyb = np.zeros_like(adv)
            if br.any():
                hyb[br] = adv[br] / (adv[br].std() + 1e-8)
            if use_gae.any():
                hyb[use_gae] = gae[use_gae] / (gae[use_gae].std() + 1e-8)
            adv = hyb
        rollout_buffer.advantages[:] = adv.astype(rollout_buffer.advantages.dtype)
        self._phases = phases
        if branch_stats:
            tb = np.array([b[0] for b in branch_stats])
            self.logger.record("branch/tb_mean", float(tb.mean()))
            self.logger.record("branch/share_pre", float((phases == PRE).mean()))
            br = adv[phases == BRANCH].reshape(-1, k) if (phases == BRANCH).sum() % k == 0 else None
            if br is not None and len(br):
                self.logger.record("branch/adv_spread", float(br.std(axis=1).mean()))
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

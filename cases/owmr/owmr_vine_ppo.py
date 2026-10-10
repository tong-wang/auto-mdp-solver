"""Coordinate vine for PPO on the CatKeep head (ESCALATION #E30; the Dirichlet head from #E32) — a TRAINING-ALGORITHM lever.

What it is for. #E29 measured, on the best explicit arm, that one decision's
stakes are ~0.01–0.1 per period against a return sd of 11–17, that a common
demand world cuts the paired contrast's sd by 7–12× (60–150× in samples), and
that the +10 residual to the heuristic sits mostly in the retailer SPLIT as a
small systematic error at every state. #E11 showed the split rule is
representable; #E12 that PPO's single-sample gradient has ~1% signal. So the
lever supplies PPO with a lower-variance gradient estimate for each action
coordinate, from paired rollouts on a common world — TRPO's "vine" rollouts
(Schulman et al. 2015 §5.2) in a coordinate-wise form the human proposed.

The mechanism. The main rollout is UNTOUCHED: n_envs × n_steps transitions,
SB3's GAE, the critic, VecNormalize, the world count per update (cnv #E98
showed that cutting distinct worlds per update is what broke the group lever
on raw frames). On top of it, at each parent step and for each env, with
probability `vine_p` the env is FORKED (the post-receipt state copied, same
episode seed, so the same demand keys — the same world) into one parent copy
and one SIBLING PER ACTION COORDINATE:

    parent copy   executes the parent's sampled action a0;
    sibling j     executes a0 with coordinate j alone REDRAWN from the policy's
                  own marginal — the Gaussian for the order and each split
                  weight x_i, the categorical for the retention option (a
                  redraw may repeat the parent's option; the contrast is then
                  zero and the sample carries no gradient).

All copies then follow the stochastic policy for H − 1 more periods on the
common world with a common action-noise stream (CRN), accumulate the window's
cost, and are discarded. (#E29: the ranking of options is identical at H = 10
and H = 20 — one decision's effect is realised within the window — so there is
no bootstrap and the critic plays no part in the branch credit.)

The credit. Sibling j's advantage is −(C_j − C_0): its window cost against the
parent copy's, in reward units. The parent copy's cost does not depend on the
coordinate sibling j redrew, so it is a valid baseline FOR THAT COORDINATE; it
shares every other coordinate with the sibling, so the sibling is scored on the
log-prob of coordinate j ONLY (`GaussianCategorical.coord_log_prob`, a mask).
That is REINFORCE on each one-dimensional marginal with a paired common-world
control variate — #E12's "true" estimator — wrapped in PPO's clipped ratio on
the masked log-prob, within the same clip range and KL valve as the main pass.
Every sibling is an exact sample of the policy (a fresh draw of one coordinate
of a diagonal Gaussian × categorical), so the ratio is on-policy at collection.

The update. `train()` runs SB3's PPO pass on the main buffer unchanged, then a
SIDE PASS over the branch samples: `n_epochs` × minibatches of `batch_size`,
clipped surrogate on the masked ratio, no value or entropy term (the main pass
has them), gradient clipping, the KL valve on the masked ratio. The side
advantages are rescaled per rollout to the main buffer's advantage sd (cnv
#E97's rule — the two estimators sit in different units) times `vine_weight`.
NOTE (#E31): under Adam a constant scale on a pass's loss barely changes its
step size, so `vine_weight` is NOT a dose control; the side pass's impact is
its learning rate × its step count. `vine_lr_scale` (#E31) decouples the two
passes: a SEPARATE optimizer (same class and kwargs as the policy's, its own
Adam moments) over the same parameters, at lr_side = scale × lr(t) on the main
schedule. None (the #E30 form) shares the policy's optimizer and lr.

Seeds. A branch's continuation noise stream is drawn from the model's own
seeded generator; the demand world is the parent's. CRN within one branch's
copies, fresh randomness across branches — the siblings never see a world a
second time (the 2048 same-seed memorisation trap is repetition, not sharing).

Cost. (1 + n_coords) copies × H periods of env time per branch; at p = 0.2,
H = 10 and 7 coordinates (N = 5) about 15× the main rollout's env steps. Env
steps are cheap next to the 20-epoch update; the gate records the fps.

Heads (#E32). A head adapter (`_CatKeepHead`, `_DirichletHead`) defines the
coordinates: for CatKeep, order, x_1..x_N and the retention option; for the
Dirichlet head, the order and EACH SHARE through its Gamma variate (the
per-share form, the human's choice — see `_DirichletHead`): a share cannot be
moved alone on the simplex, but its Gamma can, and in Gamma space the density
factorises. Each head's policy exposes `evaluate_coord_log_prob(obs, actions,
aux)`, `aux` carrying the Gammas for the Dirichlet head.

Gate: `owmr_vine_gate.py [--head catkeep|dirichlet]`. Flags: `owmr_ppo_train.py --vine --vine-p --vine-h --vine-weight --vine-lr-scale`.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

import owmr_mdp as mdp
from owmr_catkeep_policy import KEEP_MENU
from owmr_dirichlet_policy import SHARE_EPS
from owmr_gym import KEEP_MENU_FINE, LOGIT_BOX, catkeep_shares, dirichlet_shares, observation

# gym action layout under order_catkeep: [order, k_index, x_1..x_N];
# Gaussian coordinate c (0 = order, 1..N = x_1..x_N) -> gym index
def _gym_index(c: int) -> int:
    return 0 if c == 0 else c + 1


def _base_envs(env):
    e = env
    while hasattr(e, "venv"):
        e = e.venv
    return e.envs


def _advance(sc, state, order, ship):
    nxt, info = mdp.advance2(sc, state, order, ship)
    cost = float(info["cost"]["total"])
    if not nxt.terminated:
        nxt = mdp.advance1(sc, nxt)
    return nxt, cost


def step_state(sc, state, action, menu=KEEP_MENU):
    """One period from a post-receipt state under an order_catkeep action (the
    gym's decode, on the MDP directly); returns (next post-receipt state, cost)."""
    a = np.asarray(action, dtype=np.float64)
    order = float(np.clip(a[0], 0.0, sc.order_max))
    W = max(0.0, float(state.wh_stock))
    shares, _ = catkeep_shares(np.concatenate([[a[1]], np.clip(a[2:], -LOGIT_BOX, LOGIT_BOX)]), menu)
    return _advance(sc, state, order, list(shares * W))


def step_state_dirichlet(sc, state, action):
    """One period under an order_dirichlet action (the gym's decode)."""
    a = np.asarray(action, dtype=np.float64)
    order = float(np.clip(a[0], 0.0, sc.order_max))
    W = max(0.0, float(state.wh_stock))
    shares, _ = dirichlet_shares(a[1:])
    return _advance(sc, state, order, list(shares * W))


class _CatKeepHead:
    """The vine's view of the CatKeep head: coordinates = order, x_1..x_N (Gaussian), k (categorical).

    `cat_grid` (#E34): instead of ONE redraw of the retention option (which repeats the
    parent's option 31% of the time, a zero contrast), one sibling per OTHER menu option —
    the full menu as the grid. A grid enumerates rather than samples, so each menu sibling
    carries the weight pi_old(k) of its option: the side surrogate sum_k pi_old(k) r_k A_k
    = sum_k pi_new(k) A_k is the exact all-action gradient of the expected window cost
    over the retention option at that state; the parent's own option has A = 0 by
    construction (same action, same world), so leaving it out changes nothing."""
    name = "catkeep"

    def __init__(self, action_dim: int, cat_grid: bool = False, n_options: int = len(KEEP_MENU)) -> None:
        self.n_cont = action_dim - 1
        self.n_coords = self.n_cont + 1
        self.cat_coord = self.n_cont
        self.cat_grid = bool(cat_grid)
        self.n_options = int(n_options)                 # 6 (order_catkeep) or 12 (order_catkeep_fine, #E34)

    def marginals(self, dist):
        return (dist.gauss.loc.cpu().numpy().astype(np.float64), dist.gauss.scale.cpu().numpy().astype(np.float64),
                dist.cat.probs.cpu().numpy().astype(np.float64))

    def siblings(self, a0, m, b, rng):
        loc, scale, probs = m
        out = []
        for c in range(self.n_cont):                            # Gaussian coordinates
            a = a0.copy(); a[_gym_index(c)] = loc[b, c] + scale[b, c] * rng.standard_normal()
            out.append((a, c, None))
        if self.cat_grid:                                       # the full menu, weighted by pi_old(k)
            pk = probs[b] / probs[b].sum()
            kp = int(round(a0[1]))
            for k in range(self.n_options):
                if k == kp:
                    continue
                a = a0.copy(); a[1] = float(k)
                out.append((a, self.cat_coord, None, float(pk[k])))
            return out
        a = a0.copy()                                           # the categorical: one redraw
        a[1] = float(rng.choice(self.n_options, p=probs[b] / probs[b].sum()))
        out.append((a, self.cat_coord, None))
        return out

    def cont_action(self, m, j, rng):
        mu, sg, pr = m
        z = rng.standard_normal(mu.shape[1])
        uu = rng.random()
        kk = int(np.searchsorted(np.cumsum(pr[j] / pr[j].sum()), uu))
        a = np.empty(self.n_cont + 1)
        a[0] = mu[j, 0] + sg[j, 0] * z[0]
        a[1] = float(min(kk, self.n_options - 1))
        a[2:] = mu[j, 1:] + sg[j, 1:] * z[1:]
        return a

    def step(self, sc, state, action):
        return step_state(sc, state, action, KEEP_MENU_FINE if self.n_options == len(KEEP_MENU_FINE) else KEEP_MENU)

    def is_dup(self, sib, parent, coord) -> bool:
        return coord == self.cat_coord and int(round(sib[1])) == int(round(parent[1]))


class _DirichletHead:
    """The vine's view of the Dirichlet head, PER-SHARE form (#E32, the human's
    choice): coordinates = the order (Gaussian) and each of the N + 1 shares
    through its Gamma variate. A Dirichlet(alpha) draw is G_i / sum G with
    independent G_i ~ Gamma(alpha_i, 1); the normalised vector is independent of
    the total T = sum G ~ Gamma(sum alpha, 1), so the parent's Gammas are
    reconstructed as G = s * T with a fresh T — a legitimate joint draw. Sibling i
    redraws G_i alone and renormalises: an exact Dirichlet(alpha) sample that
    differs from the parent in one underlying coordinate (every share shifts
    through the renormalisation, but all of that is a deterministic function of
    the Gammas it kept), scored on log Gamma(G_i; alpha_i) alone. The side buffer
    carries the Gammas as `aux`; the environment and the main pass see only the
    shares, floored and renormalised as the policy stores them."""
    name = "dirichlet"

    def __init__(self, action_dim: int) -> None:
        self.n_shares = action_dim - 1
        self.n_coords = 1 + self.n_shares
        self.cat_coord = -1

    def marginals(self, dist):
        return (dist.gauss.loc.cpu().numpy().astype(np.float64)[:, 0], dist.gauss.scale.cpu().numpy().astype(np.float64)[:, 0],
                dist.alpha.cpu().numpy().astype(np.float64))

    @staticmethod
    def _clean(s):
        s = np.maximum(s, SHARE_EPS)
        return s / s.sum()

    def siblings(self, a0, m, b, rng):
        loc, scale, alpha = m
        G0 = self._clean(a0[1:]) * rng.gamma(alpha[b].sum())        # the parent's Gammas: shares x an independent total
        out = []
        a = a0.copy(); a[0] = loc[b] + scale[b] * rng.standard_normal()
        out.append((a, 0, G0.copy()))
        for i in range(self.n_shares):
            G = G0.copy(); G[i] = rng.gamma(alpha[b, i])
            a = a0.copy(); a[1:] = self._clean(G / G.sum())
            out.append((a, 1 + i, G))
        return out

    def cont_action(self, m, j, rng):
        loc, scale, alpha = m
        a = np.empty(self.n_shares + 1)
        a[0] = loc[j] + scale[j] * rng.standard_normal()
        a[1:] = self._clean(rng.dirichlet(alpha[j]))
        return a

    step = staticmethod(step_state_dirichlet)

    def is_dup(self, sib, parent, coord) -> bool:
        return False


def head_for(policy, action_dim: int, cat_grid: bool = False):
    from owmr_catkeep_policy import CatKeepPolicy
    from owmr_dirichlet_policy import DirichletPolicy
    if isinstance(policy, CatKeepPolicy):
        return _CatKeepHead(action_dim, cat_grid=cat_grid, n_options=int(policy.n_options))
    if isinstance(policy, DirichletPolicy):
        return _DirichletHead(action_dim)
    raise TypeError(f"the vine has no head adapter for {type(policy).__name__}")


class VinePPO(PPO):
    def __init__(self, *args, vine_p: float = 0.2, vine_h: int = 10, vine_weight: float = 1.0,
                 vine_lr_scale: float | None = None, vine_cat_grid: bool = False, vine_record: bool = False, **kwargs) -> None:
        self.vine_p, self.vine_h, self.vine_weight = float(vine_p), int(vine_h), float(vine_weight)
        self.vine_cat_grid = bool(vine_cat_grid)
        self.vine_lr_scale = None if vine_lr_scale is None else float(vine_lr_scale)
        self.vine_record = bool(vine_record)          # keep per-branch detail for the gate
        self._side_optimizer = None
        super().__init__(*args, **kwargs)
        self._vine_rng = np.random.default_rng(None if self.seed is None else int(self.seed) + 104_729)
        self._side: dict[str, list] = {}
        self._side_detail: list[dict] = []
        self._vine_stats: dict[str, float] = {}

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_vine_rng", "_side", "_side_detail", "_vine_stats", "_side_optimizer", "_head"]

    def _side_opt(self):
        """The side pass's optimizer: the policy's own (#E30 form) or, with
        `vine_lr_scale`, a separate one over the same parameters at the scaled lr."""
        if self.vine_lr_scale is None:
            return self.policy.optimizer
        if self._side_optimizer is None:
            self._side_optimizer = self.policy.optimizer_class(
                self.policy.parameters(), lr=self.lr_schedule(1.0) * self.vine_lr_scale, **self.policy.optimizer_kwargs)
        lr = self.lr_schedule(self._current_progress_remaining) * self.vine_lr_scale
        for g in self._side_optimizer.param_groups:
            g["lr"] = lr
        self.logger.record("vine/side_lr", lr)
        return self._side_optimizer

    # ------------------------------------------------------------------ rollout
    def _reset_side(self) -> None:
        self._side = {"obs": [], "actions": [], "mask": [], "old_log_prob": [], "adv_raw": [], "aux": [], "w": []}
        self._side_detail = []

    def _policy_dist(self, obs_np: np.ndarray):
        obs = self.env.normalize_obs(obs_np.astype(np.float32))
        return self.policy.get_distribution(obs_as_tensor(obs, self.device))

    def _branch(self, env_idx: list[int], obs_batch: np.ndarray, actions: np.ndarray, envs) -> None:
        """Fork the chosen envs at their current post-receipt state, run the parent
        copy and one sibling per coordinate for H periods on the common world."""
        head = self._head
        n_coords = head.n_coords
        with th.no_grad():                                     # the policy's marginals at the parents' observations
            m = head.marginals(self.policy.get_distribution(obs_as_tensor(obs_batch[env_idx], self.device)))
        members, counts = [], []
        for b, i in enumerate(env_idx):
            u = envs[i].unwrapped
            parent_state, sc = u._state, u._scenario_ep
            a0 = actions[i].astype(np.float64)
            rng = np.random.default_rng(int(self._vine_rng.integers(0, 2**63 - 1)))
            sibs = [tuple(x) + ((1.0,) if len(x) == 3 else ()) for x in head.siblings(a0, m, b, rng)]   # (action, coord, aux, weight)
            first = [(a0.copy(), -1, None, 1.0)] + sibs
            counts.append(len(first))
            for a, c, aux, w in first:
                members.append({"state": copy.deepcopy(parent_state), "sc": sc, "first": a, "env": i, "coord": c, "aux": aux, "w": w})
            # the parent's own Gammas (per-share Dirichlet form) ride on the order sibling's aux, unchanged
            members[-len(first)]["aux_parent"] = None if sibs[0][2] is None else sibs[0][2]
            # one continuation-noise stream per branch, shared by its copies (CRN)
            seed = int(rng.integers(0, 2**63 - 1))
            for mem in members[-len(first):]:
                mem["noise"] = np.random.default_rng(seed)
        # the H-period windows, batched over all members alive
        cost = np.zeros(len(members))
        for h in range(self.vine_h):
            alive = [k for k, mem in enumerate(members) if not mem["state"].terminated]
            if not alive:
                break
            if h == 0:
                for k in alive:
                    members[k]["state"], c = head.step(members[k]["sc"], members[k]["state"], members[k]["first"])
                    cost[k] += c
                continue
            obs_np = np.stack([observation(members[k]["sc"], members[k]["state"], self._obs_mode,
                                           list(members[k]["state"].demand)) for k in alive])
            with th.no_grad():
                mj = head.marginals(self._policy_dist(obs_np))
            for j, k in enumerate(alive):
                a = head.cont_action(mj, j, members[k]["noise"])
                members[k]["state"], c = head.step(members[k]["sc"], members[k]["state"], a)
                cost[k] += c
        # credit: sibling against its branch's parent copy, scored on its own coordinate
        k = 0
        n_dup = 0
        for b, i in enumerate(env_idx):
            c0 = cost[k]
            for s_ in range(1, counts[b]):
                mem = members[k + s_]
                mask = np.zeros(n_coords, dtype=np.float32); mask[mem["coord"]] = 1.0
                adv = -(cost[k + s_] - c0)
                if head.is_dup(mem["first"], actions[i], mem["coord"]):
                    n_dup += 1
                self._side["obs"].append(self._last_obs[i].copy())
                self._side["actions"].append(mem["first"].astype(np.float32))
                self._side["mask"].append(mask)
                self._side["adv_raw"].append(float(adv))
                self._side["w"].append(float(mem["w"]))
                if mem["aux"] is not None:
                    self._side["aux"].append(np.asarray(mem["aux"], dtype=np.float32))
                if self.vine_record:
                    self._side_detail.append({"env": i, "coord": mem["coord"], "parent_action": actions[i].copy(),
                                              "action": mem["first"].copy(), "aux": None if mem["aux"] is None else mem["aux"].copy(),
                                              "parent_aux": None if mem["aux"] is None else members[k]["aux_parent"].copy(),
                                              "c0": float(c0), "cj": float(cost[k + s_]),
                                              "episode_seed": int(envs[i].unwrapped._state.episode_seed),
                                              "period": int(envs[i].unwrapped._state.period)})
            k += counts[b]
        self._vine_stats["dup"] = self._vine_stats.get("dup", 0) + n_dup
        self._vine_stats["branches"] = self._vine_stats.get("branches", 0) + len(env_idx)
        self._vine_stats["env_steps"] = self._vine_stats.get("env_steps", 0) + int(cost.size * self.vine_h)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        assert isinstance(self.action_space, spaces.Box), "the vine needs a Box action (CatKeep or Dirichlet head)"
        assert self._last_obs is not None
        if not hasattr(self, "_head"):
            self._head = head_for(self.policy, int(self.action_space.shape[0]), cat_grid=self.vine_cat_grid)
        envs = _base_envs(env)
        self._obs_mode = envs[0].unwrapped.observation_mode
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        self._reset_side()
        self._vine_stats = {}
        t0 = time.time()
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            if self.vine_p > 0.0:
                chosen = [i for i in range(env.num_envs) if self._vine_rng.random() < self.vine_p]
                if chosen:
                    self._branch(chosen, self._last_obs, actions, envs)
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        # the side samples' old log-probs under the policy that collected them
        if self._side["obs"]:
            with th.no_grad():
                obs = th.as_tensor(np.stack(self._side["obs"]), device=self.device)
                acts = th.as_tensor(np.stack(self._side["actions"]), device=self.device)
                mask = th.as_tensor(np.stack(self._side["mask"]), device=self.device)
                aux = th.as_tensor(np.stack(self._side["aux"]), device=self.device) if self._side["aux"] else None
                self._side["old_log_prob"] = (self.policy.evaluate_coord_log_prob(obs, acts, aux) * mask).sum(1).cpu().numpy().tolist()
        self._vine_stats["rollout_s"] = time.time() - t0
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------ update
    def train(self) -> None:
        super().train()                                        # the main pass, unchanged
        n = len(self._side.get("obs", []))
        self.logger.record("vine/n_side", n)
        self.logger.record("vine/w_mean", float(np.mean(self._side["w"])) if n else 0.0)
        self.logger.record("vine/branches", self._vine_stats.get("branches", 0))
        self.logger.record("vine/cat_duplicates", self._vine_stats.get("dup", 0))
        self.logger.record("vine/env_steps", self._vine_stats.get("env_steps", 0))
        self.logger.record("vine/rollout_s", self._vine_stats.get("rollout_s", 0.0))
        if n == 0:
            return
        self.policy.set_training_mode(True)
        clip_range = self.clip_range(self._current_progress_remaining)
        obs = th.as_tensor(np.stack(self._side["obs"]), device=self.device)
        acts = th.as_tensor(np.stack(self._side["actions"]), device=self.device)
        mask = th.as_tensor(np.stack(self._side["mask"]), device=self.device)
        old_lp = th.as_tensor(np.asarray(self._side["old_log_prob"], dtype=np.float32), device=self.device)
        aux = th.as_tensor(np.stack(self._side["aux"]), device=self.device) if self._side["aux"] else None
        wts = th.as_tensor(np.asarray(self._side["w"], dtype=np.float32), device=self.device)   # 1 for sampled siblings, pi_old(k) on a menu grid
        adv_raw = np.asarray(self._side["adv_raw"], dtype=np.float64)
        main_sd = float(np.std(self.rollout_buffer.advantages)) + 1e-8
        side_sd = float(adv_raw.std()) + 1e-8
        adv = th.as_tensor((adv_raw / side_sd * main_sd * self.vine_weight).astype(np.float32), device=self.device)
        self.logger.record("vine/adv_raw_sd", side_sd)
        self.logger.record("vine/adv_raw_mean_abs", float(np.abs(adv_raw).mean()))
        self.logger.record("vine/main_adv_sd", main_sd)
        # per-coordinate mean advantage (a sign/strength readout, cost units)
        coords = np.stack(self._side["mask"]).argmax(1)
        for c in range(mask.shape[1]):
            sel = coords == c
            if sel.any():
                self.logger.record(f"vine/adv_raw_c{c}", float(adv_raw[sel].mean()))
        opt = self._side_opt()
        kls, clips, losses = [], [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            perm = th.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                lp = (self.policy.evaluate_coord_log_prob(obs[idx], acts[idx], None if aux is None else aux[idx]) * mask[idx]).sum(1)
                ratio = th.exp(lp - old_lp[idx])
                a = adv[idx]
                loss = -(wts[idx] * th.min(a * ratio, a * th.clamp(ratio, 1 - clip_range, 1 + clip_range))).sum() / len(idx)
                with th.no_grad():
                    log_ratio = lp - old_lp[idx]
                    kl = float(th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy())
                kls.append(kl)
                clips.append(float(th.mean((th.abs(ratio - 1) > clip_range).float()).item()))
                losses.append(float(loss.item()))
                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                opt.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                opt.step()
            if not continue_training:
                break
        self.logger.record("vine/approx_kl", float(np.mean(kls)))
        self.logger.record("vine/clip_fraction", float(np.mean(clips)))
        self.logger.record("vine/loss", float(np.mean(losses)))
        self.logger.record("vine/epochs_run", epoch + 1)

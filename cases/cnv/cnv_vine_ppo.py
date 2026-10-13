"""The vine for PPO on cnv (#E110): the owmr case's coordinate vine (its #E30/#E31) on a
one-coordinate action, with the window run to the END of the episode.

Why. The branch-and-grid lever (#E88–#E104) built its same-world comparisons OUT OF the main rollout: a group
of K envs shares one world, so the distinct worlds per update fall to n_envs·n_steps / (K·horizon) — the cut
that broke R0's high rates (#E98) and that the 64 × 400 shape only bought back at a cost (#E99–#E100). The
vine keeps the main rollout UNTOUCHED — SB3's own loop, GAE, critic, VecNormalize, every world independent —
and adds the comparisons ON THE SIDE, on forked copies of the world.

The mechanism. At each main step, for each env, with probability `vine_p` the env is FORKED (its state and
frame stack copied; the episode's realised world — demand, arrival epochs — is shared, so the copies face the
identical future) into a PARENT COPY, which executes the main rollout's sampled order a0, and `vine_siblings`
SIBLINGS, each executing an order drawn afresh from the policy at the same observation (an exact policy
sample — not a grid). Every copy then follows the stochastic policy to the END of the episode on a COMMON
action-noise stream (the k-th period's draw is the same uniform u for every copy of the fork, mapped through
that copy's own action distribution by inverse CDF), and its profit from the fork onward is summed. The full
window is deliberate (the human, 2026-10-07): an over-order's information value is realised over every later
period, which a short window would cut off.

The credit. `vine_baseline="parent"` (owmr's form): sibling j's advantage is R_j − R_0, its profit from the fork
to the end against the parent copy's; the parent copy's profit does not depend on sibling j's draw, so it is a
valid baseline for it. `vine_baseline="loo"` (the one-dimensional case's own form, the human's question
2026-10-07): with ONE action coordinate there is no mask to keep and the parent copy's order a0 is itself an
exact policy sample, so EVERY copy of the fork is scored, each against the leave-one-out mean of the others
(RLOO at a fixed state) — 1 + siblings side samples per fork instead of `siblings`, at the same simulation
cost. Common world and common continuation noise make every contrast low-variance. No bootstrap — the critic
plays no part in the branch credit.

The update. `train()` = SB3's pass on the main buffer, unchanged, then a SIDE PASS over the sibling samples:
`n_epochs` × minibatches of `batch_size`, PPO's clipped surrogate on the ratio, no value or entropy term,
gradient clipping, the KL valve. The side advantages are rescaled per rollout to unit sd when the main pass
normalises its advantages (it does under t74's recipe) — else to the main buffer's advantage sd — times
`vine_weight` (cnv #E97 / owmr #E30: the two estimators sit in different units). `vine_lr_scale` None = the
policy's own optimizer and lr (owmr #E30's form, its best arm); a number = a separate optimizer at
scale × lr(t) (owmr #E31).

Batching. A fork's copies are not run to the end on the spot (that is 2 × up to 100 serial small forwards):
they join a POOL that advances ONE period per main step in a single batched forward, and the pool is flushed
to empty at the end of the rollout, before the update — so every continuation is played by the policy that
collected the rollout, and the side buffer is complete when `train()` runs.

Cost. Per main step ≈ vine_p × (1 + vine_siblings) × (mean remaining periods ≈ 50) extra env steps: 20 at
p 0.2 with one sibling. `num_timesteps` counts MAIN steps only (the budget stays comparable to the control's
in fresh worlds); the extra simulation is logged as `vine/env_steps`.

Gate: `cnv_vine_gate.py`. Flags: `cnv_ppo_train.py --vine --vine-p --vine-siblings
--vine-weight --vine-lr-scale`.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor


def _base_envs(env):
    e = env
    while hasattr(e, "venv"):
        e = e.venv
    return e.envs


def fork_env(e):
    """A copy of a (wrapped) cnv env that shares the immutable scenario objects: the source (whose attribute
    forwarding defeats deepcopy) and the episode's realised world (frozen; sharing it is what makes the
    copy's future identical to the parent's)."""
    u = e.unwrapped
    return copy.deepcopy(e, {id(u.scenario): u.scenario, id(u._scenario_ep): u._scenario_ep})


def inverse_cdf(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Discrete draws a_b = min{a : CDF_b(a) > u_b}, row-wise — the common-random-number map."""
    cdf = np.cumsum(probs, axis=1)
    cdf /= cdf[:, -1:]
    return np.minimum((cdf <= u[:, None]).sum(axis=1), probs.shape[1] - 1)


class VinePPO(PPO):
    def __init__(self, *args, vine_p: float = 0.2, vine_siblings: int = 1, vine_weight: float = 1.0,
                 vine_lr_scale: float | None = None, vine_baseline: str = "parent", vine_record: bool = False,
                 **kwargs) -> None:
        self.vine_p, self.vine_siblings, self.vine_weight = float(vine_p), int(vine_siblings), float(vine_weight)
        assert vine_baseline in ("parent", "loo") and int(vine_siblings) >= 1
        self.vine_baseline = str(vine_baseline)
        self.vine_lr_scale = None if vine_lr_scale is None else float(vine_lr_scale)
        self.vine_record = bool(vine_record)           # keep per-fork detail for the gate
        self._side_optimizer = None
        super().__init__(*args, **kwargs)
        self._vine_rng = np.random.default_rng(None if self.seed is None else int(self.seed) + 104_729)
        self._side: dict[str, list] = {}
        self._side_detail: list[dict] = []
        self._pool: list[dict] = []
        self._forks: list[dict] = []
        self._vine_stats: dict[str, float] = {}

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_vine_rng", "_side", "_side_detail", "_pool", "_forks",
                                                  "_vine_stats", "_side_optimizer"]

    def _side_opt(self):
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

    # ------------------------------------------------------------------ the vine
    def _probs(self, obs_np: np.ndarray) -> np.ndarray:
        obs = self.env.normalize_obs(obs_np.astype(np.float32))
        with th.no_grad():
            return self.policy.get_distribution(obs_as_tensor(obs, self.device)).distribution.probs.cpu().numpy().astype(np.float64)

    def _advance_pool(self) -> None:
        """Every live copy plays one period on the stochastic policy, its fork's common uniform per period."""
        live = [m for m in self._pool if not m["done"]]
        if not live:
            return
        probs = self._probs(np.stack([m["obs"] for m in live]))
        u = np.array([m["noise"].random() for m in live])
        acts = inverse_cdf(probs, u)
        for m, a in zip(live, acts):
            self._play(m, int(a))

    def _play(self, m: dict, a: int) -> None:
        obs, r, term, trunc, _ = m["env"].step(a)
        m["obs"], m["R"], m["done"] = obs, m["R"] + float(r), bool(term or trunc)
        self._vine_stats["env_steps"] = self._vine_stats.get("env_steps", 0) + 1

    def _fork(self, envs, actions: np.ndarray) -> None:
        """Fork the chosen envs at their current state: a parent copy on a0, `vine_siblings` fresh policy draws."""
        chosen = [i for i in range(len(envs)) if self._vine_rng.random() < self.vine_p]
        if not chosen:
            return
        probs = self._probs(self._last_obs[chosen])
        for b, i in enumerate(chosen):
            rng = np.random.default_rng(int(self._vine_rng.integers(0, 2**63 - 1)))
            a0 = int(np.asarray(actions[i]).reshape(-1)[0])
            firsts = [a0] + [int(inverse_cdf(probs[b:b + 1], rng.random(1))[0]) for _ in range(self.vine_siblings)]
            seed = int(rng.integers(0, 2**63 - 1))     # one continuation-noise stream per fork, shared by its copies
            members = []
            for a in firsts:
                m = {"env": fork_env(envs[i].env), "R": 0.0, "done": False, "noise": np.random.default_rng(seed)}
                self._play(m, a)
                members.append(m)
            self._pool.extend(members)
            self._forks.append({"members": members, "obs": self._last_obs[i].copy(), "firsts": firsts, "env": i,
                                "period": int(envs[i].unwrapped._state.period) if hasattr(envs[i].unwrapped._state, "period") else -1})
        self._vine_stats["forks"] = self._vine_stats.get("forks", 0) + len(chosen)

    def _settle(self) -> None:
        """Credit every fork whose copies have all finished; drop finished copies from the pool."""
        keep = []
        for f in self._forks:
            if all(m["done"] for m in f["members"]):
                R = np.array([m["R"] for m in f["members"]])
                if self.vine_baseline == "parent":    # siblings against the parent copy
                    scored, base = range(1, len(R)), np.full(len(R), R[0])
                else:                                 # every copy against the mean of the others
                    scored, base = range(len(R)), (R.sum() - R) / (len(R) - 1)
                for k in scored:
                    self._side["obs"].append(f["obs"])
                    self._side["actions"].append(f["firsts"][k])
                    self._side["adv_raw"].append(float(R[k] - base[k]))
                    if self.vine_record:
                        self._side_detail.append({"env": f["env"], "period": f["period"], "k": k, "firsts": list(f["firsts"]),
                                                  "R": R.tolist(), "adv": float(R[k] - base[k])})
            else:
                keep.append(f)
        self._forks = keep
        self._pool = [m for m in self._pool if not m["done"]]

    # ------------------------------------------------------------------ rollout (SB3 2.6's loop + the vine hook)
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        assert isinstance(self.action_space, spaces.Discrete), "the cnv vine is written for the discrete order"
        assert self._last_obs is not None, "No previous observation was provided"
        envs = _base_envs(env)
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        self._side = {"obs": [], "actions": [], "adv_raw": [], "old_log_prob": []}
        self._side_detail, self._pool, self._forks, self._vine_stats = [], [], [], {}
        t0 = time.time()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            if self.vine_p > 0.0:                       # the vine: advance the pool, then fork at this state
                self._advance_pool()
                self._fork(envs, actions)
                self._settle()
                self._vine_stats["pool_max"] = max(self._vine_stats.get("pool_max", 0), len(self._pool))
            clipped_actions = actions
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)
            for idx, done in enumerate(dones):
                if (done and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)):
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
        if self.vine_p > 0.0:                           # flush: every fork finishes under this rollout's policy
            while self._pool:
                self._advance_pool()
                self._settle()
            assert not self._forks, "a fork outlived the flush"
            if self._side["obs"]:
                with th.no_grad():
                    obs = obs_as_tensor(self.env.normalize_obs(np.stack(self._side["obs"]).astype(np.float32)), self.device)
                    acts = th.as_tensor(np.asarray(self._side["actions"]), device=self.device)
                    _, lp, _ = self.policy.evaluate_actions(obs, acts)
                self._side["old_log_prob"] = lp.cpu().numpy().tolist()
        self._vine_stats["rollout_s"] = time.time() - t0
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------ update
    def train(self) -> None:
        super().train()                                 # the main pass, unchanged
        n = len(self._side.get("obs", []))
        for k in ("forks", "env_steps", "pool_max", "rollout_s"):
            self.logger.record(f"vine/{k}", self._vine_stats.get(k, 0))
        self.logger.record("vine/n_side", n)
        if n == 0:
            return
        self.policy.set_training_mode(True)
        clip_range = self.clip_range(self._current_progress_remaining)
        obs = obs_as_tensor(self.env.normalize_obs(np.stack(self._side["obs"]).astype(np.float32)), self.device)
        acts = th.as_tensor(np.asarray(self._side["actions"]), device=self.device)
        old_lp = th.as_tensor(np.asarray(self._side["old_log_prob"], dtype=np.float32), device=self.device)
        adv_raw = np.asarray(self._side["adv_raw"], dtype=np.float64)
        target_sd = 1.0 if self.normalize_advantage else float(np.std(self.rollout_buffer.advantages)) + 1e-8
        side_sd = float(adv_raw.std()) + 1e-8
        adv = th.as_tensor((adv_raw / side_sd * target_sd * self.vine_weight).astype(np.float32), device=self.device)
        self.logger.record("vine/adv_raw_sd", side_sd)
        self.logger.record("vine/adv_raw_mean", float(adv_raw.mean()))
        self.logger.record("vine/share_zero_contrast", float(np.mean(adv_raw == 0.0)))
        opt = self._side_opt()
        kls, clips, losses = [], [], []
        continue_training, epoch = True, 0
        for epoch in range(self.n_epochs):
            perm = th.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                _, lp, _ = self.policy.evaluate_actions(obs[idx], acts[idx])
                ratio = th.exp(lp - old_lp[idx])
                a = adv[idx]
                loss = -th.min(a * ratio, a * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
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
        self.logger.record("vine/approx_kl", float(np.mean(kls)) if kls else 0.0)
        self.logger.record("vine/clip_fraction", float(np.mean(clips)) if clips else 0.0)
        self.logger.record("vine/loss", float(np.mean(losses)) if losses else 0.0)
        self.logger.record("vine/epochs_run", epoch + 1)

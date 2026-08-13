"""A5 probe bundle (ESCALATION.md #E9): arbitrate potential (a) representation
vs (b) exploration on the Gaussian branch — no RL training.

Parts (run in this order; results accumulate under results/{scenario}/a5_probe/):

  verify    vectorized simulator vs MabEnv, bit-exact payout/obs check
  equiv     equivariance error of the shipped policy: TV(pi(perm(s)), perm(pi(s)))
  profile   regret-accrual profile + belief-vs-behavior audit, PPO vs Thompson
  distill   build a Thompson-labeled dataset, train MLP / IndexNet / ContextNet
  disteval  evaluate the distilled policies at the 8192-seed protocol
  report    aggregate the JSONs into a verdict summary

The vectorized simulator reproduces the exact seed streams of
mab_uncertainty/mab_mdp (latents on the meta branch, payouts keyed
[arm, t, 0, 1, episode_seed, seed_salt]) and is trusted only because `verify`
asserts bit-exactness against MabEnv first.

Thompson inside `profile`/`distill` is re-simulated with a free rng stream —
distribution-identical to the benchmark policy but not its bit-exact
POLICY_STREAM draws; profile numbers are diagnostic, never records. The
distilled-policy evals in `disteval` DO use the faithful payout streams and
the full 0..8191 protocol, so they are quotable next to the bar.

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_a5_probe.py --part verify
    OMP_NUM_THREADS=1 python mab_a5_probe.py --part all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SCENARIO   = "gauss_K10_T1000"
K, T       = 10, 1000
SEED_SALT  = 4729
SHIPPED    = Path("results") / SCENARIO / "PPO_obsbayes_L1_seed1_20260729_115417"
OUTDIR     = Path("results") / SCENARIO / "a5_probe"

PROTO_SEEDS   = np.arange(8192)            # the standard protocol block
PROBE_SEEDS   = np.arange(2048)            # profile diagnosis (provisional)
DISTILL_SEEDS = np.arange(2_000_000, 2_002_048)   # disjoint from protocol+selection


# ---------------------------------------------------------------------------
# Vectorized faithful simulator (bit-exactness asserted by --part verify)
# ---------------------------------------------------------------------------

def realize_means(seeds: np.ndarray) -> np.ndarray:
    """Per-episode hidden arm means — meta branch, key [0, 0, seed, salt],
    ten sequential N(0,1) draws (mab_uncertainty.LatentGaussianPayout)."""
    out = np.empty((len(seeds), K))
    for e, s in enumerate(seeds):
        rng = np.random.default_rng(
            np.random.SeedSequence([0, 0, int(s), SEED_SALT]))
        out[e] = [rng.normal(0.0, 1.0) for _ in range(K)]
    return out


def payout_draws(seeds: np.ndarray, t: int, arms: np.ndarray,
                 means: np.ndarray) -> np.ndarray:
    """Pulled-arm payouts at period t — intrinsic key [arm, t, 0, 1, seed, salt]
    (mab_uncertainty.GaussianPayout.sample with the v0.5.9 per-arm draw slot)."""
    out = np.empty(len(seeds))
    for e, s in enumerate(seeds):
        a = int(arms[e])
        rng = np.random.default_rng(
            np.random.SeedSequence([a, t, 0, 1, int(s), SEED_SALT]))
        out[e] = rng.normal(means[e, a], 1.0)
    return out


class VecSim:
    """All episodes advance in lockstep; policy sees the bayes obs batch."""

    def __init__(self, seeds: np.ndarray):
        self.seeds   = np.asarray(seeds)
        self.means   = realize_means(self.seeds)
        self.opt     = self.means.max(axis=1)
        self.best    = self.means.argmax(axis=1)
        self.pulls   = np.zeros((len(seeds), K))
        self.payouts = np.zeros((len(seeds), K))
        self.t       = 0
        self.total   = np.zeros(len(seeds))

    def obs(self) -> np.ndarray:
        """bayes obs: [post_mean x K, post_sd x K, time_to_go] (mab_gym)."""
        pm  = self.payouts / (1.0 + self.pulls)
        psd = np.sqrt(1.0 / (1.0 + self.pulls))
        ttg = np.full((len(self.seeds), 1), float(T - self.t))
        return np.concatenate([pm, psd, ttg], axis=1).astype(np.float32)

    def step(self, arms: np.ndarray) -> np.ndarray:
        pay = payout_draws(self.seeds, self.t, arms, self.means)
        idx = np.arange(len(self.seeds))
        self.pulls[idx, arms]   += 1.0
        self.payouts[idx, arms] += pay
        self.total += pay
        self.t += 1
        return pay


# ---------------------------------------------------------------------------
# Policies over obs batches
# ---------------------------------------------------------------------------

class ShippedPPO:
    """The shipped artifact + its VecNormalize stats, batched; samples like
    `--stochastic` (distribution-identical, rng controlled here)."""

    def __init__(self, run_dir: Path, rng_seed: int = 123):
        import pickle
        import torch
        from stable_baselines3 import PPO
        self.torch = torch
        self.model = PPO.load(str(run_dir / f"{SCENARIO}_ppo.zip"), device="cpu")
        self.model.policy.set_training_mode(False)
        with open(run_dir / "vecnormalize.pkl", "rb") as fh:
            vn = pickle.load(fh)
        self.mu, self.var = vn.obs_rms.mean, vn.obs_rms.var
        self.clip, self.eps = vn.clip_obs, vn.epsilon
        self.rng = np.random.default_rng(rng_seed)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.mu) / np.sqrt(self.var + self.eps),
                       -self.clip, self.clip).astype(np.float32)

    def probs(self, obs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            dist = self.model.policy.get_distribution(
                self.torch.as_tensor(self.normalize(obs)))
            return dist.distribution.probs.numpy()

    def act(self, obs: np.ndarray) -> np.ndarray:
        p = self.probs(obs)
        return (p.cumsum(axis=1) > self.rng.random((len(p), 1))).argmax(axis=1)


class VecThompson:
    """Exact conjugate Thompson, vectorized; free rng stream (see module doc)."""

    def __init__(self, rng_seed: int = 7):
        self.rng = np.random.default_rng(rng_seed)

    def act(self, sim: VecSim) -> np.ndarray:
        pm  = sim.payouts / (1.0 + sim.pulls)
        psd = np.sqrt(1.0 / (1.0 + sim.pulls))
        return (pm + psd * self.rng.standard_normal(pm.shape)).argmax(axis=1)

    @staticmethod
    def probs_mc(pm: np.ndarray, psd: np.ndarray, draws: int,
                 rng: np.random.Generator) -> np.ndarray:
        """P(arm = argmax of posterior draws), MC estimate. pm/psd: (N, K)."""
        n = len(pm)
        counts = np.zeros((n, K))
        z = rng.standard_normal((draws, 1, K))
        for d in range(draws):
            counts[np.arange(n),
                   (pm + psd * z[d]).argmax(axis=1)] += 1.0
        return counts / draws


# ---------------------------------------------------------------------------
# part: verify
# ---------------------------------------------------------------------------

def part_verify() -> dict:
    from mab_gym import MabEnv
    from mab_scenarios import SCENARIOS

    n_ep, n_steps = 8, 40
    rng = np.random.default_rng(0)
    actions = rng.integers(0, K, size=(n_ep, n_steps))
    seeds = np.array([3, 7, 11, 4096, 65537, 900001, 2_000_000, 2_000_001])

    sim = VecSim(seeds)
    env = MabEnv(scenario=SCENARIOS[SCENARIO], observation_mode="bayes")
    max_pay_err = max_obs_err = 0.0
    env_pay = np.empty((n_ep, n_steps))
    for e, s in enumerate(seeds):
        obs_env, _ = env.reset(seed=int(s))
        assert np.array_equal(obs_env, sim.obs()[e]), "reset obs mismatch"
        for t in range(n_steps):
            obs_env, r, *_ = env.step(int(actions[e, t]))
            env_pay[e, t] = r
    for t in range(n_steps):
        pay = sim.step(actions[:, t])
        max_pay_err = max(max_pay_err, float(np.abs(pay - env_pay[:, t]).max()))
    # terminal obs check (episode e loop left env at its own state; redo one)
    obs_env, _ = env.reset(seed=int(seeds[0]))
    sim2 = VecSim(seeds[:1])
    for t in range(n_steps):
        obs_env, *_ = env.step(int(actions[0, t]))
        sim2.step(actions[:1, t])
    max_obs_err = float(np.abs(np.asarray(obs_env) - sim2.obs()[0]).max())

    assert max_pay_err == 0.0, f"payouts not bit-exact: {max_pay_err}"
    assert max_obs_err == 0.0, f"obs not bit-exact: {max_obs_err}"
    return {"episodes": n_ep, "steps": n_steps,
            "payout_err": max_pay_err, "obs_err": max_obs_err, "bit_exact": True}


# ---------------------------------------------------------------------------
# part: equiv
# ---------------------------------------------------------------------------

def part_equiv(n_states: int = 512, n_perms: int = 20) -> dict:
    pol = ShippedPPO(SHIPPED)
    sim = VecSim(PROBE_SEEDS[:256])
    states, stride = [], max(1, T // (n_states // 256 + 1))
    for t in range(T):
        if t % stride == 0 and len(states) * 256 < n_states * 8:
            states.append(sim.obs())
        sim.step(pol.act(sim.obs()))
    obs = np.concatenate(states)[:n_states * 8:8]           # spread over t

    rng = np.random.default_rng(0)
    tvs, mism = [], []
    base = pol.probs(obs)
    for _ in range(n_perms):
        perm = rng.permutation(K)
        obs_p = np.concatenate(
            [obs[:, :K][:, perm], obs[:, K:2 * K][:, perm], obs[:, 2 * K:]],
            axis=1)
        p_of_perm = pol.probs(obs_p)          # pi(perm(s))
        perm_of_p = base[:, perm]             # perm(pi(s))
        tvs.append(0.5 * np.abs(p_of_perm - perm_of_p).sum(axis=1))
        mism.append(p_of_perm.argmax(1) != perm_of_p.argmax(1))
    tv = np.concatenate(tvs)
    return {"n_states": int(len(obs)), "n_perms": n_perms,
            "tv_mean": float(tv.mean()), "tv_median": float(np.median(tv)),
            "tv_p90": float(np.quantile(tv, 0.9)),
            "argmax_mismatch_rate": float(np.concatenate(mism).mean())}


# ---------------------------------------------------------------------------
# part: profile
# ---------------------------------------------------------------------------

def _run_profiled(policy_act, sim: VecSim) -> dict:
    n = len(sim.seeds)
    reg_t = np.empty(T)
    pulls_at_900 = None
    for t in range(T):
        arms = policy_act(sim)
        reg_t[t] = float((sim.opt - sim.means[np.arange(n), arms]).mean())
        if t == T - 100:
            pulls_at_900 = sim.pulls.copy()
        sim.step(arms)
    pm = sim.payouts / (1.0 + sim.pulls)
    identified = pm.argmax(axis=1) == sim.best
    late = (sim.pulls - pulls_at_900).argmax(axis=1) == sim.best
    total_reg = sim.opt * T - sim.total
    cls = {
        "A_identified_and_committed": identified & late,
        "B_identified_not_committed": identified & ~late,
        "C_not_identified":           ~identified,
    }
    cum = reg_t.cumsum()
    return {
        "reward_mean": float(sim.total.mean()),
        "cum_regret_at": {str(t): float(cum[t - 1])
                          for t in (50, 100, 200, 500, 1000)},
        "late_regret_share_after_500": float(cum[-1] - cum[499]) / float(cum[-1]),
        "classes": {k: {"frac": float(v.mean()),
                        "regret_mean": float(total_reg[v].mean()) if v.any() else None}
                    for k, v in cls.items()},
    }


def part_profile() -> dict:
    pol = ShippedPPO(SHIPPED)
    ppo = _run_profiled(lambda s: pol.act(s.obs()), VecSim(PROBE_SEEDS))
    th  = _run_profiled(VecThompson().act, VecSim(PROBE_SEEDS))
    return {"n_seeds": len(PROBE_SEEDS), "provisional": True,
            "ppo_shipped": ppo, "thompson": th}


# ---------------------------------------------------------------------------
# part: distill
# ---------------------------------------------------------------------------

def _build_nets():
    import torch
    import torch.nn as nn

    class IndexNet(nn.Module):
        """logit_i = phi(pm_i, psd_i, ttg) — the pure index-policy class."""
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(3, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh(),
                                     nn.Linear(64, 1))
        def forward(self, x):
            pm, psd, ttg = x[:, :K], x[:, K:2 * K], x[:, 2 * K:] / T
            feats = torch.stack(
                [pm, psd, ttg.expand(-1, K)], dim=2)          # (N, K, 3)
            return self.phi(feats).squeeze(-1)                # (N, K)

    class ContextNet(nn.Module):
        """IndexNet + mean-pooled cross-arm context (DeepSets)."""
        def __init__(self):
            super().__init__()
            self.psi = nn.Sequential(nn.Linear(2, 32), nn.Tanh(),
                                     nn.Linear(32, 32), nn.Tanh())
            self.phi = nn.Sequential(nn.Linear(3 + 32, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh(),
                                     nn.Linear(64, 1))
        def forward(self, x):
            pm, psd, ttg = x[:, :K], x[:, K:2 * K], x[:, 2 * K:] / T
            per = torch.stack([pm, psd], dim=2)               # (N, K, 2)
            ctx = self.psi(per).mean(dim=1, keepdim=True).expand(-1, K, -1)
            feats = torch.cat(
                [pm.unsqueeze(2), psd.unsqueeze(2),
                 ttg.unsqueeze(2).expand(-1, K, -1), ctx], dim=2)
            return self.phi(feats).squeeze(-1)

    class MLPNet(nn.Module):
        """Plain (64,64) on the flat obs — the shipped policy's class."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2 * K + 1, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh(),
                                     nn.Linear(64, K))
        def forward(self, x):
            x = torch.cat([x[:, :2 * K], x[:, 2 * K:] / T], dim=1)
            return self.net(x)

    return {"mlp": MLPNet(), "index": IndexNet(), "context": ContextNet()}


def part_distill(mc_draws: int = 512, epochs: int = 20) -> dict:
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(2)

    # -- dataset: states from Thompson's own visitation ---------------------
    sim, th = VecSim(DISTILL_SEEDS), VecThompson()
    frames = []
    for t in range(T):
        if t % 4 == 0:
            frames.append(sim.obs())
        sim.step(th.act(sim))
    obs = np.concatenate(frames)                    # (512k, 21)
    rng = np.random.default_rng(1)
    obs = obs[rng.permutation(len(obs))][:424_000]

    lab_rng = np.random.default_rng(2)
    labels = np.concatenate([
        VecThompson.probs_mc(c[:, :K], c[:, K:2 * K], mc_draws, lab_rng)
        for c in np.array_split(obs, 32)])

    x = torch.as_tensor(obs)
    y = torch.as_tensor(labels.astype(np.float32))
    n_val = 24_000
    xt, yt, xv, yv = x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]

    results = {}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, net in _build_nets().items():
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for ep in range(epochs):
            for i in torch.randperm(len(xt)).split(8192):
                loss = -(yt[i] * torch.log_softmax(net(xt[i]), dim=1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            lp = torch.log_softmax(net(xv), dim=1)
            ce = -(yv * lp).sum(1).mean().item()
            ent = -(yv * torch.log(yv.clamp_min(1e-12))).sum(1).mean().item()
        torch.save(net.state_dict(), OUTDIR / f"distill_{name}.pt")
        results[name] = {"val_ce": ce, "val_kl": ce - ent,
                         "n_params": sum(p.numel() for p in net.parameters())}
        print(f"  {name:8s} val CE {ce:.4f}  KL {ce - ent:.4f}", flush=True)
    results["label_entropy"] = ent
    results["dataset"] = {"n_train": len(xt), "n_val": n_val,
                          "mc_draws": mc_draws, "epochs": epochs,
                          "seeds": "2000000..2002047 (disjoint)"}
    return results


def part_disteval() -> dict:
    import torch

    torch.set_num_threads(2)
    out = {}
    for name, net in _build_nets().items():
        sd = torch.load(OUTDIR / f"distill_{name}.pt", weights_only=True)
        net.load_state_dict(sd); net.eval()
        rng = np.random.default_rng(11)
        sim = VecSim(PROTO_SEEDS)
        for t in range(T):
            with torch.no_grad():
                p = torch.softmax(net(torch.as_tensor(sim.obs())), dim=1).numpy()
            arms = (p.cumsum(axis=1) > rng.random((len(p), 1))).argmax(axis=1)
            sim.step(arms)
        r = sim.total
        out[name] = {"reward_mean": float(r.mean()),
                     "reward_se": float(r.std(ddof=1) / np.sqrt(len(r))),
                     "n_seeds": len(r), "protocol": "0..8191 stochastic"}
        print(f"  distilled {name:8s} {r.mean():9.2f} ± {r.std(ddof=1)/np.sqrt(len(r)):.2f}",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# part: frameavg — test-time symmetrization of the SHIPPED policy (rung a0)
# ---------------------------------------------------------------------------

def part_frameavg(n_perms: int = 16) -> dict:
    """Frame averaging: pi_bar(s) = mean_sigma sigma^-1(pi(sigma(s))) over a
    fixed sample of arm permutations (identity included). Zero retraining —
    symmetrizes the incumbent MLP at inference; full 8192-seed protocol."""
    pol = ShippedPPO(SHIPPED)
    prng = np.random.default_rng(5)
    perms = [np.arange(K)] + [prng.permutation(K) for _ in range(n_perms - 1)]
    invs  = [np.argsort(p) for p in perms]
    act_rng = np.random.default_rng(11)
    sim = VecSim(PROTO_SEEDS)
    for t in range(T):
        obs = sim.obs()
        stacked = np.concatenate([
            np.concatenate([obs[:, :K][:, p], obs[:, K:2 * K][:, p],
                            obs[:, 2 * K:]], axis=1) for p in perms])
        probs = pol.probs(stacked).reshape(n_perms, len(obs), K)
        pbar = np.mean([probs[m][:, invs[m]] for m in range(n_perms)], axis=0)
        arms = (pbar.cumsum(axis=1) > act_rng.random((len(pbar), 1))).argmax(axis=1)
        sim.step(arms)
    r = sim.total

    # spec-§9 TSV beside the shipped run, so mdp_gates can consume it
    from mab_benchmark_common import reward_stats
    st = reward_stats(r)
    oracle = float((sim.opt * T).mean())
    tsv = SHIPPED / f"ppo_eval_stoch_frameavg{n_perms}_{SCENARIO}.tsv"
    tsv.write_text(
        f"# ppo obs=bayes policy=stochastic frame_avg={n_perms} | "
        f"n_seeds={len(r)} | vectorized evaluator (bit-exact vs MabEnv, "
        f"see a5_probe verify)\n"
        "scenario\tK\tT\tfamily\treward_mean\treward_var\t"
        "semivar_d\tsemivar_u\toracle_mean\tregret_mean\n"
        f"{SCENARIO}\t{K}\t{T}\tgaussian\t{st['reward_mean']:.6f}\t"
        f"{st['reward_var']:.6f}\t{st['semivar_d']:.6f}\t{st['semivar_u']:.6f}\t"
        f"{oracle:.6f}\t{oracle - st['reward_mean']:.6f}\n")
    return {"reward_mean": float(r.mean()),
            "reward_se": float(r.std(ddof=1) / np.sqrt(len(r))),
            "n_seeds": len(r), "n_perms": n_perms,
            "protocol": "0..8191 stochastic", "tsv": str(tsv),
            "note": "shipped MLP + VecNormalize, permutation-averaged probs"}


PARTS = {"verify": part_verify, "equiv": part_equiv, "profile": part_profile,
         "distill": part_distill, "disteval": part_disteval,
         "frameavg": part_frameavg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=[*PARTS, "all", "report"])
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    names = list(PARTS) if args.part == "all" else \
        [] if args.part == "report" else [args.part]
    for name in names:
        t0 = time.time()
        print(f"=== {name} ===", flush=True)
        res = PARTS[name]()
        res["wall_seconds"] = round(time.time() - t0, 1)
        (OUTDIR / f"{name}.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2), flush=True)

    if args.part in ("all", "report"):
        print("=== A5 summary ===")
        for name in PARTS:
            f = OUTDIR / f"{name}.json"
            if f.exists():
                print(f"--- {name}: {f}")


if __name__ == "__main__":
    main()

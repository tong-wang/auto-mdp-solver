"""Anneal probe (ESCALATION.md A7-rung-c, diagnosis stage): does the crowned
policy OVER-EXPLORE late? No RL training — eval-only, on the shipped artifact.

Hypothesis (operator, 2026-07-31): Thompson's randomization is posterior-
calibrated — its noise scales with post_sd and anneals as data accumulates —
while our policy's randomization is softmax entropy trained under a CONSTANT
ent_coef, which cannot anneal by construction. If true, the policy should
still be dithering in the late game, paying known posterior gaps for
information that is nearly worthless at small time-to-go.

Parts (results accumulate under results/{scenario}/anneal_probe/):

  profile  descriptive — along played trajectories: policy entropy H_t and
           explore-rate P(a_t != argmax post_mean) as functions of t, crown
           vs Thompson, plus the instantaneous true regret split into greedy
           vs explore pulls. Distinguishes the two diseases late regret
           conflates: dithering (high entropy, self-aware suboptimal pulls)
           vs mis-identification (low entropy, confidently wrong argmax).
  commit   causal — the crown evaluated stochastically for t < t0 and argmax
           for t >= t0, over a t0 grid, full 8192-seed protocol with CRN
           pairing (shared latents + shared action draws until divergence).
           t0=1000 is the shipped baseline; t0=0 is the known-bad pure
           argmax. A peak at interior t0 is causal proof that late
           stochasticity costs reward, and its height is a LOWER bound on
           what a proper anneal buys (a step function is the crudest
           schedule).
  temper   prescription t1 (A7-t1) — deployment-time temperature schedule:
           logits scaled by beta(t) = 1 + c*(t/T)^p, i.e. probs -> probs^beta
           renormalized. The (c, p) grid is tuned on SELECTION seeds (base
           1_000_000, disjoint from reporting — same block the training
           callback uses), and only the winner is scored once on the 0..8191
           protocol, CRN-paired against beta==1. The commit part's step
           switch is this schedule's p -> inf limit, so temper >= step is
           the expectation and step is the sanity floor.
  report   read the part JSONs against the pre-registered thresholds
           (ESCALATION.md log, 2026-07-31) and print the verdict.

The vectorized simulator and its faithful payout streams are imported from
mab_a5_probe (bit-exactness vs MabEnv asserted by its `verify` part).
Thompson inside `profile` is re-simulated with a free rng stream —
distribution-identical, diagnostic only, never a record. `commit` numbers use
the faithful streams and the 0..8191 protocol, so they are quotable.

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_anneal_probe.py --part all
    OMP_NUM_THREADS=1 python mab_anneal_probe.py --part commit --n-seeds 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mab_a5_probe import (K, T, PROBE_SEEDS, PROTO_SEEDS, SCENARIO,
                          VecSim, VecThompson)

CROWN  = (Path("results") / SCENARIO
          / "PPO_obsbayes_L1_policyindex_normobsFalse_seed1_20260729_185003")
OUTDIR = Path("results") / SCENARIO / "anneal_probe"

T0_GRID    = (0, 250, 500, 750, 875, 950, 990, 1000)
LATE_START = 900                       # the "late game" reporting window

# t1 schedule grid: beta(t) = 1 + c*(t/T)^p. Selection seeds mirror the
# training callback's block (mab_ppo_train --selection-base-seed default).
TEMPER_GRID     = tuple((c, p) for c in (2.0, 8.0, 32.0, 128.0)
                        for p in (0.5, 1.0, 2.0, 4.0))
SELECTION_SEEDS = np.arange(1_000_000, 1_002_048)


class CrownPolicy:
    """The crowned IndexPolicy artifact, batched. Unlike a5's ShippedPPO
    (built for the MLP crown, norm_obs=True), this respects the run's
    norm_obs flag — the equivariant crown trains on raw observations, and
    applying the tracked-but-unused obs_rms stats would corrupt them."""

    def __init__(self, run_dir: Path):
        import pickle
        import torch
        from stable_baselines3 import PPO
        self.torch = torch
        self.model = PPO.load(str(run_dir / f"{SCENARIO}_ppo.zip"), device="cpu")
        self.model.policy.set_training_mode(False)
        with open(run_dir / "vecnormalize.pkl", "rb") as fh:
            vn_state = pickle.load(fh).__dict__       # no venv: read the dict
        self._rms = vn_state["obs_rms"] if vn_state.get("norm_obs", True) else None
        self._clip = vn_state.get("clip_obs", 10.0)

    def probs(self, obs: np.ndarray) -> np.ndarray:
        if self._rms is not None:
            obs = np.clip((obs - self._rms.mean) / np.sqrt(self._rms.var + 1e-8),
                          -self._clip, self._clip)
        with self.torch.no_grad():
            dist = self.model.policy.get_distribution(
                self.torch.as_tensor(obs.astype(np.float32)))
            return dist.distribution.probs.numpy()


def _sample(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (p.cumsum(axis=1) > rng.random((len(p), 1))).argmax(axis=1)


# ---------------------------------------------------------------------------
# part: profile
# ---------------------------------------------------------------------------

def _entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1).mean())


def part_profile(n_seeds: int) -> dict:
    seeds = PROBE_SEEDS[:n_seeds]
    idx = np.arange(len(seeds))

    # --- crown, stochastic (the deployed criterion) --------------------------
    pol, rng = CrownPolicy(CROWN), np.random.default_rng(11)
    sim = VecSim(seeds)
    H = np.empty(T); explore = np.empty(T)
    reg_greedy = np.zeros(T); reg_explore = np.zeros(T); paid_gap = np.zeros(T)
    for t in range(T):
        obs = sim.obs()
        p = pol.probs(obs)
        a = _sample(p, rng)
        greedy = obs[:, :K].argmax(axis=1)          # the policy's own leader
        exp_mask = a != greedy
        H[t] = _entropy(p)
        explore[t] = float(exp_mask.mean())
        inst = sim.opt - sim.means[idx, a]          # true instantaneous regret
        reg_greedy[t]  = float(inst[~exp_mask].sum() / len(seeds))
        reg_explore[t] = float(inst[exp_mask].sum() / len(seeds))
        # the posterior gap the policy KNOWINGLY pays when it explores
        paid_gap[t] = float((obs[idx, greedy] - obs[idx, a])[exp_mask].sum()
                            / len(seeds))
        sim.step(a)

    # --- thompson reference (free rng — diagnostic only) ---------------------
    th, th_rng = VecThompson(), np.random.default_rng(29)
    sim2 = VecSim(seeds)
    th_explore = np.empty(T)
    th_H_grid = {}
    for t in range(T):
        pm = sim2.payouts / (1.0 + sim2.pulls)
        a = th.act(sim2)
        th_explore[t] = float((a != pm.argmax(axis=1)).mean())
        if t % 25 == 0:                             # entropy is MC — subsample
            psd = np.sqrt(1.0 / (1.0 + sim2.pulls))
            th_H_grid[str(t)] = _entropy(th.probs_mc(pm, psd, 256, th_rng))
        sim2.step(a)

    def win(x, a, b):
        return float(x[a:b].mean())

    windows = {f"{a}-{b}": {
        "H_crown": win(H, a, b),
        "explore_crown": win(explore, a, b),
        "explore_thompson": win(th_explore, a, b),
    } for a, b in ((0, 100), (250, 350), (500, 600), (LATE_START, T))}

    return {
        "n_seeds": len(seeds), "provisional": True,
        "windows": windows,
        "late_explore_regret_per_episode":
            float(reg_explore[LATE_START:].sum()),
        "late_greedy_regret_per_episode":
            float(reg_greedy[LATE_START:].sum()),
        "late_paid_posterior_gap_per_episode":
            float(paid_gap[LATE_START:].sum()),
        "curves": {"H_crown": H.round(4).tolist(),
                   "explore_crown": explore.round(4).tolist(),
                   "explore_thompson": th_explore.round(4).tolist(),
                   "reg_explore_crown": reg_explore.round(4).tolist(),
                   "reg_greedy_crown": reg_greedy.round(4).tolist(),
                   "H_thompson_grid": th_H_grid},
    }


# ---------------------------------------------------------------------------
# part: commit
# ---------------------------------------------------------------------------

def part_commit(n_seeds: int, t0_grid=T0_GRID) -> dict:
    seeds = PROTO_SEEDS[:n_seeds]
    pol = CrownPolicy(CROWN)
    totals = {}
    for t0 in t0_grid:
        rng = np.random.default_rng(11)   # CRN: shared draws until divergence
        sim = VecSim(seeds)
        for t in range(T):
            p = pol.probs(sim.obs())
            arms = _sample(p, rng) if t < t0 else p.argmax(axis=1)
            sim.step(arms)
        totals[t0] = sim.total
        print(f"  t0={t0:>4}  reward_mean={sim.total.mean():9.2f}", flush=True)

    base = totals[max(t0_grid)]           # t0=T == the shipped stochastic eval
    out = {"n_seeds": len(seeds), "protocol": "0..8191, faithful streams",
           "baseline_t0": max(t0_grid), "by_t0": {}}
    for t0 in t0_grid:
        r, d = totals[t0], totals[t0] - base
        se_p = float(d.std(ddof=1) / np.sqrt(len(d))) if t0 != max(t0_grid) else 0.0
        out["by_t0"][str(t0)] = {
            "reward_mean": float(r.mean()),
            "reward_se": float(r.std(ddof=1) / np.sqrt(len(r))),
            "delta_vs_baseline": float(d.mean()),
            "delta_se_paired": se_p,
            "z_paired": float(d.mean() / se_p) if se_p else 0.0,
        }
    interior = {t0: out["by_t0"][str(t0)] for t0 in t0_grid
                if t0 not in (0, max(t0_grid))}
    best_t0 = max(interior, key=lambda t0: interior[t0]["delta_vs_baseline"])
    out["best_interior"] = {"t0": best_t0, **interior[best_t0]}
    return out


# ---------------------------------------------------------------------------
# part: temper — prescription t1, the deployment-time beta(t) schedule
# ---------------------------------------------------------------------------

def _run_tempered(pol: CrownPolicy, seeds: np.ndarray, c: float, p: float,
                  rng_seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    sim = VecSim(seeds)
    for t in range(T):
        probs = pol.probs(sim.obs())
        beta = 1.0 + c * (t / T) ** p
        # probs^beta renormalized == softmax(beta * logits); in log space —
        # at beta ~ 100 direct powers underflow every entry for a spread p
        logp = beta * np.log(np.clip(probs, 1e-300, None))
        logp -= logp.max(axis=1, keepdims=True)
        pb = np.exp(logp)
        pb /= pb.sum(axis=1, keepdims=True)
        sim.step(_sample(pb, rng))
    return sim.total


def part_temper(n_seeds: int) -> dict:
    pol = CrownPolicy(CROWN)
    sel = SELECTION_SEEDS[:n_seeds]

    scores = {}
    for c, p in TEMPER_GRID:
        r = _run_tempered(pol, sel, c, p)
        scores[(c, p)] = float(r.mean())
        print(f"  c={c:>5.1f} p={p:>3.1f}  selection_mean={r.mean():9.2f}",
              flush=True)
    (c_w, p_w) = max(scores, key=scores.get)

    # winner only, once, on the reporting protocol — CRN-paired vs beta==1
    r_win  = _run_tempered(pol, PROTO_SEEDS, c_w, p_w)
    r_base = _run_tempered(pol, PROTO_SEEDS, 0.0, 1.0)     # beta(t) == 1
    d = r_win - r_base
    se_p = float(d.std(ddof=1) / np.sqrt(len(d)))

    from mab_benchmark_common import reward_stats
    st = reward_stats(r_win)
    oracle = float((VecSim(PROTO_SEEDS).opt * T).mean())
    tsv = CROWN / f"ppo_eval_stoch_temper_{SCENARIO}.tsv"
    tsv.write_text(
        f"# ppo obs=bayes policy=stochastic temper c={c_w} p={p_w} "
        f"(A7-t1, selection-tuned on 1e6 block) | n_seeds={len(r_win)} | "
        f"vectorized evaluator (bit-exact vs MabEnv, see a5_probe verify)\n"
        "scenario\tK\tT\tfamily\treward_mean\treward_var\t"
        "semivar_d\tsemivar_u\toracle_mean\tregret_mean\n"
        f"{SCENARIO}\t{K}\t{T}\tgaussian\t{st['reward_mean']:.6f}\t"
        f"{st['reward_var']:.6f}\t{st['semivar_d']:.6f}\t{st['semivar_u']:.6f}\t"
        f"{oracle:.6f}\t{oracle - st['reward_mean']:.6f}\n")

    return {"selection_n_seeds": len(sel),
            "selection_scores": {f"c={c},p={p}": v
                                 for (c, p), v in sorted(scores.items())},
            "winner": {"c": c_w, "p": p_w},
            "protocol": "winner once on 0..8191, faithful streams",
            "reward_mean": float(r_win.mean()),
            "reward_se": float(r_win.std(ddof=1) / np.sqrt(len(r_win))),
            "delta_vs_beta1": float(d.mean()),
            "delta_se_paired": se_p,
            "z_paired": float(d.mean() / se_p) if se_p else 0.0,
            "tsv": str(tsv)}


# ---------------------------------------------------------------------------
# part: report — the pre-registered reads (ESCALATION.md log, 2026-07-31)
# ---------------------------------------------------------------------------

def part_report(_: int = 0) -> dict:
    commit = json.loads((OUTDIR / "commit.json").read_text())
    profile = json.loads((OUTDIR / "profile.json").read_text())
    best = commit["best_interior"]
    late = profile["windows"][f"{LATE_START}-{T}"]

    material   = best["delta_vs_baseline"] >= 15.0 and best["z_paired"] >= 3.0
    detectable = best["delta_vs_baseline"] > 0.0 and best["z_paired"] >= 3.0
    dithering  = late["explore_crown"] >= 2.0 * late["explore_thompson"]

    verdict = (
        "OVER-EXPLORATION CONFIRMED, MATERIAL" if material else
        "over-exploration detectable but immaterial (<15)" if detectable else
        "REFUTED at the crown: late stochasticity does not cost reward")
    return {
        "verdict": verdict,
        "commit_best_interior": best,
        "late_window": late,
        "profile_says_dithering": bool(dithering),
        "note": ("commit is causal and governs; profile classifies the "
                 "mechanism (dithering vs mis-identification). Re-check "
                 "against A3's winner at harvest — the crown trains at "
                 "ent_coef 0.01 and a tuned level may shrink the floor."),
    }


PARTS = {"profile": part_profile, "commit": part_commit,
         "temper": part_temper, "report": part_report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=[*PARTS, "all"])
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="override episode count (smoke); defaults: "
                         "profile 2048, commit 8192")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    defaults = {"profile": len(PROBE_SEEDS), "commit": len(PROTO_SEEDS),
                "temper": len(SELECTION_SEEDS), "report": 0}
    for name in (list(PARTS) if args.part == "all" else [args.part]):
        t0 = time.time()
        print(f"=== {name} ===", flush=True)
        res = PARTS[name](args.n_seeds or defaults[name])
        res["wall_seconds"] = round(time.time() - t0, 1)
        (OUTDIR / f"{name}.json").write_text(json.dumps(res, indent=2))
        summary = {k: v for k, v in res.items() if k != "curves"}
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

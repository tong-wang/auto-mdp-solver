"""Coverage round probes — never-touched arms (#E31; COVERAGE_PLAN.md).

Parts (artifacts under results/{scenario}/interpret/, coverage_ prefix):

  coverage  rung 0 — end-of-episode pull matrices for the full roster on the
            8192 CRN block: any-arm coverage, P(best untouched), first-pull
            time of the true best. Full-block runs write coverage_{name}.npz;
            coverage.json only on a full-roster invocation (smoke runs print
            and write nothing — the metrics_summary.json lesson).
  floor     P3 — the H-obj/H-opt discriminator, class side: replay the
            fitted 12-knot rule (fitc.json knots_traj) with c(ttg) floored
            (flat F, or a UCB-style lambda*sqrt(2 ln t) late term) on the
            CRN block. If some floor raises the MEAN, the deterministic
            class contains a better optimum PPO missed (H-opt); if every
            floor loses, tail insurance is unaffordable in-class (H-obj).
            Gate: the unfloored base must land on the recorded 1459.56.
  ckpt      P4 — the discriminator, trajectory side: coverage + return +
            entropy across the 20 saved 1M-step checkpoints of A8-a-s3 and
            A11-a-s2, 2048-seed CRN block (rates at the 1.5-8% level are
            resolvable; se ~ 0.3%). No retraining.
  frontier  P1 — WHERE each policy abandons: synthetic one-leader beliefs
            (leader pm = x with n = T - ttg pulls, the other 9 arms
            unpulled), per-arm pi(unpulled) over an (x, ttg) grid, and the
            lockout threshold x*(ttg) where pi < 1/ttg. Thompson reference
            via probs_mc.
  massdecay P2 — WHEN the gradient dies: pi_t mass on currently-unpulled
            arms, and for arms that end untouched the expected total
            touches R0 = sum_t pi_t and the point of no return (first t
            with remaining mass < 0.5). Net legs replay the SAME episodes
            as part coverage (same seeds, same rng streams).

Every sweep is CRN (PROTO_SEEDS latents, keyed-period-seed draws), every
policy played as deployed (nets sample their softmax, thompson samples its
posterior, rules deterministic; per-leg action rng — the per-leg-streams
lesson).

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_coverage_probe.py --n-seeds 128   # smoke
    OMP_NUM_THREADS=1 python mab_coverage_probe.py                 # rung 0
    OMP_NUM_THREADS=1 python mab_coverage_probe.py --part floor \
        --variants base,floor:2.0,log:0.5
    OMP_NUM_THREADS=1 python mab_coverage_probe.py --part ckpt \
        --run crown_s3 --ckpts 1,2,3,4,5
    OMP_NUM_THREADS=1 python mab_coverage_probe.py --part frontier
    OMP_NUM_THREADS=1 python mab_coverage_probe.py --part massdecay
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from mab_a5_probe import K, T, PROTO_SEEDS, SCENARIO, VecSim, VecThompson
from mab_anneal_probe import CrownPolicy, _sample
from mab_interpret import OUTDIR, RUNS, VecGreedy, VecUcb1, _entropy_rows
from mab_stats_probe import STATS_RUNS, StatsIndex, _stats_obs

FULL_BLOCK = 8192
CKPT_BLOCK = 2048

#: rung-0 sweep order: references first (cheap, anchor the gate), then nets
ROSTER = (["thompson", "ucb1", "greedy"]
          + list(RUNS) + list(STATS_RUNS))


def _anchor_runs() -> dict[str, Path]:
    """#E31 rung 5 — the KL-to-Thompson anchored runs. Same class, HP, budget
    and obs mode as the crown with one lever added, so they read through
    `CrownPolicy` unchanged. Deliberately NOT in ROSTER: the rung-0 census is
    a committed artifact, and a subset run must not rewrite it."""
    base = Path(__file__).resolve().parent / "results" / SCENARIO
    out = {}
    for b in ("0.003", "0.01", "0.03"):
        hits = sorted(base.glob(f"*_anchor{b}_normobsFalse_seed1_*"))
        if len(hits) == 1:
            out[f"anchor{b}"] = hits[0]
        elif hits:
            raise AssertionError(f"anchor {b}: {len(hits)} run dirs match: {hits}")
    # absent is normal: `results/` is gitignored, so a fresh clone has no runs.
    # Import must still succeed — only *asking for* a missing subject may fail.
    return out


ANCHOR_RUNS = _anchor_runs()

#: every bayes-obs artifact this probe can score, by name
BAYES_RUNS = {**RUNS, **ANCHOR_RUNS}


# ---------------------------------------------------------------------------
# part: coverage (rung 0)
# ---------------------------------------------------------------------------

def sweep(name: str, n_seeds: int) -> tuple[dict, dict]:
    """One CRN sweep; returns (summary row, per-seed arrays)."""
    seeds = PROTO_SEEDS[:n_seeds]
    n = len(seeds)
    idx = np.arange(n)
    sim = VecSim(seeds)
    best = sim.best

    kind = "bayes" if name in BAYES_RUNS else ("stats" if name in STATS_RUNS
                                               else "ref")
    if kind == "bayes":
        pol = CrownPolicy(BAYES_RUNS[name])
    elif kind == "stats":
        pol = StatsIndex(name)
    elif name == "thompson":
        pol = VecThompson(rng_seed=7)
    elif name == "ucb1":
        pol = VecUcb1()
    else:
        pol = VecGreedy()
    rng = np.random.default_rng(101)     # softmax draws, per-leg stream

    pseudo = np.zeros(n)
    first_best_t = np.full(n, T)
    tic = time.time()
    for t in range(T):
        if kind == "bayes":
            a = _sample(pol.probs(sim.obs()), rng)
        elif kind == "stats":
            a = _sample(pol.probs(_stats_obs(sim)), rng)
        else:
            a = pol.act(sim)
        pseudo += sim.opt - sim.means[idx, a]
        newly = (a == best) & (first_best_t == T)
        first_best_t[newly] = t
        sim.step(a)

    pulls = sim.pulls
    n_untouched = (pulls == 0).sum(axis=1)
    best_pulls = pulls[idx, best]
    unt = best_pulls == 0
    few = (best_pulls > 0) & (best_pulls <= 2)
    pm_final = sim.payouts / (1.0 + pulls)
    touched_first = first_best_t[~unt]

    row = {
        "policy": name, "n_seeds": n,
        "reward_mean": float(sim.total.mean()),
        "pseudo_mean": float(pseudo.mean()),
        "p_any_untouched": float((n_untouched > 0).mean()),
        "mean_n_untouched": float(n_untouched.mean()),
        "p_best_untouched": float(unt.mean()),
        "p_best_le2": float((best_pulls <= 2).mean()),
        "pseudo_best_untouched": (float(pseudo[unt].mean())
                                  if unt.any() else None),
        "pseudo_best_1to2": (float(pseudo[few].mean())
                             if few.any() else None),
        "pseudo_rest": float(pseudo[best_pulls > 2].mean()),
        "first_best_t_median": float(np.median(touched_first)),
        "first_best_t_p90": float(np.quantile(touched_first, 0.90)),
        "leader_pm_final_untouched": (float(pm_final.max(axis=1)[unt].mean())
                                      if unt.any() else None),
        "sweep_seconds": round(time.time() - tic, 1),
    }
    per_seed = {"pulls": pulls, "payouts": sim.payouts, "pseudo": pseudo,
                "first_best_t": first_best_t.astype(float)}
    return row, per_seed


def part_coverage(args) -> None:
    roster = args.policies.split(",") if args.policies else ROSTER
    record = args.n_seeds == FULL_BLOCK
    rows = []
    for name in roster:
        row, per_seed = sweep(name, args.n_seeds)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if record:
            np.savez(OUTDIR / f"coverage_{name}.npz", **per_seed)
    if record and roster == ROSTER:
        (OUTDIR / "coverage.json").write_text(json.dumps(rows, indent=2))
        print(f"wrote {OUTDIR / 'coverage.json'}", flush=True)
    elif record and args.out_tag:
        # a subset run records under its own name; the rung-0 census above is
        # a committed artifact and is never rewritten by one
        out = OUTDIR / f"coverage_{args.out_tag}.json"
        out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# part: floor (P3)
# ---------------------------------------------------------------------------

#: base + the two insurance families; a chunk invocation picks a subset
FLOOR_VARIANTS = (["base"]
                  + [f"floor:{v}" for v in (1.6, 1.8, 2.0, 2.2, 2.5, 3.0)]
                  + [f"log:{v}" for v in (0.3, 0.5, 0.7, 1.0)])


def _load_knots() -> tuple[np.ndarray, np.ndarray]:
    kn = json.loads((OUTDIR / "fitc.json").read_text())["knots_traj"]
    ttgs = np.array(sorted(int(v) for v in kn), dtype=float)
    return ttgs, np.array([kn[str(int(v))] for v in ttgs])


def sweep_rule(variant: str, n_seeds: int) -> tuple[dict, np.ndarray]:
    """Deterministic argmax(pm + c_eff(t) * psd); variant sets c_eff."""
    ttgs, cs = _load_knots()
    kind, _, raw = variant.partition(":")
    v = float(raw) if raw else None
    seeds = PROTO_SEEDS[:n_seeds]
    n = len(seeds)
    idx = np.arange(n)
    sim = VecSim(seeds)
    best = sim.best
    pseudo = np.zeros(n)
    tic = time.time()
    for t in range(T):
        c = float(np.interp(float(T - t), ttgs, cs))
        if kind == "floor":
            c = max(c, v)
        elif kind == "log":
            c = max(c, v * np.sqrt(2.0 * np.log(t + 1)))
        pm = sim.payouts / (1.0 + sim.pulls)
        psd = np.sqrt(1.0 / (1.0 + sim.pulls))
        a = (pm + c * psd).argmax(axis=1)
        pseudo += sim.opt - sim.means[idx, a]
        sim.step(a)
    bp = sim.pulls[idx, best]
    row = {
        "variant": variant, "n_seeds": n,
        "reward_mean": float(sim.total.mean()),
        "reward_se": float(sim.total.std(ddof=1) / np.sqrt(n)),
        "pseudo_mean": float(pseudo.mean()),
        "p_any_untouched": float(((sim.pulls == 0).any(axis=1)).mean()),
        "p_best_untouched": float((bp == 0).mean()),
        "p_best_le2": float((bp <= 2).mean()),
        "pseudo_p99": float(np.quantile(pseudo, 0.99)),
        "sweep_seconds": round(time.time() - tic, 1),
    }
    return row, pseudo


def part_floor(args) -> None:
    variants = (args.variants.split(",") if args.variants
                else FLOOR_VARIANTS)
    record = args.n_seeds == FULL_BLOCK
    for variant in variants:
        row, pseudo = sweep_rule(variant, args.n_seeds)
        print(json.dumps(row), flush=True)
        if record:
            tag = variant.replace(":", "")
            np.save(OUTDIR / f"coverage_floorpseudo_{tag}.npy", pseudo)


# ---------------------------------------------------------------------------
# part: ckpt (P4)
# ---------------------------------------------------------------------------

CKPT_SUBJECTS = {"crown_s3": (RUNS["crown_s3"], "bayes"),
                 "a11_s2": (STATS_RUNS["a11_s2"], "stats")}


class CkptPolicy:
    """A training checkpoint, batched. Both subjects are normobsFalse runs;
    asserted from the run's vecnormalize.pkl before trusting raw obs."""

    def __init__(self, run_dir: Path, ckpt_millions: int):
        import torch
        from stable_baselines3 import PPO
        self.torch = torch
        with open(run_dir / "vecnormalize.pkl", "rb") as fh:
            assert not pickle.load(fh).__dict__.get("norm_obs", True), \
                f"{run_dir.name} is a norm_obs run; CkptPolicy assumes raw obs"
        path = run_dir / "checkpoints" / f"ppo_mab_{ckpt_millions * 10**6}_steps.zip"
        self.model = PPO.load(str(path), device="cpu")
        self.model.policy.set_training_mode(False)

    def probs(self, obs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.model.policy.get_distribution(
                self.torch.as_tensor(obs.astype(np.float32))
            ).distribution.probs.numpy()


def part_ckpt(args) -> None:
    run_dir, kind = CKPT_SUBJECTS[args.run]
    ckpts = [int(v) for v in args.ckpts.split(",")]
    rows = []
    for m in ckpts:
        pol = CkptPolicy(run_dir, m)
        seeds = PROTO_SEEDS[:args.n_seeds]
        n = len(seeds)
        idx = np.arange(n)
        sim = VecSim(seeds)
        best = sim.best
        rng = np.random.default_rng(101)
        pseudo = np.zeros(n)
        h_sum = 0.0
        tic = time.time()
        for t in range(T):
            obs = sim.obs() if kind == "bayes" else _stats_obs(sim)
            p = pol.probs(obs)
            a = _sample(p, rng)
            h_sum += _entropy_rows(p).mean()
            pseudo += sim.opt - sim.means[idx, a]
            sim.step(a)
        bp = sim.pulls[idx, best]
        row = {
            "run": args.run, "ckpt_M": m, "n_seeds": n,
            "reward_mean": float(sim.total.mean()),
            "pseudo_mean": float(pseudo.mean()),
            "entropy_mean": float(h_sum / T),
            "p_any_untouched": float(((sim.pulls == 0).any(axis=1)).mean()),
            "mean_n_untouched": float((sim.pulls == 0).sum(axis=1).mean()),
            "p_best_untouched": float((bp == 0).mean()),
            "p_best_le2": float((bp <= 2).mean()),
            "sweep_seconds": round(time.time() - tic, 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.n_seeds == CKPT_BLOCK:
        tag = f"{min(ckpts)}-{max(ckpts)}M"
        out = OUTDIR / f"coverage_ckpt_{args.run}_{tag}.json"
        out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# part: frontier (P1)
# ---------------------------------------------------------------------------

X_GRID = np.round(np.arange(0.0, 3.001, 0.05), 2)
TTG_SET = (900, 750, 500, 250, 100, 50)


def _frontier_beliefs(ttg: int) -> tuple[np.ndarray, np.ndarray]:
    """One-leader synthetic states over X_GRID: leader (arm 0) holds all
    n = T - ttg pulls at posterior mean x; arms 1..K-1 unpulled. Returns
    (bayes obs, stats obs) batches, one row per x."""
    g = len(X_GRID)
    n_lead = float(T - ttg)
    pm = np.zeros((g, K)); pm[:, 0] = X_GRID
    psd = np.ones((g, K)); psd[:, 0] = 1.0 / np.sqrt(1.0 + n_lead)
    ttg_col = np.full((g, 1), float(ttg))
    bayes = np.concatenate([pm, psd, ttg_col], axis=1).astype(np.float32)
    pulls = np.zeros((g, K)); pulls[:, 0] = n_lead
    payouts = np.zeros((g, K)); payouts[:, 0] = X_GRID * (1.0 + n_lead)
    stats = np.concatenate([pulls, payouts, ttg_col], axis=1).astype(np.float32)
    return bayes, stats


def part_frontier(args) -> None:
    pols = {"crown_s3": (CrownPolicy(RUNS["crown_s3"]), "bayes"),
            "a11_s2": (StatsIndex("a11_s2"), "stats")}
    mc_rng = np.random.default_rng(31)
    out: dict = {}
    for ttg in TTG_SET:
        bayes, stats = _frontier_beliefs(ttg)
        pm, psd = bayes[:, :K].astype(float), bayes[:, K:2 * K].astype(float)
        laws = {name: pol.probs(bayes if kind == "bayes" else stats)
                for name, (pol, kind) in pols.items()}
        laws["thompson"] = VecThompson.probs_mc(pm, psd, 16384, mc_rng)
        for name, p in laws.items():
            per_arm = p[:, 1:].mean(axis=1)          # equivariant: any unpulled
            below = per_arm < 1.0 / ttg
            x_star = float(X_GRID[below.argmax()]) if below.any() else None
            out.setdefault(name, {})[str(ttg)] = {
                "x": X_GRID.tolist(),
                "pi_unpulled": np.round(per_arm, 8).tolist(),
                "x_star": x_star,
            }
    path = OUTDIR / "coverage_frontier.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({name: {ttg: v["x_star"] for ttg, v in d.items()}
                      for name, d in out.items()}, indent=1), flush=True)
    print(f"wrote {path}", flush=True)


# ---------------------------------------------------------------------------
# part: massdecay (P2)
# ---------------------------------------------------------------------------

def part_massdecay(args) -> None:
    n = args.n_seeds
    seeds = PROTO_SEEDS[:n]
    idx = np.arange(n)
    out = {}
    for name in ("crown_s3", "a11_s2", "thompson"):
        sim = VecSim(seeds)
        if name == "thompson":
            pol, kind = VecThompson(rng_seed=7), "thompson"
            mc_rng = np.random.default_rng(211)
        elif name in RUNS:
            pol, kind = CrownPolicy(RUNS[name]), "bayes"
        else:
            pol, kind = StatsIndex(name), "stats"
        rng = np.random.default_rng(101)     # same stream as part coverage
        P = np.empty((n, T, K), dtype=np.float32)
        mass_curve = np.zeros(T)
        tic = time.time()
        for t in range(T):
            if kind == "thompson":
                pm = sim.payouts / (1.0 + sim.pulls)
                psd = np.sqrt(1.0 / (1.0 + sim.pulls))
                p = VecThompson.probs_mc(pm, psd, 256, mc_rng)
                a = pol.act(sim)
            else:
                obs = sim.obs() if kind == "bayes" else _stats_obs(sim)
                p = pol.probs(obs)
                a = _sample(p, rng)
            P[:, t] = p
            mass_curve[t] = np.where(sim.pulls == 0, p, 0.0).sum(axis=1).mean()
            sim.step(a)
        ep, arm = np.where(sim.pulls == 0)           # arms untouched at T
        traj = P[ep, :, arm]                         # (n_pairs, T)
        r0 = traj.sum(axis=1)                        # expected total touches
        suffix = traj[:, ::-1].cumsum(axis=1)[:, ::-1]
        t_half = (suffix < 0.5).argmax(axis=1)       # point of no return
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        out[name] = {
            "n_seeds": n,
            "n_untouched_pairs": int(len(ep)),
            "mass_unpulled_curve": np.round(mass_curve, 6).tolist(),
            "untouched_R0_quantiles": dict(zip(
                map(str, qs), np.round(np.quantile(r0, qs), 4).tolist())),
            "untouched_t_half_quantiles": dict(zip(
                map(str, qs), np.quantile(t_half, qs).tolist())),
            "untouched_mean_pi_curve": np.round(
                traj.mean(axis=0), 8).tolist(),
            "sweep_seconds": round(time.time() - tic, 1),
        }
        print(json.dumps({k: v for k, v in out[name].items()
                          if "curve" not in k} | {"policy": name}),
              flush=True)
    if n == CKPT_BLOCK:
        path = OUTDIR / "coverage_massdecay.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"wrote {path}", flush=True)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="coverage",
                    choices=["coverage", "floor", "ckpt", "frontier",
                             "massdecay"])
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--policies", type=str, default=None,
                    help="coverage: comma list; default = full roster")
    ap.add_argument("--out-tag", type=str, default=None,
                    help="coverage: record a --policies subset as "
                         "coverage_{tag}.json (the full-roster census is never "
                         "rewritten by a subset run)")
    ap.add_argument("--variants", type=str, default=None,
                    help="floor: comma list; default = FLOOR_VARIANTS")
    ap.add_argument("--run", type=str, default="crown_s3",
                    choices=list(CKPT_SUBJECTS),
                    help="ckpt: which subject")
    ap.add_argument("--ckpts", type=str, default=None,
                    help="ckpt: comma list of millions, e.g. 1,2,3")
    args = ap.parse_args()
    if args.n_seeds is None:
        args.n_seeds = (FULL_BLOCK if args.part in ("coverage", "floor")
                        else CKPT_BLOCK)
    {"coverage": part_coverage, "floor": part_floor, "ckpt": part_ckpt,
     "frontier": part_frontier, "massdecay": part_massdecay}[args.part](args)


if __name__ == "__main__":
    main()

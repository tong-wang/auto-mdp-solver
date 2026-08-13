"""Is the §8.6 selection protocol picking the right checkpoint? (#E33)

The in-training selection callback ranks checkpoints on a 256-seed CRN block
(`mab_ppo_train._selection_mean`, a scalar loop — 256 is what that loop
affords, not a statistical choice). This probe asks what that cost, by scoring
EVERY saved checkpoint on the reporting block with the batched simulator:

  current pick   the checkpoint the 256-seed callback chose (read back from
                 selection_log.tsv, whose argmax is what shipped as *_ppo.zip)
  oracle         the best of the 20 saved checkpoints on the reporting block
  terminal       the 20M checkpoint (what *_ppo_final.zip holds)

`oracle - current` is the performance the 256-seed selection left on the table;
`oracle - terminal` is what selection buys over shipping the last checkpoint.
Both are PAIRED (every checkpoint sees the same CRN block), so the differences
are far tighter than the individual means.

Caveat, stated rather than hidden: the oracle is an argmax over 20 evaluated on
the same block it is scored on, so it is an optimistic upper bound — the right
reading of it is "what was available to a perfect selector", not "what a real
selector would achieve". The current pick carries no such bias here: it was
chosen on a disjoint block (base 1e6), so its score on the reporting block is
honest.

Parts:
  curve    score every checkpoint of every subject on the reporting block
  report   current vs oracle vs terminal, per subject

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_selection_probe.py --part curve
    OMP_NUM_THREADS=1 python mab_selection_probe.py --part report
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mab_a5_probe import K, T, PROTO_SEEDS, SCENARIO, VecSim
from mab_anneal_probe import _sample
from mab_coverage_probe import CkptPolicy
from mab_interpret import OUTDIR

RESULTS = Path(__file__).resolve().parent / "results" / SCENARIO

# the six finished 20M runs, all obs=bayes / norm_obs=False / K=10 / T=1000
_A8A = ("PPO_obsbayes_L1_policyindex_lr1.58e-05_lrf1.58e-06_nsteps2048_bs32_"
        "ep20_gae0.988_ent3.44e-05_vf0.381_normobsFalse_seed{s}_20260801_230542")
_ANCHOR = ("PPO_obsbayes_L1_policyindex_lr1.58e-05_lrf1.58e-06_nsteps2048_bs32_"
           "ep20_gae0.988_ent3.44e-05_vf0.381_anchor{b}_normobsFalse_seed1_"
           "20260806_1123*")


def _subjects() -> dict[str, Path]:
    """Resolve the six finished runs. Missing dirs are tolerated at import —
    `results/` is gitignored, so a fresh clone has none — and only reported
    when a part actually asks for that subject."""
    out: dict[str, Path] = {}
    for s in (1, 2, 3):
        d = RESULTS / _A8A.format(s=s)
        if d.is_dir():
            out[f"crown_s{s}"] = d
    for b in ("0.003", "0.01", "0.03"):
        hits = sorted(d for d in RESULTS.glob(_ANCHOR.format(b=b)) if d.is_dir())
        if len(hits) == 1:
            out[f"anchor{b}"] = hits[0]
        elif hits:
            raise AssertionError(f"anchor {b}: {len(hits)} run dirs match: {hits}")
    return out


SUBJECTS = _subjects()
CKPTS = tuple(range(1, 21))          # 1M .. 20M, the saved cadence
MIN_RECORD_SEEDS = 2048              # below this a run is a smoke test, not evidence


def selection_pick(run_dir: Path) -> tuple[int, float]:
    """The checkpoint the 256-seed callback chose: argmax of selection_log."""
    best_m, best_v = None, -np.inf
    for line in (run_dir / "selection_log.tsv").read_text().splitlines():
        if line.startswith("#") or line.startswith("timesteps"):
            continue
        parts = line.split("\t")
        steps, mean = int(parts[0]), float(parts[1])
        if mean > best_v:
            best_v, best_m = mean, steps // 10 ** 6
    return best_m, best_v


def score_ckpt(run_dir: Path, m: int, seeds: np.ndarray,
               rng_seed: int = 101) -> dict:
    """One checkpoint on one CRN block, sampled (the campaign's eval mode)."""
    pol = CkptPolicy(run_dir, m)
    sim = VecSim(seeds)
    idx = np.arange(len(seeds))
    rng = np.random.default_rng(rng_seed)
    pseudo = np.zeros(len(seeds))
    tic = time.time()
    for _ in range(T):
        a = _sample(pol.probs(sim.obs()), rng)
        pseudo += sim.opt - sim.means[idx, a]
        sim.step(a)
    return {"ckpt_M": m, "n_seeds": int(len(seeds)),
            "reward_mean": float(sim.total.mean()),
            "reward_se": float(sim.total.std(ddof=1) / np.sqrt(len(seeds))),
            "pseudo_mean": float(pseudo.mean()),
            "_reward": sim.total, "seconds": round(time.time() - tic, 1)}


def part_curve(n_seeds: int, only: str | None) -> None:
    seeds = PROTO_SEEDS[:n_seeds]
    targets = [only] if only else list(SUBJECTS)
    for name in targets:
        run_dir = SUBJECTS[name]
        rows, per_seed = [], {}
        for m in CKPTS:
            r = score_ckpt(run_dir, m, seeds)
            per_seed[str(m)] = r.pop("_reward")
            rows.append(r)
            print(f"  {name:12s} ckpt {m:2d}M  reward {r['reward_mean']:8.2f} "
                  f"+/-{r['reward_se']:.2f}  pseudo {r['pseudo_mean']:7.2f} "
                  f"[{r['seconds']}s]", flush=True)
        # only a full-size block writes: a low-seed smoke run must never
        # clobber a real curve that --part report would then read as evidence
        if n_seeds < MIN_RECORD_SEEDS:
            print(f"  (n_seeds={n_seeds} < {MIN_RECORD_SEEDS}: not recorded)",
                  flush=True)
            continue
        OUTDIR.mkdir(parents=True, exist_ok=True)
        (OUTDIR / f"selection_curve_{name}.json").write_text(json.dumps(
            {"subject": name, "run": run_dir.name, "n_seeds": int(n_seeds),
             "rows": rows}, indent=2))
        np.savez(OUTDIR / f"selection_curve_{name}.npz", **per_seed)
        print(f"wrote selection_curve_{name}.json", flush=True)


def part_report() -> None:
    print(f"\n{'subject':12s} {'pick':>5s} {'pick_rw':>9s} {'oracle':>6s} "
          f"{'orc_rw':>9s} {'left':>8s} {'term_rw':>9s} {'orc-term':>9s}")
    left_all, vs_term = [], []
    for name in SUBJECTS:
        f = OUTDIR / f"selection_curve_{name}.json"
        if not f.exists():
            print(f"{name:12s} (no curve yet)")
            continue
        d = json.loads(f.read_text())
        by_m = {r["ckpt_M"]: r for r in d["rows"]}
        z = np.load(OUTDIR / f"selection_curve_{name}.npz")
        pick_m, _ = selection_pick(SUBJECTS[name])
        orc_m = max(by_m, key=lambda m: by_m[m]["reward_mean"])
        term_m = max(by_m)
        # paired on the shared CRN block -> the tight statistic
        left = float((z[str(orc_m)] - z[str(pick_m)]).mean())
        ot = float((z[str(orc_m)] - z[str(term_m)]).mean())
        left_all.append(left)
        vs_term.append(ot)
        print(f"{name:12s} {pick_m:4d}M {by_m[pick_m]['reward_mean']:9.2f} "
              f"{orc_m:5d}M {by_m[orc_m]['reward_mean']:9.2f} {left:+8.2f} "
              f"{by_m[term_m]['reward_mean']:9.2f} {ot:+9.2f}")
    if left_all:
        print(f"\nmean left on the table by the 256-seed pick: "
              f"{np.mean(left_all):+.2f}  (max {np.max(left_all):+.2f})")
        print(f"mean oracle advantage over the terminal checkpoint: "
              f"{np.mean(vs_term):+.2f}")
        print("\nNote: the oracle is an argmax over 20 scored on the block it is "
              "\nranked on, so it is an optimistic ceiling, not an achievable "
              "target.\nThe 'pick' column is unbiased (chosen on a disjoint "
              "block, base 1e6).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", required=True, choices=["curve", "report"])
    ap.add_argument("--n-seeds", type=int, default=2048)
    ap.add_argument("--subject", default=None, choices=[None, *SUBJECTS])
    args = ap.parse_args()
    if args.part == "curve":
        part_curve(args.n_seeds, args.subject)
    else:
        part_report()


if __name__ == "__main__":
    main()

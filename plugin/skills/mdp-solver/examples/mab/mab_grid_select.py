"""#E36: re-select a generalist's checkpoint on the GRID, not on one cell.

The #E36 runs shipped the wrong artifact. Their selection callback built its
env from `SCENARIOS[args.scenario_name]` — the unused `-s` default,
`gauss_K10_T1000` — so checkpoints were ranked by performance at ONE cell while
the run optimised forty, and `{grid}_ppo.zip` is "best at T=1000" rather than
"best across the grid". (Fixed at source in `mab_ppo_train.py`; this module
recovers the runs that were launched before the fix.)

All 190 checkpoints per run survive, so the measurement is recoverable.

Method, following #E33's design:

  * **Score on the grid, through the grid's own sampler**, so each episode
    draws its horizon exactly as training did.
  * **CRN across checkpoints**: one fixed seed list, and `MabEnv.reset(seed=s)`
    resolves the sampler at `s`, so every checkpoint sees the SAME episodes —
    same horizons, same arm means. Comparisons are paired, which is what makes
    a modest episode count sufficient.
  * **A block disjoint from both** the reporting protocol (0..8191) and the
    original selection block (base 1e6), so the re-selection is not made on
    seeds either earlier stage already used.
  * **Mean reward is the selection statistic.** Episodes differ in length, so
    the level is not comparable to any fixed-T number — but every checkpoint
    faces the identical mixture, so the ORDERING is valid. That is all a
    selector needs. Per-cell reporting stays with `mab_ppo_eval.py`, which
    enumerates registered cells at protocol depth (spec §9.6).

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_grid_select.py --run <run_dir> [--every 10]
    OMP_NUM_THREADS=1 python mab_grid_select.py --report
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
SELECT_BASE = 2_000_000          # disjoint from protocol (0..8191) and 1e6
SELECT_SEEDS = 128


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in sorted((run_dir / "checkpoints").glob("ppo_mab_*_steps.zip")):
        m = re.search(r"ppo_mab_(\d+)_steps\.zip", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def score(model_path: Path, grid_name: str, obs_mode: str,
          seeds: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean reward over a fixed CRN block of grid episodes."""
    from stable_baselines3 import PPO
    from mab_gym import MabEnv
    from mab_grids import GRIDS

    model = PPO.load(str(model_path), device="cpu")
    model.policy.set_training_mode(False)
    env = MabEnv(scenario=GRIDS[grid_name].as_sampler(),
                 observation_mode=obs_mode)
    per = np.zeros(len(seeds))
    for i, s in enumerate(seeds):
        obs, _ = env.reset(seed=int(s))
        done, total = False, 0.0
        while not done:
            a, _ = model.predict(obs[np.newaxis], deterministic=False)
            obs, r, done, _, _ = env.step(int(a[0]))
            total += float(r)
        per[i] = total
    return float(per.mean()), per


def shipped_steps(run_dir: Path) -> int | None:
    """The checkpoint the run actually shipped, per its own selection log."""
    sel = run_dir / "selection_log.tsv"
    if not sel.exists():
        return None
    best_v, best_s = -1e18, None
    for line in sel.read_text().splitlines():
        if line.startswith("#") or line.startswith("timesteps"):
            continue
        a = line.split("\t")
        if float(a[1]) > best_v:
            best_v, best_s = float(a[1]), int(a[0])
    return best_s


def part_run(run_dir: Path, every: int, n_seeds: int) -> dict:
    args_file = next(run_dir.glob("*_ppo_args.txt"))
    txt = args_file.read_text()
    grid_name = re.search(r"grid_name[:=]\s*(\S+)", txt).group(1).strip("',\"")
    obs_mode = re.search(r"observation_mode[:=]\s*(\S+)", txt).group(1).strip("',\"")

    # The incumbent must ALWAYS be scored, whatever `--every` samples: this tool
    # exists to pair the grid winner against what the run shipped, and a stride
    # that misses the shipped step yields a nan gain -- the comparison silently
    # not happening rather than failing. (It did: 76M/133M/19M under --every 10.)
    want = {m for m, _ in checkpoints(run_dir) if (m // 1_000_000) % every == 0}
    ship = shipped_steps(run_dir)
    if ship is not None:
        want.add(ship)
    cks = [(m, p) for m, p in checkpoints(run_dir) if m in want]

    # Scoring is ~2 min per checkpoint, so reuse anything a previous pass
    # already measured on the SAME block (seeds are a deterministic function of
    # SELECT_BASE and n_seeds, so a cached column is the identical measurement).
    cache, npz = {}, run_dir / "grid_selection.npz"
    prev = run_dir / "grid_selection.json"
    if npz.exists() and prev.exists():
        old = json.loads(prev.read_text())
        if old.get("select_base") == SELECT_BASE and old.get("n_seeds") == n_seeds:
            z = np.load(npz)
            cache = {k: z[k] for k in z.files}

    seeds = np.arange(SELECT_BASE, SELECT_BASE + n_seeds)
    rows, per_seed = [], {}
    todo = sum(1 for m, _ in cks if str(m) not in cache)
    print(f"  {run_dir.name[:60]}...  grid={grid_name} obs={obs_mode} "
          f"{len(cks)} ckpts x {n_seeds} seeds ({todo} to score, "
          f"{len(cks)-todo} cached; shipped={ship})", flush=True)
    for steps, path in cks:
        if str(steps) in cache:
            per = cache[str(steps)]
            mean, secs = float(per.mean()), 0.0
        else:
            t0 = time.time()
            mean, per = score(path, grid_name, obs_mode, seeds)
            secs = round(time.time() - t0, 1)
        per_seed[str(steps)] = per
        rows.append({"steps": steps, "grid_mean": mean, "seconds": secs,
                     "shipped": steps == ship})
        print(f"    {steps/1e6:6.0f}M  grid_mean {mean:9.2f}  "
              f"[{secs}s]{'  <- shipped' if steps == ship else ''}", flush=True)

    best = max(rows, key=lambda r: r["grid_mean"])
    out = {"run": run_dir.name, "grid": grid_name,
           "select_base": SELECT_BASE, "n_seeds": n_seeds,
           "best_steps": best["steps"], "best_grid_mean": best["grid_mean"],
           "rows": rows}
    (run_dir / "grid_selection.json").write_text(json.dumps(out, indent=2))
    np.savez(run_dir / "grid_selection.npz", **per_seed)
    print(f"  -> best {best['steps']/1e6:.0f}M at {best['grid_mean']:.2f}",
          flush=True)
    return out


def part_report(grid_name: str) -> None:
    print(f"\n{'run':14s}{'grid-selected':>16s}{'ships now':>12s}"
          f"{'paired gain':>13s}")
    missing = []
    for f in sorted(RESULTS.glob(f"{grid_name}/*/grid_selection.json")):
        d = json.loads(f.read_text())
        z = np.load(f.parent / "grid_selection.npz")
        # The shipped artifact was chosen on ONE cell (the selection bug this
        # module exists to undo); pair it against the grid winner on the same
        # CRN block.
        shipped = shipped_steps(f.parent)
        tag = f.parent.name.split("_seed")[-1][:1]
        g = str(d["best_steps"])
        s = str(shipped) if shipped is not None and str(shipped) in z else None
        if s is None:
            missing.append((tag, shipped))
            gain = float("nan")
            sd = float("nan")
        else:
            diff = z[g] - z[s]
            gain = float(diff.mean())
            sd = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        print(f"seed {tag:9s}{d['best_steps']/1e6:11.0f}M "
              f"{d['best_grid_mean']:8.1f}"
              f"{(shipped or 0)/1e6:9.0f}M {gain:+9.2f} +/- {sd:5.2f}")
    print("\nPaired on one CRN block of grid episodes; the level mixes episode "
          "lengths\nand is not comparable to any fixed-T number — only the "
          "ordering is used.\n+/- is the paired standard error over the block.")
    if missing:
        print("\n*** INCOMPLETE: shipped checkpoint not in the scored set for "
              f"{len(missing)} run(s): {missing}.\n    Re-run without --report "
              "to score it (cached checkpoints are reused).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--grid", type=str, default="gauss_K10_Tlog")
    ap.add_argument("--every", type=int, default=10,
                    help="score every Nth checkpoint (in millions of steps)")
    ap.add_argument("--n-seeds", type=int, default=SELECT_SEEDS)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        part_report(a.grid)
        return
    runs = ([Path(a.run)] if a.run
            else sorted((RESULTS / a.grid).glob("PPO_*/")))
    for r in runs:
        part_run(r, a.every, a.n_seeds)


if __name__ == "__main__":
    main()

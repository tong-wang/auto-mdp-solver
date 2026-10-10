"""Post-hoc checkpoint screen for owmr — the selection layer of spec 9.7.

Training runs to its budget and saves ~20 checkpoints; the terminal checkpoint
is never the deliverable (8.6). This ranks the checkpoints on the selection
block (disjoint from the protocol block), each under its own saved VecNormalize
statistics, and prints the top-k to confirm on the protocol block.

    python owmr_select.py results/base/PPO_...            # screen
    python owmr_ppo_eval.py --model-path <winner> ...      # confirm, then quote

Screen scores rank; they are never quoted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from owmr_eval_common import SELECT_SEED_OFFSET, eval_seed_block, rollout_seeds
from owmr_gym import OwmrEnv
from owmr_scenarios import SCENARIOS

METRIC = "cost_total"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Screen a run's checkpoints on the selection block (9.7).")
    p.add_argument("run_dir", type=str)
    p.add_argument("-s", "--scenario_name", type=str, default=None,
                   help="inferred from the run directory's parent when omitted")
    p.add_argument("-o", "--observation_mode", type=str, default="raw", choices=list(OwmrEnv.OBS_MODES))
    p.add_argument("-a", "--action_mode", type=str, default="order_frac", choices=list(OwmrEnv.ACTION_MODES))
    p.add_argument("--n-seeds", type=int, default=2048)
    p.add_argument("--first-seed", type=int, default=SELECT_SEED_OFFSET)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--outfile", type=str, default=None, help="defaults to <run_dir>/screen.tsv")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def _step_of(path: Path) -> int:
    m = re.search(r"_(\d+)_steps", path.name)
    return int(m.group(1)) if m else -1


def find_checkpoints(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted((p for p in ckpt_dir.glob("*.zip") if "vecnormalize" not in p.name), key=_step_of)
    if not ckpts:
        raise SystemExit(f"no checkpoint .zip files in {ckpt_dir}")
    return ckpts


def resolve_vecnorm(ckpt: Path, run_dir: Path) -> Path | None:
    sibling = ckpt.parent / re.sub(r"_(\d+)_steps\.zip$", r"_vecnormalize_\1_steps.pkl", ckpt.name)
    if sibling.exists():
        return sibling
    terminal = run_dir / "vecnormalize.pkl"
    return terminal if terminal.exists() else None


def widen_to_ties(ranked: list[dict], top_k: int) -> list[dict]:
    """Top-k, widened when leaders sit within one screen-SE of the best (minimize)."""
    cut = ranked[0]["mean"] + ranked[0]["se"]
    keep = [r for r in ranked if r["mean"] <= cut]
    return ranked[:top_k] if len(keep) <= top_k else keep


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    scenario_name = args.scenario_name or run_dir.parent.name
    if scenario_name not in SCENARIOS:
        raise SystemExit(f"cannot infer the scenario from {run_dir}; pass -s")
    if args.first_seed == 0:
        raise SystemExit("refusing to screen on the protocol block (--first-seed 0): "
                         "the block that ranks must not be the block that reports (9.7).")
    scenario = SCENARIOS[scenario_name]
    ckpts = find_checkpoints(run_dir)
    outfile = Path(args.outfile) if args.outfile else run_dir / "screen.tsv"
    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)
    print(f"run         {run_dir}\nscenario    {scenario_name}   obs={args.observation_mode}   act={args.action_mode}")
    print(f"seeds       {args.first_seed}..{args.first_seed + args.n_seeds - 1}  (selection block)")
    print(f"checkpoints {len(ckpts)}\nmetric      {METRIC} (minimize) — ranks only, never quoted\n")
    header = f"checkpoint\tsteps\t{METRIC}_mean\t{METRIC}_se\tvecnorm"
    print(header)
    ranked: list[dict] = []
    with open(outfile, "w") as f:
        f.write(header + "\n")
        for ckpt in ckpts:
            vecnorm = resolve_vecnorm(ckpt, run_dir)
            model = PPO.load(str(ckpt), device="cpu")

            def act_fn(obs, raw_envs, _m=model):
                action, _ = _m.predict(obs, deterministic=True)
                return action

            per_seed = rollout_seeds(scenario=scenario, observation_mode=args.observation_mode,
                                     action_mode=args.action_mode, seeds=seeds, act_fn=act_fn,
                                     batch=min(args.batch_envs, len(seeds)), vecnorm_path=vecnorm,
                                     progress_every=0)
            vals = per_seed[METRIC]
            entry = {"path": ckpt, "steps": _step_of(ckpt), "mean": float(vals.mean()),
                     "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
                     "vecnorm": ("per-ckpt" if vecnorm and "vecnormalize_" in vecnorm.name
                                 else "terminal" if vecnorm else "none")}
            ranked.append(entry)
            row = f"{ckpt.name}\t{entry['steps']}\t{entry['mean']:.6f}\t{entry['se']:.6f}\t{entry['vecnorm']}"
            print(row, flush=True)
            f.write(row + "\n"); f.flush()
    stale = [r for r in ranked if r["vecnorm"] != "per-ckpt"]
    if stale:
        print(f"\nWARNING: {len(stale)} checkpoint(s) scored without a per-checkpoint normalizer.")
    ranked.sort(key=lambda r: r["mean"])
    winners = widen_to_ties(ranked, args.top_k)
    print(f"\nranking saved -> {outfile}\n\ntop-{len(winners)} for confirmation on the protocol block:")
    for i, r in enumerate(winners, 1):
        print(f"  {i}. {r['path'].name}  (screen {r['mean']:.4f} ± {r['se']:.4f}, step {r['steps']})")
    print("\nconfirm each on the protocol block, then quote the winner:")
    for r in winners:
        vecnorm = resolve_vecnorm(r["path"], run_dir)
        vn = f" --vecnorm-path {vecnorm}" if vecnorm else ""
        print(f"  python owmr_ppo_eval.py --model-path {r['path']}{vn} -s {scenario_name} "
              f"-o {args.observation_mode} -a {args.action_mode} --first-seed 0 "
              f"--outfile {run_dir}/confirm_{r['path'].stem}.tsv")


if __name__ == "__main__":
    main()

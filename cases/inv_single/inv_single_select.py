"""Post-hoc checkpoint screen — the middle layer of the three-layer evaluation (spec 9.7).

Training runs to its budget and saves ~20 checkpoints; **the terminal checkpoint
is never the deliverable** (8.6). This script ranks those checkpoints on a CRN
block disjoint from the protocol block and emits the top-k for confirmation.

    python inv_single_ppo_train.py -s simple        # produces checkpoints/
    python inv_single_select.py results/simple/PPO_...   # <- here: screen
    python inv_single_ppo_eval.py --model-path <winner>  # confirm, then quote

Three rules this file exists to keep:

1. **Screen scores are never quoted.** They rank; the protocol block confirms.
   The blocks are disjoint (`SELECT_SEED_OFFSET` vs the protocol block's 0), so
   the number that ranks a checkpoint is never the number that reports it.
2. **Each checkpoint is scored under its own normalizer.** `CheckpointCallback`
   saves the VecNormalize statistics beside every checkpoint; scoring an early
   checkpoint under the run's final statistics would rank it on observations it
   never saw.
3. **No `EvalCallback`, no live selection env, no `sync_envs_normalization`.**
   The screen reads saved checkpoints after training has finished.

This domain MINIMIZES cost, so the ranking is ascending and the tie band opens
upward from the best.

Example usage:
    python inv_single_select.py results/simple/PPO_20260824_101500_default
    python inv_single_select.py <run_dir> --n-seeds 2048 --top-k 3
"""

import argparse
import re
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from inv_single_eval_common import (
    SELECT_SEED_OFFSET,
    eval_seed_block,
    rollout_seeds,
)
from inv_single_grids import GRIDS
from inv_single_scenarios import SCENARIOS

# 9.7: the screen layer's block. Cheaper than the protocol block because it
# ranks rather than reports — but still large enough that the ranking is not
# draw noise, which the vectorized evaluator makes affordable.
DEFAULT_SCREEN_SEEDS = 2048

METRIC = "cost_total"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Screen a training run's checkpoints (9.7 screen layer).")
    p.add_argument("run_dir", type=str,
                   help="Training run directory (the one holding checkpoints/)")
    p.add_argument("-s", "--scenario_name", type=str, default=None,
                   help="Scenario the run trained on; inferred from the run "
                        "directory's parent when omitted")
    p.add_argument("-o", "--observation_mode", type=str, default="vec",
                   choices=["vec", "vec_ip", "vec_ctx",
                            "vec_ctx_slt", "vec_ip_ctx_slt"])
    p.add_argument("-a", "--action_mode", type=str, default="discrete",
                   choices=["continuous", "discrete", "hurdle"])
    p.add_argument("--n-seeds", type=int, default=DEFAULT_SCREEN_SEEDS,
                   help=f"Screen-block size (default {DEFAULT_SCREEN_SEEDS})")
    p.add_argument("--first-seed", type=int, default=SELECT_SEED_OFFSET,
                   help="First seed of the screen block; must stay disjoint "
                        "from the protocol block (0…n-1)")
    p.add_argument("--top-k", type=int, default=3,
                   help="Checkpoints passed to confirmation (widened "
                        "automatically when leaders sit within one screen-SE)")
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV of the ranking (defaults to <run_dir>/screen.tsv)")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step_of(path: Path) -> int:
    """Sort key: the step count CheckpointCallback encodes in the filename."""
    m = re.search(r"_(\d+)_steps", path.name)
    return int(m.group(1)) if m else -1


def find_checkpoints(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"no checkpoints/ under {run_dir}")
    ckpts = sorted(
        (p for p in ckpt_dir.glob("*.zip") if "vecnormalize" not in p.name),
        key=_step_of,
    )
    if not ckpts:
        raise SystemExit(f"no checkpoint .zip files in {ckpt_dir}")
    return ckpts


def resolve_vecnorm(ckpt: Path, run_dir: Path) -> Path | None:
    """The normalizer this checkpoint was trained under.

    `CheckpointCallback(save_vecnormalize=True)` writes
    `{prefix}_vecnormalize_{steps}_steps.pkl` beside `{prefix}_{steps}_steps.zip`,
    keyed on the same step count. Falls back to the run's terminal statistics
    for runs saved before that was switched on — reported, because it makes the
    early checkpoints' scores optimistic in an unknown direction.
    """
    step = _step_of(ckpt)
    sibling = ckpt.parent / re.sub(r"_(\d+)_steps\.zip$",
                                   rf"_vecnormalize_\1_steps.pkl", ckpt.name)
    if sibling.exists():
        return sibling
    terminal = run_dir / "vecnormalize.pkl"
    return terminal if terminal.exists() else None


def infer_scenario_name(run_dir: Path, explicit: str | None) -> str:
    """Runs live at results/{scenario_name}/{run_name}/ (8.4).

    A generalist's runs land under results/{grid_name}/, so a GRIDS key is a
    legal target here too — the screen then ranks checkpoints on the grid's own
    sampler, i.e. on the distribution the run was trained against.
    """
    if explicit:
        return explicit
    name = run_dir.resolve().parent.name
    if name not in SCENARIOS and name not in GRIDS:
        raise SystemExit(
            f"cannot infer the scenario from {run_dir} (got {name!r}); pass -s")
    return name


def widen_to_ties(ranked: list[dict], top_k: int) -> list[dict]:
    """Top-k, widened to include everything within one screen-SE of the best.

    8.6 makes k adjustable for exactly this reason: when the leaders sit inside
    one screen-SE the ranking cannot separate them, so confirmation — not the
    screen — should decide. Minimize sense, so the band opens upward.
    """
    best = ranked[0]
    cut = best["mean"] + best["se"]
    keep = [r for r in ranked if r["mean"] <= cut]
    return ranked[:top_k] if len(keep) <= top_k else keep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    scenario_name = infer_scenario_name(run_dir, args.scenario_name)
    # A grid screens on its SAMPLER, not on enumerated cells: the screen ranks
    # checkpoints of one generalist against the distribution it was trained on,
    # and per-cell rows belong to confirmation (9.6), where the block is the
    # reporting one. Ranking per cell here would also multiply the screen cost
    # by the cell count for a number that is never quoted.
    scenario = (GRIDS[scenario_name].as_sampler()
                if scenario_name in GRIDS else SCENARIOS[scenario_name])
    ckpts = find_checkpoints(run_dir)
    outfile = Path(args.outfile) if args.outfile else run_dir / "screen.tsv"
    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)

    if args.first_seed == 0:
        raise SystemExit(
            "refusing to screen on the protocol block (--first-seed 0): the "
            "block that ranks must not be the block that reports (9.7).")

    print(f"run         {run_dir}")
    print(f"scenario    {scenario_name}   obs={args.observation_mode}")
    print(f"seeds       {args.first_seed}…{args.first_seed + args.n_seeds - 1}"
          f"  (screen block)")
    print(f"checkpoints {len(ckpts)}")
    print(f"metric      {METRIC} (minimize)")
    print("NOTE: screen scores rank only — they are never quoted (9.7).\n")

    ranked: list[dict] = []
    header = f"checkpoint\tsteps\t{METRIC}_mean\t{METRIC}_se\tvecnorm"
    print(header)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()

        for ckpt in ckpts:
            vecnorm = resolve_vecnorm(ckpt, run_dir)
            model = PPO.load(str(ckpt), device="cpu")

            def act_fn(obs: np.ndarray, raw_envs) -> np.ndarray:
                action, _ = model.predict(obs, deterministic=True)
                return action

            per_seed = rollout_seeds(
                scenario=scenario,
                observation_mode=args.observation_mode,
                action_mode=args.action_mode,
                seeds=seeds,
                act_fn=act_fn,
                batch=min(args.batch_envs, len(seeds)),
                vecnorm_path=vecnorm,
                progress_every=0,
            )
            vals = per_seed[METRIC]
            entry = {
                "path": ckpt,
                "steps": _step_of(ckpt),
                "mean": float(vals.mean()),
                "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
                "vecnorm": ("per-ckpt" if vecnorm and "vecnormalize_" in vecnorm.name
                            else "terminal" if vecnorm else "none"),
            }
            ranked.append(entry)
            row = (f"{ckpt.name}\t{entry['steps']}\t"
                   f"{entry['mean']:.6f}\t{entry['se']:.6f}\t{entry['vecnorm']}")
            print(row, flush=True)
            f.write(row + "\n")
            f.flush()

    stale = [r for r in ranked if r["vecnorm"] != "per-ckpt"]
    if stale:
        print(f"\nWARNING: {len(stale)}/{len(ranked)} checkpoint(s) had no "
              f"per-checkpoint normalizer and were scored under the run's "
              f"terminal statistics — their ranking is not trustworthy.")

    ranked.sort(key=lambda r: r["mean"])          # minimize
    winners = widen_to_ties(ranked, args.top_k)

    print(f"\nranking saved → {outfile}")
    print(f"\ntop-{len(winners)} for confirmation on the protocol block:")
    for i, r in enumerate(winners, 1):
        print(f"  {i}. {r['path'].name}  (screen {r['mean']:.6f} "
              f"± {r['se']:.6f}, step {r['steps']})")
    if len(winners) > args.top_k:
        print(f"  (widened from {args.top_k}: leaders sit within one screen-SE)")

    print("\nconfirm each on the protocol block, then quote the winner:")
    for r in winners:
        vecnorm = resolve_vecnorm(r["path"], run_dir)
        vn = f" --vecnorm-path {vecnorm}" if vecnorm else ""
        # -a is as load-bearing as -o: an arm evaluated in an action mode it
        # did not train with is silently mis-scored, never an error. This line
        # omitted it, which cost batch 2 a full screen+eval round (#E4).
        # --outfile is as load-bearing as -a for a TOP-K confirmation: every
        # checkpoint of a run shares checkpoints/ as its model dir, and
        # write_record overwrites, so without a distinct name only the last
        # confirm to finish survives — silently, since each file still looks
        # complete. (Cost this campaign one full 30-eval confirmation round.)
        stem = Path(r["path"]).stem
        print(f"  python inv_single_ppo_eval.py --model-path {r['path']}"
              f"{vn} {'-g' if scenario_name in GRIDS else '-s'} {scenario_name} -o {args.observation_mode}"
              f" -a {args.action_mode} --first-seed 0"
              f" --outfile {run_dir}/confirm_{stem}.tsv")


if __name__ == "__main__":
    main()

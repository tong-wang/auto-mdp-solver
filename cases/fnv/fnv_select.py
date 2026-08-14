"""Post-hoc checkpoint screen — the middle layer of the three-layer evaluation (spec §9.7).

Training runs to its budget and saves ~20 checkpoints; **the terminal checkpoint
is never the deliverable** (§8.6). This script ranks those checkpoints on a CRN
block disjoint from the protocol block and emits the top-k for confirmation.

    python fnv_ppo_train.py -s FNV-aMMFE      # produces checkpoints/
    python fnv_select.py results/FNV-aMMFE/PPO_...     # <- here: screen
    python fnv_ppo_eval.py --model-path <winner>       # confirm, then quote

Three rules this file exists to keep:

1. **Screen scores are never quoted.** They rank; the protocol block confirms.
   The two blocks are disjoint (SCREEN_SEED_BASE vs PROTOCOL_SEED_BASE) so the
   number that ranks a checkpoint is never the number that reports it.
2. **A generalist is screened over its whole grid**, via the grid's sampler —
   so each screen episode draws its own cell. Selecting a generalist on one
   cell is a known failure mode, not a shortcut.
3. **No EvalCallback, no live selection env, no sync_envs_normalization.** The
   screen reads saved checkpoints after training has finished.

Example usage:
    python fnv_select.py results/simple/PPO_20260813_172426_default
    python fnv_select.py <run_dir> --n-seeds 2048 --top-k 3
"""

import argparse
import re
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from fnv_grids import GRIDS
from fnv_ppo_eval import (
    SCREEN_SEED_BASE,
    load_obs_rms,
    normalize_obs,
    resolve_vecnorm_path,
    rollout_seeds,
)
from fnv_scenarios import SCENARIOS

# §9.7: the screen layer's block. Cheaper than the protocol block because it
# ranks rather than reports — but still large enough that the ranking is not
# draw noise, which the vectorized evaluator makes affordable.
DEFAULT_SCREEN_SEEDS = 2048


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Screen a training run's checkpoints (§9.7 screen layer).")
    p.add_argument("run_dir", type=str,
                   help="Training run directory (the one holding checkpoints/)")
    p.add_argument("-s", "--scenario_name", type=str, default=None,
                   help="Scenario or grid the run trained on; inferred from the "
                        "run directory's parent when omitted")
    p.add_argument("-o", "--observation_mode", type=str, default="vec", choices=["vec"])
    p.add_argument("-a", "--action_mode",      type=str, default="box", choices=["box"])
    p.add_argument("-r", "--reward_mode",      type=str, default="profit",
                   choices=["profit", "regret"])
    p.add_argument("--n-seeds", type=int, default=DEFAULT_SCREEN_SEEDS,
                   help=f"Screen-block size (default {DEFAULT_SCREEN_SEEDS})")
    p.add_argument("--seed-base", type=int, default=SCREEN_SEED_BASE,
                   help="First seed of the screen block; must stay disjoint "
                        "from the protocol block")
    p.add_argument("--top-k", type=int, default=3,
                   help="Checkpoints passed to confirmation (widened "
                        "automatically when leaders sit within one screen-SE)")
    p.add_argument("--batch-size", type=int, default=512)
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


def infer_scenario_name(run_dir: Path, explicit: str | None) -> str:
    """Runs live at results/{scenario_name}/{run_name}/ (§8.4)."""
    if explicit:
        return explicit
    name = run_dir.resolve().parent.name
    if name not in SCENARIOS and name not in GRIDS:
        raise SystemExit(
            f"cannot infer the scenario from {run_dir} (got {name!r}); pass -s")
    return name


def screen_source(scenario_name: str):
    """The distribution a checkpoint is ranked on.

    A grid is screened through its **sampler**, so every cell is represented in
    the screen block — ranking a generalist on a single cell selects for that
    cell, not for the generality target the run was trained against (§5.6).
    """
    if scenario_name in SCENARIOS:
        return SCENARIOS[scenario_name]
    return GRIDS[scenario_name].as_sampler()


def widen_to_ties(ranked: list[dict], top_k: int) -> list[dict]:
    """Top-k, widened to include everything within one screen-SE of the best.

    §8.6 makes k adjustable for exactly this reason: when the leaders sit inside
    one screen-SE the ranking cannot separate them, so confirmation — not the
    screen — should decide.
    """
    best = ranked[0]
    cut  = best["profit_mean"] - best["profit_se"]
    keep = [r for r in ranked if r["profit_mean"] >= cut]
    return ranked[:top_k] if len(keep) <= top_k else keep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    scenario_name = infer_scenario_name(run_dir, args.scenario_name)
    source  = screen_source(scenario_name)
    ckpts   = find_checkpoints(run_dir)
    outfile = Path(args.outfile) if args.outfile else run_dir / "screen.tsv"
    seeds   = range(args.seed_base, args.seed_base + args.n_seeds)

    print(f"run       {run_dir}")
    print(f"scenario  {scenario_name}"
          + (f"  (grid: screened over all {len(GRIDS[scenario_name])} cells "
             f"via its sampler)" if scenario_name in GRIDS else ""))
    print(f"seeds     {args.seed_base}..{args.seed_base + args.n_seeds - 1}  (screen block)")
    print(f"checkpoints {len(ckpts)}")
    print("NOTE: screen scores rank only — they are never quoted (§9.7).\n")

    ranked: list[dict] = []
    header = "checkpoint\tsteps\tprofit_mean\tprofit_se"
    print(header)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()

        for ckpt in ckpts:
            vecnorm_path = resolve_vecnorm_path(ckpt, None)
            obs_rms, clip_obs = load_obs_rms(vecnorm_path)
            model = PPO.load(str(ckpt), device="cpu")

            def act_batch(obs: np.ndarray) -> np.ndarray:
                action, _ = model.predict(
                    normalize_obs(obs, obs_rms, clip_obs), deterministic=True)
                return action

            profits, _ = rollout_seeds(
                act_batch, source, seeds,
                observation_mode=args.observation_mode,
                action_mode=args.action_mode,
                reward_mode=args.reward_mode,
                batch_size=args.batch_size,
            )
            entry = {
                "path":        ckpt,
                "steps":       _step_of(ckpt),
                "profit_mean": float(profits.mean()),
                "profit_se":   float(profits.std(ddof=1) / np.sqrt(len(profits))),
            }
            ranked.append(entry)
            row = (f"{ckpt.name}\t{entry['steps']}\t"
                   f"{entry['profit_mean']:.6f}\t{entry['profit_se']:.6f}")
            print(row, flush=True)
            f.write(row + "\n")
            f.flush()

    ranked.sort(key=lambda r: -r["profit_mean"])
    winners = widen_to_ties(ranked, args.top_k)

    print(f"\nranking saved → {outfile}")
    print(f"\ntop-{len(winners)} for confirmation on the protocol block:")
    for i, r in enumerate(winners, 1):
        print(f"  {i}. {r['path'].name}  (screen {r['profit_mean']:.6f} "
              f"± {r['profit_se']:.6f}, step {r['steps']})")
    if len(winners) > args.top_k:
        print(f"  (widened from {args.top_k}: leaders sit within one screen-SE)")

    print("\nconfirm each, then ship the winner:")
    for r in winners:
        print(f"  python fnv_ppo_eval.py --model-path {r['path']} -s {scenario_name}")


if __name__ == "__main__":
    main()

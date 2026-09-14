"""Reporting-tier evaluation of a trained PPO model (spec §9).

Scores one model on one eval scenario over the CRN seed block `0…n_seeds-1`,
writes the §9.3 record plus a per-seed objective dump, and optionally reports
the **paired** delta against a benchmark's per-seed dump.

The eval scenario is independent of the training scenario (§9.6 train/eval
decoupling) and is encoded in the outfile name.

Usage:
    python inv_single_ppo_eval.py -s simple --model-path results/simple/PPO_.../best_model.zip
    python inv_single_ppo_eval.py -s simple --model-path .../best_model.zip \
        --paired-with results/simple/benchmark/benchmark_dp_eval_simple_perseed.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from inv_single_eval_common import (
    SELECT_SEED_OFFSET,
    eval_seed_block,
    paired_report,
    rollout_seeds,
    write_record,
)
from inv_single_grids import GRIDS
from inv_single_scenarios import SCENARIOS

OBS_MODES = ["vec", "vec_ip", "vec_ctx", "vec_ctx_slt",
             "vec_ip_ctx_slt"]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model on inv_single.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None,
                   help="Defaults to <model_dir>/vecnormalize.pkl when present.")
    # spec 9.6: a GRIDS key evaluates a GENERALIST by ENUMERATING its cells,
    # one leaderboard row per cell, every cell on the same seed block. The
    # aggregate is reported too, but it is a summary of the rows and never a
    # substitute for them: cells are separate leaderboards.
    p.add_argument("-g", "--grid_name", type=str, default=None,
                   help="evaluate per cell over this GRIDS entry; overrides -s")
    p.add_argument("-s", "--scenario_name", type=str, default="simple",
                   choices=list(SCENARIOS) + list(GRIDS))
    # matches the train script's default: an eval must use the mode its arm was
    # trained with, so defaults that disagree are a trap, not a preference
    p.add_argument("-o", "--observation_mode", type=str, default="vec", choices=OBS_MODES)
    p.add_argument("-a", "--action_mode", type=str, default="discrete",
                   choices=["continuous", "discrete", "hurdle"])
    p.add_argument("--n-seeds", type=int, default=65536)
    # 9.7 tier 1: the block's OFFSET, which is what makes the three evaluation
    # layers disjoint. protocol = 0 (this script's default, the only block a
    # number may be quoted from), screen = 1_000_000, a tuning trial's periodic
    # eval above that. The offsets are conventional; the disjointness is not.
    p.add_argument("--first-seed", type=int, default=0,
                   help="First seed of the eval block (default 0 = the "
                        "protocol block).")
    p.add_argument("--batch-envs", type=int, default=64,
                   help="Episodes rolled out concurrently to amortize the "
                        "policy forward pass. Does not affect results: each "
                        "slot is seeded explicitly.")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of argmax. Off by default; a "
                        "continuous ordering policy has no bandit-style "
                        "exploration value at eval time.")
    p.add_argument("--arm", type=str, default=None,
                   help="Label for the record's `arm` column; defaults to the run dir name.")
    p.add_argument("--outfile", type=str, default=None)
    p.add_argument("--paired-with", type=str, default=None,
                   help="A benchmark's *_perseed.npy; prints the paired delta.")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def _eval_grid(args, model, model_dir, vecnorm, arm, deterministic) -> None:
    """One row per cell (spec 9.6), all cells on the same seed block.

    Cells are separate leaderboards, so the per-cell rows are the deliverable.
    The aggregate printed at the end is the mean over cells of their mean cost —
    equal weight per CELL, which is what the training sampler draws and what a
    "how good is this generalist" question means. It is NOT the mean over
    episodes, which would weight cells by nothing meaningful, and it is never a
    substitute for reading the rows.
    """
    import numpy as np

    from inv_single_grids import cell_slug

    grid = GRIDS[args.grid_name]
    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)
    model_stem = Path(args.model_path).stem

    def act_fn(obs, raw_envs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    print(f"[eval] grid      : {args.grid_name}  ({len(grid.cells)} cells)"
          f"   obs={args.observation_mode}")
    block = ("protocol" if args.first_seed == 0
             else "screen" if args.first_seed == SELECT_SEED_OFFSET
             else "non-standard")
    print(f"[eval] seeds     : {args.first_seed}…"
          f"{args.first_seed + args.n_seeds - 1}   ({block} block)")
    if args.first_seed != 0:
        print("[eval] WARNING: not the protocol block — ranking only (9.7).")

    means, pooled = [], {}
    for cid, sc in grid.cells:
        per_seed = rollout_seeds(
            scenario=sc,
            observation_mode=args.observation_mode,
            action_mode=args.action_mode,
            seeds=seeds,
            act_fn=act_fn,
            batch=args.batch_envs,
            vecnorm_path=vecnorm,
            progress_every=0,
        )
        slug = cell_slug(cid)
        # the checkpoint stem is part of the NAME, not just the `arm` column:
        # confirming several checkpoints of one run puts them all in the same
        # checkpoints/ directory, and write_record overwrites, so without this
        # only the last one to finish survives -- silently, since every file
        # still looks complete. (The specialist path has the same hazard; it is
        # avoided there only by passing --outfile per checkpoint.)
        out = model_dir / f"ppo_eval_{args.grid_name}__{slug}__{model_stem}.tsv"
        write_record(out, f"{args.grid_name}__{slug}", arm, per_seed)
        m = float(np.mean(per_seed["cost_total"]))
        means.append(m)
        for k, v in per_seed.items():
            pooled.setdefault(k, []).append(np.asarray(v))
        print(f"  {slug:<14} {m:10.2f}")
    print(f"[eval] cells     : {len(means)}   mean over cells "
          f"{float(np.mean(means)):.2f}   worst cell {max(means):.2f}")

    # An AGGREGATE row wherever the caller named a file. mdp_tuning drives eval
    # with --outfile and then takes that TSV's column means as the trial score,
    # so a grid target with per-cell files only would leave it nothing to read.
    # Pooling the per-seed arrays weights every cell equally (each contributes
    # the same seed count), which is the same "mean over cells" printed above.
    # The dispersion columns pool within- and between-cell variance and are
    # therefore a summary, not a per-cell statistic — the per-cell files above
    # hold those.
    if args.outfile:
        agg = {k: np.concatenate(v) for k, v in pooled.items()}
        write_record(Path(args.outfile), args.grid_name,
                     f"{arm}::mean-over-{len(means)}-cells", agg)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    model_dir = model_path.parent

    vecnorm = (
        Path(args.vecnorm_path)
        if args.vecnorm_path
        else model_dir / "vecnormalize.pkl"
    )
    outfile = Path(
        args.outfile
        if args.outfile
        else model_dir / f"ppo_eval_{args.scenario_name}.tsv"
    )
    arm = args.arm or model_dir.name

    model = PPO.load(str(model_path), device="cpu")
    deterministic = not args.stochastic

    if args.grid_name:
        _eval_grid(args, model, model_dir, vecnorm, arm, deterministic)
        return

    scenario = SCENARIOS[args.scenario_name]

    print(f"[eval] model     : {model_path}")
    print(f"[eval] vecnorm   : {vecnorm if vecnorm.exists() else '(none)'}")
    print(f"[eval] scenario  : {args.scenario_name}   obs={args.observation_mode}")
    block = ("protocol" if args.first_seed == 0
             else "screen" if args.first_seed == SELECT_SEED_OFFSET
             else "non-standard")
    print(f"[eval] seeds     : {args.first_seed}…"
          f"{args.first_seed + args.n_seeds - 1}   ({block} block)   "
          f"deterministic={deterministic}")
    if args.first_seed != 0:
        print("[eval] WARNING: this is not the protocol block — ranking only, "
              "never quote this number (9.7).")

    def act_fn(obs: np.ndarray, raw_envs) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    per_seed = rollout_seeds(
        scenario=scenario,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        seeds=eval_seed_block(args.n_seeds, offset=args.first_seed),
        act_fn=act_fn,
        batch=args.batch_envs,
        vecnorm_path=vecnorm,
    )

    write_record(outfile, args.scenario_name, arm, per_seed)

    if args.paired_with:
        paired_report(
            per_seed["cost_total"],
            Path(args.paired_with),
            bar_name=Path(args.paired_with).stem,
        )


if __name__ == "__main__":
    main()

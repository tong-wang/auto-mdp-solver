"""DP benchmark evaluation for single-echelon inventory (spec §9).

Replays the `FiniteHorizonDP` table through `InvSingleEnv` — the same wrapper
RL trains on — over the same CRN seed block `0…n_seeds-1` and writes the same
§9.3 record as `inv_single_ppo_eval.py`, so the two are directly and *pairwise*
comparable (`inv_single_ppo_eval.py --paired-with <this run's _perseed.npy>`).

DP exactness by scenario type (settled by `inv_single_dp_exactness.py`, campaign
item A7a):

| scenario type              | DP state           | exactness         |
|----------------------------|--------------------|-------------------|
| LT=0                       | inventory          | exact             |
| deterministic LT>=1, backlog | inventory position | exact (verified) |
| stochastic LT              | inventory position | approximation     |
| lost sales                 | inventory position | not a valid bar   |

Usage:
    python inv_single_benchmark_dp_eval.py -s simple
    python inv_single_benchmark_dp_eval.py -s simple_k --n-seeds 8192
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inv_single_benchmark_dp import FiniteHorizonDP
from inv_single_eval_common import eval_seed_block, paired_report, rollout_seeds, write_record
from inv_single_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the DP benchmark on inv_single.")
    p.add_argument("-g", "--grid_name", type=str, default=None,
                   help="solve + evaluate the DP on every cell of this GRIDS "
                        "entry (de-padded twins; one record per cell, spec "
                        "9.6; every grid16 cell is backlog + deterministic "
                        "LT, so the exact role holds). Overrides -s.")
    p.add_argument("--cell", type=str, default=None,
                   help="with -g: restrict to one cell id")
    p.add_argument("-s", "--scenario_name", type=str, default="simple",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", type=str, default="vec",
                   choices=["vec", "vec_ip"],
                   help="Cosmetic for a benchmark: the DP reads simulator state, "
                        "not the agent observation. Kept so the env is built the "
                        "same way as in the RL arm.")
    p.add_argument("-a", "--action_mode", type=str, default="discrete",
                   choices=["continuous", "discrete", "hurdle"])
    p.add_argument("--n-seeds", type=int, default=65536)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--outfile", type=str, default=None)
    p.add_argument("--no-cache", action="store_true",
                   help="Force a DP re-solve and overwrite the cached V/pi tables.")
    p.add_argument("--dp-from", type=str, default=None,
                   choices=list(SCENARIOS.keys()),
                   help="Solve the DP table on a DIFFERENT scenario and replay it "
                        "in -s's world (the benchmark analogue of spec §9.6's "
                        "train/eval decoupling). The campaign's S4 bar: the `lt` "
                        "table (deterministic L=2) replayed at `slt`, whose "
                        "E[L] = 2 matches, so it is a legitimate heuristic there.")
    p.add_argument("--paired-with", type=str, default=None,
                   help="Another arm's *_perseed.npy; prints the paired delta.")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def _run_one(args, source, table_source, label: str, outfile: Path,
             arm: str) -> None:
    dp = FiniteHorizonDP(table_source, results_dir=args.outdir,
                         no_cache=args.no_cache)
    # the table is indexed by the state of the world it was SOLVED for, so the
    # rollout must read that world's statistic (inventory vs IP), not -s's
    lt_max = table_source.leadtime.max()

    print(f"[eval] scenario : {label}   DP state="
          f"{'inventory' if lt_max == 0 else 'inventory position'}")
    print(f"[eval] seeds    : 0…{args.n_seeds - 1}")

    def act_fn(obs: np.ndarray, raw_envs) -> np.ndarray:
        acts = np.empty((len(raw_envs), 1), dtype=np.float32)
        for i, env in enumerate(raw_envs):
            st = env._state
            x = st.inventory if lt_max == 0 else st.inventory + sum(st.pipeline)
            acts[i, 0] = float(dp.act(st.period, x))
        return acts

    per_seed = rollout_seeds(
        scenario=source,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        seeds=eval_seed_block(args.n_seeds),
        act_fn=act_fn,
        batch=args.batch_envs,
        vecnorm_path=None,
    )

    write_record(outfile, label, arm, per_seed)

    if args.paired_with:
        paired_report(
            per_seed["cost_total"],
            Path(args.paired_with),
            bar_name=Path(args.paired_with).stem,
        )


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent

    if args.grid_name:
        # per-cell exact DP over the DE-PADDED twins (mean(), never max() —
        # see inv_single_grids.cells_depadded); each cell is its own table
        from inv_single_grids import cells_depadded
        assert args.dp_from is None, "-g solves each cell's own table; drop --dp-from"
        assert args.outfile is None, "-g writes one file per cell; drop --outfile"
        cells = cells_depadded(args.grid_name)
        if args.cell is not None:
            cells = [c for c in cells if c[0] == args.cell]
            assert cells, f"unknown cell {args.cell!r}"
        for cid, slug, sc in cells:
            outfile = (here / args.outdir / f"{args.grid_name}__{slug}" / "benchmark"
                       / f"benchmark_dp_eval_{args.grid_name}__{slug}.tsv")
            _run_one(args, sc, sc, f"{args.grid_name}[{cid}]", outfile,
                     arm="benchmark_dp")
        return

    source = SCENARIOS[args.scenario_name]
    # a sampler target has no single table to solve; the campaign's five rungs
    # are all fixed scenarios, so refuse rather than silently solve seed 0's draw
    assert not callable(source), (
        f"scenario {args.scenario_name!r} is a world sampler; the DP benchmark "
        "needs a fixed scenario."
    )

    table_from = args.dp_from or args.scenario_name
    table_source = SCENARIOS[table_from]
    assert not callable(table_source), (
        f"--dp-from {table_from!r} is a world sampler; the DP needs a fixed scenario."
    )
    arm = "benchmark_dp" if args.dp_from is None else f"benchmark_dp[table={table_from}]"
    suffix = "" if args.dp_from is None else f"_table_{table_from}"
    outfile = Path(
        args.outfile
        if args.outfile
        else here / args.outdir / args.scenario_name / "benchmark"
        / f"benchmark_dp_eval_{args.scenario_name}{suffix}.tsv"
    )
    if args.dp_from is not None:
        print(f"[eval] DP table  : solved on {table_from!r} "
              f"(E[L]={table_source.leadtime.mean():.2f}), replayed at "
              f"{args.scenario_name!r} (E[L]={source.leadtime.mean():.2f})")
    _run_one(args, source, table_source, args.scenario_name, outfile, arm=arm)


if __name__ == "__main__":
    main()

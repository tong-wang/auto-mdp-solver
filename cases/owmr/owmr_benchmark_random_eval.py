"""Score the random floor on the shared CRN seed block (spec 9).

Usage:
    python owmr_benchmark_random_eval.py -s base --n-seeds 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path

from owmr_benchmark_random import random_act_fn
from owmr_eval_common import eval_seed_block, rollout_seeds, write_record
from owmr_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the random floor on owmr.")
    p.add_argument("-s", "--scenario_name", type=str, default="base", choices=list(SCENARIOS))
    p.add_argument("-o", "--observation_mode", type=str, default="raw")
    p.add_argument("-a", "--action_mode", type=str, default="order_frac")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--outfile", type=str, default=None)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    sc = SCENARIOS[args.scenario_name]
    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)
    print(f"[eval] scenario : {args.scenario_name}   arm=random ({args.action_mode} box)")
    per_seed = rollout_seeds(scenario=sc, observation_mode=args.observation_mode,
                             action_mode=args.action_mode, seeds=seeds,
                             act_fn=random_act_fn(), batch=args.batch_envs)
    outfile = Path(args.outfile) if args.outfile else (
        here / args.outdir / args.scenario_name / "benchmark"
        / f"benchmark_random_eval_{args.scenario_name}.tsv")
    prov = [f"# arm: random (feasible floor) — uniform in the {args.action_mode} action box",
            f"# scenario: {args.scenario_name}   seeds: {args.first_seed}..{args.first_seed + args.n_seeds - 1}"]
    write_record(outfile, args.scenario_name, "random", per_seed, seeds=seeds, provenance=prov)


if __name__ == "__main__":
    main()

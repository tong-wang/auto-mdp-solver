"""Score the LB heuristic on the shared CRN seed block (spec 9).

Usage:
    python owmr_benchmark_lb_heuristic_eval.py -s base --n-seeds 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path

from owmr_benchmark_lb_heuristic import build
from owmr_eval_common import eval_seed_block, paired_report, rollout_seeds, write_record
from owmr_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the LB heuristic benchmark on owmr.")
    p.add_argument("-s", "--scenario_name", type=str, default="base", choices=list(SCENARIOS))
    p.add_argument("-o", "--observation_mode", type=str, default="raw")
    p.add_argument("-a", "--action_mode", type=str, default="order_ship",
                   help="the policy emits quantities; keep order_ship")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--outfile", type=str, default=None)
    p.add_argument("--paired-with", type=str, default=None, help="another arm's .seeds.tsv")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    pol = build(args.scenario_name, here, args.outdir)
    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)
    print(f"[eval] scenario : {args.scenario_name}   arm=lb_heuristic (y0*={pol.y0:.3f})")
    print(f"[eval] seeds    : {args.first_seed}..{args.first_seed + args.n_seeds - 1}")
    per_seed = rollout_seeds(
        scenario=pol.scenario, observation_mode=args.observation_mode,
        action_mode="order_ship", seeds=seeds, act_fn=pol.act_fn(), batch=args.batch_envs,
    )
    outfile = Path(args.outfile) if args.outfile else (
        here / args.outdir / args.scenario_name / "benchmark"
        / f"benchmark_lb_heuristic_eval_{args.scenario_name}.tsv")
    prov = [f"# arm: lb_heuristic (feasible, spec 9.9) — order up to y0*={pol.y0:.4f}, myopic allocation with non-negativity",
            f"# scenario: {args.scenario_name}   seeds: {args.first_seed}..{args.first_seed + args.n_seeds - 1}"]
    write_record(outfile, args.scenario_name, "lb_heuristic", per_seed, seeds=seeds, provenance=prov)
    if args.paired_with:
        paired_report(per_seed["cost_total"], Path(args.paired_with), Path(args.paired_with).stem)


if __name__ == "__main__":
    main()

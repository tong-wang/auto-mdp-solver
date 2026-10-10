"""Eval record for the balance-assumption lower bound (spec 9, role `relaxed`).

The bound is analytic, so this "eval" writes the spec-9.3 record from the
solution table rather than from rollouts: `cost_total_mean` = the relaxed
system's optimum on the finite-horizon criterion (horizon periods from the
mean-cover start; `BalanceBound.finite_horizon_bound`, ESCALATION #E26),
variance 0. The `n_seeds` column carries the
protocol block size so `mdp_gates --reference` accepts it beside the rollout
arms; the per-seed sidecar is the constant repeated, since a bound has no
draw. Never a `--baseline`: it is not attainable (spec 9.9).

Usage:
    python owmr_benchmark_lb_eval.py -s base --n-seeds 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from owmr_benchmark_lb import load_or_solve
from owmr_eval_common import METRICS, eval_seed_block, write_record
from owmr_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record the balance-assumption lower bound as an eval TSV.")
    p.add_argument("-s", "--scenario_name", type=str, default="base", choices=list(SCENARIOS))
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--outfile", type=str, default=None)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    bound, sol = load_or_solve(args.scenario_name, here, args.outdir)
    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)
    n = len(seeds)
    per_seed = {m: np.full(n, np.nan) for m in METRICS}
    per_seed["cost_total"] = np.full(n, float(sol["lb_finite_T"]))
    outfile = Path(args.outfile) if args.outfile else (
        here / args.outdir / args.scenario_name / "benchmark"
        / f"benchmark_lb_eval_{args.scenario_name}.tsv")
    c = sol["lb_finite_components"]
    prov = [
        f"# arm: lb (relaxed, spec 9.9) — balance-assumption bound on the finite-horizon criterion: the relaxed "
        f"system's optimum over {sol['horizon']} periods from the mean-cover start; analytic, no rollouts (#E26)",
        f"# scenario: {args.scenario_name}   y0_star: {sol['y0_star']:.4f}   z_star: {sol['z_star']}",
        f"# lb_finite_T: {sol['lb_finite_T']:.4f} = start retailer periods {c['start_retailer_periods']:.4f} + fixed cycles "
        f"{sum(c['fixed_cycles']):.4f} ({len(c['fixed_cycles'])}) + ordered cycles {sum(c['ordered_cycles_exact']):.4f} "
        f"({len(c['ordered_cycles_exact'])} exact) + {c['ordered_cycles_at_longrun_min']} x {sol['lb_per_period']:.6f} "
        f"(the long-run minimum per cycle)",
        f"# long-run average per period {sol['lb_per_period']:.6f} (paper check only, NOT the record; the paper adds "
        f"transit {sol['transit_constant_per_period']:.4f}/period)   x horizon = {sol['lb_longrun_x_T']:.4f}",
        f"# seeds: {args.first_seed}..{args.first_seed + n - 1} (block size only; the bound has no draw)",
    ]
    write_record(outfile, args.scenario_name, "lb", per_seed, seeds=seeds, provenance=prov)


if __name__ == "__main__":
    main()

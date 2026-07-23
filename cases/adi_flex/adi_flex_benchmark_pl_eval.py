"""Evaluate a protection-level heuristic over the spec-section-9 seed protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from adi_flex_scenarios import SCENARIOS
from adi_flex_benchmark_common import default_outfile, evaluate, write_tsv
from adi_flex_benchmark_pl import POLICIES, load_or_solve, make_policy


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a PL heuristic.")
    p.add_argument("-s", "--scenario_name", type=str, default="exp4")
    p.add_argument("--policy", type=str, default="plsigma", choices=POLICIES)
    p.add_argument("--solutions", type=str, default=None)
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--outfile", type=str, default=None)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> int:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    solution = load_or_solve(args.scenario_name, args.solutions)
    policy = make_policy(scenario, args.policy, solution)

    print(f"scenario={args.scenario_name}  policy={args.policy}  "
          f"hold_back={policy.hold_back}  n_seeds={args.n_seeds}")
    stats = evaluate(scenario, lambda: policy, args.n_seeds)
    print(f"cost_mean = {stats['cost_mean']:.4f}   "
          f"(AP lower bound {solution.lower_bound:.4f}, "
          f"gap {100 * (stats['cost_mean'] - solution.lower_bound) / solution.lower_bound:.2f}%)")

    out = Path(args.outfile) if args.outfile else default_outfile(
        args.scenario_name, args.policy
    )
    write_tsv(out, scenario, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

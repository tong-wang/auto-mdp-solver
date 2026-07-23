"""Emit the AP relaxation's bound as a gate-readable reference TSV.

AP is not simulatable: it is a relaxation that permits negative deliveries, so
there is no policy to roll out and no sampling distribution to average. Its
number is an exact expectation from backward induction, not a sample mean.

This script therefore writes reward_mean = -(DP value) with zero variance. That
is honest for its only legitimate use — `--reference` in the eval gate, which
is reported as a gap and never gates. Passing this file as a `--baseline` would
produce an infinite z-score and is meaningless; use the PL heuristics for that,
which are real implementable policies with real sampling variation.

Comparing an exact expectation against seed-averaged means is exactly what the
paper does: section 4.3.2 takes C*_AP from the DP while obtaining C*_0 and
C*_sigma from 100,000 simulation runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adi_flex_scenarios import SCENARIOS
from adi_flex_benchmark_common import default_outfile, write_tsv
from adi_flex_benchmark_pl import load_or_solve


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Emit the AP reference TSV.")
    p.add_argument("-s", "--scenario_name", type=str, default="exp4")
    p.add_argument("--solutions", type=str, default=None)
    p.add_argument("--n-seeds", type=int, default=8192,
                   help="recorded for protocol symmetry; the DP value is exact")
    p.add_argument("--outfile", type=str, default=None)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> int:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    solution = load_or_solve(args.scenario_name, args.solutions)

    stats = {
        "reward_mean": -solution.lower_bound,
        "reward_var":  0.0,   # exact expectation, not a sample
        "semivar_d":   0.0,
        "semivar_u":   0.0,
        "cost_mean":   solution.lower_bound,
    }
    print(f"scenario={args.scenario_name}  AP bound (lower) = {solution.lower_bound:.4f}")

    out = Path(args.outfile) if args.outfile else default_outfile(
        args.scenario_name, "ap"
    )
    write_tsv(out, scenario, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate the random benchmark over the spec-section-9 seed protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from adi_flex_scenarios import SCENARIOS
from adi_flex_benchmark_common import default_outfile, evaluate, write_tsv
from adi_flex_benchmark_random import RandomPolicy


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the random benchmark.")
    p.add_argument("-s", "--scenario_name", type=str, default="exp4")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--outfile", type=str, default=None)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> int:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    policy = RandomPolicy(seed_salt=scenario.seed_salt)

    print(f"scenario={args.scenario_name}  policy=random  n_seeds={args.n_seeds}")
    stats = evaluate(scenario, lambda: policy, args.n_seeds)
    print(f"cost_mean = {stats['cost_mean']:.4f}")

    out = Path(args.outfile) if args.outfile else default_outfile(
        args.scenario_name, "random"
    )
    write_tsv(out, scenario, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Spec-§9 eval for the ``random`` floor benchmarks (see cnv_benchmark_common.py)."""

from __future__ import annotations

from pathlib import Path

from cnv_benchmark_common import build_arg_parser, evaluate
from cnv_benchmark_random import POLICIES, make_policy


def _build_arg_parser():
    p = build_arg_parser("Evaluate the random / no-learning floors on a cnv scenario or grid.")
    p.add_argument("--policy", type=str, default="uniform", choices=list(POLICIES))
    return p


def parse_args():
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    evaluate(f"random_{args.policy}", make_policy(args.policy), args,
             Path(__file__).resolve().parent, provenance=[f"# policy: {args.policy}"])


if __name__ == "__main__":
    main()

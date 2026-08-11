"""Spec-§9 eval for the greedy benchmark (see mab_benchmark_common.py)."""

from __future__ import annotations

from mab_benchmark_common import build_arg_parser, evaluate
from mab_benchmark_greedy import make_policy


def main() -> None:
    args = build_arg_parser("Evaluate the greedy benchmark on a mab scenario.").parse_args()
    evaluate("greedy", make_policy, args)


if __name__ == "__main__":
    main()

"""Spec-§9 eval for the distilled rule benchmark (see mab_benchmark_common.py)."""

from __future__ import annotations

from mab_benchmark_common import build_arg_parser, evaluate
from mab_benchmark_rule import make_policy


def main() -> None:
    args = build_arg_parser("Evaluate the #E26 two-constant rule on a mab scenario.").parse_args()
    evaluate("rule", make_policy, args)


if __name__ == "__main__":
    main()

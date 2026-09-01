"""Evaluate the random-valid baseline over the spec-§9 seed protocol."""

from __future__ import annotations

from game2048_benchmark_common import build_arg_parser, evaluate
from game2048_benchmark_random import make_policy

if __name__ == "__main__":
    args = build_arg_parser(__doc__).parse_args()
    evaluate("random", make_policy, args)

"""Evaluate snake-leaf expectimax over the spec-§9 seed protocol.

Writes benchmark_snake_expectimax_d{depth}_eval_{scenario}.tsv beside the
other baselines. Supports --seed-start chunking exactly like the reference
expectimax rows (the 4x4 leg runs as parallel chunks and is merged).
"""

from __future__ import annotations

from game2048_benchmark_common import build_arg_parser, evaluate
from game2048_benchmark_snake_expectimax import DEFAULT_DEPTH, make_policy

if __name__ == "__main__":
    p = build_arg_parser(__doc__)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    args = p.parse_args()
    evaluate(f"snake_expectimax_d{args.depth}",
             lambda env: make_policy(env, args.depth), args)

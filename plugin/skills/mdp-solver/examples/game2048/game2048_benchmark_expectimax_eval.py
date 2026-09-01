"""Evaluate the expectimax reference over the spec-§9 seed protocol.

``--depth`` is the number of MAX/CHANCE layer pairs. Cost grows roughly by
(4 x 2 x empty cells) per extra level, so depth 3 is ~an order of magnitude
slower than depth 2; the default is the one the leaderboard is run at.
"""

from __future__ import annotations

from game2048_benchmark_common import build_arg_parser, evaluate
from game2048_benchmark_expectimax import DEFAULT_DEPTH, make_policy

if __name__ == "__main__":
    parser = build_arg_parser(__doc__)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help=f"expectimax layer pairs (default {DEFAULT_DEPTH})")
    args = parser.parse_args()
    evaluate(f"expectimax_d{args.depth}",
             lambda env: make_policy(env, args.depth), args)

"""Evaluate the snake-heuristic baseline over the spec-§9 seed protocol.

Writes benchmark_snake_eval_{scenario}.tsv beside the other baselines, then
prints the two discussion instruments (forced-break rate, 3-move-regime
share) — aggregates only, the TSV schema stays untouched.
"""

from __future__ import annotations

from game2048_benchmark_common import build_arg_parser, evaluate
from game2048_benchmark_snake import make_policy_factory

if __name__ == "__main__":
    p = build_arg_parser(__doc__)
    p.add_argument("--base", type=float, default=4.0,
                   help="weight mode: rank weight base (>2 or the ordering "
                        "degenerates)")
    p.add_argument("--mode", choices=["weight", "lex", "proc"],
                   default="weight",
                   help="weight = v1 (defects preserved for the record); "
                        "lex = v2, lexicographic path comparison; proc = v3, "
                        "the procedural snake (hard constraint + monotone "
                        "prefix + 1-spawn safety)")
    p.add_argument("--orient-lock", dest="orient_lock", type=int, default=0,
                   help="freeze the orientation once the anchor tile reaches "
                        "this value (0 = mode default); switches before the "
                        "lock need the anchor/prefix strictly beaten")
    p.add_argument("--no-safety", dest="safety", action="store_false",
                   help="proc mode: ablate the 1-spawn safety obligation "
                        "(the EXECUTION-vs-EYES decomposition arm)")
    args = p.parse_args()
    stats: dict = {}
    method = {"weight": "snake", "lex": "snake_lex",
              "proc": "snake_proc" if args.safety else "snake_proc_ns"}[args.mode]
    evaluate(method, make_policy_factory(stats, base=args.base,
                                         mode=args.mode,
                                         orient_lock=args.orient_lock,
                                         safety=args.safety), args)
    steps = max(stats.get("steps", 0), 1)
    print(f"\nsnake instruments over {args.n_seeds} episodes:")
    print(f"  forced-break rate : {stats['forced_breaks'] / steps:.4f} "
          f"({stats['forced_breaks']} / {steps} steps)")
    print(f"  3-move-regime share: {stats['regime_steps'] / steps:.4f}")
    print(f"  orient switches/ep : "
          f"{stats['orient_switches'] / max(stats['episodes'], 1):.2f}")

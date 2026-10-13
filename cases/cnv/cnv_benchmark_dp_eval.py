"""Spec-§9 eval for the ``dp`` benchmark (see cnv_benchmark_common.py).

Solves (or loads the cached) exact DP table per target row, then plays it
through the same env and seed block every other arm uses.

    python cnv_benchmark_dp_eval.py -s base --policy timing
    python cnv_benchmark_dp_eval.py -g fig3 --policy full --n-seeds 8192
"""

from __future__ import annotations

from pathlib import Path

from cnv_benchmark_common import build_arg_parser, eval_targets, evaluate
from cnv_benchmark_dp import POLICIES, DpTable, solve, table_path

HERE = Path(__file__).resolve().parent


def _build_arg_parser():
    p = build_arg_parser("Evaluate the exact DP policy on a cnv scenario or grid.")
    p.add_argument("--policy", type=str, default="timing", choices=list(POLICIES))
    p.add_argument("--grid-m", type=int, default=40,
                   help="timing DP: T grid points per unit period")
    p.add_argument("--tail", type=float, default=1e-9,
                   help="demand / order truncation: the prior-predictive (1 - tail) quantile")
    p.add_argument("--resolve", action="store_true",
                   help="re-solve even if a cached table exists")
    return p


def parse_args():
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    target, rows = eval_targets(args)
    by_row = {row_id: src for row_id, src in rows}

    def make_policy(env_probe):
        src = env_probe.scenario
        row_id = next(rid for rid, s in by_row.items() if s is src)
        path = table_path(HERE, args.outdir, target, row_id, args.policy)
        if path.exists() and not args.resolve:
            table = DpTable.load(path)
            print(f"  dp {args.policy}: loaded {path}")
        else:
            kw = {"tail": args.tail} if args.policy == "full" else {"tail": args.tail, "grid_m": args.grid_m}
            table = solve(src, args.policy, **kw)
            table.save(path)
            print(f"  dp {args.policy}: solved and cached {path}")
        return table.act

    evaluate(f"dp_{args.policy}", make_policy, args, HERE,
             provenance=[f"# policy: {args.policy}", f"# grid_m: {args.grid_m}", f"# tail: {args.tail}"])


if __name__ == "__main__":
    main()

"""Spec-§9 eval for the ``myopic`` benchmark (see cnv_benchmark_common.py).

    python cnv_benchmark_myopic_eval.py -s base --policy timing
    python cnv_benchmark_myopic_eval.py -g fig3 --policy event --n-seeds 8192
"""

from __future__ import annotations

from pathlib import Path

from cnv_benchmark_common import build_arg_parser, evaluate
from cnv_benchmark_myopic import POLICIES, make_policy


def _build_arg_parser():
    p = build_arg_parser("Evaluate the paper's myopic policies on a cnv scenario or grid.")
    p.add_argument("--policy", type=str, default="timing", choices=list(POLICIES),
                   help="observation type the myopic newsvendor learns from")
    return p


def parse_args():
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    evaluate(f"myopic_{args.policy}", make_policy(args.policy), args,
             Path(__file__).resolve().parent, provenance=[f"# policy: {args.policy}"])


if __name__ == "__main__":
    main()

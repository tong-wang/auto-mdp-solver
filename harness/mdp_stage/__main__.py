"""CLI entry point — see the package docstring for the two modes."""

from __future__ import annotations

import sys

from mdp_stage.gate import EPISODES, OPS, run_gate, stage_table


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m mdp_stage",
        description="Pipeline entry gates: is this domain ready for the op "
        "you want to run? Without --for, print the cheap-check stage table.",
    )
    ap.add_argument("domain", help="domain directory (or its *_schema.json)")
    ap.add_argument(
        "--for", dest="op", choices=sorted(OPS), default=None,
        help="run this op's full entry gate and exit 0/1",
    )
    ap.add_argument(
        "--episodes", type=int, default=EPISODES,
        help=f"differential episodes at the solve gate (default {EPISODES})",
    )
    args = ap.parse_args(argv)

    if args.op:
        _, code = run_gate(args.domain, args.op, episodes=args.episodes)
        return code
    return stage_table(args.domain)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

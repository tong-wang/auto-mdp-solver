"""Emit the AP relaxation's bound as an eval-format TSV row (spec §9.3, §9.9).

**This deliberately does NOT play a policy through the gym**, and that is the
whole point. The other `*_eval.py` scripts here simulate: they take a solved
table, run it over seeds 0..n-1, and report a sample mean. Doing that with the
AP table would produce a perfectly good *feasible* number — a real policy under
the real information set, essentially PL(0) — and then file it under the `ap`
arm, whose declared role is `relaxed`. The bound would silently become a
heuristic, the het bracket would collapse, and every "% of bound" figure
computed against it would be wrong in the safe-looking direction.

The relaxation's value is the recursion's own C_1(0, vhat=0), not a simulated
quantity, so this reads it off the solved artifact and writes one row in the
shared eval format. Variance columns are 0 by construction: a bound is a
number, not a sample, and there is no seed protocol to average over. Consumers
that want a spread should read the SE of the arms being bracketed, not this.

Example usage:
    python adi_flex_benchmark_ap.py -s het_exp4
    python adi_flex_benchmark_ap_eval.py -s het_exp4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from adi_flex_scenarios import SCENARIOS
from adi_flex_benchmark_common import EVAL_HEADER, eval_row


def read_bound(solution_txt: Path) -> float:
    """C_1(0,0) as written into the arm's .txt header by run_arm()."""
    head = solution_txt.read_text().splitlines()[0]
    m = re.search(r"C_1\(0,0\)=([-\d.eE+]+)", head)
    if not m:
        raise SystemExit(f"no C_1(0,0) in {solution_txt}: {head!r}")
    return float(m.group(1))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Emit the AP-relaxation bound as an eval TSV row.")
    p.add_argument("-s", "--scenario_name", type=str, default="het_exp4",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--ap-solutions", type=str, default=None,
                   help="The arm's .txt (defaults to results/<scenario>/ap/<scenario>.txt)")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV output (defaults to results/<scenario>/benchmark/"
                        "benchmark_ap_eval_<scenario>.tsv)")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    here = Path(__file__).resolve().parent / "results" / args.scenario_name / "ap"
    src = Path(args.ap_solutions) if args.ap_solutions else here / f"{args.scenario_name}.txt"
    if not src.exists():
        raise SystemExit(f"{src} not found — run adi_flex_benchmark_ap.py -s {args.scenario_name} first")

    bound = read_bound(src)
    # the `benchmark_{method}_eval` marker is what lets `mdp_gates --ir` find
    # this arm's declared role=relaxed; named anything else it is inert
    outfile = (Path(args.outfile) if args.outfile else
               Path(__file__).resolve().parent / "results" / args.scenario_name
               / "benchmark" / f"benchmark_ap_eval_{args.scenario_name}.tsv")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    header = EVAL_HEADER
    # reward = -cost (§9.3): the LOWEST attainable cost is the HIGHEST attainable
    # reward, so this row is an upper bound on reward — `≽ opt` either way round
    row = (
        eval_row(scenario, {"reward_mean": -bound, "reward_var": 0.0,
                            "semivar_d": 0.0, "semivar_u": 0.0})
    )
    with open(outfile, "w") as f:
        f.write("# role=relaxed: AP-relaxation LOWER bound on cost, not attainable.\n")
        f.write("# variance columns are 0 by construction - this is a bound, not a sample.\n")
        f.write(header + "\n")
        f.write(row + "\n")
    print(header)
    print(row)
    print(f"\nAP bound cost = {bound:.4f}   (lower bound; no policy can beat it)")
    print(f"results saved -> {outfile}")


if __name__ == "__main__":
    main()

"""Myopic benchmark: safety stocks that ignore future ordering opportunities (spec §9).

The optimal offsets `b_n` (fnv_benchmark_dp.py / fnv_benchmark_prop2.py) account
for the option to order again later. This benchmark deliberately does not: it
uses the **myopic** safety stock from Wang, Atasu & Kurtuluş (2012),

    b_hat_n = sigma_tilde_n * Z_{beta_n},    beta_n = (r - c_n) / r,
    sigma_tilde_n^2 = sum_{i=n+1}^{N+1} sigma_i^2   (residual uncertainty after n)

applied at every period, i.e. order up to `mu + I + b_hat_n` (a-MMFE) or
`exp(mu + I + b_hat_n)` (m-MMFE). Z_beta is the standard normal beta-quantile,
so each period is priced as a one-shot newsvendor against the uncertainty that
still remains.

**Corollary 1 of the paper states b_n <= b_hat_n**: neglecting the option to
buy later always inflates the safety stock. So this arm systematically
over-orders early, and its gap to the optimum is the value of recognizing the
future ordering opportunities. That makes it the natural **must-beat baseline**
(`--baseline` in §13) for a learned policy, where the optimal solvers are the
`--reference` a learned policy is measured against but not expected to exceed.

Scope note: this is the myopic *multiordering heuristic*, NOT the paper's
single-ordering benchmark of §4.2 (Propositions 3 and 4), which solves a
genuinely different problem — when to place one single order. Only the safety
stock formula is borrowed from that section. b_hat_n is also what
fnv_benchmark_prop2.py checks Corollary 1 against, which is why it is defined
here, once.

Emits the same offsets TSV that fnv_benchmark_dp_eval.py consumes.

Example usage:
    python fnv_benchmark_myopic.py -s FNV-aMMFE
    python fnv_benchmark_dp_eval.py --dp-solutions results/FNV-aMMFE/myopic/FNV-aMMFE.txt -s FNV-aMMFE
"""

import argparse
import math
from pathlib import Path

import numpy as np

from fnv_grids import GRIDS
from fnv_scenarios import FnvScenario, SCENARIOS

_erf = np.vectorize(math.erf)


def _Phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _Phi_inv(p: float) -> float:
    """Inverse standard normal CDF by bisection."""
    assert 0.0 < p < 1.0, f"p must be in (0,1), got {p}."
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(_Phi(mid)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def residual_sigma(scenario: FnvScenario, n: int) -> float:
    """sigma_tilde_n — the uncertainty still unresolved after period n.

    The paper's sigma_{i} is this domain's stdevs[i-1], so
    sum_{i=n+1}^{N+1} sigma_i^2 becomes sum_{j=n}^{N} stdevs[j]^2.
    """
    sig = scenario.signal.stdevs
    return math.sqrt(sum(sig[j] ** 2 for j in range(n, scenario.N + 1)))


def myopic_offsets(scenario: FnvScenario) -> tuple[float, ...]:
    """(b_hat_1, ..., b_hat_N) — the single-order safety stocks."""
    out = []
    for n in range(1, scenario.N + 1):
        beta = (scenario.r - scenario.ordering_cost_rate(n)) / scenario.r
        out.append(residual_sigma(scenario, n) * _Phi_inv(beta)
                   if 0.0 < beta < 1.0 else float("nan"))
    return tuple(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Solve FNV myopic (future-ordering-blind) safety stocks.")
    p.add_argument("-s", "--scenario_name", type=str, default="FNV-aMMFE",
                   choices=list(SCENARIOS) + list(GRIDS),
                   help="Scenario (§5.4) or grid (§5.6); a grid solves every cell")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV of offsets (defaults to results/<scenario>/myopic/<scenario>.txt)")
    return p


def resolve_cells(scenario_name: str) -> list[tuple[str, FnvScenario]]:
    if scenario_name in SCENARIOS:
        return [(scenario_name, SCENARIOS[scenario_name])]
    if scenario_name in GRIDS:
        return list(GRIDS[scenario_name])
    raise SystemExit(f"unknown scenario/grid {scenario_name!r}")


def main() -> None:
    args  = _build_arg_parser().parse_args()
    cells = resolve_cells(args.scenario_name)

    # §8.4: benchmark solution tables live at
    # results/{scenario}/benchmark/{method}/{scenario}.txt
    outfile = (Path(args.outfile) if args.outfile else
               Path(__file__).resolve().parent / "results" / args.scenario_name
               / "benchmark" / "myopic" / f"{args.scenario_name}.txt")

    print(f"scenario  {args.scenario_name}  ({len(cells)} cell(s))")
    print(f"output    {outfile}")
    print("method    myopic safety stock b_hat_n = sigma_tilde_n * Z_beta_n\n")

    N = cells[0][1].N
    header = "stdev\tT\tlambda\t" + "\t".join(f"b{n}" for n in range(1, N + 1))
    print(header)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()
        for _, sc in cells:
            b = myopic_offsets(sc)
            row = (f"{sc.stdev}\t{sc.T}\t{sc.lamb}\t"
                   + "\t".join(f"{x:.6f}" for x in b))
            print(row, flush=True)
            f.write(row + "\n")
            f.flush()

    print(f"\noffsets saved → {outfile}")
    print(f"next: python fnv_benchmark_dp_eval.py --dp-solutions {outfile} "
          f"-s {args.scenario_name}")


if __name__ == "__main__":
    main()

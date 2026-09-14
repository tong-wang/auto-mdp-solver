"""Exact DP solver for the discretized dynamic pricing MDP.

Backward induction over (period t, remaining stock n) with the price
optimized on a fine grid per state:

    V[T][n] = q * n
    V[t][0] = 0                                (absorbing: sold out)
    V[t][n] = max_p  E[ p * min(X, n) + V[t+1][n - min(X, n)] ],
              X ~ Poisson(lambda(p) * dt),  lambda(p) = a * exp(-alpha * p)

This is the optimal policy for the discretized MDP (the same MDP the RL agent
faces), so it upper-bounds any learned policy up to the price-grid resolution.

Writes the policy table (t, n, price) to the spec's benchmark-solutions
location ``results/{scenario_name}/benchmark/dp/{scenario_name}.txt`` and
prints the optimal expected revenue V[0][n0].

Example usage:
    python dynamic_pricing_benchmark_dp.py -s simple
    python dynamic_pricing_benchmark_dp.py -s ample_stock --price-grid-step 0.001
"""

import argparse
from pathlib import Path

import numpy as np

from dynamic_pricing_scenarios import DynamicPricingScenario, SCENARIOS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Solve the dynamic pricing MDP by backward induction.")
    p.add_argument("-s", "--scenario_name", type=str, default="simple",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--price-grid-step", type=float, default=0.005,
                   help="Price grid resolution (default 0.005)")
    p.add_argument("--price-high", type=float, default=8.0,
                   help="Upper end of the price grid (default 8.0, the gym action bound)")
    p.add_argument("--outfile", type=str, default=None,
                   help="Solutions file (defaults to results/<scenario>/benchmark/dp/<scenario>.txt)")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_dp(
    scenario: DynamicPricingScenario,
    price_grid_step: float = 0.005,
    price_high: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (V, policy): V[t][n] over t=0..T, policy[t][n] over t=0..T-1.

    policy[t][0] is a placeholder (sold out: no decision matters).
    """
    T, n0, q = scenario.horizon, scenario.n0, scenario.q
    prices = np.arange(0.0, price_high + 1e-12, price_grid_step)
    rates  = scenario.a * np.exp(-scenario.alpha * prices) * scenario.dt

    # pmf[p_idx, s] = P(X = s | price), s = 0..n0-1; the s >= n mass is lumped
    # into the sold-out tail per state below
    pmf = np.zeros((len(prices), n0))
    pmf[:, 0] = np.exp(-rates)
    for s in range(1, n0):
        pmf[:, s] = pmf[:, s - 1] * rates / s

    V      = np.zeros((T + 1, n0 + 1))
    V[T]   = q * np.arange(n0 + 1)
    policy = np.zeros((T, n0 + 1))

    for t in range(T - 1, -1, -1):
        for n in range(1, n0 + 1):
            s        = np.arange(n)
            pmf_n    = pmf[:, :n]
            tail     = 1.0 - pmf_n.sum(axis=1)          # P(X >= n): sell out
            exp_sold = pmf_n @ s + tail * n
            cont     = pmf_n @ V[t + 1, n - s] + tail * V[t + 1, 0]
            vals     = prices * exp_sold + cont
            k        = int(np.argmax(vals))
            V[t, n]      = vals[k]
            policy[t, n] = prices[k]

    return V, policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    outfile = (
        Path(args.outfile) if args.outfile
        else Path(__file__).resolve().parent / "results" / args.scenario_name / "benchmark" / "dp" / f"{args.scenario_name}.txt"
    )

    V, policy = solve_dp(scenario, args.price_grid_step, args.price_high)
    v0 = V[0, scenario.n0]
    print(f"expected optimal revenue V[0][{scenario.n0}] = {v0:.4f}")
    print(f"initial price policy[0][{scenario.n0}]       = {policy[0, scenario.n0]:.3f}")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(f"# dynamic_pricing DP solutions  scenario={args.scenario_name}  "
                f"a={scenario.a} alpha={scenario.alpha} dt={scenario.dt} q={scenario.q} "
                f"n0={scenario.n0} horizon={scenario.horizon}\n")
        f.write(f"# price_grid_step={args.price_grid_step} price_high={args.price_high}\n")
        f.write(f"# expected_optimal_revenue={v0:.6f}\n")
        f.write("t\tn\tprice\n")
        for t in range(scenario.horizon):
            for n in range(1, scenario.n0 + 1):
                f.write(f"{t}\t{n}\t{policy[t, n]:.4f}\n")

    print(f"solutions saved → {outfile}")


if __name__ == "__main__":
    main()

"""Evaluate the DP optimal policy on the dynamic pricing scenario.

Loads the precomputed DP price table (t, n, price) produced by
dynamic_pricing_dp.py and plays it through the gym wrapper: at each step the
price is looked up from (period, inventory). Runs episodes with seeds
0..n_seeds-1 and reports mean revenue, variance, and semi-variances — the
same output format as dynamic_pricing_ppo_eval.py for direct comparison.

Example usage:
    python dynamic_pricing_dp_eval.py --dp-solutions results/simple/dp/simple.txt
    python dynamic_pricing_dp_eval.py --dp-solutions results/simple/dp/simple.txt -s simple --n-seeds 8192
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dynamic_pricing_gym import DynamicPricingEnv
from dynamic_pricing_scenarios import DynamicPricingScenario, SCENARIOS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the DP policy on the dynamic pricing scenario.")
    p.add_argument("--dp-solutions", type=str, required=True,
                   help="Path to the precomputed DP solutions file (from dynamic_pricing_dp.py)")
    p.add_argument("-s", "--scenario_name", type=str, default="simple",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", type=str, default="vec",
                   choices=["vec", "vec_d"])
    p.add_argument("--n-seeds", type=int, default=65536,
                   help="Number of episode seeds per scenario (seeds 0..n-1)")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV output (defaults to results/<scenario>/dp/dp_eval_<scenario>.tsv)")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_scenario(
    price_table: np.ndarray,
    scenario: DynamicPricingScenario,
    n_seeds: int,
    observation_mode: str,
) -> dict:
    """Play the DP price table for seeds 0..n_seeds-1; return revenue stats."""
    env = DynamicPricingEnv(
        scenario=scenario,
        action_mode="price",
        observation_mode=observation_mode,
    )

    revenues = np.zeros(n_seeds)

    for ep_seed in range(n_seeds):
        if ep_seed % 10000 == 0:
            print(f"  seed {ep_seed}/{n_seeds}", flush=True)

        obs, _ = env.reset(seed=ep_seed)
        terminated = False
        while not terminated:
            n = int(round(float(obs[0])))                       # inventory
            t = scenario.horizon - int(round(float(obs[1])))    # period
            action = np.array([price_table[t, n]], dtype=np.float32)
            obs, _, terminated, _, _ = env.step(action)

        revenues[ep_seed] = env.total_reward

    env.close()

    mu        = revenues.mean()
    semivar_d = float(np.sum(np.maximum(mu - revenues, 0.0) ** 2) / (n_seeds - 1))
    semivar_u = float(np.sum(np.maximum(revenues - mu, 0.0) ** 2) / (n_seeds - 1))

    return {
        "revenue_mean": float(mu),
        "revenue_var":  float(revenues.var(ddof=1)),
        "semivar_d":    semivar_d,
        "semivar_u":    semivar_u,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    dp_path = Path(args.dp_solutions)
    outfile = (
        Path(args.outfile) if args.outfile
        else Path(__file__).resolve().parent / "results" / args.scenario_name / "dp" / f"dp_eval_{args.scenario_name}.tsv"
    )

    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    sol = pd.read_csv(dp_path, sep="\t", comment="#")
    price_table = np.zeros((scenario.horizon, scenario.n0 + 1))
    price_table[sol["t"].to_numpy(), sol["n"].to_numpy()] = sol["price"].to_numpy()

    print(f"dp_solutions  {dp_path}  ({len(sol)} rows)")
    print(f"output        {outfile}")
    print(f"n_seeds       {args.n_seeds}")

    header = "a\talpha\tn0\trevenue_mean\trevenue_var\tsemivar_d\tsemivar_u"
    print(header)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()

        stats = evaluate_scenario(
            price_table=price_table,
            scenario=scenario,
            n_seeds=args.n_seeds,
            observation_mode=args.observation_mode,
        )
        row = (
            f"{scenario.a}\t{scenario.alpha}\t{scenario.n0}\t"
            f"{stats['revenue_mean']:.6f}\t{stats['revenue_var']:.6f}\t"
            f"{stats['semivar_d']:.6f}\t{stats['semivar_u']:.6f}"
        )
        print(row, flush=True)
        f.write(row + "\n")
        f.flush()

    print(f"\nresults saved → {outfile}")


if __name__ == "__main__":
    main()

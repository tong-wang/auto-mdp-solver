"""Fluid (deterministic) heuristics for the dynamic pricing problem.

Solves the paper's deterministic fluid relaxation in closed form and evaluates
the resulting static policies through the gym, giving the mandatory
non-DP yardsticks:

- ``fixed``  — the GvR fixed-price heuristic p_FP = max(p*, p0), where
               p* = 1/alpha maximizes the revenue rate p*lambda(p) and
               p0 solves lambda(p0)*H = n0 (the run-out price, H = horizon*dt).
               Asymptotically optimal as n0 and demand scale up.
- ``myopic`` — always p* = 1/alpha (ignores the stock constraint).
- ``random`` — uniform price in [0, price-high] each step (sanity floor).

Output format matches dynamic_pricing_ppo_eval.py / dynamic_pricing_dp_eval.py.

Example usage:
    python dynamic_pricing_lp.py -s simple --policy fixed
    python dynamic_pricing_lp.py -s simple --policy random --n-seeds 8192
"""

import argparse
from pathlib import Path

import numpy as np

from dynamic_pricing_gym import DynamicPricingEnv
from dynamic_pricing_scenarios import DynamicPricingScenario, SCENARIOS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate fluid heuristic policies for dynamic pricing.")
    p.add_argument("-s", "--scenario_name", type=str, default="simple",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--policy", type=str, default="fixed",
                   choices=["fixed", "myopic", "random"])
    p.add_argument("--price", type=float, default=None,
                   help="Override the static price (fixed/myopic policies only)")
    p.add_argument("--price-high", type=float, default=8.0,
                   help="Upper bound for the random policy's uniform price draw")
    p.add_argument("--n-seeds", type=int, default=65536,
                   help="Number of episode seeds (seeds 0..n-1)")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV output (defaults to results/<scenario>/dp/lp_eval_<scenario>_<policy>.tsv)")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# Fluid solution
# ---------------------------------------------------------------------------

def fluid_prices(scenario: DynamicPricingScenario) -> dict[str, float]:
    """Closed-form fluid quantities: myopic price, run-out price, fixed price,
    and the deterministic revenue bound at p_FP."""
    H = scenario.horizon * scenario.dt
    p_myopic = 1.0 / scenario.alpha
    p_runout = float(np.log(scenario.a * H / scenario.n0) / scenario.alpha)
    p_fp     = max(p_myopic, p_runout)
    demand_fp = scenario.a * float(np.exp(-scenario.alpha * p_fp)) * H
    bound     = p_fp * min(demand_fp, float(scenario.n0))
    return {"p_myopic": p_myopic, "p_runout": p_runout, "p_fp": p_fp, "fluid_bound": bound}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    scenario: DynamicPricingScenario,
    policy: str,
    price: float | None,
    price_high: float,
    n_seeds: int,
) -> dict:
    env = DynamicPricingEnv(scenario=scenario, action_mode="price")
    revenues = np.zeros(n_seeds)

    for ep_seed in range(n_seeds):
        if ep_seed % 10000 == 0:
            print(f"  seed {ep_seed}/{n_seeds}", flush=True)

        env.reset(seed=ep_seed)
        # policy-stream rng, independent of the demand stream
        rng = np.random.default_rng(np.random.SeedSequence([9999, ep_seed]))
        terminated = False
        while not terminated:
            p = float(rng.uniform(0.0, price_high)) if policy == "random" else price
            _, _, terminated, _, _ = env.step(np.array([p], dtype=np.float32))

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
    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    fluid = fluid_prices(scenario)
    print("fluid solution: " + "  ".join(f"{k}={v:.4f}" for k, v in fluid.items()))

    if args.policy == "fixed":
        price = args.price if args.price is not None else fluid["p_fp"]
    elif args.policy == "myopic":
        price = args.price if args.price is not None else fluid["p_myopic"]
    else:
        price = None

    outfile = (
        Path(args.outfile) if args.outfile
        else Path(__file__).resolve().parent / "results" / args.scenario_name / "dp"
             / f"lp_eval_{args.scenario_name}_{args.policy}.tsv"
    )
    label = f"policy={args.policy}" + (f" price={price:.4f}" if price is not None else "")
    print(f"{label}  n_seeds={args.n_seeds}  output={outfile}")

    header = "a\talpha\tn0\trevenue_mean\trevenue_var\tsemivar_d\tsemivar_u"
    print(header)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(f"# {label}\n")
        f.write(header + "\n")
        f.flush()

        stats = evaluate_policy(
            scenario=scenario,
            policy=args.policy,
            price=price,
            price_high=args.price_high,
            n_seeds=args.n_seeds,
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

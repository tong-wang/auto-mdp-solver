"""Evaluate a trained PPO model on a dynamic pricing scenario.

Runs episodes with seeds 0..n_seeds-1 and reports mean revenue, variance, and
semi-variances — the same output format as dynamic_pricing_benchmark_dp_eval.py for
direct comparison against the DP optimum and the fluid heuristics.

Seeds are injected via env_method("reset", seed=...) so that VecNormalize's
observation normalization is still applied correctly via venv.normalize_obs()
(spec §9.5). The eval must use the same observation/action modes the model
was trained with.

Example usage:
    python dynamic_pricing_ppo_eval.py --model-path results/simple/PPO_.../simple_ppo.zip
    python dynamic_pricing_ppo_eval.py --model-path ... -a intensity --n-seeds 8192
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dynamic_pricing_gym import DynamicPricingEnv
from dynamic_pricing_scenarios import DynamicPricingScenario, SCENARIOS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate PPO on a dynamic pricing scenario.")
    p.add_argument("--model-path",   type=str, required=True,
                   help="Path to the saved model (.zip)")
    p.add_argument("--vecnorm-path", type=str, default=None,
                   help="Path to vecnormalize.pkl (defaults to same dir as model)")
    p.add_argument("-s", "--scenario_name", type=str, default="simple",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", type=str, default="vec",
                   choices=["vec", "vec_d"])
    p.add_argument("-a", "--action_mode",      type=str, default="price",
                   choices=["price", "intensity"])
    p.add_argument("-r", "--reward_mode",      type=str, default="revenue",
                   choices=["revenue"])
    p.add_argument("--n-seeds", type=int, default=65536,
                   help="Number of episode seeds in the block")
    p.add_argument("--first-seed", type=int, default=0,
                   help="First episode seed (spec §9.2). The protocol block is "
                        "0; §9.7's checkpoint screen runs at 1000000 so the "
                        "layer that ranks never touches the block that quotes.")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV output (defaults to <model_dir>/ppo_eval_<scenario_name>.tsv)")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_scenario(
    model: PPO,
    vecnorm_path: Path | None,
    scenario: DynamicPricingScenario,
    n_seeds: int,
    first_seed: int,
    observation_mode: str,
    action_mode: str,
    reward_mode: str,
) -> dict:
    """Run the seed block first_seed..first_seed+n_seeds-1, return stats."""
    env = DynamicPricingEnv(
        scenario=scenario,
        observation_mode=observation_mode,
        action_mode=action_mode,
        reward_mode=reward_mode,
    )
    venv = DummyVecEnv([lambda: env])
    if vecnorm_path is not None and vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training    = False
        venv.norm_reward = False

    revenues = np.zeros(n_seeds)

    for i, ep_seed in enumerate(range(first_seed, first_seed + n_seeds)):
        if i % 10000 == 0:
            print(f"  seed {i}/{n_seeds}", flush=True)
        # inject seed via env_method so VecNormalize normalization stays active
        raw_obs, _ = venv.env_method("reset", seed=ep_seed)[0]
        obs        = venv.normalize_obs(raw_obs[np.newaxis, :])

        done    = False
        revenue = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done_arr, _ = venv.step(action)
            revenue += float(rewards[0])
            done     = bool(done_arr[0])

        revenues[i] = revenue

    venv.close()

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

    model_path   = Path(args.model_path)
    vecnorm_path = Path(args.vecnorm_path) if args.vecnorm_path else model_path.parent / "vecnormalize.pkl"
    outfile      = Path(args.outfile) if args.outfile else model_path.parent / f"ppo_eval_{args.scenario_name}.tsv"

    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    model = PPO.load(str(model_path))
    print(f"model     {model_path}")
    print(f"vecnorm   {vecnorm_path}  (exists={vecnorm_path.exists()})")
    print(f"output    {outfile}")
    print(f"seeds     {args.n_seeds} from {args.first_seed}")

    header = "a\talpha\tn0\trevenue_mean\trevenue_var\tsemivar_d\tsemivar_u"
    print(header)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()

        stats = evaluate_scenario(
            model=model,
            vecnorm_path=vecnorm_path if vecnorm_path.exists() else None,
            scenario=scenario,
            n_seeds=args.n_seeds,
            first_seed=args.first_seed,
            observation_mode=args.observation_mode,
            action_mode=args.action_mode,
            reward_mode=args.reward_mode,
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

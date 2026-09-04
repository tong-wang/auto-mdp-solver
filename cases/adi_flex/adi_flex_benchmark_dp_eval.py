"""Evaluate the exact DP policy on the ADI-flex simulator.

Loads the y*(i, u, vhat) table produced by adi_flex_benchmark_dp.py and plays it through
the gym wrapper with the spec §9 seed protocol (seeds 0..n_seeds-1), writing
the shared eval TSV (reward = -cost). The simulated mean is also compared to
the DP's expected value C*_1(0,0) — a direct validation that the reduced-state
DP accounting matches the physical simulator.

Example usage:
    python adi_flex_benchmark_dp_eval.py --dp-solutions results/homog_L0_T2/dp/homog_L0_T2_policy.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adi_flex_benchmark_common import ObsView, EVAL_HEADER, eval_row
from adi_flex_gym import AdiFlexEnv
from adi_flex_scenarios import SCENARIOS, AdiFlexScenario


def evaluate_scenario(
    ystar: np.ndarray, u_min: int, d_max: int,
    scenario: AdiFlexScenario, n_seeds: int,
) -> dict:
    """Play the DP table for seeds 0..n_seeds-1; return reward stats.

    Homogeneous branch only (alloc_enabled=False), so the env is single-phase:
    every step is an ORDER step and vhat = d[2] in the observation is last
    period's far-class draw — exactly the reduced-state coordinate.
    """
    env = AdiFlexEnv(scenario=scenario, observation_mode="vec")
    assert not env.alloc_enabled, "the exact DP arm plays the homogeneous branch"
    rewards = np.zeros(n_seeds)

    for ep_seed in range(n_seeds):
        if ep_seed % 10000 == 0:
            print(f"  seed {ep_seed}/{n_seeds}", flush=True)
        obs, _ = env.reset(seed=ep_seed)
        terminated = False
        while not terminated:
            v = ObsView(obs, env.pipe_slots, scenario.N, scenario.T_dl,
                        alloc_enabled=env.alloc_enabled)
            ui = int(np.clip(v.u - u_min, 0, ystar.shape[1] - 1))
            vi = int(np.clip(v.vhat, 0, d_max))
            order = int(np.clip(ystar[v.period, ui, vi] - v.u, 0, scenario.order_max))
            obs, _, terminated, _, _ = env.step(order)
        rewards[ep_seed] = env.total_reward

    env.close()
    mu = rewards.mean()
    return {
        "reward_mean": float(mu),
        "reward_var":  float(rewards.var(ddof=1)),
        "semivar_d":   float(np.sum(np.maximum(mu - rewards, 0.0) ** 2) / (n_seeds - 1)),
        "semivar_u":   float(np.sum(np.maximum(rewards - mu, 0.0) ** 2) / (n_seeds - 1)),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the DP policy on the ADI-flex simulator.")
    p.add_argument("--dp-solutions", type=str, required=True,
                   help="Path to the policy .npz from adi_flex_benchmark_dp.py")
    p.add_argument("-s", "--scenario_name", type=str, default="homog_L0_T2",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--n-seeds", type=int, default=8192,
                   help="Number of episode seeds (seeds 0..n-1)")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV output (defaults to results/<scenario>/benchmark/"
                        "benchmark_dp_eval_<scenario>.tsv)")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    sol = np.load(args.dp_solutions)
    ystar, u_min, d_max = sol["ystar"], int(sol["u_min"]), int(sol["d_max"])
    assert ystar.shape[0] == scenario.N, \
        f"policy horizon {ystar.shape[0]} != scenario N {scenario.N}"

    outfile = (
        Path(args.outfile) if args.outfile
        # the filename must carry the `benchmark_{method}_eval` marker: it is
        # what lets `mdp_gates --ir` find this arm's declared §9.9 role. Named
        # anything else, the role is silently inert and an `exact` arm can be
        # passed as a must-beat baseline with no complaint
        else Path(__file__).resolve().parent / "results" / args.scenario_name / "benchmark"
        / f"benchmark_dp_eval_{args.scenario_name}.tsv"
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)

    header = EVAL_HEADER
    print(header)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()
        stats = evaluate_scenario(ystar, u_min, d_max, scenario, args.n_seeds)
        row = (
            eval_row(scenario, stats)
        )
        print(row, flush=True)
        f.write(row + "\n")

    se = np.sqrt(stats["reward_var"] / args.n_seeds)
    print(f"\nsimulated cost = {-stats['reward_mean']:.4f} +- {se:.4f} (SE)")
    print(f"results saved -> {outfile}")


if __name__ == "__main__":
    main()

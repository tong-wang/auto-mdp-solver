"""DP benchmark evaluation for single-echelon inventory.

Evaluates the FiniteHorizonDP policy through InvSingleEnv (same gym
used by RL) for a fair comparison. Writes TensorBoard logs to
  results/<scenario>/benchmark/DP_YYYYMMDD_HHMMSS/
using the same key names and pseudo-timestep x-axis as SB3 PPO runs.

Usage:
    python inv_single_dp_test.py
    python inv_single_dp_test.py --scenario simple --n-episodes 1000
"""
from __future__ import annotations

import argparse
import warnings
from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from inv_single_dp_solve import FiniteHorizonDP
from inv_single_gym import InvSingleEnv
from inv_single_scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate DP policy on inventory scenarios.")
    p.add_argument("--scenario",   type=str, default="simple",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--n-episodes", type=int, default=1000)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--outdir",     type=str, default="results")
    p.add_argument("--no-cache",   action="store_true",
                   help="Force re-solve and overwrite cached V/pi tables.")
    return p


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: InvSingleEnv,
    dp: FiniteHorizonDP,
    episode_seed: int,
) -> dict:
    LT = env.scenario.leadtime.max()
    _, _ = env.reset(seed=episode_seed)

    ep_reward     = 0.0
    cost_holding  = 0.0
    cost_shortage = 0.0
    cost_ordering = 0.0

    terminated = truncated = False
    while not (terminated or truncated):
        # Read raw state for DP — env._state is always the pre-order state
        t = env._state.period
        x = (env._state.inventory if LT == 0
             else env._state.inventory + sum(env._state.pipeline))

        action = np.array([float(dp.act(t, x))], dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(action)

        ep_reward     += reward
        cost_holding  += info["cost"]["holding"]
        cost_shortage += info["cost"]["shortage"]
        cost_ordering += info["cost"]["order_fixed"] + info["cost"]["order_variable"]

    return {
        "ep_reward":     ep_reward,
        "cost_total":   -ep_reward,
        "cost_holding":  cost_holding,
        "cost_shortage": cost_shortage,
        "cost_ordering": cost_ordering,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    scenario = SCENARIOS[args.scenario]
    dp = FiniteHorizonDP(scenario, results_dir=args.outdir, no_cache=args.no_cache)

    # Suppress the continuous-action-with-discrete-demand warning:
    # DP returns integer orders; continuous action mode avoids the discrete
    # action space upper-bound constraint.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = InvSingleEnv(
            scenario=scenario,
            observation_mode="vec_d",
            action_mode="continuous",
        )

    writer = SummaryWriter(log_dir=str(dp._cache_dir))

    rng           = np.random.default_rng(args.seed)
    ep_rew_buffer = deque(maxlen=100)   # mirrors SB3's ep_info_buffer
    all_rewards   = []
    all_costs     = {"total": [], "holding": [], "shortage": [], "ordering": []}

    for ep in range(args.n_episodes):
        episode_seed = int(rng.integers(0, 2_147_483_647))
        result = run_episode(env, dp, episode_seed)

        ep_rew_buffer.append(result["ep_reward"])
        all_rewards.append(result["ep_reward"])
        all_costs["total"].append(result["cost_total"])
        all_costs["holding"].append(result["cost_holding"])
        all_costs["shortage"].append(result["cost_shortage"])
        all_costs["ordering"].append(result["cost_ordering"])

        # Pseudo-timestep: episodes * horizon — comparable to RL x-axis
        global_step = (ep + 1) * scenario.horizon

        writer.add_scalar("rollout/ep_rew_mean",  np.mean(ep_rew_buffer),  global_step)
        writer.add_scalar("rollout/ep_len_mean",  float(scenario.horizon), global_step)
        writer.add_scalar("eval/ep_rew",          result["ep_reward"],     global_step)
        writer.add_scalar("eval/cost_total",      result["cost_total"],    global_step)
        writer.add_scalar("eval/cost_holding",    result["cost_holding"],  global_step)
        writer.add_scalar("eval/cost_shortage",   result["cost_shortage"], global_step)
        writer.add_scalar("eval/cost_ordering",   result["cost_ordering"], global_step)

    writer.close()
    env.close()

    print(f"\n{'─' * 52}")
    print(f"  scenario    : {args.scenario}")
    print(f"  episodes    : {args.n_episodes}")
    print(f"  ep_reward   : {np.mean(all_rewards):8.4f}  ±  {np.std(all_rewards):.4f}")
    print(f"  cost_total  : {np.mean(all_costs['total']):8.4f}  ±  {np.std(all_costs['total']):.4f}")
    print(f"    holding   : {np.mean(all_costs['holding']):8.4f}")
    print(f"    shortage  : {np.mean(all_costs['shortage']):8.4f}")
    print(f"    ordering  : {np.mean(all_costs['ordering']):8.4f}")
    print(f"  TB logs     : {dp._cache_dir}")
    print(f"{'─' * 52}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

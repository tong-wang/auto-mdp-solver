"""Shared benchmark-eval harness for the mab domain (spec §9).

Every benchmark eval (`mab_benchmark_{method}_eval.py`) runs the same seed
protocol — episode seeds 0..n_seeds-1 on the named scenario — through the
same MabEnv used by RL, and writes one spec-§9 TSV row to

    results/{scenario}/benchmark/benchmark_{method}_eval_{scenario}.tsv

Columns: scenario  K  T  family  reward_mean  reward_var  semivar_d
semivar_u  oracle_mean  regret_mean — reward_mean is the gate metric
(sense=maximize); oracle_mean is the clairvoyant always-pull-the-best-arm
total (T * opt_mean, same seeds), so regret_mean = oracle_mean - reward_mean.

Policies see only observable quantities (pulls, payouts, period, family
constants) — never the hidden arm means.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mab_gym import MabEnv
from mab_scenarios import SCENARIOS

# policy streams (kept away from the model streams and the interpreter's 9999)
POLICY_STREAM_RANDOM   = 9996
POLICY_STREAM_THOMPSON = 9997


def build_arg_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("-s", "--scenario", type=str, default="gauss_K10_T1000",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--n-seeds", type=int, default=8192,
                   help="episode seeds 0..n-1 (shared across all evals)")
    p.add_argument("--outdir", type=str, default="results")
    return p


def reward_stats(rewards: np.ndarray) -> dict:
    n = rewards.size
    mu = rewards.mean()
    return {
        "reward_mean": float(mu),
        "reward_var":  float(rewards.var(ddof=1)),
        "semivar_d":   float(np.sum(np.maximum(mu - rewards, 0.0) ** 2) / (n - 1)),
        "semivar_u":   float(np.sum(np.maximum(rewards - mu, 0.0) ** 2) / (n - 1)),
    }


def evaluate(method: str, make_policy, args: argparse.Namespace) -> dict:
    """Run the seed loop for one policy factory and write the spec-§9 TSV.

    ``make_policy(env)`` is called after each reset; the env exposes
    ``n_arms``, ``horizon``, ``_is_gauss``, ``_episode_seed`` and the
    per-episode scenario's public constants (sigma, seed_salt).
    """
    source = SCENARIOS[args.scenario]
    env = MabEnv(scenario=source, observation_mode="stats")

    rewards = np.zeros(args.n_seeds)
    oracles = np.zeros(args.n_seeds)
    for seed in range(args.n_seeds):
        if seed % 1000 == 0:
            print(f"  seed {seed}/{args.n_seeds}", flush=True)
        env.reset(seed=seed)
        policy = make_policy(env)
        done = False
        total = 0.0
        info: dict = {}
        while not done:
            arm = policy.act(env._state.pulls, env._state.payouts,
                             env._state.period)
            _, r, done, _, info = env.step(arm)
            total += r
        rewards[seed] = total
        oracles[seed] = env.horizon * info["opt_mean"]
    env.close()

    stats = reward_stats(rewards)
    stats["oracle_mean"] = float(oracles.mean())
    stats["regret_mean"] = stats["oracle_mean"] - stats["reward_mean"]

    # anchored to this file, never the CWD (spec §8.4)
    out_dir = (Path(__file__).resolve().parent / args.outdir
               / args.scenario / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"benchmark_{method}_eval_{args.scenario}.tsv"
    family = "gaussian" if env._is_gauss else "bernoulli"
    header = ("scenario\tK\tT\tfamily\treward_mean\treward_var\t"
              "semivar_d\tsemivar_u\toracle_mean\tregret_mean")
    row = (f"{args.scenario}\t{env.n_arms}\t{env.horizon}\t{family}\t"
           f"{stats['reward_mean']:.6f}\t{stats['reward_var']:.6f}\t"
           f"{stats['semivar_d']:.6f}\t{stats['semivar_u']:.6f}\t"
           f"{stats['oracle_mean']:.6f}\t{stats['regret_mean']:.6f}")
    out.write_text(f"# {method} | n_seeds={args.n_seeds}\n{header}\n{row}\n")

    print(f"\n{'─' * 56}")
    print(f"  method      : {method}")
    print(f"  scenario    : {args.scenario}  (n_seeds={args.n_seeds})")
    print(f"  reward_mean : {stats['reward_mean']:10.4f}  "
          f"± {np.sqrt(stats['reward_var'] / args.n_seeds):.4f} (SE)")
    print(f"  oracle_mean : {stats['oracle_mean']:10.4f}")
    print(f"  regret_mean : {stats['regret_mean']:10.4f}")
    print(f"  TSV         : {out}")
    print(f"{'─' * 56}\n")
    return stats

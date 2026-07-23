"""Evaluate a trained AdiFlex PPO model on the spec-section-9 seed protocol.

Scores the model on episode seeds 0..n-1 — the same seeds, same simulator and
same TSV columns as every benchmark eval — so `mdp_gates` can compare them with
an honest standard error.

VecNormalize handling follows spec section 9.5: obs normalization is part of
the policy's input contract, so the saved obs_rms must be reused here or the
policy sees inputs it never trained on. The episode seed is injected through
`env_method("reset", seed=...)` to bypass VecNormalize's own reset, and the
returned raw observation is normalized by hand.

Also supports `--force-hold-back`, which overrides the policy's allocation with
a fixed number while leaving its ordering intact. That is the ablation for the
question this domain exists to ask: the paper's heuristics all use a *constant*
protection level, so pinning the learned policy to the same constant isolates
how much of any win comes from learning the allocation rather than the ordering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from adi_flex_gym import AdiFlexEnv
from adi_flex_scenarios import SCENARIOS
from adi_flex_benchmark_common import TSV_HEADER, format_row


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained AdiFlex PPO model.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None)
    p.add_argument("-s", "--scenario_name", type=str, default="exp4")
    p.add_argument("-o", "--observation_mode", type=str, default="vec")
    p.add_argument("-a", "--action_mode", type=str, default="joint")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--force-hold-back", type=int, default=None,
                   help="override the policy's allocation with a constant (ablation)")
    p.add_argument("--outfile", type=str, default=None)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> int:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    model_path = Path(args.model_path)
    model = PPO.load(model_path, device="cpu")

    def make_env():
        return AdiFlexEnv(
            scenario=scenario,
            action_mode=args.action_mode,
            observation_mode=args.observation_mode,
        )

    venv = DummyVecEnv([make_env])
    vecnorm_path = Path(args.vecnorm_path) if args.vecnorm_path else (
        model_path.parent / "vecnormalize.pkl"
    )
    if vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
        print(f"loaded VecNormalize stats from {vecnorm_path}")
    else:
        print(f"no VecNormalize stats at {vecnorm_path}; evaluating unnormalized")

    normalizing = isinstance(venv, VecNormalize)
    costs = np.zeros(args.n_seeds)

    for ep_seed in range(args.n_seeds):
        if ep_seed % 2048 == 0:
            print(f"  seed {ep_seed}/{args.n_seeds}", flush=True)
        raw_obs, _ = venv.env_method("reset", seed=ep_seed)[0]
        obs = (venv.normalize_obs(raw_obs[np.newaxis, :]) if normalizing
               else raw_obs[np.newaxis, :])
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if args.force_hold_back is not None:
                action = np.array(action)
                action[..., 1] = args.force_hold_back
            obs, _, done_arr, infos = venv.step(action)
            total += float(infos[0]["cost"]["total"])
            done = bool(done_arr[0])
        costs[ep_seed] = total

    rewards = -costs
    mu = float(rewards.mean())
    n = len(rewards)
    stats = {
        "reward_mean": mu,
        "reward_var":  float(rewards.var(ddof=1)),
        "semivar_d":   float(np.sum(np.maximum(mu - rewards, 0.0) ** 2) / (n - 1)),
        "semivar_u":   float(np.sum(np.maximum(rewards - mu, 0.0) ** 2) / (n - 1)),
        "cost_mean":   float(costs.mean()),
    }
    print(f"cost_mean = {stats['cost_mean']:.4f}   reward_mean = {mu:.4f}")

    suffix = "" if args.force_hold_back is None else f"_hb{args.force_hold_back}"
    out = Path(args.outfile) if args.outfile else (
        model_path.parent / f"ppo_eval_{args.scenario_name}{suffix}.tsv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(TSV_HEADER + "\n")
        row = format_row(scenario, stats)
        print(row, flush=True)
        f.write(row + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

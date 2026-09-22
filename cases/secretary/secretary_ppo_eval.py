"""Protocol/selection evaluation of a trained secretary PPO policy (spec §9)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mdp_gates.completion import provenance_lines
from secretary_gym import SecretaryEnv
from secretary_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate secretary PPO.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vecnorm-path", default=None)
    parser.add_argument("--outfile", default=None)
    parser.add_argument("-s", "--scenario_name", default="standard", choices=SCENARIOS)
    parser.add_argument("-o", "--observation_mode", default="relative", choices=["relative"])
    parser.add_argument("-a", "--action_mode", default="accept", choices=["accept"])
    parser.add_argument("--reward_mode", default="success", choices=["success"])
    parser.add_argument("--n-seeds", type=int, default=8192)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--batch-envs", type=int, default=64)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-vecnorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail rather than silently score normalized training without its sidecar.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def _normalizer(path: Path | None, scenario, args) -> VecNormalize | None:
    if path is None or not path.exists():
        return None
    dummy = DummyVecEnv([lambda: SecretaryEnv(
        scenario=scenario,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        reward_mode=args.reward_mode,
    )])
    normalizer = VecNormalize.load(path, dummy)
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def _rollout(model: PPO, normalizer: VecNormalize | None, scenario, args):
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    success = np.empty(args.n_seeds, dtype=np.float64)
    ranks = np.empty(args.n_seeds, dtype=np.float64)
    for start in range(0, args.n_seeds, args.batch_envs):
        block = seeds[start:start + args.batch_envs]
        envs = [SecretaryEnv(
            scenario=scenario,
            observation_mode=args.observation_mode,
            action_mode=args.action_mode,
            reward_mode=args.reward_mode,
        ) for _ in block]
        observations = []
        for env, seed in zip(envs, block):
            obs, _ = env.reset(seed=seed)
            observations.append(obs)
        active = np.ones(len(envs), dtype=bool)
        while active.any():
            obs_batch = np.asarray(observations, dtype=np.float32)
            model_obs = normalizer.normalize_obs(obs_batch.copy()) if normalizer else obs_batch
            actions, _ = model.predict(model_obs, deterministic=args.deterministic)
            for local, env in enumerate(envs):
                if not active[local]:
                    continue
                obs, _, terminated, truncated, info = env.step(int(actions[local]))
                observations[local] = obs
                if terminated or truncated:
                    active[local] = False
                    success[start + local] = float(info["outcome"]["success"])
                    ranks[start + local] = float(info["selected_rank"])
        for env in envs:
            env.close()
        if start % 1024 == 0:
            print(f"  seed {start}/{args.n_seeds}", flush=True)
    return seeds, success, ranks


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    run_dir = model_path.parent.parent if model_path.parent.name == "checkpoints" else model_path.parent
    vecnorm_path = (
        Path(args.vecnorm_path).resolve()
        if args.vecnorm_path
        else model_path.parent / "vecnormalize.pkl"
    )
    if not vecnorm_path.exists() and model_path.parent.name == "checkpoints":
        stem = model_path.stem
        step = stem.rsplit("_", 2)[-2] if "_steps" in stem else ""
        matches = sorted(model_path.parent.glob(f"*vecnormalize*{step}*steps.pkl"))
        if matches:
            vecnorm_path = matches[0]
    if args.require_vecnorm and not vecnorm_path.exists():
        raise SystemExit(
            f"VecNormalize sidecar not found for {model_path}; pass "
            "--vecnorm-path or explicitly use --no-require-vecnorm only for "
            "a run trained without normalization"
        )
    scenario = SCENARIOS[args.scenario_name]
    model = PPO.load(model_path, device="cpu")
    normalizer = _normalizer(vecnorm_path if vecnorm_path.exists() else None, scenario, args)
    seeds, successes, ranks = _rollout(model, normalizer, scenario, args)
    if normalizer:
        normalizer.close()
    mean = float(successes.mean())
    variance = float(successes.var(ddof=1)) if args.n_seeds > 1 else 0.0
    semivar_d = float(np.square(np.maximum(mean - successes, 0)).sum() / max(1, args.n_seeds - 1))
    semivar_u = float(np.square(np.maximum(successes - mean, 0)).sum() / max(1, args.n_seeds - 1))
    rank_mean = float(ranks.mean())
    rank_var = float(ranks.var(ddof=1)) if args.n_seeds > 1 else 0.0
    outfile = Path(args.outfile).resolve() if args.outfile else model_path.parent / f"ppo_eval_{args.scenario_name}.tsv"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    header = ("scenario\tn_candidates\tn_seeds\tsuccess_mean\tsuccess_var\t"
              "semivar_d\tsemivar_u\texpected_selected_rank_mean\t"
              "expected_selected_rank_var")
    row = (f"{args.scenario_name}\t{scenario.n_candidates}\t{args.n_seeds}\t"
           f"{mean:.10f}\t{variance:.10f}\t{semivar_d:.10f}\t{semivar_u:.10f}\t"
           f"{rank_mean:.10f}\t{rank_var:.10f}")
    provenance = provenance_lines(model_path, run_dir=run_dir)
    outfile.write_text("\n".join(provenance + [
        f"# model: {model_path}",
        f"# seed_block: {args.first_seed}..{args.first_seed + args.n_seeds - 1}",
        header,
        row,
        "",
    ]))
    sidecar = outfile.with_suffix(".seeds.tsv")
    with sidecar.open("w") as stream:
        stream.write("seed\tsuccess\texpected_selected_rank\n")
        for seed, success, rank in zip(seeds, successes, ranks):
            stream.write(f"{seed}\t{success:.0f}\t{rank:.0f}\n")
    print(f"success={mean:.10f} ± {np.sqrt(variance / args.n_seeds):.10f} SE; "
          f"selected_rank={rank_mean:.6f}")
    print(f"results saved -> {outfile}")


if __name__ == "__main__":
    main()

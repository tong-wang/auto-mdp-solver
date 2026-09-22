"""Train SB3 PPO on the standard secretary scenario (spec §8)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mdp_conformance.launch import assert_l1_current
from mdp_ir.schema import load_ir
from secretary_gym import SecretaryEnv
from secretary_scenarios import SCENARIOS


_L1_DERIVED: dict[str, object] = {
    "learning_rate": 1e-4,
    "lr_final": 1e-5,
    "clip_init": 0.2,
    "clip_final": 0.05,
    "n_envs": 4,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 10,
    "target_kl": 0.02,
    "gamma": 1.0,
    "gae_lambda": 0.99,
    "ent_coef": 0.01,
    "net_arch": [64, 64],
    "norm_obs": True,
    "norm_reward": True,
    "normalize_advantage": True,
}
_L1_BASIS: dict[str, object] = {
    "scenario_name": "standard",
    "episode_len": 100,
    "beta": 1.0,
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on secretary.")
    parser.add_argument("-s", "--scenario_name", default="standard", choices=SCENARIOS)
    parser.add_argument("-o", "--observation_mode", default="relative", choices=["relative"])
    parser.add_argument("-a", "--action_mode", default="accept", choices=["accept"])
    parser.add_argument("--reward_mode", default="success", choices=["success"])
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tag", default="")
    parser.add_argument("--level", default="L1", choices=["L0", "L1"])
    parser.add_argument("--gym-log", type=int, default=0, choices=[0, 1])
    parser.add_argument("--checkpoint-every-frac", type=float, default=0.05)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--norm-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    parser.add_argument("--learning-rate", "--lr-init", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=1e-5)
    parser.add_argument("--clip-init", type=float, default=0.2)
    parser.add_argument("--clip-final", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--net_arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--policy", choices=["auto", "mlp", "cnn"], default="auto")
    return parser


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    return lambda remaining: end + (start - end) * float(remaining)


def _resolve_policy(name: str, observation_shape: tuple[int, ...]) -> str:
    if name == "auto":
        return "CnnPolicy" if len(observation_shape) >= 3 else "MlpPolicy"
    return "MlpPolicy" if name == "mlp" else "CnnPolicy"


def _run_name(args: argparse.Namespace) -> str:
    parts = [args.level, f"obs{args.observation_mode}", f"act{args.action_mode}",
             f"rew{args.reward_mode}"]
    defaults = dict(_L1_DERIVED)
    for key, expected in defaults.items():
        actual = getattr(args, key)
        if actual != expected:
            rendered = "-".join(map(str, actual)) if isinstance(actual, list) else f"{actual:.3g}" if isinstance(actual, float) else str(actual)
            parts.append(f"{key}{rendered}")
    if args.total_timesteps != 2_000_000:
        parts.append(f"steps{args.total_timesteps}")
    if args.seed != 1:
        parts.append(f"seed{args.seed}")
    if args.tag:
        parts.append(args.tag)
    return f"PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{'_'.join(parts)}"


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    if args.level != "L0":
        assert_l1_current(
            args,
            derived=_L1_DERIVED,
            basis=_L1_BASIS,
            episode_len=scenario.n_candidates,
        )
    domain_dir = Path(__file__).resolve().parent
    run_dir = domain_dir / args.outdir / args.scenario_name / _run_name(args)
    run_dir.mkdir(parents=True, exist_ok=False)
    log_stream = (run_dir / "train.log").open("w")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, log_stream)
    sys.stderr = _Tee(original_stderr, log_stream)
    try:
        print(f"outdir={run_dir}")
        print(scenario)

        def make_env(rank: int):
            def _make():
                env = SecretaryEnv(
                    scenario=scenario,
                    observation_mode=args.observation_mode,
                    action_mode=args.action_mode,
                    reward_mode=args.reward_mode,
                )
                return Monitor(env, filename=str(run_dir / f"monitor_{rank}"))
            return _make

        venv = DummyVecEnv([make_env(rank) for rank in range(args.n_envs)])
        venv = VecNormalize(
            venv,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_reward,
            clip_obs=args.vecnorm_clip_obs,
            gamma=args.gamma,
        )
        policy = _resolve_policy(args.policy, venv.observation_space.shape)
        print(f"policy={policy} observation_shape={venv.observation_space.shape}")
        checkpoint = None
        if args.checkpoint_every_frac > 0:
            save_freq = max(
                1,
                int(args.checkpoint_every_frac * args.total_timesteps / args.n_envs),
            )
            checkpoint = CheckpointCallback(
                save_freq=save_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix=f"{args.scenario_name}_ppo",
                save_vecnormalize=True,
            )
        model = PPO(
            policy,
            venv,
            learning_rate=linear_schedule(args.learning_rate, args.lr_final),
            clip_range=linear_schedule(args.clip_init, args.clip_final),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            normalize_advantage=args.normalize_advantage,
            target_kl=None if args.target_kl <= 0 else args.target_kl,
            policy_kwargs={"net_arch": list(args.net_arch)},
            stats_window_size=500,
            tensorboard_log=str(run_dir),
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint,
            progress_bar=args.progress_bar,
        )
        # The terminal artifact is never silently presented as the selected
        # policy; post-hoc checkpoint selection later promotes its winner to
        # ``{scenario}_ppo.zip`` while this file remains the audit endpoint.
        model_path = run_dir / f"{args.scenario_name}_ppo_final"
        model.save(model_path)
        venv.save(run_dir / "vecnormalize_final.pkl")
        ir = load_ir(domain_dir / "secretary_schema.json")
        args_log = vars(args) | {
            "algo_class": type(model).__name__,
            "sb3_version": stable_baselines3.__version__,
            "ir_mdp_fingerprint": ir.mdp_fingerprint(),
            "resolved_policy": policy,
        }
        (run_dir / f"{args.scenario_name}_ppo_args.txt").write_text(
            "".join(f"{key}: {value}\n" for key, value in sorted(args_log.items()))
        )
        print(f"model={model_path}.zip steps={model.num_timesteps}")
        venv.close()
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_stream.close()


if __name__ == "__main__":
    main()

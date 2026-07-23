"""SB3 PPO training for the AdiFlex domain.

Trains a joint ordering + allocation policy on one registered instance
(specialist strategy, confirmed at Stage 0). The thing to watch while it runs
is `rollout/ep_rew_mean` against the benchmark bounds: the AP relaxation gives
an unreachable ceiling on reward (-336.19 at exp4) and PL(sigma) the real
target to beat (-341.20).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from adi_flex_gym import AdiFlexEnv
from adi_flex_scenarios import SCENARIOS

ALGO = "ppo"

_SHORT_KEYS: dict[str, str] = {
    "total_timesteps": "steps",
    "learning_rate":   "lr",
    "batch_size":      "bs",
    "n_epochs":        "ep",
    "ent_coef":        "ent",
    "vecnorm_clip_obs": "clipobs",
    "norm_reward":     "normrew",
    "net_arch":        "arch",
    "n_envs":          "envs",
}
_SKIP_KEYS = {
    "outdir", "scenario_name", "progress_bar",
    "observation_mode", "action_mode", "reward_mode",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on AdiFlex.")
    p.add_argument("-s", "--scenario_name", default="exp4",
                   type=str, choices=list(SCENARIOS.keys()))
    p.add_argument("-a", "--action_mode", default="joint", type=str)
    p.add_argument("-o", "--observation_mode", default="vec", type=str)
    p.add_argument("-r", "--reward_mode", default="neg_cost", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--n_envs", default=8, type=int,
                   help="parallel envs inside DummyVecEnv; batches the policy forward pass")

    p.add_argument("--total_timesteps", default=1_000_000, type=int)
    # `learning_rate` is the dest mdp_tuning reaches for
    p.add_argument("--learning_rate", default=3e-4, type=float)
    p.add_argument("--n_steps", default=2048, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--n_epochs", default=10, type=int)
    # alpha = 1 in every experiment the paper reports, and the horizon is a
    # finite 12 periods, so undiscounted is the faithful choice
    p.add_argument("--gamma", default=1.0, type=float)
    p.add_argument("--gae_lambda", default=0.95, type=float)
    p.add_argument("--ent_coef", default=0.0, type=float)
    # nargs list, not a delimited string: mdp_tuning infers list-valued args
    # from action.nargs and passes them as separate tokens (--net_arch 64 64)
    p.add_argument("--net_arch", default=[64, 64], type=int, nargs="+")

    p.add_argument("--vecnorm_clip_obs", default=10.0, type=float)
    p.add_argument("--no_norm_reward", action="store_false",
                   dest="norm_reward", default=True)

    p.add_argument("--outdir", default=None, type=str)
    p.add_argument("--progress_bar", action="store_true")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def build_run_name(args: argparse.Namespace) -> str:
    defaults = vars(_build_arg_parser().parse_args([]))
    parts = [
        f"obs{args.observation_mode}",
        f"act{args.action_mode}",
        f"rew{args.reward_mode}",
    ]
    for key, default_val in defaults.items():
        if key in _SKIP_KEYS:
            continue
        val = getattr(args, key)
        if val != default_val:
            label = _SHORT_KEYS.get(key, key)
            if isinstance(val, float):
                formatted = f"{val:.3g}"
            elif isinstance(val, (list, tuple)):
                formatted = "-".join(str(v) for v in val)
            else:
                formatted = str(val)
            parts.append(f"{label}{formatted}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ALGO.upper()}_{timestamp}_{'_'.join(parts)}"


def main() -> None:
    args = parse_args()

    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    run_name = build_run_name(args)
    base = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent
    outdir = base / "results" / args.scenario_name / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"outdir: {outdir}")

    with open(outdir / f"{args.scenario_name}_{ALGO}_args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}\t{v}\n")

    def make_env(rank: int):
        def _init():
            e = AdiFlexEnv(
                scenario=scenario,
                action_mode=args.action_mode,
                observation_mode=args.observation_mode,
                # only rank 0 writes the per-step trace; N copies of it would
                # be N times the I/O for no extra information
                logger_filename=str(outdir / "train_log") if rank == 0 else None,
            )
            # Monitor sits INSIDE VecNormalize so ep_rew_mean stays on the raw
            # cost scale and stays directly comparable with the benchmark TSVs
            return Monitor(e, filename=str(outdir / f"monitor{rank}"))
        return _init

    # Several envs in one DummyVecEnv (still single-process) so PPO's policy
    # forward pass runs at batch size n_envs instead of 1. Episodes here are
    # only 12 steps, so per-call torch overhead — not the simulator — sets the
    # throughput: measured ~220 fps at n_envs=1 against thousands raw.
    env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=args.norm_reward,
        clip_obs=args.vecnorm_clip_obs,
    )

    net_arch = list(args.net_arch)
    model = PPO(
        "MlpPolicy",                       # flat 3-4 element observation
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": net_arch},
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(outdir),
    )
    model.learn(total_timesteps=args.total_timesteps,
                progress_bar=args.progress_bar)

    model.save(outdir / f"{args.scenario_name}_{ALGO}.zip")
    env.save(str(outdir / "vecnormalize.pkl"))
    env.close()
    print(f"saved model to {outdir / f'{args.scenario_name}_{ALGO}.zip'}")


if __name__ == "__main__":
    main()

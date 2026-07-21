"""SB3 PPO training script for the dynamic pricing problem.

Example usage:
    python dynamic_pricing_ppo_train.py
    python dynamic_pricing_ppo_train.py -s simple -a intensity --total-timesteps 500000
"""

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dynamic_pricing_gym import DynamicPricingEnv
from dynamic_pricing_scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on the dynamic pricing problem.")
    p.add_argument("-s", "--scenario_name",    type=str, default="simple",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", type=str, default="vec",
                   choices=["vec", "vec_d"])
    p.add_argument("-a", "--action_mode",      type=str, default="price",
                   choices=["price", "intensity"])
    p.add_argument("-r", "--reward_mode",      type=str, default="revenue",
                   choices=["revenue"])
    p.add_argument("--total-timesteps",  type=int,   default=500_000)
    p.add_argument("--seed",             type=int,   default=1)
    p.add_argument("--outdir",           type=str,   default="results")
    p.add_argument("--progress-bar",     action="store_true")
    # PPO hyperparameters
    p.add_argument("--learning_rate", "-l", type=float, default=3e-4)
    p.add_argument("--n-steps",          type=int,   default=2048)
    p.add_argument("--batch-size",       type=int,   default=512)
    p.add_argument("--n-epochs",         type=int,   default=10)
    p.add_argument("--gamma",            type=float, default=1.0)
    p.add_argument("--ent-coef",         type=float, default=0.0)
    p.add_argument("--net-arch",         type=int,   nargs="+", default=[64, 64])
    # VecNormalize
    p.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    p.add_argument("--no-norm-reward",   action="store_false", dest="norm_reward", default=True)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHORT_KEYS: dict[str, str] = {
    "total_timesteps":  "steps",
    "seed":             "seed",
    "learning_rate":    "lr",
    "n_steps":          "nsteps",
    "batch_size":       "bs",
    "n_epochs":         "ep",
    "gamma":            "gamma",
    "ent_coef":         "ent",
    "net_arch":         "arch",
    "vecnorm_clip_obs": "clipobs",
    "norm_reward":      "normrew",
}
_SKIP_KEYS = {"outdir", "scenario_name", "progress_bar",
              "observation_mode", "action_mode", "reward_mode"}


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
            elif isinstance(val, list):
                formatted = "-".join(str(v) for v in val)
            else:
                formatted = str(val)
            parts.append(f"{label}{formatted}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"PPO_{timestamp}_{'_'.join(parts)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    run_name = build_run_name(args)
    outdir   = (Path(__file__).resolve().parent / args.outdir / args.scenario_name / run_name).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / f"{args.scenario_name}_ppo_args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    print(f"observation_mode={args.observation_mode}  action_mode={args.action_mode}  "
          f"reward_mode={args.reward_mode}  outdir={outdir}")

    env = DynamicPricingEnv(
        scenario=scenario,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        reward_mode=args.reward_mode,
        logger_filename=str(outdir / "train_log"),
    )
    env = Monitor(env, filename=str(outdir / "monitor"))
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=args.norm_reward,
                       clip_obs=args.vecnorm_clip_obs)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": list(args.net_arch)},
        verbose=1,
        tensorboard_log=str(outdir),
        seed=args.seed,
        device="cpu",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=args.progress_bar,
    )

    model_path   = outdir / f"{args.scenario_name}_ppo.zip"
    vecnorm_path = outdir / "vecnormalize.pkl"
    model.save(str(model_path))
    env.save(str(vecnorm_path))
    print(f"saved model        → {model_path}")
    print(f"saved vecnormalize → {vecnorm_path}")
    env.close()


if __name__ == "__main__":
    main()

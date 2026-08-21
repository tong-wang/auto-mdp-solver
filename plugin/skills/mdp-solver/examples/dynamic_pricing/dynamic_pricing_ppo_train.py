"""SB3 PPO training script for the dynamic pricing problem.

Example usage:
    python dynamic_pricing_ppo_train.py
    python dynamic_pricing_ppo_train.py -s simple -a intensity --total-timesteps 500000
"""

import argparse
from datetime import datetime
from pathlib import Path

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
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
    p.add_argument("--tag",              type=str,   default="",
                   help="escalation-log entry this run was launched under "
                        "(spec §8.2); lands verbatim in the args log")
    p.add_argument("--checkpoint-every-frac", type=float, default=0.05,
                   help="checkpoint every this fraction of the budget (spec "
                        "§9.7 wants ~20 for the post-hoc screen); 0 disables")
    p.add_argument("--progress-bar",     action="store_true")
    p.add_argument("--gym-log",          type=int,   default=0, choices=[0, 1],
                   help="Write per-episode/per-step gym logs (spec §11). Off by "
                        "default: the step stream costs hundreds of MB per run.")
    # PPO hyperparameters
    p.add_argument("--learning_rate", "-l", type=float, default=3e-4)
    p.add_argument("--n-steps",          type=int,   default=2048)
    p.add_argument("--batch-size",       type=int,   default=512)
    p.add_argument("--n-epochs",         type=int,   default=10)
    p.add_argument("--gamma",            type=float, default=1.0)
    # L1 (§8.6): the price/inventory consequence realizes within the episode's
    # remaining horizon, so credit spans it — 0.95 gives ~20 steps, re-derive if
    # T̄ moves.
    p.add_argument("--gae-lambda",       type=float, default=0.95)
    p.add_argument("--ent-coef",         type=float, default=0.0)
    # L1 (§8.6): on. PPO's per-minibatch advantage rescaling — the premise
    # §8.3 reasons FROM when it calls reward norm a critic-scaling detail, so
    # it must be checkable rather than inherited silently from SB3. Off only as
    # a logged L2 move; `mdp_tuning`'s breadth tier opens it.
    p.add_argument("--no-normalize-advantage", action="store_false",
                   dest="normalize_advantage", default=True)
    p.add_argument("--net-arch",         type=int,   nargs="+", default=[64, 64])
    p.add_argument("--n-envs",           type=int,   default=1)
    # VecNormalize
    p.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    # L1 (§8.6): inventory level and the price index sit orders of magnitude
    # apart and the obs distribution is stationary, so obs norm is on — the
    # decision is the IR's, so both values stay reachable (§8.3).
    p.add_argument("--no-norm-obs",      action="store_false", dest="norm_obs", default=True)
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
    "gae_lambda":       "gae",
    "ent_coef":         "ent",
    "normalize_advantage": "advnorm",
    "net_arch":         "arch",
    "vecnorm_clip_obs": "clipobs",
    "norm_obs":         "normobs",
    "norm_reward":      "normrew",
}
_SKIP_KEYS = {"outdir", "scenario_name", "progress_bar", "gym_log",
              "checkpoint_every_frac",
              "observation_mode", "action_mode", "reward_mode",
              # campaign metadata, not config: a run's identity must not depend
              # on which escalation entry happened to motivate it
              "tag"}


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

def _ir_fingerprint() -> str:
    """The frozen model this run trained against, or '?' if unreadable (§8.4)."""
    try:
        from mdp_ir.schema import load_ir
        return load_ir(Path(__file__).resolve().parent
                       / "dynamic_pricing_schema.json").mdp_fingerprint()
    except Exception:
        return "?"


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
        # spec §8.4 provenance set — literal keys, read without domain knowledge
        f.write("algo_class: PPO\n")
        f.write(f"sb3_version: {stable_baselines3.__version__}\n")
        f.write(f"ir_mdp_fingerprint: {_ir_fingerprint()}\n")

    print(f"observation_mode={args.observation_mode}  action_mode={args.action_mode}  "
          f"reward_mode={args.reward_mode}  outdir={outdir}")

    def make_env(rank: int):
        def _make():
            env = DynamicPricingEnv(
                scenario=scenario,
                observation_mode=args.observation_mode,
                action_mode=args.action_mode,
                reward_mode=args.reward_mode,
                logger_filename=(str(outdir / f"train_log_{rank}")
                                 if args.gym_log else None),
            )
            return Monitor(env, filename=str(outdir / f"monitor_{rank}"))
        return _make

    env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    env = VecNormalize(env, norm_obs=args.norm_obs, norm_reward=args.norm_reward,
                       clip_obs=args.vecnorm_clip_obs, gamma=args.gamma)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        normalize_advantage=args.normalize_advantage,
        policy_kwargs={"net_arch": list(args.net_arch)},
        verbose=1,
        tensorboard_log=str(outdir),
        seed=args.seed,
        device="cpu",
    )

    # §9.7's screen reads ~20 saved checkpoints; the stride is a fraction of the
    # budget so that stays true when the budget moves, and CheckpointCallback
    # counts VecEnv steps rather than environment steps.
    callback = None
    if args.checkpoint_every_frac > 0:
        callback = CheckpointCallback(
            save_freq=max(1, int(args.checkpoint_every_frac * args.total_timesteps
                                 / max(1, args.n_envs))),
            save_path=str(outdir / "checkpoints"),
            name_prefix=f"{args.scenario_name}_ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
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

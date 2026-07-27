"""SB3 PPO training script for the single-echelon inventory problem.

Trains PPO on InvSingleEnv, saves the model, then runs a short deterministic evaluation.

Example usage:
    python inv_single_ppo_train.py
    python inv_single_ppo_train.py -s paper_stochastic -o vec_d
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from inv_single_gym import InvSingleEnv
from inv_single_scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on single-echelon inventory.")
    p.add_argument("-s", "--scenario_name",    type=str,   default="simple",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", type=str,   default="vec_d",
                   choices=["vec", "vec_d", "vec_d_ip"])
    p.add_argument("-a", "--action_mode",      type=str,   default="continuous",
                   choices=["continuous", "discrete"])
    p.add_argument("--total-timesteps",   type=int,   default=2_000_000)
    p.add_argument("--seed",              type=int,   default=1)
    p.add_argument("--outdir",            type=str,   default="results")
    p.add_argument("--checkpoint-every",  type=int,   default=50_000)
    p.add_argument("--report-every",      type=int,   default=50_000)
    p.add_argument("--progress-bar",      action="store_true")
    p.add_argument("--gym-log",           type=int,   default=0, choices=[0, 1],
                   help="Write per-episode/per-step gym logs (spec §11). Off by "
                        "default: the step stream costs hundreds of MB per run.")
    # PPO hyperparameters
    # --learning_rate dest is the mdp_tuning contract (spec §8.2)
    p.add_argument("--learning_rate", "--lr-init", dest="learning_rate",
                   type=float, default=3e-4)
    p.add_argument("--lr-final",          type=float, default=3e-5)
    p.add_argument("--clip-init",         type=float, default=0.2)
    p.add_argument("--clip-final",        type=float, default=0.05)
    p.add_argument("--n-steps",           type=int,   default=2048)
    p.add_argument("--batch-size",        type=int,   default=512)
    p.add_argument("--n-epochs",          type=int,   default=10)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--gae-lambda",        type=float, default=0.95)
    p.add_argument("--ent-coef",          type=float, default=0.005)
    p.add_argument("--vf-coef",           type=float, default=0.5)
    p.add_argument("--max-grad-norm",     type=float, default=0.5)
    p.add_argument("--target-kl",         type=float, default=0.02)
    p.add_argument("--net_arch",          type=int,   nargs="+", default=[64, 64])
    p.add_argument("--n-envs",            type=int,   default=1)
    # VecNormalize
    p.add_argument("--vecnorm-clip-obs",  type=float, default=10.0)
    p.add_argument("--no-norm-reward",    action="store_false", dest="norm_reward", default=True)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    def _fn(progress_remaining: float) -> float:
        return end + (start - end) * float(progress_remaining)
    return _fn


def build_schedule(start: float, end: float) -> float | Callable[[float], float]:
    return linear_schedule(start, end) if start != end else float(start)


_SHORT_KEYS: dict[str, str] = {
    "observation_mode":        "obs",
    "action_mode":     "act",
    "total_timesteps": "steps",
    "seed":            "seed",
    "learning_rate":   "lr",
    "lr_final":        "lrf",
    "net_arch":        "arch",
    "n_envs":          "nenvs",
    "clip_init":       "clip",
    "clip_final":      "clipf",
    "n_steps":         "nsteps",
    "batch_size":      "bs",
    "n_epochs":        "ep",
    "gamma":           "gamma",
    "gae_lambda":      "gae",
    "ent_coef":        "ent",
    "vf_coef":         "vf",
    "max_grad_norm":   "grad",
    "target_kl":       "kl",
    "vecnorm_clip_obs": "clipobs",
    "norm_reward":     "normrew",
}
_SKIP_KEYS = {"outdir", "scenario_name", "progress_bar", "checkpoint_every",
              "report_every", "gym_log"}


def build_run_name(args: argparse.Namespace) -> str:
    defaults = vars(_build_arg_parser().parse_args([]))
    parts = []
    for key, default_val in defaults.items():
        if key in _SKIP_KEYS:
            continue
        val = getattr(args, key)
        if val != default_val:
            label = _SHORT_KEYS.get(key, key)
            formatted = f"{val:.3g}" if isinstance(val, float) else str(val)
            parts.append(f"{label}{formatted}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "_".join(parts) if parts else "default"
    return f"PPO_{timestamp}_{prefix}"


def resolve_paths(outdir_arg: str, scenario_name: str, run_name: str) -> tuple[Path, Path]:
    outdir   = (Path(__file__).resolve().parent / outdir_arg / scenario_name / run_name).resolve()
    ckpt_dir = outdir / "checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return outdir, ckpt_dir


class _Tee:
    """Write-through proxy mirroring a stream into a second file object."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._fh.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def tee_console(outdir: Path, name: str = "train.log") -> None:
    """Mirror stdout/stderr into `outdir/name` (spec §8.4).

    The run directory is only known once the timestamped run name exists, so a
    launcher cannot redirect into it — the script captures its own console.
    """
    fh = open(outdir / name, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def write_args(args: argparse.Namespace, outdir: Path) -> None:
    with open(outdir / f"{args.scenario_name}_ppo_args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")


# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------

class InventoryMetricsCallback(BaseCallback):
    """Log mean inventory, demand, and cost periodically."""

    def __init__(self, report_every_steps: int, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.report_every_steps = report_every_steps
        self._reset()

    def _reset(self) -> None:
        self._inventory_sum = 0.0
        self._cost_sum      = 0.0
        self._demand_sum    = 0.0
        self._n             = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            self._inventory_sum += float(info.get("inventory", 0.0))
            self._cost_sum      += float(info.get("cost", {}).get("total", 0.0))
            self._demand_sum    += float(info.get("demand", 0.0))
            self._n             += 1

        if (
            self.num_timesteps > 0
            and self.num_timesteps % self.report_every_steps == 0
            and self._n > 0
        ):
            print(
                f"step={self.num_timesteps}  "
                f"inv={self._inventory_sum / self._n:.2f}  "
                f"demand={self._demand_sum / self._n:.2f}  "
                f"cost={self._cost_sum / self._n:.4f}"
            )
            self._reset()
        return True


# ---------------------------------------------------------------------------
# Build env / model / callbacks
# ---------------------------------------------------------------------------

def build_training_env(args: argparse.Namespace, outdir: Path) -> VecNormalize:
    scenario = SCENARIOS[args.scenario_name]

    def make_env(rank: int):
        def _make():
            env = InvSingleEnv(
                scenario=scenario,
                observation_mode=args.observation_mode,
                action_mode=args.action_mode,
                logger_filename=(str(outdir / f"train_log_{rank}")
                                 if args.gym_log else None),
            )
            return Monitor(env, filename=str(outdir / f"monitor_{rank}"))
        return _make

    env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    return VecNormalize(env, norm_obs=True, norm_reward=args.norm_reward,
                        clip_obs=args.vecnorm_clip_obs, gamma=args.gamma)


def build_model(args: argparse.Namespace, env, outdir: Path) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=build_schedule(args.learning_rate, args.lr_final),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=build_schedule(args.clip_init, args.clip_final),
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        policy_kwargs={"net_arch": list(args.net_arch)},
        verbose=1,
        tensorboard_log=str(outdir),
        seed=args.seed,
        device="cpu",
        # normalize_advantage=False,
    )


def build_callbacks(args: argparse.Namespace, ckpt_dir: Path) -> CallbackList:
    checkpoint = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every),
        save_path=str(ckpt_dir),
        name_prefix="ppo_inv_single",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    metrics = InventoryMetricsCallback(
        report_every_steps=max(1, args.report_every),
    )
    return CallbackList([checkpoint, metrics])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: PPO,
    args: argparse.Namespace,
    outdir: Path,
    vecnorm_path: Path,
    n_episodes: int = 20,
) -> tuple[float, float]:
    scenario = SCENARIOS[args.scenario_name]
    raw_env = InvSingleEnv(
        scenario=scenario,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        logger_filename=str(outdir / "eval_log") if args.gym_log else None,
    )
    eval_env = DummyVecEnv([lambda: raw_env])
    if vecnorm_path.exists():
        eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
        eval_env.training   = False
        eval_env.norm_reward = False

    ep_rewards, ep_costs = [], []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = np.array([False])
        ep_reward = ep_cost = 0.0
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = eval_env.step(action)
            ep_reward += float(reward[0])
            ep_cost   += float((infos[0] or {}).get("cost", {}).get("total", 0.0))
        ep_rewards.append(ep_reward)
        ep_costs.append(ep_cost)

    eval_env.close()
    return float(np.mean(ep_rewards)), float(np.mean(ep_costs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]

    run_name = build_run_name(args)
    outdir, ckpt_dir = resolve_paths(args.outdir, args.scenario_name, run_name)
    tee_console(outdir)
    write_args(args, outdir)
    print(scenario)
    print(f"observation_mode={args.observation_mode}  outdir={outdir}")

    env       = build_training_env(args, outdir)
    model     = build_model(args, env, outdir)
    callbacks = build_callbacks(args, ckpt_dir)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=args.progress_bar,
    )

    model_path   = outdir / "ppo_inv_single.zip"
    vecnorm_path = outdir / "vecnormalize.pkl"
    model.save(str(model_path))
    env.save(str(vecnorm_path))

    mean_reward, mean_cost = evaluate(
        model=model,
        args=args,
        outdir=outdir,
        vecnorm_path=vecnorm_path,
    )
    print(f"saved model      → {model_path}")
    print(f"saved vecnormalize → {vecnorm_path}")
    print(f"eval mean_reward={mean_reward:.4f}  eval mean_cost={mean_cost:.4f}")
    env.close()


if __name__ == "__main__":
    main()

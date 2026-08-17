"""SB3 PPO training script for the FNV problem (spec §8).

Trains PPO on FnvEnv, checkpoints periodically, saves the terminal model and
the VecNormalize stats, then runs a short smoke eval.

The terminal checkpoint is NOT the deliverable (§8.6). This script only
*produces* the checkpoint set; selection happens post-hoc, outside training:

    python fnv_ppo_train.py -s FNV-aMMFE          # produces checkpoints/
    python fnv_select.py     <run_dir>            # screen -> top-k  (§9.7)
    python fnv_ppo_eval.py   --model-path <ckpt>  # confirm -> leaderboard

Example usage:
    python fnv_ppo_train.py
    python fnv_ppo_train.py -s FNV-aMMFE -r regret --total-timesteps 5000000
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fnv_gym import FnvEnv
from fnv_grids import GRIDS
from fnv_ppo_eval import tee_console
from fnv_scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# The L1 derivation (spec §8.6) — every default below is a forced move
# ---------------------------------------------------------------------------
#
# Basis, measured 2026-08-13 on `simple` (N=3) and unchanged across every
# GRIDS cell, because N is a family-agreed attribute (§5.6): T_bar = 3 steps
# per episode, obs dim 6, beta = 1.0 (IR objective.discount_factor).
# RE-DERIVE if N, the horizon, or the observation vector moves.
#
# | knob        | value        | derived from                                 |
# |-------------|--------------|----------------------------------------------|
# | gamma       | 1.0          | gamma = beta. Finite 3-period horizon, no    |
# |             |              | discounting in the objective.                |
# | gae_lambda  | 0.95         | Credit horizon 1/(1-lambda) = 20 steps. An   |
# |             |              | order pays off ONLY at the terminal demand   |
# |             |              | draw, so credit must span the whole episode: |
# |             |              | 20 >> T_bar = 3, full coverage with margin.  |
# | n_envs      | 4            | Structural (§8.3), never tuned.              |
# | n_steps     | 512          | rollout = 512 x 4 = 2048 transitions, the    |
# |             |              | floor exactly; ~683 episodes >> the 10-      |
# |             |              | episode floor. T_bar is short, so n_steps    |
# |             |              | needs no inflation beyond it.                |
# | batch_size  | 256          | rollout/8, top of the rollout/32..rollout/8  |
# |             |              | band (64..256).                              |
# | lr          | 1e-4 -> 1e-5 | Exogenous noise dominates reward variance:   |
# |             |              | profit is set by one terminal demand draw    |
# |             |              | (IR uncertainty block = a single normal      |
# |             |              | signal source). lr_final = lr_init/10.       |
# | clip        | 0.2 -> 0.05  | clip_final = clip_init/4.                    |
# | ent_coef    | 0.005        | Base rate; no premature-determinism hazard   |
# |             |              | (continuous box action, no exploration task).|
# | n_epochs    | 10           | Fixed.                                       |
# | target_kl   | 0.02         | Safety valve, never tuned — repeated         |
# |             |              | approx_kl truncation means lower the LR.     |
# | norm_obs    | on           | IR rl.obs_norm = true. Obs mixes stdev ~0.1, |
# |             |              | T ~0.9, lamb ~0.1, period 1..3, inventory    |
# |             |              | ~1, information ~0 — heterogeneous and       |
# |             |              | stationary (no drift as the agent improves). |
# | norm_reward | on           | With gamma passed (§8.3), else it normalizes |
# |             |              | against SB3's default 0.99.                  |
# | net_arch    | (64, 64)     | obs dim 6 <= ~32.                            |
# | budget      | 2M steps     | Ceiling 20k-50k episodes x T_bar = 60k-150k  |
# |             |              | steps AND >= 300 updates = 614k steps; 2M    |
# |             |              | clears both. Runs to budget — no early stop. |
# | selection   | post-hoc     | CheckpointCallback every 5% of budget (20    |
# |             |              | checkpoints); screened by fnv_select.py,     |
# |             |              | confirmed by fnv_ppo_eval.py. No             |
# |             |              | EvalCallback, no live selection env.         |
#
# Judgment beyond this table is L2 and belongs in an escalation record.

T_BAR = 3            # measured mean episode length (= N); the derivation basis
MIN_ROLLOUT_EPISODES = 10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on the FNV problem.")
    p.add_argument("-s", "--scenario_name",    type=str,   default="simple",
                   choices=list(SCENARIOS) + list(GRIDS),
                   help="a scenario (§5.4) trains a specialist; a grid (§5.6) "
                        "trains a generalist over its cells")
    p.add_argument("-o", "--observation_mode", type=str,   default="vec",
                   choices=["vec"])
    p.add_argument("-a", "--action_mode",      type=str,   default="box",
                   choices=["box"])
    p.add_argument("-r", "--reward_mode",      type=str,   default="profit",
                   choices=["profit", "regret"])
    p.add_argument("--total-timesteps",  type=int,   default=2_000_000)
    p.add_argument("--seed",             type=int,   default=1)
    p.add_argument("--outdir",           type=str,   default="results")
    p.add_argument("--checkpoint-every", type=int,   default=0,
                   help="steps between checkpoints; 0 = 5%% of the budget (§8.6)")
    p.add_argument("--progress-bar",     action="store_true")
    p.add_argument("--gym-log",          type=int,   default=0, choices=[0, 1],
                   help="write per-episode/per-step gym logs (§11); off by "
                        "default, the step stream is large")
    # PPO hyperparameters — the L1 table above
    # --learning_rate dest is the mdp_tuning contract (spec §8.2)
    p.add_argument("--learning_rate", "--lr-init", dest="learning_rate",
                   type=float, default=1e-4)
    p.add_argument("--lr-final",         type=float, default=1e-5)
    p.add_argument("--clip-init",        type=float, default=0.2)
    p.add_argument("--clip-final",       type=float, default=0.05)
    p.add_argument("--n-steps",          type=int,   default=512)
    p.add_argument("--batch-size",       type=int,   default=256)
    p.add_argument("--n-epochs",         type=int,   default=10)
    p.add_argument("--gamma",            type=float, default=1.0)
    p.add_argument("--gae-lambda",       type=float, default=0.95)
    p.add_argument("--ent-coef",         type=float, default=0.005)
    p.add_argument("--vf-coef",          type=float, default=0.5)
    p.add_argument("--max-grad-norm",    type=float, default=0.5)
    p.add_argument("--target-kl",        type=_optional_float, default=0.02,
                   help="early-stop KL guard; pass 'none' for the library "
                        "default of no guard (needed for an L0 control)")
    p.add_argument("--net_arch",         type=int,   nargs="+", default=[64, 64])
    p.add_argument("--n-envs",           type=int,   default=4)
    # VecNormalize
    p.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    p.add_argument("--no-norm-reward",   action="store_false", dest="norm_reward", default=True)
    p.add_argument("--no-norm-obs",      action="store_false", dest="norm_obs", default=True,
                   help="disable observation normalization — L0 runs without "
                        "VecNormalize (§8.6); the saved stats record the choice "
                        "so eval applies the matching transform")
    return p


def _optional_float(text: str) -> float | None:
    """Parse a float, or 'none'/'off' as None — L0 needs target_kl unset."""
    return None if text.strip().lower() in {"none", "off", ""} else float(text)


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
    "observation_mode": "obs",
    "action_mode":      "act",
    "reward_mode":      "rew",
    "total_timesteps":  "steps",
    "seed":             "seed",
    "learning_rate":    "lr",
    "lr_final":         "lrf",
    "clip_init":        "clip",
    "clip_final":       "clipf",
    "n_steps":          "nsteps",
    "batch_size":       "bs",
    "n_epochs":         "ep",
    "gamma":            "gamma",
    "gae_lambda":       "gae",
    "ent_coef":         "ent",
    "vf_coef":          "vf",
    "max_grad_norm":    "grad",
    "target_kl":        "kl",
    "net_arch":         "arch",
    "n_envs":           "nenvs",
    "vecnorm_clip_obs": "clipobs",
    "norm_reward":      "normrew",
    "norm_obs":         "normobs",
}
_SKIP_KEYS = {"outdir", "scenario_name", "progress_bar", "checkpoint_every",
              "gym_log"}


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


def _ir_fingerprint() -> str:
    """The frozen model this run trained against (§8.4).

    Deliberately NOT wrapped in a fallback. `run.provenance` checks that the
    key is *present*, not that it says anything, so a swallowed error would
    write `ir_mdp_fingerprint: ?` and pass the gate while recording nothing —
    the re-attribution this key exists to prevent, arriving through the back
    door. `write_args` runs before `model.learn()`, so raising here costs no
    training time: an unreadable IR stops the run at second zero.
    """
    from mdp_ir.schema import load_ir
    return load_ir(Path(__file__).resolve().parent / "fnv_schema.json").mdp_fingerprint()


def write_args(args: argparse.Namespace, outdir: Path) -> None:
    """Record the config and the §8.4 run-provenance set."""
    with open(outdir / f"{args.scenario_name}_ppo_args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        # spec §8.4 provenance set: the resolved class, the SB3 version, and
        # the frozen IR this run was trained against — literal keys, so a
        # checker reads them without knowing what any domain flag means.
        # `algo_class` is the adoption marker; FNV's action space is a
        # continuous box, so the class is plain PPO on every path here.
        f.write("algo_class: PPO\n")
        f.write(f"sb3_version: {stable_baselines3.__version__}\n")
        f.write(f"ir_mdp_fingerprint: {_ir_fingerprint()}\n")
        f.write(f"T_bar_basis: {T_BAR}\n")


def warn_on_rollout_floor(args: argparse.Namespace) -> None:
    """Warn (never error) when the rollout holds < 10 episodes (§8.6).

    Measured as a co-factor, not a cliff — flat arms below the floor have not
    collapsed in past campaigns — so this is a warning by design.
    """
    rollout  = args.n_steps * args.n_envs
    episodes = rollout / T_BAR
    print(f"rollout   {rollout} transitions  (~{episodes:.0f} episodes at T_bar={T_BAR})")
    if rollout < 2048:
        print(f"WARNING: rollout {rollout} < 2048 transitions (§8.6 floor).")
    if episodes < MIN_ROLLOUT_EPISODES:
        print(f"WARNING: rollout holds ~{episodes:.1f} episodes, below the "
              f"{MIN_ROLLOUT_EPISODES}-episode floor (§8.6).")


def resolve_scenario_source(scenario_name: str):
    """§5.6: the gym takes a plain scenario source — a concrete scenario, or
    the sampler derived from a grid. It never sees the grid itself."""
    if scenario_name in SCENARIOS:
        return SCENARIOS[scenario_name]
    return GRIDS[scenario_name].as_sampler()


# ---------------------------------------------------------------------------
# Build env / model / callbacks
# ---------------------------------------------------------------------------

def build_training_env(args: argparse.Namespace, outdir: Path) -> VecNormalize:
    scenario = resolve_scenario_source(args.scenario_name)

    def make_env(rank: int):
        def _make():
            env = FnvEnv(
                scenario=scenario,
                observation_mode=args.observation_mode,
                action_mode=args.action_mode,
                reward_mode=args.reward_mode,
                logger_filename=(str(outdir / f"train_log_{rank}")
                                 if args.gym_log else None),
            )
            # Monitor inside VecNormalize (§8.3): ep_rew_mean stays on the raw
            # reward scale, comparable across runs regardless of reward norm.
            return Monitor(env, filename=str(outdir / f"monitor_{rank}"))
        return _make

    env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    return VecNormalize(env, norm_obs=args.norm_obs, norm_reward=args.norm_reward,
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
    )


def build_checkpoint_callback(args: argparse.Namespace, ckpt_dir: Path) -> CheckpointCallback:
    """~20 checkpoints over the budget (§8.6).

    save_freq counts calls per env, so divide by n_envs to land on the intended
    step spacing. No EvalCallback and no live selection env by design: selection
    is post-hoc (fnv_select.py), on CRN evals of the saved checkpoints.
    """
    every = args.checkpoint_every or max(1, args.total_timesteps // 20)
    return CheckpointCallback(
        save_freq=max(1, every // args.n_envs),
        save_path=str(ckpt_dir),
        name_prefix="fnv_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )


# ---------------------------------------------------------------------------
# Smoke eval (§9.7: outside the three layers, never quoted)
# ---------------------------------------------------------------------------

def smoke_eval(
    model: PPO,
    args: argparse.Namespace,
    vecnorm_path: Path,
    n_episodes: int = 50,
) -> float:
    """"Did it learn anything" check — not a judgment input, never quoted."""
    scenario = resolve_scenario_source(args.scenario_name)
    raw_env  = FnvEnv(
        scenario=scenario,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        reward_mode=args.reward_mode,
    )
    env = DummyVecEnv([lambda: raw_env])
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training    = False
        env.norm_reward = False

    profits = []
    for _ in range(n_episodes):
        obs  = env.reset()
        done = np.array([False])
        info_last: dict = {}
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            info_last = infos[0] or {}
        profits.append(float(info_last.get("profit", 0.0)))

    env.close()
    return float(np.mean(profits))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_name = build_run_name(args)
    outdir, ckpt_dir = resolve_paths(args.outdir, args.scenario_name, run_name)
    tee_console(outdir, "train.log")
    write_args(args, outdir)

    print(resolve_scenario_source(args.scenario_name))
    print(f"observation_mode={args.observation_mode}  reward_mode={args.reward_mode}")
    print(f"outdir    {outdir}")
    warn_on_rollout_floor(args)

    env       = build_training_env(args, outdir)
    model     = build_model(args, env, outdir)
    checkpoint = build_checkpoint_callback(args, ckpt_dir)

    # Runs to budget — no early stopping (§8.6). The rollout curve is too noisy
    # to judge from; judgment happens post-hoc on CRN evals of the checkpoints.
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint,
        progress_bar=args.progress_bar,
    )

    model_path   = outdir / f"{args.scenario_name}_ppo.zip"
    vecnorm_path = outdir / "vecnormalize.pkl"
    model.save(str(model_path))
    env.save(str(vecnorm_path))
    print(f"saved model        → {model_path}")
    print(f"saved vecnormalize → {vecnorm_path}")
    print(f"checkpoints        → {ckpt_dir} ({len(list(ckpt_dir.glob('*.zip')))} saved)")

    smoke = smoke_eval(model, args, vecnorm_path)
    print(f"smoke eval (50 ep, NOT a leaderboard number): mean profit {smoke:.4f}")
    print(f"\nnext: python fnv_select.py {outdir}    # §9.7 screen → top-k")
    env.close()


if __name__ == "__main__":
    main()

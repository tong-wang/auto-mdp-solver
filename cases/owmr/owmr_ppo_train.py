"""SB3 PPO training script for owmr (spec 8).

Trains PPO on OwmrEnv for the run plan's target, saves ~20 checkpoints and the
terminal model, then runs a 20-episode smoke eval that is never quoted (9.7).
Selection is post-hoc: `owmr_select.py` screens the checkpoints, and
`owmr_ppo_eval.py` confirms the shortlist on the protocol block.

Example usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python owmr_ppo_train.py -s base                 # L1
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python owmr_ppo_train.py -s base --level L0 \\
        --learning_rate 3e-4 --lr-final 3e-4 --clip-init 0.2 --clip-final 0.2 \\
        --batch-size 64 --ent-coef 0 --target-kl -1 --n-envs 1 \\
        --no-norm-obs --no-norm-reward --checkpoint-every-frac 0 --total-timesteps 2000000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mdp_conformance.launch import assert_l1_current
from owmr_gym import OwmrEnv
from owmr_scenarios import SCENARIOS

ALGO = "ppo"


# ---------------------------------------------------------------------------
# CLI (spec 8.2 tier-1 surface + the tier-2 knobs mdp_tuning searches)
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on owmr.")
    p.add_argument("-s", "--scenario_name", type=str, default="base", choices=list(SCENARIOS))
    p.add_argument("-o", "--observation_mode", type=str, default="raw", choices=list(OwmrEnv.OBS_MODES))
    p.add_argument("-a", "--action_mode", type=str, default="order_frac", choices=list(OwmrEnv.ACTION_MODES))
    # training-only reward shaping (#E19); evaluation always scores the canonical cost
    p.add_argument("-r", "--reward_mode", type=str, default="neg_cost", choices=list(OwmrEnv.REWARD_MODES))
    # spec 8.6 solve level — RECORDED, never inferred (the caller knows what was searched)
    p.add_argument("--level", type=str, default="L1",
                   help="solve level for the run name and args log (L0, L1, L2(hp), ...)")
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--tag", type=str, default="",
                   help="ledger address this run was launched under (spec 8.2 tier 1)")
    p.add_argument("--checkpoint-every-frac", type=float, default=0.05,
                   help="checkpoint cadence as a fraction of total_timesteps (0.05 = ~20); 0 disables")
    p.add_argument("--report-every", type=int, default=100_000)
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument("--gym-log", type=int, default=0, choices=[0, 1],
                   help="per-episode/per-step gym logs (spec 11); off by default")
    # PPO knobs (dests = the mdp_tuning contract)
    p.add_argument("--learning_rate", "--lr-init", dest="learning_rate", type=float, default=1e-4)
    p.add_argument("--lr-final", type=float, default=1e-5)
    p.add_argument("--clip-init", type=float, default=0.2)
    p.add_argument("--clip-final", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True)
    # negative disables the KL valve (SB3's own default is None) — the L0 control needs that
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--net_arch", type=int, nargs="+", default=[64, 64])
    # architecture lever (#E7): "askfeat" = the flat MlpPolicy reading the raw observation
    # plus each retailer's need (owmr_ask_features.AskFeatureExtractor); needs
    # order_softmax on the raw observation, un-normalized (--no-norm-obs).
    # "askfeatnorm" (#E8) = the same, with the extractor's own running normalization
    # "split" / "splitres" (#E9) = owmr_split_policy.SplitActorCriticPolicy: separate order,
    # allocation and value trunks on the global observation; plain tanh MLPs over --net_arch,
    # or pre-norm residual blocks (one per --net_arch entry, constant width)
    p.add_argument("--policy", type=str, default="mlp",
                   choices=["mlp", "askfeat", "askfeatnorm", "split", "splitres", "seqmask", "catkeep", "dirichlet"])
    # "dirichlet" (#E23) = owmr_dirichlet_policy.DirichletPolicy (Gaussian order x Dirichlet shares); needs -a order_dirichlet
    # "catkeep" (#E21) = owmr_catkeep_policy.CatKeepPolicy (Gaussian x categorical retention); needs -a order_catkeep
    # "seqmask" (#E10) = owmr_seq_policy.PhaseMaskedPolicy, the MlpPolicy with the two-phase
    # mode's ignored components masked out of log-prob and entropy; needs -a seq_order_softmax
    p.add_argument("--n-envs", type=int, default=4)
    # the coordinate vine (#E30, owmr_vine_ppo.VinePPO): paired common-world rollouts, one sibling per
    # action coordinate, scored on that coordinate alone; needs --policy catkeep. A training-algorithm
    # lever: the main rollout, GAE, critic and world count per update are unchanged.
    p.add_argument("--vine", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--vine-p", type=float, default=0.2, help="branch probability per env per parent step")
    p.add_argument("--vine-h", type=int, default=10, help="window length in periods (no bootstrap)")
    p.add_argument("--vine-weight", type=float, default=1.0, help="side advantages' weight after rescaling to the main advantage sd (inert under Adam, #E31)")
    # continuation (#E31 extension): start from a finished run's terminal weights (policy + optimizer
    # moments via set_parameters) and its VecNormalize statistics (training continues); the
    # schedules are the NEW run's — pass constant end-of-schedule values to extend "more of the same"
    p.add_argument("--init-from", type=str, default=None, help="run dir holding base_ppo.zip + vecnormalize.pkl to continue from")
    p.add_argument("--vine-cat-grid", action=argparse.BooleanOptionalAction, default=False,
                   help="#E34: the full retention menu as the grid (one sibling per other option, weighted by pi_old) instead of one redraw")
    p.add_argument("--vine-lr-scale", type=float, default=None,
                   help="#E31: a SEPARATE optimizer for the side pass at this fraction of the main lr; unset = share the policy's optimizer (#E30)")
    # VecNormalize (spec 8.3)
    p.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    p.add_argument("--no-norm-reward", action="store_false", dest="norm_reward", default=True)
    p.add_argument("--no-norm-obs", action="store_false", dest="norm_obs", default=True)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# The L1 derivation as data (spec 8.4 / 8.6), basis: `base`, T = 100, beta = 1
#   gamma 1.0                 — gamma = beta; undiscounted objective, countdown observed.
#   gae_lambda 0.95           — credit horizon 20: an order's consequences realize over
#                               l0 + l_rt = 4 periods plus the backlog tail; 20% of T.
#   n_envs 4, n_steps 512     — rollout 2048 transitions = 20 episodes >= max(2048, 10 x T).
#   batch_size 256            — rollout / 8, power of two.
#   learning_rate 1e-4->1e-5, clip 0.2->0.05 — the noisy-reward branch (gamma demand at
#                               cv 0.5–2 dominates per-step reward variance); finals = init/10, /4.
#   ent_coef 0.005            — table default; no premature-determinism hazard recorded.
#   n_epochs 10, target_kl 0.02 — fixed rows; the valve is never tuned.
#   net_arch [64, 64]         — obs dim 15 (raw) <= 32; width row floor.
#   norm_obs / norm_reward / normalize_advantage True — 8.3 priors, checked at breadth, not asserted.
# NOT here (8.4): vf_coef, max_grad_norm, vecnorm_clip_obs (no row derives them),
# total_timesteps (the budget row derives a band: 20k–50k episodes x T = 2M–5M; 5M is the
# ceiling, the human's choice at Stage 0), checkpoint_every_frac (9.7 protocol), seed.
# ---------------------------------------------------------------------------
_L1_DERIVED: dict[str, object] = {
    "learning_rate": 1e-4, "lr_final": 1e-5,
    "clip_init": 0.2, "clip_final": 0.05,
    "n_envs": 4, "n_steps": 512, "batch_size": 256,
    "n_epochs": 10, "target_kl": 0.02,
    "gamma": 1.0, "gae_lambda": 0.95,
    "ent_coef": 0.005,
    "net_arch": [64, 64],
    "norm_obs": True, "norm_reward": True, "normalize_advantage": True,
}
_L1_BASIS: dict[str, object] = {"scenario_name": "base", "episode_len": 100, "beta": 1.0}

_GYM_KNOBS = frozenset({"norm_obs", "norm_reward", "vecnorm_clip_obs"})
_ARCH_KNOBS = frozenset({"policy"})
_LAYER_ORDER = ("hp", "gym", "arch")
_NOT_CONFIG = frozenset({"seed", "total_timesteps", "init_from"})

RUN_NAME_MAX = 240   # bytes; the filesystem caps a path component at 255

_SHORT_KEYS: dict[str, str] = {
    "total_timesteps": "steps", "seed": "seed", "learning_rate": "lr", "lr_final": "lrf",
    "net_arch": "arch", "n_envs": "nenvs", "clip_init": "clip", "clip_final": "clipf",
    "n_steps": "nsteps", "batch_size": "bs", "n_epochs": "ep", "gamma": "gamma",
    "gae_lambda": "gae", "ent_coef": "ent", "vf_coef": "vf", "max_grad_norm": "grad",
    "target_kl": "kl", "vecnorm_clip_obs": "clipobs", "norm_reward": "normrew",
    "norm_obs": "normobs", "normalize_advantage": "normadv", "policy": "pol",
    "vine": "vine", "vine_p": "vp", "vine_h": "vh", "vine_weight": "vw", "vine_lr_scale": "vls", "vine_cat_grid": "vcg",
}
_SKIP_KEYS = {"outdir", "scenario_name", "progress_bar", "checkpoint_every_frac", "init_from",
              "report_every", "gym_log", "tag", "level", "observation_mode", "action_mode", "reward_mode"}


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    def _fn(progress_remaining: float) -> float:
        return end + (start - end) * float(progress_remaining)
    return _fn


def build_schedule(start: float, end: float):
    return linear_schedule(start, end) if start != end else float(start)


def build_run_name(args: argparse.Namespace) -> str:
    defaults = vars(_build_arg_parser().parse_args([]))
    defaults.update(_L1_DERIVED)          # the derivation wins where it speaks (spec 8.4)
    # the level leads the name but a run dir is a PATH: "L2(gym)" -> "L2-gym"
    # (parentheses break every tool that interpolates the path unquoted)
    level = args.level.replace("(", "-").replace(")", "").replace("+", "-")
    parts = [level, f"obs{args.observation_mode}", f"act{args.action_mode}"]
    if args.reward_mode != "neg_cost":
        parts.append(f"rew{args.reward_mode}")
    for key, default_val in defaults.items():
        if key in _SKIP_KEYS:
            continue
        val = getattr(args, key)
        if val != default_val:
            label = _SHORT_KEYS.get(key, key)
            if isinstance(val, (list, tuple)):
                formatted = "-".join(str(v) for v in val)
            elif isinstance(val, float):
                formatted = f"{val:.3g}"
            else:
                formatted = str(val)
            parts.append(f"{label}{formatted}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    name = f"PPO_{timestamp}_{'_'.join(parts)}{suffix}"
    if len(name) > RUN_NAME_MAX:
        # a path component is capped at 255 bytes (ENAMETOOLONG hit by the 10M vine runs, #E32): keep the
        # level / obs / act / seed / tag prefix and an 8-hex digest of the full knob list; the full name
        # is written to the args file as `run_name_full`
        import hashlib
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        seed_part = [q for q in parts if q.startswith("seed")]
        name = f"PPO_{timestamp}_{'_'.join(parts[:3] + seed_part)}_h{digest}{suffix}"
        args._run_name_full = f"PPO_{timestamp}_{'_'.join(parts)}{suffix}"
    return name


def derive_deviation_tags(args: argparse.Namespace) -> str:
    """Which layers this configuration moved off the L1 derivation (a distance,
    not a level — the level is authored, spec 8.6). L0 is excepted."""
    if str(args.level).lower() == "l0":
        return ""
    baseline = vars(_build_arg_parser().parse_args([]))
    baseline.update(_L1_DERIVED)
    moved: set[str] = set()
    for dest, want in baseline.items():
        if dest in _SKIP_KEYS or dest in _NOT_CONFIG:
            continue
        got = getattr(args, dest, want)
        got = tuple(got) if isinstance(got, list) else got
        want = tuple(want) if isinstance(want, list) else want
        if got != want:
            moved.add("gym" if dest in _GYM_KNOBS else "arch" if dest in _ARCH_KNOBS else "hp")
    return "+".join(layer for layer in _LAYER_ORDER if layer in moved)


def resolve_paths(outdir_arg: str, scenario_name: str, run_name: str) -> tuple[Path, Path]:
    outdir = (Path(__file__).resolve().parent / outdir_arg / scenario_name / run_name).resolve()
    ckpt_dir = outdir / "checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return outdir, ckpt_dir


class _Tee:
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

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
    """Mirror stdout/stderr into the run dir (spec 8.4): the run name is only
    known here, so the launcher cannot redirect into it."""
    fh = open(outdir / name, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def _ir_fingerprint() -> str:
    try:
        from mdp_ir.schema import load_ir
        return load_ir(Path(__file__).resolve().parent / "owmr_schema.json").mdp_fingerprint()
    except Exception:
        return "?"


def write_args(args: argparse.Namespace, outdir: Path, deviation_tags: str) -> None:
    with open(outdir / f"{args.scenario_name}_{ALGO}_args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            if k.startswith("_"):
                continue
            f.write(f"{k}: {v}\n")
        # spec 8.4 provenance set — literal keys a checker reads without domain knowledge
        # `algo_class` is the IR-declared class (rl.algo = ppo; mdp_conformance run.provenance checks it);
        # the vine is a training-algorithm lever on PPO and is recorded as the variant (#E30/#E34)
        f.write("algo_class: PPO\n")
        if getattr(args, "vine", False):
            f.write("algo_variant: VinePPO\n")
        if getattr(args, "_run_name_full", None):
            f.write(f"run_name_full: {args._run_name_full}\n")
        f.write(f"sb3_version: {stable_baselines3.__version__}\n")
        f.write(f"ir_mdp_fingerprint: {_ir_fingerprint()}\n")
        f.write(f"solve_level: {args.level}\n")
        if deviation_tags:
            f.write(f"deviation_tags: {deviation_tags}\n")


class OwmrMetricsCallback(BaseCallback):
    """Print mean per-period cost and its split periodically (a smoke, never a judgment)."""

    def __init__(self, report_every_steps: int, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.report_every_steps = report_every_steps
        self._reset()

    def _reset(self) -> None:
        self._cost = self._wh = self._rt = self._short = self._n = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict) or "cost" not in info:
                continue
            c = info["cost"]
            self._cost += float(c["total"]); self._wh += float(c["wh_holding"])
            self._rt += float(c["rt_holding"]); self._short += float(c["rt_shortage"]); self._n += 1
        if self.num_timesteps > 0 and self.num_timesteps % self.report_every_steps == 0 and self._n:
            print(f"step={self.num_timesteps}  cost/period={self._cost / self._n:.3f}  "
                  f"(wh {self._wh / self._n:.3f}  rt {self._rt / self._n:.3f}  short {self._short / self._n:.3f})")
            self._reset()
        return True


def build_training_env(args: argparse.Namespace, outdir: Path, scenario) -> VecNormalize:
    def make_env(rank: int):
        def _make():
            env = OwmrEnv(scenario, observation_mode=args.observation_mode, action_mode=args.action_mode,
                          logger_filename=(str(outdir / f"train_log_{rank}") if args.gym_log else None),
                          reward_mode=args.reward_mode)
            return Monitor(env, filename=str(outdir / f"monitor_{rank}"))
        return _make
    env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    if getattr(args, "init_from", None):
        vn = VecNormalize.load(str(Path(args.init_from) / "vecnormalize.pkl"), env)
        vn.training = True
        assert vn.norm_obs == args.norm_obs and vn.norm_reward == args.norm_reward, "init-from run's VecNormalize flags differ"
        return vn
    return VecNormalize(env, norm_obs=args.norm_obs, norm_reward=args.norm_reward,
                        clip_obs=args.vecnorm_clip_obs, gamma=args.gamma)


def build_model(args: argparse.Namespace, env, outdir: Path, scenario=None) -> PPO:
    if args.policy in ("askfeat", "askfeatnorm"):
        from owmr_ask_features import ask_policy_kwargs
        policy_kwargs = ask_policy_kwargs(scenario, args.net_arch,
                                          running_norm=(args.policy == "askfeatnorm"))
    else:
        policy_kwargs = {"net_arch": list(args.net_arch)}
    policy = "MlpPolicy"
    if args.policy == "seqmask":
        from owmr_seq_policy import PhaseMaskedPolicy
        policy = PhaseMaskedPolicy
    if args.policy == "catkeep":
        from owmr_catkeep_policy import CatKeepPolicy
        from owmr_gym import keep_menu
        policy = CatKeepPolicy
        policy_kwargs["n_options"] = len(keep_menu(args.action_mode))      # 6 (order_catkeep) or 12 (order_catkeep_fine, #E34)
    if args.policy == "dirichlet":
        from owmr_dirichlet_policy import DirichletPolicy
        policy = DirichletPolicy
    if args.policy in ("split", "splitres"):
        from owmr_split_policy import SplitActorCriticPolicy
        policy = SplitActorCriticPolicy
        policy_kwargs["residual"] = args.policy == "splitres"
    algo, extra = PPO, {}
    if args.vine:
        assert args.policy in ("catkeep", "dirichlet"), "--vine needs a head with coordinate log-probs: --policy catkeep or dirichlet"
        from owmr_vine_ppo import VinePPO
        algo, extra = VinePPO, {"vine_p": args.vine_p, "vine_h": args.vine_h, "vine_weight": args.vine_weight,
                                "vine_lr_scale": args.vine_lr_scale, "vine_cat_grid": args.vine_cat_grid}
    return algo(
        policy=policy, env=env, **extra,
        learning_rate=build_schedule(args.learning_rate, args.lr_final),
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=build_schedule(args.clip_init, args.clip_final),
        ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl if args.target_kl >= 0 else None,
        normalize_advantage=args.normalize_advantage,
        policy_kwargs=policy_kwargs,
        stats_window_size=500,
        verbose=1, tensorboard_log=str(outdir), seed=args.seed, device="cpu",
    )


def build_callbacks(args: argparse.Namespace, ckpt_dir: Path) -> CallbackList:
    """Checkpoints + metrics only — selection is post-hoc (spec 9.7): no
    EvalCallback, no live selection env. Each checkpoint saves its own
    VecNormalize statistics so the screen scores it under its own normalizer."""
    cbs = []
    if args.checkpoint_every_frac > 0:
        save_freq = max(1, int(args.total_timesteps * args.checkpoint_every_frac / max(1, args.n_envs)))
        cbs.append(CheckpointCallback(save_freq=save_freq, save_path=str(ckpt_dir),
                                      name_prefix=f"{ALGO}_owmr", save_replay_buffer=False,
                                      save_vecnormalize=True))
    cbs.append(OwmrMetricsCallback(report_every_steps=max(1, args.report_every)))
    if args.policy == "askfeatnorm":
        # the extractor's running statistics update during rollouts only (#E8)
        from owmr_ask_features import rollout_stats_callback
        cbs.append(rollout_stats_callback())
    return CallbackList(cbs)


def smoke_eval(model: PPO, args: argparse.Namespace, scenario, vecnorm_path: Path, n_episodes: int = 20) -> float:
    raw_env = OwmrEnv(scenario, observation_mode=args.observation_mode, action_mode=args.action_mode)
    eval_env = DummyVecEnv([lambda: raw_env])
    if vecnorm_path.exists():
        eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    costs = []
    for ep in range(n_episodes):
        raw_obs, _ = eval_env.env_method("reset", seed=SMOKE_FIRST_SEED + ep)[0]
        obs = eval_env.normalize_obs(raw_obs[np.newaxis, :]) if isinstance(eval_env, VecNormalize) else raw_obs[np.newaxis, :]
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_env.step(action)
            total += float(infos[0].get("cost", {}).get("total", 0.0))   # two-phase order steps carry no cost (#E10)
            done = bool(dones[0])
        costs.append(total)
    eval_env.close()
    return float(np.mean(costs))


SMOKE_FIRST_SEED = 2_000_000   # outside both the protocol and the selection blocks


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    if args.policy in ("askfeat", "askfeatnorm"):
        # the ask is computed from stock in physical units inside the network
        assert args.action_mode == "order_softmax" and args.observation_mode == "raw", (
            "--policy askfeat reads the raw observation and emits order_softmax")
        assert not args.norm_obs, "--policy askfeat needs the un-normalized observation: pass --no-norm-obs"
    run_name = build_run_name(args)
    outdir, ckpt_dir = resolve_paths(args.outdir, args.scenario_name, run_name)

    # spec 8.6: the derivation is CHECKED at launch (refuses gamma > beta, an
    # inverted schedule, a stale rollout derivation); an L0 run skips it
    sequential = args.action_mode == "seq_order_softmax"
    assert sequential == (args.policy == "seqmask"), (
        "seq_order_softmax trains only under --policy seqmask (the ignored components must be masked), "
        "and seqmask serves only it")
    assert (args.action_mode in ("order_catkeep", "order_catkeep_fine")) == (args.policy == "catkeep"), (
        "order_catkeep / order_catkeep_fine train only under --policy catkeep (the retention component is categorical), and catkeep serves only them")
    assert (args.action_mode == "order_dirichlet") == (args.policy == "dirichlet"), (
        "order_dirichlet trains only under --policy dirichlet (the shares are sampled on the simplex), and dirichlet serves only it")
    # an episode is 2 x horizon agent steps in the two-phase mode (#E10)
    episode_len = scenario.horizon * (2 if sequential else 1)
    if str(args.level).lower() != "l0":
        assert_l1_current(args, derived=_L1_DERIVED, basis=_L1_BASIS, episode_len=episode_len)
    deviation_tags = derive_deviation_tags(args)
    print(f"LEVEL: {args.level}" + (f"   deviations from the L1 derivation: {deviation_tags}"
                                     if deviation_tags else "   sitting at the L1 derivation"))

    tee_console(outdir)
    write_args(args, outdir, deviation_tags)
    print(scenario)
    print(f"observation_mode={args.observation_mode}  action_mode={args.action_mode}  outdir={outdir}")

    env = build_training_env(args, outdir, scenario)
    model = build_model(args, env, outdir, scenario)
    if args.init_from:
        src = Path(args.init_from) / f"{args.scenario_name}_{ALGO}.zip"
        model.set_parameters(str(src), exact_match=True, device="cpu")
        print(f"INIT-FROM: weights + optimizer moments from {src}; VecNormalize statistics from {Path(args.init_from) / 'vecnormalize.pkl'}")
    model.learn(total_timesteps=args.total_timesteps, callback=build_callbacks(args, ckpt_dir),
                progress_bar=args.progress_bar)

    model_path = outdir / f"{args.scenario_name}_{ALGO}.zip"
    vecnorm_path = outdir / "vecnormalize.pkl"
    model.save(str(model_path))
    env.save(str(vecnorm_path))
    mean_cost = smoke_eval(model, args, scenario, vecnorm_path)
    print(f"saved model        -> {model_path}")
    print(f"saved vecnormalize -> {vecnorm_path}")
    print(f"smoke eval (20 episodes, seeds {SMOKE_FIRST_SEED}+): mean cost {mean_cost:.2f}  "
          f"— NOT a result (9.7); the terminal checkpoint is not the deliverable (8.6). Screen, then confirm:")
    print(f"  python owmr_select.py {outdir}")
    env.close()


if __name__ == "__main__":
    main()

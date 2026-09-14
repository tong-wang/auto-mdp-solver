"""SB3 PPO training script for the single-echelon inventory problem.

Trains PPO on InvSingleEnv, saves the model, then runs a short deterministic evaluation.

Example usage:
    python inv_single_ppo_train.py
    python inv_single_ppo_train.py -s slt -o vec_ip
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import stable_baselines3
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
from inv_single_grids import GRIDS
from mdp_conformance.launch import assert_launch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on single-echelon inventory.")
    p.add_argument("-s", "--scenario_name",    type=str,   default="simple",
                   choices=list(SCENARIOS) + list(GRIDS))
    p.add_argument("-o", "--observation_mode", type=str,   default="vec",
                   choices=["vec", "vec_ip", "vec_ctx",
                            "vec_ctx_slt", "vec_ip_ctx_slt"])
    p.add_argument("-a", "--action_mode",      type=str,   default="discrete",
                   choices=["continuous", "discrete", "hurdle"])
    # spec 9.6 target dispatch: a SCENARIOS key (specialist) or a GRIDS key
    # (generalist, trained on grid.as_sampler(); eval enumerates the cells)
    p.add_argument("-g", "--grid_name", type=str, default=None,
                   choices=list(GRIDS.keys()),
                   help="train a GENERALIST on this GRIDS entry via "
                        "grid.as_sampler(); overrides -s, and the run lands "
                        "under results/{grid_name}/")
    # spec 8.6 solve level — RECORDED, never inferred: it is the claim about
    # how much judgment produced this config, and only the caller knows that.
    # An L0 run is the library's defaults plus what the problem forces: pass
    # --level L0 with SB3's own values explicitly (lr 3e-4 constant, batch 64,
    # ent 0, kl off, arch 64x64) and --no-norm-obs --no-norm-reward, so the
    # VecNormalize wrapper is a pass-through.
    p.add_argument("--level", type=str, default="L1",
                   help="solve level for the run name and args log "
                        "(L0, L1, L2(hp), ...)")
    # CONFIG-REGISTRY (guide 12 / 13): cite a base by id instead of copying
    # its flags — `--config sc0/g0/a0/h0`. The tuple resolves through
    # inv_single_configs.py; any flag given explicitly overrides it and is
    # recorded as a DEVIATION; a design axis is write-once and never a
    # deviation (a contradiction is refused, not recorded).
    p.add_argument("--config", type=str, default=None, metavar="sc0/g0/a0/h0",
                   help="cite a CONFIG-REGISTRY tuple as this run's base")
    # upstream #62 check 5: the run this arm will be read against, named at
    # launch — a run-dir name (or path) under results/. `--settling` declares
    # a known budget mismatch against a base still being extended.
    p.add_argument("--comparator", type=str, default=None,
                   help="run name/path this arm is read against (launch check 5)")
    p.add_argument("--settling", action="store_true",
                   help="declare the comparator a settling base (budget "
                        "mismatch downgraded to a warning)")
    p.add_argument("--total-timesteps",   type=int,   default=2_000_000)
    p.add_argument("--seed",              type=int,   default=1)
    p.add_argument("--outdir",            type=str,   default="results")
    p.add_argument("--tag",               type=str,   default="",
                   help="Free-text label appended to the run name (spec 8.2 "
                        "tier 1); use it to mark an arm, never to encode a knob "
                        "the run name already diffs.")
    # 9.7 wants ~20 checkpoints whatever the budget is, so the cadence is a
    # FRACTION of total_timesteps. An absolute stride stops meaning "~20" the
    # moment the budget moves, and the post-hoc screen is sized on that count.
    p.add_argument("--checkpoint-every-frac", type=float, default=0.05,
                   help="Checkpoint cadence as a fraction of total_timesteps "
                        "(default 0.05 = ~20 checkpoints).")
    p.add_argument("--report-every",      type=int,   default=50_000)
    p.add_argument("--progress-bar",      action="store_true")
    p.add_argument("--gym-log",           type=int,   default=0, choices=[0, 1],
                   help="Write per-episode/per-step gym logs (spec §11). Off by "
                        "default: the step stream costs hundreds of MB per run.")
    # PPO hyperparameters
    # --learning_rate dest is the mdp_tuning contract (spec §8.2)
    # LR row (8.6): exogenous demand noise dominates per-step reward variance
    # (reward = -cost, cost driven by the Poisson demand draw), so the noisy
    # branch applies: 1e-4 -> 1e-5, lr_final = lr_init/10. The pre-restart
    # script defaulted 3e-4 (the near-deterministic branch) with no recorded
    # basis; the restart transcribes the derivation honestly.
    p.add_argument("--learning_rate", "--lr-init", dest="learning_rate",
                   type=float, default=1e-4)
    p.add_argument("--lr-final",          type=float, default=1e-5)
    p.add_argument("--clip-init",         type=float, default=0.2)
    p.add_argument("--clip-final",        type=float, default=0.05)
    p.add_argument("--n-steps",           type=int,   default=2048)
    p.add_argument("--batch-size",        type=int,   default=512)
    p.add_argument("--n-epochs",          type=int,   default=10)
    # gamma = beta (spec §8.6). The IR declares no objective.discount_factor
    # (beta = 1.0), so the faithful training discount is 1.0, not 0.99.
    p.add_argument("--gamma",             type=float, default=1.0)
    p.add_argument("--gae-lambda",        type=float, default=0.95)
    p.add_argument("--ent-coef",          type=float, default=0.005)
    p.add_argument("--vf-coef",           type=float, default=0.5)
    p.add_argument("--max-grad-norm",     type=float, default=0.5)
    # 8.3 reasons FROM this knob twice, so it has to be reachable: it is the
    # premise that makes reward normalization a critic-scaling detail. Opened at
    # the `breadth` tier by v0.9.20 as a PRIOR, i.e. a claim only a run settles.
    p.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="PPO advantage normalization (SB3 default: on).")
    # negative disables the safety valve (SB3's own default is None) — needed
    # to express the L0 "library defaults" control faithfully
    p.add_argument("--target-kl",         type=float, default=0.02)
    p.add_argument("--net_arch",          type=int,   nargs="+", default=[64, 64])
    # the `a` axis (guide §13.1: the policy CLASS, not a constructor argument).
    # "ordinal" installs inv_single_ordinal_head.OrdinalPolicy — same Discrete
    # action space, logits induced from 3 numbers instead of q_high+1 free ones.
    p.add_argument("--policy", type=str, default="MlpPolicy",
                   choices=["MlpPolicy", "ordinal"])
    # n_envs = 4 is structural, never tuned (spec §8.6); rollout = 2048 x 4 =
    # 8192 transitions >= max(2048, 10 episodes x T=30)
    p.add_argument("--n-envs",            type=int,   default=4)
    # VecNormalize
    p.add_argument("--vecnorm-clip-obs",  type=float, default=10.0)
    p.add_argument("--no-norm-reward",    action="store_false", dest="norm_reward", default=True)
    # with both norm flags off VecNormalize is a pass-through, which is how the
    # L0 control expresses "no VecNormalize" without a second env-build path
    p.add_argument("--no-norm-obs",       action="store_false", dest="norm_obs", default=True)
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


# ---------------------------------------------------------------------------
# The L1 derivation as data (spec 8.4/8.6). The parser defaults above ARE
# these values — `_self_check` in inv_single_configs.py asserts it — and this
# dict is what the launch check verifies a run against, what the run name
# diffs, and what a promoted value must be edited INTO (so the tag survives
# the promotion). Each row's one-line basis, from the 8.6 table at T~=30,
# beta=1.0, demand Poisson(10):
#   gamma 1.0                — gamma = beta; the objective is undiscounted.
#   gae_lambda 0.95          — credit horizon 1/(1-lambda)=20: an order's
#                              holding/shortage consequences realize within
#                              ~E[L]+1 <= 3 periods of arrival; 20 covers the
#                              longest chain (a backlog carried to horizon end)
#                              without paying full-episode variance at T~=30.
#   n_envs 4, n_steps 2048   — structural default; rollout 8192 transitions
#                              >= max(2048, 10 episodes x 30).
#   batch_size 512           — rollout/16, power of two, in [rollout/32, /8].
#   learning_rate 1e-4 -> 1e-5, clip 0.2 -> 0.05 — noisy-reward branch of the
#                              LR row (see the parser comment); final = init/10
#                              and init/4 respectively.
#   ent_coef 0.005           — default row; no premature-determinism hazard
#                              recorded for this domain.
#   n_epochs 10, target_kl 0.02 — fixed by the row; the valve is never tuned.
#   net_arch [64, 64]        — obs dim <= 7 across every mode; width row floor.
#   norm_obs True            — 8.3 prior: heterogeneous stationary magnitudes
#                              (inventory vs pipeline vs context), checked at
#                              the breadth tier, not asserted.
#   norm_reward True, normalize_advantage True — 8.3/8.6 prior rows, ditto.
# NOT here, each for a reason 8.4 gives: vf_coef, max_grad_norm (SB3's own,
# no row derives them), vecnorm_clip_obs (8.3 states a value, 8.6 derives no
# row), total_timesteps (the budget row derives a BAND — 20k-50k episodes x
# T~=30 = 600k-1.5M, and >= ~300 updates; 2M is the band's ceiling rounded to
# the checkpoint grid), checkpoint_every_frac (9.7 protocol knob) and seed
# (a replicate, not a design choice).
_L1_DERIVED: dict[str, object] = {
    "learning_rate": 1e-4, "lr_final": 1e-5,
    "clip_init": 0.2, "clip_final": 0.05,
    "n_envs": 4, "n_steps": 2048, "batch_size": 512,
    "n_epochs": 10, "target_kl": 0.02,
    "gamma": 1.0, "gae_lambda": 0.95,
    "ent_coef": 0.005,
    "net_arch": [64, 64],
    "norm_obs": True, "norm_reward": True, "normalize_advantage": True,
}
# What the derivation was measured on: the `simple` scenario's scale. Every
# arm in the restart plan shares T=30 and beta=1.0, so the basis holds across
# them; the launch check still warns on the scenario move, which is correct
# (same scale keeps every row valid, a moved T~ would not).
_L1_BASIS: dict[str, object] = {"scenario_name": "simple",
                                "episode_len": 30, "beta": 1.0}

# Which LAYER implements each derived knob (8.6's L2 sub-layers): the level is
# read off the knobs that actually moved, never off which tier searched them.
_GYM_KNOBS = frozenset({"norm_obs", "norm_reward", "vecnorm_clip_obs"})
# `policy` selects the policy CLASS, which spec §8.6 puts at the ARCH layer
# ("net_arch widths = HP layer; custom extractors = arch layer"). Bucketing
# it as hp made every ordinal run read L3(hp+gym) when it moved three layers.
_ARCH_KNOBS = frozenset({"policy"})
_LAYER_ORDER = ("hp", "gym", "arch")

# The DESIGN AXES — 8.4's always-shown tier, the IR gym block's enumerated
# mode families, fixed at Phase A. They name CELLS, never levels, and they are
# write-once: a config citation fixes them and an explicit flag contradicting
# the citation is refused, never recorded as a deviation.
# `inv_single_configs.py` reads this set by ast, so the boundary has exactly
# one definition.
_DESIGN_AXES = frozenset({"observation_mode", "action_mode"})

# A replicate and a budget are not configuration.
_NOT_CONFIG = frozenset({"seed", "total_timesteps"})


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
    "norm_obs":        "normobs",
    "normalize_advantage": "normadv",
}
# `tag` is rendered explicitly at the end of the name and level/obs/act at the
# front, so none of them is also diffed as a knob; the rest are protocol
# plumbing that says nothing about the configuration.
_SKIP_KEYS = {"outdir", "scenario_name", "grid_name", "progress_bar",
              "checkpoint_every_frac", "report_every", "gym_log", "tag",
              "level", "config", "comparator", "settling",
              "observation_mode", "action_mode"}


def build_run_name(args: argparse.Namespace) -> str:
    defaults = vars(_build_arg_parser().parse_args([]))
    defaults.update(_L1_DERIVED)      # the derivation wins where it speaks
    # the level leads the name (it is the claim), then the design axes
    parts = [args.level,
             f"obs{args.observation_mode}", f"act{args.action_mode}"]
    for key, default_val in defaults.items():
        if key in _SKIP_KEYS:
            continue
        val = getattr(args, key)
        if val != default_val:
            label = _SHORT_KEYS.get(key, key)
            # a run-dir name is a PATH: it must survive word-splitting and
            # globbing. `str([64, 64, 64, 64])` embeds spaces and brackets and
            # breaks every tool that interpolates the path unquoted (it broke
            # this campaign's eval driver on the first deep-net arm).
            if isinstance(val, (list, tuple)):
                formatted = "-".join(str(v) for v in val)
            elif isinstance(val, float):
                formatted = f"{val:.3g}"
            else:
                formatted = str(val)
            parts.append(f"{label}{formatted}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if getattr(args, "tag", "") else ""
    return f"PPO_{timestamp}_{'_'.join(parts)}{suffix}"


def _explicit_dests(argv=None) -> set[str]:
    """The dests the caller actually supplied, as opposed to those argparse
    filled in. Parsing a second time with suppressed defaults is the only way
    to tell the two apart, and the difference is what separates a deviation
    from an inherited value."""
    parser = _build_arg_parser()
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    return set(vars(parser.parse_args(argv)))


def apply_config(args: argparse.Namespace,
                 argv: list[str] | None = None) -> tuple[str, list[str]]:
    """Resolve `--config` onto `args`; return (citation, deviations).

    Precedence is registry -> explicit flag, and the flags that win are exactly
    this run's deviations — except a design axis, which the citation fixes
    write-once: a contradiction there claims one cell's address while producing
    another cell's number, so it is refused, never recorded.
    """
    import inv_single_configs as registry
    sc, g, a, h = registry.parse_tuple(args.config)
    resolved = registry.resolve(sc, g, a, h)
    explicit = _explicit_dests(argv)
    for key in ("scenario_name", "grid_name"):
        if key in explicit and getattr(args, key) != resolved.get(key):
            raise SystemExit(
                f"--{key} {getattr(args, key)!r} contradicts {sc} = "
                f"{resolved.get(key)!r} — one address, two targets")
    deviations = []
    for dest, value in sorted(resolved.items()):
        if dest in registry.NON_CLI or not hasattr(args, dest):
            continue
        if dest in explicit:
            got = getattr(args, dest)
            if (tuple(got) if isinstance(got, list) else got) != (
                    tuple(value) if isinstance(value, list) else value):
                if dest in _DESIGN_AXES:
                    raise SystemExit(
                        f"--{dest.replace('_', '-')} {got!r} contradicts the "
                        f"cited cell ({g} fixes {dest}={value!r}) — a design "
                        f"axis is never a deviation. Cite the other cell's "
                        f"tuple, or launch without --config")
                deviations.append(f"{dest} {value!r}->{got!r}")
            continue
        setattr(args, dest, list(value) if isinstance(value, list) else value)
    return f"{sc}/{g}/{a}/{h}", deviations


def derive_deviation_tags(args: argparse.Namespace) -> str:
    """How far this run sits from the L1 derivation — its DISTANCE, not its level.

    Spec §8.6 (v0.10.4, from this campaign's issue #78) settles what had been
    two incompatible definitions in one section: **the level is what a campaign
    SEARCHED** — `level >= L2 <=> a search ran on this target` — and the
    derivation table is L1's fallback source, not its definition. L1 is the best
    configuration the campaign can state without search, drawn from the
    literature, the table, or its own recorded findings.

    So layer-counting no longer answers "what level is this?". It answers a
    different and still useful question — which layers this configuration moved
    off the derivation — and §8.4 keeps that derivable while the level is
    authored. The two come apart exactly when a campaign has learned something:
    `lt_variance_k0`'s arms move hp+gym+arch and are L1, because one
    configuration was tried and no search ran on that target.

    Returns the layer tags (e.g. "hp+gym+arch"), or "" for a run sitting at the
    derivation. L0 is excepted: it is a claim about the whole configuration, not
    a distance from rows that are not in force there."""
    if str(getattr(args, "level", "") or "").lower() == "l0":
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
            moved.add("gym" if dest in _GYM_KNOBS else
                      "arch" if dest in _ARCH_KNOBS else "hp")
    return "+".join(layer for layer in _LAYER_ORDER if layer in moved)


def resolve_target(args: argparse.Namespace):
    """Spec 9.6 target dispatch: a GRIDS entry trains a generalist on the
    grid's derived sampler and lands under results/{grid_name}/; otherwise
    the named SCENARIOS entry. Never hands a grid itself to the gym."""
    if args.grid_name:
        args.scenario_name = args.grid_name
        return GRIDS[args.grid_name].as_sampler()
    # A GRIDS key in -s is the same request. mdp_tuning drives `-s` with the
    # target it was given and has no grid flag, so without this a generalist
    # could not be tuned at all — the parser accepted the name (its choices
    # list GRIDS) and then this line raised KeyError on it.
    if args.scenario_name in GRIDS:
        return GRIDS[args.scenario_name].as_sampler()
    return SCENARIOS[args.scenario_name]


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


def _ir_fingerprint() -> str:
    """The frozen model this run trained against, or '?' if unreadable (§8.4)."""
    try:
        from mdp_ir.schema import load_ir
        return load_ir(Path(__file__).resolve().parent / "inv_single_schema.json").mdp_fingerprint()
    except Exception:
        return "?"


def write_args(args: argparse.Namespace, outdir: Path,
               citation: str = "", deviations: list[str] | None = None,
               deviation_tags: str = "") -> None:
    with open(outdir / f"{args.scenario_name}_ppo_args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        # spec §8.4 provenance set: the resolved class, the SB3 version, and
        # the frozen IR this run was trained against — literal keys, so a
        # checker reads them without knowing what any domain flag means
        f.write("algo_class: PPO\n")
        f.write(f"sb3_version: {stable_baselines3.__version__}\n")
        f.write(f"ir_mdp_fingerprint: {_ir_fingerprint()}\n")
        # §8.6/§13: the authored level, the level the knobs say, and the
        # registry citation this run resolved from (empty = cited nothing)
        f.write(f"solve_level: {args.level}\n")
        if deviation_tags:
            # DISTANCE from the derivation, not a level: §8.6 (v0.10.4) makes
            # the level a search claim, so a "derived_level" key would assert
            # something the knobs cannot know
            f.write(f"deviation_tags: {deviation_tags}\n")
        f.write(f"config_citation: {citation}\n")
        f.write(f"config_deviations: {'; '.join(deviations or []) or '-'}\n")


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

def build_training_env(args: argparse.Namespace, outdir: Path,
                       scenario=None) -> VecNormalize:
    if scenario is None:
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
    return VecNormalize(env, norm_obs=args.norm_obs, norm_reward=args.norm_reward,
                        clip_obs=args.vecnorm_clip_obs, gamma=args.gamma)


def build_model(args: argparse.Namespace, env, outdir: Path) -> PPO:
    if args.policy == "ordinal":
        from inv_single_ordinal_head import OrdinalPolicy
        policy: object = OrdinalPolicy
    else:
        policy = "MlpPolicy"
    return PPO(
        policy=policy,
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
        target_kl=args.target_kl if args.target_kl >= 0 else None,
        policy_kwargs={"net_arch": list(args.net_arch)},
        verbose=1,
        tensorboard_log=str(outdir),
        seed=args.seed,
        device="cpu",
        normalize_advantage=args.normalize_advantage,
    )


def build_callbacks(
    args: argparse.Namespace,
    ckpt_dir: Path,
) -> CallbackList:
    """Checkpoints and metrics only — selection is post-hoc (spec 9.7).

    There is deliberately no live selection callback here: no `EvalCallback`, no
    selection env, no `sync_envs_normalization`. Training runs to its budget and
    saves ~20 checkpoints; `inv_single_select.py` ranks them afterwards on the
    screen block, and `inv_single_ppo_eval.py` confirms the top-k on the
    protocol block. The two blocks are disjoint, so the number that ranks a
    checkpoint is never the number that reports it.

    `CheckpointCallback` counts its own `_on_step` calls, one per vec-env step,
    so the per-env stride is the global cadence divided by `n_envs`.
    """
    save_freq = max(1, int(args.total_timesteps * args.checkpoint_every_frac
                           / max(1, args.n_envs)))
    checkpoint = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(ckpt_dir),
        name_prefix="ppo_inv_single",
        save_replay_buffer=False,
        # the screen scores each checkpoint under the normalizer it was trained
        # with, so the stats have to be saved alongside it
        save_vecnormalize=True,
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
    scenario = (GRIDS[args.scenario_name].as_sampler()
                if args.scenario_name in GRIDS else SCENARIOS[args.scenario_name])
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
    citation, deviations = "", []
    if args.config:
        citation, deviations = apply_config(args)
        print(f"CONFIG: {citation}" + (f"  deviations: {'; '.join(deviations)}"
                                       if deviations else "  (no deviations)"))
    scenario = resolve_target(args)

    run_name = build_run_name(args)
    outdir, ckpt_dir = resolve_paths(args.outdir, args.scenario_name, run_name)

    # §8.6/§13: the launch checks run AFTER every knob is resolved and BEFORE
    # any artifact exists — the L1 derivation always; the configuration and
    # comparator halves when this run cites them.
    _launch_kw = {}
    if args.config:
        import inv_single_configs as registry
        sc_id, g_id, a_id, h_id = registry.parse_tuple(citation)
        _launch_kw.update(
            base=registry.resolve(sc_id, g_id, a_id, h_id),
            declared=deviations, address=citation, owned=registry.OWNED,
            outdir=outdir, log=Path(__file__).resolve().parent / "ESCALATION.md",
            ids=registry.all_ids())
    if args.comparator:
        _launch_kw.update(
            comparator=args.comparator, settling=args.settling,
            results_root=Path(__file__).resolve().parent / args.outdir)
    assert_launch(args, derived=_L1_DERIVED, basis=_L1_BASIS,
                  episode_len=scenario.horizon, **_launch_kw)

    deviation_tags = derive_deviation_tags(args)
    # The level is a claim about SEARCH and cannot be read off the knobs
    # (§8.6, v0.10.4). What the knobs give is distance from the derivation,
    # reported here and recorded in the args log beside the authored level.
    print(f"LEVEL: {args.level} (authored: what was searched on this target)"
          + (f"   deviations from the L1 derivation: {deviation_tags}"
             if deviation_tags else "   sitting at the L1 derivation"))

    tee_console(outdir)
    write_args(args, outdir, citation, deviations, deviation_tags)
    print(scenario)
    print(f"observation_mode={args.observation_mode}  outdir={outdir}")

    env       = build_training_env(args, outdir, scenario)
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
    print("NOTE: the smoke eval above is not a result (9.7) and the terminal "
          "checkpoint is not the deliverable (8.6). Screen, then confirm:")
    print(f"  python inv_single_select.py {outdir}")
    env.close()


if __name__ == "__main__":
    main()

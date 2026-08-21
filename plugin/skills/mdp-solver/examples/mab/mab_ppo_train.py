"""SB3 PPO training script for the multi-armed bandit domain — L1 config.

Trains PPO on MabEnv (a fresh bandit instance every episode via the
scenario's world-latent sampler). Script defaults are the spec-§8.6 L1
derivation for this domain (gamma = beta = 1.0, rollout = 4 envs × 2560
steps ≈ 10 episodes, noise-class LR 1e-4 → 1e-5, lambda 0.98, ent 0.01),
with the §8.6 selection protocol: a CRN selection eval every --eval-every
steps (seeds disjoint from the 0..8191 reporting protocol), best-model
saving, and plateau early stop. The canonical saved pair
({scenario}_ppo.zip + vecnormalize.pkl) is the BEST selection checkpoint;
the terminal checkpoint is saved as *_final and is never the deliverable.

The selection criterion samples the policy to match the campaign's stochastic
reporting protocol rather than argmaxing it. `norm_obs` defaults to True for
both obs modes — a per-mode split (off for the accumulator-valued `stats`)
was tried and refuted by isolation (+71.9 for ON; see _derived_norm_obs).

`--level l0` runs the spec-§8.6 faithful-defaults control from the same
script: a frozen preset (library defaults + gamma = beta + the 2M budget,
identity VecNormalize with both norms off, no selection callback, terminal
checkpoint) that rejects any override of its knobs — L0 is definitional,
not configurable. The wrapper stack is invariant across levels; the norm
flags decide behavior, so eval always finds the same artifact pair.

Example usage:
    python mab_ppo_train.py -s gauss_K10_T1000
    python mab_ppo_train.py -s gauss_K10_T1000 -o bayes --total-timesteps 40000000
    python mab_ppo_train.py -s gauss_K10_T1000 -o bayes --level l0
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import gymnasium as gym

from mab_bayes import bayes_post_sd
from mab_gym import MabEnv
from mab_scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on the multi-armed bandit.")
    p.add_argument("-s", "--scenario_name",    type=str, default="gauss_K10_T1000",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", type=str, default="stats",
                   choices=["stats", "bayes", "bayes_h"])
    # spec §9.6 target dispatch: a SCENARIOS key (specialist) or a GRIDS key
    # (generalist, trained on grid.as_sampler()). #E36.
    p.add_argument("-g", "--grid_name", type=str, default=None,
                   help="train a GENERALIST on this GRIDS entry via "
                        "as_sampler(); mutually exclusive with -s")
    p.add_argument("-r", "--reward_mode",      type=str, default="payout",
                   choices=["payout"],
                   help="reward rendering; one legal value today, but the flag "
                        "is the name every arm joins on (spec §8.2 tier 1)")
    p.add_argument("-a", "--action_mode",      type=str, default="arm",
                   choices=["arm"])
    p.add_argument("--level", type=str, default="l1", choices=["l0", "l1"],
                   help="l1 (default): the derived config below. l0: the "
                        "spec-§8.6 faithful-defaults control — a frozen "
                        "preset; overriding its knobs is an error")
    p.add_argument("--total-timesteps",  type=int, default=20_000_000,
                   help="ceiling (§8.6: 20k episodes × T=1000); the plateau "
                        "stop usually ends the run earlier")
    p.add_argument("--seed",             type=int, default=1)
    p.add_argument("--tag",              type=str, default="",
                   help="Campaign labels for this run, comma-separated, e.g. "
                        "'A6b' or 'A6a,#E13'. Tokens starting with '#E' are "
                        "recorded as ledger ids, the rest as agenda actions. "
                        "Written to run_status.json['campaign'] only — it is "
                        "deliberately NOT part of the run name (spec §8.4 "
                        "keeps the name a derived function of the config, so "
                        "it stays reproducible from the args alone). "
                        "mab_runs_index.py reads this field to build "
                        "ESCALATION.md's RUNS table and the by_action/ "
                        "TensorBoard symlinks.")
    p.add_argument("--outdir",           type=str, default="results")
    p.add_argument("--checkpoint-every-frac", type=float, default=0.05,
                   help="checkpoint every this fraction of the budget (spec "
                        "§9.7 wants ~20 for the post-hoc screen); 0 disables")
    p.add_argument("--progress-bar",     action="store_true")
    p.add_argument("--gym-log",          type=int, default=0, choices=[0, 1],
                   help="Write per-episode/per-step gym logs (spec §11). Off by "
                        "default: the step stream costs hundreds of MB per run.")
    # PPO hyperparameters (--learning_rate dest is the mdp_tuning contract);
    # defaults = the spec-§8.6 L1 derivation for this domain
    p.add_argument("--learning_rate", "--lr-init", dest="learning_rate",
                   type=float, default=1e-4)
    p.add_argument("--lr-final",      type=float, default=1e-5)
    p.add_argument("--clip-init",     type=float, default=0.2)
    p.add_argument("--clip-final",    type=float, default=0.05)
    p.add_argument("--n-steps",       type=int,   default=2560)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--n-epochs",      type=int,   default=10)
    p.add_argument("--gamma",         type=float, default=1.0,
                   help="= objective.discount_factor (beta = 1: undiscounted "
                        "finite horizon with time-to-go observed)")
    p.add_argument("--gae-lambda",    type=float, default=0.98)
    p.add_argument("--ent-coef",      type=float, default=0.01)
    p.add_argument("--vf-coef",       type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl",     type=float, default=0.02)
    # L1 (§8.6): on. PPO's per-minibatch advantage rescaling — the premise §8.3
    # reasons FROM when it calls reward norm a critic-scaling detail, so it must
    # be checkable rather than inherited silently from SB3. Off only as a logged
    # L2 move; `mdp_tuning`'s breadth tier opens it.
    p.add_argument("--no-normalize-advantage", action="store_false",
                   dest="normalize_advantage", default=True)
    p.add_argument("--net_arch",      type=int,   nargs="+", default=[64, 64])
    p.add_argument("--n-envs",        type=int,   default=4)
    # escalation levers (ESCALATION.md #E12) — both default OFF; the plain
    # L1 backbone is unchanged unless one is asked for explicitly
    p.add_argument("--policy", type=str, default="mlp",
                   choices=["mlp", "index", "deepsets", "deepsets_max",
                            "attention", "index_ttgent", "index_temp",
                            "index_ttgent_norm", "index_temp_ttgent",
                            "index_temp_ttgent_norm", "index_temp_id",
                            "index_h"],
                   help="index = A6-rung-b equivariant per-arm scorer "
                        "(mab_equinet.IndexPolicy, L2(arch)); deepsets = "
                        "A6-rung-c, the same scorer with mean-pooled "
                        "cross-arm context in every hidden layer; "
                        "deepsets_max = A6-rung-c-max, same size but pooling "
                        "by leave-one-out max (ctx_i = best arm other than i, "
                        "the statistic the decision needs); attention = "
                        "A6-rung-d, the same slot with the pooling weights "
                        "learned per state by masked self-attention over the "
                        "arms, which spans the c / c-max endpoints; "
                        "index_ttgent = A7-t2, the index policy trained with "
                        "the entropy bonus weighted by ttg/T (anneals the "
                        "bonus in episode time); index_temp = A7-t3, the "
                        "index policy with a learned invariant temperature "
                        "head, logits = beta(s) * z. All of them force "
                        "norm_obs off — slotwise VecNormalize stats would "
                        "break the equivariance they pay for")
    p.add_argument("--anchor-coef", type=float, default=0.0,
                   help="#E31 rung 5 (COVERAGE_PLAN.md): adds anchor_coef * "
                        "KL(pi_TS || pi) to the PPO loss via "
                        "mab_kl_anchor.KlAnchorPPO — forward KL to the "
                        "Thompson action law, computed from the belief the "
                        "observation encodes. TRAINS-ONLY; selection + eval "
                        "stay on the unmodified stochastic policy. Gaussian "
                        "branch, raw obs (no --norm-obs) only.")
    p.add_argument("--shape-coef", type=float, default=0.0,
                   help="A7-rung-b (L2(gym)): potential-based shaping on the "
                        "belief, Phi(s) = -coef * sum_i post_sd_i, applied to "
                        "the TRAINING reward only; selection + eval stay on "
                        "the faithful payout (gate obligation)")
    # VecNormalize; with both norms off the wrapper is skipped entirely.
    # norm_obs defaults per observation mode — see _derived_norm_obs.
    p.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    # One BooleanOptionalAction, not a store_true/store_false pair: a parser is
    # read per *dest*, so two actions sharing `norm_obs` leave whichever came
    # first unreachable to any tool that drives this CLI — and with the default
    # derived at runtime (below), the reachable half is the wrong one.
    # `mdp_tuning` searches norm_obs at its breadth tier, so both arms must run.
    p.add_argument("--norm-obs", action=argparse.BooleanOptionalAction,
                   dest="norm_obs", default=None)
    p.add_argument("--no-norm-reward",   action="store_false",
                   dest="norm_reward", default=True)
    # §8.6 selection protocol: CRN selection eval + plateau early stop;
    # selection seeds are DISJOINT from the 0..8191 reporting protocol
    p.add_argument("--selection-deterministic", action="store_true",
                   help="select checkpoints by argmax return instead of the "
                        "campaign's stochastic reporting protocol (§8.6 wants "
                        "the two criteria to agree; see mab/ESCALATION.md O1)")
    p.add_argument("--eval-every",          type=int,   default=1_000_000)
    p.add_argument("--selection-seeds",     type=int,   default=256)
    p.add_argument("--selection-base-seed", type=int,   default=1_000_000)
    # OFF by default since 2026-08-06: plateau stopping reads the same
    # 256-seed selection curve that #E33 shows is noisy, and unlike a bad
    # checkpoint pick an early stop is NOT recoverable — the checkpoints that
    # would have followed never exist. The curve is also non-monotone here
    # (#E31 P4: the crown peaks at 3M and erodes for 17M), so a run can
    # plateau, dip and recover. Deviates from spec §8.6's `budget` row
    # pending upstream issue #7.
    p.add_argument("--patience",            type=int,   default=0,
                   help="stop after this many selection evals without "
                        "improvement; 0 disables plateau stopping (default)")
    p.add_argument("--min-delta",           type=float, default=5.0,
                   help="improvement below this does not reset patience")
    return p


# The spec-§8.6 L0 definition, executable: SB3 library defaults + the
# problem-forced gamma = beta + the declared 2M budget. target_kl 0.0 maps
# to None in build_model; eval_every 0 disables the selection callback;
# both norms off makes VecNormalize an identity wrapper (stack invariant,
# behavior off). Frozen — edits here are spec changes, not tuning.
_L0_PRESET: dict[str, object] = {
    "learning_rate": 3e-4, "lr_final": 3e-4,       # constant LR
    "clip_init": 0.2, "clip_final": 0.2,           # constant clip
    "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
    "gamma": 1.0,                                  # = beta (problem-forced)
    "gae_lambda": 0.95, "ent_coef": 0.0,
    "vf_coef": 0.5, "max_grad_norm": 0.5, "target_kl": 0.0,
    "net_arch": [64, 64], "n_envs": 1,
    "norm_obs": False, "norm_reward": False,
    "normalize_advantage": True,                   # the SB3 default, stated
    "eval_every": 0, "checkpoint_every_frac": 0.0,   # terminal checkpoint only
    "total_timesteps": 2_000_000,
    "policy": "mlp", "shape_coef": 0.0,   # escalation levers are never L0
    "anchor_coef": 0.0,
}

# --policy value -> class name in mab_equinet, for the A6 (L2(arch)) rungs.
# Membership here carries the integration contract: norm_obs is forced off,
# because slotwise VecNormalize stats would break the equivariance.
_EQUIVARIANT_POLICIES: dict[str, str] = {
    "index": "IndexPolicy",          # rung b — pure index class (#E14 crown)
    "deepsets": "DeepSetsPolicy",    # rung c — + mean-pooled cross-arm context
    "deepsets_max": "DeepSetsMaxPolicy",   # rung c-max — leave-one-out max
    "attention": "AttentionPolicy",        # rung d — learned pooling weights
    "index_ttgent": "TtgEntropyIndexPolicy",   # A7-t2 — ttg-weighted entropy
    "index_temp": "TempIndexPolicy",           # A7-t3 — temperature head
    # the 3x2 factorial completing A7 (2026-08-01): {flat, ttg/T, 2*ttg/T}
    # entropy weight x {no head, temperature head}. A6b and t2/t3 are the
    # three cells already run; these are the missing three.
    "index_ttgent_norm": "NormTtgEntropyIndexPolicy",       # A7-t2n
    "index_temp_ttgent": "TtgEntropyTempIndexPolicy",       # A7-t2-t3
    "index_temp_ttgent_norm": "NormTtgEntropyTempIndexPolicy",   # A7-t2n-t3
    "index_temp_id": "IdTempIndexPolicy",   # A8 b/c — identified temperature
    "index_h": "IndexHPolicy",   # #E36 — generalist over varying T (bayes_h)
}

# spec-§8.6 layers each escalation policy opens (solve_level); anything in
# _EQUIVARIANT_POLICIES defaults to ("arch",)
_POLICY_LAYERS: dict[str, tuple[str, ...]] = {
    "index_ttgent": ("arch", "algo"),   # index arch + a modified PPO objective
    "index_ttgent_norm": ("arch", "algo"),
    "index_temp_ttgent": ("arch", "algo"),
    "index_temp_ttgent_norm": ("arch", "algo"),
}


def resolve_target(args: argparse.Namespace):
    """The training target: a SCENARIOS source (specialist) or a grid's
    derived sampler (generalist). Spec §9.6 — never reconstruct a grid from a
    sampler's value lists; enumeration is the grid's native mode, and a grid
    itself is not callable so it can never pass as a ScenarioSource."""
    if not args.grid_name:
        return SCENARIOS[args.scenario_name]
    from mab_grids import GRIDS
    return GRIDS[args.grid_name].as_sampler()


def solve_level(args: argparse.Namespace) -> str:
    """The run's spec-§8.6 level, for `run_status.json` and §RUNS.

    `--level` selects the *config derivation* (L0 preset vs the derived L1
    backbone); it is not the whole story. §8.6 levels a run by how much
    judgment produced it: L1 is the derived config, where "every row is a
    forced move — a hard rule reading the IR", and "judgment beyond these
    rules is L2", tagged with the layer opened. A hand-designed equivariant
    scorer is judgment, not a forced move, so every A6 arch rung is L2(arch)
    and A7's shaping is L2(gym) — regardless of `--level l1` selecting the
    backbone they sit on.

    Derivable here only for the levers this script owns. L2(hp) is NOT
    derivable — a tuned config looks like an L1 config with different numbers
    — so tuned runs must be labelled at the source with
    `mab_runs_index.py --annotate <dir> --set-level 'l2(hp)'`.
    """
    if args.level == "l0":
        return "l0"
    layers = []
    if args.policy in _EQUIVARIANT_POLICIES:
        layers.extend(_POLICY_LAYERS.get(args.policy, ("arch",)))
    if args.shape_coef:
        layers.append("gym")
    if args.anchor_coef:
        layers.append("algo")
    return f"l2({'+'.join(layers)})" if layers else "l1"


def _derived_norm_obs(observation_mode: str) -> bool:
    """The §8.3 obs-normalization decision: **True for both obs modes** —
    settled empirically, not by rule-reading. Kept as a function so the
    decision point (and its history) stays greppable.

    History (mab/ESCALATION.md O2, #E8, #E10): the §8.3 rule splits on the
    observation — stationary → on, drifting/accumulator → off — and `stats`
    (counts climbing to T, unbounded payout totals) reads as the accumulator
    case, so this function briefly returned False for it (2026-07-29).
    The isolation probe refuted that reading in direction: same stream, same
    config, only the flag changed — norm_obs=True 927.63 ± 6.88 vs False
    855.73 ± 7.18 (+71.9, z≈7.2). Whatever the within-episode drift costs,
    it is second-order next to what normalization buys on raw counts and
    totals. `bayes` (bounded, self-normalizing) was never in question.

    Overridable with `--norm-obs` / `--no-norm-obs`; an explicit override is
    encoded in the run name, the derived value is not."""
    return True


def parse_args() -> argparse.Namespace:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.level == "l0":
        supplied = [s for a in parser._actions if a.dest in _L0_PRESET
                    for s in a.option_strings if s in sys.argv]
        if supplied:
            parser.error(f"--level l0 is definitional (spec §8.6) — it fixes "
                         f"these knobs; drop {supplied} or use --level l1")
        for dest, value in _L0_PRESET.items():
            setattr(args, dest, value)
    if args.norm_obs is None:                     # not supplied, not L0-forced
        args.norm_obs = _derived_norm_obs(args.observation_mode)
    if args.policy in _EQUIVARIANT_POLICIES:
        if "--norm-obs" in sys.argv:
            parser.error(f"--policy {args.policy} is equivariant by "
                         "construction; slotwise VecNormalize stats break "
                         "that — the extractor scales its own inputs")
        args.norm_obs = False
    return args


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
    "normalize_advantage": "advnorm",
    "vecnorm_clip_obs": "clipobs",
    "norm_obs":         "normobs",
    "shape_coef":       "shape",
    "anchor_coef":      "anchor",
    "norm_reward":      "normrew",
    "selection_deterministic": "seldet",
    "net_arch":         "arch",
    "n_envs":           "nenvs",
}
_SKIP_KEYS = {"outdir", "scenario_name", "progress_bar", "checkpoint_every_frac",
              "gym_log", "eval_every", "selection_seeds",
              "selection_base_seed", "patience", "min_delta", "level", "seed",
              # campaign metadata, not config: a run's identity must not
              # depend on which escalation action happened to motivate it
              # (one dir can serve several, and labels get assigned late)
              "tag"}


# arch-dimension dests (spec §8.5/§8.6 escalation layer: extractor family +
# its knobs) — emitted as their own name segment right after the level tag,
# before hp diffs: architecture defines the hypothesis class, hp the
# optimizer. Future extractor flags (--extractor, --cnn_arch, embed_dim, …)
# slot in here by name.
_ARCH_KEYS: tuple[str, ...] = ("policy", "extractor", "cnn_arch", "net_arch",
                               "embed_dim", "features_dim", "channels",
                               "kernel_size")


def _fmt_val(val: object) -> str:
    if isinstance(val, (list, tuple)):
        return "x".join(str(v) for v in val)      # [128, 128] -> 128x128
    if isinstance(val, float):
        return f"{val:.3g}"
    return str(val)


def parse_tag(tag: str) -> dict[str, list[str]]:
    """Split a ``--tag`` string into campaign actions and ledger ids.

    ``"A6a, #E13"`` -> ``{"actions": ["A6a"], "ledger": ["#E13"]}``. Ledger
    ids are usually appended after the fact (the entry is written once the
    verdict is in), so a launch-time tag normally carries actions only;
    ``mab_runs_index.py --annotate`` fills the rest in later.
    """
    actions: list[str] = []
    ledger: list[str] = []
    for tok in (t.strip() for t in tag.split(",")):
        if not tok:
            continue
        (ledger if tok.startswith("#E") else actions).append(tok)
    return {"actions": actions, "ledger": ledger}


def build_run_name(args: argparse.Namespace) -> str:
    """Config-first sort key:
    {algo}_{modes}_{level}_{arch-diffs}_{hp-diffs}_seed_{ts}.

    Problem-side fields before solution-side, mirroring the results tree
    (scenario dir → modes → solver level → arch → hp variant → instance),
    timestamp last as pure uniquifier: a sorted listing clusters one mode's
    ladder (obsX_L0, obsX_L1, …) — the leaderboard comparison set — with
    same-architecture runs and replicate seeds adjacent (`ls -t` still
    gives chronology when wanted). The name is a label only; config truth
    is args.txt."""
    parts = [f"PPO_obs{args.observation_mode}", args.level.upper()]
    if args.level != "l0":      # L0 is definitional — nothing to encode
        defaults = vars(_build_arg_parser().parse_args([]))
        # norm_obs's default is derived from the observation mode, so the
        # baseline must be too — otherwise every run encodes it as a diff
        defaults["norm_obs"] = _derived_norm_obs(args.observation_mode)

        def encode(keys) -> None:
            for key in keys:
                if key in _SKIP_KEYS or key == "observation_mode":
                    continue
                val = getattr(args, key)
                if val != defaults[key]:
                    parts.append(f"{_SHORT_KEYS.get(key, key)}{_fmt_val(val)}")

        encode([k for k in _ARCH_KEYS if k in defaults])
        encode([k for k in defaults if k not in _ARCH_KEYS])
    parts.append(f"seed{args.seed}")
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts)


def resolve_paths(outdir_arg: str, scenario_name: str, run_name: str) -> tuple[Path, Path]:
    outdir   = (Path(__file__).resolve().parent / outdir_arg / scenario_name
                / run_name).resolve()
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


def write_args(args: argparse.Namespace, outdir: Path,
               algo_class: str = "PPO") -> None:
    # spec §8.4: beside the domain's own flags, the log carries a literal
    # provenance set — the resolved class, the SB3 version, and the frozen IR
    # this run was trained against. `mask: True` names the class only to
    # someone who knows this domain; `algo_class` names it to a checker.
    with open(outdir / f"{args.grid_name or args.scenario_name}_ppo_args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write(f"algo_class: {algo_class}\n")
        f.write(f"sb3_version: {stable_baselines3.__version__}\n")
        f.write(f"ir_mdp_fingerprint: {_ir_fingerprint()}\n")


def _ir_fingerprint() -> str:
    """The frozen model this run trained against, or '?' if unreadable."""
    try:
        from mdp_ir.schema import load_ir
        return load_ir(Path(__file__).resolve().parent / "mab_schema.json").mdp_fingerprint()
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Build env / model
# ---------------------------------------------------------------------------

class BeliefPotentialShaping(gym.Wrapper):
    """A7-rung-b (L2(gym), TRAINS-ONLY): potential-based shaping on the
    belief, Phi(s) = -coef * sum_i post_sd_i(s).

    r' = r + Phi(s') - Phi(s): a pull earns an immediate bonus proportional
    to how much posterior sd it removes — largest for barely-pulled arms —
    converting part of the exploration budget from undirected (entropy) to
    uncertainty-directed. Wraps only the training env; the selection
    callback and every eval step a raw MabEnv, so checkpoints are selected
    and scored on the faithful payout (gate obligation, spec/AGENT_PLAN
    shaping rule). With gamma = 1 the telescoped sum adds Phi(s_T) -
    Phi(s_0), which is action-dependent — accepted, since the shaped reward
    is never scored."""

    def __init__(self, env: gym.Env, coef: float) -> None:
        super().__init__(env)
        self.coef = float(coef)
        self._last_phi = 0.0

    def _phi(self) -> float:
        e = self.env.unwrapped
        sd = bayes_post_sd(e._state.pulls, e._state.payouts, e._is_gauss)
        return -self.coef * float(sum(sd))

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._last_phi = self._phi()
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi = self._phi()
        reward = float(reward) + (phi - self._last_phi)
        self._last_phi = phi
        return obs, reward, terminated, truncated, info


def build_training_env(args: argparse.Namespace, outdir: Path) -> VecNormalize:
    scenario = resolve_target(args)

    def make_env(rank: int):
        def _make():
            env = MabEnv(
                scenario=scenario,
                observation_mode=args.observation_mode,
                action_mode=args.action_mode,
                reward_mode=args.reward_mode,
                logger_filename=(str(outdir / f"train_log_{rank}")
                                 if args.gym_log else None),
            )
            if args.shape_coef:
                env = BeliefPotentialShaping(env, args.shape_coef)
            return Monitor(env, filename=str(outdir / f"monitor_{rank}"))
        return _make

    env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    # the stack is invariant: VecNormalize is always present, and the norm
    # flags decide behavior — with both off it is an identity wrapper (L0),
    # so every level saves/loads the same artifact pair
    return VecNormalize(env, norm_obs=args.norm_obs,
                        norm_reward=args.norm_reward,
                        clip_obs=args.vecnorm_clip_obs, gamma=args.gamma)


def build_model(args: argparse.Namespace, env, outdir: Path) -> PPO:
    if args.policy in _EQUIVARIANT_POLICIES:
        import mab_equinet
        policy = getattr(mab_equinet, _EQUIVARIANT_POLICIES[args.policy])
        scen = resolve_target(args)
        policy_kwargs = {
            "horizon": scen.horizon,
            # A11: on stats the extractor rescales counts/totals to pull
            # share + running average; unpulled arms read the prior mean
            # (0 gauss / 0.5 bern — the frozen Phase-A priors, mab_bayes)
            "obs_mode": args.observation_mode,
            "prior_mean": 0.0 if not scen.payout.is_discrete else 0.5,
        }
    else:
        policy = "MlpPolicy"
        policy_kwargs = {"net_arch": list(args.net_arch)}
    if args.grid_name:
        assert args.observation_mode == "bayes_h", (
            "a generalist over varying T needs the episode horizon in the "
            "observation: use -o bayes_h")
        assert args.policy == "index_h", (
            "-o bayes_h is read by IndexHPolicy: use --policy index_h")
    if args.observation_mode == "bayes_h":
        assert args.policy == "index_h", "bayes_h requires --policy index_h"
    if args.anchor_coef:
        from mab_kl_anchor import KlAnchorPPO
        assert not args.norm_obs, \
            "--anchor-coef reads beliefs from raw observations; --no-norm-obs required"
        assert not resolve_target(args).payout.is_discrete, \
            "--anchor-coef implements the gaussian conjugate law only"
        ppo_cls: type[PPO] = KlAnchorPPO
        anchor_kwargs = {"anchor_coef": args.anchor_coef,
                         "anchor_obs_mode": args.observation_mode}
    else:
        ppo_cls = PPO
        anchor_kwargs = {}
    return ppo_cls(
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
        target_kl=args.target_kl if args.target_kl > 0 else None,
        normalize_advantage=args.normalize_advantage,
        policy_kwargs=policy_kwargs,
        stats_window_size=100 if args.level == "l0" else 500,
        verbose=1,
        tensorboard_log=str(outdir),
        seed=args.seed,
        device="cpu",
        **anchor_kwargs,
    )


# ---------------------------------------------------------------------------
# Selection protocol — SUPERSEDED, DO NOT COPY INTO A NEW DOMAIN.
#
# This is the pre-v0.7.0 design: a live in-training callback. Spec §8.6/§9.7
# now prescribe a post-hoc three-layer screen — CheckpointCallback only, then
# rank the checkpoints on a ~2048-seed block and confirm the top-k on the
# protocol block — and say "no EvalCallback, no live selection env". That
# change was made partly on this campaign's evidence (#E33, and #E36 V4 where
# this very callback selected a generalist's checkpoint on one wrong cell).
#
# It stays here because every number in this folder was produced by it, so
# replacing it would leave the leaderboard describing code that did not
# generate it. mab_selection_probe.py is the post-hoc screen run by hand.
# ---------------------------------------------------------------------------

class SelectionEvalCallback(BaseCallback):
    """Every --eval-every timesteps: eval on a fixed CRN seed set (disjoint
    from the 0..8191 reporting protocol), best-pair saving, and plateau early
    stop after --patience evals without --min-delta improvement. Plateau
    detection runs on this curve, not the rollout `ep_rew_mean` (spec
    §8.6/§9.7).

    The policy is sampled, not argmaxed, to match the campaign's reporting
    protocol (`mab_ppo_eval.py --stochastic`) — in a bandit the policy's
    entropy *is* its exploration mechanism, so an argmax criterion measures a
    different policy from the one that ships. Selecting on a criterion the
    deliverable is not scored by makes the saved checkpoint the argmax of the
    wrong objective; here the two criteria disagreed even about which
    observation mode wins (mab/ESCALATION.md O1). `--selection-deterministic`
    restores the old behaviour."""

    def __init__(self, args: argparse.Namespace, outdir: Path, verbose: int = 1):
        super().__init__(verbose)
        self.args = args
        self.outdir = outdir
        self.best_mean = -np.inf
        self.since_improve = 0
        self.stopped_early = False
        self._next_eval = args.eval_every
        self._raw_env: MabEnv | None = None
        self._log_path = outdir / "selection_log.tsv"
        mode = "deterministic" if args.selection_deterministic else "stochastic"
        self._log_path.write_text(
            f"# selection criterion: {mode} policy, {args.selection_seeds} CRN "
            f"seeds from {args.selection_base_seed}\n"
            "timesteps\tselection_mean\tbest_mean\tsince_improve\n")

    def _selection_mean(self) -> float:
        if self._raw_env is None:
            # resolve_target, NOT SCENARIOS[...]: under -g the training target
            # is the grid's sampler, and selecting on the -s default would rank
            # checkpoints by one cell's performance while the run optimises the
            # whole grid. (#E36 launched with this bug: its selection log is a
            # valid gauss_K10_T1000 evaluation, and its best-model choice was
            # driven by that cell alone.)
            self._raw_env = MabEnv(
                scenario=resolve_target(self.args),
                observation_mode=self.args.observation_mode,
                action_mode=self.args.action_mode,
                reward_mode=self.args.reward_mode,
            )
        venv = self.model.get_vec_normalize_env()

        def norm(o: np.ndarray) -> np.ndarray:
            return venv.normalize_obs(o[np.newaxis]) if venv is not None \
                else o[np.newaxis]

        total = 0.0
        base = self.args.selection_base_seed
        for s in range(base, base + self.args.selection_seeds):
            self._raw_env.reset(seed=s)
            obs = norm(self._raw_env._get_obs())
            done = False
            while not done:
                action, _ = self.model.predict(
                    obs, deterministic=self.args.selection_deterministic)
                o, r, done, _, _ = self._raw_env.step(int(action[0]))
                obs = norm(o)
                total += float(r)
        return total / self.args.selection_seeds

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.args.eval_every
        mean = self._selection_mean()
        if mean > self.best_mean + self.args.min_delta:
            self.best_mean = mean
            self.since_improve = 0
            self.model.save(str(self.outdir / "best_model.zip"))
            venv = self.model.get_vec_normalize_env()
            if venv is not None:
                venv.save(str(self.outdir / "vecnormalize_best.pkl"))
        else:
            self.since_improve += 1
        with open(self._log_path, "a") as f:
            f.write(f"{self.num_timesteps}\t{mean:.4f}\t{self.best_mean:.4f}\t"
                    f"{self.since_improve}\n")
        if self.verbose:
            print(f"[selection] t={self.num_timesteps:,}  mean={mean:.2f}  "
                  f"best={self.best_mean:.2f}  since_improve={self.since_improve}",
                  flush=True)
        # patience <= 0 disables plateau stopping (the default); the guard is
        # load-bearing — without it `since_improve >= 0` would stop the run at
        # its very first selection eval
        if self.args.patience > 0 and self.since_improve >= self.args.patience:
            print(f"[selection] plateau — {self.args.patience} evals without "
                  f"+{self.args.min_delta}: stopping at t={self.num_timesteps:,}",
                  flush=True)
            self.stopped_early = True
            return False
        return True


# ---------------------------------------------------------------------------
# Post-training sanity evaluation (full protocol lives in mab_ppo_eval.py)
# ---------------------------------------------------------------------------

def evaluate(
    model: PPO,
    args: argparse.Namespace,
    vecnorm_path: Path,
    n_episodes: int = 50,
) -> float:
    scenario = resolve_target(args)
    raw_env = MabEnv(
        scenario=scenario,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        reward_mode=args.reward_mode,
    )
    eval_env = DummyVecEnv([lambda: raw_env])
    if vecnorm_path.exists():
        eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
        eval_env.training    = False
        eval_env.norm_reward = False

    ep_rewards = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = np.array([False])
        total = 0.0
        while not bool(done[0]):
            action, _ = model.predict(
                obs, deterministic=args.selection_deterministic)
            obs, reward, done, _ = eval_env.step(action)
            total += float(reward[0])
        ep_rewards.append(total)
    eval_env.close()
    return float(np.mean(ep_rewards))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    scenario = resolve_target(args)

    run_name = build_run_name(args)
    outdir, ckpt_dir = resolve_paths(
        args.outdir, args.grid_name or args.scenario_name, run_name)
    tee_console(outdir)
    write_args(args, outdir)
    print(scenario)
    print(f"observation_mode={args.observation_mode}  outdir={outdir}")
    if args.level == "l0":
        why = "L0-forced"
    elif args.norm_obs == _derived_norm_obs(args.observation_mode):
        why = "derived from obs mode"
    else:
        why = "explicit override"
    print(f"norm_obs={args.norm_obs} ({why})  "
          f"selection={'deterministic' if args.selection_deterministic else 'stochastic'}")

    env   = build_training_env(args, outdir)
    model = build_model(args, env, outdir)
    # the stride is a fraction of the budget (§8.2), so "~20 checkpoints for the
    # screen" survives a budget change; CheckpointCallback counts VecEnv steps
    callbacks = []
    if args.checkpoint_every_frac > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, int(args.checkpoint_every_frac * args.total_timesteps
                                 / max(1, args.n_envs))),
            save_path=str(ckpt_dir),
            name_prefix="ppo_mab",
            save_replay_buffer=False,
            save_vecnormalize=True,
        ))
    selection = None
    if args.eval_every > 0:
        selection = SelectionEvalCallback(args, outdir)
        callbacks.append(selection)

    # run outcome as data, not log forensics: "running" now; finalized below
    # to plateau_stop / budget_complete / interrupted / error. A stale
    # "running" with no live process = killed externally (SIGKILL leaves
    # no chance to write).
    status_path = outdir / "run_status.json"

    campaign = parse_tag(args.tag)

    def write_status(outcome: str, **extra: object) -> None:
        payload = {"outcome": outcome, "level": solve_level(args),
                   "config_level": args.level,   # which --level preset was used
                   "run": outdir.name,
                   "timesteps": int(getattr(model, "num_timesteps", 0)),
                   "updated": datetime.now().isoformat(timespec="seconds")}
        if campaign["actions"] or campaign["ledger"]:
            payload["campaign"] = campaign
        payload.update(extra)
        status_path.write_text(json.dumps(payload, indent=2) + "\n")

    write_status("running", started=datetime.now().isoformat(timespec="seconds"))
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=args.progress_bar,
            # TB leaf = the run id (default would be an anonymous "PPO_1"),
            # so the tensorboard run label is unique wherever --logdir points
            tb_log_name=outdir.name,
        )
    except KeyboardInterrupt:
        write_status("interrupted")
        raise
    except BaseException as err:
        write_status("error", error=f"{type(err).__name__}: {err}"[:500])
        raise

    # §8.6: the canonical pair is the best selection checkpoint; the terminal
    # checkpoint is *_final and (at L1) never the deliverable. At L0 there is
    # no selection by definition — the terminal checkpoint is canonical.
    target       = args.grid_name or args.scenario_name
    model_path   = outdir / f"{target}_ppo.zip"
    vecnorm_path = outdir / "vecnormalize.pkl"
    final_model  = outdir / f"{target}_ppo_final.zip"
    model.save(str(final_model))
    env.save(str(outdir / "vecnormalize_final.pkl"))
    best_model, best_vec = outdir / "best_model.zip", outdir / "vecnormalize_best.pkl"
    if best_model.exists() and best_vec.exists():
        shutil.copy2(best_model, model_path)
        shutil.copy2(best_vec, vecnorm_path)
        print(f"canonical pair = best selection checkpoint "
              f"(mean {selection.best_mean:.2f})")
    else:
        shutil.copy2(final_model, model_path)
        shutil.copy2(outdir / "vecnormalize_final.pkl", vecnorm_path)
        print("no selection eval — canonical checkpoint = terminal")

    write_status(
        "plateau_stop" if (selection and selection.stopped_early)
        else "budget_complete",
        best_selection_mean=(selection.best_mean if selection else None),
        selection_evals=(selection and int(
            (selection._next_eval - args.eval_every) // args.eval_every) or 0),
    )

    mean_reward = evaluate(PPO.load(str(model_path), device="cpu"),
                           args, vecnorm_path)
    print(f"saved model        → {model_path}")
    print(f"saved vecnormalize → {vecnorm_path}")
    print(f"smoke eval mean_reward over 50 episodes: {mean_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()

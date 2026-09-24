"""Train PPO on the serial multi-echelon chain (spec §8).

Training levels (spec §8.6)
---------------------------
``--level L1`` (default) is the mandatory run: the §8.6 derivation table applied
to this domain, every derived knob logged with its rationale in
``{scenario}_ppo_args.txt``.

``--level L0`` is the control: SB3's library defaults, ``gamma = beta``, and no
VecNormalize. It is the ruler for how much the derivation actually bought, and
is reporting-only — never a gate.

Derivations for this domain, each one line:

* ``gamma = 0.95`` — the *default*, taken from the IR's
  ``objective.discount_factor`` (beta), because gamma = beta is the faithful
  starting point (spec §8.2). gamma is nonetheless a **PPO training knob**,
  distinct from beta: beta fixes how the eval and every benchmark score
  ``sum beta^t cost_t``, while gamma only shapes how far ahead the algorithm
  assigns credit while learning. Moving it is legitimate — spec §8.2 allows
  ``gamma < beta`` as a *logged escalation move*, and ``gamma > beta`` never.
  Rewards are never pre-discounted in the env either way.
* ``n_envs = 4``, ``n_steps = 2048`` — rollout 8192 transitions >= max(2048,
  10 episodes x T=50 = 500). Structural, never tuned.
* ``-a ship_fraction`` — the IR's default encoding: the action IS the fraction
  of what the source can supply, rounded iff the demand distribution is
  discrete, so one action space serves both renderings. ``ship_discrete``
  (categorical, whole units) and ``ship_absolute`` (the naive control) are the
  other two; ``--mask`` applies to ``ship_discrete`` alone, and PLAYBOOK LV3
  says not to expect it to help.
* ``gae_lambda = 0.95`` — with beta = 0.95 the effective horizon is ~20 periods
  against T = 50, so credit does not need to travel the full episode.
* VecNormalize on — the IR's ``obs_normalization`` decision. The observation box
  is a validity envelope sized to the reachable range, so raw magnitudes are far
  larger than typical play; normalization is what makes them learnable.
* ``ent_coef = 0.005`` — a small floor. Under the default ``ship_fraction``
  encoding the policy is a Gaussian over ``[0,1]`` per link, and SB3 clips an
  unbounded sample into that box, so roughly half the initial mass sits at
  "ship nothing". That is a live low-value action rather than a dead zone —
  the useful range IS the box — so the floor only slows premature collapse; it
  is not there to escape anything.

Model selection (spec §9.7) happens **after** this script, not inside it. A run
saves ~20 checkpoints with their normalizers and ``{scenario}_ppo_final.zip``,
the terminal network, which is never the deliverable. ``clark_scarf_select.py``
ranks the checkpoints on a block disjoint from the reporting block and emits
the top-k; those are confirmed at the protocol block and the winner is quoted.
There is no live selection env here: evaluating during training is what §9.7
replaced, and upstream detects it by behaviour since v0.10.14 (#89).

Usage:
    # continuous branch (default rendering, the canonical target)
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python clark_scarf_ppo_train.py \
        -s g1_n3_l2_p09 -o raw --level L1

    # lattice branch — same command, a `poisson`-pinned cell
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python clark_scarf_ppo_train.py \
        -s n3_l2_p09 -o raw --level L1
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium import spaces as gym_spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mdp_conformance.launch import assert_launch

import clark_scarf_configs as configs
from clark_scarf_gym import ClarkScarfEnv
from clark_scarf_scenarios import SCENARIOS

# The BASIS discount -- the value every derived row below was measured at,
# and the parser default for gamma. It is deliberately NOT the authority on
# what an instance is scored with: the IR names `beta` and the scenario
# carries it (F11, upstream #68), so `main` checks gamma against
# `scenario.beta` and an instance rendering another discount is caught
# there rather than silently trained at gamma = 0.95.
BETA = 0.95


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on clark_scarf.")
    p.add_argument("-s", "--scenario_name", default="n3_l2_p09",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", default="raw",
                   choices=list(ClarkScarfEnv.OBS_MODES))
    p.add_argument("-a", "--action_mode", default="ship_fraction",
                   choices=list(ClarkScarfEnv.ACTION_MODES))
    p.add_argument("--level", default="L1", choices=["L0", "L1"],
                   help="L1 = the §8.6 derivation (default); L0 = library "
                        "defaults + gamma=beta, no VecNormalize (control only)")
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outdir", default="results")
    p.add_argument("--tag", type=str, default="",
                   help="Escalation-log address this run was launched under "
                        "(e.g. 'E11', 'A3'). Recorded in the args log; never "
                        "part of the run name — identity must not depend on "
                        "which entry motivated the run (spec §8.2).")
    p.add_argument("--report-every", type=int, default=50_000)
    p.add_argument("--gym-log", type=int, default=0, choices=[0, 1],
                   help="Write per-episode/per-step gym logs (spec §11). Off by "
                        "default: the step stream costs hundreds of MB per run.")
    p.add_argument("--checkpoint-every-frac", type=float, default=0.05,
                   help="Checkpoint every FRAC of the budget (spec §8.2); 0 "
                        "disables. A fraction, not a step count, so ~20 "
                        "checkpoints stays true when total_timesteps moves.")
    # --learning_rate dest is the mdp_tuning contract (spec §8.2)
    p.add_argument("--learning_rate", "--lr-init", dest="learning_rate",
                   type=float, default=3e-4)
    p.add_argument("--lr-final", type=float, default=3e-5)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    # unset -> gamma = the instance's beta (§8.6's row, resolved in parse_args).
    # A literal default here would be the basis cell's beta wearing the clothes
    # of a derivation, which is the F11 defect one layer up.
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.005)
    # dests match the mdp_tuning knob names (spec §8.2) so the tuner can reach
    # them; clip_init/max_grad_norm/vf_coef are out of the `breadth` tier but
    # open under --knobs all
    p.add_argument("--clip-init", "--clip-range", dest="clip_init",
                   type=float, default=0.2)
    # the other half of the one-DOF pair. Its ABSENCE was the defect: with no
    # such dest, mdp_tuning's `clip_final = clip_init/4` derivation silently
    # no-ops and every run trains on a constant clip while its artifacts say
    # the schedule was tuned (spec §8.2, conformance `scripts.schedule_pairs`).
    p.add_argument("--clip-final", type=float, default=0.05)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--net_arch", type=int, nargs="+", default=[64, 64])
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--mask", action="store_true",
                   help="L2(gym) escalation #2: MaskablePPO restricted to "
                        "feasible shipment quantities (ship_discrete only)")
    p.add_argument("--ent-final", type=float, default=None,
                   help="L2(hp) escalation #4: linearly decay ent_coef to this "
                        "value; default None keeps it constant")
    p.add_argument("--order-max", type=float, default=None,
                   help="L2(gym) DIAGNOSTIC: cap what the TOP link may be asked "
                        "to ship. Caps the encoding's reach, not the env's "
                        "feasible set; the IR's ship_max is untouched")
    p.add_argument("--policy-dist", default="gaussian",
                   choices=["gaussian", "beta", "gamma", "ordinal", "dgauss", "dgauss_sig"],
                   help="action distribution for a Box action space. "
                        "gaussian = SB3 default (diagonal, GLOBAL log_std, "
                        "clipped into the box); beta = support IS [0,1], "
                        "state-dependent concentrations, never clipped")
    p.add_argument("--beta-min-conc", type=float, default=1e-3,
                   help="alpha,beta floor for --policy-dist beta. Default 1e-3 "
                        "is numerical validity only, so the head can place mass "
                        "ON the walls (the DP ships its full quantity 51%% of the "
                        "time on link 0 and nothing 16%% of the time on link 2). "
                        "1.0 restores the unimodal alpha,beta>=1 family every "
                        "pre-2026-08-23 result used")
    p.add_argument("--init-action-mean", type=float, default=None,
                   help="CONTROL for the Gamma head: bias the Gaussian's action "
                        "layer so its initial mean is this many units. The Gamma "
                        "head bundles a family change with a scale-appropriate "
                        "init (mean starts at 0.7x mean demand); this separates "
                        "them by giving the Gaussian the same starting point")
    p.add_argument("--tau-min", type=float, default=0.25,
                   help="width floor for --policy-dist ordinal: the head may "
                        "approach but never reach a degenerate point mass.")
    p.add_argument("--tau-scale", type=float, default=40.0,
                   help="softplus scale for --policy-dist ordinal. Sets how "
                        "wide the positive branch starts; 40 gives tau ~ 28 at "
                        "init, i.e. near-flat over a 41-bin range.")
    p.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    # PPO's own advantage normalization. Exposed because spec 8.3 REASONS FROM
    # it (reward norm is only a critic-scaling detail *because* the policy step
    # is already scale-invariant) — a load-bearing premise nothing could check
    # while it was unreachable. Upstream v0.9.20 (#59) puts it in the tuner's
    # breadth tier for exactly that reason; spaces.py keeps it out of
    # OPTIONAL_KNOBS, so every PPO script in this family owes the flag.
    p.add_argument("--no-normalize-advantage", action="store_false",
                   dest="normalize_advantage", default=True)
    p.add_argument("--no-norm-obs", action="store_false", dest="norm_obs", default=True)
    p.add_argument("--no-norm-reward", action="store_false", dest="norm_reward",
                   default=True)
    # -- the §13 launch surface (upstream #62 checks 4 and 5, solver v0.9.31) --
    p.add_argument("--config", default=None, metavar="sc/g/a/h",
                   help="the CONFIG-REGISTRY address this run instantiates "
                        "(e.g. 'sc2/g3/a0/h6'). Given one, the launch check "
                        "asserts the RESOLVED args are that configuration and "
                        "that every id is defined in both halves of §13. "
                        "Omitted, the config group does not run -- a tuning "
                        "trial is a point inside a study's address, not an "
                        "address, and mdp_tuning builds its commands without "
                        "one.")
    p.add_argument("--deviation", action="append", default=[], metavar="DEST=VALUE",
                   help="a knob this run deliberately moves off --config, "
                        "declared so the check can require it be present AND "
                        "be the only one. Repeatable. A move the run has and "
                        "does not declare is an undeclared second lever; a "
                        "declaration the run does not have is a banner "
                        "describing a treatment nobody is running.")
    p.add_argument("--comparator", default=None, metavar="RUN",
                   help="the run this arm will be read against -- a run "
                        "directory or a run name under --outdir. Checked at "
                        "launch for existence, the same scenario and a "
                        "comparable budget, because at read time the numbers "
                        "are in hand and a confound reads as an effect.")
    p.add_argument("--settling", action="store_true",
                   help="declare --comparator a base still being extended, so "
                        "a budget mismatch warns instead of refusing.")
    return p


def _derived_gamma(scenario_name: str) -> float:
    """§8.6's gamma row for one instance: gamma = beta, resolved not assumed."""
    return float(SCENARIOS[scenario_name].beta)


def _declared_deviations(args: argparse.Namespace) -> dict[str, object]:
    """`--deviation dest=value` into the map upstream's check 4 compares.

    A bare `dest` declares "this moved" without pinning where to; `dest=value`
    pins it and the check then refuses a run whose value disagrees with its own
    banner. Values arrive as text and are compared through the harness's
    display-text rule, so `n_epochs=30` matches the integer 30.
    """
    out: dict[str, object] = {}
    for item in getattr(args, "deviation", []) or []:
        dest, _, value = item.partition("=")
        out[dest.strip().replace("-", "_")] = value.strip() or None
    return out


def parse_args() -> argparse.Namespace:
    args = _build_arg_parser().parse_args()
    if args.gamma is None:
        args.gamma = _derived_gamma(args.scenario_name)
    if args.level == "L0":
        # faithful library defaults; gamma stays at beta because that is the
        # spec's faithful STARTING point for a control run, not because gamma
        # is untunable — it is a training knob (spec §8.2)
        args.learning_rate = 3e-4
        args.lr_final = 3e-4          # constant, no schedule
        args.n_steps = 2048
        args.batch_size = 64
        args.n_epochs = 10
        args.gae_lambda = 0.95
        args.ent_coef = 0.0
        args.clip_init = 0.2
        args.clip_final = 0.2         # constant, no schedule
        args.net_arch = [64, 64]
        args.n_envs = 1
        args.norm_obs = False
        args.norm_reward = False
    return args


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    def _fn(progress_remaining: float) -> float:
        return end + progress_remaining * (start - end)
    return _fn


def build_schedule(start: float, end: float) -> float | Callable[[float], float]:
    return start if start == end else linear_schedule(start, end)


# The §8.6 L1 derivation, as data (spec §8.4). `build_run_name` diffs against
# THIS, never against the parser defaults: §8.4 encodes a knob when it differs
# from the derivation, and §8.6 makes the defaults the L1 centre only until a
# campaign promotes a discovered value. Adopt a tuned value as a default and the
# tag survives, because the derivation did not move. `mdp_tuning` reads the same
# dict (v0.9.25): every trial's command is seeded from it, so a knob a study does
# not search runs at its DERIVED value rather than at whatever the default became.
#
# Scope is exactly what the §8.6 table derives. `vf_coef` and `max_grad_norm` are
# absent because no row derives them — adding a key here would be claiming a
# derivation this domain never made. `target_kl` is absent for a different
# reason: the table fixes it at 0.02 and this script exposes no dest for it, so
# there is nothing for the diff to fire on (owed; see README).
#
# One row is contestable and is recorded rather than quietly corrected. §8.6's
# LR row reads "exogenous noise dominates reward variance -> 1e-4 -> 1e-5;
# near-deterministic dense reward -> 3e-4 -> 3e-5". Demand is this domain's only
# stochastic source and it drives every cost term, which argues the first branch,
# while the campaign's L1 has run the second since #E3. The value below is what
# L1 *is*, so the archive stays legible; re-reading the row is an L1
# re-derivation (a new generation, §8.6), never a flag on a command line.
_L1_DERIVED: dict[str, object] = {
    # `gamma` is DELIBERATELY ABSENT — §8.4's runtime carve-out. Its §8.6 row is
    # gamma = beta, and beta is per instance since F11, so the derived value is
    # not a constant this dict could hold: it is 0.95 on the discounted cells
    # and 1.0 on the `_b1` twins. The dict omits the key, `parse_args` computes
    # it from the resolved scenario, and `build_run_name` diffs against the
    # computed centre for the run in hand — the same shape `mab` uses for a knob
    # it reads off the observation mode. Writing 0.95 here instead would make
    # every tuning trial on a twin warm-start from the WRONG centre, since
    # v0.9.25 seeds each trial's command from this dict.
    "gae_lambda": 0.95,           # credit horizon ~20 periods against T = 50
    "n_envs": 4,                  # structural, never tuned
    "n_steps": 2048,              # rollout 8192 >= max(2048, 10 episodes x 50)
    "batch_size": 512,            # rollout/16, inside rollout/32 … rollout/8
    "learning_rate": 3e-4,
    "lr_final": 3e-5,             # lr_init/10
    "clip_init": 0.2,
    "clip_final": 0.05,           # clip_init/4 (§8.2's one degree of freedom)
    "ent_coef": 0.005,            # the table's floor; no known collapse hazard
    "n_epochs": 10,
    "net_arch": [64, 64],         # obs dim <= ~32 at every swept (N, L)
    "norm_obs": True,             # §8.3 decision — a PRIOR, checked at breadth
    "norm_reward": True,          # on, with gamma passed (§8.3)
    "normalize_advantage": True,  # SB3 default, stated because §8.3 reasons from it
}

# What the derivation was measured ON (§8.6). beta lives here rather than among
# the derived values: it is a property of the problem the derivation was made
# against, and reading it off the derived gamma would make the gamma <= beta
# check tautological on a domain that derives gamma = beta.
#
# The basis cell is the LATTICE one. Every row above was measured at
# `n3_l2_p09` before F10 made the gamma density the default rendering, and the
# continuous frame runs `g1_n3_l2_p09` — same N, L, T and cost structure, so
# `assert_l1_current` warns rather than refuses, which is the honest reading:
# same scale keeps every row valid, and the warning is the record that the
# instance moved.
_L1_BASIS: dict[str, object] = {"scenario_name": "n3_l2_p09",
                                "episode_len": 50, "beta": BETA}

_SHORT = {"observation_mode": "obs", "action_mode": "act", "level": "lvl",
          "total_timesteps": "steps", "seed": "seed", "learning_rate": "lr",
          "lr_final": "lrf", "net_arch": "arch", "n_envs": "nenvs",
          "n_steps": "nsteps", "batch_size": "bs", "n_epochs": "ep",
          "gamma": "gamma", "gae_lambda": "gae", "ent_coef": "ent",
          "clip_init": "clip", "clip_final": "clipf", "vf_coef": "vf",
          "max_grad_norm": "grad", "vecnorm_clip_obs": "clipobs",
          "norm_obs": "normobs", "norm_reward": "normrew", "order_max": "omax",
          "policy_dist": "dist", "beta_min_conc": "bmc",
          "init_action_mean": "initmu",
          "normalize_advantage": "normadv"}
# tag/gym_log/checkpoint_every_frac are metadata and diagnostics: they do not
# change the policy, so a run's identity must not depend on them
_SKIP = {"outdir", "scenario_name", "report_every",
         "tag", "gym_log", "checkpoint_every_frac",
         # the §13 launch surface: an address, its declared deviations and the
         # comparator say what a run MEANS, never what it computes. Two runs
         # differing only in these are the same run, so identity must not move
         # (spec §8.2 — the same reason `tag` is here)
         "config", "deviation", "comparator", "settling"}
_SHORT.update({"mask": "mask", "ent_final": "entf"})


def build_run_name(args: argparse.Namespace) -> str:
    """Encode obs/act/level plus every non-default hyperparameter (spec §8.4)."""
    defaults = vars(_build_arg_parser().parse_args([]))
    defaults.update(_L1_DERIVED)      # the derivation wins where it speaks (§8.4)
    # the one row computed rather than written: gamma = this instance's beta
    defaults["gamma"] = _derived_gamma(args.scenario_name)
    parts = [f"obs{args.observation_mode}", f"lvl{args.level}"]
    for key, dflt in defaults.items():
        if key in _SKIP or key in ("observation_mode", "level"):
            continue
        val = getattr(args, key)
        if val != dflt:
            lbl = _SHORT.get(key, key)
            parts.append(f"{lbl}{val:.3g}" if isinstance(val, float) else f"{lbl}{val}")
    return f"PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_" + "_".join(parts)


def resolve_paths(outdir_arg: str, scenario: str, run_name: str) -> Path:
    outdir = (Path(__file__).resolve().parent / outdir_arg / scenario / run_name).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


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
    """Mirror stdout/stderr into ``outdir/train.log`` (spec §8.4).

    The run directory only exists once the timestamped name is built, so a
    launcher cannot redirect into it — the script captures its own console.
    """
    fh = open(outdir / name, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def _provenance(args: argparse.Namespace) -> dict[str, str]:
    """Spec §8.4's minimum provenance set, beside the flags this domain exposes.

    Literal keys, so a checker reads them without domain knowledge. `mask: True`
    names the trained class only to someone who knows this domain; the class
    itself is what `run.provenance` compares against the IR's declared algos —
    and `ir_mdp_fingerprint` ties the run to the frozen model it trained
    against, so a run cannot be silently re-attributed to a different IR.

    The class is resolved from the same expression `main` constructs with, not
    re-derived: a provenance field that reasons its way to an answer can be
    right about a run that did something else.
    """
    import stable_baselines3

    fingerprint = "unavailable"
    try:
        from mdp_ir.schema import load_ir
        fingerprint = load_ir(
            Path(__file__).resolve().parent / "clark_scarf_schema.json"
        ).mdp_fingerprint()
    except Exception as exc:                              # pragma: no cover
        fingerprint = f"unavailable ({type(exc).__name__})"
    return {"algo_class": (MaskablePPO if args.mask else PPO).__name__,
            "sb3_version": stable_baselines3.__version__,
            "ir_mdp_fingerprint": fingerprint}


def write_args(args: argparse.Namespace, outdir: Path) -> None:
    record = {**vars(args), **_provenance(args)}
    with open(outdir / f"{args.scenario_name}_ppo_args.txt", "w") as f:
        for k, v in sorted(record.items()):
            f.write(f"{k}: {v}\n")



class EntropyDecayCallback(BaseCallback):
    """Linearly decay ent_coef (L2(hp) escalation #4).

    Rationale (#E4): with ``norm_reward=True`` returns are rescaled to ~unit
    variance while ``ent_coef`` stays fixed. Cost falls ~20x over training, so
    after normalization the entropy bonus grows RELATIVELY more important as the
    policy converges — which is the suspected cause of both L1 runs degrading
    ~5-8% after their 1.1M-step optimum. Decaying it removes that drift.
    """

    def __init__(self, start: float, end: float, total: int):
        super().__init__()
        self.start, self.end, self.total = start, end, total

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(1, self.total))
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        return True


class ProgressCallback(BaseCallback):
    def __init__(self, every: int):
        super().__init__()
        self.every, self._next = every, every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            ep = self.model.ep_info_buffer
            if ep:
                r = float(np.mean([e["r"] for e in ep]))
                print(f"[train] step={self.num_timesteps} "
                      f"ep_rew_mean={r:.2f} (cost {-r:.2f})", flush=True)
        return True


def build_training_env(args, scenario, outdir: Path) -> VecNormalize:
    def make_env(rank: int):
        def _make():
            env = ClarkScarfEnv(scenario, observation_mode=args.observation_mode,
                                action_mode=args.action_mode,
                                order_max=args.order_max,
                                logger_filename=(str(outdir / f"gym_log_{rank}")
                                                 if args.gym_log else None))
            return Monitor(env, filename=str(outdir / f"monitor_{rank}"))
        return _make

    venv = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    # with both flags off VecNormalize is a pass-through — that is how L0
    # expresses "no VecNormalize" without a second env-build path
    return VecNormalize(venv, norm_obs=args.norm_obs, norm_reward=args.norm_reward,
                        gamma=args.gamma, clip_obs=args.vecnorm_clip_obs)


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    run_name = build_run_name(args)
    outdir = resolve_paths(args.outdir, args.scenario_name, run_name)
    tee_console(outdir)
    print(f"outdir={outdir}")
    write_args(args, outdir)
    print(f"[cfg] level={args.level} obs={args.observation_mode} "
          f"N={scenario.n_echelons} L={scenario.leadtime} T={scenario.horizon} "
          f"gamma={args.gamma}{' (=beta)' if args.gamma == scenario.beta else f' (beta={scenario.beta})'} "
          f"norm_obs={args.norm_obs}", flush=True)

    # The launch gate (spec §8.6, solver v0.9.23). Conformance asks whether the
    # code is spec-shaped before anything runs and mdp_gates asks whether a
    # number clears its baselines after everything has; this is the question in
    # between — is the experiment well-posed? — and it refuses a stale
    # derivation, gamma > beta and an inverted schedule. T~ is this run's
    # horizon, read off the scenario just resolved, so the staleness comparison
    # is against the instance actually being trained on rather than the one the
    # rows were measured at. #E12 is why a banner is not enough here: L1 printed
    # a clip schedule it had never implemented for the whole campaign.
    #
    # v0.9.31 adds two more groups, and this domain is one of the few that can
    # run them: check 4 asserts the resolved args ARE the configuration the run
    # cites, in BOTH directions, and check 5 asserts the comparator exists,
    # shares the scenario and carries a comparable budget. `resolve()` starts
    # each axis from `origin()`, which is complete over what the axis owns, so
    # the base is total over `owned` and no knob is invisible. Both groups are
    # opt-in per run rather than mandatory: mdp_tuning builds a trial's command
    # from `_L1_DERIVED` and never passes an address, and refusing those would
    # take the study down to enforce a rule about arms.
    address = args.config
    assert_launch(
        args, derived=_L1_DERIVED, basis=_L1_BASIS,
        beta=scenario.beta, episode_len=scenario.horizon,
        base=configs.resolve(address) if address else None,
        declared=_declared_deviations(args),
        address=address,
        owned=(configs.G_OWNS | configs.A_OWNS | configs.H_OWNS) if address else None,
        outdir=outdir,
        log=Path(__file__).resolve().parent / "ESCALATION.md" if address else None,
        ids=(set(configs.SC) | set(configs.G) | set(configs.A) | set(configs.H))
        if address else None,
        comparator=args.comparator,
        results_root=Path(__file__).resolve().parent / args.outdir,
        settling=args.settling,
    )
    if address:
        print(f"[cfg] address {address} resolves to {configs.level(address, _declared_deviations(args))}",
              flush=True)
    else:
        print("[cfg] no --config: this run is unaddressed, so §13's "
              "declared-equals-resolved check did not run", flush=True)

    env = build_training_env(args, scenario, outdir)
    if args.mask:
        assert args.action_mode == "ship_discrete", (
            "--mask needs a discrete encoding: masking a continuous Box is "
            f"meaningless, got {args.action_mode!r}"
        )
        print("[cfg] MaskablePPO: head restricted to feasible quantities", flush=True)
    algo = MaskablePPO if args.mask else PPO
    policy_cls = "MlpPolicy"
    head_kwargs: dict = {}
    if args.policy_dist == "beta":
        assert not args.mask, "--policy-dist beta is continuous; --mask needs a discrete head"
        assert isinstance(env.action_space, gym_spaces.Box), (
            f"--policy-dist beta needs a Box action space, got {env.action_space}; "
            f"use -a ship_fraction"
        )
        from clark_scarf_beta_policy import BetaActorCriticPolicy
        policy_cls = BetaActorCriticPolicy
        # via policy_kwargs so it is SAVED WITH THE MODEL. A class attribute
        # does not travel: `PPO.load` would rebuild the head with whatever
        # default was in force and score a policy that was never trained
        head_kwargs = {"min_concentration": float(args.beta_min_conc)}
        print(f"[cfg] Beta action head (min concentration {args.beta_min_conc}): "
              f"support IS the action space, so nothing is clipped and the "
              f"concentrations are state-dependent", flush=True)
    if args.policy_dist == "gamma":
        assert not args.mask, "--policy-dist gamma is continuous; --mask needs a discrete head"
        assert isinstance(env.action_space, gym_spaces.Box), (
            f"--policy-dist gamma needs a Box action space, got {env.action_space}")
        import clark_scarf_beta_policy as _bp
        policy_cls = _bp.GammaActorCriticPolicy
        head_kwargs = {"scale_hint": float(scenario.demand.mean())}
        print(f"[cfg] Gamma action head: support [0, inf) -- the decision's own "
              f"domain, no right-hand wall invented. scale hint "
              f"{_bp._GAMMA_SCALE_HINT[0]:.1f} (mean demand)", flush=True)
    if args.policy_dist == "ordinal":
        assert not args.mask, (
            "--policy-dist ordinal is the non-maskable head: masking is "
            "rejected here (#E5/#E6/#E15) and a masked run FAILs run.provenance")
        assert isinstance(env.action_space, gym_spaces.MultiDiscrete), (
            f"--policy-dist ordinal needs a MultiDiscrete action space, got "
            f"{env.action_space}; use -a ship_discrete")
        import clark_scarf_ordinal_head as _oh
        policy_cls = _oh.OrdinalActorCriticPolicy
        # via policy_kwargs so they are SAVED WITH THE MODEL -- the Beta head's
        # recorded bug, and both sibling implementations still carry its shape
        head_kwargs = {"tau_min": float(args.tau_min),
                       "tau_scale": float(args.tau_scale)}
        print(f"[cfg] ordinal action head: 3 parameters per link induce all "
              f"{int(env.action_space.nvec[0])} logits, with an explicit atom "
              f"at ship=0 (tau_min {args.tau_min}, tau_scale {args.tau_scale})",
              flush=True)
    if args.policy_dist in ("dgauss", "dgauss_sig"):
        assert not args.mask, (
            "--policy-dist dgauss is the non-maskable head: masking is rejected "
            "here (#E5/#E6 lattice board; run.provenance)")
        assert isinstance(env.action_space, gym_spaces.MultiDiscrete), (
            f"--policy-dist dgauss needs a MultiDiscrete action space, got "
            f"{env.action_space}; use -a ship_discrete")
        import clark_scarf_dgauss_head as _dg
        policy_cls = _dg.DGaussActorCriticPolicy
        head_kwargs = {"tau_min": float(args.tau_min),
                       "tau_scale": float(args.tau_scale),
                       "mu_param": ("sigmoid" if args.policy_dist == "dgauss_sig"
                                    else "affine")}
        print(f"[cfg] discretized-Gaussian head ({head_kwargs['mu_param']} mu): "
              f"2 numbers per link induce all "
              f"{int(env.action_space.nvec[0])} logits — adjacency pooling with "
              f"NO zero atom (tau_min {args.tau_min}, tau_scale {args.tau_scale})",
              flush=True)
    model = algo(
        policy_cls, env,
        learning_rate=build_schedule(args.learning_rate, args.lr_final),
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda, ent_coef=args.ent_coef,
        clip_range=build_schedule(args.clip_init, args.clip_final),
        vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
        normalize_advantage=args.normalize_advantage,
        policy_kwargs={"net_arch": list(args.net_arch), **head_kwargs},
        tensorboard_log=str(outdir), seed=args.seed, verbose=0,
        # MLP policy: SB3 itself warns the GPU is counterproductive here, and
        # pinning CPU keeps the four parallel runs from contending for one device
        device="cpu",
    )
    if args.init_action_mean is not None:
        assert args.policy_dist == "gaussian", (
            "--init-action-mean is the Gaussian control; the beta/gamma heads "
            "carry their own initialization")
        # SB3 initializes the action layer near zero, so a Box whose useful
        # range sits far from 0 starts outside it. This puts the Gaussian's
        # initial mean where the Gamma head's already is, so a Gamma-vs-Gaussian
        # comparison isolates the DISTRIBUTION rather than the starting point.
        import torch as _th
        with _th.no_grad():
            model.policy.action_net.bias.fill_(float(args.init_action_mean))
        print(f"[cfg] Gaussian action layer biased to an initial mean of "
              f"{args.init_action_mean:.2f} units (control for the Gamma head)",
              flush=True)

    cbs = [ProgressCallback(args.report_every)]
    if args.checkpoint_every_frac > 0:
        # CheckpointCallback counts VecEnv steps, not env steps (spec §8.2).
        # save_vecnormalize is load-bearing, not a nicety: §9.7 scores each
        # checkpoint under ITS OWN normalizer, and ranking an early checkpoint
        # under the run's terminal statistics ranks it on observations it never
        # saw.
        cbs.append(CheckpointCallback(
            save_freq=max(1, int(args.checkpoint_every_frac
                                 * args.total_timesteps / args.n_envs)),
            save_path=str(outdir / "checkpoints"),
            name_prefix=f"{args.scenario_name}_ppo",
            save_vecnormalize=True))
    if args.ent_final is not None:
        cbs.append(EntropyDecayCallback(args.ent_coef, args.ent_final,
                                        args.total_timesteps))
    model.learn(total_timesteps=args.total_timesteps, callback=cbs)
    model.save(str(outdir / f"{args.scenario_name}_ppo_final.zip"))
    env.save(str(outdir / "vecnormalize_final.pkl"))
    n_ck = len(list((outdir / "checkpoints").glob("*.zip"))) \
        if (outdir / "checkpoints").is_dir() else 0
    print(f"[done] {n_ck} checkpoint(s) saved; screen them with "
          f"clark_scarf_select.py (§9.7), then confirm the winner")
    print(f"[done] {outdir}")


if __name__ == "__main__":
    main()

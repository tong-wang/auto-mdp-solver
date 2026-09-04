"""MaskablePPO training script for the ADI-flex domain (F7 two-phase env).

The env is sequentially masked (adi_flex_gym.AdiFlexEnv.action_masks): one
shared Discrete action space serves the order phase and the allocation phase,
and the per-step mask selects the live range. One policy conditions on the
`phase` observation feature. The homogeneous branch is single-phase (all-true
masks) and trains through the same class.

Example usage:
    python adi_flex_ppo_train.py -s homog_L0_T2
    python adi_flex_ppo_train.py -s het_exp4 --total-timesteps 2000000
"""

import argparse
from datetime import datetime
from pathlib import Path

import sb3_contrib
import stable_baselines3
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mdp_conformance.launch import assert_l1_current

from adi_flex_gym import (ACTION_MODES, MIP_MODES, AdiFlexEnv,
                          agent_steps_per_period)
from adi_flex_scenarios import SCENARIOS


# torch's `Simplex` constraint demands |sum(p) - 1| < 1e-6, a FIXED tolerance,
# and a float32 softmax over this domain's Discrete(order_max + 1) — 236
# categories at N=30 — exceeds it as soon as the max probability passes ~0.9986
# (measured: peak logit 12 -> sum-1 = 1.55e-6). An (s,S) policy IS near
# deterministic, so converging is what trips it: six L1 runs died at 10-32% of
# budget with `ep_rew_mean` already at -734 against the DP's -685, no NaN, no
# inf, `approx_kl` 0.0011 and `explained_variance` 0.962 in the last table. The
# distribution was valid; only the validator's tolerance was not. Validation is
# a debug aid, not a numerical safeguard — and it scales the wrong way, since a
# wider action space accumulates more float32 summation error for the same
# policy. Disabled globally, before any distribution is built.
torch.distributions.Distribution.set_default_validate_args(False)


def _ir_fingerprint() -> str:
    """The frozen model this run trained against, or '?' if unreadable (§8.4)."""
    try:
        from mdp_ir.schema import load_ir
        return load_ir(
            Path(__file__).resolve().parent / "adi_flex_schema.json"
        ).mdp_fingerprint()
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on the ADI-flex domain.")
    p.add_argument("-s", "--scenario_name",    type=str, default="homog_L0_T2",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", type=str, default="vec",
                   choices=["vec", *MIP_MODES])
    p.add_argument("-a", "--action_mode",      type=str, default="seq_mask",
                   choices=list(ACTION_MODES))
    p.add_argument("-r", "--reward_mode",      type=str, default="neg_cost",
                   choices=["neg_cost"])
    # NOTE: timesteps are AGENT steps; a period is `1 + n_alloc` of them on the
    # heterogeneous branch (3 at T_dl = 2) and exactly 1 on the homogeneous one
    p.add_argument("--total-timesteps",  type=int,   default=1_000_000)
    # spec sec 8.6 solve level — RECORDED, never inferred: it is the claim about
    # how much judgment produced this config, and only the caller knows that.
    # L0 must additionally pass --no-vecnorm (the level forbids it outright).
    p.add_argument("--level", type=str, default="L1",
                   help="solve level tag for the run name and args log (L0, L1, L2(hp), ...)")
    p.add_argument("--seed",             type=int,   default=1)
    # §CONFIG-REGISTRY (guide §12 / upstream #60): cite a base by id instead of
    # copying its flags — `--config sc0/g2/a0/h2`. The tuple resolves through
    # adi_flex_configs.py, any flag given explicitly overrides it and is
    # recorded as a DEVIATION (R2: a run is sc+g+a+h+deviations+seed), and a
    # tuple that breaks a cross-axis constraint is refused (R1c).
    p.add_argument("--config", type=str, default=None, metavar="sc0/g2/a0/h2",
                   help="cite a §CONFIG-REGISTRY tuple as this run's base")
    # spec §9.1: the escalation-log entry this run was launched under (A8, #E5).
    # Free-form, recorded verbatim in the args log, so a run directory answers
    # "which agenda item produced this" without a filename convention.
    # RQ2's isolation instrument (#E13): the ORDER is taken from the AP y*
    # table and only the protection levels are the agent's, so the arm differs
    # from PL(sigma) in exactly one function. Not an action mode and it cannot
    # be — schema v1 requires every mode to drive every decision, and this
    # drives only `allocate`; it also depends on a SOLVED artifact the IR
    # cannot declare. Unset, this script behaves exactly as before.
    p.add_argument("--fixed-order-ap", type=str, default=None, metavar="PATH",
                   help="AP y* table (.npz): order from it, protection from the "
                        "agent (forces action_mode=order_protection)")
    p.add_argument("--tag",              type=str,   default="")
    p.add_argument("--outdir",           type=str,   default="results")
    p.add_argument("--progress-bar",     action="store_true")
    # OFF by default (F29). The gym's per-period log is a debugging artifact:
    # at 1.5M agent steps it is ~30 MB per run of rows nothing reads back, and
    # a training run is the worst place to pay for it. Turn it on for one
    # deliberate episode, not for a fleet.
    p.add_argument("--gym-log", action="store_true",
                   help="write the gym's per-period step/episode logs (debugging; off by default)")
    # PPO hyperparameters
    # --- the L1 derivation, written as the defaults (spec §8.6) -------------
    # These ARE the derived values, not library defaults, because `mdp_tuning`
    # warm-starts trial 0 from the argparse defaults: a script whose defaults
    # are SB3's makes trial 0 the L0 centre for every knob the study opens,
    # which is the failure §8.6's warm-start paragraph exists to prevent. It
    # had exactly that effect here — A8's trial 0 ran lr 3e-4 / n_steps 2048 /
    # batch 512 / ent ~0 while L1 derives 1e-4 / 512 / 128 / 0.005.
    #   learning_rate  exogenous Poisson demand dominates reward variance
    #                  (IR uncertainty block) -> the noise-dominated row,
    #                  1e-4 -> 1e-5; `lr_final` carries the /10.
    #   n_envs         4, structural, never tuned; mdp_tuning LOCKS it at this
    #                  default rather than searching it.
    #   n_steps        rollout = n_steps x n_envs >= max(2048 transitions,
    #                  10 episodes). 512 x 4 = 2048, and 10 episodes x T=30 =
    #                  300 transitions, so the transition floor binds.
    #   batch_size     rollout/32 .. rollout/8 = 64 .. 256, power of two -> 128.
    #   ent_coef       0.005, the ops default; no premature-determinism hazard
    #                  here (the action space is explored by masking, not by
    #                  a bandit-like need to keep sampling).
    #   gamma          = beta = 1.0 (finite 30-period horizon, undiscounted).
    #   gae_lambda     0.95; consequences land within the lead time, not over
    #                  the horizon.
    #   n_epochs       10, fixed.
    #   net_arch       (64,64) for obs dim <= ~32 (ours is 2-4).
    # NOT moved to L1 values: `target_kl`, `lr_final` and `clip_final` use None
    # to mean "no KL valve / constant schedule", which is what L0 REQUIRES and
    # what its launcher relies on. None of the three is warm-start-relevant —
    # target_kl is never tuned, and schedule finals are driver-derived.
    p.add_argument("--learning_rate", "-l", type=float, default=1e-4)
    p.add_argument("--n-envs",           type=int,   default=4)
    p.add_argument("--n-steps",          type=int,   default=512)
    p.add_argument("--batch-size",       type=int,   default=128)
    p.add_argument("--n-epochs",         type=int,   default=10)
    p.add_argument("--gamma",            type=float, default=1.0)
    p.add_argument("--gae-lambda",       type=float, default=0.95)
    p.add_argument("--clip_init",        type=float, default=0.2)
    p.add_argument("--ent-coef",         type=float, default=0.005)
    # SB3's default; exposed because `mdp_tuning`'s breadth tier tunes it and
    # the unmatched-knob lint flags a tier knob the script cannot receive
    p.add_argument("--vf-coef",          type=float, default=0.5)
    p.add_argument("--target-kl",        type=float, default=None)
    p.add_argument("--net-arch",         type=int,   nargs="+", default=[64, 64])
    # schedules: a FINAL value turns the corresponding knob into a linear decay
    # from its initial value (spec sec 8.6 L1: lr_final = lr_init/10,
    # clip_final = clip_init/4). Absent -> constant, which is L0's requirement.
    p.add_argument("--lr-final",         type=float, default=None)
    p.add_argument("--clip-final",       type=float, default=None)
    # post-hoc model selection (spec sec 8.6 / 9.7): the terminal checkpoint is
    # never the deliverable, so save a ladder and screen it after training
    # §8.2 defaults this to 0.05 (~20 checkpoints): a run that saves none makes
    # the terminal checkpoint the deliverable by default, which §8.6 forbids
    # outright. L0 passes 0 explicitly — that is its own row, not a default.
    p.add_argument("--checkpoint-every-frac", type=float, default=0.05,
                   help="save a checkpoint every FRAC of the budget (0.05 = ~20 of them; 0 = off)")
    # VecNormalize
    p.add_argument("--no_norm_obs", action="store_false", dest="norm_obs", default=True,
                   help="disable observation normalization — half of what L0 requires")
    p.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    p.add_argument("--no-norm-reward",   action="store_false", dest="norm_reward", default=True)
    # PPO's own advantage normalization (SB3 default True, per MINIBATCH). Worth
    # a flag here because it rescales each minibatch to unit variance, which
    # hides a real difference in advantage magnitude between observation
    # encodings — plus_a's post-critic advantage sd is 2x plus_b's (F36), and
    # normalization divides exactly that away. Neither this script nor
    # mdp_tuning exposed it before.
    p.add_argument("--no-normalize-advantage", action="store_false",
                   dest="normalize_advantage", default=True,
                   help="disable PPO's per-minibatch advantage normalization")
    # ARCH-layer lever (ESCALATION.md #E22): the ORDER head's
    # distribution parameterization. `ordinal` = mixture-at-zero + discretized
    # Gaussian (3 numbers instead of order_max+1 logits) — the action space,
    # decode, masking and algorithm (`a0` = MaskablePPO) are untouched, so this
    # is a policy-architecture knob like net_arch, NOT an action mode and NOT
    # an IR change. Gated by adi_flex_ordinal_head_probe.py (G1-G5).
    # PROTECT_MODES only: the head assumes head-0 is the order quantity of a
    # one-shot MultiDiscrete mode; the fixed-order instrument has no order head.
    p.add_argument("--order-head", type=str, default="categorical",
                   choices=["categorical", "ordinal"])
    return p


def _linear(initial: float, final: float):
    """SB3 schedule: progress_remaining goes 1 -> 0 over the budget."""
    def f(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)
    return f


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


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
    this run's deviations. R1c violations refuse: a tuple whose resolution
    contradicts a constraint declared on one of its own ids is not a run whose
    numbers mean anything, and resolving it silently is what the constraint
    exists to prevent.
    """
    import adi_flex_configs as registry
    sc, g, a, h = registry.parse_tuple(args.config)
    violations = registry.check_requires(sc, g, a, h)
    if violations:
        raise SystemExit("config citation refused (R1c):\n  "
                         + "\n  ".join(violations))
    resolved = registry.resolve(sc, g, a, h)
    explicit = _explicit_dests(argv)
    if "scenario_name" in explicit and args.scenario_name != resolved["scenario_name"]:
        raise SystemExit(
            f"-s {args.scenario_name!r} contradicts {sc} = "
            f"{resolved['scenario_name']!r} — one address, two scenarios")
    deviations = []
    for dest, value in sorted(resolved.items()):
        if dest in registry.NON_CLI or not hasattr(args, dest):
            continue
        if dest in explicit:
            got = getattr(args, dest)
            if (tuple(got) if isinstance(got, list) else got) != (
                    tuple(value) if isinstance(value, list) else value):
                if dest in _DESIGN_AXES:
                    # write-once (F50): a design axis is fixed by the citation.
                    # Overriding it would claim one cell's address while
                    # producing another cell's number — refuse, never record.
                    raise SystemExit(
                        f"--{dest.replace('_', '-')} {got!r} contradicts the "
                        f"cited cell ({g} fixes {dest}={value!r}) — a design "
                        f"axis is never a deviation (F50). Cite the other "
                        f"cell's tuple, or launch without --config if the "
                        f"cell is unregistered (a probe is ledger-addressed)")
                deviations.append(f"{dest} {value!r}->{got!r}")
            continue
        setattr(args, dest, list(value) if isinstance(value, list) else value)
    return f"{sc}/{g}/{a}/{h}", deviations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHORT_KEYS: dict[str, str] = {
    "total_timesteps":  "steps",
    "seed":             "seed",
    "learning_rate":    "lr",
    "lr_final":         "lrfin",
    "n_envs":           "nenvs",
    "n_steps":          "nsteps",
    "batch_size":       "bs",
    "n_epochs":         "ep",
    "gamma":            "gamma",
    "gae_lambda":       "lam",
    "clip_init":        "clip",
    "clip_final":       "clipfin",
    "ent_coef":         "ent",
    "vf_coef":          "vf",
    "target_kl":        "kl",
    "net_arch":         "arch",
    "norm_obs":         "normobs",
    "vecnorm_clip_obs": "clipobs",
    "norm_reward":      "normrew",
    "normalize_advantage": "advnorm",
    "order_head":       "ordhead",
}
_SKIP_KEYS = {"outdir", "scenario_name", "progress_bar", "level", "tag", "gym_log",
              "checkpoint_every_frac", "config", "fixed_order_ap",
              "observation_mode", "action_mode", "reward_mode"}

# --- the derivation as data (spec §8.4) ------------------------------------
# What the L1 table in the CLI block DERIVED — not what the parser is set to.
# The two are identical today and part company on the first promotion, and it
# is this dict that the run name, `mdp_tuning`'s trial 0 and the launch check
# are all measured against. Editing it is a RE-DERIVATION — a new L1 generation
# that re-bases every Δ(L2−L1) above it — never a default edit, so it takes a
# logged basis and a ledger entry.
_L1_DERIVED: dict[str, object] = {
    "learning_rate": 1e-4, "lr_final": 1e-5,
    "n_envs": 4, "n_steps": 512, "batch_size": 128,
    "n_epochs": 10, "target_kl": 0.02,
    "gamma": 1.0, "gae_lambda": 0.95,
    "clip_init": 0.2, "clip_final": 0.05,
    "ent_coef": 0.005,
    "net_arch": [64, 64],
    "norm_obs": True, "norm_reward": True, "normalize_advantage": True,
}
# `target_kl`, `lr_final` and `clip_final` are derived HERE while their argparse
# defaults stay None: None means "no KL valve / constant schedule", which is
# what L0 requires and what its launcher relies on (F39). Before §8.4 those two
# facts could not both be written down and the derivation was the one that
# lost — a run that forgot `--lr-final` sat off the L1 centre with nothing in
# its name to say so. Now the name diffs the derivation, so omitting it reads
# `lrfinNone`.
#   NOT here, each for a reason §8.4 gives: `vf_coef` (SB3's, no row derives
#   it), `vecnorm_clip_obs` (§8.3 states a value, §8.6 derives no row),
#   `total_timesteps` (the budget row derives a BAND — 20k-50k episodes x
#   T~30 = 600k-1.5M — and a band has no single value to diff against),
#   `checkpoint_every_frac` (a §9.7 protocol knob the launch check judges on
#   its own terms) and `seed` (a replicate, not a design choice).
_L1_BASIS: dict[str, object] = {"scenario_name": "homog_L0_T2",
                                "episode_len": 30, "beta": 1.0}

# Which LAYER implements each derived knob (§8.6's L2 sub-layers). The test is
# what implements it, never which tier searches it: `mdp_tuning`'s tiers cross
# layer lines — `breadth` opens `norm_obs`, which is the §8.3 vec-env wrapper
# stack and therefore a GYM knob — so a study is not automatically L2(hp), and
# the level is read off the knobs that actually moved (spec §8.6, v0.9.21).
_GYM_KNOBS = frozenset({"norm_obs", "norm_reward", "vecnorm_clip_obs"})
# policy-architecture knobs beyond net_arch (which predates the layer split
# and stays counted as hp so no historical level re-reads)
_ARCH_KNOBS = frozenset({"order_head"})
_LAYER_ORDER = ("hp", "gym", "arch")

# The DESIGN AXES — §8.4's always-shown tier, the IR gym block's enumerated
# mode families, fixed at Phase A. They name CELLS, never levels: moving along
# one changes which question is being answered, not how hard the answer was
# pushed, and the ladder is per-cell (§8.6). They are also write-once: an
# escalation below a cell may never overwrite them, which `apply_config`
# refuses rather than records (F50). `adi_flex_configs` reads this set by ast,
# so the boundary has exactly one definition.
_DESIGN_AXES = frozenset({"observation_mode", "action_mode", "reward_mode"})

# R3: a replicate and a budget are not configuration.
_NOT_CONFIG = frozenset({"seed", "total_timesteps"})


def derive_level(args: argparse.Namespace) -> str:
    """The level this run's KNOBS say it is (spec §8.6).

    The baseline is the COMPLETE origin — the derived value where §8.6 speaks,
    the parser default where it does not (F47) — over every configuration dest
    except the design axes, which name cells rather than levels (F49/F50). A
    knob no row derives (`vf_coef`) can therefore still make a level: someone
    moving it tried a configuration the derivation did not produce, which is
    §8.6's own invariant for L2.

    L0 is excepted: it is a claim about the whole configuration — the library's
    defaults plus what the problem forces — rather than a distance from a
    derivation whose rows are not in force there.
    """
    if str(getattr(args, "level", "") or "").lower() == "l0":
        return "L0"
    baseline = vars(_build_arg_parser().parse_args([]))
    baseline.update(_L1_DERIVED)          # the derivation wins where it speaks
    moved: set[str] = set()
    for dest, want in baseline.items():
        if dest in _SKIP_KEYS or dest in _NOT_CONFIG:
            continue                      # design axes live in _SKIP_KEYS
        got = getattr(args, dest, want)
        got = tuple(got) if isinstance(got, list) else got
        want = tuple(want) if isinstance(want, list) else want
        if got != want:
            moved.add("arch" if dest in _ARCH_KNOBS
                      else "gym" if dest in _GYM_KNOBS else "hp")
    layers = [layer for layer in _LAYER_ORDER if layer in moved]
    if not layers:
        return "L1"
    return f"L{len(layers) + 1}({'+'.join(layers)})"


def build_run_name(args: argparse.Namespace) -> str:
    defaults = vars(_build_arg_parser().parse_args([]))
    defaults.update(_L1_DERIVED)          # the derivation wins where it speaks
    parts = [
        args.level,                      # the level leads the name: it is the claim
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
    if getattr(args, "fixed_order_ap", None):
        parts.append("apfixorder")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"PPO_{timestamp}_{'_'.join(parts)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    citation, deviations = ("", [])
    if args.config:
        citation, deviations = apply_config(args)
        print(f"CONFIG: {citation}" + (f"  deviations: {'; '.join(deviations)}"
                                       if deviations else "  (no deviations)"))
    if args.fixed_order_ap:
        # the wrapper IS an order_protection env with the order taken over, so
        # the args log must say so. Leaving action_mode at its default would
        # record a `seq_mask` run that never existed — an archive that cannot
        # tell two environments apart is the phantom-base failure (F44) with
        # the sign flipped, and `run.provenance` reads exactly this field.
        args.action_mode = "order_protection"
    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    # §8.6: the derivation is CHECKED at launch, not printed at it — before any
    # artifact exists. `episode_len` is in AGENT steps, because the rollout
    # floors count agent steps and a heterogeneous period is `1 + n_alloc` of
    # them (the homogeneous branch is exactly one).
    assert_l1_current(
        args, derived=_L1_DERIVED, basis=_L1_BASIS,
        episode_len=scenario.N * agent_steps_per_period(
            scenario, args.action_mode))
    # ... and the level is DERIVED from the knobs that moved, beside the one the
    # caller authored. A WARNING rather than a refusal, for one structural
    # reason: `mdp_tuning` drives this script with `--level` at its default and
    # every trial past the warm start moves an hp knob, so refusing here would
    # kill every study at launch — and a trial is a search run by construction,
    # which is a fact about the driver, not a mislabelling by the operator.
    derived_level = derive_level(args)
    if derived_level != args.level:
        print(f"LEVEL: --level says {args.level!r}, the knobs say "
              f"{derived_level!r} — the args log records BOTH (§8.6: read the "
              f"level off what moved, never off what produced it)")
    else:
        print(f"LEVEL: {derived_level} — authored and derived agree")

    run_name = build_run_name(args)
    outdir   = (Path(__file__).resolve().parent / args.outdir / args.scenario_name / run_name).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / f"{args.scenario_name}_ppo_args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        # spec §8.4 provenance set: the resolved class, the SB3 version, and the
        # frozen IR this run trained against — literal keys, so a checker reads
        # them without knowing what any domain flag means
        f.write("algo_class: MaskablePPO\n")
        f.write(f"solve_level: {args.level}\n")
        # the level the KNOBS say (§8.6). Recorded beside the authored one so a
        # run directory answers "which layers did this open" without anyone
        # having to trust the flag — the archive before this key carries 583
        # runs whose two answers differ (F42)
        f.write(f"derived_level: {derived_level}\n")
        # §CONFIG-REGISTRY: the base this run was configured FROM, which the
        # address cannot say and the run name does not carry (F43-F46)
        f.write(f"config_ids: {citation or '-'}\n")
        f.write(f"config_deviations: {'; '.join(deviations) or '-'}\n")
        f.write(f"sb3_version: {stable_baselines3.__version__} (sb3_contrib {sb3_contrib.__version__})\n")
        f.write(f"ir_mdp_fingerprint: {_ir_fingerprint()}\n")

    print(f"observation_mode={args.observation_mode}  action_mode={args.action_mode}  "
          f"reward_mode={args.reward_mode}  outdir={outdir}")

    def _make(rank: int):
        def _init():
            log = (str(outdir / "train_log")
                   if (args.gym_log and rank == 0) else None)   # F28/F29
            if args.fixed_order_ap:
                from adi_flex_fixed_order import FixedOrderProtectEnv
                e = FixedOrderProtectEnv(
                    scenario=scenario, ap_table=args.fixed_order_ap,
                    observation_mode=args.observation_mode,
                    reward_mode=args.reward_mode, logger_filename=log,
                )
            else:
                e = AdiFlexEnv(
                    scenario=scenario,
                    observation_mode=args.observation_mode,
                    action_mode=args.action_mode,
                    reward_mode=args.reward_mode,
                    logger_filename=log,
                )
            return Monitor(e, filename=str(outdir / f"monitor{rank}"))
        return _init

    env = DummyVecEnv([_make(i) for i in range(args.n_envs)])
    # VecNormalize is always wrapped; the two normalizations are what switch.
    # L0 is `--no_norm_obs --no_norm_reward` (spec §8.6), which is behaviourally
    # identical to omitting the wrapper — with `norm_obs=False` SB3 returns the
    # observation untouched, so `clip_obs` never applies either.
    # gamma MUST be passed: VecNormalize's return accumulator defaults to 0.99
    # and would otherwise normalize against a discount this run does not use.
    env = VecNormalize(env, norm_obs=args.norm_obs, norm_reward=args.norm_reward,
                       clip_obs=args.vecnorm_clip_obs, gamma=args.gamma)

    lr   = (_linear(args.learning_rate, args.lr_final)
            if args.lr_final is not None else args.learning_rate)
    clip = (_linear(args.clip_init, args.clip_final)
            if args.clip_final is not None else args.clip_init)

    if args.order_head == "ordinal":
        from adi_flex_gym import PROTECT_MODES
        assert args.action_mode in PROTECT_MODES and not args.fixed_order_ap, (
            "--order-head ordinal assumes head 0 is the order quantity of a "
            "one-shot MultiDiscrete mode (PROTECT_MODES, no fixed-order wrap); "
            f"got action_mode={args.action_mode!r}")
        from adi_flex_ordinal_head import OrdinalMaskablePolicy
        policy_cls = OrdinalMaskablePolicy
    else:
        policy_cls = "MlpPolicy"

    model = MaskablePPO(
        policy=policy_cls,
        env=env,
        learning_rate=lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=clip,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        normalize_advantage=args.normalize_advantage,
        policy_kwargs={"net_arch": list(args.net_arch)},
        verbose=1,
        tensorboard_log=str(outdir),
        seed=args.seed,
        device="cpu",
    )

    callback = None
    if args.checkpoint_every_frac > 0:
        # in AGENT steps, and CheckpointCallback counts calls per env, so the
        # rollout width divides out of the requested fraction
        every = max(1, int(args.total_timesteps * args.checkpoint_every_frac
                           / max(1, args.n_envs)))
        callback = CheckpointCallback(
            save_freq=every, save_path=str(outdir / "checkpoints"),
            name_prefix=f"{args.scenario_name}_ppo",
            save_vecnormalize=True,
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=args.progress_bar,
    )

    # spec sec 8.4: the TERMINAL artifact, deliberately named `_ppo_final` —
    # it is never the deliverable; the screen over `checkpoints/` selects that
    model_path = outdir / f"{args.scenario_name}_ppo_final.zip"
    model.save(str(model_path))
    print(f"saved terminal model -> {model_path}")
    if True:
        vecnorm_path = outdir / "vecnormalize.pkl"
        env.save(str(vecnorm_path))
        print(f"saved vecnormalize   -> {vecnorm_path}")
    env.close()


if __name__ == "__main__":
    main()

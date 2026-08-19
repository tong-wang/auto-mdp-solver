"""Algorithm-level Optuna search spaces.

Each space is defined once per algorithm — broad, RL-Baselines3-Zoo-style
ranges over that algorithm's knobs — and intersected at runtime with the
flags the domain's train script actually exposes. This moves the judgment
from "what matters for this problem" (per-domain, arbitrary) to "what are
the algorithm's knobs and their plausible ranges" (fixed once, reusable).

The space obeys the derivation↔tuning contract (SOLVE_LEVELS_PLAN §3,
"center and radius"): the L1-derived config is the center, the space is the
radius and must contain it —

- **containment**: γ's range includes β exactly (`gamma_at_beta`), so the
  derived γ=β is always expressible;
- **tiers**: priority is membership, not weights. `core` is always worth
  tuning; `breadth` needs a larger trial budget; `all` adds the knobs that
  are frozen at the derived value by default (clip, max_grad_norm, γ).
  Extractor knobs (embed_dim/features_dim/channels) sit in `core` because
  they are exposure-gated: only arch-capable train scripts expose them;
- **one-DOF schedules**: schedule pairs are derived, never tuned jointly —
  `lr_final = learning_rate/10`, `clip_final = clip_init/4` (`DERIVED`);
- **encode**: the inverse mapping (script-default cfg → trial params) that
  lets the driver enqueue the L1 config as the warm-start trial 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import optuna


# net_arch is drawn as two ORDERED axes rather than one categorical over named
# shapes. Three reasons, in order of force: (1) a categorical's choice tuple is
# part of its identity, so growing the old four-label list made every existing
# study unresumable ("CategoricalDistribution does not support dynamic value
# space") — retiring the categorical and introducing two new params does not,
# and an int range can be widened later without the same break; (2) ordered
# axes let TPE generalize "wider helped" instead of learning 12 unrelated
# labels, which is what makes a 12-shape grid searchable at core-tier budget;
# (3) width and depth have different standing in §8.6 — see NET_DEPTH_RANGE.
# The train-script dest stays `net_arch`, so --fix and --train-arg are unmoved.
# Dropped in the move: (400, 300), the one non-uniform shape (TD3/SAC heritage,
# no domain here defaults to it); --train-arg net_arch=400,300 still pins it.
NET_WIDTH_LOG2: tuple[int, int] = (5, 8)      # 32, 64, 128, 256
NET_DEPTH_RANGE: tuple[int, int] = (2, 4)

CHANNELS_CHOICES: dict[str, tuple[int, ...]] = {
    "small": (32, 64),
    "wide":  (64, 128),
    "deep":  (32, 64, 64),
}

CLIP_INIT_CHOICES = [0.1, 0.2, 0.3, 0.4]
N_EPOCHS_CHOICES = [4, 10, 20]
MAX_GRAD_NORM_CHOICES = [0.3, 0.5, 1.0, 5.0]
EMBED_DIM_CHOICES = [4, 8, 16, 32]
FEATURES_DIM_CHOICES = [64, 128, 256]


def n_steps_log2_low(min_n_steps: int | None) -> int:
    """Lower log2 bound for n_steps, raised by the domain's rollout floor
    (≥ min_rollout_episodes·T̄/n_envs transitions per env), capped at the
    space's own upper bound."""
    lo = 8
    if min_n_steps:
        lo = max(lo, math.ceil(math.log2(min_n_steps)))
    return min(lo, 12)


def sample_ppo(trial: optuna.Trial, tunable: set[str], *,
               n_envs: int = 1, min_n_steps: int | None = None,
               beta: float = 1.0, tier: str = "all",
               net_depth_default: int = 2) -> dict[str, object]:
    """Draw one PPO configuration; only dests present in *tunable* are set.

    ``beta`` is the problem's intrinsic discount factor
    (``objective.discount_factor``): γ is sampled in (β−0.05, β] with β
    itself reachable — γ > β is never sampled (strictly dominated: more
    bias against the objective *and* more variance).
    """
    cfg: dict[str, object] = {}

    def put(dest: str, draw: Callable[[], object]) -> None:
        if dest in tunable:
            cfg[dest] = draw()

    def _draw_gamma() -> float:
        if trial.suggest_categorical("gamma_at_beta", [True, False]):
            return round(beta, 6)
        return round(beta - trial.suggest_float(
            "beta_minus_gamma", 1e-4, 0.05, log=True), 6)

    put("learning_rate", lambda: trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True))
    put("ent_coef",      lambda: trial.suggest_float("ent_coef", 1e-8, 0.1, log=True))
    # gamma / gae_lambda sampled as (β-x) / (1-x) so the log scale resolves
    # the interesting region near the ceiling
    put("gamma",         _draw_gamma)
    # Range is deliberately wide and ABSOLUTE — never coupled to T̄. Two
    # campaigns measured opposite optima (mab #E35: λ≈0.99+, long deferred
    # credit; game2048 #E34/#E38: λ≈0.90-0.95 won, dense reward + spawn
    # noise), so no episode-relative floor is imposed: λ's optimum tracks
    # where consequences realize and what the noise costs, and the tuner
    # searches the full range. Floor 5e-4 → λ ≤ 0.9995 (credit horizon up
    # to ~2000 steps), so long-horizon values are inside the space — the
    # old 0.01 floor capped the horizon at 100 steps and made mab's
    # hand-set λ=0.994 unreachable AND un-enqueueable as a warm start.
    put("gae_lambda",    lambda: round(1.0 - trial.suggest_float("one_minus_gae_lambda", 5e-4, 0.2, log=True), 6))
    put("clip_init",     lambda: trial.suggest_categorical("clip_init", CLIP_INIT_CHOICES))
    put("n_steps",       lambda: 2 ** trial.suggest_int(
        "log2_n_steps", n_steps_log2_low(min_n_steps), 12))
    put("n_epochs",      lambda: trial.suggest_categorical("n_epochs", N_EPOCHS_CHOICES))
    put("vf_coef",       lambda: trial.suggest_float("vf_coef", 0.2, 1.0))
    put("max_grad_norm", lambda: trial.suggest_categorical("max_grad_norm", MAX_GRAD_NORM_CHOICES))
    def _draw_net_arch() -> tuple[int, ...]:
        width = 2 ** trial.suggest_int("log2_net_width", *NET_WIDTH_LOG2)
        # Depth is a breadth-tier axis. §8.6's L1 rule derives a *width* from
        # obs dim and says nothing about layer count, so at core the
        # derivation's own depth stands and the search corrects only the number
        # the derivation actually produced; opening depth as well would triple
        # core's arch cardinality, which a ~25-trial tier cannot pay for.
        depth = (net_depth_default if tier == "core"
                 else trial.suggest_int("net_depth", *NET_DEPTH_RANGE))
        return (width,) * depth

    put("net_arch",      _draw_net_arch)
    # equivariant-head extractor knob — only exposed by equi-capable train scripts
    put("embed_dim",     lambda: trial.suggest_categorical("embed_dim", EMBED_DIM_CHOICES))
    # CNN-extractor knobs (spec §8.5) — only exposed by CNN-capable train scripts
    put("features_dim",  lambda: trial.suggest_categorical("features_dim", FEATURES_DIM_CHOICES))
    # NOTE: use the dict's INSERTION order (size-ordered literal), NOT sorted().
    # Optuna keys a categorical by its ordered choices tuple and refuses to
    # resume a study whose recorded order differs — the first suggest raises
    # "ValueError: CategoricalDistribution does not support dynamic value
    # space" (nothing is corrupted; the study resumes once the original order is
    # restored). sorted() once reordered net_arch's labels to alphabetical and
    # made every pre-existing study unresumable. Corollary: never reorder or
    # insert entries in this dict once studies exist — growing a categorical is
    # a breaking change, which is why net_arch became two int axes instead. The
    # enqueue_trial warm start uses external values and is order-independent.
    put("channels",      lambda: CHANNELS_CHOICES[
        trial.suggest_categorical("channels", list(CHANNELS_CHOICES))])

    # batch_size last so it can respect the sampled n_steps and the domain's
    # structural n_envs (rollout buffer = n_steps × n_envs)
    if "batch_size" in tunable:
        batch = 2 ** trial.suggest_int("log2_batch_size", 5, 9)
        if "n_steps" in cfg:
            batch = min(batch, cfg["n_steps"] * max(1, n_envs))
        cfg["batch_size"] = batch

    return cfg


def encode_ppo(cfg: dict[str, object], *, n_envs: int = 1,
               min_n_steps: int | None = None,
               beta: float = 1.0, tier: str = "all",
               net_depth_default: int = 2) -> tuple[dict[str, object], list[str]]:
    """Inverse of :func:`sample_ppo`: map a concrete config (typically the
    train script's own defaults = the L1-derived center) to trial params for
    ``study.enqueue_trial``. Returns ``(params, skipped)`` — knobs whose value
    is not representable in the space are skipped (Optuna samples them)."""
    params: dict[str, object] = {}
    skipped: list[str] = []

    def _log2_exact(v: object) -> int | None:
        try:
            exp = int(math.log2(int(v)))
        except (TypeError, ValueError, OverflowError):
            return None
        return exp if 2 ** exp == int(v) else None

    for dest, value in cfg.items():
        if dest == "learning_rate":
            if isinstance(value, (int, float)) and 1e-5 <= value <= 3e-3:
                params["learning_rate"] = float(value)
            else:
                skipped.append(dest)
        elif dest == "ent_coef":
            if isinstance(value, (int, float)) and 1e-8 <= value <= 0.1:
                params["ent_coef"] = float(value)
            else:
                skipped.append(dest)
        elif dest == "gamma":
            if not isinstance(value, (int, float)):
                skipped.append(dest)
            elif abs(value - beta) < 1e-9:
                params["gamma_at_beta"] = True
            elif 1e-4 <= beta - value <= 0.05:
                params["gamma_at_beta"] = False
                params["beta_minus_gamma"] = round(beta - value, 6)
            else:
                skipped.append(dest)
        elif dest == "gae_lambda":
            om = round(1.0 - value, 6) if isinstance(value, (int, float)) else None
            if om is not None and 5e-4 <= om <= 0.2:
                params["one_minus_gae_lambda"] = om
            else:
                skipped.append(dest)
        elif dest == "clip_init":
            if value in CLIP_INIT_CHOICES:
                params["clip_init"] = value
            else:
                skipped.append(dest)
        elif dest == "n_steps":
            exp = _log2_exact(value)
            if exp is not None and n_steps_log2_low(min_n_steps) <= exp <= 12:
                params["log2_n_steps"] = exp
            else:
                skipped.append(dest)
        elif dest == "n_epochs":
            if value in N_EPOCHS_CHOICES:
                params["n_epochs"] = value
            else:
                skipped.append(dest)
        elif dest == "vf_coef":
            if isinstance(value, (int, float)) and 0.2 <= value <= 1.0:
                params["vf_coef"] = float(value)
            else:
                skipped.append(dest)
        elif dest == "max_grad_norm":
            if value in MAX_GRAD_NORM_CHOICES:
                params["max_grad_norm"] = value
            else:
                skipped.append(dest)
        elif dest == "net_arch":
            shape = tuple(value) if isinstance(value, (list, tuple)) else ()
            exp = _log2_exact(shape[0]) if shape else None
            # depth is only range-checked where it is searched: at core the
            # script's own depth stands whatever it is, so refusing to encode it
            # would strand the warm start over an axis core never touches
            depth_ok = tier == "core" or (
                NET_DEPTH_RANGE[0] <= len(shape) <= NET_DEPTH_RANGE[1])
            if (shape and len(set(shape)) == 1 and depth_ok and exp is not None
                    and NET_WIDTH_LOG2[0] <= exp <= NET_WIDTH_LOG2[1]):
                params["log2_net_width"] = exp
                if tier != "core":
                    params["net_depth"] = len(shape)
            else:
                skipped.append(dest)
        elif dest == "embed_dim":
            if value in EMBED_DIM_CHOICES:
                params["embed_dim"] = value
            else:
                skipped.append(dest)
        elif dest == "features_dim":
            if value in FEATURES_DIM_CHOICES:
                params["features_dim"] = value
            else:
                skipped.append(dest)
        elif dest == "channels":
            label = next((k for k, v in CHANNELS_CHOICES.items()
                          if tuple(v) == tuple(value)), None) \
                if isinstance(value, (list, tuple)) else None
            if label:
                params["channels"] = label
            else:
                skipped.append(dest)
        elif dest == "batch_size":
            exp = _log2_exact(value)
            if exp is not None and 5 <= exp <= 9:
                params["log2_batch_size"] = exp
            else:
                skipped.append(dest)
        else:
            skipped.append(dest)
    return params, skipped


# The tiers are budget scopes, and membership follows one question: how much
# does the L1 derivation already know? core corrects the knobs whose derived
# value is a real guess; breadth opens what the derivation does not produce at
# all; the frozen tier is what stays at its derived value unless asked for.
PPO_TIER_CORE: tuple[str, ...] = (
    # gae_lambda earns core on evidence, not symmetry: §8.6 gives it the longest
    # rule in the L1 table, two campaigns bracket its optimum an order of
    # magnitude apart (mab #E35, where short credit left 2 of 3 seeds unable to
    # learn at all; game2048 #E34/#E38), and the spec states outright that it is
    # per-instance and never a transferable constant.
    "learning_rate", "net_arch", "n_steps", "ent_coef", "gae_lambda",
    # exposure-gated extractor knobs: only arch-capable scripts expose them,
    # and when they do, tuning them is the point
    "embed_dim", "features_dim", "channels",
)
PPO_TIER_BREADTH: tuple[str, ...] = PPO_TIER_CORE + (
    # + net_arch's DEPTH axis, which core holds at the script's own value
    "n_epochs", "batch_size", "vf_coef",
    # gamma is bounded by the problem: the sampler draws in (β-0.05, β] and can
    # never exceed β, so opening it buys a logged bias-variance move (§8.6),
    # not a free parameter
    "gamma",
)
PPO_KNOBS: tuple[str, ...] = PPO_TIER_BREADTH + (
    # frozen tier: held at the derived value unless explicitly opened
    "clip_init", "max_grad_norm",
)

# knobs that are *deliberately* absent from plain-MLP train scripts — their
# absence is the arch-layer boundary at work, not template rot, so the
# unmatched-knob lint stays silent about them
OPTIONAL_KNOBS: frozenset[str] = frozenset(
    {"embed_dim", "features_dim", "channels"})

# one-degree-of-freedom schedule pairs: dest -> (source knob, factor).
# Applied by the driver when the train script exposes the dest — schedule
# finals are derived, never tuned, so a schedule can never invert.
PPO_DERIVED: dict[str, tuple[str, float]] = {
    "lr_final":   ("learning_rate", 0.1),
    "clip_final": ("clip_init", 0.25),
}


@dataclass(frozen=True)
class Space:
    """One algorithm's search space: the knob dests it covers + the sampler."""

    knobs: tuple[str, ...]
    sample: Callable[..., dict[str, object]]
    tiers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    derived: dict[str, tuple[str, float]] = field(default_factory=dict)
    encode: Callable[..., tuple[dict[str, object], list[str]]] | None = None


SPACES: dict[str, Space] = {
    "ppo": Space(
        knobs=PPO_KNOBS,
        sample=sample_ppo,
        tiers={"core": PPO_TIER_CORE,
               "breadth": PPO_TIER_BREADTH,
               "all": PPO_KNOBS},
        derived=PPO_DERIVED,
        encode=encode_ppo,
    ),
}

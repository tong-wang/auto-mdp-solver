"""Algorithm-level Optuna search spaces.

Each space is defined once per algorithm — broad, RL-Baselines3-Zoo-style
ranges over that algorithm's knobs — and intersected at runtime with the
flags the domain's train script actually exposes. This moves the judgment
from "what matters for this problem" (per-domain, arbitrary) to "what are
the algorithm's knobs and their plausible ranges" (fixed once, reusable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import optuna


def sample_ppo(trial: optuna.Trial, tunable: set[str]) -> dict[str, object]:
    """Draw one PPO configuration; only dests present in *tunable* are set."""
    cfg: dict[str, object] = {}

    def put(dest: str, draw: Callable[[], object]) -> None:
        if dest in tunable:
            cfg[dest] = draw()

    put("learning_rate", lambda: trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True))
    put("ent_coef",      lambda: trial.suggest_float("ent_coef", 1e-8, 0.1, log=True))
    # gamma / gae_lambda sampled as 1-x so the log scale resolves the
    # interesting region near 1.0
    put("gamma",         lambda: round(1.0 - trial.suggest_float("one_minus_gamma", 1e-4, 0.05, log=True), 6))
    put("gae_lambda",    lambda: round(1.0 - trial.suggest_float("one_minus_gae_lambda", 0.01, 0.2, log=True), 6))
    put("clip_range",    lambda: trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4]))
    put("n_steps",       lambda: 2 ** trial.suggest_int("log2_n_steps", 8, 12))
    put("n_epochs",      lambda: trial.suggest_categorical("n_epochs", [4, 10, 20]))
    put("vf_coef",       lambda: trial.suggest_float("vf_coef", 0.2, 1.0))
    put("max_grad_norm", lambda: trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1.0, 5.0]))
    put("net_arch",      lambda: {
        "small":  (64, 64),
        "medium": (256, 256),
        "large":  (400, 300),
        "deep":   (128, 128, 128),
    }[trial.suggest_categorical("net_arch", ["small", "medium", "large", "deep"])])
    # equivariant-head extractor knob — only exposed by equi-capable train scripts
    put("embed_dim",     lambda: trial.suggest_categorical("embed_dim", [4, 8, 16, 32]))
    # CNN-extractor knobs (spec §8.5) — only exposed by CNN-capable train scripts
    put("features_dim",  lambda: trial.suggest_categorical("features_dim", [64, 128, 256]))
    put("channels",      lambda: {
        "small": (32, 64),
        "wide":  (64, 128),
        "deep":  (32, 64, 64),
    }[trial.suggest_categorical("channels", ["small", "wide", "deep"])])

    # batch_size last so it can respect the sampled n_steps (single-env rollouts)
    if "batch_size" in tunable:
        batch = 2 ** trial.suggest_int("log2_batch_size", 5, 9)
        if "n_steps" in cfg:
            batch = min(batch, cfg["n_steps"])
        cfg["batch_size"] = batch

    return cfg


PPO_KNOBS: tuple[str, ...] = (
    "learning_rate", "ent_coef", "gamma", "gae_lambda", "clip_range",
    "n_steps", "n_epochs", "vf_coef", "max_grad_norm", "net_arch",
    "embed_dim", "features_dim", "channels", "batch_size",
)


@dataclass(frozen=True)
class Space:
    """One algorithm's search space: the knob dests it covers + the sampler."""

    knobs: tuple[str, ...]
    sample: Callable[[optuna.Trial, set[str]], dict[str, object]]


SPACES: dict[str, Space] = {
    "ppo": Space(knobs=PPO_KNOBS, sample=sample_ppo),
}

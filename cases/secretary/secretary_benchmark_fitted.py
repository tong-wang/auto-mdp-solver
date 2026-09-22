"""Deployable skip/record rule recovered from a PPO action surface."""

from __future__ import annotations

import json
from pathlib import Path

from secretary_benchmark_threshold import ThresholdPolicy
from secretary_scenarios import SecretaryScenario


def load_policy(fit_path: str | Path) -> ThresholdPolicy:
    payload = json.loads(Path(fit_path).read_text())
    return ThresholdPolicy(int(payload["fitted_skip_count"]))


def make_policy_from_file(fit_path: str | Path):
    def make_policy(scenario: SecretaryScenario, episode_seed: int) -> ThresholdPolicy:
        del scenario, episode_seed
        return load_policy(fit_path)
    return make_policy

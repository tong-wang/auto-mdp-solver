"""Deployable wrapper for the selected secretary PPO policy.

Observation contract (``observation_mode='relative'``)
------------------------------------------------------
Supply a length-2 numeric vector in this exact order:

1. ``time_to_go`` — candidates remaining including the current candidate,
   in candidates, from 100 down to 1;
2. ``relative_rank`` — the current candidate's one-based rank among all
   candidates observed so far (1 means a record).

``act(obs)`` returns the discrete ``accept`` decision: 0 rejects and 1 accepts.
The wrapper loads the selected SB3 model on CPU and applies the saved
VecNormalize observation statistics before deterministic inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from secretary_gym import SecretaryEnv
from secretary_scenarios import SCENARIOS, SecretaryScenario, SecretaryScenarioSource


def _read_args_log(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def _realize_scenario(
    source: SecretaryScenarioSource,
    episode_seed: int,
) -> SecretaryScenario:
    """Realize the registered source before calling the raw MDP layer."""
    return source(int(episode_seed))


class SecretaryPolicy:
    """Stable inference interface around the selected trained artifact."""

    def __init__(
        self,
        model_path: str | Path,
        vecnorm_path: str | Path | None = None,
        action_mode: str = "accept",
        observation_mode: str = "relative",
        scenario: str = "standard",
    ) -> None:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}")
        if action_mode != "accept" or observation_mode != "relative":
            raise ValueError(
                "the selected policy requires action_mode='accept' and "
                "observation_mode='relative'"
            )
        self.model_path = Path(model_path).resolve()
        self.vecnorm_path = (
            Path(vecnorm_path).resolve()
            if vecnorm_path is not None
            else self.model_path.parent / "vecnormalize.pkl"
        )
        if not self.vecnorm_path.exists():
            raise FileNotFoundError(f"missing trained observation statistics: {self.vecnorm_path}")
        args_path = self.model_path.parent / f"{scenario}_ppo_args.txt"
        trained = _read_args_log(args_path)
        expected = {
            "scenario_name": scenario,
            "action_mode": action_mode,
            "observation_mode": observation_mode,
        }
        mismatches = {
            key: (trained.get(key), value)
            for key, value in expected.items()
            if trained.get(key) != value
        }
        if mismatches:
            raise ValueError(f"requested policy contract does not match training args: {mismatches}")
        self.scenario_name = scenario
        self.scenario_source = SCENARIOS[scenario]
        self.model = PPO.load(self.model_path, device="cpu")
        dummy = DummyVecEnv([lambda: SecretaryEnv(
            scenario=self.scenario_source,
            observation_mode=observation_mode,
            action_mode=action_mode,
            reward_mode="success",
        )])
        self.normalizer = VecNormalize.load(self.vecnorm_path, dummy)
        self.normalizer.training = False
        self.normalizer.norm_reward = False

    def act(self, obs: np.ndarray | list[float] | tuple[float, float]) -> int:
        raw = np.asarray(obs, dtype=np.float32)
        if raw.shape != (2,):
            raise ValueError(f"expected observation shape (2,), got {raw.shape}")
        normalized = self.normalizer.normalize_obs(raw[None, :].copy())
        action, _ = self.model.predict(normalized, deterministic=True)
        return int(action[0])

    def close(self) -> None:
        self.normalizer.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the deployed secretary policy.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vecnorm-path", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    return parser


def main() -> None:
    from secretary_mdp import advance, init_state

    args = _build_arg_parser().parse_args()
    policy = SecretaryPolicy(args.model_path, args.vecnorm_path)
    try:
        for episode_seed in range(args.episodes):
            scenario = _realize_scenario(policy.scenario_source, episode_seed)
            state, _ = init_state(scenario, episode_seed)
            while not state.terminated:
                obs = [
                    scenario.n_candidates - state.period,
                    state.relative_rank,
                ]
                state, info = advance(scenario, state, policy.act(obs))
            print(
                f"seed={episode_seed} steps={state.period} "
                f"selected_rank={info['selected_rank']} "
                f"success={info['outcome']['success']:.0f}"
            )
    finally:
        policy.close()


if __name__ == "__main__":
    main()

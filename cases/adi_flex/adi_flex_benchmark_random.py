"""Random benchmark: uniform over the action space.

The mandatory floor. It draws both decisions uniformly from the same ranges
the gym exposes to the agent, so "the learned policy beats random" means it
beat the space it was actually searching rather than some other space.

Its randomness is seeded per episode from the episode seed, on a stream well
clear of the demand streams (1, 2, 3), so the whole benchmark is reproducible
and never correlated with the demand it is reacting to.
"""

from __future__ import annotations

import argparse

import numpy as np

from adi_flex_scenarios import SCENARIOS, AdiFlexScenario
from adi_flex_mdp import AdiFlexState
from adi_flex_gym import HOLD_BACK_MAX, ORDER_MAX

_POLICY_STREAM = 101   # well clear of the demand streams (1, 2, 3)


class RandomPolicy:
    """Uniform over MultiDiscrete([ORDER_MAX + 1, HOLD_BACK_MAX + 1])."""

    def __init__(self, seed_salt: int = 0) -> None:
        self.seed_salt = seed_salt
        self._rng = np.random.default_rng(0)

    def reset(self) -> None:
        # re-seeded per episode by the rollout via seed_episode()
        pass

    def seed_episode(self, episode_seed: int) -> None:
        self._rng = np.random.default_rng(
            np.random.SeedSequence([_POLICY_STREAM, episode_seed, self.seed_salt])
        )

    def act(
        self,
        scenario: AdiFlexScenario,
        state: AdiFlexState,
        vhat: int,
    ) -> tuple[int, int]:
        return (int(self._rng.integers(0, ORDER_MAX + 1)),
                int(self._rng.integers(0, HOLD_BACK_MAX + 1)))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Uniform-random benchmark.")
    p.add_argument("-s", "--scenario_name", type=str, default="exp4")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


if __name__ == "__main__":
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    pol = RandomPolicy()
    pol.seed_episode(0)
    state = None
    draws = [pol.act(scenario, state, 0) for _ in range(5)]
    print(f"scenario : {args.scenario_name}  ({scenario.desc})")
    print(f"action space : order 0..{ORDER_MAX}, hold_back 0..{HOLD_BACK_MAX}")
    print(f"sample draws : {draws}")

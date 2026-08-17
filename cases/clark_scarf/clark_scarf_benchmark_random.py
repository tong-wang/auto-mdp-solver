"""Random benchmark — the floor of the leaderboard.

Ships a uniform-random quantity on every link each period, clipped to what the
source installation holds. It exists to bound the reward scale from below: an
RL arm that fails to beat this is a build bug, not a tuning problem (spec §8.6).

Seeded off the episode seed so the arm is reproducible and replays the same
common-random-number block as every other arm.

Usage:
    python clark_scarf_benchmark_random.py -s n3_l2_p09
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from clark_scarf_mdp import ship_capacity
from clark_scarf_scenarios import SCENARIOS, ClarkScarfScenario

# stream id for the benchmark's own randomness: distinct from every model
# stream (demand is source 0), so drawing actions can never perturb the
# realized demand path an arm is scored on
_POLICY_STREAM = 9_001


@dataclass(slots=True)
class RandomShipper:
    """Uniform-random shipments within the feasible per-link box.

    Stateless: the RNG is derived from ``(period, episode_seed, seed_salt)`` on
    each call, in the same leaf-first shape as the model's own v2 keys. So the
    arm is exactly reproducible, carries no mutable state across episodes, and
    can be evaluated in any batching order without changing a single action.
    """

    scenario: ClarkScarfScenario

    def actions(self, state) -> np.ndarray:
        sc = self.scenario
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(state.period), _POLICY_STREAM,
                 int(state.episode_seed), sc.seed_salt]
            )
        )
        caps = ship_capacity(sc, state)
        return np.array(
            [float(rng.integers(0, int(caps[k]) + 1)) for k in range(sc.n_echelons)]
        )


def _cli() -> None:
    p = argparse.ArgumentParser(description="Random benchmark (no table to solve).")
    p.add_argument("-s", "--scenario_name", default="n3_l2_p09",
                   choices=list(SCENARIOS.keys()))
    args = p.parse_args()
    sc = SCENARIOS[args.scenario_name]
    print(
        f"[random] no solution table; uniform shipments in [0, capacity] "
        f"per link, N={sc.n_echelons} L={sc.leadtime}"
    )


if __name__ == "__main__":
    _cli()

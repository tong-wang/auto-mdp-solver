"""Exact finite-horizon backward induction for the secretary problem.

Before position ``t``, let V[t] be the optimal success probability.  A current
record occurs with probability 1/t; accepting it wins with conditional
probability t/n.  A non-record can never win.  Therefore

    V[t] = V[t+1] + (1/t) * max(t/n - V[t+1], 0).

The resulting policy accepts a record exactly when t/n >= V[t+1].  This
independently recovers the Gilbert-Mosteller optimal cutoff.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from secretary_mdp import SecretaryState
from secretary_scenarios import SCENARIOS, SecretaryScenario


@dataclass(frozen=True, slots=True)
class DpSolution:
    n_candidates: int
    value: tuple[float, ...]
    accept_record: tuple[bool, ...]

    @property
    def skip_count(self) -> int:
        accepted = [t for t in range(1, self.n_candidates) if self.accept_record[t]]
        return min(accepted) - 1


@lru_cache(maxsize=None)
def solve(n_candidates: int) -> DpSolution:
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2")
    value = [0.0] * (n_candidates + 2)
    accept_record = [False] * (n_candidates + 1)
    value[n_candidates] = 1.0 / n_candidates
    for position in range(n_candidates - 1, 0, -1):
        continuation = value[position + 1]
        stop_value = position / n_candidates
        accept_record[position] = stop_value >= continuation
        value[position] = continuation + max(stop_value - continuation, 0.0) / position
    return DpSolution(n_candidates, tuple(value), tuple(accept_record))


class DpPolicy:
    def __init__(self, solution: DpSolution) -> None:
        self.solution = solution

    def act(self, state: SecretaryState) -> int:
        position = state.period + 1
        return int(
            position < self.solution.n_candidates
            and state.relative_rank == 1
            and self.solution.accept_record[position]
        )


def make_policy(scenario: SecretaryScenario, episode_seed: int) -> DpPolicy:
    del episode_seed
    return DpPolicy(solve(scenario.n_candidates))


def write_solution(scenario_name: str) -> Path:
    scenario = SCENARIOS[scenario_name]
    solution = solve(scenario.n_candidates)
    path = (Path(__file__).resolve().parent / "results" / scenario_name
            / "benchmark" / "dp" / f"{scenario_name}.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("position\tvalue_before\taccept_if_record\n")
        for position in range(1, scenario.n_candidates + 1):
            stream.write(
                f"{position}\t{solution.value[position]:.15f}\t"
                f"{int(solution.accept_record[position])}\n"
            )
    print(f"n={scenario.n_candidates} skip={solution.skip_count} "
          f"optimal_success={solution.value[1]:.12f}")
    print(f"solution saved -> {path}")
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve the secretary DP.")
    parser.add_argument("-s", "--scenario_name", default="standard", choices=SCENARIOS)
    return parser


def main() -> None:
    write_solution(_build_arg_parser().parse_args().scenario_name)


if __name__ == "__main__":
    main()

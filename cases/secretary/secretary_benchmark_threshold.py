"""Finite-N optimal threshold rule for the classical secretary problem.

For ``r`` initially rejected candidates, the exact success probability is

    p(r) = (r / n) * sum(1/k for k=r..n-1),  r > 0,

with p(0)=1/n.  Maximizing this expression gives r=37 for n=100.  This is the
no-information optimum described by Gilbert & Mosteller (1966),
doi:10.1080/01621459.1966.10502008.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

from secretary_mdp import SecretaryState
from secretary_scenarios import SecretaryScenario


def threshold_success_probability(n_candidates: int, skip: int) -> Fraction:
    if not 0 <= skip < n_candidates:
        raise ValueError("skip must be in [0, n_candidates)")
    if skip == 0:
        return Fraction(1, n_candidates)
    harmonic_tail = sum(
        (Fraction(1, position) for position in range(skip, n_candidates)),
        Fraction(0, 1),
    )
    return Fraction(skip, n_candidates) * harmonic_tail


@lru_cache(maxsize=None)
def optimal_skip_count(n_candidates: int) -> int:
    return max(
        range(n_candidates),
        key=lambda skip: threshold_success_probability(n_candidates, skip),
    )


class ThresholdPolicy:
    def __init__(self, skip: int) -> None:
        self.skip = int(skip)

    def act(self, state: SecretaryState) -> int:
        return int(state.period >= self.skip and state.relative_rank == 1)


def make_policy(scenario: SecretaryScenario, episode_seed: int) -> ThresholdPolicy:
    del episode_seed
    return ThresholdPolicy(optimal_skip_count(scenario.n_candidates))


if __name__ == "__main__":
    n = 100
    skip = optimal_skip_count(n)
    print(f"n={n} skip={skip} success={float(threshold_success_probability(n, skip)):.12f}")

"""Setting-only scenarios for the classical secretary problem.

A scenario describes the fixed problem configuration.  The realized arrival
order belongs to episode state and is drawn by ``secretary_mdp.init_state``;
it is deliberately absent from this module's scenario objects.
"""

from __future__ import annotations

from dataclasses import dataclass

SEED_SCHEME = "v2"


@dataclass(slots=True, frozen=True)
class SecretaryScenario:
    """Hyperparameters shared by every episode in this setting."""

    n_candidates: int = 100
    seed_salt: int = 4729
    scenario_name: str = "standard"
    desc: str = "100 candidates with a fresh uniformly random order per reset"

    def __post_init__(self) -> None:
        assert self.n_candidates >= 2, "n_candidates must be at least 2"
        assert self.seed_salt >= 1, "seed_salt must be >= 1 under seed scheme v2"

scenario_standard = SecretaryScenario()

SCENARIOS: dict[str, SecretaryScenario] = {
    "standard": scenario_standard,
}


def _check_registry() -> None:
    assert SCENARIOS, "SCENARIOS must not be empty"
    for name, scenario in SCENARIOS.items():
        assert isinstance(scenario, SecretaryScenario), name
        assert scenario.scenario_name == name


_check_registry()


if __name__ == "__main__":
    for name, scenario in SCENARIOS.items():
        print(f"{name}: N={scenario.n_candidates}")

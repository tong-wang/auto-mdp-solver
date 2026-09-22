"""Scenario family and per-episode sampler for the secretary problem."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mdp_ir.runtime import meta_key

SEED_SCHEME = "v2"


@dataclass(slots=True, frozen=True)
class SecretaryScenario:
    """One concrete problem instance realized for an episode."""

    n_candidates: int
    arrival_order: tuple[int, ...] = field(repr=False)
    scenario_name: str
    desc: str

    def __post_init__(self) -> None:
        assert self.n_candidates >= 2, "n_candidates must be at least 2"
        assert len(self.arrival_order) == self.n_candidates
        assert set(self.arrival_order) == set(range(1, self.n_candidates + 1))


@dataclass(slots=True, frozen=True)
class SecretaryScenarioSource:
    """Draw one concrete hidden ranking per episode using the IR meta stream."""

    n_candidates: int = 100
    substream_id: int = 0
    seed_salt: int = field(default=4729, repr=False)
    scenario_name: str = "standard"
    desc: str = "100 candidates with a fresh uniformly random order per reset"

    def __post_init__(self) -> None:
        assert self.n_candidates >= 2, "n_candidates must be at least 2"
        assert self.substream_id >= 0, "substream_id must be non-negative"
        assert self.seed_salt >= 1, "seed_salt must be >= 1 under seed scheme v2"

    def seed_key(self, episode_seed: int) -> list[int]:
        """Return the canonical key for the sampler's single reset-time draw."""
        return meta_key(self.substream_id, int(episode_seed), self.seed_salt)

    def __call__(self, episode_seed: int) -> SecretaryScenario:
        """Realize the episode instance without mutating this source."""
        rng = np.random.default_rng(
            np.random.SeedSequence(
                meta_key(self.substream_id, int(episode_seed), self.seed_salt)
            )
        )
        arrival_order = tuple(
            int(rank)
            for rank in rng.choice(
                np.arange(1, self.n_candidates + 1),
                size=self.n_candidates,
                replace=False,
            )
        )
        return SecretaryScenario(
            n_candidates=self.n_candidates,
            arrival_order=arrival_order,
            scenario_name=self.scenario_name,
            desc=f"{self.desc}; episode_seed={int(episode_seed)}",
        )

source_standard = SecretaryScenarioSource()

SCENARIOS: dict[str, SecretaryScenarioSource] = {
    "standard": source_standard,
}


def _check_registry() -> None:
    assert SCENARIOS, "SCENARIOS must not be empty"
    for name, source in SCENARIOS.items():
        assert isinstance(source, SecretaryScenarioSource), name
        assert source.scenario_name == name
        realized = source(0)
        assert isinstance(realized, SecretaryScenario), name
        assert realized.scenario_name == name


_check_registry()


if __name__ == "__main__":
    for name, source in SCENARIOS.items():
        print(f"{name}: N={source.n_candidates}")

"""Classical secretary core MDP.

This module contains only the domain simulator and does not depend on
Gymnasium. The simulator knows the hidden absolute-rank permutation; the gym
wrapper decides that the policy sees only time-to-go and current relative
rank.

``init_state`` draws the episode's hidden absolute-rank permutation and stores
it in state. Transitions are deterministic thereafter. Absolute ranks and
selected-rank outcomes travel through ``info`` and never enter the policy
observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from secretary_scenarios import SecretaryScenario
from secretary_uncertainty import meta_key


@dataclass(slots=True)
class SecretaryState:
    """Full simulator state; observability is defined in ``secretary_gym``."""

    period: int
    terminated: bool
    relative_rank: int
    selected: int
    episode_seed: int
    arrival_order: tuple[int, ...]

    def copy(self) -> "SecretaryState":
        return SecretaryState(
            period=self.period,
            terminated=self.terminated,
            relative_rank=self.relative_rank,
            selected=self.selected,
            episode_seed=self.episode_seed,
            arrival_order=self.arrival_order,
        )


def _draw_arrival_order(
    scenario: SecretaryScenario,
    episode_seed: int,
) -> tuple[int, ...]:
    """Draw the reset-time permutation using seed scheme v2."""
    # The v0.11.4 IR expresses this reset draw through a world sampler, but the
    # domain deliberately keeps the scenario settings-only and stores the
    # realization in episode state (ESCALATION.md F1).  Keep the draw here at
    # state construction while still using the canonical meta-key boundary.
    rng = np.random.default_rng(np.random.SeedSequence(
        meta_key(0, episode_seed, scenario.seed_salt)
    ))
    return tuple(int(value) for value in rng.choice(
        np.arange(1, scenario.n_candidates + 1),
        size=scenario.n_candidates,
        replace=False,
    ))


def init_state(
    scenario: SecretaryScenario,
    episode_seed: int,
) -> tuple[SecretaryState, dict]:
    """Draw and store a fresh episode instance, before the first candidate."""
    arrival_order = _draw_arrival_order(scenario, episode_seed)
    state = SecretaryState(
        period=0,
        terminated=False,
        relative_rank=1,
        selected=0,
        episode_seed=int(episode_seed),
        arrival_order=arrival_order,
    )
    return state, {
        "action_period": -1,
        "accept": 0,
        "decision_relative_rank": 0,
        "candidate_rank": 0,
        "selected_rank": 0,
        "outcome": {"success": 0.0, "total": 0.0},
    }


def advance(
    scenario: SecretaryScenario,
    state: SecretaryState,
    accept: int,
) -> tuple[SecretaryState, dict]:
    """Apply one reject/accept decision and return the next state plus outcome."""
    assert not state.terminated, "episode already terminated"
    accept = int(accept)
    assert accept in (0, 1), f"accept must be 0 or 1, got {accept}"
    assert 0 <= state.period < scenario.n_candidates

    action_period = state.period
    decision_relative_rank = state.relative_rank
    assert len(state.arrival_order) == scenario.n_candidates
    candidate_rank = int(state.arrival_order[action_period])
    selected = int(accept == 1 or action_period == scenario.n_candidates - 1)
    selected_rank = candidate_rank if selected else 0
    success = float(selected and selected_rank == 1)

    state2 = state.copy()
    state2.period += 1
    state2.selected = selected
    state2.terminated = bool(selected)
    if state2.terminated:
        state2.relative_rank = 0
    else:
        next_rank = state2.arrival_order[state2.period]
        state2.relative_rank = 1 + sum(
            1 for prior in state2.arrival_order[:state2.period]
            if prior < next_rank
        )

    info = {
        "action_period": action_period,
        "accept": accept,
        "decision_relative_rank": decision_relative_rank,
        "candidate_rank": candidate_rank,
        "selected_rank": selected_rank,
        "outcome": {"success": success, "total": success},
    }
    return state2, info


def _run_smoke(episode_seed: int = 3) -> None:
    from secretary_scenarios import scenario_standard

    scenario = scenario_standard
    state, _ = init_state(scenario, episode_seed)
    while not state.terminated:
        # Classical-looking smoke rule: reject first 37, then accept a record.
        accept = int(state.period >= 37 and state.relative_rank == 1)
        state, info = advance(scenario, state, accept)
    print(
        f"steps={state.period} selected_rank={info['selected_rank']} "
        f"success={info['outcome']['success']:.0f}"
    )


if __name__ == "__main__":
    _run_smoke()

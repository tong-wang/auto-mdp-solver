"""Domain tests for the classical secretary problem."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdp_ir.schema import load_ir, ungroup_mdp
from mdp_ir.testing import (
    assert_diverges,
    assert_laws,
    assert_match,
    mutated,
    schema_beside,
)

from secretary_mdp import advance, init_state
from secretary_scenarios import source_standard
from secretary_select import _blocks_overlap, _checkpoints, _vecnorm_for
from secretary_benchmark_dp import solve
from secretary_benchmark_threshold import (
    optimal_skip_count,
    threshold_success_probability,
)

SCHEMA = schema_beside(__file__)


def _compositions() -> list[str | None]:
    raw = json.loads(SCHEMA.read_text())
    mdp = ungroup_mdp(raw["mdp"])
    scenario = mdp["scenario"]
    return [None] + sorted(scenario.get("instances") or {}) + sorted(
        m["name"] for m in scenario.get("mixtures") or []
    )


COMPOSITIONS = _compositions()


def test_covering_set_is_not_empty():
    assert COMPOSITIONS and None in COMPOSITIONS


def test_engine_laws():
    assert_laws(SCHEMA)


@pytest.mark.parametrize("instance", COMPOSITIONS, ids=lambda x: x or "base")
def test_differential_matches_domain(instance):
    assert_match(SCHEMA, instance=instance, episodes=8)


def test_scenario_sampler_is_pure_and_init_copies_realization_to_state():
    scenario_a = source_standard(17)
    scenario_b = source_standard(17)
    scenario_c = source_standard(18)
    a, _ = init_state(scenario_a, 17)
    b, _ = init_state(scenario_b, 17)
    c, _ = init_state(scenario_c, 18)
    assert a.arrival_order == b.arrival_order
    assert a.arrival_order != c.arrival_order
    assert a.arrival_order == scenario_a.arrival_order
    assert set(a.arrival_order) == set(range(1, source_standard.n_candidates + 1))
    assert not hasattr(source_standard, "arrival_order")


def test_scenario_sampler_uses_the_declared_meta_key():
    assert source_standard.seed_key(17) == [
        0,
        0,
        17,
        source_standard.seed_salt,
    ]


def test_selector_pairs_each_checkpoint_with_its_own_normalizer(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    model = checkpoint_dir / "standard_ppo_1800000_steps.zip"
    normalizer = checkpoint_dir / "standard_ppo_vecnormalize_1800000_steps.pkl"
    model.write_bytes(b"model")
    normalizer.write_bytes(b"normalizer")

    assert _checkpoints(tmp_path) == [model]
    assert _vecnorm_for(model) == normalizer
    assert not _blocks_overlap(1_000_000, 2_048, 0, 8_192)
    assert _blocks_overlap(8_000, 2_048, 0, 8_192)


def test_relative_rank_and_forced_final_selection():
    scenario = source_standard(9)
    state, _ = init_state(scenario, 9)
    for _ in range(scenario.n_candidates):
        visible = 1 + sum(
            prior < state.arrival_order[state.period]
            for prior in state.arrival_order[:state.period]
        )
        assert state.relative_rank == visible
        state, info = advance(scenario, state, 0)
    assert state.terminated and state.selected == 1
    assert info["selected_rank"] == state.arrival_order[-1]


def test_a_corrupted_selection_update_diverges():
    def corrupt(doc):
        for transition in doc["mdp"]["dynamics"]["transitions"]:
            if transition["event"] == "SELECT":
                transition["updates"] = [
                    update.replace("selected_rank = candidate_rank", "selected_rank = 0")
                    for update in transition["updates"]
                ]

    assert_diverges(SCHEMA, mutated(load_ir(SCHEMA), corrupt), episodes=2)


def test_literature_threshold_and_dp_agree_exactly_at_n100():
    solution = solve(100)
    skip = optimal_skip_count(100)
    assert skip == 37
    assert solution.skip_count == skip
    assert solution.value[1] == pytest.approx(
        float(threshold_success_probability(100, skip)), abs=1e-15
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

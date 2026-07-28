"""Synthetic IR fixtures for the engine tests.

These tests exercise ``mdp_ir`` itself — the validator, the seed grammar, the
family dispatch, the catalog resolver — so they build their own minimal IRs
rather than borrowing a shipped example. That keeps the harness's own suite
independent of ``plugin/`` (the two published artifacts are disjoint) and
makes each test's fixture the smallest thing that can express the rule under
test. Domain-specific behaviour is tested by each domain's own
``{domain}_test.py``, next to the code it describes.

``minimal_ir()`` is the base: a one-decision, one-source, two-event MDP that
validates. Tests deep-copy it and mutate the one node they care about.
"""

from __future__ import annotations

import copy

import pytest

# A trivial stock-and-flow world: one continuous decision `act` adds to
# `level`, one Poisson source `draw` subtracts from it. Small enough to read
# in one screen, rich enough to carry a guard, an info field, a decomposition,
# an instance override and a scenario constant.
_MINIMAL: dict = {
    "ir_version": "1.0",
    "seed_scheme": "v2",
    "domain": {"name": "tiny", "class_prefix": "Tiny"},
    "mdp": {
        "horizon": {"T": 5, "period_indexing": "0-based"},
        "entity_structure": {"kind": "single"},
        "state_variables": [
            {"name": "period", "role": "time_index", "type": "int"},
            {"name": "level", "role": "core", "type": "float"},
        ],
        "info_fields": [
            {"name": "outflow", "type": "float"},
            {"name": "cost", "type": "decomposition", "components": ["carry", "total"]},
        ],
        "decisions": [
            {
                "name": "act",
                "type": {"value": "continuous", "suggested": "continuous"},
                "dim": 1,
                "bounds": {"value": [0.0, 10.0], "suggested": [0.0, 10.0]},
            }
        ],
        "uncertainty_sources": [
            {
                "name": "flow",
                "generator": "FlowGenerator",
                "stream_id": 0,
                "distribution": {"family": "poisson", "settings": {"rate": "rate"}},
                "stages": [{"name": "sample", "realization": "period"}],
            }
        ],
        "dynamics": {
            "event_sequence": ["A", "B"],
            "transitions": [
                {"event": "A", "updates": ["level += act"]},
                {
                    "event": "B",
                    "updates": ["outflow ~ flow.sample", "level -= outflow"],
                },
                {"event": "END_OF_PERIOD", "updates": ["period += 1"]},
            ],
        },
        "objective": {
            "sense": "minimize",
            "per_step_components": [{"name": "carry", "expr": "h * abs(level)"}],
        },
        "scenario": {
            "constants": [
                {"name": "h", "value": 1.0, "axis": "cost"},
                {"name": "rate", "value": 3.0, "axis": "demand"},
            ],
            "instances": {"pricey": {"h": 5.0}},
        },
        "initial_state": {"level": 0.0},
    },
    "gym": {
        "observation_modes": [
            {"name": "vec", "features": [{"ref": "level"}, {"ref": "info.outflow"}], "default": True}
        ],
        "action_modes": [
            {
                "name": "act",
                "encodes": "act",
                "type": "continuous",
                "bounds": [0.0, 10.0],
                "default": True,
            }
        ],
        "reward_modes": [{"name": "neg_cost", "expr": "-total", "default": True}],
        "termination": {},
    },
    "rl": {
        "requires_memory": {
            "value": False, "suggested": False, "source": "derived",
            "rationale": "fully observed",
        },
        "obs_normalization": {"enabled": True},
    },
}


def minimal_ir() -> dict:
    """A fresh, validating IR document. Mutate the copy, never the template."""
    return copy.deepcopy(_MINIMAL)


def minimal_catalog() -> dict:
    """The same world in catalog form: the `flow` source becomes a slot with
    two candidates — a Poisson default and a degenerate alternative — so the
    resolver, the selection record and the fingerprint have something to
    quantify over without borrowing a shipped domain."""
    doc = minimal_ir()
    src = doc["mdp"].pop("uncertainty_sources")[0]
    doc["mdp"]["uncertainty_slots"] = [{
        "name": src["name"],
        "interface": "FlowGenerator",
        "stream_id": src["stream_id"],
        "latent": False,
        "stages": src["stages"],
        "default": "poisson",
        "candidates": {
            "poisson": {
                "generator": "PoissonFlow",
                "family": "poisson",
                "settings": {"rate": "rate"},
                "is_discrete": True,
                "desc": "the base candidate",
            },
            "fixed": {
                "generator": "FixedFlow",
                "family": "deterministic",
                "settings": {"value": "rate"},
                "is_discrete": True,
                "desc": "degenerate alternative — same constant, no draw",
            },
        },
        "desc": "",
    }]
    return doc


@pytest.fixture
def ir_doc() -> dict:
    return minimal_ir()


@pytest.fixture
def catalog_doc() -> dict:
    return minimal_catalog()

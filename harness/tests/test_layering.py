"""Catalog ⊕ selection: one schema, a candidate pool per uncertainty slot.

The promise of the layer is that the *menu* is free and the *structure* is
frozen: appending a candidate, adding an instance or editing a constant's
value must not move the structural fingerprint (so it needs no re-confirmation
at the Phase-A gate), while touching dynamics — or a declared claim about them
— must.
"""

from __future__ import annotations

import copy
import json

import pytest
from conftest import minimal_catalog, minimal_ir

from mdp_ir import layering
from mdp_ir.schema import MdpIR, load_ir


def resolved(doc: dict, **kw) -> MdpIR:
    return MdpIR.model_validate(layering.resolve_catalog(doc, **kw))


# -- form detection ----------------------------------------------------------


def test_a_slot_document_is_a_catalog(catalog_doc):
    assert layering.is_catalog(catalog_doc)


def test_a_resolved_document_is_not_a_catalog(ir_doc):
    assert not layering.is_catalog(ir_doc)


def test_the_retired_structure_binding_kinds_are_reported(catalog_doc):
    with pytest.raises(layering.LayeringError, match="retired"):
        layering.is_catalog({**catalog_doc, "kind": "binding"})


# -- selection ---------------------------------------------------------------


def test_the_default_selection_is_recorded(catalog_doc):
    ir = resolved(catalog_doc)
    assert ir.selection == {"flow": "poisson"}
    assert ir.mdp.uncertainty_sources[0].distribution.family == "poisson"


def test_selecting_a_candidate_swaps_the_family(catalog_doc):
    ir = resolved(catalog_doc, select={"flow": "fixed"})
    assert ir.selection == {"flow": "fixed"}
    assert ir.mdp.uncertainty_sources[0].distribution.family == "deterministic"


def test_selecting_an_unknown_candidate_is_an_error(catalog_doc):
    with pytest.raises(layering.LayeringError):
        resolved(catalog_doc, select={"flow": "no_such"})


def test_load_ir_resolves_a_catalog_from_disk(catalog_doc, tmp_path):
    p = tmp_path / "tiny_schema.json"
    p.write_text(json.dumps(catalog_doc))
    assert load_ir(p).selection == {"flow": "poisson"}
    assert load_ir(p, select={"flow": "fixed"}).selection == {"flow": "fixed"}


# -- grids under a selection -------------------------------------------------


def _with_a_scoped_instance(doc: dict) -> dict:
    """`fixed_world` selects the non-default candidate, so every resolution but
    its own drops it; `pricey` overrides a constant only and survives all."""
    doc["mdp"]["scenario"]["instances"]["fixed_world"] = {"flow": "fixed", "h": 2.0}
    return doc


def test_a_grid_on_a_scoped_out_instance_still_validates(catalog_doc):
    """The grid's base instance is pruned under the default selection; that
    scopes the grid out of this resolution, it does not make the name unknown
    (upstream #75). The whole file must stay validatable."""
    doc = _with_a_scoped_instance(catalog_doc)
    doc["grids"] = [{"name": "sweep", "base_instance": "fixed_world",
                     "axes": {"h": [1.0, 2.0]}}]
    ir = resolved(doc)
    assert ir.pruned_instances == ["fixed_world"]
    assert "fixed_world" not in ir.mdp.scenario.instances


def test_a_grid_applies_under_the_selection_its_base_instance_names(catalog_doc):
    doc = _with_a_scoped_instance(catalog_doc)
    doc["grids"] = [{"name": "sweep", "base_instance": "fixed_world",
                     "axes": {"h": [1.0, 2.0]}}]
    ir = resolved(doc, instance="fixed_world")
    assert ir.pruned_instances == []
    assert len(ir.grids[0].cells()) == 2


def test_a_grid_axis_error_is_reported_under_that_selection(catalog_doc):
    """Skipping a scoped-out grid loses no checking: the resolution the grid
    does apply to is where its axes are checked, and the CLI walks them all."""
    doc = _with_a_scoped_instance(catalog_doc)
    doc["grids"] = [{"name": "sweep", "base_instance": "fixed_world",
                     "axes": {"no_such": [1.0]}}]
    resolved(doc)
    with pytest.raises(ValueError, match="no scenario constant"):
        resolved(doc, instance="fixed_world")


def test_a_grid_on_an_undeclared_instance_is_still_an_error(catalog_doc):
    doc = _with_a_scoped_instance(catalog_doc)
    doc["grids"] = [{"name": "sweep", "base_instance": "no_such",
                     "axes": {"h": [1.0]}}]
    for kw in ({}, {"instance": "fixed_world"}):
        with pytest.raises(ValueError, match="unknown base_instance"):
            resolved(doc, **kw)


# -- structural fingerprint --------------------------------------------------


def test_editing_a_constant_value_does_not_move_the_fingerprint(catalog_doc):
    fp = layering.structural_fingerprint(catalog_doc)
    tweaked = copy.deepcopy(catalog_doc)
    next(c for c in tweaked["mdp"]["scenario"]["constants"]
         if c["name"] == "h")["value"] = 9.9
    assert layering.structural_fingerprint(tweaked) == fp


def test_adding_an_instance_does_not_move_the_fingerprint(catalog_doc):
    fp = layering.structural_fingerprint(catalog_doc)
    tweaked = copy.deepcopy(catalog_doc)
    tweaked["mdp"]["scenario"]["instances"]["hot"] = {"rate": 12.0}
    assert layering.structural_fingerprint(tweaked) == fp


def test_appending_a_candidate_does_not_move_the_fingerprint(catalog_doc):
    fp = layering.structural_fingerprint(catalog_doc)
    tweaked = copy.deepcopy(catalog_doc)
    tweaked["mdp"]["uncertainty_slots"][0]["candidates"]["uniform_rate"] = {
        "generator": "UniformRateFlow", "family": "poisson",
        "settings": {"rate": {"draw": {"family": "uniform",
                                       "settings": {"low": 2.0, "high": 4.0}},
                              "example": 3.0}},
    }
    assert layering.structural_fingerprint(tweaked) == fp


def test_editing_dynamics_moves_the_fingerprint(catalog_doc):
    fp = layering.structural_fingerprint(catalog_doc)
    tweaked = copy.deepcopy(catalog_doc)
    tweaked["mdp"]["dynamics"]["transitions"][0]["updates"].append("level += 0")
    assert layering.structural_fingerprint(tweaked) != fp


def test_editing_a_declared_claim_moves_the_fingerprint(catalog_doc):
    """Invariants are structural: a claim is part of the formalization the
    user confirmed, not a free-floating test."""
    with_claim = copy.deepcopy(catalog_doc)
    with_claim["mdp"]["invariants"] = [{"name": "floor", "expr": "level >= -100"}]
    fp = layering.structural_fingerprint(with_claim)

    edited = copy.deepcopy(with_claim)
    edited["mdp"]["invariants"][0]["expr"] = "level >= -50"
    assert layering.structural_fingerprint(edited) != fp

    dropped = copy.deepcopy(with_claim)
    dropped["mdp"]["invariants"] = []
    assert layering.structural_fingerprint(dropped) != fp


def test_a_constant_referenced_only_by_a_claim_is_frozen(catalog_doc):
    """The fingerprint freezes the NAMES the structural core references; a
    claim is part of that core, so a constant it alone mentions counts."""
    base = copy.deepcopy(catalog_doc)
    base["mdp"]["scenario"]["constants"].append(
        {"name": "cap", "value": 100.0, "axis": "sizing"})
    without = layering.structural_fingerprint(base)

    referenced = copy.deepcopy(base)
    referenced["mdp"]["invariants"] = [{"name": "cap", "expr": "level <= cap"}]
    assert layering.structural_fingerprint(referenced) != without


def test_structural_fingerprint_requires_a_catalog(ir_doc):
    with pytest.raises(layering.LayeringError):
        layering.structural_fingerprint(ir_doc)


# -- the zero-code path ------------------------------------------------------


def test_an_appended_candidate_resolves_with_no_other_edit(catalog_doc):
    """A new candidate needs no new domain Python: it resolves, validates and
    desugars its own latent draw."""
    doc = copy.deepcopy(catalog_doc)
    doc["mdp"]["uncertainty_slots"][0]["candidates"]["uniform_rate"] = {
        "generator": "UniformRateFlow", "family": "poisson",
        "settings": {"rate": {"draw": {"family": "uniform",
                                       "settings": {"low": 2.0, "high": 4.0}},
                              "example": 3.0, "hidden": True}},
    }
    # a hidden latent forces the memory derivation, which in turn needs a
    # memory mechanism declared — frame stacking is the cheaper of the two
    doc["rl"]["requires_memory"] = {
        "value": True, "suggested": True, "source": "derived",
        "rationale": "hidden rate latent",
    }
    doc["rl"]["frame_stack"] = 4
    ir = resolved(doc, select={"flow": "uniform_rate"})
    assert ir.selection == {"flow": "uniform_rate"}
    assert ir.mdp.uncertainty_sources[0].distribution.family == "poisson"
    assert ir.mdp.scenario.samplers[0].draws[0].name == "flow_rate"


def test_the_resolved_ir_carries_no_catalog(catalog_doc):
    """Nothing downstream should have to know a catalog existed."""
    doc = layering.resolve_catalog(catalog_doc)
    assert "uncertainty_slots" not in doc["mdp"]
    assert "uncertainty_sources" in doc["mdp"]


def test_minimal_fixtures_describe_the_same_world():
    """`minimal_catalog()` must be the same world as `minimal_ir()`, or the two
    halves of this suite drift apart. Compared by trajectory, not fingerprint:
    resolution legitimately enriches the source with the candidate's generator
    name and `is_discrete`/`latent` metadata, none of which changes behaviour.
    """
    from mdp_ir.interpreter import simulate

    a = simulate(MdpIR.model_validate(minimal_ir()), 3, {"act": 4.0})
    b = simulate(resolved(minimal_catalog()), 3, {"act": 4.0})
    assert a.rows == b.rows

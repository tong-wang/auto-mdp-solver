"""Bounds entries: one resolution rule for all three forms (upstream #53).

A bounds entry is a literal, the name of a scenario constant, or an expression
over constants (and, in a catalog IR, slot read-API stats). Issue #3 gave the
name form its per-instance resolution; the expression form was collapsed at
catalog-resolution time instead, which made it catalog-only and froze any
constant inside it at its BASE value — silently correct for no instance, the
very defect the sample's "declare the derivation" advice exists to avoid.

These tests pin the rule that replaces it: whatever its form, an entry resolves
against the instance's constant pool at the read API. The catalog resolver folds
only what lives there alone — a slot's stats, derived from the *selected*
candidate — and an expression reading no overridden constant still collapses to
a number at load, so no existing IR's fingerprint moves.
"""

from __future__ import annotations

import copy
import json

import pytest
from conftest import minimal_catalog, minimal_ir
from pydantic import ValidationError

from mdp_ir import layering
from mdp_ir.schema import MdpIR, load_ir


def resolved(doc: dict, **kw) -> MdpIR:
    return MdpIR.model_validate(layering.resolve_catalog(doc, **kw))


def with_bounds(doc: dict, bounds, *, consts=(), instances=None) -> dict:
    """Put `bounds` on the `level` state variable."""
    doc = copy.deepcopy(doc)
    for sv in doc["mdp"]["state_variables"]:
        if sv["name"] == "level":
            sv["bounds"] = bounds
    doc["mdp"]["scenario"]["constants"].extend(consts)
    if instances is not None:
        doc["mdp"]["scenario"]["instances"] = instances
    return doc


# -- the expression form, outside the catalog --------------------------------


def test_an_expression_resolves_in_a_non_catalog_ir(ir_doc):
    """`uncertainty_sources` files never ran the catalog resolver, so an
    expression reached the validator unevaluated and was refused as a typo'd
    constant name — the sample's own example, rejected verbatim."""
    m = MdpIR.model_validate(with_bounds(ir_doc, [0.0, "20 * h"]))
    assert m.mdp.state_bounds("level") == (0.0, 20.0)


def test_an_expression_tracks_the_instance_override(ir_doc):
    m = MdpIR.model_validate(with_bounds(ir_doc, [0.0, "20 * h"]))
    assert m.mdp.state_bounds("level", instance="pricey") == (0.0, 100.0)


def test_the_expression_form_reads_the_core_builtins(ir_doc):
    m = MdpIR.model_validate(
        with_bounds(ir_doc, ["-1 * max(h, rate)", "sqrt(rate) * h"]))
    assert m.mdp.state_bounds("level") == (-3.0, pytest.approx(3 ** 0.5))


def test_a_categorical_constant_may_gate_an_envelope(ir_doc):
    """String literals are data, never names: the tokenizer must not read
    `'wide'` as an identifier, nor the folder rewrite inside it."""
    doc = with_bounds(
        ir_doc, [0.0, "10 * h if mode == 'wide' else h"],
        consts=[{"name": "mode", "value": "wide", "axis": "variant"}],
        instances={"narrow": {"mode": "tight"}},
    )
    m = MdpIR.model_validate(doc)
    assert m.mdp.state_bounds("level") == (0.0, 10.0)
    assert m.mdp.state_bounds("level", instance="narrow") == (0.0, 1.0)


def test_a_slot_stat_outside_the_catalog_names_its_real_cause(ir_doc):
    """`flow` IS declared here — as an uncertainty *source*, whose read API
    derives from a selected candidate and so exists only in catalog form. The
    old message ("names no scenario constant") reported it as a typo."""
    with pytest.raises(ValidationError, match="catalog form"):
        MdpIR.model_validate(with_bounds(ir_doc, [0.0, "20 * flow.mean"]))


def test_an_unknown_name_inside_an_expression_is_still_rejected(ir_doc):
    with pytest.raises(ValidationError, match=r"reads \['nope'\]"):
        MdpIR.model_validate(with_bounds(ir_doc, [0.0, "20 * nope"]))


def test_an_expression_inverted_in_one_instance_is_rejected(ir_doc):
    """Same contract the name form carries: true in the base scenario AND in
    every instance, or the declaration is not a declaration."""
    doc = with_bounds(ir_doc, ["0.5 * h", "2 * h"],
                      instances={"flip": {"h": -1.0}})
    with pytest.raises(ValidationError, match="lo >= hi"):
        MdpIR.model_validate(doc)


def test_a_non_numeric_expression_is_rejected(ir_doc):
    doc = with_bounds(
        ir_doc, [0.0, "mode"],
        consts=[{"name": "mode", "value": "wide", "axis": "variant"}])
    with pytest.raises(ValidationError, match="non-numeric"):
        MdpIR.model_validate(doc)


# -- the expression form, inside the catalog ---------------------------------


def test_a_constant_inside_an_expression_tracks_its_override(catalog_doc):
    """#53's core defect: `pricey` overrides `h`, and the entry froze at the
    base value with no error and no warning."""
    doc = with_bounds(catalog_doc, [0.0, "20 * h"])
    assert resolved(doc).mdp.state_bounds("level") == (0.0, 20.0)
    assert resolved(doc).mdp.state_bounds("level", instance="pricey") \
        == (0.0, 100.0)


def test_it_tracks_the_override_when_loaded_under_that_instance(catalog_doc):
    doc = with_bounds(catalog_doc, [0.0, "20 * h"])
    ir = resolved(doc, instance="pricey")
    assert ir.mdp.state_bounds("level", instance="pricey") == (0.0, 100.0)


def test_a_slot_stat_still_resolves_against_the_selection(catalog_doc):
    """Unchanged: a stat means the *selected* candidate's namespace, which the
    read API cannot rebuild later, so the resolver keeps evaluating it."""
    doc = with_bounds(catalog_doc, [0.0, "20 * flow.mean"])
    doc["mdp"]["uncertainty_slots"][0]["candidates"]["fixed"]["settings"] = \
        {"value": "2 * rate"}
    assert resolved(doc).mdp.state_bounds("level") == (0.0, 60.0)
    assert resolved(doc, select={"flow": "fixed"}).mdp.state_bounds("level") \
        == (0.0, 120.0)


def test_a_pure_slot_stat_expression_still_collapses_at_load(catalog_doc):
    """The no-move guarantee: an expression reading no overridden constant
    resolves exactly as before, so the freeze token of every shipped IR that
    uses the feature stays where it is."""
    doc = with_bounds(catalog_doc, [0.0, "20 * flow.mean"])
    sv = next(s for s in resolved(doc).mdp.state_variables if s.name == "level")
    assert sv.bounds == [0.0, 60.0]
    assert resolved(doc).mdp_fingerprint() == \
        resolved(with_bounds(catalog_doc, [0.0, 60.0])).mdp_fingerprint()


def test_a_mixed_expression_folds_the_stat_and_keeps_the_constant(catalog_doc):
    """The motivating shape: an envelope over one per-instance constant and one
    slot stat. The stat becomes a number (its namespace dies with the resolver);
    the constant stays symbolic and resolves per instance."""
    doc = with_bounds(catalog_doc, [0.0, "h * flow.mean"])
    ir = resolved(doc)
    sv = next(s for s in ir.mdp.state_variables if s.name == "level")
    assert sv.bounds == [0.0, "h * (3.0)"]
    assert ir.mdp.state_bounds("level") == (0.0, 3.0)
    assert ir.mdp.state_bounds("level", instance="pricey") == (0.0, 15.0)


# -- the other bounds sites --------------------------------------------------


def test_element_bounds_take_expressions_too(ir_doc):
    doc = copy.deepcopy(ir_doc)
    doc["mdp"]["state_variables"].append(
        {"name": "buffer", "role": "core", "type": "float",
         "length": 3, "element_bounds": [0.0, "20 * h"]})
    doc["mdp"]["initial_state"]["buffer"] = "zeros(3)"
    m = MdpIR.model_validate(doc)
    assert m.mdp.state_bounds("buffer", element=True) == (0.0, 20.0)
    assert m.mdp.state_bounds("buffer", instance="pricey", element=True) \
        == (0.0, 100.0)


def test_decision_bounds_take_expressions_too(ir_doc):
    doc = copy.deepcopy(ir_doc)
    doc["mdp"]["decisions"][0]["bounds"] = {
        "value": [0.0, "2 * h"], "suggested": [0.0, "2 * h"]}
    doc["gym"]["action_modes"][0]["bounds"] = [0.0, "2 * h"]
    m = MdpIR.model_validate(doc)
    assert m.mdp.decision_bounds("act") == (0.0, 2.0)
    assert m.mdp.decision_bounds("act", instance="pricey") == (0.0, 10.0)


def test_an_action_mode_expression_is_checked_per_instance(ir_doc):
    doc = copy.deepcopy(ir_doc)
    doc["gym"]["action_modes"][0]["bounds"] = [0.0, "h - 1.0"]
    with pytest.raises(ValidationError, match="lo >= hi"):
        MdpIR.model_validate(doc)          # base: h == 1.0 -> [0, 0]


# -- the read path a domain actually uses ------------------------------------


def test_load_ir_resolves_expressions_for_both_forms(tmp_path, ir_doc,
                                                     catalog_doc):
    for name, doc in (("plain", ir_doc), ("catalog", catalog_doc)):
        p = tmp_path / f"{name}_schema.json"
        p.write_text(json.dumps(with_bounds(doc, [0.0, "20 * h"])))
        ir = load_ir(p, instance="pricey")
        assert ir.mdp.state_bounds("level", instance="pricey") == (0.0, 100.0)


# -- the interpreter's random policy -----------------------------------------


def test_the_random_policy_draws_inside_a_derived_action_scale(ir_doc):
    """The differential runner's random policy resolves decision bounds itself,
    so an expression there must reach the same pool the read API uses — under
    the instance, not the base."""
    from mdp_ir.interpreter import IrInterpreter

    doc = copy.deepcopy(ir_doc)
    doc["mdp"]["decisions"][0]["bounds"] = {
        "value": [0.0, "2 * h"], "suggested": [0.0, "2 * h"]}
    doc["gym"]["action_modes"][0]["bounds"] = [0.0, "2 * h"]
    ir = MdpIR.model_validate(doc)
    for instance, hi in ((None, 2.0), ("pricey", 10.0)):
        rows = IrInterpreter(ir, instance=instance).run(5).rows
        acts = [r["act"] for r in rows]
        assert acts and all(0.0 <= a <= hi for a in acts)
    assert [r["act"] for r in IrInterpreter(ir).run(5).rows] != \
        [r["act"] for r in IrInterpreter(ir, instance="pricey").run(5).rows]

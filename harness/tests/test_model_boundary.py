"""The model/rendering boundary and its fingerprints (upstream #29).

Three layers hid behind one `mdp` block — what the theory admits, which points
a study evaluates, and what the code renders — and only the second was
expressible. So a static shape defaulted to the sweep's maximum and the cap was
then asserted back as the model: the human signed off on a sweep and received a
capacity limit. The fix is not a fourth declaration: a width NAMES a scenario
constant, and what it caps is derived from the reference.

The migration property is the one these tests guard hardest: an IR that
predates `model`/`narrowed` must hash **exactly** as before, or every
frozen fingerprint moves on upgrade and every Phase-A confirmation reads as
voided.
"""

from __future__ import annotations

import json

import pytest

from mdp_conformance.checks import check_model_boundary
from mdp_conformance.loader import DomainHandle
from mdp_ir.layering import structural_fingerprint
from mdp_ir.schema import MdpIR


MODEL = {
    "quantities": {
        "leadtime": {"domain": "integer >= 1", "source": "the paper's model; L=2 is its worked value"},
    },
    "out_of_scope": ["fixed ordering costs"],
    "dynamics": ["forall s in 1..L-1: pipe[s] <- pipe[s+1]"],
}


# --- migration: silence must cost nothing ---------------------------------

def test_an_ir_without_the_new_fields_hashes_as_before(ir_doc):
    """Byte-identical, not merely 'still valid'. The recorded value is what a
    frozen case cites, so a shifted default would void every confirmation."""
    baseline = MdpIR.model_validate(ir_doc).mdp_fingerprint()

    # spelling the absent fields explicitly, and adding an absent narrowing
    # citation, must both land on the same token as saying nothing at all
    explicit = json.loads(json.dumps(ir_doc))
    explicit["mdp"].update({"model": None})
    explicit["mdp"]["state_variables"][0]["narrowed"] = None
    assert MdpIR.model_validate(explicit).mdp_fingerprint() == baseline


def test_model_fingerprint_is_none_until_declared(ir_doc):
    assert MdpIR.model_validate(ir_doc).model_fingerprint() is None


# --- the fingerprints split -----------------------------------------------

def test_declaring_the_model_moves_the_freeze_token_but_not_the_rendering(
        catalog_doc, ir_doc):
    """The payoff: 'model unchanged, rendering re-signed' becomes sayable, and
    so does its converse."""
    before_render = structural_fingerprint(catalog_doc)
    catalog = json.loads(json.dumps(catalog_doc))
    catalog["mdp"]["model"] = MODEL
    assert structural_fingerprint(catalog) == before_render      # rendering untouched

    doc = json.loads(json.dumps(ir_doc))
    before_token = MdpIR.model_validate(doc).mdp_fingerprint()
    doc["mdp"]["model"] = MODEL
    declared = MdpIR.model_validate(doc)
    assert declared.model_fingerprint() is not None
    assert declared.mdp_fingerprint() != before_token             # the theory changed


def test_the_model_fingerprint_tracks_only_the_theory(ir_doc):
    doc = dict(ir_doc)
    doc["mdp"] = {**ir_doc["mdp"], "model": MODEL}
    first = MdpIR.model_validate(doc).model_fingerprint()

    widened = json.loads(json.dumps(doc))
    widened["mdp"]["model"]["quantities"]["leadtime"]["domain"] = "integer >= 0"
    assert MdpIR.model_validate(widened).model_fingerprint() != first

    # a design-only edit leaves the theory hash alone
    recap = json.loads(json.dumps(doc))
    recap["mdp"]["scenario"]["constants"][0]["value"] = 999
    assert MdpIR.model_validate(recap).model_fingerprint() == first


def test_a_narrowing_citation_parses_and_names_its_layer(ir_doc):
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["state_variables"][1]["narrowed"] = {
        "to": "integer lattice", "by": "selection:poisson + design:integer_actions"}
    ir = MdpIR.model_validate(doc)
    assert ir.mdp.state_variables[1].narrowed.by.startswith("selection:")


# --- the four gates --------------------------------------------------------

def _domain(tmp_path, doc):
    (tmp_path / "d_schema.json").write_text(json.dumps(doc))
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


def test_undeclared_model_skips(tmp_path, ir_doc):
    result = check_model_boundary(_domain(tmp_path, ir_doc))
    assert result.status == "SKIP" and "undeclared" in result.detail


def test_a_clean_declaration_passes(tmp_path, ir_doc):
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = MODEL
    assert check_model_boundary(_domain(tmp_path, doc)).status == "PASS"


def test_a_literal_width_for_a_declared_quantity_fails(tmp_path, ir_doc):
    """The motivating defect at its source: a sweep max frozen as a capacity.

    The width must NAME a constant, so that what it caps is a design choice on
    the record rather than a number the model appears to state."""
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = MODEL
    doc["mdp"]["state_variables"].append(
        {"name": "leadtime_pipe", "role": "core", "type": "float_vector", "length": 2})
    doc["mdp"]["initial_state"]["leadtime_pipe"] = 0
    result = check_model_boundary(_domain(tmp_path, doc))
    assert result.status == "FAIL" and "literal length 2" in result.detail


def test_a_named_width_the_sweep_outruns_fails(tmp_path, ir_doc):
    """The width constant must cover every designed value."""
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = MODEL
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "leadtime_max", "value": 2, "axis": "size"})
    doc["mdp"]["state_variables"].append(
        {"name": "pipe", "role": "core", "type": "float_vector", "length": "leadtime_max"})
    doc["mdp"]["initial_state"]["pipe"] = 0
    doc["mdp"]["scenario"]["instances"] = {"long": {"leadtime_max": 4}}
    result = check_model_boundary(_domain(tmp_path, doc))
    assert result.status == "FAIL" and "outrun by a designed" in result.detail


def test_a_value_outside_the_declared_domain_fails(tmp_path, ir_doc):
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = MODEL                      # leadtime: integer >= 1
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "leadtime", "value": 0, "axis": ""})
    result = check_model_boundary(_domain(tmp_path, doc))
    assert result.status == "FAIL" and "below its declared domain" in result.detail


def test_the_statement_may_not_name_rendered_slots(tmp_path, ir_doc):
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = {**MODEL, "dynamics": ["level1 <- level2"]}
    result = check_model_boundary(_domain(tmp_path, doc))
    assert result.status == "FAIL" and "rendered slots" in result.detail


@pytest.mark.parametrize("prose", [
    "kept integral so the DP reference stays exact rather than gridded",
    "bounded for tractability of the benchmark",
])
def test_model_prose_arguing_from_a_benchmark_warns(tmp_path, ir_doc, prose):
    """The mechanical detector: a tractability or benchmark rationale inside
    the theory layer is a design choice filed as model."""
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = {**MODEL, "out_of_scope": [prose]}
    result = check_model_boundary(_domain(tmp_path, doc))
    assert result.status == "WARN" and "design choice filed as model" in result.detail


# --- the grouped file layout, and the stochastic structure ------------------

def test_the_grouped_layout_round_trips_to_the_same_hashes(ir_doc, catalog_doc):
    """The grouping is for human readers; a recorded fingerprint is what a
    frozen case cites. A cosmetic regroup that moved one would be a defect."""
    from mdp_ir.schema import group_mdp

    flat_token = MdpIR.model_validate(ir_doc).mdp_fingerprint()
    grouped = json.loads(json.dumps(ir_doc))
    grouped["mdp"] = group_mdp(grouped["mdp"])
    assert set(grouped["mdp"]) <= {"model", "design", "rendering"}
    assert MdpIR.model_validate(grouped).mdp_fingerprint() == flat_token

    cat = json.loads(json.dumps(catalog_doc))
    cat["mdp"] = group_mdp(cat["mdp"])
    assert structural_fingerprint(cat) == structural_fingerprint(catalog_doc)


def test_grouping_is_idempotent_and_reversible(ir_doc):
    from mdp_ir.schema import group_mdp, ungroup_mdp

    once = group_mdp(ir_doc["mdp"])
    assert group_mdp(once) == once                      # idempotent
    assert ungroup_mdp(once) == ungroup_mdp(ir_doc["mdp"])   # reversible


def test_a_quantity_declared_stochastic_needs_an_uncertainty_source(tmp_path, ir_doc):
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = {"quantities": {
        "leadtime": {"domain": "integer >= 1", "stochastic": True}}}
    result = check_model_boundary(_domain(tmp_path, doc))
    assert result.status == "FAIL" and "dropped randomness" in result.detail


def test_a_quantity_declared_deterministic_may_not_be_a_source(tmp_path, ir_doc):
    """clark_scarf's case: the paper holds leadtime deterministic BY ASSUMPTION,
    so an slt candidate would be out of scope rather than a new option."""
    doc = json.loads(json.dumps(ir_doc))
    source = doc["mdp"]["uncertainty_sources"][0]["name"]
    doc["mdp"]["model"] = {"quantities": {
        source: {"domain": "integer >= 1", "stochastic": False}}}
    result = check_model_boundary(_domain(tmp_path, doc))
    assert result.status == "FAIL" and "out of scope" in result.detail


def test_a_declared_source_satisfies_the_stochastic_claim(tmp_path, ir_doc):
    """inv_single's case: the slot exists, and which candidate a scenario picks
    is design — a degenerate default is not a scope reduction."""
    doc = json.loads(json.dumps(ir_doc))
    source = doc["mdp"]["uncertainty_sources"][0]["name"]
    doc["mdp"]["model"] = {"quantities": {
        source: {"domain": "any distribution on [0, inf)", "stochastic": True}}}
    assert check_model_boundary(_domain(tmp_path, doc)).status == "PASS"

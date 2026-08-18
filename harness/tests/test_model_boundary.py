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
from pydantic import ValidationError

from mdp_conformance.checks import check_model_boundary
from mdp_conformance.loader import DomainHandle
from mdp_ir.layering import structural_fingerprint
from mdp_ir.schema import MdpIR, _model_payload, _prune_absent


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


# --- migration, part two: the block itself keeps moving ---------------------
#
# The property above ("silence costs nothing") was written about the arrival of
# `model`. It said nothing about what happens when the block GAINS a field, and
# v0.9.2's `stochastic` proved the difference: absent from the prune list, it
# put a `null` in every declared quantity and re-tokenized the one downstream
# that had adopted the layer, its schema untouched (upstream #34). Adoption was
# the trigger, so the shipped suite could not see it — no example declares a
# model. These literals stand in for that: they were computed under v0.9.0 and
# are hard-coded, so any future field that escapes the prune fails here first.

V090_MDP_TOKEN = "be72884e6d65"
V090_MODEL_TOKEN = "00a8513dfd82"


def test_a_field_added_after_the_model_shipped_does_not_move_the_token(ir_doc):
    """Both hashes, against values recorded before the field existed."""
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = MODEL
    ir = MdpIR.model_validate(doc)
    assert ir.mdp_fingerprint() == V090_MDP_TOKEN
    assert ir.model_fingerprint() == V090_MODEL_TOKEN


def test_declaring_determinism_is_not_the_same_as_saying_nothing(ir_doc):
    """`stochastic` is tri-state, and the pruning must not flatten it.

    `false` is a claim — the source holds this quantity deterministic *by
    assumption* — while unset is an unasked question. They differ from each
    other and from `true`. Pruning on falsiness rather than on absence would
    hash the first two alike, which is why the prune list is the wrong home
    for this field even though joining it would have fixed the token.
    """
    def token(**q):
        doc = json.loads(json.dumps(ir_doc))
        doc["mdp"]["model"] = {"quantities": {"leadtime": {"domain": "integer >= 1", **q}}}
        return MdpIR.model_validate(doc).model_fingerprint()

    assert len({token(), token(stochastic=False), token(stochastic=True)}) == 3


def test_only_absence_prunes_and_only_inside_the_model():
    """The scope is load-bearing in both directions.

    Falsy-but-stated values survive; and the rule stops at the model block,
    because the rendering has explicit nulls that every existing token already
    hashes — pruning those would move every fingerprint in existence.
    """
    assert _model_payload({"a": None, "b": False, "c": 0, "d": "", "e": []}) == {
        "b": False, "c": 0, "d": "", "e": []}
    assert _model_payload({"q": {"x": {"deep": None, "kept": False}}}) == {
        "q": {"x": {"kept": False}}}
    assert _model_payload({"l": [{"n": None, "k": 1}]}) == {"l": [{"k": 1}]}


def test_the_rendering_keeps_its_explicit_nulls(ir_doc):
    """Guards the scoping from the other side: a null in the rendering is part
    of the hashed payload, so widening the prune would be caught here."""
    payload = _prune_absent(MdpIR.model_validate(ir_doc).mdp.model_dump(mode="json"))
    assert json.dumps(payload).count(": null") > 0


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


# --- the review round: #30 #31 #32 #33 -------------------------------------

def test_a_width_naming_the_swept_value_is_not_outrun(tmp_path, ir_doc):
    """#30: an axis-tagged width IS the design value, resolved per instance —
    comparing it against its own overrides failed the pattern §5.0 recommends,
    which is what examples/inv_single ships (`length: "pipeline_len"`)."""
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = {"quantities": {"leadtime": {"domain": "integer >= 1"}}}
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "pipeline_len", "value": 1, "axis": "leadtime"})
    doc["mdp"]["state_variables"].append(
        {"name": "pipeline", "role": "core", "type": "float_vector",
         "length": "pipeline_len"})
    doc["mdp"]["initial_state"]["pipeline"] = 0
    doc["mdp"]["scenario"]["instances"] = {"lt6": {"pipeline_len": 6}}
    assert check_model_boundary(_domain(tmp_path, doc)).status == "PASS"


def test_a_genuine_cap_outrun_by_its_quantity_fails(tmp_path, ir_doc):
    """#30's false negative: cap and capped are DIFFERENT constants."""
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = {"quantities": {"leadtime": {"domain": "integer >= 1"}}}
    doc["mdp"]["scenario"]["constants"] += [
        {"name": "leadtime", "value": 1, "axis": "leadtime"},
        {"name": "leadtime_max", "value": 2, "axis": ""},
    ]
    doc["mdp"]["state_variables"].append(
        {"name": "pipe", "role": "core", "type": "float_vector", "length": "leadtime_max"})
    doc["mdp"]["initial_state"]["pipe"] = 0
    doc["mdp"]["scenario"]["instances"] = {"overrun": {"leadtime": 9}}
    result = check_model_boundary(_domain(tmp_path, doc))
    assert result.status == "FAIL" and "outruns its own rendering" in result.detail


def test_a_multi_dimensional_width_names_each_axis(ir_doc):
    """#31: a 2-D state had no width site, so a flat derived constant was
    carried instead and neither dimension was visible."""
    from mdp_conformance.checks import _axis_tiers

    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "grid_size", "value": 3, "axis": "board"})
    doc["mdp"]["state_variables"].append(
        {"name": "board", "role": "core", "type": "int_vector",
         "length": ["grid_size", "grid_size"]})
    doc["mdp"]["initial_state"]["board"] = 0
    ir = MdpIR.model_validate(doc)
    assert ir.mdp.state_variables[-1].length == ["grid_size", "grid_size"]
    assert "sets axis" in _axis_tiers(ir)["grid_size"]


def test_a_cap_inside_a_bounds_expression_is_still_a_width(tmp_path, ir_doc):
    """#53's companion: an envelope may be derived, and the boundary machinery
    reads the constants the derivation *references* — otherwise writing
    `2 * leadtime_max` instead of `leadtime_max` would hide the cap from the
    coverage check that exists to catch a sweep outrunning its rendering."""
    from mdp_conformance.checks import _axis_tiers

    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["model"] = {"quantities": {"leadtime": {"domain": "integer >= 1"}}}
    doc["mdp"]["scenario"]["constants"] += [
        {"name": "leadtime", "value": 1, "axis": "leadtime"},
        {"name": "leadtime_max", "value": 2, "axis": ""},
    ]
    for sv in doc["mdp"]["state_variables"]:
        if sv["name"] == "level":
            sv["bounds"] = [0, "2 * leadtime_max"]
    doc["mdp"]["scenario"]["instances"] = {"overrun": {"leadtime": 9}}
    ir = MdpIR.model_validate(doc)
    assert "sets bounds" in _axis_tiers(ir)["leadtime_max"]
    result = check_model_boundary(_domain(tmp_path, doc))
    assert result.status == "FAIL" and "outruns its own rendering" in result.detail


def test_a_decision_width_may_name_a_constant(ir_doc):
    """#33: `bounds` could name a constant and `dim` — the width of the same
    decision — could not, so an action count a study sweeps had to be padded."""
    doc = json.loads(json.dumps(ir_doc))
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "n_products", "value": 3, "axis": "size"})
    doc["mdp"]["decisions"][0]["dim"] = "n_products"
    doc["mdp"]["scenario"]["instances"] = {"wide": {"n_products": 8}}
    ir = MdpIR.model_validate(doc)
    assert ir.mdp.decision_dim("act") == 3
    assert ir.mdp.decision_dim("act", "wide") == 8


@pytest.mark.parametrize("expr", [
    "sum(sum(pipe[k]) for k in range(n_echelons))",
    "[pipe[k][1:] + [0] for k in range(n_echelons)]",
    "[x + y for x, y in pairs]",
])
def test_a_comprehension_binds_its_own_index(expr):
    """#32: for/in/range were whitelisted and the evaluator ran comprehensions,
    but the validator rejected every one on its own index."""
    from mdp_ir.schema import _check_expr

    _check_expr(expr, {"pipe", "n_echelons", "pairs"}, "probe")


def test_an_undeclared_name_still_fails_inside_a_comprehension():
    from mdp_ir.schema import _check_expr

    with pytest.raises(ValueError, match="undeclared_name"):
        _check_expr("sum(undeclared_name[k] for k in range(3))", {"pipe"}, "probe")


# --- #35: the binding has to reach the site that asked for it ---------------
#
# Every test above uses an EXPRESSION. `updates` entries are statements, and
# `mode="eval"` alone returned no bindings for them — so #32's fix landed on
# invariants and cost components and missed dynamics, the case that motivated
# it. Invariants were missed too, by a second cause: `prev.x` was blanked to a
# space before the check, leaving an unparseable string.

@pytest.mark.parametrize("stmt", [
    "a = [pipe[k] for k in range(n_echelons)]",
    "total += sum([pipe[k][0] for k in range(n_echelons)])",
    "stock = [stock[k] + pipe[k][0] for k in range(n_echelons)] + stock[n_echelons:]",
])
def test_a_comprehension_binds_its_index_in_a_statement_too(stmt):
    from mdp_ir.schema import _check_expr

    _check_expr(stmt, {"pipe", "n_echelons", "a", "total", "stock"}, "probe")


@pytest.mark.parametrize("stmt", [
    "a = [undeclared_name[k] for k in range(3)]",
    "total += sum([undeclared_name[k] for k in range(n_echelons)])",
])
def test_a_statement_is_not_a_blanket_pass(stmt):
    """The exec fallback must widen what BINDS, never what resolves."""
    from mdp_ir.schema import _check_expr

    with pytest.raises(ValueError, match="undeclared_name"):
        _check_expr(stmt, {"pipe", "n_echelons", "a", "total"}, "probe")


def _with_invariant(doc, expr):
    doc = json.loads(json.dumps(doc))
    doc["mdp"]["invariants"] = [{"name": "conserve", "expr": expr}]
    return doc


def test_a_conservation_law_may_quantify_over_a_prev_reference(ir_doc):
    """The idiom this unblocks: a balance law over a multi-dimensional state,
    which is exactly where a comprehension is wanted and where `prev` is
    unavoidable."""
    MdpIR.model_validate(_with_invariant(
        ir_doc,
        "close(sum([level for k in range(1)]), sum([prev.level for k in range(1)]))",
    ))


def test_an_undeclared_name_still_fails_in_a_prev_referencing_invariant(ir_doc):
    with pytest.raises(ValidationError, match="undeclared_name"):
        MdpIR.model_validate(_with_invariant(
            ir_doc, "close(sum([undeclared_name[k] for k in range(1)]), prev.level)"))


def test_prev_must_still_name_a_per_period_value(ir_doc):
    """`rate` is a scenario constant — fixed for the episode, so `prev.rate`
    is meaningless. The rewrite must not smuggle it into scope."""
    with pytest.raises(ValidationError, match="names no per-period value"):
        MdpIR.model_validate(_with_invariant(ir_doc, "close(level, prev.rate)"))


def test_the_error_quotes_what_the_author_wrote(ir_doc):
    """The checker sees `prev.x` rewritten to `x`; quoting that back would
    point at a line the file does not contain."""
    with pytest.raises(ValidationError, match=r"prev\.level"):
        MdpIR.model_validate(_with_invariant(
            ir_doc, "close(undeclared_name, prev.level)"))

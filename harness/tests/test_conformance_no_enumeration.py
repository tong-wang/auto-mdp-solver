"""The IR declares vectors; it never enumerates them (spec §5.0, upstream #45).

`model.boundary` governs the *width* and SKIPs entirely when the theory layer
is undeclared — which is the state a hand-unrolled IR tends to be in. This
check governs *what is being sized*, on the rendering, unconditionally.

Synthetic IRs only: the harness suite must not depend on `plugin/`.
"""

from __future__ import annotations

from mdp_conformance.checks import _enumeration_findings


def _findings(doc) -> str:
    problems, _ = _enumeration_findings(doc)
    return " | ".join(problems)


def test_a_clean_ir_reports_nothing(ir_doc):
    assert _findings(ir_doc) == ""


# -- decorative constants ----------------------------------------------------

def test_a_constant_nobody_reads_is_decorative(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "demand_window", "value": 2, "axis": ""})
    assert "decorative" in _findings(ir_doc)
    assert "demand_window" in _findings(ir_doc)


def test_naming_a_constant_in_its_own_desc_is_not_a_reference(ir_doc):
    """Otherwise a decorative constant hides behind its own documentation."""
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "demand_window", "value": 2, "axis": "",
         "desc": "demand_window sizes the ADI pipeline"})
    assert "demand_window" in _findings(ir_doc)


def test_a_constant_named_at_a_width_site_is_referenced(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "pipe_len", "value": 3, "axis": "size"})
    ir_doc["mdp"]["state_variables"].append(
        {"name": "pipe", "role": "core", "type": "float_vector",
         "length": "pipe_len", "element_bounds": [0, 100],
         "observability": "observable"})
    assert "pipe_len" not in _findings(ir_doc)


def test_a_constant_the_gym_observes_is_referenced(ir_doc):
    """Regression: scanning only the `mdp` block called `fnv`'s `t_last`
    decorative, when it is a feature of the gym's observation vector — a
    generalist trained across a design grid has to see which cell it is in
    (spec §5.6). A constant can be consumed outside `mdp` entirely."""
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "t_last", "value": 0.9, "axis": "epoch"})
    assert "t_last" in _findings(ir_doc)          # not yet read anywhere
    ir_doc["gym"]["observation_modes"][0]["features"].append(
        {"derived": "t_last", "expr": "t_last"})
    assert "t_last" not in _findings(ir_doc)


def test_a_constant_only_an_instance_overrides_is_referenced(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "swept", "value": 1.0, "axis": "cost"})
    ir_doc["mdp"]["scenario"].setdefault("instances", {})["hot"] = {"swept": 2.0}
    assert "swept" not in _findings(ir_doc)


# -- enumerated families -----------------------------------------------------

def test_ordinal_siblings_are_an_enumerated_vector(ir_doc):
    for name in ("due_now", "due_next"):
        ir_doc["mdp"]["state_variables"].append(
            {"name": name, "role": "core", "type": "int", "bounds": [0, 9],
             "observability": "observable"})
    assert "enumerated 'due'" in _findings(ir_doc)


def test_numbered_siblings_are_an_enumerated_vector(ir_doc):
    for name, value in (("lambda1", 3), ("lambda2", 1)):
        ir_doc["mdp"]["scenario"]["constants"].append(
            {"name": name, "value": value, "axis": "demand"})
    assert "enumerated 'lambda'" in _findings(ir_doc)


def test_a_lone_indexed_name_is_not_a_family(ir_doc):
    """`c1` with no `c2` is a name, not an unrolled vector — two members is
    the whole signal, so a single one must never trip it."""
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "c1", "value": 1.0, "axis": "cost"})
    ir_doc["mdp"]["objective"]["per_step_components"].append(
        {"name": "extra", "expr": "c1", "desc": "reads it"})
    assert "enumerated" not in _findings(ir_doc)


# -- re-baked vectors --------------------------------------------------------

def test_an_instance_restating_a_numeric_vector_is_re_baked(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "h_install", "value": [2.0, 1.0, 0.5], "axis": "cost"})
    ir_doc["mdp"]["scenario"].setdefault("instances", {})["deep"] = {
        "h_install": [3.0, 2.0, 1.0]}
    assert "re-baked" in _findings(ir_doc)
    assert "deep.h_install" in _findings(ir_doc)


def test_a_categorical_list_override_is_not_a_re_bake(ir_doc):
    """`event_order = ['O','R','D']` is a permutation an instance legitimately
    selects, not a computed vector. Numeric-only is what keeps them apart."""
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "event_order", "value": ["O", "R", "D"], "axis": "variant"})
    ir_doc["mdp"]["scenario"].setdefault("instances", {})["rod"] = {
        "event_order": ["R", "O", "D"]}
    assert "re-baked" not in _findings(ir_doc)


def test_a_scalar_override_is_not_a_re_bake(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "rate2", "value": 1.0, "axis": "cost"})
    ir_doc["mdp"]["scenario"].setdefault("instances", {})["hot"] = {"rate2": 2.0}
    assert "re-baked" not in _findings(ir_doc)

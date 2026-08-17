"""`--decision NAME=VALUE`, shared by both CLIs.

#38's sweep concluded that no module constructs a decision value without
consulting the declared width. True of the libraries, and both argument parsers
were still calling `float(val)` — so on a vector decision the two CLIs died
with `TypeError: 'float' object is not subscriptable` (#39).

That matters more than a convenience: spec Phase-A step 7b names the
interpreter CLI as the route to a REQUIRED artifact, the annotated sample
trajectory a human reads to check the model. A domain adopting a symbolic `dim`
could not produce it as documented.
"""

from __future__ import annotations

import pytest
from conftest import minimal_ir

from mdp_ir.interpreter import parse_decisions
from mdp_ir.schema import MdpIR


def _ir(width=3, instances=None):
    doc = minimal_ir()
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "n_echelons", "value": width, "axis": "size"})
    doc["mdp"]["decisions"][0]["dim"] = "n_echelons"
    doc["mdp"]["scenario"]["instances"] = instances or {}
    return MdpIR.model_validate(doc)


# --- migration: the scalar form must not move -------------------------------

def test_a_scalar_decision_parses_to_a_scalar(ir_doc):
    """Every existing invocation. `dim: 1` stays a float, not `[float]`."""
    parsed = parse_decisions(["act=40"], MdpIR.model_validate(ir_doc))
    assert parsed == {"act": 40.0}
    assert isinstance(parsed["act"], float)


# --- the vector forms -------------------------------------------------------

def test_one_value_broadcasts_to_the_declared_width():
    """`--decision ship=10` should mean the obvious thing, not an error."""
    assert parse_decisions(["act=10"], _ir())["act"] == [10.0, 10.0, 10.0]


def test_a_comma_separated_list_sets_components():
    assert parse_decisions(["act=10,0,5"], _ir())["act"] == [10.0, 0.0, 5.0]


def test_the_width_resolves_per_instance():
    """The same spec parses to different shapes across a width sweep — the
    part a single-instance test would miss."""
    ir = _ir(3, {"n2": {"n_echelons": 2}, "n4": {"n_echelons": 4}})
    assert parse_decisions(["act=10"], ir, "n2")["act"] == [10.0, 10.0]
    assert len(parse_decisions(["act=10"], ir, "n4")["act"]) == 4
    assert parse_decisions(["act=1,2"], ir, "n2")["act"] == [1.0, 2.0]


# --- errors are messages, not tracebacks ------------------------------------

def test_a_length_mismatch_names_the_width_and_the_instance():
    ir = _ir(3, {"n2": {"n_echelons": 2}})
    with pytest.raises(ValueError, match=r"width 2 for instance 'n2'; got 3"):
        parse_decisions(["act=1,2,3"], ir, "n2")


def test_an_unknown_decision_is_caught_before_any_width_lookup():
    """`decision_dim` raises StopIteration on a name it does not know, which is
    not a message anyone can act on."""
    with pytest.raises(ValueError, match="unknown decision 'shp'"):
        parse_decisions(["shp=1"], _ir())


def test_a_non_numeric_value_is_reported_as_such():
    with pytest.raises(ValueError, match="not a number"):
        parse_decisions(["act=abc"], _ir())


# --- the mixture guard ------------------------------------------------------

def test_a_mixture_name_resolves_as_base_rather_than_raising():
    """`--instance` accepts a mixture name, which is NOT a scenario instance —
    a width lookup against one would KeyError. Same guard as #38."""
    doc = minimal_ir()
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "n_echelons", "value": 3, "axis": "size"})
    doc["mdp"]["decisions"][0]["dim"] = "n_echelons"
    doc["mdp"]["scenario"]["instances"] = {"pricey": {"h": 5.0}}
    doc["mdp"]["scenario"]["mixtures"] = [
        {"name": "mix", "substream_id": 9,
         "components": [[0.5, "pricey"], [0.5, ""]]}
    ]
    ir = MdpIR.model_validate(doc)
    assert parse_decisions(["act=10"], ir, "mix")["act"] == [10.0, 10.0, 10.0]


def test_nothing_to_parse_is_an_empty_dict(ir_doc):
    assert parse_decisions([], MdpIR.model_validate(ir_doc)) == {}

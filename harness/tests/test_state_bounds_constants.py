"""State-variable bounds naming scenario constants (upstream issue #3).

Bounds are the envelope the gym materializes, and when the envelope follows a
per-instance constant (a counter bounded by the horizon), the declaration
should name the constant — a union-over-instances literal is correct for no
instance and drags the structural fingerprint on every widen. `Decision.bounds`
and `StateVariable.length` already resolve names; these tests lock in the same
contract for `bounds`/`element_bounds`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mdp_ir.schema import MdpIR


def _with_counter(ir: dict, bounds=None, element_bounds=None,
                  extra_consts=(), instances=None) -> dict:
    sv = {"name": "pulls", "role": "core", "type": "int"}
    if bounds is not None:
        sv["bounds"] = bounds
    if element_bounds is not None:
        sv["element_bounds"] = element_bounds
        sv["length"] = "n_arms" if any(
            c["name"] == "n_arms" for c in extra_consts) else 2
    ir["mdp"]["state_variables"].append(sv)
    ir["mdp"]["initial_state"]["pulls"] = 0
    ir["mdp"]["scenario"]["constants"].extend(extra_consts)
    if instances is not None:
        ir["mdp"]["scenario"]["instances"] = instances
    return ir


def test_bounds_may_name_an_instance_overridden_constant(ir_doc):
    ir = _with_counter(
        ir_doc, bounds=[0, "horizon_T"],
        extra_consts=[{"name": "horizon_T", "value": 100, "axis": "size"}],
        instances={"long": {"horizon_T": 40000}},
    )
    m = MdpIR.model_validate(ir)
    sv = next(s for s in m.mdp.state_variables if s.name == "pulls")
    assert sv.bounds == [0, "horizon_T"]


def test_state_bounds_resolves_per_instance(ir_doc):
    ir = _with_counter(
        ir_doc, bounds=[0, "horizon_T"],
        extra_consts=[{"name": "horizon_T", "value": 100, "axis": "size"}],
        instances={"long": {"horizon_T": 40000}},
    )
    m = MdpIR.model_validate(ir)
    assert m.mdp.state_bounds("pulls") == (0.0, 100.0)
    assert m.mdp.state_bounds("pulls", instance="long") == (0.0, 40000.0)


def test_element_bounds_resolve_too(ir_doc):
    ir = _with_counter(
        ir_doc, element_bounds=["neg_cap", "cap"],
        extra_consts=[{"name": "cap", "value": 60.0, "axis": "size"},
                      {"name": "neg_cap", "value": -60.0, "axis": "size"},
                      {"name": "n_arms", "value": 10, "axis": "size"}],
        instances={"big": {"cap": 6e4, "neg_cap": -6e4, "n_arms": 20}},
    )
    m = MdpIR.model_validate(ir)
    assert m.mdp.state_bounds("pulls", element=True) == (-60.0, 60.0)
    assert m.mdp.state_bounds("pulls", instance="big", element=True) \
        == (-6e4, 6e4)


def test_no_bounds_resolves_to_none(ir_doc):
    m = MdpIR.model_validate(ir_doc)
    assert m.mdp.state_bounds("level") is None


def test_unknown_constant_name_rejected(ir_doc):
    ir = _with_counter(ir_doc, bounds=[0, "nope"])
    with pytest.raises(ValidationError, match="name no scenario constant"):
        MdpIR.model_validate(ir)


def test_bound_inverted_in_one_instance_rejected(ir_doc):
    """The named constant must give lo < hi in the base AND every instance —
    the whole point is that the declaration stays true as the family grows."""
    ir = _with_counter(
        ir_doc, bounds=[0, "horizon_T"],
        extra_consts=[{"name": "horizon_T", "value": 100, "axis": "size"}],
        instances={"degenerate": {"horizon_T": 0}},
    )
    with pytest.raises(ValidationError, match="lo >= hi"):
        MdpIR.model_validate(ir)


def test_non_numeric_constant_rejected(ir_doc):
    ir = _with_counter(
        ir_doc, bounds=[0, "mode"],
        extra_consts=[{"name": "mode", "value": "long", "axis": "mode"}],
    )
    with pytest.raises(ValidationError, match="non-numeric"):
        MdpIR.model_validate(ir)


def test_literal_bounds_still_shape_checked(ir_doc):
    ir = _with_counter(ir_doc, bounds=[3.0, 1.0])
    with pytest.raises(ValidationError, match="lo >= hi"):
        MdpIR.model_validate(ir)

"""Grid-axis tier classification (spec §5.6 "Choosing axes", upstream #5).

The classifier reads tiers off the IR's own declarations — no hand-kept tag
list: a constant named in decision/action-mode bounds is tier-1, in a state
variable's length/(element_)bounds tier-2 obs-dim, the horizon constant
tier-2 horizon, everything else tier-3.
"""

from __future__ import annotations

from mdp_conformance.checks import _axis_tiers
from mdp_ir.schema import MdpIR


def _classify(ir_doc, **mdp_edits):
    for key, value in mdp_edits.items():
        ir_doc["mdp"][key] = value
    return _axis_tiers(MdpIR.model_validate(ir_doc))


def test_plain_constants_are_tier3(ir_doc):
    tiers = _classify(ir_doc)
    assert "h" not in tiers and "rate" not in tiers


def test_horizon_constant_is_tier2(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "horizon_T", "value": 5, "axis": "size"})
    tiers = _classify(ir_doc, horizon={"T": "horizon_T",
                                       "period_indexing": "0-based"})
    assert tiers["horizon_T"] == "tier-2 horizon"


def test_decision_bound_constant_is_tier1(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "arm_max", "value": 9, "axis": "size"})
    ir_doc["mdp"]["decisions"][0]["bounds"] = {
        "value": [0, "arm_max"], "suggested": [0, "arm_max"]}
    tiers = _classify(ir_doc)
    assert tiers["arm_max"].startswith("tier-1")


def test_state_length_constant_is_tier2_obs_dim(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "n_arms", "value": 3, "axis": "size"})
    ir_doc["mdp"]["state_variables"].append(
        {"name": "pulls", "role": "core", "type": "int_vector",
         "length": "n_arms"})
    ir_doc["mdp"]["initial_state"]["pulls"] = 0
    tiers = _classify(ir_doc)
    assert tiers["n_arms"].startswith("tier-2 obs-dim")


def test_tier1_wins_over_tier2_when_both_apply(ir_doc):
    """A constant that sizes a state vector AND the action space is tier-1 —
    the action space is the harder break."""
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "n_arms", "value": 3, "axis": "size"})
    ir_doc["mdp"]["state_variables"].append(
        {"name": "pulls", "role": "core", "type": "int_vector",
         "length": "n_arms"})
    ir_doc["mdp"]["initial_state"]["pulls"] = 0
    ir_doc["mdp"]["decisions"][0]["bounds"] = {
        "value": [0, "n_arms"], "suggested": [0, "n_arms"]}
    tiers = _classify(ir_doc)
    assert tiers["n_arms"].startswith("tier-1")

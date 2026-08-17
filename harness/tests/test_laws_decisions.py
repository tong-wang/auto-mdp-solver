"""The decision dict the engine laws run their policies on.

`mdp_ir.laws` builds its own fixed-policy decisions rather than using the
interpreter's random policy, so it needed the vector-decision fix separately —
and did not get it. Every policy law raised `TypeError: 'float' object is not
subscriptable` on the first domain to declare a vector decision, while that
domain's differential passed on the same IR (#38). Two gates, one shape
question, opposite answers.

`laws.py` had no unit tests before this: the same structural gap that hid the
`--all-instances` defect in #37. These cover the decision dict specifically,
which is where all three of its instance-resolution questions live.
"""

from __future__ import annotations

import json

import pytest
from conftest import minimal_ir

from mdp_ir import laws
from mdp_ir.schema import MdpIR


def _vector_doc(width=3, instances=None, index=True):
    """A decision whose width is a swept constant, indexed by the dynamics —
    so a scalar reaches the rule and raises, exactly as reported."""
    doc = minimal_ir()
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "n_echelons", "value": width, "axis": "size"})
    doc["mdp"]["decisions"][0]["dim"] = "n_echelons"
    if index:
        doc["mdp"]["dynamics"]["transitions"][0]["updates"] = ["level += act[0]"]
    doc["mdp"]["scenario"]["instances"] = instances or {}
    return MdpIR.model_validate(doc)


POLICY_LAWS = [
    "law_determinism",
    "law_seed_sensitivity",
    "law_decision_path_independence",
    "law_invariants",
]


# --- the reported failure --------------------------------------------------

@pytest.mark.parametrize("law", POLICY_LAWS)
def test_a_policy_law_runs_on_a_vector_decision(law):
    """Was: TypeError on every one of these."""
    result = getattr(laws, law)(_vector_doc(), None, None)
    assert result.status in {"PASS", "SKIP"}


def test_the_dict_carries_the_declared_width():
    assert laws._fixed_decisions(_vector_doc(), 0.5)["act"] == [5.0, 5.0, 5.0]


# --- the part a single-instance test would miss ----------------------------

def test_the_width_resolves_per_instance():
    """The naive three-line fix — `[v] * decision_dim(d.name)` — passes the
    base width to every instance, handing a 3-wide action to a 2-echelon
    world. The width has to be resolved against the instance being run."""
    ir = _vector_doc(3, {"n2": {"n_echelons": 2}, "n4": {"n_echelons": 4}})
    assert len(laws._fixed_decisions(ir, 0.5)["act"]) == 3
    assert len(laws._fixed_decisions(ir, 0.5, "n2")["act"]) == 2
    assert len(laws._fixed_decisions(ir, 0.5, "n4")["act"]) == 4


@pytest.mark.parametrize("law", POLICY_LAWS)
@pytest.mark.parametrize("inst", [None, "n2", "n4"])
def test_every_policy_law_gets_the_instance_width(law, inst):
    ir = _vector_doc(3, {"n2": {"n_echelons": 2}, "n4": {"n_echelons": 4}})
    assert getattr(laws, law)(ir, inst, None).status in {"PASS", "SKIP"}


def test_the_bounds_resolve_per_instance_too():
    """Same parameter, same bug, not in the report: `decision_bounds` takes an
    instance and was not given one, so a law tested an action the instance
    cannot take. `mab` narrows `arm` from (0, 9) to (0, 4) on its 5-arm
    instances, and got 9."""
    doc = minimal_ir()
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "cap", "value": 10.0, "axis": "scale"})
    doc["mdp"]["decisions"][0]["bounds"] = {"value": [0.0, "cap"],
                                            "suggested": [0.0, "cap"]}
    doc["mdp"]["scenario"]["instances"] = {"tight": {"cap": 4.0}}
    ir = MdpIR.model_validate(doc)
    assert laws._fixed_decisions(ir, 1.0)["act"] == 10.0
    assert laws._fixed_decisions(ir, 1.0, "tight")["act"] == 4.0


def test_a_mixture_name_resolves_as_base_rather_than_raising():
    """A mixture is not a scenario instance, so resolving a width or a bound
    against its name would KeyError — and laws ARE called with mixture names.
    `_episode_rows` already guards this way; threading the instance through
    without the same guard would have crashed every domain with a mixture."""
    doc = minimal_ir()
    doc["mdp"]["scenario"]["instances"] = {"pricey": {"h": 5.0}}
    doc["mdp"]["scenario"]["mixtures"] = [
        # "" names the base constants (ScenarioMixture)
        {"name": "mix", "substream_id": 9,
         "components": [[0.5, "pricey"], [0.5, ""]]}
    ]
    ir = MdpIR.model_validate(doc)
    assert "mix" not in (ir.mdp.scenario.instances or {})
    assert laws._fixed_decisions(ir, 0.5, "mix") == laws._fixed_decisions(ir, 0.5)


# --- migration -------------------------------------------------------------

def test_a_scalar_decision_gets_the_identical_dict(ir_doc):
    """Every existing IR has `dim: 1` and indexes the name directly, so the
    scalar must not become a one-element list."""
    ir = MdpIR.model_validate(json.loads(json.dumps(ir_doc)))
    dec = laws._fixed_decisions(ir, 0.5)
    assert dec == {"act": 5.0}
    assert isinstance(dec["act"], float)

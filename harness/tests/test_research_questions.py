"""Declared tier-2 stances and the §14 deliverables they owe (upstream #24).

`2-structural` is three claims — confirm, discover, bypass — taking different
evidence and owing different artifacts. Undeclared, the same measurement reads
three ways: `raw ≈ echelon` is a success under bypass, inconclusive under
confirm, irrelevant under discover. The block records which was signed up for;
the check turns that into an obligation.

Synthetic throughout: `harness/tests` must not depend on `plugin/`.
"""

from __future__ import annotations

import json

import pytest

from mdp_conformance.checks import check_research_questions
from mdp_conformance.loader import DomainHandle
from mdp_ir.schema import MdpIR


def _rq(stances):
    return {"tier2": [dict(s) for s in stances]}


def test_ir_without_the_block_still_validates(ir_doc):
    assert MdpIR.model_validate(ir_doc).research_questions is None


def test_probe_requirement_defaults_by_stance(ir_doc):
    ir_doc["research_questions"] = _rq([
        {"stance": "confirm", "structure": "(s,S)"},
        {"stance": "discover", "structure": "unknown"},
        {"stance": "bypass", "structure": "echelon coordinates"},
    ])
    ir = MdpIR.model_validate(ir_doc)
    assert [q.probe_required for q in ir.research_questions.tier2] == [True, True, False]
    assert ir.research_questions.probe_required is True


def test_bypass_only_owes_nothing(ir_doc):
    ir_doc["research_questions"] = _rq([{"stance": "bypass", "structure": "echelon"}])
    assert MdpIR.model_validate(ir_doc).research_questions.probe_required is False


def test_explicit_probe_required_overrides_the_default(ir_doc):
    """A bypass campaign may still opt in; a confirm campaign may not opt out
    silently — but the field is the place to argue it, not the code."""
    ir_doc["research_questions"] = _rq([
        {"stance": "bypass", "structure": "echelon", "probe_required": True}])
    assert MdpIR.model_validate(ir_doc).research_questions.probe_required is True


def test_two_stances_on_one_structure_are_legal(ir_doc):
    """The motivating shape: bypass primary, confirm secondary, same structure."""
    ir_doc["research_questions"] = _rq([
        {"stance": "bypass", "structure": "echelon coordinates", "priority": "primary"},
        {"stance": "confirm", "structure": "echelon coordinates", "priority": "secondary"},
    ])
    ir = MdpIR.model_validate(ir_doc)
    assert len(ir.research_questions.tier2) == 2
    assert ir.research_questions.probe_required is True


def test_declaring_questions_does_not_move_the_fingerprint(ir_doc):
    before = MdpIR.model_validate(ir_doc).mdp_fingerprint()
    ir_doc["research_questions"] = _rq([{"stance": "confirm", "structure": "(s,S)"}])
    assert MdpIR.model_validate(ir_doc).mdp_fingerprint() == before


def test_unknown_stance_is_rejected(ir_doc):
    ir_doc["research_questions"] = _rq([{"stance": "prove", "structure": "x"}])
    with pytest.raises(Exception):
        MdpIR.model_validate(ir_doc)


# --- the conformance check ------------------------------------------------

def _domain(tmp_path, ir_doc, *, probe: bool, interpret: bool):
    (tmp_path / "d_schema.json").write_text(json.dumps(ir_doc))
    if probe:
        (tmp_path / "d_policy_probe.py").write_text("# probe\n")
    if interpret:
        (tmp_path / "INTERPRET.md").write_text("# readback\n")
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


def test_undeclared_domain_skips(tmp_path, ir_doc):
    h = _domain(tmp_path, ir_doc, probe=False, interpret=False)
    assert check_research_questions(h).status == "SKIP"


def test_confirm_without_the_artifacts_fails(tmp_path, ir_doc):
    ir_doc["research_questions"] = _rq([{"stance": "confirm", "structure": "the snake"}])
    h = _domain(tmp_path, ir_doc, probe=False, interpret=False)
    result = check_research_questions(h)
    assert result.status == "FAIL"
    assert "d_policy_probe.py" in result.detail and "INTERPRET.md" in result.detail


def test_confirm_with_both_artifacts_passes(tmp_path, ir_doc):
    ir_doc["research_questions"] = _rq([{"stance": "confirm", "structure": "(s,S)"}])
    h = _domain(tmp_path, ir_doc, probe=True, interpret=True)
    assert check_research_questions(h).status == "PASS"


def test_a_missing_interpret_alone_still_fails(tmp_path, ir_doc):
    ir_doc["research_questions"] = _rq([{"stance": "discover", "structure": "?"}])
    h = _domain(tmp_path, ir_doc, probe=True, interpret=False)
    result = check_research_questions(h)
    assert result.status == "FAIL" and "INTERPRET.md" in result.detail


def test_bypass_only_passes_with_no_artifacts(tmp_path, ir_doc):
    """The narrowing: this domain used to owe a probe by §1's problem-shaped
    condition and has no use for one."""
    ir_doc["research_questions"] = _rq([{"stance": "bypass", "structure": "echelon"}])
    h = _domain(tmp_path, ir_doc, probe=False, interpret=False)
    result = check_research_questions(h)
    assert result.status == "PASS" and "no §14 artifact owed" in result.detail

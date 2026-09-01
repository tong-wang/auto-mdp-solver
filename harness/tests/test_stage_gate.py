"""Entry gates for the split pipeline (AGENT_PLAN §15).

The freeze core is the contract under test: an op may start only from a
validated, fully-confirmed IR whose mdp block is still the one the human
signed off — and a gym/rl edit must NOT void that, or Phase B's mutable
layers stop being mutable. Expensive checks (conformance / laws /
differential) are exercised only on their failure plumbing here: their happy
paths need a generated domain, which is each domain's own test file's job.

Synthetic throughout: ``harness/tests`` must not depend on ``plugin/``.
"""

from __future__ import annotations

import json

import pytest

from mdp_ir.schema import load_ir
from mdp_stage.gate import (
    OPS,
    DomainContext,
    check_conformance,
    check_differential,
    check_freeze,
    check_interpret_owed,
    check_laws,
    run_gate,
    stage_table,
)


def _confirm_all(doc: dict) -> dict:
    """Resolve every Confirmable the minimal IR leaves at source=derived."""
    doc["rl"]["requires_memory"]["source"] = "human_confirmed"
    for dec in doc["mdp"]["decisions"]:
        dec["type"]["source"] = "human_confirmed"
        dec["bounds"]["source"] = "human_confirmed"
    return doc


def _write_domain(tmp_path, doc: dict, *, sign_off: bool = True):
    d = tmp_path / "tiny"
    d.mkdir(exist_ok=True)
    schema = d / "tiny_schema.json"
    schema.write_text(json.dumps(doc))
    if sign_off:
        fp = load_ir(schema).mdp_fingerprint()
        (d / "tiny.signoff.json").write_text(json.dumps({
            "mdp_fingerprint": fp,
            "signed_off": "2026-09-01",
            "restatement": "tiny.restatement.md",
        }))
        (d / "tiny.restatement.md").write_text("# tiny\n")
    return d


def _statuses(results):
    return {r.check: r.status for r in results}


# -- freeze core -------------------------------------------------------------


def test_build_entry_passes_on_signed_off_domain(tmp_path, ir_doc, capsys):
    d = _write_domain(tmp_path, _confirm_all(ir_doc))
    results, code = run_gate(d, "build")
    assert code == 0
    assert set(_statuses(results).values()) == {"PASS"}


def test_unconfirmed_confirmable_blocks_build(tmp_path, ir_doc, capsys):
    doc = _confirm_all(ir_doc)
    doc["rl"]["requires_memory"]["source"] = "derived"
    d = _write_domain(tmp_path, doc)
    results, code = run_gate(d, "build")
    assert code == 1
    assert _statuses(results)["ir.unconfirmed"] == "FAIL"
    (bad,) = [r for r in results if r.check == "ir.unconfirmed"]
    assert "rl.requires_memory" in bad.detail


def test_missing_signoff_blocks_and_freeze_skips(tmp_path, ir_doc, capsys):
    d = _write_domain(tmp_path, _confirm_all(ir_doc), sign_off=False)
    results, code = run_gate(d, "build")
    assert code == 1
    s = _statuses(results)
    assert s["signoff.readable"] == "FAIL"
    assert s["signoff.fingerprint"] == "SKIP"


def test_post_signoff_mdp_edit_voids_the_freeze(tmp_path, ir_doc):
    doc = _confirm_all(ir_doc)
    d = _write_domain(tmp_path, doc)
    doc["mdp"]["scenario"]["constants"][0]["value"] = 2.0  # h: 1.0 -> 2.0
    (d / "tiny_schema.json").write_text(json.dumps(doc))
    r = check_freeze(DomainContext(d))
    assert r.status == "FAIL"
    assert "moved" in r.detail and "re-confirm" in r.detail


def test_gym_and_rl_edits_do_not_void_the_freeze(tmp_path, ir_doc):
    doc = _confirm_all(ir_doc)
    d = _write_domain(tmp_path, doc)
    doc["gym"]["observation_modes"][0]["features"].append({"ref": "period"})
    doc["rl"]["obs_normalization"]["enabled"] = False
    (d / "tiny_schema.json").write_text(json.dumps(doc))
    assert check_freeze(DomainContext(d)).status == "PASS"


# -- solve exit artifacts (escalate / interpret / package entries) -----------


def _with_runplan(d, target="base"):
    (d / "tiny.runplan.json").write_text(json.dumps({
        "target": target, "strategy": "specialist",
        "escalation_budget": "2 rounds", "confirmed": "2026-09-01",
    }))
    return d


def test_escalate_entry_wants_runplan_artifacts_and_log(tmp_path, ir_doc, capsys):
    d = _with_runplan(_write_domain(tmp_path, _confirm_all(ir_doc)))
    results, code = run_gate(d, "escalate")
    assert code == 1
    s = _statuses(results)
    assert s["runplan.readable"] == "PASS"
    assert s["baselines.tsv"] == "FAIL"
    assert s["rl.artifacts"] == "FAIL"
    assert s["escalation.log"] == "FAIL"


def test_package_entry_passes_with_full_artifact_tree(tmp_path, ir_doc, capsys):
    d = _with_runplan(_write_domain(tmp_path, _confirm_all(ir_doc)))
    bench = d / "results" / "base" / "benchmark"
    bench.mkdir(parents=True)
    (bench / "benchmark_random_eval_base.tsv").write_text("x\n")
    (bench / "benchmark_myopic_eval_base.tsv").write_text("x\n")
    run = d / "results" / "base" / "obs_vec"
    run.mkdir()
    (run / "model.zip").write_text("")
    (run / "tiny_ppo_eval_base.tsv").write_text("x\n")
    fp = load_ir(d / "tiny_schema.json").mdp_fingerprint()
    (run / "base_ppo_args.txt").write_text(f"ir_mdp_fingerprint: {fp}\n")
    results, code = run_gate(d, "package")
    assert code == 0
    assert set(_statuses(results).values()) == {"PASS"}


def test_one_baseline_tsv_is_not_enough(tmp_path, ir_doc, capsys):
    d = _with_runplan(_write_domain(tmp_path, _confirm_all(ir_doc)))
    bench = d / "results" / "base" / "benchmark"
    bench.mkdir(parents=True)
    (bench / "benchmark_random_eval_base.tsv").write_text("x\n")
    results, code = run_gate(d, "package")
    assert code == 1
    assert _statuses(results)["baselines.tsv"] == "FAIL"


# -- artifact currency: runs must belong to the CURRENT mdp block ------------


def _run_with_fp(d, name, fp):
    run = d / "results" / "base" / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "model.zip").write_text("")
    (run / "eval.tsv").write_text("x\n")
    if fp is not None:
        (run / "base_ppo_args.txt").write_text(f"seed: 1\nir_mdp_fingerprint: {fp}\n")
    return run


def test_all_runs_stale_after_refreeze_blocks(tmp_path, ir_doc, capsys):
    """The freeze check alone misses this: formalize legitimately re-confirms
    (sign-off rewritten, every gate unblocked) while the runs on disk still
    answer the previous model's question."""
    from mdp_stage.gate import check_rl_current
    d = _with_runplan(_write_domain(tmp_path, _confirm_all(ir_doc)))
    _run_with_fp(d, "obs_vec", "000000000000")  # not the current fingerprint
    r = check_rl_current(DomainContext(d))
    assert r.status == "FAIL"
    assert "retrain" in r.detail


def test_stale_beside_current_warns_but_does_not_block(tmp_path, ir_doc, capsys):
    from mdp_stage.gate import check_rl_current
    d = _with_runplan(_write_domain(tmp_path, _confirm_all(ir_doc)))
    fp = load_ir(d / "tiny_schema.json").mdp_fingerprint()
    _run_with_fp(d, "old_arm", "000000000000")
    _run_with_fp(d, "new_arm", fp)
    r = check_rl_current(DomainContext(d))
    assert r.status == "WARN" and "old_arm" in r.detail
    bench = d / "results" / "base" / "benchmark"
    bench.mkdir(parents=True)
    (bench / "a.tsv").write_text("x\n")
    (bench / "b.tsv").write_text("x\n")
    _, code = run_gate(d, "package")
    assert code == 0  # WARN does not fail the gate


def test_pre_provenance_runs_warn_unverifiable(tmp_path, ir_doc):
    from mdp_stage.gate import check_rl_current
    d = _with_runplan(_write_domain(tmp_path, _confirm_all(ir_doc)))
    _run_with_fp(d, "obs_vec", None)  # no args log at all
    r = check_rl_current(DomainContext(d))
    assert r.status == "WARN" and "unverifiable" in r.detail


# -- interpret informational branch ------------------------------------------


def test_interpret_owed_branches(tmp_path, ir_doc):
    doc = _confirm_all(ir_doc)
    d = _write_domain(tmp_path, doc)
    assert check_interpret_owed(DomainContext(d)).status == "SKIP"

    doc["research_questions"] = {"tier2": [
        {"stance": "confirm", "structure": "(s,S)"},
    ]}
    (d / "tiny_schema.json").write_text(json.dumps(doc))
    assert check_interpret_owed(DomainContext(d)).status == "PASS"

    doc["research_questions"] = {"tier2": [
        {"stance": "bypass", "structure": "echelon"},
    ]}
    (d / "tiny_schema.json").write_text(json.dumps(doc))
    r = check_interpret_owed(DomainContext(d))
    assert r.status == "SKIP" and "bypass" in r.detail


# -- expensive checks: failure plumbing only ---------------------------------


def test_solve_gate_runs_the_three_upstream_gates():
    names = [c.__name__ for c in OPS["solve"]]
    assert names[-3:] == ["check_conformance", "check_laws", "check_differential"]


def test_differential_without_adapter_reports_fail_not_crash(tmp_path, ir_doc, capsys):
    d = _write_domain(tmp_path, _confirm_all(ir_doc))
    r = check_differential(DomainContext(d), episodes=2)
    assert r.status == "FAIL"


def test_conformance_on_codeless_folder_reports_fail_not_crash(tmp_path, ir_doc):
    d = _write_domain(tmp_path, _confirm_all(ir_doc))
    r = check_conformance(DomainContext(d))
    assert r.status == "FAIL"


def test_laws_runs_on_the_synthetic_ir(tmp_path, ir_doc):
    d = _write_domain(tmp_path, _confirm_all(ir_doc))
    r = check_laws(DomainContext(d))
    assert r.status in {"PASS", "FAIL"}  # plumbing: it ran and reported


# -- table mode --------------------------------------------------------------


def test_stage_table_is_a_report_not_a_gate(tmp_path, ir_doc, capsys):
    d = _write_domain(tmp_path, _confirm_all(ir_doc))  # signed off, no run plan
    assert stage_table(d) == 0
    out = capsys.readouterr().out
    assert "BLOCKED" in out          # escalate/package: no run plan yet
    assert "--for solve" in out      # the expensive checks are named as unrun


def test_invalid_ir_fails_validates_check(tmp_path, ir_doc, capsys):
    doc = _confirm_all(ir_doc)
    doc["mdp"]["horizon"]["T"] = 0  # < 1: schema rejects
    d = _write_domain(tmp_path, doc, sign_off=False)
    results, code = run_gate(d, "build")
    assert code == 1
    assert _statuses(results)["ir.validates"] == "FAIL"

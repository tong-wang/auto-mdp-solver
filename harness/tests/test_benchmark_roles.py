"""Benchmark roles: the IR declaration and the two gate behaviours (upstream #23).

Spec §9.9 names three positions relative to the optimum and states one
invariant over them — a `feasible` arm cannot beat an `exact` or `relaxed` one,
and if it appears to, the eval/bound/simulator is wrong. Before this block the
role lived in prose, so the invariant could be written down but never checked.

Synthetic IRs and hand-written TSVs only: `harness/tests` must not depend on
`plugin/`.
"""

from __future__ import annotations

import pytest

from mdp_gates.compare import (
    IllTypedBaseline,
    _method_from_path,
    compare_evals,
    roles_from_ir,
)
from mdp_ir.schema import MdpIR


# --- the declaration ------------------------------------------------------

def test_ir_without_benchmarks_still_validates(ir_doc):
    """Additive: every existing IR keeps working, with an empty list."""
    ir = MdpIR.model_validate(ir_doc)
    assert ir.benchmarks == []


def test_benchmarks_block_parses_and_keeps_roles(ir_doc):
    ir_doc["benchmarks"] = [
        {"name": "dp", "role": "exact", "basis": "value iteration to 1e-9"},
        {"name": "clairvoyant", "role": "relaxed", "basis": "reads the latent"},
        {"name": "myopic", "role": "feasible", "basis": "one-step lookahead",
         "policies": ["fixed", "adaptive"], "source": "human_confirmed"},
    ]
    ir = MdpIR.model_validate(ir_doc)
    assert [b.role.value for b in ir.benchmarks] == ["exact", "relaxed", "feasible"]
    assert ir.benchmarks[2].policies == ["fixed", "adaptive"]


def test_declaring_a_benchmark_does_not_move_the_fingerprints(ir_doc, catalog_doc):
    """The whole reason the block sits at the root, as with eval_metrics:
    benchmarks are built in Phase B, so declaring one must not void the
    Phase-A freeze."""
    from mdp_ir.layering import structural_fingerprint

    before = MdpIR.model_validate(ir_doc)
    doc = dict(ir_doc)
    doc["benchmarks"] = [{"name": "dp", "role": "exact"}]
    assert MdpIR.model_validate(doc).mdp_fingerprint() == before.mdp_fingerprint()

    catalog_after = dict(catalog_doc)
    catalog_after["benchmarks"] = [{"name": "dp", "role": "exact"}]
    assert structural_fingerprint(catalog_after) == structural_fingerprint(catalog_doc)


def test_unknown_role_is_rejected(ir_doc):
    ir_doc["benchmarks"] = [{"name": "dp", "role": "optimal"}]
    with pytest.raises(Exception):
        MdpIR.model_validate(ir_doc)


def test_roles_from_ir_reads_the_block(tmp_path, ir_doc):
    ir_doc["benchmarks"] = [{"name": "dp", "role": "exact"},
                            {"name": "myopic", "role": "feasible"}]
    path = tmp_path / "d_schema.json"
    import json
    path.write_text(json.dumps(ir_doc))
    assert roles_from_ir(path) == {"dp": "exact", "myopic": "feasible"}


# --- resolving an eval file back to its declaration -----------------------

@pytest.mark.parametrize("name,expected", [
    ("inv_benchmark_dp_eval_simple.tsv", "dp"),
    ("benchmark_myopic_eval_lt.tsv", "myopic"),
    ("mab_benchmark_thompson_eval_gauss_K10_T1000.tsv", "thompson"),
    ("inv_ppo_eval_simple.tsv", None),
])
def test_method_from_path(tmp_path, name, expected):
    assert _method_from_path(tmp_path / name) == expected


# --- the gate behaviours --------------------------------------------------

def _tsv(path, mean, var=4.0):
    path.write_text(f"cost_total_mean\tcost_total_var\n{mean}\t{var}\n")
    return path


def test_baseline_on_an_exact_arm_is_refused(tmp_path):
    """--baseline is must-beat; an exact solver is the optimum."""
    cand = _tsv(tmp_path / "ppo_eval_s.tsv", 100.0)
    base = _tsv(tmp_path / "d_benchmark_dp_eval_s.tsv", 90.0)
    with pytest.raises(IllTypedBaseline, match="exact"):
        compare_evals(cand, [base], [], n_seeds=100, sense="minimize",
                      roles={"dp": "exact"})


def test_baseline_on_a_feasible_arm_is_fine(tmp_path):
    cand = _tsv(tmp_path / "ppo_eval_s.tsv", 90.0)
    base = _tsv(tmp_path / "d_benchmark_myopic_eval_s.tsv", 100.0)
    report = compare_evals(cand, [base], [], n_seeds=100, sense="minimize",
                           roles={"myopic": "feasible"})
    assert report.ok and not report.role_violations


def test_candidate_beating_an_exact_reference_is_a_bug_report(tmp_path):
    """The §9.9 invariant: this must fail the gate, not crown the policy."""
    cand = _tsv(tmp_path / "ppo_eval_s.tsv", 80.0)          # lower = better here
    ref = _tsv(tmp_path / "d_benchmark_dp_eval_s.tsv", 90.0)
    report = compare_evals(cand, [], [ref], n_seeds=100, sense="minimize",
                           roles={"dp": "exact"})
    assert report.role_violations and not report.ok
    assert "bound" in report.role_violations[0]
    assert "[BUG ]" in report.render()


def test_the_invariant_is_sense_aware(tmp_path):
    """Same numbers, maximize domain: 80 vs 90 is now the candidate LOSING."""
    cand = _tsv(tmp_path / "ppo_eval_s.tsv", 80.0)
    ref = _tsv(tmp_path / "d_benchmark_dp_eval_s.tsv", 90.0)
    report = compare_evals(cand, [], [ref], n_seeds=100, sense="maximize",
                           roles={"dp": "exact"})
    assert not report.role_violations


def test_candidate_at_the_bound_is_not_a_violation(tmp_path):
    """`= opt` is attainable; only strictly better is impossible."""
    cand = _tsv(tmp_path / "ppo_eval_s.tsv", 90.0)
    ref = _tsv(tmp_path / "d_benchmark_dp_eval_s.tsv", 90.0)
    report = compare_evals(cand, [], [ref], n_seeds=100, sense="minimize",
                           roles={"dp": "exact"})
    assert not report.role_violations


def test_no_roles_declared_means_no_new_behaviour(tmp_path):
    """Undeclared domains gate exactly as before — the block is optional."""
    cand = _tsv(tmp_path / "ppo_eval_s.tsv", 80.0)
    base = _tsv(tmp_path / "d_benchmark_dp_eval_s.tsv", 90.0)
    report = compare_evals(cand, [base], [], n_seeds=100, sense="minimize")
    assert report.ok and not report.role_violations


# --- the ordering band, and column-sourced arms (upstream #27, #26 Part C) ---

def test_inside_the_band_reports_rather_than_fails(tmp_path):
    """§9.9's claim is about what is possible in principle; the check must not
    test it to the last float on two Monte-Carlo means."""
    cand = _tsv(tmp_path / "ppo_eval_s.tsv", 89.9999, var=4.0)
    ref = _tsv(tmp_path / "d_benchmark_dp_eval_s.tsv", 90.0, var=4.0)
    report = compare_evals(cand, [], [ref], n_seeds=100, sense="minimize",
                           roles={"dp": "exact"})
    assert not report.role_violations          # not a defect claim
    assert report.warnings and "within the band" in report.warnings[-1]
    assert report.ok


def test_outside_the_band_still_violates(tmp_path):
    cand = _tsv(tmp_path / "ppo_eval_s.tsv", 80.0, var=0.01)
    ref = _tsv(tmp_path / "d_benchmark_dp_eval_s.tsv", 90.0, var=0.01)
    report = compare_evals(cand, [], [ref], n_seeds=100, sense="minimize",
                           roles={"dp": "exact"})
    assert report.role_violations and not report.ok


def test_declared_tolerance_widens_the_band(tmp_path):
    """Implementation slack declared by whoever wrote the solver, not inferred."""
    cand = _tsv(tmp_path / "ppo_eval_s.tsv", 89.0, var=0.0001)
    ref = _tsv(tmp_path / "d_benchmark_dp_eval_s.tsv", 90.0, var=0.0001)
    strict = compare_evals(cand, [], [ref], n_seeds=100, sense="minimize",
                           roles={"dp": "exact"})
    assert strict.role_violations              # 1.0 ahead, SE ~ 0
    lenient = compare_evals(cand, [], [ref], n_seeds=100, sense="minimize",
                            roles={"dp": "exact"}, tolerances={"dp": 2.0})
    assert not lenient.role_violations and lenient.warnings


def test_z_role_zero_restores_the_strict_comparison(tmp_path):
    cand = _tsv(tmp_path / "ppo_eval_s.tsv", 89.9999, var=4.0)
    ref = _tsv(tmp_path / "d_benchmark_dp_eval_s.tsv", 90.0, var=4.0)
    report = compare_evals(cand, [], [ref], n_seeds=100, sense="minimize",
                           roles={"dp": "exact"}, z_role=0.0)
    assert report.role_violations


def test_column_sourced_benchmark_parses_and_carries_tolerance(ir_doc):
    ir_doc["benchmarks"] = [
        {"name": "oracle", "role": "relaxed", "column": "oracle_mean",
         "basis": "clairvoyant: reads the realized arm means", "tolerance": 0.5},
    ]
    ir = MdpIR.model_validate(ir_doc)
    assert ir.benchmarks[0].column == "oracle_mean"
    assert ir.benchmarks[0].tolerance == 0.5


def test_tolerances_from_ir_skips_the_zero_default(tmp_path, ir_doc):
    import json
    ir_doc["benchmarks"] = [{"name": "dp", "role": "exact", "tolerance": 1.5},
                            {"name": "myopic", "role": "feasible"}]
    path = tmp_path / "d_schema.json"
    path.write_text(json.dumps(ir_doc))
    from mdp_gates.compare import tolerances_from_ir
    assert tolerances_from_ir(path) == {"dp": 1.5}

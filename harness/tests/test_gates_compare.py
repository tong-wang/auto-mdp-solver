"""Eval gate: the sense-aware verdict.

A maximize metric passes when the candidate is higher; a minimize metric
(cost/regret) passes when the candidate is LOWER. The minimize cases are the
regression guard for the silently-inverted gate a cost-reporting domain would
otherwise get under the maximize default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mdp_gates.compare import compare_evals

# SE = sqrt((var_c + var_b) / N); with var=1 each, N=1000 → SE ≈ 0.0447, so a
# 1.0 mean gap is decisive either way and the verdict turns purely on sense
N = 1000


def _tsv(dir_: Path, name: str, metric: str, mean: float, var: float) -> Path:
    """A one-row spec-§9 eval TSV with a {metric}/{metric_var} pair."""
    p = dir_ / name
    p.write_text(f"{metric}\t{metric.replace('_mean', '_var')}\n{mean}\t{var}\n")
    return p


@pytest.fixture
def evals(tmp_path: Path) -> dict[str, Path]:
    return {
        "hi": _tsv(tmp_path, "cand_hi.tsv", "revenue_mean", 10.0, 1.0),
        "lo": _tsv(tmp_path, "base_lo.tsv", "revenue_mean", 9.0, 1.0),
        "cost_lo": _tsv(tmp_path, "cand_cost_lo.tsv", "cost_total_mean", 40.0, 1.0),
        "cost_hi": _tsv(tmp_path, "base_cost_hi.tsv", "cost_total_mean", 50.0, 1.0),
    }


def test_maximize_higher_candidate_passes(evals):
    rep = compare_evals(evals["hi"], [evals["lo"]], [], n_seeds=N, sense="maximize")
    assert rep.ok and rep.comparisons[0].z > 0


def test_maximize_lower_candidate_fails(evals):
    rep = compare_evals(evals["lo"], [evals["hi"]], [], n_seeds=N, sense="maximize")
    assert not rep.ok and rep.comparisons[0].z < 0


def test_minimize_lower_cost_candidate_passes(evals):
    rep = compare_evals(evals["cost_lo"], [evals["cost_hi"]], [], n_seeds=N,
                        sense="minimize")
    assert rep.ok and rep.comparisons[0].z > 0
    # the raw candidate-baseline difference is negative while the gate PASSES
    assert rep.comparisons[0].diff < 0


def test_minimize_higher_cost_candidate_fails(evals):
    rep = compare_evals(evals["cost_hi"], [evals["cost_lo"]], [], n_seeds=N,
                        sense="minimize")
    assert not rep.ok and rep.comparisons[0].z < 0


def test_sense_flips_the_verdict_on_identical_data(evals):
    """The whole point: same TSVs, wrong sense, inverted answer."""
    args = (evals["cost_lo"], [evals["cost_hi"]], [])
    as_min = compare_evals(*args, n_seeds=N, sense="minimize")
    as_max = compare_evals(*args, n_seeds=N, sense="maximize")
    assert as_min.ok and not as_max.ok


def test_default_sense_is_maximize(evals):
    rep = compare_evals(evals["hi"], [evals["lo"]], [], n_seeds=N)
    assert rep.sense == "maximize" and rep.ok


def test_invalid_sense_raises(evals):
    with pytest.raises(ValueError):
        compare_evals(evals["hi"], [evals["lo"]], [], n_seeds=N, sense="min")


def test_explicit_metric_emits_no_warning(evals):
    rep = compare_evals(evals["hi"], [evals["lo"]], [], n_seeds=N,
                        metric="revenue_mean")
    assert not rep.warnings


def test_auto_selected_metric_warns_and_names_the_column(evals):
    rep = compare_evals(evals["hi"], [evals["lo"]], [], n_seeds=N)
    assert len(rep.warnings) == 1
    assert "revenue_mean" in rep.warnings[0] and "auto-selected" in rep.warnings[0]


def test_ambiguous_auto_pick_lists_the_alternatives(tmp_path: Path):
    multi = tmp_path / "multi.tsv"
    multi.write_text(
        "cost_total_mean\tcost_total_var\treturn_mean\treturn_var\n"
        "40.0\t1.0\t-40.0\t1.0\n")
    rep = compare_evals(multi, [multi], [], n_seeds=N, sense="minimize")
    assert rep.warnings and "return_mean" in rep.warnings[0]

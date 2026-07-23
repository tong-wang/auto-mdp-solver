"""Sanity tests for the eval gate — run from the repo root:

    python -m mdp_gates.compare_test

Covers the sense-aware verdict: a maximize metric passes when the candidate is
higher, a minimize metric (cost/regret) passes when the candidate is LOWER.
The minimize case is the regression guard for the silently-inverted gate a
cost-reporting domain would otherwise get under the maximize default.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mdp_gates.compare import compare_evals

_checks = 0


def check(cond: bool, label: str) -> None:
    global _checks
    assert cond, label
    _checks += 1
    print(f"ok  {label}")


def _tsv(dir_: Path, name: str, metric: str, mean: float, var: float) -> Path:
    """Write a one-row spec-§9 eval TSV with a {metric}/{metric_var} pair."""
    var_col = metric.replace("_mean", "_var")
    p = dir_ / name
    p.write_text(f"{metric}\t{var_col}\n{mean}\t{var}\n")
    return p


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        N = 1000  # SE = sqrt((var_c+var_b)/N); var=1 each → SE ≈ 0.0447

        # -- maximize: higher mean is better ---------------------------------
        hi = _tsv(d, "cand_hi.tsv", "revenue_mean", mean=10.0, var=1.0)
        lo = _tsv(d, "base_lo.tsv", "revenue_mean", mean=9.0, var=1.0)
        rep = compare_evals(hi, [lo], [], n_seeds=N, sense="maximize")
        check(rep.ok and rep.comparisons[0].z > 0,
              "maximize: higher-mean candidate beats lower-mean baseline")
        rep = compare_evals(lo, [hi], [], n_seeds=N, sense="maximize")
        check(not rep.ok and rep.comparisons[0].z < 0,
              "maximize: lower-mean candidate fails")

        # -- minimize: lower mean is better (the regression guard) -----------
        cost_lo = _tsv(d, "cand_cost_lo.tsv", "cost_total_mean", mean=40.0, var=1.0)
        cost_hi = _tsv(d, "base_cost_hi.tsv", "cost_total_mean", mean=50.0, var=1.0)
        rep = compare_evals(cost_lo, [cost_hi], [], n_seeds=N, sense="minimize")
        check(rep.ok and rep.comparisons[0].z > 0,
              "minimize: lower-cost candidate PASSES vs higher-cost baseline")
        check(rep.comparisons[0].diff < 0,
              "minimize: raw cand-base diff is negative while the gate PASSES")
        rep = compare_evals(cost_hi, [cost_lo], [], n_seeds=N, sense="minimize")
        check(not rep.ok and rep.comparisons[0].z < 0,
              "minimize: higher-cost candidate FAILS")

        # -- the inversion: same data, wrong sense flips the verdict ---------
        as_max = compare_evals(cost_lo, [cost_hi], [], n_seeds=N, sense="maximize")
        as_min = compare_evals(cost_lo, [cost_hi], [], n_seeds=N, sense="minimize")
        check(as_min.ok and not as_max.ok,
              "same cost TSVs: minimize PASSES, maximize (wrong) FAILS — sense matters")

        # -- default sense is maximize (back-compatible) ---------------------
        rep_default = compare_evals(hi, [lo], [], n_seeds=N)
        check(rep_default.sense == "maximize" and rep_default.ok,
              "default sense is maximize (back-compatible)")

        # -- bad sense rejected ----------------------------------------------
        try:
            compare_evals(hi, [lo], [], n_seeds=N, sense="min")
            check(False, "invalid sense must raise")
        except ValueError:
            check(True, "invalid sense value raises ValueError")

        # -- auto-metric warning ---------------------------------------------
        # explicit --metric: no warning
        rep = compare_evals(hi, [lo], [], n_seeds=N, metric="revenue_mean")
        check(not rep.warnings, "explicit metric: no auto-select warning")
        # auto-pick (metric=None): one warning naming the picked column
        rep = compare_evals(hi, [lo], [], n_seeds=N)
        check(len(rep.warnings) == 1 and "revenue_mean" in rep.warnings[0]
              and "auto-selected" in rep.warnings[0],
              "auto-selected metric emits a warning naming the column")
        # ambiguity: multiple *_mean columns → warning lists the others
        (d / "multi.tsv").write_text(
            "cost_total_mean\tcost_total_var\treturn_mean\treturn_var\n"
            "40.0\t1.0\t-40.0\t1.0\n")
        base_multi = d / "multi.tsv"
        rep = compare_evals(base_multi, [base_multi], [], n_seeds=N, sense="minimize")
        check(rep.warnings and "return_mean" in rep.warnings[0],
              "auto-pick with several *_mean columns warns and lists the others")

    print(f"\nall {_checks} checks passed")


if __name__ == "__main__":
    main()

"""Sanity tests for the tuning metric resolver — run from the repo root:

    python -m mdp_tuning.resolve_metric_test

Guards the domain-agnostic auto-pick: the first *_mean column in header order
is the objective (spec §9.3), with no metric name privileged. The regression
this locks in is that a non-pricing domain is no longer overridden by a
hard-coded `profit_mean` preference.
"""

from __future__ import annotations

from mdp_tuning.driver import resolve_metric

_checks = 0


def check(cond: bool, label: str) -> None:
    global _checks
    assert cond, label
    _checks += 1
    print(f"ok  {label}")


def main() -> None:
    # -- explicit request wins --------------------------------------------
    cols = ["stdev", "T", "profit_mean", "regret_mean", "profit_var"]
    means = {"profit_mean": 1.0, "regret_mean": 0.0, "profit_var": 1.0}
    check(resolve_metric("regret_mean", cols, means) == "regret_mean",
          "explicit metric is returned as requested")
    try:
        resolve_metric("nope_mean", cols, means)
        check(False, "unknown explicit metric must raise")
    except ValueError:
        check(True, "unknown explicit metric raises ValueError")

    # -- auto = first *_mean in header order ------------------------------
    check(resolve_metric("auto", cols, means) == "profit_mean",
          "auto picks the first *_mean column (pricing: profit_mean)")

    # a cost domain leads with cost_total_mean; nothing privileges profit
    cost_cols = ["holding", "cost_total_mean", "cost_total_var", "service_mean"]
    cost_means = {"cost_total_mean": 40.0, "cost_total_var": 1.0, "service_mean": 0.9}
    check(resolve_metric("auto", cost_cols, cost_means) == "cost_total_mean",
          "auto picks cost_total_mean for a cost domain (no profit privileging)")

    # profit_mean present but NOT first → auto must NOT jump to it
    reordered = ["service_mean", "service_var", "profit_mean", "profit_var"]
    reordered_means = {"service_mean": 0.9, "service_var": 0.01,
                       "profit_mean": 5.0, "profit_var": 1.0}
    check(resolve_metric("auto", reordered, reordered_means) == "service_mean",
          "auto follows header order, not a hard-coded profit_mean preference")

    # -- no *_mean column → error -----------------------------------------
    try:
        resolve_metric("auto", ["a", "b"], {"a": 1.0, "b": 2.0})
        check(False, "auto with no *_mean column must raise")
    except ValueError:
        check(True, "auto with no *_mean column raises ValueError")

    print(f"\nall {_checks} checks passed")


if __name__ == "__main__":
    main()

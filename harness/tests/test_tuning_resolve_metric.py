"""Tuning metric resolver: the domain-agnostic auto-pick.

The first ``*_mean`` column in header order is the objective (spec §9.3), with
no metric name privileged. The regression locked in here is that a non-pricing
domain is not overridden by a hard-coded ``profit_mean`` preference.
"""

from __future__ import annotations

import pytest

from mdp_tuning.driver import resolve_metric

COLS = ["stdev", "T", "profit_mean", "regret_mean", "profit_var"]
MEANS = {"profit_mean": 1.0, "regret_mean": 0.0, "profit_var": 1.0}


def test_explicit_metric_is_returned_as_requested():
    assert resolve_metric("regret_mean", COLS, MEANS) == "regret_mean"


def test_unknown_explicit_metric_raises():
    with pytest.raises(ValueError):
        resolve_metric("nope_mean", COLS, MEANS)


def test_auto_picks_the_first_mean_column():
    assert resolve_metric("auto", COLS, MEANS) == "profit_mean"


def test_auto_picks_cost_for_a_cost_domain():
    """Nothing privileges profit: a cost domain leads with cost_total_mean."""
    cols = ["holding", "cost_total_mean", "cost_total_var", "service_mean"]
    means = {"cost_total_mean": 40.0, "cost_total_var": 1.0, "service_mean": 0.9}
    assert resolve_metric("auto", cols, means) == "cost_total_mean"


def test_auto_follows_header_order_not_a_profit_preference():
    """profit_mean present but NOT first — auto must not jump to it."""
    cols = ["service_mean", "service_var", "profit_mean", "profit_var"]
    means = {"service_mean": 0.9, "service_var": 0.01,
             "profit_mean": 5.0, "profit_var": 1.0}
    assert resolve_metric("auto", cols, means) == "service_mean"


def test_auto_with_no_mean_column_raises():
    with pytest.raises(ValueError):
        resolve_metric("auto", ["a", "b"], {"a": 1.0, "b": 2.0})

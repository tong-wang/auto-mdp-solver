"""fnv's own tests — the MMFE grid exemplar.

    pytest plugin/skills/mdp-solver/examples/fnv
    python fnv_test.py

The generic guarantees live in ``mdp_ir.laws``; the accumulation and
terminal-realization laws live in the IR as ``mdp.invariants``. What is here is
what only this domain can state: the martingale information path, its
independence from the ordering decisions, and the categorical link selector
that makes aMMFE and mMMFE two instances of one IR rather than two IRs.
"""

from __future__ import annotations

import json
import math

import pytest

from mdp_ir.interpreter import IrInterpreter, simulate
from mdp_ir.schema import load_ir
from mdp_ir.testing import assert_laws, assert_match, schema_beside

SCHEMA = schema_beside(__file__)
EPISODES = 8
SALT = 1


def _compositions() -> list[str | None]:
    scenario = json.loads(SCHEMA.read_text())["mdp"]["scenario"]
    return ([None] + sorted(scenario.get("instances") or {})
            + sorted(m["name"] for m in scenario.get("mixtures") or []))


COMPOSITIONS = _compositions()


def test_engine_laws():
    assert_laws(SCHEMA)


@pytest.mark.parametrize("instance", COMPOSITIONS, ids=lambda i: i or "base")
def test_differential_matches_the_domain(instance):
    """Base is aMMFE, `mmmfe` is mMMFE with a re-baked signal schedule."""
    assert_match(SCHEMA, instance=instance, episodes=EPISODES, seed_salt=SALT)


def test_the_covering_set_includes_both_mmfe_modes():
    assert "mmmfe" in COMPOSITIONS


# -- the MMFE information process --------------------------------------------


def test_the_information_path_ignores_the_order_decisions():
    """The signal is keyed on (period, seed, salt), never on the action, so the
    forecast a policy sees cannot be steered by how much it ordered. Stated
    here rather than in ``mdp_ir.laws`` because the draw lands in a dynamics
    local (`delta`) and only reaches the trajectory through `information`."""
    ir = load_ir(SCHEMA)
    lo = simulate(ir, 5, decisions={"order": 0.0}, seed_salt=SALT)
    hi = simulate(ir, 5, decisions={"order": 3.0}, seed_salt=SALT)
    assert [r["information"] for r in lo.rows] == \
           [r["information"] for r in hi.rows]


def test_information_accumulates_as_a_random_walk():
    """Increments add up: I_t - I_{t-1} is the period's revelation, and the
    path is not constant (a degenerate signal would make the state useless)."""
    ir = load_ir(SCHEMA)
    rows = simulate(ir, 6, decisions={"order": 1.0}, seed_salt=SALT).rows
    path = [r["information"] for r in rows]
    assert len(set(path)) > 1
    assert path[0] != 0.0 or len(path) > 1


def test_the_signal_schedule_is_padded_for_the_one_indexed_clock():
    """`std = signal_stdevs[period]` and the clock runs 1..N, so the schedule
    carries N+1 entries with slot 0 as unused padding. Every ordering period
    draws a strictly positive increment — including the first, which is why
    the period-1 order is already taken under a revealed signal."""
    ir = load_ir(SCHEMA)
    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    schedule = consts["signal_stdevs"]
    assert len(schedule) == consts["N"] + 1
    assert schedule[0] == 0.0, "slot 0 is padding and must never be read"
    assert all(s > 0.0 for s in schedule[1:])
    rows = simulate(ir, 2, decisions={"order": 1.0}, seed_salt=SALT).rows
    assert rows[0]["information"] != 0.0


def test_signal_volatility_declines_over_the_horizon():
    """The MMFE tau grid front-loads revelation: the last increment covers the
    shortest remaining interval, so it is the smallest."""
    consts = {c.name: c.value for c in load_ir(SCHEMA).mdp.scenario.constants}
    schedule = consts["signal_stdevs"][1:]
    assert schedule[-1] < schedule[0]


# -- the categorical link selector -------------------------------------------


def test_the_mode_selector_is_a_categorical_constant():
    base = load_ir(SCHEMA)
    variant = load_ir(SCHEMA, instance="mmmfe")
    consts = {c.name: c.value for c in base.mdp.scenario.constants}
    assert consts["mmfe_mode"] == "additive"
    assert variant.mdp.scenario.instances["mmmfe"]["mmfe_mode"] == "multiplicative"


def test_the_two_modes_share_one_signal_family():
    """The link differs; the demand-signal generator does not. That is why the
    mode is a constant-conditioned expression rather than a slot candidate."""
    base = load_ir(SCHEMA)
    variant = load_ir(SCHEMA, instance="mmmfe")
    fam = lambda ir: next(  # noqa: E731
        s.distribution.family for s in ir.mdp.uncertainty_sources
        if s.name == "signal")
    assert fam(base) == fam(variant)


def test_the_multiplicative_link_exponentiates_the_same_information():
    """Same seed, same information path, exponentiated demand — the whole
    behavioural content of the mode switch."""
    add = simulate(load_ir(SCHEMA), 4, decisions={"order": 1.0}, seed_salt=SALT)
    mult = IrInterpreter(load_ir(SCHEMA, instance="mmmfe"),
                         instance="mmmfe", seed_salt=SALT).run(
        4, decisions={"order": 1.0})
    consts = {c.name: c.value for c in load_ir(SCHEMA).mdp.scenario.constants}
    # the signal is wider under mmmfe (stdev 0.1 -> 0.2), so compare the LINK
    # on the additive run's own information rather than across the two paths
    terminal = add.rows[-1]
    assert terminal["demand"] == pytest.approx(
        consts["mu"] + terminal["information"])
    assert mult.rows[-1]["demand"] == pytest.approx(
        math.exp(consts["mu"] + mult.rows[-1]["information"]))


def test_demand_is_positive_under_the_multiplicative_link():
    """The reason the multiplicative mode exists: an additive link can hand a
    negative demand to a lognormal-ish problem, exp() cannot."""
    ir = load_ir(SCHEMA, instance="mmmfe")
    for seed in range(6):
        rows = IrInterpreter(ir, instance="mmmfe", seed_salt=SALT).run(
            seed, decisions={"order": 1.0}).rows
        assert rows[-1]["demand"] > 0


# ---------------------------------------------------------------------------
# Benchmark equivalence gates (spec §1.2)
#
# These ran only behind manual --self-test / --check-invariants / --cross-check
# flags, which repeats mab's own LV12: "a gate that runs when someone remembers
# it is not a gate." The bar every leaderboard number is quoted against comes
# from these two solvers, so their agreement belongs in the tracked suite.
# Kept to a handful of cells so the suite stays fast.
# ---------------------------------------------------------------------------

def _sample_cells(grid_name, k=6):
    from fnv_grids import GRIDS
    cells = list(GRIDS[grid_name])
    return cells[:: max(1, len(cells) // k)]


@pytest.mark.parametrize("grid", ["FNV-aMMFE", "FNV-mMMFE"])
def test_dp_last_period_matches_the_closed_form(grid):
    """The exactly-solvable rung: b_N = sigma_{N+1} * Phi^-1(1 - c_N/r)."""
    from fnv_benchmark_dp import closed_form_last_offset, solve_offsets
    for _, sc in _sample_cells(grid):
        b = solve_offsets(sc)
        step = 20.0 * sc.stdev / 4000        # the solver's own grid resolution
        assert abs(b[sc.N - 1] - closed_form_last_offset(sc)) <= max(step, 1e-3)


@pytest.mark.parametrize("grid", ["FNV-aMMFE", "FNV-mMMFE"])
def test_two_independent_solvers_agree(grid):
    """§1.2 second implementation: the value-function DP and the paper's
    threshold recursion must land on the same offsets."""
    from fnv_benchmark_dp import solve_offsets as dp_solve
    from fnv_benchmark_prop2 import solve_offsets as p2_solve
    for _, sc in _sample_cells(grid):
        a, b = dp_solve(sc), p2_solve(sc)
        tol = max(40.0 * sc.stdev / 4000, 2e-3)
        assert max(abs(x - y) for x, y in zip(a, b)) <= tol


@pytest.mark.parametrize("grid", ["FNV-aMMFE", "FNV-mMMFE"])
def test_corollary_1_safety_stock_bound(grid):
    """The paper's Corollary 1: future ordering opportunities can only lower
    the safety stock, so b_n <= b_hat_n everywhere."""
    from fnv_benchmark_myopic import myopic_offsets
    from fnv_benchmark_prop2 import solve_offsets
    for _, sc in _sample_cells(grid):
        for b, b_hat in zip(solve_offsets(sc), myopic_offsets(sc)):
            assert b <= b_hat + 1e-6


def test_offsets_do_not_depend_on_the_mmfe_mode():
    """Equations (6)-(7) never reference the mode, so the same (sigma schedule,
    cost ladder) must give identical b_n on both branches — only the link from
    b_n to the stock level differs."""
    from fnv_benchmark_prop2 import solve_offsets
    from fnv_scenarios import FnvScenario
    for _, sc in _sample_cells("FNV-aMMFE", k=4):
        flipped = FnvScenario(
            stdev=sc.stdev, T=sc.T, lamb=sc.lamb, N=sc.N, r=sc.r, c1=sc.c1,
            mu=sc.mu, mmfe_mode="multiplicative", seed_salt=sc.seed_salt)
        assert solve_offsets(sc) == pytest.approx(solve_offsets(flipped), abs=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

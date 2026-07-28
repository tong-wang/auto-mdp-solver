"""dynamic_pricing's own tests.

    pytest plugin/skills/mdp-solver/examples/dynamic_pricing
    python dynamic_pricing_test.py

The generic guarantees live in ``mdp_ir.laws``; the sales/stock consistency
laws live in the IR as ``mdp.invariants``. What is here is what only this
domain can state: a demand intensity that responds to the *current* price
while remaining independent of the price *path*, and the absorbing sell-out.
"""

from __future__ import annotations

import json

import pytest

from mdp_ir.interpreter import IrInterpreter, simulate
from mdp_ir.schema import load_ir
from mdp_ir.testing import assert_laws, assert_match, schema_beside

SCHEMA = schema_beside(__file__)
EPISODES = 8


def _compositions() -> list[str | None]:
    scenario = json.loads(SCHEMA.read_text())["mdp"]["scenario"]
    return ([None] + sorted(scenario.get("instances") or {})
            + sorted(m["name"] for m in scenario.get("mixtures") or []))


COMPOSITIONS = _compositions()


def test_engine_laws():
    assert_laws(SCHEMA)


@pytest.mark.parametrize("instance", COMPOSITIONS, ids=lambda i: i or "base")
def test_differential_matches_the_domain(instance):
    assert_match(SCHEMA, instance=instance, episodes=EPISODES)


def test_the_covering_set_includes_the_declared_instance():
    assert "ample_stock" in COMPOSITIONS


# -- a state-conditioned generator that is still path-independent ------------


def test_arrivals_depend_on_the_current_price_only():
    """Arrival intensity is a function of the price set *this* period, so two
    histories that differ earlier but agree now must see identical arrivals.
    ``mdp_ir.laws`` cannot assert this — it skips state-conditioned draws,
    precisely because the naive form of the law does not hold for them."""
    ir = load_ir(SCHEMA)
    flat = simulate(ir, 11, decisions={"price": 1.0})
    switch = simulate(
        ir, 11, decisions=lambda ns: {"price": 5.0 if ns["period"] < 5 else 1.0})
    n = min(len(flat.rows), len(switch.rows))
    assert [r["arrivals"] for r in flat.rows[5:n]] == \
           [r["arrivals"] for r in switch.rows[5:n]]


def test_a_higher_price_suppresses_arrivals():
    """The demand curve's direction — the sign that makes the pricing problem
    a trade-off rather than a giveaway."""
    ir = load_ir(SCHEMA)
    cheap = simulate(ir, 3, decisions={"price": 0.5})
    dear = simulate(ir, 3, decisions={"price": 8.0})
    assert sum(r["arrivals"] for r in dear.rows) < \
           sum(r["arrivals"] for r in cheap.rows)


def test_ample_stock_instance_lowers_the_arrival_scale():
    ir = load_ir(SCHEMA, instance="ample_stock")
    base = simulate(load_ir(SCHEMA), 9, decisions={"price": 1.0})
    ample = IrInterpreter(ir, instance="ample_stock").run(
        9, decisions={"price": 1.0})
    assert sum(r["arrivals"] for r in ample.rows) < \
           sum(r["arrivals"] for r in base.rows)


# -- the absorbing state -----------------------------------------------------


def test_giving_stock_away_sells_out_and_terminates_early():
    ir = load_ir(SCHEMA)
    traj = simulate(ir, 1, decisions={"price": 0.0})
    assert traj.terminated_early
    assert traj.rows[-1]["inventory"] == 0
    assert len(traj.rows) < ir.mdp.horizon_T()


def test_reward_follows_the_maximize_sense():
    """A maximize domain's reward is +total, not -total: the inversion this
    pins is the one that silently trains a policy to lose money."""
    ir = load_ir(SCHEMA)
    assert ir.mdp.objective.sense.value == "maximize"
    traj = simulate(ir, 9, decisions={"price": 1.0})
    assert all(r["reward"] == pytest.approx(r["total"]) for r in traj.rows)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

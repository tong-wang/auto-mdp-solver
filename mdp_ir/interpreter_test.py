"""Sanity tests for the IR interpreter — run from the repo root:

    python -m mdp_ir.interpreter_test

Checks the invariants the interpreter must guarantee by construction
(determinism, decision-path independence, seeding) plus IR-derived property
checks (conservation laws, instance overrides, termination) on the two
example IRs. These same invariants are what the Stage-1 differential gate
will assert against generated ``_mdp`` code.
"""

from __future__ import annotations

from pathlib import Path

from mdp_ir.interpreter import IrInterpreter, simulate
from mdp_ir.schema import load_ir

ROOT = Path(__file__).resolve().parent.parent  # repo root; fixtures under examples/

_checks = 0


def check(cond: bool, label: str) -> None:
    global _checks
    assert cond, label
    _checks += 1
    print(f"ok  {label}")


def main() -> None:
    inv = load_ir(ROOT / "examples" / "inv_single" / "inv_single_schema.json")
    vrz = load_ir(ROOT / "examples" / "dynamic_pricing" / "vanryzin_pricing_schema.json")

    # -- determinism ---------------------------------------------------------
    for ir, dec in ((inv, {"order": 40.0}), (vrz, {"price": 1.0})):
        a = simulate(ir, episode_seed=7, decisions=dec)
        b = simulate(ir, episode_seed=7, decisions=dec)
        check(a.rows == b.rows, f"{ir.domain.name}: same seed -> identical trajectory")
        c = simulate(ir, episode_seed=8, decisions=dec)
        check(a.rows != c.rows, f"{ir.domain.name}: different seed -> different trajectory")

    # random-policy episodes are deterministic too (policy seeded from episode)
    a = simulate(inv, episode_seed=3)
    b = simulate(inv, episode_seed=3)
    check(a.rows == b.rows, "inv_single: random policy is seed-deterministic")

    # -- decision-path independence ------------------------------------------
    lo = simulate(inv, episode_seed=5, decisions={"order": 0.0})
    hi = simulate(inv, episode_seed=5, decisions={"order": 50.0})
    check(
        [r["demand"] for r in lo.rows] == [r["demand"] for r in hi.rows],
        "inv_single: demand independent of the order path",
    )
    check(
        [r["leadtime"] for r in hi.rows]
        and all(r["leadtime"] == 0 for r in lo.rows),
        "inv_single: leadtime drawn only when the O guard fires",
    )

    # vanryzin: arrivals at t depend on the CURRENT price only — histories
    # that differ before t but agree at t see identical period-t arrivals
    base = simulate(vrz, episode_seed=11, decisions={"price": 1.0})
    switch = simulate(
        vrz, episode_seed=11,
        decisions=lambda ns: {"price": 5.0 if ns["period"] < 5 else 1.0},
    )
    n = min(len(base.rows), len(switch.rows))
    check(
        [r["arrivals"] for r in base.rows[5:n]]
        == [r["arrivals"] for r in switch.rows[5:n]],
        "vanryzin: arrivals independent of the past price path",
    )

    # -- episode-level realization ---------------------------------------------
    demands = {r["demand"] for r in simulate(inv, episode_seed=2, decisions={"order": 0.0}).rows}
    check(len(demands) <= 5, "inv_single: per-episode demand support has <= 5 values")

    # -- conservation (IR-derived property checks) ------------------------------
    t = simulate(inv, episode_seed=9, decisions={"order": 30.0})
    inv0 = 0.0
    pipe0 = 0.0
    for r in t.rows:
        check_ok = abs(r["inventory"] - (inv0 + r["received"] - r["demand"])) < 1e-9
        assert check_ok, f"inventory balance broken at t={r['t']}"
        pipe_ok = abs(sum(r["pipeline"]) - (pipe0 + r["order"] - r["received"])) < 1e-9
        assert pipe_ok, f"pipeline balance broken at t={r['t']}"
        inv0, pipe0 = r["inventory"], sum(r["pipeline"])
    check(True, "inv_single: inventory + pipeline conservation over the episode")

    v = simulate(vrz, episode_seed=9, decisions={"price": 1.0})
    check(
        all(r["units_sold"] == min(r["arrivals"], r["inventory"] + r["units_sold"]) for r in v.rows)
        and all(r["lost_demand"] == r["arrivals"] - r["units_sold"] for r in v.rows),
        "vanryzin: sales capped by stock; lost_demand consistent",
    )

    # -- reward modes ------------------------------------------------------------
    check(
        all(abs(r["reward"] + r["total"]) < 1e-9 for r in t.rows),
        "inv_single: reward == -cost.total (sense=minimize)",
    )
    check(
        all(abs(r["reward"] - r["total"]) < 1e-9 for r in v.rows),
        "vanryzin: reward == revenue.total (sense=maximize)",
    )

    # -- termination ---------------------------------------------------------------
    check(
        len(t.rows) == inv.mdp.horizon_T() and not t.terminated_early,
        "inv_single: runs exactly T periods",
    )
    sellout = simulate(vrz, episode_seed=1, decisions={"price": 0.0})
    check(
        sellout.terminated_early and sellout.rows[-1]["inventory"] == 0,
        "vanryzin: price=0 sells out -> early termination at inventory==0",
    )
    check(
        all(r["inventory"] >= 0 for r in sellout.rows),
        "vanryzin: inventory never negative (no backlog)",
    )

    # -- scenario instances -----------------------------------------------------
    ls = IrInterpreter(inv, instance="lost_sales").run(5, decisions={"order": 0.0})
    check(
        all(r["inventory"] >= 0 for r in ls.rows) and any(r["lost_sales"] > 0 for r in ls.rows),
        "inv_single[lost_sales]: no backlog, unmet demand recorded as lost_sales",
    )
    ample = IrInterpreter(vrz, instance="ample_stock").run(9, decisions={"price": 1.0})
    check(
        sum(r["arrivals"] for r in ample.rows) < sum(r["arrivals"] for r in v.rows),
        "vanryzin[ample_stock]: a=20 -> fewer arrivals than base a=100",
    )

    # -- horizon as a scenario constant (T varies per instance) ------------------
    from mdp_ir.schema import MdpIR

    hz = inv.model_dump(mode="json")
    hz["mdp"]["scenario"]["constants"].append(
        {"name": "n_periods", "value": inv.mdp.horizon_T(), "axis": "sizing"}
    )
    hz["mdp"]["horizon"]["T"] = "n_periods"
    hz["mdp"]["scenario"]["instances"]["short"] = {"n_periods": 3}
    hz_ir = MdpIR.model_validate(hz)
    base_len = len(simulate(hz_ir, episode_seed=0, decisions={"order": 0.0}).rows)
    short = IrInterpreter(hz_ir, instance="short").run(0, decisions={"order": 0.0})
    check(
        base_len == inv.mdp.horizon_T() and len(short.rows) == 3,
        "horizon.T names a constant: base length unchanged, instance override -> 3 periods",
    )
    try:
        MdpIR.model_validate(
            {**hz, "mdp": {**hz["mdp"], "horizon": {**hz["mdp"]["horizon"], "T": "no_such"}}}
        )
        check(False, "horizon.T naming an unknown constant must fail validation")
    except ValueError:
        check(True, "horizon.T naming an unknown constant fails validation")

    # -- observation modes -------------------------------------------------------
    interp = IrInterpreter(inv)
    row = t.rows[0]
    obs = interp.observe("vec_ip", row)
    check(
        abs(obs["inventory_position"] - (row["inventory"] + sum(row["pipeline"]))) < 1e-9,
        "inv_single: vec_ip derived feature evaluates correctly",
    )
    obs = interp.observe("vec_d", {**row, "demand": row["demand"]})
    check(
        obs["info.demand"] == row["demand"] and obs["pipeline"] == row["pipeline"],
        "inv_single: vec_d exposes info.demand + state refs",
    )

    # -- differential vs the handwritten inv_single domain ------------------------
    from mdp_ir.differential import make_inv_single_adapter, run_differential

    for inst in (None, "lost_sales"):
        rep = run_differential(
            inv, make_inv_single_adapter(inv, instance=inst),
            episode_seeds=[0, 1, 2], instance=inst,
        )
        check(
            rep.ok and rep.periods == 3 * inv.mdp.horizon_T(),
            f"differential[{inst or 'base'}]: interpreter == real inv_single, bit-exact",
        )

    # a corrupted dynamics update must diverge (the gate actually gates)
    import copy

    bad = copy.deepcopy(inv.model_dump(mode="json"))
    bad["mdp"]["dynamics"]["transitions"][1]["updates"] = [
        "received = pipeline[0]", "inventory += received",   # missing shift
    ]
    bad_ir = MdpIR.model_validate(bad)
    rep = run_differential(bad_ir, make_inv_single_adapter(bad_ir), episode_seeds=[0])
    check(not rep.ok, "differential: corrupted dynamics (missing pipeline shift) diverges")

    print(f"\nall {_checks} checks passed")


if __name__ == "__main__":
    main()

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

# file lives at <repo>/harness/mdp_ir/interpreter_test.py; fixtures are the
# shipped example domains under the plugin's skill directory
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "plugin" / "skills" / "mdp-solver"  # holds examples/

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
    # the adapter lives beside the IR (portable-domain contract), so it is
    # discovered from the IR path rather than imported from this package
    from mdp_ir.differential import load_adapter_factory, run_differential

    make_inv_single_adapter = load_adapter_factory(
        ROOT / "examples" / "inv_single" / "inv_single_schema.json", inv
    )

    for inst in (None, "lost_sales"):
        rep = run_differential(
            inv, make_inv_single_adapter(inv, instance=inst, seed_salt=1),
            episode_seeds=[0, 1, 2], instance=inst, seed_salt=1,
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
    rep = run_differential(bad_ir, make_inv_single_adapter(bad_ir, seed_salt=1),
                           episode_seeds=[0], seed_salt=1)
    check(not rep.ok, "differential: corrupted dynamics (missing pipeline shift) diverges")

    # =====================================================================
    # scenario redesign (scenario_redesign.md; spec §5, §6.3): world-layer
    # samplers/mixtures, design-layer grids, seed scheme v2
    # =====================================================================

    # inv_single is natively seed_scheme v2 with hidden world-latent samplers
    check(inv.seed_scheme == "v2" and len(inv.mdp.scenario.samplers) == 2
          and all(s.hidden for s in inv.mdp.scenario.samplers),
          "inv_single IR: v2 scheme, two hidden paper-demand samplers")

    # -- v2 forbids treatment A (episode-realization stages) -----------------
    ta = inv.model_dump(mode="json")
    next(s for s in ta["mdp"]["uncertainty_sources"] if s["name"] == "demand")[
        "stages"].append({"name": "support", "realization": "episode", "sub_stream": 0})
    try:
        MdpIR.model_validate(ta)
        check(False, "v2 + episode-realization stage must fail validation")
    except ValueError:
        check(True, "v2 scheme rejects episode-realization stages (treatment A)")

    # -- hidden latent wiring: derivation tracked, constants unobservable ----
    rm = inv.model_dump(mode="json")
    rm["rl"]["requires_memory"]["suggested"] = False
    try:
        MdpIR.model_validate(rm)
        check(False, "hidden sampler must flip the requires_memory derivation")
    except ValueError:
        check(True, "hidden sampler flips requires_memory derivation (validator catches)")
    d4 = {r["demand"] for r in simulate(inv, episode_seed=4, decisions={"order": 0.0}).rows}
    d5 = {r["demand"] for r in simulate(inv, episode_seed=5, decisions={"order": 0.0}).rows}
    check(len(d4) <= 5 and d4 != d5,
          "v2 sampler: <=5-value support per episode, varying across seeds")

    # -- v2 seed keys: period below source, branch word, leaf-first ----------
    v2_ir = inv
    src = next(s for s in v2_ir.mdp.uncertainty_sources if s.name == "demand")
    stage = src.stages[0]
    expected = ([f"sub:{stage.sub_stream}"] if stage.sub_stream is not None else []) + [
        "period", f"source:{src.stream_id}", "branch:1", "episode_seed", "seed_salt"]
    check(stage.seed_key(src.stream_id, False, scheme="v2") == expected,
          "v2 symbolic key: [sub?, period, source, branch:1, episode_seed, seed_salt]")
    from mdp_ir.interpreter import _meta_seed_key, _numeric_seed_key
    nk = _numeric_seed_key(stage, src.stream_id, period=7, episode_seed=9,
                           seed_salt=13, scheme="v2")
    check(nk[-5:] == [7, src.stream_id, 1, 9, 13],
          "v2 numeric key ends [period, source, 1, episode_seed, seed_salt]")
    check(_meta_seed_key("v2", 2, 9, 13) == [2, 0, 9, 13]
          and _meta_seed_key("v1", 0, 9, 13) == [0, 9, 13],
          "meta keys: v2 [sub, 0, e, salt]; v1 legacy [0, e, salt]")

    # -- duplicate meta substream ids fail -----------------------------------
    dup = inv.model_dump(mode="json")
    dup["mdp"]["scenario"]["samplers"].append(
        {"name": "second", "substream_id": 0,
         "draws": [{"name": "demand_probs", "distribution": {
             "family": "normalized_uniform_weights", "settings": {"size": 5}}}]})
    try:
        MdpIR.model_validate(dup)
        check(False, "duplicate substream_id must fail validation")
    except ValueError:
        check(True, "duplicate meta substream_id fails validation")

    # -- mixture: standalone equivalence over components ---------------------
    mx = inv.model_dump(mode="json")
    mx["mdp"]["scenario"]["mixtures"] = [{
        "name": "mix", "substream_id": 3,
        "components": [[0.5, ""], [0.5, "lost_sales"]],
    }]
    mx_ir = MdpIR.model_validate(mx)
    pure = {
        comp: {e: IrInterpreter(mx_ir, instance=comp or None)
               .run(e, decisions={"order": 0.0}).rows for e in range(10)}
        for comp in ("", "lost_sales")
    }
    hit = {"": 0, "lost_sales": 0}
    for e in range(10):
        rows = IrInterpreter(mx_ir, instance="mix").run(e, decisions={"order": 0.0}).rows
        matched = [c for c in hit if rows == pure[c][e]]
        assert len(matched) == 1, f"mixture episode {e} matches {matched}"
        hit[matched[0]] += 1
    check(all(v > 0 for v in hit.values()),
          f"mixture: every episode equals one component verbatim; both hit {hit}")

    # -- design-layer grid: canonical cells ----------------------------------
    axes_consts = [c.name for c in inv.mdp.scenario.constants
                   if isinstance(c.value, (int, float)) and not isinstance(c.value, bool)]
    g = inv.model_dump(mode="json")
    g["grids"] = [{"name": "sweep",
                   "axes": {axes_consts[0]: [1, 2], axes_consts[1]: [3, 4, 5]}}]
    g_ir = MdpIR.model_validate(g)
    cells = g_ir.grids[0].cells()
    check(len(cells) == 6
          and cells[0][0] == f"{axes_consts[0]}=1,{axes_consts[1]}=3"
          and cells[1][1] == {axes_consts[0]: 1, axes_consts[1]: 4},
          "grid: row-major cells, axis-derived ids")
    try:
        MdpIR.model_validate(
            {**g, "grids": [{"name": "bad", "axes": {"no_such": [1]}}]})
        check(False, "grid axis naming no constant must fail")
    except ValueError:
        check(True, "grid axis naming no constant fails validation")

    # -- domain-owned expression builtins (mdp.expr_builtins) ----------------
    # declared in the IR, implemented in a module next to it, lazily resolved
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        eb = inv.model_dump(mode="json")
        eb["mdp"]["expr_builtins"] = [
            {"name": "double_it", "module": "inv_single_test_builtins"}]
        comp = eb["mdp"]["objective"]["per_step_components"][0]
        comp_name = comp["name"]
        comp["expr"] = f"double_it({comp['expr']})"
        (tdir / "inv_single_schema.json").write_text(json.dumps(eb))
        (tdir / "inv_single_test_builtins.py").write_text(
            "def double_it(x):\n    return 2 * x\n")
        eb_ir = load_ir(tdir / "inv_single_schema.json")
        base_rows = simulate(inv, episode_seed=6, decisions={"order": 25.0}).rows
        eb_rows = simulate(eb_ir, episode_seed=6, decisions={"order": 25.0}).rows
        check(
            all(abs(e[comp_name] - 2 * b[comp_name]) < 1e-9
                for e, b in zip(eb_rows, base_rows)),
            "expr_builtins: declared builtin resolved next to the IR and applied",
        )

    undecl = inv.model_dump(mode="json")
    c0 = undecl["mdp"]["objective"]["per_step_components"][0]
    c0["expr"] = f"triple_it({c0['expr']})"
    try:
        MdpIR.model_validate(undecl)
        check(False, "undeclared function in an expression must fail validation")
    except ValueError:
        check(True, "undeclared function in an expression fails validation")

    shadow = inv.model_dump(mode="json")
    shadow["mdp"]["expr_builtins"] = [
        {"name": "min", "module": "some_domain_extras"}]
    try:
        MdpIR.model_validate(shadow)
        check(False, "expr_builtins shadowing a core builtin must fail")
    except ValueError:
        check(True, "expr_builtins shadowing a core builtin fails validation")

    print(f"\nall {_checks} checks passed")


if __name__ == "__main__":
    main()

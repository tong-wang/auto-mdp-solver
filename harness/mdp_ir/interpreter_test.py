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
    dp = load_ir(ROOT / "examples" / "dynamic_pricing" / "dynamic_pricing_schema.json")
    fnv = load_ir(ROOT / "examples" / "fnv" / "fnv_schema.json")

    # -- determinism ---------------------------------------------------------
    for ir, dec in ((inv, {"order": 40.0}), (dp, {"price": 1.0}), (fnv, {"order": 0.5})):
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

    # dynamic_pricing: arrivals at t depend on the CURRENT price only — histories
    # that differ before t but agree at t see identical period-t arrivals
    base = simulate(dp, episode_seed=11, decisions={"price": 1.0})
    switch = simulate(
        dp, episode_seed=11,
        decisions=lambda ns: {"price": 5.0 if ns["period"] < 5 else 1.0},
    )
    n = min(len(base.rows), len(switch.rows))
    check(
        [r["arrivals"] for r in base.rows[5:n]]
        == [r["arrivals"] for r in switch.rows[5:n]],
        "dynamic_pricing: arrivals independent of the past price path",
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

    v = simulate(dp, episode_seed=9, decisions={"price": 1.0})
    check(
        all(r["units_sold"] == min(r["arrivals"], r["inventory"] + r["units_sold"]) for r in v.rows)
        and all(r["lost_demand"] == r["arrivals"] - r["units_sold"] for r in v.rows),
        "dynamic_pricing: sales capped by stock; lost_demand consistent",
    )

    # -- reward modes ------------------------------------------------------------
    check(
        all(abs(r["reward"] + r["total"]) < 1e-9 for r in t.rows),
        "inv_single: reward == -cost.total (sense=minimize)",
    )
    check(
        all(abs(r["reward"] - r["total"]) < 1e-9 for r in v.rows),
        "dynamic_pricing: reward == revenue.total (sense=maximize)",
    )

    # -- termination ---------------------------------------------------------------
    check(
        len(t.rows) == inv.mdp.horizon_T() and not t.terminated_early,
        "inv_single: runs exactly T periods",
    )
    sellout = simulate(dp, episode_seed=1, decisions={"price": 0.0})
    check(
        sellout.terminated_early and sellout.rows[-1]["inventory"] == 0,
        "dynamic_pricing: price=0 sells out -> early termination at inventory==0",
    )
    check(
        all(r["inventory"] >= 0 for r in sellout.rows),
        "dynamic_pricing: inventory never negative (no backlog)",
    )

    # -- scenario instances -----------------------------------------------------
    ls = IrInterpreter(inv, instance="lost_sales").run(5, decisions={"order": 0.0})
    check(
        all(r["inventory"] >= 0 for r in ls.rows) and any(r["lost_sales"] > 0 for r in ls.rows),
        "inv_single[lost_sales]: no backlog, unmet demand recorded as lost_sales",
    )
    ample = IrInterpreter(dp, instance="ample_stock").run(9, decisions={"price": 1.0})
    check(
        sum(r["arrivals"] for r in ample.rows) < sum(r["arrivals"] for r in v.rows),
        "dynamic_pricing[ample_stock]: a=20 -> fewer arrivals than base a=100",
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

    # -- differential vs the handwritten fnv domain (grid exemplar) ---------
    # 1-indexed MMFE ordering: N=3 order periods; terminal-only stochastic
    # payoff realized under the `period == N` guard; baked signal-volatility
    # schedule keeps the interpreter's normal draw bit-exact with the domain.
    make_fnv_adapter = load_adapter_factory(
        ROOT / "examples" / "fnv" / "fnv_schema.json", fnv
    )
    rep = run_differential(
        fnv, make_fnv_adapter(fnv, seed_salt=1),
        episode_seeds=[0, 1, 2], seed_salt=1,
    )
    check(rep.ok and rep.periods == 3 * 3,
          "differential[fnv]: interpreter == real fnv, bit-exact (3 order periods)")

    # the MMFE information path is decision-path independent (§4.3): the signal
    # is keyed on (period, seed, salt), so different order paths see the same I
    lo = simulate(fnv, episode_seed=5, decisions={"order": 0.0}, seed_salt=1)
    hi = simulate(fnv, episode_seed=5, decisions={"order": 3.0}, seed_salt=1)
    check([r["information"] for r in lo.rows] == [r["information"] for r in hi.rows],
          "fnv: MMFE information path independent of the order decisions")

    # demand/sales are a terminal-only realization (0 until the horizon)
    run = simulate(fnv, episode_seed=4, decisions={"order": 1.0}, seed_salt=1)
    check(all(r["demand"] == 0 for r in run.rows[:-1]) and run.rows[-1]["demand"] != 0,
          "fnv: demand/sales realized only at the terminal period")

    # =====================================================================
    # scenario redesign (spec §5, §6.3): world-layer
    # samplers/mixtures, design-layer grids, seed scheme v2
    # =====================================================================

    # inv_single is natively seed_scheme v2; the demand slot's latent draws
    # desugar to ONE hidden sampler at the slot's stream id (catalog model:
    # instances share it — common random numbers across instances)
    check(inv.seed_scheme == "v2" and len(inv.mdp.scenario.samplers) == 1
          and inv.mdp.scenario.samplers[0].substream_id == 0
          and inv.mdp.scenario.samplers[0].hidden,
          "inv_single IR: v2 scheme, one hidden demand-latent sampler at the slot's stream id")

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
         "draws": [{"name": "demand_probabilities", "distribution": {
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
    # replacing the mixture list orphans the loader's resolutions for the
    # shipped mix_demand — drop them (loader-set data, mixture-list-coupled)
    mx["mixture_resolutions"] = None
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

    # -- generic numpy dispatch: any Generator scalar distribution by name ---
    import numpy as np

    from mdp_ir.interpreter import _NUMPY_SCALAR_DISTS

    interp0 = IrInterpreter(inv)
    for fam, kw in (
        ("gamma", {"shape": 9.0, "scale": 3.3333}),
        ("binomial", {"n": 20, "p": 0.3}),
        ("exponential", {"scale": 2.0}),
        ("negative_binomial", {"n": 5, "p": 0.4}),
    ):
        check(fam in _NUMPY_SCALAR_DISTS, f"{fam} in numpy scalar-dist allowlist")
        got = interp0._sample_family(np.random.default_rng(7), fam, dict(kw), {})
        want = getattr(np.random.default_rng(7), fam)(**kw)
        want = want.item() if hasattr(want, "item") else want
        check(got == want, f"generic dispatch reproduces numpy {fam} bit-for-bit")
    check(
        isinstance(
            interp0._sample_family(
                np.random.default_rng(1), "binomial", {"n": 5, "p": 0.5}, {}
            ),
            int,
        ),
        "generic dispatch: integer-valued distribution coerces to int",
    )
    # settings resolve as exprs over the namespace, same as explicit families
    check(
        interp0._sample_family(np.random.default_rng(2), "gamma",
                               {"shape": "3 + 6", "scale": 3.3333}, {})
        == np.random.default_rng(2).gamma(shape=9, scale=3.3333),
        "generic dispatch: settings evaluate as namespace exprs",
    )
    try:
        interp0._sample_family(np.random.default_rng(1), "no_such_dist", {}, {})
        check(False, "unknown family must raise")
    except NotImplementedError:
        check(True, "unknown distribution family still raises NotImplementedError")

    # =====================================================================
    # catalog ⊕ selection (IR_LAYERING_PLAN §10)
    # =====================================================================
    import copy as _copy

    from mdp_ir import layering

    inv_path = ROOT / "examples" / "inv_single" / "inv_single_schema.json"
    raw = json.loads(inv_path.read_text())
    check(layering.is_catalog(raw), "inv_single schema is a catalog document")
    check(inv.selection == {"demand": "discrete", "leadtime": "discrete"},
          "base load records the default selection")

    poi = load_ir(inv_path, instance="poisson")
    demand_src = next(s for s in poi.mdp.uncertainty_sources if s.name == "demand")
    check(poi.selection["demand"] == "poisson"
          and demand_src.distribution.family == "poisson",
          "instance 'poisson' selects the poisson candidate")
    check(inv.mdp.decision_bounds("order") == (0.0, 200.0)
          and poi.mdp.decision_bounds("order") == (0.0, 600.0),
          "symbolic bounds resolve per selection from derived demand.mean (10 vs 30)")
    sel = load_ir(inv_path, select={"demand": "poisson"})
    check(sel.mdp_fingerprint() == poi.mdp_fingerprint(),
          "--select demand=poisson resolves identically to instance 'poisson'")
    # "poisson" is selection-inconsistent under the base load but survives as
    # a mixture component (constants-only after stripping) — the interpreter
    # needs its overrides when mix_demand draws it; "poisson_lost_sales" is
    # no component, so it drops
    check(sorted(inv.mdp.scenario.instances) == ["lost_sales", "poisson"]
          and inv.mdp.scenario.instances["poisson"] == {}
          and sorted(poi.mdp.scenario.instances)
          == ["lost_sales", "poisson", "poisson_lost_sales"]
          and poi.mdp.scenario.instances["poisson"] == {},
          "instances inconsistent with the selection drop unless mixture "
          "components; slot keys strip from survivors")

    # every declared composition passes the differential (the covering set)
    for inst in ("poisson", "poisson_lost_sales"):
        ir_i = load_ir(inv_path, instance=inst)
        rep = run_differential(
            ir_i, make_inv_single_adapter(ir_i, instance=inst, seed_salt=1),
            episode_seeds=[0, 1, 2], instance=inst, seed_salt=1,
        )
        check(rep.ok, f"differential[{inst}]: selected composition matches, bit-exact")

    # structural fingerprint: values, new instances, new candidates are free;
    # a dynamics edit is not
    fp = layering.structural_fingerprint(raw)
    tweaked = _copy.deepcopy(raw)
    next(c for c in tweaked["mdp"]["scenario"]["constants"] if c["name"] == "h")["value"] = 9.9
    tweaked["mdp"]["scenario"]["instances"]["hot"] = {"b": 4.0}
    tweaked["mdp"]["uncertainty_slots"][0]["candidates"]["poisson_u"] = {
        "generator": "PoissonDemand", "family": "poisson",
        "settings": {"rate": {"draw": {"family": "uniform",
                                       "settings": {"low": 20.0, "high": 40.0}},
                              "example": 30.0}}}
    check(layering.structural_fingerprint(tweaked) == fp,
          "structural fingerprint: value edits / new instance / new candidate do not move it")
    dyn_edit = _copy.deepcopy(raw)
    dyn_edit["mdp"]["dynamics"]["transitions"][1]["updates"].append("received = received + 0")
    check(layering.structural_fingerprint(dyn_edit) != fp,
          "structural fingerprint: a dynamics edit moves it")

    # a freshly appended candidate resolves + validates with zero other edits
    novel = MdpIR.model_validate(
        layering.resolve_catalog(tweaked, select={"demand": "poisson_u"}))
    nsrc = next(s for s in novel.mdp.uncertainty_sources if s.name == "demand")
    check(novel.selection["demand"] == "poisson_u"
          and nsrc.distribution.family == "poisson"
          and novel.mdp.decision_bounds("order") == (0.0, 600.0)
          and novel.mdp.scenario.samplers[0].draws[0].name == "demand_rate",
          "appended candidate: resolves, derives bounds (uniform-latent mean 30), desugars its draw")

    # read-API is lazy: underivable attrs only error when referenced
    dp_raw = json.loads(
        (ROOT / "examples" / "dynamic_pricing" / "dynamic_pricing_schema.json").read_text())
    check(layering.is_catalog(dp_raw)
          and load_ir(ROOT / "examples" / "dynamic_pricing" / "dynamic_pricing_schema.json")
          .selection == {"demand": "poisson"},
          "state-dependent settings are legal while nothing references the slot's read-API")

    # =====================================================================
    # §10f increments (2026-07-28): generic family runtime, cross-family
    # mixtures, event_sequence as a constant
    # =====================================================================
    from mdp_ir import runtime

    import inv_single_uncertainty as unc  # dir on sys.path via adapter load

    class _Ctx:
        """Minimal SamplingContext for direct generator draws."""

        def __init__(self, period: int, episode_seed: int, seed_salt: int):
            self.period = period
            self.episode_seed = episode_seed
            self.seed_salt = seed_salt

    # -- generic runtime ≡ hand-written latent generators, bit-for-bit -------
    gd = runtime.FamilyGenerator.from_ir(inv, "demand")
    hd = unc.LatentDiscreteDemand(
        support_size=3, support_low=8, support_high=12, source_id=0)
    same = True
    for e in (0, 3, 11):
        rg, rh = gd.realize(e, 1), hd.realize(e, 1)
        same &= rg.settings["values"] == rh.vals.tolist()
        same &= all(abs(a - b) < 1e-12 for a, b in
                    zip(rg.settings["probabilities"], rh.probs.tolist()))
        same &= all(rg.sample(_Ctx(p, e, 1)) == rh.sample(_Ctx(p, e, 1))
                    for p in range(5))
    check(same, "FamilyGenerator(from_ir) ≡ LatentDiscreteDemand: latents + samples bit-exact")
    check(gd.min() == 8.0 and gd.max() == 12.0 and gd.mean() == 10.0,
          "derived min/mean/max envelope over the discrete latent (support bounds)")

    gp = runtime.FamilyGenerator.from_ir(poi, "demand", instance="poisson")
    hp = unc.LatentPoissonDemand(alpha=9.0, beta=0.3, source_id=0)
    same = True
    for e in (0, 4):
        rg, rh = gp.realize(e, 1), hp.realize(e, 1)
        same &= rg.settings["rate"] == rh.rate
        same &= all(rg.sample(_Ctx(p, e, 1)) == rh.sample(_Ctx(p, e, 1))
                    for p in range(4))
    check(same, "FamilyGenerator(from_ir) ≡ LatentPoissonDemand: rate + samples bit-exact")
    check(abs(gp.mean() - hp.mean()) < 1e-9 and abs(gp.max() - hp.max()) < 1e-9,
          "FamilyGenerator envelopes reproduce the hand-written mean/max formulas")

    # -- appended candidate: full differential with ZERO domain edits --------
    # poisson_u (uniform-rate latent) exists in no hand-written class; the
    # adapter falls back to the FamilyDemand bridge (the zero-code path)
    ir_u = MdpIR.model_validate(
        layering.resolve_catalog(tweaked, select={"demand": "poisson_u"}))
    rep = run_differential(ir_u, make_inv_single_adapter(ir_u, seed_salt=1),
                           episode_seeds=[0, 1, 2], seed_salt=1)
    check(rep.ok,
          "appended candidate (uniform-rate poisson): differential MATCH, zero domain code")

    # -- shipped cross-family mixture: resolution + envelope + differential --
    mix_ir = load_ir(inv_path, instance="mix_demand")
    mres = (mix_ir.mixture_resolutions or {}).get("mix_demand") or {}
    check(set(mres) == {"poisson"}
          and mres["poisson"].selection["demand"] == "poisson"
          and any(s.distribution.family == "poisson" for s in mres["poisson"].sources),
          "mixture load: divergent component carries its own resolution")
    check(mix_ir.mdp.decision_bounds("order") == (0.0, 400.0),
          "mixture bounds envelope: 20 * weighted demand mean (0.5*10 + 0.5*30)")
    rep = run_differential(
        mix_ir, make_inv_single_adapter(mix_ir, instance="mix_demand", seed_salt=1),
        episode_seeds=list(range(8)), instance="mix_demand", seed_salt=1)
    check(rep.ok, "differential[mix_demand]: cross-family mixture matches, bit-exact")

    # standalone equivalence across families: every mixture episode equals
    # the drawn component's standalone run verbatim, and both regimes occur
    pure_by_comp = {
        "": {e: IrInterpreter(inv).run(e, decisions={"order": 0.0}).rows
             for e in range(12)},
        "poisson": {e: IrInterpreter(poi, instance="poisson")
                    .run(e, decisions={"order": 0.0}).rows for e in range(12)},
    }
    hit = {"": 0, "poisson": 0}
    mix_interp = IrInterpreter(mix_ir, instance="mix_demand")
    for e in range(12):
        rows = mix_interp.run(e, decisions={"order": 0.0}).rows
        matched = [c for c in hit if rows == pure_by_comp[c][e]]
        assert len(matched) == 1, f"mixture episode {e} matches {matched}"
        hit[matched[0]] += 1
    check(all(v > 0 for v in hit.values()),
          f"cross-family mixture: every episode ≡ one component verbatim; both hit {hit}")

    # -- event_sequence as a constant (event-order variants are instances) ---
    ev = inv.model_dump(mode="json")
    ev["mdp"]["scenario"]["constants"].append(
        {"name": "evt_order", "value": ["O", "R", "D"], "axis": "variant",
         "desc": "event order — a structural parameter (§10d)"})
    ev["mdp"]["dynamics"]["event_sequence"] = "evt_order"
    ev["mdp"]["scenario"]["instances"]["rdo"] = {"evt_order": ["R", "D", "O"]}
    ev_ir = MdpIR.model_validate(ev)
    check(ev_ir.mdp.event_sequence() == ["O", "R", "D"]
          and ev_ir.mdp.event_sequence("rdo") == ["R", "D", "O"],
          "event_sequence names a constant; instance overrides the order")
    check(simulate(ev_ir, episode_seed=7, decisions={"order": 40.0}).rows
          == simulate(inv, episode_seed=7, decisions={"order": 40.0}).rows,
          "constant event order == literal order: base trajectories identical")
    rep = run_differential(ev_ir, make_inv_single_adapter(ev_ir, instance="rdo", seed_salt=1),
                           episode_seeds=[0, 1, 2], instance="rdo", seed_salt=1)
    check(rep.ok, "differential[rdo]: R-D-O event-order instance matches the domain, bit-exact")

    bad_seq = _copy.deepcopy(ev)
    bad_seq["mdp"]["dynamics"]["event_sequence"] = "no_such_const"
    try:
        MdpIR.model_validate(bad_seq)
        check(False, "event_sequence naming an unknown constant must fail")
    except ValueError:
        check(True, "event_sequence naming an unknown constant fails validation")
    bad_seq = _copy.deepcopy(ev)
    bad_seq["mdp"]["scenario"]["instances"]["rdo"] = {"evt_order": ["R", "D"]}
    try:
        MdpIR.model_validate(bad_seq)
        check(False, "an instance order missing a transition event must fail")
    except ValueError:
        check(True, "an instance order missing a transition event fails validation")
    bad_seq = inv.model_dump(mode="json")
    ts = bad_seq["mdp"]["dynamics"]["transitions"]
    ts[0], ts[1] = ts[1], ts[0]
    try:
        MdpIR.model_validate(bad_seq)
        check(False, "literal sequence with out-of-order transitions must fail")
    except ValueError:
        check(True, "literal event_sequence: out-of-order transition declaration fails")

    print(f"\nall {_checks} checks passed")


if __name__ == "__main__":
    main()

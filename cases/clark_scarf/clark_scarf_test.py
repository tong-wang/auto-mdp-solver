"""clark_scarf's own tests — the claims only this domain can make.

Run from anywhere (the portable-domain contract):

    pytest clark_scarf
    python clark_scarf_test.py

Everything true of *any* IR — determinism, decision-path independence,
termination, the declared invariants — is checked by ``mdp_ir.laws`` and only
invoked here. What lives in this file is clark_scarf-specific:

* the bit-exact differential over the structural covering set;
* negative controls proving the gates can actually fail;
* the **coordinate-change claim** underpinning the campaign's headline
  experiment — that `raw` and `echelon` observations are related by an
  invertible map, so the comparison tests representation and not information.
  No IR expression can state this: it is a property relating two *gym*
  observation modes, which the IR describes but does not execute;
* the **shipping-box claim** — no installation ever dispatches more than it
  holds, under adversarial actions.

The conservation laws live in the IR as ``mdp.invariants`` (retailer balance,
system conservation, non-negative upper levels), so every instance inherits
them. Two former laws are gone with the widths they policed:
`pipe_slots_beyond_leadtime_empty` guarded the unused tail of a fixed-width
register (F7 named the lead time at the width site instead), and
`no_flow_through_padding` guarded inert links (F9 named the chain length at
all three width sites, so there are none).
"""

from __future__ import annotations

import json
import re

import sys
from pathlib import Path

import numpy as np
import pytest

from mdp_ir.testing import (
    assert_diverges,
    assert_laws,
    assert_match,
    mutated,
    schema_beside,
)

SCHEMA = schema_beside(__file__)


def _raw_flat() -> dict:
    """The schema document with its `mdp` block FLATTENED.

    v0.9.2 lets the file present `mdp` in three headed groups — model (theory),
    design (which points this study evaluates), rendering (what the interpreter
    runs) — so a reader sees at a glance which section moves which fingerprint.
    This domain uses that layout. `ungroup_mdp` is the harness's own flattener
    and a flat block passes through unchanged, so every raw-JSON assertion in
    this file is written against the flat form and holds under either layout.
    (`load_ir` flattens internally, so the pydantic path never needed this.)
    """
    from mdp_ir.schema import ungroup_mdp

    doc = json.loads(SCHEMA.read_text())
    doc["mdp"] = ungroup_mdp(doc["mdp"])
    return doc


EPISODES = 8


def _declared_compositions() -> list[str | None]:
    """Base + every declared instance, read from the schema so the sweep cannot
    drift from the catalog it is meant to cover."""
    raw = _raw_flat()
    scenario = raw["mdp"]["scenario"]
    return [None] + sorted(scenario.get("instances") or {})


COMPOSITIONS = _declared_compositions()


# -- engine laws -------------------------------------------------------------


def test_engine_laws():
    assert_laws(SCHEMA)


# -- the differential, over the structural covering set ----------------------


@pytest.mark.parametrize(
    "instance", COMPOSITIONS, ids=[i or "base" for i in COMPOSITIONS]
)
def test_differential_matches_the_domain(instance):
    assert_match(SCHEMA, instance, episodes=EPISODES)


def test_the_covering_set_spans_every_structural_combination():
    """Chain length and lead time both change the shape of the problem, so the
    differential must exercise every (n_echelons, leadtime) pair — not just a
    convenient one. A covering set that quietly lost a combination would let a
    padding or pipeline bug through."""
    raw = _raw_flat()
    scenario = raw["mdp"]["scenario"]
    consts = {c["name"]: c["value"] for c in scenario["constants"]}
    seen = set()
    for over in [{}] + list(scenario["instances"].values()):
        n = over.get("n_echelons", consts["n_echelons"])
        lead = over.get("leadtime", consts["leadtime"])
        seen.add((int(n), int(lead)))
    expected = {(n, lead) for n in (2, 3, 4) for lead in (1, 2)}
    assert expected <= seen, f"covering set is missing {sorted(expected - seen)}"


# -- negative controls: the gates must be able to fail ------------------------


def test_a_corrupted_dynamics_update_diverges():
    """If the differential cannot fail, it proves nothing. Break the demand
    subtraction and the interpreter must stop agreeing with the domain."""

    def break_demand(doc):
        d_event = _event(doc, "D")
        d_event["updates"] = [
            u.replace("stock[0] -= demand", "stock[0] -= demand + 1")
            for u in d_event["updates"]
        ]

    assert_diverges(SCHEMA, mutated(_load(), break_demand), episodes=EPISODES)


def test_dropping_the_source_debit_diverges():
    """The load-bearing coupling of the whole problem is that a level cannot
    ship what it does not hold. Remove the debit — making stock appear from
    nowhere as it moves down the chain — and the differential must catch it."""

    def break_debit(doc):
        s_event = _event(doc, "S")
        # the debit is one quantified rule since F9; neutralise it by making
        # every link draw from nowhere
        s_event["updates"] = [
            u.replace("(shipped[k - 1] if k >= 1 else 0)", "0")
            for u in s_event["updates"]
        ]

    assert_diverges(SCHEMA, mutated(_load(), break_debit), episodes=EPISODES)


def _event(doc: dict, event: str) -> dict:
    """A transition from the raw IR document. ``mutated`` edits the JSON doc,
    not the pydantic model, so edits address dict keys."""
    return next(t for t in doc["mdp"]["dynamics"]["transitions"] if t["event"] == event)


def _load():
    from mdp_ir.schema import load_ir

    return load_ir(SCHEMA)


# -- claims no IR expression can state ---------------------------------------
#
# These relate two GYM observation modes, or quantify over adversarial actions.
# The IR declares the modes but does not execute them, so only a test can hold
# these down.


@pytest.fixture(scope="module")
def domain():
    import importlib
    import sys
    from pathlib import Path

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    return (
        importlib.import_module("clark_scarf_scenarios"),
        importlib.import_module("clark_scarf_mdp"),
        importlib.import_module("clark_scarf_gym"),
    )


@pytest.mark.parametrize(
    "name", ["n2_l1_p09", "n2_l2_p09", "n3_l1_p09", "n3_l2_p09", "n4_l1_p09", "n4_l2_p09"]
)
def test_raw_and_echelon_are_a_pure_change_of_coordinates(domain, name):
    """THE claim the headline experiment rests on.

    If `echelon` carried information `raw` lacks, a raw-vs-echelon gap would
    measure information rather than representation and the campaign's question
    would be ill-posed. Differencing the echelon view must recover the raw view
    exactly, at every step of a full episode, under random actions.

    This is why `echelon` sums stock and in-flight stock SEPARATELY: one
    running sum over their total would be genuinely lossy at leadtime 2, and
    this test is what would catch that regression.
    """
    scen, _, gymmod = domain
    sc = scen.SCENARIOS[name]
    n = sc.n_echelons
    e_raw = gymmod.ClarkScarfEnv(sc, observation_mode="raw")
    e_ech = gymmod.ClarkScarfEnv(sc, observation_mode="echelon")
    o_raw, _ = e_raw.reset(seed=11)
    o_ech, _ = e_ech.reset(seed=11)

    assert o_raw.shape == o_ech.shape, (
        "the two modes must have identical width — a dimension difference is "
        "already an information difference"
    )

    for step in range(sc.horizon):
        blocks_r = o_raw[1:].reshape(-1, n)
        blocks_e = o_ech[1:].reshape(-1, n)
        recovered = np.asarray(gymmod.from_echelon([list(b) for b in blocks_e]))
        # tolerance scales with the running sum: differencing a float32 cumsum
        # loses absolute precision proportional to the largest partial sum
        # (~1e-7 relative). A real invertibility failure would be an error of
        # order the values themselves.
        atol = 1e-4 * max(1.0, float(np.abs(blocks_e).max()))
        assert np.allclose(recovered, blocks_r, atol=atol), (
            f"{name} step {step}: echelon view does not inverse to raw\n"
            f"  raw={blocks_r}\n  recovered={recovered}"
        )
        a = e_raw.action_space.sample()
        o_raw, _, t1, _, _ = e_raw.step(a)
        o_ech, _, t2, _, _ = e_ech.step(a)
        if t1 or t2:
            break


@pytest.mark.parametrize("name", ["n2_l2_p09", "n3_l2_p09", "n4_l2_p09"])
def test_no_installation_ever_ships_more_than_it_holds(domain, name):
    """Adversarial actions: demand the maximum on every link every period.

    The IR's `upper_levels_non_negative` invariant states the consequence; this
    states the mechanism — that the clip is a per-component box read from the
    PRE-dispatch state, so two links cannot both claim the same units, and a
    link cannot forward stock that another link delivers the same period.
    """
    scen, mdp, _ = domain
    sc = scen.SCENARIOS[name]
    state, _ = mdp.init_state(sc, episode_seed=5)
    while not state.terminated:
        state1 = mdp.advance1(sc, state)
        pre = list(state1.stock)
        caps = mdp.ship_capacity(sc, state1)
        ship = [min(sc.ship_max, caps[k]) for k in range(sc.n_echelons)]  # ask for everything
        state, info = mdp.advance2(sc, state1, ship)
        for k in range(1, sc.n_echelons):
            assert info["shipped"][k - 1] <= pre[k] + 1e-9, (
                f"{name}: link {k} shipped {info['shipped'][k - 1]} but its "
                f"source held only {pre[k]}"
            )
        for k in range(1, sc.n_echelons):
            assert state.stock[k] >= -1e-9, (
                f"{name}: installation {k + 1} driven negative to {state.stock[k]}"
            )


def test_no_width_outruns_its_named_constant(domain):
    """F9 retired the padding this file used to police.

    `test_padding_is_economically_invisible` guarded inert levels: four slots
    always rendered, `n_echelons` live, the rest required to move nothing and
    cost nothing. There are no inert levels now — `stock`, `pipe` and the
    `ship` decision each NAME `n_echelons` at their width site, so the
    rendering is exactly as wide as the instance and padding cannot exist to
    be invisible.

    What replaces it is the claim that made the padding unnecessary: every
    width is the constant, on every declared instance.
    """
    scen, mdp, _ = domain
    doc = _schema_doc()
    widths = {v["name"]: v.get("length") for v in doc["mdp"]["state_variables"]}
    assert widths["stock"] == "n_echelons", widths
    assert widths["pipe"] == ["n_echelons", "leadtime"], widths
    assert doc["mdp"]["decisions"][0]["dim"] == "n_echelons", doc["mdp"]["decisions"][0]

    for name in COMPOSITIONS:
        if name is None:
            continue
        sc = scen.SCENARIOS[name]
        state, _ = mdp.init_state(sc, episode_seed=3)
        assert len(state.stock) == sc.n_echelons, name
        assert len(state.pipeline) == sc.n_echelons, name
        assert all(len(row) == sc.leadtime for row in state.pipeline), name


def test_clark_scarf_decomposition_is_exactly_optimal():
    """The second-implementation gate behind the DP row (spec §1.2).

    The shipped benchmark solves the chain as N one-dimensional problems on the
    strength of Theorems 1–2. This checks that claim against a **brute-force DP
    over the full joint state**, on an instance small enough to enumerate.

    It is not ceremony. The first draft of the DP charged each decision against
    demand over the L transit periods; cost is actually assessed on the
    END-OF-PERIOD state, one draw further on (the periodic-review "lead time
    plus one"). That set every target far too low and left a **30% gap** to the
    true optimum — while still producing a plausible-looking, monotone,
    time-varying table of critical numbers, and while every other gate in the
    pipeline stayed green. Only this comparison caught it.
    """
    from clark_scarf_dp_exactness import compare

    r = compare("verify_tiny")
    assert abs(r["gap"]) < 1e-6, (
        f"Clark-Scarf decomposition is not optimal on {r['scenario']}: "
        f"{r['clark_scarf']:.6f} vs joint optimum {r['joint_optimal']:.6f} "
        f"(gap {r['gap']:+.6f}, {r['gap_pct']:+.4f}%)"
    )


def test_clark_scarf_decomposition_is_exactly_optimal_undiscounted():
    """The same gate on the beta = 1 twin (F12).

    The undiscounted frames are worth having only if their bar is still a
    verified optimum, and "backward induction takes beta = 1 as the ordinary
    case" is an argument, not a check. Both recursions here read
    `scenario.beta`, so the decomposition and the joint brute force are solved
    at the same discount — which is the property that would break silently if
    either one ever went back to a literal.
    """
    from clark_scarf_dp_exactness import compare

    r = compare("verify_tiny_b1")
    assert abs(r["gap"]) < 1e-6, (
        f"the decomposition is not optimal undiscounted on {r['scenario']}: "
        f"{r['clark_scarf']:.6f} vs joint optimum {r['joint_optimal']:.6f} "
        f"(gap {r['gap']:+.6f}, {r['gap_pct']:+.4f}%)"
    )


def test_beta_twins_differ_only_in_the_discount():
    """A twin is the same rendering at another discount — nothing else moved.

    The whole claim of F12 is that naming the constant makes a second frame
    cost one field. If a twin ever drifts on structure, cost or demand, the
    frames stop being comparable *as renderings of one model* and the entry
    that opened them is false.
    """
    from dataclasses import fields
    from clark_scarf_scenarios import SCENARIOS

    for twin, src in (("n3_l2_p09_b1", "n3_l2_p09"),
                      ("verify_tiny_b1", "verify_tiny"),
                      ("g1_n3_l2_p09_b1", "g1_n3_l2_p09")):
        a, b = SCENARIOS[twin], SCENARIOS[src]
        assert a.beta == 1.0 and b.beta == 0.95
        for f in fields(a):
            if f.name in ("scenario_name", "beta"):
                continue
            assert getattr(a, f.name) == getattr(b, f.name), (
                f"{twin} differs from {src} on {f.name!r}: a twin varies the "
                f"discount and nothing else"
            )


@pytest.mark.parametrize("name", ["n3_l2_p09", "n2_l1_p09"])
def test_unseeded_resets_draw_fresh_episodes(domain, name):
    """Training-loop contract: gymnasium's auto-reset passes NO seed, so an
    unseeded reset MUST draw a fresh episode from the per-env stream.

    This is the regression test for ESCALATION #E17: the first build only set
    `_episode_seed` when a seed was passed, so every training env replayed one
    fixed demand path forever — the entire Stage-4 campaign trained on n_envs
    fixed demand sequences while every eval (which seeds explicitly) scored the
    artifacts on fresh ones. No existing gate looks at this: conformance checks
    the MDP layer's seed keys and the gym's step contract, the laws check the
    interpreter — none of them run two unseeded resets back to back.

    Seeded resets must stay deterministic (CRN protocol depends on it).
    """
    scen, _, gymmod = domain
    sc = scen.SCENARIOS[name]
    env = gymmod.ClarkScarfEnv(sc, observation_mode="raw")
    env.reset(seed=7)

    def demand_path():
        env.reset()                      # unseeded, like the training loop
        out = []
        while True:
            _, _, term, _, info = env.step(env.action_space.sample())
            out.append(info["demand"])
            if term:
                return out

    paths = [demand_path() for _ in range(3)]
    assert paths[0] != paths[1] and paths[1] != paths[2], (
        "unseeded resets replayed an identical demand path — training would "
        "silently run on a fixed episode (#E17)"
    )

    env.reset(seed=11)
    a = [env.step(env.action_space.sample())[4]["demand"] for _ in range(5)]
    env.reset(seed=11)
    env.action_space.seed(0)
    # deterministic demands under a fixed seed (actions differ; demand is
    # decision-path independent, so the comparison is valid)
    b = [env.step(env.action_space.sample())[4]["demand"] for _ in range(5)]
    assert a == b, "seeded resets must stay deterministic (CRN protocol)"


def test_integer_demand_keeps_every_stock_integral(domain):
    """Poisson demand plus integer shipments must leave every stock integral.

    This is what lets the DP reference enumerate exactly instead of on a grid;
    if it ever broke, the DP row would silently become an approximation.
    """
    scen, mdp, _ = domain
    sc = scen.SCENARIOS["n3_l2_p09"]
    state, _ = mdp.init_state(sc, episode_seed=9)
    rng = np.random.default_rng(0)
    while not state.terminated:
        state1 = mdp.advance1(sc, state)
        caps = mdp.ship_capacity(sc, state1)
        ship = [float(int(rng.integers(0, 25))) for _ in range(sc.n_echelons)]
        ship = [min(ship[k], caps[k]) for k in range(sc.n_echelons)]
        state, _ = mdp.advance2(sc, state1, ship)
        for vec_name, vec in [("stock", state.stock)] + [
            (f"pipe[{s}]", row) for s, row in enumerate(state.pipeline)
        ]:
            for x in vec:
                assert float(x).is_integer(), (
                    f"{vec_name} went non-integral: {vec}"
                )


def test_pipeline_impulse_lands_after_exactly_leadtime_arrivals(domain):
    """A dispatched impulse lands after exactly `leadtime` arrival events, at
    every lead time — including values no instance sweeps.

    This is the gate behind the model-domain claim: the model admits any
    integer lead time >= 1 and the rendering now carries exactly that many
    slots (`length: "leadtime"`), so there is no cap to outgrow. Sweeping past
    the designed {1, 2, 3} is what shows the recursion is general rather than
    fitted to the instances.
    """
    scen, mdp, _ = domain
    for lead in range(1, 7):
        sc = scen.ClarkScarfScenario(
            scenario_name=f"impulse_l{lead}",
            horizon=lead + 3,
            n_echelons=2,
            leadtime=lead,
            demand=scen.PoissonDemand(0.0001),   # demand ~ 0: isolate the flow
            h_install=(1.5, 0.5, 0.0, 0.0),
            p_short=9.0,
        )
        impulse = 7.0
        state, _ = mdp.init_state(sc, episode_seed=1)
        state.pipeline = [[0.0] * lead for _ in range(sc.n_echelons)]
        state.stock = [50.0, 50.0]
        assert all(len(row) == lead for row in state.pipeline), (
            f"L={lead}: rendering carries {[len(r) for r in state.pipeline]} slots, "
            f"expected exactly the lead time"
        )
        landed_at = None
        for t in range(sc.horizon):
            state1 = mdp.advance1(sc, state)
            if state1.arrived[1] > 0 and landed_at is None:
                landed_at = t
            ship = [0.0] * sc.n_echelons
            if t == 0:
                ship[1] = impulse
            state, _ = mdp.advance2(sc, state1, ship)
        assert landed_at == lead, (
            f"L={lead}: impulse dispatched at t=0 landed at the arrival event of "
            f"t={landed_at}, expected t={lead}"
        )


# -- the model / design / implementation boundary -----------------------------
#
# The model layer now lives in the schema as `mdp.model` (v0.9.0, upstream #29),
# and `mdp_conformance`'s `model.boundary` check covers four of the five
# relationships: the statement is free of rendered names, a literal at a width
# site fails, a width constant covers every designed value, and every instance
# sits inside its declared domain.
#
# The residual these tests were written for is now CLOSED. It was a cap
# rendered as a COUNT OF VARIABLES (`pipe1..pipe4`) rather than at a width
# site: `_axis_tiers` derives widths from references, so a cap nothing
# references is invisible and the pre-F3 two-slot pipeline would have passed.
# F7 named the lead time at the width site and F9 named the chain length at
# all three, so every width this domain has is now derivable — `model.boundary`
# reports `leadtime` and `n_echelons` among its widths.
#
# The tests stay as the regression that keeps it that way: a literal creeping
# back into any width site is exactly the F3 defect returning, and the check
# can only see it while the constant is named.


def _schema_doc() -> dict:
    return _raw_flat()


def test_every_width_names_a_constant_and_none_is_a_literal(domain):
    """Both of this domain's structural dimensions are NAMED WIDTHS.

    `pipe` declares `["n_echelons", "leadtime"]` and the `ship` decision
    declares `dim: "n_echelons"`, so `model.boundary` derives both from their
    references and an instance renders exactly the shape it selects. A literal
    at any of these sites is the sweep-frozen-as-capacity defect (F3/F7/F9)
    returning, and it is invisible to the check the moment the name goes.
    """
    doc = _schema_doc()
    svs = {v["name"]: v for v in doc["mdp"]["state_variables"]}

    assert "pipe" in svs, f"expected one pipeline matrix, got {sorted(svs)}"
    assert not [n for n in svs if re.fullmatch(r"pipe_?\d+", n)], (
        "per-link pipeline variables are back; the shape is one `pipe` matrix"
    )
    assert svs["pipe"]["length"] == ["n_echelons", "leadtime"], svs["pipe"]["length"]
    assert svs["stock"]["length"] == "n_echelons", svs["stock"]["length"]

    decisions = doc["mdp"]["decisions"]
    assert len(decisions) == 1, [d["name"] for d in decisions]
    assert decisions[0]["dim"] == "n_echelons", decisions[0]["dim"]

    for site in (svs["pipe"]["length"], [svs["stock"]["length"]], [decisions[0]["dim"]]):
        for entry in site:
            assert isinstance(entry, str), (
                f"literal {entry!r} at a width site: the cap becomes a capacity "
                f"the model appears to state, and model.boundary cannot see it"
            )


def test_no_cap_constant_survives_in_schema_or_code(domain):
    """`model.boundary` checks widths against the DESIGN; nothing checks them
    against the CODE. This domain used to reconcile `n_levels_max` with
    `N_LEVELS_MAX` here. Both are gone (F9), and the claim worth holding is
    that they stay gone: a reintroduced cap is a capacity nobody declared.
    """
    scen, _, _ = domain
    consts = {c["name"]: c["value"] for c in _schema_doc()["mdp"]["scenario"]["constants"]}
    assert "n_levels_max" not in consts, (
        "a chain-length cap is back in the schema; widths NAME n_echelons now"
    )
    assert not hasattr(scen, "N_LEVELS_MAX"), (
        "a chain-length cap is back in the code; vectors are n_echelons long now"
    )


def test_every_narrowed_element_names_its_layer(domain):
    """`narrowed.by` must name selection / design / implementation. Upstream
    stores the field; completeness across the elements that need one is a
    domain claim — every core state vector and every decision here is narrowed
    to an integer lattice, and an uncited narrowing is how a design choice
    launders itself into the model."""
    doc = _schema_doc()
    layers = ("selection:", "design:", "implementation:")
    core = [v for v in doc["mdp"]["state_variables"] if v["role"] == "core"]
    for el in core + doc["mdp"]["decisions"]:
        n = el.get("narrowed")
        assert n, f"{el['name']} renders an integer lattice with no narrowing citation"
        assert any(b.strip().startswith(layers) for b in n["by"].split("+")), (
            f"{el['name']}: narrowing does not name its layer: {n['by']!r}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# -- `ship_fraction`: one decision, rounded iff the RENDERING is discrete ----


@pytest.mark.parametrize("name,discrete", [
    ("n3_l2_p09", True), ("n2_l1_p09", True), ("verify_tiny", True),
    ("g1_n3_l2_p09", False), ("g1_n2_l1_p09", False), ("g1_verify_tiny", False),
])
def test_ship_fraction_rounds_iff_demand_is_discrete(domain, name, discrete):
    """THE property of the unified mode.

    The decision is a fraction either way; whether the quantity it resolves to
    must land on an integer is a property of the RENDERING, and the IR says so
    — `narrowed` on `ship` reads "integer feasibility under the `poisson`
    selection ONLY". A second action mode for the discrete case would put a
    rendering choice into the action space.
    """
    scen, mdp, gym_mod = domain
    sc = scen.SCENARIOS[name]
    assert sc.demand.is_discrete is discrete, f"{name}: fixture assumption wrong"
    env = gym_mod.ClarkScarfEnv(sc, "raw", "ship_fraction")
    env.reset(seed=11)
    rng = np.random.default_rng(5)
    saw_fraction = False
    for _ in range(150):
        act = rng.uniform(0.0, 1.0, size=env.n_live)
        q = env._to_quantities(act)
        if discrete:
            assert np.allclose(q, np.rint(q)), f"{name}: non-integral shipment {q}"
        else:
            saw_fraction |= bool(np.any(np.abs(q - np.rint(q)) > 1e-9))
        obs, _, term, _, _ = env.step(act)
        if discrete:
            for k, v in enumerate(env._state.stock):
                assert abs(v - round(v)) < 1e-9, f"{name}: stock[{k}] = {v} left the lattice"
        if term:
            env.reset()
    if not discrete:
        assert saw_fraction, f"{name}: continuous rendering never produced a fractional shipment"


@pytest.mark.parametrize("name", ["n3_l2_p09", "g1_n3_l2_p09"])
def test_ship_fraction_extremes_are_meaningful(domain, name):
    """The action IS the fraction: 0 ships nothing, 0.5 half, 1 everything available.

    No affine remap, so an action reads the same everywhere it is printed — in
    a trajectory, a figure, or a §14 readback. "Ship everything available"
    sitting at a = 1 in every state is LV1 applied to the availability
    constraint.
    """
    scen, mdp, gym_mod = domain
    sc = scen.SCENARIOS[name]
    env = gym_mod.ClarkScarfEnv(sc, "raw", "ship_fraction")
    env.reset(seed=7)
    rng = np.random.default_rng(3)
    for _ in range(80):
        caps = np.asarray(mdp.ship_capacity(sc, env._state), dtype=float)
        want_top = np.floor(caps + 0.5) if sc.demand.is_discrete else caps
        assert np.allclose(env._to_quantities(np.full(env.n_live, 1.0)), want_top), (
            f"{name}: a=1 must ship the whole declared capacity"
        )
        # the fraction reads directly: 0 ships nothing, 0.5 ships half
        assert np.all(env._to_quantities(np.zeros(env.n_live)) == 0.0)
        half = env._to_quantities(np.full(env.n_live, 0.5))
        want_half = 0.5 * caps
        if sc.demand.is_discrete:
            want_half = np.floor(want_half + 0.5)
        assert np.allclose(half, want_half), (
            f"{name}: a=0.5 shipped {half}, expected {want_half} — the action "
            f"must BE the fraction, with no affine remap"
        )
        _, _, term, _, _ = env.step(rng.uniform(0, 1, size=env.n_live))
        if term:
            env.reset()


@pytest.mark.parametrize("name", ["n3_l2_p09", "g1_n3_l2_p09"])
def test_ship_fraction_never_needs_the_clip(domain, name):
    """`feasibility_strategy: clip` is declared but must never fire.

    The fraction is taken OF the cap, so no action can exceed it — which is
    LV3's point that re-parametrizing relative to the binding constraint makes
    a mask (and the clip) vacuous.
    """
    scen, mdp, gym_mod = domain
    sc = scen.SCENARIOS[name]
    env = gym_mod.ClarkScarfEnv(sc, "raw", "ship_fraction")
    env.reset(seed=31)
    rng = np.random.default_rng(13)
    for _ in range(150):
        act = rng.uniform(0.0, 1.0, size=env.n_live)
        caps = np.asarray(mdp.ship_capacity(sc, env._state), dtype=float)
        q = env._to_quantities(act)
        assert np.all(q <= caps + 1e-9), f"{name}: asked {q} against caps {caps}"
        assert np.all(q >= -1e-12), f"{name}: negative shipment {q}"
        _, _, term, _, _ = env.step(act)
        if term:
            env.reset()


def test_ship_fraction_is_injective_on_a_continuous_rendering(domain):
    """No rounding means no duplicated actions — the lattice's cost, isolated."""
    scen, _, gym_mod = domain
    env = gym_mod.ClarkScarfEnv(scen.SCENARIOS["g1_n3_l2_p09"], "raw", "ship_fraction")
    env.reset(seed=23)
    caps = env._effective_caps()
    a = np.linspace(0.0, 1.0, 41)
    for k in range(env.n_live):
        if caps[k] <= 0:
            continue
        q = np.array([env._to_quantities(np.full(env.n_live, x))[k] for x in a])
        assert len(np.unique(q)) == len(q), (
            f"link {k}: {len(a)} actions -> {len(np.unique(q))} quantities"
        )


# -- custom action heads must persist their own hyperparameters --------------


def test_custom_head_params_survive_save_and_load(domain, tmp_path):
    """A head's hyperparameters must live INSIDE the model, not beside it.

    They were a class attribute (`BetaDistribution.MIN_CONCENTRATION`) and a
    module global (the Gamma scale hint), both set by the train script. Neither
    travels with a saved model, so `PPO.load` rebuilt the head with whatever
    default happened to be in force and then scored a policy that had never
    been trained. It was not subtle once measured -- a Gamma arm whose
    selection score was ~1004 evaluated at ~13000, because its scale hint
    reverted from 10.0 to 1.0 and the policy shipped a tenth of what it had
    learned to ship -- but nothing failed, and it would have gone on producing
    plausible, wrong leaderboard rows.

    The globals are SABOTAGED here before reloading: if a value is being read
    from them rather than from the checkpoint, this test fails.
    """
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    scen, _, gym_mod = domain
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import clark_scarf_beta_policy as bp

    sc = scen.SCENARIOS["g1_n3_l2_p09"]
    common = dict(n_steps=64, batch_size=32, verbose=0, device="cpu")

    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "raw", "ship_fraction")])
    m = PPO(bp.BetaActorCriticPolicy, env,
            policy_kwargs={"net_arch": [8, 8], "min_concentration": 0.25}, **common)
    m.save(str(tmp_path / "beta"))
    keep = bp.BetaDistribution.MIN_CONCENTRATION
    try:
        bp.BetaDistribution.MIN_CONCENTRATION = 99.0
        got = PPO.load(str(tmp_path / "beta"), device="cpu").policy.action_dist.min_concentration
    finally:
        bp.BetaDistribution.MIN_CONCENTRATION = keep
    assert got == pytest.approx(0.25), (
        f"beta min_concentration came back as {got}: it is being read from the "
        f"class attribute, not from the checkpoint"
    )

    env2 = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "raw", "ship_absolute")])
    g = PPO(bp.GammaActorCriticPolicy, env2,
            policy_kwargs={"net_arch": [8, 8], "scale_hint": 10.0}, **common)
    g.save(str(tmp_path / "gamma"))
    keep2 = bp._GAMMA_SCALE_HINT[0]
    try:
        bp._GAMMA_SCALE_HINT[0] = 1.0
        got2 = PPO.load(str(tmp_path / "gamma"), device="cpu").policy.action_dist.scale_hint
    finally:
        bp._GAMMA_SCALE_HINT[0] = keep2
    assert got2 == pytest.approx(10.0), (
        f"gamma scale_hint came back as {got2}: it is being read from the "
        f"module global, not from the checkpoint"
    )


# --------------------------------------------------------------------------
# §CONFIG-REGISTRY (guide §13) — the two halves are checked against each other
# --------------------------------------------------------------------------
def _configs():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import clark_scarf_configs as cfg
    return cfg


def _registry_table_ids() -> dict[str, str | None]:
    """`{id: parent}` as ESCALATION.md §CONFIG-REGISTRY renders them."""
    text = (Path(__file__).resolve().parent / "ESCALATION.md").read_text()
    start = text.index("## CONFIG-REGISTRY")
    section = text[start:text.index("## LEDGER", start)]
    ids: dict[str, str | None] = {}
    for line in section.splitlines():
        if not line.startswith("| <a id="):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        cid = cells[0].split("`")[1]
        parent = None
        if len(cells) > 1 and "`" in cells[1]:
            parent = cells[1].split("`")[1]
        ids[cid] = parent
    return ids


def test_config_registry_halves_agree():
    """Every id is in both halves, with the same parent (guide §13.5).

    The redundancy is deliberate — a launch has to *resolve* an id and parsing
    markdown to do it would be the fragile thing §13 exists to avoid — and it
    is only safe because it is checked. The module is authoritative for what a
    config is; the table for why it exists.
    """
    cfg = _configs()
    table = _registry_table_ids()
    module = {cid: parent for axis in cfg.AXES.values() for cid, (parent, _) in axis.items()}
    module.update({cid: None for cid in cfg.SC})
    assert set(module) == set(table), (
        f"module-only ids {sorted(set(module) - set(table))}, "
        f"table-only ids {sorted(set(table) - set(module))}"
    )
    for cid, parent in module.items():
        if parent is not None:
            assert table[cid] == parent, (
                f"{cid}: module parents {parent}, the table says {table[cid]}"
            )


def test_config_origins_are_read_from_the_derivation():
    """`g0`/`a0`/`h0` are the train script's own output, never re-stated.

    An origin written by hand is one that can disagree with the derivation it
    names — the drift §13.2 forbids by making the origin *read*.
    """
    cfg = _configs()
    derived = cfg.l1_derived()
    h0 = cfg.origin("h")
    for knob, value in derived.items():
        if knob in cfg.H_OWNS:
            assert h0[knob] == value, f"h0[{knob}] = {h0[knob]!r}, derived {value!r}"
    for axis in "gah":
        assert cfg.AXES[axis][f"{axis}0"] == (None, {}), (
            f"{axis}0 must be the computed origin, not a written delta"
        )
    # every knob the CLI exposes lands in exactly one axis, or is excluded
    owned = set().union(*cfg.OWNS.values())
    for axis_a in "gah":
        for axis_b in "gah":
            if axis_a < axis_b:
                assert not (cfg.OWNS[axis_a] & cfg.OWNS[axis_b])
    assert not (owned & (cfg.EXCLUDED | cfg.PROTOCOL))


def test_config_level_is_read_off_the_knobs_that_moved():
    """A design-axis value names a cell, not a level (guide §13.4)."""
    cfg = _configs()
    assert cfg.level("sc0/g0/a0/h0") == "L1"
    assert cfg.level("sc0/g2/a0/h0") == "L1", (
        "a different action mode is a different CELL whose ladder restarts at "
        "L1 — reading it as an escalation is the mistake that relabelled half "
        "of a declared research instrument upstream"
    )
    assert cfg.level("sc0/g3/a0/h0") == "L1", "the echelon cell has its own ladder"
    assert cfg.level("sc0/g2/a1/h0") == "L2(arch)", (
        "the ordinal head is a policy CLASS, so citing it IS an escalation — "
        "unlike an action mode, which names a cell"
    )


def test_config_constraints_refuse_an_impossible_citation():
    """A forced knob is declared on the constraining id, never resolved silently."""
    cfg = _configs()
    with pytest.raises(cfg.ConfigError):
        cfg.resolve("sc0/g0/a1/h0")      # ordinal head against a Box encoding
    with pytest.raises(cfg.ConfigError):
        cfg.resolve("sc0/g9/a0/h0")      # unknown id
    assert cfg.resolve("sc0/g2/a1/h0")["policy_dist"] == "ordinal"


def test_registered_design_axis_values_are_declared_in_the_ir():
    """No id may cite a mode the IR does not declare (guide §3.1, #65).

    And §7 makes it bite in the other direction too: the gym may not implement
    a mode the IR does not declare. Two did on the previous board
    (`ship_fraction_bins`, `ship_scaled`) and were reported for a whole
    campaign without closing; this board removes them, so the set is empty and
    stays that way.
    """
    cfg = _configs()
    declared = cfg.ir_modes()
    for cid, (_, delta) in cfg.G.items():
        for knob, value in delta.items():
            if knob in cfg.DESIGN_AXES:
                assert value in declared[knob], f"{cid} cites undeclared {knob}={value}"
    for axis, extra in cfg.gym_only_modes().items():
        assert not extra, (
            f"the gym implements {sorted(extra)} on {axis}, which the IR "
            f"declares nowhere — §7: what the gym renders, the IR declares. "
            f"Declare it (the gym block moves no fingerprint) or remove it."
        )


# -- what the gym renders, the IR declares (spec §7, upstream #71) ------------


def _catalog_instances() -> list[str]:
    """The instances `load_ir` keeps — the catalog resolves the `demand` slot,
    so the poisson-pinned siblings drop out and only the `g1_*` set survives.
    Both renderings share one gym, so checking the survivors checks both."""
    from mdp_ir import load_ir

    return sorted(load_ir(str(SCHEMA)).mdp.scenario.instances or {})


def _declared_obs_modes() -> list[str]:
    """Read the menu from the IR, so a mode added there is covered here without
    anyone remembering to widen a list."""
    from mdp_ir import load_ir

    return [m.name for m in load_ir(str(SCHEMA)).gym.observation_modes]


@pytest.mark.parametrize("instance", _catalog_instances())
@pytest.mark.parametrize("mode", _declared_obs_modes())
def test_declared_features_are_exactly_what_the_gym_renders(domain, instance, mode):
    """Spec §7: the rendered vector IS the declared vector. Upstream ships no
    check for this — a prototype reddens a frozen example — so the rule is a
    review item everywhere and an executable one only where a domain writes it.
    This is that.

    Two defects lived here until 2026-08-28 and neither was reachable by
    any gate:

    * every mode rendered a leading `period / horizon` that no `features` list
      named, so a reader of the IR could not tell the observation carried time
      at all (this domain is one of the two upstream #71 surveyed);
    * `echelon_transit_per_slot` summed OVER the pipeline slots while the gym
      keeps one echelon running-sum PER slot. Those agree only at
      `leadtime == 2`, where there is exactly one observable slot — which is
      the cell the expression was written against. At `verify_l3` the
      declaration was both too narrow and numerically wrong, and at
      `leadtime == 1` it declared an all-zero vector the gym never renders.
      `to_echelon`'s own docstring says collapsing the slots "would be lossy";
      the IR said otherwise since the mode existed.

    Exact equality is the right instrument HERE and not in general: spec §7
    (v0.9.33) rules a mode may re-encode what it declares, so upstream must
    check provenance rather than width. This domain re-encodes nothing, so the
    stronger claim is available and is worth more.
    """
    from mdp_ir import load_ir
    from mdp_ir.interpreter import IrInterpreter

    scen, mdpmod, gymmod = domain
    sc = scen.SCENARIOS[instance]
    st, _ = mdpmod.init_state(sc, episode_seed=7)
    for _ in range(3):                      # a few periods in, so nothing is 0
        st = mdpmod.advance1(sc, st)
        st, _ = mdpmod.advance2(sc, st, [1.0] * sc.n_echelons)
    st = mdpmod.advance1(sc, st)            # the observation point: post-A

    ns = {
        "stock": list(st.stock[: sc.n_echelons]),
        "pipe": [list(st.pipeline[k]) for k in range(sc.n_echelons)],
        "period": st.period,
    }
    declared = IrInterpreter(load_ir(str(SCHEMA)), instance=instance).observe(mode, ns)
    flat = np.concatenate(
        [np.asarray(v, dtype=float).ravel() for v in declared.values()]
    )
    rendered = gymmod.observation(sc, st, mode).astype(float)

    assert flat.shape == rendered.shape, (
        f"{instance}/{mode}: the IR declares {flat.shape[0]} components "
        f"({', '.join(declared)}) and the gym renders {rendered.shape[0]}"
    )
    assert np.allclose(flat, rendered, atol=1e-5), (
        f"{instance}/{mode}: declared {flat.round(3).tolist()} != "
        f"rendered {rendered.round(3).tolist()}"
    )


def test_the_time_feature_is_declared_on_every_observation_mode():
    """The component §7 says goes missing, pinned by name.

    The width test above would catch its removal from the gym, but not its
    removal from the IR alone — and the IR is the artifact a reader trusts.
    """
    modes = _raw_flat().get("gym", {}).get("observation_modes")
    if modes is None:                       # grouped layout keeps gym under rendering
        modes = json.loads(SCHEMA.read_text())["rendering"]["gym"]["observation_modes"]
    for m in modes:
        names = [f.get("derived") or f.get("ref") for f in m["features"]]
        want = "time_to_go" if m["name"].endswith("_ttg") else "period_frac"
        assert names[0] == want, (
            f"observation mode {m['name']!r} must declare {want} first, "
            f"in render order; got {names}"
        )


def test_the_log_is_readable_by_the_harness_reader():
    """§13's two halves are only checked at launch if the harness can FIND the
    table — and for the whole of this campaign it could not.

    `mdp_conformance.launch.registry_ids` looks for `^##\\s+CONFIG-REGISTRY`.
    This log headed the section `## §CONFIG-REGISTRY`, matching the prose form
    (`see §CONFIG-REGISTRY`) rather than the guide's own §11 template, which
    heads all five sections without the sign. So the reader returned `None`,
    check 4 downgraded the cross-half assertion to a note, and the launch check
    would have reported a clean config on an address whose ids no reader could
    resolve. That is the #E12 shape exactly: a rule that silently does not run.

    `test_config_registry_halves_agree` above cannot catch it — it parses the
    log with its own reader, so it agrees with itself. This one uses the
    harness's.
    """
    from mdp_conformance.launch import registry_ids

    cfg = _configs()
    log = Path(__file__).resolve().parent / "ESCALATION.md"
    defined = registry_ids(log)
    assert defined is not None, (
        "the harness reader finds no CONFIG-REGISTRY section in ESCALATION.md "
        "— check the heading is `## CONFIG-REGISTRY`, not `## §CONFIG-REGISTRY`"
    )
    data = set(cfg.SC) | set(cfg.G) | set(cfg.A) | set(cfg.H)
    assert defined == data, (
        f"log-only ids {sorted(defined - data)}, data-only {sorted(data - defined)}"
    )


def test_the_launch_surface_is_protocol_and_never_run_identity():
    """`--config`/`--deviation`/`--comparator` say what a run MEANS.

    Two runs differing only in those are the same run (spec §8.2), so they must
    not reach the run name — otherwise adopting the check would re-key every
    result dir and orphan the campaign's own join keys.
    """
    pytest.importorskip("sb3_contrib")
    import importlib

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    train = importlib.import_module("clark_scarf_ppo_train")
    cfg = _configs()

    surface = {"config", "deviation", "comparator", "settling"}
    assert surface <= train._SKIP, f"not skipped in the run name: {surface - train._SKIP}"
    assert surface <= cfg.PROTOCOL, f"not protocol in the registry: {surface - cfg.PROTOCOL}"

    plain = train._build_arg_parser().parse_args(["-s", "n3_l2_p09"])
    cited = train._build_arg_parser().parse_args(
        ["-s", "n3_l2_p09", "--config", "sc0/g0/a0/h0", "--comparator", "PPO_x",
         "--deviation", "n_epochs=30"])
    for a in (plain, cited):
        a.gamma = train._derived_gamma(a.scenario_name)
    strip = lambda n: n.split("_", 3)[3] if n.count("_") >= 3 else n
    assert strip(train.build_run_name(plain)) == strip(train.build_run_name(cited))


def test_declared_deviations_parse_both_forms():
    """`dest` alone declares "this moved"; `dest=value` pins where to, and the
    check then refuses a run whose value disagrees with its own banner."""
    pytest.importorskip("sb3_contrib")
    import importlib

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    train = importlib.import_module("clark_scarf_ppo_train")
    args = train._build_arg_parser().parse_args(
        ["-s", "n3_l2_p09", "--deviation", "n_epochs=30", "--deviation", "ent_coef"])
    assert train._declared_deviations(args) == {"n_epochs": "30", "ent_coef": None}


# -- the mixture-at-zero ordinal head (a-axis) --------------------------------


def _ordinal():
    import importlib

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module("clark_scarf_ordinal_head")


def test_ordinal_head_is_a_proper_distribution():
    """Every component must normalize, for arbitrary raw parameters.

    The expansion builds log-probabilities by hand — a mixture weight plus a
    renormalized quadratic branch — so nothing downstream would notice if they
    failed to sum to one: `Categorical(logits=...)` silently re-softmaxes, and
    the head would train against a distribution nobody declared.
    """
    pytest.importorskip("torch")
    import torch as th

    oh = _ordinal()
    dims = [41, 41, 41]
    th.manual_seed(0)
    raw = th.randn(64, oh.N_PARAMS_PER_COMPONENT * len(dims)) * 3.0
    logits = oh.expand_ordinal_logits(raw, dims)
    assert logits.shape == (64, sum(dims))
    for j, n in enumerate(dims):
        part = logits[:, j * n : (j + 1) * n]
        total = th.logsumexp(part, dim=1)
        assert th.allclose(total, th.zeros_like(total), atol=1e-5), (
            f"component {j} does not normalize: logsumexp ranges "
            f"[{total.min():.3e}, {total.max():.3e}]"
        )
    assert th.isfinite(logits).all(), "non-finite logit in the expansion"


def test_ordinal_head_can_represent_every_shipment_quantity():
    """Three parameters induce 41 logits, so representability is a real claim.

    A head that cannot put its mode on the DP's action cannot reproduce the
    reference policy however well it is trained, and the failure would look
    like a tuning problem. Checked for EVERY quantity on EVERY link.
    """
    pytest.importorskip("torch")
    import math

    import torch as th

    oh = _ordinal()
    dims = [41, 41, 41]
    n = dims[0]
    for k in range(n):
        p = th.zeros(1, len(dims), oh.N_PARAMS_PER_COMPONENT)
        if k == 0:
            p[:, :, 0] = 20.0                      # w -> 1: the zero atom
        else:
            p[:, :, 0] = -20.0                     # w -> 0
            r = min(max((k - 1) / (n - 2), 1e-9), 1 - 1e-9)
            p[:, :, 1] = math.log(r / (1 - r))     # mu -> k
            p[:, :, 2] = -20.0                     # tau -> tau_min: sharp
        logits = oh.expand_ordinal_logits(p.view(1, -1), dims)
        for j in range(len(dims)):
            got = int(th.argmax(logits[:, j * n : (j + 1) * n], dim=1)[0])
            assert got == k, f"link {j}: head puts its mode on {got}, wanted {k}"


def test_ordinal_head_init_is_the_documented_prior():
    """The init is a CHOICE, inherited from the siblings, so it is pinned.

    A zero-init categorical over 41 bins starts at ~0.024 everywhere. This head
    starts with HALF its mass on "ship nothing" and spreads the rest nearly
    flat. That is a real prior toward not shipping — and NOT defensible here
    (#E6): this domain has no fixed cost, the DP ships ~ demand nearly every
    period, and under deterministic evaluation the 0.5 atom gates the argmax
    behind ~3.7 logit-units of one scalar. Kept pinned because a1 is a frozen
    record, not a recommendation. If either number moves, the
    exploration these arms begin from has changed and every comparison against
    the categorical head is measuring something new.
    """
    pytest.importorskip("torch")
    import torch as th
    from torch.nn import functional as F

    oh = _ordinal()
    dims = [41, 41, 41]
    probs = oh.expand_ordinal_logits(
        th.zeros(1, oh.N_PARAMS_PER_COMPONENT * len(dims)), dims).exp()[0, :41]

    tau = oh.TAU_MIN + oh.TAU_SCALE * float(F.softplus(th.tensor(0.0)))
    assert tau == pytest.approx(27.98, abs=0.01)
    assert float(probs[0]) == pytest.approx(0.5, abs=1e-6), (
        "the zero atom no longer starts at 0.5"
    )
    pos = probs[1:]
    assert float(pos.max() / pos.min()) == pytest.approx(1.275, abs=0.01), (
        "the positive branch is no longer near-flat at init"
    )


def test_ordinal_head_refuses_a_continuous_action_space():
    """An encoding that is only sound under one action space must REFUSE the
    others rather than document it — the rule this folder already applies to
    `--mask`."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    oh = _ordinal()
    scen = __import__("clark_scarf_scenarios")
    gym_mod = __import__("clark_scarf_gym")
    sc = scen.SCENARIOS["g1_n3_l2_p09"]
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "raw", "ship_absolute")])
    with pytest.raises(AssertionError, match="MultiDiscrete"):
        PPO(oh.OrdinalActorCriticPolicy, env,
            policy_kwargs={"net_arch": [8, 8]}, n_steps=64, batch_size=32,
            verbose=0, device="cpu")


def test_ordinal_head_params_survive_save_and_load(domain, tmp_path):
    """`tau_min`/`tau_scale` must live INSIDE the model.

    Both sibling implementations keep these as module globals, which is the
    shape of the bug `test_custom_head_params_survive_save_and_load` documents:
    a head rebuilt at load time with a different width is a policy that was
    never trained. Sabotaging the module global would NOT catch it here — the
    defaults are bound when the `def` executes — so the check is that a
    non-default value comes back intact, which fails if
    `_get_constructor_parameters` stops carrying it.
    """
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    oh = _ordinal()
    scen, _, gym_mod = domain
    sc = scen.SCENARIOS["g1_n3_l2_p09"]
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "raw", "ship_discrete")])
    m = PPO(oh.OrdinalActorCriticPolicy, env,
            policy_kwargs={"net_arch": [8, 8], "tau_min": 0.5, "tau_scale": 7.0},
            n_steps=64, batch_size=32, verbose=0, device="cpu")
    m.save(str(tmp_path / "ord"))
    back = PPO.load(str(tmp_path / "ord"), device="cpu")
    assert back.policy.action_dist.tau_scale == pytest.approx(7.0), (
        f"tau_scale came back as {back.policy.action_dist.tau_scale}: it is not "
        f"being saved with the model"
    )
    assert back.policy.action_dist.tau_min == pytest.approx(0.5)


def test_ordinal_head_trains_and_its_actions_are_legal(domain):
    """End to end: the head drives PPO and emits in-space actions.

    The expansion sits between the network and SB3's inherited machinery, so a
    layout error would surface as a shape or a log_prob mismatch during the
    first update rather than as a bad score.
    """
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    oh = _ordinal()
    scen, _, gym_mod = domain
    sc = scen.SCENARIOS["g1_n3_l2_p09"]
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "echelon", "ship_discrete")])
    m = PPO(oh.OrdinalActorCriticPolicy, env,
            policy_kwargs={"net_arch": [8, 8]}, n_steps=64, batch_size=32,
            n_epochs=1, verbose=0, device="cpu", seed=0)
    m.learn(total_timesteps=128)
    obs = env.reset()
    for _ in range(20):
        act, _ = m.predict(obs, deterministic=True)
        assert env.envs[0].action_space.contains(act[0]), f"illegal action {act[0]}"
        obs, _, _, _ = env.step(act)
    assert m.policy.action_net.out_features == 9, (
        f"action layer emits {m.policy.action_net.out_features}, expected 9 "
        f"(3 parameters x 3 links)"
    )


def test_registry_resolves_the_derived_gamma():
    """`resolve()` must fill gamma from the instance, or check 4 refuses
    every addressed run.

    F11 made beta per instance and gamma derived from it, so the train script's
    CLI default became None and `_L1_DERIVED` deliberately omits the row —
    there is no constant it could hold. That left a hole in `origin('h')` that
    only the resolved scenario can fill, and the symptom was invisible until an
    addressed launch: `gamma resolves to 0.95 where sc0/g2/a0/h1 fixes None`.
    An id that searched gamma must still win over the derivation.
    """
    cfg = _configs()
    from clark_scarf_scenarios import SCENARIOS

    for addr in ("sc0/g2/a0/h0", "sc0/g3/a1/h0"):
        r = cfg.resolve(addr)
        assert r["gamma"] == pytest.approx(SCENARIOS[r["scenario_name"]].beta), (
            f"{addr}: gamma {r['gamma']!r} is not the instance's beta"
        )
    # an id that SEARCHES gamma must survive the fill. No such id exists on this
    # board yet, so the property is checked against a synthetic delta rather
    # than left unchecked until the first tuning study mints one.
    probe = dict(cfg.resolve("sc0/g2/a0/h0"))
    probe["gamma"] = 0.9131
    assert cfg._fill_derived(probe, "n3_l2_p09")["gamma"] == pytest.approx(0.9131)


def test_the_derived_gamma_fill_does_not_inflate_levels():
    """The fill belongs to the axis that OWNS gamma, and nowhere else.

    Filling it into `origin('g')` / `origin('a')`, which do not own it, makes
    every axis read as moved the moment an id searches gamma — `sc2/g3/a0/h6`
    inflated L2(hp) -> L2(gym+arch+hp) and `sc1/g4/a1/h3` L2(arch+hp) ->
    L2(gym+arch+hp). Upstream measured 33 level disagreements in 105 runs when
    a campaign read §13.4 loosely, every one in the direction that inflates.
    """
    cfg = _configs()
    assert "gamma" in cfg.H_OWNS and "gamma" not in cfg.G_OWNS | cfg.A_OWNS
    for addr, want in (("sc0/g0/a0/h0", "L1"),
                       ("sc0/g2/a0/h0", "L1"),
                       ("sc0/g3/a0/h0", "L1"),
                       ("sc0/g2/a1/h0", "L2(arch)")):
        assert cfg.level(addr) == want, f"{addr} reads {cfg.level(addr)}, recorded {want}"
    for axis in ("g", "a"):
        assert "gamma" not in cfg._fill_derived(cfg.origin(axis), "n3_l2_p09")


# -- §9.7: selection is a post-hoc screen, not a live callback ----------------


def test_no_live_selection_machinery_exists():
    """§9.7's rule, now enforced upstream by behaviour rather than by name.

    §9.7 has specified post-hoc checkpoint selection since v0.7.0 — "no
    EvalCallback, no live selection env". This domain shipped a live
    `CRNSelectionCallback` and the check PASSED throughout, because it detected
    by class NAME and this one was called something else (#E1, upstream #89,
    fixed in v0.10.14 — the detector is now behavioural).

    Turning it off by default was not enough: the check reads the SOURCE, and
    the class was still defined. It was kept "to reproduce runs that used it",
    and that rationale is board-specific — it belongs to the board with ~1100
    such runs, not to this one, which has none. So it is gone, and this test
    pins its absence rather than its default.
    """
    pytest.importorskip("sb3_contrib")
    import importlib

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    train = importlib.import_module("clark_scarf_ppo_train")

    args = train._build_arg_parser().parse_args(["-s", "n3_l2_p09"])
    src = (Path(d) / "clark_scarf_ppo_train.py").read_text()
    for gone in ("CRNSelectionCallback", "--live-select", "rollout_seeds"):
        assert gone not in src, (
            f"{gone!r} is back in the train script: §9.7 forbids evaluating or "
            f"shipping in-training, and upstream detects it by BEHAVIOUR since "
            f"v0.10.14 — a flag defaulting to off does not clear it"
        )
    assert args.checkpoint_every_frac > 0, (
        "a run that saves no checkpoints has nothing for the screen to rank"
    )
    assert "save_vecnormalize=True" in src, (
        "§9.7 scores each checkpoint under ITS OWN normalizer; without "
        "save_vecnormalize the screen ranks early checkpoints on observations "
        "they never saw"
    )
    assert (Path(d) / "clark_scarf_select.py").exists(), (
        "the conformant route needs the screen to exist, not just the callback "
        "to be off"
    )


def test_deployable_policy_refuses_a_missing_normalizer(domain, tmp_path):
    """§12's artifact may not silently ship unnormalized.

    A policy trained under VecNormalize and loaded without its statistics is
    not a worse policy, it is a different one — and the only symptom is a drift
    against `ppo_eval` that nothing is obliged to check. The failure mode is
    this campaign's own: a head hyperparameter that did not travel scored ~1004
    at selection and ~13000 at eval.

    Reachable because §9.7's fix moved the file: `vecnormalize.pkl` was written
    only by the live selection callback, so with that off, the old default path
    resolves to nothing.
    """
    pytest.importorskip("stable_baselines3")
    import importlib

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    pol = importlib.import_module("clark_scarf_policy")

    scen, _, gym_mod = domain
    sc = scen.SCENARIOS["n3_l2_p09"]           # for the env only; the policy
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "raw", "ship_discrete")])
    m = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [8, 8]},
            n_steps=64, batch_size=32, verbose=0, device="cpu")
    m.save(str(tmp_path / "n3_l2_p09_ppo"))
    (tmp_path / "n3_l2_p09_ppo_args.txt").write_text(
        "observation_mode: raw\naction_mode: ship_discrete\nnorm_obs: True\n")

    with pytest.raises(FileNotFoundError, match="norm_obs=True"):
        pol.ClarkScarfPolicy(model_path=tmp_path / "n3_l2_p09_ppo.zip",
                             scenario="n3_l2_p09", observation_mode="raw",
                             action_mode="ship_discrete")

    # a run that genuinely did not normalize is not refused
    (tmp_path / "n3_l2_p09_ppo_args.txt").write_text(
        "observation_mode: raw\naction_mode: ship_discrete\nnorm_obs: False\n")
    p = pol.ClarkScarfPolicy(model_path=tmp_path / "n3_l2_p09_ppo.zip",
                             scenario="n3_l2_p09", observation_mode="raw",
                             action_mode="ship_discrete")
    assert p._obs_rms is None


def test_eval_finds_the_run_args_from_a_checkpoint_path():
    """§9.7 evaluates CHECKPOINTS, whose parent is `checkpoints/`.

    `model_path.parent` is the run directory only for the terminal artifact.
    For a checkpoint it is one level too deep, the args log is not found, and
    every field falls back to a parser default — an echelon checkpoint scores
    as `raw`, the level renders `L?`, and the missing-normalizer refusal cannot
    fire because it tests `norm_obs` from an empty mapping. The `L?` in an arm
    string was the visible half of that.
    """
    pytest.importorskip("sb3_contrib")
    import importlib.util as u

    d = Path(__file__).resolve().parent
    spec = u.spec_from_file_location("_ev", d / "clark_scarf_ppo_eval.py")
    ev = u.module_from_spec(spec)
    spec.loader.exec_module(ev)

    runs = sorted((d / "results" / "n3_l2_p09").glob("PPO_*")) \
        if (d / "results" / "n3_l2_p09").is_dir() else []
    run = next((r for r in runs
                if (r / "n3_l2_p09_ppo_args.txt").exists()
                and (r / "checkpoints").is_dir()
                and any((r / "checkpoints").glob("*_steps.zip"))), None)
    if run is None:
        pytest.skip("no run with checkpoints on disk")

    ckpt = next((run / "checkpoints").glob("*_steps.zip"))
    assert ev._resolve_run_dir(ckpt, "n3_l2_p09") == run
    assert ev._resolve_run_dir(run / "n3_l2_p09_ppo_final.zip", "n3_l2_p09") == run
    got = ev._read_run_args(ev._resolve_run_dir(ckpt, "n3_l2_p09"), "n3_l2_p09")
    for key in ("observation_mode", "action_mode", "level", "norm_obs"):
        assert got.get(key), f"{key} unreadable from a checkpoint path"


# -- the discretized-Gaussian head (a2): adjacency pooling, NO zero atom ------


def _dgauss():
    import importlib

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module("clark_scarf_dgauss_head")


def test_dgauss_head_is_a_proper_distribution():
    pytest.importorskip("torch")
    import torch as th

    dg = _dgauss()
    dims = [41, 41, 41]
    th.manual_seed(0)
    raw = th.randn(64, dg.N_PARAMS_PER_COMPONENT * len(dims)) * 3.0
    logits = dg.expand_dgauss_logits(raw, dims)
    assert logits.shape == (64, sum(dims))
    for j, n in enumerate(dims):
        total = th.logsumexp(logits[:, j * n : (j + 1) * n], dim=1)
        assert th.allclose(total, th.zeros_like(total), atol=1e-5)
    assert th.isfinite(logits).all()


def test_dgauss_head_reaches_every_bin_at_moderate_m():
    """The affine mu is the #E6 fix: EVERY bin — the boundaries included — is
    the mode at |m| <= 1, with no saturating input. The hurdle head needed
    ~3.7 logit-units of one scalar before its argmax could leave zero, and a
    sigmoid mu would have rebuilt that cliff at the support edges."""
    pytest.importorskip("torch")
    import torch as th

    dg = _dgauss()
    dims = [41, 41, 41]
    n = dims[0]
    for k in range(n):
        m = 2.0 * k / (n - 1) - 1.0                 # |m| <= 1 by construction
        p = th.zeros(1, len(dims), dg.N_PARAMS_PER_COMPONENT)
        p[:, :, 0] = m
        p[:, :, 1] = -20.0                          # tau -> tau_min: sharp
        logits = dg.expand_dgauss_logits(p.view(1, -1), dims)
        for j in range(len(dims)):
            got = int(th.argmax(logits[:, j * n : (j + 1) * n], dim=1)[0])
            assert got == k, f"link {j}: mode at {got}, wanted {k} (m={m:+.3f})"


def test_dgauss_init_is_genuinely_flat_with_a_live_argmax():
    """The property whose absence produced #E6, pinned.

    The hurdle head started with HALF its mass on one action and its
    deterministic argmax on 'ship nothing' in every state. This head must
    start near-flat (no bin over ~1.3x any other) with its argmax mid-range —
    a live action that moves continuously with mu, so an undertrained policy
    evaluates as a mediocre shipper, never as a constant catastrophe."""
    pytest.importorskip("torch")
    import torch as th

    dg = _dgauss()
    dims = [41, 41, 41]
    probs = dg.expand_dgauss_logits(
        th.zeros(1, dg.N_PARAMS_PER_COMPONENT * len(dims)), dims).exp()[0, :41]
    assert float(probs.max() / probs.min()) < 1.35, "init is no longer near-flat"
    assert int(probs.argmax()) == 20, "init argmax is no longer mid-range"
    assert abs(float(probs[0]) - 1 / 41) < 0.005, (
        f"bin 0 starts at {float(probs[0]):.4f}: the zero action must carry "
        f"~flat mass, not an atom"
    )


def test_dgauss_head_refuses_a_continuous_action_space():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    dg = _dgauss()
    scen = __import__("clark_scarf_scenarios")
    gym_mod = __import__("clark_scarf_gym")
    sc = scen.SCENARIOS["g1_n3_l2_p09"]
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "raw", "ship_absolute")])
    with pytest.raises(AssertionError, match="MultiDiscrete"):
        PPO(dg.DGaussActorCriticPolicy, env,
            policy_kwargs={"net_arch": [8, 8]}, n_steps=64, batch_size=32,
            verbose=0, device="cpu")


def test_dgauss_head_params_survive_save_and_load(domain, tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    dg = _dgauss()
    scen, _, gym_mod = domain
    sc = scen.SCENARIOS["n3_l2_p09"]
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "raw", "ship_discrete")])
    m = PPO(dg.DGaussActorCriticPolicy, env,
            policy_kwargs={"net_arch": [8, 8], "tau_min": 0.5, "tau_scale": 7.0},
            n_steps=64, batch_size=32, verbose=0, device="cpu")
    m.save(str(tmp_path / "dg"))
    back = PPO.load(str(tmp_path / "dg"), device="cpu")
    assert back.policy.action_dist.tau_scale == pytest.approx(7.0)
    assert back.policy.action_dist.tau_min == pytest.approx(0.5)


def test_dgauss_head_trains_and_its_actions_are_legal(domain):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    dg = _dgauss()
    scen, _, gym_mod = domain
    sc = scen.SCENARIOS["n3_l2_p09"]
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "echelon", "ship_discrete")])
    m = PPO(dg.DGaussActorCriticPolicy, env,
            policy_kwargs={"net_arch": [8, 8]}, n_steps=64, batch_size=32,
            n_epochs=1, verbose=0, device="cpu", seed=0)
    m.learn(total_timesteps=128)
    obs = env.reset()
    for _ in range(20):
        act, _ = m.predict(obs, deterministic=True)
        assert env.envs[0].action_space.contains(act[0])
        obs, _, _, _ = env.step(act)
    assert m.policy.action_net.out_features == 6, (
        f"action layer emits {m.policy.action_net.out_features}, expected 6 "
        f"(2 numbers x 3 links)"
    )


def test_dgauss_sigmoid_variant_is_the_atomless_twin():
    """a3 (#E6 follow-up): same machinery as a2, sibling's sigmoid mu.

    Three pins: (i) the init is IDENTICAL to the affine variant's — flat, live
    mid-range argmax — so an a2-vs-a3 contrast is parameterization-only from
    step one; (ii) interior bins are the mode at moderate m; (iii) mu_param
    survives save/load, because a head rebuilt with the wrong variant is a
    policy that never trained.
    """
    pytest.importorskip("torch")
    import torch as th

    dg = _dgauss()
    dims = [41, 41, 41]
    pa = dg.expand_dgauss_logits(th.zeros(1, 6), dims, mu_param="affine").exp()[0, :41]
    ps = dg.expand_dgauss_logits(th.zeros(1, 6), dims, mu_param="sigmoid").exp()[0, :41]
    assert th.allclose(pa, ps, atol=1e-6), "the two variants must start identical"

    import math
    for k in (1, 10, 20, 30, 39):          # interior bins at moderate m
        m = math.log((k / 40) / (1 - k / 40))
        p = th.zeros(1, len(dims), 2); p[:, :, 0] = m; p[:, :, 1] = -20.0
        logits = dg.expand_dgauss_logits(p.view(1, -1), dims, mu_param="sigmoid")
        assert int(th.argmax(logits[:, :41], dim=1)[0]) == k

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import tempfile, os
    scen = __import__("clark_scarf_scenarios")
    gym_mod = __import__("clark_scarf_gym")
    sc = scen.SCENARIOS["n3_l2_p09"]
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "raw", "ship_discrete")])
    m2 = PPO(dg.DGaussActorCriticPolicy, env,
             policy_kwargs={"net_arch": [8, 8], "mu_param": "sigmoid"},
             n_steps=64, batch_size=32, verbose=0, device="cpu")
    with tempfile.TemporaryDirectory() as td:
        m2.save(os.path.join(td, "s"))
        back = PPO.load(os.path.join(td, "s"), device="cpu")
        assert back.policy.action_dist.mu_param == "sigmoid"


def test_every_model_loader_refuses_a_missing_normalizer(domain, tmp_path):
    """The §12 policy was not the only consumer, and the probe proved it.

    Three other modules load a model and inject VecNormalize statistics, each
    resolving the file by name. Two names are in play: `vecnormalize.pkl` from
    the removed live-selection callback, `vecnormalize_final.pkl` from the end
    of training. A loader that looks for only the first finds nothing and
    scores raw — which is not a worse number, it is a different policy.

    That is not hypothetical: the probe shipped with exactly that defect and
    read two tuned nets back on observations they never saw, producing a
    readback that had to be retracted (#E11).

    **This test EXERCISES the refusal; it does not grep for it.** The first
    version asserted the string "REFUS" appeared in each source, and passed on
    a `select.py` whose normalizer refusal had been deleted — because that file
    refuses `--first-seed 0` elsewhere and the substring was still there. A
    name check that cannot fail is the defect this campaign filed upstream as
    issue #89, reproduced in its own test file.
    """
    pytest.importorskip("stable_baselines3")
    import importlib
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    probe = importlib.import_module("clark_scarf_policy_probe")

    scen, _, gym_mod = domain
    sc = scen.SCENARIOS["n3_l2_p09"]
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "raw", "ship_discrete")])
    m = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [8, 8]},
            n_steps=64, batch_size=32, verbose=0, device="cpu")
    m.save(str(tmp_path / "n3_l2_p09_ppo_final"))
    args_f = tmp_path / "n3_l2_p09_ppo_args.txt"
    args_f.write_text(
        "observation_mode: raw\naction_mode: ship_discrete\nnorm_obs: True\n")

    # the probe: refuses rather than reading a normalized policy back raw
    with pytest.raises(SystemExit, match="norm_obs=True"):
        probe._load_policy(tmp_path / "n3_l2_p09_ppo_final.zip", sc,
                           "raw", "ship_discrete")

    # a run that genuinely did not normalize is not refused
    args_f.write_text(
        "observation_mode: raw\naction_mode: ship_discrete\nnorm_obs: False\n")
    assert probe._load_policy(tmp_path / "n3_l2_p09_ppo_final.zip", sc,
                              "raw", "ship_discrete") is not None


def test_select_refuses_to_rank_checkpoints_without_normalizers(
        domain, tmp_path, monkeypatch):
    """The screen RANKS, and an unnormalized ranking still prints a top-k.

    `clark_scarf_select.py` had the two-name fallback but no refusal, so a run
    whose normalizers were absent would be screened on raw observations and
    emit a confident top-k for confirmation with nothing marking it. The
    refusal must fire before any checkpoint is scored — this drives `main()` to
    prove it does.
    """
    pytest.importorskip("stable_baselines3")
    import importlib
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    sel = importlib.import_module("clark_scarf_select")

    scen, _, gym_mod = domain
    sc = scen.SCENARIOS["n3_l2_p09"]
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "raw", "ship_discrete")])
    m = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [8, 8]},
            n_steps=64, batch_size=32, verbose=0, device="cpu")

    run_dir = tmp_path / "n3_l2_p09" / "PPO_test"
    (run_dir / "checkpoints").mkdir(parents=True)
    m.save(str(run_dir / "checkpoints" / "n3_l2_p09_ppo_1000_steps"))
    (run_dir / "n3_l2_p09_ppo_args.txt").write_text(
        "observation_mode: raw\naction_mode: ship_discrete\nnorm_obs: True\n")

    assert sel.resolve_vecnorm(
        sel.find_checkpoints(run_dir)[0], run_dir) is None, (
        "fixture is wrong: a normalizer was found, so the refusal cannot fire")

    monkeypatch.setattr(sys, "argv", ["clark_scarf_select.py", str(run_dir)])
    with pytest.raises(SystemExit, match="norm_obs=True"):
        sel.main()


def test_deployable_policy_resolves_modes_from_the_run(domain, tmp_path, monkeypatch):
    """`-a` is as load-bearing as `--model-path`, and its default was a trap.

    The §12 CLI defaulted `--action-mode` to `ship_fraction`. Replaying a
    `ship_discrete` net without passing `-a` therefore decoded a MultiDiscrete
    action as a fraction and reported ~35x the true cost — the magnitude of the
    random baseline, so it reads as a plausibly bad policy rather than as a
    wiring mistake, and nothing raised. The modes must come off the run, and an
    explicit mode that contradicts the run must be refused.
    """
    pytest.importorskip("stable_baselines3")
    import importlib
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    pol = importlib.import_module("clark_scarf_policy")

    scen, _, gym_mod = domain
    sc = scen.SCENARIOS["n3_l2_p09"]
    env = DummyVecEnv([lambda: gym_mod.ClarkScarfEnv(sc, "echelon", "ship_discrete")])
    m = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [8, 8]},
            n_steps=64, batch_size=32, verbose=0, device="cpu")
    m.save(str(tmp_path / "n3_l2_p09_ppo_final"))
    (tmp_path / "n3_l2_p09_ppo_args.txt").write_text(
        "observation_mode: echelon\naction_mode: ship_discrete\nnorm_obs: False\n")

    # the run dir is found from a CHECKPOINT path too, not just from beside the model
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    m.save(str(ck / "n3_l2_p09_ppo_100_steps"))
    assert pol._resolve_run_dir(ck / "n3_l2_p09_ppo_100_steps.zip",
                                "n3_l2_p09") == tmp_path

    # an explicit mode contradicting the run is refused, not silently scored
    monkeypatch.setattr(sys, "argv", [
        "clark_scarf_policy.py", "--model-path",
        str(tmp_path / "n3_l2_p09_ppo_final.zip"), "-s", "n3_l2_p09",
        "-a", "ship_fraction", "--episodes", "1"])
    with pytest.raises(SystemExit, match="did not train with"):
        pol.main()

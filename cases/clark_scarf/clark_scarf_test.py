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

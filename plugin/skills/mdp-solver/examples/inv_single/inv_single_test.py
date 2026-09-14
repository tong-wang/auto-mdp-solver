"""inv_single's own tests — the claims only this domain can make.

Run from anywhere (the portable-domain contract):

    pytest plugin/skills/mdp-solver/examples/inv_single
    python inv_single_test.py

Everything true of *any* IR — determinism, decision-path independence,
termination, declared invariants, mixture standalone equivalence — is checked
by ``mdp_ir.laws`` and only invoked here. What lives in this file is
inv_single-specific: the bit-exact differential over its covering set, the
equivalence between the family-generic runtime and the hand-written latent
generators, and the negative controls that prove the gates can fail.

The conservation laws that used to be asserted here in Python now live in the
IR as ``mdp.invariants`` (inventory balance, pipeline balance, the stockout
modes), so every instance and every appended candidate inherits them.
"""

from __future__ import annotations

import json

import pytest

from mdp_ir import families, layering, runtime
from mdp_ir.interpreter import IrInterpreter
from mdp_ir.schema import MdpIR, load_ir
from mdp_ir.testing import (
    assert_diverges,
    assert_laws,
    assert_match,
    differential,
    mutated,
    schema_beside,
)

SCHEMA = schema_beside(__file__)
EPISODES = 8


def _declared_compositions() -> list[str | None]:
    """Base + every declared instance + every mixture, read from the schema so
    the sweep cannot drift from the catalog it is meant to cover."""
    raw = json.loads(SCHEMA.read_text())
    scenario = raw["mdp"]["scenario"]
    return ([None] + sorted(scenario.get("instances") or {})
            + sorted(m["name"] for m in scenario.get("mixtures") or []))


COMPOSITIONS = _declared_compositions()


# -- engine laws -------------------------------------------------------------


def test_engine_laws():
    assert_laws(SCHEMA)


# -- the covering set: one node per declared composition ---------------------


@pytest.mark.parametrize("instance", COMPOSITIONS,
                         ids=lambda i: i or "base")
def test_differential_matches_the_domain(instance):
    """The interpreter and the hand-written domain must agree bit-for-bit on
    every declared composition — all three demand candidates, both lead-time
    candidates, both stockout modes, the event-order variants, per-instance
    pipeline_len, and the cross-family mixture's re-selection path."""
    assert_match(SCHEMA, instance=instance, episodes=EPISODES)


def test_the_covering_set_is_not_silently_empty():
    """Guards the derivation above: if the schema's instances vanished, every
    parametrized case would pass by not existing."""
    assert len(COMPOSITIONS) >= 8, COMPOSITIONS
    assert "mix_demand" in COMPOSITIONS


# -- negative controls: the gates must be able to fail -----------------------


def test_a_corrupted_dynamics_update_diverges():
    """Drop the pipeline shift from the R event: the domain still shifts, the
    interpreter no longer does, so the trajectories must part."""
    def drop_shift(doc):
        doc["mdp"]["dynamics"]["transitions"][1]["updates"] = [
            "received = pipeline[0]", "inventory += received",
        ]

    assert_diverges(SCHEMA, mutated(load_ir(SCHEMA), drop_shift), episodes=2)


def test_a_mis_stated_balance_claim_is_caught_while_the_sides_still_agree():
    """The reason invariants exist: drop `lost_sales` from the balance and the
    claim breaks under the lost_sales instance even though the interpreter and
    the domain remain bit-identical."""
    def weaken(doc):
        doc["mdp"]["invariants"][0]["expr"] = (
            "close(inventory, prev.inventory + received - demand)")

    ir = mutated(load_ir(SCHEMA, instance="lost_sales"), weaken)
    traj = IrInterpreter(ir, instance="lost_sales", seed_salt=1).run(0)
    assert traj.violations, "a wrong balance claim went unnoticed"


# -- symbolic bounds derive from the selected candidate ----------------------


@pytest.mark.parametrize("instance,expected_mean", [
    (None, 10.0),        # simple: fixed rate 10
    ("poisson", 30.0),   # gamma-rate latent: alpha/beta = 9 / 0.3
])
def test_order_bound_tracks_the_selected_demand_scale(instance, expected_mean):
    """`bounds = [0, 20 * demand.mean]` — the point of a symbolic bound is
    that a candidate with a different scale resizes the action space without
    anyone editing a number."""
    ir = load_ir(SCHEMA, instance=instance)
    src = next(s for s in ir.mdp.uncertainty_sources if s.name == "demand")
    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    derived = families.mean(
        src.distribution.family, src.distribution.settings,
        lambda v, _a=None: consts.get(v, v) if isinstance(v, str) else v,
    )
    assert derived == pytest.approx(expected_mean)
    assert ir.mdp.decision_bounds("order") == (0.0, 20.0 * expected_mean)


def test_a_selection_and_its_instance_resolve_identically():
    by_instance = load_ir(SCHEMA, instance="poisson")
    by_select = load_ir(SCHEMA, select={"demand": "poisson", "leadtime": "slt"})
    assert by_select.mdp_fingerprint() == by_instance.mdp_fingerprint()


# -- which instances survive a selection ------------------------------------


def test_surviving_instances_are_consistent_with_the_selection():
    """An instance that re-selects a slot is dropped unless it is a mixture
    component; survivors keep only their constant overrides. Asserted as the
    predicate rather than as a name list, so adding an instance to the catalog
    stays free — which is what the structural fingerprint promises."""
    raw = json.loads(SCHEMA.read_text())
    slots = {s["name"] for s in raw["mdp"]["uncertainty_slots"]}
    ir = load_ir(SCHEMA)
    components = {c for m in ir.mdp.scenario.mixtures for _, c in m.components}
    for name, overrides in ir.mdp.scenario.instances.items():
        assert not (set(overrides) & slots), (
            f"surviving instance {name!r} still carries slot key(s) "
            f"{sorted(set(overrides) & slots)}"
        )
        declared = raw["mdp"]["scenario"]["instances"][name]
        if set(declared) & slots:
            assert name in components, (
                f"instance {name!r} re-selects a slot but is no mixture "
                f"component, so it should have dropped"
            )


# -- the family-generic runtime vs. the hand-written generators --------------


class _Ctx:
    """The minimal SamplingContext a generator needs for a direct draw."""

    def __init__(self, period: int, episode_seed: int, seed_salt: int):
        self.period = period
        self.episode_seed = episode_seed
        self.seed_salt = seed_salt


@pytest.fixture(scope="module")
def uncertainty():
    """The domain's own module — a sibling import, which works because pytest
    puts each test file's directory on sys.path."""
    import inv_single_uncertainty

    return inv_single_uncertainty


def test_generic_runtime_matches_the_discrete_latent_generator(uncertainty):
    """A catalog candidate driven by FamilyGenerator must be bit-identical to
    the hand-written class it replaces — that equivalence is what lets a new
    candidate ship with zero domain Python."""
    ir = load_ir(SCHEMA, instance="discrete")
    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    generic = runtime.FamilyGenerator.from_ir(ir, "demand", instance="discrete")
    handwritten = uncertainty.LatentDiscreteDemand(
        support_size=consts["demand_support_size"],
        support_low=consts["demand_support_low"],
        support_high=consts["demand_support_high"],
        source_id=0,
    )
    for episode in (0, 3, 11):
        g, h = generic.realize(episode, 1), handwritten.realize(episode, 1)
        assert g.settings["values"] == h.vals.tolist()
        assert g.settings["probabilities"] == pytest.approx(h.probs.tolist())
        for period in range(5):
            ctx = _Ctx(period, episode, 1)
            assert g.sample(ctx) == h.sample(ctx)


def test_generic_runtime_matches_the_poisson_latent_generator(uncertainty):
    ir = load_ir(SCHEMA, instance="poisson")
    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    generic = runtime.FamilyGenerator.from_ir(ir, "demand", instance="poisson")
    handwritten = uncertainty.LatentPoissonDemand(
        alpha=consts["demand_alpha"], beta=consts["demand_beta"], source_id=0)
    for episode in (0, 4):
        g, h = generic.realize(episode, 1), handwritten.realize(episode, 1)
        assert g.settings["rate"] == h.rate
        for period in range(4):
            ctx = _Ctx(period, episode, 1)
            assert g.sample(ctx) == h.sample(ctx)
    assert generic.mean() == pytest.approx(handwritten.mean())
    assert generic.max() == pytest.approx(handwritten.max())


def test_discrete_candidate_has_a_small_per_episode_support():
    """The latent is an episode-level support of at most demand_support_size
    values — the property that makes this instance a memory problem."""
    ir = load_ir(SCHEMA, instance="discrete")
    size = {c.name: c.value for c in ir.mdp.scenario.constants}["demand_support_size"]
    for episode in (2, 4, 5):
        rows = IrInterpreter(ir, instance="discrete").run(
            episode, decisions={"order": 0.0}).rows
        assert len({r["demand"] for r in rows}) <= size


# -- the zero-code path, end to end -----------------------------------------


def test_an_appended_candidate_passes_the_differential_with_no_domain_edit():
    """A uniform-rate Poisson exists in no hand-written class: the adapter has
    to fall through to the FamilyGenerator bridge, and the result must still
    match the domain bit-for-bit."""
    raw = json.loads(SCHEMA.read_text())
    raw["mdp"]["uncertainty_slots"][0]["candidates"]["poisson_u"] = {
        "generator": "PoissonDemand", "family": "poisson",
        "settings": {"rate": {"draw": {"family": "uniform",
                                       "settings": {"low": 20.0, "high": 40.0}},
                              "example": 30.0}},
    }
    ir = MdpIR.model_validate(
        layering.resolve_catalog(raw, select={"demand": "poisson_u"}))
    assert ir.mdp.decision_bounds("order") == (0.0, 600.0)   # 20 * uniform mean 30
    report = differential(SCHEMA, ir=ir, episodes=3)
    assert report.ok, report.render()


def test_appending_a_candidate_leaves_the_structure_frozen():
    raw = json.loads(SCHEMA.read_text())
    before = layering.structural_fingerprint(raw)
    raw["mdp"]["uncertainty_slots"][0]["candidates"]["poisson_u"] = {
        "generator": "PoissonDemand", "family": "poisson",
        "settings": {"rate": 30.0},
    }
    raw["mdp"]["scenario"]["instances"]["hot"] = {"b": 4.0}
    next(c for c in raw["mdp"]["scenario"]["constants"]
         if c["name"] == "h")["value"] = 9.9
    assert layering.structural_fingerprint(raw) == before


# -- the cross-family mixture ------------------------------------------------


def test_mixture_components_carry_their_own_resolution():
    ir = load_ir(SCHEMA, instance="mix_demand")
    res = (ir.mixture_resolutions or {}).get("mix_demand") or {}
    assert set(res) == {"discrete", "poisson"}
    assert res["poisson"].selection["demand"] == "poisson"
    assert res["discrete"].selection["demand"] == "discrete"


def test_mixture_bounds_envelope_the_components():
    """0.5 * 10 + 0.5 * 30 = 20 mean demand, so 20x that is the action cap."""
    ir = load_ir(SCHEMA, instance="mix_demand")
    assert ir.mdp.decision_bounds("order") == (0.0, 400.0)


# -- the second implementation, gated ----------------------------------------
#
# `inv_single_dp_exactness.py` independently re-derives the cost recursion and
# backward induction, and its verdict is load-bearing: it is what upgraded the
# `lt` bar from "approximation" to *optimal*, and it is cited by name in the
# IR's `benchmarks[dp]` basis, whose role is `exact` with source
# `human_confirmed`. `mdp_gates --ir` refuses any candidate that beats an
# `exact` arm, so an unchecked second implementation would be silently defining
# this domain's optimality bar.
#
# Spec 1.2 / guide rule 9: gate every second implementation, and the gate lives
# here rather than behind a flag someone has to remember to run. The claim class
# is bit-exact, so the assertion is exact equality — not a tolerance.


def test_the_shipped_dp_is_exact_where_the_second_implementation_says_it_is(tmp_path):
    """DP-on-IP == the exact (u, p1) DP at deterministic L > 1, action by action.

    Forces `no_cache` so the gate checks the solver rather than a cached table,
    and writes into `tmp_path` so it neither reads nor pollutes `results/`.
    """
    from inv_single_dp_exactness import compare_exact_vs_shipped

    r = compare_exact_vs_shipped("lt", no_cache=True, results_dir=str(tmp_path))

    assert r["L"] >= 2, "the claim is about deterministic L > 1; 'lt' must supply it"
    assert r["n_bad"] == 0, (
        f"the shipped DP-on-IP departs from exact in {r['n_bad']}/{r['T']} periods "
        f"(worst |dq| = {r['max_dq']}) — the `exact` benchmark role is not earned"
    )
    assert r["max_dq"] == 0
    assert r["exact"]
    # a comparison that excluded every cell as action-ceiling-saturated would
    # pass vacuously, so the gate also checks it actually compared something
    assert r["n_sat"] < r["T"] * len(r["ip_grid"]), "every cell was saturated"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# -- campaign-2 additions: the grid, the context mode, the registry -----------
#
# grid16 (inv_single_grids.py, IR root `grids`) pads LT=0 cells' pipelines to
# the family maximum so one policy serves every cell. The padding claim — a
# zero-probability lead-time tail changes NOTHING but the observation length —
# is a second-implementation-style equivalence, so it is gated here rather
# than asserted in a docstring.

def test_grid16_matches_the_ir_declaration():
    """The code grid and the IR `grids` block enumerate the same cells."""
    from inv_single_grids import GRIDS

    ir = load_ir(SCHEMA)
    declared = {g.name: g for g in ir.grids}
    assert set(GRIDS) == set(declared), "GRIDS and IR grids disagree on names"
    for name, grid in GRIDS.items():
        ir_cells = declared[name].cells()
        assert [cid for cid, _ in grid] == [cid for cid, _ in ir_cells], (
            f"{name}: cell ids/order differ from the IR's enumeration")
        for (cid, sc), (_, overrides) in zip(grid, ir_cells):
            assert sc.shortage_cost == overrides["b"], cid
            assert sc.order_cost_fixed == overrides["K"], cid
            # the lead-time axis is spelled differently per grid: grid16 sweeps
            # a deterministic VALUE, lt_variance a whole probability LAW
            if "leadtime_value" in overrides:
                assert sc.leadtime.mean() == overrides["leadtime_value"], (
                    f"{cid}: a padded cell's effective lead time is its mean")
            else:
                assert list(sc.leadtime.probabilities) == list(
                    overrides["leadtime_probs"]), (
                    f"{cid}: the cell's law differs from the IR's — cell ids "
                    f"embed the float repr, so the arithmetic path is part of "
                    f"the join key")


def test_grid16_padded_cell_is_the_unpadded_cell_plus_zeros():
    """A zero-probability lead-time tail (the §5.6 obs-dim padding) leaves
    every transition bit-identical; only the pipeline length changes."""
    import dataclasses

    from inv_single_grids import GRIDS
    from inv_single_mdp import advance1, advance2, init_state
    from inv_single_uncertainty import DeterministicLeadtime

    padded = GRIDS["grid16"]["b=9,K=0,leadtime_value=0"]
    assert padded.leadtime.max() == 2 and padded.leadtime.mean() == 0
    unpadded = dataclasses.replace(padded, leadtime=DeterministicLeadtime(0))

    for episode_seed in range(5):
        states = []
        for sc in (padded, unpadded):
            state, _ = init_state(scenario=sc, episode_seed=episode_seed)
            state = advance1(sc, state)
            rows = []
            for t in range(sc.horizon):
                state, info = advance2(sc, state, order=float(10 + (t % 3)))
                rows.append((state.inventory, tuple(state.pipeline),
                             info["demand"], info["cost"]["total"]))
                if not state.terminated:
                    state = advance1(sc, state)
            states.append(rows)
        for (inv_p, pipe_p, d_p, c_p), (inv_u, pipe_u, d_u, c_u) in zip(*states):
            assert (inv_p, d_p, c_p) == (inv_u, d_u, c_u)
            assert pipe_p[:len(pipe_u)] == pipe_u
            assert all(x == 0 for x in pipe_p[len(pipe_u):]), (
                "padding slots must stay empty forever")


def test_vec_ctx_reads_the_episode_cell_not_the_family():
    """Under the grid sampler the cost context must come from the EPISODE's
    resolved cell — a policy conditioning on one cell's costs for every cell
    would silently be the blind arm."""
    from inv_single_grids import GRIDS
    from inv_single_gym import InvSingleEnv

    env = InvSingleEnv(scenario=GRIDS["grid16"].as_sampler(),
                      observation_mode="vec_ctx")
    seen = set()
    for episode_seed in range(40):
        obs, _ = env.reset(seed=episode_seed)
        assert env.observation_space.contains(obs)
        sc = env._scenario_ep
        b, h, K = sc.shortage_cost, sc.holding_cost, sc.order_cost_fixed
        assert obs[-2] == pytest.approx(b / (b + h))
        assert obs[-1] == pytest.approx(K)
        seen.add((b, K, sc.leadtime.mean()))
    assert len(seen) > 4, "40 episodes should visit several cells"


def test_config_registry_self_check_and_canonical_l1():
    """Importing the registry runs its structural self-check; `{sc}/L1` is
    the canonical spelling of the origins; every base is total over OWNED."""
    import inv_single_configs as registry

    assert registry.parse_tuple("sc2/L1") == ("sc2", "g0", "a0", "h0")
    assert registry.parse_tuple("sc6/g2/a0/h0") == ("sc6", "g2", "a0", "h0")
    for spec in ("sc0/L0", "sc0/g0/a0", "sc0/g0/g1/a0/h0"):
        with pytest.raises(ValueError):
            registry.parse_tuple(spec)
    for sc in registry.SC:
        base = registry.resolve(sc, "g0", "a0", "h0")
        missing = registry.OWNED - set(base)
        assert not missing, f"{sc}/L1 resolves no value for {sorted(missing)}"


def test_action_mode_is_a_value_not_a_rule():
    """`auto` is gone. It derived 'discrete' for every scenario this domain
    ships, so it was a second name for a value rather than a third mode — and
    a config cell recording `auto` named a rule where every other cell named
    the space it builds. Removing it must change no space anywhere."""
    from inv_single_grids import GRIDS
    from inv_single_gym import InvSingleEnv
    from inv_single_scenarios import SCENARIOS

    with pytest.raises(AssertionError):
        InvSingleEnv(SCENARIOS["simple"], action_mode="auto")

    sources = [(n, SCENARIOS[n]) for n in SCENARIOS]
    sources += [(gn, g.as_sampler()) for gn, g in GRIDS.items()]
    for name, source in sources:
        env = InvSingleEnv(source, observation_mode="vec")
        assert env.action_mode == "discrete", f"{name} default is not discrete"
        # the space the old `auto` branch would have built, built directly
        explicit = InvSingleEnv(source, observation_mode="vec",
                                action_mode="discrete")
        assert env.action_space == explicit.action_space, name

    import inv_single_configs as registry
    assert registry.resolve("sc0", "g0", "a0", "h0")["action_mode"] == "discrete"


def test_an_id_is_a_promotion_not_a_record_that_something_ran():
    """Guide §13.2: rows are minted for ADOPTED bases — crowned, shipped or
    parented — never for probes or failed arms. `g3`/`g5` were minted ahead of
    adoption and withdrawn; their slots stay occupied so the numbers are never
    reused, and the resolver refuses them."""
    import inv_single_configs as registry

    assert set(registry.WITHDRAWN) == {"g3", "g5"}
    for cid in registry.WITHDRAWN:
        assert cid not in registry.G, f"{cid} is withdrawn but still a base"
        with pytest.raises(ValueError, match="WITHDRAWN"):
            registry.parse_tuple(f"sc0/{cid}/a0/h0")

    # the slot is occupied: density holds across live + withdrawn, so the next
    # g id is g5+1, never g3 or g5 again
    nums = sorted(int(c[1:]) for c in list(registry.G) + list(registry.WITHDRAWN))
    assert nums == list(range(len(nums))), nums

    # and the adopted ids still resolve to what they always meant
    assert registry.resolve("sc0", "g4", "a0", "h0")["norm_obs"] is False
    assert registry.parse_tuple("sc0/L1") == ("sc0", "g0", "a0", "h0")


def test_hurdle_reaches_the_same_orders_as_discrete():
    """`hurdle` splits one decision into two heads — order at all, and how many
    in [1, q_high] — without changing the reachable order set. That is what
    makes it a fair swap for `discrete` and what lets the quantity head be
    re-parameterized later (§9 Q-B) without also moving the order/don't
    decision."""
    from inv_single_gym import InvSingleEnv
    from inv_single_scenarios import SCENARIOS

    for name in ("simple", "simple_k", "lt", "slt"):
        sc = SCENARIOS[name]
        d = InvSingleEnv(sc, observation_mode="vec", action_mode="discrete")
        h = InvSingleEnv(sc, observation_mode="vec", action_mode="hurdle")
        q_high = d.action_space.n - 1
        assert list(h.action_space.nvec) == [2, q_high], name

        reachable_d = {d.decode_action([a]) for a in range(d.action_space.n)}
        reachable_h = {h.decode_action([0, 0])}
        reachable_h |= {h.decode_action([1, k]) for k in range(q_high)}
        assert reachable_d == reachable_h, name

        # the zero atom is reachable ONLY through head 0, whatever head 1 says
        for k in (0, 5, q_high - 1):
            assert h.decode_action([0, k]) == 0.0
        assert h.decode_action([1, 0]) == 1.0        # head 1 is 1-based


def test_observation_leads_with_time_to_go():
    """obs[0] counts DOWN to the terminal boundary, in every mode (F5). The
    period index and time-to-go carry the same information for a fixed horizon,
    so nothing about the MDP changes — but which one is rendered decides what
    every trained arm saw, and a silent swap would orphan a leaderboard without
    saying so."""
    from inv_single_gym import InvSingleEnv
    from inv_single_scenarios import SCENARIOS

    for mode in ("vec", "vec_ip", "vec_ctx"):
        env = InvSingleEnv(SCENARIOS["simple"], observation_mode=mode)
        obs, _ = env.reset(seed=0)
        assert obs[0] == pytest.approx(30.0), mode          # horizon - 0
        seen = [obs[0]]
        for _ in range(29):
            obs, _, term, _, _ = env.step(0)
            seen.append(obs[0])
        assert seen[-1] == pytest.approx(1.0), mode         # one period left
        assert seen == sorted(seen, reverse=True), mode     # monotone down
        assert env.observation_space.contains(obs), mode


def test_ordinal_head_is_a_proper_distribution():
    """The three parameters induce an exact log-probability vector: normalized,
    with a controllable zero atom and a controllable location."""
    pytest.importorskip("torch")
    import torch as th
    from inv_single_ordinal_head import expand_order_logits

    n = 46
    for t in (-3.0, 0.0, 3.0):
        params = th.tensor([[t, 0.0, 0.0]])
        lp = expand_order_logits(params, n)
        assert lp.shape == (1, n)
        assert float(lp.exp().sum()) == pytest.approx(1.0, abs=1e-5)
        assert float(lp[0, 0].exp()) == pytest.approx(1 / (1 + pow(2.718281828, -t)), abs=1e-4)

    # the location parameter moves the mode of the POSITIVE part monotonically
    modes = []
    for m in (-4.0, -1.0, 1.0, 4.0):
        lp = expand_order_logits(th.tensor([[-6.0, m, -4.0]]), n)
        modes.append(int(lp[0, 1:].argmax()) + 1)
    assert modes == sorted(modes) and modes[0] < modes[-1], modes


def test_ordinal_head_can_represent_the_optimal_policy():
    """It must be able to place a sharp mode on the levels the DP actually
    uses — S=14 at `simple`, S=24 at `simple_k` — and to switch the order/
    don't-order decision. If it could not, no amount of training would help."""
    pytest.importorskip("torch")
    import torch as th
    from inv_single_ordinal_head import expand_order_logits, TAU_MIN, TAU_SCALE

    n = 46
    for target in (14, 24, 1, 45):
        # invert mu = 1 + (n-2)*sigmoid(m)
        frac = (target - 1) / (n - 2)
        frac = min(max(frac, 1e-6), 1 - 1e-6)
        m = th.log(th.tensor(frac / (1 - frac)))
        s = th.tensor(-8.0)                       # tau -> TAU_MIN, sharp
        lp = expand_order_logits(th.stack([th.tensor(-8.0), m, s])[None, :], n)
        assert int(lp[0, 1:].argmax()) + 1 == target, (target, int(lp[0, 1:].argmax()) + 1)
        assert float(lp[0, target].exp()) > 0.9, target
    # and the trigger can turn ordering off
    lp = expand_order_logits(th.tensor([[8.0, 0.0, 0.0]]), n)
    assert float(lp[0, 0].exp()) > 0.99


def test_the_rendered_vector_is_the_declared_vector():
    """Spec §7 (v0.9.32, from this campaign's issue #71; **corrected by
    v0.9.33**): what the gym renders, the IR declares — every component must
    DERIVE from a quantity the features list names, and a mode the IR names is
    constructible under that name.

    **Width equality is NOT the general instrument, and v0.9.33 says so.**
    `game2048` is the counterexample the proposal never surveyed: its `vec`,
    `grid` and `onehot` modes each declare exactly `{"ref": "board"}` and
    render it 4-, 4- and 20-wide. All three are correct — one declared feature
    under three encodings — and a width check marks the third non-compliant.
    Width cannot distinguish an expanding encoding (correct) from an undeclared
    prepended scalar (the defect); they have the same symptom.

    So the width half below is a **domain-local** invariant, valid here only
    because no `inv_single` mode re-encodes: every declared feature is an
    identity `ref` to a state variable or a scalar `derived`. If a re-encoding
    mode is ever added — one-hot inventory, say — this half must be dropped for
    that mode rather than the mode being called non-compliant. The
    constructibility half is general and stays."""
    from mdp_ir import load_ir
    from inv_single_gym import InvSingleEnv
    from inv_single_mdp import init_state
    from inv_single_scenarios import SCENARIOS

    ir = load_ir(SCHEMA)

    # A mode may be NARROWED by a declared width (the `_slt` modes carry the
    # lead-time law as a `dim: 3` feature, which only a 3-point law can fill),
    # so the scenario is chosen per mode rather than once: constructing a
    # narrowed mode outside its width is supposed to fail, and would be read
    # here as non-compliance.
    def _for(mode_name):
        return SCENARIOS["slt" if mode_name.endswith("_slt") else "lt"]

    declared_modes = {m.name for m in ir.gym.observation_modes}
    for name in declared_modes:
        scenario = _for(name)
        # the vector state variable's width comes from the MDP layer, not the
        # gym — reading it off the gym would make this check compare the gym to
        # itself. The lead-time law's width is the IR's own `dim` declaration
        # (v0.10.7), read off the feature below.
        state, _ = init_state(scenario=scenario, episode_seed=0)
        widths = {"pipeline": len(state.pipeline)}

        # the declared menu is the implemented menu — constructible by name
        env = InvSingleEnv(scenario, observation_mode=name)
        rendered = env.observation_space.shape[0]

        mode = next(m for m in ir.gym.observation_modes if m.name == name)
        # domain-local: no mode here re-encodes, so widths must agree exactly
        flat = 0
        for f in mode.features:
            f = f if isinstance(f, dict) else f.model_dump()
            ref = f.get("ref") or f.get("derived")
            flat += widths.get(ref) or f.get("dim") or 1
        assert flat == rendered, (
            f"{name}: declared features flatten to {flat}, gym renders "
            f"{rendered}. Either the IR does not describe what the gym builds, "
            f"or this mode re-encodes a declared quantity — in which case the "
            f"width half of this test does not apply to it (spec §7, v0.9.33)")


# --------------------------------------------------------------------------
# The composition probe's fitted IP-only rule (#E12)
# --------------------------------------------------------------------------

def test_the_fitted_ip_rule_is_a_function_of_inventory_position_alone():
    """The control must be restricted to IP, or it proves nothing.

    `fit_ip_only` reads a net's own behaviour; the resulting rule may depend on
    period and inventory position and on NOTHING else. If it could see the
    pipeline's composition it would not be a test of sufficiency.
    """
    pytest.importorskip("stable_baselines3")
    from inv_single_policy_probe import _ip_rule_lookup, fit_ip_only

    actions = {(3, 10): [5.0, 5.0, 7.0], (3, 12): [4.0], (4, 10): [6.0]}
    lookup = _ip_rule_lookup(*fit_ip_only(actions))

    # modal action per cell, not the mean
    assert lookup(3, 10) == 5.0
    # the same (period, IP) always answers the same thing, however it was reached
    assert lookup(3, 10) == lookup(3, 10)
    assert lookup(4, 10) == 6.0 and lookup(3, 12) == 4.0
    # the signature carries no pipeline argument at all
    import inspect
    assert list(inspect.signature(lookup).parameters) == ["period", "ip"]


def test_an_unvisited_inventory_position_extrapolates_instead_of_ordering_nothing():
    """The bug that inverted #E12's `lt` reading, kept as a gate.

    Answering 0 at an unvisited IP is not neutral: an unvisited IP is one the
    net's policy avoids, so ordering nothing there drives the episode further
    out and it runs away. At `lt` that put 41,671 cost on 7 of 8192 seeds and
    flipped the mean to +15.43 while the median moved -1. The fallback must
    extrapolate from the nearest visited position.
    """
    pytest.importorskip("stable_baselines3")
    from inv_single_policy_probe import _ip_rule_lookup, fit_ip_only

    actions = {(3, 10): [5.0], (3, 12): [4.0]}
    lookup = _ip_rule_lookup(*fit_ip_only(actions))

    assert lookup(3, -50) == 5.0, "far below the visited band -> nearest is IP 10"
    assert lookup(3, 99) == 4.0, "far above the visited band -> nearest is IP 12"
    # an unseen period falls back to the pooled table, still never to a silent 0
    assert lookup(27, 11) in (4.0, 5.0)
    # ... and the empty table is the only case that may answer 0
    assert _ip_rule_lookup({}, {})(0, 0) == 0.0


def test_the_composition_cap_subsamples_at_random_never_a_prefix():
    """The bug that inflated #E12's gradients to +2.79/+0.84, kept as a gate.

    `composition_probe` may cap how many compositions it queries per inventory
    position, but the candidate list is sorted by (inventory, pipeline), so its
    head is the low-inventory / heavy-pipeline states — exactly those with the
    largest lateness effect. Taking a prefix therefore biases the very quantity
    being measured, and every IP level had 390-1106 candidates, so the cap bound
    everywhere. Measured three ways at slt: prefix-60 +2.78, random-60 +1.12,
    all compositions +1.21.
    """
    import inspect

    pytest.importorskip("stable_baselines3")
    from inv_single_policy_probe import _build_arg_parser, composition_probe

    src = inspect.getsource(composition_probe)
    assert "comps[:max_comps]" not in src, (
        "a prefix of the sorted composition list is a biased subsample, not a cap"
    )
    assert "rng.permutation" in src, "the cap must subsample at random"
    assert "seed" in inspect.signature(composition_probe).parameters, (
        "a random subsample must be seeded or the probe stops reproducing"
    )
    # and the shipped default must be 'no cap at all'
    default = _build_arg_parser().parse_args(["--model-path", "x"]).comp_max_comps
    assert default == 0, f"default cap should be 0 (uncapped), got {default}"


def test_the_composition_figure_and_the_probe_read_the_same_states():
    """Figure and ledger must not drift.

    The figure exists to illustrate #E12's numbers, so it harvests visited
    states through the probe's own `_roll_visited` rather than a second
    implementation. If the plot script grew its own roll, the two could disagree
    -- which is exactly how the prefix bug survived: a figure drawn over all
    compositions disagreed with a table drawn over 60, and only then was one of
    them wrong.
    """
    import inspect

    import inv_single_plot_composition as plot

    src = inspect.getsource(plot)
    assert "from inv_single_policy_probe import _roll_visited" in src
    assert "def _roll_visited" not in src, "the figure must not re-implement the roll"


# --------------------------------------------------------------------------
# The information bottleneck and the weighted base-stock rule (#E13)
# --------------------------------------------------------------------------

def test_the_bottleneck_has_no_bias_and_pins_the_inventory_coefficient():
    """Both degeneracies must stay closed or the weights are not identifiable.

    `e = c*(inv + sum w_k pipe_k) + b` is the same statistic for any scale `c`
    and shift `b`, so without pinning them the recovered weights would be an
    arbitrary member of a family. Scale is killed by fixing the inventory
    coefficient to 1; shift by having no bias -- and a bias would be redundant
    anyway, since the MLP's first Linear carries one that absorbs it exactly.
    """
    pytest.importorskip("torch")
    import torch as th

    from inv_single_distill_bottleneck import Bottleneck

    stats = {"t_mu": 0.0, "t_sd": 1.0, "e_mu": 0.0, "e_sd": 1.0,
             "y_mu": 0.0, "y_sd": 1.0}
    net = Bottleneck(n_pipe=3, d=1, stats=stats)
    with th.no_grad():
        net.w.copy_(th.tensor([[0.5, 0.25, 0.125]]))
        obs = th.tensor([[7.0, 10.0, 4.0, 8.0, 16.0]])   # ttg, inv, p0, p1, p2
        e = float(net.embed(obs))
    # inventory enters with coefficient exactly 1, and nothing is added
    assert e == 10.0 + 0.5 * 4.0 + 0.25 * 8.0 + 0.125 * 16.0
    # a zero stock vector must embed to exactly zero -- that is "no bias"
    zero = th.zeros(1, 5)
    assert float(net.embed(zero)) == 0.0


def test_all_ones_weights_are_plain_inventory_position():
    """The null the whole experiment is measured against.

    With every weight at 1 the embedding must be textbook inventory position,
    so `basestock_opt` is the w=(1,1,1) member of the weighted family and the
    comparison in #E13 is like-for-like rather than two implementations.
    """
    pytest.importorskip("torch")
    import torch as th

    from inv_single_distill_bottleneck import Bottleneck

    stats = {"t_mu": 0.0, "t_sd": 1.0, "e_mu": 0.0, "e_sd": 1.0,
             "y_mu": 0.0, "y_sd": 1.0}
    net = Bottleneck(n_pipe=3, d=1, stats=stats)
    with th.no_grad():
        net.w.fill_(1.0)
        obs = th.tensor([[7.0, -3.0, 4.0, 8.0, 16.0]])
        assert float(net.embed(obs)) == -3.0 + 4.0 + 8.0 + 16.0


def test_the_weighted_rule_reduces_to_basestock_at_all_ones():
    """`weighted_basestock` must contain the standing bar, not approximate it."""
    import numpy as np

    from inv_single_benchmark_weighted import act_fn_for

    class _E:
        class _S:
            inventory, pipeline = 12.0, [3, 5, 7, 0]
        _state = _S()

    envs = [_E()]
    ip = 12.0 + 3 + 5 + 7                       # the dead 4th slot is not read
    got = float(act_fn_for((1.0, 1.0, 1.0), S=41, lt_max=3)(None, envs)[0, 0])
    assert got == max(0.0, 41 - ip)
    # and a discounted far slot orders MORE at the same inventory position
    more = float(act_fn_for((1.0, 1.0, 0.70), S=41, lt_max=3)(None, envs)[0, 0])
    assert more > got


# --------------------------------------------------------------------------
# lt_variance: lead-time variability at matched mean (RQ-4 generality target)
# --------------------------------------------------------------------------

def test_lt_variance_holds_the_mean_and_sweeps_the_variance():
    """The grid's whole claim is that only ONE thing varies.

    If E[L] moved with p, the order-up-to level would move too and any change
    in the recovered weights would be unattributable — mean and variance would
    be confounded. So the matched mean is the design, not a detail.
    """
    from inv_single_grids import GRIDS

    grid = GRIDS["lt_variance_k0"]
    means, variances = set(), []
    for _cid, sc in grid:
        lt = sc.leadtime
        E = sum(v * q for v, q in zip(lt.values, lt.probabilities))
        V = sum((v - E) ** 2 * q for v, q in zip(lt.values, lt.probabilities))
        means.add(round(E, 12))
        variances.append(round(V, 12))
    assert means == {2.0}, f"E[L] must be 2 at every cell, got {means}"
    assert min(variances) == 0.0, "the p=0 endpoint must be deterministic"
    assert max(variances) == pytest.approx(1.0), (
        "the p=1/2 endpoint must be the maximum-variance law on this support: "
        "all mass on {1,3}, which is Var=1 at E[L]=2")
    # p > 1/2 is not merely unused but inexpressible -- and normalisation would
    # not catch it, since [2/3, -1/3, 2/3] already sums to 1
    from inv_single_uncertainty import DiscreteLeadtime
    with pytest.raises(AssertionError, match="non-negative"):
        DiscreteLeadtime(values=[1, 2, 3], probabilities=[2/3, -1/3, 2/3])
    # one pipeline length across the family => the observation never changes shape
    assert len({sc.leadtime.max() for _, sc in grid}) == 1


def test_lt_variance_endpoints_reproduce_lt_and_slt():
    """p=0 and p=1/3 must be the two cells the campaign already measured.

    #E13's readings (far-slot weight 1.00 and ~0.70) are the sweep's anchors,
    and they only anchor it if the endpoint cells really are those scenarios.
    """
    from inv_single_grids import GRIDS
    from inv_single_scenarios import SCENARIOS

    cells = dict(GRIDS["lt_variance_k0"].cells)
    b, K = SCENARIOS["slt"].shortage_cost, 0.0
    lo = [sc for cid, sc in cells.items()
          if cid.startswith("leadtime_probs=[0.0,") and sc.shortage_cost == b
          and sc.order_cost_fixed == K]
    hi = [sc for cid, sc in cells.items()
          if "0.3333333333333333, 0.3333333333333333" in cid
          and sc.shortage_cost == b and sc.order_cost_fixed == K]
    assert len(lo) == 1 and len(hi) == 1

    # p = 0: degenerate at L = 2, i.e. `lt`'s dynamics in the slt family
    assert lo[0].leadtime.probabilities == [0.0, 1.0, 0.0]
    assert lo[0].leadtime.mean() == 2.0
    # p = 1/3: exactly the `slt` scenario's law
    assert hi[0].leadtime.probabilities == SCENARIOS["slt"].leadtime.probabilities
    assert hi[0].leadtime.values == SCENARIOS["slt"].leadtime.values


def test_cells_depadded_refuses_a_genuinely_stochastic_grid():
    """The helper collapses a cell to a deterministic twin at its mean.

    That is right for grid16, where a "stochastic" law is really padding with a
    zero-probability tail. Applied to lt_variance it would erase the variance
    the grid exists to sweep — every cell would become L=2 and the whole sweep
    would silently collapse to one point.
    """
    from inv_single_grids import cells_depadded

    assert len(cells_depadded("grid16")) == 16          # padded cells: fine
    for name in ("lt_variance_k0", "lt_variance_k20"):
        with pytest.raises(AssertionError, match="genuinely stochastic"):
            cells_depadded(name)


def test_the_two_lt_variance_grids_differ_only_in_K():
    """K=0 and K>0 are separate grids ON PURPOSE, not an axis.

    Base-stock is optimal at K=0 and (s,S) at K>0, so one generalist spanning
    both would have to switch policy STRUCTURE mid-family and a cross-K row
    comparison would be meaningless. Splitting them is the same rule that makes
    every scenario its own leaderboard — so the split must be real: everything
    except K has to match, cell for cell.
    """
    from inv_single_grids import GRIDS

    a, b = GRIDS["lt_variance_k0"], GRIDS["lt_variance_k20"]
    assert len(a.cells) == len(b.cells) == 45
    assert {sc.order_cost_fixed for _, sc in a} == {0.0}
    assert {sc.order_cost_fixed for _, sc in b} == {20.0}, (
        "K=20 is the campaign's fixed cost — simple_k and grid16 both use "
        "it, so this grid stays comparable with them")
    for (cid_a, sa), (cid_b, sb) in zip(a, b):
        assert cid_a.replace(",K=0", "") == cid_b.replace(",K=20", ""), (
            "cells must line up 1:1 apart from K")
        assert sa.shortage_cost == sb.shortage_cost
        assert sa.leadtime.probabilities == sb.leadtime.probabilities
        assert sa.leadtime.values == sb.leadtime.values


def test_lt_variance_sweeps_b_uniformly_over_one_to_nine():
    """b is an axis, so `uniform(1..9)` means the sampler draws cells equally.

    Reported per cell at evaluation, uniform in training — the two readings the
    grid has to serve at once.
    """
    from inv_single_grids import GRIDS

    grid = GRIDS["lt_variance_k0"]
    bs = sorted({sc.shortage_cost for _, sc in grid})
    assert bs == [float(x) for x in range(1, 10)]
    # critical fractile b/(b+h) with h=1 runs 0.50 -> 0.90
    h = {sc.holding_cost for _, sc in grid}.pop()
    assert min(b / (b + h) for b in bs) == pytest.approx(0.5)
    assert max(b / (b + h) for b in bs) == pytest.approx(0.9)
    # every (law, b) pair appears exactly once
    pairs = [(tuple(sc.leadtime.probabilities), sc.shortage_cost) for _, sc in grid]
    assert len(pairs) == len(set(pairs)) == 45


def test_vec_ctx_slt_shows_the_law_and_refuses_the_wrong_width():
    """The narrowing is a declared width, enforced — not a family rule.

    The IR declares the law feature `dim: 3` (the categorical candidate's
    support). The law itself is defined under EVERY candidate — a deterministic
    source is the one-point pmf — so what the gym refuses is a generator whose
    law cannot fill three components: the `lt` instance (1 point) and, new
    with this rule, `grid16`'s padded short-lead-time cells (2 points), which
    the v0.10.4 family check (`hasattr(..., "probabilities")`) let through.
    """
    from inv_single_gym import InvSingleEnv, LEADTIME_LAW_DIM, leadtime_law
    from inv_single_grids import GRIDS
    from inv_single_scenarios import SCENARIOS

    env = InvSingleEnv(SCENARIOS["slt"], observation_mode="vec_ctx_slt")
    obs, _ = env.reset(seed=0)
    n = LEADTIME_LAW_DIM
    # [ttg, inv, pipe..., law..., fractile, K]
    assert len(obs) == 2 + len(env._state.pipeline) + n + 2
    law = list(obs[2 + len(env._state.pipeline):][:n])
    assert law == pytest.approx(SCENARIOS["slt"].leadtime.probabilities)

    # the law must track the CELL, which is the whole point of the mode
    cells = dict(GRIDS["lt_variance_k0"].cells)
    for cid, sc in list(cells.items())[:1] + list(cells.items())[-1:]:
        e = InvSingleEnv(sc, observation_mode="vec_ctx_slt")
        o, _ = e.reset(seed=0)
        got = list(o[2 + len(e._state.pipeline):][:n])
        assert got == pytest.approx(sc.leadtime.probabilities), cid

    # one point: the deterministic instance
    assert leadtime_law(SCENARIOS["lt"].leadtime) == [1.0]
    with pytest.raises(AssertionError, match="3-wide feature"):
        InvSingleEnv(SCENARIOS["lt"], observation_mode="vec_ctx_slt")
    # two points: a padded grid16 cell (DiscreteLeadtime([lt, lt_max], [1, 0]))
    cid, sc = next((c, x) for c, x in GRIDS["grid16"].cells
                   if "leadtime_value=0" in c)
    assert len(leadtime_law(sc.leadtime)) == 2, cid
    with pytest.raises(AssertionError, match="3-wide feature"):
        InvSingleEnv(sc, observation_mode="vec_ip_ctx_slt")


def test_the_law_feature_reads_the_law_not_the_constant():
    """Upstream #77 (v0.10.7): the IR feature `leadtime.probs` folds from the
    SELECTED candidate at load, so under the deterministic candidate it is the
    point mass — never the `leadtime_probs` constant, which still has its base
    value there and is what this feature read until this change.

    Three things held together: the folded literal under `slt` is the
    scenario's law; under `lt` it is `[1.0]`; and the width the gym renders is
    the width the IR declares.
    """
    from mdp_ir import load_ir
    from inv_single_gym import LEADTIME_LAW_DIM, leadtime_law
    from inv_single_scenarios import SCENARIOS

    def _law_expr(instance):
        ir = load_ir(SCHEMA, instance=instance)
        mode = next(m for m in ir.gym.observation_modes if m.name == "vec_ctx_slt")
        feat = next(f for f in mode.features if f.derived == "leadtime_law")
        assert feat.dim == LEADTIME_LAW_DIM
        assert "leadtime_probs" not in feat.expr, feat.expr
        return eval(feat.expr)

    assert _law_expr("slt") == pytest.approx(SCENARIOS["slt"].leadtime.probabilities)
    assert _law_expr("slt") == pytest.approx(leadtime_law(SCENARIOS["slt"].leadtime))
    assert _law_expr("lt") == [1.0] == leadtime_law(SCENARIOS["lt"].leadtime)

    # the constant is still declared — the slt candidate's probabilities and
    # the lt_variance grid axis read it — but no observation feature does
    import json
    doc = json.load(open(SCHEMA))
    assert any(c["name"] == "leadtime_probs"
               for c in doc["mdp"]["scenario"]["constants"])
    for mode in doc["gym"]["observation_modes"]:
        for f in mode["features"]:
            assert f.get("expr") != "leadtime_probs", mode["name"]


def test_the_ip_context_mode_differs_from_the_full_one_only_in_the_pipeline():
    """`vec_ip_ctx_slt` is the RESTRICTED-INFORMATION control, so the
    restriction has to be the only difference.

    If the two modes differed in their conditioning as well — say the IP arm
    could not see the lead-time law — a gap between them would confound
    "the pipeline is worth something" with "one arm was handicapped". The
    contrast is only interpretable while the context is identical and the
    collapse is exact.
    """
    import numpy as np

    from inv_single_gym import InvSingleEnv, LEADTIME_LAW_DIM
    from inv_single_grids import GRIDS

    for cid, sc in list(GRIDS["lt_variance_k0"].cells)[::11]:
        full = InvSingleEnv(sc, observation_mode="vec_ctx_slt")
        rest = InvSingleEnv(sc, observation_mode="vec_ip_ctx_slt")
        of, _ = full.reset(seed=3)
        orr, _ = rest.reset(seed=3)
        n_pipe = len(full._state.pipeline)
        n_ctx = LEADTIME_LAW_DIM + 2                    # law + fractile + K

        # identical conditioning
        assert list(of[-n_ctx:]) == pytest.approx(list(orr[-n_ctx:])), cid
        # identical time-to-go
        assert of[0] == orr[0]
        # and the collapse is exactly the sum it claims to be
        assert orr[1] == pytest.approx(of[1] + float(np.sum(of[2:2 + n_pipe])))
        # widths differ by exactly what was collapsed away: `inventory` plus
        # the n_pipe slots (n_pipe + 1 entries) become the single IP scalar
        assert len(of) - len(orr) == n_pipe


def test_every_cli_offers_exactly_the_observation_modes_the_ir_declares():
    """The mode list is duplicated across the gym and three CLIs, and drift is
    silent until someone tries to use the missing one.

    `inv_single_select.py` sat at four modes while the IR declared five, so a
    `vec_ctx` run could never have been screened — invisible only because no
    grid arm had been run yet. A menu that is narrower than the IR makes a
    declared mode unreachable; one that is wider offers a mode nothing
    implements.
    """
    import pathlib
    import re

    from mdp_ir import load_ir

    here = pathlib.Path(__file__).resolve().parent
    declared = {m.name for m in load_ir(SCHEMA).gym.observation_modes}
    # scope the scan to each file's own list literal: a bare search for
    # "vec*" across the file also matches flag names like `vecnorm_clip_obs`
    sources = {
        "inv_single_ppo_train.py": r'"--observation_mode".*?choices=\[(.*?)\]',
        "inv_single_ppo_eval.py": r'OBS_MODES = \[(.*?)\]',
        "inv_single_select.py": r'"--observation_mode".*?choices=\[(.*?)\]',
    }
    for fname, pat in sources.items():
        text = (here / fname).read_text()
        m = re.search(pat, text, re.S)
        assert m, f"{fname}: could not find the observation-mode list"
        found = set(re.findall(r'"([^"]+)"', m.group(1)))
        assert declared <= found, (
            f"{fname} is missing declared mode(s) {sorted(declared - found)}; "
            f"a mode the IR declares must be reachable from every CLI")
        assert found <= declared, (
            f"{fname} offers undeclared mode(s) {sorted(found - declared)}")


# --------------------------------------------------------------------------
# Solve-level semantics (spec §8.6 v0.10.4, from this campaign's issue #78)
# --------------------------------------------------------------------------

def test_the_level_is_a_search_claim_and_the_knobs_only_give_distance():
    """`level >= L2 <=> a search ran on this target` — the knobs cannot say it.

    The spec used to define the level twice: once by that invariant, once by
    counting layers moved off a fixed `_L1_DERIVED`. The two agree only while
    the best configuration a campaign can state equals the table, so they
    diverge the moment a campaign learns anything. v0.10.4 settled it in favour
    of the invariant and demoted the table to L1's fallback source.

    So a configuration assembled from prior findings, with ONE run and no
    search on this target, is L1 however many layers it moves — and what the
    layers give is DISTANCE, reported separately.
    """
    pytest.importorskip("stable_baselines3")
    from inv_single_ppo_train import _build_arg_parser, derive_deviation_tags

    p = _build_arg_parser()

    # our lt_variance_k0 carried arms: four knobs, three layers, still L1
    carried = p.parse_args(["-s", "simple", "--policy", "ordinal", "--no-norm-obs",
                            "--no-normalize-advantage", "--ent-coef", "5.78e-06"])
    assert carried.level == "L1", "the level is authored; nothing here searched"
    assert derive_deviation_tags(carried) == "hp+gym+arch"

    # a run sitting at the derivation has no distance to report
    assert derive_deviation_tags(p.parse_args(["-s", "simple"])) == ""

    # L0 is a claim about the whole configuration, not a distance from rows
    # that are not in force there
    assert derive_deviation_tags(p.parse_args(["-s", "simple", "--level", "L0"])) == ""

    # and the function must not hand back anything level-shaped: returning
    # "L4(hp+gym+arch)" is the defect issue #78 removed
    import inspect
    src = inspect.getsource(derive_deviation_tags)
    assert "L{" not in src and 'f"L' not in src, (
        "derive_deviation_tags must return layer tags, never a level token")


def test_every_cli_parser_builds():
    """A parser that raises at import is invisible to a text-scanning test.

    `-s` was widened to accept a GRIDS key — the dispatch has always resolved
    one — but the eval script imported GRIDS only *inside* `_eval_grid`, so
    `_build_arg_parser` raised NameError while every existing test stayed green:
    they read these files as text and never build the parsers. mdp_tuning
    introspects them, so this failed only when a tuning study was launched.
    """
    pytest.importorskip("stable_baselines3")
    import inv_single_ppo_eval as ev
    import inv_single_ppo_train as tr
    import inv_single_select as sel

    from inv_single_grids import GRIDS
    from inv_single_scenarios import SCENARIOS

    for mod in (ev, tr, sel):
        p = mod._build_arg_parser()
        assert p is not None, mod.__name__

    # and a grid target must be reachable from the two scripts tuning drives
    for mod in (ev, tr):
        choices = next((a.choices for a in mod._build_arg_parser()._actions
                        if a.dest == "scenario_name" and a.choices), None)
        assert choices is not None, f"{mod.__name__}: no -s choices"
        assert set(GRIDS) <= set(choices), (
            f"{mod.__name__}: -s rejects grid target(s) "
            f"{sorted(set(GRIDS) - set(choices))}, which the dispatch resolves")
        assert set(SCENARIOS) <= set(choices)


def test_a_grid_target_resolves_through_either_flag():
    """`-s <grid>` and `-g <grid>` are the same request.

    The parser's choices list GRIDS, but `resolve_target` dispatched only on
    `--grid_name` and fell through to `SCENARIOS[...]`, so `-s lt_variance_k0`
    parsed and then raised KeyError. mdp_tuning drives `-s` and has no grid
    flag, so this was the difference between a generalist being tunable and not.
    """
    from inv_single_grids import GRIDS
    pytest.importorskip("stable_baselines3")
    from inv_single_ppo_train import _build_arg_parser, resolve_target

    p = _build_arg_parser()
    name = next(iter(GRIDS))
    via_s = resolve_target(p.parse_args(["-s", name]))
    via_g = resolve_target(p.parse_args(["-g", name]))
    for got in (via_s, via_g):
        assert callable(got), "a grid target must resolve to a sampler"
        assert got(7) is got(7), "the sampler must be pure"
    assert via_s(11).scenario_name == via_g(11).scenario_name


def test_the_far_pipeline_slot_is_unidentifiable_under_a_deterministic_leadtime():
    """#E20's identifiability gate.

    At Var(L)=0 the order enters slot 2 and shifts out before the next decision,
    so pipe_2 is identically zero at every decision epoch and its weight is
    unidentifiable — a fitted value for it is the initialisation, not a
    measurement. Reported without this check it reads as "the weight is 1 under
    a deterministic lead time, exactly as Karlin-Scarf says", which is a
    fabrication that happens to agree with theory.
    """
    import numpy as np

    from inv_single_grids import GRIDS
    from inv_single_gym import InvSingleEnv

    cells = dict(GRIDS["lt_variance_k0"].cells)
    det = next(sc for cid, sc in cells.items()
               if cid.startswith("leadtime_probs=[0.0,") and sc.shortage_cost == 9)
    var = next(sc for cid, sc in cells.items()
               if "0.5, 0.0, 0.5" in cid and sc.shortage_cost == 9)
    for sc, expect_zero in ((det, True), (var, False)):
        env = InvSingleEnv(sc, observation_mode="vec")
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(40):
            env.reset(seed=int(rng.integers(1 << 30)))
            for _t in range(sc.horizon):
                st = env._state
                rows.append(list(st.pipeline))
                ip = st.inventory + sum(st.pipeline)
                env.step(np.array([max(0, 41 - ip)], dtype=np.int64))
        sd2 = float(np.asarray(rows, dtype=float)[:, 2].std())
        if expect_zero:
            assert sd2 == 0.0, (
                "pipe_2 must be constant under a deterministic lead time; if it "
                "varies, the identifiability argument in #E20 needs revisiting")
        else:
            assert sd2 > 1.0, "pipe_2 must vary at maximum lead-time variance"

def test_cost_leadtime_overlaps_grid16_bit_exactly():
    """RQ-7 scores a grid16-trained policy on `cost_leadtime`, so the 12 cells
    the two grids share must be the SAME problem, and every cell must render the
    same observation width.

    Compared field by field with the uncertainty generators compared by `repr`:
    they define no `__eq__`, so two `DeterministicLeadtime(value=2)` are unequal
    by identity while being the same law. Comparing the objects directly reports
    all 12 cells as differing, which is an artefact and not a finding."""
    import dataclasses

    from inv_single_grids import GRIDS

    g16 = dict(GRIDS["grid16"].cells)
    cl = dict(GRIDS["cost_leadtime"].cells)
    assert len(g16) == 16 and len(cl) == 54

    def content(sc):
        d = dataclasses.replace(sc, scenario_name="", desc="")
        return {
            f.name: repr(getattr(d, f.name)) if f.name in ("leadtime", "demand")
            else getattr(d, f.name)
            for f in dataclasses.fields(d)
        }

    overlap = set(g16) & set(cl)
    assert len(overlap) == 12, f"expected 12 shared cells, got {len(overlap)}"
    for cid in sorted(overlap):
        assert content(g16[cid]) == content(cl[cid]), cid

    # one observation width across both grids, or a grid16-trained policy could
    # not be fed a cost_leadtime cell at all (spec 5.6)
    widths = {s.leadtime.max() + 1 for s in list(g16.values()) + list(cl.values())}
    assert widths == {3}, widths

    # the held-out strata separate the two interpolation axes
    def parse(cid):
        d = dict(kv.split("=") for kv in cid.split(","))
        return float(d["b"]), float(d["K"]), int(float(d["leadtime_value"]))

    held = set(cl) - set(g16)
    seen_b = {1.0, 4.0, 9.0}
    b_only = [c for c in held if parse(c)[0] not in seen_b and parse(c)[2] in (0, 2)]
    lt_only = [c for c in held if parse(c)[0] in seen_b and parse(c)[2] == 1]
    both = [c for c in held if parse(c)[0] not in seen_b and parse(c)[2] == 1]
    assert (len(held), len(b_only), len(lt_only), len(both)) == (42, 24, 6, 12)

    # b=19 stays OUT of the eval grid on purpose: it anchors the top of the
    # training range so b in 5..8 interpolate rather than extrapolate
    assert not any(parse(c)[0] == 19.0 for c in cl)

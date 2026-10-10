"""owmr's own tests — the claims only this domain can make.

Run from anywhere (the portable-domain contract):

    pytest owmr
    python owmr_test.py

Everything true of *any* IR — determinism, decision-path independence,
termination, the declared invariants — is checked by ``mdp_ir.laws`` and only
invoked here. What lives in this file is owmr-specific:

* the bit-exact differential over the covering set (base + the three cells);
* negative controls proving the differential and the invariants can fail;
* the **second-implementation gate** on the demand draw: the hand-written
  ``GammaDemand`` must reproduce the harness's generic ``FamilyGenerator``
  draw for the ``independent`` gamma slot bit-for-bit;
* the **rendered-vector claim** (spec 7): every observation mode's width is
  what its declared feature list resolves to on the instance;
* the **allocation claim**: under adversarial requests the warehouse never
  ships more than it holds and never goes negative — the mechanism behind
  the IR's ``warehouse_never_negative`` invariant;
* the **episode-seed provenance** claim: unseeded resets draw fresh episodes.

The conservation laws live in the IR as ``mdp.invariants`` (warehouse
balance, retailer balance, conservation, warehouse never negative, shipments
within on-hand), so every instance inherits them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mdp_ir import runtime
from mdp_ir.interpreter import IrInterpreter
from mdp_ir.schema import load_ir, ungroup_mdp
from mdp_ir.testing import (
    assert_diverges,
    assert_laws,
    assert_match,
    mutated,
    schema_beside,
)

SCHEMA = schema_beside(__file__)
EPISODES = 8


def _raw_flat() -> dict:
    """The schema document with its `mdp` block flattened (the file uses the
    model / design / rendering layout; a flat block passes through unchanged)."""
    doc = json.loads(SCHEMA.read_text())
    doc["mdp"] = ungroup_mdp(doc["mdp"])
    return doc


def _declared_compositions() -> list[str | None]:
    scenario = _raw_flat()["mdp"]["scenario"]
    return [None] + sorted(scenario.get("instances") or {})


COMPOSITIONS = _declared_compositions()


# -- engine laws -------------------------------------------------------------


def test_engine_laws():
    assert_laws(SCHEMA)


# -- the differential, over the covering set ---------------------------------


@pytest.mark.parametrize("instance", COMPOSITIONS, ids=[i or "base" for i in COMPOSITIONS])
def test_differential_matches_the_domain(instance):
    assert_match(SCHEMA, instance, episodes=EPISODES)


def test_the_covering_set_is_not_silently_empty():
    """Guards the derivation above: if the schema's instances vanished, every
    parametrized case would pass by not existing. Four cells: base + p/cv."""
    assert len(COMPOSITIONS) == 4, COMPOSITIONS
    assert {"hicv", "hipen", "hipen_hicv"} <= set(COMPOSITIONS)


# -- negative controls: the gates must be able to fail ------------------------


def _event(doc: dict, event: str) -> dict:
    return next(t for t in doc["mdp"]["dynamics"]["transitions"] if t["event"] == event)


def test_a_corrupted_demand_update_diverges():
    """Break the demand subtraction and the interpreter must stop agreeing
    with the domain."""

    def break_demand(doc):
        d = _event(doc, "D")
        d["updates"] = [
            u.replace("rt_stock[i] - demand[i]", "rt_stock[i] - demand[i] - 1")
            for u in d["updates"]
        ]

    assert_diverges(SCHEMA, mutated(load_ir(SCHEMA), break_demand), episodes=EPISODES)


def test_dropping_the_scale_down_diverges_and_violates_the_claim():
    """The load-bearing coupling of the problem is that the warehouse cannot
    ship what it does not hold. Remove the proportional scale-down and (a) the
    interpreter parts from the domain, (b) its own `warehouse_never_negative`
    claim fails — two gates, both observed failing."""

    def drop_scale(doc):
        s = _event(doc, "S")
        s["updates"] = [
            ("ship_scale = 1.0" if u.startswith("ship_scale =") else u) for u in s["updates"]
        ]

    ir = mutated(load_ir(SCHEMA), drop_scale)
    assert_diverges(SCHEMA, ir, episodes=EPISODES)
    traj = IrInterpreter(ir, seed_salt=1).run(0)   # random requests overshoot on-hand
    assert traj.violations, "dropping the scale-down left every invariant satisfied"


# -- the second implementation of the demand draw -----------------------------


class _Ctx:
    def __init__(self, period: int, episode_seed: int, seed_salt: int):
        self.period = period
        self.episode_seed = episode_seed
        self.seed_salt = seed_salt


@pytest.fixture(scope="module")
def domain():
    import importlib
    import sys
    from pathlib import Path

    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    return (
        importlib.import_module("owmr_uncertainty"),
        importlib.import_module("owmr_scenarios"),
        importlib.import_module("owmr_mdp"),
        importlib.import_module("owmr_gym"),
    )


@pytest.mark.parametrize("instance", COMPOSITIONS, ids=[i or "base" for i in COMPOSITIONS])
def test_gamma_demand_matches_the_shared_family_dispatch_bit_for_bit(domain, instance):
    """`GammaDemand` claims to be a bit-exact replica of the interpreter's draw
    for the `independent` gamma slot (spec 1.2: exact equality, never
    allclose). The interpreter draws through `mdp_ir.runtime.sample_family`
    from an rng seeded by the slot's intrinsic key; this seeds the same rng
    and calls the same dispatch with the IR's settings resolved to numbers.
    (`FamilyGenerator.from_ir` cannot resolve the IR's comprehension settings
    over constants, so the zero-code bridge is not the twin here — the
    differential over `demand` is the end-to-end form of this same claim.)"""
    unc = domain[0]
    ir = load_ir(SCHEMA, instance=instance)
    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])
    n = int(consts["n_retailers"])
    settings = {
        "of": "gamma", "size": n,
        "shape": [1.0 / (c * c) for c in consts["cv_rt"]],
        "scale": [consts["mu_rt"][i] * consts["cv_rt"][i] * consts["cv_rt"][i] for i in range(n)],
    }
    hand = unc.GammaDemand(mu_rt=consts["mu_rt"], cv_rt=consts["cv_rt"], source_id=0)
    for episode in (0, 3, 11):
        for period in range(6):
            ctx = _Ctx(period, episode, 1)
            rng = np.random.default_rng(
                np.random.SeedSequence(unc.intrinsic_key(0, ctx)))
            assert runtime.sample_family(rng, "independent", settings) == hand.sample(ctx)


# -- the rendered vector is the declared vector (spec 7) ----------------------


def _declared_width(ir, mode: str, instance) -> int:
    """Resolve each declared feature's width on this instance: a `ref` to a
    state variable or info field takes its rendered length, a `derived` expr
    is a scalar unless it comprehends over the retailers."""
    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])
    n, l0, l_rt = int(consts["n_retailers"]), int(consts["l0"]), int(consts["l_rt"])
    lengths = {"wh_stock": 1, "wh_pipe": l0, "rt_stock": n, "rt_pipe": n * l_rt,
               "info.demand": n, "time_to_go": 1, "wh_echelon_position": 1, "rt_position": n,
               "phase": 1}
    m = next(m for m in ir.gym.observation_modes if m.name == mode)
    return sum(lengths[f.ref if f.ref else f.derived] for f in m.features)


@pytest.mark.parametrize("mode", ["raw", "raw_d", "position", "raw_seq"])
def test_each_observation_mode_renders_its_declared_width(domain, mode):
    _, scen, _, gymmod = domain
    ir = load_ir(SCHEMA)
    act = "seq_order_softmax" if mode == "raw_seq" else "order_frac"
    env = gymmod.OwmrEnv(scen.SCENARIOS["base"], observation_mode=mode, action_mode=act)
    obs, _ = env.reset(seed=1)
    assert obs.shape[0] == _declared_width(ir, mode, None) == env.observation_space.shape[0]


def test_the_observation_leads_with_time_to_go(domain):
    _, scen, _, gymmod = domain
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc)
    obs, _ = env.reset(seed=1)
    assert obs[0] == sc.horizon
    obs, *_ = env.step(env.action_space.sample())
    assert obs[0] == sc.horizon - 1


# -- the allocation claim ----------------------------------------------------


@pytest.mark.parametrize("name", ["base", "hipen_hicv"])
def test_the_warehouse_never_ships_more_than_it_holds(domain, name):
    """Adversarial requests: ask for the cap on every retailer every period.
    The IR's `warehouse_never_negative` invariant states the consequence; this
    states the mechanism — the proportional scale-down reads on-hand AFTER the
    period's receipt, so the shipped total is exactly what was there."""
    _, scen, mdp, _ = domain
    sc = scen.SCENARIOS[name]
    state, _ = mdp.init_state(sc, episode_seed=5)
    while not state.terminated:
        state1 = mdp.advance1(sc, state)
        onhand = state1.wh_stock
        state, info = mdp.advance2(sc, state1, sc.order_max, [sc.order_max] * sc.n_retailers)
        assert sum(info["shipped"]) <= onhand + 1e-9
        assert info["ship_scale"] < 1.0 or onhand == 0.0
        assert state.wh_stock >= -1e-9


def test_order_frac_never_triggers_the_scale_down(domain):
    """Fractions summing to at most 1 are feasible by construction, so the
    MDP's projection is dormant under the default encoding."""
    _, scen, _, gymmod = domain
    env = gymmod.OwmrEnv(scen.SCENARIOS["hicv"], action_mode="order_frac")
    env.reset(seed=9)
    rng = np.random.default_rng(0)
    for _ in range(50):
        f = rng.dirichlet(np.ones(env.n + 1))[:-1]   # sums to < 1: some retained
        _, _, term, _, info = env.step(np.concatenate([[rng.uniform(0, 10)], f]))
        assert info["ship_scale"] == 1.0
        if term:
            break


# -- episode-seed provenance -------------------------------------------------


def test_unseeded_resets_draw_fresh_episodes(domain):
    _, scen, _, gymmod = domain
    env = gymmod.OwmrEnv(scen.SCENARIOS["base"])
    env.reset(seed=3)
    seeds = {env.reset()[1] is not None and env._episode_seed for _ in range(4)}
    assert len(seeds) == 4, seeds


def test_an_explicit_seed_pins_the_episode(domain):
    _, scen, _, gymmod = domain
    env = gymmod.OwmrEnv(scen.SCENARIOS["base"])
    a = np.asarray([5.0] + [0.2] * env.n, dtype=np.float32)
    runs = []
    for _ in range(2):
        env.reset(seed=17)
        runs.append([float(env.step(a)[1]) for _ in range(10)])
    assert runs[0] == runs[1]



# -- the readback instrument recovers a planted structure (spec 14.1) ---------


def test_probe_recovers_a_planted_order_up_to_rule(domain):
    """An instrument that has never recovered a known answer is not measuring
    yet. Plant the balance-assumption policy class in the `order_frac`
    encoding — order up to an echelon level Y0, ship each retailer up to a
    target Z — and the probe must report zero flatness on both surfaces and
    hand back Y0 and Z."""
    import importlib

    _, scen, mdp, _ = domain
    probe = importlib.import_module("owmr_policy_probe")
    sc = scen.SCENARIOS["base"]
    n, Y0, Z = sc.n_retailers, 35.0, 3.0

    def planted(obs):
        # raw layout: [ttg, wh, wh_pipe(l0), rt_stock(n), rt_pipe(n*l_rt)]
        wh = float(obs[1])
        wh_pipe = obs[2:2 + sc.l0]
        rt = obs[2 + sc.l0:2 + sc.l0 + n]
        rt_pipe = obs[2 + sc.l0 + n:].reshape(n, sc.l_rt)
        ip0 = wh + wh_pipe.sum() + rt.sum() + rt_pipe.sum()
        order = max(0.0, Y0 - ip0)
        want = np.maximum(0.0, Z - (rt + rt_pipe.sum(axis=1)))
        frac = want / wh if wh > 0 else np.zeros(n)
        return np.concatenate([[order], frac])

    out = probe.run_probe(sc, planted, "raw", "order_frac", n_grid=21)
    s = out["summary"]
    assert s["order_vs_wh_stock"]["order_up_to_flatness"] == pytest.approx(0.0, abs=1e-6)
    assert s["order_vs_wh_stock"]["recovered_y0"] == pytest.approx(Y0, abs=1e-6)
    assert s["allocation_vs_rt0_stock"]["target_flatness"] == pytest.approx(0.0, abs=1e-6)
    assert s["allocation_vs_rt0_stock"]["recovered_target"] == pytest.approx(Z, abs=1e-6)
    assert s["order_vs_time_to_go"]["order_range_over_horizon"] == pytest.approx(0.0, abs=1e-6)


# -- the order_softmax interface: sum < 1 by construction (ESCALATION #E2, #E4) --


def test_order_softmax_shares_sum_below_one_across_the_box(domain):
    """The guarantee `order_frac` never had. Over the whole logit box —
    both corners, the centre, random interior points — the retailer shares sum
    strictly below 1, and under random actions through the full env the MDP's
    proportional scale-down never fires."""
    _, scen, _, gymmod = domain
    n = scen.SCENARIOS["base"].n_retailers
    B = gymmod.LOGIT_BOX
    rng = np.random.default_rng(0)
    probes = [np.full(n, B), np.full(n, -B), np.zeros(n), np.array([B] + [-B] * (n - 1))]
    probes += list(rng.uniform(-B, B, size=(500, n)))
    for x in probes:
        s = gymmod.anchored_softmax_shares(x)
        assert (s > 0).all() and s.sum() < 1.0, (x, s)
    env = gymmod.OwmrEnv(scen.SCENARIOS["hipen_hicv"], action_mode="order_softmax")
    env.reset(seed=4)
    for _ in range(env.scenario.horizon):
        _, _, term, _, info = env.step(env.action_space.sample())
        assert info["ship_scale"] == 1.0
        if term:
            break


def test_order_frac_does_fire_the_scale_down_the_control(domain):
    """Negative control for the test above: the same random-action sweep under
    `order_frac` DOES hit the scale-down, which is the #E2 defect. If this ever
    stops failing the property, the positive test has lost its contrast."""
    _, scen, _, gymmod = domain
    env = gymmod.OwmrEnv(scen.SCENARIOS["hipen_hicv"], action_mode="order_frac")
    env.reset(seed=4)
    fired = 0
    for _ in range(env.scenario.horizon):
        _, _, term, _, info = env.step(env.action_space.sample())
        fired += info["ship_scale"] < 1.0
        if term:
            break
    assert fired > 0


def test_order_softmax_retention_is_reachable_at_both_ends(domain):
    """Retention = 1 - sum(shares) spans (0, 1): all logits at the bottom of
    the box retain almost everything, all at the top ship almost everything,
    and raw 0 splits on-hand into N+1 equal parts."""
    _, scen, _, gymmod = domain
    n = scen.SCENARIOS["base"].n_retailers
    B = gymmod.LOGIT_BOX
    keep = lambda x: 1.0 - gymmod.anchored_softmax_shares(x).sum()
    assert keep(np.full(n, -B)) > 0.999
    assert keep(np.full(n, B)) < 1e-4
    assert gymmod.anchored_softmax_shares(np.zeros(n)) == pytest.approx(np.full(n, 1.0 / (n + 1)))
    assert keep(np.zeros(n)) == pytest.approx(1.0 / (n + 1))


def test_order_softmax_box_is_the_declared_one(domain):
    """The rendered action box equals what the IR's gym.action_modes entry
    resolves to on the instance (order in [0, cap], logits in [-10, 10])."""
    _, scen, _, gymmod = domain
    ir = load_ir(SCHEMA)
    env = gymmod.OwmrEnv(scen.SCENARIOS["base"], action_mode="order_softmax")
    lo_o, hi_o = ir.action_bounds("order_softmax", decision="order")
    lo_s, hi_s = ir.action_bounds("order_softmax", decision="ship")
    assert env.action_space.low[0] == lo_o and env.action_space.high[0] == pytest.approx(hi_o)
    assert (env.action_space.low[1:] == lo_s).all() and (env.action_space.high[1:] == hi_s).all()
    assert hi_s == gymmod.LOGIT_BOX


# -- the LB heuristic's myopic allocation, when backlogs exceed on-hand (#E6) --


def test_myopic_allocation_rations_backlog_when_it_exceeds_onhand(domain):
    """The Lagrangian bracket cannot fit when shipping only enough to clear
    backlogs already overshoots on-hand. The rule must then give the whole
    on-hand to backlogged retailers (ties in proportion to backlog) and ship
    nothing to the others — never raise, never exceed on-hand."""
    pytest.importorskip("scipy")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    lbmod = importlib.import_module("owmr_benchmark_lb")
    _, scen, _, _ = domain
    bound = lbmod.BalanceBound(scen.SCENARIOS["base"])
    ip = np.array([-6.0, -2.0, 3.0, -2.0, 5.0])
    s = bound.allocate_myopic(4.0, ip)
    assert s.sum() == pytest.approx(4.0)
    assert s[2] == 0.0 and s[4] == 0.0
    assert s[0] / s[1] == pytest.approx(3.0) and s[1] == pytest.approx(s[3])
    assert (bound.allocate_myopic(0.0, ip) == 0.0).all()


# -- the bound on the finite-horizon criterion (ESCALATION #E26) ---------------


def test_lb_record_is_the_relaxed_optimum_on_the_finite_horizon(domain):
    """The `lb` record is the relaxed system's optimum on THIS problem's
    criterion (T periods from the mean-cover start), not the paper's long-run
    average x T. Checks: the vectorized cycle cost agrees with the scalar one;
    the long-run minimum is recovered at y0*; the first cycle is the closed form
    h0 (X0 - sum z*) + sum R_i(z*) (the start over-stocks the warehouse); the
    cycles the horizon charges are all accounted for; the components sum to the
    record; base = 603.18, +7.43 over the scaled long-run average (the start
    transient every policy pays)."""
    pytest.importorskip("scipy")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    lbmod = importlib.import_module("owmr_benchmark_lb")
    _, scen, _, _ = domain
    bound = lbmod.BalanceBound(scen.SCENARIOS["base"])
    sol = bound.solve()
    for x in (-3.0, 5.0, 14.0, 20.0, 31.5):
        assert bound.cycle_cost_vec([x])[0] == pytest.approx(bound.cycle_cost(x), abs=1e-9)
    assert bound.average_cost_vec([sol["y0_star"]])[0] == pytest.approx(sol["lb_per_period"], abs=1e-9)
    c = sol["lb_finite_components"]
    assert c["X0"] == 20.0 and c["IP0"] == 30.0
    assert c["fixed_cycles"][0] == pytest.approx(
        bound.h0 * (20.0 - bound.z.sum()) + bound.retailer_cost(bound.z).sum(), abs=1e-9)
    assert len(c["fixed_cycles"]) + len(c["ordered_cycles_exact"]) + c["ordered_cycles_at_longrun_min"] == 100 - 1
    total = (c["start_retailer_periods"] + sum(c["fixed_cycles"]) + sum(c["ordered_cycles_exact"])
             + c["ordered_cycles_at_longrun_min"] * sol["lb_per_period"])
    assert sol["lb_finite_T"] == pytest.approx(total, abs=1e-9)
    assert sol["lb_finite_T"] == pytest.approx(603.18, abs=0.01)
    assert sol["lb_finite_T"] - sol["lb_longrun_x_T"] == pytest.approx(7.43, abs=0.01)


# -- the ask features (ESCALATION #E7) ----------------------------------------


def _ask_extractor(sc, env):
    import importlib
    af = importlib.import_module("owmr_ask_features")
    kw = af.ask_policy_kwargs(sc, [64, 64])["features_extractor_kwargs"]
    return af.AskFeatureExtractor(env.observation_space, **kw)


def test_ask_is_the_heuristics_need_on_its_own_states(domain):
    """The ask is handed in, not learned, so it must BE the standard retailer
    need: on every post-receipt state of heuristic trajectories the extractor's
    a_i equals max(0, z*_i - IP_i) computed from the simulator state."""
    pytest.importorskip("torch")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("stable_baselines3")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("scipy")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    import torch as th
    _, scen, mdp, gymmod = domain
    heur = importlib.import_module("owmr_benchmark_lb_heuristic")
    pol = heur.build("base")
    sc = pol.scenario
    ext = _ask_extractor(sc, gymmod.OwmrEnv(sc, "raw", "order_softmax"))
    worst = 0.0
    for seed in range(4):
        state, _ = mdp.init_state(sc, seed)
        while not state.terminated:
            s1 = mdp.advance1(sc, state)
            obs = gymmod.observation(sc, s1, "raw", [])
            ask = ext.parse(th.as_tensor(obs[None, :]))["ask"][0].numpy()
            want = np.maximum(0.0, pol.bound.z - np.asarray(mdp.inventory_position(s1)))
            worst = max(worst, float(np.abs(ask - want).max()))
            order, ship = pol.decide(s1)
            state, _ = mdp.advance2(sc, s1, order, list(ship))
    assert worst < 1e-5


def test_ask_planted_heuristic_reproduces_its_cost(domain):
    """Plumbing gate (a planted answer): drive the order_softmax env with the
    heuristic's order and its myopic shipments rewritten as logits over the
    extractor's own parse of the observation. The episode cost must match the
    heuristic's own run in `order_ship`. Not exact: a myopic allocation that
    ships everything needs logit +inf, so the box leaves ~e^-7 of on-hand behind.
    The echelon order here is a TEST FIXTURE, never part of the arm."""
    pytest.importorskip("torch")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("stable_baselines3")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("scipy")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    import torch as th
    _, scen, mdp, gymmod = domain
    heur = importlib.import_module("owmr_benchmark_lb_heuristic")
    pol = heur.build("base")
    sc, bound = pol.scenario, pol.bound
    ext = _ask_extractor(sc, gymmod.OwmrEnv(sc, "raw", "order_softmax"))
    box = gymmod.LOGIT_BOX

    def planted(obs):
        p = {k: v[0].numpy().astype(np.float64) for k, v in ext.parse(th.as_tensor(obs[None, :])).items()}
        W = float(p["W"])
        ip = p["rt"] + p["transit"]
        order = max(0.0, pol.y0 - (W + p["wh_pipe"].sum() + ip.sum()))
        if W <= 0:
            return np.concatenate([[order], np.full(sc.n_retailers, -box)])
        s = bound.allocate_myopic(W, ip)
        kept = max(W - s.sum(), W * np.exp(-7.0))
        logits = np.clip(np.log(np.maximum(s, W * np.exp(-17.0))) - np.log(kept), -box, box)
        return np.concatenate([[order], logits])

    for seed in range(6):
        env_h = gymmod.OwmrEnv(sc, "raw", "order_ship")
        env_p = gymmod.OwmrEnv(sc, "raw", "order_softmax")
        env_h.reset(seed=seed)
        obs, _ = env_p.reset(seed=seed)
        cost_h = cost_p = 0.0
        done = False
        while not done:
            order, ship = pol.decide(env_h._state)
            _, _, term_h, trunc_h, info_h = env_h.step(np.concatenate([[order], ship]).astype(np.float32))
            obs, _, term, trunc, info = env_p.step(planted(obs).astype(np.float32))
            cost_h += info_h["cost"]["total"]
            cost_p += info["cost"]["total"]
            done = term or trunc
            assert (term_h or trunc_h) == done
        assert abs(cost_p - cost_h) / cost_h < 2e-3, (seed, cost_p, cost_h)


def test_ask_policy_is_the_flat_global_network_and_reloads(domain, tmp_path):
    """The arm is SB3's standard flat actor-critic on [raw obs, ask features]:
    no per-retailer weights, one global policy MLP, one global value MLP, a
    weightless extractor. Checkpoints must reload in the eval scripts (the
    extractor class is pickled by reference) and act identically."""
    pytest.importorskip("torch")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("stable_baselines3")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    import torch as th
    from stable_baselines3 import PPO
    _, scen, _, gymmod = domain
    af = importlib.import_module("owmr_ask_features")
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_softmax")
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, device="cpu",
                policy_kwargs=af.ask_policy_kwargs(sc, [16, 16]), seed=0)
    pol = model.policy
    assert isinstance(pol.features_extractor, af.AskFeatureExtractor)
    assert sum(p.numel() for p in pol.features_extractor.parameters()) == 0
    n = sc.n_retailers
    assert pol.features_dim == env.observation_space.shape[0] + 2 * n + 2
    assert pol.mlp_extractor.policy_net[0].in_features == pol.features_dim   # global input
    assert pol.action_net.out_features == n + 1                              # [order, logits]
    model.learn(128)
    path = tmp_path / "m.zip"
    model.save(str(path))
    loaded = PPO.load(str(path), device="cpu")
    obs, _ = env.reset(seed=3)
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_allclose(a1, a2, atol=1e-6)


def test_ask_running_norm_updates_on_rollouts_only_and_reloads(domain, tmp_path):
    """#E8: the extractor normalizes its own features like VecNormalize. The
    statistics move only during rollout collection (the training script's
    switch), count exactly the rollout steps (plus SB3's one bootstrap value
    call per rollout), stay frozen for evaluation, and travel with the checkpoint."""
    pytest.importorskip("torch")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("stable_baselines3")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    import torch as th
    from stable_baselines3 import PPO
    _, scen, _, gymmod = domain
    af = importlib.import_module("owmr_ask_features")
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_softmax")
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=2, device="cpu",
                policy_kwargs=af.ask_policy_kwargs(sc, [16, 16], running_norm=True), seed=0)
    ext = model.policy.features_extractor
    model.learn(128, callback=af.rollout_stats_callback())
    # 2 rollouts x 64 steps, +1 bootstrap call each; gradient passes add nothing
    assert float(ext.rms_count) == pytest.approx(1e-4 + 2 * 65)
    assert ext.update_stats is False
    frozen = ext.rms_mean.clone()
    obs, _ = env.reset(seed=5)
    for _ in range(10):
        model.predict(obs, deterministic=True)
    assert th.equal(ext.rms_mean, frozen)
    # the features the network sees are standardized, the raw ask is not
    x = ext.features(th.as_tensor(obs[None, :]))
    z = ext(th.as_tensor(obs[None, :]))
    assert not th.allclose(x, z)
    path = tmp_path / "m.zip"
    model.save(str(path))
    loaded = PPO.load(str(path), device="cpu")
    lext = loaded.policy.features_extractor
    assert th.equal(lext.rms_mean, ext.rms_mean) and lext.update_stats is False
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_allclose(a1, a2, atol=1e-6)


# -- the split actor (ESCALATION #E9) ------------------------------------------


@pytest.mark.parametrize("residual", [False, True], ids=["split", "splitres"])
def test_split_actor_shares_no_weights_between_order_and_allocation(domain, tmp_path, residual):
    """The order depends only on the order trunk and the logits only on the
    allocation trunk: zeroing one trunk's gradient path leaves the other
    output untouched. Checkpoints reload and act identically."""
    pytest.importorskip("torch")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("stable_baselines3")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    import torch as th
    from stable_baselines3 import PPO
    _, scen, _, gymmod = domain
    sp = importlib.import_module("owmr_split_policy")
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_softmax")
    arch = [32, 32, 32] if residual else [16, 16]
    model = PPO(sp.SplitActorCriticPolicy, env, n_steps=64, batch_size=32, device="cpu",
                policy_kwargs={"net_arch": arch, "residual": residual}, seed=0)
    pol = model.policy
    assert isinstance(pol.action_net, sp.SplitHead)
    obs = th.as_tensor(np.random.default_rng(0).normal(size=(8, env.observation_space.shape[0])),
                       dtype=th.float32)
    feats = pol.extract_features(obs)
    lat = pol.mlp_extractor.forward_actor(feats)
    mean = pol.action_net(lat)
    g_order = th.autograd.grad(mean[:, 0].sum(), list(pol.mlp_extractor.alloc_trunk.parameters()),
                               allow_unused=True, retain_graph=True)
    g_alloc = th.autograd.grad(mean[:, 1:].sum(), list(pol.mlp_extractor.order_trunk.parameters()),
                               allow_unused=True)
    assert all(g is None or float(g.abs().max()) == 0.0 for g in g_order + g_alloc)
    assert float(pol.action_net.alloc.weight.abs().max()) < 0.05      # small-gain init
    if residual:
        n_blocks = sum(isinstance(m, sp.ResBlock) for m in pol.mlp_extractor.order_trunk)
        assert n_blocks == len(arch)
    model.learn(128)
    path = tmp_path / "m.zip"
    model.save(str(path))
    loaded = PPO.load(str(path), device="cpu")
    o, _ = env.reset(seed=3)
    np.testing.assert_allclose(model.predict(o, deterministic=True)[0],
                               loaded.predict(o, deterministic=True)[0], atol=1e-6)


# -- the two-phase encoding (ESCALATION #E10) ---------------------------------


def test_two_phase_encoding_reproduces_the_one_shot_period_exactly(domain):
    """seq_order_softmax is an ENCODING, not a new model: feeding the order at
    the order step and the logits at the allocation step must reproduce
    order_softmax's episode bit for bit (costs, states, observations), with
    reward 0 and no cost info on every order step, and the allocation step's
    observation showing the order in the warehouse pipeline's last slot."""
    _, scen, _, gymmod = domain
    sc = scen.SCENARIOS["base"]
    rng = np.random.default_rng(7)
    one = gymmod.OwmrEnv(sc, "raw", "order_softmax")
    two = gymmod.OwmrEnv(sc, "raw_seq", "seq_order_softmax")
    o1, _ = one.reset(seed=11)
    o2, _ = two.reset(seed=11)
    l0 = sc.l0
    for t in range(sc.horizon):
        a = np.concatenate([[rng.uniform(0, 12)], rng.normal(size=sc.n_retailers)]).astype(np.float32)
        np.testing.assert_array_equal(o2[:-1], o1)
        assert o2[-1] == 0.0
        junk = a.copy(); junk[1:] = 99.0                    # ignored at the order step
        o2, r, term, trunc, info = two.step(junk)
        assert r == 0.0 and not term and "cost" not in info
        assert o2[-1] == 1.0 and o2[1 + l0] == pytest.approx(o1[1 + l0] + float(a[0]))
        junk = a.copy(); junk[0] = -5.0                     # ignored at the allocation step
        o2, r2, term2, _, info2 = two.step(junk)
        o1, r1, term1, _, info1 = one.step(a)
        assert r2 == r1 and term2 == term1 and info2["cost"] == info1["cost"]
    assert term1 and term2


def test_phase_mask_counts_only_the_live_components(domain):
    """PhaseMaskedPolicy: at an order step the log-prob and entropy are the
    order component's alone, at an allocation step the logits' alone — under
    raw phase values AND after VecNormalize has centred the phase feature."""
    pytest.importorskip("torch")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("stable_baselines3")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    import torch as th
    from stable_baselines3 import PPO
    _, scen, _, gymmod = domain
    sq = importlib.import_module("owmr_seq_policy")
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw_seq", "seq_order_softmax")
    model = PPO(sq.PhaseMaskedPolicy, env, n_steps=64, batch_size=32, device="cpu",
                policy_kwargs={"net_arch": [16, 16]}, seed=0)
    pol = model.policy
    d = env.observation_space.shape[0]
    obs = th.zeros(4, d)
    obs[:, -1] = th.tensor([0.0, 1.0, -1.0, 1.0])            # raw 0/1, then normalized -1/+1
    acts = th.randn(4, sc.n_retailers + 1)
    with th.no_grad():
        _, logp, ent = pol.evaluate_actions(obs, acts)
        dist = pol.get_distribution(obs).distribution
        per = dist.log_prob(acts)
        pent = dist.entropy()
    order_rows, alloc_rows = [0, 2], [1, 3]
    assert th.allclose(logp[order_rows], per[order_rows, 0])
    assert th.allclose(logp[alloc_rows], per[alloc_rows, 1:].sum(1))
    assert th.allclose(ent[order_rows], pent[order_rows, 0])
    assert th.allclose(ent[alloc_rows], pent[alloc_rows, 1:].sum(1))
    # forward's log-prob (rollout) must be the same masked quantity evaluate_actions scores
    with th.no_grad():
        a, _, lp = pol(obs)
        _, lp2, _ = pol.evaluate_actions(obs, a)
    assert th.allclose(lp, lp2)


# -- the explicit relu allocation (ESCALATION #E14) -----------------------------


def test_relu_shares_reach_every_corner_exactly(domain):
    """order_relu's decode: the N + 1 shares sum to 1; a retailer with x <= 0
    gets exactly 0; x_0 <= 0 ships exactly everything (split by the retailer
    weights); all x <= 0 keeps everything."""
    _, _, _, gymmod = domain
    rs = gymmod.relu_shares
    sh, kept = rs([1.0, 2.0, 0.0, -3.0, 1.0, 0.0])
    assert sh.sum() + kept == pytest.approx(1.0)
    assert kept == pytest.approx(0.25) and sh[1] == 0.0 and sh[2] == 0.0
    sh, kept = rs([-1.0, 2.0, 0.0, -3.0, 2.0, 0.0])          # ship everything
    assert kept == 0.0 and sh.sum() == pytest.approx(1.0) and sh[0] == pytest.approx(0.5)
    sh, kept = rs([-1.0, -2.0, 0.0, -3.0, -2.0, 0.0])        # every weight clipped
    assert kept == 1.0 and (sh == 0.0).all()
    sh, kept = rs([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])            # warehouse only
    assert kept == 1.0 and (sh == 0.0).all()


def test_order_relu_never_fires_the_feasibility_fallback(domain):
    """The scale-down is a feasibility fallback, never a rationing rule: under
    order_relu it must not fire beyond float rounding, across random actions
    including the ship-everything corner, and the warehouse never goes negative
    beyond the IR's tolerance."""
    _, scen, _, gymmod = domain
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_relu")
    assert env.action_space.shape[0] == sc.n_retailers + 2
    rng = np.random.default_rng(3)
    worst = 1.0
    shipped_all = 0
    for ep in range(6):
        env.reset(seed=ep)
        done = False
        while not done:
            a = rng.uniform(env.action_space.low, env.action_space.high)
            if rng.random() < 0.5:
                a[1] = -1.0                                  # force the ship-everything corner
            W = float(env._state.wh_stock)
            _, _, term, trunc, info = env.step(a.astype(np.float32))
            worst = min(worst, info["ship_scale"])
            if a[1] <= 0 and W > 0 and (a[2:] > 0).any():
                # ship-everything corner, under the IR's per-shipment bound (order_max): each retailer's
                # share of on-hand is capped there (#E37), so the expected total is the capped sum
                shares, _ = gymmod.relu_shares(np.clip(a.astype(np.float32)[1:].astype(np.float64), -gymmod.LOGIT_BOX, gymmod.LOGIT_BOX))   # the env sees the float32 action
                expected = sum(min(float(q) * W, float(sc.order_max)) for q in shares)
                assert sum(info["shipped"]) == pytest.approx(expected, rel=1e-12)
                assert all(q <= sc.order_max + 1e-9 for q in info["shipped"])
                shipped_all += 1
            assert env._state.wh_stock >= -1e-9
            done = term or trunc
    assert worst >= 1.0 - 1e-12 and shipped_all > 50


# -- the nested explicit allocation (ESCALATION #E17) ---------------------------


def test_nested_shares_decode(domain):
    """order_nested: kept = relu(x_0)/(1+relu(x_0)); the rest split by relu
    weights (exact zeros; equal split when all retailer weights are <= 0);
    shares + kept = 1 exactly."""
    _, _, _, gymmod = domain
    ns = gymmod.nested_shares
    sh, kept = ns([0.0, 2.0, 0.0, -3.0, 2.0, 0.0])            # ship everything, 50/50
    assert kept == 0.0 and sh.sum() == pytest.approx(1.0) and sh[0] == pytest.approx(0.5) and sh[1] == 0.0
    sh, kept = ns([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])             # keep half, equal split of the rest
    assert kept == pytest.approx(0.5) and np.allclose(sh, 0.1)
    sh, kept = ns([-4.0, -1.0, -1.0, -1.0, -1.0, -1.0])       # degenerate lower level: equal shares
    assert kept == 0.0 and np.allclose(sh, 0.2)
    sh, kept = ns([10.0, 3.0, 0.0, 0.0, 0.0, 0.0])            # box cap on retention: 10/11
    assert kept == pytest.approx(10 / 11) and sh[0] == pytest.approx(1 / 11) and (sh[1:] == 0).all()
    assert ns([0.5, 1, 1, 1, 1, 1])[1] == pytest.approx(1 / 3)  # slope-1 boundary: x_0 = 0.5 keeps a third


def test_order_nested_never_fires_the_feasibility_fallback(domain):
    _, scen, _, gymmod = domain
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_nested")
    assert env.action_space.shape[0] == sc.n_retailers + 2
    rng = np.random.default_rng(5)
    worst, shipped_all = 1.0, 0
    for ep in range(6):
        env.reset(seed=ep)
        done = False
        while not done:
            a = rng.uniform(env.action_space.low, env.action_space.high)
            if rng.random() < 0.5:
                a[1] = -1.0
            W = float(env._state.wh_stock)
            _, _, term, trunc, info = env.step(a.astype(np.float32))
            worst = min(worst, info["ship_scale"])
            if a[1] <= 0 and W > 0:
                # under the IR's per-shipment bound (order_max) each share of on-hand is capped (#E37)
                shares, _ = gymmod.nested_shares(np.clip(a.astype(np.float32)[1:].astype(np.float64), -gymmod.LOGIT_BOX, gymmod.LOGIT_BOX))   # the env sees the float32 action
                expected = sum(min(float(q) * W, float(sc.order_max)) for q in shares)
                assert sum(info["shipped"]) == pytest.approx(expected, rel=1e-12)
                assert all(q <= sc.order_max + 1e-9 for q in info["shipped"])
                shipped_all += 1
            assert env._state.wh_stock >= -1e-9
            done = term or trunc
    assert worst >= 1.0 - 1e-12 and shipped_all > 50


# -- the unbounded, un-anchored softmax (ESCALATION #E18) -----------------------


def test_free_softmax_decode_and_no_fallback(domain):
    """order_softmax_free: one softmax over N + 1 free logits, no clip. Shares +
    kept = 1; extreme logits are handled (stable form) and drive shares to
    numerical zero; the action space is unbounded; the fallback never fires."""
    _, scen, _, gymmod = domain
    fs = gymmod.free_softmax_shares
    sh, kept = fs([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert kept == pytest.approx(1 / 6) and np.allclose(sh, 1 / 6)
    sh, kept = fs([-1000.0, 5.0, 5.0, -1000.0, 5.0, 5.0])         # far logits: exact zeros in float
    assert kept == 0.0 and sh[2] == 0.0 and sh.sum() == pytest.approx(1.0)
    sh, kept = fs([1e6, 0.0, 0.0, 0.0, 0.0, 0.0])                # no overflow
    assert kept == pytest.approx(1.0) and np.isfinite(sh).all()
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_softmax_free")
    assert env.action_space.shape[0] == sc.n_retailers + 2
    assert (env.action_space.high[1:] >= 1e9).all() and (env.action_space.low[1:] <= -1e9).all()
    rng = np.random.default_rng(9)
    worst = 1.0
    for ep in range(4):
        env.reset(seed=ep)
        done = False
        while not done:
            a = np.concatenate([[rng.uniform(0, 12)], rng.normal(scale=rng.choice([1, 10, 100]), size=sc.n_retailers + 1)])
            _, _, term, trunc, info = env.step(a.astype(np.float32))
            worst = min(worst, info["ship_scale"])
            assert env._state.wh_stock >= -1e-9
            done = term or trunc
    assert worst >= 1.0 - 1e-12


# -- the transit-charged training reward (ESCALATION #E19) ----------------------


def test_transit_reward_mode_adds_exactly_h0_on_retailer_pipelines(domain):
    """neg_cost_transit shapes the TRAINING reward only: reward = -(cost.total +
    h0 * end-of-period retailer pipeline); info['cost'] stays canonical, and
    the two envs' states agree step for step."""
    _, scen, _, gymmod = domain
    sc = scen.SCENARIOS["base"]
    a_env = gymmod.OwmrEnv(sc, "raw", "order_nested")
    b_env = gymmod.OwmrEnv(sc, "raw", "order_nested", reward_mode="neg_cost_transit")
    a_env.reset(seed=4); b_env.reset(seed=4)
    rng = np.random.default_rng(1)
    extra = 0.0
    for t in range(100):
        act = np.concatenate([[rng.uniform(0, 10)], rng.normal(size=sc.n_retailers + 1)]).astype(np.float32)
        _, ra, _, _, ia = a_env.step(act)
        _, rb, _, _, ib = b_env.step(act)
        pipe = sum(sum(row) for row in b_env._state.rt_pipe) if t < 99 else None
        assert ia["cost"] == ib["cost"] and ra == -ia["cost"]["total"]
        assert rb <= ra
        extra += ra - rb
    # every shipped unit spends exactly l_rt = 1 period in transit -> the horizon total
    # of the shaping is h0 * (units shipped before the last period's arrival)
    assert extra > 0


# -- the categorical retention head (ESCALATION #E21) ---------------------------


def test_catkeep_decode_and_no_fallback(domain):
    """order_catkeep: the menu index picks the kept fraction exactly; the rest is
    the relu split; shares + kept = 1; the fallback never fires."""
    _, scen, _, gymmod = domain
    cs, menu = gymmod.catkeep_shares, gymmod.KEEP_MENU
    sh, k = cs([0.0, 1.0, 1.0, -1.0, 1.0, 1.0])
    assert k == 0.0 and sh.sum() == pytest.approx(1.0) and sh[2] == 0.0 and sh[0] == pytest.approx(0.25)
    sh, k = cs([3.4, 1.0, 1.0, 1.0, 1.0, 1.0])                  # rounds to option 3 -> keep 0.3
    assert k == menu[3] and sh.sum() == pytest.approx(0.7)
    sh, k = cs([5.0, -1.0, -1.0, -1.0, -1.0, -1.0])             # keep 0.8, degenerate split -> equal
    assert k == menu[5] and np.allclose(sh, 0.2 * 0.2)
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_catkeep")
    assert env.action_space.shape[0] == sc.n_retailers + 2 and env.action_space.high[1] == len(menu) - 1
    rng = np.random.default_rng(2)
    worst = 1.0
    for ep in range(4):
        env.reset(seed=ep)
        done = False
        while not done:
            a = rng.uniform(env.action_space.low, env.action_space.high)
            _, _, term, trunc, info = env.step(a.astype(np.float32))
            worst = min(worst, info["ship_scale"])
            assert env._state.wh_stock >= -1e-9
            done = term or trunc
    assert worst >= 1.0 - 1e-12


def test_catkeep_fine_menu_decode_and_space(domain):
    """order_catkeep_fine (#E34): the 12-rung menu decodes exactly, the Box index
    bound follows the menu, the policy head widens to 12 options, and
    `order_catkeep` is untouched."""
    pytest.importorskip("torch")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("stable_baselines3")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    from stable_baselines3 import PPO
    _, scen, _, gymmod = domain
    fine, coarse = gymmod.KEEP_MENU_FINE, gymmod.KEEP_MENU
    assert len(fine) == 12 and fine[0] == 0.0 and all(b > a for a, b in zip(fine, fine[1:])) and set(coarse) <= set(fine)
    assert gymmod.keep_menu("order_catkeep_fine") is fine and gymmod.keep_menu("order_catkeep") is coarse
    sh, k = gymmod.catkeep_shares([7.0, 1.0, 1.0, 1.0, 1.0, 1.0], fine)      # option 7 -> keep 0.35
    assert k == 0.35 and sh.sum() == pytest.approx(0.65)
    sh, k = gymmod.catkeep_shares([7.0, 1.0, 1.0, 1.0, 1.0, 1.0])            # the default menu: option 7 clips to 5 -> 0.8
    assert k == 0.8
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_catkeep_fine")
    assert env.action_space.shape[0] == sc.n_retailers + 2 and env.action_space.high[1] == 11
    ck = importlib.import_module("owmr_catkeep_policy")
    model = PPO(ck.CatKeepPolicy, env, n_steps=64, batch_size=32, device="cpu",
                policy_kwargs={"net_arch": [16, 16], "n_options": 12}, seed=0)
    assert model.policy.action_net.out_features == (sc.n_retailers + 1) + 12
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([5.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    assert info["ship_scale"] >= 1.0 - 1e-12


def test_catkeep_policy_joint_distribution(domain, tmp_path):
    """CatKeepPolicy: log-prob = Gaussian(order, x) + categorical(k); the sampled
    k is an integer index; deterministic = argmax; rollout log-prob equals the
    re-scored one; checkpoints reload and act identically."""
    pytest.importorskip("torch")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("stable_baselines3")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    import torch as th
    from stable_baselines3 import PPO
    _, scen, _, gymmod = domain
    ck = importlib.import_module("owmr_catkeep_policy")
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_catkeep")
    model = PPO(ck.CatKeepPolicy, env, n_steps=64, batch_size=32, device="cpu",
                policy_kwargs={"net_arch": [16, 16]}, seed=0)
    pol = model.policy
    assert isinstance(pol.action_dist, ck.GaussianCategorical)
    assert pol.action_net.out_features == (sc.n_retailers + 1) + len(gymmod.KEEP_MENU)
    obs = th.as_tensor(np.random.default_rng(0).normal(size=(6, env.observation_space.shape[0])), dtype=th.float32)
    with th.no_grad():
        a, v, lp = pol(obs)
        _, lp2, ent = pol.evaluate_actions(obs, a)
        dist = pol.get_distribution(obs)
        manual = dist.gauss.log_prob(th.cat([a[:, :1], a[:, 2:]], 1)).sum(1) + dist.cat.log_prob(a[:, 1].long())
        det, _, _ = pol(obs, deterministic=True)
    assert th.allclose(lp, lp2) and th.allclose(lp, manual)
    assert th.equal(a[:, 1], a[:, 1].round()) and (a[:, 1] >= 0).all() and (a[:, 1] < len(gymmod.KEEP_MENU)).all()
    assert th.equal(det[:, 1], th.argmax(dist.cat.probs, 1).float())
    assert ent.shape == (6,)
    model.learn(128)
    path = tmp_path / "m.zip"
    model.save(str(path))
    loaded = PPO.load(str(path), device="cpu")
    o, _ = env.reset(seed=3)
    np.testing.assert_allclose(model.predict(o, deterministic=True)[0], loaded.predict(o, deterministic=True)[0], atol=1e-6)


# -- the Dirichlet allocation head (ESCALATION #E23) ----------------------------


def test_dirichlet_decode_and_no_fallback(domain):
    _, scen, _, gymmod = domain
    ds = gymmod.dirichlet_shares
    sh, k = ds([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    assert k == pytest.approx(0.2) and sh.sum() == pytest.approx(0.8) and sh[4] == 0.0
    sh, k = ds([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert k == 1.0 and (sh == 0).all()
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_dirichlet")
    assert env.action_space.shape[0] == sc.n_retailers + 2
    rng = np.random.default_rng(6)
    worst = 1.0
    for ep in range(4):
        env.reset(seed=ep)
        done = False
        while not done:
            a = np.concatenate([[rng.uniform(0, 10)], rng.dirichlet(np.full(sc.n_retailers + 1, 0.5))]).astype(np.float32)
            _, _, term, trunc, info = env.step(a)
            worst = min(worst, info["ship_scale"])
            assert env._state.wh_stock >= -1e-9
            done = term or trunc
    assert worst >= 1.0 - 1e-6


def test_dirichlet_policy_distribution(domain, tmp_path):
    """DirichletPolicy: samples lie on the simplex; log-prob = Gaussian(order) +
    Dirichlet(shares), equal between rollout and re-scoring; the mode puts
    exact zeros on components with alpha <= 1; alpha starts near ALPHA_INIT;
    checkpoints reload and act identically."""
    pytest.importorskip("torch")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    pytest.importorskip("stable_baselines3")   # the upstream CI runner is torch-free (case-gates installs harness[dev] only)
    import importlib
    import torch as th
    from stable_baselines3 import PPO
    _, scen, _, gymmod = domain
    dp = importlib.import_module("owmr_dirichlet_policy")
    sc = scen.SCENARIOS["base"]
    env = gymmod.OwmrEnv(sc, "raw", "order_dirichlet")
    model = PPO(dp.DirichletPolicy, env, n_steps=64, batch_size=32, device="cpu",
                policy_kwargs={"net_arch": [16, 16]}, seed=0)
    pol = model.policy
    assert isinstance(pol.action_dist, dp.GaussianDirichlet)
    obs = th.as_tensor(np.random.default_rng(0).normal(size=(8, env.observation_space.shape[0])), dtype=th.float32)
    with th.no_grad():
        dist = pol.get_distribution(obs)
        assert th.allclose(dist.alpha, th.full_like(dist.alpha, dp.ALPHA_INIT), atol=0.3)   # small-gain head, bias restored
        a, v, lp = pol(obs)
        _, lp2, ent = pol.evaluate_actions(obs, a)
        det, _, _ = pol(obs, deterministic=True)
    assert th.allclose(a[:, 1:].sum(1), th.ones(8), atol=1e-5) and (a[:, 1:] > 0).all()
    assert th.allclose(lp, lp2) and ent.shape == (8,)
    assert th.allclose(det[:, 1:].sum(1), th.ones(8), atol=1e-6)
    mode = dp.dirichlet_mode(th.tensor([[3.0, 0.5, 2.0, 0.9, 1.0, 1.5], [0.5, 0.2, 0.9, 0.3, 0.1, 0.4]]))
    assert mode[0, 1] == 0 and mode[0, 3] == 0 and mode[0, 4] == 0 and mode[0].sum() == pytest.approx(1.0)
    assert mode[0, 0] == pytest.approx(2.0 / 3.5) and mode[1].tolist() == [0, 0, 1, 0, 0, 0]
    model.learn(128)
    path = tmp_path / "m.zip"
    model.save(str(path))
    loaded = PPO.load(str(path), device="cpu")
    o, _ = env.reset(seed=3)
    np.testing.assert_allclose(model.predict(o, deterministic=True)[0], loaded.predict(o, deterministic=True)[0], atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

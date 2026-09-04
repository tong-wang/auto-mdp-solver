"""Domain-owned gates for `adi_flex` (spec §1.2).

Helpers come from `mdp_ir.testing`, never a repo conftest, so this file works
wherever the folder lives — the portable-domain contract that also puts
`adi_flex_ir_adapter.py` in here.

Covers: the engine laws, the differential over the full declared covering set,
the model/rendering boundary that F1 established, and negative controls for
each of the two gates this domain claims.

Standalone:
    python adi_flex_test.py
"""

from __future__ import annotations

import json
import re

import pytest

from mdp_ir.testing import assert_laws, assert_match, mutated, schema_beside

SCHEMA = schema_beside(__file__)
EPISODES = 8


def _raw() -> dict:
    """The schema document with its `mdp` block flattened.

    This domain uses the FLAT layout, but read it through `ungroup_mdp` anyway:
    a flat block passes through unchanged, so every raw-JSON assertion below
    keeps holding if the grouped (model / design / rendering) layout is ever
    adopted. Going through the raw `["mdp"][...]` path without this is the trap
    that killed `fnv_test.py` at collection the moment `--regroup` ran.
    """
    from mdp_ir.schema import ungroup_mdp

    doc = json.loads(SCHEMA.read_text())
    doc["mdp"] = ungroup_mdp(doc["mdp"])
    return doc


def _load():
    from mdp_ir.schema import load_ir

    return load_ir(SCHEMA)


def _declared_compositions() -> list[str | None]:
    """Base + every declared instance, read from the schema so the sweep cannot
    drift from the catalog it is meant to cover."""
    scenario = _raw()["mdp"]["scenario"]
    base: list[str | None] = [None]
    return base + sorted(scenario.get("instances") or {})


COMPOSITIONS = _declared_compositions()


def test_the_covering_set_is_not_silently_empty():
    """Without this guard every parametrized case below "passes" by not
    existing. 30 declared IR instances across four boards, plus a fifth board
    that is scenario-only: the 15 homogeneous
    (L,T) grid cells, het_exp0..7, the six T_dl=3 het3_exp1..6 (F10), and the
    T_dl=1 Case-1 instance homog_L0_Tdl1 (F26) and the T_dl=0 null control
    homog_L0_Tdl0 (F27). The boards are counted
    separately because they are `cases` siblings — a total alone would let one
    board absorb another's loss."""
    # 30 IR instances + base. The T_dl=0 board is a SCENARIO but not an IR
    # instance: `Decision.dim` must resolve >= 1 upstream, and at T_dl=0 the
    # allocation decision genuinely has zero components (F27). Runnable and
    # gate-covered through the gym; NOT differentially verified.
    assert len(COMPOSITIONS) == 31, COMPOSITIONS
    grid = [c for c in COMPOSITIONS
            if c and c.startswith("homog_L") and "Tdl" not in c]
    assert len(grid) == 15, grid
    assert sum(1 for c in COMPOSITIONS if c and c.startswith("het_exp")) == 8
    assert sum(1 for c in COMPOSITIONS if c and c.startswith("het3_exp")) == 6
    # the declared-width ladder: T_dl = 0, 1, 2 at L = 0, which is the paper's
    # own numeric range (§3.3) and RQ4's null / Case-1 / Case-2 boards
    assert "homog_L0_Tdl1" in COMPOSITIONS
    assert "homog_L0_Tdl0" not in COMPOSITIONS, (
        "the T_dl=0 board is declared as a scenario only — if it has become an "
        "IR instance, upstream must first admit a zero-width decision")
    from adi_flex_scenarios import SCENARIOS
    assert SCENARIOS["homog_L0_Tdl0"].n_alloc == 0


# -- engine laws -------------------------------------------------------------


def test_engine_laws():
    assert_laws(SCHEMA)


# -- the differential, over the declared covering set ------------------------


@pytest.mark.parametrize(
    "instance", COMPOSITIONS, ids=[i or "base" for i in COMPOSITIONS]
)
def test_differential_matches_the_domain(instance):
    assert_match(SCHEMA, instance, episodes=EPISODES)


# -- negative controls: a gate never observed failing is not known to gate ----
#
# These drive a FIXED policy instead of using `mdp_ir.testing.assert_diverges`,
# and the reason is the finding that writing them produced: the differential's
# default is a random policy within the declared bounds, and this domain's order
# bound is [0, 100] against a demand mean of 6. The random policy therefore
# orders ~50/period, inventory diverges upward, and two economically central
# regimes are NEVER VISITED — stock never goes negative (no backlog), and the
# surplus is so far above the protection level that sigma never binds. Both
# corruptions below sail through a default `assert_diverges` with a clean MATCH.
#
# So the covering differential above says the rendering is faithful *in the
# region a random policy reaches*, which is not the region the campaign runs in.
# `order=6, allocate=2` — the §7b trajectory's own policy — reaches both. Chosen
# deliberately per spec §1.2: "two gates on the same object are blind to
# different faults", and here the blindness is measurable —
#   order=6, allocate=20 is blind to the allocation fault (an allocation at or
#     above the surplus IS the maximal fill, so the corrupted expression agrees), and
#   order=8, allocate=20 is blind to the backorder fault (it never backlogs).


_CONTROL_DECISIONS = {"order": 6, "allocate": [6, 2]}   # n_alloc = T_dl = 2


def _diverges_under(edit, decisions=None) -> bool:
    """True if the mutated IR stops matching the domain under a fixed policy."""
    from mdp_ir.differential import load_adapter_factory, run_differential

    ir = mutated(_load(), edit)
    factory = load_adapter_factory(SCHEMA, ir)
    report = run_differential(
        ir,
        factory(ir, instance=None, seed_salt=1),
        episode_seeds=list(range(EPISODES)),
        decisions=decisions or _CONTROL_DECISIONS,
        seed_salt=1,
    )
    return not report.ok


def _break_allocation(doc: dict) -> None:
    """Drop the allocation decision out of the early-fill term (maximal fill
    regardless of what the agent chose — the pre-F7 sigma=0 behavior)."""
    for t in doc["mdp"]["dynamics"]["transitions"]:
        if t["event"] != "D":
            continue
        t["updates"] = [
            "want = [adv[k] for k in range(n_alloc)]" if u.startswith("want =") else u
            for u in t["updates"]
        ]
        assert any(u.startswith("want =") for u in t["updates"]), (
            "the mutation found no `want =` statement — the D-event dynamics were "
            "renamed and this control silently became a no-op"
        )


def _break_backorder(doc: dict) -> None:
    for c in doc["mdp"]["objective"]["per_step_components"]:
        if c["name"] == "backorder":
            c["expr"] = "0"


def test_a_corrupted_fill_cascade_diverges():
    """Control for the DYNAMICS. The five-case evolution of eq. (18)-(19) is
    rendered as a branch-free min/max cascade; drop the allocation decision out
    of the early-fill term and the IR must stop matching the simulator."""
    assert _diverges_under(_break_allocation)


def test_dropping_the_backorder_charge_diverges():
    """Control for the OBJECTIVE half. Costs travel in `info`, so a mis-stated
    cost component is invisible to any state-only comparison."""
    assert _diverges_under(_break_backorder)


def test_the_control_policy_is_what_makes_those_controls_bite():
    """Guard the paragraph above, so nobody "simplifies" the fixed policy back
    to the random default and silently un-gates both controls."""
    assert not _diverges_under(_break_allocation, {"order": 6, "allocate": [20, 20]})
    assert not _diverges_under(_break_backorder, {"order": 8, "allocate": [20, 20]})


# -- path independence, which the laws gate can no longer see -----------------


def test_demand_is_independent_of_the_decision_path():
    """Realized demand must not depend on what the agent did (spec §6.3).

    `mdp_ir.laws` checked this until F3 and now **SKIPs** it — "no unguarded
    state-independent draw lands in a row field". The property still holds: the
    seed key is `[tau, stream, period, episode_seed, seed_salt]`, none of it
    state. But keying the draw on `tau`, a dynamics local, is enough for the
    checker to stop being able to prove it, so the coverage moved here rather
    than quietly disappearing. Two very different order/allocation paths through
    the same episode seed must see identical demand vectors every period.
    """
    from adi_flex_mdp import advance1, advance2, init_state
    from adi_flex_scenarios import AdiFlexScenario, SCENARIOS

    scenario = SCENARIOS["het_exp4"]
    assert isinstance(scenario, AdiFlexScenario)   # not a sampler

    def draws(order: int, allocate: int):
        state, _ = init_state(scenario, episode_seed=11)
        out = []
        while not state.terminated:
            state1, info1 = advance1(scenario, state, order=order)
            state, _ = advance2(scenario, state1, allocate=allocate)
            out.append(info1["d"])
        return out

    idle = draws(0, 0)
    busy = draws(17, 5)
    assert idle == busy, "demand realization depends on the decision path"
    assert any(sum(d) > 0 for d in idle), "degenerate episode proves nothing"


# -- the model/rendering boundary (F1) ---------------------------------------


def test_no_state_width_is_a_literal():
    """The defect §5.0 exists for: a literal at a width site, for a quantity the
    model declares, is how a sweep maximum becomes a capacity limit nobody
    chose. `pipe` and `adv` both carried one until F1."""
    for sv in _raw()["mdp"]["state_variables"]:
        length = sv.get("length")
        if length is None:
            continue
        parts = length if isinstance(length, list) else [length]
        assert all(isinstance(p, str) for p in parts), (
            f"state {sv['name']!r} has a literal width {length!r} — name a "
            f"scenario constant so the width stays a design choice"
        )


def test_the_pipeline_width_is_the_supply_lead_time():
    """W_i = (w_i, ..., w_{i+L-1}) — the dimension IS L (paper §2, eq. 1).

    Replaces the F2-era guard that checked a `pipe_len` cap covered the L sweep.
    That cap is gone: a width that cannot be outrun by its own design axis is
    better than one that is merely checked against it. What can still drift is
    the rendering falling back to a fixed register, so this pins the width to L
    at both ends — the declaration and the simulator — including the L=0 case
    where the paper's state is (x_i, V_i) and the vector is empty.
    """
    from adi_flex_mdp import init_state
    from adi_flex_scenarios import SCENARIOS

    raw = _raw()
    pipe = next(sv for sv in raw["mdp"]["state_variables"] if sv["name"] == "pipe")
    assert pipe["length"] == "L", (
        f"pipe width names {pipe['length']!r}, not the supply lead time itself"
    )
    assert "pipe_len" not in {c["name"] for c in raw["mdp"]["scenario"]["constants"]}, (
        "the pipe_len cap is back — the width is L, per instance"
    )

    for name in ("homog_L0_T2", "homog_L2_T1", "homog_L4_T2", "het_exp4"):
        sc = SCENARIOS[name]
        state, _ = init_state(sc, episode_seed=0)
        assert len(state.pipe) == sc.L, (
            f"{name}: simulator pipeline is {len(state.pipe)} wide at L={sc.L}"
        )


def test_a_sampler_observation_covers_every_member_leadtime():
    """The padding is a rendering choice, and it has to cover the family.

    A specialist observes its own L slots; a sampler's Box is fixed at
    construction, so it pads to the members' maximum. If a member ever carries
    a longer pipeline than the space was built for, the observation silently
    truncates — this is what catches that.
    """
    import numpy as np

    from adi_flex_gym import AdiFlexEnv
    from adi_flex_scenarios import SCENARIOS

    sampler = SCENARIOS["homog_grid"]
    env = AdiFlexEnv(scenario=sampler, observation_mode="vec")
    assert env.pipe_slots == max(m.L for m in sampler.members)

    for ep in range(12):
        obs, _ = env.reset(seed=ep)
        assert env.observation_space.contains(obs), f"episode {ep}: obs outside space"
        assert len(state_pipe := env._state.pipe) <= env.pipe_slots, (
            f"episode {ep}: pipeline {len(state_pipe)} exceeds {env.pipe_slots} obs slots"
        )
        obs, _, _, _, _ = env.step(6)
        assert env.observation_space.contains(obs), f"episode {ep}: obs outside space"


def test_the_allocation_vector_is_derived_from_the_demand_lead_time():
    """The allocation is a VECTOR of dimension T − 1 (one component per
    crossover-eligible class — the same classes §4.2 counts protection levels
    for: "when T > 2, more than one protection level is needed"). At this
    campaign's T_dl = 2 it has exactly one component, which is why a scalar
    action renders it correctly — F4's lesson, carried through F7's
    redefinition: `dim` must NAME the constant, or widening T_dl silently
    leaves the allocation decision behind.
    """
    raw = _raw()
    consts = {c["name"]: c["value"] for c in raw["mdp"]["scenario"]["constants"]}
    assert consts["n_alloc"] == consts["T_dl"], (
        f"n_alloc={consts['n_alloc']} != T_dl={consts['T_dl']} — only the due-now "
        f"class is forced, so every class ahead is a decision (F10)"
    )
    dim = next(d for d in raw["mdp"]["decisions"] if d["name"] == "allocate")["dim"]
    assert dim == "n_alloc", (
        f"allocate.dim is {dim!r} — it must NAME the constant, not restate its value, "
        f"or widening T_dl silently leaves the allocation decision behind"
    )
    # and the rate vector spans one more class than there are allocation slots
    assert len(consts["lambda_seg"]) == consts["T_dl"] + 1


def test_the_declared_order_cap_is_the_derived_one():
    """F14/F16. `order_max` comes from the dynamics — a bound on the horizon's
    total demand, the most an order could ever be useful for — and the IR now
    carries that derivation directly rather than restating its value
    (auto-mdp-solver#53, shipped v0.9.11: an expression over constants resolves
    per instance).

    Two renderings of one statement still have to agree, so this drives the
    IR's own resolver against `AdiFlexScenario.order_max` on every declared
    instance. It also pins the *form*: a literal or a bare constant name here
    would be a restatement, and a restatement drifts — widening a horizon or a
    rate vector would silently leave an action space that binds.
    """
    from adi_flex_scenarios import SCENARIOS

    ir  = _load()
    raw = _raw()["mdp"]["decisions"]
    hi  = next(d for d in raw if d["name"] == "order")["bounds"]["value"][1]
    assert isinstance(hi, str) and not hi.isidentifier(), (
        f"order's upper bound is {hi!r} — a literal or a bare constant name is a "
        f"restatement of the derivation, not the derivation"
    )
    for name in _raw()["mdp"]["scenario"]["instances"]:
        declared = ir.mdp.decision_bounds("order", name)[1]
        derived  = SCENARIOS[name].order_max
        assert int(declared) == derived, (
            f"{name}: IR resolves order's cap to {declared}, the dynamics give {derived}"
        )


def test_the_rendering_honours_a_wider_demand_lead_time():
    """F10. The test above checks the IR *declares* T_dl generally — and it
    passed all campaign while the code was frozen at T_dl = 2: `adv` typed as a
    2-tuple, `d[2]` indexed by literal, one allocation micro-step, a module
    constant `T_DL`. A declaration no rendering honours is decoration, so this
    test drives an actual T_dl = 3 scenario through both layers.

    T_dl = 3 is outside the paper's own numerics (§3.3 limits those to
    T = 0,1,2) but inside its model (§4: "there are T + 1 segments, with demand
    lead times ranging from 0 to T"), which is the line this domain renders.
    """
    import numpy as np

    from adi_flex_gym import AdiFlexEnv
    from adi_flex_mdp import advance1, advance2, init_state, valid_allocation
    from adi_flex_scenarios import AdiFlexScenario

    sc = AdiFlexScenario(
        lambda_seg=(2.0, 1.0, 2.0, 1.0), L=1, N=8, K=100.0, h=1.0, p=9.0,
        alloc_enabled=True, scenario_name="t3_probe", desc="F10 guard",
    )
    assert (sc.T_dl, sc.n_alloc) == (3, 3), (sc.T_dl, sc.n_alloc)

    # -- simulator: every width follows the scenario, none is a literal
    state, info = init_state(sc, episode_seed=7)
    assert len(state.adv) == sc.T_dl, f"adv is {len(state.adv)} wide, want T_dl={sc.T_dl}"
    assert len(info["d"]) == sc.T_dl + 1
    assert len(info["allocate"]) == sc.n_alloc

    state1, info1 = advance1(sc, state, order=25)
    assert len(info1["d"]) == sc.T_dl + 1
    assert len(state1.adv) == sc.n_alloc, "one class per allocation component"

    # the components share ONE surplus: later caps shrink by what earlier ones
    # took, so a per-component mask can never oversell the stock
    caps, taken = [], 0
    for k in range(sc.n_alloc):
        caps.append(valid_allocation(sc, state1, k, taken))
        taken += caps[k]
    assert taken <= max(state1.inv, 0), (
        f"components oversold the free stock: {caps} > {max(state1.inv, 0)}"
    )

    # total transition at every width: an absurd vector clips, it does not raise
    nxt, info2 = advance2(sc, state1, (10**6,) * sc.n_alloc)
    assert len(nxt.adv) == sc.T_dl
    assert info2["early_fill"] <= max(state1.inv, 0)

    # -- gym: a period is 1 + n_alloc agent steps, and the spaces contain it
    env = AdiFlexEnv(scenario=sc, observation_mode="vec")
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs), "reset obs outside space at T_dl=3"
    rng = np.random.default_rng(0)
    steps, terminated = 0, False
    while not terminated:
        mask = env.action_masks()
        assert mask.any(), "mask must never be empty"
        obs, _, terminated, _, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert env.observation_space.contains(obs), f"step {steps} obs outside space"
        steps += 1
    assert steps == sc.N * (1 + sc.n_alloc), (
        f"{steps} agent steps for {sc.N} periods; a period is 1 + n_alloc = "
        f"{1 + sc.n_alloc} steps at T_dl={sc.T_dl}"
    )
    env.close()


def test_the_two_decisions_sit_at_different_information_sets():
    """F7's core claim, pinned at both layers. The order is chosen BEFORE the
    period's demand is observed; the allocation AFTER (paper §2 event
    sequence). Simulator: advance1 realizes the demand — so it must come back
    identical whatever the order was (same seed key), and advance2 must consume
    it. Gym: the order-phase observation must show the PREVIOUS period's demand
    (the reduced-state vhat information), never the current draw.
    """
    import numpy as np

    from adi_flex_gym import AdiFlexEnv
    from adi_flex_mdp import advance1, init_state
    from adi_flex_scenarios import SCENARIOS

    sc = SCENARIOS["het_exp4"]
    state, _ = init_state(sc, episode_seed=7)
    d_small = advance1(sc, state, order=0)[1]["d"]
    d_big   = advance1(sc, state, order=100)[1]["d"]
    assert d_small == d_big, "the demand draw depends on the order — wrong information set"

    env = AdiFlexEnv(scenario=sc)
    obs, _ = env.reset(seed=7)
    n, T = env.pipe_slots, env.T_dl
    # [inv, pipe(n), adv(T), ttg, d(T+1), phase, surplus, outstanding] — computed,
    # never written out: F10 is exactly the bug a literal offset here would hide,
    # and F24 moved `d` ahead of `phase`, which is the same bug one layer up
    d_lo    = 1 + n + T + 1
    d_block = slice(d_lo, d_lo + T + 1)
    phase_i = d_lo + T + 1        # het branch: the block is present
    assert not obs[d_block].any(), "order-phase obs at t=0 must show no demand yet"
    obs, _, _, _, _ = env.step(6)
    assert obs[phase_i] == 1.0 and tuple(int(x) for x in obs[d_block]) == d_small, (
        "allocation-phase obs must show this period's realized demand"
    )
    for _ in range(env.n_alloc):        # one micro-step per allocation component
        obs, _, _, _, _ = env.step(0)
    assert obs[phase_i] == 0.0 and tuple(int(x) for x in obs[d_block]) == d_small, (
        "next order-phase obs must show LAST period's demand, not a new draw"
    )


def test_the_action_mask_is_the_feasible_set_and_the_transition_is_total():
    """The mask exposes exactly {0..min(surplus, outstanding)} at the
    allocation phase (validity computed in the mdp layer, spec §7.1), and an
    UNMASKED overshoot must clip inside the simulator rather than crash or
    over-fill — the invariant: unmasked play is safe, masking is for
    exploration. (This is what keeps random-action conformance runs green,
    where `retailer` fails with its own constraint error.)
    """
    from adi_flex_gym import AdiFlexEnv
    from adi_flex_mdp import advance2, valid_allocation
    from adi_flex_scenarios import SCENARIOS

    sc = SCENARIOS["het_exp4"]
    env = AdiFlexEnv(scenario=sc)
    checked = 0
    for ep in range(6):
        obs, _ = env.reset(seed=ep)
        terminated = False
        while not terminated:
            mask = env.action_masks()
            if env._phase == 1.0:
                mid  = env._mid
                # the mask is for the component the env is asking about, against
                # what the earlier components of this same allocation took
                k    = len(env._alloc_buf)
                took = sum(env._alloc_buf)
                fmax = valid_allocation(sc, mid, k, took)
                assert mask.sum() == min(fmax, sc.order_max) + 1, "mask != feasible set"
                # total transition: overshoot lands exactly on the maximal fill
                s_over, _ = advance2(sc, mid, allocate=sc.order_max)
                s_max,  _ = advance2(sc, mid, allocate=(10**6,) * sc.n_alloc)
                assert s_over == s_max, "an infeasible allocation must clip, not overfill"
                checked += 1
            obs, _, terminated, _, _ = env.step(int(mask.argmax()))
    assert checked > 0


def test_the_homogeneous_branch_is_the_maximal_fill_policy():
    """The unified-model cross-check, post-F7: het_exp7 (0,0,6) under a
    maximal-fill allocation must reproduce homog_L0_T2 trajectories bit-exactly
    (alloc_enabled=False forces what allocate=max chooses — the policy the
    pre-F7 encoding wrote as sigma=0)."""
    from adi_flex_mdp import advance1, advance2, init_state
    from adi_flex_scenarios import SCENARIOS

    het, hom = SCENARIOS["het_exp7"], SCENARIOS["homog_L0_T2"]
    assert het.alloc_enabled and not hom.alloc_enabled

    for seed in (0, 5, 11):
        s_het, _ = init_state(het, seed)
        s_hom, _ = init_state(hom, seed)
        for _ in range(het.N):
            i_het, j_het = advance1(het, s_het, order=6)
            i_hom, j_hom = advance1(hom, s_hom, order=6)
            s_het, info_het = advance2(het, i_het, allocate=10**6)  # clips to max
            s_hom, info_hom = advance2(hom, i_hom, allocate=0)      # forced max
            assert (s_het.inv, s_het.adv) == (s_hom.inv, s_hom.adv)
            assert info_het["cost"] == info_hom["cost"]
            assert j_het["d"] == j_hom["d"]


def test_the_model_layer_names_no_retired_identifier_in_the_present_tense():
    """Prose that describes the CURRENT schema must not name a dead identifier.

    Every correction in this campaign's IR-CHANGELOG left prose behind pointing
    at what had just been removed — `narrowed` describing literals F2 had
    replaced, a `notation` entry calling σ a scalar after F4 made it a vector,
    info descs citing `lambda0..lambda2` after F3 replaced them with one vector.
    The docs kept asserting the previous state while the schema had moved.

    Past-tense history is *wanted* — "was order_c/protect_c" is how a reader
    learns what changed — so this checks only strings that make no such marker.
    """
    retired = ("lambda0", "lambda1", "lambda2", "demand0", "demand1", "demand2",
               "order_d", "protect_d", "order_c", "protect_c",
               "protect", "sigma_enabled", "n_protect", "protect_max",
               "sigma_eff", "pipe_len", "vec_d", "key_exprs")
    past = ("was ", "were ", "until", "before", "pre-", "enumerat", "replaced",
            "had been", "earlier", "no longer", "predate", "died", "retired",
            "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
    declared = {c["name"] for c in _raw()["mdp"]["scenario"]["constants"]}
    assert not (set(retired) & declared), "a retired name is declared again"

    offenders = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            hit = [n for n in retired if re.search(rf"\b{n}\b", node)]
            if hit and not any(mark.lower() in node.lower() for mark in past):
                offenders.append(f"{path}: names {hit} with no past-tense marker")

    walk(_raw(), "mdp")
    assert not offenders, "\n".join(offenders)


def test_the_notation_map_records_each_symbol_once():
    """Two entries for one symbol is two sources of truth, and the stale one
    wins whichever a reader happens to read first — which is how `sigma_i`
    survived F4 still describing σ as a scalar."""
    note = _raw()["mdp"]["model"]["notation"]
    # the symbol is the leading run of letters: `sigma_i` and `sigma_i^{i+k}`
    # are both "sigma", `L` and `lambda^tau` stay distinct
    stems = [m.group(0).lower()
             for m in (re.match(r"[A-Za-z]+", k) for k in note) if m]
    dupes = {s for s in stems if stems.count(s) > 1}
    assert not dupes, f"notation records {dupes} under more than one key: {list(note)}"


def test_every_action_mode_drives_every_decision():
    """The mdp layer consumes both canonical decisions each period, so a mode
    encoding only one cannot step the env. This is what the pre-F1 IR got wrong
    (four half-modes naming one decision each) and it is worth a local guard:
    the schema validator rejects it, but only once someone loads the schema."""
    ir = _load()
    names = {d.name for d in ir.mdp.decisions}
    for mode in ir.gym.action_modes:
        assert set(mode.encoded_decisions()) == names, (
            f"action mode {mode.name!r} drives {mode.encoded_decisions()}, "
            f"but the mdp layer consumes {sorted(names)} every period"
        )


def test_declared_benchmarks_have_their_roles_and_files():
    """§9.9: the role is what makes a leaderboard readable, and `relaxed` +
    `feasible` is what brackets the het optimum without an exact solver."""
    ir = _load()
    roles = {b.name: b.role.value for b in ir.benchmarks}
    assert roles == {"dp": "exact", "ap": "relaxed", "rule": "feasible"}, roles
    here = SCHEMA.parent
    for name in roles:
        assert (here / f"adi_flex_benchmark_{name}.py").exists(), name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_the_mip_observation_is_the_papers_state_and_not_the_whole_profile():
    """F18. `vec_mip` carries Wang & Toktay's sufficient statistic, exactly.

    Eq. (7) defines the MIP as the inventory position less ALL the advance
    demands — not Gallego & Ozer's, which deducts only the protection period
    [i, i+L]; the paper flags the difference itself. What survives BESIDE it is
    the Case boundary, and it is the whole content of the mode:

    - Case 1, T <= L + 1 (Prop. 1): nothing. The scalar u_i is the entire
      state and evolves linearly.
    - Case 2, T > L + 1 (eq. 11): V~ = (v_i^{i+L+1}, ..., v_i^{i+T-1}) only,
      of dimension T - L - 1.

    Re-appending the full `adv` — which is what this mode did until F18 —
    restates what `mip` has already deducted and makes the mode something
    other than the paper's state, which would have made an RL comparison
    against `vec` answer a question nobody asked.
    """
    import numpy as np

    from adi_flex_gym import AdiFlexEnv, mip_tail_slots
    from adi_flex_scenarios import SCENARIOS

    seen_case1 = seen_case2 = 0
    for name, sc in SCENARIOS.items():
        if not hasattr(sc, "L"):
            continue
        expected = max(0, sc.T_dl - sc.L - 1)
        assert mip_tail_slots(sc) == expected, (
            f"{name}: MIP tail is {mip_tail_slots(sc)} wide, eq. (11) says "
            f"T_dl - L - 1 = {expected}"
        )
        env = AdiFlexEnv(scenario=sc, observation_mode="vec_mip")
        # layout: [mip, V~(expected), time_to_go] + the within-period block
        # (phase, d(T_dl+1), surplus, outstanding), which F23 renders only when
        # the period HAS an interior
        block = ((sc.T_dl + 1) + 3) if sc.alloc_enabled else 0
        assert env.observation_space.shape[0] == 1 + expected + 1 + block

        obs, _ = env.reset(seed=3)
        assert env.observation_space.contains(obs)
        # the MIP itself is eq. (7): ALL of adv deducted
        s = env._state
        assert obs[0] == np.float32(s.inv + sum(s.pipe) - sum(s.adv))
        # and what follows is adv[L+1:], not adv
        assert list(obs[1 : 1 + expected]) == [float(x) for x in s.adv[sc.L + 1 :]]
        if expected == 0:
            seen_case1 += 1
            assert sc.T_dl <= sc.L + 1
        else:
            seen_case2 += 1
            assert sc.T_dl > sc.L + 1
            # the tail is a STRICT sub-vector: the full profile would be wider
            assert expected < sc.T_dl, f"{name}: tail is the whole profile"

    # the boundary is exercised in both directions, or this test proves nothing
    assert seen_case1 and seen_case2, (
        f"Case 1 seen {seen_case1}x, Case 2 {seen_case2}x — the registry must "
        f"cover both sides of T <= L + 1 for this to be a check"
    )


def test_the_allocation_observes_the_interstate_not_the_period_opening():
    """F19. The allocation decides on the interstate, so it must observe it.

    `advance1` places the order into the pipeline, lands the arrival in `inv`
    and joins this period's demand into `adv`; the allocation then weighs all
    three. Reading the period-OPENING state instead left the observation one
    event-step stale, and at L >= 1 nothing else carried the order — so a
    memoryless policy could not see its own action from one step earlier,
    contradicting `rl.requires_memory = False`. At L = 0 the bug was invisible
    (the order lands in `inv` and leaks through `surplus`), which is why only
    the L >= 1 instances expose it and why this test must use one.
    """
    import numpy as np

    from adi_flex_gym import AdiFlexEnv, PHASE_ALLOC
    from adi_flex_scenarios import SCENARIOS

    live = [
        (n, s) for n, s in SCENARIOS.items()
        if hasattr(s, "L") and s.L >= 1 and s.alloc_enabled and s.n_alloc > 0
    ]
    assert live, "no allocating instance with L >= 1 — this test cannot bite"

    for name, sc in live:
        for mode in ("vec", "vec_mip", "vec_mip2"):
            seen = {}
            for order in (0, 7, 40):
                env = AdiFlexEnv(scenario=sc, observation_mode=mode)
                env.reset(seed=11)
                obs, _, _, _, _ = env.step(order)
                assert env._phase == PHASE_ALLOC, "expected the allocation phase"
                assert env.observation_space.contains(obs)
                seen[order] = obs
            assert not np.array_equal(seen[0], seen[40]), (
                f"{name}/{mode}: the allocation observation is identical whether "
                f"0 or 40 was just ordered — the order is invisible (F19)"
            )

        # and the state read is the interstate itself, component by component
        env = AdiFlexEnv(scenario=sc, observation_mode="vec")
        env.reset(seed=11)
        env.step(9)
        mid = env._mid
        assert mid is not None
        obs = env._get_obs()
        n_pipe = env.pipe_slots
        assert obs[0] == np.float32(mid.inv), f"{name}: obs inv is not the interstate's"
        assert list(obs[1 : 1 + len(mid.pipe)]) == [float(x) for x in mid.pipe], (
            f"{name}: obs pipeline is not the interstate's — the order is in its "
            f"last slot, which is the whole point"
        )
        assert list(obs[1 + n_pipe : 1 + n_pipe + len(mid.adv)]) == [
            float(x) for x in mid.adv
        ], f"{name}: obs profile is not the interstate's (demand not yet joined)"


def test_the_record_separates_the_request_from_the_realization():
    """F20. `allocate` is what was asked; `fills` is what happened.

    On the homogeneous branch `alloc_enabled` is False, `advance2` forces the
    maximal fill and never reads its `allocate` argument — so the requested
    vector is INERT there, and a record carrying it alone shows a decision of
    zero beside a fill it did not cause. Both are now reported, and `fills` is
    what the differential compares, so the forcing is checked against the IR's
    own rendering of it rather than trusted.
    """
    import adi_flex_mdp as mdp
    from adi_flex_scenarios import SCENARIOS

    homog = SCENARIOS["homog_L0_T2"]
    assert not homog.alloc_enabled, "this test needs the forced-fill branch"
    state, _ = mdp.init_state(homog, 7)
    mid, _ = mdp.advance1(homog, state, order=30)

    realized = None
    for asked in (0, 1, 99):
        nxt, info = mdp.advance2(homog, mid, asked)
        assert info["allocate"] == (asked,) * homog.n_alloc, "request not recorded as driven"
        assert sum(info["fills"]) == info["early_fill"], "fills must sum to early_fill"
        if realized is None:
            realized = (tuple(info["fills"]), nxt.inv, nxt.adv)
        # the argument is inert: every realization is identical
        assert (tuple(info["fills"]), nxt.inv, nxt.adv) == realized, (
            f"asking for {asked} changed the outcome on a branch that forces "
            f"the maximal fill — the argument is supposed to be ignored"
        )
    # ...and the forced fill is not the trivial one, or the check is vacuous
    assert realized is not None and sum(realized[0]) > 0, (
        "the forced fill was zero, so 'the argument is ignored' proves nothing"
    )
    # nothing is left fillable: that is what maximal means
    assert min(realized[1], 0) == realized[1] or all(v == 0 for v in realized[2]), (
        "stock remained while a class ahead was still outstanding"
    )

    # the heterogeneous branch is the opposite: the request drives, and `fills`
    # is it after the shared-surplus clip
    het = SCENARIOS["het_exp4"]
    assert het.alloc_enabled
    state, _ = mdp.init_state(het, 7)
    mid, _ = mdp.advance1(het, state, order=30)
    seen = set()
    for asked in (0, 1, 99):
        _, info = mdp.advance2(het, mid, asked)
        assert sum(info["fills"]) == info["early_fill"]
        assert all(f <= a for f, a in zip(info["fills"], info["allocate"])), (
            "a class was filled beyond what was asked for it"
        )
        seen.add(tuple(info["fills"]))
    assert len(seen) > 1, "the request never moved the realization — the decision is inert"


def test_the_within_period_block_exists_only_when_the_period_has_an_interior():
    """F23, restored at F25 after F24 wrongly put `d` back.

    The block — `d`, `phase`, `surplus`, `outstanding` — describes what happens
    BETWEEN a period's two decisions, and a single-phase period has no between.
    `d` at the order phase is LAST period's draw: history, not state. F24
    restored it because the exact DP indexes on `d[2]` and `d[2]` is not a
    function of the state; both true, neither decisive. Substitution is:
    playing the same `y*` table with `vhat` read from the state's `adv[1]`
    costs 685.263 against `d[2]`'s 685.263 — identical over 61440 decisions —
    while `vhat = 0` costs 791.486. The coordinate matters and the STATE
    carries it.
    """
    import numpy as np

    from adi_flex_gym import AdiFlexEnv
    from adi_flex_scenarios import SCENARIOS

    seen_on = seen_off = 0
    for name, sc in SCENARIOS.items():
        if not hasattr(sc, "L"):
            continue
        for mode in ("vec", "vec_mip", "vec_mip2"):
            env = AdiFlexEnv(scenario=sc, observation_mode=mode)
            core = (1 + env.mip_tail_slots if mode.startswith("vec_mip")
                    else 1 + env.pipe_slots + env.adv_slots) + 1   # + time_to_go
            # the within-period block only when the period has an interior (F25)
            want = core + ((sc.T_dl + 1) + 3 if sc.alloc_enabled else 0)
            assert env.observation_space.shape[0] == want, (
                f"{name}/{mode}: obs is {env.observation_space.shape[0]} wide, want {want} "
                f"(core {core}"
                f"{' + block ' + str(sc.T_dl + 4) if sc.alloc_enabled else ', no block'})"
            )
        seen_on += sc.alloc_enabled
        seen_off += not sc.alloc_enabled
    assert seen_on and seen_off, "registry must cover both sides of alloc_enabled"

    # and nothing constant survives on the branch the block was hiding
    sc = SCENARIOS["homog_L0_T2"]
    env = AdiFlexEnv(scenario=sc, observation_mode="vec")
    rows = []
    for ep in range(20):
        env.reset(seed=ep)
        rng = np.random.default_rng(ep)
        while True:
            obs, _, term, _, _ = env.step(int(rng.integers(0, 20)))
            rows.append(obs)
            assert env.observation_space.contains(obs)
            if term:
                break
    arr = np.array(rows)
    const = [i for i in range(arr.shape[1]) if np.all(arr[:, i] == arr[0, i])]
    # d[0] and d[1] are identically zero at lambda_seg=(0,0,6) — a property of
    # the RATE vector, not the rendering, so they stay (narrowing on a zero rate
    # is the F10 trap). What must NOT survive is a feature dead by LAYOUT.
    n_pipe, n_adv = env.pipe_slots, env.adv_slots
    layout_dead = [i for i in const if i < 1 + n_pipe + n_adv + 1]
    assert not layout_dead, (
        f"homog_L0_T2 carries constant features at {layout_dead}, outside the "
        f"rate-driven `d` slots — those are dead by layout")


def test_the_two_mip_coordinates_carry_the_same_information():
    """F33. `vec_mip` and `vec_mip2` are the same state in a different basis.

    `vec_mip` is Wang & Toktay's `u = IP - sum(adv)` (eq. 7); `vec_mip2` is
    Gallego & Ozer's `w = IP - sum(adv[0..L])`, which W&T name and distinguish
    themselves from. Both carry the SAME tail `V~ = adv[L+1:]`, so
    `u = w - sum(V~)` recovers one from the other exactly and neither knows
    anything the other does not. That is the whole point: a performance
    difference between them is about the COORDINATE, not the information, which
    no other pair in this campaign isolates.
    """
    import numpy as np

    from adi_flex_gym import AdiFlexEnv
    from adi_flex_scenarios import SCENARIOS

    checked = differed = 0
    for name, sc in SCENARIOS.items():
        if not hasattr(sc, "L"):
            continue
        a = AdiFlexEnv(scenario=sc, observation_mode="vec_mip")
        b = AdiFlexEnv(scenario=sc, observation_mode="vec_mip2")
        assert a.observation_space.shape == b.observation_space.shape, (
            f"{name}: the two coordinates must have identical width — same tail, "
            f"one scalar each"
        )
        n_tail = a.mip_tail_slots
        for ep in range(4):
            a.reset(seed=ep)
            b.reset(seed=ep)
            rng = np.random.default_rng(ep)
            while True:
                oa, ob = a._get_obs(), b._get_obs()
                u, w = float(oa[0]), float(ob[0])
                tail = float(sum(ob[1:1 + n_tail]))
                assert abs(u - (w - tail)) < 1e-4, (
                    f"{name}: u={u} but w-sum(V~)={w - tail} — the two "
                    f"coordinates have stopped being a change of basis"
                )
                assert np.allclose(oa[1:], ob[1:]), f"{name}: tails diverged"
                checked += 1
                differed += (u != w)
                act = int(rng.integers(0, 12))
                _, _, ta, _, _ = a.step(act)
                _, _, tb, _, _ = b.step(act)
                assert ta == tb
                if ta:
                    break
    assert checked > 500, checked
    assert differed > 0, (
        "the two coordinates never differed — the test proves nothing unless "
        "some visited state has a non-empty tail"
    )


def test_the_plus_modes_are_vec_in_mip_coordinates():
    """F34. `vec_mip_plus_a` / `_plus_b` restore what `vec_mip` compressed away.

    `vec_mip` is lossy: from `(mip, V~)` the split of `mip = inv - adv[0] -
    sum(V~)` into its parts is gone. Hand back EITHER part and the loss is
    undone -- `_plus_a` carries `inv`, `_plus_b` carries `adv[0]`, and each
    recovers the other by that identity. So both modes carry exactly `vec`'s
    information at exactly `vec`'s width, and the only thing that varies across
    `vec` / `_plus_a` / `_plus_b` is WHICH basis the same knowledge arrives in.

    That is what makes them a test of redundancy rather than of information:
    every information-theoretic account of the vec-vs-vec_mip gap has already
    failed (see the F34 assumptions_log entry), so what is left to vary is how
    many coordinates gradient descent has to fit one relationship along.

    SCOPE: the equivalence holds AT L = 0 ONLY, which is where the experiment
    runs. With a pipeline, `mip` folds `sum(pipe)` into the statistic, so
    handing back one of `inv` / `adv[0]` still leaves the per-slot breakdown
    unrecoverable and a `_plus_` mode stays strictly narrower than `vec`. The
    test asserts equivalence on the L = 0 instances and asserts the narrowness
    everywhere else, so neither claim is quietly over-extended.
    """
    import numpy as np

    from adi_flex_gym import AdiFlexEnv, PHASE_ALLOC
    from adi_flex_scenarios import SCENARIOS

    checked = nonzero_adv0 = 0
    for name, sc in SCENARIOS.items():
        if not hasattr(sc, "L") or sc.T_dl == 0:
            continue                     # no advance demand => nothing compressed
        raw = AdiFlexEnv(scenario=sc, observation_mode="vec")
        pa  = AdiFlexEnv(scenario=sc, observation_mode="vec_mip_plus_a")
        pb  = AdiFlexEnv(scenario=sc, observation_mode="vec_mip_plus_b")
        assert pa.observation_space.shape == pb.observation_space.shape, (
            f"{name}: the two _plus_ modes trade one raw component for another "
            f"and must have identical width"
        )
        if sc.L != 0:
            # a pipeline is summarized inside `mip` and no single restored
            # component brings the slots back — out of the equivalence's scope
            assert pa.observation_space.shape < raw.observation_space.shape, (
                f"{name}: at L>0 a _plus_ mode must still be strictly narrower "
                f"than `vec` — it cannot have recovered the pipeline"
            )
            continue
        assert raw.observation_space.shape == pa.observation_space.shape, (
            f"{name}: at L=0 a _plus_ mode must be exactly as wide as `vec` — it "
            f"trades one raw component for the statistic, it does not add a feature"
        )
        n_tail = pa.mip_tail_slots
        for ep in range(4):
            raw.reset(seed=ep); pa.reset(seed=ep); pb.reset(seed=ep)
            rng = np.random.default_rng(1000 + ep)
            while True:
                oa, ob = pa._get_obs(), pb._get_obs()
                st = raw._mid if (raw._phase == PHASE_ALLOC and raw._mid is not None) \
                    else raw._state
                u_a, u_b = float(oa[0]), float(ob[0])
                tail_a = float(sum(oa[1:1 + n_tail]))
                inv_a, adv0_b = float(oa[1 + n_tail]), float(ob[1 + n_tail])
                assert u_a == u_b, f"{name}: both _plus_ modes use W&T's mip"
                # each recovers the component the other was given
                assert abs((inv_a - u_a - tail_a) - float(st.adv[0])) < 1e-4, (
                    f"{name}: adv[0] is not recoverable from _plus_a — the mode "
                    f"is not information-equivalent to `vec`"
                )
                assert abs((u_b + adv0_b + tail_a) - float(st.inv)) < 1e-4, (
                    f"{name}: inv is not recoverable from _plus_b"
                )
                checked += 1
                nonzero_adv0 += (float(st.adv[0]) != 0.0)
                act = int(rng.integers(0, 12))
                _, _, t0, _, _ = raw.step(act)
                _, _, t1, _, _ = pa.step(act)
                _, _, t2, _, _ = pb.step(act)
                assert t0 == t1 == t2
                if t0:
                    break
    assert checked > 500, checked
    assert nonzero_adv0 > 0, (
        "adv[0] was zero at every visited state — the recovery identity was "
        "never actually exercised, so the test proves nothing"
    )


# ---------------------------------------------------------------------------
# The action modes whose decode the IR cannot declare
#
# Schema v1 refuses a per-component `transform` on a multi-decision mode, so
# `order_protection`, `target_ip` and `target_mip` all ship `transform: ""` and
# their decodes live in `adi_flex_gym` alone. The differential drives the mdp
# layer through `mdp.decisions`, never through a gym mode, so it cannot see a
# wrong decode; conformance and laws cannot either. These are the gates that
# stand in for the one that does not exist.
#
# The END-TO-END replays — PL(sigma) and the DP tables reproduced to the float
# through each new mode — live in `adi_flex_action_mode_probe.py` instead,
# because they read solver tables out of `results/` and this file must work
# wherever the folder lives (the portable-domain contract in the module
# docstring). Everything below is self-contained.
# ---------------------------------------------------------------------------


def test_the_declared_protection_cap_is_the_derived_one():
    """The sibling of the `order_max` test above, and the same rule (F14): a
    cap is a statement about the dynamics, never a measurement of solved runs.

    `sigma_t` is held back instead of serving `adv[t]`, so it can only ever be
    useful for demand arriving LATER but due SOONER — `sum_{j<t} (t-j)*D[j]`, a
    WEIGHTED sum of independent Poissons — and the IR carries that derivation
    rather than a number. This drives the IR's resolver against
    `AdiFlexScenario.sigma_max` on every declared instance, and pins the form:
    a literal here would be a restatement, and a restatement drifts the moment
    a rate vector moves.
    """
    from adi_flex_scenarios import SCENARIOS

    modes = {m["name"]: m for m in _raw()["gym"]["action_modes"]}
    hi = modes["order_protection"]["bounds"][1][1]
    assert isinstance(hi, str) and not hi.isidentifier(), (
        f"the protection cap is {hi!r} — a literal or a bare constant name is a "
        f"restatement of the derivation, not the derivation"
    )
    # Resolved through the API, one line, exactly as its `order_max` sibling
    # does — `MdpIR.action_bounds` is upstream #70 (this campaign's proposal,
    # shipped v0.9.30). It used to be eight lines re-implementing the namespace
    # rule (base constants under the instance's overrides) and reaching past
    # `load_ir` into the raw document for the constants pool, which was a second
    # implementation of a rule the harness already owned.
    ir = _load()
    for name in SCENARIOS:
        if name not in _raw()["mdp"]["scenario"]["instances"]:
            continue
        lo, declared = ir.action_bounds("order_protection", "allocate", name)
        assert lo == 0 and int(declared) == SCENARIOS[name].sigma_max, (
            f"{name}: IR resolves the protection cap to [{lo}, {declared}], "
            f"the dynamics give [0, {SCENARIOS[name].sigma_max}]"
        )
        # #69 (shipped v0.9.30) is what lets the derivation be declared as
        # itself: on an instance where no reserve can exist the cap is 0, so the
        # bound is the degenerate [0, 0] that schema v1 refused. Assert the
        # degeneracy rather than pinning sigma = 0 by hand elsewhere.
        if SCENARIOS[name].n_sigma and SCENARIOS[name].alloc_enabled:
            # zero exactly when NO crossover-eligible class carries rate. Not
            # `lambda_0 == 0` alone: sigma_t protects against
            # sum_{j<t} (t-j)*D[j], so at t >= 2 a later class still preempts —
            # het3_exp4 is (0, 1, 1, 4), lambda_0 = 0 and the cap is 5.
            eligible = sum(SCENARIOS[name].lambda_seg[: SCENARIOS[name].T_dl - 1])
            assert (declared == 0) == (eligible == 0), (
                f"{name}: cap {declared} disagrees with eligible rate "
                f"{eligible} — the cap is 0 exactly when nothing can preempt "
                f"the class being protected"
            )


def test_the_protection_cascade_is_the_papers_own_rule():
    """`order_protection`'s whole claim is that it re-parametrizes the SAME
    decision, so at T_dl = 2 its cascade must be paper §4.2 exactly.

    The cascade pours available stock into `sigma_0, adv[0], sigma_1, adv[1],
    ...` — a sigma box retains, an adv box ships — and §4.2 fills the due-next
    class then protects `sigma` from every crossover-eligible one. Those are
    the same function; this asserts it over the state space rather than
    trusting the algebra, including the negative-inventory and zero-surplus
    corners where an off-by-one would otherwise hide.
    """
    import dataclasses

    import numpy as np

    from adi_flex_gym import AdiFlexEnv
    from adi_flex_scenarios import SCENARIOS

    sc = SCENARIOS["het_exp4"]
    env = AdiFlexEnv(scenario=sc, action_mode="order_protection")
    env.reset(seed=0)
    proto = env._state
    rng = np.random.default_rng(11)

    for _ in range(4000):
        inv = int(rng.integers(-15, 60))
        adv = tuple(int(x) for x in rng.integers(0, 16, size=sc.n_alloc))
        sigma = int(rng.integers(0, sc.sigma_max + 1))
        st = dataclasses.replace(proto, inv=inv, adv=adv)

        want, taken = [], 0
        for k, out in enumerate(adv):                     # §4.2, component-wise
            surplus = max(inv, 0) - taken
            a = min(surplus, out) if k == 0 else \
                int(np.clip(surplus - sigma, 0, out))
            want.append(a)
            taken += a
        assert env._sigma_cascade(st, [sigma]) == tuple(want), (
            f"cascade diverges from §4.2 at inv={inv} adv={adv} sigma={sigma}"
        )


def test_a_protection_level_of_zero_is_the_maximal_fill_policy():
    """Where no reserve can be useful the mode must reduce to forced maximal
    fill, which §3 proves optimal for a homogeneous base.

    Two instances reach that state by different routes: `het_exp7` has
    `lambda_0 = 0`, so nothing can ever preempt the class being protected and
    the derived cap is 0; and `sigma = 0` on any board means the cascade
    reserves nothing. Both must agree with the incumbent mode playing a
    maximal-fill allocation, step for step.
    """
    import numpy as np

    from adi_flex_gym import AdiFlexEnv
    from adi_flex_scenarios import SCENARIOS

    for name in ("het_exp4", "het_exp7"):
        sc = SCENARIOS[name]
        a = AdiFlexEnv(scenario=sc, action_mode="order_protection")
        b = AdiFlexEnv(scenario=sc, action_mode="seq_mask")
        for seed in range(4):
            obs_a, _ = a.reset(seed=seed)
            b.reset(seed=seed)
            rng = np.random.default_rng(seed)
            done = False
            while not done:
                order = int(rng.integers(0, 40))
                obs_a, _, done, _, _ = a.step([order] + [0] * a.n_sigma)
                _, _, db, _, _ = b.step(order)          # order phase
                while not db:                            # then maximal fill
                    m = b.action_masks()
                    _, _, db, _, _ = b.step(int(np.flatnonzero(m)[-1]))
                    if b._phase == 0:
                        break
            assert a.total_reward == b.total_reward, (
                f"{name} seed {seed}: sigma=0 gave {-a.total_reward:.6f}, "
                f"maximal fill gave {-b.total_reward:.6f}"
            )


def test_the_no_order_sentinel_is_outside_the_target_range():
    """A target encoding cannot express "do not order" once the reference goes
    negative — `y <= reference` is the no-order condition, and with backlog the
    reference sits below every non-negative target. The optimal policy here DOES
    decline to order under backlog (K = 100 against p = 9), so a bare
    `[0, order_max]` target would force an order in exactly the (s,S) trigger
    region.

    Index 0 is therefore a sentinel meaning "do not order", and targets are
    `1..order_max+1`. It sits OUTSIDE the legitimate target range rather than
    consuming a value inside it, so no target level is sacrificed. This pins
    both halves: the sentinel never orders at any reference, and every target
    still resolves to `clip(target - reference, 0, order_max)`.
    """
    import dataclasses

    from adi_flex_gym import AdiFlexEnv
    from adi_flex_scenarios import SCENARIOS

    from adi_flex_gym import TARGET_MODES

    sc = SCENARIOS["homog_L0_T2"]
    for mode in TARGET_MODES:
        env = AdiFlexEnv(scenario=sc, action_mode=mode, observation_mode="vec")
        env.reset(seed=0)
        proto = env._state
        # the combined modes carry the same order head inside a MultiDiscrete,
        # so the width is read off the head rather than off a Discrete `.n`
        head = (int(env.action_space.nvec[0])
                if hasattr(env.action_space, "nvec") else int(env.action_space.n))
        assert head == sc.order_max + 2, (
            f"{mode}: expected order_max + 2 categories (targets plus the "
            f"sentinel), got {head}"
        )
        for inv in (-60, -12, -1, 0, 5, 40):
            st = dataclasses.replace(proto, inv=inv, adv=(3, 2))
            assert env._target_order(st, 0) == 0, (
                f"{mode}: the sentinel ordered at inv={inv} — it must never order"
            )
            # the reference is the MIP for every mip-flavoured mode, which now
            # includes target_mip_protection — matching the gym's own
            # `"mip" in action_mode`, not an equality against one name
            ref = st.inv + sum(st.pipe) - (sum(st.adv) if "mip" in mode else 0)
            for tgt in (0, 1, 7, sc.order_max):
                got = env._target_order(st, tgt + 1)
                assert got == max(0, min(tgt - ref, sc.order_max)), (
                    f"{mode}: target {tgt} at ref {ref} gave {got}"
                )


def test_a_target_policy_and_a_quantity_policy_are_the_same_policy():
    """The target modes re-parametrize the ORDER decision and nothing else, so
    any order-up-to policy played through them must equal the same policy
    played as quantities through `seq_mask` — to the float, not to a tolerance.

    This is the self-contained half of the acceptance suite: it uses a fixed
    order-up-to level rather than a solver table, so it needs no `results/`.
    The DP and AP replays, which are the sharper test because the optimal
    policy is natively an order-up-to policy, live in the probe.
    """
    import numpy as np

    from adi_flex_gym import AdiFlexEnv
    from adi_flex_scenarios import SCENARIOS

    for name, S in (("homog_L0_T2", 36), ("het_exp4", 18), ("homog_L0_Tdl1", 30)):
        sc = SCENARIOS[name]
        for mode in ("target_ip", "target_mip"):
            tgt = AdiFlexEnv(scenario=sc, action_mode=mode, observation_mode="vec")
            qty = AdiFlexEnv(scenario=sc, action_mode="seq_mask", observation_mode="vec")
            for seed in range(3):
                tgt.reset(seed=seed)
                qty.reset(seed=seed)
                for env in (tgt, qty):
                    done = False
                    rng = np.random.default_rng(1000 + seed)
                    while not done:
                        if env._phase == 0:
                            st = env._state
                            ref = st.inv + sum(st.pipe)
                            if mode == "target_mip":
                                ref -= sum(st.adv)
                            order = int(np.clip(S - ref, 0, sc.order_max))
                            act = (0 if order == 0 else
                                   int(np.clip(ref + order, 0, sc.order_max)) + 1) \
                                if env is tgt else order
                        else:
                            m = env.action_masks()
                            act = int(rng.choice(np.flatnonzero(m)))
                        _, _, done, _, _ = env.step(act)
                assert tgt.total_reward == qty.total_reward, (
                    f"{name}/{mode} seed {seed}: target {-tgt.total_reward:.6f} "
                    f"vs quantity {-qty.total_reward:.6f}"
                )


def test_the_combined_modes_compose_their_parents_and_add_no_decode():
    """`target_*_protection` is the composition of a target order decode and the
    protection cascade, with nothing of its own — that is why it was declarable
    at no marginal risk once both parents were verified.

    Composition is asserted directly: the combined mode's order head must decode
    exactly as its target parent's does, and its allocation must be the same
    cascade `order_protection` runs. If either drifts, the mode has grown a
    third decode nobody gated.
    """
    import dataclasses

    import numpy as np

    from adi_flex_gym import AdiFlexEnv
    from adi_flex_scenarios import SCENARIOS

    sc = SCENARIOS["het_exp4"]
    prot = AdiFlexEnv(scenario=sc, action_mode="order_protection")
    prot.reset(seed=0)
    proto = prot._state
    rng = np.random.default_rng(5)

    for parent, combined in (("target_ip", "target_ip_protection"),
                             ("target_mip", "target_mip_protection")):
        a = AdiFlexEnv(scenario=sc, action_mode=parent, observation_mode="vec")
        b = AdiFlexEnv(scenario=sc, action_mode=combined, observation_mode="vec")
        a.reset(seed=0)
        b.reset(seed=0)
        for _ in range(300):
            inv = int(rng.integers(-40, 60))
            adv = tuple(int(x) for x in rng.integers(0, 16, size=sc.n_alloc))
            st = dataclasses.replace(proto, inv=inv, adv=adv)
            for act in (0, 1, 9, sc.order_max + 1):
                assert a._target_order(st, act) == b._target_order(st, act), (
                    f"{combined}: order head diverges from {parent} at "
                    f"inv={inv} act={act}"
                )
            sig = int(rng.integers(0, sc.sigma_max + 1))
            assert (prot._sigma_cascade(st, [sig])
                    == b._sigma_cascade(st, [sig])), (
                f"{combined}: cascade diverges from order_protection at "
                f"inv={inv} adv={adv} sigma={sig}"
            )


def test_a_one_shot_period_renders_no_within_period_block():
    """F23's rule, applied to the MODE rather than to the branch.

    The within-period block (`phase`, `d`, `surplus`, `outstanding`) exists to
    describe what happens BETWEEN a period's two decisions. `order_protection`
    takes both at the pre-demand information set, so its period has no interior
    and the block would be structurally dead — the exact condition F23 removed
    it under. Gating on `alloc_enabled` alone would have rendered four dead
    features on every het board.
    """
    from adi_flex_gym import AdiFlexEnv
    from adi_flex_scenarios import SCENARIOS

    from adi_flex_gym import PROTECT_MODES, agent_steps_per_period

    sc = SCENARIOS["het_exp4"]
    two = AdiFlexEnv(scenario=sc, action_mode="seq_mask")
    width = 1 + two.pipe_slots + two.adv_slots + 1          # inv, pipe, adv, ttg
    assert two.two_phase
    assert two.observation_space.shape[0] == width + sc.T_dl + 1 + 3

    for mode in PROTECT_MODES:                 # every one-shot mode, not one
        one = AdiFlexEnv(scenario=sc, action_mode=mode, observation_mode="vec")
        assert not one.two_phase, mode
        assert one.observation_space.shape[0] == width, (
            f"{mode}: one-shot obs is {one.observation_space.shape[0]} wide, "
            f"expected {width} — the within-period block should not be rendered"
        )
        assert agent_steps_per_period(sc, mode) == 1, mode
        one.reset(seed=0)
        steps = 0
        done = False
        while not done:
            _, _, done, _, _ = one.step(one.action_space.sample())
            steps += 1
        assert steps == sc.N, f"{mode}: ran {steps} agent steps over {sc.N} periods"

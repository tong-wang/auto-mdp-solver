"""Execution semantics of the interpreter, on synthetic IRs.

These are the guarantees generated domain code must reproduce bit-for-bit, so
they are stated here against a fixture small enough that a failure points at
the engine rather than at a domain. The same guarantees are re-checked against
every real IR by :mod:`mdp_ir.laws`.
"""

from __future__ import annotations

import json

import pytest
from conftest import minimal_ir

from mdp_ir.interpreter import IrInterpreter, simulate
from mdp_ir.schema import MdpIR, load_ir

ACT = {"act": 4.0}


def build(doc: dict) -> MdpIR:
    return MdpIR.model_validate(doc)


# -- determinism and seeding -------------------------------------------------


def test_same_seed_replays_identically(ir_doc):
    ir = build(ir_doc)
    assert simulate(ir, 7, ACT).rows == simulate(ir, 7, ACT).rows


def test_different_seeds_diverge(ir_doc):
    ir = build(ir_doc)
    assert simulate(ir, 7, ACT).rows != simulate(ir, 8, ACT).rows


def test_random_policy_is_seed_deterministic(ir_doc):
    ir = build(ir_doc)
    assert simulate(ir, 3).rows == simulate(ir, 3).rows


def test_exogenous_draws_ignore_the_decision_path(ir_doc):
    """The flow source's settings reference no state, so its realized values
    must be identical under any decision sequence at the same seed."""
    ir = build(ir_doc)
    lo = simulate(ir, 5, {"act": 0.0})
    hi = simulate(ir, 5, {"act": 10.0})
    assert [r["outflow"] for r in lo.rows] == [r["outflow"] for r in hi.rows]


# -- scenario constants in control positions ---------------------------------


def test_horizon_constant_is_overridable_per_instance(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "n_periods", "value": 5, "axis": "sizing"})
    ir_doc["mdp"]["horizon"]["T"] = "n_periods"
    ir_doc["mdp"]["scenario"]["instances"]["short"] = {"n_periods": 2}
    ir = build(ir_doc)
    assert len(simulate(ir, 0, ACT).rows) == 5
    assert len(IrInterpreter(ir, instance="short").run(0, decisions=ACT).rows) == 2


def test_event_order_constant_reorders_execution(ir_doc):
    """A→B adds before draining; B→A drains before adding. With a floor on
    the level the two orders separate — without one, `+act -outflow` commutes
    and the order would be unobservable."""
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "evt", "value": ["A", "B"], "axis": "variant"})
    ir_doc["mdp"]["dynamics"]["event_sequence"] = "evt"
    ir_doc["mdp"]["scenario"]["instances"]["ba"] = {"evt": ["B", "A"]}
    ir_doc["mdp"]["dynamics"]["transitions"][1]["updates"] = [
        "outflow ~ flow.sample", "level -= outflow", "level = max(0.0, level)",
    ]
    ir = build(ir_doc)
    base = simulate(ir, 4, ACT).rows
    flipped = IrInterpreter(ir, instance="ba").run(4, decisions=ACT).rows
    assert [r["outflow"] for r in base] == [r["outflow"] for r in flipped]
    assert [r["level"] for r in base] != [r["level"] for r in flipped]


def test_instance_overrides_reach_the_objective(ir_doc):
    ir = build(ir_doc)
    base = simulate(ir, 1, ACT).total
    pricey = IrInterpreter(ir, instance="pricey").run(1, decisions=ACT).total
    assert pricey == pytest.approx(5.0 * base)


def test_unknown_instance_is_reported_with_the_alternatives(ir_doc):
    with pytest.raises(KeyError, match="pricey"):
        IrInterpreter(build(ir_doc), instance="nope").run(0)


# -- observation modes -------------------------------------------------------


def test_observation_mode_evaluates_refs_and_derived_features(ir_doc):
    ir_doc["gym"]["observation_modes"][0]["features"].append(
        {"derived": "headroom", "expr": "10 - level"})
    ir = build(ir_doc)
    row = simulate(ir, 2, ACT).rows[0]
    obs = IrInterpreter(ir).observe("vec", row)
    assert obs["level"] == row["level"]
    assert obs["info.outflow"] == row["outflow"]
    assert obs["headroom"] == pytest.approx(10 - row["level"])


# -- domain-owned expression builtins ----------------------------------------


def test_expr_builtin_resolves_next_to_the_ir(ir_doc, tmp_path):
    """Declared in the IR, implemented in a module beside it, resolved lazily
    — the portable-domain contract, with no repo-root assumption."""
    ir_doc["mdp"]["expr_builtins"] = [
        {"name": "double_it", "module": "tiny_test_builtins"}]
    comp = ir_doc["mdp"]["objective"]["per_step_components"][0]
    comp["expr"] = f"double_it({comp['expr']})"
    (tmp_path / "tiny_schema.json").write_text(json.dumps(ir_doc))
    (tmp_path / "tiny_test_builtins.py").write_text(
        "def double_it(x):\n    return 2 * x\n")

    base = simulate(build(minimal_ir()), 6, ACT).rows
    eb = simulate(load_ir(tmp_path / "tiny_schema.json"), 6, ACT).rows
    assert all(e["carry"] == pytest.approx(2 * b["carry"]) for e, b in zip(eb, base))


# -- world-layer mixtures ----------------------------------------------------


def test_mixture_episodes_equal_a_component_verbatim(ir_doc):
    """Standalone equivalence (spec §5.3): a mixture episode must be exactly
    what the drawn component would have produced on its own."""
    ir_doc["mdp"]["scenario"]["mixtures"] = [{
        "name": "mix", "substream_id": 3,
        "components": [[0.5, ""], [0.5, "pricey"]],
    }]
    ir = build(ir_doc)
    pure = {
        comp: {e: IrInterpreter(ir, instance=comp or None).run(e, decisions=ACT).rows
               for e in range(12)}
        for comp in ("", "pricey")
    }
    hit = {"": 0, "pricey": 0}
    mix = IrInterpreter(ir, instance="mix")
    for e in range(12):
        rows = mix.run(e, decisions=ACT).rows
        matched = [c for c in hit if rows == pure[c][e]]
        assert len(matched) == 1, f"episode {e} matched {matched}"
        hit[matched[0]] += 1
    assert all(n > 0 for n in hit.values()), f"one component never drawn: {hit}"


# -- list-valued world latents (the bandit shape) -----------------------------


def bandit_doc() -> dict:
    """A 3-armed Bernoulli bandit: the arm means are one `iid` sampler draw
    (a hidden latent VECTOR realized per episode), and the per-period pull is
    a scalar bernoulli whose `p` indexes that vector by the decision."""
    doc = minimal_ir()
    doc["mdp"]["scenario"]["constants"] += [
        {"name": "n_arms", "value": 3, "axis": "sizing"},
        {"name": "arm_means", "value": [0.5, 0.5, 0.5]},
    ]
    doc["mdp"]["scenario"]["samplers"] = [{
        "name": "world", "substream_id": 7, "hidden": True,
        "draws": [{"name": "arm_means", "distribution": {
            "family": "iid",
            "settings": {"of": "uniform", "size": "n_arms",
                         "low": 0.1, "high": 0.9},
        }}],
    }]
    doc["mdp"]["decisions"] = [{
        "name": "arm",
        "type": {"value": "discrete", "suggested": "discrete"},
        "dim": 1,
        "bounds": {"value": [0.0, 2.0], "suggested": [0.0, 2.0]},
    }]
    doc["mdp"]["uncertainty_sources"] = [{
        "name": "pull", "generator": "PullGenerator", "stream_id": 0,
        "distribution": {"family": "bernoulli",
                         "settings": {"p": "arm_means[arm]"}},
        "stages": [{"name": "sample", "realization": "period"}],
    }]
    doc["mdp"]["state_variables"][1] = {
        "name": "wins", "role": "core", "type": "float"}
    doc["mdp"]["info_fields"] = [
        {"name": "payout", "type": "int"},
        {"name": "econ", "type": "decomposition", "components": ["payoff", "total"]},
    ]
    doc["mdp"]["dynamics"] = {
        "event_sequence": ["A"],
        "transitions": [
            {"event": "A", "updates": ["payout ~ pull.sample", "wins += payout"]},
            {"event": "END_OF_PERIOD", "updates": ["period += 1"]},
        ],
    }
    doc["mdp"]["objective"] = {
        "sense": "maximize",
        "per_step_components": [{"name": "payoff", "expr": "payout"}],
    }
    doc["mdp"]["initial_state"] = {"wins": 0.0}
    doc["gym"]["observation_modes"] = [{
        "name": "vec", "default": True,
        "features": [{"ref": "wins"}, {"ref": "info.payout"}],
    }]
    doc["gym"]["action_modes"] = [{
        "name": "arm", "encodes": "arm", "type": "discrete",
        "bounds": [0.0, 2.0], "default": True,
    }]
    doc["gym"]["reward_modes"] = [{"name": "payoff", "expr": "total", "default": True}]
    doc["rl"]["requires_memory"] = {
        "value": True, "suggested": True, "source": "human_confirmed",
        "rationale": "hidden arm means must be inferred from pulls",
    }
    doc["rl"]["frame_stack"] = 4
    return doc


def test_latent_vector_realizes_per_episode_and_replays():
    ir = build(bandit_doc())
    interp = IrInterpreter(ir)
    first = interp.run(0, decisions={"arm": 0})
    means = interp.constants["arm_means"]
    assert len(means) == 3 and all(0.1 <= m <= 0.9 for m in means)
    assert means != [0.5, 0.5, 0.5]      # the placeholder was overwritten
    interp.run(1, decisions={"arm": 0})
    assert interp.constants["arm_means"] != means     # a fresh world per episode
    assert interp.run(0, decisions={"arm": 0}).rows == first.rows


def test_pull_thresholds_one_uniform_by_the_chosen_arm():
    """Period draws key on the period, not the action, so at one seed the
    bernoulli thresholds the SAME uniform by each arm's mean — a higher-mean
    arm dominates pointwise (common random numbers across arms), which is the
    decision-path-independence contract in bandit form."""
    interp = IrInterpreter(build(bandit_doc()))
    interp.run(0, decisions={"arm": 0})
    means = interp.constants["arm_means"]
    lo = means.index(min(means))
    hi = means.index(max(means))
    lo_pay = [r["payout"] for r in interp.run(0, decisions={"arm": lo}).rows]
    hi_pay = [r["payout"] for r in interp.run(0, decisions={"arm": hi}).rows]
    assert set(lo_pay + hi_pay) <= {0, 1}
    assert all(a <= b for a, b in zip(lo_pay, hi_pay))


def test_hidden_latent_vector_is_barred_from_observation():
    doc = bandit_doc()
    doc["gym"]["observation_modes"][0]["features"].append(
        {"derived": "cheat", "expr": "arm_means[arm]"})
    with pytest.raises(Exception, match="latent"):
        build(doc)


# -- declared invariants -----------------------------------------------------


def _with_invariants(doc: dict, *invs: dict) -> MdpIR:
    doc["mdp"]["invariants"] = list(invs)
    return build(doc)


def test_a_true_claim_produces_no_violations(ir_doc):
    ir = _with_invariants(
        ir_doc, {"name": "balance", "expr": "close(level, prev.level + act - outflow)"})
    assert simulate(ir, 9, ACT).violations == []


def test_a_false_claim_is_reported_on_every_row(ir_doc):
    """Unconditionally false, so the count is exact — a claim that merely
    *usually* fails would depend on what the source happened to draw."""
    ir = _with_invariants(
        ir_doc, {"name": "wrong", "expr": "close(level, level + 1)"})
    traj = simulate(ir, 9, ACT)
    assert len(traj.violations) == len(traj.rows)
    assert traj.violations[0].name == "wrong"


def test_violations_are_collected_not_raised(ir_doc):
    """Phase A must be able to show a failing claim next to the trajectory
    that broke it, so evaluation cannot abort the episode."""
    ir = _with_invariants(ir_doc, {"name": "boom", "expr": "level / 0 > 1"})
    traj = simulate(ir, 0, ACT)
    assert len(traj.rows) == 5
    assert "ZeroDivisionError" in traj.violations[0].detail


def test_prev_is_the_initial_state_on_the_first_period(ir_doc):
    ir = _with_invariants(
        ir_doc, {"name": "first", "expr": "t > 0 or close(prev.level, 0.0)"})
    assert simulate(ir, 1, ACT).violations == []


def test_prev_info_is_zero_on_the_first_period(ir_doc):
    ir = _with_invariants(
        ir_doc, {"name": "first_info", "expr": "t > 0 or close(prev.outflow, 0.0)"})
    assert simulate(ir, 1, ACT).violations == []


def test_terminal_scope_fires_only_on_the_last_row(ir_doc):
    ir = _with_invariants(
        ir_doc, {"name": "end", "expr": "False", "scope": "terminal"})
    traj = simulate(ir, 0, ACT)
    assert len(traj.violations) == 1
    assert traj.violations[0].period == traj.rows[-1]["t"]


def test_t_is_the_input_period_not_the_advanced_clock(ir_doc):
    """END_OF_PERIOD has already incremented the time-index variable by the
    time claims are evaluated; `t` is what the row is keyed by."""
    ir = _with_invariants(ir_doc, {"name": "clock", "expr": "period == t + 1"})
    assert simulate(ir, 0, ACT).violations == []


def test_close_honours_an_explicit_tolerance(ir_doc):
    ir = _with_invariants(
        ir_doc,
        {"name": "loose", "expr": "close(level, level + 0.001, 0.01)"},
        {"name": "tight", "expr": "close(level, level + 0.001, 1e-9)"},
    )
    names = {v.name for v in simulate(ir, 0, ACT).violations}
    assert names == {"tight"}


def test_checking_can_be_switched_off(ir_doc):
    ir = _with_invariants(ir_doc, {"name": "wrong", "expr": "False"})
    assert simulate(ir, 0, ACT, check_invariants=False).violations == []

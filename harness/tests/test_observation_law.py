"""The law in force is observable (upstream #77).

A catalog IR selects one candidate per uncertainty slot. A generalist trained
over a grid of laws must condition on *which* law it is being asked to adapt
to, and nothing could say it: the slot read-API was moment-only, a feature
expr was checked against the constant pool (a constant that parameterises one
candidate still has a value under every other one, so naming it is silently
wrong under the sibling), and the one identifier that looks like the answer —
the slot's own name — is the realized draw.

Three things close it, each pinned here:

* the registry derives ``support`` / ``probs`` for finite-support families
  (one-point law under a deterministic candidate, so a feature is defined
  under every selection), and composes them across a mixture as the marginal;
* an observation feature's ``slot.attr`` reads fold at load, like bounds —
  left symbolic, the interpreter would evaluate them at runtime against the
  realized draw, the same leak by a different door;
* the latent guard is about the realization (reading one of the issue's two):
  a hidden draw's name is still barred from a feature, a slot's law is its
  prior and passes through folded.
"""

from __future__ import annotations

import copy

import pytest
from conftest import minimal_catalog, minimal_ir

from mdp_ir import families, layering
from mdp_ir.interpreter import IrInterpreter, simulate
from mdp_ir.schema import MdpIR

IDENTITY = (lambda v, _attr=None: v)
ACT = {"act": 1.0}


def law(family: str, **settings):
    return families.law(family, settings, IDENTITY)


def resolved(doc: dict, **kw) -> MdpIR:
    return MdpIR.model_validate(layering.resolve_catalog(doc, **kw))


def bounded(doc: dict, entry: str, **kw) -> float:
    doc = copy.deepcopy(doc)
    for sv in doc["mdp"]["state_variables"]:
        if sv["name"] == "level":
            sv["bounds"] = [-1e6, f"1 + ({entry})"]
    return resolved(doc, **kw).mdp.state_bounds("level")[1] - 1.0


def _grid_doc() -> dict:
    """The motivating shape: one slot, a categorical law and a degenerate
    sibling, an instance selecting each — and a feature that reads the law."""
    doc = minimal_catalog()
    slot = doc["mdp"]["uncertainty_slots"][0]
    slot["candidates"]["cat"] = {
        "generator": "CategoricalFlow", "family": "categorical",
        "settings": {"values": [1, 2, 3], "probabilities": [0.25, 0.5, 0.25]},
        "desc": "three-point law",
    }
    doc["mdp"]["scenario"]["instances"].update(
        {"cat_c": {"flow": "cat"}, "fixed_c": {"flow": "fixed"}})
    doc["gym"]["observation_modes"][0]["features"].append(
        {"derived": "flow_law", "expr": "flow.probs", "dim": 3})
    return doc


def _feature_expr(ir: MdpIR, name: str = "flow_law") -> str:
    mode = ir.gym.observation_modes[0]
    return next(f.expr for f in mode.features if f.derived == name)


# -- the derivation -----------------------------------------------------------


def test_the_law_derives_for_finite_support_families():
    assert law("deterministic", value=2.0) == ([2.0], [1.0])
    assert law("categorical", values=[1, 2, 3], probabilities=[0.25, 0.5, 0.25]) \
        == ([1.0, 2.0, 3.0], [0.25, 0.5, 0.25])
    assert law("bernoulli", p=0.3) == ([0.0, 1.0], pytest.approx([0.7, 0.3]))
    assert law("iid", of="bernoulli", p=0.3, size=4)[0] == [0.0, 1.0]


def test_an_unbounded_family_has_no_law_and_names_the_hatch():
    """Poisson has no pmf to hand back as a vector; the refusal says what to
    reach for, rather than truncating somewhere and calling it the law."""
    with pytest.raises(families.FamilyError, match="read_api.probs"):
        law("poisson", rate=3.0)
    with pytest.raises(families.FamilyError, match="read_api.probs"):
        law("normal", mean=0.0, std=1.0)


def test_a_latent_setting_refuses_rather_than_reporting_the_placeholder():
    """The marginal law under a latent needs the latent's whole distribution,
    not a moment: it is not derivable family-generically, and the example
    value is a placeholder, not the prior."""
    with pytest.raises(families.FamilyError, match="marginal"):
        law("categorical", values=[1, 2, 3],
            probabilities={"draw": {"family": "normalized_uniform_weights",
                                    "settings": {"size": 3}},
                           "example": [0.25, 0.5, 0.25]})


# -- through the slot read-API ------------------------------------------------


def test_probs_and_support_are_read_api_attributes():
    assert "probs" in layering.READ_API and "support" in layering.READ_API
    doc = _grid_doc()
    assert bounded(doc, "sum(flow.probs)", instance="cat_c") == pytest.approx(1.0)
    assert bounded(doc, "flow.support[2]", instance="cat_c") == pytest.approx(3.0)


def test_the_degenerate_sibling_has_the_one_point_law():
    """Defined under EVERY candidate, so a mode reading the law needs no
    narrowing — the deterministic candidate answers with a point mass."""
    doc = _grid_doc()
    assert bounded(doc, "sum(flow.probs)", instance="fixed_c") == pytest.approx(1.0)
    assert bounded(doc, "flow.support[0]", instance="fixed_c") == pytest.approx(3.0)
    assert bounded(doc, "len(flow.probs)", instance="fixed_c") == pytest.approx(1.0)


def test_a_declared_law_still_wins_over_the_derivation():
    doc = _grid_doc()
    doc["mdp"]["uncertainty_slots"][0]["candidates"]["cat"]["read_api"] = {
        "probs": [0.1, 0.8, 0.1]}
    assert bounded(doc, "flow.probs[1]", instance="cat_c") == pytest.approx(0.8)
    assert bounded(doc, "flow.support[1]", instance="cat_c") == pytest.approx(2.0)


def test_an_unbounded_candidate_says_so_at_resolution():
    doc = _grid_doc()
    with pytest.raises(layering.LayeringError, match="read_api.probs"):
        bounded(doc, "sum(flow.probs)")          # default selection: poisson


# -- the observation feature --------------------------------------------------


def test_a_feature_folds_the_law_at_load_under_every_selection():
    """The feature carries the SELECTED candidate's law, as numbers, and the
    document validates — no `probs` identifier survives to the checker."""
    doc = _grid_doc()
    assert _feature_expr(resolved(doc, instance="cat_c")) == "([0.25, 0.5, 0.25])"
    assert _feature_expr(resolved(doc, instance="fixed_c")) == "([1.0])"


def test_the_folded_feature_is_a_law_not_a_draw():
    """Every row of an episode observes the same vector — the pmf — where the
    realized draw would have differed row to row."""
    ir = resolved(_grid_doc(), instance="cat_c")
    traj = simulate(ir, 3, ACT)
    seen = {tuple(IrInterpreter(ir, instance="cat_c").observe("vec", row)["flow_law"])
            for row in traj.rows}
    assert seen == {(0.25, 0.5, 0.25)}


def test_the_rest_of_the_expression_stays_symbolic():
    doc = _grid_doc()
    doc["gym"]["observation_modes"][0]["features"][-1]["expr"] = \
        "[p * h for p in flow.probs]"
    assert _feature_expr(resolved(doc, instance="cat_c")) == \
        "[p * h for p in ([0.25, 0.5, 0.25])]"
    ir = resolved(doc, instance="cat_c")
    row = simulate(ir, 1, ACT).rows[0]
    assert IrInterpreter(ir, instance="cat_c").observe("vec", row)["flow_law"] \
        == pytest.approx([0.25, 0.5, 0.25])      # h == 1.0 under cat_c


def test_a_resolved_ir_cannot_reach_the_law():
    """Only the resolver knows the selection, so a legacy resolved IR has no
    read-API at all: the reference is an unresolved identifier, never a
    runtime lookup against the draw."""
    doc = minimal_ir()
    doc["gym"]["observation_modes"][0]["features"].append(
        {"derived": "flow_law", "expr": "flow.probs"})
    with pytest.raises(ValueError, match="unresolved identifier"):
        MdpIR.model_validate(doc)


def test_the_selection_not_a_constant_decides_the_law():
    """Route 2 of the issue: a constant parameterising one candidate has a
    value under every other one. The law does not — under the degenerate
    sibling the feature is the point mass, not the categorical's vector."""
    doc = _grid_doc()
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "cat_probs", "value": [0.25, 0.5, 0.25], "axis": ""})
    doc["mdp"]["uncertainty_slots"][0]["candidates"]["cat"]["settings"]["probabilities"] = \
        "cat_probs"
    assert _feature_expr(resolved(doc, instance="cat_c")) == "([0.25, 0.5, 0.25])"
    assert _feature_expr(resolved(doc, instance="fixed_c")) == "([1.0])"


# -- the latent guard: about the realization, not the slot --------------------


def _hidden_rate_doc() -> dict:
    doc = minimal_catalog()
    doc["mdp"]["uncertainty_slots"][0]["candidates"]["poisson"]["settings"]["rate"] = {
        "draw": {"family": "uniform", "settings": {"low": 2.0, "high": 4.0}},
        "example": 3.0, "hidden": True,
    }
    doc["rl"]["requires_memory"] = {
        "value": True, "suggested": True, "source": "derived",
        "rationale": "hidden rate latent",
    }
    doc["rl"]["frame_stack"] = 4
    return doc


def test_a_hidden_draw_is_still_barred_from_a_feature():
    doc = _hidden_rate_doc()
    doc["gym"]["observation_modes"][0]["features"].append(
        {"derived": "leak", "expr": "flow_rate"})
    with pytest.raises(ValueError, match="latent"):
        resolved(doc)


def test_a_latent_slots_prior_passes_through_folded():
    """Reading one: the guard bars what the policy must infer, and the prior
    is not that. `flow.mean` composes through the latent (E[rate]) and folds
    to a number the guard has no reason to see."""
    doc = _hidden_rate_doc()
    doc["gym"]["observation_modes"][0]["features"].append(
        {"derived": "prior_mean", "expr": "flow.mean"})
    assert _feature_expr(resolved(doc), "prior_mean") == "(3.0)"


# -- mixtures -----------------------------------------------------------------


def _mixture_doc() -> dict:
    doc = _grid_doc()
    doc["mdp"]["scenario"]["mixtures"] = [{
        "name": "mix", "substream_id": 1,
        "components": [[0.5, "cat_c"], [0.5, "fixed_c"]],
        "desc": "half the three-point law, half a point mass at 3",
    }]
    return doc


def test_a_mixture_folds_the_marginal_law_over_the_union_support():
    """Nature picks the component per episode; before it does, the law in
    force is the mixture's marginal — the prior a policy may know. The point
    mass at 3 lands on the categorical's own support point."""
    doc = _mixture_doc()
    assert _feature_expr(resolved(doc, instance="mix")) == "([0.125, 0.25, 0.625])"
    # a bound resolves under the mixture AND under each divergent component
    # (their per-episode resolutions walk it too), so it is written to be
    # valid under the point mass as well
    assert bounded(doc, "flow.support[-1]", instance="mix") == pytest.approx(3.0)
    assert bounded(doc, "len(flow.support)", instance="mix") == pytest.approx(3.0)


# -- the declaration ----------------------------------------------------------


def test_a_feature_declares_its_width_and_a_mode_says_why():
    doc = _grid_doc()
    doc["gym"]["observation_modes"][0]["desc"] = "carries the law in force"
    ir = resolved(doc, instance="cat_c")
    mode = ir.gym.observation_modes[0]
    assert mode.desc == "carries the law in force"
    assert next(f.dim for f in mode.features if f.derived == "flow_law") == 3
    assert all(f.dim is None for f in mode.features if f.ref)


def test_a_width_below_one_is_rejected():
    doc = _grid_doc()
    doc["gym"]["observation_modes"][0]["features"][-1]["dim"] = 0
    with pytest.raises(ValueError):
        resolved(doc, instance="cat_c")


def test_a_folded_feature_moves_no_fingerprint():
    """The gym block sits outside the structural fingerprint; a mode that
    reads the law re-signs nothing."""
    plain = minimal_catalog()
    assert layering.structural_fingerprint(_grid_doc()) == \
        layering.structural_fingerprint(plain)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

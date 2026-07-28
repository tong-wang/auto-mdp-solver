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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

"""Bound construction: the spread a bound is built from (upstream #54).

`max` answers "how wide a box" at one fixed multiple — `_SIGMAS = 4` for every
unbounded family, at every site. That single number cannot say the two things a
bound needs to say:

* **the multiple belongs to the site.** An observation envelope is checked
  every period and a loose one costs nothing (SB3 leaves a non-image Box
  untouched); an action cap is consulted once an episode and a loose one is
  paid in exploration. Same draw, two exposures.
* **an aggregate is not a multiple of a per-period bound.** Over `n` periods
  the mean scales by `n` and the spread by `sqrt(n)`, so `n * max` inflates
  the tail term by `sqrt(n)` — and reads as obviously correct.

`sd` is what both need, so it joins the slot read API next to `mean`/`max`.
With #53 having made any expression legal at a bounds site, that is enough for
a domain to state its own envelope in the IR, with no per-domain Python.
"""

from __future__ import annotations

import copy
import math

import pytest
from conftest import minimal_catalog

from mdp_ir import families, layering
from mdp_ir.schema import MdpIR

IDENTITY = (lambda v, _attr=None: v)


def sd(family: str, **settings) -> float:
    return families.sd(family, settings, IDENTITY)


def resolved(doc: dict, **kw) -> MdpIR:
    return MdpIR.model_validate(layering.resolve_catalog(doc, **kw))


def bounded(doc: dict, entry: str, **kw) -> float:
    """Upper bound of `level` when its envelope is `entry`, resolved."""
    doc = copy.deepcopy(doc)
    for sv in doc["mdp"]["state_variables"]:
        if sv["name"] == "level":
            sv["bounds"] = [-1e6, f"1 + ({entry})"]
    return resolved(doc, **kw).mdp.state_bounds("level")[1] - 1.0


# -- the derivations ---------------------------------------------------------


@pytest.mark.parametrize("family,settings,expected", [
    ("deterministic", {"value": 5.0}, 0.0),
    ("poisson", {"rate": 6.0}, math.sqrt(6.0)),
    ("normal", {"mean": 0.0, "std": 2.5}, 2.5),
    ("uniform", {"low": 0.0, "high": 12.0}, 12.0 / math.sqrt(12.0)),
    ("gamma", {"shape": 9.0, "scale": 2.0}, 3.0 * 2.0),
    ("exponential", {"scale": 4.0}, 4.0),
    ("bernoulli", {"p": 0.25}, math.sqrt(0.25 * 0.75)),
    ("binomial", {"n": 10.0, "p": 0.4}, math.sqrt(10 * 0.4 * 0.6)),
    ("beta", {"a": 2.0, "b": 3.0}, math.sqrt(6 / (25 * 6))),
    ("geometric", {"p": 0.2}, math.sqrt(0.8) / 0.2),
    ("negative_binomial", {"n": 5.0, "p": 0.4}, math.sqrt(5 * 0.6) / 0.4),
])
def test_sd_matches_the_closed_form(family, settings, expected):
    assert sd(family, **settings) == pytest.approx(expected)


def test_categorical_sd_is_the_weighted_spread():
    got = sd("categorical", values=[0.0, 10.0, 20.0],
             probabilities=[0.5, 0.25, 0.25])
    assert got == pytest.approx(math.sqrt(68.75))


def test_lognormal_sd_is_the_marginal_not_the_log_scale():
    m, s = 1.0, 0.5
    want = math.sqrt(math.exp(s * s) - 1.0) * math.exp(m + s * s / 2.0)
    assert sd("lognormal", mean=m, sigma=s) == pytest.approx(want)


def test_an_iid_vector_takes_its_base_family_sd():
    assert sd("iid", of="poisson", size=4, rate=9.0) == pytest.approx(3.0)


def test_an_inid_vector_refuses_and_names_the_aggregate():
    """Same stance `mean` takes: the sd of an `independent` vector is a vector,
    and the scalar a bound wants is the sd of the SUM — which is neither the
    max nor the average of the components'."""
    with pytest.raises(families.FamilyError, match="sqrt\\(sum"):
        sd("independent", of="poisson", size=2, rate=[1.0, 2.0])


def test_a_latent_setting_refuses_rather_than_understating():
    """`mean` and `max` compose through a latent; spread does not — the
    marginal carries the latent's own variance (law of total variance). A
    conditional sd would be too small exactly where uncertainty is largest."""
    with pytest.raises(families.FamilyError, match="world latent"):
        sd("poisson", rate={"draw": {"family": "gamma",
                                     "settings": {"shape": 9.0, "scale": 3.0}},
                            "example": 27.0})


def test_an_underivable_family_says_so():
    with pytest.raises(families.FamilyError, match="no sd derivation"):
        sd("weibull", a=1.5)


# -- through the slot read API -----------------------------------------------


def test_sd_is_a_read_api_attribute(catalog_doc):
    assert "sd" in layering.READ_API
    assert bounded(catalog_doc, "flow.sd") == pytest.approx(math.sqrt(3.0))


def test_the_shipped_envelope_is_mean_plus_four_sd(catalog_doc):
    """The decomposition that makes the multiple a parameter: `max` IS
    `mean + 4*sd` for an unbounded family, so a site that wants six sigma can
    now write it instead of inheriting four."""
    assert bounded(catalog_doc, "flow.mean + 4 * flow.sd") \
        == pytest.approx(bounded(catalog_doc, "flow.max"))
    assert bounded(catalog_doc, "flow.mean + 6 * flow.sd") \
        > bounded(catalog_doc, "flow.max")


def test_the_aggregate_bound_is_expressible_and_tighter(catalog_doc):
    """`n * max` vs the bound on the sum. Both are declarations in the IR now
    (#53); this is the one that is right."""
    n = 30
    naive = bounded(catalog_doc, f"{n} * flow.max")
    correct = bounded(catalog_doc, f"{n} * flow.mean + 4 * sqrt({n}) * flow.sd")
    assert correct == pytest.approx(n * 3.0 + 4 * math.sqrt(n * 3.0))
    assert naive > 2 * correct          # the sqrt(n) inflation, measured


def test_an_unknown_read_api_attribute_still_lists_what_exists(catalog_doc):
    with pytest.raises(layering.LayeringError, match="no read-API attribute"):
        bounded(catalog_doc, "flow.variance")


# -- the escape hatch outside the derived set (upstream #77) -----------------


def _law_doc(on: str = "poisson") -> dict:
    """The catalog fixture with one candidate declaring stats the registry
    has no notion of — quantiles and an entropy. (The law itself, `probs` /
    `support`, was the motivating case and has since joined the derived
    vocabulary; see test_observation_law.py.)"""
    doc = minimal_catalog()
    doc["mdp"]["uncertainty_slots"][0]["candidates"][on]["read_api"] = {
        "quantiles": [1.0, 2.0, 3.0],
        "entropy": 1.04,
    }
    doc["mdp"]["scenario"]["instances"].update(
        {"poisson_c": {"flow": "poisson"}, "fixed_c": {"flow": "fixed"}})
    return doc


def test_a_candidate_can_declare_a_stat_the_registry_cannot_derive():
    """The defect: `read_api` is documented as the escape hatch for what the
    family registry cannot derive, but the READ_API membership test ran BEFORE
    the override was consulted — so the hatch reached only the five attributes
    that never need it, and a law could not be named at all."""
    doc = _law_doc()
    assert "quantiles" not in layering.READ_API
    assert bounded(doc, "sum(flow.quantiles)", instance="poisson_c")     == pytest.approx(6.0)
    assert bounded(doc, "flow.entropy", instance="poisson_c")            == pytest.approx(1.04)


def test_a_declared_stat_folds_as_a_vector_into_a_bound():
    doc = _law_doc()
    assert bounded(doc, "flow.quantiles[0] + flow.quantiles[2]", instance="poisson_c") == pytest.approx(4.0)


def test_a_sibling_that_declares_nothing_fails_loudly_rather_than_wrongly():
    """The trap the hatch has to avoid being: under a candidate that does not
    declare the law, the reference must not resolve to something that is not
    this law. It raises, and names whose vocabulary it checked."""
    doc = _law_doc(on="poisson")
    with pytest.raises(layering.LayeringError) as exc:
        bounded(doc, "sum(flow.quantiles)", instance="fixed_c")
    msg = str(exc.value)
    assert "no read-API attribute" in msg
    assert "declares no read_api block" in msg


def test_the_derived_vocabulary_is_unchanged_and_typos_still_caught():
    """Consulting the override first must not open the namespace: a name
    neither derived nor declared is still an error, under both candidates."""
    doc = _law_doc()
    for inst in ("poisson_c", "fixed_c"):
        with pytest.raises(layering.LayeringError, match="no read-API attribute"):
            bounded(doc, "flow.variance", instance=inst)
    # and a derived attribute still derives on the candidate that also declares
    assert bounded(doc, "flow.mean", instance="poisson_c") == pytest.approx(3.0)


def _law_mixture_doc(both: bool) -> dict:
    doc = _law_doc()
    if both:
        doc["mdp"]["uncertainty_slots"][0]["candidates"]["fixed"]["read_api"] = {
            "quantiles": [30.0], "entropy": 0.0,
        }
    doc["mdp"]["uncertainty_slots"][0]["candidates"]["fixed"]["settings"] = \
        {"value": "10 * rate"}
    doc["mdp"]["scenario"]["mixtures"] = [{
        "name": "mix", "substream_id": 1,
        "components": [[0.5, "poisson_c"], [0.5, "fixed_c"]],
        "desc": "half Poisson(3), half a point mass at 30",
    }]
    return doc


def test_a_declared_stat_does_not_compose_across_a_mixture():
    """A mixture's envelope rules cover the derived five, each with a §5.3
    composition (max of maxes, law of total variance). A declared attribute is
    a statement about one law and has no such rule, so taking either
    component's answer for the mixture would be a fiction — refused, with the
    reason, even when every component can answer."""
    with pytest.raises(layering.LayeringError, match="does not compose"):
        bounded(_law_mixture_doc(both=True), "sum(flow.quantiles)", instance="mix")


def test_a_mixture_component_that_cannot_answer_fails_at_its_own_resolution():
    """The earlier guard: a mixture resolves each component in turn, so a
    component whose candidate declares nothing fails there — before any
    composition question is reached."""
    with pytest.raises(layering.LayeringError) as exc:
        bounded(_law_mixture_doc(both=False), "sum(flow.quantiles)", instance="mix")
    assert "declares no read_api block" in str(exc.value)


def test_the_derived_five_still_compose_across_a_mixture():
    """The declared-attribute refusal must not touch the envelope rules."""
    doc = _law_mixture_doc(both=True)
    assert bounded(doc, "flow.mean", instance="mix") \
        == pytest.approx(0.5 * 3.0 + 0.5 * 30.0)


# -- under a mixture ---------------------------------------------------------


def _mixture_doc() -> dict:
    doc = minimal_catalog()
    doc["mdp"]["uncertainty_slots"][0]["candidates"]["fixed"]["settings"] = \
        {"value": "10 * rate"}
    doc["mdp"]["scenario"]["instances"].update(
        {"poisson_c": {"flow": "poisson"}, "fixed_c": {"flow": "fixed"}})
    doc["mdp"]["scenario"]["mixtures"] = [{
        "name": "mix", "substream_id": 1,
        "components": [[0.5, "poisson_c"], [0.5, "fixed_c"]],
        "desc": "half Poisson(3), half a point mass at 30",
    }]
    return doc


def test_a_mixture_sd_carries_the_between_component_spread():
    """A mixture is wider than its components: the §5.3 envelope for spread is
    the law of total variance, not a weighted mean of the sds — which here
    would report 0.87 for a distribution straddling 3 and 30."""
    doc = _mixture_doc()
    m1, m2, s1 = 3.0, 30.0, math.sqrt(3.0)
    mu = 0.5 * m1 + 0.5 * m2
    want = math.sqrt(0.5 * (s1 ** 2 + m1 ** 2) + 0.5 * m2 ** 2 - mu ** 2)
    assert bounded(doc, "flow.sd", instance="mix") == pytest.approx(want)
    assert bounded(doc, "flow.sd", instance="mix") > 10 * (0.5 * s1 + 0.5 * 0.0)


def test_a_mixture_component_keeps_its_own_sd():
    doc = _mixture_doc()
    assert bounded(doc, "flow.sd", instance="poisson_c") \
        == pytest.approx(math.sqrt(3.0))
    assert bounded(doc, "flow.sd", instance="fixed_c") == pytest.approx(0.0)


# -- the rationale gate ------------------------------------------------------


def _domain(tmp_path, doc):
    import json

    from mdp_conformance.loader import DomainHandle

    (tmp_path / "d_schema.json").write_text(json.dumps(doc))
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


def _with_rationale(ir_doc: dict, why: str) -> dict:
    doc = copy.deepcopy(ir_doc)
    doc["mdp"]["decisions"][0]["bounds"]["rationale"] = why
    doc["mdp"]["decisions"][0]["bounds"]["source"] = "human_confirmed"
    return doc


def test_a_bound_justified_by_solved_runs_warns(tmp_path, ir_doc):
    """The shape `adi_flex` shipped, quoted in #54: every quantity in it is an
    observation of instances already solved, and it reads as diligence."""
    from mdp_conformance.checks import check_bound_rationale

    doc = _with_rationale(ir_doc, "user-confirmed cap with >=2x headroom over "
                                  "the largest optimal order observed in the DP "
                                  "tables; must never bind at the optimum")
    result = check_bound_rationale(_domain(tmp_path, doc))
    assert result.status == "WARN"
    assert "nobody has run" in result.detail


def test_a_bound_derived_from_the_dynamics_passes(tmp_path, ir_doc):
    """The same cap, argued from the model: true of instances nobody has run,
    and it never needs revisiting when the design set grows."""
    from mdp_conformance.checks import check_bound_rationale

    doc = _with_rationale(ir_doc, "bounds the horizon's total demand: "
                                  "T*rate + 4*sqrt(T*rate); no order beyond it "
                                  "can be useful under any policy")
    assert check_bound_rationale(_domain(tmp_path, doc)).status == "PASS"


def test_the_word_optimal_alone_is_not_the_tell(tmp_path, ir_doc):
    """A claim about what can never be optimal is a statement about the model.
    Flagging it would train authors to launder the vocabulary."""
    from mdp_conformance.checks import check_bound_rationale

    doc = _with_rationale(ir_doc, "an order beyond the horizon's remaining "
                                  "demand can never be optimal")
    assert check_bound_rationale(_domain(tmp_path, doc)).status == "PASS"

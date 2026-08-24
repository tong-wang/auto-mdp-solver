"""§5.0 at its real scope: the IR holds no bare design value (upstream #68).

The width-site rule always said a literal for a model-declared quantity is a
defect. These cover the two sites where obeying it was impossible until the
schema widened — `objective.discount_factor` and `benchmarks[].tolerance` —
and the line that keeps the rule from becoming "name every numeral".
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mdp_ir.schema import MdpIR

from conftest import minimal_ir


def _ir(beta=None, consts=None, instances=None, benchmarks=None) -> MdpIR:
    doc = minimal_ir()
    if beta is not None:
        doc["mdp"]["objective"]["discount_factor"] = beta
    if consts:
        doc["mdp"]["scenario"]["constants"] += consts
    if instances:
        doc["mdp"]["scenario"]["instances"].update(instances)
    if benchmarks is not None:
        doc["benchmarks"] = benchmarks
    return MdpIR.model_validate(doc)


# --- the discount ---------------------------------------------------------

def test_a_literal_beta_behaves_exactly_as_before():
    assert _ir(beta=0.95).mdp.discount_factor() == 0.95
    assert _ir().mdp.discount_factor() == 1.0          # the default


def test_beta_may_name_a_scenario_constant_and_an_instance_may_move_it():
    """The point of the change: two renderings of one model side by side.

    Before, β was a value inside the frozen core, so the second frame could
    only be had by re-basing the first."""
    ir = _ir(beta="beta",
             consts=[{"name": "beta", "value": 0.95}],
             instances={"undiscounted": {"beta": 1.0}})
    assert ir.mdp.discount_factor() == 0.95
    assert ir.mdp.discount_factor("undiscounted") == 1.0


def test_naming_beta_leaves_the_model_fingerprint_still():
    """The theory is unchanged by how the rendering names its discount."""
    lit = _ir(beta=0.95)
    sym = _ir(beta="beta", consts=[{"name": "beta", "value": 0.95}])
    assert lit.model_fingerprint() == sym.model_fingerprint()


def test_the_bound_is_enforced_on_a_literal_at_parse_time():
    with pytest.raises(ValidationError, match=r"not in \(0, 1\]"):
        _ir(beta=1.5)


def test_the_bound_is_enforced_on_a_resolved_override():
    """A symbol has no value until an instance picks one, so the (0, 1] bound
    moves to the moment it resolves — which is the first point it can be
    wrong, not a weakening."""
    ir = _ir(beta="beta",
             consts=[{"name": "beta", "value": 0.95}],
             instances={"bad": {"beta": 1.5}})
    assert ir.mdp.discount_factor() == 0.95
    with pytest.raises(ValueError, match=r"resolved to 1.5"):
        ir.mdp.discount_factor("bad")


def test_a_symbol_naming_nothing_is_reported_not_silently_defaulted():
    ir = _ir(beta="nope")
    with pytest.raises(KeyError, match="not a scenario constant"):
        ir.mdp.discount_factor()


def test_a_non_identifier_string_is_refused_at_parse_time():
    with pytest.raises(ValidationError, match="neither a number nor"):
        _ir(beta="0.95 * 2")


# --- the tolerance --------------------------------------------------------

def _bench(tol):
    return [{"name": "dp", "role": "exact", "tolerance": tol}]


def test_tolerance_may_name_a_constant_and_resolve_per_instance():
    """One solver file, two renderings: exact on a lattice, approximate on a
    binned density. The band the §9.9 gate takes belongs to the rendering
    being judged, not to the entry."""
    ir = _ir(consts=[{"name": "dp_slack", "value": 0.001}],
             instances={"gamma_cell": {"dp_slack": 0.5}},
             benchmarks=_bench("dp_slack"))
    assert ir.benchmark_tolerance("dp") == 0.001
    assert ir.benchmark_tolerance("dp", "gamma_cell") == 0.5


def test_a_literal_tolerance_behaves_exactly_as_before():
    assert _ir(benchmarks=_bench(0.001)).benchmark_tolerance("dp") == 0.001


def test_a_negative_resolved_tolerance_is_refused():
    ir = _ir(consts=[{"name": "slack", "value": -1.0}], benchmarks=_bench("slack"))
    with pytest.raises(ValueError, match="< 0"):
        ir.benchmark_tolerance("dp")


def test_adoption_moves_the_freeze_token_but_never_the_theory():
    """What adoption actually costs, measured rather than assumed.

    `Benchmark` is root-level and outside the mdp block, so it is tempting to
    call the tolerance half fingerprint-neutral. It is not: the constant the
    symbol names has to be declared in `mdp.scenario.constants`, which IS
    inside the block, so the freeze token moves either way — once, at adoption,
    with an §IR-CHANGELOG entry (guide §7 rule 7). What never moves is the
    model fingerprint: the theory is unchanged by how the rendering names its
    discount or its slack.
    """
    lit = _ir(beta=0.95, benchmarks=_bench(0.001))
    sym = _ir(beta="beta", benchmarks=_bench("dp_slack"),
              consts=[{"name": "beta", "value": 0.95},
                      {"name": "dp_slack", "value": 0.001}])
    assert lit.mdp_fingerprint() != sym.mdp_fingerprint()
    assert lit.model_fingerprint() == sym.model_fingerprint()


def test_the_second_rendering_is_free_once_the_choice_is_named():
    """The payoff, and it lands on the RENDERING identity rather than on the
    freeze token: `structural_fingerprint` excludes the scenario node, so the
    beta = 1 twin is an instance override that moves nothing. Before, beta was
    a value inside the frozen core — the second frame could only be had by
    re-basing the first."""
    import copy

    from conftest import minimal_catalog
    from mdp_ir.layering import structural_fingerprint

    base = minimal_catalog()
    base["mdp"]["objective"]["discount_factor"] = "beta"
    base["mdp"]["scenario"]["constants"].append({"name": "beta", "value": 0.95})
    twin = copy.deepcopy(base)
    twin["mdp"]["scenario"]["instances"]["undiscounted"] = {"beta": 1.0}
    assert structural_fingerprint(base) == structural_fingerprint(twin)


# --- identity vs choice ---------------------------------------------------

def test_identities_are_not_findings():
    """The half of the rule that keeps it from becoming ceremony: an
    undiscounted beta and a zero tolerance name nothing, because there is no
    degree of freedom to name. `examples/mab` is the live case — it declares
    discount_factor in its model layer AND renders it 1.0, correctly."""
    from mdp_conformance.checks import _hardcode_findings

    doc = minimal_ir()
    doc["mdp"]["model"] = {"quantities": {"discount_factor": {"domain": "real in [0, 1]"}}}
    doc["mdp"]["objective"]["discount_factor"] = 1.0
    doc["benchmarks"] = _bench(0.0)
    findings, sites = _hardcode_findings(doc)
    assert findings == []
    assert sites == 2


def test_a_non_identity_literal_is_a_finding_and_a_declaration_aggravates_it():
    from mdp_conformance.checks import _hardcode_findings

    doc = minimal_ir()
    doc["mdp"]["model"] = {"quantities": {"discount_factor": {"domain": "real in [0, 1]"}}}
    doc["mdp"]["objective"]["discount_factor"] = 0.95
    findings, _ = _hardcode_findings(doc)
    assert len(findings) == 1
    assert "mdp.model declares the quantity" in findings[0]


def test_a_symbol_is_never_a_finding():
    from mdp_conformance.checks import _hardcode_findings

    doc = minimal_ir()
    doc["mdp"]["objective"]["discount_factor"] = "beta"
    doc["mdp"]["scenario"]["constants"].append({"name": "beta", "value": 0.95})
    doc["benchmarks"] = _bench("dp_slack")
    assert _hardcode_findings(doc)[0] == []

"""`independent` — iid's non-identically-distributed sibling (upstream #49).

`iid` gives one draw statement a vector value, but resolves ONE settings dict
shared by every component, so it cannot express independent-but-not-identical
ones. The workaround is to enumerate `size` draw statements in the dynamics
body and smuggle the component index into the seed key through a local, which
costs the `path_independence` law.

The two families stay apart deliberately: `iid`'s moments legitimately drop
`size` because its components agree, and `independent`'s cannot.
"""

import numpy as np
import pytest

from mdp_ir import families
from mdp_ir.runtime import sample_family

_IDENT = lambda v, attr: v  # noqa: E731 — settings already concrete


def _rng(seed=11):
    return np.random.default_rng(seed)


# --- sampling ---------------------------------------------------------------


def test_components_are_drawn_in_index_order_from_one_rng():
    """The declared semantics: per-component draws, index order, one stream.

    Stated in the spec rather than left to numpy, so the IR's meaning does not
    depend on whether an implementation vectorizes.
    """
    got = sample_family(_rng(), "independent",
                        {"of": "poisson", "size": 3, "rate": [2, 1, 3]})
    r = _rng()
    want = [sample_family(r, "poisson", {"rate": k}) for k in (2, 1, 3)]
    assert got == want


def test_all_scalar_settings_reproduce_iid_bit_for_bit():
    """The degenerate case is not merely semantic — a domain migrating from
    `iid` to `independent` keeps its numbers, and the stream position too."""
    a = _rng(5)
    from_iid = sample_family(a, "iid", {"of": "poisson", "size": 4, "rate": 2.0})
    after_a = a.random()

    b = _rng(5)
    from_ind = sample_family(b, "independent",
                             {"of": "poisson", "size": 4, "rate": 2.0})
    after_b = b.random()

    assert from_iid == from_ind
    assert after_a == after_b


def test_scalars_broadcast_alongside_vectors():
    got = sample_family(_rng(3), "independent",
                        {"of": "normal", "size": 3,
                         "mean": [0.0, 10.0, 100.0], "std": 1.0})
    assert len(got) == 3
    # each component sits near its own mean, so the vector setting is being
    # consumed positionally rather than broadcast from entry 0
    assert got[0] < got[1] < got[2]


def test_an_expression_setting_is_resolved_once_not_per_component():
    calls = []

    def counting_resolve(value, *_):
        calls.append(value)
        return [2, 1, 3] if value == "lambda_seg" else value

    sample_family(_rng(), "independent",
                  {"of": "poisson", "size": 3, "rate": "lambda_seg"},
                  counting_resolve)
    assert calls.count("lambda_seg") == 1


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize("rate,size", [([2, 1], 3), ([2, 1, 3, 4], 3)])
def test_wrong_length_vector_is_an_error_never_recycled_or_truncated(rate, size):
    with pytest.raises(families.FamilyError, match="entries but size is"):
        sample_family(_rng(), "independent",
                      {"of": "poisson", "size": size, "rate": rate})


@pytest.mark.parametrize("of", ["iid", "independent", 7, None])
def test_of_must_name_a_base_family(of):
    with pytest.raises(families.FamilyError, match="must name a base family"):
        families.independent_parts({"of": of, "size": 2})


# --- read API ---------------------------------------------------------------


def test_envelopes_reduce_over_components():
    """What a bound site means: sized to hold ANY component."""
    s = {"of": "poisson", "size": 3, "rate": [2, 1, 3]}
    per_component = [families.max_value("poisson", {"rate": k}, _IDENT)
                     for k in (2, 1, 3)]
    assert families.max_value("independent", s, _IDENT) == max(per_component)
    assert families.min_value("independent", s, _IDENT) == 0.0


def test_mean_refuses_rather_than_guessing():
    """E[X] of an inid vector is a vector. The scalar a bound usually wants is
    the SUM over components (rates (2,1,3) -> 6, not 2), so a guess would size
    a bound a third too small with nothing to say so."""
    s = {"of": "poisson", "size": 3, "rate": [2, 1, 3]}
    with pytest.raises(families.FamilyError, match="no scalar mean"):
        families.mean("independent", s, _IDENT)


def test_envelope_validates_lengths_too():
    """The check runs wherever settings first resolve — including moment
    derivation at load, which is well before any episode."""
    s = {"of": "poisson", "size": 3, "rate": [2, 1]}
    with pytest.raises(families.FamilyError, match="entries but size is"):
        families.max_value("independent", s, _IDENT)


def test_is_discrete_is_undelegated_like_iid():
    """Recorded, not asserted as good: `is_discrete` takes only the family
    name, so neither wrapper can consult its base. Every shipped candidate
    declares `is_discrete` explicitly, which is the working path."""
    assert families.is_discrete("independent") is families.is_discrete("iid") is None


# --- stage level, end to end (upstream #52) ---------------------------------
#
# The v0.9.9 tests above drive `sample_family` and `families` directly, so they
# never loaded an IR — and the stage-level path was broken for BOTH wrappers
# while the docs said it was legal. `examples/mab`'s `iid` is a setting-level
# draw spec, which desugars into a sampler at load and never reaches the
# validator's settings loop, so nothing else covered it either.


def _stage_ir(ir_doc, family, settings):
    """The synthetic IR with its one source rendered as a wrapper family."""
    ir_doc["mdp"]["uncertainty_sources"][0]["distribution"] = {
        "family": family, "settings": settings,
    }
    return ir_doc


@pytest.mark.parametrize("family", ["iid", "independent"])
def test_a_wrapper_family_loads_as_a_stage_family(ir_doc, tmp_path, family):
    """`of` names a base family, not a value: validating it as an expression
    reported `poisson` as an unresolved identifier and no such IR could load."""
    import json

    from mdp_ir.schema import load_ir

    doc = _stage_ir(ir_doc, family, {"of": "poisson", "size": "2", "rate": "rate"})
    p = tmp_path / "stage.json"
    p.write_text(json.dumps(doc))
    ir = load_ir(p)
    assert ir.mdp.uncertainty_sources[0].distribution.family == family


@pytest.mark.parametrize("bad_key,settings", [
    ("size", {"of": "poisson", "size": "nope + 1", "rate": "rate"}),
    ("rate", {"of": "poisson", "size": "2", "rate": "no_such_constant"}),
])
def test_the_non_structural_settings_are_still_validated(
    ir_doc, tmp_path, bad_key, settings
):
    """Exempting `of` must not exempt the rest. `size` in particular is an
    ordinary expression over constants, not a structural key."""
    import json

    import pytest as _pytest

    from mdp_ir.schema import load_ir

    p = tmp_path / "stage.json"
    p.write_text(json.dumps(_stage_ir(ir_doc, "independent", settings)))
    with _pytest.raises(ValueError, match=rf"settings\['{bad_key}'\]"):
        load_ir(p)


def test_structural_keys_come_from_one_table():
    """The validator and the runtime split on the same contract — they
    disagreed, which is the whole of #52."""
    assert families.structural_settings("iid") == {"of"}
    assert families.structural_settings("independent") == {"of"}
    assert families.structural_settings("poisson") == frozenset()
    # `size` is deliberately NOT structural: it is an expression
    assert "size" not in families.structural_settings("independent")


def test_a_stage_family_draws_one_component_per_element(ir_doc, tmp_path):
    """Loads AND draws: the whole claim `MDP_IR_SAMPLE.md` makes for the
    stage-level path, which nothing exercised end to end before #52."""
    import json

    from mdp_ir.interpreter import IrInterpreter
    from mdp_ir.schema import load_ir

    doc = _stage_ir(ir_doc, "independent",
                    {"of": "poisson", "size": "3", "rate": "rates"})
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "rates", "value": [2.0, 1.0, 3.0], "desc": "per-component rates"}
    )
    # the draw target is a vector now, so the body consumes it as one
    for t in doc["mdp"]["dynamics"]["transitions"]:
        t["updates"] = [u.replace("level -= outflow", "level -= sum(outflow)")
                        for u in t["updates"]]
    p = tmp_path / "stage.json"
    p.write_text(json.dumps(doc))

    rows = IrInterpreter(load_ir(p), seed_salt=1).run(0).rows
    drawn = [r["outflow"] for r in rows if isinstance(r.get("outflow"), list)]
    assert drawn, "the stage never produced a vector draw"
    assert all(len(v) == 3 for v in drawn)          # one component per rate
    assert all(all(float(x).is_integer() for x in v) for v in drawn)   # poisson

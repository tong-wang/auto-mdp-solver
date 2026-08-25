"""An action bound: what may be declared, and how it reads back (#69, #70).

Two halves of one gap, found by the same campaign and shipped together.

**#69** — the non-degeneracy check ran without knowing whether the bound it
checked was discrete or continuous, so `[a, a]` was refused everywhere. For a
continuous bound that is right: a point is not a usable interval. For a
discrete one it is wrong — `[0, 0]` is `Discrete(1)`, the honest declaration of
an action that exists with exactly one legal value on this instance. §5.0 and
#53/#54 push a bound toward being stated as its *derivation*, and a derivation
correct across a family will legitimately collapse on a member of it, so the
old rule taxed exactly the files that took the guidance.

**#70** — the resolution was performed on every load and thrown away: the root
validator resolved each action-mode bound per instance, but only `decisions`
and `state_variables` had a reader. A domain gating its rendered action space
had to re-implement the namespace rule, which is #53's defect one layer up.

Both are two-sided here: what the old rule wrongly refused must now pass, and
what it rightly refused must still fail.
"""

from __future__ import annotations

import copy

import pytest
from conftest import minimal_catalog, minimal_ir
from pydantic import ValidationError

from mdp_ir import layering
from mdp_ir.schema import MdpIR


@pytest.fixture
def ir_doc() -> dict:
    return minimal_ir()


def with_action(doc: dict, bounds, *, type_="discrete", consts=(), instances=None) -> dict:
    """Put one [lo, hi] on both the decision and the mode that encodes it —
    the two sites the same space is declared at."""
    doc = copy.deepcopy(doc)
    d = doc["mdp"]["decisions"][0]
    d["type"] = {"value": type_, "suggested": type_}
    d["bounds"] = {"value": list(bounds), "suggested": list(bounds)}
    m = doc["gym"]["action_modes"][0]
    m["type"] = type_
    m["bounds"] = list(bounds)
    doc["mdp"]["scenario"]["constants"].extend(consts)
    if instances is not None:
        doc["mdp"]["scenario"]["instances"] = instances
    return doc


# -- #69: a discrete space of width one --------------------------------------


def test_a_discrete_action_may_declare_one_legal_value(ir_doc):
    """`Discrete(1)`, spelled as a literal."""
    m = MdpIR.model_validate(with_action(ir_doc, [0, 0]))
    assert m.mdp.decision_bounds("act") == (0.0, 0.0)
    assert m.action_bounds("act") == (0.0, 0.0)


def test_a_continuous_action_still_needs_a_non_empty_interval(ir_doc):
    """Unchanged: `[a, a]` is a point, `lo > hi` is empty, neither is a Box."""
    with pytest.raises(ValidationError, match="lo < hi"):
        MdpIR.model_validate(with_action(ir_doc, [0.0, 0.0], type_="continuous"))


def test_an_inverted_discrete_bound_is_still_rejected(ir_doc):
    """The mis-resolution the check exists to catch — a swapped pair, a
    constant read from the wrong place — fails for discrete too."""
    with pytest.raises(ValidationError, match="lo <= hi"):
        MdpIR.model_validate(with_action(ir_doc, [3, 1]))


def test_a_derivation_collapsing_on_one_instance_is_accepted(ir_doc):
    """The shape the campaign hit: one symbolic cap, correct across the family
    *including* where it returns zero, and 0 on the member where the quantity
    it caps cannot exist. Under the old rule the only way through was
    `max(1, <derivation>)` — a padded action mirrored into the gym and a
    paragraph of `desc` disowning part of the bound."""
    doc = with_action(ir_doc, [0, "cap"],
                      consts=[{"name": "cap", "value": 8, "axis": "reserve"}],
                      instances={"homog": {"cap": 0}})
    m = MdpIR.model_validate(doc)
    assert m.action_bounds("act") == (0.0, 8.0)
    assert m.action_bounds("act", instance="homog") == (0.0, 0.0)


def test_the_same_derivation_is_still_refused_when_continuous(ir_doc):
    doc = with_action(ir_doc, [0.0, "cap"], type_="continuous",
                      consts=[{"name": "cap", "value": 8.0, "axis": "reserve"}],
                      instances={"homog": {"cap": 0.0}})
    with pytest.raises(ValidationError, match="lo >= hi"):
        MdpIR.model_validate(doc)


def test_an_inverted_derivation_is_rejected_on_the_instance_that_inverts_it(ir_doc):
    """Per-scope, as before: the base resolves fine and one instance does not."""
    doc = with_action(ir_doc, [0, "cap"],
                      consts=[{"name": "cap", "value": 8, "axis": "reserve"}],
                      instances={"broken": {"cap": -1}})
    with pytest.raises(ValidationError, match="lo > hi"):
        MdpIR.model_validate(doc)


def test_a_state_envelope_still_needs_a_non_empty_interval(ir_doc):
    """Scoped out deliberately: a state variable carries no discrete/continuous
    type, and a zero-width envelope is an observation dimension that cannot
    vary — likelier a mis-resolved constant than a design."""
    doc = copy.deepcopy(ir_doc)
    for sv in doc["mdp"]["state_variables"]:
        if sv["name"] == "level":
            sv["bounds"] = [0.0, "zero"]
    doc["mdp"]["scenario"]["constants"].append(
        {"name": "zero", "value": 0.0, "axis": ""})
    with pytest.raises(ValidationError, match="lo >= hi"):
        MdpIR.model_validate(doc)


# -- #70: the reader ---------------------------------------------------------


def test_the_reader_resolves_every_entry_form(ir_doc):
    """The three forms `decision_bounds` already serves, at the gym site."""
    doc = with_action(ir_doc, [0, "20 * h"],
                      instances={"pricey": {"h": 5.0}})
    m = MdpIR.model_validate(doc)
    assert m.action_bounds("act") == (0.0, 20.0)
    assert m.action_bounds("act", instance="pricey") == (0.0, 100.0)


def test_the_reader_and_the_raw_entries_are_both_available(ir_doc):
    """`bounds_per_decision()` keeps returning the unresolved entries — the
    validator needs them to say which one failed — and the reader is added
    beside it rather than in place of it."""
    m = MdpIR.model_validate(with_action(ir_doc, [0, "20 * h"]))
    assert m.gym.action_modes[0].bounds_per_decision() == [[0, "20 * h"]]
    assert m.action_bounds("act") == (0.0, 20.0)


def test_the_reader_selects_a_multi_decision_pair_by_decision_name(ir_doc):
    doc = copy.deepcopy(ir_doc)
    doc["mdp"]["decisions"].append({
        "name": "act2",
        "type": {"value": "continuous", "suggested": "continuous"},
        "dim": 1,
        "bounds": {"value": [0.0, "h"], "suggested": [0.0, "h"]},
    })
    doc["gym"]["action_modes"][0].update(
        encodes=["act", "act2"], bounds=[[0.0, 10.0], [0.0, "h"]])
    m = MdpIR.model_validate(doc)
    assert m.action_bounds("act", "act") == (0.0, 10.0)
    assert m.action_bounds("act", "act2") == (0.0, 1.0)
    assert m.action_bounds("act", "act2", instance="pricey") == (0.0, 5.0)
    with pytest.raises(ValueError, match="name which decision"):
        m.action_bounds("act")


def test_the_reader_names_what_it_could_not_find(ir_doc):
    m = MdpIR.model_validate(ir_doc)
    with pytest.raises(KeyError, match="no action mode named 'nope'"):
        m.action_bounds("nope")
    with pytest.raises(KeyError, match="does not encode 'nope'"):
        m.action_bounds("act", "nope")


def test_the_reader_serves_the_catalog_form():
    """A catalog IR folds slot stats and collapses whatever reads no overridden
    constant, so what reaches the reader is already the plain form — the same
    pool, the same three entry forms, one code path."""
    doc = minimal_catalog()
    doc["gym"]["action_modes"][0]["bounds"] = [0.0, "20 * h"]
    doc["mdp"]["decisions"][0]["bounds"] = {
        "value": [0.0, "20 * h"], "suggested": [0.0, "20 * h"]}
    m = MdpIR.model_validate(layering.resolve_catalog(doc))
    assert m.action_bounds("act") == (0.0, 20.0)
    assert m.action_bounds("act", instance="pricey") == (0.0, 100.0)

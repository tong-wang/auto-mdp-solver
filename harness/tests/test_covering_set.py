"""Which instances `--all-instances` actually sweeps.

The covering-set sweep is the differential's headline gate: it is what proves
the interpreter and the generated domain agree on every instance a study
declares. It had no test at all, and the branch that discovers the list lived
inside an argparse handler — so when v0.9.2 added the grouped `mdp` layout,
a grouped IR started sweeping the base alone, reporting one confident MATCH
and exiting 0 (#37). Silence, not failure.

The invariant pinned here is deliberately layout-independent: flat and grouped
presentations of the SAME document must sweep the SAME list. That survives the
next layout change too.
"""

from __future__ import annotations

import json

from conftest import minimal_catalog, minimal_ir

from mdp_ir.differential import covering_set
from mdp_ir.schema import group_mdp


def _write(tmp_path, doc, name="d_schema.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def _grouped(doc):
    doc = json.loads(json.dumps(doc))
    doc["mdp"] = group_mdp(doc["mdp"])
    return doc


def test_the_base_and_every_named_instance_are_swept(tmp_path):
    doc = minimal_ir()
    doc["mdp"]["scenario"]["instances"] = {"pricey": {"h": 5.0}, "cheap": {"h": 0.5}}
    assert covering_set(_write(tmp_path, doc)) == [None, "cheap", "pricey"]


def test_the_grouped_layout_sweeps_the_same_list(tmp_path):
    """The regression. `group_mdp` moves `scenario` under `design`, and the
    reader was looking at `mdp.scenario`."""
    doc = minimal_ir()
    doc["mdp"]["scenario"]["instances"] = {"pricey": {"h": 5.0}, "cheap": {"h": 0.5}}
    flat = covering_set(_write(tmp_path, doc, "flat.json"))
    grouped = covering_set(_write(tmp_path, _grouped(doc), "grouped.json"))
    assert grouped == flat
    assert len(grouped) == 3          # not [None] — the silent-failure shape


def test_a_catalog_sweeps_the_same_list_in_either_layout(tmp_path):
    """Catalogs are where the layout matters most: their instances select
    candidates, so a dropped instance is a whole composition never exercised."""
    doc = minimal_catalog()
    doc["mdp"]["scenario"]["instances"] = {
        "fixed_flow": {"flow": "fixed"}, "pricey": {"h": 5.0},
    }
    flat = covering_set(_write(tmp_path, doc, "flat.json"))
    grouped = covering_set(_write(tmp_path, _grouped(doc), "grouped.json"))
    assert flat == [None, "fixed_flow", "pricey"]
    assert grouped == flat


def test_an_instance_the_base_resolution_would_drop_is_still_swept(tmp_path):
    """Why this reads the raw document and not a loaded `ir`.

    `resolve_catalog` drops instances inconsistent with the active selection,
    so `fixed_flow` — which selects the non-default candidate, and is therefore
    the instance the sweep most needs — is absent from a base-resolved IR.
    Sourcing the list from `ir.mdp.scenario.instances` would look tidier and
    quietly shrink the covering set.
    """
    from mdp_ir import load_ir

    doc = minimal_catalog()
    doc["mdp"]["scenario"]["instances"] = {"fixed_flow": {"flow": "fixed"}}
    path = _write(tmp_path, doc)

    assert "fixed_flow" not in (load_ir(path).mdp.scenario.instances or {})
    assert "fixed_flow" in covering_set(path)


def test_no_instances_still_sweeps_the_base(tmp_path):
    doc = minimal_ir()
    doc["mdp"]["scenario"].pop("instances", None)
    assert covering_set(_write(tmp_path, doc)) == [None]

"""Documentation text leaves the freeze token alone from IR 0.5 (spec §5.1).

`desc` and `note` say what a quantity is for; they state no fact the
interpreter reads, and no trajectory moves when one is rewritten. Hashing them
meant a prose correction voided a Phase-A sign-off, and a case paid for that
twice in one afternoon: a same-day re-signing to accommodate a fix, then a
description knowingly left wrong because putting it right would have orphaned
its artifacts.

The exclusion is version-gated for the reason #34 established for the model
block — an IR at 0.4 must hash byte-identically to what it always did, so every
recorded token stays valid and adoption is the author's move. Synthetic IRs
only: the harness's suite must not depend on `plugin/`.
"""

from __future__ import annotations

import copy

import pytest

from mdp_ir.schema import MdpIR, hashes_prose


def token(doc: dict) -> str:
    return MdpIR.model_validate(doc).mdp_fingerprint()


def at(ir_doc: dict, version: str, **edits) -> dict:
    """The fixture at one IR version, with optional prose edits applied."""
    doc = copy.deepcopy(ir_doc)
    doc["ir_version"] = version
    constant = doc["mdp"]["scenario"]["constants"][0]
    if "desc" in edits:
        constant["desc"] = edits["desc"]
    if "note" in edits:
        doc["mdp"]["uncertainty_sources"][0]["distribution"]["note"] = edits["note"]
    if "value" in edits:
        constant["value"] = edits["value"]
    return doc


# -- 0.4: nothing moves, including what we would rather it did not hash ------

def test_a_legacy_ir_still_hashes_its_prose(ir_doc):
    """The property that makes adoption safe: 0.4 behaves exactly as before."""
    assert token(at(ir_doc, "0.4")) != token(at(ir_doc, "0.4", desc="reworded"))


def test_an_unparseable_version_is_treated_as_legacy():
    """A freeze token is not the place to guess."""
    assert hashes_prose("draft") and hashes_prose("") and hashes_prose("0.4")
    assert not hashes_prose("0.5") and not hashes_prose("1.0")


# -- 0.5: prose is documentation, facts are not ------------------------------

def test_rewriting_a_desc_does_not_move_the_token(ir_doc):
    assert token(at(ir_doc, "0.5")) == token(at(ir_doc, "0.5", desc="reworded"))


def test_rewriting_a_distribution_note_does_not_move_the_token(ir_doc):
    assert token(at(ir_doc, "0.5")) == token(at(ir_doc, "0.5", note="qualified"))


def test_a_stated_fact_still_moves_the_token(ir_doc):
    """The point is to stop hashing prose, not to stop hashing the model."""
    assert token(at(ir_doc, "0.5")) != token(at(ir_doc, "0.5", value=999))


# -- adoption: once, and never again -----------------------------------------

def test_adopting_moves_the_token_exactly_once(ir_doc):
    """Which is the truth about it: its statement stopped including the prose.

    A later version in the same prose-free lineage must not move it a second
    time — the failure #34 recorded, in its version-gated form.
    """
    legacy, adopted = token(at(ir_doc, "0.4")), token(at(ir_doc, "0.5"))
    assert legacy != adopted
    assert token(at(ir_doc, "0.6")) == adopted
    assert token(at(ir_doc, "0.5", desc="again")) == adopted


def test_blanking_the_prose_does_not_dodge_the_one_time_move(ir_doc):
    """Adoption drops the keys, so an empty `desc` is not the same as none.

    Which is the honest reading: the bump is a decision about what the IR's
    statement contains, not a text edit a domain can pre-apply to keep its
    token still.
    """
    blank = at(ir_doc, "0.4", desc="")
    adopted = copy.deepcopy(blank)
    adopted["ir_version"] = "0.5"
    assert token(blank) != token(adopted)


# -- the theory layer is untouched either way --------------------------------

def test_the_model_fingerprint_is_unaffected(ir_doc):
    """`model_fingerprint` hashes the theory, which carries no desc or note."""
    doc = at(ir_doc, "0.5")
    doc["mdp"]["model"] = {"quantities": {"level": {"domain": "float >= 0"}}}
    edited = copy.deepcopy(doc)
    edited["mdp"]["scenario"]["constants"][0]["desc"] = "reworded"
    a, b = MdpIR.model_validate(doc), MdpIR.model_validate(edited)
    assert a.model_fingerprint() == b.model_fingerprint() is not None
    assert a.mdp_fingerprint() == b.mdp_fingerprint()

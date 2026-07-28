"""Seed-key grammar (spec §6.3).

The keys are the contract between the interpreter and generated domain code:
if the slot order drifts on either side, every differential goes red for a
reason no trajectory diff explains. These tests pin both grammars — v1's
``stream_id`` before ``period``, and v2's leaf-first ``period`` below the
source with a branch word — symbolically and numerically.
"""

from __future__ import annotations

from mdp_ir.interpreter import _meta_seed_key, _numeric_seed_key
from mdp_ir.schema import MdpIR, Realization, UncertaintyStage


def stage(**kw) -> UncertaintyStage:
    return UncertaintyStage(name="sample", realization=Realization.period, **kw)


# -- v2: the canonical scheme for new domains --------------------------------


def test_v2_symbolic_key_is_leaf_first():
    st = stage()
    assert st.seed_key(4, False, scheme="v2") == [
        "period", "source:4", "branch:1", "episode_seed", "seed_salt"]


def test_v2_sub_stream_prefixes_the_key():
    st = stage(sub_stream=2)
    assert st.seed_key(4, False, scheme="v2")[0] == "sub:2"


def test_v2_numeric_key_matches_the_symbolic_slots():
    st = stage()
    assert _numeric_seed_key(st, 4, period=7, episode_seed=9, seed_salt=13,
                             scheme="v2") == [7, 4, 1, 9, 13]


def test_v2_episode_realization_takes_no_period_slot():
    """An episode-realization draw is constant across the episode, so keying
    it on the period would make it vary."""
    st = UncertaintyStage(name="support", realization=Realization.episode)
    key = _numeric_seed_key(st, 4, period=7, episode_seed=9, seed_salt=13,
                            scheme="v2")
    assert key == [4, 1, 9, 13]


def test_v2_keyed_realization_takes_no_period_slot():
    st = UncertaintyStage(name="tab", realization=Realization.keyed,
                          key_exprs=["1"])
    key = _numeric_seed_key(st, 4, period=7, episode_seed=9, seed_salt=13,
                            key_vals=[5], scheme="v2")
    assert key == [5, 4, 1, 9, 13]


def test_v2_meta_key_is_substream_then_branch_zero():
    assert _meta_seed_key("v2", 2, 9, 13) == [2, 0, 9, 13]


# -- v1: legacy, still read by older domains ---------------------------------


def test_v1_keys_the_stream_before_the_period():
    st = stage()
    assert _numeric_seed_key(st, 4, period=7, episode_seed=9, seed_salt=13,
                             scheme="v1") == [4, 7, 9, 13]


def test_v1_lone_drawer_meta_key_keeps_its_historical_shape():
    assert _meta_seed_key("v1", 0, 9, 13) == [0, 9, 13]
    assert _meta_seed_key("v1", 2, 9, 13) == [0, 2, 9, 13]


# -- the derivation the schema exposes ---------------------------------------


def test_entity_id_leads_a_multi_entity_key():
    st = stage()
    key = _numeric_seed_key(st, 4, period=7, episode_seed=9, seed_salt=13,
                            entity_id=3, scheme="v2")
    assert key[0] == 3


def test_declared_scheme_reaches_the_interpreter(ir_doc):
    ir_doc["seed_scheme"] = "v2"
    assert MdpIR.model_validate(ir_doc).seed_scheme == "v2"

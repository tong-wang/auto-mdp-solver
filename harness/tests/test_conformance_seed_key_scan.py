"""The v2 drift guard reads the mdp layer too (spec §6.3).

A domain with deterministic dynamics ships no `_uncertainty.py` (spec §1,
§4.3), so the one draw it owns — the reset-time one — lands in `_mdp.py`,
which the scan did not look at. A key written out by hand there collected a
PASS line saying "all SeedSequence calls go through the key helpers". World
layers still FAIL; the mdp layer WARNs, because no domain in that shape was
ever asked to route through the helper.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

from mdp_conformance.checks import check_seed_key_helpers, raw_seed_key_sites
from mdp_conformance.loader import DomainHandle

RAW = """
import numpy as np

def _draw(episode_seed, seed_salt):
    return np.random.default_rng(
        np.random.SeedSequence([0, 0, episode_seed, seed_salt]))
"""

ROUTED = """
import numpy as np
from mdp_ir.runtime import meta_key

def _draw(episode_seed, seed_salt):
    return np.random.default_rng(
        np.random.SeedSequence(meta_key(0, episode_seed, seed_salt)))
"""

DEFINES_HELPER = """
import numpy as np

def meta_key(substream_id, episode_seed, seed_salt):
    return [substream_id, 0, episode_seed, seed_salt]

def _draw(episode_seed, seed_salt):
    return np.random.default_rng(np.random.SeedSequence(
        meta_key(0, episode_seed, seed_salt)))
"""


def _sites(source: str, label: str = "mdp") -> list[str]:
    return raw_seed_key_sites(ast.parse(source), label)


def _domain(**trees) -> DomainHandle:
    return DomainHandle(
        name="d", directory=None, files={}, modules={"s": SimpleNamespace(SEED_SCHEME="v2")},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object, state_cls=None,
        init_state=lambda *a, **k: (None, {}),
        trees={role: ast.parse(src) for role, src in trees.items()},
    )


def test_a_hand_written_key_is_a_site():
    assert _sites(RAW) == ["mdp:6"]


def test_a_helper_built_key_is_clean():
    assert _sites(ROUTED) == []


def test_a_call_inside_the_helper_itself_is_clean():
    """The helper has to build the raw key somewhere — that is its job."""
    assert _sites(DEFINES_HELPER) == []


def test_the_mdp_layer_warns_where_a_world_layer_fails():
    warn = check_seed_key_helpers(_domain(mdp=RAW, scenarios=ROUTED))
    assert warn.status == "WARN" and "mdp:6" in warn.detail

    fail = check_seed_key_helpers(_domain(mdp=ROUTED, scenarios=RAW))
    assert fail.status == "FAIL" and "scenarios:6" in fail.detail


def test_a_world_layer_stray_outranks_an_mdp_one():
    """One line, one verdict: the harder one, so a FAIL is never softened by
    where the other stray happened to sit."""
    result = check_seed_key_helpers(_domain(mdp=RAW, scenarios=RAW))
    assert result.status == "FAIL"


def test_a_clean_domain_still_passes():
    result = check_seed_key_helpers(_domain(mdp=ROUTED, scenarios=ROUTED))
    assert result.status == "PASS"


def test_a_v1_domain_is_untouched():
    """v1 keys are frozen so recorded results stay reproducible (spec §6.3)."""
    h = DomainHandle(
        name="d", directory=None, files={}, modules={"s": SimpleNamespace(SEED_SCHEME="v1")},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object, state_cls=None,
        init_state=lambda *a, **k: (None, {}), trees={"mdp": ast.parse(RAW)},
    )
    assert check_seed_key_helpers(h).status == "SKIP"

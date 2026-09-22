"""A sampler the IR declares owes a twin in `SCENARIOS` (spec §5.2).

`scenario.samplers` read the registry alone, so *absence* certified itself: a
domain whose IR declares a world sampler and whose `SCENARIOS` holds no
callable SKIPped the check entirely, and the draw it implements somewhere else
— an `init_state` seeding its own permutation — was covered here by nothing.
A SKIP reads as "nothing to check"; this one meant "the drawer is out of
sight". Synthetic IRs only: the harness's suite must not depend on `plugin/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mdp_conformance.checks import check_sampler_registry
from mdp_conformance.loader import DomainHandle


@dataclass(frozen=True)
class _Concrete:
    rate: float


def _domain(tmp_path, ir_doc, scenarios: dict | None = None) -> DomainHandle:
    (tmp_path / "d_schema.json").write_text(json.dumps(ir_doc))
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS=scenarios or {}, env_cls=object,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


def _declare_sampler(ir_doc, name="world"):
    ir_doc["mdp"]["scenario"]["samplers"] = [{
        "name": name, "substream_id": 0, "hidden": False,
        "draws": [{"name": "rate", "distribution": {
            "family": "uniform", "settings": {"low": 1.0, "high": 2.0}}}],
    }]
    return ir_doc


def test_an_ir_with_no_sampler_and_an_empty_registry_skips(tmp_path, ir_doc):
    assert check_sampler_registry(_domain(tmp_path, ir_doc)).status == "SKIP"


def test_a_declared_sampler_with_no_registry_twin_warns(tmp_path, ir_doc):
    """The secretary shape: the IR declares the reset draw, the domain
    implements it in its mdp layer, and the registry is empty."""
    result = check_sampler_registry(_domain(tmp_path, _declare_sampler(ir_doc)))
    assert result.status == "WARN"
    assert "world" in result.detail and "SCENARIOS" in result.detail


def test_a_pure_twin_passes(tmp_path, ir_doc):
    h = _domain(tmp_path, _declare_sampler(ir_doc),
                {"world": lambda seed: _Concrete(rate=1.0 + seed)})
    assert check_sampler_registry(h).status == "PASS"


def test_an_impure_twin_still_fails(tmp_path, ir_doc):
    """The purity half is unchanged — reading the IR only removed a blind
    spot, it did not soften what a present sampler must do."""
    calls = iter(range(100))
    h = _domain(tmp_path, _declare_sampler(ir_doc),
                {"world": lambda seed: _Concrete(rate=float(next(calls)))})
    result = check_sampler_registry(h)
    assert result.status == "FAIL" and "impure" in result.detail


def test_an_unreadable_ir_skips_rather_than_warns(tmp_path, ir_doc):
    """No schema beside the domain is the mid-build state, not a finding."""
    h = DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )
    assert check_sampler_registry(h).status == "SKIP"

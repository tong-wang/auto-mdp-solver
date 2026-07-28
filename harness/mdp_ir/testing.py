"""Helpers for a domain's own ``{domain}_test.py``.

A domain folder is portable (spec §1): it must work wherever it lives, so its
test cannot reach for a conftest in this repo. These helpers ship in the
package instead, which makes them equally available to a shipped example, a
case under ``cases/``, and a domain in a downstream research repo.

Typical use, from ``{domain}_test.py`` sitting next to ``{domain}_schema.json``::

    from mdp_ir.testing import assert_laws, assert_match, schema_beside

    SCHEMA = schema_beside(__file__)

    def test_laws():
        assert_laws(SCHEMA)

    def test_differential():
        assert_match(SCHEMA, instance="lost_sales", episodes=8)

What belongs in a domain test is what only that domain can state: its
conservation laws in Python where they are awkward to declare (declare them
in ``mdp.invariants`` where they are not), its equivalences against
hand-written classes, its benchmark checks. Everything true of *any* IR is
already covered by :mod:`mdp_ir.laws`, which ``assert_laws`` runs for you.
"""

from __future__ import annotations

from pathlib import Path

from mdp_ir.differential import (
    DifferentialReport,
    load_adapter_factory,
    run_differential,
)
from mdp_ir.laws import resolve_schema, run_laws
from mdp_ir.schema import MdpIR, load_ir

__all__ = [
    "schema_beside",
    "differential",
    "assert_match",
    "assert_diverges",
    "assert_laws",
    "mutated",
]


def schema_beside(test_file: str | Path) -> Path:
    """The one ``*_schema.json`` in the same folder as the calling test."""
    return resolve_schema(Path(test_file).resolve().parent)


def differential(
    schema: str | Path,
    instance: str | None = None,
    episodes: int = 3,
    first_seed: int = 0,
    seed_salt: int = 1,
    ir: MdpIR | None = None,
) -> DifferentialReport:
    """Replay the IR interpreter against the domain's real implementation.

    ``ir`` overrides the loaded IR — pass a mutated one to check that a
    deliberate corruption is actually caught (see :func:`assert_diverges`).
    """
    schema = Path(schema)
    resolved = ir if ir is not None else load_ir(schema, instance=instance)
    factory = load_adapter_factory(schema, resolved)
    return run_differential(
        resolved,
        factory(resolved, instance=instance, seed_salt=seed_salt),
        episode_seeds=list(range(first_seed, first_seed + episodes)),
        instance=instance,
        seed_salt=seed_salt,
    )


def assert_match(schema: str | Path, instance: str | None = None, **kw) -> None:
    """The interpreter and the domain must agree bit-for-bit, with no
    declared-invariant violation on either side of the replay."""
    report = differential(schema, instance=instance, **kw)
    assert report.ok, report.render()


def assert_diverges(schema: str | Path, ir: MdpIR, **kw) -> None:
    """A deliberately corrupted IR must NOT match — the negative control that
    proves the gate can fail."""
    report = differential(schema, ir=ir, **kw)
    assert not report.ok, (
        "corrupted IR still matched the domain: the differential is not gating\n"
        + report.render()
    )


def assert_laws(schema: str | Path, instance: str | None = None) -> None:
    """Every engine law (``mdp_ir.laws``) must pass for this IR."""
    schema = resolve_schema(schema)
    ir = load_ir(schema, instance=instance)
    results = run_laws(ir, instance, schema)
    failed = [r for r in results if r.status == "FAIL"]
    assert not failed, "\n".join(f"{r.law}: {r.detail}" for r in failed)


def mutated(ir: MdpIR, edit) -> MdpIR:
    """A copy of ``ir`` with ``edit(doc)`` applied to its plain-dict form —
    for negative controls that must break the gate."""
    import copy

    doc = copy.deepcopy(ir.model_dump(mode="json"))
    edit(doc)
    return MdpIR.model_validate(doc)

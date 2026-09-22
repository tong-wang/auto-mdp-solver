"""`benchmarks.declared` — the declaration and its file, and when each is due
(upstream #92).

The set-equality half is older: a `role` nobody can trace to a file is a second
source of truth. What #92 adds is timing. The roles are settled in the
formalize interview — the bracket is part of the objective conversation — while
the solvers are written in Phase B, so "declared, no file yet" is the normal
state of a Stage-2 folder and must not block `--for solve`. The run plan is the
signal that Stage 3 has opened and the files are due.

Synthetic throughout: `harness/tests` must not depend on `plugin/`.
"""

from __future__ import annotations

import json

from mdp_conformance.checks import check_benchmarks
from mdp_conformance.loader import DomainHandle


def _domain(tmp_path, ir_doc, *, on_disk=(), runplan: bool = False):
    (tmp_path / "d_schema.json").write_text(json.dumps(ir_doc))
    for method in on_disk:
        (tmp_path / f"d_benchmark_{method}.py").write_text("# solver\n")
    if runplan:
        (tmp_path / "d.runplan.json").write_text(
            json.dumps({"target": "base", "strategy": "single"})
        )
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


def _declare(ir_doc, *names):
    ir_doc["benchmarks"] = [{"name": n, "role": "feasible"} for n in names]
    return ir_doc


def test_no_declaration_skips(tmp_path, ir_doc):
    assert check_benchmarks(_domain(tmp_path, ir_doc)).status == "SKIP"


def test_declared_and_present_passes(tmp_path, ir_doc):
    h = _domain(tmp_path, _declare(ir_doc, "myopic"), on_disk=("myopic",))
    result = check_benchmarks(h)
    assert result.status == "PASS" and "myopic=feasible" in result.detail


def test_declared_before_stage_3_is_owed(tmp_path, ir_doc):
    """The Phase-A record the status quo cost `owmr`: three roles declared in
    the interview, no solver written yet, and a FAIL that blocked solve."""
    h = _domain(tmp_path, _declare(ir_doc, "myopic"))
    result = check_benchmarks(h)
    assert result.status == "WARN"
    assert "owed" in result.detail and "d.runplan.json" in result.detail


def test_declared_with_no_file_fails_once_the_run_plan_exists(tmp_path, ir_doc):
    h = _domain(tmp_path, _declare(ir_doc, "myopic"), runplan=True)
    result = check_benchmarks(h)
    assert result.status == "FAIL" and "declared with no d_benchmark_*.py" in result.detail


def test_on_disk_but_undeclared_fails_at_every_stage(tmp_path, ir_doc):
    """Unchanged by #92: an untraceable arm is a second source of truth
    whatever stage the folder is at."""
    h = _domain(tmp_path, _declare(ir_doc, "myopic"), on_disk=("myopic", "oracle"))
    result = check_benchmarks(h)
    assert result.status == "FAIL" and "on disk but undeclared: ['oracle']" in result.detail


def test_an_owed_declaration_does_not_soften_an_undeclared_file(tmp_path, ir_doc):
    h = _domain(tmp_path, _declare(ir_doc, "myopic"), on_disk=("oracle",))
    result = check_benchmarks(h)
    assert result.status == "FAIL" and "on disk but undeclared: ['oracle']" in result.detail
    # the owed declaration is still reported, not swallowed by the failure
    assert "owed, built in Stage 3" in result.detail

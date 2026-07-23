"""Run the conformance checks over one or more domains and format a report.

The per-check report is the dual-use output of the harness: in the prompted
agent it is the accept/reject gate for a generated domain; the same
pass/fail vector can serve as a reward signal for a trained generator.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from .loader import CheckResult, DomainHandle, load_domain
from .checks import REGISTRY

_ICON = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP", "ERROR": "ERR "}
_FAILING = {"FAIL", "ERROR"}


def run_domain(directory) -> tuple[str, list[CheckResult]]:
    """Load and check one domain. Import failures become a single ERROR result."""
    try:
        handle: DomainHandle = load_domain(directory)
    except Exception as e:
        name = Path(directory).name
        return name, [CheckResult("load", "ERROR", f"{type(e).__name__}: {e}")]

    results: list[CheckResult] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for check in REGISTRY:
            try:
                out = check(handle)
            except Exception as e:
                out = CheckResult(getattr(check, "__name__", "check"), "ERROR",
                                  f"{type(e).__name__}: {e}")
            results.extend(out if isinstance(out, list) else [out])
    return handle.name, results


def format_report(name: str, results: list[CheckResult]) -> str:
    width = max((len(r.check) for r in results), default=10)
    lines = [f"=== {name} ==="]
    for r in results:
        lines.append(f"  [{_ICON[r.status]}] {r.check.ljust(width)}  {r.detail}")
    passed = sum(r.status == "PASS" for r in results)
    lines.append(f"  {passed}/{len(results)} passed"
                 f"{'' if not any(r.status in _FAILING for r in results) else '  <-- GATE FAILED'}")
    return "\n".join(lines)


def run(directories) -> int:
    """Run every domain, print reports, return process exit code."""
    any_fail = False
    for directory in directories:
        name, results = run_domain(directory)
        print(format_report(name, results))
        print()
        any_fail |= any(r.status in _FAILING for r in results)
    return 1 if any_fail else 0

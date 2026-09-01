"""The entry-gate checks and the two runners (table / --for gate).

Checks come in two costs. Cheap ones read files and re-derive facts (IR
validity, the freeze comparison, artifact presence); expensive ones *run* the
upstream gates — conformance, laws, the differential — and only ``--for
solve`` pays for them, because solve's upstream exit (build) *is* those three
gates. Every op from build on carries the freeze core: the mdp block a later
op works from must still be the block the human signed off.

Nothing here writes. The two state files are single-writer by contract —
formalize owns ``{name}.signoff.json``, solve owns ``{name}.runplan.json`` —
and a gate that repaired its own preconditions would be no gate at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from mdp_ir.laws import resolve_schema
from mdp_ir.schema import MdpIR, load_ir

_FAILING = {"FAIL", "ERROR"}
_ICON = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP", "ERROR": "ERR "}

# episodes per differential run at the solve gate — the skill's Stage-1 gate
# figure, not the differential CLI's own default (20): the gate re-runs the
# command the skill states, so the two must quote the same number
EPISODES = 40


@dataclass
class GateResult:
    """Mirrors ``mdp_conformance.CheckResult`` / ``laws.LawResult`` so the
    three gates read the same in a terminal and in a transcript."""

    check: str
    status: str  # PASS | FAIL | SKIP | ERROR
    detail: str = ""


class DomainContext:
    """One domain folder, its schema, and the two state files — resolved once,
    loaded lazily, shared by every check in a run."""

    def __init__(self, directory: str | Path):
        self.schema = resolve_schema(Path(directory))
        self.directory = self.schema.parent
        self.name = self.schema.stem.removesuffix("_schema")
        self.signoff_path = self.directory / f"{self.name}.signoff.json"
        self.runplan_path = self.directory / f"{self.name}.runplan.json"

    @cached_property
    def raw(self) -> dict:
        return json.loads(self.schema.read_text())

    @cached_property
    def instance_names(self) -> list[str]:
        """Names the validity check must resolve — mirrors ``python -m mdp_ir``:
        for a catalog, base + every instance + every mixture; for a legacy
        resolved file, base only (instance resolution stays with consumers)."""
        from mdp_ir import layering

        if not layering.is_catalog(self.raw):
            return []
        scenario_raw = layering._flat_mdp(self.raw).get("scenario") or {}
        return sorted(scenario_raw.get("instances") or {}) + sorted(
            m["name"] for m in scenario_raw.get("mixtures") or []
        )

    @cached_property
    def ir(self) -> MdpIR:
        return load_ir(self.schema)

    def _state_file(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path.name}: expected a JSON object")
        return data

    @cached_property
    def signoff(self) -> dict | None:
        return self._state_file(self.signoff_path)

    @cached_property
    def runplan(self) -> dict | None:
        return self._state_file(self.runplan_path)

    def results_dir(self) -> Path | None:
        """``results/{target}`` for the runplan's target, if both resolve."""
        if not self.runplan or "target" not in self.runplan:
            return None
        return self.directory / "results" / str(self.runplan["target"])


# ---------------------------------------------------------------------------
# Cheap checks
# ---------------------------------------------------------------------------


def check_ir_validates(ctx: DomainContext) -> GateResult:
    try:
        ir = ctx.ir
        for inst in ctx.instance_names:
            load_ir(ctx.schema, instance=inst)
    except Exception as exc:
        return GateResult("ir.validates", "FAIL", f"{type(exc).__name__}: {exc}")
    covered = "base" + (
        ", " + ", ".join(ctx.instance_names) if ctx.instance_names else ""
    )
    return GateResult(
        "ir.validates",
        "PASS",
        f"fingerprint={ir.mdp_fingerprint()} (resolved: {covered})",
    )


def check_unconfirmed(ctx: DomainContext) -> GateResult:
    try:
        pending = ctx.ir.unconfirmed()
    except Exception as exc:
        return GateResult("ir.unconfirmed", "FAIL", f"{type(exc).__name__}: {exc}")
    if pending:
        return GateResult(
            "ir.unconfirmed", "FAIL",
            f"{len(pending)} Confirmable(s) still source=derived: "
            + ", ".join(pending),
        )
    return GateResult("ir.unconfirmed", "PASS", "every judgment call reviewed")


def check_signoff(ctx: DomainContext) -> GateResult:
    try:
        data = ctx.signoff
    except Exception as exc:
        return GateResult("signoff.readable", "FAIL", f"{type(exc).__name__}: {exc}")
    if data is None:
        return GateResult(
            "signoff.readable", "FAIL",
            f"{ctx.signoff_path.name} not found — formalize's gate writes it "
            "at human sign-off",
        )
    if "mdp_fingerprint" not in data:
        return GateResult(
            "signoff.readable", "FAIL",
            f"{ctx.signoff_path.name} has no 'mdp_fingerprint' key",
        )
    return GateResult(
        "signoff.readable", "PASS",
        f"signed off {data.get('signed_off', '(date not recorded)')}",
    )


def check_freeze(ctx: DomainContext) -> GateResult:
    """The always-on post-freeze drift detector: the signed-off fingerprint
    must equal the fingerprint of the mdp block as it stands NOW. gym/rl
    edits leave it still; any later mdp edit voids the confirmation."""
    try:
        data = ctx.signoff
        current = ctx.ir.mdp_fingerprint()
    except Exception as exc:
        return GateResult("signoff.fingerprint", "FAIL", f"{type(exc).__name__}: {exc}")
    if data is None or "mdp_fingerprint" not in data:
        return GateResult(
            "signoff.fingerprint", "SKIP", "no readable sign-off to compare against"
        )
    recorded = data["mdp_fingerprint"]
    if recorded != current:
        return GateResult(
            "signoff.fingerprint", "FAIL",
            f"recorded {recorded} != current {current} — the mdp block moved "
            "after sign-off; return to formalize and re-confirm",
        )
    return GateResult("signoff.fingerprint", "PASS", current)


def check_restatement(ctx: DomainContext) -> GateResult:
    try:
        data = ctx.signoff
    except Exception:
        data = None
    named = (data or {}).get("restatement", f"{ctx.name}.restatement.md")
    path = ctx.directory / named
    if not path.exists():
        return GateResult("restatement.exists", "FAIL", f"{named} not found")
    return GateResult(
        "restatement.exists", "PASS",
        f"{named} (currency is conformance's docs.restatement_current)",
    )


def check_runplan(ctx: DomainContext) -> GateResult:
    try:
        data = ctx.runplan
    except Exception as exc:
        return GateResult("runplan.readable", "FAIL", f"{type(exc).__name__}: {exc}")
    if data is None:
        return GateResult(
            "runplan.readable", "FAIL",
            f"{ctx.runplan_path.name} not found — solve writes it at run-plan "
            "confirmation",
        )
    missing = [k for k in ("target", "strategy") if k not in data]
    if missing:
        return GateResult(
            "runplan.readable", "FAIL",
            f"{ctx.runplan_path.name} missing key(s): {', '.join(missing)}",
        )
    return GateResult(
        "runplan.readable", "PASS",
        f"target={data['target']} strategy={data['strategy']}",
    )


def check_baselines(ctx: DomainContext) -> GateResult:
    rdir = ctx.results_dir()
    if rdir is None:
        return GateResult("baselines.tsv", "SKIP", "no readable run plan names a target")
    bench = rdir / "benchmark"
    tsvs = sorted(bench.glob("*.tsv")) if bench.is_dir() else []
    if len(tsvs) < 2:
        return GateResult(
            "baselines.tsv", "FAIL",
            f"{len(tsvs)} baseline eval TSV(s) under {bench} — Stage 3 owes at "
            "least random + a heuristic",
        )
    return GateResult(
        "baselines.tsv", "PASS", f"{len(tsvs)} TSVs: "
        + ", ".join(t.name for t in tsvs),
    )


def _qualifying_runs(rdir: Path) -> list[Path]:
    """Run dirs under ``results/{target}/`` (not ``benchmark/``) holding both
    a model and an eval TSV."""
    return [
        d for d in sorted(rdir.iterdir())
        if d.is_dir() and d.name != "benchmark"
        and any(d.glob("*.zip")) and any(d.glob("*.tsv"))
    ]


def _recorded_fingerprint(run_dir: Path) -> str | None:
    """The ``ir_mdp_fingerprint`` a run's args log recorded at launch
    (spec §8.4 provenance keys), or None for a pre-provenance run."""
    logs = sorted(run_dir.glob("*_args.txt"))
    if not logs:
        return None
    for line in logs[0].read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "ir_mdp_fingerprint":
            return value.strip() or None
    return None


def check_rl_artifacts(ctx: DomainContext) -> GateResult:
    """At least one trained run: a run dir under ``results/{target}/`` (not
    ``benchmark/``) holding a model and its eval TSV."""
    rdir = ctx.results_dir()
    if rdir is None:
        return GateResult("rl.artifacts", "SKIP", "no readable run plan names a target")
    if not rdir.is_dir():
        return GateResult("rl.artifacts", "FAIL", f"{rdir} does not exist")
    runs = _qualifying_runs(rdir)
    if not runs:
        return GateResult(
            "rl.artifacts", "FAIL",
            f"no run dir under {rdir} holds both a model (*.zip) and an eval TSV",
        )
    return GateResult(
        "rl.artifacts", "PASS", f"{len(runs)} run(s): "
        + ", ".join(d.name for d in runs[:4]) + ("…" if len(runs) > 4 else ""),
    )


def check_rl_current(ctx: DomainContext) -> GateResult:
    """Trained artifacts belong to the model as it stands NOW. Every run's
    args log records ``ir_mdp_fingerprint`` at launch (spec §8.4); comparing
    it to the current fingerprint catches the drift the freeze check alone
    cannot: a legitimate re-freeze (formalize re-confirmed, sign-off
    rewritten) leaves every op unblocked while the runs on disk still answer
    the *previous* model's question. Stale-but-accompanied is a warning —
    old arms are legitimate history; all-stale is a FAIL — there is nothing
    current to read."""
    rdir = ctx.results_dir()
    if rdir is None or not rdir.is_dir():
        return GateResult("rl.current", "SKIP", "no run tree to read")
    runs = _qualifying_runs(rdir)
    if not runs:
        return GateResult("rl.current", "SKIP", "no qualifying runs to date")
    try:
        current = ctx.ir.mdp_fingerprint()
    except Exception as exc:
        return GateResult("rl.current", "FAIL", f"{type(exc).__name__}: {exc}")
    recorded = {d.name: _recorded_fingerprint(d) for d in runs}
    stale = sorted(n for n, fp in recorded.items() if fp and fp != current)
    fresh = sorted(n for n, fp in recorded.items() if fp == current)
    unknown = sorted(n for n, fp in recorded.items() if fp is None)
    if stale and not fresh and not unknown:
        return GateResult(
            "rl.current", "FAIL",
            f"every run predates the current mdp block ({current}): "
            + ", ".join(stale) + " — retrain before reading these artifacts",
        )
    if stale:
        return GateResult(
            "rl.current", "WARN",
            f"stale run(s) from a superseded mdp block: {', '.join(stale)}"
            " — select and report only from current runs",
        )
    if unknown and not fresh:
        return GateResult(
            "rl.current", "WARN",
            "runs record no ir_mdp_fingerprint (pre-provenance) — currency "
            "unverifiable; treat with care",
        )
    return GateResult("rl.current", "PASS", f"all runs match {current}")


def check_escalation_log(ctx: DomainContext) -> GateResult:
    log = ctx.directory / "ESCALATION.md"
    if not log.exists():
        return GateResult(
            "escalation.log", "FAIL",
            "ESCALATION.md not found — no run without an address; seed the "
            "campaign log per ESCALATION_LOG_GUIDE before opening L2+",
        )
    return GateResult("escalation.log", "PASS", "ESCALATION.md present")


def check_interpret_owed(ctx: DomainContext) -> GateResult:
    """Informational, never blocking: does a declared tier-2 stance owe the
    §14 probe deliverables, or do only the generic pieces apply?"""
    try:
        rq = ctx.ir.research_questions
    except Exception as exc:
        return GateResult("interpret.owed", "FAIL", f"{type(exc).__name__}: {exc}")
    if rq is None:
        return GateResult(
            "interpret.owed", "SKIP",
            "no research_questions block — only the generic §14 pieces apply",
        )
    if rq.probe_required:
        return GateResult(
            "interpret.owed", "PASS",
            "a declared stance owes {domain}_policy_probe.py + INTERPRET.md",
        )
    return GateResult(
        "interpret.owed", "SKIP", "declared stances owe no probe (bypass)"
    )


# ---------------------------------------------------------------------------
# Expensive checks — run the upstream gates
# ---------------------------------------------------------------------------


def check_conformance(ctx: DomainContext) -> GateResult:
    from mdp_conformance.runner import run_domain

    _, results = run_domain(ctx.directory)
    bad = [r for r in results if r.status in _FAILING]
    if bad:
        return GateResult(
            "gate.conformance", "FAIL",
            f"{len(bad)} failing: " + ", ".join(r.check for r in bad),
        )
    passed = sum(r.status == "PASS" for r in results)
    return GateResult("gate.conformance", "PASS", f"{passed}/{len(results)} passed")


def check_laws(ctx: DomainContext) -> GateResult:
    from mdp_ir.laws import run_laws

    try:
        results = run_laws(ctx.ir, None, ctx.schema)
    except Exception as exc:
        return GateResult("gate.laws", "FAIL", f"{type(exc).__name__}: {exc}")
    bad = [r for r in results if r.status == "FAIL"]
    if bad:
        return GateResult(
            "gate.laws", "FAIL",
            f"{len(bad)} failing: " + ", ".join(r.law for r in bad),
        )
    passed = sum(r.status == "PASS" for r in results)
    return GateResult("gate.laws", "PASS", f"{passed}/{len(results)} passed")


def check_differential(ctx: DomainContext, episodes: int = EPISODES) -> GateResult:
    """Re-run the skill's Stage-1 gate command: ``--all-instances`` at the
    gate's episode count, bit-exact MATCH required. The CLI prints its own
    per-instance report above ours."""
    from mdp_ir import differential

    argv = [str(ctx.schema), "--all-instances", "--episodes", str(episodes)]
    try:
        rc = differential.main(argv)
    except SystemExit as exc:  # argparse .error() — e.g. no adapter found
        return GateResult("gate.differential", "FAIL", f"CLI error (exit {exc.code})")
    except Exception as exc:
        return GateResult("gate.differential", "FAIL", f"{type(exc).__name__}: {exc}")
    if rc != 0:
        return GateResult(
            "gate.differential", "FAIL",
            "MISMATCH or invariant violation — treat as a modeling error to "
            "diagnose, never a tolerance to relax",
        )
    return GateResult(
        "gate.differential", "PASS",
        f"MATCH, --all-instances --episodes {episodes}",
    )


# ---------------------------------------------------------------------------
# Op -> checks
# ---------------------------------------------------------------------------

# the freeze core — every op from build on: the IR must validate, hold no
# unreviewed judgment call, and still be the block the human signed off
_FREEZE_CORE = [
    check_ir_validates,
    check_unconfirmed,
    check_signoff,
    check_freeze,
    check_restatement,
]

OPS: dict[str, list] = {
    "build": list(_FREEZE_CORE),
    "solve": _FREEZE_CORE + [check_conformance, check_laws, check_differential],
    "escalate": _FREEZE_CORE
    + [check_runplan, check_baselines, check_rl_artifacts, check_rl_current,
       check_escalation_log],
    "interpret": _FREEZE_CORE
    + [check_runplan, check_rl_artifacts, check_rl_current,
       check_interpret_owed],
    "package": _FREEZE_CORE
    + [check_runplan, check_baselines, check_rl_artifacts, check_rl_current],
}

# checks whose cost is a training-gate re-run, listed so the table can say
# what it did NOT check rather than silently passing over it
_EXPENSIVE = {check_conformance, check_laws, check_differential}


def _run_checks(ctx: DomainContext, checks, episodes: int) -> list[GateResult]:
    out: list[GateResult] = []
    for check in checks:
        try:
            if check is check_differential:
                out.append(check(ctx, episodes=episodes))
            else:
                out.append(check(ctx))
        except Exception as exc:  # a check must report, not take down the gate
            out.append(
                GateResult(
                    getattr(check, "__name__", "check"), "ERROR",
                    f"{type(exc).__name__}: {exc}",
                )
            )
    return out


def format_report(title: str, results: list[GateResult]) -> str:
    width = max((len(r.check) for r in results), default=10)
    lines = [f"=== {title} ==="]
    for r in results:
        lines.append(f"  [{_ICON[r.status]}] {r.check.ljust(width)}  {r.detail}")
    failed = any(r.status in _FAILING for r in results)
    lines.append("  " + ("ENTRY BLOCKED" if failed else "ENTRY OK"))
    return "\n".join(lines)


def run_gate(
    directory: str | Path, op: str, episodes: int = EPISODES
) -> tuple[list[GateResult], int]:
    """One op's full entry gate. Returns (results, exit code)."""
    ctx = DomainContext(directory)
    results = _run_checks(ctx, OPS[op], episodes)
    print(format_report(f"{ctx.name} --for {op}", results))
    return results, 1 if any(r.status in _FAILING for r in results) else 0


def stage_table(directory: str | Path) -> int:
    """The cheap-check overview: every op's entry status at a glance. A
    report, not a gate — always exits 0 once the folder resolves; the
    expensive solve checks are named as unrun, never presumed."""
    ctx = DomainContext(directory)
    cheap = [c for checks in OPS.values() for c in checks if c not in _EXPENSIVE]
    ordered = list(dict.fromkeys(cheap))  # first-seen order, deduplicated
    results = _run_checks(ctx, ordered, EPISODES)
    by_name = {c.__name__: r for c, r in zip(ordered, results)}

    lines = [f"=== {ctx.name} — stage table ==="]
    width = max(len(r.check) for r in results)
    for r in results:
        lines.append(f"  [{_ICON[r.status]}] {r.check.ljust(width)}  {r.detail}")
    lines.append("")
    for op, checks in OPS.items():
        blocking = [
            by_name[c.__name__].check
            for c in checks
            if c not in _EXPENSIVE and by_name[c.__name__].status in _FAILING
        ]
        unrun = [c for c in checks if c in _EXPENSIVE]
        if blocking:
            verdict = "BLOCKED — " + ", ".join(blocking)
        elif unrun:
            verdict = (
                f"cheap checks pass; run --for {op} for the full gate "
                "(conformance + laws + differential)"
            )
        else:
            verdict = "READY"
        lines.append(f"  {op.ljust(9)} {verdict}")
    print("\n".join(lines))
    return 0

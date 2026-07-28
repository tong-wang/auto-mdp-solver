"""Engine laws: invariants the IR execution semantics must satisfy, for ANY IR.

Where ``mdp_conformance`` checks the *shape of generated code* against the
spec, this module checks the *execution semantics of the IR itself*. It knows
no domain: every law is derived from the IR — which sources are unguarded,
which settings are state-dependent, which instances and mixtures are
declared, what the IR *claims* under ``mdp.invariants``.

That makes it the third gate the pipeline can run on a freshly generated
case, alongside conformance and the differential:

    python -m mdp_ir.laws <domain-dir-or-schema.json> [more ...]

A domain directory is resolved to its single ``*_schema.json``. Laws that do
not apply to an IR report SKIP with the reason (a domain with no mixtures
cannot fail the mixture law), so the report says what was *not* covered
rather than silently passing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from mdp_ir.interpreter import IrInterpreter
from mdp_ir.schema import DecisionType, MdpIR, PeriodIndexing, StateRole, load_ir

# how many episodes each behavioural law samples; enough to make a
# probabilistic law (seed sensitivity, mixture coverage) meaningful without
# turning a gate into a benchmark
EPISODES = 12


@dataclass
class LawResult:
    """Outcome of one law. Mirrors ``mdp_conformance.CheckResult`` so the two
    gates read the same in a terminal and in a transcript."""

    law: str
    status: str          # PASS | FAIL | SKIP
    detail: str = ""


def _pass(law: str, detail: str = "") -> LawResult:
    return LawResult(law, "PASS", detail)


def _fail(law: str, detail: str) -> LawResult:
    return LawResult(law, "FAIL", detail)


def _skip(law: str, detail: str) -> LawResult:
    return LawResult(law, "SKIP", detail)


# ---------------------------------------------------------------------------
# IR-derived facts the laws quantify over
# ---------------------------------------------------------------------------


def _fixed_decisions(ir: MdpIR, at: float) -> dict:
    """A constant decision dict at a fraction ``at`` of each decision's range —
    a concrete policy for any IR, without knowing what the decisions mean."""
    out: dict = {}
    for d in ir.mdp.decisions:
        lo, hi = ir.mdp.decision_bounds(d.name)
        v = lo + at * (hi - lo)
        out[d.name] = (
            int(round(v)) if d.type.value is DecisionType.discrete else float(v)
        )
    return out


def _episode_rows(ir: MdpIR, inst: str | None) -> int:
    """Rows a full episode produces: the clock runs from the indexing origin
    up to ``horizon.T``, so a 1-indexed domain yields one row fewer than T."""
    base = inst if inst in ir.mdp.scenario.instances else None
    origin = (
        0 if ir.mdp.horizon.period_indexing is PeriodIndexing.zero_based else 1
    )
    return ir.mdp.horizon_T(base) - origin


def _time_var(ir: MdpIR) -> str:
    return next(
        sv.name for sv in ir.mdp.state_variables if sv.role is StateRole.time_index
    )


def _row_fields(ir: MdpIR) -> set[str]:
    names = {sv.name for sv in ir.mdp.state_variables}
    names |= {f.name for f in ir.mdp.info_fields}
    names |= {d.name for d in ir.mdp.decisions}
    return names


def _path_independent_draws(ir: MdpIR) -> list[str]:
    """Row fields whose value must not depend on the decision path.

    Derived, not declared: a draw qualifies when its transition is UNGUARDED
    (a guarded draw legitimately differs when the guard is decision-sensitive)
    and its distribution settings reference no per-period name (a
    state-conditioned intensity legitimately tracks the decision). The draw
    target must itself be a row field — a draw into a dynamics local is not
    observable in the trajectory, so it is out of scope here.
    """
    from mdp_ir.interpreter import _DRAW

    per_period = ir.mdp.per_period_names
    fields = _row_fields(ir)
    sources = {s.name: s for s in ir.mdp.uncertainty_sources}
    out: list[str] = []
    for t in ir.mdp.dynamics.transitions:
        if t.guard:
            continue
        for u in t.updates:
            m = _DRAW.match(u)
            if not m:
                continue
            target, src_name, _ = m.groups()
            if target not in fields:
                continue
            src = sources.get(src_name)
            if src is None:
                continue
            from mdp_ir.schema import _identifiers

            dependent = any(
                isinstance(v, str) and (_identifiers(v) & per_period)
                for v in src.distribution.settings.values()
            )
            if not dependent:
                out.append(target)
    return sorted(set(out))


# ---------------------------------------------------------------------------
# The laws
# ---------------------------------------------------------------------------


def law_determinism(ir: MdpIR, inst: str | None, path: Path | None) -> LawResult:
    """Same (instance, seed, decisions) must replay bit-identically."""
    dec = _fixed_decisions(ir, 0.5)
    for seed in range(3):
        a = IrInterpreter(ir, instance=inst).run(seed, decisions=dec).rows
        b = IrInterpreter(ir, instance=inst).run(seed, decisions=dec).rows
        if a != b:
            return _fail("determinism", f"seed {seed} replayed differently")
    r = IrInterpreter(ir, instance=inst).run(0)
    s = IrInterpreter(ir, instance=inst).run(0)
    if r.rows != s.rows:
        return _fail("determinism", "random policy is not seed-deterministic")
    return _pass("determinism", "fixed + random policy, 3 seeds")


def law_seed_sensitivity(ir: MdpIR, inst: str | None, path: Path | None) -> LawResult:
    """Different episode seeds must produce different trajectories — unless
    every source is degenerate, in which case there is nothing to vary."""
    fams = {s.distribution.family for s in ir.mdp.uncertainty_sources}
    if fams <= {"deterministic"}:
        return _skip("seed_sensitivity", "all sources are deterministic")
    dec = _fixed_decisions(ir, 0.5)
    runs = [IrInterpreter(ir, instance=inst).run(s, decisions=dec).rows
            for s in range(EPISODES)]
    if all(r == runs[0] for r in runs[1:]):
        return _fail("seed_sensitivity",
                     f"all {EPISODES} seeds gave one trajectory")
    return _pass("seed_sensitivity", f"{EPISODES} seeds differ")


def law_decision_path_independence(ir: MdpIR, inst: str | None, path: Path | None) -> LawResult:
    """Exogenous draws must not react to the decision path (spec §4.3)."""
    targets = _path_independent_draws(ir)
    if not targets:
        return _skip("path_independence",
                     "no unguarded state-independent draw lands in a row field")
    lo = IrInterpreter(ir, instance=inst).run(5, decisions=_fixed_decisions(ir, 0.0))
    hi = IrInterpreter(ir, instance=inst).run(5, decisions=_fixed_decisions(ir, 1.0))
    n = min(len(lo.rows), len(hi.rows))
    for fld in targets:
        a = [r[fld] for r in lo.rows[:n]]
        b = [r[fld] for r in hi.rows[:n]]
        if a != b:
            return _fail("path_independence",
                         f"{fld!r} changed with the decision path")
    return _pass("path_independence", f"exogenous: {targets}")


def law_invariants(ir: MdpIR, inst: str | None, path: Path | None) -> LawResult:
    """The IR's own declared claims must hold on every sampled trajectory."""
    if not ir.mdp.invariants:
        return _skip("invariants", "none declared")
    names = [i.name for i in ir.mdp.invariants]
    for at in (0.0, 0.5, 1.0, None):
        dec = None if at is None else _fixed_decisions(ir, at)
        for seed in range(EPISODES):
            traj = IrInterpreter(ir, instance=inst).run(seed, decisions=dec)
            if traj.violations:
                v = traj.violations[0]
                return _fail("invariants", f"seed={seed} {v}")
    return _pass("invariants", f"{len(names)} claim(s) over 4 policies: {names}")


def law_termination(ir: MdpIR, inst: str | None, path: Path | None) -> LawResult:
    """No episode may run past the horizon; without an absorbing condition it
    must run the full length."""
    full = _episode_rows(ir, inst)
    early = ir.gym.termination.early_terminated_when
    for seed in range(EPISODES):
        traj = IrInterpreter(ir, instance=inst).run(seed)
        if len(traj.rows) > full:
            return _fail("termination",
                         f"seed {seed}: {len(traj.rows)} rows > horizon {full}")
        if not early and len(traj.rows) != full:
            return _fail("termination",
                         f"seed {seed}: {len(traj.rows)} rows != horizon {full} "
                         f"with no early-termination condition")
    return _pass("termination",
                 f"{full} periods" + (" (absorbing allowed)" if early else ""))


def law_finite_reward(ir: MdpIR, inst: str | None, path: Path | None) -> LawResult:
    """Every reward mode must evaluate finite on every row."""
    import math

    interp = IrInterpreter(ir, instance=inst)
    modes = [m.name for m in ir.gym.reward_modes]
    for seed in range(3):
        traj = interp.run(seed)
        for row in traj.rows:
            for m in ir.gym.reward_modes:
                ns = interp._base_ns()
                ns.update(row)
                try:
                    v = float(interp._eval(m.expr, ns))
                except Exception as exc:  # noqa: BLE001
                    return _fail("finite_reward",
                                 f"mode {m.name!r}: {type(exc).__name__}: {exc}")
                if not math.isfinite(v):
                    return _fail("finite_reward", f"mode {m.name!r} gave {v}")
    return _pass("finite_reward", f"modes: {modes}")


def law_observation_modes(ir: MdpIR, inst: str | None, path: Path | None) -> LawResult:
    """Every observation mode must evaluate against every row."""
    interp = IrInterpreter(ir, instance=inst)
    traj = interp.run(0)
    modes = [m.name for m in ir.gym.observation_modes]
    # a row keys the clock as `t`; a feature refers to the time-index state
    # variable by its own name, so bind it before evaluating
    tv = _time_var(ir)
    for m in modes:
        for row in traj.rows:
            try:
                interp.observe(m, {**row, tv: row["t"]})
            except Exception as exc:  # noqa: BLE001
                return _fail("observation_modes",
                             f"mode {m!r}: {type(exc).__name__}: {exc}")
    return _pass("observation_modes", f"modes: {modes}")


def law_instances_run(ir: MdpIR, _inst: str | None, path: Path | None) -> LawResult:
    """Every declared instance must construct and roll an episode."""
    names = sorted(ir.mdp.scenario.instances)
    if not names:
        return _skip("instances_run", "no instances declared")
    for name in names:
        try:
            traj = IrInterpreter(ir, instance=name).run(0)
        except Exception as exc:  # noqa: BLE001
            return _fail("instances_run",
                         f"instance {name!r}: {type(exc).__name__}: {exc}")
        if not traj.rows:
            return _fail("instances_run", f"instance {name!r} produced no rows")
    return _pass("instances_run", f"{len(names)}: {names}")


def law_mixture_equivalence(ir: MdpIR, _inst: str | None, path: Path | None) -> LawResult:
    """Standalone equivalence (spec §5.3): every mixture episode must equal
    one component's standalone episode verbatim, and over enough episodes
    every component must actually occur."""
    if not ir.mdp.scenario.mixtures:
        return _skip("mixture_equivalence", "no mixtures declared")
    # ONE decision dict, computed from the base IR and reused everywhere: a
    # random policy would resolve its bounds per component, so a difference in
    # bounds — not in the world — could masquerade as a component mismatch
    dec = _fixed_decisions(ir, 0.0)
    resolutions = ir.mixture_resolutions or {}
    for m in ir.mdp.scenario.mixtures:
        comps = [c for _, c in m.components]
        # a CROSS-FAMILY component re-selects slot candidates, so its
        # standalone run is the catalog re-resolved under that component —
        # not this IR (whose slots resolved to the loaded selection). That
        # needs the schema path; without one, say so instead of comparing
        # against the wrong world.
        divergent = sorted(resolutions.get(m.name) or {})
        if divergent and path is None:
            return _skip("mixture_equivalence",
                         f"{m.name!r} has re-selecting component(s) {divergent}; "
                         f"pass the schema path to compare standalone")

        def standalone(comp: str) -> IrInterpreter:
            if comp in divergent:
                return IrInterpreter(load_ir(path, instance=comp), instance=comp)
            return IrInterpreter(ir, instance=comp or None)

        pure = {
            c: {e: standalone(c).run(e, decisions=dec).rows
                for e in range(EPISODES)}
            for c in comps
        }
        hit = dict.fromkeys(comps, 0)
        mix = IrInterpreter(ir, instance=m.name)
        for e in range(EPISODES):
            rows = mix.run(e, decisions=dec).rows
            matched = [c for c in comps if rows == pure[c][e]]
            if len(matched) != 1:
                return _fail("mixture_equivalence",
                             f"{m.name!r} episode {e} matches {matched or 'no'} "
                             f"component(s)")
            hit[matched[0]] += 1
        missing = [c or "<base>" for c, n in hit.items() if n == 0]
        if missing:
            return _fail("mixture_equivalence",
                         f"{m.name!r}: component(s) {missing} never drawn in "
                         f"{EPISODES} episodes")
    return _pass("mixture_equivalence",
                 f"{len(ir.mdp.scenario.mixtures)} mixture(s), "
                 f"{EPISODES} episodes each")


REGISTRY = [
    law_determinism,
    law_seed_sensitivity,
    law_decision_path_independence,
    law_invariants,
    law_termination,
    law_finite_reward,
    law_observation_modes,
    law_instances_run,
    law_mixture_equivalence,
]


# ---------------------------------------------------------------------------
# Runner + CLI
# ---------------------------------------------------------------------------


def run_laws(
    ir: MdpIR, instance: str | None = None, path: str | Path | None = None
) -> list[LawResult]:
    """Every law against one IR. A law that raises becomes a FAIL rather than
    taking down the gate. ``path`` is the IR's own schema file: only the
    mixture law needs it (to re-resolve a re-selecting component), and it
    SKIPs with a reason when it is missing."""
    out: list[LawResult] = []
    p = Path(path) if path is not None else None
    for law in REGISTRY:
        try:
            out.append(law(ir, instance, p))
        except Exception as exc:  # noqa: BLE001
            out.append(LawResult(getattr(law, "__name__", "law"), "FAIL",
                                 f"{type(exc).__name__}: {exc}"))
    return out


def format_report(name: str, results: list[LawResult]) -> str:
    width = max((len(r.law) for r in results), default=10)
    lines = [f"=== {name} ==="]
    for r in results:
        lines.append(f"  [{r.status.ljust(4)}] {r.law.ljust(width)}  {r.detail}")
    passed = sum(r.status == "PASS" for r in results)
    failed = any(r.status == "FAIL" for r in results)
    lines.append(f"  {passed}/{len(results)} passed"
                 f"{'  <-- LAWS FAILED' if failed else ''}")
    return "\n".join(lines)


def resolve_schema(path: str | Path) -> Path:
    """Accept a schema file or a domain directory holding exactly one."""
    p = Path(path)
    if p.is_dir():
        found = sorted(p.glob("*_schema.json"))
        if len(found) != 1:
            raise FileNotFoundError(
                f"{p}: expected exactly one *_schema.json, found {len(found)}"
            )
        return found[0]
    return p


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m mdp_ir.laws",
        description="Check IR execution semantics — the laws any IR must obey.",
    )
    ap.add_argument("targets", nargs="+", metavar="SCHEMA_OR_DIR")
    ap.add_argument("--instance", default=None,
                    help="run the laws under one instance (default: the base)")
    args = ap.parse_args(argv)

    failed = False
    for target in args.targets:
        path = resolve_schema(target)
        ir = load_ir(path, instance=args.instance)
        results = run_laws(ir, args.instance, path)
        print(format_report(ir.domain.name, results))
        print()
        failed |= any(r.status == "FAIL" for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

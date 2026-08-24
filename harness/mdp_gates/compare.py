"""Stage-4 eval gate: does the candidate beat each baseline by a real margin?

Reads the spec-§9 eval TSVs (shared by ``{domain}_ppo_eval.py`` /
``{domain}_benchmark_{method}_eval.py``) and compares a candidate against
each baseline on a ``*_mean`` metric column, using the matching ``*_var``
column and the seed count to form a standard error:

    advantage = (cand_mean - base_mean) * sense_sign      # sense_sign = +1 max, -1 min
    z         = advantage / sqrt((cand_var + base_var) / n_seeds)

``--sense`` sets whether the metric is maximized (default; higher is better,
e.g. ``revenue_mean``) or minimized (lower is better, e.g. ``cost_total_mean``
or ``regret_mean``). It MUST match the domain's objective sense — a cost- or
regret-reporting domain gated with the maximize default gets a silently
inverted verdict. The gate passes when z >= z_min for every baseline, so a
positive ``z`` always means "candidate is better under this sense".

All evals must have been run with the same seed protocol (seeds 0..n-1); the
unpaired SE above ignores the positive correlation induced by shared seeds,
which *overstates* the error, so a PASS is conservative.

``--reference`` files (e.g. the DP optimum) are reported — gap and % of
reference — but never gate.

Multi-row TSVs (scenario grids) are aggregated by averaging means and
variances across rows before comparison; per-cell gating is out of scope.

CLI:

    python -m mdp_gates --candidate ppo_eval_simple.tsv \
        --baseline benchmark_random_eval_simple.tsv --baseline benchmark_myopic_eval_simple.tsv \
        --reference benchmark_dp_eval_simple.tsv --n-seeds 8192 [--metric revenue_mean] [--z 2.0]

Exit status: 0 if every baseline comparison passes, else 1.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .completion import Harvest, harvest_verdict, read_provenance


@dataclass
class EvalStats:
    label: str
    metric: str
    mean: float
    var: float
    rows: int
    available_means: list[str] = field(default_factory=list)  # all *_mean columns present


@dataclass
class Comparison:
    baseline: EvalStats
    diff: float
    se: float
    z: float
    passed: bool


@dataclass
class GateReport:
    candidate: EvalStats
    n_seeds: int
    z_min: float
    sense: str = "maximize"
    comparisons: list[Comparison] = field(default_factory=list)
    references: list[EvalStats] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # spec §9.9 violations: a feasible arm (the candidate) strictly beating an
    # exact or relaxed one. Not a lost comparison — a bug report
    role_violations: list[str] = field(default_factory=list)
    # §8.6's "a training run runs to its budget" as a precondition: a number
    # from a run that stopped early is not a worse result, it is not a result
    harvest: Harvest | None = None

    @property
    def ok(self) -> bool:
        return (all(c.passed for c in self.comparisons)
                and not self.role_violations
                and not (self.harvest is not None and self.harvest.blocks))

    def render(self) -> str:
        c = self.candidate
        se_c = math.sqrt(c.var / self.n_seeds)
        better = "higher" if self.sense == "maximize" else "lower"
        lines = [
            f"candidate  {c.label}: {c.metric} = {c.mean:.4f} ± {se_c:.4f} "
            f"(SE, n={self.n_seeds}; sense={self.sense}, {better} is better)"
        ]
        if self.harvest is not None:
            tag = {"complete": "run ", "declared": "run ",
                   "unknown": "WARN", "short": "STOP"}[self.harvest.status]
            lines.append(f"  [{tag}] {self.harvest.detail}")
        lines += [f"  [WARN] {w}" for w in self.warnings]
        lines += [f"  [BUG ] {v}" for v in self.role_violations]
        for cmp in self.comparisons:
            b = cmp.baseline
            verdict = "PASS" if cmp.passed else "FAIL"
            lines.append(
                f"  [{verdict}] vs {b.label}: {b.mean:.4f}  "
                f"cand-base = {cmp.diff:+.4f}  z = {cmp.z:.2f} (need >= {self.z_min:g})"
            )
        for ref in self.references:
            gap = c.mean - ref.mean
            pct = 100.0 * c.mean / ref.mean if ref.mean else float("nan")
            lines.append(
                f"  [ref ] vs {ref.label}: {ref.mean:.4f}  "
                f"gap = {gap:+.4f}  ({pct:.1f}% of reference)"
            )
        lines.append("GATE " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(lines)


def _read_eval_tsv(path: Path, metric: str | None) -> EvalStats:
    """Aggregate one spec-§9 eval TSV; auto-picks the first *_mean column."""
    with open(path) as f:
        rows = list(csv.DictReader(
            (line for line in f if not line.startswith("#")), delimiter="\t",
        ))
    if not rows:
        raise ValueError(f"{path}: no data rows")

    mean_cols = [c for c in rows[0] if c.endswith("_mean")]
    if metric is None:
        metric = mean_cols[0] if mean_cols else None
        if metric is None:
            raise ValueError(f"{path}: no *_mean column; have {list(rows[0])}")
    if metric not in rows[0]:
        raise ValueError(f"{path}: no column {metric!r}; have {list(rows[0])}")
    var_col = metric.replace("_mean", "_var")
    if var_col not in rows[0]:
        raise ValueError(f"{path}: no matching variance column {var_col!r}")

    means = [float(r[metric]) for r in rows]
    varis = [float(r[var_col]) for r in rows]
    return EvalStats(
        label=path.stem,
        metric=metric,
        mean=sum(means) / len(means),
        var=sum(varis) / len(varis),
        rows=len(rows),
        available_means=mean_cols,
    )


def _method_from_path(path: Path) -> str | None:
    """The `{method}` of a spec-§9 benchmark eval TSV, or None if not one.

    `{domain}_benchmark_{method}_eval_{scenario}.tsv` — the same `{method}` that
    names the solver file and keys the results directory, which is what lets a
    declared role find its eval.
    """
    stem = path.name
    marker = "benchmark_"
    if marker not in stem:
        return None
    rest = stem[stem.index(marker) + len(marker):]
    if "_eval" not in rest:
        return None
    return rest[:rest.index("_eval")] or None


def roles_from_ir(schema_path: Path) -> dict[str, str]:
    """{method: role} from an IR's `benchmarks` block; empty if none declared."""
    from mdp_ir.schema import load_ir
    ir = load_ir(schema_path)
    return {b.name: b.role.value for b in getattr(ir, "benchmarks", [])}


def tolerances_from_ir(schema_path: Path,
                       instance: str | None = None) -> dict[str, float]:
    """{method: tolerance} — declared implementation slack, in metric units.

    Resolved under `instance`, because an entry's slack may name a scenario
    constant (§5.0): one solver file can serve two renderings whose numerical
    error differs, and the band this gate takes is a claim about the rendering
    being judged, not about the entry.
    """
    from mdp_ir.schema import load_ir
    ir = load_ir(schema_path)
    out: dict[str, float] = {}
    for b in getattr(ir, "benchmarks", []):
        try:
            t = ir.benchmark_tolerance(b.name, instance)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"benchmark {b.name!r}: {exc}") from exc
        if t:
            out[b.name] = t
    return out


class IllTypedBaseline(ValueError):
    """`--baseline` (must-beat) on a benchmark that cannot be beaten (§9.9)."""


def compare_evals(
    candidate: Path,
    baselines: list[Path],
    references: list[Path],
    n_seeds: int,
    metric: str | None = None,
    z_min: float = 2.0,
    sense: str = "maximize",
    roles: dict[str, str] | None = None,
    tolerances: dict[str, float] | None = None,
    z_role: float = 2.0,
    short_ok: str | None = None,
    budget_tolerance: float = 0.05,
) -> GateReport:
    if sense not in ("maximize", "minimize"):
        raise ValueError(f"sense must be 'maximize' or 'minimize', got {sense!r}")
    sense_sign = 1.0 if sense == "maximize" else -1.0
    cand = _read_eval_tsv(candidate, metric)
    report = GateReport(candidate=cand, n_seeds=n_seeds, z_min=z_min, sense=sense)
    # the precondition, before any comparison: a short run and a bad arm produce
    # the same low mean, and the gate would report the second correctly and the
    # first as a finding
    report.harvest = harvest_verdict(read_provenance(candidate),
                                     tolerance=budget_tolerance,
                                     declared=short_ok)
    if metric is None:
        # the metric was auto-picked (first *_mean column); the wrong pick
        # gates on the wrong quantity, and its sense may not match --sense.
        msg = (
            f"metric auto-selected as {cand.metric!r} (first *_mean column); "
            f"pass --metric to choose explicitly"
        )
        others = [m for m in cand.available_means if m != cand.metric]
        if others:
            msg += f". Other *_mean columns present: {others}"
        report.warnings.append(msg)
    roles = roles or {}
    tolerances = tolerances or {}
    # §9.9: --baseline is must-beat, so nominating an arm that cannot be beaten
    # by construction is a category error. Refuse the comparison rather than
    # running it and reporting a routine loss. Checked before any file is read,
    # so an ill-typed gate fails the same way whatever the numbers say.
    for path in baselines:
        role = roles.get(_method_from_path(path) or "")
        if role in ("exact", "relaxed"):
            raise IllTypedBaseline(
                f"--baseline {path.name} is declared role={role} (spec §9.9): "
                f"{'an exact solver is the optimum' if role == 'exact' else 'a relaxation is ≽ the optimum'}, "
                f"so a deployable policy cannot beat it. Pass it as --reference."
            )
    for path in baselines:
        base = _read_eval_tsv(path, metric or cand.metric)
        diff = cand.mean - base.mean            # raw mean difference (cand - base)
        advantage = sense_sign * diff           # >0 ⇔ candidate better under sense
        se = math.sqrt((cand.var + base.var) / n_seeds)
        z = advantage / se if se > 0 else math.copysign(math.inf, advantage)
        report.comparisons.append(
            Comparison(baseline=base, diff=diff, se=se, z=z, passed=z >= z_min)
        )
    for path in references:
        ref = _read_eval_tsv(path, metric or cand.metric)
        report.references.append(ref)
        # the candidate is `feasible` by construction (a real policy under the
        # real information set), so beating an exact or relaxed arm is
        # impossible — §9.9: a bug report, not a result.
        #
        # Compared against a BAND, not to the last float. Two independent
        # reasons, and either alone would justify it: an `exact` solver is
        # exact in its logic, not its arithmetic (it truncates a tail,
        # discretizes a state, stops at a tolerance — §9.9 says `basis` carries
        # that), and both arms are Monte-Carlo means, so CRN shrinks the paired
        # variance without removing it. The band is the declared implementation
        # slack or the statistical term, whichever is larger. Inside it, report;
        # outside it, indict. Firing on 1e-6 would tell the one campaign that
        # closed on its bar that its simulator is broken, and the quickest way
        # to silence that is to demote the bar — destroying the bracket the role
        # exists to make checkable (upstream #27).
        role = roles.get(_method_from_path(path) or "")
        if role in ("exact", "relaxed"):
            advantage = sense_sign * (cand.mean - ref.mean)
            se = math.sqrt((cand.var + ref.var) / n_seeds)
            declared = tolerances.get(_method_from_path(path) or "", 0.0)
            band = max(declared, z_role * se)
            if advantage > band:
                report.role_violations.append(
                    f"candidate ({cand.metric} = {cand.mean:.4f}) beats {ref.label} "
                    f"({ref.mean:.4f}) by {advantage:.4f} > band {band:.4f} "
                    f"(declared {declared:.4f}, {z_role:g}·SE {z_role * se:.4f}), "
                    f"declared role={role} — impossible under §9.9: the eval, "
                    f"the bound, or the simulator is wrong. Not a result."
                )
            elif advantage > 0:
                report.warnings.append(
                    f"candidate is {advantage:.4f} ahead of {ref.label} "
                    f"(role={role}) but within the band {band:.4f} — not "
                    f"separable from implementation slack and eval noise, so "
                    f"this is a `~`, not a defect claim and not a win"
                )
    return report


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m mdp_gates",
        description="Gate a candidate eval TSV against baseline eval TSVs (mean ± SE).",
    )
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--baseline", action="append", default=[], type=Path,
                    help="baseline eval TSV the candidate must beat (repeatable)")
    ap.add_argument("--reference", action="append", default=[], type=Path,
                    help="reference eval TSV reported but not gated, e.g. the DP optimum")
    ap.add_argument("--n-seeds", required=True, type=int,
                    help="seed count the evals were run with (all must match)")
    ap.add_argument("--metric", default=None,
                    help="metric column (default: first *_mean column)")
    ap.add_argument("--z", type=float, default=2.0,
                    help="required z-score margin per baseline (default 2.0)")
    ap.add_argument("--sense", choices=("maximize", "minimize"), default="maximize",
                    help="objective sense of the metric: maximize (higher is better, "
                         "default) or minimize (lower is better, e.g. a cost/regret "
                         "column). Must match the domain's objective sense.")
    ap.add_argument("--z-role", type=float, default=2.0,
                    help="z multiplier for the §9.9 ordering band: a candidate "
                         "ahead of an exact/relaxed arm by less than "
                         "max(declared tolerance, z*SE) is reported, not failed "
                         "(default 2.0)")
    ap.add_argument("--short-ok", default=None, metavar="REASON",
                    help="record that the candidate's run was stopped "
                         "deliberately, with the reason. A killed run and a "
                         "silently dead one leave identical artifacts, so this "
                         "is the one input no file can carry — and stating it "
                         "here puts it in the gate output the ledger quotes.")
    ap.add_argument("--budget-tolerance", type=float, default=0.05,
                    help="how far short of its declared budget a run may end "
                         "and still read as complete (default 0.05 — one §8.6 "
                         "checkpoint cadence, so a completed run whose last "
                         "artifact predates its final step still passes)")
    ap.add_argument("--instance", default=None,
                    help="IR instance the candidate was produced under; a "
                         "benchmark tolerance naming a scenario constant "
                         "resolves against it (§5.0). Only read with --ir")
    ap.add_argument("--ir", default=None, type=Path,
                    help="domain schema declaring benchmark roles (spec §9.9). With it, "
                         "--baseline on an exact/relaxed arm is refused, and a candidate "
                         "beating one fails the gate as a bug report")
    args = ap.parse_args(argv)

    if not args.baseline and not args.reference:
        ap.error("nothing to compare: pass --baseline and/or --reference")

    roles: dict[str, str] = {}
    tolerances: dict[str, float] = {}
    if args.ir is not None:
        try:
            roles = roles_from_ir(args.ir)
            tolerances = tolerances_from_ir(args.ir, args.instance)
        except Exception as e:
            ap.error(f"--ir {args.ir}: {type(e).__name__}: {e}")

    try:
        report = compare_evals(
            candidate=args.candidate,
            baselines=args.baseline,
            references=args.reference,
            n_seeds=args.n_seeds,
            metric=args.metric,
            z_min=args.z,
            sense=args.sense,
            roles=roles,
            tolerances=tolerances,
            z_role=args.z_role,
            short_ok=args.short_ok,
            budget_tolerance=args.budget_tolerance,
        )
    except IllTypedBaseline as e:
        print(f"GATE REFUSED: {e}", file=sys.stderr)
        return 2
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

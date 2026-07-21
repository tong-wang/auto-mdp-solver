"""Stage-4 eval gate: does the candidate beat each baseline by a real margin?

Reads the spec-§9 eval TSVs (shared by ``{domain}_ppo_eval.py`` /
``{domain}_benchmark_{method}_eval.py``) and compares a candidate against
each baseline on a ``*_mean`` metric column, using the matching ``*_var``
column and the seed count to form a standard error:

    z = (cand_mean - base_mean) / sqrt((cand_var + base_var) / n_seeds)

The gate passes when z >= z_min for every baseline. All evals must have been
run with the same seed protocol (seeds 0..n-1); the unpaired SE above ignores
the positive correlation induced by shared seeds, which *overstates* the
error, so a PASS is conservative.

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


@dataclass
class EvalStats:
    label: str
    metric: str
    mean: float
    var: float
    rows: int


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
    comparisons: list[Comparison] = field(default_factory=list)
    references: list[EvalStats] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.comparisons)

    def render(self) -> str:
        c = self.candidate
        se_c = math.sqrt(c.var / self.n_seeds)
        lines = [
            f"candidate  {c.label}: {c.metric} = {c.mean:.4f} ± {se_c:.4f} "
            f"(SE, n={self.n_seeds})"
        ]
        for cmp in self.comparisons:
            b = cmp.baseline
            verdict = "PASS" if cmp.passed else "FAIL"
            lines.append(
                f"  [{verdict}] vs {b.label}: {b.mean:.4f}  "
                f"diff = {cmp.diff:+.4f}  z = {cmp.z:.2f} (need >= {self.z_min:g})"
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

    if metric is None:
        metric = next((c for c in rows[0] if c.endswith("_mean")), None)
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
    )


def compare_evals(
    candidate: Path,
    baselines: list[Path],
    references: list[Path],
    n_seeds: int,
    metric: str | None = None,
    z_min: float = 2.0,
) -> GateReport:
    cand = _read_eval_tsv(candidate, metric)
    report = GateReport(candidate=cand, n_seeds=n_seeds, z_min=z_min)
    for path in baselines:
        base = _read_eval_tsv(path, metric or cand.metric)
        diff = cand.mean - base.mean
        se = math.sqrt((cand.var + base.var) / n_seeds)
        z = diff / se if se > 0 else math.copysign(math.inf, diff)
        report.comparisons.append(
            Comparison(baseline=base, diff=diff, se=se, z=z, passed=z >= z_min)
        )
    for path in references:
        report.references.append(_read_eval_tsv(path, metric or cand.metric))
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
    args = ap.parse_args(argv)

    if not args.baseline and not args.reference:
        ap.error("nothing to compare: pass --baseline and/or --reference")

    report = compare_evals(
        candidate=args.candidate,
        baselines=args.baseline,
        references=args.reference,
        n_seeds=args.n_seeds,
        metric=args.metric,
        z_min=args.z,
    )
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

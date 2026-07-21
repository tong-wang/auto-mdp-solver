"""Differential runner: IR interpreter vs a real ``_mdp`` implementation.

The Stage-1 codegen gate (SKILL.md Stage 1): replay identical ``(instance,
episode_seed, decisions)`` through both the restricted IR interpreter
(``mdp_ir/interpreter.py``) and a domain's ``init_state`` / ``advance``
functions, and require the trajectories to match field-for-field. A generated
simulator that passes the conformance harness but models the *wrong problem*
fails here — this is the executable oracle for formalization correctness:
formalization has no oracle of its own, so a self-consistent but wrongly
modeled domain passes every unit test and is caught only here.

A **domain adapter** hides the domain's API shape: it is a callable
``(episode_seed, decisions_per_period) -> rows`` returning per-period dicts
keyed like interpreter rows. Only the keys the adapter emits are compared, so
an adapter chooses the fields it vouches for. The adapter builds its scenario
object *from the IR's scenario constants* (exactly what generated code will
do), so both sides consume one source of truth.

Adapters live **with their domain**, not here (portable-domain contract): the
IR's directory contains ``{ir.domain.name}_ir_adapter.py`` exposing
``make_adapter(ir, instance=None, seed_salt=0, domain_dir=None) -> DomainAdapter``
with ``domain_dir`` defaulting to the adapter's own directory, so a domain
folder works unchanged wherever it lives. ``examples/inv_single`` ships the
reference adapter.

CLI:

    python -m mdp_ir.differential examples/inv_single/inv_single_schema.json \
        [--episodes 20] [--instance lost_sales] [--seed-salt 0] \
        [--decision order=40] [--max-report 10]

Exit status is non-zero if any episode diverges.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from mdp_ir.interpreter import IrInterpreter
from mdp_ir.schema import MdpIR

# adapter: (episode_seed, decisions per period) -> per-period rows to compare
DomainAdapter = Callable[[int, list[dict]], list[dict]]

_ATOL = 1e-9


@dataclass
class Divergence:
    episode_seed: int
    period: int
    fld: str
    ir_value: object
    domain_value: object

    def __str__(self) -> str:
        return (
            f"seed={self.episode_seed} t={self.period} field={self.fld!r}: "
            f"interpreter={self.ir_value!r} domain={self.domain_value!r}"
        )


@dataclass
class DifferentialReport:
    domain: str
    instance: str | None
    episodes: int = 0
    periods: int = 0
    fields: list[str] = field(default_factory=list)
    divergences: list[Divergence] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.divergences

    def render(self, max_report: int = 10) -> str:
        inst = f" instance={self.instance}" if self.instance else ""
        head = (
            f"{self.domain}{inst}: {self.episodes} episodes, "
            f"{self.periods} periods, fields={self.fields}"
        )
        if self.ok:
            return f"MATCH  {head}"
        lines = [f"DIVERGED  {head}  ({len(self.divergences)} divergence(s))"]
        lines += [f"  {d}" for d in self.divergences[:max_report]]
        if len(self.divergences) > max_report:
            lines.append(f"  ... ({len(self.divergences) - max_report} more)")
        return "\n".join(lines)


def _values_differ(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) > _ATOL
    if isinstance(a, list) and isinstance(b, list):
        return len(a) != len(b) or any(_values_differ(x, y) for x, y in zip(a, b))
    return a != b


def diff_rows(
    ir_rows: list[dict], domain_rows: list[dict], episode_seed: int
) -> list[Divergence]:
    """Compare trajectories field-by-field over the domain rows' keys."""
    out: list[Divergence] = []
    if len(ir_rows) != len(domain_rows):
        out.append(
            Divergence(episode_seed, -1, "episode_length",
                       len(ir_rows), len(domain_rows))
        )
    for ir_row, dom_row in zip(ir_rows, domain_rows):
        for fld, dom_val in dom_row.items():
            if fld not in ir_row:
                out.append(Divergence(episode_seed, ir_row["t"], fld,
                                      "<missing>", dom_val))
            elif _values_differ(ir_row[fld], dom_val):
                out.append(Divergence(episode_seed, ir_row["t"], fld,
                                      ir_row[fld], dom_val))
    return out


def run_differential(
    ir: MdpIR,
    adapter: DomainAdapter,
    episode_seeds: list[int],
    decisions: dict | None = None,
    instance: str | None = None,
    seed_salt: int = 0,
) -> DifferentialReport:
    """Replay each episode through interpreter and domain; diff trajectories.

    The interpreter runs first (random policy within bounds unless
    ``decisions`` fixes them); its per-period decisions are then fed verbatim
    to the adapter, so both sides see the identical action sequence."""
    interp = IrInterpreter(ir, instance=instance, seed_salt=seed_salt)
    decision_names = [d.name for d in ir.mdp.decisions]
    report = DifferentialReport(domain=ir.domain.name, instance=instance)

    for seed in episode_seeds:
        traj = interp.run(seed, decisions=decisions)
        acts = [{n: row[n] for n in decision_names} for row in traj.rows]
        domain_rows = adapter(seed, acts)
        if domain_rows and not report.fields:
            report.fields = sorted(domain_rows[0].keys())
        report.divergences += diff_rows(traj.rows, domain_rows, seed)
        report.episodes += 1
        report.periods += len(traj.rows)

    return report


# ---------------------------------------------------------------------------
# Adapter discovery
# ---------------------------------------------------------------------------


def _import_domain(domain_dir: Path, modules: list[str]) -> list[object]:
    """Import a domain's flat modules with its directory on sys.path."""
    import importlib

    path = str(domain_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    return [importlib.import_module(m) for m in modules]


def load_adapter_factory(
    ir_path: str | Path, ir: MdpIR
) -> Callable[..., DomainAdapter]:
    """Locate the adapter module next to the IR file and return its factory.

    Convention: ``{ir.domain.name}_ir_adapter.py`` in the IR's directory,
    exposing ``make_adapter(ir, instance=None, seed_salt=0, domain_dir=None)``.
    If that exact name is absent but the directory holds exactly one
    ``*_ir_adapter.py``, that one is used.
    """
    directory = Path(ir_path).resolve().parent
    module_name = f"{ir.domain.name}_ir_adapter"
    if not (directory / f"{module_name}.py").exists():
        candidates = sorted(directory.glob("*_ir_adapter.py"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"no adapter for {ir.domain.name!r}: expected "
                f"{directory / (module_name + '.py')}"
                + (f" (found {[c.name for c in candidates]})" if candidates else "")
            )
        module_name = candidates[0].stem
    (module,) = _import_domain(directory, [module_name])
    return module.make_adapter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    import argparse

    from mdp_ir.schema import load_ir

    ap = argparse.ArgumentParser(
        prog="python -m mdp_ir.differential",
        description="Diff IR-interpreter trajectories against a real domain implementation.",
    )
    ap.add_argument("ir_file")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--seed-salt", type=int, default=0)
    ap.add_argument(
        "--decision", action="append", default=[], metavar="NAME=VALUE",
        help="hold a decision constant (default: random within bounds per episode)",
    )
    ap.add_argument("--max-report", type=int, default=10)
    args = ap.parse_args(argv)

    ir = load_ir(args.ir_file)
    try:
        factory = load_adapter_factory(args.ir_file, ir)
    except FileNotFoundError as exc:
        ap.error(str(exc))

    fixed: dict[str, float] = {}
    for spec in args.decision:
        name, _, val = spec.partition("=")
        fixed[name.strip()] = float(val)

    adapter = factory(ir, instance=args.instance, seed_salt=args.seed_salt)
    report = run_differential(
        ir,
        adapter,
        episode_seeds=list(range(args.first_seed, args.first_seed + args.episodes)),
        decisions=fixed or None,
        instance=args.instance,
        seed_salt=args.seed_salt,
    )
    print(report.render(max_report=args.max_report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

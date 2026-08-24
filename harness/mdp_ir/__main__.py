"""Validate IR JSON files against the schema.

    python -m mdp_ir plugin/skills/mdp-solver/examples/inv_single/inv_single_schema.json [more.json ...]
    python -m mdp_ir <schema.json> --regroup    # rewrite into model/design/rendering groups

A catalog schema (per-slot candidates, IR_LAYERING_PLAN §10) is validated by
resolving its base selection **and every named instance** — each must produce
a valid ``MdpIR``. Exit status is non-zero if any file fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from mdp_ir import layering
from mdp_ir.schema import load_ir


def regroup(paths: list[str]) -> int:
    """Rewrite each file into the three headed groups, in place.

    Refuses to write if any fingerprint would move: the grouping is a file
    layout for human readers, and a recorded fingerprint is what a frozen case
    cites. A cosmetic change that moved one would be a defect, not a feature.
    """
    from mdp_ir.schema import group_mdp

    failed = 0
    for path in paths:
        p = Path(path)
        raw = json.loads(p.read_text())
        before = (load_ir(p).mdp_fingerprint(), _rendering_hash(raw))
        grouped = dict(raw)
        grouped["mdp"] = group_mdp(raw["mdp"])
        tmp = p.with_suffix(".regrouped.tmp")
        tmp.write_text(json.dumps(grouped, indent=2) + "\n")
        try:
            after = (load_ir(tmp).mdp_fingerprint(), _rendering_hash(grouped))
        finally:
            if before != after:
                tmp.unlink(missing_ok=True)
        if before != after:
            print(f"REFUSED  {path}: fingerprints would move {before} -> {after}")
            failed += 1
            continue
        tmp.replace(p)
        print(f"regrouped {path}  (fingerprints unmoved: {before[0]}"
              + (f" / {before[1]}" if before[1] else "") + ")")
    return 1 if failed else 0


def _rendering_hash(raw: dict) -> str | None:
    return layering.structural_fingerprint(raw) if layering.is_catalog(raw) else None


def _beta(ir, m) -> str:
    """β as the CLI should show it: the number where the IR pins one, and
    `name->value` where it names a scenario constant (§5.0). An unresolvable
    symbol prints as `name->?` rather than raising — validation reports it."""
    raw = m.objective.discount_factor
    if not isinstance(raw, str):
        return str(raw)
    try:
        return f"{raw}->{m.discount_factor()}"
    except (KeyError, TypeError, ValueError):
        return f"{raw}->?"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    if "--regroup" in argv:
        return regroup([a for a in argv if a != "--regroup"])

    failures = 0
    for path in argv:
        try:
            raw = json.loads(Path(path).read_text())
            catalog = layering.is_catalog(raw)
            if catalog:
                scenario_raw = (layering._flat_mdp(raw).get("scenario") or {})
                inst_names = sorted(scenario_raw.get("instances") or {})
                inst_names += sorted(
                    m["name"] for m in scenario_raw.get("mixtures") or []
                )
                ir = load_ir(path)                      # base selection
                for inst in inst_names:                 # every instance + mixture resolves
                    load_ir(path, instance=inst)
            else:
                inst_names = []
                ir = load_ir(path)
        except (ValidationError, ValueError, OSError) as exc:
            failures += 1
            print(f"FAIL  {path}")
            print(f"      {exc}")
            continue
        m, g, r = ir.mdp, ir.gym, ir.rl
        unconfirmed = ir.unconfirmed()
        print(
            f"OK    {path}  [{ir.domain.name} v{ir.ir_version}]\n"
            f"      mdp: states={len(m.state_variables)} info={len(m.info_fields)} "
            f"decisions={len(m.decisions)} sources={len(m.uncertainty_sources)} "
            f"constants={len(m.scenario.constants)} instances={len(m.scenario.instances)} "
            f"beta={_beta(ir, m)} fingerprint={ir.mdp_fingerprint()}\n"
            f"      gym: obs_modes={len(g.observation_modes)} act_modes={len(g.action_modes)} "
            f"reward_modes={len(g.reward_modes)} horizon_end={g.termination.horizon_end.value}\n"
            f"      rl : algo={r.algo.value} requires_memory={r.requires_memory.value}"
            f"({r.requires_memory.source.value}) frame_stack={r.frame_stack} "
            f"obs_norm={r.obs_normalization.enabled}"
        )
        if catalog:
            slots = layering._flat_mdp(raw)["uncertainty_slots"]
            menu = "  ".join(
                f"{s['name']}: "
                + " | ".join(
                    c + ("*" if c == s["default"] else "")
                    for c in s["candidates"]
                )
                for s in slots
            )
            print(
                f"      catalog: {menu}\n"
                f"      structural_fingerprint={layering.structural_fingerprint(raw)} "
                f"(instances validated: base"
                + (", " + ", ".join(inst_names) if inst_names else "")
                + ")"
            )
        if unconfirmed:
            print(f"      unconfirmed ({len(unconfirmed)}): {', '.join(unconfirmed)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

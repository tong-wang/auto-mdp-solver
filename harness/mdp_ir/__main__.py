"""Validate IR JSON files against the schema.

    python -m mdp_ir plugin/skills/mdp-solver/examples/inv_single/inv_single_schema.json [more.json ...]

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


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    failures = 0
    for path in argv:
        try:
            raw = json.loads(Path(path).read_text())
            catalog = layering.is_catalog(raw)
            if catalog:
                scenario_raw = (raw["mdp"].get("scenario") or {})
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
            f"beta={m.objective.discount_factor} fingerprint={ir.mdp_fingerprint()}\n"
            f"      gym: obs_modes={len(g.observation_modes)} act_modes={len(g.action_modes)} "
            f"reward_modes={len(g.reward_modes)} horizon_end={g.termination.horizon_end.value}\n"
            f"      rl : algo={r.algo.value} requires_memory={r.requires_memory.value}"
            f"({r.requires_memory.source.value}) frame_stack={r.frame_stack} "
            f"obs_norm={r.obs_normalization.enabled}"
        )
        if catalog:
            slots = raw["mdp"]["uncertainty_slots"]
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

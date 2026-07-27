"""Validate IR JSON files against the schema.

    python -m mdp_ir plugin/skills/mdp-solver/examples/inv_single/inv_single_schema.json [more.json ...]

Exit status is non-zero if any file fails validation.
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

from mdp_ir.schema import load_ir


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    failures = 0
    for path in argv:
        try:
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
        if unconfirmed:
            print(f"      unconfirmed ({len(unconfirmed)}): {', '.join(unconfirmed)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

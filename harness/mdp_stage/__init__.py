"""Pipeline entry gates: is a domain folder ready for the op you want to run?

The split pipeline (AGENT_PLAN §15) cuts the mdp-solver skill at
durable-artifact seams — formalize / build / solve / escalate / interpret /
package — and each op re-validates its upstream gate at its own entry rather
than trusting that someone "just signed off." This tool is that re-validation
in executable form:

    python -m mdp_stage <domain-dir>              # stage table, cheap checks
    python -m mdp_stage <domain-dir> --for solve  # one op's full entry gate, exit 0/1

Cross-op state splits by derivability. Everything derivable is re-derived
here (IR validity, ``unconfirmed()``, the fingerprint, conformance / laws /
differential verdicts, artifact presence) — never recorded, because a
recorded verdict goes stale the moment the folder moves on. What is *not*
derivable lives in two single-writer files beside the schema, and this tool
only ever reads them:

- ``{name}.signoff.json`` — written by formalize at its gate: the
  ``mdp_fingerprint`` at human sign-off, the sign-off date, the restatement
  filename. The always-on freeze check compares it against the *current*
  fingerprint, so a post-freeze edit of the mdp block is caught at every
  later op's entry instead of relying on one conversation's memory.
- ``{name}.runplan.json`` — written by solve at run-plan confirmation:
  ``target``, ``strategy``, the escalation budget. Downstream ops locate
  their artifacts through it.

Judgment stays out by design: the tool checks artifacts and verdicts, never
"is the gap big enough to escalate" — that diagnosis belongs to the brain
reading the leaderboard, per the skill.
"""

from mdp_stage.gate import GateResult, OPS, run_gate, stage_table

__all__ = ["GateResult", "OPS", "run_gate", "stage_table"]

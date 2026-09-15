---
name: mdp-interpret
description: >
  Stage 5 of the MDP pipeline: read the winning policy back into the
  domain's policy-structure vocabulary (spec §14) — probe, fitted rule scored
  paired, overlay figure — anchored to a reference when one exists. Use when
  a winning artifact exists and a declared confirm/discover stance owes the
  readback. Full pipeline: mdp-solver.
---

# mdp-interpret — Stage 5 (policy-structure readback)

One op of the split MDP pipeline; `mdp-solver` is the conductor. The governing
docs are at `${CLAUDE_SKILL_DIR}/../mdp-solver/` — read them from that path,
never from a filesystem search (a dev checkout on disk would silently
substitute unreleased content). Read `CONTRACTS.md` there first: it maps the
ops and says what each reads. **This op needs:** spec `MDP_PROJECT_SPEC.md`
§14 (this op's specification — consult it while writing each file; do not code
from memory of it), plus §1.2 and §8.4 where named below — nothing else of the
spec, no guide, no templates. Everything else in that directory is consulted
only when a step below names it. Venv and run discipline per `ENVIRONMENT.md`.

**Entry gate — run first, exit 0 required:**

```bash
<python> -m mdp_stage {domain} --for interpret
```

Its `interpret.owed` line states whether a declared stance owes the §14
deliverables or only the generic pieces apply (spec §14.0) — quote it rather
than re-deciding.

What did the policy learn? Read the winning artifact back into the domain's
policy-structure vocabulary per spec §14. **Anchored** whenever a reference
policy or a predicted structural class exists — any domain with a DP
baseline qualifies. With neither, only the generic pieces apply (the
action-surface figure, the feature-sensitivity sweeps); record what they
show and move on.

- `{domain}_policy_probe.py` per spec §14.1: the action surface over a
  state grid, the structural-form statistic, recovered thresholds + action
  agreement vs the reference, feature-sensitivity sweeps. Validate the
  probe first on an instance whose optimal structure is known before
  trusting it where none is. If a probe carries its own implementation of
  the MDP core (a vectorized sim for sweeps), it is licensed but gated:
  declare the equivalence it claims and prove it in `{domain}_test.py`
  (spec §1.2's second-implementation gates) — probe outputs land in the
  run's `probe/` dir, readback conclusions in `interpret/` (spec §8.4).
- Fit the predicted structural rule and **score it under the Stage-4 eval
  protocol** (same seeds, paired): report reference / fitted rule / net,
  with the verdict branches pre-decided per spec §14.2. The fitted rule
  beating its own net is a real outcome — when it happens, the fitted rule
  is a shippable artifact; ship it and say so.
- `{domain}_plot_policy.py` per spec §14.3: the overlay figure in canonical
  coordinates; committed static render in `{domain}/figures/`, interactive
  HTML beside the runs in `results/{scenario}/figures/` (gitignored);
  committed markdown cites the regenerating command.

No hard gate: a failed structural recovery fires spec §14.2's diagnosis
branches but does not block packaging — the finding (including "it is not
doing the classical thing") is a verdict on a declared stance and goes in the
README's `2-structural` results (spec §1.3).

Exit: probe outputs in the run's `probe/` dir, readback conclusions in
`interpret/`, the committed figure in `{domain}/figures/`, and the verdict on
each declared stance stated for the README's `2-structural` block — hand to
`mdp-package` (or back to the conductor).

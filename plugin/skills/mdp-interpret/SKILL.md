---
name: mdp-interpret
description: >
  Stage 5 of the MDP pipeline: read the winning trained policy back into the
  domain's policy-structure vocabulary per spec §14 — the probe, the fitted
  structural rule scored under the paired protocol, the overlay figure —
  anchored against a reference policy when one exists. Use when a winning
  artifact exists and a declared tier-2 stance (confirm/discover) owes the
  readback, or on request for the generic pieces. Entry: python -m mdp_stage
  {domain} --for interpret. The full automatic pipeline is mdp-solver.
---

# mdp-interpret — Stage 5 (policy-structure readback)

One op of the split MDP pipeline (`mdp-solver` is the conductor; its
`CONTRACTS.md` maps the ops). The governing docs live beside the conductor in
this plugin's `mdp-solver` skill directory —
`${CLAUDE_SKILL_DIR}/../mdp-solver/` holds `MDP_PROJECT_SPEC.md` (§14 is this
op's specification — consult it while writing each file; do not code from
memory of it), `ENVIRONMENT.md`, `CONTRACTS.md` and `examples/`. **Read them
from that path. Do not search the filesystem for them**: a development
checkout of the solver may also be on disk, and reading that instead silently
substitutes unreleased content for the version you are installed at. Use the
workspace venv per `ENVIRONMENT.md`.

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

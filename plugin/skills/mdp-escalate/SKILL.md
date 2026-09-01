---
name: mdp-escalate
description: >
  L2+ escalation rounds of the MDP pipeline: after mdp-solve's L1 gate shows
  a gap against the baselines, open the hp / gym / arch levers —
  diagnosis-driven, playbook-consulted, ledger-addressed — within the run
  plan's escalation budget. Use when an L1 leaderboard row exists and
  underperforms, and the diagnosis says the build is healthy. Entry: python
  -m mdp_stage {domain} --for escalate, plus the campaign log. The full
  automatic pipeline is mdp-solver.
---

# mdp-escalate — L2+ rounds (hp / gym / arch)

One op of the split MDP pipeline (`mdp-solver` is the conductor; its
`CONTRACTS.md` maps the ops). The governing docs live beside the conductor in
this plugin's `mdp-solver` skill directory —
`${CLAUDE_SKILL_DIR}/../mdp-solver/` holds `MDP_PROJECT_SPEC.md` (§8.6 is
this op's backbone reference), `ESCALATION_LOG_GUIDE.md` (**the campaign-log
format — its §-numbers are cited from case logs and never renumber**),
`PLAYBOOK.md`, `ENVIRONMENT.md`, `CONTRACTS.md` and `examples/`. **Read them
from that path. Do not search the filesystem for them**: a development
checkout of the solver may also be on disk, and reading that instead silently
substitutes unreleased content for the version you are installed at. Use the
workspace venv per `ENVIRONMENT.md` — its retry budgets (1 repair attempt
per failing design axis), >30-minute ask, and background-run rules govern
every launch below.

**Entry gate — run first, exit 0 required:**

```bash
<python> -m mdp_stage {domain} --for escalate
```

It re-derives the freeze check plus solve's exit artifacts (run plan,
baselines, an RL row) and requires `{domain}/ESCALATION.md` — the guide's
no-run-without-an-address rule made mechanical: rounds live at MAP addresses
and runs are cited from the §LEDGER, so the log exists *before* the first
L2+ launch, seeded per `ESCALATION_LOG_GUIDE.md`. Also confirm the premise,
not just the artifacts: escalation is only licensed by `mdp-solve`'s exit
diagnosis — a gap **with a healthy build**. If L1 sits at or below random,
that verdict is a build bug; go back instead of escalating.

## Rounds

A round is: diagnose the symptom from cheap evidence → choose the lever(s) —
**hp** (tuning), **gym** (obs/action/reward reshaping), **arch**
(`policy_kwargs`) — → run within budget → evaluate under the *same* Stage-4
protocol and `mdp_gates` gate as every other arm → record. A tuned result is
always L2: level ≥ L2 ⟺ more than one training config was tried (spec §8.6).
Tag each leaderboard row by the layers that actually moved (`L2(hp)`,
`L3(hp+arch)`, …), and label every row clearly against L0/L1.

Before designing any L2+ escalation, consult the shipped playbook:
`PLAYBOOK.md` in the `mdp-solver` skill directory (path above) is the
curated index, linking into
`examples/*/PLAYBOOK.md` — each a campaign's distilled experiences. Match on
*symptom*, read the matched entry **with its context**, and analogize —
entries are experiences, not rules; their scope conditions decide whether one
applies here. An entry's `#E` citations resolve in that example's
`ESCALATION.md`; drill down only when the entry's story needs chasing.

- Tuning — the escalation opened only when the L1 gate shows a
  gap (or on request): `{domain}_ppo_tune.py` thin wrapper over `mdp_tuning`
  pre-filling `--metric <metric>_mean` (and `--beta`/`--episode-len` from
  the IR). The driver warm-starts trial 0 from the script defaults (the L1
  center) and tunes the `core` knob tier by default — `--knobs breadth`
  needs ≥ ~40 trials, and is also where the two normalization priors
  (`norm_obs`, `normalize_advantage`) get checked, since §8.6's rows for
  them are settled by a run and by nothing earlier. **Tag the round by what
  moved, not by the fact that a study produced it:** the tiers cross layer
  lines (`core` reaches the extractor knobs, `breadth` reaches `norm_obs`),
  so a winner can be `L3(hp+arch)` or `L3(hp+gym)` — `--show-space` says
  which layers a study can reach before it runs, `--fix` closes one.
  (The diagnosis that *licenses* opening a study at all is `mdp-solve`'s
  exit, quoted at the entry gate above.)
  **Selection-bias rule:** the study winner was selected on its tuning
  seeds — always re-evaluate the winning artifact with the full protocol
  before comparing or shipping, and expect the score to drop. A config
  tuned at a small budget does not necessarily improve when retrained
  longer; prefer shipping the tuned artifact itself. If you do retrain a
  winner, read its config off the trial's own args log: a trial command is
  the **derivation plus the delta the study searched** (spec §8.6), so
  "the script's defaults plus the winning cfg" is a different configuration
  wherever a default has moved off the derivation.


Gym-layer rounds carry the one upstream hazard: an obs/reward feature that
needs information `_mdp` does not publish is not a gym edit — it re-opens
`mdp-build` (info amendment → differential re-run) or the IR itself (new
information → `mdp-formalize`). Say so and route it; never patch it in the
wrapper.

## Stopping

Two exits, both honest: **competitive** — the escalated arm clears the
`mdp_gates` gate after full-protocol re-evaluation (selection-bias rule
above) — or **budget exhausted** per the run plan. Budget exhaustion without
clearing the bar is a real result: record it in the log (failed experiments
are §LEDGER content too) and report it plainly — do not quietly extend the
budget. Either way, update `ESCALATION.md` per the guide, put the final rows
on the leaderboard, and hand the winning artifact (which may still be L1) to
`mdp-interpret` — or back to the conductor.

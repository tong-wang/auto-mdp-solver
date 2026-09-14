# CLAUDE.md — `inv_single/` (single-item periodic-review inventory control)

Domain-local operating brief for an agent about to change this folder.
**This file points up: at the specs that govern the folder, and the version
of them its numbers were produced under.** Everything the campaign produced —
its documents, its code, its numbers — is inventoried in `README.md` and is
linked from here, never copied.

One stocking point, one item, 30 periods. Each period places a replenishment
order that arrives after a lead time; unmet demand is backlogged or lost
depending on the instance. Minimize undiscounted holding + shortage + ordering
cost (β = 1.0). This is the smallest domain with an exactly-solvable DP
reference, so it doubles as the pipeline's correctness fixture.

**Every scenario in `SCENARIOS` is a SEPARATE LEADERBOARD**, and so is every
`grid16` cell. They differ in lead time, stockout mode, demand family and fixed
cost — `simple` (L=0, K=0) and `slt` (stochastic L∈{1,2,3}) are different
problems, not two settings of one. Never compare a score across them or quote
an aggregate over them.

## The specs that govern this folder

`inv_single/` was produced by the **auto-mdp-solver** skill from
`inv_single_schema.json`. It is not a hand-written project, and it must not
drift into one.

**Before changing anything here, read the skill docs — they are authoritative
over this file, over the code, and over local habit:**

| doc | governs |
|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG/seed tree, script + eval conventions (§1–§14) |
| `CONTRACTS.md` + the step skills (`mdp-formalize` … `mdp-package`) | the pipeline ops, their entry gates (`python -m mdp_stage`), and what each op must emit |
| `ESCALATION_LOG_GUIDE.md` | the campaign log (§MAP, §FRAME-CHANGELOG, §IR-CHANGELOG, §CONFIG-REGISTRY, §LEDGER) and the §10 playbook digest |

All of these live **in the `mdp-solver` skill's own directory** (the step
skills are its siblings) — wherever the plugin is installed, they sit beside
its `SKILL.md`. Read them from there.
**Do not search the filesystem for them by name**: a development checkout of
the solver may also be on disk, and reading that instead silently swaps
unreleased content for the version this folder was built against, which is
exactly what the two provenance lines below exist to pin down.

Solver provenance — **two versions, two meanings** (keep both; they answer
different questions):

- **Built and gated at: v0.10.9** (`4c82d69`) — full suite re-run
  **2026-09-14** on that pin: validator OK, conformance **31/31**, laws
  **9/9**, differential **MATCH 13/13**, pytest **61**. Fingerprints
  `mdp 84a22814f666` / `model 17606699e437` / `structural 40f9149f9a0e` — the
  first two moved at F6 (2026-09-07), the theory hash has never moved. This
  line never moves without a full re-run.
  The folder's *runs* span pins: #E1–#E18 were produced at ≤ v0.10.2, #E19–#E24
  at v0.10.4; no run has yet been produced at v0.10.7 or later. The stamp above is the
  gate battery, not a claim that every PPO run was re-executed.
- **Conformance maintained through: v0.10.9** — the two are equal right now,
  which is the healthy state. When they diverge, the second names a version
  carried forward without re-running.

Upgrading: read the release notes, move the harness to the new tag, re-run
every gate below, then log the move. **The pin drifted three times during this
campaign** — a shared harness moves without any file in the case changing, and
each occurrence misattributed a block of escalations to the wrong version. All
three, and what each cost, are in `ESCALATION.md` §FRAME-CHANGELOG.

v0.10.2 itself is this campaign's own upstream fix (issue #75, `d793b88`):
`grids.base_instance` was checked after selection-pruning, so a grid based on
an instance selecting a non-default candidate — `slt`, here — could never
validate. It now resolves under the selection its base instance names, and the
loader records what it dropped in `pruned_instances` (for this IR:
`discrete_lost_sales`, `poisson_lost_sales`, `slt`). This is what makes the
matched-mean lead-time-variance grid expressible; before it, the only route was
explicit instances plus a mixture, which costs an `mdp` fingerprint move.

`README.md` carries the case: what the problem is, what is in the folder, what
the arms scored, and the commands that reproduce them.

## Hard rules

Each fires at a moment and names where the answer lives. None of them states
the answer — that is what makes them safe to keep current.

- **Before inventing a mechanism, check the spec.** The pipeline usually has
  one already, and its version composes with the gates.
- **Structural changes go through the IR first**, then the code, then the
  gates — never code-first. `inv_single_schema.json` is the source of truth;
  `inv_single_scenarios.py` and friends materialize it.
  Observation/architecture levers are NOT IR changes — they live in the gym /
  train script with their own executable gate; only the *problem* goes
  through the schema.
- **Re-run the gates after any IR change and after any solver bump** — the
  IR depends on host behavior, not just on this folder. A moved fingerprint
  owes a §IR-CHANGELOG entry (guide §5, tripwire 2).
- **Before quoting a number, say which artifact produced it.** Selection and
  terminal networks are different objects (`{scenario}_ppo.zip` vs
  `_ppo_final.zip`, spec §8.4), and the protocol every number is quoted at —
  including which eval mode is the record here — is stated once in
  `README.md`'s results section.
- **Editing `_L1_DERIVED` is a re-derivation, not a default edit** (spec
  §8.4). It re-bases every Δ(L2−L1) above it, so it takes a logged basis and
  a ledger entry — see spec §8.6 for when re-deriving is the right move.
- **Before pointing `mdp_gates --reference` at the DP, check that scenario's
  DP role.** It is `exact` only where its basis holds; `README.md`'s DP-notes
  table says where, and the claim itself is gated by
  `inv_single_test.py::test_the_shipped_dp_is_exact_where_the_second_implementation_says_it_is`.
- **An eval must use the observation mode its arm trained with.** Pass `-o`
  explicitly when in doubt; the modes and their widths are in `README.md`.
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in `ESCALATION.md`, and, if the spec should change, as an
  upstream issue filed via the mdp-propose skill. The issue is the record and
  the log carries its URL; the draft behind it is scratch (see File hygiene).

## Gate commands

Run from `inv_single/` unless noted. These are the checks; the commands that
*run* the code — train, select, eval, benchmark, probe, plot — are in
`README.md`'s technical appendix. Always pin threads for training — torch
oversubscribes.

```bash
python -m mdp_ir inv_single_schema.json
python -m mdp_conformance .
python -m mdp_ir.laws inv_single_schema.json
python -m mdp_ir.differential inv_single_schema.json --episodes 40 --all-instances
pytest inv_single_test.py
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `inv_single/` only if it is (a) spec-§1 layout, (b) an
implementation a finding cites and someone must re-run to reproduce it
(probes, extra gates), or (c) a campaign document (`README`, `ESCALATION`,
`INTERPRET`, `PLAYBOOK`). **Everything else goes to
`scratch/` (gitignored): launchers, monitors, one-off checks, throwaway
analysis. All output goes to `results/` (gitignored).**

**Round plans and upstream-proposal drafts are never tracked.** A `*_PLAN.md`
lives in `scratch/` while it
is being drafted AND while it is being executed; findings go into
`ESCALATION.md` and `README.md` **as they land**, and the plan is deleted once
written up. If a plan is the only place a result exists, that is a bug in
`ESCALATION.md`. **The probe a plan drives is the opposite — it stays,
permanently**, and is committed in the same commit as the escalation entry that
cites it.

An `UPSTREAM_PROPOSAL_*.md` is the same shape and for the same reason: it is
drafted in `scratch/`, and once the mdp-propose skill files the issue, the
issue is the proposal — the draft goes, and `ESCALATION.md` carries the issue
URL and, later, the maintainer's disposition. A proposal is pinned to a spec
version and goes stale; a tracked copy goes stale *silently*. The three drafts
this campaign filed were removed on 2026-09-07 for exactly that reason.

Check with `git status --short inv_single/`: untracked files should be rare
and deliberate.

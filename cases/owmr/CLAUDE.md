# CLAUDE.md — `owmr/` (one-warehouse, N-retailer periodic-review distribution)

Domain-local operating brief for an agent about to change this folder.
**This file points up: at the specs that govern the folder, and the version
of them its numbers were produced under.** Everything the campaign produced —
its documents, its code, its numbers — is inventoried in `README.md` and is
linked from here, never copied.

One warehouse orders from an ample supplier and each period allocates its
on-hand stock to N retailers under i.i.d. gamma demand with backorders;
minimize the undiscounted holding + backorder cost over a finite horizon. The
optimum is unknown and the reference is a bracket (a relaxed bound, its
feasible heuristic). **Every cell in `SCENARIOS` is a SEPARATE LEADERBOARD**,
scored against its own bracket: never compare a score across cells.

## The specs that govern this folder

`owmr/` was produced by the **auto-mdp-solver** skill from
`owmr_schema.json`. It is not a hand-written project, and it must not
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
different questions, and a case that is later contributed upstream needs the
distinction — see `cases/README.md`, "After a case is merged"):

- **Built and gated at: auto-mdp-solver 0.11.0** (PyPI harness) under plugin
  **0.11.3** (`c52e4e0`), 2026-09-22 — the checkout every number in this
  folder was produced under. This line never moves. Moving it would claim the
  results were re-measured.
- **Conformance maintained through: 0.11.3** — how far declarations, drawing
  conventions and gate compatibility have been carried forward. Updated when
  the case is brought to a newer spec *without* re-running anything; log each
  such move in `ESCALATION.md` §FRAME-CHANGELOG.
- **Solved with: Claude Code 2.1.278–2.1.283, claude-fable-5-1** — the agent that drove
  the campaign (the host version moved during it; the model did not). A case built by a different agent is a
  different data point for the auto-solving objective; a folder without this
  line was built with Claude Code before v0.11.0.

While the campaign is live the first two are the same tag. They diverge only once
someone conforms the case to a spec that landed after its results did.

The repo-root `CLAUDE.md` carries the cross-domain rules. `README.md` carries
the case: what the problem is, what is in the folder, what the arms scored,
and the commands that reproduce them.

## Hard rules

Each fires at a moment and names where the answer lives. None of them states
the answer — that is what makes them safe to keep current.

- **Before inventing a mechanism, check the spec.** The pipeline usually has
  one already, and its version composes with the gates.
- **Structural changes go through the IR first**, then the code, then the
  gates — never code-first. `owmr_schema.json` is the source of truth;
  `owmr_scenarios.py` and friends materialize it.
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
- **Before crowning an arm for the discover claim, check its action mode.**
  `target_frac` decodes through the echelon order-up-to form — the structure
  the claim is about; which modes are structure-free is stated in
  `owmr_gym.py`'s class docstring and the IR's `action_modes` descriptions.
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in `ESCALATION.md`, and, if the spec should change, as an
  upstream issue filed via the mdp-propose skill. The issue is the record and
  the log carries its URL; the draft behind it is scratch (see File hygiene).

## Gate commands

Run from `owmr/` unless noted. These are the checks; the commands that
*run* the code — train, eval, benchmark, probe, plot — are in `README.md`'s
technical appendix. Always pin threads for training — torch oversubscribes.

```bash
# conformance / laws / differential run from the PARENT of owmr/
python -m mdp_ir owmr_schema.json
python -m mdp_conformance owmr
python -m mdp_ir.laws owmr
python -m mdp_ir.differential owmr/owmr_schema.json --episodes 40 --all-instances
pytest owmr_test.py
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `owmr/` only if it is (a) spec-§1 layout, (b) an
implementation a finding cites and someone must re-run to reproduce it
(probes, extra gates), or (c) a campaign document (`README`, `ESCALATION`,
`INTERPRET`, `PLAYBOOK`). **Everything else goes to `scratch/` (gitignored):
launchers, monitors, one-off checks, throwaway analysis. All output goes to
`results/` (gitignored).**

**Round plans and upstream-proposal drafts are never tracked.** A `*_PLAN.md`
lives in `scratch/` while it is being drafted AND while it is being executed;
findings go into `ESCALATION.md` and `README.md` **as they land**, and the
plan is deleted
once written up. If a plan is the only place a result exists, that is a bug
in `ESCALATION.md`. **The probe a plan drives is the opposite — it stays,
permanently**: an escalation entry cites numbers that only exist if the code
behind them can be re-run. A probe is written in `owmr/` from the start
(it imports its siblings) and **committed in the same commit as the
escalation entry that cites it** — `git log ESCALATION.md` then shows each
finding beside the code that produced it. Design rationale that must outlive
the round has two tracked homes, neither of them the plan: the escalation
entry and the probe's module docstring.

An `UPSTREAM_PROPOSAL_*.md` is the same shape and for the same reason: it is
drafted in `scratch/`, and once the mdp-propose skill files the issue, the
issue is the proposal — the draft goes, and `ESCALATION.md` carries the issue
URL and, later, the maintainer's disposition. A proposal is pinned to a spec
version and goes stale; a tracked copy goes stale *silently*, which is how a
folder ends up asserting a claim upstream has already rejected.

Check with `git status --short owmr/`: untracked files should be rare
and deliberate.

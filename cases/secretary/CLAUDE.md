# CLAUDE.md — `secretary/` (classical secretary problem)

Domain-local operating brief for an agent about to change this folder.
**This file points up: at the specs that govern the folder, and the version
of them its numbers were produced under.** Everything the campaign produced —
its documents, its code, its numbers — is inventoried in `README.md` and is
linked from here, never copied.

One policy observes position and relative rank, then irrevocably rejects or
accepts one of 100 randomly ordered candidates to maximize the probability of
selecting the unique overall best. There is one `standard` leaderboard; no
independent branches or incommensurable scenario modes exist.

## The specs that govern this folder

`secretary/` was produced by the **auto-mdp-solver** skill from
`secretary_schema.json`. It is not a hand-written project, and it must not
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

- **Built and gated at: [auto-mdp-solver v0.11.4](https://github.com/tong-wang/auto-mdp-solver/releases/tag/v0.11.4), commit [`d0e1a50a5fd7bcd0996f82da2db90006ca75b4cc`](https://github.com/tong-wang/auto-mdp-solver/commit/d0e1a50a5fd7bcd0996f82da2db90006ca75b4cc)** — the checkout every number in this folder was produced under. This line never moves. Moving it would claim the results were re-measured.
- **Conformance maintained through: [auto-mdp-solver v0.11.4](https://github.com/tong-wang/auto-mdp-solver/releases/tag/v0.11.4)** — how far declarations, drawing conventions and gate compatibility have been carried forward. Updated when the case is brought to a newer spec *without* re-running anything; log each such move in `ESCALATION.md` §FRAME-CHANGELOG.
- **Solved with: Codex API workspace agent, GPT-5** — the agent that drove this campaign.

While the campaign is live the first two are the same tag. They diverge only once
someone conforms the case to a spec that landed after its results did.

The repo-root `CLAUDE.md` carries the cross-domain rules. `README.md` carries
the case: what the problem is, what is in the folder, what the arms scored,
and the commands that reproduce them.

## Hard rules

Each fires at a moment and names where the answer lives. None of them states
the answer — that is what makes them safe to keep current.

- **Before inventing a mechanism, check the spec.** The pipeline usually has one already, and its version composes with the gates.
- **Structural changes go through the IR first**, then the code, then the gates — never code-first. `secretary_schema.json` is the source of truth; `secretary_scenarios.py` and friends materialize it. Observation/architecture levers are NOT IR changes — they live in the gym / train script with their own executable gate; only the *problem* goes through the schema.
- **Re-run the gates after any IR change and after any solver bump** — the IR depends on host behavior, not just on this folder. A moved fingerprint owes a §IR-CHANGELOG entry (guide §5, tripwire 2).
- **Before quoting a number, say which artifact produced it.** Selection and terminal networks are different objects (`standard_ppo.zip` vs `_ppo_final.zip`, spec §8.4), and the protocol every number is quoted at — including which eval mode is the record here — is stated once in `README.md`'s results section.
- **Editing `_L1_DERIVED` is a re-derivation, not a default edit** (spec §8.4). It re-bases every Δ(L2−L1) above it, so it takes a logged basis and a ledger entry — see spec §8.6 for when re-deriving is the right move.
- **A deliberate deviation from the spec is recorded, not silent** — in the code comment, in `ESCALATION.md`, and, if the spec should change, as an upstream issue filed via the mdp-propose skill. The issue is the record and the log carries its URL; the draft behind it is scratch (see File hygiene).

## Gate commands

Run from `secretary/` unless noted. These are the checks; the commands that
*run* the code — train, eval, benchmark, probe, plot — are in `README.md`'s
technical appendix. Always pin threads for training — torch oversubscribes.

```bash
# conformance / laws / differential run from the PARENT of secretary/
python -m mdp_ir secretary/secretary_schema.json
python -m mdp_conformance secretary
python -m mdp_ir.laws secretary
python -m mdp_ir.differential secretary/secretary_schema.json --episodes 40 --all-instances
# Exercise the complete horizon; the default random decision policy stops early.
python -m mdp_ir.differential secretary/secretary_schema.json --decision accept=0 --episodes 10 --all-instances

# from secretary/
pytest secretary_test.py
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `secretary/` only if it is (a) spec-§1 layout, (b) an
implementation a finding cites and someone must re-run to reproduce it
(probes, extra gates), or (c) a campaign document (`README`, `ESCALATION`,
`INTERPRET`, `PLAYBOOK`). **Everything else goes to `scratch/` (gitignored):
launchers, monitors, one-off checks, throwaway analysis. All output goes to
`results/` (gitignored).**

**Round plans and upstream-proposal drafts are never tracked.** A `*_PLAN.md`
lives in `scratch/` while it is being drafted AND while it is being executed;
findings go into `ESCALATION.md` and `README.md` **as they land**, and the plan
is deleted once written up. If a plan is the only place a result exists, that
is a bug in `ESCALATION.md`. **The probe a plan drives is the opposite — it
stays, permanently**: an escalation entry cites numbers that only exist if the
code behind them can be re-run. A probe is written in `secretary/` from the
start and committed in the same commit as the escalation entry that cites it.

An `UPSTREAM_PROPOSAL_*.md` is drafted in `scratch/`; once mdp-propose files
the issue, the issue is the proposal, the draft goes, and `ESCALATION.md`
carries the issue URL and later disposition.

Check with `git status --short secretary/`: untracked files should be rare
and deliberate.

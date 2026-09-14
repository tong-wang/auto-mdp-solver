# CLAUDE.md — `fnv/` (fresh-newsvendor sequential ordering under MMFE)

Domain-local operating brief for an agent about to change this folder.
**This file points up: at the specs that govern the folder, and the version
of them its numbers were produced under.** Everything the campaign produced —
its documents, its code, its numbers — is inventoried in `README.md` and is
linked from here, never copied.

Place `N` sequential orders at rising unit cost while a martingale demand
signal is progressively revealed (Heath & Jackson 1994 MMFE); demand realizes
once at the horizon and is sold against cumulative inventory. State is
`(period, inventory, information)`; the decision is one continuous order
quantity; the objective is expected profit.

**`FNV-aMMFE` (additive, `D = mu + I`) and `FNV-mMMFE` (multiplicative,
`D = exp(mu + I)`) are SEPARATE LEADERBOARDS** — different demand scales,
different bars. Their scores must never be compared. Both are grids in
`GRIDS`, not entries in `SCENARIOS`; within a grid each cell is its own row.

## The specs that govern this folder

`fnv/` was produced by the **auto-mdp-solver** skill from `fnv_schema.json`.
It is not a hand-written project, and it must not drift into one.

**Before changing anything here, read the skill docs — they are authoritative
over this file, over the code, and over local habit:**

| doc | governs |
|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG/seed tree, script + eval conventions (§1–§14) |
| `CONTRACTS.md` + the step skills (`mdp-formalize` … `mdp-package`) | the pipeline ops, their entry gates (`python -m mdp_stage`), and what each op must emit |
| `ESCALATION_LOG_GUIDE.md` | the campaign log (§MAP, §FRAME-CHANGELOG, §IR-CHANGELOG, §CONFIG-REGISTRY, §LEDGER) and the §10 playbook digest |

All of these live **in the `mdp-solver` skill's own directory** (the step
skills are its siblings) — in this repo, `plugin/skills/mdp-solver/`. This
case lives **inside** the auto-mdp-solver repo, so the harness it builds
against is the repo's own `harness/` at whatever revision is checked out;
the two provenance lines below say which revisions its numbers and its
declarations belong to. Never trust a version number written in a document
over `git describe --tags`.

Solver provenance — **two versions, two meanings** (keep both; they answer
different questions — see `cases/README.md`, "After a case is merged"):

- **Built and gated at: v0.8.0** — the checkout every number in this folder
  was produced under (campaign 2026-08-13/14). This line never moves. Moving
  it would claim the results were re-measured.
- **Conformance maintained through: v0.10.9** — re-gated 2026-09-14 by the
  author at the v0.10.9 pin: IR OK, `mdp` fingerprint `d415b34e8c33` and structural
  `36f5c3aaf7a6` unmoved; conformance 24/31, no FAIL; laws 7/9; differential
  MATCH on both instances; 19 domain tests. How far declarations, drawing
  conventions and gate compatibility have been carried forward (declarations
  to v0.9.5, the log to v0.9.26/v0.9.27, the two documents to v0.10.9's §1.3
  shapes). Every move is logged in `ESCALATION.md` §FRAME-CHANGELOG.

Upgrading: read the tag's notes, re-run every gate below at the new
revision, then log the move. The author's working copy lives downstream
(the `fnv/` folder of a research repo) and re-ports here; a release that
invalidates a statement in this folder names it in its commit body rather
than editing it (repo `CLAUDE.md`, Rules).

The repo-root `CLAUDE.md` carries the cross-domain rules. `README.md` carries
the case: what the problem is, what is in the folder, what the arms scored,
and the commands that reproduce them.

## Hard rules

Each fires at a moment and names where the answer lives. None of them states
the answer — that is what makes them safe to keep current.

- **Before inventing a mechanism, check the spec.** The pipeline usually has
  one already, and its version composes with the gates.
- **Structural changes go through the IR first**, then the code, then the
  gates — never code-first. `fnv_schema.json` is the source of truth;
  `fnv_scenarios.py` and friends materialize it.
  Observation/architecture levers are NOT IR changes — they live in the gym /
  train script with their own executable gate; only the *problem* goes
  through the schema.
- **Re-run the gates after any IR change and after any solver bump** — the
  IR depends on host behavior, not just on this folder. A moved fingerprint
  owes a §IR-CHANGELOG entry (guide §5, tripwire 2).
- **Before quoting a number, say which artifact produced it.** Selection and
  terminal networks are different objects (`{scenario}_ppo.zip` vs a
  `checkpoints/*.zip` chosen by the screen, spec §8.4), and the protocol
  every number is quoted at — including that the record eval is the
  deterministic argmax here — is stated once in `README.md`'s results
  section.
- **Reading the raw IR JSON by path** → `ESCALATION.md` §IR-CHANGELOG F2: the
  `mdp` block is in the grouped layout, and a flat-shape read breaks
  collection. Go through `load_ir`.
- **Adding a quantity to `mdp.model`** → spec §5.0's tie-breaker and F2,
  which records the three quantities this domain classes as design.
- **Changing either offset solver** (`fnv_benchmark_dp.py`,
  `fnv_benchmark_prop2.py`) → their independence is the spec §1.2 gate;
  re-run `--cross-check` and `--self-test`, and read the DP module's
  docstring on extrapolation before touching its grid. The invariants
  `--check-invariants` asserts are the paper's Corollary 1 and
  mode-independence.
- **Fitting or quoting the policy's structure** → `INTERPRET.md` "Method"
  and `PLAYBOOK.md` LV5–LV7: acting periods only, one fit per cell, in logs
  on the multiplicative branch, the acting count beside every slope.
- **Quoting a grid mean** → `PLAYBOOK.md` LV8: seeds are the replication
  unit, and a median with a sign test goes beside it.
- **Screening a generalist** → over its whole grid, never one cell (`#E1`);
  `fnv_select.py` does this by construction.
- **Launching a run** → cite a §CONFIG-REGISTRY address (`sc{0,1}/g0/a0/h0b`
  is the crown). No `fnv_configs.py` exists yet; the registry's residue says
  it is owed the moment an arm is launched here again.
- **Editing `_L1_DERIVED` is a re-derivation, not a default edit** (spec
  §8.4). It re-bases every Δ(L2−L1) above it, so it takes a logged basis and
  a ledger entry — see spec §8.6 for when re-deriving is the right move.
  Here the derivation is still a comment table in `fnv_ppo_train.py`
  (conformance WARNs `scripts.l1_derived`); the same rule applies to it.
- **Writing to the log** → §FRAME-CHANGELOG, §IR-CHANGELOG and §LEDGER are
  append-only, new entries at the end; §MAP and §CONFIG-REGISTRY are living.
  The rung spelled `L1′` (two apostrophes in the wild) and `L1b` is one rung;
  grep for all three.
- **Changing a file here** → the author's downstream copy is the working
  tree; a divergence is logged in §FRAME-CHANGELOG with its direction.
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in `ESCALATION.md`, and, if the spec should change, as an
  upstream issue filed via the mdp-propose skill. The issue is the record and
  the log carries its URL; the draft behind it is scratch (see File hygiene).
- **Never** use `param`, `params`, or `param_*`; conformance fails.

## Gate commands

Run from `cases/fnv/` unless noted. These are
the checks; the commands that *run* the code — train, eval, benchmark, probe,
plot — are in `README.md`'s technical appendix. Always pin threads for
training — torch oversubscribes.

```bash
# conformance / laws / differential run from cases/, the parent of fnv/
python -m mdp_ir fnv/fnv_schema.json
python -m mdp_conformance fnv
python -m mdp_ir.laws fnv
python -m mdp_ir.differential fnv/fnv_schema.json --episodes 40 --all-instances
cd fnv && pytest fnv_test.py
```

The spec §9.9 **role gate** is a check too, so it lives here. `--ir` is what
activates the role reasoning: a `feasible` arm beating an `exact` one fails
as a bug report. Its z-test against the baseline is *unpaired* (pooled
per-cell SE), so it reads about 1 SE where the leaderboard's paired per-cell
Δ reads 14; the paired number is the record (`README.md`, Protocol).

```bash
python -m mdp_gates --ir fnv_schema.json --n-seeds 2048 --metric profit_mean --sense maximize \
  --candidate results/FNV-aMMFE/PPO_<run>/ppo_eval_FNV-aMMFE_<ckpt>.tsv \
  --baseline  results/FNV-aMMFE/benchmark/benchmark_myopic_eval_FNV-aMMFE_n2048.tsv \
  --reference results/FNV-aMMFE/benchmark/benchmark_prop2_eval_FNV-aMMFE_n2048.tsv
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `fnv/` only if it is (a) spec-§1 layout, (b) an
implementation a finding cites and someone must re-run to reproduce it
(probes, extra gates), or (c) a campaign document (`README`, `ESCALATION`,
`INTERPRET`, `PLAYBOOK`). **Everything else goes to `scratch/` (gitignored):
launchers, monitors, one-off checks, throwaway analysis. All output goes to
`results/` (gitignored).**

**Round plans and upstream-proposal drafts are never tracked.** A `*_PLAN.md`
lives in `scratch/` while it is being drafted AND while it is being executed;
findings go into `ESCALATION.md` and `README.md` **as they land**, and the
plan is deleted once written up. If a plan is the only place a result exists,
that is a bug in `ESCALATION.md`. **The probe a plan drives is the opposite —
it stays, permanently**: an escalation entry cites numbers that only exist if
the code behind them can be re-run. A probe is written in `fnv/` from the
start (it imports its siblings) and **committed in the same commit as the
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

Check with `git status --short fnv/`: untracked files should be rare
and deliberate.

# CLAUDE.md — `mab/` (standard stochastic multi-armed bandit)

Domain-local operating brief for an agent about to change this folder.
**This file points up: at the specs that govern the folder, and the version
of them its numbers were produced under.** Everything the campaign produced —
its documents, its code, its numbers — is inventoried in `README.md` and is
linked from here, never copied.

K arms, T rounds. Nature draws each arm's mean payout fresh per episode (a
world latent the agent never sees); the agent pulls one arm per round,
observes only that arm's payout, and maximizes the undiscounted total over a
finite horizon. **Bernoulli and Gaussian are SEPARATE LEADERBOARDS** — payout
scales differ by ~1000×, so their scores may never be compared — and likewise
**every `gauss_K{K}_T{T}` instance is its own leaderboard**, since
`reward_mean` is not comparable across horizons at all.

## The specs that govern this folder

`mab/` was produced by the **auto-mdp-solver** skill from `mab_schema.json`.
It is not a hand-written project, and it must not drift into one.

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

- **Built and gated at: v0.7.0** (`2e50c60`) — the checkout every number in
  this folder was produced under. This line never moves. Moving it would
  claim the results were re-measured.
- **Conformance maintained through: v0.10.9** — how far declarations, drawing
  conventions and gate compatibility have been carried forward **without
  re-running anything**: the `benchmarks` block (including the column-sourced
  oracle), the `research_questions` stances, the spec-§8.4 provenance emit, the
  `mdp.model` theory layer and grouped layout (**F6**), and, at v0.10.0, the
  split-skill pipeline state, the §MAP taxonomy retype, and the README/CLAUDE
  rebuild to spec §1.3. **This round** adds the two §1.3 blocks `README.md`
  was missing: the **lead** (v0.10.9) and, below it, the **TL;DR** (v0.10.6),
  the latter written from the finished boards after the seven-question
  checklist pass. Both are document conventions, reaching no code and no
  declaration. Logged in `ESCALATION.md`
  §FRAME-CHANGELOG; **no §IR-CHANGELOG entry is due, because the `mdp` block
  did not move** — the fingerprint is still F6's `5bb25e684405`. **No verdict,
  mark or number moved.**

The two lines differing is the intended state, not drift. Before relying on
either, read the tag off the solver checkout that is actually installed — the
installed skill's own directory is the authority, never this file.

The repo-root `CLAUDE.md` carries the cross-domain rules. `README.md` carries
the case: what the problem is, what is in the folder, what the arms scored,
and the commands that reproduce them.

## Hard rules

Each fires at a moment and names where the answer lives. None of them states
the answer — that is what makes them safe to keep current.

- **Before inventing a mechanism, check the spec.** The pipeline usually has
  one already, and its version composes with the gates.
- **Structural changes go through the IR first**, then the code, then the
  gates — never code-first. `mab_schema.json` is the source of truth;
  `mab_scenarios.py` and friends materialize it.
  Observation/architecture levers are NOT IR changes — they live in the gym /
  train script with their own executable gate; only the *problem* goes
  through the schema.
- **Re-run the gates after any IR change and after any solver bump** — the
  IR depends on host behavior, not just on this folder. A moved fingerprint
  owes a §IR-CHANGELOG entry (guide §5, tripwire 2) and a re-confirmed
  `mab.signoff.json`, which the always-on freeze check compares against.
- **Before reading the raw `["mdp"]` JSON, flatten it.** The block ships in
  the grouped layout (`model` / `design` / `rendering`), so `["mdp"]["scenario"]`
  does not exist — and `load_ir` and `ungroup_mdp` are **not** interchangeable
  here. See `ESCALATION.md` §IR-CHANGELOG **F6** and `mab_test.py::_raw_flat`,
  which is the only raw reader and carries the reasoning.
- **Before quoting a number, say which artifact produced it.** Selection and
  terminal networks are different objects (`{scenario}_ppo.zip` vs
  `_ppo_final.zip`, spec §8.4), and the protocol every number is quoted at —
  including which eval mode is the record here — is stated once in
  `README.md`'s Results section.
- **Before re-running training, read `README.md`'s note on the selection
  path.** This folder's `SelectionEvalCallback` predates v0.7.0's spec §8.6/§9.7
  post-hoc screen; the code is kept as it ran, so a new run should not simply
  inherit it. `mab_selection_probe.py` is the post-hoc screen run by hand.
- **Editing `_L1_DERIVED` is a re-derivation, not a default edit** (spec
  spec §8.4). It re-bases every Δ(L2−L1) above it, so it takes a logged basis and
  a ledger entry — see spec §8.6 for when re-deriving is the right move.
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in `ESCALATION.md`, and, if the spec should change, as an
  upstream issue filed via the mdp-propose skill. The issue is the record and
  the log carries its URL; the draft behind it is scratch (see File hygiene).

## Gate commands

Run from `plugin/skills/mdp-solver/examples/` (the parent of this folder).
These are the checks; the commands that *run* the code — train, eval,
benchmark, probe, plot — are in `README.md`'s technical appendix. Always pin
threads for training — torch oversubscribes.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# pipeline entry gates — where the folder stands, and per-op readiness
python -m mdp_stage           mab
python -m mdp_stage           mab --for solve   # + conformance, laws, differential

# validation / conformance / laws / differential
python -m mdp_ir              mab/mab_schema.json
python -m mdp_conformance     mab
python -m mdp_ir.laws         mab
python -m mdp_ir.differential mab/mab_schema.json --all-instances --episodes 40
pytest mab
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `mab/` only if it is (a) spec-§1 layout, (b) an
implementation a finding cites and someone must re-run to reproduce it
(probes, extra gates), or (c) a campaign document (`README`, `ESCALATION`,
`INTERPRET`, `PLAYBOOK`). The two pipeline-state files
(`mab.signoff.json`, `mab.runplan.json`) are committed beside the schema and
are written by the pipeline, never by hand-editing to make a gate pass.
**Everything else goes to `scratch/` (gitignored): launchers, monitors,
one-off checks, throwaway analysis. All output goes to `results/`, which is
gitignored.**

**Round plans and upstream-proposal drafts are never tracked.** A `*_PLAN.md`
lives in `scratch/` while it is being drafted AND while it is being executed;
findings go into `ESCALATION.md` and `README.md` **as they land**, and the
plan is deleted once written up. If a plan is the only place a result exists,
that is a bug in `ESCALATION.md`. **The probe a plan drives is the opposite —
it stays, permanently**: an escalation entry cites numbers that only exist if
the code behind them can be re-run. A probe is written in `mab/` from the
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
folder ends up asserting a claim upstream has already rejected. All ten this
campaign filed are dispositioned; `ESCALATION.md` §UPSTREAM records where each
went.

Check with `git status --short mab/`: untracked files should be rare and
deliberate.

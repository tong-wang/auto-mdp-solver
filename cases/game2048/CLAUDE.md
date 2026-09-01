# CLAUDE.md — `game2048` (the 2048 sliding-tile game)

Domain-local operating brief for an agent about to change this folder.
**This file points up: at the specs that govern the folder, and the version
of them its numbers were produced under.** Everything the campaign produced —
its documents, its code, its numbers — is inventoried in `README.md` and is
linked from here, never copied.

An `n x n` board; slide + merge along one axis, then a tile spawns in a
uniformly-drawn empty cell. Objective = the game's own **merge score**.
Instances are (board size × spawn regime), and **`3x3_20` and `4x4_20` are
SEPARATE LEADERBOARDS whose scores may never be compared** — different
frames, different bars, different problem scales.

## The specs that govern this folder

`game2048/` was produced by the **auto-mdp-solver** skill from
`game2048_schema.json`. It is not a hand-written project, and it must not
drift into one.

**Before changing anything here, read the skill docs — they are authoritative
over this file, over the code, and over local habit:**

| doc | governs |
|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG/seed tree, script + eval conventions (§1–§14) |
| `SKILL.md` | the pipeline stages, their gates, and what each stage must emit |
| `ESCALATION_LOG_GUIDE.md` | the campaign log (§MAP, §FRAME-CHANGELOG, §IR-CHANGELOG, §LEDGER) and the §10 playbook digest |

They ship with the plugin, at `plugin/skills/mdp-solver/` in this repository.

Solver provenance — **two versions, two meanings** (keep both; they answer
different questions):

- **Built and gated at: NOT RECORDED.** The campaign ran from 2026-07-29
  across many solver versions and nothing in the folder stamps one, so no
  number here may be claimed as re-measured under a newer tag. Recovering it
  means reading the release history against the ledger dates; until then this
  line stays honest rather than plausible.
- **Conformance maintained through: v0.9.35** (`d108e24`) — the campaign docs
  were brought to spec §1.3 on 2026-09-01. **No runs were re-executed.**

The repository root's `CLAUDE.md` carries the cross-case rules. `README.md`
carries the case: what the problem is, what is in the folder, what the arms
scored, and the commands that reproduce them.

**One thing about this folder that is not true of every case:
`ESCALATION.md` here is a scoped subset** — 32 of the campaign's 85 numbered
entries. Its own header states what is held back, marks every line it
redacted, and warns that some `#E` citations therefore do not resolve. Treat a
dangling citation as expected, not as damage.

## Hard rules

Each fires at a moment and names where the answer lives. None of them states
the answer — that is what makes them safe to keep current.

- **Before inventing a mechanism, check the spec.** The pipeline usually has
  one already, and its version composes with the gates.
- **Structural changes go through the IR first**, then the code, then the
  gates — never code-first. `game2048_schema.json` is the source of truth;
  `game2048_scenarios.py` and friends materialize it.
  Observation/architecture levers are NOT IR changes — they live in the gym /
  train script with their own executable gate; only the *problem* goes
  through the schema.
- **Re-run the gates after any IR change and after any solver bump** — the
  IR depends on host behavior, not just on this folder. A moved
  `structural_fingerprint` owes a §IR-CHANGELOG entry (guide §5, tripwire 2).
- **Before quoting a number, say which artifact produced it.** Selection and
  terminal networks are different objects (`best_model.zip` vs
  `{scenario}_{algo}.zip`, spec §8.4), and the protocol every number is
  quoted at — including which eval mode is the record here — is stated once
  in `README.md`'s results section.
- **Before calling an arm the best, check what the board says is absent.**
  Both leaderboards here ship the best arm *of this case*, and the 3×3 board
  is a statistical tie with arms that are not published in it. `README.md`
  states this at each board; do not quote either crown as a campaign-wide
  best.
- **Re-deriving the `--level L1` config is a re-derivation, not a default
  edit** (spec §8.4). It re-bases every Δ measured above it, so it takes a
  logged basis and a ledger entry — see spec §8.6 for when that is the right
  move.
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in the log, and as an upstream proposal (via the mdp-propose
  skill) if the spec should change.

## Gate commands

Run from `cases/` (the parent of this folder). These are the checks; the
commands that *run* the code — train, eval, benchmark, probe — are in
`README.md`'s technical appendix. Always pin threads for training: torch
oversubscribes on a shared box.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m mdp_ir              game2048/game2048_schema.json
python -m mdp_conformance     game2048
python -m mdp_ir.laws         game2048
python -m mdp_ir.differential game2048/game2048_schema.json --all-instances --episodes 40
pytest game2048
```

Expected at v0.9.35: `mdp_ir` OK with `structural_fingerprint=b040fb828165`;
laws 7/9 (`path_independence` and `mixture_equivalence` both correct SKIPs —
the spawn support is state-conditioned, and no mixture is declared);
conformance 21/30 with **zero FAIL**; differential MATCH bit-exact on all 8
instances; pytest 39 passed.

The campaign's own executable gates (`game2048_*_gate.py`) are inventoried in
`README.md`; each is cited by the escalation entry that owes it, and what any
of them established is that entry's to state, not this file's.

## File hygiene — the folder is the deliverable, not the workbench

- Tracked here: the spec §1 layout, the gates a finding cites (someone must be
  able to re-run them), and the campaign documents (`README`, `ESCALATION`,
  `INTERPRET`, `PLAYBOOK`, the restatement). **All output goes to `results/`,
  which is gitignored** — every number in `README.md` regenerates from the
  commands in its appendix.
- **Round plans are never tracked.** Findings go into `ESCALATION.md` and
  `README.md` **as they land**. If a plan is the only place a result exists,
  that is a bug in `ESCALATION.md`.
- **The probe a plan drives is the opposite — it stays, permanently.** An
  escalation entry cites numbers that only exist if the code behind them can
  be re-run, so a probe or gate is committed **in the same commit as the
  escalation entry that cites it** — `git log ESCALATION.md` then shows each
  finding beside the code that produced it.
- Spec §14.3's committed figure contract is **owed**, not met: there is no
  `game2048_plot_policy.py` and no `figures/`. Plotting is inline in
  `game2048_interpret.py` and its output is gitignored. `README.md` lists this
  as owed rather than dropping it.

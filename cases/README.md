# cases/ — auto-solve test cases

Each subfolder is one end-to-end run of the mdp-solver skill on a new
problem: IR + restatement (Phase A), generated domain + benchmarks + trained
policy (Phase B), and a README with the leaderboard. The purpose of this
directory is to grow the solver's capability envelope — every case should
stress something the pipeline hasn't handled before, and a case that forces
an `mdp_ir/` extension is a good case.

Trained artifacts (`results/`) are gitignored; each case README's commands
must reproduce them.

## Contributing a case

External case contributions are welcome — they are exactly how this directory
is meant to grow. The easiest path is the **`mdp-contribute` skill** (ships
with the plugin): it assembles your case to the contract below, runs the gates,
and opens the PR via `gh` (fork-and-PR; nothing is sent without your approval).
It also supports a **re-skinned** contribution — an isomorphic rename that
shares the MDP structure while abstracting away your business context. A case
too sensitive even re-skinned is simply not contributed; there is no partial
path. Spec/schema extension proposals are a separate flow — the
**`mdp-propose` skill**, one `upstream-proposal` issue per proposal, filed
the moment a campaign hits the wall rather than at case close.

Contributing manually instead: add one folder `cases/<name>/` containing the
frozen IR (`<name>_schema.json`), the restatement, the spec-conformant domain
modules (`_uncertainty`/`_scenarios`/`_mdp`/`_gym` + `<name>_ir_adapter.py`),
benchmarks with their eval scripts, `_ppo_train`/`_ppo_eval`, `<name>_policy.py`,
the domain's `CLAUDE.md` operating brief (spec §1, from the skill's template),
and a README whose commands reproduce every leaderboard number (no `results/`).
Include your campaign record — `ESCALATION.md` and its case-close
`PLAYBOOK.md` (ESCALATION_LOG_GUIDE §10: digest entries — necessary context /
symptom / diagnosis / prescription / failed attempts, real names and numbers,
ledger cited not copied) — it is half the value. State the
provenance of the problem in the PR (paper cases: cite it; business cases:
confirm you may publish it — the confirmation covers the campaign record
too, since a promoted case ships whole). CI (`case-gates`) runs IR
validation, spec conformance, and the differential on every case dir the PR
touches; all three must be green. Negative cases — where no pipeline
deliverable beats the baselines — are accepted when stated plainly; they
stress the pipeline too (they just don't get promoted to the examples set).

## Promotion to the examples set

Promotion out of `cases/` into `plugin/skills/mdp-solver/examples/` is an
internal maintainer decision, separate from contribution — and it is also a
**publication decision**: the whole folder moves, campaign record included,
and everything under `examples/` ships with the plugin (which is the point —
the in-folder `PLAYBOOK.md` becomes reachable by the installed skill).
Criteria:

- **deliverable-competitive** — a pipeline deliverable beats the classical
  baselines at the case's protocol. The deliverable may be the trained
  policy *or* its §14 readback: a fitted rule that outperforms its own net
  is a Stage-5 success, not a negative case.
- **coverage** — the case covers a shape the manifest's coverage note lists
  as missing, or is the regression case for an `mdp_ir` change.
- **no declared debt** — coverage debt is paid before the freeze; frozen
  entries take no research edits afterward.
- **gates green** on the folder as it will ship, and the case README's
  commands reproduce every number.

Mechanics: move the whole folder — never copy; two folders with the same
`{name}_*` modules break pytest's `prepend` import mode — add the manifest
row, update its shape-coverage note, record the promotion in the table
below, and bump the plugin version. See `examples/MANIFEST.md` for the
scoped freeze (code/schema/tests frozen; campaign-record docs
maintainer-curated).

| promoted case | when | now at |
|---|---|---|
| `mab` — stochastic multi-armed bandit | 2026-08-13 | `plugin/skills/mdp-solver/examples/mab/` |

Demotion is the same move in reverse, and is not a failure verdict: an
example that no longer earns its place as a *few-shot exemplar* — too
specific a problem, or incomplete as a pipeline run — returns here, where it
still gates. `fnv` moved back on 2026-08-13: it ships no train/eval pair, so
it never exercised the solve leg, and its MMFE ordering problem is narrower
than the shapes an exemplar should teach. Note its solve half exists in the
originating research repo, so completing it is a contribution of the missing
scripts, not a fresh case.

## After a case is merged

Merging transfers **maintenance**, not authorship. The argument stays the
contributor's — their record, their verdicts, their numbers. What upstream
takes on is keeping the folder *gating* as the spec moves, and a merged case is
a regression fixture for the pipeline rather than a living project: the
contributor's own campaign continues in their repo against a pinned tag and
never needs to track this copy.

**When the spec moves under a merged case**, exactly one question decides what
happens: *can the change be made without invalidating a number?*

| | what upstream does | example |
|---|---|---|
| **yes** — declarations, drawing conventions, conformance edits | maintainer updates the case, with a changelog line in its own `ESCALATION.md` §FRAME-CHANGELOG | `mab` declaring its six `benchmarks` and two tier-2 stances (2026-08-16) |
| **no** — the fix needs the campaign re-run | **caveat it** in `examples/MANIFEST.md` or the case README; disclose, never quietly re-run someone's numbers | `mab`'s selection machinery predating v0.7.0 — every number was produced under the old design, so the code stays and the caveat says so |
| the case no longer earns its shipped place | demote (above); still gates from `cases/` | `fnv`, 2026-08-13 |

The line a maintainer does not cross: conformance edits touch code, schema,
declarations and add changelog lines. **They never rewrite a contributor's
reasoning, verdicts, or numbers.** Redrawing a campaign map to a new taxonomy
sits at the edge of that line — it changes form, not substance — and for an
externally contributed case it is worth asking the author first.

A contributor who disagrees with a conformance edit files an
`upstream-proposal` (the `mdp-propose` skill) rather than needing write access.

## Updating your own merged case

Contribution is one-shot for *ownership* and continuous for *improvement* —
returning with more work is expected, not an imposition. Which path depends on
where the case currently lives and whether numbers move:

1. **Still in `cases/`** — just open a PR against the folder, the same way the
   case arrived. Numbers may move; bring the record with them (`ESCALATION.md`
   entries, an updated `PLAYBOOK.md` if the digest changed) and make sure the
   README's commands still reproduce every number. CI re-runs `case-gates` on
   the dirs you touch. **No demotion, no re-contribution.**
2. **Promoted to `examples/`, and the change moves no number** — declarations,
   conformance to a new spec, a record correction. This lands as a maintainer
   edit under the scoped freeze; send it as a PR or an issue and it is applied
   in place.
3. **Promoted to `examples/`, and the change moves numbers or re-opens the
   campaign** — new arms, another escalation round, retuned results. **Demote
   first**: the folder moves back to `cases/`, your update lands there, and
   re-promotion is a fresh maintainer decision against the same criteria. This
   is not a demerit — it is what keeps "what shipped at tag `v0.X.Y`" a stable
   answer, which is the whole reason `examples/` is frozen. Say so in the PR
   and the demotion is part of accepting it.
4. **A different question on the same domain** — a new IR, a new campaign
   framing — is a **new case with a new name**, not an update. The old one
   keeps gating the shape it was admitted for, and two folders sharing
   `{name}_*` module names would break pytest's import mode anyway.

Cost asymmetry worth knowing before you choose: a `cases/` update ships in
neither published artifact, so it is cheap. An `examples/` update is a release
event — it bumps the plugin version and moves the tag every installed copy is
compared against.

**Every case records two versions, and they mean different things**: the tag it
was *built and gated against*, which is the provenance of its numbers and never
changes, and the tag its conformance has been *maintained through*, which the
maintainer updates. A case whose two lines differ has had its declarations
moved forward and its results left alone — which is exactly the intended state.

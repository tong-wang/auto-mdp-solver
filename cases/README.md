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

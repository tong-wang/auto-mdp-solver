# CLAUDE.md

This file guides a coding agent (Claude Code, Codex, or any host that reads
`CLAUDE.md`) working in **auto-mdp-solver** — the standalone MDP-solver
pipeline (public brand `auto-mdp-solver`; split out of a private research
workspace on 2026-07-21).

## What this repo is

**North star:** a generic solver for the vast family of Markov Decision
Processes (MDPs) — collapsing their notorious per-problem bespokeness by pairing
Claude's power to *formalize* any verbal problem into a standardized,
differentially-verified representation (its "digital twin") with RL's (PPO)
generality to *solve* it competitively — and to *interpret* the learned policy
back into structural insight, the way classical DP analysis does.

The automatic solver that turns a verbal dynamic-decision problem into a
trained, deployable RL policy. Its development objective is **strengthening
auto-solving capability** — mainly by adding test cases under `cases/`.
Manual, human-guided research on individual domains happens in downstream
research repos, which install this repo as an editable package; do not mix
the two kinds of work.

## Where things are

The full per-file map lives in `README.md` §Repository layout — keep it
there, not here. The short version: **two independently-published artifacts**
in their own subtrees, packaging disjoint — the PyPI wheel ships only the
`harness/` packages (`mdp_ir`, `mdp_conformance`, `mdp_gates`, `mdp_tuning`,
`mdp_stage`); the plugin ships only `plugin/` (the `mdp-solver` conductor +
six step skills `mdp-formalize` … `mdp-package`, plus `mdp-contribute` and
`mdp-propose`). `cases/` and the root-level material ship in neither.

Canonical documents — each the single source for its subject; consult, don't
restate:

- `plugin/skills/mdp-solver/MDP_PROJECT_SPEC.md` — per-domain conventions
  (architecture, naming, RNG, scripts, eval, §14 interpret). Cited by
  §-number from case docs.
- `plugin/skills/mdp-solver/ESCALATION_LOG_GUIDE.md` — campaign-log format.
  Both docs' §-numbers are append-only: cited from append-only case logs,
  never renumbered.
- `plugin/skills/mdp-solver/CONTRACTS.md` — the split pipeline's op map,
  entry gates (`python -m mdp_stage`), and the two single-writer state files.
- `plugin/skills/mdp-solver/examples/MANIFEST.md` — what each frozen example
  teaches and the gate lines it must keep green.
- `scratch/{AGENT_PLAN,SOLVE_LEVELS_PLAN,IR_LAYERING_PLAN,AGENT_COMPETITION_PLAN}.md`
  — design records; read before resuming the work they pin. **Untracked**
  (`scratch/` is gitignored; this checkout holds the only current copies —
  never `git clean -fdX` here), though spec/guide/code still cite their
  §-numbers, which stay append-only like the rest.

## Rules

- **`cases/` and `plugin/skills/mdp-solver/examples/` are author-owned — spec,
  doc and harness work here does not edit them.** A change to the spec, the
  guide, a schema or the harness lands *without* touching anything inside those
  folders — their code, schemas, tests, `README.md`, `CLAUDE.md` and
  campaign-record docs alike — and instead **names in its commit body which
  statements it invalidated**, so the contributing author moves the folder
  forward on their own clock. The reason is divergence, not etiquette: these
  folders are mirrored in the authors' research repos, and an upstream edit
  drifts from the downstream copy *silently*, which is a worse failure than the
  stale line it was trying to fix — a stale line is visible. If a change cannot
  be expressed without touching contributed material, stop and ask.
  `DOMAIN_CLAUDE_TEMPLATE.md` and `DOMAIN_README_TEMPLATE.md` are the source
  templates and are maintained here; the per-domain `CLAUDE.md` and `README.md`
  files emitted from them into cases/examples are not.
- `plugin/skills/mdp-solver/examples/` is additionally frozen in its code,
  schemas, and tests: those change only when pipeline/spec work requires it,
  never as research — and then through the entry's author, per the rule above.
  Campaign-record docs (`PLAYBOOK.md`, `ESCALATION.md`, `INTERPRET.md`) are
  maintainer-curated instead (see the manifest). Each entry must keep the
  manifest gates green.
- Adding an example (promoting a finished domain from a research repo, or a
  `cases/` folder per the criteria in `cases/README.md` — deliverable-
  competitive incl. §14 readback, coverage, debt paid): move the **whole**
  folder into `plugin/skills/mdp-solver/examples/` (campaign record included
  — promotion is also the publication decision, since examples ship with the
  plugin), add its gate lines to that dir's `MANIFEST.md`. Zero code edits —
  domains are location-independent by the portable-domain contract (in-folder
  `{domain}_ir_adapter.py`, IR-relative builtin resolution, path-taking tools,
  folder-relative outputs).
- New test cases go under `cases/<case_name>/` and follow the skill end to
  end (Phase A interview → gates → leaderboard). Trained artifacts
  (`results/`) are gitignored; the case README's commands must reproduce them.
- Run everything through one resolved interpreter: `$MDP_SOLVER_PYTHON`, else
  `.venv/bin/python` here, else any venv with this repo installed. Use
  `pip install -e "./harness[domain]"` — the bare install is deliberately
  torch-free (harness only), so the `[domain]` extra is what the
  example/generated domain scripts need (SB3 + torch + tensorboard + pandas,
  plus pytest so a domain's own `{domain}_test.py` runs out of the box).
  `[dev]` is pytest alone — the torch-free path for harness work, enough to
  run the whole suite. This repo ships no `.venv`; create one if absent.
- **Host neutrality lives at the root, not in the folders.** `AGENTS.md` is a
  symlink to this file (for hosts that read only that name) and
  `.codex/config.toml` names `CLAUDE.md` in `project_doc_fallback_filenames`
  so Codex reads every `CLAUDE.md` on its root→cwd walk; `.agents/skills/`
  mirrors `.claude/skills/` so this checkout's skills load in a Codex session.
  Domain folders carry only `CLAUDE.md` — no host file is ever added there,
  and the pipeline ops read a folder's brief at entry on every host
  (spec §1.3), since only Claude Code pushes files below the cwd.
- Never use `param`, `params`, or `param_*` as identifiers (spec rule).
- **Plugin versioning:** the plugin is the versioned unit — individual SKILL.md /
  doc files carry no per-file version stamps. Any change under `plugin/` bumps
  `plugin/.claude-plugin/plugin.json` in the same commit, and each publish to
  `main` gets a matching `git tag v<version>`, so "what version is live" is
  always answerable from the tag and installed copies are comparable to it.
  There is no Codex manifest: Codex installs from `.claude-plugin/` as a
  compatibility fallback (verified 2026-09-16 on Codex 0.154), so one manifest
  serves both hosts.

## Regression suite

Needs pytest — either extra supplies it (`[dev]` is the torch-free one). From
the repo root:

```bash
pytest                    # everything: harness/tests + every domain's own tests
```

`pytest.ini` at the root sets the paths, so a bare run covers all three kinds
of test. They are deliberately separate:

| what | where | why there |
|---|---|---|
| engine tests | `harness/tests/` | synthetic IRs only (`conftest.py` builds them) — the harness's suite must not depend on `plugin/`, since the two published artifacts are disjoint |
| per-domain tests | `{domain}_test.py` **inside** the domain folder | a claim about *that* world is only checkable once the domain exists, and the folder must stay portable — so the helpers come from `mdp_ir.testing`, never a repo conftest |
| case tests | `cases/*/{case}_test.py` | same contract as an example |

Two things follow from the layout. Domain folders must never contain
same-named unprefixed modules (pytest's `prepend` import mode would silently
share the first one) — the spec's `{domain}_*` naming rule is what keeps that
true. And each `{domain}_test.py` stays runnable on its own
(`python inv_single_test.py`), which is what the portable-domain contract
requires.

The CLI gates below are the same checks in tool form — what the skill runs
between stages, and what `$E/MANIFEST.md` lists per example:

```bash
E=plugin/skills/mdp-solver/examples
python -m mdp_conformance $E/inv_single $E/mab $E/game2048  # generated-code shape
python -m mdp_ir.laws     $E/inv_single $E/mab $E/game2048  # IR execution semantics
python -m mdp_ir $E/inv_single/inv_single_schema.json \
                 $E/mab/mab_schema.json \
                 $E/game2048/game2048_schema.json                   # IR validation
python -m mdp_ir.differential $E/inv_single/inv_single_schema.json --all-instances --episodes 40
```

Run `pytest` after any change to `harness/` or an example.
Domains in downstream research repos also depend on these packages —
breaking changes to the IR schema, seed-key construction, or adapter
discovery need a coordinated check there before release.

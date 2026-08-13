# CLAUDE.md

This file guides Claude Code when working in **auto-mdp-solver** — the
standalone MDP-solver pipeline (public brand `auto-mdp-solver`; split out of
a private research workspace on 2026-07-21).

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

## Layout

The repo carries **two independently-published artifacts** in their own
subtrees, plus root-level material for public browsers:

| path | role |
|---|---|
| `harness/` | **the PyPI package** `auto-mdp-solver` (its own `pyproject.toml`) |
| `harness/mdp_ir/` | IR schema (pydantic), interpreter, differential runner |
| `harness/mdp_ir/layering.py` | catalog ⊕ selection resolution + symbolic bounds (one `{domain}_schema.json`; each uncertainty slot declares a `candidates` pool + `default`, instances select candidates and override constants; `load_ir(path, instance=, select=)` resolves, so nothing downstream sees the catalog) |
| `harness/mdp_ir/families.py` | distribution-family registry: derived `mean`/`max`/`min`/`is_discrete` (with latent composition) that symbolic bounds resolve against — never hand-authored per domain |
| `harness/mdp_ir/runtime.py` | generic generator runtime: the one family→numpy sampling dispatch (shared with the interpreter) + `FamilyGenerator`, a spec-§4.2 generator over any registered family; domains bridge it once per slot, so a new catalog candidate needs zero new domain Python |
| `harness/mdp_conformance/` | spec-conformance harness (`python -m mdp_conformance <domain-dir>`) |
| `harness/mdp_gates/` | eval-gate comparison (`python -m mdp_gates`) |
| `harness/mdp_tuning/` | Optuna tuning driver (`python -m mdp_tuning <domain-dir> ...`) |
| `plugin/` | **the Claude Code plugin** (`.claude-plugin/plugin.json` + `skills/`) |
| `plugin/skills/mdp-solver/SKILL.md` | the pipeline skill (source of truth) |
| `plugin/skills/mdp-solver/MDP_PROJECT_SPEC.md` | **canonical** per-domain architecture/naming/RNG/script conventions |
| `plugin/skills/mdp-solver/MDP_IR_SAMPLE.md` | annotated MDP-IR reference |
| `plugin/skills/mdp-solver/DOMAIN_CLAUDE_TEMPLATE.md` | template for the per-domain `{domain}/CLAUDE.md` emitted at Stage 1 (provenance block, file hygiene, seeded traps; campaign-specific slots) |
| `plugin/skills/mdp-solver/examples/` | frozen exemplar domains — ship with the plugin as few-shot exemplars **and** are the regression suite (see `examples/MANIFEST.md` there) |
| `plugin/skills/mdp-solver/PLAYBOOK.md` | shipped escalation-playbook **index** — maintainer-curated; links into promoted examples' in-folder `PLAYBOOK.md`s (digest entries per ESCALATION_LOG_GUIDE §10) |
| `plugin/skills/mdp-contribute/SKILL.md` | the contribution skill (case PRs with the playbook riding in the folder / re-skinned cases; a case too sensitive even re-skinned is not contributed; `.github/workflows/case-gates.yml` is its CI counterpart) |
| `plugin/skills/mdp-propose/SKILL.md` | the proposal skill — mid-campaign spec/schema/process extension proposals, one `upstream-proposal` issue each; maintainer disposition accept (case becomes the regression) / reject / defer, recorded on the issue |
| `.claude-plugin/marketplace.json` | marketplace manifest (points at `./plugin`) |
| `cases/` | auto-solve test cases: one folder per case, built end-to-end by the skill |
| `README.md`, `docs/` | public-facing landing + guides/FAQ (browse on GitHub) |

The two packaging systems are disjoint: the PyPI wheel contains only the
`harness/` packages; the plugin ships only `plugin/`'s recognized dirs (so
`examples/` must live **inside** `plugin/skills/mdp-solver/` to ship). Neither
looks at the other's metadata. `cases/`, `docs/`, `README` ship in neither —
they exist for the public GitHub repo.

## Rules

- `plugin/skills/mdp-solver/examples/` is frozen in its code, schemas, and
  tests: those change only when pipeline/spec work requires it, never as
  research. Campaign-record docs (`PLAYBOOK.md`, `ESCALATION.md`,
  `INTERPRET.md`) are maintainer-curated instead (see the manifest). Each
  entry must keep the manifest gates green.
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
- Never use `param`, `params`, or `param_*` as identifiers (spec rule).
- **Plugin versioning:** the plugin is the versioned unit — individual SKILL.md /
  doc files carry no per-file version stamps. Any change under `plugin/` bumps
  `plugin/.claude-plugin/plugin.json` in the same commit, and each publish to
  `main` gets a matching `git tag v<version>`, so "what version is live" is
  always answerable from the tag and installed copies are comparable to it.

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
python -m mdp_conformance $E/inv_single $E/dynamic_pricing $E/fnv $E/mab  # generated-code shape
python -m mdp_ir.laws     $E/inv_single $E/dynamic_pricing $E/fnv $E/mab  # IR execution semantics
python -m mdp_ir $E/inv_single/inv_single_schema.json \
                 $E/dynamic_pricing/dynamic_pricing_schema.json \
                 $E/fnv/fnv_schema.json \
                 $E/mab/mab_schema.json                             # IR validation
python -m mdp_ir.differential $E/inv_single/inv_single_schema.json --all-instances --episodes 40
```

Run `pytest` after any change to `harness/` or an example.
Domains in downstream research repos also depend on these packages —
breaking changes to the IR schema, seed-key construction, or adapter
discovery need a coordinated check there before release.

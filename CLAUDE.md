# CLAUDE.md

This file guides Claude Code when working in **auto-mdp-solver** — the
standalone MDP-solver pipeline (public brand `auto-mdp-solver`; split out of
a private research workspace on 2026-07-21).

## What this repo is

The automatic solver that turns a verbal dynamic-decision problem into a
trained, deployable RL policy. Its development objective is **strengthening
auto-solving capability** — mainly by adding test cases under `cases/`.
Manual, human-guided research on individual domains happens in downstream
research repos, which install this repo as an editable package; do not mix
the two kinds of work.

## Layout

| path | role |
|---|---|
| `skills/mdp-solver/SKILL.md` | the pipeline skill (source of truth; a user-level `~/.claude/skills/mdp-solver` symlink may point here) |
| `skills/mdp-solver/MDP_PROJECT_SPEC.md` | **canonical** per-domain architecture/naming/RNG/script conventions |
| `skills/mdp-solver/MDP_IR_SAMPLE.md` | annotated MDP-IR reference |
| `MDP_AGENT_PLAN.md` | design rationale / packaging notes (background only) |
| `.claude-plugin/` | plugin + marketplace manifests (repo root = the plugin) |
| `mdp_ir/` | IR schema (pydantic), interpreter, differential runner |
| `mdp_conformance/` | spec-conformance harness (`python -m mdp_conformance <domain-dir>`) |
| `mdp_gates/` | eval-gate comparison (`python -m mdp_gates`) |
| `mdp_tuning/` | Optuna tuning driver (`python -m mdp_tuning <domain-dir> ...`) |
| `examples/` | frozen exemplar domains — the regression suite (see `examples/MANIFEST.md`) |
| `cases/` | auto-solve test cases: one folder per case, built end-to-end by the skill |

## Rules

- `examples/` is frozen: entries change only when pipeline/spec work requires
  it, never as research. Each entry must keep the manifest gates green.
- Adding an example (promoting a finished domain from a research repo): move
  the domain folder in, add its gate lines to `examples/MANIFEST.md`. Zero
  code edits — domains are location-independent by the portable-domain
  contract (in-folder `{domain}_ir_adapter.py`, IR-relative builtin
  resolution, path-taking tools, folder-relative outputs).
- New test cases go under `cases/<case_name>/` and follow the skill end to
  end (Phase A interview → gates → leaderboard). Trained artifacts
  (`results/`) are gitignored; the case README's commands must reproduce them.
- Run everything with this repo's venv (`.venv/bin/python`) or any venv with
  `pip install -e .` of this repo.
- Never use `param`, `params`, or `param_*` as identifiers (spec rule).

## Regression suite

From the repo root (see `examples/MANIFEST.md` for the authoritative list):

```bash
python -m mdp_ir.interpreter_test
python -m mdp_conformance examples/inv_single examples/dynamic_pricing
python -m mdp_ir examples/inv_single/inv_single_schema.json
python -m mdp_ir.differential examples/inv_single/inv_single_schema.json --episodes 40
python -m mdp_ir.differential examples/dynamic_pricing/vanryzin_pricing_schema.json --episodes 40
```

Run these after any change to `mdp_ir/`, `mdp_conformance/`, or an example.
Domains in downstream research repos also depend on these packages —
breaking changes to the IR schema, seed-key construction, or adapter
discovery need a coordinated check there before release.

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

The repo carries **two independently-published artifacts** in their own
subtrees, plus root-level material for public browsers:

| path | role |
|---|---|
| `harness/` | **the PyPI package** `auto-mdp-solver` (its own `pyproject.toml`) |
| `harness/mdp_ir/` | IR schema (pydantic), interpreter, differential runner |
| `harness/mdp_conformance/` | spec-conformance harness (`python -m mdp_conformance <domain-dir>`) |
| `harness/mdp_gates/` | eval-gate comparison (`python -m mdp_gates`) |
| `harness/mdp_tuning/` | Optuna tuning driver (`python -m mdp_tuning <domain-dir> ...`) |
| `plugin/` | **the Claude Code plugin** (`.claude-plugin/plugin.json` + `skills/`) |
| `plugin/skills/mdp-solver/SKILL.md` | the pipeline skill (source of truth) |
| `plugin/skills/mdp-solver/MDP_PROJECT_SPEC.md` | **canonical** per-domain architecture/naming/RNG/script conventions |
| `plugin/skills/mdp-solver/MDP_IR_SAMPLE.md` | annotated MDP-IR reference |
| `plugin/skills/mdp-solver/examples/` | frozen exemplar domains — ship with the plugin as few-shot exemplars **and** are the regression suite (see `examples/MANIFEST.md` there) |
| `.claude-plugin/marketplace.json` | marketplace manifest (points at `./plugin`) |
| `cases/` | auto-solve test cases: one folder per case, built end-to-end by the skill |
| `README.md`, `docs/` | public-facing landing + guides/FAQ (browse on GitHub) |

The two packaging systems are disjoint: the PyPI wheel contains only the
`harness/` packages; the plugin ships only `plugin/`'s recognized dirs (so
`examples/` must live **inside** `plugin/skills/mdp-solver/` to ship). Neither
looks at the other's metadata. `cases/`, `docs/`, `README` ship in neither —
they exist for the public GitHub repo.

## Rules

- `plugin/skills/mdp-solver/examples/` is frozen: entries change only when
  pipeline/spec work requires it, never as research. Each entry must keep the
  manifest gates green.
- Adding an example (promoting a finished domain from a research repo): move
  the domain folder into `plugin/skills/mdp-solver/examples/`, add its gate
  lines to that dir's `MANIFEST.md`. Zero code edits — domains are
  location-independent by the portable-domain contract (in-folder
  `{domain}_ir_adapter.py`, IR-relative builtin resolution, path-taking tools,
  folder-relative outputs).
- New test cases go under `cases/<case_name>/` and follow the skill end to
  end (Phase A interview → gates → leaderboard). Trained artifacts
  (`results/`) are gitignored; the case README's commands must reproduce them.
- Run everything through one resolved interpreter: `$MDP_SOLVER_PYTHON`, else
  `.venv/bin/python` here, else any venv with this repo installed. Use
  `pip install -e "./harness[domain]"` — the bare install is deliberately
  torch-free (harness only), so the `[domain]` extra is what the
  example/generated domain scripts need (SB3 + torch + tensorboard + pandas).
  This repo ships no `.venv`; create one if absent.
- Never use `param`, `params`, or `param_*` as identifiers (spec rule).

## Regression suite

From the repo root, with the harness installed (`pip install -e ./harness`);
`E` shortens the examples path (see `$E/MANIFEST.md` for the authoritative list):

```bash
E=plugin/skills/mdp-solver/examples
python -m mdp_ir.interpreter_test
python -m mdp_gates.compare_test
python -m mdp_tuning.resolve_metric_test
python -m mdp_conformance $E/inv_single $E/dynamic_pricing $E/fnv
python -m mdp_ir $E/inv_single/inv_single_schema.json
python -m mdp_ir.differential $E/inv_single/inv_single_schema.json --episodes 40
python -m mdp_ir.differential $E/dynamic_pricing/vanryzin_pricing_schema.json --episodes 40
python -m mdp_ir.differential $E/fnv/fnv_schema.json --episodes 40
```

Run these after any change to `harness/` or an example.
Domains in downstream research repos also depend on these packages —
breaking changes to the IR schema, seed-key construction, or adapter
discovery need a coordinated check there before release.

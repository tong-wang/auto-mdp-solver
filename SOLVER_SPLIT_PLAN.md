# MDP-solver / projects split — migration plan (DRAFT, awaiting approval)

Date: 2026-07-21. Status: **proposal only — nothing has been moved or edited.**

## Objective

Fork two diverging tracks into two repos:

1. **`mdp_solver` repo** — the automatic solver: skill, spec docs, tooling
   packages, frozen exemplar domains, and a growing suite of auto-solve test
   cases. Development = strengthening auto-solving capability.
2. **`rl_test` repo (this one)** — the research projects: topk_id, adi_flex,
   retailer, … where the work is manual, human-guided fine-tuning. Depends on
   the solver repo as an installed package.

## Target layout

### New repo `~/projects/mdp_solver`

```
mdp_solver/
  pyproject.toml            # installable: pip install -e .
  CLAUDE.md                 # solver-repo instructions (derived from spec)
  MDP_PROJECT_SPEC.md       # canonical conventions   (moved from rl_test)
  MDP_IR_SAMPLE.md          # annotated IR reference  (moved)
  MDP_AGENT_PLAN.md         # design rationale        (moved)
  skill/mdp-solver/SKILL.md # source of truth for the skill
  mdp_ir/                   # schema, interpreter, differential (moved)
  mdp_conformance/          # (moved)
  mdp_gates/                # (moved)
  mdp_tuning/               # (moved)
  examples/                 # frozen exemplar domains (COPIED, see decision D1)
    fnv/ cnv/ inv_single/ owmr/ 2048/ sudoku/ dynamic_pricing/
  cases/                    # objective (1): new auto-solve test cases go here,
                            # one folder per case, run end-to-end by the skill
```

- The skill is installed **user-level**: `~/.claude/skills/mdp-solver` →
  symlink to `mdp_solver/skill/mdp-solver/`. It then works in *any* repo:
  in `mdp_solver` for test-case development, in `rl_test` (or a future repo)
  when a research problem needs a new domain scaffolded.
- Package deps in `pyproject.toml`: numpy, gymnasium, sb3_contrib, optuna
  (versions from today's `requirements.txt`); torch arrives via sb3.

### `rl_test` after the split

- Keeps: all domain folders, ops files (`alicloud_monitor/`, `nus_hpc.md`,
  `ALICLOUD_QUEUE.md`, watch scripts), its venv.
- Removes: `mdp_ir/ mdp_conformance/ mdp_gates/ mdp_tuning/`, the three
  `MDP_*.md` docs, `.claude/skills/mdp-solver/`.
- Adds: `mdp-solver` as an editable install (`pip install -e ../mdp_solver`),
  noted in `requirements.txt`; CLAUDE.md updated to point at the solver repo
  for spec/conventions.
- Every existing invocation (`python -m mdp_tuning …`, `python -m mdp_ir …`,
  `python -m mdp_conformance …`, `python -m mdp_gates …`) keeps working
  unchanged once the package is installed — module names don't change.

## Code refactor (the 3 hard couplings)

1. **Differential adapters out of the shared package.**
   `mdp_ir/differential.py` currently imports domain modules directly
   (`ADAPTERS` for inv_single, vanryzin_pricing, adi_flex, topk_id). Change:
   each domain owns `{domain}/{domain}_ir_adapter.py`; `differential.py`
   discovers it from the IR file's directory (same sys.path-insertion pattern
   `mdp_conformance/loader.py` already uses). inv_single + dynamic_pricing
   adapters land in `examples/`; topk_id + adi_flex adapters land in their
   rl_test folders. Gate: differential still bit-exact MATCH for all four.
2. **Interpreter domain-builtin hook made IR-relative.**
   `mdp_ir/interpreter.py` `_domain_builtin("topk_id", "topk_id_probit_map", …)`
   resolves the module relative to the repo root today; change resolution to
   the IR schema file's own directory. Gate: topk_id IR validates + interprets
   from rl_test with only the package installed.
3. **Test fixtures in-repo.**
   `mdp_ir/interpreter_test.py` loads `inv_single_schema.json` and
   `vanryzin_pricing_schema.json` from domain folders; after the exemplar copy
   these resolve inside `examples/`. Gate: interpreter_test 25/25.

Plus skill-text updates: exemplar pointers ("`examples/` in the solver repo,
or a matching domain in the current workspace"), venv wording ("current
workspace venv with mdp-solver installed" instead of repo-root `.venv`).

## Portable-domain contract (cheap later moves, by construction)

A domain folder is **portable** — movable/copyable into `mdp_solver/examples/`
with zero code edits — when:

1. all its code lives in the folder, siblings imported by bare name (already
   the spec convention; every loader sys.path-inserts the folder);
2. its differential adapter is in-folder (`{domain}_ir_adapter.py`,
   refactor 1) — no central registry entry to update;
3. any interpreter domain-builtin resolves relative to the IR file's own
   directory (refactor 2) — no repo-root lookup;
4. its outputs (`results/`, tuning journals) are folder-relative (already true).

All four tools already take explicit paths (`python -m mdp_conformance
<dir>`, differential takes the schema path, mdp_tuning takes the domain dir),
so location is irrelevant to them. One tweak: `mdp_conformance`'s no-arg
discovery scans the package's parent directory — change it to scan the cwd.

**"Promote a finished project" procedure (later, e.g. adi_flex or topk_id):**
`git mv` (or copy) the folder into `mdp_solver/examples/`, then add one line
to the examples regression manifest (the list of gate commands run per
example). Nothing else.

## Domain roster (for hand-picking examples/)

| domain | IR | last touched | status | notes for the pick |
|---|---|---|---|---|
| fnv | – | 2026-07-07 (spec refactor) | frozen | exemplar: single-entity continuous |
| cnv | – | 2026-07-07 (spec refactor) | frozen | exemplar: competitive newsvendor, reward modes |
| inv_single | IR | 2026-07-10 (IR relocation) | frozen | **interpreter_test fixture** — schema must ship with solver repo |
| owmr | – | 2026-07-06 (spec refactor) | frozen; possible future dive | spec's reference implementation; multi-agent exemplar |
| retailer | IR (Phase A only) | 2026-07-15 | frozen; no Phase B planned | reverse-engineered IR, legacy seeds ⇒ differential not bit-exact; remote-server component — poor example |
| 2048 | – | 2026-07-07 (layer split) | frozen | exemplar: discrete + masking |
| sudoku | – | 2026-07-10 | parked (FUTURE_WORK.md queued) | exemplar: deterministic dynamics + sampled instances; also a planned dive |
| dynamic_pricing | IR | 2026-07-10 (IR relocation) | frozen, pipeline-born | **interpreter_test fixture**; exact-DP baseline exemplar |
| adi_flex | IR | 2026-07-15 | in progress (Phase B pass 1 done) | stays in rl_test; promote when finished |
| topk_id | IR | 2026-07-20 | **active** (bayes11 campaigns running) | stays in rl_test; promote when finished |

Constraint: `inv_single` and `dynamic_pricing` schemas are interpreter_test
fixtures — if their full domains are not picked, at minimum their
`*_schema.json` files ship as test fixtures (full domains preferred, since
both also back differential adapters).

**PICKED (user):** _pending_

## Decision D1 — exemplar domains: COPY (recommended) vs MOVE

- **COPY (recommended):** `examples/` gets frozen snapshots; rl_test keeps its
  copies untouched. Why: sudoku has queued future work (FUTURE_WORK.md) and
  owmr is both the spec's reference implementation and a possible future
  deep-dive — under objective (2) those dives belong in rl_test. Cost:
  ~700 KB duplication + a stated rule that `examples/` copies change only via
  solver-pipeline work, never via research edits.
- **MOVE:** cleaner (single copy) but any later manual dive on sudoku/owmr/2048
  would then happen inside the solver repo — re-mixing exactly what this split
  separates.

## Sequencing (protects in-flight campaigns)

Atlas tuning fleets and the queued alicloud campaigns run from clones of
rl_test and invoke `python -m mdp_tuning` etc. from the repo root. Order:

1. **Phase 1 — build solver repo (zero risk to rl_test).** Create
   `~/projects/mdp_solver`, copy packages/docs/exemplars, apply refactors 1–3,
   fresh venv, validate from scratch: conformance 11/11 per exemplar,
   differential MATCH ×4, interpreter_test 25/25, one small mdp_tuning smoke.
   rl_test is not touched at all in this phase.
2. **Phase 2 — rl_test switches over, on a branch.** `pip install -e
   ../mdp_solver` into rl_test's venv; add `{domain}_ir_adapter.py` to
   topk_id/adi_flex; delete root `mdp_*` packages + `MDP_*.md` +
   `.claude/skills/`; re-run topk_id + adi_flex gates. Merge only when green.
3. **Phase 3 — remotes, only when their campaigns drain.** Atlas/alicloud
   stay on their current commit until running studies finish; then: clone
   mdp_solver beside rl_test, `pip install -e`, fast-forward rl_test.
   (Alicloud is down / watches paused — nothing to do there until it's back.)
4. Install the user-level skill symlink; remove the project-level skill copy.

## Going-forward workflow

- **Objective (1) — solver capability:** new test cases are built in
  `mdp_solver/cases/<case_name>/` by running the skill end-to-end there; the
  exemplar set and `cases/` leaderboards become the solver's regression
  suite. IR/interpreter extensions happen in one repo, with
  interpreter_test + differential as the safety net.
- **Objective (2) — project deep dives:** continue in rl_test exactly as this
  week's topk_id work; the solver package is a stable dependency. If a dive
  needs an IR extension, that's a solver-repo change first (released by
  git pull + reinstall), then used from rl_test — an explicit, versioned
  handoff instead of lockstep same-commit edits.

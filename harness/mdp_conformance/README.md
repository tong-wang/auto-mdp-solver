# mdp_conformance

A domain-parametrized checker for the conventions in `MDP_PROJECT_SPEC.md`. It
loads any `{domain}/` directory by the spec's file-naming convention and runs a
suite of static and behavioral invariant checks against it.

It is the reusable backbone for the MDP solver pipeline
(`plugin/skills/mdp-solver/SKILL.md`): the
same per-check report is a **gate** for a generated domain and a **reward signal**
for a trained generator.

## Usage

```bash
# from the repo root, using the project venv
.venv/bin/python -m mdp_conformance plugin/skills/mdp-solver/examples/inv_single plugin/skills/mdp-solver/examples/mab plugin/skills/mdp-solver/examples/game2048

# no args: auto-discover every sibling domain (any dir with a *_mdp.py)
.venv/bin/python -m mdp_conformance
```

Exit code is `0` if no domain has a `FAIL`/`ERROR`, else `1`. `WARN`/`SKIP` are
informational and do not fail the gate.

## What it checks

Static (source/AST, no execution):
- `static.file_layout` — required `_mdp`/`_scenarios`/`_gym`; optional
  `_uncertainty`/`_exceptions`/`_grids`.
- `static.layering` — acyclic `uncertainty ← scenarios ← mdp ← gym` (§1.1);
  `_grids` imported by drivers only (never by the model layers).
- `static.no_param` — the `param`/`params`/`param_*` ban (§2).
- `static.mdp_no_reward` — reward is not computed in the MDP layer (§6.4).
- `static.state_slots` — `{Domain}State` is `@dataclass(slots=True)`.

Scenario architecture (spec §5, §6.3) — **scheme-aware**: a domain declares
`SEED_SCHEME = "v1" | "v2"`; undeclared is treated as v1 (frozen legacy keys,
WARN nudge) so pre-redesign domains keep passing until they migrate:
- `scheme.declared` — `SEED_SCHEME` present, valid, unmixed across modules.
- `scenario.samplers` — every sampler entry in `SCENARIOS` is a pure function:
  same seed twice ⇒ equal concrete scenarios (both schemes).
- `scheme.v2_ids` (v2 only) — `seed_salt >= 1` everywhere; meta drawers carry
  distinct `substream_id`; generator instances carry distinct `source_id`.
- `scheme.v2_keys` (v2 only) — every `SeedSequence` call in the world layers
  goes through `intrinsic_key()` / `meta_key()` (template-drift guard).
- `grids.*` — no grid object inside `SCENARIOS`; `GRIDS` entries are
  enumerable, non-callable, uniquely-celled, and `as_sampler()` is pure.

Behavioral (constructs the gym and runs the simulator):
- `behavior.scenarios_valid` — every `SCENARIOS` entry constructs; `scenario_name`
  matches its registry key.
- `behavior.init_state` — `init_state` returns `(state, info)` with the
  `action_period == state.period - 1` invariant.
- `behavior.gym_contract` — Gymnasium `reset`/`step` shapes; finite reward.
- `behavior.determinism` — same seed + same actions ⇒ identical trajectory.
- `behavior.purity` — `advance*` does not mutate the input state.
- `rng.generators` — each generator's `sample()` is reproducible from
  `(period, episode_seed, seed_salt)`. Generators that yield a value from only
  those fields are **decision-path independent by construction**; generators that
  need more state (e.g. a board-game spawn conditioned on the current board) are
  classified **state-conditioned** and skipped, not failed.

Scripts (reads the parsers, without importing the training stack):
- `scripts.cli_contract` — the train/eval CLIs expose §8.2's **tier-1** names:
  the problem and rendering modes, `total_timesteps`/`outdir`/`seed`/`tag`,
  the VecNormalize trio, `gym_log`/`checkpoint_every_frac`, and on the eval side
  `model_path`/`vecnorm_path`/`outfile`/`n_seeds`/`first_seed`. Tier 2
  (algorithm knobs) is not enumerated here — `mdp_tuning --show-space` is its
  oracle — and tier 3 (domain-specific) is checked by nothing, by design. Two
  groups are conditional and read from the domain: a rendering mode is owed when
  the env's `__init__` accepts it (a domain whose env has no `reward_mode`
  parameter owes no flag), the VecNormalize names when the script builds the
  wrapper. Dests are read from the AST, so the gate needs no SB3/torch. A
  script that does not exist yet is not a violation — a domain mid-formalization
  owes no eval script. **WARN for one release, then FAIL**: every existing
  script fails it on the day it lands, so a hard gate would make every domain
  non-conformant at once.

- `scripts.schedule_pairs` — an exposed schedule init has its final
  (`learning_rate`/`lr_final`, `clip_init`/`clip_final`, §8.2). `mdp_tuning`
  guards the one-DOF derivation on the final's dest existing, so a half-exposed
  pair makes it a silent no-op: `build_schedule` collapses to a constant and the
  run trains flat while its args log and the study both record a tuned init.
  Exposing *neither* half is out of scope — that is tier-2 completeness, which
  `mdp_tuning --show-space` owns. Same WARN-then-FAIL adoption path.

Documents (reads the artifacts beside the code, not the code):
- `docs.restatement_current` — the restatement's sample trajectory still
  renders. A ```` ```step7b ```` fence declares the command that produced it;
  the next fenced block is its output, pasted verbatim; the check re-runs the
  command from the domain folder's parent and diffs the result. No other gate
  can reach this artifact — when the model changes, the interpreter and the
  generated domain move *together*, so the differential stays MATCH while the
  document drifts from both. WARN, and SKIP where nothing is declared, so
  adoption is one fenced block and nothing breaks on upgrade. Only
  `python -m mdp_ir.interpreter` invocations are re-run; anything else in the
  fence is reported, never executed.

## Design notes

- **Behavior is driven through the gym** so single-step (`advance`) and two-step
  (`advance1`/`advance2`) domains are handled uniformly.
- **Discovery is file-based**, not folder-based: the domain prefix is read from
  the files, so a folder may hold a domain whose IR/adapter use an unrelated
  prefix.
- Checks are written against the normalized `DomainHandle`, never a concrete
  domain, so the harness generalizes to freshly generated domains.

# mdp_conformance

A domain-parametrized checker for the conventions in `MDP_PROJECT_SPEC.md`. It
loads any `{domain}/` directory by the spec's file-naming convention and runs a
suite of static and behavioral invariant checks against it.

It is the reusable backbone for the MDP solver pipeline
(`skills/mdp-solver/SKILL.md`): the
same per-check report is a **gate** for a generated domain and a **reward signal**
for a trained generator.

## Usage

```bash
# from the repo root, using the project venv
.venv/bin/python -m mdp_conformance inv_single cnv fnv owmr 2048

# no args: auto-discover every sibling domain (any dir with a *_mdp.py)
.venv/bin/python -m mdp_conformance
```

Exit code is `0` if no domain has a `FAIL`/`ERROR`, else `1`. `WARN`/`SKIP` are
informational and do not fail the gate.

## What it checks

Static (source/AST, no execution):
- `static.file_layout` — required `_mdp`/`_scenarios`/`_gym`; optional
  `_uncertainty`/`_exceptions`.
- `static.layering` — acyclic `uncertainty ← scenarios ← mdp ← gym` (§1.1).
- `static.no_param` — the `param`/`params`/`param_*` ban (§2).
- `static.mdp_no_reward` — reward is not computed in the MDP layer (§6.4).
- `static.state_slots` — `{Domain}State` is `@dataclass(slots=True)`.

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
  need more state (e.g. 2048's board) are classified **state-conditioned** and
  skipped, not failed.

## Design notes

- **Behavior is driven through the gym** so single-step (`advance`) and two-step
  (`advance1`/`advance2`) domains are handled uniformly.
- **Discovery is file-based**, not folder-based: the domain prefix is read from
  the files (e.g. the `2048/` directory uses the `game2048_` prefix).
- Checks are written against the normalized `DomainHandle`, never a concrete
  domain, so the harness generalizes to freshly generated domains.

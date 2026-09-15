---
name: mdp-build
description: >
  Stages 1–2 of the MDP pipeline: generate the spec-conformant domain
  (_uncertainty/_scenarios/_mdp/_gym, adapter, tests, campaign docs) from a
  frozen IR, gated by conformance + laws + differential. Use when
  {name}.signoff.json exists beside the schema and the domain code does not,
  or must be regenerated. Full pipeline: mdp-solver.
---

# mdp-build — Stages 1–2 (IR → domain code + gym)

One op of the split MDP pipeline; `mdp-solver` is the conductor. The governing
docs are at `${CLAUDE_SKILL_DIR}/../mdp-solver/` — read them from that path,
never from a filesystem search (a dev checkout on disk would silently
substitute unreleased content). Read `CONTRACTS.md` there first: it maps the
ops and says what each reads. **This op needs:** spec `MDP_PROJECT_SPEC.md`
§1–§7 (consult it while writing each file; do not code from memory of it),
`MDP_IR_SAMPLE.md`, the two `DOMAIN_*_TEMPLATE.md` files,
`examples/MANIFEST.md` and the one exemplar matched below — not spec §8–§14,
not the escalation guide. Everything else in that directory is consulted only
when a step below names it. Venv and run discipline per `ENVIRONMENT.md`
(retry budgets, background runs).

**Entry gate — run first, exit 0 required:**

```bash
<python> -m mdp_stage {domain} --for build
```

Never start from a prior op's (or a human's) report that the IR is ready —
the gate re-derives it: schema validates, `unconfirmed()` empty, sign-off
present and its fingerprint still current. A blocked entry goes back to
`mdp-formalize`, not around the gate.

Pick few-shot exemplars by problem shape. Three ship with this plugin in the
`examples/` directory of the `mdp-solver` skill (path above; see
`examples/MANIFEST.md` there):
`inv_single` vector state + two-step advance + episode-support demand +
exact-DP baseline; `mab` exploration/exploitation over an equivariant state +
censored feedback + a fitted rule scored as a benchmark; `game2048` a
variable, long horizon + board state + CNN extractor + action masking. If the
current workspace contains other spec-conformant domains, prefer whichever
matches the problem shape (multi-entity, discrete + action masking,
deterministic dynamics + scenario-sampled instances, …); otherwise
generalize from the shipped two plus the spec's patterns.

### Stage 1 — model layers

Write in dependency order: `{domain}_exceptions.py` (optional) →
`{domain}_uncertainty.py` (omit if dynamics deterministic) →
`{domain}_scenarios.py` → `{domain}_mdp.py`.

- Seed keys follow the **§6.3 v2 seed tree**: declare `SEED_SCHEME = "v2"`
  and set `seed_scheme: "v2"` in the IR; build every key through the
  domain's `intrinsic_key()` / `meta_key()` helpers (never a raw
  `SeedSequence`). The interpreter implements both schemes and the
  differential gate is bit-exact under v2 (`examples/inv_single` is the
  reference); `"v1"` exists only for pre-redesign domains with recorded
  results.
- A generator whose distribution depends on the current decision takes it
  as an extra `sample(ctx, <decision>)` argument; the seed key must still
  contain only `(period, episode_seed, seed_salt)` slots so realized
  randomness is decision-path independent.
- Write a differential adapter **in the domain folder**:
  `{name}/{ir_name}_ir_adapter.py` exposing
  `make_adapter(ir, instance=None, seed_salt=0, domain_dir=None)` with
  `domain_dir` defaulting to the adapter's own directory
  (`mdp_ir.differential.load_adapter_factory` discovers it from the IR
  file's directory; `examples/inv_single/inv_single_ir_adapter.py` is the
  reference). Build the scenario **from the IR's constants**; emit
  row keys named exactly like interpreter rows; pass `seed_salt` through.
- Give `{domain}_mdp.py` a `__main__` smoke episode and run it.

- Write `{domain}_test.py` in the domain folder (spec §1.2): the engine laws,
  the differential parametrized over the covering set derived from the schema,
  at least one negative control, and any claim an IR expression cannot state.

- Emit the folder's two campaign docs from the templates beside this skill,
  in the shapes spec §1.3 fixes. They divide by direction: `CLAUDE.md` points
  **up**, at the specs governing the folder; `README.md` points **across**, at
  everything the campaign produces. A campaign document is inventoried in
  exactly one of them.
  - `{domain}/CLAUDE.md` from `DOMAIN_CLAUDE_TEMPLATE.md` — five sections:
    fill the problem statement, the leaderboard-commensurability line, the
    two provenance lines and the gate-command slots; keep the fixed text
    verbatim. The template's filling rules are binding — a pointer may name a
    destination, never describe or score what is inside it, and a hard rule is
    phrased trigger → destination.
  - `{domain}/README.md` from `DOMAIN_README_TEMPLATE.md` — the lead, the
    TL;DR block and four sections: write the one- or two-sentence lead (what
    the case is, its source, generated from the IR) and fill **The problem**
    and the two layout tables now (documents the campaign owes are listed
    "owed", never omitted); the TL;DR keeps its one-line placeholder, and the
    results and appendix sections are seeded as headings and grow as findings
    land, not at Stage 6.

- Emit `{domain}/.gitignore` beside it — three lines: `results/`, `scratch/`,
  `__pycache__/`. The folder must carry its own ignore rules (portable-domain
  contract): a contributed or promoted folder cannot assume the destination
  repo's root `.gitignore` covers them.

**GATE:** all three must exit 0 —
`python -m mdp_conformance {domain}` (generated-code shape),
`python -m mdp_ir.laws {domain}` (IR execution semantics), and
`python -m mdp_ir.differential {name}/{name}_schema.json --all-instances --episodes 40`.
The differential must be MATCH (bit-exact) with **no invariant violation**.
Then run `pytest {domain}` for the domain's own tests, and `pytest` at the repo
root to prove no regression to the shipped examples.

**Why bit-exact, and not just "tests pass":** formalization has no oracle —
dynamics can be modeled plausibly but wrongly, and a generated domain that
is self-consistent will pass every unit test while encoding the wrong
problem. The IR interpreter is the only independent implementation of those
dynamics, so replaying identical `(instance, episode_seed)` through both and
demanding an exact trajectory match is what actually catches a
mis-formalization. This is the reason the differential gate exists; treat a
MATCH failure as a modeling error to diagnose, never as a tolerance to relax.

### Stage 2 — gym wrapper

`{domain}_gym.py` from the IR's `gym` block: every observation mode (never
exposing latent state), every action mode with its declared strategy
(mask → `valid_actions` in the MDP layer + `action_masks()` on the env;
clip / reparametrize per spec §7.2), reward modes from objective
components in `info`, `terminated` at horizon + any early-termination expr.

- **The rendered vector is the declared vector** (spec §7): the modes the
  wrapper accepts are the modes the IR names, and each renders exactly that
  mode's `features`, in order. When the wrapper wants a component the IR does
  not declare, edit the IR first — `gym` is a mutable block, so a feature line
  costs no re-confirmation and moves no fingerprint. **The time feature is the
  one this loses**, in the domains that have one: where the horizon is the
  problem's own boundary (`termination.horizon_end: "terminated"`) the gym
  prepends one as a formatting decision, so it is written in the code and
  nowhere else unless you declare it. Where the horizon is only a cap
  (`"truncated"`) the problem is stationary, the policy must not see the clock,
  and there is nothing to declare — spec §7. No check catches either half; read the vector against
  the list by hand at the gate below — by *provenance*, not by width, since
  a mode may re-encode what it declares (a one-hot expansion of a declared
  board is compliant; an undeclared scalar prepended to it is not, and the
  two look alike if you only count). Spec §7 also states which encoding to
  prefer (`time_to_go = T - t`) and the horizon-proportional exception.

- **Action-box design lesson:** SB3 PPO's Gaussian initializes at raw
  action 0. If raw 0 maps to a dead zone (e.g. a shut-off intensity),
  learning stalls. Prefer encodings where raw 0 is a live, low-value
  action; keep the paper-native encoding as a non-default mode.

**GATE:** the module `__main__` runs random-action episodes in every
obs/action mode with `observation_space.contains(obs)` asserted each step,
and `python -m mdp_conformance {domain}` still passes.

Exit: both stage gates green. `<python> -m mdp_stage {domain} --for solve`
runs the same three checks as its entry — hand off to `mdp-solve` (or back to
the conductor).

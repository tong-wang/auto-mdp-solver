# examples/ manifest — the solver regression suite

One row per example domain; the gate commands are run from the repo root and
must all pass. Two promotion sources: a finished project from a research
repo, or a `cases/` folder promoted **whole** — campaign record
(`PLAYBOOK.md`, `ESCALATION.md`, `INTERPRET.md`, figures) included, since
everything here ships with the plugin and the in-folder playbook must be
reachable by the installed skill. Admission criteria live in
`cases/README.md`. Either way: move the folder (never copy — two folders
with the same `{name}_*` modules break pytest's import mode), add its
row(s), update the shape-coverage note below.

The freeze is scoped: **code, schema, and tests are frozen** — no research
edits, ever. The campaign-record docs are maintainer-curated instead:
recurrence marks and scope corrections may land there, each shipping as a
plugin version bump so "what shipped" stays answerable from the tag.

**Where an entry is not exemplary, say so here.** These folders are read as
few-shot examples, so a domain built before a convention changed will teach
the superseded pattern by demonstration — a prose disclaimer inside the
folder keeps its own record honest but does not stop the code being copied.
Current caveats:

- `mab` — its **selection machinery predates v0.7.0**. The train script uses
  a live in-training `SelectionEvalCallback`; §8.6/§9.7 now specify a
  post-hoc three-layer screen and state "no `EvalCallback`, no live selection
  env" — a change this campaign's own #E33/#E36 V4 motivated. Read `mab` for
  the shapes listed below; take the selection pattern from the spec, not from
  this script. The code stays as it is because every number in the folder was
  produced by it (see its `CLAUDE.md` Traps and `README`); `mab_selection_probe.py`
  is the post-hoc screen the campaign ran by hand. As of v0.9.23 the
  `scripts.selection_protocol` check names this in the report — a **WARN**, so
  the gates stay green and the caveat is no longer prose alone.
- `inv_single` and `dynamic_pricing` **declare no `research_questions` stance**,
  and that is correct rather than an omission: neither ships a
  `{domain}_policy_probe.py` or an `INTERPRET.md`, so neither makes a tier-2
  claim. Spec §14.0 ties the deliverable to the declared stance, so declaring
  `confirm` here would fail `research.deliverables` — the check would be right.
  `mab` is the entry that carries the §14 leg, and the one whose stances are
  declared. Both do declare `benchmarks` (their exact DPs, and
  `dynamic_pricing`'s fluid policies) as of 2026-08-16.
- `mab`'s **§MAP was retyped on 2026-08-16** to the guide's
  `cases`/`means`/`designs` taxonomy and its drawing contract (upstream
  #19/#20/#22), so the folder no longer teaches the withdrawn `S`/`P` typing by
  demonstration. Campaign-record curation, not research: no number, verdict,
  mark or ranking moved, and the retype's own §FRAME-CHANGELOG line lists the
  four structural defects the old typing had permitted — chief among them `L0`
  drawn as a crown-eligible sibling of `L1`, which spec §8.6 forbids. The
  *code* still predates v0.7.0 per the caveat above; only the record was
  redrawn.

- `mab`'s **restatement was corrected on 2026-08-18** for the defect upstream
  #55 was filed about. Both trajectory blocks carried three rows and an
  ellipsis, and the Bernoulli one was attributed to the bare `mab_schema.json`
  invocation — but the `payout` slot's catalog `default` is `gaussian`, so that
  command has rendered the Gaussian branch since promotion. The numbers were
  right; the command printed beside them was not, which no gate could see (the
  `mdp` fingerprint never moved). Both blocks now sit under `step7b` fences
  with verbatim output, so `docs.restatement_current` re-runs them. Campaign-
  record curation, not research: no number, verdict or ranking moved.

- `game2048` was **promoted with declared debt outstanding**, which the
  admission criteria in `cases/README.md` otherwise forbid — the criterion says
  the debt is paid *before* the freeze, and here it was not. Two things are
  owed: spec §14.3's figure contract is only partly met (`figures/` carries
  committed replay animations, but there is no `game2048_plot_policy.py`
  emitting a static *and* an interactive render of the action surface from one
  spec), and the fitted rule that its `discover` stance owes as a first-class
  §9 benchmark was not delivered. The folder's own `README.md` and
  `INTERPRET.md` both say so; this line records that the maintainer waived the
  criterion rather than that the folder hid the gap. **Read this as "does not
  meet §14.3 today", not "can never".** The freeze bars *research* edits; a
  spec-contract deliverable is exactly what still lands here when pipeline or
  spec work calls for it, through the entry's author — which is how every other
  example's code has moved. The fitted rule is the harder of the two, since
  fitting and scoring one is research and would need a spec change to arrive by
  that route.
- `game2048`'s **campaign record is a deliberate 32-of-85 subset**, so it is a
  weaker drill-down than `mab`'s. Nine `#E` citations do not resolve (12, 16,
  18, 19, 20, 30, 34, 35, 45 — `#E12` dangles in the source record too, having
  been reserved and never authored), and `§CONFIG-REGISTRY` is omitted. The
  held-back material — the afterstate-factored policy, the critic-architecture
  programme, the GAE-λ axis, the adversarial-spawn programme and the visitation
  channel — is ongoing research on a chassis this folder does not publish. Read
  the entries that are here; do not chase a citation that is not.
- `game2048`'s **`PLAYBOOK.md` carries no lever entries**, so it contributes no
  rows to the shipped `PLAYBOOK.md` index. That is the campaign's own verdict,
  not an omission: its transferable findings are either textbook, or general
  method belonging in the escalation guide, or — for the one genuinely
  non-obvious lever it measured, the GAE-λ descent — established on the
  unpublished chassis and **not used by either shipped crown**, both of which
  run at the λ = 0.95 default. Read this entry for the *shapes* it covers
  below, not for levers.
- `game2048` is the first example whose deliverable **loses at one of its two
  scales**, and it is promoted anyway because the criterion is beating the
  classical baselines, which it does at both. At `3x3_20` the trained policy
  clears everything on the board (2688.37 against a 1154.25 bar). At `4x4_20`
  it beats every baseline and expectimax d2 but reaches only 82% of d3, and the
  bar — expectimax d3 with a hand-shaped monotone-chain leaf evaluator — stands
  3.0× above it. The folder states that negative as its headline. An exemplar
  that only ever won would teach that the pipeline always wins.

Every example carries three gates, all run from the repo root with
`E=plugin/skills/mdp-solver/examples`:

| example | origin | gates |
|---|---|---|
| `inv_single` | imported 2026-07-21 | `python -m mdp_conformance $E/inv_single` · `python -m mdp_ir $E/inv_single/inv_single_schema.json` · `pytest $E/inv_single` |
| `dynamic_pricing` | imported 2026-07-21 | `python -m mdp_conformance $E/dynamic_pricing` · `python -m mdp_ir $E/dynamic_pricing/dynamic_pricing_schema.json` · `pytest $E/dynamic_pricing` |
| `mab` | promoted from `cases/` 2026-08-13 (contributed PR #13) | `python -m mdp_conformance $E/mab` · `python -m mdp_ir $E/mab/mab_schema.json` · `pytest $E/mab` |
| `game2048` | promoted from `cases/` 2026-09-01 (contributed PR #73) | `python -m mdp_conformance $E/game2048` · `python -m mdp_ir $E/game2048/game2048_schema.json` · `pytest $E/game2048` |

The third gate is each domain's own `{domain}_test.py`, sitting beside the code
it describes. It runs the engine laws (`mdp_ir.laws`) and **parametrizes the
differential over the covering set read from the schema** — one test node per
declared instance and mixture, so a break names the composition instead of
stopping at the first one. Because the set is derived rather than listed,
adding an instance to a catalog extends the sweep with no edit here.

`inv_single`'s covering set: the `simple` base + `simple_k` + lead-time `lt` +
event-order `rdo`/`rod` + `lost_sales` + latent-demand `discrete`/`poisson`
(+ their `*_lost_sales`) + the cross-family mixture `mix_demand` — all three
demand candidates, both lead-time candidates, both stockout modes, the
event-order-as-constant mode, per-instance `pipeline_len`, and the per-episode
re-selection path.

Each domain is ONE catalog schema (`{domain}_schema.json`, IR_LAYERING_PLAN
§10): slots declare candidate pools, instances select candidates + override
constants. The CLI form
(`python -m mdp_ir.differential <schema> --all-instances --episodes 40`) runs
the same sweep in one process and stays the tool for manual investigation.
Each schema also declares its conservation laws under `mdp.invariants`, which
the differential and the laws gate both enforce.

`mab`'s covering set: both `payout` candidates (`bernoulli` base +
`gaussian`) × the K/T grid — 33 declared instances spanning K ∈ {5, 10, 20}
and T ∈ {10 … 40000}, plus the `gaussian` composition itself. Its size axes
are symbolic (`n_arms`, `horizon_T`), so registering a further cell extends
the sweep without touching the structure.

`game2048`'s covering set: the `3x3_20` base + seven instances spanning board
size × spawn regime (2×2, 3×3, 4×4, 5×5 × `prob_4` ∈ {0.0, 0.2}) — eight
compositions in all. `3x3_20` is the base itself and is therefore correctly
absent from the `instances` block; `3x3_0` overrides only `prob_4` and
`4x4_20` only the geometry, so the grid is expressed as deviations rather
than enumerated. The board dimension is symbolic (`grid_size`, `n_cells`,
`T_cap`), so a further board size is one instance row.

Shape coverage note: these cover continuous single-entity control, two-step
advance, episode-support demand, decision-conditioned generators, exact-DP
benchmarks, and a **cross-family world mixture** (`inv_single`'s
`mix_demand`, exercising per-episode candidate re-selection and the
family-generic runtime); and, from `mab`, the **design layer** with a
trained generalist over a `{domain}_grids.py` grid, **discrete actions** (the
first — the other two are continuous), **censored information** (only the
pulled arm is observed, so the agent must infer the rest), a **decision that
selects which exogenous stream is read** (`key_exprs` on a period stage — the
regression case for v0.5.9), **symbolic size axes** (bounds and lengths
naming scenario constants, so one IR spans 34 compositions), and the **§14
interpret leg** end to end — a policy probe, a fitted structural rule scored
as a first-class §9 benchmark, and the campaign record (`ESCALATION.md`,
`PLAYBOOK.md`, `INTERPRET.md`, `CLAUDE.md`) that makes the in-folder playbook
reachable by the installed skill.

And, from `game2048`, the **spatial board** — a 2-D grid observation with CNN
extractors (the case carries `SmallBoardCnn` per spec §8.5, since SB3's
NatureCNN cannot run on a 3×3 board), **action masking** (`masked`, with a
`masked`/`free` feasibility-strategy menu declared in one IR), a
**state-dependent categorical support** (the spawn cell is `categorical` over
`empty_cells(board)`, resolved from the live namespace at draw time — which is
why `mdp_ir.laws` correctly declines to assert `path_independence` here),
**uncertainty stages keyed on something other than `period`** (both spawn
slots key on `spawn_count`), **domain expression builtins**
(`mdp.expr_builtins` → `game2048_board.py`, so the slide/merge rules have one
implementation shared by the IR and the `_mdp` layer), and the shape where
**neither an `exact` nor a `relaxed` node exists** — tile values are unbounded,
so the state space cannot be enumerated and every bar is `feasible`. That last
one matters as an exemplar: the optimum is never bracketed, so the strongest
reference is the best policy anyone built rather than a ceiling, and the role
vocabulary has to carry that without a DP to anchor it.

Not yet covered by a shipped example:
multi-entity, deterministic dynamics + sampled instances, competitive
information modes, an information-accumulation (MMFE) state, and a
terminal-only stochastic payoff.

`mab`'s grid trips the §5.6 `grids.axes` tier warning (its axis is the
horizon), so that branch of the check has a live regression; the tier-3 PASS
branch is covered by `harness/tests/test_conformance_grid_axes.py` on
synthetic IRs, which is where a classifier's own regression belongs. `mab`'s
results are **Gaussian-only by declared scope** — the `bernoulli` branch is
implemented and differentially gated but deliberately not evaluated, which is
a scope boundary, not an unpaid debt; the folder states what that forgoes.

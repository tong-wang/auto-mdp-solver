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
  is the post-hoc screen the campaign ran by hand.
- `mab`'s **§MAP was retyped on 2026-08-16** to the guide's
  `cases`/`means`/`designs` taxonomy and its drawing contract (upstream
  #19/#20/#22), so the folder no longer teaches the withdrawn `S`/`P` typing by
  demonstration. Campaign-record curation, not research: no number, verdict,
  mark or ranking moved, and the retype's own §FRAME-CHANGELOG line lists the
  four structural defects the old typing had permitted — chief among them `L0`
  drawn as a crown-eligible sibling of `L1`, which spec §8.6 forbids. The
  *code* still predates v0.7.0 per the caveat above; only the record was
  redrawn.

Every example carries three gates, all run from the repo root with
`E=plugin/skills/mdp-solver/examples`:

| example | origin | gates |
|---|---|---|
| `inv_single` | imported 2026-07-21 | `python -m mdp_conformance $E/inv_single` · `python -m mdp_ir $E/inv_single/inv_single_schema.json` · `pytest $E/inv_single` |
| `dynamic_pricing` | imported 2026-07-21 | `python -m mdp_conformance $E/dynamic_pricing` · `python -m mdp_ir $E/dynamic_pricing/dynamic_pricing_schema.json` · `pytest $E/dynamic_pricing` |
| `mab` | promoted from `cases/` 2026-08-13 (contributed PR #13) | `python -m mdp_conformance $E/mab` · `python -m mdp_ir $E/mab/mab_schema.json` · `pytest $E/mab` |

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
reachable by the installed skill. Not yet covered by a shipped example:
multi-entity, action masking, deterministic dynamics + sampled instances,
competitive information modes, an information-accumulation (MMFE) state, and
a terminal-only stochastic payoff.

`mab`'s grid trips the §5.6 `grids.axes` tier warning (its axis is the
horizon), so that branch of the check has a live regression; the tier-3 PASS
branch is covered by `harness/tests/test_conformance_grid_axes.py` on
synthetic IRs, which is where a classifier's own regression belongs. `mab`'s
results are **Gaussian-only by declared scope** — the `bernoulli` branch is
implemented and differentially gated but deliberately not evaluated, which is
a scope boundary, not an unpaid debt; the folder states what that forgoes.

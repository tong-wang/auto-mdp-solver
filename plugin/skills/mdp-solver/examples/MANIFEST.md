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

Every example carries three gates, all run from the repo root with
`E=plugin/skills/mdp-solver/examples`:

| example | origin | gates |
|---|---|---|
| `inv_single` | imported 2026-07-21 | `python -m mdp_conformance $E/inv_single` · `python -m mdp_ir $E/inv_single/inv_single_schema.json` · `pytest $E/inv_single` |
| `dynamic_pricing` | imported 2026-07-21 | `python -m mdp_conformance $E/dynamic_pricing` · `python -m mdp_ir $E/dynamic_pricing/dynamic_pricing_schema.json` · `pytest $E/dynamic_pricing` |
| `fnv` | added 2026-07-23 | `python -m mdp_conformance $E/fnv` · `python -m mdp_ir $E/fnv/fnv_schema.json` · `pytest $E/fnv` |

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

Shape coverage note: these cover continuous single-entity control, two-step
advance, episode-support demand, decision-conditioned generators, exact-DP
benchmarks, and a **cross-family world mixture** (`inv_single`'s
`mix_demand`, exercising per-episode candidate re-selection and the
family-generic runtime); and the **design layer** — a `{domain}_grids.py`
with `GRIDS`, generalist-over-a-grid training, an information-accumulation
(MMFE) state, and a terminal-only stochastic payoff (`fnv`, the reference
grid exemplar). Not yet covered by a shipped example: multi-entity,
discrete + action masking, deterministic dynamics + sampled instances,
competitive/censored information modes.

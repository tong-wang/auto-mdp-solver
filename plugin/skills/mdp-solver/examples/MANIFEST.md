# examples/ manifest — the solver regression suite

One row per example domain; the gate commands are run from the repo root and
must all pass. Promoting a finished project from a research repo = move the
folder here + add its row(s). Entries are frozen: no research edits.

| example | origin | gates |
|---|---|---|
| `inv_single` | imported 2026-07-21 | `python -m mdp_conformance plugin/skills/mdp-solver/examples/inv_single` · `python -m mdp_ir plugin/skills/mdp-solver/examples/inv_single/inv_single_schema.json` · `python -m mdp_ir.differential plugin/skills/mdp-solver/examples/inv_single/inv_single_schema.json --episodes 40` (+ `--instance lost_sales`) |
| `dynamic_pricing` | imported 2026-07-21 | `python -m mdp_conformance plugin/skills/mdp-solver/examples/dynamic_pricing` · `python -m mdp_ir plugin/skills/mdp-solver/examples/dynamic_pricing/vanryzin_pricing_schema.json` · `python -m mdp_ir.differential plugin/skills/mdp-solver/examples/dynamic_pricing/vanryzin_pricing_schema.json --episodes 40` |
| `fnv` | added 2026-07-23 | `python -m mdp_conformance plugin/skills/mdp-solver/examples/fnv` · `python -m mdp_ir plugin/skills/mdp-solver/examples/fnv/fnv_schema.json` · `python -m mdp_ir.differential plugin/skills/mdp-solver/examples/fnv/fnv_schema.json --episodes 40` |

All three schemas are also `harness/mdp_ir/interpreter_test.py` fixtures.

Shape coverage note: these cover continuous single-entity control, two-step
advance, episode-support demand, decision-conditioned generators, and exact-DP
benchmarks (`inv_single`, `dynamic_pricing`); and the **design layer** — a
`{domain}_grids.py` with `GRIDS`, generalist-over-a-grid training, an
information-accumulation (MMFE) state, and a terminal-only stochastic payoff
(`fnv`, the reference grid exemplar). Not yet covered by a shipped example:
multi-entity, discrete + action masking, deterministic dynamics + sampled
instances, competitive/censored information modes.

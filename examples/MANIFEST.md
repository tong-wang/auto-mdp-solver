# examples/ manifest — the solver regression suite

One row per example domain; the gate commands are run from the repo root and
must all pass. Promoting a finished project from a research repo = move the
folder here + add its row(s). Entries are frozen: no research edits.

| example | origin | gates |
|---|---|---|
| `inv_single` | imported 2026-07-21 | `python -m mdp_conformance examples/inv_single` · `python -m mdp_ir examples/inv_single/inv_single_schema.json` · `python -m mdp_ir.differential examples/inv_single/inv_single_schema.json --episodes 40` (+ `--instance lost_sales`) |
| `dynamic_pricing` | imported 2026-07-21 | `python -m mdp_conformance examples/dynamic_pricing` · `python -m mdp_ir examples/dynamic_pricing/vanryzin_pricing_schema.json` · `python -m mdp_ir.differential examples/dynamic_pricing/vanryzin_pricing_schema.json --episodes 40` |

Both schemas are also `mdp_ir/interpreter_test.py` fixtures.

Shape coverage note: these two cover continuous single-entity control,
two-step advance, episode-support demand, decision-conditioned generators,
and exact-DP benchmarks. Not yet covered by a shipped example: multi-entity,
discrete + action masking, deterministic dynamics + sampled instances,
competitive/censored information modes.

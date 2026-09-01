# examples/ manifest — the solver regression suite

Three domains, each carried for a distinct reason. They serve two purposes at
once: they are the **regression suite** (every gate below must pass) and the
**few-shot exemplars** the skill reads when building a new domain. Everything
here ships with the plugin, which is why the set is kept small — a fourth
entry has to teach something the other three cannot.

## What is here

| example | why it earns a place | origin |
|---|---|---|
| `inv_single` | The **classical MDP** in its standard form: a vector state under stochastic demand, a fixed ordering cost, and a stochastic lead time. This is the shape most new domains resemble, so it is the default thing to read first. | imported 2026-07-21 |
| `mab` | The **standard bandit** — exploration against exploitation, with a per-episode latent the agent must infer from censored feedback (only the pulled arm is observed). Its state is **equivariant**: the arms are exchangeable, so the policy must not depend on their ordering. | promoted from `cases/` 2026-08-13 (PR #13) |
| `game2048` | A **variable and long horizon** — the episode ends when the board jams, not on a clock — over a **board state** read by a **CNN feature extractor**, with masked discrete actions. The one entry with no DP: tile values are unbounded, so the optimum is never bracketed. | promoted from `cases/` 2026-09-01 (PR #73) |

## Gates

Three per entry, identical in form. From the repo root, `D` = the entry's
directory:

```bash
python -m mdp_conformance $D
python -m mdp_ir          $D/{name}_schema.json
pytest                    $D
```

Each folder's `CLAUDE.md` carries the same three folder-relative — that is the
copy to run from inside one.

The third gate is the domain's own `{domain}_test.py`: it runs the engine laws
(`mdp_ir.laws`) and **parametrizes the differential over the covering set read
from the schema** — one node per declared instance and mixture, so a break
names the composition, and adding an instance extends the sweep with no edit
here. Each domain is ONE catalog schema (IR_LAYERING_PLAN §10): slots declare
candidate pools, instances select and override. The CLI form
(`python -m mdp_ir.differential <schema> --all-instances --episodes 40`) runs
the same sweep in one process, and each schema's `mdp.invariants` are enforced
by both the differential and the laws gate.

## Adding and removing an entry

Admission criteria live in `cases/README.md`. Either source — a finished
project from a research repo, or a `cases/` folder — moves **whole**, campaign
record included, since the in-folder `PLAYBOOK.md` must be reachable by the
installed skill. Move the folder, never copy: two folders with the same
`{domain}_*` modules break pytest's import mode. Then update the tables above
and bump the plugin version.

Demotion is the same move in reverse and is not a failure verdict — it means
the entry no longer earns its slot as a *few-shot exemplar*. `dynamic_pricing`
went back to `cases/` on 2026-09-01: its continuous single-entity control is
the shape `inv_single` already teaches, and the set is better at three.

**The freeze is scoped.** Code, schema and tests take **no research edits** —
but they do change when pipeline or spec work requires it, through the entry's
author. Campaign-record docs are maintainer-curated. Every change here ships
as a plugin version bump, so "what shipped" stays answerable from the tag.

## Read with these caveats

- `mab`'s **selection machinery predates v0.7.0**: its train script uses a live
  in-training `SelectionEvalCallback`, where §8.6/§9.7 now specify a post-hoc
  three-layer screen. Take the selection pattern from the spec, not from this
  script. The code stays because every number in the folder was produced by it;
  the `scripts.selection_protocol` check WARNs on it.
- `inv_single` **declares no `research_questions` stance**, correctly: it ships
  no policy probe and no `INTERPRET.md`, so it makes no tier-2 claim. `mab` and
  `game2048` are where the §14 interpret leg is exemplified.
- `game2048` was promoted with **declared debt outstanding** — §14.3's figure
  contract is only partly met (no `game2048_plot_policy.py`) and its `discover`
  stance's fitted rule was not delivered. Read that as "not met today", not
  "cannot be": a spec-contract deliverable still lands through the author. Its
  `ESCALATION.md` is also a **32-of-85 subset**, so some `#E` citations do not
  resolve, and its `PLAYBOOK.md` carries no lever entries.

## Shape coverage

Covered: vector-state single-entity control, two-step advance, episode-support
demand, decision-conditioned generators, exact-DP benchmarks, cross-family
world mixtures (`inv_single`'s `mix_demand`); the design layer over a grid,
discrete actions, censored information, equivariant state, a decision that
selects which exogenous stream is read, symbolic size axes, and the §14
interpret leg end to end (`mab`); spatial board observations with CNN
extractors, action masking, a state-dependent categorical support,
uncertainty stages keyed on something other than `period`, domain expression
builtins, and the no-DP/unbounded-state shape (`game2048`).

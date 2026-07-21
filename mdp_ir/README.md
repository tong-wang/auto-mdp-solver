# mdp_ir

Pydantic schema for the **MDP intermediate representation** (v0.4) — the
Phase-A artifact of the MDP agent (`MDP_AGENT_PLAN.md` §3). An IR is a
machine-checkable JSON description of a dynamic decision problem that the user
confirms *before* any domain code is generated; it then drives templated
Phase-B codegen.

`MDP_IR_SAMPLE.md` (repo root) is the annotated reference instance and explains
every field; this package is its executable form.

## Validate

```bash
python -m mdp_ir examples/inv_single/inv_single_schema.json examples/dynamic_pricing/vanryzin_pricing_schema.json
```

```python
from mdp_ir import load_ir
ir = load_ir("examples/inv_single/inv_single_schema.json")
ir.mdp_fingerprint()   # freeze token of the problem block
ir.unconfirmed()       # Confirmable fields not yet human-reviewed
```

## Execute (the IR interpreter)

`interpreter.py` runs an IR **directly** — dynamics exprs, derived seed keys,
objective, reward, termination — with no generated code. It produces the
Phase-A round-trip trajectories the user confirms, and later serves as the
Stage-1 **differential oracle**: generated `_mdp` code must reproduce its
trajectories on shared `(instance, episode_seed, decisions)`.

```bash
python -m mdp_ir.interpreter examples/inv_single/inv_single_schema.json --decision order=40
python -m mdp_ir.interpreter examples/dynamic_pricing/vanryzin_pricing_schema.json \
    --decision price=1.0 --instance ample_stock --episode-seed 3
python -m mdp_ir.interpreter_test        # invariant checks on both examples
```

```python
from mdp_ir import simulate
traj = simulate(ir, episode_seed=0, decisions={"order": 40})   # or a callable / per-period list
print(traj.render())
```

Decisions not fixed are drawn uniformly within their declared bounds,
deterministically from the episode seed. Draws are keyed by the schema-derived
seed keys — `[entity_id?, sub_stream?, stream_id, period?, episode_seed,
seed_salt]`, matching the reference generators (n.b. spec §6.3's snippet shows
`period` before `stream_id`; the actual generator code puts the stream first,
and the IR follows the code) — so trajectories are reproducible and realized
uncertainty is decision-path independent by construction
(`interpreter_test.py` asserts this, plus conservation and termination
properties, on both examples). Supported distribution families:
`episode_categorical`, `categorical`, `poisson`, `normal`, `lognormal`
(extension 2026-07-15, retailer), `uniform`, `bernoulli`.

## Diff against real code (the differential runner)

`differential.py` is the Stage-1 gate mechanism: it replays identical
`(instance, episode_seed, decisions)` through the interpreter **and** a
domain's `init_state`/`advance` implementation, then diffs the trajectories
field-for-field. A simulator that is spec-conformant but models the wrong
problem fails here.

```bash
python -m mdp_ir.differential examples/inv_single/inv_single_schema.json --episodes 20
# MATCH  inv_single: 20 episodes, 600 periods, fields=[...14 fields...]
python -m mdp_ir.differential examples/inv_single/inv_single_schema.json --instance lost_sales
```

A **domain adapter** (`{ir.domain.name}_ir_adapter.py` in the IR's own
directory, discovered by `load_adapter_factory`; the module exposes
`make_adapter(ir, instance=None, seed_salt=0, domain_dir=None)`) maps the domain's
API onto interpreter-row field names and builds its scenario object *from the
IR's scenario constants* — one source of truth, exactly as generated code
will. Consequence: the differential validates **logic** (dynamics, seeding,
economics exprs — a corrupted update or cost expr diverges immediately), not
constant transcription, since both sides read the same constants. The shipped
`inv_single` adapter matches the handwritten domain **bit-exactly** across
base and `lost_sales` instances under random policies; that exactness is what
makes the gate meaningful for generated `_mdp` code.

## The three layers

The root is `{domain, mdp, gym, rl, assumptions_log}`. The blocks mirror the
codebase's layering (spec §1.1) and the Phase-B stages, and differ in owner,
lifecycle, and *oracle*:

| Block | Contents | Oracle | Lifecycle | Drives |
|---|---|---|---|---|
| `mdp` | horizon, entities, state/info, decisions, uncertainty, dynamics, objective components, scenario constants, initial state | **none** → human confirmation (`Confirmable`) | frozen at the Phase-A gate (`mdp_fingerprint()`) | Stage 1: `_uncertainty` / `_scenarios` / `_mdp` |
| `gym` | observation / action / **reward** mode menus, termination semantics | empirical (train and compare) | mutable design axes | Stage 2: `_gym` |
| `rl` | `requires_memory`, algo, frame-stack, obs-normalization decision (spec §8.3), net-arch hint | empirical | mutable design axes | Stage 4: training config |

Dependencies point strictly downward; root-level validators enforce each edge
(gym features reference mdp names, latent state appears in no mode, rl memory
machinery matches the mdp-derived heuristic, masking requires a discrete
default action mode).

## What the schema enforces

- **Identifier resolution** — every name in any expression (dynamics updates,
  objective/feature/reward exprs, transforms, guards, triggers, distribution
  settings, initial state) must resolve to a declared state var, info field,
  decision, scenario constant, source/stage, or builtin. Typos and phantom
  names fail validation, not codegen.
- **Scenario constants are the single home for numbers** — distributions,
  objective exprs, dynamics, and `initial_state` reference them by name; the
  value namespace is collision-checked. `axis` tags mark scenario dimensions;
  `instances` (constant overrides by name) become the `SCENARIOS` registry.
- **Derived seed keys** — `UncertaintyStage.seed_key()` computes the spec-§6.3
  key from `realization` + `entity_id_in_seed` + stream ids; it is never
  hand-written. `event` stages require a `trigger`; `stream_id=0` is reserved
  for scenario sampling.
- **State vs info** — borderline calls carry a `Confirmable` `placement` that
  must match its containing list; economics live in `objective`
  (a `decomposition` info field must carry exactly those components + `total`).
- **Observability** — a `latent` state variable may be referenced by **no**
  observation mode.
- **Menus** — exactly one default each of observation / action / reward modes;
  action modes encode a declared decision; `mask` requires discrete,
  `reparametrize` requires a `transform`.
- **Gymnasium termination** — finite `T` that is part of the problem ends
  episodes as `terminated` (no bootstrapping past it); `truncated` is only for
  external limits. Optional absorbing-state `early_terminated_when` expr.
- **rl consistency** — `requires_memory.suggested` must equal the derived
  latent-correlation heuristic; memory machinery (recurrent / frame-stack)
  present iff `requires_memory.value`.

## Confirmable&lt;T&gt;

Agent-proposed-but-human-owned fields (`requires_memory`, decision
`type`/`bounds`, borderline `placement`) use `Confirmable[T]`:

```jsonc
{ "value": <T>, "suggested": <T>, "source": "derived" | "human_confirmed" | "human_override", "rationale": "…" }
```

`derived` = proposed, not yet reviewed; `human_confirmed` = reviewed, proposal
kept; `human_override` = reviewed, changed (requires `value != suggested`).
The Phase-A gate requires `ir.unconfirmed()` to be empty before the mdp block
is frozen.

## Scope notes (v1)

Finite horizon only (no `horizon.kind`), but `horizon.T` may name a scenario
constant so the horizon varies per instance (resolve with
`MdpBlock.horizon_T(instance)`); single scalar `[lo, hi]` bounds per
decision (no per-dim bounds); observation features are deterministic exprs of
true values (noisy sensor channels are unrepresentable).

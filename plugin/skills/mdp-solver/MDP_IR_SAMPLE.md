# MDP-IR — Sample Instance (v0.4)

A worked example of the Phase-A intermediate representation (see `SKILL.md`
Phase A). It is instantiated for the **existing** `inv_single`
domain so every field can be checked against real code in `inv_single/`. This
is what the agent would produce from a text description *before writing any
code*, and what the user confirms at the Phase-A gate.

The IR is JSON (a pydantic model at runtime, `mdp_ir/schema.py`). Comments
below the block explain the design intent of each section.

## 0. The three layers

The root is `{domain, mdp, gym, rl, assumptions_log}` — three blocks that
mirror the codebase's own layering (spec §1.1) and differ in **owner,
lifecycle, and oracle**:

| Block | What it is | Oracle | Lifecycle |
|---|---|---|---|
| `mdp` | the *problem*: state, decisions, uncertainty, dynamics, economics, scenario constants | none → human confirmation | **frozen** at the Phase-A gate; `mdp_fingerprint()` is the freeze token |
| `gym` | the *interface menus*: observation / action / reward modes, termination | empirical (train & compare) | mutable Phase-B design axis |
| `rl` | the *training config*: memory/policy class, algo, obs-norm decision | empirical | mutable Phase-B design axis |

Dependencies point strictly downward (gym → mdp, rl → gym/mdp); root
validators enforce each edge. `Confirmable[T]` fields live (almost) entirely
in `mdp` — that is exactly where no oracle exists. Editing `gym`/`rl` does not
change the fingerprint and needs no re-confirmation.

---

## 1. The IR (filled instance)

```jsonc
{
  "ir_version": "0.4",
  "seed_scheme": "v2",                   // spec §6.3 seed tree; canonical for new domains

  "domain": {
    "name": "inv_single",
    "class_prefix": "InvSingle",
    "one_line": "Single-echelon periodic-review inventory with stochastic lead time and backlogging.",
    "problem_class": "inventory_control"
  },

  // ════════════════════════ mdp — the problem (frozen) ════════════════════════
  "mdp": {

    "horizon": { "T": 30, "period_indexing": "0-based" },   // v1 = finite only
    // T may instead name a scenario constant (e.g. "T": "n_periods") so the
    // horizon varies per instance like any other scenario attribute; resolve
    // with MdpBlock.horizon_T(instance).

    "entity_structure": {
      "kind": "single",
      "entity_id_in_seed": false      // no entity dimension → derived seed keys omit entity_id
    },

    // --- STATE: minimal sufficient statistic for the next transition ONLY (§5.1) ---
    "state_variables": [
      {
        "name": "period", "role": "time_index", "type": "int",
        "bounds": [0, 30], "observability": "observable",
        "desc": "current period t; horizon ends when period == T"
      },
      {
        "name": "inventory", "role": "core", "type": "int",
        // scale-derived bounds are SYMBOLIC over the slot read-API (derived by
        // mdp_ir.families, §7) and resolve per selected demand candidate —
        // never freeze one candidate's scale into the structure
        "bounds": ["-40 * demand.mean", "40 * demand.mean"],
        "observability": "observable",
        "desc": "net on-hand inventory after all period events; read by next transition"
      },
      {
        "name": "pipeline", "role": "core", "type": "int_vector",
        "length": 6,                    // leadtime.max()+1  (max L = 5); shapes stay literal (frozen)
        "element_bounds": [0, "40 * demand.mean"],
        "observability": "observable",
        "desc": "pipeline[k] = quantity arriving at the k-th future R (receive) event"
      }
    ],

    // --- INFO: per-transition outputs; NEVER read back by the simulator (§5.1) ---
    "info_fields": [
      {
        "name": "demand", "type": "int", "desc": "realized demand this period (D event)",
        // borderline state-vs-info call (§5.1) → Confirmable<enum>, reviewed → human_confirmed
        "placement": {
          "value": "info", "suggested": "info", "source": "human_confirmed",
          "rationale": "no later transition reads it; exposed only via the vec_d observation mode"
        }
      },
      { "name": "received",     "type": "int", "desc": "units received this period (R event)" },
      { "name": "leadtime",     "type": "int", "desc": "sampled lead time of this period's order" },
      { "name": "lost_sales",   "type": "int", "desc": "unmet demand dropped (only if allow_backlog=false)" },
      { "name": "action_period","type": "int", "desc": "period the emitting decision was taken (boundary invariant)" },
      // must carry exactly the objective components + "total" (validated):
      { "name": "cost", "type": "decomposition",
        "components": ["holding", "shortage", "order_fixed", "order_variable", "total"] }
    ],

    "decisions": [
      {
        "name": "order",
        // Confirmable<enum> — integrality is a no-oracle call (§6); still `derived` here
        "type": {
          "value": "continuous", "suggested": "continuous", "source": "derived",
          "rationale": "'how much to order' stated no integrality; continuous is easier for PPO"
        },
        "dim": 1,
        // Confirmable<[lo,hi]> — the upper action scale is a codegen guess;
        // symbolic, so it re-resolves when a different demand candidate is selected
        "bounds": {
          "value": [0, "20 * demand.mean"], "suggested": [0, "20 * demand.mean"],
          "source": "derived",
          "rationale": "20x mean demand; caps the action scale, not a physical limit"
        },
        "feasibility": ["non_negative"],
        "desc": "replenishment quantity placed at the O (order) event"
      }
    ],

    // --- UNCERTAINTY: authored as SLOTS with a candidate pool (catalog ⊕
    //     selection, §7 below; IR_LAYERING_PLAN §10). A slot fixes the
    //     structural skeleton — name, interface, stream_id, stages (whose
    //     `realization` DERIVES the seed key, §5.3 — never hand-written) —
    //     and declares the candidate families that can fill it, the way
    //     _uncertainty.py declares generator classes. `load_ir` resolves the
    //     selected candidate into a plain `uncertainty_sources` entry, so
    //     nothing downstream sees the catalog. (A file carrying
    //     `uncertainty_sources` directly is the legacy resolved form and
    //     still loads.)
    "uncertainty_slots": [
      {
        "name": "demand", "interface": "DemandGenerator", "stream_id": 0,
        "latent": false,
        // one per-period stage → seed_key() = [period, source:0, 1, episode_seed, seed_salt]
        "stages": [ { "name": "sample", "realization": "period" } ],
        "default": "discrete",
        "candidates": {
          "discrete": {
            "generator": "DiscreteDemand",
            // `family` is any explicitly-aliased family (categorical, poisson,
            // normal, lognormal, uniform, bernoulli, choice_without_replacement,
            // normalized_uniform_weights) OR any numpy Generator scalar
            // distribution by name (gamma, beta, binomial, exponential, zipf, …),
            // whose `settings` are numpy's own kwargs. See interpreter._sample_family.
            "family": "categorical",
            "settings": {
              // a setting whose value is a DRAW SPEC is this slot's WORLD
              // LATENT (spec §5.2): drawn once per episode on the meta branch
              // at THIS SLOT's stream_id, then consumed as a plain constant.
              // The loader desugars it into a scenario sampler + a synthesized
              // placeholder constant ({slot}_{setting}, value = `example`).
              // hidden: true flips the requires_memory derivation and bars the
              // realized value from observation modes.
              "values": {
                "draw": { "family": "choice_without_replacement",
                          "settings": { "low": "demand_support_low",
                                        "high": "demand_support_high",
                                        "size": "demand_support_size" } },
                "example": [10, 20, 30, 40, 50], "hidden": true
              },
              "probabilities": {
                "draw": { "family": "normalized_uniform_weights",
                          "settings": { "size": "demand_support_size" } },
                "example": [0.2, 0.2, 0.2, 0.2, 0.2], "hidden": true
              }
            },
            "is_discrete": true
          },
          // a second candidate is ~10 lines appended HERE — selected per
          // instance ({"demand": "poisson"}) or --select demand=poisson —
          // never a second schema file. Symbolic bounds re-resolve to its
          // scale automatically (derived demand.mean: 30 either way here).
          "poisson": {
            "generator": "PoissonDemand", "family": "poisson",
            "settings": {
              "rate": { "draw": { "family": "gamma",
                                  "settings": { "shape": "demand_alpha",
                                                "scale": "1.0 / demand_beta" } },
                        "example": 30.0, "hidden": true }
            },
            "is_discrete": true
          }
        }
      },
      {
        "name": "leadtime", "interface": "LeadtimeGenerator", "stream_id": 1,
        "latent": false,
        // event: decision-triggered; trigger REQUIRED (validated)
        // → seed_key() = [period, source:1, 1, episode_seed, seed_salt]
        "stages": [ { "name": "draw", "realization": "event", "trigger": "O && order>0" } ],
        "default": "discrete",
        "candidates": {
          "discrete": {
            "generator": "DiscreteLeadtime", "family": "categorical",
            // plain settings (no draw spec) = no world latent for this slot
            "settings": { "values": "leadtime_values", "probabilities": "leadtime_probs" },
            "is_discrete": true
          }
        }
      }
    ],

    "dynamics": {
      "event_sequence": ["O", "R", "D"],    // order → receive → demand
      "observation_point": "pre-order",
      // every identifier in updates/guards must resolve (state ∪ info ∪ decisions
      // ∪ constants ∪ sources/stages ∪ locals like L) — validated, no phantom names
      "transitions": [
        { "event": "O", "guard": "order > 0",
          "updates": ["L ~ leadtime.draw", "leadtime = L", "pipeline[L] += order"] },
        { "event": "R",
          "updates": ["received = pipeline[0]", "inventory += received",
                       "pipeline = pipeline[1:] + [0]"] },
        { "event": "D",
          "updates": ["demand ~ demand.sample", "inventory -= demand",
                       "if not allow_backlog: lost_sales = max(0, -inventory); inventory = max(0, inventory)"] },
        { "event": "END_OF_PERIOD", "updates": ["action_period = period", "period += 1"] }
      ]
    },

    // --- OBJECTIVE: mdp-layer economics; travel in info (spec §6.4). Evaluated at
    //     END_OF_PERIOD on end-of-period state. Reward construction is gym's job.
    "objective": {
      "sense": "minimize",
      // β — the problem's intrinsic discount (time value of money /
      // continuation probability), from the Phase-A interview. Part of the
      // objective, never a solver knob: eval + all baselines score Σ β^t r_t,
      // training gamma defaults to β (spec §8.6), rewards are never
      // pre-discounted in the env. 1.0 = undiscounted (the default).
      "discount_factor": 1.0,
      "per_step_components": [
        { "name": "holding",        "expr": "h * max(0,  inventory)" },
        { "name": "shortage",       "expr": "b * max(0, -inventory) if allow_backlog else b * lost_sales" },
        { "name": "order_fixed",    "expr": "K if order > 0 else 0" },
        { "name": "order_variable", "expr": "c * order" }
      ]
    },

    // --- SCENARIO: the single home for every number. Exprs reference constants by
    //     name; `axis` tags scenario dimensions; `instances` → the SCENARIOS registry.
    "scenario": {
      "constants": [
        { "name": "h", "value": 0.2,  "axis": "cost", "desc": "holding cost / unit" },
        { "name": "b", "value": 2.0,  "axis": "cost", "desc": "shortage cost / unit" },
        { "name": "K", "value": 25.0, "axis": "cost", "desc": "fixed cost / order" },
        { "name": "c", "value": 1.0,  "axis": "cost", "desc": "variable cost / unit" },
        { "name": "allow_backlog", "value": true, "axis": "variant" },
        { "name": "demand_support_size", "value": 5,  "axis": "demand" },
        { "name": "demand_support_low",  "value": 10, "axis": "demand" },
        { "name": "demand_support_high", "value": 50, "axis": "demand" },
        { "name": "demand_alpha", "value": 9.0, "axis": "demand",
          "desc": "poisson candidate: Gamma shape of the hidden rate" },
        { "name": "demand_beta",  "value": 0.3, "axis": "demand",
          "desc": "poisson candidate: Gamma rate; E[lambda] = alpha/beta = 30" },
        { "name": "leadtime_values", "value": [2, 3, 4, 5], "axis": "leadtime" },
        { "name": "leadtime_probs",  "value": [0.125, 0.375, 0.375, 0.125], "axis": "leadtime" }
        // NOTE: no hand-authored placeholders and no `samplers` node — the
        // draw specs inside the demand candidates desugar at load time into a
        // `demand_latent` sampler (substream = the slot's stream_id) plus
        // synthesized placeholder constants (demand_values / demand_probabilities
        // / demand_rate). Instances sharing a candidate share the latent stream:
        // common random numbers across instances by construction.
      ],
      // instances = the SCENARIOS registry: constant overrides AND slot
      // selections compose freely — a permutation is one line, never a file.
      // At resolution, instances inconsistent with the active selection are
      // dropped and slot keys stripped, so the resolved IR is consistent.
      "instances": {
        "lost_sales":         { "allow_backlog": false },
        "poisson":            { "demand": "poisson" },
        "poisson_lost_sales": { "demand": "poisson", "allow_backlog": false }
      }
    },

    // literal or expr over constants; must cover every `core` state var (validated)
    "initial_state": { "inventory": 0, "pipeline": "zeros(6)" },

    // OPTIONAL: domain-owned expression builtins. Expressions may call only
    // the core builtins (min/max/exp/phi/topk/...); a domain needing more
    // declares each extra function here and ships `def {name}` in
    // `{module}.py` next to this IR (portable-domain contract). The
    // interpreter resolves the module lazily on first call — nothing
    // domain-specific is ever hard-coded in mdp_ir.
    "expr_builtins": [
      // { "name": "bayes_topk", "module": "mydomain_probit_map",
      //   "desc": "Bayes-optimal top-k selection via probit MAP" }
    ]
  },

  // ═══════════════ gym — interface menus (mutable design axes) ═══════════════
  "gym": {

    // the MENU of modes (§5.2); a latent var may appear in NO mode (validated)
    "observation_modes": [
      { "name": "vec", "default": true,
        "features": [ {"ref": "inventory"}, {"ref": "pipeline"} ] },
      { "name": "vec_d",
        "features": [ {"ref": "inventory"}, {"ref": "pipeline"}, {"ref": "info.demand"} ] },
      { "name": "vec_ip",
        "features": [ {"derived": "inventory_position", "expr": "inventory + sum(pipeline)"} ] }
    ],

    // encodings of the canonical decision(s), possibly several at once
    // (e.g. an order&allocations vector encoding);
    // strategy: clip (continuous), mask (discrete), reparametrize (needs transform)
    "action_modes": [
      { "name": "order", "default": true, "encodes": "order",
        "type": "continuous", "bounds": [0, "20 * demand.mean"],
        "transform": "", "feasibility_strategy": "clip",
        "desc": "identity: agent emits the order quantity directly" }
    ],

    // how reward is assembled from the objective components in info —
    // the mdp stays reward-agnostic; sense=minimize ⇒ negate
    "reward_modes": [
      { "name": "neg_cost", "default": true, "expr": "-total",
        "desc": "negated per-step cost.total" }
    ],

    // Gymnasium semantics: T is part of the problem ⇒ `terminated` (no
    // bootstrapping past it); `truncated` is only for external time limits.
    "termination": { "horizon_end": "terminated", "early_terminated_when": null }
  },

  // ═══════════════ rl — training configuration (Stage-4 input) ═══════════════
  "rl": {
    // Confirmable<bool>; `suggested` must equal the derived heuristic (validated):
    // a hidden world-latent sampler ⇒ suggested=true. Overridden to false here
    // because the realized demand distribution is fixed within an episode, so no
    // within-episode inference (memory) is needed — a worked human_override.
    "requires_memory": {
      "value": false, "suggested": true, "source": "human_override",
      "rationale": "the demand slot's latent draw is hidden ⇒ derivation suggests memory; overridden: the demand distribution is constant within an episode, nothing to infer step-to-step"
    },
    "algo": "ppo",                       // ppo | maskable_ppo | recurrent_ppo
    "frame_stack": 1,                    // >1 iff requires_memory (validated)
    // the spec-§8.3 decision, explicit instead of folklore:
    "obs_normalization": {
      "enabled": true,
      "rationale": "inventory and pipeline are stationary features of heterogeneous magnitude (scale set by the selected demand candidate)"
    }
  },

  "assumptions_log": [
    "Order quantity modeled as continuous (float) and clipped to [0, 20*demand.mean]; integrality unstated.",
    "Initial inventory and pipeline assumed empty (0).",
    "Unmet demand backlogged; the lost-sales variant is the 'lost_sales' scenario instance.",
    "Cost components evaluated at END_OF_PERIOD on end-of-period inventory.",
    "`demand` placed in info (not state): no later transition reads it.",
    "Period indexing 0-based; T=30; horizon end is `terminated`, not `truncated`.",
    "Seed keys derived from stage `realization` (spec §6.3)."
  ]
}
```

---

## 2. Round-trip confirmation artifacts (what the user actually reviews)

The raw IR is for the machine. Per `SKILL.md` Phase A, before freezing the `mdp` block
the agent renders two human-readable views and asks the user to confirm.

### 2a. Plain-English restatement

> Each period (30 total) you manage a single stock point. You first **place a
> replenishment order** of any non-negative size. That order arrives after a
> random lead time of 2–5 periods (most often 3 or 4). Any shipment scheduled to
> arrive this period is **received** into inventory, then **demand** hits —
> drawn from a 5-value distribution (values in 10–50) that is re-drawn fresh at
> the start of each episode. Unmet demand is **backlogged** (inventory goes
> negative). At period end you pay: holding 0.2/unit of positive stock, shortage
> 2.0/unit of backlog, a fixed 25 per order placed, and 1.0/unit ordered. The
> goal is to **minimize total cost** over the 30 periods.

### 2b. Sample trajectory

Real output of `python -m mdp_ir.interpreter inv_single/inv_single_schema.json
--decision order=40` — the IR's `dynamics` exprs **executed directly by the
restricted interpreter** (`mdp_ir/interpreter.py`) under a fixed
`episode_seed`; no generated code exists yet at this point:

```
t  order  demand  received  leadtime  inventory               pipeline  holding  shortage  order_fixed  order_variable   total   reward
0  40.00      43         0         3        -43      [0,0,40.00,0,0,0]     0.00     86.00        25.00           40.00  151.00  -151.00
1  40.00      22         0         5        -65  [0,40.00,0,0,40.00,0]     0.00    130.00        25.00           40.00  195.00  -195.00
2  40.00      32         0         4        -97  [40.00,0,0,80.00,0,0]     0.00    194.00        25.00           40.00  259.00  -259.00
3  40.00      22     40.00         3     -79.00     [0,0,120.00,0,0,0]     0.00    158.00        25.00           40.00  223.00  -223.00
...
episode total = 5698.00   reward total = -5698.00   periods = 30
```

The same interpreter is reused in Stage 1 as the **differential-test oracle**
(`mdp_ir/differential.py`): a domain's `_mdp` code must reproduce interpreter
trajectories field-for-field on shared `(instance, episode_seed, decisions)` —
the direct attack on the core risk that "formalization correctness has no
oracle". This is not hypothetical: the runner already diffs this IR against
the handwritten `inv_single/` domain and matches **bit-exactly** (all state,
info, and cost fields) across base and `lost_sales` instances under random
policies:

```
$ python -m mdp_ir.differential inv_single/inv_single_schema.json --episodes 20
MATCH  inv_single: 20 episodes, 600 periods, fields=[..., 'demand', 'holding',
       'inventory', 'leadtime', 'pipeline', 'shortage', ..., 'total']
```

**Gate to Phase B:** user confirms 2a + 2b; the IR validates; `unconfirmed()`
is empty (every `Confirmable` reviewed → `human_confirmed` / `human_override`);
the `mdp` block is frozen and `mdp_fingerprint()` recorded.

---

## 3. Notes on how the IR drives codegen (Phase B)

| IR field | Drives |
|---|---|
| `mdp.uncertainty_sources[].generator` + `distribution` | `_uncertainty.py` generator classes; `stream_id` → `_STREAM` constant |
| `mdp.uncertainty_sources[].stages[].realization` (+ entity structure) | the **derived** `SeedSequence` key per stage — `UncertaintyStage.seed_key()`, §5.3 |
| `mdp.scenario.constants` / `instances` | `{Domain}Scenario` fields and the `SCENARIOS` registry in `_scenarios.py` |
| `mdp.state_variables` vs `info_fields` | the state dataclass vs the `info` dict returned by `advance()` (§5.1) |
| `mdp.dynamics.transitions` | the body of `advance()` |
| `mdp.objective.per_step_components` | the `info["cost"]` dict in `_mdp` |
| `mdp.horizon.T`, `period_indexing` | termination check, period counter |
| `gym.observation_modes` (+ latent tags) | the gym's `observation_mode`s (§5.2) |
| `gym.action_modes` (type, strategy, transform) | action space + constraint handling: mask (spec §7.1) / clip / reparametrize (spec §7.2) |
| `gym.reward_modes` | reward assembly from `info` in `_gym` (mdp stays reward-agnostic, spec §6.4) |
| `gym.termination` | Gymnasium `terminated` / `truncated` flags |
| `rl.requires_memory`, `algo`, `frame_stack` | policy class: `PPO` / `MaskablePPO` / `RecurrentPPO`, frame-stacking |
| `rl.obs_normalization` | the `VecNormalize` decision (spec §8.3) |

---

## 4. Latent / partial-observability variant (POMDP path)

`inv_single` is fully observed. To show the POMDP path concretely, suppose demand
follows a **hidden two-state regime** (high/low) fixed for the episode and never
directly observed. Only these deltas from §1:

```jsonc
// 1) mdp: a fundamentally-hidden state variable
"state_variables": [
  /* ...period, inventory, pipeline... */
  {
    "name": "demand_regime", "role": "core", "type": "categorical",
    "categories": ["low", "high"],
    "observability": "latent",        // appears in NO gym mode (validated)
    "desc": "hidden demand regime, constant within an episode"
  }
],

// 2) mdp: an episode-level latent source that fixes the regime
"uncertainty_sources": [
  /* ...demand, leadtime... */
  {
    "name": "regime", "generator": "RegimeSampler", "stream_id": 3, "latent": true,
    "distribution": { "family": "bernoulli", "settings": { "p_high": 0.5 } },
    "stages": [ { "name": "draw", "realization": "episode" } ]
    // seed_key() = [stream:3, episode_seed, seed_salt] — period-free, derived
  }
],

// 3) rl: suggested flips BECAUSE a latent source realizes at episode level;
//    memory machinery must follow (validated)
"rl": {
  "requires_memory": {
    "value": true, "suggested": true, "source": "human_confirmed",
    "rationale": "latent 'regime' source realizes at episode level (temporally correlated)"
  },
  "algo": "recurrent_ppo",
  "frame_stack": 1,
  "obs_normalization": { "enabled": true, "rationale": "…" }
}
```

The layer split shows its value here: the *problem fact* (a latent,
episode-correlated source) lives in `mdp`; its *consequences* (recurrent
policy, heuristic-only baselines) live in `rl` and are derived + validated,
not hand-synchronized.

---

## 5. The three tricky modeling calls, and how the schema handles them

### 5.1 State vs info
`state_variables` carries **only the minimal sufficient statistic** — a variable
is in `state` iff *some later transition reads it given the decision*; everything
else (realized demand, receipts, costs, diagnostics) is an `info_field`.
This is a **modeling call with no oracle**, so borderline assignments carry a
`Confirmable` `placement` surfaced in the round-trip confirmation. Economics
have one home — `objective.per_step_components` — and a `decomposition` info
field must mirror exactly those components (validated), so dynamics never
compute costs.

### 5.2 Observability is a gym decision — a *menu*, not a per-variable boolean
Two distinct concepts, now in two different *layers*:
- **Latency** (`observability: latent` on an mdp state var / `latent: true` on a
  source) — *fundamentally hidden from any agent*. A problem fact; drives
  `rl.requires_memory` and baseline tractability; may appear in **no** mode.
- **Mode selection** (`gym.observation_modes`) — among the *observable*
  quantities, which subset a given mode exposes. An interface choice (cf.
  `inv_single`'s `vec`/`vec_d`/`vec_ip` modes).

Features reference a **state var**, an **info field** (`info.demand`), or a
**derived expression** (`inventory_position`). Validation: refs resolve, exprs
resolve, and no mode touches anything latent.

### 5.3 Uncertainty realization timing drives the seed key — derived, not written
Each source is a list of **stages** with `realization` ∈ `{period, event,
keyed}` (plus `episode`, v1-only — see below):
- `period` → drawn every period; key includes `period`.
- `event` → decision-triggered (`trigger` required, e.g. leadtime at `O` when
  `order > 0`); period-keyed, event-gated.
- `episode` → drawn once; key omits `period`. **v1 only**: under v2 an
  episode-level latent is a **scenario sampler** (world layer, §6.3 / the
  `paper_demand` sampler above), not an intrinsic stage — the schema rejects
  episode-realization stages in v2.

The key itself is **computed** by `UncertaintyStage.seed_key()` from
`realization` + `entity_id_in_seed` + `stream_id`/`sub_stream` — there is no
hand-written `seed_keys` field to drift out of sync. **Slot order depends on
`seed_scheme`.** v2 (spec §6.3, canonical for new domains) keys the intrinsic
branch `[entity_id?, sub_stream?, period?, stream_id, 1, episode_seed,
seed_salt]` — `period` *below* the source, branch word `1` above it. v1
(frozen legacy) instead keys `stream_id` *before* `period`. Generated code
must match **its own scheme**; the differential runner proves the interpreter
reproduces it bit-exactly, and the conformance harness's **determinism** and
**decision-path-independence** tests validate the generated code honors it.

---

## 6. The `Confirmable<T>` convention (no-oracle fields)

Several IR fields are *agent-proposed but human-owned*: the agent derives a
sensible default, but no compiler or oracle can check it. The schema wraps
every such field in one reusable shape:

```jsonc
Confirmable<T> = {
  "value":     <T>,                     // what Phase B actually uses
  "suggested": <T>,                     // the agent's derived default
  "source":    "derived" | "human_confirmed" | "human_override",
  "rationale": "<why this value>"
}
```

**Invariants (schema-enforced):** `human_override` ⟺ `value != suggested`;
`derived` and `human_confirmed` both require `value == suggested` and differ
only in whether the human has *reviewed* the proposal. The Phase-A walkthrough
targets exactly `ir.unconfirmed()` — the fields still at `derived` — and the
gate requires that list to be empty, so "confirmed" is recorded in the
artifact, not implied by conversation history. Overrides are echoed into
`assumptions_log`.

| Field | T | Why it's no-oracle |
|---|---|---|
| `rl.requires_memory` | bool | latent-correlation heuristic; frame-stack vs recurrent is judgment (§5.3) |
| `mdp.decisions[].type` | enum | integrality rarely stated in the prompt |
| `mdp.decisions[].bounds` | [lo,hi] | action *scale* is a codegen guess, not a physical limit |
| `placement` (state/info vars) | enum | the state-vs-info boundary is a modeling call (§5.1); tag only borderline vars |

Note the pattern: `Confirmable` lives in the **mdp block** (plus
`requires_memory`, whose *derivation* reads mdp facts) — the no-oracle layer.
Gym/rl choices have an empirical oracle (train and measure), so they are plain
fields: experiment axes, not judgment calls.

---

## 6. Scenario-redesign additions (seed scheme v2)

Added 2026-07-22 (spec §5, §6.3). All fields are
additive — every pre-existing IR validates unchanged as `seed_scheme: "v1"`
with bit-identical draws.

- **`seed_scheme`** (root, default `"v1"`): `"v2"` selects the seed-tree
  grammar — intrinsic keys `[(draw,) period, source_id, 1, episode_seed,
  seed_salt]`, meta keys `[substream_id, 0, episode_seed, seed_salt]` — and
  **forbids episode-realization stages** (treatment A): world latents must be
  `scenario.samplers`. Under v2, `stream_id` is the per-instance source id
  (may be 0) and must be distinct across sources.
- **`mdp.scenario.samplers`** — world latents (spec §5.2), realized once per
  episode on the meta branch BEFORE any period draw; draws execute in order
  from one rng seeded by the sampler's meta key. `hidden: true` flips the
  `requires_memory` derivation and bars the realized constants from
  observation exprs. **In the catalog form these are never authored**: a draw
  spec inside a candidate's settings (§1) desugars at load time into one
  sampler per slot — named `{slot}_latent`, `substream_id` = the slot's
  `stream_id` (one stream identity per source of randomness, both branches),
  draws in settings-declaration order, targets synthesized as
  `{slot}_{setting}` placeholder constants carrying the spec's `example`.
  The per-period source then consumes them as plain constants, and every
  expr/validator sees one namespace. (A legacy resolved file may still carry
  the node explicitly.)
- **`mdp.scenario.mixtures`** — world mixtures (spec §5.3): weighted
  components naming instances (`""` = base), drawn once per episode on the
  mixture's own substream; a mixture name is usable anywhere an instance
  name is (`--instance`, gates). Weights are a modeling commitment.
- **`grids`** (root, design layer — spec §5.5/§5.6): generality targets for
  generalist training, `{ name, base_instance, axes: {constant: [values]} }`;
  `ScenarioGrid.cells()` yields `(cell_id, overrides)` row-major with
  axis-derived ids. Grids never appear inside the `scenario` node.
- **Differential**: `--seed-salt` defaults to 1 (v2 domains assert
  `seed_salt >= 1`); sampler-bearing instances diff bit-exactly — the
  adapter builds the domain's sampler on the IR sampler's substream
  (`examples/inv_single/inv_single_ir_adapter.py` is the reference).

## 7. Catalog ⊕ selection (one schema file, a pool of compositions)

Added 2026-07-28 (IR_LAYERING_PLAN §10). The authored IR is **one**
`{domain}_schema.json` whose slots each carry a `candidates` pool + a
`default` (§1 above); a file carrying `uncertainty_sources` directly is the
legacy resolved form and still loads unchanged.

- **Selection** = `{slot: candidate}`: slot defaults ⊕ the named instance's
  slot-valued keys ⊕ an explicit `--select slot=candidate` /
  `load_ir(path, instance=…, select=…)`. Declarations scale per slot;
  compositions are references — the slot×family cross-product is never
  enumerated anywhere, and `poisson × lost_sales` is the one-line instance
  `{"demand": "poisson", "allow_backlog": false}`.
- **Derived read-API**: symbolic bounds (`"20 * demand.mean"`) resolve
  against `mean`/`max`/`is_discrete` computed by `mdp_ir.families` from the
  selected candidate's family + settings, composing through the latent
  hierarchy (a Gamma-latent Poisson gets `max = envelope(poisson,
  envelope(gamma))`). Derivation is lazy per attribute — a slot nothing
  references may use state-dependent settings (dynamic_pricing's
  price-dependent rate). Escape hatch: an explicit `read_api` block on the
  candidate.
- **What is frozen**: `layering.structural_fingerprint()` — the structural
  core plus the constant *names* its expressions reference (computed, never
  hand-tagged). Constant values, candidates, instances, mixtures are free,
  hash-tracked data: appending a candidate or editing a value never voids
  Phase-A confirmation. `mdp_fingerprint()` still fingerprints one resolved
  composition (`MdpIR.selection` records which).
- **Verification**: every *composition you run* is differentially verified
  on demand; `python -m mdp_ir.differential {schema} --all-instances` sweeps
  base + every named instance (the covering set — each candidate exercised
  at least once). `python -m mdp_ir {schema}` validates base + every
  instance's resolution.
- **Python mirror** (the layer rule): candidates ↔ `_uncertainty.py`
  generator classes; instances ↔ `SCENARIOS` registry entries; mixtures ↔
  the mixture combinator; slot `stream_id` ↔ generator `source_id`.

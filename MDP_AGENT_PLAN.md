# MDP Agent — Design Plan

> **Status: background design rationale — not the runbook.** The executable
> two-phase pipeline now lives in the `mdp-solver` skill
> (`.claude/skills/mdp-solver/SKILL.md`), which is self-contained and
> authoritative for *how to run it*. This document is retained for the design
> rationale behind the architecture and as the blueprint for later packaging
> the skill into a standalone agent (see §6, §8). Where this doc and the skill
> differ on operational detail, the skill wins; `MDP_PROJECT_SPEC.md` remains
> canonical for domain conventions.

A general-purpose agent that takes a **verbal/text description of a dynamic
decision-making problem** and produces a **trained, deployable RL policy** (a
Python file wrapping the trained model). The agent generates a spec-conformant
domain (`_uncertainty` / `_scenarios` / `_mdp` / `_gym`), baselines, and a tuned
RL model, following `MDP_PROJECT_SPEC.md`.

---

## 1. Scope (v1)

In scope:
- **Finite-horizon** problems.
- **Low-dimensional** state/action spaces.
- **Single-agent.**
- **Latent state / partial observability is allowed.** The `_mdp` layer already
  represents the *full simulator* state; the gym's `observation_mode` decides
  what the agent sees. Latent variables are simply state fields the gym does not
  expose. Partial observability affects the *policy class* (may need memory), not
  the simulator layering.

Out of scope for v1: high-dimensional / image observations, genuine multi-agent,
infinite-horizon average-reward formulations.

### Partial-observability implications
- The IR (§3) tags every state variable **observable** vs **latent**.
- When latent state is temporally correlated, the agent needs memory:
  **frame-stacking** (`FrameStackObservation`, already used in `cnv`) for short
  memory, or **`RecurrentPPO`** (`sb3_contrib`) for longer/again-belief-state
  cases. The IR carries a `requires_memory` flag to drive this choice.
- Optimal baselines under partial observability require a belief-MDP and are
  often intractable — baselines degrade gracefully to heuristics (§Phase B/3).

---

## 2. Two-phase architecture

The pipeline splits at the natural human/compute boundary:

```
PHASE A — Interactive formalization (human in the loop, cheap)
    text description ──► MDP-IR ──► round-trip confirm ──► frozen IR

PHASE B — Automated build (hands-off, compute-intensive)
    IR ─► model layers ─► gym ─► baselines ─► train+tune+eval ─► deployable .py
          └gate┘         └gate┘  └gate┘        └gate┘
```

- **Phase A** is conversational: extract structure, surface assumptions, confirm
  with the user. Ends with a frozen, machine-checkable IR.
- **Phase B** is an orchestrated pipeline of stages, each with a **verify-and-
  repair loop** and a **gate** that must pass before the next stage runs. Bounded
  retry budgets; checkpoint between stages so a late failure doesn't redo early
  work.

---

## 3. Phase A — Interactive formalization → MDP-IR

The LLM does **not** jump to code. It populates a formal **intermediate
representation** (pydantic/JSON schema). The IR shrinks codegen degrees of
freedom, is machine-checkable, drives templated scaffolding, and is the artifact
the user confirms.

### IR structure: three layers

The IR root is `{domain, mdp, gym, rl, assumptions_log}` (schema in `mdp_ir/`,
annotated instance in `MDP_IR_SAMPLE.md`). The blocks mirror the codebase
layering and differ in **oracle and lifecycle**:

- **`mdp` — the problem** (no oracle → human confirmation; **frozen** at the
  Phase-A gate, `mdp_fingerprint()` is the freeze token). State variables
  (observable | latent), canonical decisions (type/bounds/feasibility),
  uncertainty sources (distribution + stages whose `realization` *derives* the
  spec-§6.3 seed key, each tagged `timing: per_transition | episode_start` —
  an `episode_start` source with no per-transition sibling is **problem-instance
  selection** and compiles to a `ScenarioSampler` + concrete-instance scenario,
  never to an `_uncertainty` generator, per spec §4.3; `dynamics: deterministic`
  with an empty source list is a valid, expressible state and suppresses
  `_uncertainty.py` generation), dynamics (event sequence + update exprs), per-step
  objective decomposition (reward-agnostic, spec §6.4), horizon, entity
  structure, **scenario constants** (the single home for every number; `axis`
  tags + `instances` drive `_scenarios.py`), initial state. Judgment calls use
  `Confirmable<T>` (`derived` → `human_confirmed`/`human_override`).
- **`gym` — the interface menus** (empirical oracle; mutable Phase-B design
  axes). Observation modes, action modes (encoding + mask/clip/reparametrize
  strategy), **reward modes** (assembled from objective components), and
  Gymnasium termination semantics (`terminated` at T vs `truncated`).
- **`rl` — the training config** (empirical oracle; mutable). `requires_memory`
  (suggested from latent+episode-correlated sources, validated), algo
  (`PPO`/`MaskablePPO`/`RecurrentPPO`), frame-stack, and the explicit
  obs-normalization decision (spec §8.3).

Cross-layer validators enforce downward-only references (gym → mdp, rl →
gym/mdp), identifier resolution in every expression, and latent state appearing
in no observation mode.

### Round-trip confirmation (the correctness checkpoint)
Formalization has **no compiler / no oracle** — dynamics can be plausibly but
wrongly modeled. Before any code is written, the agent:
1. Restates the IR back in plain English — written to a co-located
   `{name}.restatement.md` **and printed in full in the confirmation
   message** (a confirmation request must never reference text the user has
   not been shown) — **including an explicit randomness
   classification**: "the dynamics are deterministic / stochastic via ⟨sources⟩;
   the randomness in your description is problem-instance selection (which
   puzzle/config an episode poses), not uncertainty in the dynamics" — or the
   converse. Verbal descriptions almost always carry this ambiguity ("a random
   Sudoku puzzle", "demand is uncertain"), and it is cheap to confirm here but
   expensive to discover after Stage 1 has built the wrong layering (spec §4.3).
2. Generates a few **human-readable sample trajectories** ("period 0: demand=7,
   ordered 5, backlog 2, cost …") by **executing the IR's `dynamics` exprs with
   a restricted interpreter** (no generated code exists yet). The same
   interpreter is reused in Stage 1 as a differential-test oracle against the
   generated `_mdp` (same seeds → same trajectories), directly attacking
   challenge #1.
3. Asks the user to confirm.
4. Records every default-filled gap in an **assumptions log**.

**Gate to Phase B:** user confirms; IR validates; `unconfirmed()` is empty
(every `Confirmable` reviewed); the `mdp` block is frozen and its fingerprint
recorded. `gym`/`rl` blocks stay revisable in Phase B without re-confirmation.

---

## 4. Phase B — Automated build

### Stage 1 — Model layers + conformance gate
Generate in dependency order (spec §1.1) **from the IR's `mdp` block**:
`_exceptions` → `_uncertainty` (omitted when the IR declares deterministic
dynamics, spec §4.3) → `_scenarios` (from `scenario.constants` + `instances`)
→ `_mdp`. Few-shot from existing domains (owmr = multi-entity / two-step
reference; fnv/cnv = single-entity; 2048 = discrete + masking; sudoku =
deterministic dynamics + scenario-sampled problem instances).

**Gate = the conformance harness** (see §5) **+ differential test against the
IR interpreter** (same `(scenario, episode_seed)` → same trajectory). Harness
checks:
- **Static**: imports, type-checks, acyclic layering (`_uncertainty` imports no
  domain code), naming conventions, no `param`/`params`, no reward in `_mdp`.
- **Purity**: `advance()` does not mutate input `state` (deep compare).
- **Determinism**: same `(scenario, episode_seed)` → identical trajectory.
- **Decision-path independence**: uncertainty realized at period *t* is
  independent of actions at *t' < t* (validates reverse-tree seeding, §6.3 —
  easy to get subtly wrong; weight heavily).
- **Randomness placement** (spec §4.3): no episode-keyed generator in
  `_uncertainty` — a generator whose draw varies with `episode_seed` but not
  with `period` is scenario sampling in disguise and must be a
  `ScenarioSampler`.
- **`state`/`info` boundary**: `action_period` invariant; economics in `info`.
- **Well-posedness**: N random rollouts → finite, non-NaN, bounded costs.

### Stage 2 — Gym wrapper + gate
Template-driven from the IR's `gym` block: `observation_mode`s (respecting
observable vs latent tagging), action modes with their declared constraint
strategy — **masking** for discrete (§7.1), **clipping / reparametrization**
for continuous (§7.2) — and `reward_mode`s assembled from the objective
components in `info`.

**Gate**: valid spaces; `reset`/`step` obey the Gymnasium contract; masks match
`valid_actions`; obs dtype/shape stable across a rollout.

### Stage 3 — Baselines (mandatory, not optional)
A trained policy is meaningless on a novel problem without a yardstick. Auto-
generate at minimum **random** and a **myopic/greedy heuristic**; add **LP/DP**
where the IR structure permits and the state is low-dim / tractable. Baselines
also fix the reward scale, feeding the normalization decision in Stage 4. Under
partial observability, fall back to heuristic baselines.

**Gate**: baselines run to completion and bound the reward scale.

### Stage 4 — Train + tune + eval
- **Config from the IR's `rl` block**: `algo` (`PPO`/`MaskablePPO`/
  `RecurrentPPO`), `frame_stack`, `net_arch`, and the explicit
  `obs_normalization` decision (spec §8.3: on for heterogeneous stationary
  features; off under distribution shift) — all recorded fields, not folklore.
  The run manifest additionally pins which gym modes (obs/action/reward) the
  model was trained with; Stage 5 reads that binding.
- **Tuning**: implemented as the domain-generic **`mdp_tuning/` harness**
  (Optuna TPE; like `mdp_conformance`, parametrized by domain with zero
  per-domain code). The search space is *algorithm-level* (`spaces.py`,
  RL-Zoo-style broad ranges) intersected with the knobs the generated train
  script exposes via `_build_arg_parser()` introspection — no per-problem
  judgment. In short: sensible defaults + short screening runs + a *small* Optuna search
  (lr, entropy, net width) with **learning-curve early stopping**. Full grid
  search is too costly to be the default; a budget controller caps trials.
- **Eval**: spec seed-loop TSV format (§9), reported **against the baselines**.
- **Gate**: learned policy beats random and myopic by a margin; training curve
  stable (no divergence/collapse).

### Stage 5 — Deployable artifact (a Python file)
Emit `{domain}_policy.py` exposing a small, stable interface:

```python
class {Domain}Policy:
    def __init__(self, model_path, vecnorm_path=None): ...
    def act(self, obs) -> action:      # deterministic by default
        ...
```

It bundles: load the SB3 `.zip`; load `vecnormalize.pkl` and apply
`normalize_obs` (part of the input contract, spec §9.5); apply any **action
transform** (reparametrization → feasible action). It ships with a **documented
observation contract** (fields, order, units the caller must supply).

**Note — not fully standalone for constrained/masked domains.** Masked-discrete
policies need `valid_actions`, and reparametrized-continuous policies need the
transform; both live in the `_mdp` layer. So `{domain}_policy.py` imports the
domain's mask/transform logic rather than duplicating it. Document this
dependency explicitly.

---

## 5. Cross-cutting: the conformance harness (build this first)

A domain-parametrized test harness that encodes the spec's invariants as
executable checks (the list in Stage 1). Point it at any `{domain}/` directory.
It is **simultaneously**:
- the **Stage 1/2 gate** in the prompted design,
- a **regression suite** as the spec evolves,
- and the **reward signal** if a trained generator is pursued later.

**Validate it today against the 5 existing domains** (`owmr`, `cnv`, `fnv`,
`inv_single`, `2048`) — they should all pass. This is the single highest-leverage
piece and a prerequisite for either agent-implementation strategy.

---

## 6. Agent implementation: prompted vs trained

**Prompted orchestration** (recommended for v1): base model + spec + exemplars +
verify/repair loops; the harness is a runtime accept/reject/retry gate. Low build
cost, instant iteration, reacts to spec changes by editing text.

**Trained generator** (later, if justified): fine-tune via RL (GRPO/PPO) or DPO,
using the harness score + downstream training success as reward / preference
signal. Higher ceiling on your specific distribution, but needs a dataset,
training infra, and **goes stale on every spec change**.

Decision drivers: the spec is still actively changing, and the verify/repair loop
closes most of the gap training would buy — so **start prompted**. Move to trained
only if prompting plateaus on a recurring failure mode, the spec has stabilized,
and there's volume to justify it. The harness is needed either way, so starting
prompted wastes nothing.

---

## 7. Challenges & gaps (ranked)

1. **Formalization correctness has no oracle.** Passing tests ≠ correct dynamics.
   Mitigations: IR-derived conservation/property checks (e.g. inventory balance),
   human round-trip confirmation, cross-check against an independent baseline.
   A human confirm at the IR gate is likely unavoidable for correctness.
2. **Certifying policy quality on novel problems.** No known optimum ⇒ baselines
   are mandatory and "≥ baseline" is a weak bar. Deliverable is explicitly
   "best-effort + benchmarked," not "optimal."
3. **Compute / time / cost.** Tuning dominates. Needs budget controller, cheap
   proxies, early stopping, parallel trials.
4. **RL brittleness — reward scale & normalization.** Encode the spec's §8.3
   heuristics as an explicit IR-driven decision policy.
5. **Seeding correctness.** Reverse-tree scheme is precise; the decision-path-
   independence test is the guard.
6. **Partial observability.** Adds memory/recurrent policies and makes optimal
   baselines hard; keep other dims (low-dim, finite, single-agent) tight to
   compensate.
7. **Bounded self-repair.** Retry caps + escalation-to-human, or the agent loops
   forever on a mis-formalized problem.
8. **Orchestration state.** Checkpoint between stages so a Stage-4 failure doesn't
   redo Stages 0–3.

---

## 8. Recommended build order (MVP first)

1. **Conformance harness** — validate against the 5 existing domains.
2. **IR schema** + Phase-A elicitation with round-trip confirmation.
3. Codegen Stages 1–2 for **one new held-out simple domain** (a newsvendor
   variant not already in the repo), gated by the harness — prove the loop
   end-to-end.
4. Add baselines + **fixed-hyperparameter** train/eval (no tuning yet).
5. Add tuning + the deployable `{domain}_policy.py` packaging.
6. Generalize prompts/exemplars across problem classes; add repair loops and the
   budget controller.

---

## 9. Remaining open questions

- **Baseline tractability policy**: when the state is low-dim but partially
  observed, how hard do we try for a DP/belief baseline vs. defaulting to
  heuristics? (Proposal: heuristic by default; DP only when fully observed and
  small.)
- **Tuning budget**: default wall-clock / trial cap per problem?
- **Repair budget**: max regeneration attempts per stage before escalating to
  the user?

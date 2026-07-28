# Layered MDP-IR: problem **structure** vs. concrete **bindings**

**Status:** SUPERSEDED as an artifact design — the binding artifact (§3) was
built, reviewed, and **scrapped the same day** for the catalog ⊕ selection
model (§10, now implemented). §§1–2 (the problem, the executability
constraint), §3d (the axis discriminator), and §4's naming constraints remain
valid analysis; read §10 for the design that shipped. Created 2026-07-27.
**Owner:** tong. **Trigger:** the `inv_single` two-schema awkwardness (below).

> This is a spec proposal, not a scheduled change. The IR schema is a shared
> contract (`harness/mdp_ir/`) that downstream research repos depend on, so any
> implementation needs a coordinated migration — see §6.

---

## 1. The problem

`inv_single` now carries **two** IR schemas:

| file | demand source | latent recipe | demand constants |
|---|---|---|---|
| `inv_single_schema.json` | `categorical` (`vals`/`probs`) | `choice_without_replacement` + `normalized_uniform_weights` | `demand_vals`, `demand_probs`, `demand_support_*` |
| `inv_single_poisson_schema.json` | `poisson` (`rate`) | `gamma` | `demand_alpha`, `demand_beta`, `demand_rate` |

A structural diff shows the two files are **identical except for the demand
uncertainty source, its constants, and the sampler recipe**. Everything the
first-pass diff flagged as differing between them is serialization noise:
`mdp.dynamics` differs by exactly 3 lines (all `"guard": null`), and
`state_variables`/`decisions` differ only in `0` vs `0.0` and optional `null`
fields present in one file and omitted in the other — the actual bound values
(30, 200, 400) match. The state, dynamics, event-sequence, cost/objective, and
decision space are the same inventory MDP.

So we copy-pasted an entire problem to swap one uncertainty source. **The IR is
supposed to describe a problem *structure*; today it is welded to a concrete
scenario** (a concrete demand family *and* the numeric bounds derived from it).

Root cause: an IR `UncertaintySource` declares a **single** distribution
`family: str` ([`schema.py:408`](harness/mdp_ir/schema.py)), and an *instance*
is only a dict of constant-value overrides
([`schema.py:345`](harness/mdp_ir/schema.py)). Instances vary **values inside a
fixed structure** (that is why `lost_sales` — flipping `allow_backlog` — is an
instance of the discrete schema). A different demand **family** is a different
*structure*, which instances cannot express — hence a whole second schema.

## 2. The hard constraint (why we can't just go abstract)

The MDP-IR is not only a structural description — it is an **executable,
differentially-verified twin**. `python -m mdp_ir.differential` *runs* the
interpreter and matches its trajectories bit-for-bit against the Python domain.
To run, the interpreter must **sample** demand and **build concrete gym spaces**
— which requires a concrete family (a `rate`, or `vals`/`probs`) and numeric
bounds. You cannot sample an abstract `DemandGenerator` or size a space from an
interface.

**Therefore a *purely* abstract IR (demand as "some DemandGenerator") is not
differentially verifiable — and differential verification is the entire reason
the twin is trustworthy.** Abstraction alone would trade away the IR's core
value. The abstraction has to live in a layer that a concrete layer can still
execute.

## 3. Proposed design: structure ⊕ bindings

Split the IR into two layers instead of duplicating whole schemas.

### 3a. Structure layer — one per domain (family-agnostic)
Captures *what problem this is*:
- state variables, dynamics, event-sequence, cost/objective, decision space;
- **typed uncertainty slots**: `demand: DemandGenerator`, `leadtime:
  LeadtimeGenerator` — a slot names the *interface* (its read-API: `mean`,
  `max`, `phi`, `is_discrete`, sampling seed-key contract), not a family;
- the abstract **scenario / scenario-sampler shape**: which constants are
  *structural* (shared) vs. *latent* (supplied by a binding), and that a
  world-latent sampler draws the latent slot once per episode.

This is literally *"define the IR around the abstract classes"* — the slots are
`DemandGenerator` / `LeadtimeGenerator`, and the abstract sampler/scenario shape
mirrors the `InvSingleScenarioSampler` base and `InvSingleScenario`.

### 3b. Binding layer — one per concrete family choice (executable)
Binds each slot to a concrete family + latent recipe + family-specific
constants:
- `discrete` binding → `demand = categorical(vals, probs)`, latent via
  `choice_without_replacement` + `normalized_uniform_weights`;
- `poisson` binding → `demand = poisson(rate)`, latent via `gamma(alpha, beta)`.

Each binding is executable and **differentially verified against its Python
sibling** (`InvSingleDiscreteSampler` / `InvSinglePoissonSampler`).

### 3c. The symmetry
This is the base-plus-children hierarchy you just built in Python, lifted into
the IR:

| Python | IR |
|---|---|
| `DemandGenerator` (abstract) | structure **slot** `demand: DemandGenerator` |
| `DiscreteDemand` / `PoissonDemand` | **binding** picks the family |
| `InvSingleScenarioSampler` (abstract base) | structure's abstract sampler shape |
| `InvSingleDiscreteSampler` / `InvSinglePoissonSampler` | one **binding** each |

**One structure, N bindings, zero duplication.** Structure = "what problem";
binding = "which concrete families realize it." Differential runs each binding;
conformance checks the structure once plus each binding.

### 3d. The precise boundary — what a binding may and may not change
A **binding realizes an uncertainty slot** — the distribution family, its
latent-draw recipe, and its family-specific constants — **without changing the
structure's shapes (state / observation / decision) or its dynamics equations.**
*Shape- and dynamics-preserving* is the test. Anything that touches the decision
space, a state's shape, or *which* dynamics branch executes is **structural**,
not a binding. Variation therefore sorts onto **three** axes, not two:

| what varies | belongs to |
|---|---|
| distribution family + latent recipe filling a slot (shape/dynamics-preserving) | **binding** |
| constant *values* within a fixed structure+family | **instance** |
| a flag gating optional decisions/dynamics via **guards**, in one parametric structure | **structure (parametric mode)** |
| different state vars / genuinely different dynamics, no shared parametric form | **structure (hard fork)** |

So "structure" is **not monolithic**: it splits into a *parametric* form (mode
flags, dimension K / entity count — one structure, many settings via guards and
parameters) and a *hard fork* (genuinely separate structures). The IR already
carries the mechanism for parametric modes: **guards** on dynamics steps (the
`"guard": null` fields) and conditionally-present decisions.

**Operational discriminator — encapsulatability (not file location).** The
reliable test is *not* which file a change touches: a flag can be declared in
`_scenario` and branched in `_mdp` and be *either* axis. It is — **can the
variation be sealed inside an uncertainty slot's generator, leaving every space
*shape* (state / obs / decision) and the transition/reward *as functions of the
realized quantities and the decision* unchanged?** Yes → **binding** (only
scale-derived numeric bounds may move, and those resolve per binding). It changes
a space *shape*, adds/removes a decision, or alters a transition *function* not
reducible to a slot realization → **structural**. (Corollary: the intuitive
"changes `_mdp` = structural, changes `_scenario` = binding" heuristic is
**wrong** — both examples below are `_scenario`-declared and `_mdp`-branched, yet
classify oppositely.)

**Equivalent one-line invariant: `_mdp` is binding-agnostic.** In a properly
layered domain, `_mdp` consumes uncertainty slots and decisions *generically*,
unaware of which binding is in play. So a **binding** change = another scenario
(+ a slot generator in `_uncertainty`/`_scenarios` if the family is new) with
`_mdp` untouched **and unaware**; a **structural** change = `_mdp` must gain
variant-aware logic (a transition branch/guard, a new decision, a new state var).
The airtight test is the *normative* **"could `_mdp` be written not to know about
this variant?"** — not the as-coded "does the diff touch `_mdp`?" They diverge at
fnv: `mmfe_mode` *touches* `_mdp` today yet is a **binding** (the link function
belongs in the slot; the inlined branch is a smell to refactor out). Yes → binding;
no → structural. (Footnotes: a binding may still need a *new generator class* in
`_uncertainty`/`_scenarios`, never `_mdp`; and pure `_gym` menu changes — obs /
reward modes — are a *third*, interface axis, neither structural nor binding.)

**Worked example — `adi_flex` (rl_test), homogeneous vs heterogeneous.**
Tempting to call these two *bindings*; they are not. The demand is a length-3
Poisson vector (`SegmentDemand` ×3) in **both** branches — homogeneous is just a
one-hot rate vector — so the demand **binding is identical**. What splits them is
`sigma_enabled`, a **structural mode**: it swaps the action space
(`MultiDiscrete[Z,Σ]` vs `Discrete[Z]`) and guards a dynamics branch
(`sigma_eff = protect if sigma_enabled else 0`). Within each branch, the
`homog_L{l}_T{t}` grid and `het_exp{n}` mixes are **instances** (value overrides).
Net: **one parametric structure** (sigma mode + shared Poisson-vector binding),
two mode settings, each with an instance family — *not* two bindings, *not* two
independent structures.

**Contrast — `fnv` (rl_test), additive vs multiplicative MMFE = a *binding*.**
The mirror image. `mmfe_mode` branches only the terminal demand realization —
`D = mu + I` vs `D = exp(mu + I)` ([`fnv_mdp.py:143-158`]) — while
`sales = min(D, inv)`, revenue and cost consume the scalar `D` identically. The
action space is a 1-D `Box` in both modes (only its numeric `high` differs — a
scale-derived bound); the observation shape is mode-invariant (the agent sees the
raw signal `I`). The whole difference is *how the demand slot is realized from the
shared signal* (`mu` reinterpreted as mean vs log-scale) — encapsulatable in a
`DemandGenerator`. So aMMFE/mMMFE are **two bindings of one structure**; the
inline `if mmfe_mode` in `_mdp` is a coding choice the layered discipline would
push into the slot (cleaning up the Python too). The framing "special case"
(adi_flex homog ⊂ het) vs "parallel siblings" (fnv a ‖ m) does **not** predict the
axis — encapsulatability does.

Caveat this raises for the UX (§7): a mode flag *feels* lightweight to a user
("just toggle sigma") but is **structural** — adding one is a structure edit
(re-authors, re-verifies all bindings), not a cheap binding add. The forward/
expansion routing (§7d) must recognize a mode toggle as structural despite its
small surface.

## 4. Terminology
The schema already uses `Realization` for **draw timing** (period/event/
episode/keyed — [`schema.py:127`](harness/mdp_ir/schema.py)). Do **not** overload
it. Call the new axis a **binding** (structure ⊕ binding). A `# variant would
add a `kind` discriminator here` breadcrumb already exists at
[`schema.py:221`](harness/mdp_ir/schema.py) — a tagged-union discriminator is the
likely mechanism.

**Naming & surfacing (decided 2026-07-27).** `binding` is the term **throughout**
— schema fields, gate output, docs, plan. No prose synonym: "world" was proposed
and **dropped**, because the spec already spends the word at two *other* scales
(the **world layer** = `{domain}_scenarios.py`, §5; the **world latent** = the
per-episode draw, §5.2), and because "world" and "scenario" are near-synonyms in
English while denoting different levels here. The precise hierarchy is
**form → numbers → draw**:

| level | fixes | cardinality |
|---|---|---|
| structure | the problem shape; slots unfilled | one per domain |
| **binding** | *which families* fill the slots | one per family choice; fixed per run |
| instance | *which numbers* the constants take | one per named override |
| scenario (spec §5) | everything — a fully concrete problem instance | one per episode |

A binding is a **space** (all scenarios sharing a functional form); a
`{Domain}Scenario` is a **point**, and that spec meaning is fixed. Between them
sits the existing **scenario source** (binding + instance → concrete scenario, or
a sampler that draws the remaining latents per episode).

Ruled out as alternative names: **`variant`** — already taken *and* it denotes the
opposite side of the §3d boundary (`allow_backlog` carries `"axis": "variant"` and
is **structural**); **`realization`** (draw timing); **`instance`** (the
neighbouring axis); **`model`** (collides with the `/model` built-in).

**The noun stays off the command surface.** A user says "also try Poisson demand";
classifying that as binding-vs-instance is the agent's mechanical job (does it
change a distribution family, or only constant values?), and its only visible
consequence is how much sign-off is needed — surfaced as a question about content
("what prior on λ?"), never about category. Name artifacts and result columns by
*what varies* (`inv_single.poisson.json`, a `demand` column), so the abstraction
needs naming only in docs — where `binding` is the right precise word anyway.

## 5. What a layered `inv_single` looks like
- `inv_single_structure.json` (or a `structure` block): the inventory MDP with
  `demand`/`leadtime` slots — **no** demand family, no `demand_*` latent
  constants.
- `inv_single.discrete.json` / `inv_single.poisson.json` (or a `bindings: []`
  array): each supplies the demand family, the latent recipe, and its
  `demand_*` constants; both `$ref`/point at the one structure.
- The shared portable-domain adapter already branches on
  `demand.distribution.family` to pick the Python sampler
  ([`inv_single_ir_adapter.py:56-68`](plugin/skills/mdp-solver/examples/inv_single/inv_single_ir_adapter.py))
  — that logic is exactly the binding→sibling dispatch, so the adapter side is
  mostly there.

## 6. Scope & migration (when we build it)
Touches the shared harness:
- **`schema.py`** — new constructs: a structure with typed slots + a binding
  that fills them; validators (a binding must fill every slot; a slot's read-API
  must be satisfiable; structural vs. latent constant partition).
- **`interpreter.py`** — "bind-then-execute": resolve the binding's families
  into the structure, then sample as today.
- **`differential.py`** — iterate bindings; the adapter dispatch already exists.
- **`mdp_conformance`** — validate the structure once, then each binding.
- **`mdp_tuning` / `mdp_gates`** — confirm they key off a resolved
  (structure+binding), not a whole-schema fingerprint.
- **Existing domains** (`dynamic_pricing`, `fnv`) — re-expressed as structure +
  a single binding. Must stay green with **no behavior change**.
- **Downstream research repos** — coordinated check before release (per
  CLAUDE.md); provide a back-compat path so a legacy single-file schema loads as
  "structure + one implicit binding."
- **MANIFEST / examples** — gate lines become per-binding.

Migration order: prototype on `inv_single` as the reference (it already has two
bindings), then fold `dynamic_pricing`/`fnv` to single-binding, then downstream.

**Interim state until then (decided 2026-07-27).** `inv_single_poisson_schema.json`
stays on disk **untracked**, as the fixture the layering will be prototyped
against — but it is *not* a shipped gate: its two `MANIFEST.md` lines were
reverted. Rationale: `examples/` are few-shot exemplars for the skill, so a
committed duplicate schema would teach every generated domain to duplicate — the
exact failure this plan exists to remove. Consequences accepted meanwhile:
`InvSinglePoissonSampler` is exercised by conformance's sampler checks but is
**not** differentially verified (it gets its differential when it gets a
binding), and the adapter's `if demand.distribution.family == "poisson"` branch
is dead code — kept deliberately, since that dispatch is precisely the seam that
becomes a binding lookup.

## 7. The launch/agent UX — two workflows

The layering is right for the *artifact*; the question is how it surfaces when a
user launches the agent. Key realization (2026-07-27): the two workflows the user
cares about are the two ends of the structure⊕binding split, **not** a
contradiction.

### 7a. Forward-from-scratch = establish structure + binding[0]
The `mdp-solver` pipeline (verbal problem → IR → trained policy) produces a
structure **and its first binding** in one shot, presented as a **single IR /
case**. The split is latent — the user never sees "structure vs binding." The
simple, end-to-end experience we have today is untouched. **Layering is *free*
at one binding.**

### 7b. Expansion = add binding[k] to a frozen structure
"Add more scenarios on top of an existing case" reuses the **frozen** structure
and only authors a **new binding** (a new demand family + latent recipe + its
constants, wired to a Python sibling generator). It runs **only that binding's**
differential + gates + training. The structure IR is **not re-authored** — which
is exactly the "we do not want to touch the IR every time" requirement.

### 7c. The principle that dissolves the tension
The **structure is the durable, once-authored, differentially-verified backbone**
(expensive: LLM formalization). **Bindings — and instances — are cheap accretions
that never re-touch it.** So the line for *"when must I touch the structure IR"*
is sharp: **only when the *structure itself* changes** (a new state variable, a
new decision, new dynamics) — **never** for a new binding / family / scenario.
Progressive disclosure: the structure is a real, reusable on-disk artifact (so
bindings can point at it) but hidden in the forward run and surfaced only in
expansion. **Simple by default, structured when expanding.**

### 7d. Two subtleties this exposes
- **Scale-derived bounds must be binding-resolved, not frozen in the structure.**
  Today both `inv_single` schemas carry the *same* numeric bounds (30/200/400) —
  binding[0]'s demand scale leaked into what should be shared structure. For the
  structure to be genuinely family-agnostic, demand-scale-derived bounds
  (inventory/order ranges, obs/act spaces) must live in the binding, or be held
  **symbolically** in the structure (`f(demand.max())`) and resolved numerically
  per binding. Else a Poisson binding (mean 30) inherits bounds sized for a
  discrete {8..12} demand.
- **Expansion's core routing decision: binding vs instance vs structural change.**
  The agent must classify an incoming "add a scenario" request:
  same structure + new family → **binding** (structure untouched);
  same family + different values → **instance** override (cheapest);
  genuinely new structure (a second product, a new decision) → **structure edit**
  (re-authors, and re-verifies *all* existing bindings). The user shouldn't have
  to know which — the agent routes.

### Still open
- When does the forward run *proactively* emit >1 binding vs. exactly one? Lean:
  exactly one unless the problem explicitly poses alternative families or the
  user asks — multi-binding is opt-in.
- Is **expansion a distinct mode/command** of the agent? (Ties into the
  agentization plan's mode split.)
- Leaderboard / results presentation for "one structure, N bindings" without it
  reading as N problems. Label columns by *what varies* (`demand=poisson`), not
  by the category noun (§4).
- On-disk shape a user sees: one file with sections vs. a structure file +
  binding files.

Guiding principle: **structure⊕binding is an internal correctness win that must
not tax the single-binding user — and it is precisely what makes the *expansion*
workflow cheap.** Multi-binding is opt-in; the structure is authored once.

## 8. Alternatives considered
- **Purely abstract IR** (slots only, no families anywhere) — *rejected*: not
  executable, so not differentially verifiable; forfeits the twin's value (§2).
- **`$ref`/include dedup** (two files share one structure fragment, families
  stay inline) — *fallback*: kills the copy-paste but is cosmetic; it does not
  give the abstract-slot model, so a domain still has no single "structure"
  object, just text reuse.
- **Status quo** (N full schemas) — *rejected as the target*: the duplication in
  §1 is the smell we are removing.

## 9. Decision log
- 2026-07-27 — Direction agreed (structure⊕binding, not pure abstraction).
  Implementation gated on the §7 UX questions. No harness code touched yet.
- 2026-07-27 — UX framing resolved (§7): forward-run = establish structure +
  binding[0] as one IR (simple path unchanged); expansion = add a binding to the
  frozen structure without re-authoring it. Structure = once-authored backbone;
  bindings/instances = cheap accretions. Residual: multi-binding trigger, whether
  expansion is its own mode, results presentation, on-disk shape.
- 2026-07-27 — Boundary sharpened (§3d) against `adi_flex`: a binding is a
  shape/dynamics-preserving slot realization; three axes = binding (family) /
  instance (values) / structure (parametric mode via guards, or hard fork). A
  *mode flag* (adi_flex `sigma_enabled`) is structural despite feeling small.
  adi_flex homog↔het = one parametric structure (not two bindings).
- 2026-07-27 — Discriminator settled (§3d) via `fnv` contrast: the axis is
  **encapsulatability in an uncertainty slot**, NOT file location (both
  `sigma_enabled` and `mmfe_mode` are `_scenario`-declared + `_mdp`-branched, yet
  opposite). fnv aMMFE/mMMFE = a **binding** (demand link function `D=f(mu,I)`);
  adi_flex `sigma_enabled` = **structural**. "special case" vs "parallel" framing
  does not predict the axis.
- 2026-07-27 — **Built.** On-disk shape = **always split** (user's call over the
  recommended inline-until-second); bounds = **symbolic in the structure** (over
  the recommended structure-literal + binding-override). New
  `harness/mdp_ir/layering.py` (resolution) and `split.py` (migration);
  `load_ir` dispatches on `kind` and still accepts a single file as
  structure + one implicit binding, so downstream repos keep working.
  Layering lives in the **loader** — it resolves to the existing `MdpIR`, and
  symbolic bounds resolve to numbers at load, so interpreter/differential/
  conformance saw no change. All three examples migrated; 6 differentials MATCH
  bit-for-bit and the discrete binding's resolved IR is byte-identical to the
  retired single file (same `mdp_fingerprint`). Poisson now resizes correctly
  (inventory ±1200, order 600 vs discrete's ±400/200) — §7d's bounds leak fixed
  by construction — and `poisson × lost_sales` runs with **no new artifact**.
  Known cost accepted: every binding restates *all* slots, so `leadtime` is
  duplicated across `inv_single`'s two bindings (~12 lines); a binding
  `extends` mechanism is the candidate follow-up, deliberately not invented
  mid-implementation.
- 2026-07-27 — Naming settled (§4): `binding` throughout, no prose synonym.
  "world" dropped (spec spends it on the *world layer* and *world latent*, at
  other scales; near-synonym of `scenario`, which is a fixed spec term for a
  fully concrete point). Hierarchy = form → numbers → draw (structure /
  **binding** / instance / scenario). `variant` explicitly ruled out — taken, and
  it denotes the *structural* side (`allow_backlog` has `"axis": "variant"`).
  Category noun never appears on the command surface; the agent classifies the
  request, and artifacts/columns are named by what varies.
- 2026-07-27 — Interim state chosen (§6): un-ship the duplicate rather than
  build the layering under time pressure. `inv_single_poisson_schema.json`
  reverted out of `MANIFEST.md`, kept untracked as the prototype fixture.
  Layering (option B) remains gated on the §7 residuals — chiefly the on-disk
  shape (one file with sections vs. structure + binding files).
- 2026-07-27 — **Binding artifact scrapped** (review, same day it was built).
  Three fatal commitments it bundled: (1) a binding is a *total* function over
  all slots, so bindings enumerate the slot×family cross-product and every
  binding restates every slot (`leadtime` duplicated; fnv/dynamic_pricing
  forced to carry near-empty boilerplate binding files — taxing the simple
  case against §7's own principle); (2) family knowledge (read-API formulas,
  latent recipes) authored per-domain-per-binding though it is generic math —
  the Poisson envelope formula existed twice, hand-written in JSON and in
  Python; (3) *verified* conflated with *frozen* — the differential is a
  runner, so verification needs executability, not freezing. Root cause: a
  binding is pure data (family + settings + latent recipe per slot) that was
  given artifact/schema status. Superseded by §10.
- 2026-07-27 — **Catalog ⊕ selection model agreed** (§10): per-slot
  `candidates` declared in the schema (mirror of `_uncertainty.py`'s class
  pool), instances extended to carry per-slot candidate *selections* (mirror
  of the `SCENARIOS` registry), latent draws nested in candidate settings
  (`rate: {draw: gamma(...)}`) replacing hand-wired sampler blocks, read-API
  derived from a harness family registry instead of authored. Declarations
  scale as M+N per slot; compositions are references (one line or a CLI flag),
  never files. Cross-family mixtures become expressible (inexpressible under
  bindings — they would have spanned artifact files).
- 2026-07-27 — **Frozen-set defined computationally** (§10c): frozen = the
  structural core (state/decisions/dynamics/objective/slots/stages/stream_ids
  and all type/dim/length shapes) plus the transitive closure of constant
  *names* its expressions reference. Values, candidates, instances, mixtures,
  and catalog-side constants are free data with hash provenance. Candidate/
  instance names are labels, never frozen. Mirrors the existing precedent that
  `gym`/`rl` sit outside `mdp_fingerprint`.
- 2026-07-27 — **Structural vs value parameters derived, never user-declared**
  (§10d): a constant referenced by the core in *control position* (guard,
  conditional, event order, anything a space shape depends on) is a structural
  parameter — each used setting needs differential coverage; *arithmetic
  position* = value parameter — one representative value verifies the path.
  `allow_backlog`/`evt_order`/adi_flex `sigma_enabled` vs `h`/`demand_rate`.
  Declaring a NEW structural parameter is a structure edit; selecting among
  declared settings is an instance. Event-sequence variants (`test_rdo/rod/
  ord`) become instances once `dynamics.event_sequence` may name a constant
  (the `horizon.T` precedent) — deferred, not in the first implementation.
- 2026-07-27 — **Stream identity unified** (§10e): one identity per source of
  randomness — the slot's `stream_id`, declared once in the structure, keys
  BOTH branches: intrinsic draws (branch 1, as before) and that slot's episode
  latents (branch 0). The `substream_id` field disappears from samplers/
  candidates; mixtures (the only composition-scoped drawer) allocate above the
  slot range. Consequence: common random numbers across instances sharing a
  candidate (the intrinsic branch already behaved this way; per-drawer meta
  substreams were the odd one out, bought nothing — two drawers never co-run —
  and cost duplicated sampler blocks and broken cross-instance pairing).
- 2026-07-27 — **Python layer rule set** (mirror of §10, stage 2):
  everything about ONE source of randomness — its families, episode latents,
  both seed branches, envelope — belongs in `_uncertainty.py` (generators grow
  `realize(episode_seed)`); `_scenarios.py` keeps only composition (the
  scenario dataclass, the named registry, mixtures). Kills the per-family
  sampler subclasses and `FamilyDemandBounds`.
- 2026-07-28 — **Stage 2 built** (the Python mirror). `inv_single`
  restructured: `LatentDiscreteDemand` / `LatentPoissonDemand` own their
  episode latents (`realize(episode_seed, seed_salt)` keyed on the meta
  branch at their own `source_id`; `meta_key` moved into `_uncertainty.py`);
  the generic `InvSingleScenarioSource` (template + per-seed realization +
  attribute delegation) replaces `InvSingleScenarioSampler` and both
  per-family subclasses; `FamilyDemandBounds` folded into the latent
  generators' envelope methods; registry keys unchanged, entries now
  `source_{name}`. "Scenario vs sampler" is derived from content in Python
  too. Adapter rebuilt on the same classes — all 4 differentials MATCH
  bit-for-bit at 40 episodes, conformance green, 67 interpreter checks pass.
  MDP_PROJECT_SPEC updated throughout (§1 files/chain, §2 naming —
  `Latent{Variant}{Source}`, `{Domain}ScenarioSource`, `source_{name}` —
  §4.3 boundary, §5.2 rewritten, §5.3/§5.4 aligned; deterministic-instance
  sources and mixtures are the only drawers with their own `substream_id`,
  allocating ≥ `N_SOURCE_IDS`). Conformance `check_meta_v2` relaxed to
  match: a callable source without `substream_id` is the
  generator-owned-latent convention (its meta stream is the generator's
  `source_id`, already checked for distinctness); substream distinctness now
  binds only composition-scoped drawers.

- 2026-07-28 — **§10f residuals closed** (three of four; latent salt stays
  deferred by user call — no use case, `--seed-salt` is the coarse opt-out).
  (1) *Generic runtime*: `mdp_ir/runtime.py` — `sample_family` becomes the
  single family→numpy dispatch (interpreter delegates; twin sides cannot
  drift), `FamilyGenerator` implements the spec-§4.2 shape over any
  registered family incl. nested latent recipes on the v2 template;
  `families.min_value` + `min` read-API added. Domain cost: one
  `Family{Source}` bridge class per slot; `inv_single`'s adapter dispatches
  by candidate name with the bridge as fallback — an appended uniform-rate
  candidate passes the full differential with zero domain edits.
  (2) *Cross-family mixtures*: per-component resolutions at the root
  (`MdpIR.mixture_resolutions`, loader-set like `selection`, outside
  `mdp_fingerprint`); interpreter swaps sources/samplers per episode
  (standalone equivalence bit-exact); mixture components survive instance
  filtering (constants-only); bounds envelope per §5.3. `mix_demand`
  (0.5 discrete ⊕ 0.5 poisson) ships in `inv_single` — catalog mixture +
  `source_mix_demand` registry twin — and joins the `--all-instances`
  covering set (7 differentials MATCH at 40 episodes). Conformance's
  registry-name check now reads the *source*'s name (a mixture resolves to
  a component scenario that rightly keeps the component's name).
  (3) *`event_sequence` as a constant*: `Dynamics.event_sequence:
  list|str` + `MdpBlock.event_sequence(instance)` (the `horizon_T`
  precedent); the interpreter sorts transitions by the resolved order per
  episode, so `test_rdo/rod/ord`-style variants are instances (R-D-O
  differential MATCHes the domain bit-exact). Literal sequences now require
  transitions declared in order (loud error, not silent reorder — the one
  behavior-affecting edge for downstream IRs). The event-order constant
  name freezes into `structural_fingerprint` (§10d control position).
  All three examples' structural fingerprints unmoved; suite: 82
  interpreter checks, 3×conformance green, 7 differentials MATCH.
  Plugin 0.4.0 → 0.5.0.

## 10. The design that shipped: **catalog ⊕ selection**

The binding artifact solved duplication by *moving* the enumeration; the
catalog dissolves it by copying how the Python side already avoids it:
`_uncertainty.py` declares the menu of generator classes once, and
`_scenarios.py` composes them by reference. Lifted into the IR, level by
level (the §3c table, completed):

| Python | IR |
|---|---|
| numpy rng primitives (`rng.poisson`, `rng.choice`) | harness **family registry** (`mdp_ir/families.py`) |
| `_uncertainty.py` generator classes | per-slot **candidates** (declared once, side by side) |
| a concrete scenario (fixed generators + costs) | **instance** whose selected candidates carry no `draw`s |
| a world-latent sampler subclass | **instance** whose selected candidates carry `draw`s |
| the mixture combinator | **mixture** (existing `ScenarioMixture`) |
| the `SCENARIOS` registry | the named instances + mixtures pool |
| constructing a scenario ad hoc in a script | `--select slot=candidate` at the CLI |

"Fixed scenario vs sampler" stops being a declared type distinction — it is
derived from whether any selected candidate setting carries a `draw`.

### 10a. On-disk shape
One file per domain again: `{domain}_schema.json`. No `kind` marker — the
loader discriminates on `mdp.uncertainty_slots` (catalog form, has
`candidates` + `default` per slot) vs `mdp.uncertainty_sources` (legacy
resolved single file, still loads as-is). The structure⊕binding split files
are gone; `split.py` deleted with them (the format it emitted no longer
exists; nothing was ever published).

### 10b. Slot, candidate, draw, selection
A slot keeps its structural skeleton (name, interface, `stream_id`, stages)
and gains `default` + `candidates: {name: {generator, family, settings,
is_discrete?, read_api?, desc?}}`. A setting value is a literal, an expr
string over constants (state/decision refs allowed exactly as before), or a
**draw spec** `{"draw": {family, settings}, "example": <placeholder>,
"hidden": bool}` — the world latent, nested where it belongs. The loader
desugars each slot's draws into the existing `ScenarioSampler` machinery
(synthesized constant `{slot}_{setting}` carrying `example` as its
placeholder; one sampler per slot at `substream_id = slot.stream_id`; draws
in settings-declaration order from one rng, preserving multi-draw recipes),
so the interpreter/differential/conformance are unchanged. **Selection** =
`{slot: candidate}`: defaults ⊕ the named instance's slot-valued keys ⊕
explicit `--select`. Instances therefore carry constants *and* selections;
at resolution, instances inconsistent with the active selection are dropped
from the resolved IR and slot keys are stripped from the survivors, so a
resolved `MdpIR` is always internally consistent. `MdpIR.binding` (never
shipped) is replaced by `MdpIR.selection`.

### 10c. What is frozen
`layering.structural_fingerprint()` hashes the structural core — everything
except candidate blocks, defaults, instances, mixtures, grids, and constant
*values* — plus the sorted set of constant names the core's expressions
reference (computed with the schema's own tokenizer, never hand-tagged).
Shape and names frozen; numbers and menu free. Appending a candidate or an
instance never voids Phase-A confirmation; provenance of results is
`(structural_fingerprint, selection, constants-hash)`.

### 10d. Verification policy
Registry families are unit-tested once, centrally. The structure validates
once. Each *composition you run* is differentially verified on demand
(`--all-instances` sweeps the declared pool); gates pin a covering set —
every candidate and every structural-parameter setting exercised at least
once, size ~max over slots, never the cross-product. A candidate whose
family no registry entry covers declares a custom family (the `ExprBuiltin`
pattern: name + module next to the IR) and supplies `read_api` explicitly.

### 10e. Derived read-API
Symbolic bounds (`20 * demand.mean`) resolve against a namespace computed
from the registry: `mean`/`max`/`is_discrete` per family, composing through
the latent hierarchy (a latent setting contributes its draw-family's
mean/envelope; envelopes use the 4-sigma convention, reproducing e.g.
`poisson(rate ~ gamma(9, 1/0.3))` → rate_max 70 → max 70+4√70, the exact
formula previously hand-written). Derivation is lazy per attribute: a slot
whose read-API nothing references (fnv, dynamic_pricing) never derives, so
state-dependent settings (`rate = a*exp(-alpha*price)*dt`) are legal there.
A referenced-but-underivable attribute is a load error naming the fix
(`read_api` override on the candidate).

### 10f. Residual / next increments
- **Stage 2 (Python mirror) — DONE 2026-07-28** (see decision log): latents
  moved into `_uncertainty.py` (`Latent{...}` generators, `realize()` on the
  meta branch at their own `source_id`), `_scenarios.py` reduced to
  composition (`{Domain}Scenario` ⊕ the one generic `{Domain}ScenarioSource`
  ⊕ mixtures), per-family sampler subclasses and `FamilyDemandBounds`
  deleted, spec §5.2 rewritten.
- **Generic generator runtime — DONE 2026-07-28** (decision log): new
  `mdp_ir/runtime.py` — the one family→numpy sampling dispatch (interpreter
  delegates to it) + `FamilyGenerator`, a spec-§4.2 generator over any
  registered family with nested latent recipes, both seed branches, and
  moments derived via `mdp_ir.families` (`min_value` added; `min` joined
  the read-API). Domains bridge it once per slot
  (`FamilyDemand(FamilyGenerator, DemandGenerator)`); adapters fall back to
  `from_parts`/`from_ir` for candidates no hand-written class implements —
  **a new family needs zero new domain Python** (proven in
  `inv_single_test.py`: an appended uniform-rate candidate passes the full
  differential with zero domain edits).
- **Cross-family mixtures — DONE 2026-07-28** (decision log): the loader
  resolves each selection-divergent component separately
  (`MdpIR.mixture_resolutions`, root-level like `selection`); the
  interpreter re-selects sources/samplers per episode; bounds under a
  mixture load resolve against the §5.3 envelope (weighted mean of means,
  max of maxes). `inv_single` ships `mix_demand` (discrete ⊕ poisson) as
  the reference; `--all-instances` and `python -m mdp_ir` sweep mixtures.
- **`dynamics.event_sequence` as a constant — DONE 2026-07-28** (decision
  log): literal list or constant name (the `horizon.T` precedent); the
  interpreter executes transitions in the resolved order, so event-order
  variants are instances. Literal form: transitions must be declared in
  sequence order (validated). Per-instance validity (R before D; R-O-D ⇒
  leadtime.min ≥ 1) stays domain-side as planned.
- **Per-instance latent salt** as the CRN opt-out — **stays deferred**
  (2026-07-28, user call): no use case has appeared, and a differing
  `--seed-salt` per run already provides a coarse decorrelation. Revisit
  only if an analysis needs latents decorrelated while intrinsic draws stay
  paired.

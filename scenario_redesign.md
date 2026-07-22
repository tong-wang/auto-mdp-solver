# Scenario redesign

Status: **design settled; spec, skill, and conformance rewritten**
(2026-07-22, §12 steps 1–3 done — `MDP_PROJECT_SPEC.md` carries the
two-layer architecture in §5, the v2 seed tree in §6.3, the grid dispatch in
§9.6; `SKILL.md` carries the split interview and the transitional seed-key
pin; `mdp_conformance` enforces the new architecture scheme-aware, frozen
examples staying green; `mdp_ir` expresses samplers/mixtures/grids and both
seed schemes; both examples — `inv_single` and `dynamic_pricing` — migrated
to v2 with differentials bit-exact and the Stage-1 pin lifted — §12 steps
1–5 complete in this repo).
Remaining: `MDP_IR_SAMPLE.md` reference update and downstream rl_test
migrations (§10) at their own pace. Prototypes in `scratch/` (§11).

## 1. Motivation

Across the research domains we have met three recurring shapes of scenario
randomness:

1. **Fixed scenario** — all parameters concrete and constant within the
   scenario; the only randomness is intrinsic, realized per period
   (`inv_single`'s Poisson scenarios).
2. **Latent episode-level randomness** — a parameter realized once at episode
   start, then per-period intrinsic randomness unfolds around it (retailer:
   mean demand ~ LogNormal at t=0, full-price demand ~ Normal each period).
   Two treatments exist in the wild: **A** (retailer) embeds the episode-level
   draw inside a generator in `{domain}_uncertainty.py`; **B** (FNV) uses a
   `{Domain}ScenarioSampler` that realizes the draw at reset.
3. **Mixture** — a meta-scenario that is a probabilistic mixture of other
   scenarios with given weights, drawn once at episode start (Beergame:
   demand is Normal(10, 2) w.p. 0.5, Deterministic(4) w.p. 0.5).

Working through these surfaced a deeper distinction that the final design is
built on: FNV's sampler (uniform over 540 cost-parameter combinations) is
*not* the same kind of thing as retailer's. Retailer's draw is **nature's**:
the prior over market size is part of the MDP, and the problem — pricing
under an unobservable latent — collapses without it. FNV's draw is the
**experimenter's**: each cost combination is a complete standalone problem,
and the draw exists only because the training objective was a *generalist*
policy across the family rather than a specialist for one combination.
Design-level sampling occurs **if and only if** the objective is a
generalist (benchmarking across a grid is enumeration, not sampling;
domain randomization is generalist training with hidden family parameters;
curriculum is excluded, §6).

## 2. Architecture: two layers (settled)

One boundary rule (§4.3: meta-level randomness is realized once at episode
start, keyed on `episode_seed` only), and two layers with a strict direction:

| layer | module | objects | registry | meaning |
|---|---|---|---|---|
| **World** | `{domain}_scenarios.py` | `{Domain}Scenario`, `{Domain}ScenarioSampler`, `{Domain}MixtureSampler` | `SCENARIOS` | models the problem; samplers are **world latents only** — draws nature makes |
| **Design** | `{domain}_grids.py` | `{Domain}ScenarioGrid` | `GRIDS` | experiment design; describes the *generality* of a generalist |

**Layering rule:** a grid may contain anything from the world layer; nothing
in the world layer may contain or import a grid. The `_mdp` and `_gym`
layers never see a grid — the gym receives a plain scenario source per
episode; only training / eval / tuning drivers consume grids. Dependency
chain: `uncertainty ← scenarios ← grids`, grids imported by drivers only.

Types 2 and 3 of §1 are both world-layer samplers (the mixture is a sampler
whose latent is *which component runs*). FNV's "type 2" usage moves to the
design layer as a grid. Within each layer the shapes are extensible
(parametric draw, mixture, whole-instance draw for the world layer;
cross-product and explicit-list construction for grids) — no new top-level
concepts needed for the next domain.

## 3. Naming (settled)

- `{Domain}Scenario` — unchanged (§5.1): a fully concrete problem instance.
- `{Domain}ScenarioSampler` — unchanged in name, **narrowed in meaning**:
  world latents only. No `role` field is needed — anything design-level is a
  grid, not a sampler.
- `{Domain}MixtureSampler` — the mixture combinator; **is a** sampler
  (callable, sits in `SCENARIOS`, passes the gym's `callable(self.scenario)`
  check, instances use `sampler_{name}`). Chosen over `ScenarioMixer` for the
  visible is-a relationship and the standard statistics term.
- `{Domain}ScenarioGrid` — the design-layer object. "Grid" is fully general:
  a hand-picked list of scenarios is a grid with one categorical axis.
- **`ScenarioSource`** — spec term (and optional per-domain type alias) for
  `{Domain}Scenario | {Domain}ScenarioSampler`; used by `SCENARIOS` values,
  the gym's `scenario` argument, mixture components, and grid cells.
- Per the portable-domain contract, all of these are **generated per-domain**
  from canonical reference implementations in the spec (domains import only
  numpy/typing — no harness-package dependency). Each class is small;
  duplication is cheap.

## 4. World layer

### 4.1 Treatment B is canonical (settled); A is deprecated

§4.3 test 1 already rules this: a draw keyed only on
`(episode_seed, seed_salt)` is scenario sampling in disguise. Retailer's
`DemandGenerator.episode_params()` violates it. **A is also present in the
frozen exemplar**: `inv_single`'s `EpisodeDemand` draws its support/weights
on an episode-only sub-stream (`_SUB_SUPPORT`) — a hyperparameter draw, not
"the first step of the same process" (§4.3 test 3). It evades the
conformance harness because `sample()` also keys on `period`; only the
*internal* episode-level sub-stream is episode-only.

Consequences:

- **Conformance check must be strengthened** (open, §8): the rule is not
  "`sample()` must key on `period`" but "a generator may not derive its
  distribution's parameters from an episode-only stream internally."
- **Pre-empt the information-hiding objection**: retailer's mean demand is
  *hidden* from the policy, and treatment A makes hiding automatic. The
  canonical answer: observability is the observation-mode layer's decision —
  a realized latent field on the scenario is **not** automatically observed.
- Name the two sub-cases of world latents (same machinery, different
  problems — affects what `{domain}_policy.py` takes as input):
  **observed latent** (contextual MDP; FNV's forecast) vs **hidden latent**
  (policy must infer; retailer's market size).

### 4.2 Seed hierarchy (settled; scheme v2)

`seed_salt` stays what it is everywhere else in the spec: a single per-domain
reproducibility knob (`repr=False`, `>= 1`, users never reason about it),
universal across a project/gym — **never** a decorrelation device.
Decorrelation between independent stochastic objects is structural: the
seed key encodes a path through one tree,

```
seed_salt (root, >= 1)
└─ episode_seed
   ├─ branch 0 (meta)      └─ substream                       [drawn once per episode]
   └─ branch 1 (intrinsic) └─ source └─ period └─ (draw)      [drawn per period]
```

with **one uniqueness rule applied at every node: children carry distinct
ids** (meta drawers under branch 0; source *instances* — not classes, so
repeated generators like OWMR's per-retailer leadtimes get distinct ids —
under branch 1; draw indices under a period). Ids default to 0 for a lone
child. The meta/intrinsic asymmetry is expressed by tree shape: branch 0 has
no period level. An intrinsic key without a period level is *syntactically
malformed* — treatment A (§4.1) becomes impossible to express, and the
conformance check reduces to a shape check.

**Canonical encoding is leaf-first (root last):**

```
intrinsic:  SeedSequence([(draw,) period, source, 1, episode_seed, seed_salt])
meta:       SeedSequence([substream, 0, episode_seed, seed_salt])
```

Why leaf-first and not the tree-reading root-first order (checked
2026-07-22, numpy 2.5.1):

1. **numpy's documented direction.** The parallel-RNG guide recommends
   varying ids *before* the fixed root seed: "`spawn` *appends* integers
   after the user-provided seed, so … it is a little bit safer to prepend
   your worker IDs rather than append them to avoid a collision."
2. **`SeedSequence` zero-pads short entropy lists to its 4-word pool**, so
   keys differing only in trailing zeros within the first 4 words collide:
   `[5,0] ≡ [5]`, `[7,0,0] ≡ [7]` (5-word lists are safe; leading and mid
   zeros always significant — all verified empirically). Root-first puts the
   routinely-zero leaf ids (period 0, substream 0, source 0) in trailing
   position — e.g. root-first meta `[salt, e, 0, 0] ≡ [salt, e]`, a silent
   collision. Leaf-first ends every key with `seed_salt`, constant and
   `>= 1`, so the truncation can never bite (hence the salt rule above).
3. **Statistical quality is order-independent** — 200k-key collision,
   uniformity, and adjacent-stream correlation tests pass identically both
   ways. The choice is structural, not about hash quality.

**Scheme versioning.** The key grammar is versioned per domain
(`seed_scheme: "v1" | "v2"`, declared in the scenarios module and the IR;
never mixed within a domain):

- **v2** (above) is canonical: the spec teaches only it; all newly generated
  domains and this repo's examples use it.
- **v1** is the frozen historical grammar of existing research domains, kept
  solely so recorded results stay reproducible: meta `[0, e, salt]` (or
  `[0, sub, e, salt]`), intrinsic orders varying per domain
  (`[stream, period, e, salt]`, `[period, stream, e, salt]`,
  `[sub, stream, period, e, salt]`). Documented in the spec in three lines;
  a v1 domain migrates only when its numbers are invalidated anyway.
- Even accidental cross-scheme mixing cannot collide: the v1 3-word meta key
  pads to `[0, e, salt, 0]`, which ends in 0 — no v2 key ends in 0.

**Validation from the wild**: `cnv` has a live collision today — its sampler
and its Gamma latent draw both key on `[0, e, salt]`, coupling `lambda_true`
deterministically to the sampled scenario parameters. Exactly the bug class
the hierarchy eliminates; cnv adopts v2 during the fix.

The meta/intrinsic symmetry is a **timing parallel, not a missing method**:

| layer | hook | called | seed key |
|---|---|---|---|
| intrinsic generator | `sample(ctx)` | every period, by the transition | `[(draw,) period, source, 1, episode_seed, seed_salt]` |
| meta-level drawer | `__call__(episode_seed)` | once per episode, by the gym at `reset()` | `[substream, 0, episode_seed, seed_salt]` |

No `episode_init()` lifecycle hook (rejected): the name connotes stateful
initialization, inviting per-episode state inside samplers — the exact
anti-pattern. Also rejected: a uniform `resolve(seed)` protocol giving fixed
scenarios a trivial identity method — episode-level ceremony on objects that
have none, to save one `callable()` branch in the gym.

**Purity rule**: a sampler is a stateless pure function `seed → Scenario`;
`sampler(s)` must return an equal scenario every time for the same `s`, and
must not mutate the sampler (vectorized envs share sampler instances).
"Runs once at episode start" is enforced by who calls it and by the seed key
(no `period` in it) — not by the method name.

**Vocabulary rule**: "episode" is gym-layer vocabulary. The world layer
defines a seed-indexed family of concrete problems (a distribution over
sample paths); the *gym* binds one episode to one seed at `reset()`.
World-layer definitions and docstrings phrase things as "per seed" / "per
draw", never "per episode". The parameter name `episode_seed` is kept for
cross-layer consistency with `SamplingContext` — a layer-neutral rename
(e.g. `path_seed`) would be API-breaking across all domains (though not
reproducibility-breaking; seed keys hash ints, not names) and stays a noted
option, out of scope here.

### 4.3 `{Domain}MixtureSampler` contract (settled)

```python
@dataclass
class {Domain}MixtureSampler:
    """Draws one component scenario source per seed."""

    # (weight, component); components may be fixed scenarios OR samplers,
    # including other mixtures (see below)
    components: list[tuple[float, ScenarioSource]]
    substream_id: int = 0                    # child id under branch 0 (§4.2)
    seed_salt: int = field(default=<domain_default>, repr=False)
    scenario_name: str | None = None
    desc: str = ""

    def __call__(self, episode_seed: int) -> {Domain}Scenario:
        # 1. draw the component index on the §4.2 meta key
        #    [substream_id, 0, episode_seed, seed_salt]
        # 2. if the chosen component is callable (a sampler), delegate
        #    episode_seed to it VERBATIM; else return the fixed scenario as-is
        ...
```

Weights are normalized in `__post_init__` (assert positive, sum > 0). A
world mixture's weights are a **modeling commitment** (Beergame's 50/50 is a
claim about demand) — see the litmus tests in §5; non-uniform *training*
pools are grid weights, not mixtures.

**Mixtures of mixtures are allowed** — deliberately: components are
`ScenarioSource`s and a mixture *is* one, so nesting type-checks with no
special case. Two points make this well-defined:

- Nesting adds **structure, not expressive power**: a mixture of mixtures
  collapses mathematically to a flat mixture (weights multiply through). Its
  value is reuse and narrative fidelity — a named world object
  (`sampler_demand_regimes`) can be embedded as one branch of a larger
  mixture without hand-flattening weight products, which is error-prone and
  destroys the named object you'd want to evaluate against.
- **Standalone equivalence**: because `episode_seed` is delegated verbatim
  and every drawer uses its own substream key (§4.2), a nested
  component behaves *identically* standalone and embedded — episode `e`
  through a branch equals episode `e` run on that component directly. Any
  branch is reproducible in isolation.

### 4.4 Grid-shaped world latents

A world latent may still be grid-*shaped* (nature draws uniformly from a
finite set). Shape does not decide the layer; the litmus tests (§5) do.

## 5. World vs. design: litmus tests (settled)

The rigorous distinction is the mathematical object, not the usage: a world
latent requires a **probability measure as a modeling commitment**; a grid
requires only a **finite set**, and any measure on it is a training
hyperparameter. Sampling-vs-enumeration follows as a corollary (you
integrate over measures, you enumerate sets), rather than being the
definition. Tests, in order of usefulness:

1. **Weights test.** "If I changed the weights, would I be changing the
   world model or just the training recipe?" World model → world latent;
   recipe → grid.
2. **Complete-problem test.** "Is one draw from this a complete problem
   someone might want a specialist policy for?" One FNV cost combo — yes →
   grid cell. One realized mean demand in retailer — no, the
   pricing-under-uncertainty problem is gone → world latent.
3. **Deployment test.** "Would the environment draw this afresh each episode
   in the real system?" Yes → world latent. Known and fixed at deployment,
   training across variants → grid.

Phase A interview consequence (the original motivation for the split): the
two are asked in **different phases**, not distinguished by a subtle answer.
Scenario elicitation asks only about world randomness (using test 3;
follow-up: observed or hidden?). The generalist question arrives later, at
training-target setup: "one policy for this instance, or one that works
across a range? over what range?" — and its answer builds a grid.

Edge cases pinned down:

- **Diagnostic enumeration of world latents** (performance conditional on
  mean-demand deciles) is slicing one problem's expectation for analysis —
  not a leaderboard basis, not a counterexample.
- **Continuous generality ranges** (h ∈ [0.1, 0.5], sim2real-style
  randomization) are discretized at grid construction time — stated v1
  limitation, matching practice (eval needs a finite test set anyway).
- **Non-uniform training pools** ("train 80% on the hard scenario") are grid
  weights, not world mixtures — test 1 sorts them.
- **Unbounded procedurally-generated families** (sudoku's boards carved from
  random solved grids) stay world-layer instance samplers even though no
  "nature" draws them: the family is not finite/enumerable, so the grid's
  generality claim cannot be stated, and eval *samples* fresh instances
  (integrates) rather than enumerating. The grid requires a finite,
  enumerable generality claim.
- **Domain randomization** (family params deliberately hidden from the
  policy) is generalist training over a grid; the hiding is an
  observation-mode decision and should be stated intent, not an accident.

## 6. Design layer: `{Domain}ScenarioGrid` (settled)

Sampled in training, enumerated in eval, never both for world latents — the
grid is the set describing the *generality* of a generalist; its use as a
training pool is incidental.

```python
@dataclass
class {Domain}ScenarioGrid:
    """A finite set of complete scenario sources — the generality target
    of a generalist policy. NOT callable: deliberately fails the gym's
    sampler check so it cannot be passed where a ScenarioSource belongs."""

    # (cell_id, source); cell_id derives from axis values ("h=0.2,b=2.0")
    # and names the per-cell leaderboard row
    cells: list[tuple[str, ScenarioSource]]
    grid_name: str | None = None
    desc: str = ""

    @classmethod
    def from_axes(cls, base: {Domain}Scenario, axes: dict[str, list], ...):
        """Cross product of constant overrides on a base scenario,
        row-major in axis-declaration order."""

    @classmethod
    def from_sources(cls, sources: list[ScenarioSource], ...):
        """Explicit list — a grid with one categorical axis; definition
        order preserved; cell ids default to the sources' scenario_names."""

    def as_sampler(self, weights=None, substream_id=0) -> {Domain}ScenarioSampler-like:
        """Derive an ordinary sampler over the cells (uniform by default)
        for training. The gym receives this, never the grid. Keys per the
        domain's §4.2 scheme; a v1 domain (fnv, adi_flex) preserves its
        recorded numbers by keeping the v1 legacy key AND replicating the
        previous draw procedure (cell order + rng call pattern)."""

    # enumeration is data access, not behavior — no enumerate() method:
    def __len__(self): ...                   # number of cells
    def __iter__(self): ...                  # yields (cell_id, source) in canonical order
    def __getitem__(self, key): ...          # by cell_id (str) or index (int)
```

**Enumeration contract.** The grid *is* the enumeration — a finite set with
named cells — so it is exposed as data with the standard dunders above;
`as_sampler()` stays the only verb (sampling is derived behavior,
enumeration is what the object already is). **Canonical, deterministic cell
order is part of the contract**, for three reasons:

1. **`as_sampler()` correctness** — the derived sampler draws an index `k`
   on its stream and returns `cells[k]`; unstable order would map the same
   `(episode_seed, stream_id)` to different cells across runs/machines and
   silently break reproducibility.
2. **Leaderboard stability** — per-cell rows (`{grid_name}/{cell_id}`) come
   out in the same order every run.
3. **Cell identity** — `cell_id` is a pure function of the axis values
   (`"h=0.2,b=2.0"`, axes in declaration order), never of insertion
   accidents.

**Enumeration yields sources, not scenarios.** If a cell is itself a world
sampler (retailer crossed over cost structures), the grid does not draw it —
the eval driver decides episodes and seeds per cell. The grid never touches
an `episode_seed`: pure design-layer data, consistent with the vocabulary
rule (§4.2).

**Common random numbers fall out for free.** Every cell is a source keyed by
the same seed protocol, so the eval driver can reuse one identical block of
episode seeds across all cells, making per-cell comparisons paired rather
than independent. Stated in the spec so drivers do it deliberately.

**Family-level attributes.** Like §5.2 samplers, a grid exposes the fixed
attributes all cells agree on (adi_flex: `sigma_enabled`, `N`) so the gym
wrapper and eval scripts can build spaces and read dimensions without
resolving a cell first; `from_axes`/`from_sources` assert the agreement.

- Cells are `ScenarioSource`s, not just scenarios: a generalist across cost
  params of a domain *with* a world latent (retailer crossed over cost
  structures, each cell keeping its market-size sampler) composes legally in
  exactly one direction under the layering rule (§2).
- **Not callable** is the isolation made mechanical: implementing
  `__call__(episode_seed)` would let a grid pass the gym's `callable` check
  and re-blur the layers. Training incidentally samples via `as_sampler()`;
  eval enumerates `cells`; the grid itself is just the set.
- Per-cell evaluation resolves what an earlier draft deferred as the
  "enumeration gap": enumeration was never a missing sampler feature — it is
  the native mode of the object that was missing. Dataset-backed instance
  pools (real traces, benchmark files) are grids via `from_sources`
  (respecting the portable-domain contract: folder-relative data) — unless
  the traces model an empirical world distribution, in which case the litmus
  tests put them in a world sampler.
- Gates/leaderboard: grids produce per-cell rows (baselines knowing cell
  params are per-instance baselines — not cheating); world latents produce a
  single expected metric (a baseline knowing the latent is **clairvoyant** —
  an upper bound, labeled as such).
- Deployment: a generalist `{domain}_policy.py` takes the cell parameters as
  input; a world latent must never be required by it.

## 7. Boundary clarifications to add to the spec

- **Within-episode regime switching** (Markov-modulated demand; a "mixture"
  that can flip at t=17) is **intrinsic** randomness — a generator in
  `{domain}_uncertainty.py`, not a scenario concern. The line: mixture drawn
  at episode start = world sampler; regime evolving within the episode =
  intrinsic generator.
- **Nonstationary training distributions** (curriculum, annealed
  randomization) are **out of scope** for both layers: samplers keyed only on
  `episode_seed` are deliberately stationary and reproducible; which
  scenario source or grid weights to use per episode is the training
  driver's business. No global episode counters inside samplers.
- **Adversarial / agent-conditioned scenario selection** breaks the MDP
  contract entirely — one-line exclusion.

## 8. Open layer: conformance

- **v2 domains — template shape checks (§4.2)**: every intrinsic key
  matches `[(draw,) period, source, 1, e, salt]` (the mandatory period
  level makes treatment A syntactically impossible), every meta key matches
  `[substream, 0, e, salt]`, child ids distinct at every tree level,
  `seed_salt >= 1`, `seed_scheme` declared and unmixed.
- **v1 domains — frozen-key checks**: flag generators whose internal
  sub-streams are keyed on `(episode_seed, seed_salt)` only (treatment A
  hiding inside a two-stream generator — retailer, cnv, inv_single today);
  at most one bare `[0, e, salt]` meta drawer per module, further ids
  unique. Would have caught cnv's live sampler/latent collision.
- Registry smoke test: call every `SCENARIOS` entry that is a sampler with a
  handful of seeds; each call must terminate in a concrete
  `{Domain}Scenario`. Doubles as the guard against the one pathological case
  nesting enables — a self-referential mixture built by post-construction
  mutation.
- Purity/determinism (§4.2): call each sampler twice with the same seed and
  require equal scenarios — catches hidden state and impure draws.
- Layering (§2): grids must not appear in `SCENARIOS`, world samplers carry
  no enumeration API, `_mdp`/`_gym` must not import `{domain}_grids.py`.

## 9. Open layer: IR representation

`mdp_ir/schema.py` today has only `ScenarioConstant` + per-instance literal
overrides; stream 0 is reserved in a comment but nothing produces it. The IR
cannot express world samplers, mixtures, or grids, so Phase A cannot capture
a retailer- or Beergame-shaped problem (nor a generalist target), and the
interpreter / differential runner cannot check one.

Sketch (to be settled): a sampler block on `Scenario` for world latents
(reusing the distribution vocabulary from uncertainty sources, staged at
episode start on stream 0), a mixture form referencing named instances or
nested samplers with weights, and a separate **grid block outside the
`scenario` node** — the IR should mirror the two-layer split, with grids
expressed as axes over scenario constants (dovetailing with the existing
`axis` tag on `ScenarioConstant`). Sampler and mixture entries declare their
meta-level `substream_id` (§4.2) just as uncertainty sources declare their
`stream_id`, with per-level uniqueness validated at schema load. The IR also
records the domain's `seed_scheme` (v1 frozen keys / v2 hierarchy) so the
interpreter and differential runner replicate draws exactly.

## 10. Migration consequences (open layer)

Full survey of both repos (2026-07-22). The §4.2 scheme flag splits the cost
cleanly: this repo's examples adopt v2 and regenerate their cheap scripted
gates once; downstream research domains stay v1 and keep their expensive
recorded results bit-for-bit — the only re-run is cnv, as a bug fix.

| project | change | numbers preserved? |
|---|---|---|
| `examples/inv_single` | `EpisodeDemand` → `InvSingleScenarioSampler` (world latent, realized support held on the scenario); `scenario_paper_*` become sampler entries in `SCENARIOS`; gym gains the `callable` branch (has none today); schema + differential + MANIFEST gates re-expressed; **adopts scheme v2** (the exemplar must demonstrate the canonical hierarchy) | **regenerated once** — gates are cheap scripted runs; supersedes the earlier `substream_id=1` preservation trick |
| `examples/dynamic_pricing` | **migrated to v2** (2026-07-22): pure re-keying — `source_id` + `intrinsic_key()` on the arrivals generator, `SEED_SCHEME`/salt rule, IR `seed_scheme: v2` with source id 0. No latents, no samplers, no pruning needed | trajectories re-keyed (gates are commands; trained artifacts reproducible per README) |
| `fnv` | design draw → `FnvScenarioGrid` (`from_axes`, 6×9×10 = 540 cells); `"FNV-aMMFE"`/`"FNV-mMMFE"` move `SCENARIOS` → `GRIDS`; training via `as_sampler()`; eval gains per-cell rows; stays **v1** | **yes — demonstrated** (§11 spike): `as_sampler(draw="v1_axes")` reproduces the retired sampler bit-for-bit over 700+ seeds |
| `adi_flex` | uniform draw over member list → textbook `from_sources` grid; stays **v1** | **yes**, same condition as fnv |
| `sudoku` | stays a world-layer instance sampler (see §5 edge case); stays **v1** | **yes** — lone sampler keeps its key |
| `retailer` | `DemandGenerator.episode_params()` → `RetailerScenarioSampler` (hidden world latent); stays **v1** | **yes, exactly** — the extracted sampler reuses the `[0, e, salt]` key bit-for-bit |
| `cnv` | double migration: Gamma latent (treatment A) → world sampler, plus existing sampler; **adopts v2** during the fix | **no — deliberately**: today both draws collide on `[0, e, salt]` (§4.2); the current numbers are defective |
| `owmr`, `topk_id`, `2048` | fixed scenarios only (topk_id says so explicitly) — vocabulary/naming conformance at most | n/a |

Harness impact (this repo): `mdp_ir` schema additive (old schemas stay
valid), but interpreter + differential must implement stream-0 semantics —
the largest new code; `mdp_conformance` gains the §8 checks — **sequencing
trap**: the episode-only-stream check fails inv_single/retailer/cnv until
they migrate, so checks land with or after migrations (or behind a grace
flag); `mdp_gates` gains per-cell rows; skill Phase A interview split per
§5. `cases/` are unaffected retroactively. Downstream repos coordinate per
the release rule in CLAUDE.md.

## 11. Domain-layer spike on inv_single (done 2026-07-22)

Prototyped the full v2 design on a copy of `examples/inv_single`
(`scratch/inv_single_v2/`, gitignored; frozen example untouched; the copy
running standalone also re-confirmed the portable-domain contract — run
`python spike_test.py` from that folder to reproduce). 20/20 checks pass: registry
smoke + double-call purity, distinct-seed / distinct-substream decorrelation,
v2 key shapes (leaf-first, salt last), the three mdp event-sequence smoke
tests, full-episode reproducibility through a sampler, mixture standalone
equivalence over 200 seeds + mixture-of-mixtures, one sampler instance
shared by interleaved gyms without cross-contamination, and `phi()` on
realized scenarios.

Findings to fold into the spec text:

- **Family-attribute mirroring works and should be the stated rule**: the
  sampler exposes family-level attributes under the *same names* as the
  scenario fields (`horizon`, `leadtime`, `allow_backlog`, and a `demand`
  property returning a bounds object with the generator's read API —
  `max()`, `mean()`, `is_discrete`). The gym's space-building code needed
  **zero changes**; only `reset()` gained the resolve branch and `step()`
  switched to the per-episode scenario.
- **Ship per-domain key helpers** (`intrinsic_key()` / `meta_key()`) in the
  reference implementation so the v2 template cannot drift; conformance can
  then flag any `SeedSequence` construction outside the helpers.
- `EpisodeDemand` decomposed cleanly into `InvSingleEpisodeDemandSampler`
  (world latent) + `FixedDistributionDemand` (intrinsic, realized) — and the
  realized generator *gains* a working `phi()`, which `EpisodeDemand` could
  not offer (its marginal required integrating over episode randomness).
  Baselines get strictly stronger.
- Per-instance `source_id` with class-level defaults (demand 0, leadtime 1)
  plus a distinctness assert in the scenario's `__post_init__` was enough;
  no ceremony in scenario definitions.
- Sequential `substream_id` assignment across the five paper samplers (0–4)
  is mild ceremony; an in-module `_check_meta_substreams()` assert next to
  the registry guards it locally, conformance externally.
- Not exercised (next layers): `{domain}_benchmark_*` scripts take one
  concrete scenario — under a sampler entry the eval protocol must realize
  per episode (part of §9/§10 work); `ir_adapter`/schema untouched, pending
  the IR sampler block.

### 11.1 Design-layer spike on FNV (done 2026-07-22)

Prototyped `FnvScenarioGrid` against the real FNV domain
(`scratch/fnv_v2/`, gitignored; `fnv_scenarios.py` copied *unmodified* so
the retired `FnvScenarioSampler` serves as the v1 reference). 18/18 checks
pass: 540-cell shape, unique axis-derived cell ids, deterministic row-major
order across constructions, the full enumeration contract (not callable,
`__getitem__` by id and index, iteration, family attrs without resolving a
cell), canonical index draw with purity/marginal-coverage/weights, CRN
across cells (same seed block → identical standardized draws), and
`from_sources`. Findings:

- **Number preservation is demonstrated, not argued**:
  `as_sampler(draw="v1_axes")` reproduces the retired sampler **bit-for-bit
  over 700+ seeds** (dynamics fields + derived signal schedule). The
  mechanism: the old sampler draws one `rng.choice` per axis in declaration
  order, not a cell index — so the v1-compat mode must replicate that call
  pattern. On a full cross product the two are the same distribution
  (uniform-over-cells = independent-uniform-per-axis); preservation is
  purely about the rng call pattern. Spec note for §6: a migrating v1 grid
  offers `draw="v1_axes"`; new work uses the canonical index draw.
- One real grid was 6×9×10 = **540** cells (not 510 as remembered) —
  corrected throughout this doc.
- The old sampler left `scenario_name=None` on generated scenarios; grid
  cells carry their `cell_id` as `scenario_name` — an improvement (rollout
  logs become self-identifying), and benign for dynamics.
- CRN paired evaluation is real: FNV's signal generator keys stdev-free, so
  two cells under the same episode-seed block produce identical
  standardized draws — per-cell comparisons are exactly paired.

## 12. Suggested implementation order

1. ~~Spec rewrite~~ **done 2026-07-22**: `MDP_PROJECT_SPEC.md` §1 (layout +
   chain gain `{domain}_grids.py`), §2 (naming), §4.2 (per-instance
   `source_id`, `intrinsic_key()`), §4.3 (strengthened boundary + regime /
   curriculum lines), §5 restructured (5.2 world sampler + mirroring rule +
   observed/hidden, 5.3 mixture, 5.4 registry, 5.5 litmus tests, 5.6 grid),
   §6.3 (seed tree v2 + scheme flag), §7 (`ScenarioSource` gym contract),
   §8 (generalist training via `GRIDS`), §9.6 (three-target eval dispatch
   with CRN), plus a transition note in the header.
2. ~~Skill (Phase A interview)~~ **done 2026-07-22**: `SKILL.md` Phase A
   step 3 (three-bin randomness classification: intrinsic incl. regime
   switches / world latent with deployment test + observed-hidden follow-up
   + extract-hidden-treatment-A note / training-target explicitly deferred)
   and step 4 (specialist-vs-generalist in plain words, grid as the
   generalist's range); Phase B Stage 0 (target may be a `GRIDS` entry;
   strategy wording) and Stage 4. Stage 1 gains an explicit **transitional
   pin**: generated seed keys must match the interpreter's pre-v2 key until
   step 4 lands — do not upgrade generated keys ahead of `mdp_ir`.
3. ~~Conformance checks (§8)~~ **done 2026-07-22**: `mdp_conformance` gains
   `scheme.declared` (SEED_SCHEME present/valid/unmixed; undeclared → v1
   with a WARN nudge — the grace mechanism that resolves the sequencing
   trap), `scenario.samplers` (registry smoke + double-call purity, both
   schemes), `scheme.v2_ids` (salt >= 1, distinct substream/source ids),
   `scheme.v2_keys` (AST drift guard: SeedSequence only via
   `intrinsic_key()`/`meta_key()`), `grids.*` (no grid in SCENARIOS; GRIDS
   enumerable, non-callable, unique cells, pure `as_sampler()`), and the
   layering check now forbids `_grids` imports in model layers. Loader
   knows the `_grids.py` role. Validated: frozen examples stay green
   (exit 0, WARN only), `scratch/inv_single_v2` passes all v2 checks,
   `scratch/fnv_v2` (+ copied `fnv_mdp`/`fnv_gym`) passes v1 + grids
   checks, and planted violations (duplicate substream, raw SeedSequence)
   FAIL with precise diagnostics. Full regression suite green.
4. ~~IR schema + interpreter + differential support (§9)~~ **done
   2026-07-22** (additive; every pre-existing IR bit-identical, default
   `seed_scheme: "v1"`): schema gains `seed_scheme`, `Scenario.samplers`
   (`SampledConstant` draws targeting *existing* constants, so all expr
   validation is unchanged; `hidden` flag feeds the requires_memory
   derivation and bars the constants from observation exprs),
   `Scenario.mixtures` (usable anywhere an instance name is), root-level
   `grids` (`ScenarioGrid.cells()`, row-major, axis-derived ids), v2
   branches in `UncertaintyStage.seed_key`, and scheme rules (v2 forbids
   episode-realization stages — treatment A rejected at validation; v1
   keeps `stream_id >= 1`). Interpreter gains `_meta_seed_key`, v2
   `_numeric_seed_key`, `_sample_family` (+ `choice_without_replacement`,
   `normalized_uniform_weights` recipe families), and per-episode
   `_episode_setup` (mixture component draw + sampler draws before
   anything reads constants/horizon). interpreter_test grows 25 → 36
   checks incl. mixture standalone-equivalence verbatim vs pure
   components. **Deferred to step 5**: differential runs against
   sampler/mixture instances (needs a migrated domain adapter) and the
   `MDP_IR_SAMPLE.md` annotated-reference update; the SKILL.md Stage-1
   seed-key pin lifts only when the inv_single migration proves v2
   end-to-end through the differential gate.
5. **inv_single migrated 2026-07-22** — the design proven end to end:
   domain files from the §11 spike (with cleanup: unused `NormalDemand`
   dropped; registry pruned to `simple`, `simple-k`, `paper_stochastic`,
   `paper_lost_sales` + the `test_*` smoke scenarios — the seven
   referenced-nowhere variants removed); IR rewritten to `seed_scheme: v2`
   (demand source = plain categorical over sampled `demand_vals`/
   `demand_probs`; two hidden `paper_demand` samplers, substreams 0/1 for
   base/`lost_sales` — sampler `instances` may name `""` = base;
   `requires_memory` suggested=True with human_override value=False);
   adapter resolves the shared sampler per episode. Differential is
   bit-exact for base and `lost_sales` (the step-4 deferred piece);
   conformance v2 checks all PASS; `mdp_ir.differential` default salt is
   now 1 (v2 domains assert salt >= 1; MATCH is salt-invariant). The
   SKILL.md Stage-1 transitional pin is **lifted** — generated domains now
   use v2. Remaining: `MDP_IR_SAMPLE.md` annotated-reference update;
   downstream rl_test migrations (retailer, cnv, fnv, adi_flex) per §10 at
   their own pace.

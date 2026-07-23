# MDP Project Specification

Canonical patterns for MDP simulation domains. Shipped conformant examples:
`examples/inv_single` and `examples/dynamic_pricing`; hypothetical domains
(a one-warehouse multi-retailer system, a tile-merging board game, …) are
described inline where a pattern needs an illustration richer than the
examples provide. All new MDP domains should follow these conventions.

> **Transition note (2026-07-22):** the scenario architecture (§5) and seed
> scheme (§6.3) were redesigned — world/design layers, mixture samplers,
> grids, seed tree v2. Both shipped examples follow the new design, which is
> specified in full in §5 and §6.3 below.

---

## 1. File Layout

Each domain lives in its own subfolder `{domain}/`. Every file is prefixed with the domain name.

| File | Purpose |
|---|---|
| `{domain}_exceptions.py` | Custom exception hierarchy (optional) |
| `{domain}_uncertainty.py` | Stochastic primitives: the `SamplingContext` protocol and all `{Source}Generator` classes (demand, leadtime, …) — **omit entirely when the dynamics are deterministic** (§4.3) |
| `{domain}_scenarios.py` | **World layer**: the `{Domain}Scenario` / `{Domain}ScenarioSampler` / `{Domain}MixtureSampler` classes, their predefined instances, and the `SCENARIOS` registry |
| `{domain}_grids.py` | **Design layer** (optional): the `{Domain}ScenarioGrid` class and the `GRIDS` registry — generality targets for generalist training (§5.6) |
| `{domain}_mdp.py` | Core MDP simulator: `{Domain}State` and the state-transition functions |
| `{domain}_gym.py` | Gymnasium single-agent wrapper |
| `{domain}_ppo_train.py` | SB3 PPO training (drop "gym" if it is the only env type) |
| `{domain}_ppo_tune.py` | Optional thin wrapper over the repo-level `mdp_tuning` harness, pre-filling domain defaults (e.g. `--metric`) |
| `{domain}_dreamerv3_train.py` | RLlib DreamerV3 training |
| `{domain}_ppo_eval.py` | Evaluate a trained RL model over the full parameter grid |
| `{domain}_benchmark_{method}.py` | Non-RL benchmark solver — `{method}` names the method (`lp`, `dp`, `myopic`, `greedy`, `fluid`, or a domain-custom heuristic). One per method; a solver may expose several related policies via `--policy` (§9.7) |
| `{domain}_benchmark_{method}_eval.py` | Evaluate a benchmark over the full parameter grid, same TSV format as the RL eval |
| `{domain}_policy.py` | Deployable policy wrapper over the trained artifact (§12) |

### 1.1 Dependency chain

The three domain-model files form a strict, acyclic layering — each imports only from the ones to its left:

```
{domain}_uncertainty  ←  {domain}_scenarios  ←  {domain}_mdp
   (generators,            ({Domain}Scenario,       ({Domain}State,
    SamplingContext)        SCENARIOS)               transitions)
```

- **`{domain}_uncertainty.py`** is the base: it has no domain imports (only `numpy` and `typing`). It must **not** import `{Domain}State` — see the `SamplingContext` protocol in §4.1.
- A domain with **deterministic dynamics** has no `{domain}_uncertainty.py` at all; the chain shortens to `{domain}_scenarios ← {domain}_mdp`. Initial conditions belong to the scenario, and per-episode variety comes from a `{Domain}ScenarioSampler` (§4.3, §5.2) — e.g. a Sudoku domain, where the puzzle *is* the scenario and placements are deterministic.
- **`{domain}_scenarios.py`** imports the generator classes from `{domain}_uncertainty` and composes them into `{Domain}Scenario` instances.
- **`{domain}_mdp.py`** imports `{Domain}Scenario` from `{domain}_scenarios` (for type annotations) and the generator classes from `{domain}_uncertainty` (only where it needs to construct them, e.g. its `__main__` smoke test).
- **Gym wrappers** sit on top of all three: `{Domain}State` and transition functions from `{domain}_mdp`, `{Domain}Scenario` from `{domain}_scenarios`, generator classes from `{domain}_uncertainty` as needed.
- **`{domain}_grids.py`** (optional, §5.6) sits *beside* the chain, not in it: it imports only from `{domain}_scenarios`, and is imported **only by training / eval / tuning drivers** — never by `{domain}_mdp` or `{domain}_gym`. The gym receives a plain scenario source (a scenario, or a sampler derived via `grid.as_sampler()`); grids stay invisible to the model layers.

---

## 2. Naming Conventions

| Concept | Pattern | Example |
|---|---|---|
| Generator base class | `{Source}Generator` | `DemandGenerator`, `LeadtimeGenerator` |
| Generator leaf class | `{Variant}{Source}` | `PoissonDemand`, `NormalDemand`, `DiscreteLeadtime` |
| Sampling-context protocol | `SamplingContext` or `{Source}Context` | `SamplingContext`, `SpawnContext`, `DemandContext` |
| Scenario config class | `{Domain}Scenario` | `InvSingleScenario` |
| Scenario sampler class (world latents, §5.2) | `{Domain}ScenarioSampler` | `InvSingleScenarioSampler` |
| Mixture sampler class (§5.3) | `{Domain}MixtureSampler` | `InvSingleMixtureSampler` |
| Scenario grid class (design layer, §5.6) | `{Domain}ScenarioGrid` | `InvSingleScenarioGrid` |
| Scenario source (term / type alias) | `ScenarioSource` | `{Domain}Scenario \| {Domain}ScenarioSampler` (any callable sampler counts) |
| MDP state class | `{Domain}State` | `InvSingleState` |
| Gym env class | `{Domain}Env` | `InvSingleEnv` |
| Scenario instances | `scenario_{name}` | `scenario_simple` |
| Scenario sampler instances | `sampler_{name}` | `sampler_default` |
| Grid instances | `grid_{name}` | `grid_costs` |
| Scenario registry dict | `SCENARIOS` | `SCENARIOS = {"simple": scenario_simple, "random": sampler_default, ...}` |
| Grid registry dict | `GRIDS` | `GRIDS = {"cost-sweep": grid_costs, ...}` |
| Scenario registry key | `scenario_name` | `scenario_name="simple"` |
| Scenario description | `desc` | `desc="zero lead times everywhere"` |
| Constructor argument for scenario | `scenario` | `def __init__(self, scenario, ...)` |
| Attribute storing scenario on env | `self.scenario` | `self.scenario: InvSingleScenario` |

Which file each class lives in:

- **`{domain}_uncertainty.py`**: `SamplingContext`, `{Source}Generator` and its leaf classes, the `intrinsic_key()` helper.
- **`{domain}_scenarios.py`**: `{Domain}Scenario`, `{Domain}ScenarioSampler`, `{Domain}MixtureSampler`, the `meta_key()` helper, `scenario_{name}` / `sampler_{name}` instances, `SCENARIOS`.
- **`{domain}_grids.py`**: `{Domain}ScenarioGrid`, `grid_{name}` instances, `GRIDS`.
- **`{domain}_mdp.py`**: `{Domain}State`, transition functions.

**Never use `param`, `params`, or `param_*`** anywhere in the codebase.

---

## 3. Exceptions (`{domain}_exceptions.py`)

**This file is optional.** A domain with no domain-specific constraint failures
may omit it and rely on plain `assert`s / built-in exceptions (as both shipped
examples do). Provide it when the MDP raises structured, catchable errors that
a caller may want to distinguish (e.g. an invalid-move error in a board game,
or an infeasible-allocation error in a multi-entity logistics domain).

```python
class {Domain}Error(Exception):
    """Base exception for {domain} problems."""
    pass

class SomeConstraintViolationError({Domain}Error):
    """Descriptive message about what was violated."""
    pass
```

All constraint-checking code in the MDP raises these instead of generic exceptions.

---

## 4. Uncertainty Primitives (`{domain}_uncertainty.py`)

This module holds every independent source of intrinsic MDP randomness (demand, leadtime, …) as an explicit generator class, plus the `SamplingContext` protocol they sample against. It is the **base of the dependency chain** (§1.1): it imports only `numpy` and `typing`, never `{Domain}State`, `{Domain}Scenario`, or the MDP.

**This file is optional.** It exists only when the dynamics themselves are stochastic. A domain where all randomness is problem-instance selection (§4.3) has deterministic dynamics and omits this file entirely — that is a first-class case, not a degenerate one.

### 4.0 Module docstring

```
"""{Domain} stochastic primitives.

Defines the SamplingContext protocol and all generator classes for
{source1}, {source2}, .... These are the stochastic building blocks that
{Domain}Scenario is composed from; they have no dependency on the MDP dynamics.

Dependency order: {domain}_uncertainty  ←  {domain}_scenarios  ←  {domain}_mdp
"""
```

### 4.1 `SamplingContext` protocol

Generators need `period`, `episode_seed`, and `seed_salt` to build a reproducible seed. Those fields live on `{Domain}State` — but `{Domain}State` is defined in `{domain}_mdp.py`, which sits at the **top** of the dependency chain. Importing it here would create a cycle (`uncertainty → mdp → scenarios → uncertainty`).

Resolve this with **structural typing**: declare the minimal interface as a `typing.Protocol` and annotate `sample()` against it. `{Domain}State` satisfies it structurally — no import required.

Name it `SamplingContext` by default. A domain with a single dominant source of randomness may instead name it after that source — `{Source}Context` (e.g. `SpawnContext` in a tile-spawning board game) — when that reads more naturally; the structural-typing mechanism is identical either way.

```python
from typing import Protocol

class SamplingContext(Protocol):
    """Minimal interface a state must expose for generator.sample().

    {Domain}State satisfies this structurally — no import of the concrete type
    is needed here, avoiding circular dependencies.
    """

    period:       int
    episode_seed: int
    seed_salt:    int
```

### 4.2 Generator classes

Each independent source of randomness is modeled as an explicit generator class — not as inline math in `advance()`. This makes the randomness structure visible, testable, and extensible.

```python
def intrinsic_key(
    source_id: int, ctx: SamplingContext, draw: int | None = None
) -> list[int]:
    """v2 intrinsic seed key: [(draw,) period, source_id, 1, episode_seed, seed_salt].

    Leaf-first so the trailing word is always seed_salt (>= 1); see §6.3.
    """
    key = [ctx.period, source_id, 1, ctx.episode_seed, ctx.seed_salt]
    if draw is not None:
        key.insert(0, draw)
    return key


class {Source}Generator:
    """Abstract base for {source} generators."""

    is_discrete: bool
    source_id:   int  # child id under branch 1 of the seed tree (§6.3); per instance

    def sample(self, ctx: SamplingContext) -> float | int:
        """Draw one sample, fully reproducible given ctx.period / episode_seed."""
        raise NotImplementedError

    def mean(self) -> float:
        """Expected value, for deterministic planners (e.g. LP)."""
        raise NotImplementedError

    def max(self) -> float:
        """Upper bound on the sampled value (used for space construction)."""
        raise NotImplementedError


class Normal{Source}({Source}Generator):
    """Normally distributed {source}."""

    is_discrete = False   # whether the value is integer-valued

    def __init__(self, mu: float, sigma: float, source_id: int = 0) -> None:
        self.mu        = mu
        self.sigma     = sigma
        self.source_id = source_id

    def sample(self, ctx: SamplingContext) -> float:
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, ctx))
        )
        return rng.normal(loc=self.mu, scale=self.sigma)

    def mean(self) -> float:
        return self.mu

    def max(self) -> float:
        return self.mu + 4.0 * self.sigma
```

- **`sample(ctx)` owns the seed construction, via `intrinsic_key()`** — `advance()` calls `scenario.demand.sample(state)` and never builds an rng itself; generators never build a raw `SeedSequence` outside the helper (so the §6.3 template cannot drift, and conformance can flag stray keys). The `ctx` argument is any `SamplingContext` (in practice the `{Domain}State`).
- **`source_id`** is the generator *instance's* child id under branch 1 of the seed tree (§6.3) — per instance, not per class. Give each source a conventional default (demand 0, leadtime 1, …); multi-entity domains give each entity's generator instance its own id (e.g. `base + entity_index`) so two retailers with identical parameters still draw independent values. `{Domain}Scenario.__post_init__` asserts the composed instances carry distinct ids.
- A source that draws **several times per period** distinguishes the draws with the optional leading `draw` index — never by mutating internal state.
- **`mean()`** returns the expected value for deterministic planners (e.g. LP/DP benchmarks).
- **`max()`** returns a practical upper bound (e.g. mean + 4σ); used by the gym wrapper to set action/observation space bounds without isinstance checks. Leadtime generators additionally expose **`min()`**.
- **`is_discrete`** (class attribute) records whether samples are integer-valued, so consumers can pick integer vs. float spaces without isinstance checks.
- The generator instance is stored as a field on `{Domain}Scenario` (e.g. `scenario.demand`, `scenario.rt_leadtimes[i]`).

### 4.3 Intrinsic vs. meta-level randomness — the boundary

*Intrinsic* randomness unfolds **during** the dynamics: it is realized per
transition and keyed by something that advances within the episode (`period`,
a draw counter). *Meta-level* randomness selects **which problem instance**
the episode poses: it is realized once at episode start and keyed by
`episode_seed` only. Only intrinsic randomness belongs in this file;
meta-level randomness is `{Domain}ScenarioSampler` territory (§5.2, branch 0
of the seed tree, §6.3). Under the v2 seed scheme the boundary is enforced
by grammar: an intrinsic key *must* contain the period level, so a
meta-level draw cannot be expressed inside a generator without leaving the
key template. Apply the tests below to every source whose classification is
not obvious:

1. **Timing.** A draw that depends only on `(episode_seed, seed_salt)` — no
   `period`, no per-transition counter — is scenario sampling in disguise.
   Model it as a `{Domain}ScenarioSampler` and make the scenario hold the
   *realized* value. This includes draws hiding *inside* a two-stream
   generator whose `sample()` keys on `period` but whose internal
   distribution parameters come from an episode-only sub-stream (the
   conformance harness flags both forms).
2. **Nameability.** If a fixed realization of the draw is something you would
   hand-craft, name, or benchmark against (a specific puzzle board, a specific
   cost structure), it is a scenario. A scenario must be a fully concrete
   problem instance (§5.1); distributions over instances are expressed only
   through samplers.
3. **Episode-start draws.** An episode-start draw is intrinsic only when it is
   the first step of the *same stochastic process* that continues per
   transition (a tile-merging board game's initial tile spawn, drawn by the
   same `SpawnGenerator` as every later spawn). A one-shot draw with no
   per-transition sibling defaults to meta-level.

Two boundary lines that trip the unwary, stated explicitly:

- **Within-episode regime switching is intrinsic.** A latent that can *change
  during* the episode (Markov-modulated demand, a regime that flips at t=17)
  is a generator here, however mixture-like it looks. The line: a mixture
  drawn once at episode start = world sampler (§5.3); a regime evolving
  within the episode = intrinsic generator.
- **Nonstationary training distributions are out of scope everywhere.**
  Samplers keyed only on `episode_seed` are deliberately stationary and
  reproducible; curriculum / annealed randomization is the training driver's
  business (which scenario source or grid weights to use per episode) — no
  global episode counters inside generators or samplers. Adversarial /
  agent-conditioned scenario selection breaks the MDP contract entirely.

A domain where **all** randomness is meta-level has deterministic dynamics and
no `{domain}_uncertainty.py` at all (e.g. a Sudoku domain: the puzzle is the
scenario; placements are deterministic).

---

## 5. Scenarios (`{domain}_scenarios.py`) and Grids (`{domain}_grids.py`)

Scenario-level structure lives in **two layers with a strict direction**:

- **World layer** (`{domain}_scenarios.py`): models the *problem*. Fixed
  `{Domain}Scenario` instances, plus samplers for **world latents only** —
  draws nature makes once per episode (§5.2, §5.3). Registry: `SCENARIOS`.
- **Design layer** (`{domain}_grids.py`, optional): models the *experiment*.
  A `{Domain}ScenarioGrid` is a finite set of complete scenario sources — the
  generality target of a generalist policy (§5.5, §5.6). Registry: `GRIDS`.

A grid may contain anything from the world layer; nothing in the world layer
may contain or import a grid, and the `_mdp` / `_gym` layers never see one
(§1.1). The union `{Domain}Scenario | {Domain}ScenarioSampler` is called a
**`ScenarioSource`** — the type of `SCENARIOS` values, the gym's `scenario`
argument, mixture components, and grid cells.

`{domain}_scenarios.py` imports generator classes from `{domain}_uncertainty`
and composes them:

```python
from {domain}_uncertainty import {Source}Generator, Normal{Source}, ...
from {domain}_exceptions import ...
```

### 5.1 `{Domain}Scenario`

```python
@dataclass(slots=True)
class {Domain}Scenario:
    """{Domain} scenario parameters."""

    # structural / sizing parameters      (numerical → grid axes)
    ...

    # stochastic model instances (demand, leadtime, etc.)
    ...

    # cost / reward parameters            (numerical → grid axes)
    ...

    # categorical flags & enums           (scenario modes; see below)
    some_flag: bool = False

    # reproducibility
    seed_salt: int = field(default=<default_int>, repr=False)

    # identifier and description
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        # all structural assertions go here
        ...
```

- Use `@dataclass(slots=True)`.
- Required fields first, optional (with defaults) last.
- **Two kinds of configuration attribute, distinguished by value type — this
  is the axis of experiment design.** *Numerical* attributes (sizing, cost /
  reward coefficients, distributional parameters) are the **grid axes**: the
  values a study sweeps — exactly what `{Domain}ScenarioGrid.from_axes`
  crosses into cells (§5.6). *Categorical*
  attributes (bool flags and string enums — e.g. `allow_backlog`,
  `event_sequence`, `mmfe_mode`) select a **scenario mode**: a qualitatively
  distinct variant of the dynamics, not a value to interpolate. Group the two
  separately in the field list. An experiment design is then a grid over the
  numerical axes crossed with a chosen set of scenario modes; *how* modes
  compose with the grid, and whether they are compared head-to-head or kept as
  separate branches, is a per-study choice, not a property of the class. (These
  scenario modes live on the `mdp` layer — distinct from the gym's
  obs/action/reward *modes*, §7.)
- `seed_salt` is the domain's universal reproducibility knob (§6.3) and must be **`>= 1`** (the v2 seed grammar relies on a nonzero trailing word); `scenario_name` is the registry key. A deterministic-dynamics domain (§4.3) has **no `seed_salt` on the scenario or the state** — there is no intrinsic randomness to salt; only its `{Domain}ScenarioSampler` keeps one, for the per-episode instance draw.
- **A scenario is a fully concrete problem instance**: together with an `episode_seed` it must pin the episode completely. Anything one might hand-craft, name, or benchmark against (a specific puzzle board, a specific demand profile) must be representable as a fixed scenario; distributions over instances live only in samplers (§5.2, world latents) — and finite *sets* of instances in grids (§5.6, design families). If a "scenario" still needs a random draw to become a concrete problem, that draw belongs in a `{Domain}ScenarioSampler`, not in `{domain}_uncertainty.py` (§4.3).
- **`seed_salt` must be declared `field(default=<default_int>, repr=False)`** so it is excluded from the dataclass `repr`. It is an internal reproducibility knob, not a meaningful configuration value to a reader; keeping it out of the `repr` avoids cluttering the `print(scenario)` output that training scripts emit at startup (§8.2).
- **`scenario_name`** is the short registry key (e.g. `"simple"`, `"test1a"`); **`desc`** is a human-readable one-line description of the scenario. Keep the name short and put the explanation in `desc` rather than encoding everything into a long name. Prefer `desc` over a leading `#` comment on the instance, so the description travels with the object: drivers and eval scripts can print a self-describing header, e.g. `print(f"=== {scenario.scenario_name}: {scenario.desc} ===")`.
- Validate all structural constraints in `__post_init__`.

### 5.2 `{Domain}ScenarioSampler` (optional) — world latents

Define a sampler when the **problem itself** contains a latent draw nature
makes once per episode: a hidden market size, a demand-regime pick, a whole
problem instance (a puzzle board). A sampler is **world modeling** — its
distribution is part of the MDP, and changing it changes the problem. A
distribution used merely to train one policy across a family of complete
problems is *not* a sampler: it is a grid (§5.5–§5.6). When in doubt, apply
the litmus tests in §5.5.

```python
def meta_key(substream_id: int, episode_seed: int, seed_salt: int) -> list[int]:
    """v2 meta seed key: [substream_id, 0, episode_seed, seed_salt] (§6.3)."""
    return [substream_id, 0, episode_seed, seed_salt]


@dataclass
class {Domain}ScenarioSampler:
    """Draws the {latent} and returns the concrete scenario for that seed.

    Pure function: episode_seed -> {Domain}Scenario.
    """

    # family-level attributes (fixed across all generated scenarios), under
    # the SAME NAMES as the {Domain}Scenario fields — see below
    horizon: int
    # ...

    # latent-draw configuration (distribution hyper-parameters)
    # ...

    # meta-level substream id (§6.3 branch 0); distinct per drawer per module
    substream_id: int = 0

    seed_salt: int = field(default=<default_int>, repr=False)
    scenario_name: str | None = None
    desc: str = ""

    def __call__(self, episode_seed: int) -> {Domain}Scenario:
        rng = np.random.default_rng(np.random.SeedSequence(
            meta_key(self.substream_id, episode_seed, self.seed_salt)))
        latent = ...   # realize the world latent
        return {Domain}Scenario(..., scenario_name=self.scenario_name)
```

- Lives in `{domain}_scenarios.py`, alongside `{Domain}Scenario`. Instances
  are named `sampler_{name}`; `SCENARIOS` may hold either kind.
- **Pure function** `episode_seed -> {Domain}Scenario`: calling twice with
  the same seed must return equal scenarios, and a call must not mutate the
  sampler — vectorized envs share one sampler instance across workers. There
  is deliberately **no lifecycle hook** (`episode_init()` or similar):
  "drawn once per episode" is enforced by who calls it (the gym's `reset()`)
  and by the seed key (no `period` level), not by a method name.
- **Vocabulary**: "episode" is gym-layer vocabulary. The world layer defines
  a seed-indexed family of concrete problems; the *gym* binds one episode to
  one seed at `reset()`. Phrase docstrings "per seed" / "per draw", never
  "per episode". (The parameter name `episode_seed` is kept for consistency
  with `SamplingContext`.)
- Keys on the **meta branch** of the seed tree (§6.3), via `meta_key()` —
  never a raw `SeedSequence`. Every meta-level drawer in the module (samplers
  and mixtures, at any nesting depth) carries a distinct `substream_id`;
  assert uniqueness next to the registry.
- This is **meta-level** randomness, **not** intrinsic MDP randomness. Do not
  model it as a `{Source}Generator` — see the boundary tests in §4.3.
- A sampler may draw **entire problem instances** (a full Sudoku board carved
  from a random solved grid), not just scalar parameters. Unbounded,
  procedurally generated instance families stay samplers even though the
  draw is arguably the experimenter's — the grid alternative requires a
  finite, enumerable set (§5.5). Generation helpers live in
  `{domain}_scenarios.py` as module-level functions; the returned scenario
  holds the realized instance.
- The gym wrapper checks `callable(self.scenario)` and calls
  `self.scenario(episode_seed)` in `reset()` to get the concrete scenario
  for that episode (§7).
- **`scenario_name`**: set to the same key used in `SCENARIOS` (e.g.
  `scenario_name="hidden-market-size"`) so scripts can read it without an
  isinstance check.
- **Family-level attributes — the mirroring rule**: expose every attribute
  the gym or eval scripts need before a draw *under the same name as the
  corresponding `{Domain}Scenario` field* (`horizon`, `leadtime`,
  `allow_backlog`, mode flags, …). For the *latent* part, expose a small
  bounds object under the generator's field name with the generator's read
  API (`max()`, `mean()`, `is_discrete`). Done this way, the gym's
  space-building code works unchanged against a scenario or a sampler.
- **Observed vs hidden latent** — same machinery, different problems, so
  state which one you mean in `desc`: an *observed* latent (a forecast the
  agent sees) makes a contextual MDP — an observation mode may expose the
  realized value; a *hidden* latent (an unobserved market size) must be
  inferred — a realized latent field on the scenario is **not** automatically
  observed; observability is decided only by the observation modes (§7). The
  deployable policy (§12) must never require a hidden latent as input, and a
  baseline that reads one is *clairvoyant* — an upper bound, labeled as such
  in eval output.

### 5.3 `{Domain}MixtureSampler` (optional)

A mixture is a world-latent sampler whose latent is *which component runs*
(nature picks the demand regime): drawn once per seed, components with given
probabilities. **The weights are a modeling commitment** — "50/50" is a claim
about the world. A non-uniform *training pool* ("train 80% on the hard
scenario") is not a mixture; it is grid weights (§5.5, §5.6).

```python
@dataclass
class {Domain}MixtureSampler:
    """Draws one component scenario source per seed."""

    # (weight, component); components may be fixed scenarios OR samplers,
    # including other mixtures
    components: list[tuple[float, ScenarioSource]]

    substream_id: int = 0        # meta-level substream id (§6.3 branch 0)
    seed_salt: int = field(default=<default_int>, repr=False)
    scenario_name: str | None = None
    desc: str = ""

    def __call__(self, episode_seed: int) -> {Domain}Scenario:
        rng = np.random.default_rng(np.random.SeedSequence(
            meta_key(self.substream_id, episode_seed, self.seed_salt)))
        k = int(rng.choice(len(self.components), p=self._probs))
        component = self.components[k][1]
        return component(episode_seed) if callable(component) else component
```

- Normalize weights in `__post_init__` (assert positive); assert the
  components agree on the family-level attributes one gym must serve
  (horizon, backlog mode, …), and expose those family attributes under the
  mirroring rule of §5.2 (bounds combine across components: max of maxes,
  weighted mean of means).
- **`episode_seed` is delegated verbatim** to the chosen component. Combined
  with per-drawer substream ids this gives *standalone equivalence*: episode
  seed `e` through a mixture branch equals seed `e` run on that component
  directly — any branch is reproducible in isolation.
- **Mixtures of mixtures are allowed** — components are `ScenarioSource`s and
  a mixture is one. Nesting adds structure, not expressive power (weights
  multiply through to a flat mixture); its value is reuse — embed a named
  world object as one branch without hand-flattening weight products.

### 5.4 Instances and registry

```python
scenario_simple = {Domain}Scenario(scenario_name="simple", desc="...", ...)
scenario_{name} = {Domain}Scenario(scenario_name="{name}", desc="...", ...)

sampler_hidden_mu = {Domain}ScenarioSampler(scenario_name="hidden-mu", substream_id=0, ...)

SCENARIOS: dict[str, ScenarioSource] = {
    "simple":    scenario_simple,
    "{name}":    scenario_{name},
    "hidden-mu": sampler_hidden_mu,   # world-latent sampler entries allowed
}
```

- Fixed scenario instances are named `scenario_{name}`; sampler instances are named `sampler_{name}`.
- `SCENARIOS` values are `ScenarioSource`s: `{Domain}Scenario`, `{Domain}ScenarioSampler`, or `{Domain}MixtureSampler` — **world layer only; a grid must never appear in `SCENARIOS`** (it is not callable and deliberately fails the gym's sampler check).
- Assert meta-level `substream_id` uniqueness across the module's drawers next to the registry (a `_check_meta_substreams()` helper); the conformance harness checks it externally.
- `SCENARIOS` dict provides the canonical lookup for training and evaluation scripts.
- Small smoke-test scenarios used only by the `{domain}_mdp.py` `__main__` driver may be defined here as plain module-level instances and imported by the driver; they need not be added to `SCENARIOS` if they are not meant as training/eval targets.

### 5.5 World vs. design — the litmus tests

A world latent requires a **probability measure as a modeling commitment**; a
grid requires only a **finite set**, and any measure on it is a training
hyper-parameter. Sampling-vs-enumeration follows as a corollary — you
integrate over measures, you enumerate sets. Tests, in order of usefulness:

1. **Weights test.** "If I changed the weights, would I be changing the world
   model or just the training recipe?" World model → sampler/mixture;
   recipe → grid.
2. **Complete-problem test.** "Is one draw from this a complete problem
   someone might want a specialist policy for?" One cost-parameter combo —
   yes → grid cell. One realized hidden market size — no, the
   inference-under-uncertainty problem is gone → world latent.
3. **Deployment test.** "Would the environment draw this afresh each episode
   in the real system?" Yes → world latent. Known and fixed at deployment,
   trained across variants → grid.

Edge cases, pinned:

- **Diagnostic enumeration of world latents** (performance conditional on
  latent deciles) is slicing one problem's expectation for analysis — not a
  leaderboard basis, not a counterexample.
- **Continuous generality ranges** (a cost coefficient in `[0.1, 0.5]`,
  sim2real-style randomization) are discretized at grid construction time —
  eval needs a finite test set anyway.
- **Unbounded procedurally-generated families** (random puzzle boards) stay
  world-layer instance samplers: the grid's generality claim must be finite
  and enumerable.
- **Domain randomization with hidden family parameters** is generalist
  training over a grid; the hiding is an observation-mode decision and must
  be stated intent, not an accident.

**Design-level sampling exists if and only if the training objective is a
generalist over a family** — benchmarking across variants is enumeration,
not sampling. So the interview/order of questions is: world randomness is
settled while formalizing the MDP; the grid question ("one policy for this
instance, or one that works across a range — over what range?") is settled
when choosing the training target.

### 5.6 `{Domain}ScenarioGrid` (`{domain}_grids.py`, optional)

The grid is the finite set describing the *generality* of a generalist
policy. Training uses it as a pool (incidentally — via a derived sampler);
evaluation **enumerates** it, one leaderboard row per cell. It is
deliberately **not callable**, so it can never pass where a `ScenarioSource`
belongs.

```python
@dataclass
class {Domain}ScenarioGrid:
    """A finite set of complete scenario sources — the generality target
    of a generalist policy. NOT callable."""

    # (cell_id, source) in canonical order; cell_id derives from axis values
    # ("h=0.2,b=2.0") and names the per-cell leaderboard row
    cells: list[tuple[str, ScenarioSource]]
    grid_name: str | None = None
    desc: str = ""

    @classmethod
    def from_axes(cls, axes: dict[str, tuple], **base_kwargs): ...
        # row-major cross product of constant overrides, axes in declaration order

    @classmethod
    def from_sources(cls, sources: list[ScenarioSource], ...): ...
        # explicit list — a grid with one categorical axis; definition order;
        # cell ids default to the sources' scenario_names

    def as_sampler(self, weights=None, substream_id=0): ...
        # derive an ordinary sampler over the cells (uniform by default)

    # enumeration is data access, not behavior — no enumerate() method:
    def __len__(self): ...              # number of cells
    def __iter__(self): ...             # yields (cell_id, source) in canonical order
    def __getitem__(self, key): ...     # by cell_id (str) or index (int)
```

- **Canonical, deterministic cell order is part of the contract**: (a) the
  derived sampler maps a drawn index `k` to `cells[k]` — unstable order
  breaks reproducibility silently; (b) leaderboard rows
  (`{grid_name}/{cell_id}`) come out in the same order every run; (c)
  `cell_id` is a pure function of axis names/values in declaration order,
  never of insertion accidents.
- **Cells are `ScenarioSource`s, not just scenarios** — a generalist across
  cost structures of a domain *with* a world latent keeps each cell's
  sampler. The grid never draws a cell's latent: enumeration yields sources;
  the eval driver decides episodes and seeds per cell. The grid never
  touches an `episode_seed`.
- **`as_sampler()` return contract**: an ordinary sampler — callable, pure,
  `scenario_name` set from `grid_name`, family attributes delegated to the
  grid (mirroring rule, §5.2), defined as a **module-level class** so it
  pickles into `SubprocVecEnv` workers. The class itself is an
  implementation detail; domains never construct it directly. Weights on the
  cells are a training choice (§5.5) and live only here, never on the grid.
- **Family-level attributes**: the grid exposes the attributes all cells
  agree on (mirroring rule, §5.2) so gym/eval scripts can build spaces and
  read dimensions without resolving a cell; the constructors assert the
  agreement.
- **Common random numbers**: every cell is keyed by the same seed protocol,
  so eval drivers reuse one identical block of episode seeds across all
  cells — per-cell comparisons are paired, not independent. Do this
  deliberately.
- Per-cell baselines that read the cell's parameters are ordinary
  baselines — not cheating (contrast with clairvoyant baselines on world
  latents, §5.2).
- Instances are named `grid_{name}`, registered in `GRIDS`; a migrating
  domain that previously trained via an ad-hoc sampler may offer a
  legacy-exact draw mode on its derived sampler to preserve recorded
  numbers.

---

## 6. Core MDP (`{domain}_mdp.py`)

This module contains `{Domain}State` and the pure state-transition functions. It imports `{Domain}Scenario` from `{domain}_scenarios` (for annotations) and generator classes from `{domain}_uncertainty` only where it constructs them (e.g. its `__main__` smoke test).

### 6.1 Module-level docstring

```
"""{Domain} core MDP.

This module contains the domain-level simulator only.
It does not depend on Gym/Gymnasium.

State/info is represented from the *simulator's* full perspective.  What is
observable to the RL agent is NOT defined here — observations are constructed
in the gym wrapper ({domain}_gym.py) based on the ``observation_mode`` it is
configured with.

Key components:
1. {Domain}State: keep track of the running state, including:
    a. running status: period, terminated
    b. state variables: ...
    c. episode specific seed
2. state transition functions:
    a. init_state(scenario, episode_seed)
    b. advance(scenario, state, decision)

The scenario config ({Domain}Scenario) and the stochastic primitives it is
built from ({Source}Generator) live in {domain}_scenarios.py and
{domain}_uncertainty.py respectively; this module imports {Domain}Scenario for
type annotations only.

Notations:
    - ...

NOTE: ...
"""
```

### 6.2 `{Domain}State`

```python
@dataclass(slots=True)
class {Domain}State:
    """Full internal state of the simulator.

    This captures everything the simulator needs to generate transitions — it is
    the MDP state from the *simulator's* perspective, not the agent's.  Whether
    a field is observable to the RL agent is NOT determined here; observation
    construction is delegated to the gym wrapper ({domain}_gym.py) according to
    the ``observation_mode`` it is configured with.
    """

    # --- simulator state fields ---
    period: int
    terminated: bool
    # ... domain-specific fields ...

    ## internal RNG state
    episode_seed: int
    seed_salt: int = field(repr=False)
```

- Use `@dataclass(slots=True)`.
- `episode_seed` and `seed_salt` are **public** fields — generators receive the state (as a `SamplingContext`, §4.1) and read these directly. Deterministic-dynamics domains (§4.3) omit `seed_salt`; `episode_seed` stays as the episode identifier.
- **`seed_salt` must be declared `field(repr=False)`** (same rationale as §5.1): it is an internal reproducibility knob and is kept out of the state `repr` so it does not clutter debug prints of the state. `episode_seed` stays in the `repr` — it identifies which episode the state belongs to.
- The docstring must always state the simulator-perspective stance and point to the gym wrapper.
- `{Domain}State` structurally satisfies `SamplingContext`; it deliberately exposes `period`, `episode_seed`, and `seed_salt` with those exact names so generators can sample against it without a concrete import.

### 6.3 RNG pattern — the seed tree (scheme v2)

All random draws use `numpy.random.SeedSequence`. Every draw's key encodes a
path through **one tree**:

```
seed_salt (root, >= 1)
└─ episode_seed
   ├─ branch 0 (meta)      └─ substream_id                    [drawn once per episode]
   └─ branch 1 (intrinsic) └─ source_id └─ period └─ (draw)   [drawn per period]
```

with **one rule applied at every node: children carry distinct ids** — meta
drawers under branch 0 (§5.2–§5.3), generator *instances* under branch 1
(§4.2; per instance, not per class, so repeated generators — e.g.
per-retailer leadtimes — get distinct ids), draw indices under a period. Ids
default to 0 for a lone child. The meta/intrinsic asymmetry is tree shape:
branch 0 has no period level, so a meta-level draw cannot be expressed in an
intrinsic key without leaving the template (§4.3).

**Canonical encoding is leaf-first (root last)** — the *reverse tree*:

```python
# intrinsic (generators; via intrinsic_key(), §4.2):
ss = np.random.SeedSequence([(draw,) period, source_id, 1, episode_seed, seed_salt])
# meta (samplers / mixtures; via meta_key(), §5.2):
ss = np.random.SeedSequence([substream_id, 0, episode_seed, seed_salt])
```

Why leaf-first and not the tree-reading root-first order:

1. **`SeedSequence` zero-pads entropy lists shorter than its 4-word pool**,
   so keys differing only in trailing zeros within the first 4 words collide
   (`[5,0] ≡ [5]`; 5-word lists are safe; leading/mid zeros are always
   significant). Root-first would put the routinely-zero leaf ids (period 0,
   substream 0) in trailing position — e.g. root-first meta
   `[salt, e, 0, 0] ≡ [salt, e]`, a silent collision. Leaf-first ends every
   key with `seed_salt`, constant and **`>= 1`** (hence that rule, §5.1).
2. **numpy's documented direction**: the parallel-RNG guide recommends
   varying ids *before* the fixed root seed, because `spawn()` *appends*
   integers to the entropy list.
3. Statistical quality is order-independent — the choice is structural.

- Draws never build a raw `SeedSequence` inline: intrinsic draws go through
  `intrinsic_key()` (§4.2), meta draws through `meta_key()` (§5.2), so the
  templates cannot drift and the conformance harness can flag strays.
- This structure makes each draw uniquely identified and independent of the
  decision path.

**Scheme versioning.** The grammar above is **v2**, the canonical scheme for
all new domains. Pre-existing research domains carry frozen historical keys
(**v1**: meta `[0, e, salt]` or `[0, sub, e, salt]`; intrinsic word orders
varying per domain) kept solely so recorded results stay reproducible; a
domain declares its scheme (`SEED_SCHEME = "v1" | "v2"` in the scenarios
module), never mixes schemes, and migrates only when its numbers are
invalidated anyway. Even accidental cross-scheme mixing cannot collide: the
v1 3-word meta key pads to `[0, e, salt, 0]`, ending in 0 — no v2 key ends
in 0.

### 6.4 Transition functions

```python
def init_state(scenario: {Domain}Scenario, episode_seed: int) -> tuple[{Domain}State, dict]:
    ...

def advance(scenario: {Domain}Scenario, state: {Domain}State, decision) -> tuple[{Domain}State, dict]:
    """Return (next_state, info)."""
    ...
```

- Pure functions — never mutate `state`.
- **No reward in the MDP layer.** Reward is a Gym-facing concept; the `info` dict carries all cost/revenue quantities (e.g. `info["cost"]["total"]`), and the gym wrapper computes `reward` from them in `step()`.
- For two-step sequential decisions, split into `advance1` (returns intermediate state) and `advance2` (returns `(next_state, info)`).

#### `state` vs `info` — the boundary

`advance()` returns two things and they have **opposite directions**, which is exactly what distinguishes them:

- **`state`** = the *minimal* information needed to compute the **next** transition. It flows **back into** the next `advance()`.
- **`info`** = everything else that happened **during** this transition. It flows **out to** the gym wrapper and is **never read by the simulator again**.

Concretely, `info` carries three kinds of quantity and nothing else:

1. **Time index** — `action_period`: the period the decision was applied to (= the **input** `state.period`). Every other value in the same `info` dict describes this same period, so the label matches its siblings. Keep it distinct from `state.period`, which is the running counter and, on the **returned** state, has already advanced to the *next* period. Invariant: after any `advance*`, `action_period` is the period just processed; for `init_state` (no decision yet) it is `state.period - 1`. For two-step domains, `advance1` and `advance2` share the same `action_period` (both process one period). Kept in `info` despite mirroring the input period, as a convenience label for logging and analysis.
2. **Transition diagnostics / events** — realized demand, leadtimes, arrivals, sales, backlog / lost sales, stockout events, spawned tiles, move-validity flags, etc. Records of *what the transition did*, not persistent state.
3. **Economic / reward quantities** — per-step `cost`, `profit`, `revenue`, and regret benchmarks (`profit_max`, `revenue_opt`), plus running economic accumulators (`total_cost`). These are the whole reason `info` exists: the per-step ones must stay **out of `state`** so the MDP remains reward-agnostic, and the gym reads them to compute `reward`.

**Timeline — the two counters (`state.period` vs `info["action_period"]`).**
A 0-indexed domain (periods P0, P1, …):

```
      init_state()            advance(state, d0)          advance(state, d1)
           |                         |                           |
           v                         v                           v
     +-----------+   d0    +------------------+   d1   +------------------+
     | set up    | ------> | process period 0 | -----> | process period 1 | --> ... --> terminated
     | episode   |         | (realize P0)     |        | (realize P1)     |
     +-----------+         +------------------+        +------------------+

  state.period          0          0  -->  1                 1  -->  2
  (running counter)   start     (in)     (out)            (in)     (out)

  info["action_period"]  -1              0                        1
  (= the INPUT period)                P0's demand/             P1's demand/
                       no action      cost/sales/…            cost/sales/…
                       taken yet      (all for period 0)      (all for period 1)
```

- `state.period` is the **running counter**. On the state passed **in** it equals the period being processed; on the state returned **out** it has already advanced by one.
- `info["action_period"]` = the **input** period = the period the decision acted on. It labels every diagnostic in that same `info` dict, so they never disagree. Equivalently it is `(returned state).period - 1`; for `init_state` that is `state.period - 1` (`-1` here; no decision yet).
- A 1-indexed domain starts at `state.period = 1`, so `init` gives `action_period = 0`.
- Two-step domains (e.g. `inv_single`): `advance1` and `advance2` share the same `action_period = n`; only `advance2` advances `state.period` to `n+1`.

**Do not put in `state`** anything that does not feed the next transition — per-step outcomes, per-step costs, profits, and benchmarks belong in `info` (or, for an episode-level benchmark, a public helper like `optimal_revenue(scenario, episode_seed)`). A field that would be `None`/`0` for most of the episode (a terminal-only benchmark) is a signal it belongs in `info`, not `state`.

**Do not duplicate the physical state snapshot in `info`.** Inventory levels and pipelines that merely mirror the returned `state` are redundant — the gym already has the state — so keep them out of `info` and repoint any consumer to read them off `state`. Three kinds of value are kept in `info` *on purpose* even though they relate to `state`, because the small duplication buys clarity or is spec-mandated: the **time index** (`action_period`, the input period — see above), **running economic accumulators** the simulator carries forward (`total_cost`), and **realized random outcomes** (`demand`, `rt_demands`) that `state` stores only so the gym can observe them — these are genuine category-2 diagnostics and belong in `info`.

---

## 7. Gymnasium Wrapper (`{domain}_gym.py`)

```python
class {Domain}Env(gym.Env):
    """{Domain} Gymnasium wrapper.

    action_mode:
    - "...": description

    observation_mode:
    - "dict": structured dict observation
    - "vec":  flat float32 vector
    - "vec_d": flat vector including demand signal
    - "vec_ip": flat vector with inventory positions

    Notes:
    - The core simulator is the source of truth for dynamics and reward.
    - This wrapper handles only Gym-facing concerns: spaces, formatting, reset/step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioSource,
        action_mode: str = "...",
        observation_mode: str = "vec",
        logger_filename: str | None = None,
    ) -> None:
        ...
        # a fixed scenario, or a sampler resolved per episode in reset().
        # Space building reads only family-level attributes (§5.2 mirroring
        # rule), which samplers expose under the same names as a scenario.
        self.scenario: ScenarioSource = scenario
        self._scenario_ep: {Domain}Scenario | None = None
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()
        ...

    def reset(self, seed=None, options=None) -> tuple[obs, info]:
        ...
        self._scenario_ep = (
            self.scenario(self._episode_seed)
            if callable(self.scenario) else self.scenario
        )
        # init_state / advance* are called with self._scenario_ep
        ...

    def step(self, action) -> tuple[obs, reward, terminated, truncated, info]: ...
```

- **Convention:** call MDP *functions* through a module alias (`import {domain}_mdp as mdp`, then `mdp.init_state(...)`, `mdp.advance(...)`, `mdp.valid_actions(...)`); reserve `from ... import ...` for *types* used in annotations — `{Domain}State` from `{domain}_mdp`, `{Domain}Scenario` from `{domain}_scenarios`, and generator classes from `{domain}_uncertainty` if needed. This keeps every transition-function call visibly namespaced and avoids importing the same module two ways for the same kind of name.
- `self.scenario` is the stored attribute name (the source, as passed in); `self._scenario_ep` is the per-episode concrete scenario, and everything episode-scoped (`init_state`, `advance*`) uses it. For a fixed scenario they are the same object.
- Action and observation spaces built in private `_build_action_space()` / `_build_observation_space()` methods, reading only family-level attributes so a sampler works unchanged (§5.2 mirroring rule).
- Supports `logger_filename` for per-episode and per-step log output.

### 7.1 Action Masking (discrete action spaces only)

When some actions are structurally invalid at certain states — not merely suboptimal, but guaranteed to leave the state unchanged or violate a hard constraint — **action masking** removes them from the policy's support before sampling.

**This pattern applies only to discrete action spaces.** With a `Discrete(n)` space, "valid" is a clean boolean concept: each of the n actions either is or is not executable. With continuous spaces, the notion of valid vs. invalid is not a finite enumerable set; feasibility is better handled by projection or clipping inside `advance()`.

**Implementation**:

1. **MDP layer** — expose a public function alongside `advance()`:

   ```python
   def valid_actions(state: {Domain}State) -> np.ndarray:
       """Return bool array of shape (n_actions,) marking structurally valid actions."""
       mask = np.zeros(N_ACTIONS, dtype=bool)
       for a, direction in _ACTIONS.items():
           mask[a] = _try_move(state.board, direction)[1]
       return mask
   ```

   The validity logic belongs here — not in the gym wrapper — because it is a property of the MDP transition function, not of the observation format.

2. **Gym wrapper** — add `action_masks()` delegating to the MDP:

   ```python
   def action_masks(self) -> np.ndarray:
       """Boolean mask of valid actions (required by MaskablePPO)."""
       return valid_actions(self._state)
   ```

3. **Training script** — use `MaskablePPO` from `sb3_contrib` instead of `PPO`:

   ```python
   from sb3_contrib.ppo_mask import MaskablePPO
   model = MaskablePPO("CnnPolicy", env, ...)
   model.learn(total_timesteps=...)
   ```

   Masking is applied automatically during rollout collection; no extra flags needed.

**Keep `valid_actions` cheap.** It is called every step during training. If checking validity requires re-running the transition function (e.g. a board game that must attempt each candidate move to test it), ensure the underlying check is O(board size), not O(horizon).

### 7.2 Handling Action Constraints (continuous action spaces)

With continuous actions, invalid decisions cannot be enumerated as a boolean mask. Three strategies exist, in order of preference:

**1. Reparametrization (preferred when the feasible set has algebraic structure)**

Transform the policy's raw output into a guaranteed-valid action inside `step()`, before calling `mdp.advance()`. The policy never sees the constraint; it always produces feasible actions.

```python
def step(self, action):
    valid_action = self._transform_action(action)   # gym wrapper's responsibility
    self._state, info = mdp.advance(self.scenario, self._state, valid_action)
    ...
```

The action space declared in `_build_action_space()` describes the *raw* (pre-transform) domain — what the policy actually outputs. The transform maps it to the true feasible set.

Common transforms:

| Constraint | Raw space | Transform |
|---|---|---|
| Allocations sum to 1 (simplex) | `Box([0,∞)^n)` or `Box([0,1]^n)` | L1-normalize or softmax |
| Bounded scalar `[lo, hi]` | `Box([-∞, ∞])` | sigmoid scaled to `[lo, hi]` |
| Non-negative quantity | `Box([-∞, ∞])` | softplus or exp |

**Simplex reparametrization in detail (worked example)**

Consider a one-warehouse multi-retailer domain: inventory ships from a warehouse to N retailers, and the allocation decisions must satisfy `sum(shipments) <= wh_onhand` — i.e., they lie on a scaled simplex.

An action mode `"order&allocations"` in its `{domain}_gym.py`:

- **Raw action space**: `Box([0,1]^(N+1))` — unnormalized allocation weights, one per retailer plus warehouse self-retention. The policy outputs values in `[0, 1]`; their relative magnitudes matter, not their absolute values.
- **Transform in `step()`**: L1-normalize the weights, then scale by available inventory:

  ```python
  def _shipment_by_allocation(self, onhand, allocations):
      total = sum(allocations)
      if total > 1e-8:
          normalized = np.array(allocations) / total   # L1 normalization → simplex
      else:
          normalized = np.ones(len(allocations)) / len(allocations)  # fallback: uniform
      return onhand * normalized  # scale to available stock
  ```

- **Result**: `advance()` always receives shipments that exactly exhaust available inventory, with no constraint violation possible.

An alternative when the raw output is unbounded is **softmax**: `normalized = exp(logits) / sum(exp(logits))`. Softmax and L1-normalization achieve the same mapping to the simplex; softmax is numerically stable for unconstrained logits while L1 is simpler when the raw space is already non-negative.

**2. Clipping in `advance()` (simplest fallback)**

Accept any action and project it to the nearest feasible point inside `advance()`. The agent learns implicitly that out-of-bound outputs are wasteful. Works reliably but the policy may waste representational capacity near boundaries, and gradient signal near clipped boundaries is zero.

**3. Penalty in reward (last resort)**

Add a negative reward proportional to constraint violation magnitude. Fragile — penalty scale needs careful tuning relative to the task reward, and large violations early in training can dominate the learning signal.

---

## 8. Training Scripts

### 8.1 Structure (all scripts)

```python
import argparse
from {domain}_gym import {Domain}Env
from {domain}_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="...")
    p.add_argument("-s", "--scenario_name",    default="simple",
                   type=str, choices=list(SCENARIOS.keys()))
    p.add_argument("-a", "--action_mode",      default="...",     type=str)
    p.add_argument("-o", "--observation_mode", default="vec",     type=str)
    p.add_argument("--seed",                   default=42,        type=int)
    # ... algorithm-specific hyperparameters ...
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main():
    args = parse_args()

    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    env = {Domain}Env(
        scenario=scenario,
        action_mode=args.action_mode,
        observation_mode=args.observation_mode,
        logger_filename=...,
    )

    # ... training logic ...


if __name__ == "__main__":
    main()
```

### 8.2 Mandatory conventions

- Always `_build_arg_parser()` (private) + `parse_args()` (public) + `main()` + `if __name__ == "__main__": main()`.
- `_build_arg_parser()` returns the parser; `parse_args()` calls `.parse_args()` on it and returns the namespace. This split lets other scripts reuse the parser (e.g. to add extra arguments) without re-implementing it.
- Scenario lookup via `SCENARIOS[args.scenario_name]` (not `globals()`).
- **Generalist training** targets a grid: look up `GRIDS[args.grid_name]` and pass `scenario=grid.as_sampler()` (optionally with weights). The env never sees the grid itself (§5.6); the derived sampler carries `scenario_name` from `grid_name` for run identity.
- `print(scenario)` at the top of `main()` to confirm configuration.
- Env constructed with `scenario=scenario` (keyword argument, never positional).
- RLlib `env_config` dict must use `"scenario"` as the key (RLlib calls `Env(**config)`).
- **Do not hard-code `gamma`** in PPO kwargs — expose it as a CLI argument instead.
- **Pin BLAS/torch threads when launching training** (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`): the policies in these domains are tiny, so torch's default all-cores threading adds sync overhead rather than speed, and on a shared machine it oversubscribes cores already used by other jobs (measured on a small-board CNN domain: 9 min → 11 s for 2048 steps on a box concurrently running an 8-core workload; expect a smaller but still real gain on an idle box). The `mdp_tuning` harness sets this for its subprocesses automatically.

### 8.3 SB3 env-wrapper stack (VecNormalize)

For SB3 single-agent PPO scripts, wrap the env in this order:

```python
env = {Domain}Env(scenario=scenario, ..., logger_filename=...)
env = FrameStackObservation(env, stack_size=..., padding_type="zero")  # only if the domain uses it
env = Monitor(env, filename=str(outdir / "monitor"))
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=args.norm_reward, clip_obs=args.vecnorm_clip_obs)
```

- **Order matters.** `Monitor` must sit **inside** `VecNormalize` so `rollout/ep_rew_mean` is logged on the **raw** reward scale — this keeps the metric comparable across runs regardless of reward normalization. (Only `train/value_loss`, `train/explained_variance`, etc. are on the normalized scale.)
- **Expose two CLI flags:** `--vecnorm_clip_obs` (default `10.0`) and `--no_norm_reward` (`action="store_false", dest="norm_reward", default=True`). Encode both in the run name when non-default.
- **Save the stats:** `env.save(outdir / "vecnormalize.pkl")` after `model.learn(...)`, then `env.close()`.
- **What each normalization is for:**
  - *Obs norm* conditions the network **inputs** (features often span several orders of magnitude — e.g. inventory vs. a price index). It is **part of the policy's input contract**: the saved `obs_rms` **must** be reused at eval/deploy time (§9.5), otherwise the policy sees inputs it was never trained on.
  - *Reward norm* scales the reward by the running std of the discounted return (no centering). It is **training-only** — never needed for inference/deployment. Because SB3 PPO's `normalize_advantage=True` already makes the policy-gradient step scale-invariant, reward norm mostly affects the **critic** (value-loss scale / `vf_coef` balance), not the policy behavior.
- **Not for non-SB3 frameworks.** VecNormalize is SB3-specific; RLlib/PettingZoo/DreamerV3 scripts use their own normalization (e.g. RLlib's `rllib_vecnormalize`).

**When obs norm helps vs. when it hurts** — decide per domain, don't apply blindly:

- **Helps** when obs dimensions span very different magnitudes (e.g. inventory ~1e3 alongside a price index ~1) *and* the obs distribution is roughly **stationary** across training. This is the common case and the main reason to enable it.
- **Little benefit** when obs are already **homogeneous and bounded** on a common small scale (e.g. one-hot planes, or log2-scaled tiles on a game board) — there is no cross-feature imbalance to fix; a static encoding is enough.
- **Risk** when the obs distribution is **non-stationary** — i.e. what the agent observes shifts as it improves (e.g. a tile-merging game whose reachable tile values keep growing). The running `obs_rms` never stabilizes and the frozen eval stats match no single stage ("stats-mismatch under distribution shift"). Domains with this property may deliberately omit VecNormalize.

Reward norm is training-only and largely redundant with PPO's `normalize_advantage`, so the decision above is really about **obs** norm.

### 8.4 Output directory structure and run name encoding

Training outputs go to `results/{scenario_name}/{run_name}/` relative to the domain directory:

```
results/
  {scenario_name}/                   # e.g. simple/, cost-sweep/
    {run_name}/                      # e.g. PPO_20260630_224416_default/
      {scenario_name}_{algo}.zip     # saved SB3 model
      vecnormalize.pkl               # VecNormalize running stats (if used)
      {scenario_name}_{algo}_args.txt  # all CLI args at training time
      train_log_episode.log          # per-episode log
      train_log_step.log             # per-step log
      monitor.monitor.csv            # SB3 Monitor CSV
      {ALGO}_1/                      # TensorBoard event files
      ppo_eval_{eval_scenario}.tsv   # eval output (eval scenario encoded in filename)
    benchmark/
      {method}/                      # one dir per benchmark method (dp, lp, ...)
        {scenario_name}.txt          # precomputed solution table for this scenario, if any
      benchmark_{name}_eval_{scenario_name}.tsv  # benchmark eval output ({name} = method or policy variant)
  tuning/                            # `mdp_tuning` studies — domain-level, NOT scenario-nested
    optuna.db                        # sqlite store; holds every study for this domain
    {study_name}/                    # e.g. dynamic_pricing_simple_ppo (domain + scenario + algo)
      trial_{NNNN}/                  # passed to the train script as --outdir
        train.log  eval.log  eval.tsv
        {scenario_name}/{run_name}/  # the tree above, re-applied by the train script
```

**Why `tuning/` is not under `{scenario_name}/`**: the harness hands each
`trial_{NNNN}/` to the domain's train script as `--outdir`, and that script
re-applies its own `{scenario_name}/{run_name}/` nesting — so the scenario
already appears inside every trial. Nesting the study under a scenario as well
would repeat that segment, and `optuna.db` is per-domain (studies for
different scenarios live side by side in it), so it needs a domain-level home.

**Run name encoding**: encode experiment design axes and all non-default hyperparameters in the run name, with a timestamp prefix for uniqueness. Use a `build_run_name(args)` helper:

Two tiers of keys:

- **Always-shown** (experiment design axes — define what problem is being solved):
  `observation_mode` → `obs`, `action_mode` → `act`, `reward_mode` → `rew`.
  Add these to `_SKIP_KEYS` and prepend them unconditionally before the defaults loop.
- **Only when non-default** (hyperparameters — tuning knobs):
  `lr`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `ent_coef`, `total_timesteps`, etc.
- **Always skipped**: `outdir`, `scenario_name`, `progress_bar`.

```python
_SHORT_KEYS: dict[str, str] = {
    "total_timesteps": "steps",
    ...
}
_SKIP_KEYS = {"outdir", "scenario_name", "progress_bar", "observation_mode", "action_mode", "reward_mode"}

def build_run_name(args: argparse.Namespace) -> str:
    defaults = vars(_build_arg_parser().parse_args([]))
    parts = [
        f"obs{args.observation_mode}",
        f"act{args.action_mode}",
        f"rew{args.reward_mode}",
    ]
    for key, default_val in defaults.items():
        if key in _SKIP_KEYS:
            continue
        val = getattr(args, key)
        if val != default_val:
            label = _SHORT_KEYS.get(key, key)
            formatted = f"{val:.3g}" if isinstance(val, float) else str(val)
            parts.append(f"{label}{formatted}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ALGO}_{timestamp}_{'_'.join(parts)}"
```

Result: `PPO_20260630_224416_obssales_actdiscrete_rewprofit` (all defaults) or `PPO_20260630_224416_obstiming_actdiscrete_rewprofit_steps5000000`.

**File naming rules**:
- Model: `{scenario_name}_{algo}.zip` (e.g. `simple_ppo.zip`)
- Args log: `{scenario_name}_{algo}_args.txt`
- Eval output: `ppo_eval_{eval_scenario}.tsv` — the eval scenario is encoded in the filename so that evaluating the same model on different scenarios produces non-overwriting files (see §9.6)

### 8.5 Policy selection (MLP vs CNN)

Pick the SB3 policy family from the **observation shape**, with a CLI override:

- `ndim >= 3` (channels-first spatial obs, e.g. `(C, N, N)` from an `array3d`/`onehot` observation_mode) → `CnnPolicy`.
- `ndim <= 2` (flat vectors; non-spatial matrices such as entity × pipeline-slot tables) → `MlpPolicy`.
- Expose `--policy {auto,mlp,cnn}` with default `auto`; resolve once in `main()` via a `_resolve_policy(policy, obs_shape)` helper and print the resolution.

The shape is a reliable proxy only because the gym conventions make it one: spatial observation modes deliberately emit channels-first 3D arrays (`(1, N, N)` for a raw board), while non-spatial features stay flat. A bare 2D obs is treated as non-spatial — genuinely spatial boards are already `(1, N, N)` by convention.

**SB3's default NatureCNN must not be used** — its Atari-sized kernels (8×8 stride 4) crash on small boards. The default extractor is a small-board CNN (`SmallBoardCnn`, defined in the train script — extractors live there):

- stacked stride-1, same-padding convolutions, channels `(32, 64)`, kernel 3 clamped to `min(3, H, W)`;
- **no pooling** — absolute position matters on game boards;
- flatten → `Linear(features_dim=128)` + ReLU;
- `--net_arch` keeps meaning the MLP / pi-vf head widths in both families.

Expose `--features_dim`, `--channels`, `--kernel_size` as CLI arguments so the `mdp_tuning` harness can reach them (its PPO space tunes `features_dim`/`channels` whenever the script exposes them). Domains may add specialized extractors beyond the default (e.g. row/column kernels for a grid board) behind a `--cnn_arch` choice.

---

## 9. Evaluation Scripts

Parallel evaluation scripts are expected for each domain: one for the trained RL model (`{domain}_ppo_eval.py`) and one per non-RL benchmark (`{domain}_benchmark_{method}_eval.py`). They share the same output format for direct comparison. **Benchmark** is the umbrella term for any non-RL solution used to bound or compare policy quality — whether it comes from the original paper/document or is synthesized on the fly. In a gate (§13) a benchmark plays a role — `--baseline` (must-beat) or `--reference` (reported) — but that role is a per-comparison choice, not part of the artifact's identity.

### 9.1 Common structure

```python
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="...")
    # for PPO eval:
    p.add_argument("--model-path",   type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None)
    # for a benchmark eval that replays a precomputed solution table:
    p.add_argument("--solutions", type=str, required=True)
    # shared:
    p.add_argument("-o", "--observation_mode", type=str, default="vec")
    p.add_argument("-a", "--action_mode",      type=str, default="box")
    p.add_argument("--n-seeds", type=int, default=65536)
    p.add_argument("--outfile", type=str, default=None)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()
```

### 9.2 Seed loop

For each `(scenario_param_combo)`, run exactly `n_seeds` episodes using episode seeds `0, 1, ..., n_seeds-1`, accumulating the domain's per-episode objective (the episode return, or whatever the domain reports as its outcome):

```python
returns = np.zeros(n_seeds)          # the domain's per-episode objective

for ep_seed in range(n_seeds):
    if ep_seed % 10000 == 0:
        print(f"  seed {ep_seed}/{n_seeds}", flush=True)
    obs, _ = env.reset(seed=ep_seed)
    ...
    returns[ep_seed] = info_last["<objective>"]   # e.g. profit (maximize) / cost (minimize)
```

Default `n_seeds=65536` provides tight confidence intervals without tuning.

### 9.3 Output columns (TSV)

Columns are the varied scenario parameters followed by per-metric
`{name}_mean` / `{name}_var` pairs. **The domain's primary objective column
comes first among the `*_mean` columns** — that ordering is the contract both
`mdp_tuning` and `mdp_gates` rely on to auto-pick the metric when `--metric`
is not given (neither privileges any particular name). The objective's
**sense** (maximize vs minimize) is the domain's, and callers pass it through
(`mdp_gates --sense`, `mdp_tuning --minimize`).

Example — a pricing domain (`dynamic_pricing`), whose primary objective is
`profit` (maximize):

```
stdev  T  lamb  profit_mean  regret_mean  profit_var  semivar_d  semivar_u
```

Where:
- `profit_mean` / `regret_mean`: sample mean over `n_seeds` episodes (`profit`
  first — it is the objective)
- `profit_var`: sample variance (`ddof=1`)
- `semivar_d`: downside semi-variance — `sum(max(mu - x, 0)^2) / (n-1)` for x in profits
- `semivar_u`: upside semi-variance — `sum(max(x - mu, 0)^2) / (n-1)`

A cost-minimizing domain would instead lead with `cost_total_mean` /
`cost_total_var` and be gated with `--sense minimize`.

### 9.4 Incremental output

Open the output file before the parameter grid loop and write + flush each row immediately. This makes results visible in real time without buffering:

```python
with open(outfile, "w") as f:
    f.write(header + "\n")
    f.flush()
    for combo in grid:
        stats = evaluate_scenario(...)
        row = format_row(combo, stats)
        print(row, flush=True)
        f.write(row + "\n")
        f.flush()
```

### 9.5 VecNormalize injection (PPO eval only)

The trained PPO model may have been trained with `VecNormalize`. To evaluate with seeded episodes while preserving normalization:

1. Wrap env in `DummyVecEnv`, load `VecNormalize` with `training=False`, `norm_reward=False`.
2. Inject the episode seed via `venv.env_method("reset", seed=ep_seed)[0]` — this returns `(raw_obs, info)` from the underlying env, bypassing VecNormalize's `reset()`.
3. Normalize the returned observation manually: `obs = venv.normalize_obs(raw_obs[np.newaxis, :])`.
4. Continue with `venv.step(action)` as usual — subsequent observations are normalized automatically.

```python
raw_obs, _ = venv.env_method("reset", seed=ep_seed)[0]
obs        = venv.normalize_obs(raw_obs[np.newaxis, :])
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done_arr, infos = venv.step(action)
    done = bool(done_arr[0])
```

The `vecnorm-path` defaults to `<model_dir>/vecnormalize.pkl`. If the file does not exist, evaluation proceeds without normalization.

### 9.6 Scenario dispatch and outfile defaults

**Scenario dispatch**: eval scripts accept `-s/--scenario_name` (a
`SCENARIOS` key) or `-g/--grid_name` (a `GRIDS` key). The three targets have
three distinct evaluation semantics — never reconstruct a grid from a
sampler's value lists; enumeration is the grid's native mode (§5.6):

```python
if args.grid_name:                     # generalist target: one row per cell
    grid = GRIDS[args.grid_name]
    targets = list(grid)               # [(cell_id, source), ...] canonical order
else:
    source = SCENARIOS[args.scenario_name]
    targets = [(args.scenario_name, source)]

seeds = eval_seed_block(args)          # ONE block, reused across all targets (CRN, §5.6)
for cell_id, source in targets:
    for episode_seed in seeds:
        sc = source(episode_seed) if callable(source) else source
        ...                            # roll out on the concrete scenario
```

- A **fixed scenario** target reports its metric directly.
- A **world-sampler** target (hidden/observed latent, mixture) reports one
  *expected* metric over the seed block — its latent is integrated, never
  enumerated (per-latent slices are diagnostics only, §5.5).
- A **grid** target reports one row per cell (`{grid_name}/{cell_id}`), all
  cells on the same seed block so comparisons are paired.

**Train/eval decoupling**: a model trained on one scenario can be evaluated on any other. The eval script's `-s` argument names the *eval* scenario, independent of what scenario the model was trained on. The eval scenario is encoded in the output filename to prevent overwrites when the same model is evaluated on multiple scenarios.

**Outfile defaults**:
- PPO eval: `model_path.parent / f"ppo_eval_{args.scenario_name}.tsv"` — co-located with the model
- Benchmark eval: `{domain_dir}/results/{args.scenario_name}/benchmark/benchmark_{name}_eval_{scenario_name}.tsv`, where `{name}` is the method (`dp`) or the specific policy variant (`myopic`, `random`, …) — under the scenario's `benchmark/` folder

**Benchmark solutions file**: a precomputed solution table (e.g. a DP policy table) lives at `results/{scenario_name}/benchmark/{method}/{scenario_name}.txt`. These are artifacts analogous to a trained model: the table covers one scenario family and is referenced by `--solutions` at eval time. A model trained on `simple` can still reference the parent scenario's solutions file if `simple`'s parameters are a subset of that grid.

### 9.7 Multiple policies per benchmark solver

A benchmark solver may expose several closely-related policies through a `--policy` argument (e.g. a fluid-relaxation file offering `fixed`, `myopic`, and `random`), rather than one file per policy. Name the file after the method family it embodies (`{domain}_benchmark_fluid.py`), and encode the selected policy in the eval TSV name (`benchmark_myopic_eval_{scenario}.tsv`) so each benchmark run writes a distinct, gate-referenceable file.

---

## 10. Comments and Docstrings Style

- **Module docstring**: must be the **first statement** in the file, before all imports. Describes purpose, lists key components, notes important constraints. For `{domain}_uncertainty.py` and the other domain-model files, the docstring states the dependency position (§1.1).
- **Class docstrings**: one-paragraph purpose; multi-line only for classes with non-obvious behaviour.
- **`{Domain}State` docstring**: always includes the simulator-perspective disclaimer and pointer to the gym wrapper.
- **Inline comments**: only for non-obvious WHY (hidden constraint, subtle invariant, workaround). No narration of what the code does.
- No multi-line comment blocks; no task or PR references in comments.

---

## 11. Logging

- Two log streams: one per episode, one per step.
- Logger names passed to `logging.getLogger`: `"episode_logger"` and `"step_logger"` (these are the verbal/canonical names shared across domains).
- Attribute names on the env instance are short aliases (e.g. `self.logger_e`, `self.logger_s`) — the verbal names above are what matter for cross-domain consistency, not the attribute names.
- Both loggers initialized in the gym wrapper when `logger_filename is not None`.
- Log filenames: `{filename}_episode.log` and `{filename}_step.log`.
- `filename` / `outdir` encodes run identity (scenario, modes, algo, hyperparams, timestamp).

---

## 12. Deployable Policy (`{domain}_policy.py`)

The training pipeline's final artifact is a Python file exposing the trained
model behind a small, stable interface, so a caller needs no SB3/Gym knowledge:

```python
class {Domain}Policy:
    def __init__(self, model_path, vecnorm_path=None, action_mode="...", scenario="..."):
        ...
    def act(self, obs) -> decision:   # deterministic by default
        ...
```

It bundles the three things that are part of the trained model's I/O contract:

1. **The SB3 model** (`.zip`), loaded on CPU.
2. **VecNormalize observation stats** (`vecnormalize.pkl`, §8.3/§9.5):
   `act()` applies `(obs - obs_rms.mean) / sqrt(obs_rms.var + eps)` with the
   saved clip — the policy was trained on normalized inputs, so the stats are
   mandatory at inference, not an optimization. Default the path to
   `vecnormalize.pkl` next to the model.
3. **The action transform** for the trained `action_mode` (reparametrized
   continuous actions, §7.2) or the action mask hookup (§7.1). Import this
   from the domain module that owns it (`{domain}_gym` / `{domain}_mdp`) —
   do not duplicate the math; the policy file is intentionally *not*
   standalone for constrained/masked domains, and its docstring states that
   dependency.

Requirements:

- **Documented observation contract** in the module docstring: the exact
  feature list, order, and units the caller must supply for each supported
  `observation_mode`.
- **`action_mode` / `observation_mode` / scenario must match training.** The
  run directory's `{scenario_name}_{algo}_args.txt` records them; the policy
  constructor takes them explicitly rather than guessing.
- **`__main__` smoke test** that replays the policy against the raw
  `{domain}_mdp` loop (not the gym) for a few episodes and prints per-episode
  outcomes — proving the wrapper's normalization + transform reproduce the
  eval-time behavior without the gym stack.

---

## 13. Eval Gate (`mdp_gates`)

The repo-level `mdp_gates` package is the executable form of the "learned
policy beats the baselines" criterion. It compares spec-§9 eval TSVs on a
`*_mean` metric with a standard error built from the matching `*_var` column
and the shared seed count:

```bash
python -m mdp_gates \
  --candidate results/simple/PPO_.../ppo_eval_simple.tsv \
  --baseline  results/simple/benchmark/benchmark_random_eval_simple.tsv \
  --baseline  results/simple/benchmark/benchmark_myopic_eval_simple.tsv \
  --reference results/simple/benchmark/benchmark_dp_eval_simple.tsv \
  --n-seeds 8192
```

- The gate **passes** (exit 0) when the candidate beats every `--baseline` by
  `z >= 2` standard errors; `--reference` files (e.g. the DP optimum) are
  reported as a gap / % of reference but never gate.
- **`--sense` must match the domain's objective sense.** The default is
  `maximize` (higher `*_mean` is better, e.g. `revenue_mean`); pass
  `--sense minimize` when the gated metric is lower-is-better (a raw
  `cost_total_mean`, `regret_mean`, loss, …). Getting this wrong silently
  inverts the verdict — a minimize domain gated with the default passes only
  when the candidate is *worse*. Prefer gating on a higher-is-better return
  column when the eval TSV reports one; otherwise set `--sense minimize`
  explicitly. (This mirrors `mdp_tuning`'s `--minimize`.)
- All TSVs must come from the same seed protocol (seeds `0..n-1`, §9.2).
  Shared seeds make the unpaired SE conservative, so a PASS is trustworthy.
- **Post-tuning rule**: a tuning study's winner was *selected* on its tuning
  eval seeds, so its selection score is optimistic. Always re-evaluate the
  winning artifact with the full protocol (more seeds than the tuning eval)
  before reporting or gating it.

# MDP Project Specification

Canonical patterns for MDP simulation domains; originally derived from a
multi-entity reference implementation (`owmr`, a one-warehouse multi-retailer
domain, cited throughout as an illustration — its patterns are described
inline wherever referenced). Shipped conformant examples: `examples/inv_single`
and `examples/dynamic_pricing`. All new MDP domains should follow these
conventions.

---

## 1. File Layout

Each domain lives in its own subfolder `{domain}/`. Every file is prefixed with the domain name.

| File | Purpose |
|---|---|
| `{domain}_exceptions.py` | Custom exception hierarchy (optional) |
| `{domain}_uncertainty.py` | Stochastic primitives: the `SamplingContext` protocol and all `{Source}Generator` classes (demand, leadtime, …) — **omit entirely when the dynamics are deterministic** (§4.3) |
| `{domain}_scenarios.py` | The `{Domain}Scenario` / `{Domain}ScenarioSampler` classes, their predefined instances, and the `SCENARIOS` registry |
| `{domain}_mdp.py` | Core MDP simulator: `{Domain}State` and the state-transition functions |
| `{domain}_gym.py` | Gymnasium single-agent wrapper |
| `{domain}_ppo_train.py` | SB3 PPO training (drop "gym" if it is the only env type) |
| `{domain}_ppo_tune.py` | Optional thin wrapper over the repo-level `mdp_tuning` harness, pre-filling domain defaults (e.g. `--metric`) |
| `{domain}_dreamerv3_train.py` | RLlib DreamerV3 training |
| `{domain}_ppo_eval.py` | Evaluate a trained RL model over the full parameter grid |
| `{domain}_dp_eval.py` | Evaluate the DP / heuristic baseline over the full parameter grid |
| `{domain}_lp.py` | LP / heuristic baseline solver |
| `{domain}_policy.py` | Deployable policy wrapper over the trained artifact (§12) |

### 1.1 Dependency chain

The three domain-model files form a strict, acyclic layering — each imports only from the ones to its left:

```
{domain}_uncertainty  ←  {domain}_scenarios  ←  {domain}_mdp
   (generators,            ({Domain}Scenario,       ({Domain}State,
    SamplingContext)        SCENARIOS)               transitions)
```

- **`{domain}_uncertainty.py`** is the base: it has no domain imports (only `numpy` and `typing`). It must **not** import `{Domain}State` — see the `SamplingContext` protocol in §4.1.
- A domain with **deterministic dynamics** has no `{domain}_uncertainty.py` at all; the chain shortens to `{domain}_scenarios ← {domain}_mdp`. Initial conditions belong to the scenario, and per-episode variety comes from a `{Domain}ScenarioSampler` (§4.3, §5.2) — e.g. `sudoku`, where the puzzle *is* the scenario and placements are deterministic.
- **`{domain}_scenarios.py`** imports the generator classes from `{domain}_uncertainty` and composes them into `{Domain}Scenario` instances.
- **`{domain}_mdp.py`** imports `{Domain}Scenario` from `{domain}_scenarios` (for type annotations) and the generator classes from `{domain}_uncertainty` (only where it needs to construct them, e.g. its `__main__` smoke test).
- **Gym wrappers** sit on top of all three: `{Domain}State` and transition functions from `{domain}_mdp`, `{Domain}Scenario` from `{domain}_scenarios`, generator classes from `{domain}_uncertainty` as needed.

---

## 2. Naming Conventions

| Concept | Pattern | Example |
|---|---|---|
| Generator base class | `{Source}Generator` | `DemandGenerator`, `LeadtimeGenerator` |
| Generator leaf class | `{Variant}{Source}` | `PoissonDemand`, `NormalDemand`, `DiscreteLeadtime` |
| Sampling-context protocol | `SamplingContext` or `{Source}Context` | `SamplingContext`, `SpawnContext`, `DemandContext` |
| Scenario config class | `{Domain}Scenario` | `OwmrScenario` |
| Scenario sampler class | `{Domain}ScenarioSampler` | `FnvScenarioSampler` |
| MDP state class | `{Domain}State` | `OwmrState` |
| Gym env class | `{Domain}Env` | `OwmrEnv` |
| Scenario instances | `scenario_{name}` | `scenario_simple` |
| Scenario sampler instances | `sampler_{name}` | `sampler_default` |
| Scenario registry dict | `SCENARIOS` | `SCENARIOS = {"simple": scenario_simple, "random": sampler_default, ...}` |
| Scenario registry key | `scenario_name` | `scenario_name="simple"` |
| Scenario description | `desc` | `desc="zero lead times everywhere"` |
| Constructor argument for scenario | `scenario` | `def __init__(self, scenario, ...)` |
| Attribute storing scenario on env | `self.scenario` | `self.scenario: OwmrScenario` |

Which file each class lives in:

- **`{domain}_uncertainty.py`**: `SamplingContext`, `{Source}Generator` and its leaf classes.
- **`{domain}_scenarios.py`**: `{Domain}Scenario`, `{Domain}ScenarioSampler`, `scenario_{name}` / `sampler_{name}` instances, `SCENARIOS`.
- **`{domain}_mdp.py`**: `{Domain}State`, transition functions.

**Never use `param`, `params`, or `param_*`** anywhere in the codebase.

---

## 3. Exceptions (`{domain}_exceptions.py`)

**This file is optional.** A domain with no domain-specific constraint failures
may omit it and rely on plain `assert`s / built-in exceptions (as `inv_single`,
`cnv`, and `fnv` do). Provide it when the MDP raises structured, catchable errors
that a caller may want to distinguish (as `owmr` and `2048` do).

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

Name it `SamplingContext` by default. A domain with a single dominant source of randomness may instead name it after that source — `{Source}Context` (e.g. `SpawnContext` in `2048`) — when that reads more naturally; the structural-typing mechanism is identical either way.

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
class {Source}Generator:
    """Abstract base for {source} generators."""

    _STREAM_ID: int   # unique stream id for this source; subclasses set it

    def __init__(self, id: int) -> None:
        self.id = id   # entity index (e.g. retailer); leads the seed key

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
    _STREAM_ID = 1

    def __init__(self, id: int, mu: float, sigma: float) -> None:
        super().__init__(id=id)
        self.mu    = mu
        self.sigma = sigma

    def sample(self, ctx: SamplingContext) -> float:
        ss = np.random.SeedSequence(
            [self.id, ctx.period, self._STREAM_ID, ctx.episode_seed, ctx.seed_salt]
        )
        rng = np.random.default_rng(ss)
        return rng.normal(loc=self.mu, scale=self.sigma)

    def mean(self) -> float:
        return self.mu

    def max(self) -> float:
        return self.mu + 4.0 * self.sigma
```

- **`sample(ctx)` owns the full seed construction** — `advance()` calls `scenario.demand.sample(state)` and never builds an rng itself. The `ctx` argument is any `SamplingContext` (in practice the `{Domain}State`).
- **`_STREAM_ID`** is the per-source stream id in the seed key (§6.3): `1 = demand`, `2 = leadtime`, etc. All leaf classes of one source share the same id.
- **`id`** (constructor arg) is the entity index (e.g. retailer index) and leads the seed key so two retailers with identical parameters still draw independent values.
- **`mean()`** returns the expected value for deterministic planners (LP/DP baselines).
- **`max()`** returns a practical upper bound (e.g. mean + 4σ); used by the gym wrapper to set action/observation space bounds without isinstance checks. Leadtime generators additionally expose **`min()`**.
- **`is_discrete`** (class attribute) records whether samples are integer-valued, so consumers can pick integer vs. float spaces without isinstance checks.
- The generator instance is stored as a field on `{Domain}Scenario` (e.g. `scenario.demand`, `scenario.rt_leadtimes[i]`).

### 4.3 Intrinsic vs. meta-level randomness — the boundary

*Intrinsic* randomness unfolds **during** the dynamics: it is realized per
transition and keyed by something that advances within the episode (`period`,
a draw counter). *Meta-level* randomness selects **which problem instance**
the episode poses: it is realized once at episode start and keyed by
`episode_seed` only. Only intrinsic randomness belongs in this file;
meta-level randomness is `{Domain}ScenarioSampler` territory (§5.2, stream 0).
Getting this wrong produces a plausible-looking but mislayered domain, so
apply the tests below to every source whose classification is not obvious:

1. **Timing.** A draw that depends only on `(episode_seed, seed_salt)` — no
   `period`, no per-transition counter — is scenario sampling in disguise.
   Model it as a `{Domain}ScenarioSampler` and make the scenario hold the
   *realized* value. (The conformance harness flags such generators.)
2. **Nameability.** If a fixed realization of the draw is something you would
   hand-craft, name, or benchmark against (a specific puzzle board, a specific
   cost structure), it is a scenario. A scenario must be a fully concrete
   problem instance (§5.1); distributions over instances are expressed only
   through samplers.
3. **Episode-start draws.** An episode-start draw is intrinsic only when it is
   the first step of the *same stochastic process* that continues per
   transition (2048's initial tile spawn, drawn by the same `SpawnGenerator`
   as every later spawn). A one-shot draw with no per-transition sibling
   defaults to meta-level.

A domain where **all** randomness is meta-level has deterministic dynamics and
no `{domain}_uncertainty.py` at all (e.g. `sudoku`: the puzzle is the
scenario; placements are deterministic).

---

## 5. Scenarios (`{domain}_scenarios.py`)

This module defines the `{Domain}Scenario` config class (and optional `{Domain}ScenarioSampler`), all predefined instances, and the `SCENARIOS` registry. It imports generator classes from `{domain}_uncertainty` and composes them.

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
  values a study sweeps to build a family of `instances`. *Categorical*
  attributes (bool flags and string enums — e.g. `allow_backlog`,
  `event_sequence`, `mmfe_mode`) select a **scenario mode**: a qualitatively
  distinct variant of the dynamics, not a value to interpolate. Group the two
  separately in the field list. An experiment design is then a grid over the
  numerical axes crossed with a chosen set of scenario modes; *how* modes
  compose with the grid, and whether they are compared head-to-head or kept as
  separate branches, is a per-study choice, not a property of the class. (These
  scenario modes live on the `mdp` layer — distinct from the gym's
  obs/action/reward *modes*, §7.)
- `seed_salt` controls scenario-level RNG; `scenario_name` is the registry key. A deterministic-dynamics domain (§4.3) has **no `seed_salt` on the scenario or the state** — there is no intrinsic randomness to salt; only its `{Domain}ScenarioSampler` keeps one, for the per-episode instance draw.
- **A scenario is a fully concrete problem instance**: together with an `episode_seed` it must pin the episode completely. Anything one might hand-craft, name, or benchmark against (a specific puzzle board, a specific demand profile) must be representable as a fixed scenario; distributions over instances live only in samplers (§5.2). If a "scenario" still needs a random draw to become a concrete problem, that draw belongs in a `{Domain}ScenarioSampler`, not in `{domain}_uncertainty.py` (§4.3).
- **`seed_salt` must be declared `field(default=<default_int>, repr=False)`** so it is excluded from the dataclass `repr`. It is an internal reproducibility knob, not a meaningful configuration value to a reader; keeping it out of the `repr` avoids cluttering the `print(scenario)` output that training scripts emit at startup (§8.2).
- **`scenario_name`** is the short registry key (e.g. `"simple"`, `"test1a"`); **`desc`** is a human-readable one-line description of the scenario. Keep the name short and put the explanation in `desc` rather than encoding everything into a long name. Prefer `desc` over a leading `#` comment on the instance, so the description travels with the object: drivers and eval scripts can print a self-describing header, e.g. `print(f"=== {scenario.scenario_name}: {scenario.desc} ===")`.
- Validate all structural constraints in `__post_init__`.

### 5.2 `{Domain}ScenarioSampler` (optional)

When the training distribution spans a family of scenarios rather than a single fixed one, define a callable sampler class:

```python
@dataclass(slots=True)
class {Domain}ScenarioSampler:
    """Samples a {Domain}Scenario at the start of each episode."""

    # parameter grids / ranges
    stdev_values: tuple[float, ...]
    # ...

    # fixed family-level attributes (same value across all generated scenarios)
    # expose here so gym wrapper / eval scripts can read them without generating a scenario first
    some_mode: str = "default"

    # reproducibility
    seed_salt: int = <default_int>

    # identifier (same field as on {Domain}Scenario)
    scenario_name: str | None = None

    def __call__(self, episode_seed: int) -> {Domain}Scenario:
        ss  = np.random.SeedSequence([0, episode_seed, self.seed_salt])
        rng = np.random.default_rng(ss)
        stdev = rng.choice(self.stdev_values)
        # ... sample other parameters ...
        return {Domain}Scenario(stdev=stdev, ...)
```

- Lives in `{domain}_scenarios.py`, alongside `{Domain}Scenario`.
- Callable: `sampler(episode_seed) -> {Domain}Scenario`.
- Uses **stream 0** (`[0, episode_seed, seed_salt]`) — episode-level draws omit `period`.
- This is **meta-level** randomness (sampling scenario parameters), **not** intrinsic MDP randomness (demand, leadtimes). Do not model it as a `{Source}Generator` — see the boundary tests in §4.3.
- A sampler may draw **entire problem instances** (e.g. a full sudoku board carved from a random solved grid), not just scalar parameters. Generation helpers for such instances live in `{domain}_scenarios.py` as module-level functions, and the returned `{Domain}Scenario` holds the realized instance.
- The gym wrapper checks `callable(self.scenario)` and calls `self.scenario(episode_seed)` in `reset()` to get the concrete scenario for that episode.
- Instances are named `sampler_{name}`.
- The `SCENARIOS` dict may hold either a `{Domain}Scenario` or a `{Domain}ScenarioSampler` as values.
- **`scenario_name`**: set to the same key used in `SCENARIOS` (e.g. `scenario_name="FNV-aMMFE"`) so training/eval scripts can read it without an isinstance check.
- **Family-level attributes**: if the gym wrapper or eval scripts need a mode attribute (e.g. `mmfe_mode`) before generating a concrete scenario, expose it as a field on the sampler with the same name and value as the family's fixed setting.

### 5.3 Instances and registry

```python
scenario_simple = {Domain}Scenario(scenario_name="simple", desc="...", ...)
scenario_{name} = {Domain}Scenario(scenario_name="{name}", desc="...", ...)

sampler_default = {Domain}ScenarioSampler(scenario_name="random", ...)

SCENARIOS = {
    "simple":  scenario_simple,
    "{name}":  scenario_{name},
    "random":  sampler_default,   # sampler entries allowed
}
```

- Fixed scenario instances are named `scenario_{name}`; sampler instances are named `sampler_{name}`.
- `SCENARIOS` values may be either a `{Domain}Scenario` or a `{Domain}ScenarioSampler`.
- `SCENARIOS` dict provides the canonical lookup for training and evaluation scripts.
- Small smoke-test scenarios used only by the `{domain}_mdp.py` `__main__` driver may be defined here as plain module-level instances and imported by the driver; they need not be added to `SCENARIOS` if they are not meant as training/eval targets.

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

### 6.3 RNG pattern

All random draws use `numpy.random.SeedSequence`. The key list is structured as a **reverse tree**: from the most specific (lowest-level) identifier to the most general (highest-level), so that draws at different scopes never collide:

```
[most specific ← ... → most general]
[entity_id,  period,  stream_id,  episode_seed,  seed_salt]
 └ draw-level ─────────────────┘   └ episode ┘   └ scenario ┘
```

```python
ss = np.random.SeedSequence([
    entity_id,          # e.g. retailer index; omit if there is no entity structure
    state.period,       # current period
    stream_id,          # 1 = demand, 2 = leadtime, 0 = scenario sampling, etc.
    state.episode_seed,
    state.seed_salt,
])
rng = np.random.default_rng(ss)
```

- Omit `entity_id` when the domain has no multi-entity structure (e.g. single-agent FNV).
- `stream_id` distinguishes different sources of randomness at the same period (demand vs. leadtime); it is the generator's `_STREAM_ID` (§4.2).
- Episode-level draws (e.g. scenario parameter sampling) omit `period` and use a dedicated stream (typically `stream_id=0`).
- This structure makes each draw uniquely identified and independent of the decision path.

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
- 1-indexed domains (`cnv`, `fnv`) start at `state.period = 1`, so `init` gives `action_period = 0`.
- Two-step domains (`owmr`, `inv_single`): `advance1` and `advance2` share the same `action_period = n`; only `advance2` advances `state.period` to `n+1`.

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
        scenario: {Domain}Scenario,
        action_mode: str = "...",
        observation_mode: str = "vec",
        logger_filename: str | None = None,
    ) -> None:
        ...
        self.scenario: {Domain}Scenario = scenario
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()
        ...

    def reset(self, seed=None, options=None) -> tuple[obs, info]: ...
    def step(self, action) -> tuple[obs, reward, terminated, truncated, info]: ...
```

- **Convention:** call MDP *functions* through a module alias (`import {domain}_mdp as mdp`, then `mdp.init_state(...)`, `mdp.advance(...)`, `mdp.valid_actions(...)`); reserve `from ... import ...` for *types* used in annotations — `{Domain}State` from `{domain}_mdp`, `{Domain}Scenario` from `{domain}_scenarios`, and generator classes from `{domain}_uncertainty` if needed. This keeps every transition-function call visibly namespaced and avoids importing the same module two ways for the same kind of name.
- `self.scenario` is the stored attribute name.
- Action and observation spaces built in private `_build_action_space()` / `_build_observation_space()` methods.
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

**Keep `valid_actions` cheap.** It is called every step during training. If checking validity requires re-running the transition function (as in 2048, where `_try_move` is called up to n times), ensure the underlying check is O(board size), not O(horizon).

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

**Simplex reparametrization in detail (OWMR example)**

OWMR ships inventory from a warehouse to N retailers. The allocation decisions must satisfy: `sum(shipments) <= wh_onhand` — i.e., they lie on a scaled simplex.

Action mode `"order&allocations"` in `owmr_gym.py`:

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
- `print(scenario)` at the top of `main()` to confirm configuration.
- Env constructed with `scenario=scenario` (keyword argument, never positional).
- RLlib `env_config` dict must use `"scenario"` as the key (RLlib calls `Env(**config)`).
- **Do not hard-code `gamma`** in PPO kwargs — expose it as a CLI argument instead.
- **Pin BLAS/torch threads when launching training** (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`): the policies in these domains are tiny, so torch's default all-cores threading adds sync overhead rather than speed, and on a shared machine it oversubscribes cores already used by other jobs (measured on sudoku 9x9: 9 min → 11 s for 2048 steps on a box concurrently running an 8-core workload; expect a smaller but still real gain on an idle box). The `mdp_tuning` harness sets this for its subprocesses automatically.

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
- **Little benefit** when obs are already **homogeneous and bounded** on a common small scale (e.g. one-hot planes, or log2-scaled tiles in 2048) — there is no cross-feature imbalance to fix; a static encoding is enough.
- **Risk** when the obs distribution is **non-stationary** — i.e. what the agent observes shifts as it improves (e.g. 2048's reachable tiles keep growing). The running `obs_rms` never stabilizes and the frozen eval stats match no single stage ("stats-mismatch under distribution shift"). Domains with this property may deliberately omit VecNormalize (see `inventory/inventory_train_PPO.py`).

Reward norm is training-only and largely redundant with PPO's `normalize_advantage`, so the decision above is really about **obs** norm.

### 8.4 Output directory structure and run name encoding

Training outputs go to `results/{scenario_name}/{run_name}/` relative to the domain directory:

```
results/
  {scenario_name}/                   # e.g. simple/, FNV-aMMFE/
    {run_name}/                      # e.g. PPO_20260630_224416_default/
      {scenario_name}_{algo}.zip     # saved SB3 model
      vecnormalize.pkl               # VecNormalize running stats (if used)
      {scenario_name}_{algo}_args.txt  # all CLI args at training time
      train_log_episode.log          # per-episode log
      train_log_step.log             # per-step log
      monitor.monitor.csv            # SB3 Monitor CSV
      {ALGO}_1/                      # TensorBoard event files
      ppo_eval_{eval_scenario}.tsv   # eval output (eval scenario encoded in filename)
    dp/
      {scenario_name}.txt            # precomputed DP/heuristic solutions for this scenario
      dp_eval_{scenario_name}.tsv    # DP eval output
```

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
- Model: `{scenario_name}_{algo}.zip` (e.g. `FNV-aMMFE_ppo.zip`)
- Args log: `{scenario_name}_{algo}_args.txt`
- Eval output: `ppo_eval_{eval_scenario}.tsv` — the eval scenario is encoded in the filename so that evaluating the same model on different scenarios produces non-overwriting files (see §9.6)

### 8.5 Policy selection (MLP vs CNN)

Pick the SB3 policy family from the **observation shape**, with a CLI override:

- `ndim >= 3` (channels-first spatial obs, e.g. `(C, N, N)` from an `array3d`/`onehot` observation_mode) → `CnnPolicy`.
- `ndim <= 2` (flat vectors; non-spatial matrices such as entity × pipeline-slot tables) → `MlpPolicy`.
- Expose `--policy {auto,mlp,cnn}` with default `auto`; resolve once in `main()` via a `_resolve_policy(policy, obs_shape)` helper and print the resolution.

The shape is a reliable proxy only because the gym conventions make it one: spatial observation modes deliberately emit channels-first 3D arrays (`(1, N, N)` for a raw board), while non-spatial features stay flat. A bare 2D obs is treated as non-spatial — genuinely spatial boards are already `(1, N, N)` by convention.

**SB3's default NatureCNN must not be used** — its Atari-sized kernels (8×8 stride 4) crash on small boards. The default extractor is a small-board CNN (`SmallBoardCnn` in the train script, following the 2048 precedent of extractors living there):

- stacked stride-1, same-padding convolutions, channels `(32, 64)`, kernel 3 clamped to `min(3, H, W)`;
- **no pooling** — absolute position matters on game boards;
- flatten → `Linear(features_dim=128)` + ReLU;
- `--net_arch` keeps meaning the MLP / pi-vf head widths in both families.

Expose `--features_dim`, `--channels`, `--kernel_size` as CLI arguments so the `mdp_tuning` harness can reach them (its PPO space tunes `features_dim`/`channels` whenever the script exposes them). Domains may add specialized extractors beyond the default (e.g. 2048's row/column kernels) behind a `--cnn_arch` choice.

---

## 9. Evaluation Scripts

Two parallel evaluation scripts are expected for each domain: one for the trained RL model (`{domain}_ppo_eval.py`) and one for the DP/heuristic baseline (`{domain}_dp_eval.py`). They share the same output format for direct comparison.

### 9.1 Common structure

```python
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="...")
    # for PPO eval:
    p.add_argument("--model-path",   type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None)
    # for DP eval:
    p.add_argument("--dp-solutions", type=str, required=True)
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

For each `(scenario_param_combo)`, run exactly `n_seeds` episodes using episode seeds `0, 1, ..., n_seeds-1`:

```python
profits = np.zeros(n_seeds)
regrets = np.zeros(n_seeds)

for ep_seed in range(n_seeds):
    if ep_seed % 10000 == 0:
        print(f"  seed {ep_seed}/{n_seeds}", flush=True)
    obs, _ = env.reset(seed=ep_seed)
    ...
    profits[ep_seed] = info_last.get("profit", 0.0)
    regrets[ep_seed] = info_last.get("profit", 0.0) - info_last.get("profit_max", 0.0)
```

Default `n_seeds=65536` provides tight confidence intervals without tuning.

### 9.3 Output columns (TSV)

```
stdev  T  lamb  profit_mean  regret_mean  profit_var  semivar_d  semivar_u
```

Where:
- `profit_mean` / `regret_mean`: sample mean over `n_seeds` episodes
- `profit_var`: sample variance (`ddof=1`)
- `semivar_d`: downside semi-variance — `sum(max(mu - x, 0)^2) / (n-1)` for x in profits
- `semivar_u`: upside semi-variance — `sum(max(x - mu, 0)^2) / (n-1)`

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

**Scenario dispatch**: eval scripts accept `-s/--scenario_name`. The looked-up value may be a `{Domain}Scenario` (evaluate a single point) or a `{Domain}ScenarioSampler` (evaluate the full parameter grid). Branch on type:

```python
scenario_or_sampler = SCENARIOS[args.scenario_name]
if isinstance(scenario_or_sampler, {Domain}ScenarioSampler):
    sampler = scenario_or_sampler
    grid = [
        (stdev, T, lamb)
        for stdev in sampler.stdev_values
        for T     in sampler.T_values
        for lamb  in sampler.lamb_values
    ]
else:
    sc   = scenario_or_sampler
    grid = [(sc.stdev, sc.T, sc.lamb)]
```

**Train/eval decoupling**: a model trained on one scenario can be evaluated on any other. The eval script's `-s` argument names the *eval* scenario, independent of what scenario the model was trained on. The eval scenario is encoded in the output filename to prevent overwrites when the same model is evaluated on multiple scenarios.

**Outfile defaults**:
- PPO eval: `model_path.parent / f"ppo_eval_{args.scenario_name}.tsv"` — co-located with the model
- DP eval: `{domain_dir}/results/{args.scenario_name}/dp/dp_eval_{scenario_name}.tsv` — under the scenario's DP folder

**DP solutions file**: precomputed DP/heuristic solutions live at `results/{scenario_name}/dp/{scenario_name}.txt`. These are artifacts analogous to a trained model: the DP solutions file covers one scenario family and is referenced by `--dp-solutions` at eval time. A model trained on `simple` can still reference the parent scenario's solutions file if `simple`'s parameters are a subset of that grid.

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
  --baseline  results/simple/dp/lp_eval_simple_random.tsv \
  --baseline  results/simple/dp/lp_eval_simple_myopic.tsv \
  --reference results/simple/dp/dp_eval_simple.tsv \
  --n-seeds 8192
```

- The gate **passes** (exit 0) when the candidate beats every `--baseline` by
  `z >= 2` standard errors; `--reference` files (e.g. the DP optimum) are
  reported as a gap / % of reference but never gate.
- All TSVs must come from the same seed protocol (seeds `0..n-1`, §9.2).
  Shared seeds make the unpaired SE conservative, so a PASS is trustworthy.
- **Post-tuning rule**: a tuning study's winner was *selected* on its tuning
  eval seeds, so its selection score is optimistic. Always re-evaluate the
  winning artifact with the full protocol (more seeds than the tuning eval)
  before reporting or gating it.

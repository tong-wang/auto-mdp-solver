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

Each domain lives in its own subfolder `{domain}/`. Every code file is
prefixed with the domain name; the campaign docs (`CLAUDE.md`, `README.md`,
`ESCALATION.md`, …) are deliberately not.

| File | Purpose |
|---|---|
| `CLAUDE.md` | Domain-local operating brief — emitted at Stage 1 from the skill's `DOMAIN_CLAUDE_TEMPLATE.md`. Pointer-first: where to look + what bites, never a second copy of the README |
| `{domain}_exceptions.py` | Custom exception hierarchy (optional) |
| `{domain}_uncertainty.py` | Stochastic primitives: the `SamplingContext` protocol and all `{Source}Generator` classes (demand, leadtime, …) — including latent-bearing generators that own their source's per-episode world latent (§5.2) — **omit entirely when the dynamics are deterministic** (§4.3) |
| `{domain}_scenarios.py` | **World layer — composition only**: the `{Domain}Scenario` / `{Domain}ScenarioSource` / `{Domain}MixtureSampler` classes, their predefined instances, and the `SCENARIOS` registry. Owns no sampling: latents live on generators (§5.2) |
| `{domain}_grids.py` | **Design layer** (optional): the `{Domain}ScenarioGrid` class and the `GRIDS` registry — generality targets for generalist training (§5.6) |
| `{domain}_mdp.py` | Core MDP simulator: `{Domain}State` and the state-transition functions |
| `{domain}_gym.py` | Gymnasium single-agent wrapper |
| `{domain}_ppo_train.py` | SB3 PPO training (drop "gym" if it is the only env type) |
| `{domain}_ppo_tune.py` | Optional thin wrapper over the repo-level `mdp_tuning` harness, pre-filling domain defaults (e.g. `--metric`) |
| `{domain}_dreamerv3_train.py` | RLlib DreamerV3 training |
| `{domain}_ppo_eval.py` | Evaluate a trained RL model over the full parameter grid |
| `{domain}_benchmark_{method}.py` | Non-RL benchmark solver — `{method}` names the method (`lp`, `dp`, `myopic`, `greedy`, `fluid`, or a domain-custom heuristic). One per method; a solver may expose several related policies via `--policy` (§9.8) |
| `{domain}_benchmark_{method}_eval.py` | Evaluate a benchmark over the full parameter grid, same TSV format as the RL eval |
| `{domain}_policy.py` | Deployable policy wrapper over the trained artifact (§12) |
| `{domain}_policy_probe.py` | Policy-interpretation probe: action-surface sweep, structural-form fit + paired scoring of the fitted rule, agreement vs the reference, feature-sensitivity sweeps (§14). **Required when a declared tier-2 stance is `confirm` or `discover`** (`research_questions`, §14.0) — a claim-shaped condition, not a problem-shaped one: a `bypass` campaign owes no probe |
| `{domain}_plot_policy.py` | Policy-overlay figure — one plot spec, static + interactive renders (§14.3) |
| `{domain}_test.py` | The domain's own tests: the engine laws + the differential parametrized over the covering set, plus the claims only this domain can state. **Required for a new domain**; see §1.2 |

### 1.2 `{domain}_test.py`

Each domain owns its tests, in its own folder, beside the code they describe —
a domain-specific claim is only checkable once the domain exists, and a domain
folder must stay portable. Helpers come from `mdp_ir.testing` (never a repo
conftest), so the file works wherever the folder lives:

```python
from mdp_ir.testing import assert_laws, assert_match, schema_beside
SCHEMA = schema_beside(__file__)

def test_engine_laws():
    assert_laws(SCHEMA)                       # every law in mdp_ir.laws

@pytest.mark.parametrize("instance", COMPOSITIONS, ids=lambda i: i or "base")
def test_differential_matches_the_domain(instance):
    assert_match(SCHEMA, instance=instance, episodes=8)
```

Rules:

- **Derive the covering set, never list it.** Read `mdp.scenario.instances` +
  `mixtures` from the schema at collection time, so adding a candidate or an
  instance extends the sweep with no edit to the test. Add one guard that the
  derived list is not silently empty — otherwise every parametrized case
  "passes" by not existing.
- **Put what generalizes in the IR, not here.** A conservation law belongs in
  `mdp.invariants` (every instance and every appended candidate inherits it,
  and both the laws gate and the differential enforce it). Reach for Python
  only where an expression cannot say it: an equivalence against a hand-written
  class, a cross-episode statistic, a claim about a dynamics local.
- **Carry a negative control.** At least one test must corrupt the IR and
  assert the gate *fails* (`assert_diverges`, or a mis-stated claim that
  produces violations). A gate never observed failing is not known to gate.
  The control covers *each* gate the domain claims: two gates on the same
  object are blind to different faults (a σ-drop is invisible to an exact
  agreement gate run at σ's default; a prior-drop is invisible to a
  calibration gate), so one demonstrated failure does not certify the others.
- **Gate every second implementation.** A domain may carry additional
  implementations of its MDP core (a vectorized probe, a batched replay) —
  the differential itself is one, so plurality is licensed, not banned. Each
  declares the equivalence it claims and proves it here:

  | claim | gate |
  |---|---|
  | **bit-exact replica** — same trajectories | exact equality (`== 0.0`, never `allclose`) against `{domain}_mdp`/`{domain}_gym` over shared seeds and actions |
  | **distributional** — same laws, different draws | its deterministic core agrees with the functions the IR names (`expr_builtins`); it *should* also state what stands in for the diff in the region it generalizes into (a calibration check, a known-answer probe) |

  Gates live in this file, never behind a manual `--part` someone must
  remember to run. Any scenario constant a fast path assumes is **derived
  from or asserted against the resolved instance**, so an IR edit fails
  loudly instead of silently narrowing every result to one stale cell. And
  a distributionally-gated implementation licenses **contrasts, not
  levels**: paired differences on its own seed block may be quoted;
  absolute levels come only from the canonical eval (§9).
- **Stay runnable standalone**: end with
  `if __name__ == "__main__": raise SystemExit(pytest.main([__file__]))`.
- **Never add an unprefixed module** to a domain folder. pytest's `prepend`
  import mode puts each test file's directory on `sys.path`, so two folders
  holding the same unprefixed name silently share whichever loaded first. The
  `{domain}_*` prefix rule above is what prevents it.

`[domain]` includes pytest, so the extra that makes a generated domain runnable
also makes its gates runnable — `pip install "auto-mdp-solver[domain]"` is
enough to build a case *and* check it. `[dev]` is pytest on its own, the
torch-free path: `pytest` over `harness/tests` and over a domain's own
`{domain}_test.py` both work under it, since neither imports the training
stack.

### 1.1 Dependency chain

The three domain-model files form a strict, acyclic layering — each imports only from the ones to its left:

```
{domain}_uncertainty  ←  {domain}_scenarios  ←  {domain}_mdp
   (generators,            ({Domain}Scenario,       ({Domain}State,
    SamplingContext)        SCENARIOS)               transitions)
```

- **`{domain}_uncertainty.py`** is the base: it has no domain imports (only `numpy` and `typing`). It must **not** import `{Domain}State` — see the `SamplingContext` protocol in §4.1.
- A domain with **deterministic dynamics** has no `{domain}_uncertainty.py` at all; the chain shortens to `{domain}_scenarios ← {domain}_mdp`. Initial conditions belong to the scenario, and per-episode variety comes from a bespoke callable instance source in `{domain}_scenarios.py` — the one case with no generator to own the draw, so the source carries its own `substream_id` like a mixture (§4.3, §5.2) — e.g. a Sudoku domain, where the puzzle *is* the scenario and placements are deterministic.
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
| Latent-bearing generator (world latent, §5.2) | `Latent{Variant}{Source}` | `LatentPoissonDemand`, `LatentDiscreteDemand` |
| Sampling-context protocol | `SamplingContext` or `{Source}Context` | `SamplingContext`, `SpawnContext`, `DemandContext` |
| Scenario config class | `{Domain}Scenario` | `InvSingleScenario` |
| Scenario source wrapper (realizes latents, §5.2) | `{Domain}ScenarioSource` | `InvSingleScenarioSource` |
| Mixture sampler class (§5.3) | `{Domain}MixtureSampler` | `InvSingleMixtureSampler` |
| Scenario grid class (design layer, §5.6) | `{Domain}ScenarioGrid` | `InvSingleScenarioGrid` |
| Scenario source (term / type alias) | `ScenarioSource` | `{Domain}Scenario \| {Domain}ScenarioSource \| {Domain}MixtureSampler` (any callable counts) |
| MDP state class | `{Domain}State` | `InvSingleState` |
| Gym env class | `{Domain}Env` | `InvSingleEnv` |
| Scenario instances | `scenario_{name}` | `scenario_simple` |
| Scenario source instances | `source_{name}` | `source_poisson` |
| Grid instances | `grid_{name}` | `grid_costs` |
| Scenario registry dict | `SCENARIOS` | `SCENARIOS = {"simple": scenario_simple, "random": source_random, ...}` |
| Grid registry dict | `GRIDS` | `GRIDS = {"cost-sweep": grid_costs, ...}` |
| Scenario registry key | `scenario_name` | `scenario_name="simple"` |
| Scenario description | `desc` | `desc="zero lead times everywhere"` |
| Constructor argument for scenario | `scenario` | `def __init__(self, scenario, ...)` |
| Attribute storing scenario on env | `self.scenario` | `self.scenario: InvSingleScenario` |

Which file each class lives in:

- **`{domain}_uncertainty.py`**: `SamplingContext`, `{Source}Generator` and its leaf classes (concrete and `Latent{...}`), the `intrinsic_key()` **and** `meta_key()` helpers — both seed branches of a source live with the source.
- **`{domain}_scenarios.py`**: `{Domain}Scenario`, `{Domain}ScenarioSource`, `{Domain}MixtureSampler`, `scenario_{name}` / `source_{name}` instances, `SCENARIOS`.
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
- **`max()`** returns a practical upper bound — `mean + k·sd` at `k = 4`, the same convention `mdp_ir.families` derives for a symbolic bound in the IR — used by the gym wrapper to size spaces without isinstance checks. Leadtime generators additionally expose **`min()`**. **`sd()`** returns the spread that bound is built from, so a site can state its own multiple (`mdp_ir.families.sd`, read as `demand.sd` in an IR expression, mirrors it). Take `k` at the call site rather than baking one number into the generator, because the sites do not want the same bound:
  - **Set `k` from exposure, not from habit.** The criterion is that the expected number of breaches across a campaign stays below one: `P(exceed) × draws << 1`. For Poisson demand at `lambda = 6`, a per-period observation bound sees ~3e6 draws in a 9M-step run and a per-episode action cap ~8e3, which puts the breach count at 1.5e3 vs 4.1 at `k = 4`, and 53 vs 0.14 at `k = 5`. The same multiple is loose at one site and wrong at the other; `4` is an example, never a default to inherit unexamined.
  - **An observation envelope should be generous — ideally the model's own domain.** Looseness is free: SB3 leaves a non-image `Box` untouched (`preprocess_obs` is the identity, `VecNormalize` reads its shape), so a policy handed the same observation under a 7× wider box selects the same action. What it is *not* free in is compatibility — `check_for_correct_spaces` compares `low`/`high` at `load()`, and gymnasium's `Box.__eq__` compares them elementwise, so retuning a derived envelope invalidates every saved artifact. Declaring `[0, inf)` where the theory says non-negative is stable by construction, still fails `contains()` on the sign violations a mis-indexed slot produces, and never moves.
  - **An action bound should be as tight as the coverage argument supports.** Slack here is paid directly: cardinality a discrete policy must explore, dead range a `Box` policy must learn to avoid.
  - **An aggregate is not a multiple of a per-period bound.** For a quantity accumulating over `n` periods the mean scales by `n` and the spread by `sqrt(n)` — a sum of independent Poissons is Poisson at the summed rate — so `n * max()` inflates the tail term by `sqrt(n)` (over 2× too loose at `n = 30`). Write `n*mean + k*sqrt(n)*sd`, on the sum. The wrong form reads as obviously correct, which is why it is called out here.
- **`is_discrete`** (class attribute) records whether samples are integer-valued, so consumers can pick integer vs. float spaces without isinstance checks.
- The generator instance is stored as a field on `{Domain}Scenario` (e.g. `scenario.demand`, `scenario.rt_leadtimes[i]`).
- **Family-generic bridge (one per slot, catalog model §10f).** The harness
  ships `mdp_ir.runtime.FamilyGenerator` — this same generator shape over
  *any* registered distribution family (latent recipes included, same v2
  seed template, moments derived from `mdp_ir.families`). Declare one bridge
  per slot after the hand-written classes:

  ```python
  from mdp_ir.runtime import FamilyGenerator

  class Family{Source}(FamilyGenerator, {Source}Generator):
      """Any registered family as {source} — the zero-code path."""
  ```

  A catalog candidate no hand-written class implements is then constructed
  as `Family{Source}.from_ir(ir, "{slot}")` (or `from_parts`) — **a new
  family needs zero new domain code**. Keep hand-written classes where they
  add domain value: an exact `phi()` (DP benchmarks), state-dependent
  settings, domain-named constructor arguments.

### 4.3 Intrinsic vs. meta-level randomness — the boundary

*Intrinsic* randomness unfolds **during** the dynamics: it is realized per
transition and keyed by something that advances within the episode (`period`,
a draw counter). *Meta-level* randomness selects **which problem instance**
the episode poses: it is realized once at episode start and keyed by
`episode_seed` only. Both live with the source that owns them, on different
seed branches: intrinsic draws in `sample()` (branch 1), meta-level draws in
a latent-bearing generator's `realize()` (§5.2, branch 0 of the seed tree,
§6.3) — never inside `sample()`. Under the v2 seed scheme the boundary is
enforced by grammar: an intrinsic key *must* contain the period level, so a
meta-level draw cannot be expressed inside `sample()` without leaving the
key template. Apply the tests below to every source whose classification is
not obvious:

1. **Timing.** A draw that depends only on `(episode_seed, seed_salt)` — no
   `period`, no per-transition counter — is a world latent. Model it as a
   `Latent{...}` generator whose `realize()` returns the realized concrete
   generator (§5.2). This includes draws hiding *inside* a two-stream
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

- **World layer** (`{domain}_scenarios.py`): *composes* the problem. Fixed
  `{Domain}Scenario` instances, `{Domain}ScenarioSource` wrappers realizing
  templates whose generators carry **world latents** — draws nature makes
  once per episode, owned by the generators themselves (§5.2) — and mixtures
  (§5.3). Registry: `SCENARIOS`.
- **Design layer** (`{domain}_grids.py`, optional): models the *experiment*.
  A `{Domain}ScenarioGrid` is a finite set of complete scenario sources — the
  generality target of a generalist policy (§5.5, §5.6). Registry: `GRIDS`.

A grid may contain anything from the world layer; nothing in the world layer
may contain or import a grid, and the `_mdp` / `_gym` layers never see one
(§1.1). The union `{Domain}Scenario | {Domain}ScenarioSource |
{Domain}MixtureSampler` is called a **`ScenarioSource`** — the type of
`SCENARIOS` values, the gym's `scenario` argument, mixture components, and
grid cells.

`{domain}_scenarios.py` imports generator classes from `{domain}_uncertainty`
and composes them:

```python
from {domain}_uncertainty import {Source}Generator, Normal{Source}, ...
from {domain}_exceptions import ...
```

### 5.0 Three layers: model, design, rendering

**The file says which is which.** The `mdp` block is presented in three headed groups, so a reader sees at a glance what an edit will move:

```jsonc
"mdp": {
  "model":     { "quantities": {...}, "out_of_scope": [...], "dynamics": [...], "notation": {...} },
  "design":    { "scenario": { "constants": [...], "instances": {...} } },
  "rendering": { "horizon": {...}, "state_variables": [...], "decisions": [...],
                 "uncertainty_sources": [...], "dynamics": {...}, "objective": {...},
                 "invariants": [...], "expr_builtins": [...], "initial_state": {...} }
}
```

| group | moves | reading |
|---|---|---|
| `model` | freeze token **+** model fingerprint | the theory's statement changed |
| `design` | freeze token | which points this study evaluates changed |
| `rendering` | freeze token **+** rendering fingerprint | the executable form changed |

The grouping is a **file layout only**: in memory the block is flat (`ir.mdp.state_variables`), both layouts load, and every hash is computed after flattening — so `python -m mdp_ir <schema> --regroup` converts a flat file in place and **refuses to write if any fingerprint would move**.

**The tie-breaker for placing anything new**: *delete it — does the model change? does the system evolve differently?* If both answers are no, it is not model. That is why **invariants are rendering**: they are derived from the theory, add no information, and verify rather than dictate — and their expressions are written in rendered names (`sum(pipeline)`), sometimes conditioned on a selection (`stockout_mode == 'lost_sales' or …`). And why the **horizon** splits: "finite horizon, T ≥ 1" is model, the value of `T` is a design constant, and `rendering.horizon` is the executable form that names it.

**Randomness is a model statement.** Each quantity carries `stochastic`, describing what the *theory* permits rather than what a scenario realizes:

```jsonc
"demand":   {"domain": "any density on [0, inf); may differ period to period", "stochastic": true},
"leadtime": {"domain": "integer >= 1", "stochastic": false, "source": "deterministic BY ASSUMPTION"}
```

Two domains can look identical from outside — a leadtime slot with a degenerate default — and mean opposite things: `inv_single` permits stochastic lead time and a scenario selects a point mass (design); `clark_scarf`'s source holds it deterministic (model), so a stochastic candidate would be out of scope. `model.boundary` checks both directions, reading the **catalog**: a declared-stochastic quantity with no uncertainty slot is randomness the rendering dropped; a declared-deterministic quantity that *is* a slot is either out of scope or a wrong model statement. Which candidate an instance selects is design and never triggers a model gate.

Formalizing is a statement about **theory** — what the model admits, what the source puts out of scope. Three layers follow, and Phase A lives entirely in the first:

| layer | question | where it lives |
|---|---|---|
| **model** | what does the theory admit? | `mdp.model` — "lead time L: integer ≥ 1" |
| **design** | which points does this study evaluate? | a scenario constant's `axis`, and the instances — "sweep L ∈ {1, 2}" |
| **rendering** | what does the executable form cap? | a **named scenario constant** at the width site — `length: "leadtime_max"` |

The executable `mdp` block is always a *rendering*: static shapes, unrolled slots and concrete types are what an interpreter can run. The failure this separation prevents is not that the code is specific — it is that **the specific thing gets asserted back as the model.** With no field for the theory, a static width defaults to the sweep's maximum, and the cap is then restated as a property of the problem ("lead time must be 1 or 2"). The human confirmed a *sweep* at sign-off and received a *capacity limit*.

```jsonc
"model": {
  "quantities": {
    "leadtime":   {"domain": "integer >= 1", "source": "the paper's model; L=2 is its worked value"},
    "n_echelons": {"domain": "integer >= 2", "source": "the [N] -> ... -> [1] scheme"}
  },
  "out_of_scope": ["fixed ordering costs (paper §5)"],
  "dynamics": ["forall s in 1..L-1: pipe[s] <- pipe[s+1]"],
  "notation": {"L": "leadtime (paper p. 478)"}
}
```

A rendering width is **not a fourth declaration**. Every width site takes a symbol — `state_variables[].length` (a scalar, or a **list for a multi-dimensional state**: `["grid_size", "grid_size"]`) / `.bounds` / `.element_bounds`, `decisions[].dim`, `decisions[].bounds`, `horizon.T` — so a width *names a scenario constant*, and what it caps and what it renders are **derived from the reference** (the same classification `grids.axes` uses: a constant sizing a state vector is an obs-dim width, one in a decision's bounds is an action-scale width, `horizon.T` is a horizon width). Declaring caps separately would restate what the IR already knows.

**A width takes a symbol; an envelope may take a derivation.** The bounds sites — `state_variables[].bounds` / `.element_bounds`, `decisions[].bounds`, and the `gym.action_modes[].bounds` that mirror them — are the one kind of width that caps no declared design value: they are the *envelope* the gym materializes, and an envelope is often a **function** of several constants (`N * order_max`, `T_dl * demand.max`) rather than one of them. So a bounds entry may be a literal, the name of a scenario constant, or an **expression** over constants (and, in a catalog IR, over a slot's read-API stats — `40 * demand.mean`), and all three forms resolve at the same time against the same pool: per instance, at `mdp.state_bounds()` / `.decision_bounds()` / `action_bounds()` — the last on the IR root, the one place the gym block and the constants pool are both visible, so a domain gating its rendered action space reads the declaration instead of re-implementing how declarations resolve (upstream #70). Nothing above is weakened by this — the constants an expression *reads* are the references `grids.axes` and `model.boundary` classify, exactly as a bare name is. What stays a defect is freezing the **result**: a literal computed from constants is correct for no instance, nothing keeps it honest as the family grows, and the width machinery cannot see what it caps (upstream #53).

**An envelope must be non-empty; a discrete one may be a single value.** Every bounds pair is checked in the base scenario **and in every declared instance** — one contract for all three forms — and an inverted pair (`lo > hi`) is a mis-resolution wherever it appears. A *continuous* bound additionally needs `lo < hi`, since a point is not a Box; a **discrete** one may resolve to `[a, a]`, which is `Discrete(1)` — the honest declaration of an action that exists with exactly one legal value on this instance. Stating an envelope as a derivation is what makes that ordinary rather than exotic: a derivation correct across a family will legitimately collapse on a member of it (a reserve nothing can preempt, a cap with no eligible class), and padding it to `max(1, ·)` to satisfy a checker puts a number in the file that the argument does not contain — the anti-pattern the paragraph above exists to remove. State bounds keep the strict form: a state variable carries no discrete/continuous type, and a zero-width envelope is an observation dimension that cannot vary (upstream #69).

A `decisions[].dim` above 1 makes the action a **vector of that width**, resolved per instance, and the interpreter draws and seeds it at that shape — so a sweep over the action width needs no padding and no surviving enumeration. Width 1 stays a scalar, not a one-element list.

```jsonc
"state_variables": [
  {"name": "pipe", "type": "float_vector", "length": "leadtime_max"}
],
"scenario": {"constants": [{"name": "leadtime_max", "value": 4, "axis": "size"}]}
```

Rules, each mechanically checked by `mdp_conformance model.boundary`:

- **The model speaks in quantified rules, never rendered names.** `forall s in 1..L-1: pipe[s] <- pipe[s+1]`, not `pipe1 <- pipe2`. Expressions may use comprehensions (`sum(sum(pipe[k]) for k in range(n_echelons))`); the index binds itself. This holds wherever an expression is written — including `dynamics.transitions[].updates`, whose entries are assignments, and including invariants that reference `prev.<name>`. A conservation law over a multi-dimensional state is the idiom to reach for: `close(sum(stock) + sum([sum(pipe[k]) for k in range(n)]), sum(prev.stock) + …)` — do not flatten by hand to keep a validator happy.
- **A width site holding a literal, for a quantity the model declares, is a defect.** That is exactly how a sweep maximum becomes a capacity limit nobody chose — so name the constant, and the cap becomes a design choice on the record instead of a number the model appears to state.
- **The IR holds no bare design value — the rule above is the width case of it.** Wherever a value site can hold a number an instance could legitimately need to differ on, it takes a **symbol**. A literal there is a defect for the reason just given, plus one the width case does not state and which is the one that bites: **a value with no name cannot be overridden by an instance.** It is not merely mislabelled, it is *immobile* — the second rendering can only be had by re-basing the first.
  - The line is **identity vs choice**, not literal vs symbol. A number nothing could vary stays literal: a lower bound of `0` on a non-negative quantity, an index, a stream id, the `1` in a shift, an undiscounted `discount_factor` of `1.0`, a `tolerance` of `0`. Naming those buys no degree of freedom, and a rule that named every numeral would be ceremony.
  - Anything an instance could legitimately need to differ on names a constant, so what the file pins is visible as a choice, overridable per instance, and — where the model layer declares the quantity — checkable against its declared domain. A model declaration is the clearest case, not the only one: `benchmarks[].tolerance` is a claim about a reference *implementation's* numerical error, which the model layer says nothing about and which still varies by instance.
  - Both `objective.discount_factor` and `benchmarks[].tolerance` accept a scenario-constant name, resolved per instance. β stays **frame-defining and never tuned** — this is not a licence to search it; it is frame-defining the same way `n_echelons` or `leadtime` are, and two instances at two discounts are two `cases` (own frame, incomparable scores), which is the shape the IR supports everywhere else. Read them through `MdpBlock.discount_factor(instance=)` and `MdpIR.benchmark_tolerance(name, instance=)`, which apply the `(0, 1]` and `>= 0` bounds at the moment the value resolves.
  - **What adoption costs, and what it buys.** Naming either one moves the **mdp fingerprint** once — the constant is declared inside the `mdp` block — so it is a freeze-token move with an §IR-CHANGELOG entry, exactly like any other rendering edit. The **model** fingerprint never moves: the theory is unchanged by how the rendering names its discount. What it buys is that every *later* instance is free — `structural_fingerprint` excludes the scenario node, so a β = 1 twin beside a β = 0.95 base moves no rendering identity at all.
  - `mdp_conformance`'s `schema.no_hardcode` reports a non-identity literal at either site. It **WARNs** rather than FAILs, in the shape `schema.no_enumeration` established: these sites only became expressible as symbols in the release that added the check, so every shipped domain is mid-migration by construction. It promotes to FAIL once the example set is clean.
- **The same defect one level down: a vector enumerated instead of declared.** The width rule above governs the *size*; this governs *what is being sized*, and it is not reachable by `model.boundary` at all — that check SKIPs whenever the theory layer is undeclared, which is exactly the state a hand-unrolled IR tends to be in. Three shapes, all WARNed by `mdp_conformance schema.no_enumeration`:
  - a **decorative constant** — declared, referenced nowhere, while the rendering hardcodes the same quantity independently. `adi_flex` declares `demand_window: 2` and mentions it *once in the whole file*, its own declaration, while `due_now`/`due_next` carry the pipeline by hand 11 and 12 times over. Set `demand_window` to 3 and nothing happens: no third slot, no error, no complaint. The constant is decoration, and the IR silently disagrees with itself.
  - an **enumerated family** — one vector written as siblings (`due_now`/`due_next`, `lambda_now`/`lambda_next`/`lambda_later`, `pipe1`/`pipe2`). Declare `due` with `length: "demand_window"` and quantify over it. Unrolling deletes the width site, so nothing can sweep the dimension and no width check can see it — the rule above is evaded rather than broken.
  - a **re-baked vector** — a numeric list restated per instance because its *derivation* cannot be declared (`ScenarioConstant.value` is a literal by design). `clark_scarf`'s `h_install` states its own formula in `desc` and then hardcodes the result in **ten** instances, each re-computed by hand and checked by nothing. Until a derived-constant mechanism exists (issue #45) this one is sometimes unavoidable — when it is, the `desc` must state the derivation and every instance that must re-bake it. Baking tends to *strand the inputs* too: the constants the formula consumed have nothing left to feed, so they read as decorative unless something else happens to use them (`fnv`'s `t_last` survives only because the gym observes it).

  **Why this is a hard rule and not a style note: the IR is the first artifact a user reads.** It is what the pipeline hands back after Phase A, and it is read as the statement of *their* problem. A model definition that enumerates `lambda_now, lambda_next, lambda_later` does not read as a formalization — it reads as a transcript of one particular case, and a reader who sees that stops trusting the layer that produced it. Every other guarantee in this spec is downstream of someone believing that file.
- **A width site has two legal shapes, and they are checked differently.** A width may *name the swept design value itself* — `length: "pipeline_len"`, `axis`-tagged, resolved per instance — in which case it renders exactly what each instance selects and cannot be outrun. Or it may name a *cap*: an untagged constant bounding a different quantity (`leadtime_max` over `leadtime`). **A cap must cover every designed value of the quantity it caps**; a sweep that outruns its own rendering is caught before it is frozen. The `axis` tag is what distinguishes them, so no new declaration is needed — and comparing a design value against its own overrides would fail the first shape, which is the recommended one.
- **Every declared instance sits inside the declared domains.**
- **No model-layer rationale argues from a benchmark, a solver, or tractability.** That vocabulary inside the theory layer is the tell that a design choice has been filed as model — "kept integral so the DP reference stays exact" is a *design* statement, and it belongs to the layer that chose it.
- **One vocabulary.** The model object uses the schema's own identifiers; a source's symbols are recorded in `notation` with citations and never adopted, which would create a third vocabulary nobody consumes. Verify quotes against the source rather than memory — a false citation is how a design choice launders itself into the model layer.

Where the rendered form is stricter than the theory, say so and name the layer that did it: `"narrowed": {"to": "integer lattice", "by": "selection:poisson + design:integer_actions"}`.

**Fingerprints follow the layers.** `model_fingerprint()` hashes the theory and moves only when the model's statement changes; the structural fingerprint is explicitly the *rendering* hash; `mdp_fingerprint()` remains the freeze token over the whole block. A repair that re-slots a pipeline then reads as "model unchanged, rendering re-signed" — which the single-hash instrument could not say. All three fields are optional and are omitted from the freeze token when absent, so an IR written before them hashes exactly as before; declaring them moves the token once, which is correct.

**Once, and not again when the layer grows.** Inside `model`, an unset field is an unmade statement and is pruned from both hashes — so a field added after the block shipped does not re-tokenize the IRs that already adopted it. Only absence prunes, never a falsy value: `"stochastic": false` is the claim that the source holds a quantity deterministic *by assumption*, and it must not hash as if the question had never been asked. The rule stops at the model block, because the rendering carries explicit nulls that existing tokens already hash.

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
- `seed_salt` is the domain's universal reproducibility knob (§6.3) and must be **`>= 1`** (the v2 seed grammar relies on a nonzero trailing word); `scenario_name` is the registry key. A deterministic-dynamics domain (§4.3) has **no `seed_salt` on the scenario or the state** — there is no intrinsic randomness to salt; only its callable instance source keeps one, for the per-episode instance draw.
- **A simulated scenario is a fully concrete problem instance**: together with an `episode_seed` it must pin the episode completely. Anything one might hand-craft, name, or benchmark against (a specific puzzle board, a specific demand profile) must be representable as a fixed scenario; distributions over instances live only in latent generators realized through a `{Domain}ScenarioSource` (§5.2, world latents) — and finite *sets* of instances in grids (§5.6, design families). A scenario holding an unrealized `Latent{...}` generator is a **template**: register it wrapped in a source, never simulate it directly.
- **`seed_salt` must be declared `field(default=<default_int>, repr=False)`** so it is excluded from the dataclass `repr`. It is an internal reproducibility knob, not a meaningful configuration value to a reader; keeping it out of the `repr` avoids cluttering the `print(scenario)` output that training scripts emit at startup (§8.2).
- **`scenario_name`** is the short registry key (e.g. `"simple"`, `"test1a"`); **`desc`** is a human-readable one-line description of the scenario. Keep the name short and put the explanation in `desc` rather than encoding everything into a long name. Prefer `desc` over a leading `#` comment on the instance, so the description travels with the object: drivers and eval scripts can print a self-describing header, e.g. `print(f"=== {scenario.scenario_name}: {scenario.desc} ===")`.
- Validate all structural constraints in `__post_init__`.

### 5.2 World latents — generator-owned, realized through `{Domain}ScenarioSource`

Model a world latent when the **problem itself** contains a draw nature makes
once per episode: a hidden market size, a demand-regime pick, a whole problem
instance (a puzzle board). A world latent is **world modeling** — its
distribution is part of the MDP, and changing it changes the problem. A
distribution used merely to train one policy across a family of complete
problems is *not* a latent: it is a grid (§5.5–§5.6). When in doubt, apply
the litmus tests in §5.5.

**The latent belongs to the source of randomness it parameterizes** (catalog
model, IR_LAYERING_PLAN §10): a `Latent{...}` generator class in
`{domain}_uncertainty.py` carries the latent's hyper-parameters and realizes
it in `realize(episode_seed, seed_salt)` — keyed on the **meta branch at the
generator's own `source_id`**, one stream identity per source on both
branches — returning the realized concrete generator. A concrete generator's
`realize()` returns itself, so every composition resolves uniformly.

```python
# {domain}_uncertainty.py
def meta_key(substream_id: int, episode_seed: int, seed_salt: int) -> list[int]:
    """v2 meta seed key: [substream_id, 0, episode_seed, seed_salt] (§6.3).
    For a source's latents substream_id IS the source_id; composition-scoped
    drawers (mixtures, deterministic-instance sources) use >= N_SOURCE_IDS."""
    return [substream_id, 0, episode_seed, seed_salt]


class Latent{Variant}{Source}({Source}Generator):
    """{Latent} drawn once per seed; realized as a concrete {Source} generator."""

    latent = True

    def __init__(self, <hyper-params>, source_id: int = <slot id>): ...

    def realize(self, episode_seed: int, seed_salt: int) -> {Source}Generator:
        rng = np.random.default_rng(np.random.SeedSequence(
            meta_key(self.source_id, episode_seed, seed_salt)))
        <latent> = ...
        return {Variant}{Source}(<latent>, source_id=self.source_id)

    def mean(self): ...   # family-level ENVELOPE over the latent prior —
    def max(self): ...    # what gym wrappers size spaces from (§7)

    def sample(self, state):   # unrealized: fail loudly, never draw
        raise RuntimeError("unrealized latent — resolve the source first")
```

`{domain}_scenarios.py` then only **composes**: a `{Domain}Scenario` holding
latent generators is a *template*, registered through the single generic
wrapper —

```python
# {domain}_scenarios.py
class {Domain}ScenarioSource:
    """Pure function episode_seed -> {Domain}Scenario: realizes each latent
    generator; all other attribute access delegates to the template."""

    def __init__(self, template: {Domain}Scenario):
        self._template = template

    def __call__(self, episode_seed: int) -> {Domain}Scenario:
        t = self._template
        return dataclasses.replace(
            t,
            demand=t.demand.realize(episode_seed, t.seed_salt),
            leadtime=t.leadtime.realize(episode_seed, t.seed_salt),
        )

    def __getattr__(self, item):
        return getattr(self._template, item)
```

- Instances are named `source_{name}`; `SCENARIOS` may hold either kind.
  "Fixed scenario vs per-seed source" is thereby **derived from content** —
  whether any composed generator is latent — not a per-family class split:
  swapping the demand family or its latent recipe touches only
  `{domain}_uncertainty.py` and the template line, never a sampler subclass.
- **Pure function** `episode_seed -> {Domain}Scenario`: calling twice with
  the same seed must return equal scenarios, and a call must not mutate the
  source — vectorized envs share one source instance across workers. There
  is deliberately **no lifecycle hook** (`episode_init()` or similar):
  "drawn once per episode" is enforced by who calls it (the gym's `reset()`)
  and by the seed key (no `period` level), not by a method name.
- **Vocabulary**: "episode" is gym-layer vocabulary. The world layer defines
  a seed-indexed family of concrete problems; the *gym* binds one episode to
  one seed at `reset()`. Phrase docstrings "per seed" / "per draw", never
  "per episode". (The parameter name `episode_seed` is kept for consistency
  with `SamplingContext`.)
- Latents key on the **meta branch** of the seed tree (§6.3) via
  `meta_key()` — never a raw `SeedSequence` — at the owning generator's
  `source_id`. Corollary: scenarios sharing a latent recipe share latent
  draws per seed (**common random numbers across instances**, sharpening
  paired comparisons). The only drawers with their *own* `substream_id` are
  composition-scoped — mixtures (§5.3) and a deterministic-dynamics domain's
  instance source (§4.3) — allocating at or above `N_SOURCE_IDS` so they
  never collide with a source's latent stream.
- A latent may be an **entire problem instance** (a full Sudoku board carved
  from a random solved grid), not just scalar parameters. Unbounded,
  procedurally generated instance families stay world latents even though
  the draw is arguably the experimenter's — the grid alternative requires a
  finite, enumerable set (§5.5). In the deterministic-dynamics case there is
  no generator to own the draw, so the bespoke callable instance source in
  `{domain}_scenarios.py` owns it directly (with its own `substream_id`);
  generation helpers live there as module-level functions.
- The gym wrapper checks `callable(self.scenario)` and calls
  `self.scenario(episode_seed)` in `reset()` to get the concrete scenario
  for that episode (§7).
- **`scenario_name`**: the template's name is the `SCENARIOS` key; the
  source delegates attribute access, so scripts read it without an
  isinstance check.
- **Family-level attributes — the mirroring rule, now automatic**: the gym
  and eval scripts read pre-draw attributes (`horizon`, `allow_backlog`,
  `demand.max()`, …) off the source; `__getattr__` delegation covers the
  scenario fields, and the latent generator itself exposes the read API
  (`max()`, `mean()`, `is_discrete`) as envelopes over its latent prior. No
  separate bounds object; space-building code works unchanged against a
  scenario or a source.
- **Observed vs hidden latent** — same machinery, different problems, so
  state which one you mean in `desc`: an *observed* latent (a forecast the
  agent sees) makes a contextual MDP — an observation mode may expose the
  realized value; a *hidden* latent (an unobserved market size) must be
  inferred — a realized latent field on the scenario is **not** automatically
  observed; observability is decided only by the observation modes (§7). The
  deployable policy (§12) must never require a hidden latent as input, and a
  baseline that reads one is *clairvoyant*: it solves an information
  relaxation, so its value is `≽` the true optimum (§9.3) and it is **not
  attainable by any deployable policy** — the `relaxed` role of §9.9. Label it
  as such in eval output, with the direction taken from `objective.sense`.
  Never call it "an upper bound": the same construction bounds from above in a
  maximize domain and from below in a minimize one, so the word names the
  domain rather than the benchmark — and every inventory domain here is a
  minimize domain, which is where following the old wording literally wrote an
  inverted label into the file §5.2 says the label must appear.

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

    # (weight, component); components may be fixed scenarios OR sources,
    # including other mixtures
    components: list[tuple[float, ScenarioSource]]

    # meta-level substream id (§6.3 branch 0); composition-scoped, so it
    # allocates at or above N_SOURCE_IDS — never in the sources' latent range
    substream_id: int = N_SOURCE_IDS
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
  with the mixture's own out-of-range substream id this gives *standalone
  equivalence*: episode seed `e` through a mixture branch equals seed `e` run
  on that component directly — any branch is reproducible in isolation.
- **Mixtures of mixtures are allowed** — components are `ScenarioSource`s and
  a mixture is one. Nesting adds structure, not expressive power (weights
  multiply through to a flat mixture); its value is reuse — embed a named
  world object as one branch without hand-flattening weight products.
- **Components may differ in distribution family** (cross-family mixtures,
  IR_LAYERING_PLAN §10f): the generator-owned-latent design makes this free
  on the Python side — each component's generators realize on their own slot
  streams; the mixture only picks. The IR mirror: mixture components may
  carry different candidate *selections*; the loader resolves each divergent
  component separately (`MdpIR.mixture_resolutions`) and the interpreter
  swaps in its sources per episode, preserving standalone equivalence
  bit-for-bit (`inv_single`'s `mix_demand` — discrete vs poisson demand —
  is the shipped reference).

### 5.4 Instances and registry

```python
scenario_simple = {Domain}Scenario(scenario_name="simple", desc="...", ...)
scenario_{name} = {Domain}Scenario(scenario_name="{name}", desc="...", ...)

source_hidden_mu = {Domain}ScenarioSource({Domain}Scenario(
    scenario_name="hidden-mu", desc="...",
    demand=Latent{Variant}Demand(<hyper-params>), ...))

SCENARIOS: dict[str, ScenarioSource] = {
    "simple":    scenario_simple,
    "{name}":    scenario_{name},
    "hidden-mu": source_hidden_mu,   # latent-bearing source entries allowed
}
```

- Fixed scenario instances are named `scenario_{name}`; source instances are named `source_{name}`.
- `SCENARIOS` values are `ScenarioSource`s: `{Domain}Scenario`, `{Domain}ScenarioSource`, or `{Domain}MixtureSampler` — **world layer only; a grid must never appear in `SCENARIOS`** (it is not callable and deliberately fails the gym's sampler check).
- Next to the registry, assert every callable entry resolves seed 0 to a fully concrete scenario (a `_check_registry()` helper); the conformance harness checks purity and source-id distinctness externally.
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

**Choosing axes.** Not every scenario constant may become one. For one
policy to serve every cell, three independent conditions must hold: **(a)
shape** — observation and action spaces identical across cells; **(b) HP
validity** — every scalar hyperparameter simultaneously right for every cell
(they are set once, at construction, for the whole run); **(c)
commensurability** — cell scores on one scale, so the aggregate the sampler
optimizes means something. Classify a candidate axis by which it breaks
(`mdp_conformance` reads the tiers off the IR's own declarations and warns):

- **Tier 1 — structural: do not, unless you pay for it.** Anything that
  changes the **action space** (a constant named in decision or action-mode
  bounds). One policy cannot emit two action spaces. This is a *plumbing*
  limit, not a mathematical one — an equivariant per-entity scorer is
  already count-agnostic — so it is liftable: pad the action space to the
  family maximum and mask the surplus (§7.1). Budget for it honestly (wasted
  capacity, permanently masked logits, a normalizer that no longer sees a
  fixed slot count), and declare it if you do it.
- **Tier 2 — costly but survivable.** Two distinct mechanisms, and the
  interventions are not interchangeable. *Changes the observation dimension*
  (a constant named in a state variable's `length` or bounds — e.g. a lead
  time setting pipeline length): pad the affected vectors to the family
  maximum; benign once declared. *Changes the horizon* (the `horizon.T`
  constant): changes no space's shape, so it is easy to add and hard to get
  right — it breaks (b) and (c) instead. Horizon-dependent scalar HPs
  (`gae_lambda`, rollout composition) silently change *meaning* across
  cells; the measured optima in two campaigns bracket opposite directions
  (§8.6), so re-derive per cell, never transfer.
- **Tier 3 — free.** Constants entering only the reward computation or the
  uncertainty parameters (costs, prices, distribution parameters). These
  change what the optimal policy *is* without changing what a policy *is* —
  exactly what a generality target should vary.
- **Rider 1 — is the axis observable?** Cuts across all tiers. If the axis
  value is in the observation, the policy conditions on it directly. If not,
  the policy must *infer which cell it is in* from within-episode experience
  — the generalist becomes an adaptive policy, and the sufficient-statistic
  question (`requires_memory`) re-opens. A tier-3 axis the agent can neither
  observe nor identify from its own history is the quietly bad case.
- **Rider 2 — does the axis change reward scale?** Then the sampler's
  uniform-over-cells is *not* uniform-over-reward: a cell at the high end
  contributes proportionally more to every aggregate, and that weighting
  silently becomes the training objective — and the checkpoint-selection
  criterion. The horizon does this by construction; so does any cost axis
  spanning orders of magnitude. Report **per cell** (§9.6) and state the
  weighting; never quote a single aggregate as the generalist's score. This
  binds *any* comparison over the grid, not only the generalist's own score —
  §9.7 carries the reporting rules, including the one failure this rider does
  not cover (the SE's replication unit, which bites a scale-homogeneous grid
  just as hard).

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
  decision path. A domain whose IR keys a stage on a decision (`key_exprs` on
  a period/event stage — K parallel exogenous streams, one read per period,
  e.g. per-arm bandit payouts) still conforms: the extra leading key words
  select *which* pre-determined stream is read, while every stream's contents
  remain a function of `(episode_seed, seed_salt)` alone.

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

**Latent in the reward vs. the metric.** A hindsight/clairvoyant benchmark (`optimal_revenue`, `profit_max`) reads the realized latent, so referencing hidden state in an *objective* is legitimate — but a **per-step reward the policy consumes** must not reveal hidden state the agent is meant to *infer*: a memory policy (fed `r_{t-1}`) can back it out and the censoring is gone. Latent is safe in exactly three places — an eval-only **metric** (never touches the policy), a **terminal** reward (no later decision to contaminate), or an **action-independent normalizer** (a per-episode clairvoyant gap, e.g. `100 * revenue / revenue_opt`, as in `retailer`'s `gap` reward mode). Keep per-step, decision-relevant rewards on observable quantities — forgone margin on *observed* sales, never a charge on a latent quantity (e.g. unobserved lost sales) the agent is supposed to be uncertain about.

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
    - The core simulator is the reference implementation of the schema's
      dynamics and reward; this wrapper adds none of its own.
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
- **What the gym renders, the IR declares.** A mode's `gym.observation_modes[].features` list is **exhaustive, in render order** — every component the vector carries is named there — and the declared menu must *be* the menu: a mode the IR names is constructible under that name. Neither half is gated by any check, so both are review items at the Phase-A gate and whenever a mode is added or a feature is prepended.
  **The rule is about content, not width.** A mode may freely *re-encode* what it declares: `game2048`'s `vec`, `grid` and `onehot` modes each declare exactly `{"ref": "board"}` and render it as flat cells, as `(1, n, n)`, and as `(n_cells+1, n, n)` one-hot planes — one declared feature, three encodings, and only the last has a width matching nothing. What must hold is **provenance**: every component of the vector derives from a quantity the list names. This is also why a width-equality check is the wrong instrument on its own — an expanding encoding and an undeclared prepended scalar are the same symptom to it, and the first is correct while the second is the defect.
  The component that goes missing is **the time feature**, in the domains that have one. Where the horizon is the problem's own boundary, the policy is a function of where in it the episode stands, so the gym prepends a time feature — and that is the component least likely to be written down, because it enters the gym as a formatting decision rather than as a state variable someone chose to expose. Upstream #71 surveyed three live domains and found two rendering a time feature their IR never declares; a reader of those IRs cannot tell the observation carries time at all, let alone which convention. The asymmetry is the argument: a latent variable's *absence* from every mode is validated, while a rendered feature's absence from the list it belongs to is not.
- **Prefer `time_to_go = T - t` — absolute, counting down — as the finite-horizon time feature**, declared like any other derived feature (`{"derived": "time_to_go", "expr": "T - period"}`; `T` is the horizon symbol, so the expression resolves per instance). This is a **recommendation with a stated exception**, not a rule.
  *Structural reason.* Finite-horizon value functions recurse backward from the boundary at `t = T`, and the consequence that matters is stationarity in the remaining-time coordinate: `π*(x, T − t) → π*_inf(x)` as `T − t` grows. The same feature value therefore carries the same decision whether `T` is 30 or 100, so a backward-encoded policy transfers across horizons where a forward-encoded one relearns the map. A **relative** encoding destroys exactly this: "3 periods left" is `0.10` at `T = 30` and `0.03` at `T = 100`, so one decision arrives as two different inputs. Measured on `inv_single` by fitting an exact DP's actions with PPO's own network and observation — which isolates representability from learning — adding plain `time_to_go` moved accuracy over all `(x, t)` from 95.9% to 99.7%, while two hand-crafted tail encodings (`1/(1 + time_to_go)`, a last-3-periods one-hot) both scored *worse than nothing*.
  *The exception, in the same breath.* Absolute backward is right when the terminal effect has a **fixed depth in periods** (`inv_single`'s optimal-policy ramp is confined to the last two, independent of horizon). Where the structure instead **scales with the horizon** — a markdown season, a budget depleting proportionally, anything genuinely phase-like — **relative forward (`t / T`) is correct**, and the recommendation must not push against it. What is not optional either way is that the convention be *visible* in the declared feature's `expr`.
  *Which horizon this is about — and the IR already says.* **"Finite horizon" is not the test**, because every IR is finite-horizon by construction: `Horizon` carries no `kind` discriminator in v1, so a stationary problem still declares a `T`. The discriminator is **`gym.termination.horizon_end`**.
  - **`terminated`** — the horizon is the *problem's boundary*. The value function recurses from `t = T`, the same board or inventory means something different near the end, and **time is state**: render a time feature, declare it, and prefer the counting-down form above. `inv_single` is this case.
  - **`truncated`** — the horizon is a **cap, not a boundary**. The problem is stationary within it and the cut-off exists to bound wall-clock or to stop a degenerate episode. Here the policy **must not see the clock**: a countdown the dynamics never read is a nonstationary input carrying no signal, it spends capacity, and it teaches an end-of-episode artifact to a value function that has no end. What handles the cut-off is **bootstrapping at truncation** — the reason Gymnasium separates the two flags at all — not an observation feature. `game2048` is this case: `T_cap = 1000` is a safety net (its own constant `desc` records that games reach `game_over` long before it under masked actions, and that under free actions the cap is what stops a policy choosing illegal slides forever), `horizon_end` is `truncated`, and all three of its modes render the board alone. That is **compliant, and "fixing" it would be the defect** — the declaration rule asks only that what *is* rendered be declared, which board-only satisfies.

  A **budget** is the case that looks like a clock and is not one: moves, fuel or attempts remaining that the *dynamics or reward actually read* are state, belong in `mdp.state_variables`, and are observed as themselves. The test is whether anything in the model consumes it — a quantity only the episode loop reads is a cap; a quantity a transition reads is state.
  *Two things this does not buy.* It is **not retroactive**: changing a live domain's time convention changes every trained arm's input and orphans its whole RL table, so it is a default for new domains, not a migration. And it does **not** cure a finite-horizon terminal blind spot — `inv_single` spent a campaign day on one and eliminated representation as the cause before re-encoding anything (PPO's own network fit the DP's terminal actions 100% from the raw period index). What it buys is representational convenience and cross-horizon transfer.
- **An action bound is derived from the MDP definition and its dynamics, never from existing runs.** The tempting justification is the one the author can see — run the solver, take the largest action an optimal policy places, add headroom — and it produces a number that looks well-founded. It is not a bound: it is a **measurement of the instances already solved**, and its failure mode is silent. A cap that binds does not raise; it truncates the action set and reports a worse policy, so the first instance with a longer horizon, a heavier rate or a larger lead time inherits a cap justified against a study that no longer exists. §5.2's coverage rule cannot catch it either — that fires only for a cap *named over a declared design value*, which an action scale generally is not. So an action bound's recorded rationale must cite model quantities (a horizon, a rate, a lead time, a capacity the model declares) and must not cite the behaviour of any solved policy, tuned baseline or completed training run. `"≥2× headroom over the largest optimal order we observed"` is the anti-pattern by name, because it is what an author reaches for first and it reads as diligence. One line settles it: *would this justification still hold for an instance nobody has run?* `mdp_conformance schema.bound_rationale` WARNs on the vocabulary of measurement; it is evidence, not proof, and the question above is the actual test.
  Magnitude and shape are separate questions: §7.2's transform table is about the *shape* of the action set, §4.1 about how wide it should be, and the two space kinds want different answers (an observation envelope is generous, an action bound tight).
- Supports `logger_filename` for per-episode and per-step log output.
- **Episode-seed provenance.** `reset()` derives the episode seed like this — and never from the global `np.random`:

  ```python
  def reset(self, *, seed=None, options=None):
      super().reset(seed=seed, options=options)   # maintains self.np_random
      if seed is not None:
          self._episode_seed = seed               # eval/benchmark contract: explicit seed pins the instance
      else:
          self._episode_seed = int(self.np_random.integers(0, 2_147_483_647))
  ```

  `super().reset(seed=...)` seeds gymnasium's per-env `self.np_random` once (SB3 passes `training_seed + rank` per env at the first reset); unseeded resets then draw the episode-seed stream from it — per-env independent, reproducible, and safe under any vec-env. Drawing from **global** `np.random` is forbidden: under `SubprocVecEnv` (fork) every worker inherits identical global state and the "parallel" envs replay the *same* episode-seed sequence from episode 2 on; under `DummyVecEnv` the stream is reset-order-fragile. Everything below `episode_seed` is already counter-keyed (§6.3) and unaffected.

  The **`else` branch is not optional**, and `behavior.gym_reseed` (`mdp_conformance`) is what enforces it: three unseeded resets under a fixed action stream must not replay one episode, and the same explicit seed must reproduce one. Omitting it leaves `_episode_seed` untouched on the auto-reset path SB3 takes, so every training episode after the first replays a single exogenous path — silent in both directions that normally catch things, since training only degrades and §9 evaluation seeds every episode explicitly. The leaderboard stays a valid measurement of the artifact while the training distribution is wrong.

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
    # train scripts only (§8.6): check the derivation against what this run
    # actually resolved to, before any artifact exists
    assert_l1_current(args, derived=_L1_DERIVED, basis=_L1_BASIS,
                      episode_len=scenario.horizon)
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

**The CLI tier contract.** Every argument of a train or eval script belongs to
exactly one of three tiers. The tier decides who owns the name, who may change
the value, and what enforces it. It does **not** decide who may *search* the
value: `norm_obs` is tier 1 — every domain spells its normalization flag the
same way — and is also a knob `mdp_tuning` opens at its `breadth` tier (§8.6).
A tier-1 name inside a tuning space is the ordinary case, not a conflict.

**Tier 1 — the invariant surface.** Arguments that mean the same thing in every
domain and under every algorithm family. Tooling and humans join across domains
on these, so their *names* are the contract:

| group | train | eval |
|---|---|---|
| problem & rendering | `scenario_name` `observation_mode` `action_mode` `reward_mode` | `scenario_name` `observation_mode` `action_mode` `reward_mode` |
| process & identity | `total_timesteps` `outdir` `seed` `tag` | `model_path` `vecnorm_path` `outfile` |
| protocol | — | `n_seeds` `first_seed` |
| env / wrapper stack | `n_envs` `norm_obs` `norm_reward` `vecnorm_clip_obs` | — |
| artifacts & diagnostics | `gym_log` `checkpoint_every_frac` | — |

`mdp_conformance`'s `scripts.cli_contract` check reads the parsers that exist
and compares their dests against this table. A parser assembled by a helper in
the domain folder is a parser that exists: factoring the common eval surface
into one `{domain}_benchmark_common.build_arg_parser` is the *right* shape, and
the check resolves the builders a script imports from its own folder — and the
calls those builders make — to find it. The table is a contract over the names
the surface offers, not over which file spells them. The reach is per-import,
not per-folder: a helper's flags count for the scripts that import it, so the
`__main__` demo CLI in a `{domain}_policy.py` never credits an evaluator that
imports the policy class from it.
Two groups are owed only where they apply, and the check reads that from the
domain rather than from prose: a **rendering** mode is owed when the env's
`__init__` accepts it — including where only one value is legal today, since
the flag is the join point and a second rendering must not change the CLI's
shape — and the **VecNormalize** names are owed when the script reaches for the
wrapper in code (§8.3): `norm_obs` `norm_reward` `vecnorm_clip_obs` where a
train script *builds* one, `vecnorm_path` where an eval script *loads* the
saved stats (§9.5). Naming it in a comment owes nothing — a domain that
implements its own normalization and mentions `VecNormalize` only to say what
it is an analog of would otherwise ship three knobs that control nothing, and
an args log recording a normalization the run never had is a worse artifact
than the warning it silences. Everything else is owed unconditionally. A
script that does not exist yet is not a violation — a
domain still in Phase A owes no `{domain}_ppo_eval.py`, and the check reads the
parsers it finds and skips the rest, so a Phase-B-stage-0 gate never reports a
stage the campaign has not claimed.

**Tier 2 — algorithm knobs**, owned by the family's tuning space and
deliberately **not** enumerated here. `learning_rate`, `n_steps`, `batch_size`,
`n_epochs`, `gae_lambda`, `ent_coef`, `vf_coef`, `clip_init`, `max_grad_norm`,
`gamma`, `net_arch`, `normalize_advantage` and any extractor knobs are
whatever `mdp_tuning`'s space
for that family declares; `python -m mdp_tuning <domain-dir> --show-space` is
the machine-readable oracle. §8.6 already obliges a new family to bring its own
L0 one-liner and its own L1 derivation table, so its knobs are family-property
by construction — freezing PPO's set in prose would create a second source of
truth that goes stale on the first off-policy family. What this section asks
for is enforcement, not enumeration: `mdp_tuning` **fails the study at launch**
when an in-tier knob has no matching dest and `--knobs` was passed explicitly
(or `--strict-knobs` is set). Silence is the failure mode that earns the
fatality — a study accepts `--knobs breadth`, finds one dest missing, warns
once, and then searches seven knobs while its own banner says eight, with
nothing downstream of that log able to tell.

**Tier 3 — domain-specific.** `frame_stack`, `cnn_arch`, `mask`, `edge_dim`,
`anchor_coef`, and anything else one problem needs. Named locally, checked by
nothing, and that is correct.

- Always `_build_arg_parser()` (private) + `parse_args()` (public) + `main()` + `if __name__ == "__main__": main()`.
- `_build_arg_parser()` returns the parser; `parse_args()` calls `.parse_args()` on it and returns the namespace. This split lets other scripts reuse the parser (e.g. to add extra arguments) without re-implementing it.
- Scenario lookup via `SCENARIOS[args.scenario_name]` (not `globals()`).
- **Generalist training** targets a grid: look up `GRIDS[args.grid_name]` and pass `scenario=grid.as_sampler()` (optionally with weights). The env never sees the grid itself (§5.6); the derived sampler carries `scenario_name` from `grid_name` for run identity.
- `print(scenario)` at the top of `main()` to confirm configuration.
- Env constructed with `scenario=scenario` (keyword argument, never positional).
- RLlib `env_config` dict must use `"scenario"` as the key (RLlib calls `Env(**config)`).
- **Do not hard-code `gamma`** in PPO kwargs — expose it as a CLI argument instead. Its default is the IR's `objective.discount_factor` (β): training γ = β is the faithful setting; γ < β only as a logged escalation move, γ > β never (§8.6).
- **Expose `--net_arch`** (`nargs="+", type=int`, default `64 64` → `policy_kwargs`): network size is a core tuning knob, and `mdp_tuning` can only reach dests the script exposes.
- **Expose `--n-envs`** (structural, never tuned; default from the §8.6 derivation). All env access goes through the VecEnv API — **never `venv.envs`** — with zero-arg env factories and rank-suffixed Monitor filenames, so `DummyVecEnv`/`SubprocVecEnv` stay a one-argument swap. `DummyVecEnv` is the default; `SubprocVecEnv` only when a *measured* env-step cost (≳1 ms) justifies the IPC overhead. Eval scripts stay single-env (§9.5) regardless.
- **Expose `--tag <ledger-address>`** (free-form, default empty): the
  escalation-log entry this run was launched under (`A8a`, `E14`). It costs
  nothing — the args log records every CLI arg verbatim, so the tag lands in the
  run dir's immutable `{scenario}_{algo}_args.txt` and a reader standing in the
  run dir can find its birth entry without grep.
- **Expose `--checkpoint-every-frac`** (default `0.05`, `0` disables): §9.7's
  screen wants ~20 checkpoints, and a **fraction of the budget** is what keeps
  that true when the budget moves — an absolute step count stops meaning "~20
  checkpoints" the moment `total_timesteps` changes. Convert at the call site:
  `save_freq = max(1, int(frac * total_timesteps / n_envs))`, since
  `CheckpointCallback` counts VecEnv steps, not environment steps.
- **Schedule pairs are one degree of freedom, and both halves must exist:**
  `lr_final = lr_init/10`, `clip_final = clip_init/4`. The two flags are
  separate dests, but defaults obey the ratios and `mdp_tuning` derives the
  final from the tuned init, so a schedule can never invert. **Exposing an init
  without its final is the failure mode to avoid**: the derivation is guarded on
  the final's dest existing, so it silently no-ops, the script's
  `build_schedule(start, end)` collapses to a constant, and the run trains on a
  flat schedule while every artifact says it was tuned. Nothing in the run
  distinguishes that from a schedule that was derived — which is why
  `mdp_conformance`'s `scripts.schedule_pairs` reports a half-exposed pair.
  A script exposing *neither* half is a tier-2 completeness question, not this
  one — `mdp_tuning --show-space` is where that surfaces (§8.6).
- For high instance-variance domains pass `stats_window_size=500` (or more) to the model: the default 100-episode rolling `ep_rew_mean` swings even under a static policy.
- **Pin BLAS/torch threads when launching training** (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`): the policies in these domains are tiny, so torch's default all-cores threading adds sync overhead rather than speed, and on a shared machine it oversubscribes cores already used by other jobs (measured on a small-board CNN domain: 9 min → 11 s for 2048 steps on a box concurrently running an 8-core workload; expect a smaller but still real gain on an idle box). The `mdp_tuning` harness sets this for its subprocesses automatically.

### 8.3 SB3 env-wrapper stack (VecNormalize)

For SB3 single-agent PPO scripts, wrap the env in this order:

```python
def make_env(rank: int):
    def _make():
        env = {Domain}Env(scenario=scenario, ..., logger_filename=...)
        env = FrameStackObservation(env, stack_size=..., padding_type="zero")  # only if the domain uses it
        return Monitor(env, filename=str(outdir / f"monitor_{rank}"))
    return _make

env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
env = VecNormalize(env, norm_obs=args.norm_obs, norm_reward=args.norm_reward,
                   clip_obs=args.vecnorm_clip_obs, gamma=args.gamma)
```

- **Pass `gamma=args.gamma` to VecNormalize.** Its reward normalization divides by the std of a running *discounted* return with its **own** gamma defaulting to 0.99 — left unset it normalizes against a different discount than training.
- **n_envs is structural** (§8.2): the rollout buffer is `n_steps × n_envs`; more envs decorrelate the buffer (more instances per update, stabler `obs_rms`), which matters most for long-episode domains. Rank-suffixed Monitor files keep per-env logs from interleaving.

- **Order matters.** `Monitor` must sit **inside** `VecNormalize` so `rollout/ep_rew_mean` is logged on the **raw** reward scale — this keeps the metric comparable across runs regardless of reward normalization. (Only `train/value_loss`, `train/explained_variance`, etc. are on the normalized scale.)
- **Expose three CLI flags:** `--vecnorm_clip_obs` (default `10.0`), `--no_norm_obs` (`action="store_false", dest="norm_obs"`) and `--no_norm_reward` (`action="store_false", dest="norm_reward", default=True`). Encode each in the run name when non-default. **`norm_obs` is a flag, never a literal**: §8.6's L1 table derives it per domain and L0 requires it off, so a skeleton that hardcodes `norm_obs=True` can express neither — which is exactly how a script written to this section ends up unable to run its own domain's derived config. **Where the derived value is computed at runtime** rather than written as the argparse default — `mab` reads the observation mode inside `parse_args` and leaves `default=None` — declare the flag as one `argparse.BooleanOptionalAction`, not a `store_true`/`store_false` pair: a parser is read per *dest*, so two actions sharing one leave whichever was declared first unreachable to anything driving the CLI, and `mdp_tuning` searches this knob (§8.6). The runtime derivation is still legitimate; what it costs is the warm start, which reports trial 0 as *not* the L1 centre for that knob because the script states no single default to enqueue.
- **Save the stats:** `env.save(outdir / "vecnormalize.pkl")` after `model.learn(...)`, then `env.close()`.
- **What each normalization is for:**
  - *Obs norm* conditions the network **inputs** (features often span several orders of magnitude — e.g. inventory vs. a price index). It is **part of the policy's input contract**: the saved `obs_rms` **must** be reused at eval/deploy time (§9.5), otherwise the policy sees inputs it was never trained on.
  - *Reward norm* scales the reward by the running std of the discounted return (no centering). It is **training-only** — never needed for inference/deployment. Because SB3 PPO's `normalize_advantage=True` already makes the policy-gradient step scale-invariant, reward norm mostly affects the **critic** (value-loss scale / `vf_coef` balance), not the policy behavior. That premise is a §8.6 derivation row with a flag behind it rather than an SB3 default inherited in silence — it carries a large lever arm, so it is checkable: `adi_flex` A14 measured a **+108-point** main effect for ON and roughly 6× worse seed dispersion for OFF.
- **Not for non-SB3 frameworks.** VecNormalize is SB3-specific; RLlib/PettingZoo/DreamerV3 scripts use their own normalization (e.g. RLlib's `rllib_vecnormalize`).

**When obs norm helps vs. when it hurts** — decide per domain, don't apply blindly:

- **Helps** when obs dimensions span very different magnitudes (e.g. inventory ~1e3 alongside a price index ~1) *and* the obs distribution is roughly **stationary** across training. This is the common case and the main reason to enable it.
- **Little benefit** when obs are already **homogeneous and bounded** on a common small scale (e.g. one-hot planes, or log2-scaled tiles on a game board) — there is no cross-feature imbalance to fix; a static encoding is enough.
- **Risk** when the obs distribution is **non-stationary** — i.e. what the agent observes shifts as it improves (e.g. a tile-merging game whose reachable tile values keep growing). The running `obs_rms` never stabilizes and the frozen eval stats match no single stage ("stats-mismatch under distribution shift"). Domains with this property may deliberately omit VecNormalize.

Reward norm is training-only and largely redundant with PPO's `normalize_advantage`, so the decision above is really about **obs** norm. That is also why the two sit in different tuning scopes: `norm_obs` is opened at `breadth`, `norm_reward` only in `all`.

**The list above is a prior, not a verdict.** Every signal in it — "span very
different magnitudes", "roughly stationary", "shifts as it improves" — is a
claim about the state distribution a **trained** policy visits, and the only
magnitudes available when the derivation is made are the declared observation
bounds, which §4.2 says to keep generous ("Looseness is free"). So the call is
made from information that cannot settle it, and both readings have now been
measured wrong:

- `mab` (#E8/#E10) read climbing counts and unbounded payout totals as the
  accumulator case → derived **off**; the isolation probe measured
  927.63 ± 6.88 for ON vs 855.73 ± 7.18 for OFF (**+71.9**, z≈7.2).
- `adi_flex` (F36–F38) read the declared envelopes as heterogeneous → derived
  **on**; the true feature sds on the optimal-policy state distribution were
  homogeneous (8–11 against an accumulated VecNormalize divisor of ~216), and a
  12-run factorial found a **74.6-point** observation-encoding gap that exists
  only at `norm_obs=on` and is statistically absent without it.

Derive it, log the rationale, and treat an L2 study that opens the knob as the
thing that settles it (§8.6).

### 8.4 Output directory structure and run name encoding

Training outputs go to `results/{scenario_name}/{run_name}/` relative to the domain directory:

```
results/
  {scenario_name}/                   # e.g. simple/, cost-sweep/
    {run_name}/                      # e.g. PPO_20260630_224416_default/
      {scenario_name}_{algo}.zip     # saved SB3 model
      vecnormalize.pkl               # VecNormalize running stats (if used)
      {scenario_name}_{algo}_args.txt  # all CLI args at training time
      train.log                      # captured stdout/stderr of the training run
      eval.log                       # captured stdout/stderr of the eval run
      train_log_episode.log          # per-episode log   (only with --gym-log 1, §11)
      train_log_step.log             # per-step log      (only with --gym-log 1, §11)
      monitor.monitor.csv            # SB3 Monitor CSV
      {ALGO}_1/                      # TensorBoard event files
      ppo_eval_{eval_scenario}.tsv   # eval output (eval scenario encoded in filename)
      checkpoints/                   # periodic checkpoints (§8.6: every 5% of budget)
      probe/                         # §14.1 instrument outputs for THIS model — raw
                                     #   measurements: action surfaces, sweeps, phase curves
      interpret/                     # §14 readback of THIS model — conclusions: fitted
                                     #   structural form + parameters, agreement stats, verdict
    benchmark/
      {method}/                      # one dir per benchmark method (dp, lp, ...)
        {scenario_name}.txt          # precomputed solution table for this scenario, if any
      benchmark_{name}_eval_{scenario_name}.tsv  # benchmark eval output ({name} = method or policy variant)
      benchmark_{name}.log           # captured stdout/stderr of that benchmark eval
  tuning/                            # `mdp_tuning` studies — domain-level, NOT scenario-nested
    optuna.db                        # sqlite store; holds every study for this domain
    {study_name}/                    # e.g. dynamic_pricing_simple_ppo (domain + scenario + algo)
      trial_{NNNN}/                  # passed to the train script as --outdir
        train.log  eval.log  eval.tsv
        {scenario_name}/{run_name}/  # the tree above, re-applied by the train script
  insights/                          # across-run comparisons — domain-level, like tuning/:
    {question-slug}/                 #   rule-vs-benchmark contrasts, multi-seed syntheses,
                                     #   generalist-vs-specialist studies (conventions inside
                                     #   are still accumulating; the home is fixed)
```

**Model-specific artifacts live with the model.** A probe measures *one*
trained model and a readback concludes about *one* trained model, so both
land inside that model's run dir — the same logic that puts `eval.log` there.
The write direction is the boundary: a probe script *writes* `probe/`
(instrument output, data, regenerable); an interpret script *reads* `probe/`
and the eval TSVs and *writes* `interpret/` (analysis: the fitted rule, its
constants, the verdict). Both anchor to the run dir of the model they were
pointed at (`model_path.parent / "probe"`) — they already take
`--model-path`, and §8.4's derive-from-input-path rule makes that the
anchored form, so no hard-coded run lists. Anything comparing *across* runs
or policies — a fitted rule scored against a benchmark, a cross-seed
synthesis — is not about one model and goes to `insights/`.

**Anchor every output path to the script, never to the CWD.** Any script that
writes into `results/` resolves it as
`Path(__file__).resolve().parent / args.outdir / ...` — train, eval, and
benchmark alike. A bare `Path(args.outdir)` is CWD-relative, so the same
command run from the repo root silently creates a second `results/` tree
there; the correctness of the layout must not depend on the caller's `cd`.
Scripts that derive their destination from an input path instead (e.g. an eval
that writes next to `--model-path`) are already anchored and need no `outdir`.

**No level between `{scenario_name}/` and `{run_name}/`.** The tree above is
exact: run directories sit directly under the scenario, alongside
`benchmark/`. Do not interpose an algorithm-family directory (`RL/`, `PPO/`) —
the algorithm already prefixes the run name.

**Console output belongs with the run it describes.** Captured stdout/stderr
is an artifact, not scratch: a training run goes to `{run_dir}/train.log`, an
eval to `{run_dir}/eval.log`, and a benchmark eval to
`{scenario_name}/benchmark/benchmark_{name}.log` — never to the domain folder
or the CWD. `mdp_tuning` already does this for trials
(`trial_{NNNN}/train.log`); manual and agent-driven launches follow the same
rule. Because `{run_name}` carries a timestamp computed inside the train
script, the launcher cannot name `{run_dir}` in advance — so the train script
tees its own stdout/stderr into `{run_dir}/train.log` as soon as it has
resolved the path, and the launcher's own redirect target stops mattering.

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
- **Only when it deviates from the derivation** (hyperparameters — tuning
  knobs): `lr`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `ent_coef`,
  `total_timesteps`, etc. The baseline is `_L1_DERIVED`, **not** the parser
  default — see "the hyperparameter tier diffs against the derivation" below.
- **Always skipped**: `outdir`, `scenario_name`, `progress_bar`, `gym_log`
  (diagnostic toggles are not design axes — §11).

```python
_SHORT_KEYS: dict[str, str] = {
    "total_timesteps": "steps",
    ...
}
_SKIP_KEYS = {"outdir", "scenario_name", "progress_bar", "gym_log",
              "observation_mode", "action_mode", "reward_mode"}

# The §8.6 L1 derivation, as data: what the table comment says was derived.
_L1_DERIVED: dict[str, object] = {"gamma": 0.99, "gae_lambda": 0.95, ...}

def build_run_name(args: argparse.Namespace) -> str:
    defaults = vars(_build_arg_parser().parse_args([]))
    defaults.update(_L1_DERIVED)          # the derivation wins where it speaks
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

**The hyperparameter tier diffs against the derivation, not against the
defaults.** Two statements are load-bearing and only jointly true until a
campaign does what campaigns exist to do — adopt a discovered value. This
section encodes a knob when it differs from the script's default; §8.6 says
those defaults *are* the L1 centre. Promote by editing a default and the knob
stops differing from it, so every later run name drops the tag — and run dirs
are immutable, so the archive then holds two textually identical names meaning
different configurations, which is exactly the property the ledger's join key
relies on. Refusing to promote is the other horn, and a campaign has taken it:
carrying a discovered value as an explicit flag forever because the naming rule
made adoption unsafe. Diffing against `_L1_DERIVED` dissolves the dilemma —
adopt the value as a default for convenience, and the tag survives because the
*derivation* did not move.

Three consequences, none of which the default-relative rule has:

- **A deviation is visible forever**, whatever the defaults later become. The
  quieter half of the same gap closes with it: a knob sitting at a default that
  is wrong for the current scale appears in no run name, no leaderboard row and
  no ledger citation, recoverable only by reading a train log by hand.
- **The level stops being authored and becomes derivable.** An empty
  hyperparameter tier means the run sat at the L1 centre; any tag means the hp
  layer moved, and which tags appear says how far. An authored `L1_…` token is
  a claim nothing checks — one campaign carries it in ~90 run dirs of an
  L2(gym) arm.
- **Re-deriving is distinguishable from tuning.** §8.6 already mandates
  re-derivation when "the instance, scale, or measured T̄ moves". That is an
  edit to `_L1_DERIVED` with a logged basis — a new L1 generation, whose
  results are not comparable to the old one's — and it is mechanically distinct
  from an L2 knob search, which is a flag on a command line.

Scope and two carve-outs. `_L1_DERIVED` holds **what the §8.6 L1 table
derived** and nothing else: a knob the table does not derive (`vf_coef`,
`max_grad_norm`) has no derived value to diff against and keeps diffing its
default, so adding one means claiming a derivation, with a basis. And where the
derived value is computed at **runtime** rather than written down — mab reads
the observation mode inside `parse_args` — the dict omits that key and
`build_run_name` supplies the computed centre for the run in hand.

`mdp_tuning` reads the same dict: the warm start enqueues the derivation, so
"trial 0 = the L1 centre" survives a promotion too; **every** trial's command is
seeded from it, so a knob the study does not search runs at its derived value
rather than at whatever the default has since become; and the launch banner
names any knob where the two differ, a null default included (§8.6).

**File naming rules**:
- Model: `{scenario_name}_{algo}.zip` (e.g. `simple_ppo.zip`)
- Args log: `{scenario_name}_{algo}_args.txt`
- Eval output: `ppo_eval_{eval_scenario}.tsv` — the eval scenario is encoded in the filename so that evaluating the same model on different scenarios produces non-overwriting files (see §9.6)

**The args log carries a minimum provenance set**, beside whatever flags the domain exposes. Keys are literal, so a checker reads them without domain knowledge:

| key | value | why |
|---|---|---|
| `algo_class` | the **resolved class actually constructed** — `PPO`, `MaskablePPO`, `RecurrentPPO` — not the flag that selected it | a flag like `mask: True` says the class only to someone who knows that domain; the class itself is what a generic check can compare against `rl.algos` |
| `sb3_version` | `stable_baselines3.__version__` | already asked for by §8.6; stated here so the whole set lives in one place |
| `ir_mdp_fingerprint` | the IR's `mdp_fingerprint()` at training time | ties the run to the frozen model it was trained against, so a run cannot be silently re-attributed to a different IR |

Domains stay free to add their own keys. `mdp_conformance run.provenance` reads the set: **`algo_class` is the adoption marker** — a log without it is pre-convention and WARNs, a log with it but missing another key FAILs, and no run directory SKIPs. Keying adoption on any-key-present would fail the domains that honoured §8.6's older `sb3_version` line while excusing those that ignored it. It also fails a run whose `algo_class` is not among the IR's declared classes (`rl.algo`/`rl.algos`) — the artifact-vs-declaration check, which is only possible because the class is recorded rather than inferred.

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

### 8.6 Solve levels and the L1 derivation

Training runs are **leveled** by how much judgment produced their config.
The invariant: **level ≥ L2 ⟺ more than one training configuration was
tried** — replicate seeds are copies, not search; within-run model selection
(the post-hoc checkpoint screen, §9.7) is part of L1's single run; a tuned
result is always an escalation.

- **L0 — faithful defaults** (control, reporting only, never a gate). The
  library's defaults plus only what the *problem* forces: γ = β
  (`objective.discount_factor`), the algorithm class the IR's validators
  force, the env as built, the shared eval protocol, and a declared 2M-step
  budget constant. No VecNormalize, constant LR 3e-4, constant clip 0.2,
  `ent_coef` 0, batch 64, MLP (64,64), 1 env, terminal checkpoint
  (`--no_norm_obs --no_norm_reward --n-envs 1 --checkpoint-every-frac 0`). In one
  line: `PPO("MlpPolicy", env, gamma=β, seed=s).learn(2_000_000)` — anything
  that can't fit in that line is L1+. Record the SB3 version in args.txt.
  L0 is the ruler: Δ(L1−L0) measures the configuration layer per case.
- **L1 — the derived config** (the mandatory backbone). Every row below is a
  *forced move* — a hard rule reading the IR, logged with a one-line
  rationale (the `obs_normalization` decision pattern, generalized).
  Judgment beyond these rules is L2.
- **L2+ — escalations**, one level per round, tagged with the layer(s)
  opened: `L2(hp)`, `L2(gym)`, `L2(arch)`, `L3(hp+arch)`, … **Levels are
  per-family.** A *family* is the set of algorithm classes sharing one L0 form
  and one L1 derivation table: `ppo`, `maskable_ppo` and `recurrent_ppo` are
  one family, so their `Δ(L1−L0)` values are commensurable and comparing those
  classes on one leaderboard is legitimate (declare the axis as `rl.algos`).
  A genuinely different family — anything off-policy, anything model-based —
  has a different L0 line and different L1 knobs, so its delta is *not*
  comparable to this family's; two families meet only at the protocol layer, as
  `means` siblings. **Admitting a new family obliges two artifacts**: its own
  L0 faithful-defaults one-liner, and an L1 derivation table in its own knobs.
  Without them the family has no ruler and no backbone, and its levels are not
  levels. Budget extension
  ("still climbing at ceiling") is the cheapest L2 move.

**The L1 derivation table** (signal → knob):

| knob | rule |
|---|---|
| `gamma` | γ = β. Undiscounted indefinite horizon: γ = 1 − 1/T̄. γ < β only as a logged L2(hp) bias-variance move; γ > β never (more bias *and* more variance). |
| `gae_lambda` | λ sets the credit horizon `1/(1−λ)`. Choose the L1 value from **where action consequences realize** — which may be far shorter than the episode (a 2048 merge chain pays within ~10–20 moves; short credit won there, λ 0.90–0.95, game2048 #E34/#E38) or span the whole remaining horizon (an exploratory pull pays over everything after it; long credit was mandatory there, λ 0.99+, mab #E35 — at 167 steps of credit over T=20000, 2 of 3 seeds never learned at all). Coverage `(1/(1−λ))/T̄` is the diagnostic lens, not a formula: a λ carried across a horizon change silently changes what it means, so **re-examine it — in either direction — whenever T̄ moves**; it is per-instance, never a transferable tuned constant. No universal floor or band exists: the two campaigns' optima bracket ~2% to ~17% coverage. |
| `n_envs`, `n_steps` | rollout = n_steps × n_envs ≥ max(2048 transitions, 10 episodes); n_envs = 4 default, structural, never tuned; long T̄ → raise n_envs before inflating n_steps. That ordering is an **L1 move**, made once when the train script's defaults are written. At L2 it no longer applies: `mdp_tuning` holds n_envs at that default — it is a forced-move lock, not a knob — and meets the same floor from the other side, raising the n_steps lower bound to ⌈10·T̄/n_envs⌉ (`--episode-len`, `--min-rollout-episodes`). |
| `batch_size` | rollout/32 … rollout/8, power of two. |
| LR schedule | exogenous noise dominates reward variance (read the IR's uncertainty block) → 1e-4 → 1e-5; near-deterministic dense-reward → 3e-4 → 3e-5. `lr_final = lr_init/10`. |
| clip schedule | 0.2 → 0.05 (`clip_final = clip_init/4`). |
| `ent_coef` | 0.005; 0.01+ where premature determinism is a known hazard (bandit-like exploration, sparse success). |
| `n_epochs`, `target_kl` | 10 / 0.02 fixed; target_kl is the safety valve, never tuned — repeated `approx_kl` truncation is the LR-too-high signal, fix the LR. |
| `norm_obs` | §8.3 decision per the IR (heterogeneous stationary → on; drifting/accumulator obs → off, prefer sufficient-statistic obs). Both values are reachable from the CLI (`--no_norm_obs`, §8.2 tier 1) — a derivation whose result the script cannot express is not a derivation. **The signal is a prior, not a verdict**: both of its branches are claims about the state distribution a *trained* policy visits, so neither is observable when the row is applied, and two campaigns have falsified it in opposite directions (mab #E8/#E10 → +71.9 for the ON the rule ruled out; adi_flex F36–F38 → a 74.6-point observation-encoding artifact that exists only at the ON the rule prescribed; §8.3). So derive it *and* let `mdp_tuning`'s `breadth` tier check it — a campaign that opens it records the outcome the way γ's bias-variance move is recorded. Where the derived value is computed at runtime, §8.3's `BooleanOptionalAction` rule applies. |
| `norm_reward` | on, with `gamma=args.gamma` passed (§8.3). Reachable in `mdp_tuning`'s `all` scope and held at this value below it: §8.3's own reasoning — training-only, largely redundant with `normalize_advantage` — is what puts it in the frozen tier rather than at `breadth`, and no campaign has yet moved it. It is there for reachability, since a derivation nothing can contradict is not one. |
| `normalize_advantage` | on. PPO's per-minibatch rescaling of advantages to zero mean / unit variance, exposed as `--no-normalize-advantage`. It is stated here rather than inherited silently from SB3 because §8.3 *reasons from* it — it is the premise that downgrades reward norm to a critic-scaling detail — and a load-bearing premise no script exposes is one no campaign can check. `mdp_tuning` opens it at `breadth`. Off only as a logged L2 move: the one measurement is decisively for the default (adi_flex A14, +108.05 main effect for ON, ~6× worse seed dispersion for OFF), which is also why searching it is cheap rather than dangerous. |
| `net_arch` | (64,64) for obs dim ≤ ~32; scale the first hidden layer to ~2–4× obs dim above. Structured obs (set/permutation, grid, sequence) is never a width problem — record a *predicted escalation: arch* note. Boundary: `net_arch` widths = HP layer; custom extractors = arch layer. This row derives a **width**, not a layer count, which is why `mdp_tuning` searches width at its `core` tier and opens depth only at `breadth`: at core the derivation's own depth stands. The searched grid is width {32, 64, 128, 256} × depth {2, 3, 4}, uniform layers. |
| budget | ceiling = 20k–50k episodes × T̄ steps AND ≥ ~300 updates. **A training run runs to its budget — no early stopping.** The training trajectory is too noisy to make any judgment from; judgment happens post-hoc, on CRN evals of saved checkpoints (the selection row). Early stopping exists only in tuning trials (§9.7), where it reads the periodic CRN eval, never the rollout curve, and the arm is one of many. |
| model selection | **Post-hoc, three-layer (§9.7).** `CheckpointCallback` every ~5% of budget (~20 checkpoints — `--checkpoint-every-frac 0.05`, §8.2); after training, evaluate every checkpoint on the selection block (~2048 CRN seeds, disjoint from the protocol block — `--first-seed`, §9.7), take the top-k (k≈3, adjustable — widen when the leaders sit within one screen-SE), confirm those on the protocol block (~8192), ship the winner. No `EvalCallback`, no live selection env, no `sync_envs_normalization` — the machinery that selected a generalist's checkpoint on one wrong cell (mab #E36 V4) simply isn't there. **The terminal checkpoint is never the deliverable** — the marginal gain of the screen over a working callback is small (+1.4, mab #E33) but selecting *at all* is worth +43, and the post-hoc form buys the robustness. |

**Which knobs a tuning study opens follows this table.** `mdp_tuning`'s
tiers are budget scopes ordered by how much the derivation above already knows:
`core` corrects the knobs whose derived value is a real guess (`learning_rate`,
`net_arch` width, `n_steps`, `ent_coef`, `gae_lambda` — the row with the widest
measured spread), `breadth` opens what no row derives at all (`net_arch` depth,
`n_epochs`, `batch_size`, `vf_coef`, `normalize_advantage`) plus the two knobs
whose derivation is *bounded* rather than free — `gamma`, since γ > β is never
sampled, and `norm_obs`, whose row states a prior neither branch of which is
observable when it is made — and `all` adds what stays at its derived value
unless asked for (`clip_init`, `max_grad_norm`, `norm_reward`). Membership
answers *what does the derivation still not know*, which is a different
question from *which layer implements the knob* — so a tier spans layers, and
the paragraph below the diagnosis rules says how to read a winner's level. Schedule *finals* are in no tier — §8.2's
one-degree-of-freedom rule derives them from the tuned inits.

**The derivation records its basis, as data.** The train script's L1 table
comment states what each derived value was derived *from* — the measured T̄
(and at which scale/instance, under roughly what policy strength, since episode
length drifts as the agent improves in open-ended domains) — and `_L1_BASIS`
carries the machine-readable part of it beside `_L1_DERIVED` (§8.4):

```python
_L1_BASIS: dict[str, object] = {"scenario_name": "simple", "episode_len": 30,
                                "beta": 0.99}
```

β belongs here rather than among the derived values: it is a property of the
problem the derivation was made *against*, and reading it off the derived γ
instead would make the γ ≤ β rule tautological on a domain that derives γ = β
and wrong on one that derives γ = 1 − 1/T̄.

A derivation whose basis moved is stale even though every number in it is
unchanged: the one campaign that recorded its basis (`T~48`, game2048) is the
one that caught its own violation when the campaign changed scale; re-derive
when the instance, scale, or measured T̄ moves.

**The derivation is checked at launch, not printed at it.** `mdp_conformance`
asks *is this code spec-shaped?* before anything runs and `mdp_gates` asks
*does this number clear its baselines?* after everything has; the launch — *is
this experiment well-posed?* — is between them, and a banner is not a check. A
campaign printed `rollout=512x4=2048: >=2048 and >=10 episodes` unconditionally
at a scale where its runs held 2–6, for the whole campaign, at the scale
carrying every headline number. So `main()` calls
`assert_l1_current(args, derived=_L1_DERIVED, basis=_L1_BASIS,
episode_len=...)` (from `mdp_conformance.launch`) once the args are resolved,
which prints the derivation's rows as *checked*, lists this run's deviations
from them, and **refuses** three things §8.6 calls never:

- a **stale derivation**, in the one form that is machine-decidable — the
  derived rollout satisfied the ≥10-episode row at the T̄ it was measured on
  and does not at this run's. The remedy is a re-derivation, never an override:
  patching it with a flag records a broken derivation as an ordinary L2(hp)
  escalation and leaves the campaign citing a ruler it has replaced. A basis
  measured on a *different instance* only warns — a sibling scenario at the
  same scale leaves every row valid, and refusing it would make the check
  something to work around;
- **γ > β**;
- an **inverted schedule** (a final above its init, §8.2).

Everything else reports rather than refuses, and two exclusions are load-
bearing. The rollout floors stay **warnings** — measured as a co-factor, not a
cliff (game2048: r = +0.31, flat arms at 2.4 episodes did not collapse) — and
the **prior** rows (`norm_obs`, `normalize_advantage`) are never asserted at
all: their rows above state a prior only a run can settle, so deviating there
is the experiment the row asks for. An L0 run skips the check entirely, since
L0 is defined as the library's defaults and the L1 rows are not in force.

**Where a campaign keeps a §CONFIG-REGISTRY, two more groups join it, and they ask a different question.** The rows above ask *are these knobs derived*; these ask *is this run the run it says it is*. Both take their inputs as data — `assert_launch(args, derived=…, base=…, declared=…, address=…, owned=…, comparator=…, results_root=…)` — because the registry's own resolver computes the deviation list as the diff it would be checked against, so a check that re-resolved the citation would agree with itself. What makes it a check is *when* it runs: at the point of use, after every knob the script derives from another. The resolver runs at the top of `main()`, and anything that moves a knob afterwards silently invalidates the record it just wrote.

- **The configuration.** The args diff against the cited base must equal the declared deviation list *exactly*, in both directions: an undeclared move is a lever nobody chose, and a declared one the run does not have is a banner describing a treatment that is not running. The base must fix every knob an axis claims — one owned and unfixed declares nothing and diffs against nothing, so it is invisible both ways. The run must land in its own `results/{scenario_name}/` tree. And every cited id must have a §CONFIG-REGISTRY anchor — pass the data half's id set as `ids=` and the two halves are asserted against each other in both directions, so the reader-facing table cannot drift away from the module that configures runs. A run that cites **nothing** is an error rather than a skip: that is the state the registry exists to end.
- **The comparator.** The run this arm will be read against must exist, share the scenario, and carry a comparable budget. A base that was never run, a contrast across two scenarios, and a budget confound all read as treatment effects once the numbers are in hand, and the launch is the last moment any of them is visible. `settling=True` is the declared exception for a base still being extended — the mismatch downgrades to a warning instead of being tolerated silently.

An L0 run skips the configuration group as well (guide §13.4 puts L0 outside the registry, so it cites `{sc}/L0` and nothing resolves) but keeps the comparator check, since an L0 baseline is very often the run another arm is read against. `assert_l1_current` stays the L1-only entry point: a domain with no registry is not a domain that owes one.

Diagnosis at the L1 gate: L1 ≤ random → suspect the build, don't escalate;
L1 < L0 → the derivation misfired; L1 competitive vs baselines → done; gap →
open L2. Tuning (`mdp_tuning`) is the escalation the hp layer is searched from:
it warm-starts from the script's `_L1_DERIVED` (§8.4) — the parser defaults
where a script declares none — searches the
`core` knob tier by default, and derives schedule finals — see its `--knobs`,
`--fix`, `--beta`, `--episode-len` flags. A script that does not expose an
in-tier knob makes the study search a smaller space than it reports, so naming
a tier explicitly turns that into a launch failure rather than a warning (§8.2
tier 2).

**A study is not automatically `L2(hp)`, and the tier tables say which layers
it can reach.** Two of them cross the boundary this section draws elsewhere.
`core` searches the extractor knobs (`embed_dim`, `features_dim`, `channels`)
on the scripts that expose them — and the `net_arch` row above puts custom
extractors at the **arch** layer, not the hp one. `breadth` searches
`norm_obs` and `all` searches `norm_reward` — the §8.3 vec-env wrapper stack,
which this table derives from the IR's observation structure and objective and
which §8.3 makes part of the saved artifact's input contract, so it is a
**gym**-layer knob, not an optimizer setting. A winner that moved one of those
is `L3(hp+arch)` or `L3(hp+gym)` in the notation above, and the escalation
entry says so: **read the level off the knobs that actually moved**, never off
the fact that a study produced it. Two flags make this answerable rather than
archaeological — `--show-space` prints the resolved tunable set before launch,
and `--fix` holds a knob at its derived value, so a study that means to stay
inside one layer can say so (`--fix norm_obs`, `--fix embed_dim`).

**Every trial carries the derivation, not only trial 0.** A trial's command
sets the knobs the study *searches* and lets every other dest fall to the train
script's parser default — which was the derivation only while §8.4's identity
held. So each trial's command is seeded from `_L1_DERIVED` first, in the order
*derivation → study-level (scenario, budget, outdir, seed) → `--train-arg` →
sampled*: a trial then **is** the derivation plus the delta the study searched,
which is what makes Δ(L2−L1) readable off the study at all. A knob whose derived
value the script computes at runtime is left off the command line, since that is
how its derivation is applied (§8.4's carve-out). Two things follow. The launch
banner names every knob whose default has moved off its derived value — a
**null** default included, which is exactly what a script writes when the
derivation *cannot* be the default (`target_kl` and `clip_final` left null so L0
can have no KL valve and a constant schedule). And a hand-run of the train
script without those flags is a **different configuration** from the trial it
means to reproduce, which is the failure a stage-2 replication walks into.

**Trial 0 is the L1 centre only as far as the space can represent it.** The
warm start enqueues the derived values, and the space is a grid: `n_steps`
and `batch_size` are powers of two, `net_arch` is uniform width on a power of
two, `learning_rate` and `ent_coef` live inside log ranges. A derived value off
that grid — a rollout floor derived to 2560, an `ent_coef` of exactly 0 — is
**moved to its nearest representable neighbour**, and the move is reported in
both `--show-space` and the launch banner and recorded on the study. Rounding,
not dropping, because dropping hands the knob to the sampler and puts trial 0 an
unbounded distance from L1 (a 0 `ent_coef` becomes any value across seven
decades), where a rounding step is bounded by the grid. Read the report: trial 0
is then *L1 rounded to the grid*, and Δ(L2−L1) is measured against that.

**A refusal is not a rounding error.** Three values are rejected outright rather
than moved, because each is a signal that rounding would hide: a γ above β
(§8.6 rules it out entirely, so the script is stating a spec violation), an
`n_steps` below the domain's rollout floor (the floor says the default was
derived under a different T̄, so it is *stale* and wants re-deriving, not
rounding up), and a CNN channel stack (a structure has no nearest neighbour).
Those are sampled, and the study measures no delta against L1 for them.

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
    p.add_argument("-s", "--scenario_name",    type=str, default="simple")
    p.add_argument("-o", "--observation_mode", type=str, default="vec")
    p.add_argument("-a", "--action_mode",      type=str, default="box")
    p.add_argument("--n-seeds",    type=int, default=65536)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--outfile", type=str, default=None)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()
```

Two conventions the parsers follow:

- **Criterion-selecting boolean flags use `argparse.BooleanOptionalAction`**
  (`--stochastic` / `--no-stochastic`), never `store_true`: both values stay
  reachable from `mdp_tuning`'s `--eval-arg KEY=VALUE` pass-through, and the
  flag's meaning is visible in `--help`. A `store_true` whose other value is
  the silent default is exactly how 73 trials get ranked on the wrong
  criterion without an error.
- **The parser's names are §8.2's tier-1 table**, eval column: the problem and
  rendering modes, `model_path` / `vecnorm_path` / `outfile`, and the protocol
  pair `n_seeds` / `first_seed`. `first_seed` is what makes §9.7's *disjoint*
  blocks expressible — the protocol block starts at 0 and the screen block at
  `1_000_000`, and without an offset flag the requirement is unsatisfiable as
  written. (The corresponding train-side flag is `--tag`, §8.2.)

### 9.2 Seed loop

For each `(scenario_param_combo)`, run exactly `n_seeds` episodes over the
seed block `first_seed, ..., first_seed + n_seeds - 1` (the default block is
the 0-based one), accumulating the domain's per-episode objective (the episode
return, or whatever the domain reports as its outcome):

```python
returns = np.zeros(n_seeds)          # the domain's per-episode objective

for i, ep_seed in enumerate(range(args.first_seed, args.first_seed + n_seeds)):
    if i % 10000 == 0:
        print(f"  seed {i}/{n_seeds}", flush=True)
    obs, _ = env.reset(seed=ep_seed)
    ...
    returns[i] = info_last["<objective>"]   # e.g. profit (maximize) / cost (minimize)
```

**The block is the unit of comparison, so it must be identical across arms and
disjoint across layers.** Every arm on one leaderboard reads the same
`(first_seed, n_seeds)` — that is what makes the per-seed differences paired
(§9.7). A layer that *ranks* must not read the block a layer that *quotes*
reads, which is the whole reason the offset is a flag.

Default `n_seeds=65536` provides tight confidence intervals without tuning;
see §9.7 for the reporting tiers actually used per run level.

**Score with β.** When the IR's `objective.discount_factor` β < 1, the
per-episode objective is the *discounted* return `Σ β^t r_t` — in the PPO
eval **and in every benchmark eval identically** (the leaderboard must
compare one objective). β is applied here and as the training γ (§8.6),
never by pre-discounting rewards inside the env or `_mdp` (that would make
the reward non-stationary and pollute the differential contract). With the
default β = 1 the plain sum stands.

### 9.3 Output columns (TSV)

Columns are the varied scenario parameters followed by per-metric
`{name}_mean` / `{name}_var` pairs. **The domain's primary objective column
comes first among the `*_mean` columns** — that ordering is the contract both
`mdp_tuning` and `mdp_gates` rely on to auto-pick the metric when `--metric`
is not given (neither privileges any particular name). The objective's
**sense** (maximize vs minimize) is the domain's, and callers pass it through
(`mdp_gates --sense`, `mdp_tuning --minimize`).

**Write `≽` for *at least as good as*** — larger under `maximize`, smaller under
`minimize`. State every bound direction with `≽`/`≼`, **never** as
"upper"/"lower": those invert with the sense, so the word ends up naming the
domain rather than the benchmark. This is the vocabulary §5.2 and §9.9 use.

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

Leading `#` lines in an eval TSV are **provenance, not data** — model path,
seed block, scenario — and every §9 reader (`mdp_gates`, `mdp_tuning`) skips
them. Stamp them liberally: a TSV that states its own block is what makes a
cross-study comparison auditable.

**Three of them are a contract**, written by every eval script that scores a
trained artifact, because §13's harvest precondition reads them and the run
directory may not still be beside the TSV when it does:

```
# steps_scored: 8375000     <- num_timesteps of the model that produced this number
# steps_run: 10000000       <- the furthest step count any artifact in its run dir reached
# steps_budget: 10000000    <- the budget that run declared (_total_timesteps)
```

`mdp_gates.completion.provenance_lines(model_path)` renders them, reading SB3
zips with `zipfile` + `json` so the torch-free install keeps working. The budget
comes from inside the archive rather than from the args log on purpose: a copied
or stale log can disagree with the model it claims to describe, and the artifact
cannot.

A cost-minimizing domain would instead lead with `cost_total_mean` /
`cost_total_var` and be gated with `--sense minimize`.

**Bystander metrics (`eval_metrics`).** When the IR declares root-level
`eval_metrics` (each: `name`, `expr` over the END_OF_PERIOD namespace,
`reduce` ∈ last/sum/max/min across the episode), every eval script emits one
`{name}_mean` column per metric **after** the objective columns — the
first-`*_mean` contract above keeps gates and tuning blind to them by
construction. Bystander metrics *describe, never decide*: they never gate,
never crown a record, and never drive model selection. Typical origin: an
objective candidate the human declined at Phase A but wants to keep seeing
(recorded in the metric's `source`); since a metric never feeds the policy,
it is also a legal home for latent references (§6.4's reward-latent rule
does not apply to it).

Two conventions travel with the columns:

- **Per-seed sidecar.** Each eval also writes `<name>.seeds.tsv` — one row
  per episode (`seed`, objective, then one column per metric). This is the
  ground truth for any distribution question; since every eval shares one
  seed list (§9.2), it also enables *paired* per-seed comparisons, which are
  far sharper than comparing means.
- **Distribution columns are derived, never curated.** If a metric's support
  is enumerable from the scenario (e.g. a max-tile metric whose ceiling is
  known from the board size), threshold/CDF columns are enumerated over the
  full support — degenerate ends reading 1.000/0.000 are self-evident and
  harmless; trimming is the presentation layer's job. Hand-picked threshold
  dicts are per-domain judgment that goes stale as policies improve.

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

### 9.7 Eval tiers — the three-layer evaluation

One principle organizes all judgment: **the training trajectory is never a
judgment input** — too noisy for any decision, in training or tuning. Every
judgment reads a CRN eval, and the three layers trade randomness against
efficiency (the numbers are defaults, adjustable case by case; the *blocks
are mutually disjoint* — a layer that ranks must not touch the block that
quotes).

Disjointness is a property of the seed block, so it is set by
`--first-seed` / `--n-seeds` (§8.2 tier 1, §9.1): the **protocol** block is
`first_seed = 0`, the **screen** block is `first_seed = 1_000_000`, and a
trial's periodic eval reads its own block above that. The offsets are
conventional, the disjointness is not.

| layer | where | episodes | job |
|---|---|---|---|
| trial | periodic eval inside a tuning trial, every ~5% of budget | ~512 CRN seeds | the tuner's signal: trial value = the **last** eval (it describes the artifact the trial ships); early stopping reads THIS curve, never the rollout curve — patience ~4 evals with a min-evals guard ~5, strict comparison (the fixed CRN block pairs the evals, so policy differences are not draw noise); `trial.report` on the same curve enables population pruning. **Its scores are never quoted** — a trial value is a tuning signal on a different instrument, and comparing one to a protocol number manufactures a result in either direction |
| screen | post-hoc over a training run's ~20 checkpoints (§8.6, `--checkpoint-every-frac 0.05`) | ~2048 CRN seeds from `--first-seed 1000000` | rank the checkpoints, pass the top-k to confirmation; its scores are never quoted |
| protocol | `{domain}_ppo_eval.py` / benchmark evals, `--first-seed 0` | **~8192 evidence-grade** (2048 default for cheap domains) | confirm the top-k, crown the winner — the leaderboard number; a ladder run *exists* only once this TSV does |

A smoke eval (train-script tail, ~50 episodes, "did it learn anything")
stays outside the layers and is never quoted.

**A tuning winner is an artifact, not a result.** It enters a leaderboard, a
comparison, or a claim only through a protocol-layer TSV produced by
`{domain}_ppo_eval.py`. Its trial value *selected* it; that value never
*scores* it. The two are different instruments — the last eval of one run on a
~512-seed block versus ~8192 seeds on the record block — so a trial-vs-protocol
comparison manufactures a verdict in whichever direction the layers happen to
differ. A campaign that skips the re-score can close a tuning round as a null
result on numbers that were never comparable.

**The reported best also carries a selection effect**, which changes how the
re-score reads: it is a *maximum* over the study's trials, so the winner's
protocol score is an honest measurement of **that artifact** but an optimistic
estimate of what re-running the tuning *procedure* would yield. Claims about
the procedure's value need a multi-seed retrain of the winning configuration;
claims about the shipped artifact do not.

**The leaderboard is a table**, in the domain `README.md`: one row per arm —
the trained policy, every baseline, the reference/oracle if one exists —
with the metric, the paired delta against the stated comparison, and its
significance, plus a column or note carrying each row's scope where arms
differ in it. Numbers a reader must compare across arms do not belong in
prose; the same applies to any multi-arm readout in the campaign docs
(ESCALATION_LOG_GUIDE §6).

The screen layer is affordable because the **selection evaluator is
vectorized over episodes** — the old "~256–512 seeds" tier reflected a
scalar `predict`-per-step loop's budget, not a statistical judgment; batched
eval runs 20–100× cheaper per step than training, and the seed count stops
being the constraint.

**Comparing two arms over a grid.** When the eval enumerates a §5.6 grid, the
comparison is per cell (§5.6 Rider 2: report per cell, never quote a single
aggregate as the score) — and if an aggregate is unavoidable, three things
govern it:

- **The replication unit is the seed, not the cell.** The grid is *enumerated*,
  so `std(cell Δs)/√n_cells` is a dispersion measure, not a sampling error:
  it estimates how much the answer would move on a fresh draw of cells, which
  is not the claim being made. Take the SE across seeds over the pooled paired
  rows. The two differ by more than rounding — CRN correlates the cells, and a
  campaign measuring both found the cell-based form understating the SE by
  ~1.8× on three separate arms, enough to turn a 3.3 SE result into 1.8 SE.
- **State the weighting, because equal weight is one.** Where cells differ in
  scale — any multiplicative link, any cost or horizon axis — an equal-weight
  mean of absolute deltas is a scale-weighted vote, and the largest corner
  decides. Normalize per cell, or weight deliberately, or don't aggregate.
- **Report the median and the fraction of cells favouring the arm beside the
  mean.** A mean whose median has the opposite sign is not a weak result, it
  is a different result; the fraction favouring is the cheapest one-number
  tell that this has happened.

Sizing: from a ~500-episode pilot SD σ, `n ≈ (2σ/ε)²` for the smallest delta
ε worth resolving. Make comparative claims on **per-seed paired differences**
(shared episode seeds are common random numbers — instance luck cancels, and
the paired SE is typically several times smaller than the naive per-policy
SE). Cost scales with `n × T̄`, so pick the smallest power of two that meets
precision, not 8192 by reflex. At evidence-grade (3 training seeds), each
replicate gets the full reporting eval; retrain spread and eval SE are
different uncertainties — report both, never one as the other.

### 9.8 Multiple policies per benchmark solver

A benchmark solver may expose several closely-related policies through a `--policy` argument (e.g. a fluid-relaxation file offering `fixed`, `myopic`, and `random`), rather than one file per policy. Name the file after the method family it embodies (`{domain}_benchmark_fluid.py`), and encode the selected policy in the eval TSV name (`benchmark_myopic_eval_{scenario}.tsv`) so each benchmark run writes a distinct, gate-referenceable file.

### 9.9 Benchmark roles — where a benchmark sits relative to the optimum

A benchmark's `--baseline`/`--reference` role is a per-comparison choice (§9). Its **position relative to the optimum** is not: it follows from how the benchmark is built, and it is what makes a leaderboard readable. Three positions, stated with `≽` (§9.3) so they hold under either sense:

| role | how it is built | vs the optimum | attainable? |
|---|---|---|---|
| `relaxed` | drops a constraint, or grants information the deployed policy will not have (LP/Lagrangian relaxation; clairvoyant / hindsight — §5.2) | `≽ opt` | no |
| `exact` | solves the MDP (a converged DP over the true model) | `= opt` | yes |

| `feasible` | runs a real policy under the real information set — heuristics, fitted rules, **and the RL artifact itself** | `≼ opt` | yes |

**`role` describes the algorithm's logic, not its numerical implementation.** A converged DP over the true model is `exact` even though it truncates a distribution, discretizes a state, or stops at a tolerance — record that in `basis`, which is what `basis` is for. Read otherwise, `exact` is a category no implementation can hold: value iteration stops at a tolerance, unbounded supports are truncated, float64 sums round. A benchmark whose *logic* solves the MDP is `exact`; a benchmark whose logic is a heuristic is `feasible` however precisely it is computed.

Two consequences worth stating because they are checkable rather than editorial:

- **A `feasible` arm scoring strictly better than an `exact` or `relaxed` arm is impossible.** It means the eval, the bound, or the simulator is wrong — the RL policy beating a verified-optimal DP is not a result, it is a bug report. **Judged against a band, not to the last float**: `max(tolerance, z·SE)`, where `tolerance` is the entry's declared implementation slack and the statistical term comes from the same CRN estimate the gate already computes for baselines. That slack is a claim about the rendering being judged, not about the entry — one solver file can serve two renderings whose error differs in size and in kind (a lattice decomposition that is exact, a binned density that is not) — so `tolerance` may name a scenario constant and resolve per instance (§5.0), and `mdp_gates --instance` is what tells the gate which one to resolve against. Inside the band the gate reports and does not fail — a `~` in the guide's marks, never a `✗`. This matters precisely for the domains the pipeline is meant to produce: a policy that closes on a discretized bar would otherwise be told its simulator is broken, and the quickest way to silence that is to demote the bar to `feasible`, which destroys the bracket.
- **`relaxed` together with `feasible` brackets the optimum**, so a domain carrying both has a certified optimality gap *without* an exact solver — which is the common case for the domains that most need one.

**Declare the roles in the IR**, in the root-level `benchmarks` block — one entry per `{domain}_benchmark_{method}.py`, keyed by the same `{method}`:

```json
"benchmarks": [
  {"name": "dp", "role": "exact",
   "basis": "value iteration to 1e-9; exact up to Poisson tail truncation at mass < 1e-9",
   "source": "human_confirmed"},
  {"name": "myopic", "role": "feasible", "basis": "one-step lookahead"}
]
```

`basis` is one line on *how* it is built — what determines the role, and where a caveat belongs; `policies` lists a §9.8 multi-policy solver's policies. Root level for the same reason as `eval_metrics`: benchmarks are built in Phase B, so declaring one never moves `mdp_fingerprint`. The RL artifact needs no entry — a trained policy under the real information set is `feasible` by construction. The block is optional and additive; an IR without it validates unchanged.

Two gates read it, which is the point of declaring rather than describing:

- **`mdp_conformance`** (`benchmarks.declared`) — every declared benchmark has a file and every file is declared, so a renamed method or a deleted solver cannot go unnoticed. SKIPs when nothing is declared.
- **`mdp_gates --ir {domain}_schema.json`** — refuses `--baseline` on an `exact` or `relaxed` arm (exit 2: must-beat asks the candidate to beat a bound it cannot beat), and **fails the gate when the candidate beats an `exact` or `relaxed` reference**, quoting §9.9. Without `--ir` the gate behaves exactly as before.

Name the role beside the leaderboard too; a relaxation and a heuristic look alike in a table of numbers, and the bracket they jointly form goes unstated unless something says which is which.

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
- **`logger_filename` is always an absolute path under the run directory** —
  `str(outdir / "train_log")`. A bare name resolves against the CWD (§8.4),
  which is how a domain ends up littering the folder it was launched from.

**These loggers are off by default.** The train script exposes
`--gym-log {0,1}`, default **`0`**, and passes
`logger_filename=str(outdir / "train_log") if args.gym_log else None`. It is a
diagnostic toggle, not a design axis, so it goes in `_SKIP_KEYS` and never
appears in the run name.

Rationale: the per-step stream writes one line per environment step, from
inside the hot loop. Measured on this codebase, a single `train_log_step.log`
reaches 150–830 MB, and per-domain tuning trees ran to 6.5 GB (adi_flex) and
30.1 GB (topk_id) of step logs — 97% of that study's total footprint — for
traces nothing subsequently read. Under `mdp_tuning` the cost is also
per-trial and lands on exactly the runs whose wall-clock is being compared.
Enable it deliberately when debugging a transition or a reward, for one run,
and leave it off otherwise. No tuning-side special case is needed: the default
already holds for studies, so a domain that never adds the flag is still
correct.

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
- **The harvest precondition: the run that produced the candidate reached the
  budget it declared.** §8.6 says a training run runs to its budget and nothing
  checked it, at any stage — a run that stopped at 6% of budget leaves a valid
  `best_model.zip`, feeds a well-formed TSV with correct standard errors, and is
  indistinguishable from a bad arm in the output, so the gate reports the second
  correctly and the first as a finding. The check reads the `steps_run` /
  `steps_budget` provenance the eval stamped (§9.3) and asks whether the step
  count is **known and declared**, not whether the run finished:

  | state | verdict |
  |---|---|
  | reached budget (within one checkpoint cadence, `--budget-tolerance`) | pass |
  | short, and declared with `--short-ok "<reason>"` | pass — the reason lands in the gate output the ledger quotes |
  | undeterminable — no provenance | WARN, and the comparison still runs |
  | short, with nothing declaring it | **FAIL**, reporting the fraction instead of a verdict |

  Two states look exactly like truncation from outside and are legitimate, which
  is why the question is not "did it finish": a run directory that is a **partial
  mirror** of a completed one (the copy stops wherever it stopped, so
  `checkpoints/` ends early), and a run the operator **killed on purpose**. A
  killed run and a silently dead one leave identical artifacts, so the
  declaration is the one input no file can carry — hence a flag rather than a
  marker file. And a scored model below the budget is *not* the signal: §9.7
  makes the deliverable a mid-run checkpoint chosen post-hoc, so the reading is
  the **run's** terminus, taken as the furthest step count any artifact in the
  run dir reached.
- **Trial-layer numbers are out of scope.** A tuning trial may legitimately stop
  early on its CRN eval curve (§8.6), and §9.7 already forbids quoting a trial
  score as a result — the precondition applies to a campaign candidate.
- **Post-tuning rule**: a tuning study's winner was *selected* on its tuning
  eval seeds, so its selection score is optimistic. Always re-evaluate the
  winning artifact with the full protocol (more seeds than the tuning eval)
  before reporting or gating it. And the gap is not only optimism: a
  different seed block is a different draw of the problem, so **the oracle
  itself moves** between blocks (mab measured 16.24 points) — the shift can
  go either direction and affects every trial alike, which is why a study's
  absolute scores are comparable only within that study.

---

## 14. Policy Interpretation (`{domain}_policy_probe.py`, `{domain}_plot_policy.py`)

*(Added 2026-07-30, distilled from the `inv_single` escalation campaign — the
first end-to-end execution of the pipeline's interpret step.)*

The pipeline's aim is a competitive policy **and** the structural insight
classical DP analysis used to deliver: after `solve`, read the trained policy
back into the domain's policy-structure vocabulary (a base-stock level,
(s, S) thresholds, an index rule, protection levels, …). Interpretation sits
between `solve` and `package`; its findings feed the README's
empirical-findings section and the final report.

Two forms, mirroring formalize's replicate/elicit duality:

- **Anchored** — a reference policy or a predicted structural class exists
  (an exact DP, a paper's policy, a classical form from theory). This
  section specifies the anchored form; it is **required** whenever the
  campaign declares a `confirm` or `discover` tier-2 stance (§14.0). Shipping
  a DP/reference baseline makes the anchored form *available*; declaring the
  stance is what makes it owed.
- **Open** — no reference and no predicted class. Only the generic pieces
  below apply (the action-surface figure, the feature-sensitivity sweeps);
  discovery-style probes are deliberately not yet spec'd — conventions
  accumulate from live campaigns first.

Scope note: this section specifies the **readback that ships** — the
deliverable read from the winning artifact at Stage 5. Probes run *inside*
an escalation campaign (diagnosing a failing arm, checking a mechanism
before spending budget) are campaign record, governed by the escalation
guide, not by this section; their outputs land in the run's `probe/` dir
(§8.4) and their lessons in the campaign's `PLAYBOOK.md`.

### 14.0 Declare the tier-2 stance first (`research_questions`)

Whether a readback is owed at all is a property of the **claim**, not of the
problem — and `2-structural` (ESCALATION_LOG_GUIDE §3.1) is three claims, taking
different evidence:

| stance | the claim | evidence | §14 artifacts owed |
|---|---|---|---|
| `confirm` | RL recovers a structure we already know — `(s,S)`, an index rule, echelon coordinates | the readback **agrees with the reference** policy (§14.3) | probe + `INTERPRET.md` |
| `discover` | the policy has structure we did *not* know | **structural-form fit + paired scoring** of the fitted rule, which is then a first-class §9 benchmark | probe + `INTERPRET.md` |
| `bypass` | RL does as well *without* the structure, or without the machinery that produces it | the **outcome comparison alone** — typically an ordered `means` split whose cross-link delta is the price of generality | none |

The same measurement reads three ways, which is why the stance is declared
rather than inferred: `raw ≈ echelon` is a **success** under `bypass` (the
structure was not needed), **inconclusive** under `confirm` (consistent with
having found it, and no demonstration), and irrelevant under `discover`.
Undeclared, a finding and a gap are indistinguishable in the record.

Declare it at Phase A, in the same round that fixes the objective and the mode
stance — it is the same kind of commitment, and it decides a Stage-5
deliverable:

```jsonc
"research_questions": {
  "tier2": [
    {"stance": "bypass", "structure": "echelon coordinates",
     "instrument": "gym.observation_mode extension chain (echelon ⊂ raw)",
     "claim": "a policy from raw installation stock matches one given the echelon transform"},
    {"stance": "confirm", "structure": "echelon coordinates", "priority": "secondary",
     "instrument": "policy readback on the crowned artifact",
     "claim": "the learned policy is echelon-structured"}
  ]
}
```

`probe_required` defaults from the stance (`true` for confirm/discover, `false`
for bypass) and may be set explicitly to argue an exception in the file rather
than in prose. Two stances on one structure is a legitimate shape, as above:
bypass is the primary claim, and confirm rides along because the structure is
*known* optimal, so a good enough policy must arrive there and the readback
says whether it did.

Tier 1 (how does RL compare with the existing solutions, exact and heuristic?)
is **standing** — every campaign asks it, so it is never declared. Tier 3 is
engineering.

`mdp_conformance research.deliverables` enforces this: a declared
`confirm`/`discover` stance requires `{domain}_policy_probe.py` and
`INTERPRET.md`; a bypass-only campaign passes with neither; an IR with no block
SKIPs.

### 14.1 The probe (`{domain}_policy_probe.py`)

Loads a trained artifact exactly as `{domain}_ppo_eval.py` does — §9.5
VecNormalize injection, deterministic actions — so the probed policy is the
same object the leaderboard scored. Core operation: sweep the policy over a
declared state grid at a fixed period (remaining state coordinates held at
declared values) and report, per probed period:

- **the raw action surface** (state → action), dumped so figures and fits
  never re-run the sweep;
- **a structural-form statistic** for the predicted class, stated as a
  number, never an impression — e.g. order-up-to flatness = the spread of
  `state + action` over the acting region (0 = exact form); an (s, S) fit =
  the largest still-acting state `s` plus the mean post-action level `S`
  over acting states;
- **recovered thresholds vs the reference's** (`S_hat` vs `S*`, per period),
  when a reference exists;
- **action agreement** with the reference over the probed grid;
- **feature-sensitivity sweeps**: hold the state fixed, sweep one
  observation feature over its range, report how far the action moves. This
  is the *mechanism* check — does the net condition on this feature? — and
  it is independent of any score verdict: a feature can be score-neutral at
  the arm level while the net demonstrably latches onto it (and the reverse
  claim, "the net ignores it", is only checkable here).

Rules:

- **Validate the instrument where the answer is known.** This binds every
  instrument, not just this probe: a replay harness, a fast
  re-implementation (§1.2's second-implementation gates), a critic readout —
  each is trusted for discovery only after it has recovered a fact the
  domain already certifies (the exactly-solvable rung, the DP case, a
  recorded protocol number). An instrument that has never recovered a known
  answer is not measuring yet.
- **The probe doubles as the training diagnostic.** Before escalating a
  failing arm (§8.6 diagnosis), probe it: a reading like "flatness 25,
  agreement 0.04" says the net is not conditioning on the state at all,
  which localizes the failure to the interface (action encoding, observation
  scaling) faster than any learning curve — and before any tuning money is
  spent.

### 14.2 Scoring the fitted rule

The fitted structural rule is itself a policy. Evaluate it under the **same
§9 protocol** as every other arm — same CRN seed block, same paired report —
and publish the trio **reference / fitted rule / raw net** with paired Δs.

Rationale: the fitted rule can **beat** the net it was read from — fitting a
constant threshold deletes the raggedness (plateau oscillation) the net
carries. Interpretation is therefore a candidate policy *improvement*, not
merely an explanation; when the fitted rule wins, it is a shippable artifact
(a few numbers per period replacing a network) and the README must say so.

- **Pre-decide the verdict branches** before scoring:
  1. clean recovery and |fitted − net| ≤ 1 % of the bar → the net
     implements the predicted structure; claim it;
  2. |fitted − net| > 1 % of the bar → the net is doing something else —
     open a diagnosis before claiming the structural form;
  3. no structure recovered → suspect the action interface first (encoding,
     bounds, masking), not the theory.
- **Full-horizon assertion.** A fitted rule scored from a partial period
  sweep silently acts arbitrarily in the unprobed periods and returns a
  plausible-looking garbage score (the observed failure mode inflated cost
  13× without erroring). The scorer must assert the fit covers every period
  of the horizon.
- **Quote absolute gaps beside percentages.** Bars differ in magnitude
  across instances, so the same absolute gap can read as +1 % on one rung
  and +4 % on another. A cross-instance percentage comparison without the
  absolute number misleads.

### 14.3 The figure contract (`{domain}_plot_policy.py`)

The plot is usually how the insight is delivered; it has its own contract:

- **Canonical coordinates.** Plot the action surface in coordinates where
  the predicted structure is a canonical shape — e.g. post-decision level
  vs pre-decision state, where a base-stock policy is a flat line and an
  (s, S) policy is a flat plateau at `S` joining the `y = x` no-act diagonal
  at `s`. Deviations from the form then show as visible raggedness, not
  arithmetic.
- **Overlay, never side-by-side**: the reference policy underneath the
  trained arms' per-seed traces, so agreement and deviation regions are
  directly readable — including where the seeds disagree with each other
  (threshold states are where they blow open).
- **One plot spec, two renders**: a static figure (SVG/PNG) and an
  interactive HTML written from the same spec in one run, so the two can
  never drift.
- **Where figures land** (the figure companion to §8.4):
  - the **committed static** figure → `{domain}/figures/` — this is what
    README / campaign markdown inlines;
  - the **interactive HTML** → `results/{scenario}/figures/` — gitignored,
    beside the runs it derives from. The location split *is* the gitignore
    boundary; no extra ignore rules are needed.
  - Committed markdown cites the **regenerating command**, never a link to
    the interactive file — that file does not exist in a fresh clone.

"""Conformance checks for MDP domains.

Each check takes a ``DomainHandle`` and returns a ``CheckResult`` (or a list of
them). Checks fall into two groups:

- **Static** — parse the source (AST) without running it: file layout, acyclic
  layering (§1.1), the ``param`` ban (§2), no reward in the MDP layer (§6.4),
  and ``@dataclass(slots=True)`` on the state.
- **Behavioral** — construct the gym and run the simulator: scenarios build,
  ``init_state`` invariants, the Gymnasium reset/step contract, whole-episode
  determinism, episode-seed provenance across resets (§7), ``advance`` purity,
  and the per-generator RNG seeding contract.

Behavioral checks drive the domain through its gym wrapper so that single-step
(``advance``) and two-step (``advance1``/``advance2``) domains are handled
uniformly. The RNG check works at the generator level: a generator that yields a
value from only ``(period, episode_seed, seed_salt)`` is decision-path
independent *by construction*; one that needs more state (e.g. a board-game
spawn conditioned on the current board) is classified state-conditioned and
skipped rather than failed.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import math

import numpy as np

from .loader import DomainHandle, CheckResult


# ---------------------------------------------------------------------------
# Equality helper (handles dataclasses, ndarrays, nested containers, NaN)
# ---------------------------------------------------------------------------

def _eq(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b) and not isinstance(a, type):
        fa = {f.name: getattr(a, f.name) for f in dataclasses.fields(a)}
        fb = {f.name: getattr(b, f.name) for f in dataclasses.fields(b)}
        return _eq(fa, fb)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            if math.isnan(a) and math.isnan(b):
                return True
        except (TypeError, ValueError):
            pass
    try:
        if bool(a == b):
            return True
    except Exception:
        pass
    # plain objects (e.g. generator instances held by a scenario) usually lack
    # __eq__; compare them structurally so sampler-purity checks can compare
    # two independently constructed scenarios field by field
    if type(a) is type(b):
        if getattr(a, "__dict__", None):
            return _eq(vars(a), vars(b))
        slots = getattr(type(a), "__slots__", None)
        if slots:
            return all(_eq(getattr(a, s, None), getattr(b, s, None)) for s in slots)
    return False


# ---------------------------------------------------------------------------
# Shared behavioral helpers
# ---------------------------------------------------------------------------

def _pick_scenario(h: DomainHandle):
    """Choose a small, fast scenario; prefer a 'simple'-ish key."""
    keys = list(h.SCENARIOS)
    for k in keys:
        if "simple" in k.lower():
            return h.SCENARIOS[k]
    return h.SCENARIOS[keys[0]]


def _concrete(value):
    """Resolve a scenario-or-sampler to a concrete scenario (samplers are callable)."""
    return value(0) if callable(value) else value


def _max_steps(scenario) -> int:
    horizon = getattr(scenario, "horizon", None)
    return int(horizon) + 5 if isinstance(horizon, int) else 3000


def _build_env(h: DomainHandle):
    return h.env_cls(scenario=_concrete(_pick_scenario(h)))


def _rollout(h: DomainHandle, action_seed: int, max_steps: int) -> tuple[list, bool]:
    """Run one episode; return (per-step records, terminated?)."""
    env = _build_env(h)
    obs, info = env.reset(seed=0)
    env.action_space.seed(action_seed)
    records = [(obs, 0.0, False, False, info)]
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        records.append((obs, reward, terminated, truncated, info))
        done = bool(terminated or truncated)
        steps += 1
    return records, done


def _find_state_obj(env, state_cls):
    """Return the {Domain}State instance held by the env (not the scenario)."""
    if state_cls is None:
        return getattr(env, "_state", None)
    val = getattr(env, "_state", None)
    if isinstance(val, state_cls):
        return val
    for value in vars(env).values():
        if isinstance(value, state_cls):
            return value
    return None


def _discover_generators(scenario) -> list[tuple[str, object]]:
    """Collect (label, generator) pairs reachable from the scenario fields."""
    out: list[tuple[str, object]] = []
    seen: set[int] = set()

    def visit(obj, label):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "sample") and callable(getattr(obj, "sample")):
            out.append((label, obj))
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{label}[{i}]")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{label}[{k!r}]")

    if dataclasses.is_dataclass(scenario) and not isinstance(scenario, type):
        for f in dataclasses.fields(scenario):
            visit(getattr(scenario, f.name), f.name)
    return out


class _MinimalCtx:
    """A SamplingContext exposing only the exogenous seed-key fields."""

    def __init__(self, seed_salt: int, period: int = 0, episode_seed: int = 0):
        self.period = period
        self.episode_seed = episode_seed
        self.seed_salt = seed_salt


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

def check_file_layout(h: DomainHandle) -> CheckResult:
    # {domain}_exceptions.py is optional (spec §3), so its absence is not flagged.
    missing_required = [r for r in ("mdp", "scenarios", "gym") if r not in h.files]
    if missing_required:
        return CheckResult("static.file_layout", "FAIL",
                           f"missing required files: {missing_required}")
    if "uncertainty" not in h.files:
        return CheckResult("static.file_layout", "WARN",
                           "no {domain}_uncertainty.py — valid only if the dynamics "
                           "are deterministic (spec §1, §4.3)")
    return CheckResult("static.file_layout", "PASS", f"prefix '{h.name}'")


def _imported_modules(tree: ast.Module) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def check_layering(h: DomainHandle) -> CheckResult:
    # grids sit beside the chain (spec §1.1): only drivers may import them
    forbidden = {
        "uncertainty": ("_scenarios", "_mdp", "_gym", "_grids"),
        "scenarios":   ("_mdp", "_gym", "_grids"),
        "mdp":         ("_gym", "_grids"),
        "gym":         ("_grids",),
    }
    violations = []
    for role, suffixes in forbidden.items():
        if role not in h.trees:
            continue
        for mod in _imported_modules(h.trees[role]):
            if any(mod.endswith(s) for s in suffixes):
                violations.append(f"{role} imports {mod}")
    if violations:
        return CheckResult("static.layering", "FAIL", "; ".join(violations))
    return CheckResult("static.layering", "PASS",
                       "acyclic uncertainty←scenarios←mdp←gym; grids driver-only")


def check_no_param(h: DomainHandle) -> CheckResult:
    hits = []
    for role, tree in h.trees.items():
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.arg):
                name = node.arg
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            if name and (name in ("param", "params") or name.startswith("param_")):
                hits.append(f"{role}:{getattr(node, 'lineno', '?')} '{name}'")
    if hits:
        return CheckResult("static.no_param", "FAIL", "; ".join(hits[:8]))
    return CheckResult("static.no_param", "PASS", "no param/params/param_* identifiers")


def check_mdp_no_reward(h: DomainHandle) -> CheckResult:
    """Flag reward being *computed* in the MDP layer (§6.4).

    Detects assignments to a ``reward`` variable and ``reward`` used as a dict /
    subscript key. Mentions inside docstrings and comments (e.g. a NOTE stating
    that reward is derived in the gym) are ignored — they are good practice.
    """
    tree = h.trees["mdp"]
    hits = []
    for node in ast.walk(tree):
        # assignment target named 'reward'
        targets = []
        if isinstance(node, (ast.Assign,)):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id == "reward":
                hits.append(f"line {node.lineno}: reward = ...")
        # 'reward' used as a dict key
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "reward":
                    hits.append(f"line {node.lineno}: dict key 'reward'")
        # 'reward' used as a subscript index, e.g. info["reward"] = ...
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and node.slice.value == "reward":
            hits.append(f"line {node.lineno}: subscript ['reward']")
    if hits:
        return CheckResult("static.mdp_no_reward", "FAIL",
                           "reward computed in MDP layer — " + "; ".join(hits[:5]))
    return CheckResult("static.mdp_no_reward", "PASS", "MDP layer is reward-agnostic")


def check_state_slots(h: DomainHandle) -> CheckResult:
    tree = h.trees["mdp"]
    results = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name.endswith("State")):
            continue
        has_slots = False
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, "id", getattr(dec.func, "attr", "")) == "dataclass":
                for kw in dec.keywords:
                    if kw.arg == "slots" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        has_slots = True
        results.append((node.name, has_slots))
    if not results:
        return CheckResult("static.state_slots", "WARN", "no *State class found in MDP module")
    bad = [n for n, ok in results if not ok]
    if bad:
        return CheckResult("static.state_slots", "WARN",
                           f"State without @dataclass(slots=True): {bad}")
    return CheckResult("static.state_slots", "PASS",
                       f"{', '.join(n for n, _ in results)} slotted")


# ---------------------------------------------------------------------------
# Behavioral checks
# ---------------------------------------------------------------------------

def check_scenarios_valid(h: DomainHandle) -> CheckResult:
    if not isinstance(h.SCENARIOS, dict) or not h.SCENARIOS:
        return CheckResult("behavior.scenarios_valid", "FAIL", "SCENARIOS missing or empty")
    problems = []
    for key, value in h.SCENARIOS.items():
        try:
            sc = _concrete(value)
        except Exception as e:
            problems.append(f"{key}: construct failed ({type(e).__name__}: {e})")
            continue
        # the key names the SOURCE: check the registry value's own name when
        # it carries one — a mixture resolves to a *component* scenario that
        # rightly keeps the component's name (standalone equivalence)
        name = getattr(value, "scenario_name", None)
        if name is None:
            name = getattr(sc, "scenario_name", None)
        if name is not None and name != key:
            problems.append(f"{key}: scenario_name={name!r} != registry key")
    if problems:
        return CheckResult("behavior.scenarios_valid", "FAIL", "; ".join(problems[:6]))
    return CheckResult("behavior.scenarios_valid", "PASS",
                       f"{len(h.SCENARIOS)} scenarios construct and are keyed consistently")


def check_init_state(h: DomainHandle) -> CheckResult:
    scenario = _concrete(_pick_scenario(h))
    state, info = h.init_state(scenario, 0)
    if not isinstance(info, dict):
        return CheckResult("behavior.init_state", "FAIL", "init_state info is not a dict")
    if "action_period" not in info:
        return CheckResult("behavior.init_state", "WARN", "info lacks 'action_period'")
    expected = state.period - 1
    if info["action_period"] != expected:
        return CheckResult("behavior.init_state", "FAIL",
                           f"action_period={info['action_period']} != state.period-1={expected}")
    return CheckResult("behavior.init_state", "PASS",
                       f"period={state.period}, action_period={info['action_period']}")


def check_gym_contract(h: DomainHandle) -> CheckResult:
    env = _build_env(h)
    reset_out = env.reset(seed=0)
    if not (isinstance(reset_out, tuple) and len(reset_out) == 2):
        return CheckResult("behavior.gym_contract", "FAIL", "reset() must return (obs, info)")
    obs, _ = reset_out
    contained = env.observation_space.contains(obs)
    action = env.action_space.sample()
    step_out = env.step(action)
    if not (isinstance(step_out, tuple) and len(step_out) == 5):
        return CheckResult("behavior.gym_contract", "FAIL",
                           "step() must return (obs, reward, terminated, truncated, info)")
    _, reward, terminated, truncated, _ = step_out
    if not np.isfinite(float(reward)):
        return CheckResult("behavior.gym_contract", "FAIL", f"non-finite reward {reward}")
    if not isinstance(bool(terminated), bool):
        return CheckResult("behavior.gym_contract", "FAIL", "terminated is not boolean")
    if not contained:
        return CheckResult("behavior.gym_contract", "WARN",
                           "reset obs not in observation_space (dtype/bounds mismatch)")
    return CheckResult("behavior.gym_contract", "PASS", "reset/step contract, finite reward")


def check_determinism(h: DomainHandle) -> CheckResult:
    max_steps = _max_steps(_concrete(_pick_scenario(h)))
    a, done_a = _rollout(h, action_seed=123, max_steps=max_steps)
    b, done_b = _rollout(h, action_seed=123, max_steps=max_steps)
    if len(a) != len(b):
        return CheckResult("behavior.determinism", "FAIL",
                           f"episode length differs: {len(a)} vs {len(b)}")
    for t, (ra, rb) in enumerate(zip(a, b)):
        if not _eq(ra, rb):
            return CheckResult("behavior.determinism", "FAIL",
                               f"divergence at step {t} under identical seed+actions")
    guard = "" if done_a and done_b else " (hit step cap, not terminated)"
    return CheckResult("behavior.determinism", "PASS",
                       f"identical over {len(a)-1} steps{guard}")


def _episode_trace(env, seed: int | None, action_seed: int, max_steps: int) -> list:
    """One episode as (obs, reward) per step, under a reproducible action stream.

    Reward rides along with the observation because a censored domain may not
    show its exogenous draw in the obs at all (a bandit shows only the arm it
    pulled) — the payoff still moves with the draw, so including it keeps the
    trace a faithful signature of the episode without any domain knowledge.
    """
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(action_seed)
    trace = [(obs, 0.0)]
    done, steps = False, 0
    while not done and steps < max_steps:
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        trace.append((obs, float(reward)))
        done = bool(terminated or truncated)
        steps += 1
    return trace


def _traces_eq(a: list, b: list) -> bool:
    return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))


def check_gym_reseed(h: DomainHandle) -> CheckResult:
    """Unseeded ``reset()`` must draw a FRESH episode seed (spec §7).

    The mandated pattern's ``else`` branch is what makes that true. Omitting it
    leaves ``_episode_seed`` untouched on the auto-reset path SB3 takes, so every
    training episode after the first replays one exogenous path. The deviation is
    silent in both directions that usually catch things: training only degrades,
    and evaluation is unaffected because §9 seeds every episode explicitly — so
    the leaderboard stays a valid measurement while the training distribution is
    wrong. ``behavior.determinism`` is the mirror property (same explicit seed ⇒
    identical) and passes happily alongside the bug.
    """
    max_steps = _max_steps(_concrete(_pick_scenario(h)))
    env = _build_env(h)
    # Is the episode seed even legible in the trace? Two different EXPLICIT
    # seeds must part ways before "unseeded resets replay" means anything. This
    # asks the env rather than introspecting generators, so a state-conditioned
    # or censored domain is judged on the path that actually runs.
    if _traces_eq(_episode_trace(env, 1, 321, max_steps),
                  _episode_trace(env, 2, 321, max_steps)):
        return CheckResult("behavior.gym_reseed", "SKIP",
                           "episode seed does not move the (obs, reward) trace: "
                           "deterministic dynamics, or the draw is invisible "
                           "from outside — nothing to assert")
    env.reset(seed=0)  # seed np_random first, as SB3 does at training start
    traces = [_episode_trace(env, None, action_seed=321, max_steps=max_steps)
              for _ in range(3)]
    if all(_traces_eq(t, traces[0]) for t in traces[1:]):
        return CheckResult("behavior.gym_reseed", "FAIL",
                           "3 unseeded reset()s replayed one episode — reset() "
                           "must draw a fresh _episode_seed from self.np_random "
                           "when seed is None (spec §7); training would run on "
                           "n_envs fixed exogenous paths")
    # the converse half of the contract: an explicit seed pins the episode
    a = _episode_trace(env, 7, action_seed=321, max_steps=max_steps)
    b = _episode_trace(env, 7, action_seed=321, max_steps=max_steps)
    if not _traces_eq(a, b):
        return CheckResult("behavior.gym_reseed", "FAIL",
                           "reset(seed=7) twice under identical actions "
                           "diverged — an explicit seed must pin the episode")
    return CheckResult("behavior.gym_reseed", "PASS",
                       "unseeded resets draw fresh episodes; explicit seed pins one")


def check_purity(h: DomainHandle) -> CheckResult:
    env = _build_env(h)
    env.reset(seed=0)
    env.action_space.seed(5)
    state_obj = _find_state_obj(env, h.state_cls)
    if state_obj is None:
        return CheckResult("behavior.purity", "SKIP", "could not locate state object on env")
    before = copy.deepcopy(state_obj)
    env.step(env.action_space.sample())
    if not _eq(state_obj, before):
        return CheckResult("behavior.purity", "FAIL",
                           "advance() mutated the input state in place")
    return CheckResult("behavior.purity", "PASS", "transition did not mutate input state")


def check_rng_generators(h: DomainHandle) -> CheckResult:
    scenario = _concrete(_pick_scenario(h))
    generators = _discover_generators(scenario)
    if not generators:
        return CheckResult("rng.generators", "SKIP", "no generators found on scenario")
    salt = getattr(scenario, "seed_salt", 0)
    exogenous, state_cond, failures, episode_keyed = [], [], [], []
    for label, gen in generators:
        ctx = _MinimalCtx(salt)
        try:
            draws = [gen.sample(ctx) for _ in range(8)]
        except Exception:
            state_cond.append(label)
            continue
        # same (period, seed, salt) must always yield the same value; comparing a
        # single pair can coincide by chance for low-entropy draws, so require all
        # repeated draws to match.
        if all(_eq(d, draws[0]) for d in draws[1:]):
            exogenous.append(label)
        else:
            failures.append(label)
            continue
        # boundary probe (spec §4.3): a generator whose draw varies with
        # episode_seed but never with period is an episode-level draw —
        # scenario sampling in disguise. Constant generators (sensitive to
        # neither, e.g. a fixed leadtime) are legitimate and not flagged.
        try:
            base = draws[0]
            period_sensitive = any(
                not _eq(gen.sample(_MinimalCtx(salt, period=p)), base)
                for p in (1, 2, 3, 5, 11)
            )
            seed_sensitive = any(
                not _eq(gen.sample(_MinimalCtx(salt, episode_seed=s)), base)
                for s in (1, 2, 3, 5, 11)
            )
            if seed_sensitive and not period_sensitive:
                episode_keyed.append(label)
        except Exception:
            pass
    if failures:
        return CheckResult("rng.generators", "FAIL",
                           f"non-deterministic given (period,seed,salt): {failures}")
    if episode_keyed:
        return CheckResult("rng.generators", "WARN",
                           f"episode-keyed (vary with episode_seed but not period): "
                           f"{episode_keyed} — likely scenario sampling in disguise; "
                           "model as a ScenarioSampler (spec §4.3)")
    detail = f"exogenous+deterministic: {exogenous}"
    if state_cond:
        detail += f"; state-conditioned (path-indep not asserted): {state_cond}"
    return CheckResult("rng.generators", "PASS", detail)


# ---------------------------------------------------------------------------
# Scenario-architecture checks (spec §5, §6.3)
#
# Scheme-aware: a domain declares SEED_SCHEME = "v1" | "v2" in its scenarios
# (or uncertainty) module. Undeclared domains are treated as v1 — frozen
# legacy keys, strict template checks skipped — so pre-redesign domains keep
# passing until they migrate. v2 opts into the strict checks.
# ---------------------------------------------------------------------------

def _declared_scheme(h: DomainHandle) -> str | None:
    """The domain's declared SEED_SCHEME, or None. Raises on mixed declarations."""
    found = {
        role: getattr(m, "SEED_SCHEME")
        for role, m in h.modules.items()
        if hasattr(m, "SEED_SCHEME")
    }
    values = set(found.values())
    if len(values) > 1:
        raise ValueError(f"mixed SEED_SCHEME declarations: {found}")
    return values.pop() if values else None


def check_seed_scheme(h: DomainHandle) -> CheckResult:
    try:
        scheme = _declared_scheme(h)
    except ValueError as e:
        return CheckResult("scheme.declared", "FAIL", str(e))
    if scheme is None:
        return CheckResult("scheme.declared", "WARN",
                           "no SEED_SCHEME declared — treated as v1 (frozen legacy "
                           "keys); new domains must declare SEED_SCHEME = 'v2' "
                           "(spec §6.3)")
    if scheme not in ("v1", "v2"):
        return CheckResult("scheme.declared", "FAIL",
                           f"SEED_SCHEME must be 'v1' or 'v2', got {scheme!r}")
    return CheckResult("scheme.declared", "PASS", f"SEED_SCHEME = {scheme!r}")


def check_sampler_registry(h: DomainHandle) -> CheckResult:
    """Registry smoke + purity (spec §5.2): every sampler entry in SCENARIOS is
    a pure function — same seed twice -> equal concrete scenarios."""
    samplers = [(k, v) for k, v in h.SCENARIOS.items() if callable(v)]
    if not samplers:
        return CheckResult("scenario.samplers", "SKIP", "no sampler entries in SCENARIOS")
    problems = []
    for key, sampler in samplers:
        for seed in (0, 1, 42):
            try:
                a, b = sampler(seed), sampler(seed)
            except Exception as e:
                problems.append(f"{key}(seed={seed}) raised {type(e).__name__}: {e}")
                break
            if callable(a):
                problems.append(f"{key}(seed={seed}) did not resolve to a concrete scenario")
                break
            if not _eq(a, b):
                problems.append(f"{key}(seed={seed}) impure: two calls differ")
                break
    if problems:
        return CheckResult("scenario.samplers", "FAIL", "; ".join(problems[:5]))
    return CheckResult("scenario.samplers", "PASS",
                       f"{len(samplers)} sampler(s) pure and concrete over seeds (0, 1, 42)")


def check_meta_v2(h: DomainHandle) -> CheckResult:
    """v2 template (spec §6.3): composition-scoped meta drawers (mixtures)
    carry distinct substream_id; every seed_salt >= 1; generator instances
    carry distinct source_id. A callable source without a substream_id is the
    generator-owned-latent convention (catalog model): its meta stream is the
    latent generator's own source_id, covered by the source_id checks below."""
    if _declared_scheme(h) != "v2":
        return CheckResult("scheme.v2_ids", "SKIP", "v1/undeclared domain (frozen keys)")
    problems = []
    substreams: dict[str, int] = {}
    for key, value in h.SCENARIOS.items():
        salt = getattr(value, "seed_salt", None)
        if salt is not None and salt < 1:
            problems.append(f"{key}: seed_salt={salt} < 1")
        if callable(value):
            sub = getattr(value, "substream_id", None)
            if sub is not None:
                substreams[key] = sub
    dupes = {}
    for key, sub in substreams.items():
        dupes.setdefault(sub, []).append(key)
    for sub, keys in dupes.items():
        if len(keys) > 1:
            problems.append(f"substream_id={sub} shared by {keys}")
    for key, value in h.SCENARIOS.items():
        try:
            sc = _concrete(value)
        except Exception:
            continue  # construction failures are check_scenarios_valid's job
        gens = _discover_generators(sc)
        ids = {}
        for label, gen in gens:
            sid = getattr(gen, "source_id", None)
            if sid is None:
                problems.append(f"{key}: generator {label} without source_id")
            else:
                ids.setdefault(sid, []).append(label)
        for sid, labels in ids.items():
            if len(labels) > 1:
                problems.append(f"{key}: source_id={sid} shared by {labels}")
    if problems:
        return CheckResult("scheme.v2_ids", "FAIL", "; ".join(problems[:6]))
    return CheckResult("scheme.v2_ids", "PASS",
                       "salts >= 1; substream/source ids present and distinct")


def check_seed_key_helpers(h: DomainHandle) -> CheckResult:
    """v2 drift guard (spec §4.2, §5.2): every SeedSequence(...) in the world
    layers is either inside intrinsic_key()/meta_key() or takes a key built by
    one of them — raw inline keys cannot silently diverge from the template."""
    if _declared_scheme(h) != "v2":
        return CheckResult("scheme.v2_keys", "SKIP", "v1/undeclared domain (frozen keys)")
    helper_names = {"intrinsic_key", "meta_key"}

    def call_name(func) -> str:
        return getattr(func, "attr", None) or getattr(func, "id", "") or ""

    strays = []
    for role in ("uncertainty", "scenarios", "grids"):
        tree = h.trees.get(role)
        if tree is None:
            continue

        def visit(node, fn_stack):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_stack = fn_stack + [node.name]
            if isinstance(node, ast.Call) and call_name(node.func) == "SeedSequence":
                inside_helper = any(n in helper_names for n in fn_stack)
                arg_ok = (
                    len(node.args) == 1
                    and isinstance(node.args[0], ast.Call)
                    and call_name(node.args[0].func) in helper_names
                )
                if not (inside_helper or arg_ok):
                    strays.append(f"{role}:{node.lineno}")
            for child in ast.iter_child_nodes(node):
                visit(child, fn_stack)

        visit(tree, [])
    if strays:
        return CheckResult("scheme.v2_keys", "FAIL",
                           f"raw SeedSequence outside intrinsic_key()/meta_key(): {strays}")
    return CheckResult("scheme.v2_keys", "PASS",
                       "all SeedSequence calls go through the key helpers")


def check_grids(h: DomainHandle) -> list[CheckResult]:
    """Design layer (spec §5.6): grids stay out of SCENARIOS; GRIDS entries are
    enumerable, non-callable, uniquely-celled, and derive pure samplers."""
    results = []
    grid_like = [k for k, v in h.SCENARIOS.items()
                 if not callable(v) and hasattr(v, "cells")]
    if grid_like:
        results.append(CheckResult("grids.not_in_scenarios", "FAIL",
                                   f"grid object(s) in SCENARIOS: {grid_like}"))
    if "grids" not in h.modules:
        results.append(CheckResult("grids.registry", "SKIP", "no {domain}_grids.py"))
        return results
    GRIDS = getattr(h.modules["grids"], "GRIDS", None)
    if not isinstance(GRIDS, dict) or not GRIDS:
        results.append(CheckResult("grids.registry", "FAIL", "GRIDS missing or empty"))
        return results
    problems = []
    for key, grid in GRIDS.items():
        if callable(grid):
            problems.append(f"{key}: grid is callable (must never pass as a ScenarioSource)")
            continue
        try:
            cells = list(grid)
            ids = [cid for cid, _ in cells]
        except Exception as e:
            problems.append(f"{key}: not enumerable ({type(e).__name__}: {e})")
            continue
        if not cells:
            problems.append(f"{key}: no cells")
            continue
        if len(ids) != len(set(ids)):
            problems.append(f"{key}: duplicate cell ids")
        try:
            sampler = grid.as_sampler()
            a, b = sampler(0), sampler(0)
            if callable(a) or not _eq(a, b):
                problems.append(f"{key}: as_sampler() not pure/concrete")
        except Exception as e:
            problems.append(f"{key}: as_sampler failed ({type(e).__name__}: {e})")
    if problems:
        results.append(CheckResult("grids.registry", "FAIL", "; ".join(problems[:5])))
    else:
        total = sum(len(list(g)) for g in GRIDS.values())
        results.append(CheckResult("grids.registry", "PASS",
                                   f"{len(GRIDS)} grid(s), {total} cells; samplers pure"))
    results.append(_check_grid_axes(h, GRIDS))
    return results


def _axis_tiers(ir) -> dict[str, str]:
    """Constant name -> tier label, read off the IR's own declarations
    (spec §5.6 "Choosing axes"). Assignment order makes tier-1 win when a
    constant plays several roles — the action space is the harder break."""
    tiers: dict[str, str] = {}
    m = ir.mdp
    for sv in m.state_variables:
        if isinstance(sv.length, str):
            tiers[sv.length] = f"tier-2 obs-dim (sets length of state {sv.name!r})"
        for raw in (sv.bounds, sv.element_bounds):
            for x in raw or []:
                if isinstance(x, str):
                    tiers.setdefault(
                        x, f"tier-2 obs-dim (sets bounds of state {sv.name!r})")
    if isinstance(m.horizon.T, str):
        tiers[m.horizon.T] = "tier-2 horizon"
    for d in m.decisions:
        for x in d.bounds.value:
            if isinstance(x, str):
                tiers[x] = f"tier-1 (sets action bounds of decision {d.name!r})"
    for am in ir.gym.action_modes:
        for pair in am.bounds_per_decision():
            for x in pair:
                if isinstance(x, str):
                    tiers[x] = f"tier-1 (sets action bounds of mode {am.name!r})"
    return tiers


def check_benchmarks(h: DomainHandle) -> CheckResult:
    """Declared benchmarks and benchmark files must be the same set (spec §9.9).

    The declaration is what carries `role`, and a role nobody can trace to a
    file is a second source of truth: a renamed method, a deleted solver, or an
    arm quoted from an older run are all invisible without this check.
    """
    schemas = sorted(h.directory.glob("*_schema.json"))
    if len(schemas) != 1:
        return CheckResult("benchmarks.declared", "SKIP",
                           "no single *_schema.json to read declarations from")
    try:
        from mdp_ir.schema import load_ir
        ir = load_ir(schemas[0])
    except Exception as e:
        return CheckResult("benchmarks.declared", "SKIP",
                           f"schema not loadable ({type(e).__name__}: {e})")
    declared = {b.name: b for b in getattr(ir, "benchmarks", [])}
    prefix = f"{h.name}_benchmark_"
    # `{domain}_benchmark_common.py` is the shared eval-loop helper several
    # domains factor out (mab, adi_flex), not a solver — it names no method
    on_disk = {
        p.name[len(prefix):-len(".py")]
        for p in h.directory.glob(f"{prefix}*.py")
        if not p.name.endswith("_eval.py") and p.name != f"{prefix}common.py"
    }
    if not declared:
        detail = "IR declares no benchmarks"
        if on_disk:
            detail += f"; {len(on_disk)} benchmark file(s) present: {sorted(on_disk)}"
        return CheckResult("benchmarks.declared", "SKIP", detail)
    # a column-sourced entry declares an arm that is emitted by the eval rather
    # than run as a solver (a clairvoyant/hindsight bound reading the same
    # realized latents) — it has no file by construction, and exempting it is
    # what keeps the bracket declarable on such domains
    from_column = {n for n, b in declared.items() if getattr(b, "column", None)}
    missing_file = sorted(declared.keys() - on_disk - from_column)
    undeclared = sorted(on_disk - declared.keys())
    problems = []
    if missing_file:
        problems.append(f"declared with no {prefix}*.py: {missing_file}")
    if undeclared:
        problems.append(f"on disk but undeclared: {undeclared}")
    if problems:
        return CheckResult("benchmarks.declared", "FAIL", "; ".join(problems))
    roles = ", ".join(
        f"{n}={b.role.value}" + (f"@{b.column}" if getattr(b, "column", None) else "")
        for n, b in sorted(declared.items()))
    detail = f"{len(declared)} declared, all matched: {roles}"
    if from_column:
        detail += f" ({len(from_column)} column-sourced, no file expected)"
    return CheckResult("benchmarks.declared", "PASS", detail)


def check_research_questions(h: DomainHandle) -> CheckResult:
    """A declared confirm/discover stance owes the §14 artifacts (spec §1, §14).

    The probe's obligation used to be a fact about the *problem* ("required
    when a reference policy or predicted structural class exists"), which
    obliged an artifact a bypass campaign has no use for and named nothing as a
    confirm campaign's own deliverable. Tied to the declared stance it becomes
    checkable, and narrower: a bypass-only campaign owes neither file.
    """
    schemas = sorted(h.directory.glob("*_schema.json"))
    if len(schemas) != 1:
        return CheckResult("research.deliverables", "SKIP",
                           "no single *_schema.json to read declarations from")
    try:
        from mdp_ir.schema import load_ir
        ir = load_ir(schemas[0])
    except Exception as e:
        return CheckResult("research.deliverables", "SKIP",
                           f"schema not loadable ({type(e).__name__}: {e})")
    rq = getattr(ir, "research_questions", None)
    if rq is None or not rq.tier2:
        return CheckResult("research.deliverables", "SKIP",
                           "IR declares no tier-2 research question")
    owing = [q for q in rq.tier2 if q.probe_required]
    stances = ", ".join(f"{q.stance.value}({q.structure})" for q in rq.tier2)
    if not owing:
        return CheckResult("research.deliverables", "PASS",
                           f"{stances} — bypass only, no §14 artifact owed")
    missing = []
    if not (h.directory / f"{h.name}_policy_probe.py").exists():
        missing.append(f"{h.name}_policy_probe.py")
    if not (h.directory / "INTERPRET.md").exists():
        missing.append("INTERPRET.md")
    if missing:
        return CheckResult("research.deliverables", "FAIL",
                           f"{stances} owes the §14 readback; missing: "
                           f"{', '.join(missing)}")
    return CheckResult("research.deliverables", "PASS",
                       f"{stances} — probe and INTERPRET.md present")


def _check_grid_axes(h: DomainHandle, GRIDS: dict) -> CheckResult:
    """Classify each grid axis by what it breaks (spec §5.6): tier 1 changes
    the action space (one policy cannot emit two); tier-2 horizon changes no
    space's shape but silently re-scales horizon-dependent HPs and cell
    rewards; tier-2 obs-dim needs padding to the family maximum. Tier 3
    (reward/uncertainty constants) is what a generality target should vary."""
    schemas = sorted(h.directory.glob("*_schema.json"))
    if len(schemas) != 1:
        return CheckResult("grids.axes", "SKIP", "no single *_schema.json to classify against")
    try:
        from mdp_ir.schema import load_ir
        ir = load_ir(schemas[0])
    except Exception as e:
        return CheckResult("grids.axes", "SKIP",
                           f"schema not loadable ({type(e).__name__}: {e})")
    tiers = _axis_tiers(ir)
    warns, classified = [], 0
    for key, grid in GRIDS.items():
        axes = getattr(grid, "axes", None)
        if not isinstance(axes, dict):
            continue
        for name, values in axes.items():
            classified += 1
            tier = tiers.get(name)
            if tier is None:
                continue  # tier 3: reward/uncertainty constant — free
            if tier == "tier-2 horizon":
                nums = [v for v in values if isinstance(v, (int, float))]
                span = (f"{max(nums) / min(nums):.0f}x ({min(nums)}->{max(nums)})"
                        if nums and min(nums) > 0 else "?")
                warns.append(
                    f"{key}: axis {name!r} is the horizon (span {span}) — "
                    f"horizon-dependent HPs (gae_lambda, rollout composition) "
                    f"change meaning across cells, re-derive per cell; cell "
                    f"rewards are scale-weighted, report per cell (§9.6), "
                    f"never one aggregate")
            else:
                warns.append(f"{key}: axis {name!r} is {tier} — see §5.6 "
                             f"Choosing axes for the obligations")
    if warns:
        return CheckResult("grids.axes", "WARN", "; ".join(warns[:4]))
    if classified:
        return CheckResult("grids.axes", "PASS",
                           f"{classified} axis(es), all tier-3 (reward/uncertainty)")
    return CheckResult("grids.axes", "SKIP", "no grid declares axes metadata")


REGISTRY = [
    check_file_layout,
    check_layering,
    check_no_param,
    check_mdp_no_reward,
    check_state_slots,
    check_seed_scheme,
    check_scenarios_valid,
    check_sampler_registry,
    check_meta_v2,
    check_seed_key_helpers,
    check_grids,
    check_benchmarks,
    check_research_questions,
    check_init_state,
    check_gym_contract,
    check_determinism,
    check_gym_reseed,
    check_purity,
    check_rng_generators,
]

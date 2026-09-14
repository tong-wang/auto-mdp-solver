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
import inspect
import math
import re
from pathlib import Path

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


def _bound_constants(entry, constants: set[str]) -> list[str]:
    """The scenario constants a bounds entry references: the entry itself when
    it names one, else the constants its expression reads. A derived envelope
    is a width site like any other, so the constants inside it have to reach
    the boundary machinery — otherwise writing `40 * cap` instead of `cap`
    makes the cap invisible to coverage (#53)."""
    from mdp_ir.schema import _root_identifiers

    if not isinstance(entry, str):
        return []
    if entry in constants:
        return [entry]
    return sorted(_root_identifiers(entry) & constants)


def _axis_tiers(ir) -> dict[str, str]:
    """Constant name -> tier label, read off the IR's own declarations
    (spec §5.6 "Choosing axes"). Assignment order makes tier-1 win when a
    constant plays several roles — the action space is the harder break."""
    tiers: dict[str, str] = {}
    m = ir.mdp
    constants = {c.name for c in m.scenario.constants}
    for sv in m.state_variables:
        for axis_pos, part in enumerate(
                sv.length if isinstance(sv.length, list) else [sv.length]):
            if isinstance(part, str):
                where = (f"sets length of state {sv.name!r}"
                         if not isinstance(sv.length, list)
                         else f"sets axis {axis_pos} of state {sv.name!r}")
                tiers[part] = f"tier-2 obs-dim ({where})"
        for raw in (sv.bounds, sv.element_bounds):
            for x in raw or []:
                for c in _bound_constants(x, constants):
                    tiers.setdefault(
                        c, f"tier-2 obs-dim (sets bounds of state {sv.name!r})")
    if isinstance(m.horizon.T, str):
        tiers[m.horizon.T] = "tier-2 horizon"
    for d in m.decisions:
        if isinstance(d.dim, str):
            tiers[d.dim] = f"tier-1 (sets dim of decision {d.name!r})"
        for x in d.bounds.value:
            for c in _bound_constants(x, constants):
                tiers[c] = f"tier-1 (sets action bounds of decision {d.name!r})"
    for am in ir.gym.action_modes:
        for pair in am.bounds_per_decision():
            for x in pair:
                for c in _bound_constants(x, constants):
                    tiers[c] = f"tier-1 (sets action bounds of mode {am.name!r})"
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


_PROVENANCE_KEYS = ("algo_class", "sb3_version", "ir_mdp_fingerprint")

# resolved SB3 class -> the IR's Algo value. The spec records the class the run
# actually constructed, so this map is the whole translation a generic check
# needs — no domain knowledge about which flag selects what.
_CLASS_TO_ALGO = {
    "PPO": "ppo",
    "MaskablePPO": "maskable_ppo",
    "RecurrentPPO": "recurrent_ppo",
}


def _args_logs(h: DomainHandle) -> list[Path]:
    results = h.directory / "results"
    return sorted(results.rglob("*_args.txt")) if results.is_dir() else []


def _parse_args_log(path: Path) -> dict[str, str]:
    # one reader, in the module the train script imports at launch — the check
    # and the run must agree about what the log says
    from .launch import read_args_log
    return read_args_log(path)


_LAYER_LEAK = ("benchmark", "solver", "tractab", "gridded", "dp reference")


def _domain_bounds(spec: str) -> tuple[float | None, float | None, bool]:
    """(low, high, integral) parsed from a model domain string, best effort.

    Handles the forms the spec's examples use — "integer >= 1", "real, [0, inf)",
    "integer >= 2" — and returns Nones for anything it cannot read, which the
    caller treats as un-checkable rather than as a failure.
    """
    text = spec.lower()
    integral = "integer" in text or "int " in text
    low = high = None
    m = re.search(r">=\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        low = float(m.group(1))
    m = re.search(r"<=\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        high = float(m.group(1))
    m = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(inf|-?\d+(?:\.\d+)?)\s*[\])]", text)
    if m:
        low = float(m.group(1)) if low is None else low
        if high is None and m.group(2) != "inf":
            high = float(m.group(2))
    return low, high, integral



_CAP_SUFFIXES = ("_cap", "_max", "_limit")


def _capped_quantity(cap_name: str, constants: dict, quantities: dict) -> str | None:
    """The quantity a cap constant caps, or None if the pairing is not stated.

    A cap is named for what it bounds — `leadtime_max` caps `leadtime` — so the
    suffix is the only link the IR carries today. No match means no claim: the
    check reports the width as unpaired rather than inventing a comparison.
    """
    for suffix in _CAP_SUFFIXES:
        if cap_name.endswith(suffix):
            base = cap_name[: -len(suffix)]
            if base in constants or base in quantities:
                return base
    return None


def check_model_boundary(h: DomainHandle) -> CheckResult:
    """The model layer says theory; widths are named constants (spec §5.0, #29).

    A rendering width is not a separate declaration — it is a scenario constant
    named at the width site, so what it caps and what it renders are *derived*
    from the reference (`_axis_tiers`, already used by the grid check). What the
    IR must add is the theory it was previously silent about, and the discipline
    that keeps the two apart:

    * a **literal** at a width site, for a quantity the model declares, is the
      defect this section exists for — that is how a sweep maximum becomes a
      capacity limit nobody chose;
    * a width constant must **cover every designed value**, or the study
      outruns its own rendering;
    * every declared instance must sit inside the declared domain;
    * model-layer prose must not argue from a benchmark or from tractability.
    """
    schemas = sorted(h.directory.glob("*_schema.json"))
    if len(schemas) != 1:
        return CheckResult("model.boundary", "SKIP", "no single *_schema.json")
    try:
        from mdp_ir.schema import load_ir
        ir = load_ir(schemas[0])
    except Exception as e:
        return CheckResult("model.boundary", "SKIP",
                           f"schema not loadable ({type(e).__name__}: {e})")
    model = ir.mdp.model
    if model is None:
        return CheckResult("model.boundary", "SKIP",
                           "IR declares no mdp.model — the theory layer is "
                           "undeclared, so nothing separates it from the rendering")
    problems: list[str] = []
    constants = {c.name: c for c in ir.mdp.scenario.constants}
    instances = getattr(ir.mdp.scenario, "instances", {}) or {}

    def designed(name: str) -> list[float]:
        vals = [c.value for n, c in constants.items()
                if n == name and isinstance(c.value, (int, float))]
        for over in instances.values():
            v = (over or {}).get(name)
            if isinstance(v, (int, float)):
                vals.append(v)
        return vals

    # gate 1 — the statement speaks in quantified rules, not rendered names
    rendered = {v.name for v in ir.mdp.state_variables}
    statement = " ".join(model.dynamics + model.out_of_scope
                         + [q.domain for q in model.quantities.values()])
    named = sorted(n for n in rendered if re.search(rf"\b{re.escape(n)}\d+\b", statement))
    if named:
        problems.append(f"model statement names rendered slots {named}")

    # gate 2 — a width site holding a LITERAL, for a quantity the theory
    # declares, is the sweep-frozen-as-capacity defect at its source
    widths = _axis_tiers(ir)                       # constant -> what it renders
    for sv in ir.mdp.state_variables:
        parts = sv.length if isinstance(sv.length, list) else [sv.length]
        if any(isinstance(part, int) for part in parts):
            literal = next(part for part in parts if isinstance(part, int))
            for qname in model.quantities:
                if qname in sv.name or sv.name in qname:
                    problems.append(
                        f"state {sv.name!r} has a literal length {literal} while the "
                        f"model declares {qname!r} ({model.quantities[qname].domain!r}) — "
                        f"name a scenario constant so the width is a design choice, "
                        f"not a capacity the model appears to state")

    # gate 3 — a genuine CAP must cover the quantity it caps.
    #
    # The two shapes a width site can have are already distinguished by the IR:
    # an `axis`-tagged constant IS the design value, resolved per instance, so
    # "outrun" is impossible by construction — comparing it against its own
    # overrides fails the very pattern §5.0 recommends (#30). An untagged
    # constant is a cap, and the thing it caps is a DIFFERENT constant.
    unpaired = []
    for cname, role in widths.items():
        const = constants.get(cname)
        if const is None:
            continue
        if getattr(const, "axis", ""):
            continue                     # the design value itself; nothing to outrun
        capped = _capped_quantity(cname, constants, model.quantities)
        if capped is None:
            unpaired.append(cname)
            continue
        vals = designed(capped)
        cap = const.value
        if isinstance(cap, (int, float)) and vals and max(vals) > cap:
            problems.append(
                f"cap {cname}={cap} ({role}) is outrun by a designed "
                f"{capped}={max(vals)} — the study outruns its own rendering")

    # gate 4 — the stochastic structure the theory states must be the one the
    # IR renders. Read at the CATALOG level: a slot whose selected candidate is
    # a point mass is still a slot, so choosing inv_single's `deterministic`
    # leadtime is a design choice, never a scope reduction
    import json as _json
    raw = _json.loads(schemas[0].read_text())
    from mdp_ir.layering import _flat_mdp
    flat = _flat_mdp(raw)
    slot_names = {s["name"] for s in
                  (flat.get("uncertainty_slots") or flat.get("uncertainty_sources") or [])
                  if isinstance(s, dict) and "name" in s}
    for qname, q in model.quantities.items():
        if q.stochastic is True and qname not in slot_names:
            problems.append(
                f"{qname!r} is declared stochastic by the model but is no "
                f"uncertainty source — the rendering dropped randomness the "
                f"theory states")
        if q.stochastic is False and qname in slot_names:
            problems.append(
                f"{qname!r} is declared deterministic by the model but is an "
                f"uncertainty source — either the source is out of scope or the "
                f"model statement is wrong")

    # gate 5 — declared instances sit inside the declared domains
    for qname, q in model.quantities.items():
        low, high, integral = _domain_bounds(q.domain)
        for val in designed(qname):
            if low is not None and val < low:
                problems.append(f"{qname}={val} is below its declared domain {q.domain!r}")
            if high is not None and val > high:
                problems.append(f"{qname}={val} is above its declared domain {q.domain!r}")
            if integral and float(val) != int(val):
                problems.append(f"{qname}={val} is not integral, domain {q.domain!r}")

    leaks = [w for w in _LAYER_LEAK if w in statement.lower()]
    if problems:
        return CheckResult("model.boundary", "FAIL", "; ".join(problems[:4]))
    if leaks:
        return CheckResult("model.boundary", "WARN",
                           f"model-layer prose argues from {leaks} — a benchmark or "
                           f"tractability rationale inside the theory layer is a "
                           f"design choice filed as model")
    inventory = ", ".join(f"{n} ({r.split('(')[0].strip()})" for n, r in sorted(widths.items()))
    detail = (f"{len(model.quantities)} quantities declared; widths derived "
              f"from their references [{inventory or 'none'}]")
    if unpaired:
        detail += (f"; unpaired caps {sorted(unpaired)} — named for no declared "
                   f"quantity, so coverage is unchecked")
    return CheckResult("model.boundary", "PASS", detail)


def check_run_provenance(h: DomainHandle) -> CheckResult:
    """Run directories record what produced them (spec §8.4).

    The args log was already the declared home for run provenance — §8.6 asks it
    to carry the SB3 version — but nothing said what it must contain, so its
    contents were whatever flags a domain happened to expose. That is why the
    trained artifact's class could only be *derived*, and only with domain
    knowledge (`mask: True` names the class to someone who knows the domain).
    Recording `algo_class` is what makes the artifact-vs-declaration check
    possible at all.
    """
    logs = _args_logs(h)
    if not logs:
        return CheckResult("run.provenance", "SKIP", "no run directory to read")
    declared: set[str] = set()
    schemas = sorted(h.directory.glob("*_schema.json"))
    if len(schemas) == 1:
        try:
            from mdp_ir.schema import load_ir
            ir = load_ir(schemas[0])
            declared = {a.value for a in ir.rl.declared_algos()}
        except Exception:
            declared = set()
    legacy, partial, wrong_class = [], [], []
    for log in logs:
        rec = _parse_args_log(log)
        # `algo_class` is the adoption marker — the one key only the §8.4
        # convention writes. Keying "legacy" on *any* required key present
        # would punish the domains that honoured §8.6's older sb3_version line
        # and excuse the ones that ignored it, which is exactly backwards.
        if "algo_class" not in rec:
            legacy.append(log.name)
            continue
        missing = [k for k in _PROVENANCE_KEYS if k not in rec]
        if missing:
            partial.append(f"{log.parent.name}/{log.name}: missing {missing}")
        cls = rec.get("algo_class")
        if cls and declared:
            algo = _CLASS_TO_ALGO.get(cls)
            if algo is None or algo not in declared:
                wrong_class.append(
                    f"{log.parent.name}: algo_class={cls!r} not in declared "
                    f"{sorted(declared)}")
    if partial or wrong_class:
        return CheckResult("run.provenance", "FAIL",
                           "; ".join((partial + wrong_class)[:4]))
    if legacy and len(legacy) == len(logs):
        return CheckResult("run.provenance", "WARN",
                           f"{len(legacy)} run(s) predate the §8.4 provenance set "
                           f"(no {list(_PROVENANCE_KEYS)}); re-runs will carry it")
    detail = f"{len(logs) - len(legacy)} run(s) carry the full provenance set"
    if legacy:
        detail += f"; {len(legacy)} pre-convention"
    if declared:
        detail += f"; algo_class within declared {sorted(declared)}"
    return CheckResult("run.provenance", "PASS", detail)


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


_PROSE_KEYS = {"desc", "source", "rationale", "basis", "notation", "out_of_scope",
               "domain", "claim", "instrument", "structure", "narrowed",
               "assumptions_log"}
_ORDINALS = ("now", "next", "later", "first", "second", "third", "cur", "prev", "last")


def _referenced_tokens(doc: dict, mdp: dict) -> set[str]:
    """Every identifier the *rendering* actually reads.

    Prose fields are skipped: naming a constant in its own `desc` is not a
    reference, and counting it would let a decorative constant hide behind its
    documentation. The `model` block is skipped entirely — it is the theory,
    written in the theory's own vocabulary (spec §5.0).

    `gym` and `rl` are in scope, and must be: a constant can be consumed
    outside the `mdp` block entirely. `fnv`'s `t_last` is a feature of the gym's
    observation vector — one policy trains across a whole design grid and has to
    condition on which cell it is in (spec §5.6) — so scanning only `mdp` called
    a load-bearing constant decorative. Nothing else in the document is scanned:
    `benchmarks` and `research_questions` are prose about solutions, not
    consumers of constants.
    """
    found: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k not in _PROSE_KEYS:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            found.append(node)

    scenario = mdp.get("scenario") or {}
    walk({k: v for k, v in mdp.items() if k not in ("model", "scenario")})
    walk({k: v for k, v in scenario.items() if k != "constants"})
    walk({k: v for k, v in doc.items() if k in ("gym", "rl")})
    toks = set()
    for s in found:
        toks |= set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", s))
    # an instance override and a sampler draw both name a constant
    for over in (scenario.get("instances") or {}).values():
        toks |= set(over)
    for sampler in scenario.get("samplers") or []:
        toks |= {d["name"] for d in sampler.get("draws", [])}
    return toks


def _enumeration_findings(doc: dict) -> tuple[list[str], int]:
    """The three shapes, on a whole IR document. Pure, so it is testable on
    synthetic IRs; returns (problems, constant count).

    Known under-report: the re-bake detector is numeric-**list**-only. `fnv`'s
    `order_max` is the same defect at scalar width (`mu + 5*stdev`, re-baked
    1.5 -> 7.389 in `mmmfe`) and is not reported, because most scalar instance
    overrides are genuine design inputs rather than derived values — `stdev:
    0.2` in that same instance is exactly one. Separating them needs the
    derivation to be declarable, which is the point of upstream #45.
    """
    from mdp_ir.schema import ungroup_mdp
    mdp = ungroup_mdp(copy.deepcopy(doc["mdp"]))
    scenario = mdp.get("scenario") or {}
    consts = {c["name"]: c for c in (scenario.get("constants") or [])}
    seen = _referenced_tokens(doc, mdp)
    problems: list[str] = []

    dead = [n for n in consts if n not in seen]
    if dead:
        problems.append(f"decorative constant(s) {sorted(dead)} — declared, "
                        f"referenced nowhere; the rendering hardcodes them instead")

    families: dict[str, list[str]] = {}
    for name in list(consts) + [s["name"] for s in mdp.get("state_variables") or []]:
        m = (re.match(r"^(.*?)_?(\d+)$", name)
             or re.match(rf"^(.*?)_({'|'.join(_ORDINALS)})$", name))
        if m and m.group(1):
            families.setdefault(m.group(1), []).append(name)
    enumerated = {k: sorted(v) for k, v in families.items() if len(v) > 1}
    if enumerated:
        problems.append("; ".join(
            f"enumerated {k!r}: {v} — declare one vector with a named length"
            for k, v in sorted(enumerated.items())))

    rebaked: list[str] = []
    for iname, over in (scenario.get("instances") or {}).items():
        for key, val in over.items():
            base = consts.get(key, {}).get("value")
            # numeric only: a list of strings is a categorical choice (an event
            # permutation), not a computed vector, and overriding it is correct
            if (isinstance(val, list) and isinstance(base, list)
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                            for x in val)):
                rebaked.append(f"{iname}.{key}")
    if rebaked:
        problems.append(f"re-baked numeric vector(s) {sorted(rebaked)} — a derived "
                        f"value restated per instance, verified by nothing")
    return problems, len(consts)


def _flat(doc: dict) -> dict:
    """The mdp block, grouped or flat — the same shape both readers see."""
    mdp = doc.get("mdp", doc)
    if isinstance(mdp.get("model"), dict) or isinstance(mdp.get("rendering"), dict):
        out = dict(mdp.get("rendering") or {})
        for k, v in mdp.items():
            if k not in ("model", "design", "rendering"):
                out.setdefault(k, v)
        return out
    return mdp


def _hardcode_findings(doc: dict) -> tuple[list[str], int]:
    """Non-identity literals at design-bearing value sites, and the site count.

    An **identity** is a number nothing could vary: `1.0` for a discount (the
    undiscounted problem) and `0` for a tolerance (no declared slack). Naming
    those buys no degree of freedom, which is the half of §5.0's rule that
    keeps it from becoming "name every numeral".

    A model-layer declaration **aggravates** a finding rather than making one.
    `examples/mab` declares `discount_factor` in its theory and renders it
    `1.0`: model-declared, literal, and entirely correct — an undiscounted
    bandit has no second discount to render.
    """
    findings: list[str] = []
    sites = 0
    flat = _flat(doc)
    declared = set(((doc.get("mdp", doc)).get("model") or {}).get("quantities") or {})

    beta = (flat.get("objective") or {}).get("discount_factor")
    if beta is not None:
        sites += 1
        if (isinstance(beta, (int, float)) and not isinstance(beta, bool)
                and float(beta) != 1.0):
            note = (" — and mdp.model declares the quantity, so the rendering "
                    "is stating a choice in the one place that reads as a model "
                    "claim" if "discount_factor" in declared else "")
            findings.append(
                f"objective.discount_factor is the literal {beta}{note}; name a "
                f"scenario constant so an instance can render this model at "
                f"another discount (§5.0)")

    for b in doc.get("benchmarks", []) or []:
        tol = b.get("tolerance")
        if tol is None:
            continue
        sites += 1
        if (isinstance(tol, (int, float)) and not isinstance(tol, bool)
                and float(tol) != 0.0):
            findings.append(
                f"benchmarks[{b.get('name')!r}].tolerance is the literal {tol}; "
                f"one entry may serve renderings whose numerical error differs, "
                f"so name a constant and resolve it per instance (§5.0/§9.9)")
    return findings, sites


def check_no_hardcode(h: DomainHandle) -> CheckResult:
    """The IR holds no bare design value (spec §5.0).

    §5.0's width-site rule generalizes: a value site that can hold a number an
    instance could legitimately need to differ on takes a symbol, because a
    literal there is both a design choice wearing a model claim's clothes *and*
    — the cost the width rule does not state — **immobile**. A value with no
    name cannot be overridden by an instance, so the second rendering can only
    be had by re-basing the first.

    The line is identity vs choice, not literal vs symbol: `0` and `1` stay
    literal (the undiscounted β, the no-slack tolerance), and so do bounds of
    zero, indices and stream ids, which this check does not look at.

    WARN, not FAIL, in the shape `schema.no_enumeration` established: the sites
    named here only became expressible as symbols in the release that added
    this check, so shipped domains are all mid-migration by construction. It
    promotes to FAIL once the example set carries no non-identity literal here.
    """
    schemas = sorted(h.directory.glob("*_schema.json"))
    if len(schemas) != 1:
        return CheckResult("schema.no_hardcode", "SKIP", "no single *_schema.json")
    try:
        import json
        doc = json.loads(schemas[0].read_text())
        findings, sites = _hardcode_findings(doc)
    except Exception as e:
        return CheckResult("schema.no_hardcode", "SKIP",
                           f"schema not readable ({type(e).__name__}: {e})")
    if findings:
        return CheckResult("schema.no_hardcode", "WARN", "; ".join(findings)[:400])
    return CheckResult("schema.no_hardcode", "PASS",
                       f"{sites} design-bearing value site(s) declared; each is "
                       f"a symbol or an identity")


def check_no_enumeration(h: DomainHandle) -> CheckResult:
    """The IR is a model definition, not a rendering transcript (spec §5.0).

    §5.0 already calls a literal at a *width site* a defect. The three shapes
    here are the same defect one level down — in what is being sized rather
    than in the size — and none of them is reachable by `model.boundary`, which
    SKIPs entirely when the theory layer is undeclared. They are also the first
    thing a reader sees, which is the other reason they matter.

    * **decorative constant** — declared and referenced nowhere. It names a
      quantity the rendering then hardcodes independently, so changing it
      changes nothing and the IR silently disagrees with itself.
    * **enumerated family** — one vector hand-unrolled into siblings
      (`due_now`/`due_next`, `pipe1`/`pipe2`). The width stops being a symbol,
      so nothing can sweep it and no width check can see it.
    * **re-baked vector** — a numeric list an instance restates. The value is
      derived, the derivation is not declarable (`ScenarioConstant.value` is a
      literal by design), so every instance re-computes it by hand and nothing
      verifies the arithmetic.

    WARN, not FAIL: the third shape has no declarable alternative today, so a
    hard failure would punish authors for a schema gap. Promote once that lands.
    """
    schemas = sorted(h.directory.glob("*_schema.json"))
    if len(schemas) != 1:
        return CheckResult("schema.no_enumeration", "SKIP", "no single *_schema.json")
    try:
        import json
        doc = json.loads(schemas[0].read_text())
        problems, n_consts = _enumeration_findings(doc)
    except Exception as e:
        return CheckResult("schema.no_enumeration", "SKIP",
                           f"schema not readable ({type(e).__name__}: {e})")

    if problems:
        return CheckResult("schema.no_enumeration", "WARN", "; ".join(problems)[:400])
    return CheckResult("schema.no_enumeration", "PASS",
                       f"{n_consts} constant(s) all referenced; no enumerated "
                       f"families; no re-baked vectors")


# vocabulary of a bound justified by MEASUREMENT rather than by the dynamics.
# Deliberately narrow: it flags the *observation of solved instances*, not the
# word "optimal" — "an order beyond the horizon's demand can never be optimal"
# is a statement about the model and must not trip this.
_MEASURED = (
    "observed", "we ran", "measured", "empirical", "in practice we",
    "training run", "trained", "tuned", "baseline", "benchmark",
    "dp table", "dp policy", "headroom over", "never bind", "must not bind",
)


def check_bound_rationale(h: DomainHandle) -> CheckResult:
    """An action bound is a claim about the dynamics, never a measurement of
    the instances already solved (spec §7).

    The tempting justification is the one the author can see: run the solver,
    take the largest action an optimal policy places, add headroom. It reads as
    diligence and produces a defensible-looking number. It is not a bound — it
    is a summary of a design set, and its failure mode is silent. A cap that
    binds does not raise; it truncates the action set and reports a worse
    policy, so the first instance with a longer horizon or a heavier rate
    inherits a cap justified against a study that no longer exists. §5.2's
    coverage rule cannot catch it either: that fires only for a cap NAMED over
    a declared design value, which an action scale generally is not.

    The question the rationale has to survive is one line: *would this
    justification still hold for an instance nobody has run?* Bounds
    themselves are checked elsewhere — this reads only the prose that says
    why. WARN, because the vocabulary is evidence and not proof; a rationale
    that cites a benchmark to *illustrate* a derivation is a false positive
    worth one look.
    """
    schemas = sorted(h.directory.glob("*_schema.json"))
    if len(schemas) != 1:
        return CheckResult("schema.bound_rationale", "SKIP", "no single *_schema.json")
    try:
        from mdp_ir.schema import load_ir
        ir = load_ir(schemas[0])
    except Exception as e:
        return CheckResult("schema.bound_rationale", "SKIP",
                           f"schema not loadable ({type(e).__name__}: {e})")

    hits, checked = [], 0
    for d in ir.mdp.decisions:
        why = (d.bounds.rationale or "").lower()
        checked += 1
        words = [w for w in _MEASURED if w in why]
        if words:
            hits.append(f"decision {d.name!r} bounds cite {words}")
    if hits:
        return CheckResult(
            "schema.bound_rationale", "WARN",
            "; ".join(hits) + " — an action bound justified by what solved runs "
            "did is a measurement of the instances already in the study, and it "
            "binds silently on the first one that outgrows it; derive it from "
            "the model and the horizon instead (would the justification hold "
            "for an instance nobody has run?)")
    return CheckResult("schema.bound_rationale", "PASS",
                       f"{checked} decision bound rationale(s) argue from the "
                       f"model, not from solved runs")



# ---------------------------------------------------------------------------
# The round-trip artifact (spec Phase-A step 7b)
# ---------------------------------------------------------------------------

_STEP7B_FENCE = "step7b"
_INTERPRETER_MODULE = "mdp_ir.interpreter"


def _fenced_blocks(text: str) -> list[tuple[str, str, int]]:
    """Split markdown into ``(info_string, body, opening_line_number)`` triples.

    Deliberately a line scanner and not a markdown parser: the only structure
    this check needs is "which fence came after which", and a dependency-free
    scan is what lets the harness read a document it does not own.
    """
    blocks, fence, info, body, start = [], None, "", [], 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if fence is None:
            if stripped.startswith("```"):
                ticks = len(stripped) - len(stripped.lstrip("`"))
                fence, info, body, start = "`" * ticks, stripped[ticks:].strip(), [], lineno
            continue
        if stripped.startswith(fence) and not stripped[len(fence):].strip():
            blocks.append((info, "\n".join(body), start))
            fence = None
            continue
        body.append(line)
    return blocks


def _join_continuations(command: str) -> str:
    return re.sub(r"\\\s*\n\s*", " ", command.strip())


def check_restatement_current(h: DomainHandle) -> CheckResult:
    """The restatement's sample trajectory still renders from the IR beside it.

    The restatement is the one artifact whose *purpose* is to be believed by a
    human, and it is the one nothing else can reach. The differential proves
    the interpreter and the generated domain agree — when the model changes
    both sides move together and the document drifts from both while the
    differential stays MATCH. Conformance reads Python, the laws runner reads
    execution semantics, the fingerprints hash the IR; none of them opens a
    ``.md``. So the failure mode is silent by construction, and it has been
    observed three times: a stale artifact showing pre-fix numbers for a whole
    campaign, one stale across eleven findings, and a shipped case whose
    documented command renders a different branch than the block it labels.

    A recorded fingerprint is the cheap check and it is not enough — it catches
    neglect but not *partial diligence*, the likelier failure where an author
    updates the one-line token and skips the expensive regeneration. So this
    re-runs the render and diffs it, which requires two things of the document:
    the invocation must be recoverable (a ``step7b``-tagged fence) and the
    output must be pasted verbatim rather than trimmed or hand-renamed. Both
    are cheap to satisfy and neither is checkable any other way.

    Paths in the declared command are written relative to the domain folder's
    *parent* (``{domain}/{domain}_schema.json``), which is what keeps the
    command portable with the folder. WARN, not FAIL, on the §7
    ``bound_rationale`` precedent: a re-render can differ for a reason the
    author has already accepted, and one look settles it.
    """
    name = "docs.restatement_current"
    docs = sorted(h.directory.glob("*.restatement.md"))
    if not docs:
        return CheckResult(name, "SKIP", "no *.restatement.md in the domain folder")

    import shlex
    import subprocess
    import sys

    problems, rendered = [], 0
    for doc in docs:
        blocks = _fenced_blocks(doc.read_text())
        declared = [(i, b) for i, b in enumerate(blocks) if b[0] == _STEP7B_FENCE]
        if not declared:
            return CheckResult(
                name, "SKIP",
                f"{doc.name} declares no ```{_STEP7B_FENCE} render command "
                f"(step 7b) — nothing to re-run")
        for index, (_, command, lineno) in declared:
            label = f"{doc.name}:{lineno}"
            if index + 1 >= len(blocks):
                problems.append(f"{label}: no output block follows the "
                                f"```{_STEP7B_FENCE} command")
                continue
            expected = blocks[index + 1][1]
            try:
                argv = shlex.split(_join_continuations(command))
            except ValueError as e:
                problems.append(f"{label}: command is not parseable ({e})")
                continue
            if len(argv) < 3 or argv[1] != "-m" or argv[2] != _INTERPRETER_MODULE:
                problems.append(f"{label}: not a `python -m {_INTERPRETER_MODULE}` "
                                f"invocation — only that is re-run")
                continue
            cwd = h.directory.parent
            schema = next((a for a in argv[3:] if a.endswith("_schema.json")), None)
            if schema is None:
                problems.append(f"{label}: command names no *_schema.json")
                continue
            target = (cwd / schema).resolve()
            if not target.exists() or target.parent != h.directory.resolve():
                problems.append(
                    f"{label}: {schema!r} does not resolve to an IR inside "
                    f"{h.directory.name}/ from its parent directory — write the "
                    f"path as {h.directory.name}/<ir>.json so the command "
                    f"travels with the folder")
                continue
            try:
                proc = subprocess.run([sys.executable, *argv[1:]], cwd=cwd,
                                      capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                problems.append(f"{label}: render timed out")
                continue
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()
                problems.append(f"{label}: render failed ({tail[-1] if tail else 'no stderr'})")
                continue
            live = [ln.rstrip() for ln in proc.stdout.splitlines()]
            documented = [ln.rstrip() for ln in expected.splitlines()]
            rendered += 1
            if live == documented:
                continue
            for row, (a, b) in enumerate(zip(documented, live), start=1):
                if a != b:
                    problems.append(
                        f"{label}: declared render differs at output line {row} "
                        f"— doc {a[:60]!r} vs live {b[:60]!r}")
                    break
            else:
                problems.append(
                    f"{label}: declared render has {len(documented)} line(s), "
                    f"live has {len(live)}")

    if problems:
        return CheckResult(
            name, "WARN",
            "; ".join(problems) + " — the artifact a human reads to check the "
            "world no longer matches the IR beside it; re-render it and re-read "
            "the prose around it, since the annotations drift with the numbers")
    return CheckResult(name, "PASS",
                       f"{rendered} declared step-7b render(s) reproduce verbatim")


# ---------------------------------------------------------------------------
# §8.2 — the CLI tier contract
# ---------------------------------------------------------------------------

# Tier 1: the invariant surface. These names mean the same thing in every domain
# and under every algorithm family, which is why they are the ones tooling and
# humans join across domains — so the *names* are the contract.
_TIER1_TRAIN = ("scenario_name", "total_timesteps", "outdir", "seed", "tag",
                "n_envs", "gym_log", "checkpoint_every_frac")
_TIER1_EVAL = ("scenario_name", "model_path", "outfile", "n_seeds", "first_seed")

# Owed only where they apply, read from the domain rather than from prose: a
# rendering mode when the env's __init__ accepts it — one legal value still owes
# the flag, since the name is what other arms join on — and the VecNormalize
# names when the script actually reaches for the wrapper in code (§8.3): built
# on the train side, loaded on the eval side (§9.5).
_TIER1_RENDER = ("observation_mode", "action_mode", "reward_mode")
_TIER1_VECNORM_TRAIN = ("norm_obs", "norm_reward", "vecnorm_clip_obs")
_TIER1_VECNORM_EVAL = ("vecnorm_path",)

# Tier 2 is deliberately not listed here. `mdp_tuning`'s space for the family is
# its oracle and §8.6 obliges a new family to bring its own; a second list in the
# checker would be a second source of truth that goes stale on the first
# off-policy family. Tier 3 is checked by nothing, by design.

# Adoption path (§8.2): every existing script fails this on the day it lands —
# upstream's own examples included — so a hard gate would make every domain
# non-conformant simultaneously. WARN for one release, then FAIL.
_CLI_CONTRACT_SEVERITY = "WARN"


def _add_argument_dests(source: str) -> set[str]:
    """Every dest the ``add_argument`` calls in one script declare.

    Static, and deliberately so: opening a train script's parser for real means
    importing SB3 and torch, and the gate that reports a missing flag must not
    need the training stack to do it. The dest rule is argparse's own — an
    explicit ``dest=`` wins, else the first long option with ``--`` stripped and
    ``-`` folded to ``_``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return _dests_in(tree)


def _dests_in(root: ast.AST) -> set[str]:
    """The dests declared anywhere under one AST node — a file, or one `def`."""
    dests: set[str] = set()
    for node in ast.walk(root):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        explicit = next((kw.value.value for kw in node.keywords
                         if kw.arg == "dest" and isinstance(kw.value, ast.Constant)
                         and isinstance(kw.value.value, str)), None)
        if explicit:
            dests.add(explicit)
            continue
        options = [a.value for a in node.args
                   if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not options:
            continue
        flags = [o for o in options if o.startswith("-")]
        name = next((f for f in flags if f.startswith("--")), flags[0] if flags else options[0])
        dests.add(name.lstrip("-").replace("-", "_"))
    return dests


def _local_module(name: str, directory: Path) -> Path | None:
    """A module name resolved against the domain folder, or None.

    Only the folder is searched, so `argparse` and SB3 resolve to nothing: the
    reader never leaves the directory it was handed.
    """
    for candidate in (name, name.split(".")[0]):
        path = directory / f"{candidate}.py"
        if path.exists():
            return path
    return None


def _module_index(tree: ast.Module, directory: Path):
    """`(top-level symbols, imported name -> (module, original), alias -> module)`."""
    symbols = {n.name: n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imported: dict[str, tuple[Path, str]] = {}
    aliases: dict[str, Path] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            dep = _local_module(node.module, directory)
            if dep:
                for a in node.names:
                    imported[a.asname or a.name] = (dep, a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                dep = _local_module(a.name, directory)
                if dep:
                    aliases[a.asname or a.name] = dep
    return symbols, imported, aliases


def _symbol_dests(path: Path, name: str, directory: Path,
                  seen: set[tuple[str, str]]) -> set[str]:
    """The dests one imported callable declares, following the calls it makes.

    Scoped to the symbol's own body, not its module: a domain's `_policy.py`
    has a `__main__` demo CLI of its own, and an eval that imports the policy
    class from it must not be credited with the demo's flags — that would hide
    a genuinely missing name behind an unrelated file.
    """
    key = (str(path.resolve()), name)
    if key in seen:
        return set()
    seen.add(key)
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return set()
    symbols, imported, aliases = _module_index(tree, directory)
    if name not in symbols:
        if name in imported:  # re-exported by this module; keep walking
            dep, original = imported[name]
            return _symbol_dests(dep, original, directory, seen)
        return set()
    node = symbols[name]
    dests = _dests_in(node)
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Name):
            if func.id in symbols:
                dests |= _symbol_dests(path, func.id, directory, seen)
            elif func.id in imported:
                dep, original = imported[func.id]
                dests |= _symbol_dests(dep, original, directory, seen)
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id in aliases):
            dests |= _symbol_dests(aliases[func.value.id], func.attr, directory, seen)
    return dests


def _cli_dests(path: Path, directory: Path) -> set[str]:
    """Every dest a script's CLI surface offers, a shared builder's included.

    A domain that factors the common eval surface into one
    `{domain}_benchmark_common.build_arg_parser` offers those names from every
    script that calls it, and §8.2's table is a contract over the names the
    surface offers, not over which file spells them. Reading one file at a time
    reports them missing, and the only way to clear that warning is to copy the
    builder's flags into each script — a second source of truth for the very
    surface tier 1 exists to standardize.

    So the reader resolves the callables a script imports from its own folder,
    transitively through the calls they make. Nothing is imported for real, so
    the reason this is static at all — reporting a missing flag must not need
    SB3 and torch — survives.
    """
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return set()
    dests = _dests_in(tree)
    symbols, imported, aliases = _module_index(tree, directory)
    seen: set[tuple[str, str]] = set()
    for dep, original in imported.values():
        dests |= _symbol_dests(dep, original, directory, seen)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            dests |= _symbol_dests(aliases[node.value.id], node.attr, directory, seen)
    return dests


def _vecnormalize_refs(source: str) -> tuple[bool, bool]:
    """``(builds, uses)`` the wrapper — read from the code, never from prose.

    §8.2 states the condition as a construction, so a substring search over the
    file is the wrong instrument: it fires on the comment in a script that
    implements its *own* normalization and mentions VecNormalize only to say
    what it is an analog of. Owing three knobs that control nothing is worse
    than the warning it would silence — an args log recording a normalization
    the run never had is a false record.

    Two answers, because the two sides use the wrapper differently: a train
    script **builds** one (`VecNormalize(env, ...)`), while an eval script
    loads the saved stats (`VecNormalize.load(...)`) and tests for them
    (`isinstance(env, VecNormalize)`) — §9.5 — and still owes `vecnorm_path`
    for the file it reads. An import alone obliges nothing.

    The known blind spot of reading the symbol: an evaluator that unpickles the
    saved stats itself, never touching the name, is invisible here. That is the
    cheap direction to be wrong in — the flag goes unasked-for rather than a
    working script being told to add knobs it cannot honour.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False, False

    def _is_vecnorm(node) -> bool:
        return ((isinstance(node, ast.Name) and node.id == "VecNormalize")
                or (isinstance(node, ast.Attribute) and node.attr == "VecNormalize"))

    builds = uses = False
    for node in ast.walk(tree):
        if _is_vecnorm(node):
            uses = True
        if isinstance(node, ast.Call) and _is_vecnorm(node.func):
            builds = True
    return builds, uses


def _env_accepts(env_cls, name: str) -> bool:
    try:
        return name in inspect.signature(env_cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _cli_findings(path: Path, owed: list[str], directory: Path) -> str | None:
    missing = sorted(set(owed) - _cli_dests(path, directory))
    return f"{path.name}: missing {', '.join(missing)}" if missing else None


def check_script_cli_contract(h: DomainHandle) -> CheckResult:
    """Train and eval scripts expose the tier-1 CLI surface (spec §8.2, §9.1).

    The convention existed and had converged — the four IR-era domains agreed on
    every contested name — but it was written down nowhere, so each new domain
    re-derived it and sometimes missed. Three of the names exist because a spec
    section imposed a capability while naming no flag for it: `first_seed` makes
    §9.7's *disjoint* selection block expressible at all, `checkpoint_every_frac`
    keeps "~20 checkpoints" true when the budget moves, and `norm_obs` is the one
    §8.3 hardcoded, so a script written to that skeleton could express neither L0
    nor its own L1 derivation.

    A script that does not exist yet is not a violation: the check reads the
    parsers it finds and skips the rest, so it never reports a stage the campaign
    has not claimed.
    """
    trains = sorted(h.directory.glob(f"{h.name}_*_train.py"))
    if not trains:
        return CheckResult("scripts.cli_contract", "SKIP",
                           "no train script yet — a domain mid-formalization owes none")
    render = [m for m in _TIER1_RENDER if _env_accepts(h.env_cls, m)]
    findings: list[str] = []
    n_read = 0
    absent: list[str] = []
    for train in trains:
        algo = train.name[len(h.name) + 1: -len("_train.py")]
        n_read += 1
        owed = list(_TIER1_TRAIN) + render
        if _vecnormalize_refs(train.read_text())[0]:
            owed += list(_TIER1_VECNORM_TRAIN)
        findings.append(_cli_findings(train, owed, h.directory))
        ev = h.directory / f"{h.name}_{algo}_eval.py"
        if not ev.exists():
            absent.append(ev.name)
            continue
        n_read += 1
        owed = list(_TIER1_EVAL) + render
        if _vecnormalize_refs(ev.read_text())[1]:
            owed += list(_TIER1_VECNORM_EVAL)
        findings.append(_cli_findings(ev, owed, h.directory))
    findings = [f for f in findings if f]
    tail = f"; no {', '.join(absent)} yet (not a violation)" if absent else ""
    if findings:
        return CheckResult("scripts.cli_contract", _CLI_CONTRACT_SEVERITY,
                           "; ".join(findings) + tail)
    return CheckResult("scripts.cli_contract", "PASS",
                       f"{n_read} script(s) expose the tier-1 surface" + tail)


# §8.2 schedule pairs. The names come from the spec's own text, not from
# mdp_tuning's space — conformance must not learn the tuning knob list (that is
# tier 2's whole point, and mdp_tuning imports this package, so the dependency
# only runs one way).
_SCHEDULE_PAIRS = (("learning_rate", "lr_final"), ("clip_init", "clip_final"))


def check_schedule_pairs(h: DomainHandle) -> CheckResult:
    """A schedule's init and final are exposed together (spec §8.2).

    The derivation that keeps a schedule from inverting — `lr_final = lr/10`,
    `clip_final = clip_init/4` — is guarded on the final's dest existing, so a
    script exposing only the init makes it a silent no-op: `build_schedule`
    collapses to a constant and the run trains flat while its args log and the
    study both record a tuned init. No artifact downstream distinguishes that
    from a schedule that was actually derived, which is the whole reason this is
    checked statically rather than left to the run.

    Exposing *neither* half is out of scope here — that is tier-2 completeness,
    which `mdp_tuning --show-space` owns.
    """
    trains = sorted(h.directory.glob(f"{h.name}_*_train.py"))
    if not trains:
        return CheckResult("scripts.schedule_pairs", "SKIP", "no train script yet")
    findings = []
    for train in trains:
        dests = _cli_dests(train, h.directory)
        for init, final in _SCHEDULE_PAIRS:
            if init in dests and final not in dests:
                findings.append(f"{train.name}: --{init} without --{final} "
                                f"(schedule is silently constant)")
            elif final in dests and init not in dests:
                findings.append(f"{train.name}: --{final} without --{init}")
    if findings:
        return CheckResult("scripts.schedule_pairs", _CLI_CONTRACT_SEVERITY,
                           "; ".join(findings))
    return CheckResult("scripts.schedule_pairs", "PASS",
                       f"{len(trains)} train script(s): every exposed schedule "
                       f"init has its final")


def _l1_derived(source: str) -> tuple[list[str] | None, bool]:
    """``(keys, is_read)`` for a module-level ``_L1_DERIVED``; keys is None when
    the script declares none.

    Read from the AST rather than by importing: conformance runs on a folder
    whose train script imports SB3, and the check must work in the torch-free
    install like every other one here.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, False
    keys: list[str] | None = None
    for node in tree.body:
        names = ([t.id for t in node.targets if isinstance(t, ast.Name)]
                 if isinstance(node, ast.Assign)
                 else [node.target.id] if isinstance(node, ast.AnnAssign)
                 and isinstance(node.target, ast.Name) else [])
        if "_L1_DERIVED" not in names:
            continue
        value = node.value
        keys = ([k.value for k in value.keys
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                if isinstance(value, ast.Dict) else [])
    read = any(isinstance(n, ast.Name) and n.id == "_L1_DERIVED"
               and isinstance(n.ctx, ast.Load) for n in ast.walk(tree))
    return keys, read


def _l1_basis(source: str) -> tuple[dict, bool]:
    """``(basis, declared)`` for a module-level ``_L1_BASIS`` of constants."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, False
    for node in tree.body:
        names = ([t.id for t in node.targets if isinstance(t, ast.Name)]
                 if isinstance(node, ast.Assign)
                 else [node.target.id] if isinstance(node, ast.AnnAssign)
                 and isinstance(node.target, ast.Name) else [])
        if "_L1_BASIS" in names and isinstance(node.value, ast.Dict):
            return {k.value: (v.value if isinstance(v, ast.Constant) else None)
                    for k, v in zip(node.value.keys, node.value.values)
                    if isinstance(k, ast.Constant)}, True
    return None, False


def check_l1_derived(h: DomainHandle) -> CheckResult:
    """The L1 derivation is data, and the run name diffs against it (spec §8.4).

    The rule it enforces exists because two spec statements are only jointly
    true until a campaign promotes a discovered value: §8.4 encodes a
    hyperparameter when it differs from the script default, and §8.6 says those
    defaults *are* the L1 centre. Edit a default to adopt a finding and the tag
    disappears from every later run name, leaving an archive of textually
    identical names meaning different configurations — so `_L1_DERIVED` is what
    the name is measured against, and a default is free to move.

    Two failures are worth separating from "not adopted yet". A key that is not
    a CLI dest is a knob the diff can never fire on — a typo, or a dest renamed
    out from under the dict. A dict nothing reads is worse: the script looks
    adopted and its run names still diff the defaults, which is the silent
    no-op this check exists to make loud.
    """
    trains = sorted(h.directory.glob(f"{h.name}_*_train.py"))
    if not trains:
        return CheckResult("scripts.l1_derived", "SKIP",
                           "no train script yet — a domain mid-formalization owes none")
    legacy, findings, sizes = [], [], []
    for train in trains:
        keys, read = _l1_derived(train.read_text())
        if keys is None:
            legacy.append(train.name)
            continue
        sizes.append(f"{train.name}: {len(keys)} derived value(s)")
        if not keys:
            findings.append(f"{train.name}: _L1_DERIVED is empty — a derivation "
                            f"that states nothing is not one")
            continue
        unknown = sorted(set(keys) - _cli_dests(train, h.directory))
        if unknown:
            findings.append(f"{train.name}: _L1_DERIVED names {unknown}, which "
                            f"the CLI has no dest for — the run name can never "
                            f"diff them")
        if not read:
            findings.append(f"{train.name}: _L1_DERIVED is declared but never "
                            f"read — build_run_name still diffs the parser "
                            f"defaults, so a promoted value would go untagged")
    if findings:
        return CheckResult("scripts.l1_derived", "FAIL", "; ".join(findings))
    if legacy:
        return CheckResult("scripts.l1_derived", "WARN",
                           f"{', '.join(legacy)}: no _L1_DERIVED — pre-convention, "
                           f"so the run name diffs mutable defaults and a promoted "
                           f"value would go untagged (§8.4)")
    return CheckResult("scripts.l1_derived", "PASS", "; ".join(sizes))


def _argument_default(source: str, dest: str) -> tuple[bool, object]:
    """``(declared, default)`` for one dest's ``add_argument`` call.

    A dest can exist and still be unusable: §8.2 requires
    `--checkpoint-every-frac` and §8.6 wants ~20 checkpoints from it, but a
    script that declares the flag with ``default=None`` saves none unless every
    launch remembers to pass it. Reading the dest alone cannot see that, which
    is the difference between forcing a capability and forcing its use.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False, None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if dest not in _dests_in(node):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                return True, (kw.value.value
                              if isinstance(kw.value, ast.Constant) else "?")
        return True, None
    return False, None


def _names_used(source: str, predicate) -> set[str]:
    """Names this script USES — not merely imports or mentions — filtered.

    Any load counts: a call, but also a binding (`cb_cls = MaskableEvalCallback
    if masked else EvalCallback`), a wrapper's argument, a base class; an
    `import ... as` alias resolves to the imported name. Matching call names
    alone missed the shipped case — game2048 constructs through `cb_cls(...)`.
    An unused import and a docstring naming the class stay silent.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    aliases = {a.asname: a.name.rsplit(".", 1)[-1]
               for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom))
               for a in node.names if a.asname}
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = aliases.get(node.id, node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            name = node.attr
        else:
            continue
        if predicate(name):
            out.add(name)
    return out


def check_launch_check(h: DomainHandle) -> CheckResult:
    """The train script checks its own derivation at launch (spec §8.6).

    Conformance runs before anything is launched and the eval gate runs after
    everything has, so the launch — *is this experiment well-posed?* — had no
    gate at all, and that is where the expensive mistakes are made. §8.6 already
    asks the script to state what it derived and from what; `assert_l1_current`
    turns the statement into a check, and this reports whether the script makes
    it. A basis naming a scenario the registry does not have is worse than none:
    the staleness comparison silently never fires.
    """
    trains = sorted(h.directory.glob(f"{h.name}_*_train.py"))
    if not trains:
        return CheckResult("scripts.launch_check", "SKIP",
                           "no train script yet — a domain mid-formalization owes none")
    silent, findings = [], []
    for train in trains:
        source = train.read_text()
        # either entry point: `assert_launch` is the same check plus the
        # config/comparator groups a domain with a §13 registry can supply
        if not any(name in source
                   for name in ("assert_l1_current", "assert_launch")):
            silent.append(train.name)
        basis, _ = _l1_basis(source)
        if basis is None:
            continue
        scen = basis.get("scenario_name")
        if scen is not None and h.SCENARIOS and scen not in h.SCENARIOS:
            findings.append(f"{train.name}: _L1_BASIS names scenario {scen!r}, "
                            f"which SCENARIOS does not have — the staleness "
                            f"comparison can never fire")
    if findings:
        return CheckResult("scripts.launch_check", "FAIL", "; ".join(findings))
    if silent:
        return CheckResult("scripts.launch_check", "WARN",
                           f"{', '.join(silent)}: no assert_l1_current/assert_launch "
                           f"call — the "
                           f"L1 derivation is printed, not checked, so a run at "
                           f"an instance it was not derived for is launched "
                           f"silently (§8.6)")
    return CheckResult("scripts.launch_check", "PASS",
                       f"{len(trains)} train script(s) check the derivation at launch")


def check_selection_protocol(h: DomainHandle) -> CheckResult:
    """Selection is the post-hoc screen, not a live callback (spec §8.6/§9.7).

    §9.7 has specified post-hoc checkpoint selection since v0.7.0, in terms that
    exclude the alternative outright — "no `EvalCallback`, no live selection
    env". Neither existing gate can see a violation: the code is conformant, and
    the numbers clear their baselines. The cost is not hypothetical — one
    campaign selected among ~400 callback candidates instead of ~20 checkpoints,
    paying roughly a 5x selection-noise multiplier on every arm for *more*
    compute than the mandated protocol costs.

    A warning rather than a failure, deliberately: a domain generated before the
    rule produced every number it reports with the machinery it has, and
    rewriting that would leave its leaderboard describing code that did not
    generate it (`examples/mab` is exactly this case, and says so in its own
    caveat). What the campaign owes is the knowledge, at the moment of use.
    """
    trains = sorted(h.directory.glob(f"{h.name}_*_train.py"))
    if not trains:
        return CheckResult("scripts.selection_protocol", "SKIP",
                           "no train script yet — a domain mid-formalization owes none")
    findings = []
    for train in trains:
        source = train.read_text()
        live = sorted(_names_used(source, lambda n: n.endswith("EvalCallback")))
        if live:
            findings.append(f"{train.name}: selects live with {', '.join(live)} — §9.7 "
                            f"selects post-hoc from checkpoints, with no live "
                            f"selection env")
        declared, default = _argument_default(source, "checkpoint_every_frac")
        if declared and (default is None or default == 0):
            findings.append(f"{train.name}: checkpoint_every_frac defaults to "
                            f"{default!r}, so a run that does not pass it saves "
                            f"no checkpoints and §9.7's screen cannot run "
                            f"(§8.2 defaults it to 0.05)")
    if findings:
        return CheckResult("scripts.selection_protocol", "WARN", "; ".join(findings))
    return CheckResult("scripts.selection_protocol", "PASS",
                       f"{len(trains)} train script(s): checkpoints saved by "
                       f"default, no live selection callback")


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
    check_run_provenance,
    check_model_boundary,
    check_no_enumeration,
    check_no_hardcode,
    check_bound_rationale,
    check_restatement_current,
    check_script_cli_contract,
    check_schedule_pairs,
    check_l1_derived,
    check_launch_check,
    check_selection_protocol,
    check_init_state,
    check_gym_contract,
    check_determinism,
    check_gym_reseed,
    check_purity,
    check_rng_generators,
]

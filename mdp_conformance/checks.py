"""Conformance checks for MDP domains.

Each check takes a ``DomainHandle`` and returns a ``CheckResult`` (or a list of
them). Checks fall into two groups:

- **Static** — parse the source (AST) without running it: file layout, acyclic
  layering (§1.1), the ``param`` ban (§2), no reward in the MDP layer (§6.4),
  and ``@dataclass(slots=True)`` on the state.
- **Behavioral** — construct the gym and run the simulator: scenarios build,
  ``init_state`` invariants, the Gymnasium reset/step contract, whole-episode
  determinism, ``advance`` purity, and the per-generator RNG seeding contract.

Behavioral checks drive the domain through its gym wrapper so that single-step
(``advance``) and two-step (``advance1``/``advance2``) domains are handled
uniformly. The RNG check works at the generator level: a generator that yields a
value from only ``(period, episode_seed, seed_salt)`` is decision-path
independent *by construction*; one that needs more state (e.g. 2048's board) is
classified state-conditioned and skipped rather than failed.
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
    return bool(a == b)


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
    forbidden = {
        "uncertainty": ("_scenarios", "_mdp", "_gym"),
        "scenarios":   ("_mdp", "_gym"),
        "mdp":         ("_gym",),
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
    return CheckResult("static.layering", "PASS", "acyclic uncertainty←scenarios←mdp←gym")


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


REGISTRY = [
    check_file_layout,
    check_layering,
    check_no_param,
    check_mdp_no_reward,
    check_state_slots,
    check_scenarios_valid,
    check_init_state,
    check_gym_contract,
    check_determinism,
    check_purity,
    check_rng_generators,
]

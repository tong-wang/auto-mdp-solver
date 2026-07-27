"""Restricted interpreter that executes an MDP-IR directly — no generated code.

Two roles (SKILL.md Phase A; ``MDP_IR_SAMPLE.md`` §2b):

  * **Phase A**: produce the human-readable sample trajectories the user
    confirms at the round-trip gate, before any domain code exists.
  * **Stage 1**: differential-test oracle for codegen — the generated
    ``_mdp`` must reproduce the interpreter's trajectories on shared
    ``(instance, episode_seed, decisions)``.

Semantics implemented:

  * ``mdp.dynamics.transitions`` run in order each period; a transition's
    ``guard`` gates its updates. Updates are either draw statements
    (``x ~ source.stage``) or restricted Python statements executed against
    the IR namespace (state ∪ info ∪ decisions ∪ scenario constants ∪ locals
    ∪ whitelisted functions — the same namespace the schema validates).
  * Seeding follows the schema's *derived* key (``UncertaintyStage.seed_key``),
    whose slot order depends on ``seed_scheme``: v1 keys ``stream_id`` before
    ``period``; v2 (spec §6.3, canonical for new domains) keys ``period`` below
    the source with a branch word — see that method for both grammars. An
    episode-realization stage takes a period-less key, so its draw is constant
    across the episode; period/event draws key on the current period, so
    realized uncertainty is decision-path independent by construction.
  * ``objective.per_step_components`` are evaluated at END_OF_PERIOD on
    end-of-period state; the decomposition info field carries them + ``total``.
  * The gym block supplies the default reward mode (evaluated per step) and
    termination: ``early_terminated_when`` (absorbing) or ``period == T``.

CLI:

    python -m mdp_ir.interpreter IR.json [--episode-seed N] [--seed-salt N]
        [--instance NAME] [--decision name=value ...] [--max-periods N]

Decisions not fixed with ``--decision`` are drawn uniformly at random within
their declared bounds (deterministically from the episode seed).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from mdp_ir.schema import (
    DecisionType,
    MdpIR,
    PeriodIndexing,
    Realization,
    StateRole,
    UncertaintySource,
    UncertaintyStage,
)

_DRAW = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*~\s*([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*$"
)

# the function whitelist matching the schema's _BUILTINS (minus keywords).
# Core functions only — domain-owned builtins are declared per IR in
# `mdp.expr_builtins` and injected in `IrInterpreter.__init__`.
_FUNCS: dict[str, object] = {
    "min": min, "max": max, "sum": sum, "abs": abs, "len": len,
    "round": round, "int": int, "float": float,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "floor": math.floor, "ceil": math.ceil,
    "zeros": lambda n: [0] * int(n),
    "phi": lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))),
    "topk": lambda values, k: sorted(values, reverse=True)[: int(k)],
    "range": range,
}

# numpy Generator scalar-distribution methods callable by name: a family not
# handled explicitly in `_sample_family` is dispatched to `rng.<family>(...)`
# with its `settings` passed straight through as numpy's own keyword arguments
# (e.g. gamma -> {shape, scale}; binomial -> {n, p}; beta -> {a, b}). Curated
# once from numpy's univariate distributions, so a new distribution needs no
# code here — just name it in the IR. Array-valued draws (dirichlet,
# multinomial, multivariate_*) are intentionally excluded: this samples one
# scalar. Families with friendlier aliases (poisson.rate, normal.mean/std) are
# handled explicitly above and take precedence over this generic path.
_NUMPY_SCALAR_DISTS: frozenset[str] = frozenset({
    "beta", "binomial", "chisquare", "exponential", "f", "gamma",
    "geometric", "gumbel", "hypergeometric", "laplace", "logistic",
    "lognormal", "logseries", "negative_binomial", "noncentral_chisquare",
    "noncentral_f", "normal", "pareto", "poisson", "power", "rayleigh",
    "standard_cauchy", "standard_exponential", "standard_gamma",
    "standard_normal", "standard_t", "triangular", "uniform", "vonmises",
    "wald", "weibull", "zipf",
})


def _domain_builtin(domain: str, module: str, func: str):
    """Import a domain-owned builtin implementation on first use.

    Resolution order: a module already importable (conformance/differential
    put the domain dir on sys.path), then the directory of every IR loaded
    this process (``schema.DOMAIN_DIRS``) and its ``{domain}/`` subdirectory —
    the implementation is found next to the IR that calls it, wherever that
    folder lives (portable-domain contract; no repo-root assumption).
    """
    import importlib
    import sys as _sys

    from mdp_ir.schema import DOMAIN_DIRS

    try:
        return getattr(importlib.import_module(module), func)
    except ModuleNotFoundError:
        pass
    for base in DOMAIN_DIRS:
        for cand in (base, base / domain):
            if (cand / f"{module}.py").exists():
                path = str(cand)
                if path not in _sys.path:
                    _sys.path.insert(0, path)
                return getattr(importlib.import_module(module), func)
    raise ModuleNotFoundError(
        f"domain builtin {module}.{func}: {module}.py is neither importable "
        f"nor found next to any loaded IR ({[str(d) for d in DOMAIN_DIRS]})"
    )


def _declared_funcs(ir: MdpIR) -> dict[str, object]:
    """Callables for the IR's declared ``mdp.expr_builtins``: each lazily
    resolves ``def {name}`` in ``{module}.py`` next to the IR on first call,
    so IRs that never call one never trigger the import."""

    def lazy(domain: str, module: str, func: str):
        return lambda *args: _domain_builtin(domain, module, func)(*args)

    return {
        b.name: lazy(ir.domain.name, b.module, b.name)
        for b in ir.mdp.expr_builtins
    }

# policy stream: keeps random fallback decisions out of the model streams
_POLICY_STREAM = 9999


def _meta_seed_key(
    scheme: str, substream_id: int, episode_seed: int, seed_salt: int
) -> list[int]:
    """Meta-branch key for samplers/mixtures (spec §6.3). v2: leaf-first
    ``[substream, 0, episode_seed, seed_salt]``. v1 legacy: the historical
    ``[0, episode_seed, seed_salt]`` for the lone (substream 0) drawer, else
    ``[0, substream, episode_seed, seed_salt]``."""
    if scheme == "v2":
        return [substream_id, 0, episode_seed, seed_salt]
    if substream_id == 0:
        return [0, episode_seed, seed_salt]
    return [0, substream_id, episode_seed, seed_salt]


def _numeric_seed_key(
    stage: UncertaintyStage,
    stream_id: int,
    *,
    period: int,
    episode_seed: int,
    seed_salt: int,
    entity_id: int | None = None,
    key_vals: list[int] | None = None,
    scheme: str = "v1",
) -> list[int]:
    """Numeric form of ``UncertaintyStage.seed_key`` (same slot order, per
    ``scheme``): v1 keys ``stream_id`` before ``period``; v2 (spec §6.3) keys
    ``period`` below the source, then the branch word ``1`` — see
    ``UncertaintyStage.seed_key`` for both symbolic grammars. Interpreter draws
    are bit-identical to conforming domain code. ``key_vals`` are the evaluated
    ``key_exprs`` of a keyed stage; keyed stages carry no period slot (the draw
    is a fixed per-episode latent table indexed by the key values)."""
    key: list[int] = []
    if entity_id is not None:
        key.append(entity_id)
    if key_vals:
        key += key_vals
    has_period = stage.realization not in (Realization.episode, Realization.keyed)
    if scheme == "v2":
        # v2 seed tree (spec §6.3), leaf-first: [.., sub?, period?, source, 1, e, salt]
        if stage.sub_stream is not None:
            key.append(stage.sub_stream)
        if has_period:
            key.append(period)
        key += [stream_id, 1]
    else:
        if stage.sub_stream is not None:
            key.append(stage.sub_stream)
        key.append(stream_id)
        if has_period:
            key.append(period)
    key += [episode_seed, seed_salt]
    return key


@dataclass
class Trajectory:
    episode_seed: int
    instance: str | None
    rows: list[dict] = field(default_factory=list)
    terminated_early: bool = False
    total: float = 0.0
    total_reward: float = 0.0

    def render(self, max_rows: int | None = None) -> str:
        if not self.rows:
            return "(empty trajectory)"
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        cols = list(rows[0].keys())

        def fmt(v: object) -> str:
            if isinstance(v, float):
                return f"{v:.2f}"
            if isinstance(v, list):
                return "[" + ",".join(fmt(x) for x in v) + "]"
            return str(v)

        cells = [[fmt(r[c]) for c in cols] for r in rows]
        widths = [
            max(len(c), *(len(row[i]) for row in cells))
            for i, c in enumerate(cols)
        ]
        out = ["  ".join(c.rjust(w) for c, w in zip(cols, widths))]
        out += ["  ".join(v.rjust(w) for v, w in zip(row, widths)) for row in cells]
        if max_rows is not None and len(self.rows) > max_rows:
            out.append(f"... ({len(self.rows) - max_rows} more periods)")
        tag = " (terminated early)" if self.terminated_early else ""
        out.append(
            f"episode total = {self.total:.2f}   "
            f"reward total = {self.total_reward:.2f}   "
            f"periods = {len(self.rows)}{tag}"
        )
        return "\n".join(out)


class IrInterpreter:
    """Executes one IR. Stateless across episodes; all randomness flows
    through the derived seed keys, exactly as generated code must."""

    def __init__(self, ir: MdpIR, instance: str | None = None, seed_salt: int = 0):
        self.ir = ir
        self.instance = instance
        self.seed_salt = seed_salt
        self._scheme = ir.seed_scheme
        self._funcs = {**_FUNCS, **_declared_funcs(ir)}
        # a mixture name is usable wherever an instance name is (spec §5.3);
        # its component is drawn per episode in _episode_setup()
        self._mixture = next(
            (m for m in ir.mdp.scenario.mixtures if m.name == instance), None
        )
        base_instance = None if self._mixture is not None else instance
        self.constants: dict[str, object] = {
            c.name: c.value for c in ir.mdp.scenario.constants
        }
        if base_instance is not None:
            overrides = ir.mdp.scenario.instances.get(base_instance)
            if overrides is None:
                raise KeyError(
                    f"unknown scenario instance {base_instance!r}; "
                    f"have {sorted(ir.mdp.scenario.instances)} "
                    f"+ mixtures {[m.name for m in ir.mdp.scenario.mixtures]}"
                )
            self.constants.update(overrides)
        self._base_constants = dict(self.constants)
        self.T = ir.mdp.horizon_T(base_instance)
        self._sources = {s.name: s for s in ir.mdp.uncertainty_sources}
        self._decomposition = next(
            (f for f in ir.mdp.info_fields if f.type == "decomposition"), None
        )
        self._scalar_info = [
            f.name for f in ir.mdp.info_fields if f.type != "decomposition"
        ]
        self._reward_mode = next(m for m in ir.gym.reward_modes if m.default)

    # -- expression plumbing -------------------------------------------------

    def _base_ns(self) -> dict:
        ns: dict = {"__builtins__": {}}
        ns.update(self._funcs)
        ns.update(self.constants)
        ns["T"] = self.T
        return ns

    @staticmethod
    def _eval(expr: str, ns: dict) -> object:
        return eval(compile(expr, "<ir-expr>", "eval"), ns)  # noqa: S307 — namespace is restricted

    @staticmethod
    def _exec(stmt: str, ns: dict) -> None:
        exec(compile(stmt, "<ir-stmt>", "exec"), ns)  # noqa: S102 — namespace is restricted

    def _setting(self, value: object, ns: dict) -> object:
        """Distribution settings: literals pass through, strings are exprs
        over the namespace (constants, state, current decisions)."""
        return self._eval(value, ns) if isinstance(value, str) else value

    # -- sampling -------------------------------------------------------------

    def _rng(self, source: UncertaintySource, stage: UncertaintyStage,
             period: int, episode_seed: int,
             ns: dict | None = None) -> np.random.Generator:
        key_vals = None
        if stage.key_exprs:
            assert ns is not None, f"stage {stage.name!r}: keyed draw needs a namespace"
            key_vals = [int(self._eval(e, ns)) for e in stage.key_exprs]
        key = _numeric_seed_key(
            stage, source.stream_id,
            period=period, episode_seed=episode_seed, seed_salt=self.seed_salt,
            key_vals=key_vals, scheme=self._scheme,
        )
        return np.random.default_rng(np.random.SeedSequence(key))

    # -- episode setup: world-layer meta draws (spec §5.2, §5.3) --------------

    def _episode_setup(self, episode_seed: int) -> None:
        """Resolve this episode's constants and horizon: draw the mixture
        component (if the target is a mixture), then run every applicable
        sampler's draws in declaration order from its meta-keyed rng."""
        scenario = self.ir.mdp.scenario
        if self._mixture is None and not scenario.samplers:
            return
        consts = dict(self._base_constants)
        inst = self.instance
        if self._mixture is not None:
            m = self._mixture
            weights = np.asarray([w for w, _ in m.components], dtype=float)
            rng = np.random.default_rng(np.random.SeedSequence(
                _meta_seed_key(self._scheme, m.substream_id,
                               episode_seed, self.seed_salt)))
            k = int(rng.choice(len(m.components), p=weights / weights.sum()))
            inst = m.components[k][1] or None
            if inst:
                consts.update(scenario.instances[inst])
        ns: dict = {"__builtins__": {}, **self._funcs, **consts}
        for smp in scenario.samplers:
            if smp.instances and (inst or "") not in smp.instances:
                continue
            # one rng per sampler; draws execute in order (multi-draw recipes
            # like support-then-weights are bit-reproducible by construction)
            rng = np.random.default_rng(np.random.SeedSequence(
                _meta_seed_key(self._scheme, smp.substream_id,
                               episode_seed, self.seed_salt)))
            for draw in smp.draws:
                value = self._sample_family(
                    rng, draw.distribution.family, draw.distribution.settings, ns)
                consts[draw.name] = value
                ns[draw.name] = value
        self.constants = consts
        t = self.ir.mdp.horizon.T
        self.T = int(consts[t]) if isinstance(t, str) else int(t)

    def _draw(self, source: UncertaintySource, stage: UncertaintyStage,
              ns: dict, period: int, episode_seed: int) -> object:
        rng = self._rng(source, stage, period, episode_seed, ns)
        return self._sample_family(
            rng, source.distribution.family, source.distribution.settings, ns)

    def _sample_family(self, rng: np.random.Generator, fam: str,
                       settings: dict, ns: dict) -> object:
        """Sample one value of a distribution family from ``rng``. Shared by
        per-period/event draws and scenario-sampler draws (spec §5.2)."""
        if fam == "categorical":
            vals = self._setting(settings["values"], ns)
            probs = self._setting(settings["probabilities"], ns)
            return int(rng.choice(np.asarray(vals), p=np.asarray(probs, dtype=float)))
        if fam == "poisson":
            return int(rng.poisson(float(self._setting(settings["rate"], ns))))
        if fam == "normal":
            return float(rng.normal(float(self._setting(settings["mean"], ns)),
                                    float(self._setting(settings["std"], ns))))
        if fam == "lognormal":
            # settings are the underlying normal's (mean, sigma)
            return float(rng.lognormal(float(self._setting(settings["mean"], ns)),
                                       float(self._setting(settings["sigma"], ns))))
        if fam == "uniform":
            return float(rng.uniform(float(self._setting(settings["low"], ns)),
                                     float(self._setting(settings["high"], ns))))
        if fam == "bernoulli":
            # single-parameter family; accept any single setting key (p, p_high, …)
            (p,) = [self._setting(v, ns) for v in settings.values()]
            return int(rng.random() < float(p))
        # sampler-recipe families (scenario samplers, spec §5.2): list-valued
        if fam == "choice_without_replacement":
            lo = int(self._setting(settings["low"], ns))
            hi = int(self._setting(settings["high"], ns))
            size = int(self._setting(settings["size"], ns))
            return [int(v) for v in
                    rng.choice(np.arange(lo, hi + 1), size=size, replace=False)]
        if fam == "normalized_uniform_weights":
            size = int(self._setting(settings["size"], ns))
            lo = float(self._setting(settings.get("low", 1.0), ns))
            hi = float(self._setting(settings.get("high", 10.0), ns))
            raw = rng.uniform(lo, hi, size=size)
            return [float(v) for v in raw / raw.sum()]
        # any other numpy Generator scalar distribution, by name: settings map
        # straight to numpy's own parameters. New distributions need no code
        # here (see _NUMPY_SCALAR_DISTS). `.item()` surfaces numpy's native
        # int/float, so discrete families coerce to int and continuous to float.
        if fam in _NUMPY_SCALAR_DISTS:
            draw = getattr(rng, fam)(
                **{k: self._setting(v, ns) for k, v in settings.items()}
            )
            return draw.item() if hasattr(draw, "item") else draw
        raise NotImplementedError(f"distribution family {fam!r}")

    # -- episode --------------------------------------------------------------

    def _initial_state(self, ns: dict) -> dict:
        state: dict = {}
        origin = (
            0 if self.ir.mdp.horizon.period_indexing is PeriodIndexing.zero_based else 1
        )
        for sv in self.ir.mdp.state_variables:
            if sv.name in self.ir.mdp.initial_state:
                v = self.ir.mdp.initial_state[sv.name]
                state[sv.name] = self._eval(v, ns) if isinstance(v, str) else v
            elif sv.role is StateRole.time_index:
                state[sv.name] = origin
            else:  # unreachable: schema requires core vars in initial_state
                state[sv.name] = 0
        return state

    def _random_policy(self, episode_seed: int) -> Callable[[dict], dict]:
        rng = np.random.default_rng(
            np.random.SeedSequence([_POLICY_STREAM, episode_seed, self.seed_salt])
        )

        def policy(_ns: dict) -> dict:
            out = {}
            for d in self.ir.mdp.decisions:
                # bound entries may name scenario constants (already merged
                # with the instance overrides in self.constants)
                lo, hi = (
                    self.constants[x] if isinstance(x, str) else x
                    for x in d.bounds.value
                )
                if d.type.value is DecisionType.discrete:
                    out[d.name] = int(rng.integers(int(lo), int(hi) + 1))
                else:
                    out[d.name] = float(rng.uniform(lo, hi))
            return out

        return policy

    def run(
        self,
        episode_seed: int,
        decisions: Callable[[dict], dict] | list[dict] | dict | None = None,
        max_periods: int | None = None,
    ) -> Trajectory:
        """Roll one episode. ``decisions``: a callable(namespace)->dict, a
        per-period list of dicts, a constant dict, or None (random within
        bounds, seeded from the episode seed)."""
        ir = self.ir
        # world-layer meta draws first: mixture component + sampler draws set
        # this episode's constants/horizon before anything reads them
        self._episode_setup(episode_seed)
        if decisions is None:
            policy = self._random_policy(episode_seed)
        elif callable(decisions):
            policy = decisions
        elif isinstance(decisions, dict):
            policy = lambda _ns, _d=dict(decisions): dict(_d)  # noqa: E731
        else:
            seq = [dict(d) for d in decisions]
            policy = lambda ns, _s=seq: _s[min(len(_s) - 1, ns["_step"])]  # noqa: E731

        time_var = next(
            sv.name for sv in ir.mdp.state_variables if sv.role is StateRole.time_index
        )
        early = ir.gym.termination.early_terminated_when
        comp_names = [c.name for c in ir.mdp.objective.per_step_components]

        ns = self._base_ns()
        ns.update(self._initial_state(ns))
        traj = Trajectory(episode_seed=episode_seed, instance=self.instance)
        limit = self.T if max_periods is None else min(self.T, max_periods)

        for step in range(limit):
            period = int(ns[time_var])
            ns["_step"] = step
            for f in self._scalar_info:
                ns[f] = 0
            acts = policy(ns)
            ns.update(acts)

            for t in ir.mdp.dynamics.transitions:
                if t.guard and not self._eval(t.guard, ns):
                    continue
                for u in t.updates:
                    m = _DRAW.match(u)
                    if m:
                        target, src_name, stage_name = m.groups()
                        src = self._sources[src_name]
                        stage = next(s for s in src.stages if s.name == stage_name)
                        ns[target] = self._draw(src, stage, ns, period, episode_seed)
                    else:
                        self._exec(u, ns)

            comps = {
                c.name: float(self._eval(c.expr, ns))
                for c in ir.mdp.objective.per_step_components
            }
            comps["total"] = sum(comps.values())
            ns.update(comps)
            reward = float(self._eval(self._reward_mode.expr, ns))

            snap = lambda v: list(v) if isinstance(v, list) else v  # noqa: E731 — detach from live ns
            row: dict = {"t": period}
            row.update({d.name: snap(ns[d.name]) for d in ir.mdp.decisions})
            row.update({f: snap(ns[f]) for f in self._scalar_info})
            row.update({
                sv.name: snap(ns[sv.name])
                for sv in ir.mdp.state_variables
                if sv.role is not StateRole.time_index
            })
            row.update({c: comps[c] for c in comp_names})
            row["total"] = comps["total"]
            row["reward"] = reward
            traj.rows.append(row)
            traj.total += comps["total"]
            traj.total_reward += reward

            if early and self._eval(early, ns):
                traj.terminated_early = True
                break
            if int(ns[time_var]) >= self.T:
                break

        return traj

    def observe(self, mode_name: str, ns_row: dict) -> dict:
        """Evaluate an observation mode's features against a namespace (e.g.
        a trajectory row merged over constants) — what the agent would see."""
        mode = next(m for m in self.ir.gym.observation_modes if m.name == mode_name)
        ns = self._base_ns()
        ns.update(ns_row)
        out = {}
        for feat in mode.features:
            if feat.ref:
                name = feat.ref.split(".")[-1]
                out[feat.ref] = ns[name]
            else:
                out[feat.derived] = self._eval(feat.expr, ns)
        return out


def simulate(
    ir: MdpIR,
    episode_seed: int = 0,
    decisions: Callable[[dict], dict] | list[dict] | dict | None = None,
    instance: str | None = None,
    seed_salt: int = 0,
    max_periods: int | None = None,
) -> Trajectory:
    """One-call convenience wrapper around ``IrInterpreter``."""
    return IrInterpreter(ir, instance=instance, seed_salt=seed_salt).run(
        episode_seed, decisions=decisions, max_periods=max_periods
    )


def main(argv: list[str]) -> int:
    import argparse

    from mdp_ir.schema import load_ir

    ap = argparse.ArgumentParser(
        prog="python -m mdp_ir.interpreter",
        description="Execute an MDP-IR directly and print a sample trajectory.",
    )
    ap.add_argument("ir_file")
    ap.add_argument("--episode-seed", type=int, default=0)
    ap.add_argument("--seed-salt", type=int, default=0)
    ap.add_argument("--instance", default=None)
    ap.add_argument(
        "--decision", action="append", default=[], metavar="NAME=VALUE",
        help="hold a decision constant (others are random within bounds)",
    )
    ap.add_argument("--max-periods", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=20)
    args = ap.parse_args(argv)

    # instance flows into load_ir too: for a catalog schema it may select
    # slot candidates (IR_LAYERING_PLAN §10); run-time constant overrides
    # stay with IrInterpreter as always
    ir = load_ir(args.ir_file, instance=args.instance)
    interp = IrInterpreter(ir, instance=args.instance, seed_salt=args.seed_salt)

    fixed: dict[str, float] = {}
    for spec in args.decision:
        name, _, val = spec.partition("=")
        fixed[name.strip()] = float(val)
    unknown = set(fixed) - {d.name for d in ir.mdp.decisions}
    if unknown:
        ap.error(f"unknown decision(s) {sorted(unknown)}")

    if len(fixed) == len(ir.mdp.decisions):
        decisions: object = fixed
    elif fixed:
        rand = interp._random_policy(args.episode_seed)
        decisions = lambda ns: {**rand(ns), **fixed}  # noqa: E731
    else:
        decisions = None

    traj = interp.run(args.episode_seed, decisions=decisions, max_periods=args.max_periods)
    inst = f" instance={args.instance}" if args.instance else ""
    print(f"{ir.domain.name} v{ir.ir_version}  episode_seed={args.episode_seed}{inst}")
    print(traj.render(max_rows=args.max_rows))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))

"""Restricted interpreter that executes an MDP-IR directly — no generated code.

Two roles (plan §3, sample §2b):

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
  * Seeding follows the schema's *derived* key (``UncertaintyStage.seed_key``):
    ``[entity_id?, sub_stream?, stream_id, period?, episode_seed, seed_salt]``.
    Episode-realization draws are cached per episode; period/event draws are
    keyed on the current period, so realized uncertainty is decision-path
    independent by construction.
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

# the function whitelist matching the schema's _BUILTINS (minus keywords)
_FUNCS: dict[str, object] = {
    "min": min, "max": max, "sum": sum, "abs": abs, "len": len,
    "round": round, "int": int, "float": float,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "floor": math.floor, "ceil": math.ceil,
    "zeros": lambda n: [0] * int(n),
    # extension 2026-07-10 (topk_id): must stay in lockstep with schema._BUILTINS
    "phi": lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))),
    "topk": lambda values, k: sorted(values, reverse=True)[: int(k)],
    "range": range,
    # extension 2026-07-13 (topk_id): deterministic probit-MAP top-k selection.
    # The implementation lives with the domain that motivated it
    # (topk_id/topk_id_probit_map.py — one shared function, bit-exact by
    # construction); resolved lazily so mdp_ir keeps no hard domain dependency
    # and IRs that never call bayes_topk never trigger the import.
    "bayes_topk": lambda *args: _domain_builtin(
        "topk_id", "topk_id_probit_map", "bayes_topk"
    )(*args),
    "post_mu": lambda *args: _domain_builtin(
        "topk_id", "topk_id_probit_map", "post_mu"
    )(*args),
    "post_sd": lambda *args: _domain_builtin(
        "topk_id", "topk_id_probit_map", "post_sd"
    )(*args),
}


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

# policy stream: keeps random fallback decisions out of the model streams
_POLICY_STREAM = 9999


def _numeric_seed_key(
    stage: UncertaintyStage,
    stream_id: int,
    *,
    period: int,
    episode_seed: int,
    seed_salt: int,
    entity_id: int | None = None,
    key_vals: list[int] | None = None,
) -> list[int]:
    """Numeric form of ``UncertaintyStage.seed_key`` (same slot order):
    ``[entity_id?, key_exprs..., sub_stream?, stream_id, period?,
    episode_seed, seed_salt]`` — matching the reference generators (e.g.
    inv_single ``EpisodeDemand`` keys ``[sub, stream, period, episode_seed,
    seed_salt]``), so interpreter draws are bit-identical to conforming
    domain code. ``key_vals`` are the evaluated ``key_exprs`` of a keyed
    stage; keyed stages carry no period slot (the draw is a fixed
    per-episode latent table indexed by the key values)."""
    key: list[int] = []
    if entity_id is not None:
        key.append(entity_id)
    if key_vals:
        key += key_vals
    if stage.sub_stream is not None:
        key.append(stage.sub_stream)
    key.append(stream_id)
    if stage.realization not in (Realization.episode, Realization.keyed):
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
        self.constants: dict[str, object] = {
            c.name: c.value for c in ir.mdp.scenario.constants
        }
        if instance is not None:
            overrides = ir.mdp.scenario.instances.get(instance)
            if overrides is None:
                raise KeyError(
                    f"unknown scenario instance {instance!r}; "
                    f"have {sorted(ir.mdp.scenario.instances)}"
                )
            self.constants.update(overrides)
        self.T = ir.mdp.horizon_T(instance)
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
        ns.update(_FUNCS)
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
            key_vals=key_vals,
        )
        return np.random.default_rng(np.random.SeedSequence(key))

    def _draw(self, source: UncertaintySource, stage: UncertaintyStage,
              ns: dict, period: int, episode_seed: int, cache: dict) -> object:
        fam = source.distribution.family
        settings = source.distribution.settings

        if fam == "episode_categorical":
            support = self._episode_support(source, ns, episode_seed, cache)
            if stage.realization is Realization.episode:
                return support
            vals, probs = support
            rng = self._rng(source, stage, period, episode_seed, ns)
            return int(rng.choice(vals, p=probs))

        rng = self._rng(source, stage, period, episode_seed, ns)
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
            # extension 2026-07-15 (retailer): episode-level LogNormal market
            # size; settings are the underlying normal's (mean, sigma)
            return float(rng.lognormal(float(self._setting(settings["mean"], ns)),
                                       float(self._setting(settings["sigma"], ns))))
        if fam == "uniform":
            return float(rng.uniform(float(self._setting(settings["low"], ns)),
                                     float(self._setting(settings["high"], ns))))
        if fam == "bernoulli":
            # single-parameter family; accept any single setting key (p, p_high, …)
            (p,) = [self._setting(v, ns) for v in settings.values()]
            return int(rng.random() < float(p))
        raise NotImplementedError(f"distribution family {fam!r}")

    def _episode_support(self, source: UncertaintySource, ns: dict,
                         episode_seed: int, cache: dict) -> tuple:
        """Draw-once (episode stage) support for episode_categorical, using
        the inv_single EpisodeDemand recipe."""
        stage = next(
            st for st in source.stages if st.realization is Realization.episode
        )
        key = (source.name, stage.name)
        if key not in cache:
            s = int(self._setting(source.distribution.settings["support_size"], ns))
            lo = int(self._setting(source.distribution.settings["support_low"], ns))
            hi = int(self._setting(source.distribution.settings["support_high"], ns))
            rng = self._rng(source, stage, 0, episode_seed)
            vals = rng.choice(np.arange(lo, hi + 1), size=s, replace=False)
            raw = rng.uniform(1.0, 10.0, size=s)
            cache[key] = (vals.astype(int), raw / raw.sum())
        return cache[key]

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
        cache: dict = {}
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
                        ns[target] = self._draw(src, stage, ns, period, episode_seed, cache)
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

    ir = load_ir(args.ir_file)
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

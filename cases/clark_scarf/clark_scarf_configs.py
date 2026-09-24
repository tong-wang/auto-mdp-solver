"""The §CONFIG-REGISTRY data half — configuration ids as data (guide §13).

`§CONFIG-REGISTRY` graduated from a local extension to a **rule** at solver
v0.9.26 (upstream #60), on two campaigns: `game2048`, which invented it, and
`adi_flex`, which adopted the shape verbatim. This module is this campaign's
adoption.

**Authority split (§13.5).** This module is authoritative for *what a config
is* — a run resolves against it. `ESCALATION.md` §CONFIG-REGISTRY is
authoritative for *why* — what promoted an id, what it supersedes, what it
cost. Every id here owes a table row and every row an id here;
`clark_scarf_test.py` asserts that in both directions, because deliberate
redundancy is only safe when it is checked.

**Components, never bundles (§13.5).** There is no config object. A run cites a
*tuple* — `sc1/g4/a2/h0` — plus its declared deviations, and the cross product
is implicit. Registering bundles would make the axis that moved recoverable
only by diffing two definitions, which is the failure the axes exist to
prevent, and it would kill the query the registry exists for: *every arm that
used `a2`* is a grep, not an expansion.

**L0 is not here at all.** §8.6 defines it as the library's defaults plus what
the problem forces, so no id names it and an L0 run cites `{sc}/L0` (§13.4).

**What a run cites is not what a run is.** The origins are *read* from
`clark_scarf_ppo_train.py` — the derived value where §8.6 speaks, the parser
default where it does not — so `g0`/`a0`/`h0` cannot drift from the derivation
they name. `_L1_DERIVED` alone is not enough: it holds only what §8.6 derived,
which is right for a run name and wrong for an origin, and a knob no row
derives (`vf_coef`) would be silently absent, letting two runs differing only
there cite one tuple. **A config must be complete; a run name must be a diff.**

Read by `ast` rather than by import: the train script pulls in SB3 and torch,
and a registry that cannot be resolved without them is one no gate can run.
"""

from __future__ import annotations

import ast
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TRAIN = _HERE / "clark_scarf_ppo_train.py"


# --------------------------------------------------------------------------
# sc — aliases into SCENARIOS (§13.1: it REFERENCES, never defines)
# --------------------------------------------------------------------------
# Two cells, two frames. They are not comparable and the registry is where that
# stops being a convention: an `sc` id is the first component of every address,
# so a subtraction across frames is visible in the address before anyone reads
# a number (ESCALATION §MAP, both frames).
SC: dict[str, str] = {
    # the only cell this board studies. Every derived row in `_L1_DERIVED` was
    # measured here and the `dp` benchmark is role `exact` on it (brute-force
    # verified to 0.000000), so "% of optimal" means it.
    "sc0": "n3_l2_p09",
}


# --------------------------------------------------------------------------
# axis ownership (§13.1) — decided by the layer that implements the knob
# --------------------------------------------------------------------------
# `g` — the env as presented to the algorithm: the design axes plus the §8.3
# vec-env wrapper stack. The wrapper booleans are the easiest knobs here to
# misfile: they read as hyperparameters because they sit next to the learner,
# and they change what the agent SEES and what it is PAID.
G_OWNS = frozenset({"observation_mode", "action_mode",
                    "norm_obs", "norm_reward", "vecnorm_clip_obs",
                    # caps what the ENCODING may request; the IR's ship_max is
                    # untouched, so this is the gym's presentation, not the model
                    "order_max"})

# `a` — policy family, extractor, critic form. The test is *a custom class, not
# a constructor argument*: `--policy-dist beta|gamma` swaps the policy class
# itself, `--mask` swaps PPO for MaskablePPO, and `--beta-min-conc` /
# `--init-action-mean` configure that class at construction.
A_OWNS = frozenset({"policy_dist", "mask", "beta_min_conc", "init_action_mean",
                    # the ordinal head's two widths. They belong to `a` with
                    # `policy_dist`, not to `h`: they parameterize the policy
                    # CLASS, and a knob has one home (§13.5)
                    "tau_min", "tau_scale"})

# `h` — optimizer, schedules, rollout geometry, loss weights, epochs. `gamma`
# stays here even though §8.3 hands it to the normalizer: the wrapper CONSUMES
# the discount, it does not define it. `net_arch` width is an §8.6 row, so it is
# hp; a custom extractor would be `a` (there is none).
H_OWNS = frozenset({"learning_rate", "lr_final", "n_steps", "batch_size",
                    "n_epochs", "gamma", "gae_lambda", "ent_coef", "ent_final",
                    "clip_init", "clip_final", "vf_coef", "max_grad_norm",
                    "net_arch", "n_envs", "normalize_advantage"})

# not config (§13.2): a budget is recoverable from the run log and gets extended
# constantly; a seed is a replicate, not a design choice. A comparison must
# still STATE its budget — that is a reporting rule, not a config one.
EXCLUDED = frozenset({"seed", "total_timesteps"})

# protocol and diagnostics: they do not change the policy, so a run's identity
# must not depend on them (the same set `build_run_name` skips, plus `level`,
# which §13.4 makes derivable rather than authored)
PROTOCOL = frozenset({"scenario_name", "level", "tag", "outdir", "gym_log",
                      "checkpoint_every_frac", "report_every",
                      # the §13 launch surface itself (upstream #62 checks 4
                      # and 5). An address, its declared deviations and the
                      # comparator are how a run SAYS what it is; they are
                      # inputs to the check, never knobs it checks, and a
                      # registry that owned them could address itself
                      "config", "deviation", "comparator", "settling"})

# The §8.4 always-shown tier: values name a CELL, never a level (§13.4). An
# escalation below a cell may not overwrite one — that is a different cell's
# number produced under this cell's address, refused rather than recorded.
DESIGN_AXES = frozenset({"observation_mode", "action_mode"})


# --------------------------------------------------------------------------
# the id tables — parent + delta, append-only (§13.2)
# --------------------------------------------------------------------------
# An id is a PROMOTION, not a record that something ran (§13.2): rows exist for
# adopted bases — crowned, shipped, or parented — never for probes or failed
# arms, however fully replicated. Everything else is ledger-addressed and reads
# `unregistered cell` on the tree.
G: dict[str, tuple[str | None, dict]] = {
    "g0": (None, {}),                                   # L1 origin, computed
    "g1": ("g0", {"observation_mode": "echelon"}),      # the tier-2 comparison arm
    "g2": ("g0", {"action_mode": "ship_discrete"}),     # the echelon-free interface
    "g3": ("g1", {"action_mode": "ship_discrete"}),     # its echelon twin
}

A: dict[str, tuple[str | None, dict]] = {
    "a0": (None, {}),                                   # L1 origin: PPO + MlpPolicy
    "a1": ("a0", {"policy_dist": "ordinal"}),           # hurdle-dgauss (misnomer kept; #E6)
    "a2": ("a0", {"policy_dist": "dgauss"}),            # the plain body: adjacency pooling, no atom
    "a3": ("a0", {"policy_dist": "dgauss_sig"}),        # atomless twin of a1's body (sigmoid mu)
}

H: dict[str, tuple[str | None, dict]] = {
    "h0": (None, {}),                                   # L1 origin, read from the derivation
}

AXES: dict[str, dict[str, tuple[str | None, dict]]] = {"g": G, "a": A, "h": H}
OWNS: dict[str, frozenset] = {"g": G_OWNS, "a": A_OWNS, "h": H_OWNS}

# Cross-axis constraints (§13.3): declared on the CONSTRAINING id and refused at
# resolution, not left implicit. `--mask` needs a discrete head, and a
# continuous distribution family cannot be masked — the assertion already lives
# in the train script; declaring it here is what makes it bind a SAMPLER too.
REQUIRES: dict[str, list[tuple[str, object]]] = {
    # the ordinal head is a MultiDiscrete parameterization: it asserts the space
    # in its own `_build`, and declaring the force here refuses the citation at
    # launch instead of at model construction
    "a1": [("g.action_mode", ("ship_discrete",))],
    "a2": [("g.action_mode", ("ship_discrete",))],
    "a3": [("g.action_mode", ("ship_discrete",))],
}


class ConfigError(ValueError):
    """A citation that does not resolve, or a registry that does not hold."""


# --------------------------------------------------------------------------
# reading the train script (ast — the origins are read, never written)
# --------------------------------------------------------------------------
def _module() -> ast.Module:
    return ast.parse(_TRAIN.read_text())


def _literal(node: ast.AST, consts: dict) -> object:
    """A constant, a list of constants, or a module-level name (`BETA`)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(e, consts) for e in node.elts]
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand, consts)          # type: ignore[operator]
    return None


def _module_constants(tree: ast.Module) -> dict:
    out: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value.value
    return out


def l1_derived() -> dict:
    """The train script's `_L1_DERIVED` (spec §8.4), read as data."""
    tree = _module()
    consts = _module_constants(tree)
    for node in tree.body:
        names = ([t.id for t in node.targets if isinstance(t, ast.Name)]
                 if isinstance(node, ast.Assign)
                 else [node.target.id] if isinstance(node, ast.AnnAssign)
                 and isinstance(node.target, ast.Name) else [])
        if "_L1_DERIVED" in names and isinstance(node.value, ast.Dict):
            return {k.value: _literal(v, consts)
                    for k, v in zip(node.value.keys, node.value.values)
                    if isinstance(k, ast.Constant)}
    raise ConfigError("clark_scarf_ppo_train.py declares no _L1_DERIVED")


def cli_defaults() -> dict:
    """Every CLI dest the train script exposes, with its parser default."""
    tree = _module()
    consts = _module_constants(tree)
    out: dict = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        if "dest" in kw and isinstance(kw["dest"], ast.Constant):
            dest = kw["dest"].value
        else:
            flags = [a.value for a in node.args
                     if isinstance(a, ast.Constant) and str(a.value).startswith("--")]
            if not flags:
                continue
            dest = flags[0].lstrip("-").replace("-", "_")
        default = _literal(kw["default"], consts) if "default" in kw else None
        action = _literal(kw.get("action"), consts) if "action" in kw else None
        if default is None and action == "store_true":
            default = False
        out[dest] = default
    return out


# --------------------------------------------------------------------------
# origins, resolution
# --------------------------------------------------------------------------
def origin(axis: str) -> dict:
    """`g0`/`a0`/`h0` — the derivation's output on that axis, COMPLETE.

    Derived value where §8.6 speaks, parser default where it does not. This is
    the one place the ladder attaches to the registry (§13.4), and it is read
    rather than written so it cannot drift from the derivation it names.
    """
    base = cli_defaults()
    base.update(l1_derived())
    return {k: v for k, v in base.items() if k in OWNS[axis]}


def _fill_derived(cfg: dict, scenario_name: str) -> dict:
    """Fill the rows whose derived value is PER INSTANCE, in place.

    `gamma` is the only one, and it exists because F11 made beta a scenario
    field: §8.6's row is `gamma = beta`, so the train script leaves the CLI
    default None and computes it in `parse_args`, and `_L1_DERIVED` omits the
    key because no constant could hold it. `cli_defaults()` therefore reads
    None and `origin()` has a hole only the resolved scenario can fill.

    Used by BOTH `resolve` and `level`, which is the point: filling it in one
    and not the other makes every address read L2(hp), because gamma appears to
    have moved off a None baseline. Only fills where nothing has set a value —
    an id that searched gamma (h6) must win over the derivation.
    """
    # `in cfg`, not `.get(...) is None`: `origin('g')` and `origin('a')` do not
    # own gamma and must not GAIN it here, or every axis reads as moved and a
    # tuned-gamma id inflates from L2(hp) to L2(gym+arch+hp)
    if "gamma" in cfg and cfg["gamma"] is None:
        from clark_scarf_scenarios import SCENARIOS
        cfg["gamma"] = float(SCENARIOS[scenario_name].beta)
    return cfg


def resolve(tuple_str: str) -> dict:
    """Expand `sc1/g4/a1/h0` into one complete configuration."""
    parts = tuple_str.split("/")
    if len(parts) != 4:
        raise ConfigError(f"an address is sc/g/a/h, got {tuple_str!r}")
    sc, ids = parts[0], dict(zip("gah", parts[1:]))
    if sc not in SC:
        raise ConfigError(f"unknown scenario id {sc!r}")
    cfg: dict = {"scenario_name": SC[sc]}
    for axis, cid in ids.items():
        if cid not in AXES[axis]:
            raise ConfigError(f"unknown {axis} id {cid!r}")
        chain, cur = [], cid
        while cur is not None:
            chain.append(cur)
            cur = AXES[axis][cur][0]
        cfg.update(origin(axis))
        for node in reversed(chain):
            cfg.update(AXES[axis][node][1])
    _fill_derived(cfg, cfg["scenario_name"])
    for cid in ids.values():
        for knob, allowed in REQUIRES.get(cid, []):
            got = cfg.get(knob.split(".", 1)[1])
            ok = got in allowed if isinstance(allowed, tuple) else got == allowed
            if not ok:
                raise ConfigError(
                    f"{cid} requires {knob} in {allowed!r}, the citation "
                    f"resolves to {got!r} — a forced knob is declared on the "
                    f"constraining id and refused here, never resolved silently")
    return cfg


def level(tuple_str: str, deviations: dict | None = None) -> str:
    """Read the level off the knobs that MOVED, never off the id (§13.4).

    A design-axis value names a cell, not a level: `sc0/g2/a0/h0` is L1 in its
    own cell, and only a non-design knob off its origin makes an escalation.
    """
    cfg = resolve(tuple_str)
    moved = set()
    for axis in "gah":
        base = _fill_derived(origin(axis), cfg["scenario_name"])
        for knob, value in base.items():
            if knob in DESIGN_AXES:
                continue
            if cfg.get(knob) != value:
                moved.add(axis)
    for knob in (deviations or {}):
        for axis in "gah":
            if knob in OWNS[axis] and knob not in DESIGN_AXES:
                moved.add(axis)
    if not moved:
        return "L1"
    names = {"g": "gym", "a": "arch", "h": "hp"}
    return f"L2({'+'.join(names[a] for a in 'gah' if a in moved)})"


# --------------------------------------------------------------------------
# what the module checks at IMPORT (§13.5) — a rule with no enforcer is prose
# --------------------------------------------------------------------------
def _check_one_knob_one_home() -> None:
    homes: dict[str, list[str]] = {}
    for axis, owned in OWNS.items():
        for knob in owned:
            homes.setdefault(knob, []).append(axis)
    doubled = {k: v for k, v in homes.items() if len(v) > 1}
    if doubled:
        raise ConfigError(f"knobs in two axes: {doubled} — the partition is by "
                          f"implementing layer, so no dest may appear twice")
    known = set().union(*OWNS.values()) | EXCLUDED | PROTOCOL
    unowned = sorted(set(cli_defaults()) - known)
    if unowned:
        raise ConfigError(
            f"CLI dest(s) {unowned} belong to no axis and are not excluded — a "
            f"knob the registry cannot express is a run it cannot address")


def _check_ids_dense() -> None:
    for axis, table in AXES.items():
        want = {f"{axis}{i}" for i in range(len(table))}
        if set(table) != want:
            raise ConfigError(f"{axis} ids are not dense from the origin: "
                              f"{sorted(table)} against {sorted(want)}")
        for cid, (parent, _) in table.items():
            if parent is not None and parent not in table:
                raise ConfigError(f"{cid} parents unknown id {parent!r}")
        if table[f"{axis}0"][0] is not None:
            raise ConfigError(f"{axis}0 is the L1 origin and has no parent")


def _check_deltas_owned() -> None:
    for axis, table in AXES.items():
        for cid, (_, delta) in table.items():
            stray = sorted(set(delta) - OWNS[axis])
            if stray:
                raise ConfigError(f"{cid}'s delta sets {stray}, which the {axis} "
                                  f"axis does not own")
            if cid.endswith("0") and delta:
                raise ConfigError(f"{cid} is an origin; its delta is computed "
                                  f"from the derivation, never written")


def _check_scenarios_live() -> None:
    try:
        from clark_scarf_scenarios import SCENARIOS
    except ImportError:                                   # pragma: no cover
        return                        # not importable from here; the test asserts it
    missing = sorted(k for k in SC.values() if k not in SCENARIOS)
    if missing:
        raise ConfigError(f"sc alias(es) name {missing}, which SCENARIOS does "
                          f"not have — an alias may never outlive its instance")


def _check_design_axis_values() -> None:
    """Every design-axis value a `g` delta sets must be DECLARED in the IR.

    Guide §3.1 (upstream #65) makes a `design-axes` split the drawing of an IR
    declaration; the same discipline binds the registry, because an id citing a
    mode the IR does not declare is an address for a problem nobody froze.
    """
    for cid, (_, delta) in G.items():
        for knob, value in delta.items():
            if knob not in DESIGN_AXES:
                continue
            declared = ir_modes()[knob]
            if declared and value not in declared:
                raise ConfigError(
                    f"{cid} sets {knob}={value!r}, which the IR does not "
                    f"declare (has {sorted(declared)}) — declare the mode in "
                    f"clark_scarf_schema.json or leave the cell unregistered")


def ir_modes() -> dict[str, set[str]]:
    """`observation_mode` / `action_mode` values the IR declares."""
    try:
        from mdp_ir import load_ir
        ir = load_ir(str(_HERE / "clark_scarf_schema.json"))
    except Exception:                                     # pragma: no cover
        return {"observation_mode": set(), "action_mode": set()}
    return {"observation_mode": {m.name for m in ir.gym.observation_modes},
            "action_mode": {m.name for m in ir.gym.action_modes}}


def gym_only_modes() -> dict[str, set[str]]:
    """Modes the GYM implements that the IR does not declare.

    Spec §7 says the menu holds in both directions, so this set should be
    EMPTY and `test_registered_design_axis_values_are_declared_in_the_ir`
    asserts that it is. Two modes were in that state on the previous board
    (`ship_fraction_bins`, `ship_scaled`), reported for a whole campaign and
    never closed; this board removed them from the gym rather than carry a
    rendering nobody froze.
    """
    tree = ast.parse((_HERE / "clark_scarf_gym.py").read_text())
    found = {"observation_mode": set(), "action_mode": set()}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        key = ("observation_mode" if "OBS_MODES" in names
               else "action_mode" if "ACTION_MODES" in names else None)
        if key and isinstance(node.value, (ast.Tuple, ast.List)):
            found[key] = {e.value for e in node.value.elts
                          if isinstance(e, ast.Constant)}
    declared = ir_modes()
    return {k: (v - declared[k]) if declared[k] else set() for k, v in found.items()}


_check_one_knob_one_home()
_check_ids_dense()
_check_deltas_owned()
_check_scenarios_live()
_check_design_axis_values()


if __name__ == "__main__":                                # pragma: no cover
    print("origins (read from clark_scarf_ppo_train.py):")
    for axis in "gah":
        print(f"  {axis}0: {origin(axis)}")
    print("\nids:")
    for axis, table in AXES.items():
        for cid, (parent, delta) in table.items():
            print(f"  {cid:4} parent={parent or '—':4} delta={delta or '(the derivation)'}")
    print("\nresolution check:")
    # addresses from THIS board's registry -- the h axis holds only its origin
    # until a tuning winner is adopted, so no h1 exists to resolve (#E10)
    for addr in ("sc0/g2/a2/h0", "sc0/g3/a1/h0"):
        print(f"  {addr} -> {level(addr)}")
    gaps = {k: sorted(v) for k, v in gym_only_modes().items() if v}
    print(f"\ngym modes the IR does not declare: {gaps or 'none'}")

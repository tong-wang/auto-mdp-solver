"""The §CONFIG-REGISTRY data half — configuration ids as data (guide §12/§13).

**Authority split.** This module is authoritative for *what a config is* — a
run resolves against it (`--config sc0/g0/a0/h0`). `ESCALATION.md`
§CONFIG-REGISTRY is authoritative for *why* — what promoted an id, what it
supersedes, what it cost. Every id here must have a table anchor there and
vice versa; the v0.9.31 launch check asserts both directions when the train
script hands it `all_ids()`.

**Components, never bundles.** There is no config object. A run cites a
*tuple* — `sc2/g1/a0/h0` — plus its deviations, and the cross product is
implicit. `{sc}/L1` is the canonical spelling of `{sc}/g0/a0/h0`.

**L0 is not here at all.** §8.6 defines it (the library's defaults plus what
the problem forces), so no id names it and an L0 run records `{sc}/L0`.

Born with the campaign restart (2026-08-27), so unlike its adi_flex parent it
starts at the origins: one generation, no escalation ids yet — those are
minted by runs, never in advance.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# --- sc: aliases into SCENARIOS / GRIDS ------------------------------------
# `sc` references, never defines: the alias is recorded here and an import-time
# check asserts the key is still live, so an alias can never outlive its
# target or silently re-point. A retired alias is never reused.
SC: dict[str, str] = {
    "sc0": "simple",         # the _L1_BASIS scenario; grid16 cell (b=9,K=0,LT=0)
    "sc1": "simple_k",       # K=20 — the (s,S) regime (RQ-1)
    "sc2": "lt",             # deterministic L=2 (RQ-2)
    "sc3": "slt",            # stochastic L~uniform{1,2,3}, E[L]=2 (RQ-2/RQ-4)
    "sc4": "lost_sales",     # lost sales, L=0 control (RQ-3)
    "sc5": "lt_lost_sales",  # lost sales, L=2 — the discover target (RQ-3)
    "sc6": "grid16",         # GRIDS entry — the RQ-6 generality target
    "sc7": "lt_variance_k0",   # GRIDS entry — RQ-4's generality target, K=0
    "sc8": "lt_variance_k20",  # GRIDS entry — the same family at K=20
}

# --- axis ownership, declared and then enforced ----------------------------
# Every dest the train script exposes is owned by exactly one axis, excluded
# as a replicate/budget, or protocol — asserted at import, so a knob added to
# the CLI and to no axis is a registry failure before it can produce an
# inexpressible run.
G_OWNS = frozenset({"observation_mode", "action_mode",
                    "norm_obs", "norm_reward", "vecnorm_clip_obs"})
H_OWNS = frozenset({"learning_rate", "lr_final", "clip_init", "clip_final",
                    "n_steps", "batch_size", "n_epochs", "n_envs",
                    "gamma", "gae_lambda", "ent_coef", "vf_coef",
                    "max_grad_norm", "target_kl", "net_arch",
                    "normalize_advantage"})
R3_EXCLUDED = frozenset({"seed", "total_timesteps"})    # not configuration
PROTOCOL = frozenset({"scenario_name", "grid_name", "level", "tag", "outdir",
                      "progress_bar", "gym_log", "checkpoint_every_frac",
                      "report_every", "config", "comparator", "settling"})

# --- g: the env as presented to the algorithm ------------------------------
# The design axes (observation_mode, action_mode — cells, never levels) plus
# the §8.3 vec-env wrapper stack. g0's delta is COMPUTED from the train
# script's parser defaults + _L1_DERIVED, not written, so it cannot drift.
G: dict[str, tuple[str | None, dict]] = {
    "g0": (None, {}),                               # computed below
    "g1": ("g0", {"observation_mode": "vec_ip"}),   # RQ-2's sufficient statistic
    "g2": ("g0", {"observation_mode": "vec_ctx"}),  # RQ-6's context generalist
    # #E2: PPO reached only 93.1% of a bar base-stock attains exactly at
    # `simple`, where the optimum is a threshold rule. Obs normalization is
    # the suspect — VecNormalize rescales `period` and `inventory` by running
    # moments, so the lattice a threshold sits on moves while the policy is
    # still learning where the threshold is.
    "g4": ("g0", {"norm_obs": False}),              # #E3: obs-norm off
    # NOTE g3 and g5 are WITHDRAWN (see WITHDRAWN below), not reused. They were minted for the
    # continuous-head arms and never adopted; guide §13.2 mints a row for a
    # base that is crowned, shipped or parented, never for a probe. The
    # numbers stay unoccupied so #E4's historical citation still resolves to
    # "an id that was withdrawn" rather than to some later cell.
}

# --- a: policy family, extractor, critic form ------------------------------
# `algo_class` and `policy` are not CLI dests — they are facts the run records
# (§8.4 provenance) and this axis constrains against.
A: dict[str, tuple[str | None, dict]] = {
    "a0": (None, {"algo_class": "PPO", "policy": "MlpPolicy"}),
    # ADOPTED #E9. The ordinal (mixture-at-zero) order head: 3 parameters induce
    # all q_high+1 logits, so the location gradient pools every sample instead of
    # nudging one category's logit. It wins on both targets and is what the
    # crowned artifacts run.
    "a1": ("a0", {"policy": "ordinal"}),
}
# The `a` axis gained a CLI knob when the ordinal head landed: `policy` selects
# the policy CLASS (guide §13.1's own test for the axis), so it is settable and
# diffable like any other. `algo_class` stays non-CLI — it is a fact the run
# records for §8.4 provenance, not a flag.
A_OWNS = frozenset({"policy"})
NON_CLI = frozenset({"algo_class"})

# --- h: optimizer, schedules, rollout geometry, loss weights ---------------
# h0's delta is not written here: it is read from the train script's
# `_L1_DERIVED` (§8.4) plus parser defaults where no row derives a value, so
# the origin cannot drift from the derivation it is supposed to be.
H: dict[str, tuple[str | None, dict]] = {
    "h0": (None, {}),                               # computed below
    # ADOPTED 2026-08-27 by #E5. The 2x2x2 over the normalization triple makes
    # `sc/g4/a0/h1` the campaign operating point on both targets — g4 already
    # carries (norm_obs off, norm_reward on), so only the advantage half is new.
    # The effect is small and one-sided: -4.01 at `simple` (t=-23), -0.05 at
    # `simple_k`. It is adopted because it never loses (8/8 holdings, both
    # targets), not because it is large.
    "h1": ("h0", {"normalize_advantage": False}),
    # ADOPTED #E9 — the tuned hp sets, one per target. They differ because the
    # two problems want opposite exploration: `simple`'s optimum is a single
    # sharp order-up-to level, `simple_k` has a trigger to discover (#E8). Both
    # read off the winning TRIAL'S OWN args log, never rebuilt from a printed
    # cfg (spec §8.6: a trial is the derivation plus the searched delta).
    "h2": ("h1", {                                   # crowned on `simple`
        "learning_rate": 1.6246311347329354e-05, "lr_final": 1.6246e-06,
        "ent_coef": 5.782445444250473e-06, "gae_lambda": 0.972416,
        "n_steps": 2048, "n_epochs": 20, "batch_size": 32,
        "vf_coef": 0.4779015401552382, "net_arch": [128, 128, 128]}),
    "h3": ("h1", {                                   # crowned on `simple_k`
        "learning_rate": 1.635665121506356e-05, "lr_final": 1.6357e-06,
        "ent_coef": 4.568239779936761e-08, "gae_lambda": 0.813899,
        "n_steps": 4096, "n_epochs": 20, "batch_size": 128,
        "vf_coef": 0.6105895200528771, "net_arch": [256, 256, 256, 256]}),
}

# Ids minted and then withdrawn, never adopted. Guide §13.2 mints a row for a
# base that is crowned, shipped or parented — never for a probe or a failed arm
# — and g3/g5 were minted ahead of that for the continuous-head arms, which the
# campaign then withdrew (RQ-5 -> §9 Q-B) and whose results were deleted.
#
# They are recorded rather than deleted because ids are append-only: the slot
# must stay OCCUPIED so it is never reused, and #E4's historical citation must
# resolve to "withdrawn" rather than silently to some later cell. They satisfy
# the density rule and are refused by `resolve` and `parse_tuple`.
WITHDRAWN: dict[str, str] = {
    "g3": "minted for the continuous head (then RQ-5, now §9 Q-B); never adopted, "
          "arms parked and results deleted 2026-08-27",
    "g5": "g3 + norm_obs=False, same withdrawal",
}

# cross-axis constraints (R1c): none yet — minted by escalations, not ahead.
REQUIRES: dict[str, dict[str, object]] = {}

AXES = {"sc": SC, "g": G, "a": A, "h": H}
_ORIGINS = {"g": "g0", "a": "a0", "h": "h0"}


# --- ast readers: one definition of every boundary -------------------------

def _train_source() -> str:
    return (_HERE / "inv_single_ppo_train.py").read_text()


def _from_train_source(name: str):
    """A module-level literal read out of the train script without importing
    it (it pulls in torch; the same reason mdp_conformance reads it by ast)."""
    for node in ast.parse(_train_source()).body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            value = node.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id in ("frozenset", "set", "tuple", "list")
                    and len(value.args) == 1):
                return ast.literal_eval(value.args[0])
            return ast.literal_eval(value)
    raise RuntimeError(f"inv_single_ppo_train.py declares no {name}")


def _train_dests() -> set[str]:
    """Every dest the train script's parser exposes, by ast."""
    out = set()
    for node in ast.walk(ast.parse(_train_source())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        dest = next((kw.value.value for kw in node.keywords if kw.arg == "dest"), None)
        if dest is None:
            flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
            long = [f for f in flags if f.startswith("--")] or flags
            dest = long[0].lstrip("-").replace("-", "_")
        out.add(dest)
    return out


def _argument_default(dest: str):
    """`(declared, default)` for one dest's add_argument call, by ast."""
    for node in ast.walk(ast.parse(_train_source())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        this = None
        default = None
        for kw in node.keywords:
            if kw.arg == "dest":
                this = kw.value.value
            elif kw.arg == "default":
                try:
                    default = ast.literal_eval(kw.value)
                except ValueError:
                    default = None
        if this is None:
            flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
            long = [f for f in flags if f.startswith("--")] or flags
            this = long[0].lstrip("-").replace("-", "_")
        if this == dest:
            return True, default
    return False, None


_DERIVED = _from_train_source("_L1_DERIVED")
DESIGN_AXES = frozenset(_from_train_source("_DESIGN_AXES"))


def _origin(owned: frozenset) -> dict:
    """An origin must be COMPLETE, or a citation cannot express a run: the
    derived value where §8.6 speaks, the parser default where it does not —
    both READ from the script, so neither is a hand-copy that can drift."""
    out = {}
    for dest in sorted(owned):
        if dest in _DERIVED:
            out[dest] = _DERIVED[dest]
        else:
            declared, default = _argument_default(dest)
            if not declared:
                raise RuntimeError(f"{dest} is owned by an axis but the train "
                                   f"script exposes no such argument")
            out[dest] = default
    return out


G["g0"] = (None, _origin(G_OWNS))
H["h0"] = (None, _origin(H_OWNS))


# --- resolution ------------------------------------------------------------

def _walk(axis: dict[str, tuple[str | None, dict]], cid: str) -> dict:
    """Resolve one id by walking to its origin and applying deltas outward."""
    chain, seen = [], set()
    while cid is not None:
        if cid in seen:
            raise ValueError(f"cycle in the registry at {cid!r}")
        if cid in WITHDRAWN:
            raise KeyError(
                f"config id {cid!r} was WITHDRAWN and is not a base: "
                f"{WITHDRAWN[cid]}. The slot stays occupied so it is never "
                f"reused; cite an adopted id and declare the rest as deviations")
        if cid not in axis:
            raise KeyError(f"unknown config id {cid!r}")
        seen.add(cid)
        parent, delta = axis[cid]
        chain.append(delta)
        cid = parent
    out: dict = {}
    for delta in reversed(chain):
        out.update(delta)
    return out


def resolve(sc: str, g: str, a: str, h: str) -> dict:
    """The knob values a cited tuple denotes. Order-independent: the axes'
    deltas are disjoint, asserted at import, so no key can be written twice."""
    if sc not in SC:
        raise KeyError(f"unknown scenario id {sc!r}")
    target = SC[sc]
    out = {"scenario_name": target,
           "grid_name": target if _is_grid(target) else None}
    for axis, cid in ((G, g), (A, a), (H, h)):
        out.update(_walk(axis, cid))
    return out


def parse_tuple(spec: str) -> tuple[str, str, str, str]:
    """`sc0/g1/a0/h0` -> the four ids, in any order within the string.

    `{sc}/L1` is the canonical spelling of `{sc}/g0/a0/h0`. `{sc}/L0` is
    refused: §8.6 defines L0, so there is no derived config to name — an L0
    run passes L0's flags and records `{sc}/L0` as its address.
    """
    parts = [p for p in spec.replace(",", "/").split("/") if p]
    if any(p.upper() == "L0" for p in parts):
        raise ValueError(
            f"{spec!r}: L0 is not configured from the registry — §8.6 defines "
            f"it as the library's defaults plus what the problem forces. "
            f"Launch it with --level L0 and L0's flags")
    if any(p.upper() == "L1" for p in parts):
        parts = [p for p in parts if p.upper() != "L1"]
        parts += [cid for axis, cid in _ORIGINS.items()
                  if not any(p in AXES[axis] for p in parts)]
    picked: dict[str, str] = {}
    for p in parts:
        for name, axis in AXES.items():
            if p in axis:
                if name in picked:
                    raise ValueError(f"two {name} ids in {spec!r}")
                picked[name] = p
                break
        else:
            if p in WITHDRAWN:
                raise ValueError(
                    f"{p!r} was WITHDRAWN and is not a base: {WITHDRAWN[p]}. "
                    f"The slot stays occupied so it is never reused; cite an "
                    f"adopted id and declare the rest as deviations")
            raise ValueError(f"{p!r} is not a registered config id")
    missing = [k for k in ("sc", "g", "a", "h") if k not in picked]
    if missing:
        raise ValueError(f"{spec!r} names no {'/'.join(missing)} id — a run "
                         f"is sc + g + a + h + deviations + seed")
    return picked["sc"], picked["g"], picked["a"], picked["h"]


def all_ids() -> set[str]:
    """Every id the data half KNOWS — what the launch check asserts against the
    log's §CONFIG-REGISTRY anchors, in both directions.

    Withdrawn ids count. The table documents them precisely so a reader meeting
    a historical citation can look the number up and find "withdrawn"; if they
    were absent here, the both-directions check would read the table as
    describing bases that configure nothing and refuse every launch. They are
    known ids that are not bases — `resolve`/`parse_tuple` still refuse them."""
    return set(SC) | set(G) | set(A) | set(H) | set(WITHDRAWN)


# the union of governed dests: what the launch check requires a cited base to
# be total over (an owned-but-unfixed knob is invisible in both directions)
OWNED = G_OWNS | A_OWNS | H_OWNS


def _is_grid(name: str) -> bool:
    from inv_single_grids import GRIDS
    return name in GRIDS


# --- import-time self-check ------------------------------------------------

def _self_check() -> None:
    """The registry's structural rules, enforced at import rather than
    documented: aliases live, one boundary one definition, ownership total
    and disjoint, deltas inside their axis, origins complete, ids dense."""
    from inv_single_scenarios import SCENARIOS
    from inv_single_grids import GRIDS
    live = set(SCENARIOS) | set(GRIDS)
    dead = {k: v for k, v in SC.items() if v not in live}
    if dead:
        raise RuntimeError(f"sc alias(es) point at targets neither SCENARIOS "
                           f"nor GRIDS has: {dead}")

    stray = set(_from_train_source("_GYM_KNOBS")) - G_OWNS
    if stray:
        raise RuntimeError(f"the train script calls {sorted(stray)} gym knobs "
                           f"and G_OWNS does not — one boundary, two definitions")
    if not DESIGN_AXES <= G_OWNS:
        raise RuntimeError(f"design axes {sorted(DESIGN_AXES - G_OWNS)} not "
                           f"owned by g — the cell coordinates live in g")

    # cells REFERENCE the IR's gym block, never define: every observation_mode
    # a g id sets must be a declared mode; every action_mode value must be
    # a declared action-mode *type*: the script's selector vocabulary
    # ('continuous'/'discrete') selects the IR mode of that type. There is no
    # 'auto' any more — it derived 'discrete' for every scenario this domain
    # ships, so it was a second name for a value, never a third mode.
    from mdp_ir.schema import load_ir
    _gym = load_ir(_HERE / "inv_single_schema.json").gym
    obs_declared = {m.name for m in _gym.observation_modes}
    act_declared = {str(m.type.value) for m in _gym.action_modes}
    for cid, (_, delta) in G.items():
        if "observation_mode" in delta and delta["observation_mode"] not in obs_declared:
            raise RuntimeError(
                f"{cid} sets observation_mode={delta['observation_mode']!r}, "
                f"which the IR does not declare ({sorted(obs_declared)})")
        if "action_mode" in delta and delta["action_mode"] not in act_declared:
            raise RuntimeError(
                f"{cid} sets action_mode={delta['action_mode']!r}, which "
                f"selects no declared IR action mode ({sorted(act_declared)})")

    # ownership partition: total and disjoint over the CLI surface
    dests = _train_dests()
    groups = {"g": G_OWNS, "a": A_OWNS, "h": H_OWNS,
              "R3": R3_EXCLUDED, "protocol": PROTOCOL}
    names = list(groups)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            both = groups[x] & groups[y]
            if both:
                raise RuntimeError(f"{sorted(both)} claimed by both {x} and "
                                   f"{y} — one knob, two homes")
    claimed = G_OWNS | A_OWNS | H_OWNS | R3_EXCLUDED | PROTOCOL
    orphan = dests - claimed
    if orphan:
        raise RuntimeError(
            f"{sorted(orphan)} are train-script arguments no axis owns and "
            f"neither the replicate/budget exclusion nor the protocol list "
            f"covers — a run setting one could not be expressed as "
            f"sc+g+a+h+deviations+seed")
    ghost = claimed - dests
    if ghost:
        raise RuntimeError(f"{sorted(ghost)} are owned but the train script "
                           f"exposes no such argument")

    # deltas stay inside their axis; origins are complete
    for name, axis, owns in (("g", G, G_OWNS), ("h", H, H_OWNS)):
        for cid, (_, delta) in axis.items():
            stray = set(delta) - owns
            if stray:
                raise RuntimeError(f"{cid} sets {sorted(stray)}, which {name} "
                                   f"does not own")
        missing = owns - set(axis[f"{name}0"][1])
        if missing:
            raise RuntimeError(f"{name}0 names no value for {sorted(missing)} "
                               f"— an origin must be COMPLETE")

    # parser defaults and the derivation may not drift apart: the script's
    # defaults ARE the L1 centre (spec §8.6), so a promoted value is edited
    # into _L1_DERIVED and the default together, never one of them
    for dest, want in _DERIVED.items():
        declared, default = _argument_default(dest)
        if declared and default is not None and default != want:
            raise RuntimeError(
                f"parser default {dest}={default!r} != _L1_DERIVED "
                f"{want!r} — the script's defaults are the L1 centre; "
                f"promote a value into both or neither")

    # id grammar: `{axis}{int}` with an optional generation letter, dense per
    # generation (a re-derivation mints g0b under L1b, never reuses g0)
    for name, axis in (("g", G), ("a", A), ("h", H)):
        gens: dict[str, list[int]] = {}
        for cid in list(axis) + [w for w in WITHDRAWN if w.startswith(name)
                                 and not w[len(name):len(name)+1].isalpha()]:
            m = re.fullmatch(rf"{name}(\d+)([a-z]?)", cid)
            if not m:
                raise RuntimeError(f"{cid!r} is not `{name}{{int}}[{{gen}}]`")
            gens.setdefault(m.group(2), []).append(int(m.group(1)))
        for gen, nums in gens.items():
            if sorted(nums) != list(range(len(nums))):
                raise RuntimeError(f"{name} ids in generation "
                                   f"{gen or '(first)'} not dense: {sorted(nums)}")
        for cid in axis:
            _walk(axis, cid)
    for sc in SC:
        if not re.fullmatch(r"sc\d+[a-z]?", sc):
            raise RuntimeError(f"{sc!r} is not `sc{{int}}[{{gen}}]`")


_self_check()


if __name__ == "__main__":
    print(f"ids: {sorted(all_ids())}")
    for sc in sorted(SC):
        print(f"  {sc} -> {SC[sc]}")
    for name, axis in (("g", G), ("a", A), ("h", H)):
        for cid in sorted(axis):
            parent, delta = axis[cid]
            shown = delta if cid not in ("g0", "h0") else f"<origin, {len(_walk(axis, cid))} knobs>"
            print(f"  {cid} (<- {parent}): {shown}")
    print("self-check: OK")

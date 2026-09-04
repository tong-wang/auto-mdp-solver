"""The §CONFIG-REGISTRY data half — configuration ids as data.

Guide §12 / upstream auto-mdp-solver#60, whose rules R1–R6 this implements.
That issue is **deferred, not accepted**: one of its three deferral grounds is
that this module "is built in no worktree", so R1a–R1d are untested. This file
is that test, in a domain deliberately unlike the one that proposed it.

**Authority split (R1d).** This module is authoritative for *what a config is*
— a run resolves against it. `ESCALATION.md` §CONFIG-REGISTRY is authoritative
for *why* — what promoted an id, what it supersedes, what it cost. Every id
here has a table row and every row has an id here; `adi_flex_level_audit.py`
asserts that, because deliberate redundancy is only safe when it is checked.

**Components, never bundles (R1).** There is no config object. A run cites a
*tuple* — `sc0/g2/a0/h2` — plus its deviations, and the cross product is
implicit. Registering bundles would make the axis that moved recoverable only
by diffing two definitions, which is the failure the axes exist to prevent.

**L0 is not here at all.** §8.6 defines it, so no id names it and a run cites
`{sc}/L0` (guide §12's own ladder table, and F44).
"""

from __future__ import annotations

import ast
import collections
import functools
import itertools
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# --- sc: aliases into SCENARIOS (R1b) --------------------------------------
# `sc` references, never defines — so it carries two obligations the defining
# axes do not: the alias is recorded here, and an import-time check asserts the
# key is still live in SCENARIOS, so an alias can never outlive its instance or
# silently re-point at a different one. A retired alias is never reused (R4).
SC: dict[str, str] = {
    "sc0": "homog_L0_T2",     # the _L1_BASIS scenario — every derived row was measured here
    "sc1": "homog_L0_Tdl1",   # RQ4's Case-1 board (F26)
    "sc2": "homog_L0_Tdl0",   # the T = 0 null control, off-tree fixture (F27)
    # A1 (#E9), minted only once its 18 runs existed — the second attempt at this
    # id and the first legitimate one; the first minted it "reserved for A1" and
    # was withdrawn (F44). The headline branch, and the ONLY sc where the
    # allocation is live: alloc_enabled, n_alloc = 2, so a period is 3 agent
    # steps (F11) and T~ = 36 against every other sc's 30. Every h0 row was
    # derived at T~ 30; A1 re-checked each T~-dependent one rather than
    # re-deriving, so no L1 generation was minted.
    "sc3": "het_exp4",
    # A23 (#E17), minted once its 36 runs existed. The THIRD leaderboard: T_dl
    # 2 -> 3 and L 0 -> 1 move together, so no number here may be read against
    # sc3 or the homog boards (F10). First board where the general machinery
    # runs rather than a degenerate case of it — n_sigma = 2, so the protection
    # cascade is not a scalar; L = 1, so IP is not on-hand. T~ = 12 for every
    # run on it, because the board carries ONE encoding (F53): `seq_mask` was
    # dropped after A23 measured it, so T~ no longer differs between two arms
    # being compared. The derivation is still the T~ 30 one and its rows were
    # NOT re-derived — see F53 for the one row (gae_lambda) where that bites.
    "sc4": "het3_exp2",
}

# --- the design axes: cells, not levels (F49/F50) ---------------------------
# §8.4's always-shown tier and the IR gym block's enumerated mode families —
# the axes "determined by the research questions", fixed at Phase A. On the
# tree they are the `design-axes` kind (the root region, with `cases`); below
# them sit the `escalations`. Two rules bind them here:
#   read side  (F49): a design-axis value can never make a level;
#   write side (F50): an escalation below a cell can never overwrite one — a
#                     deviation touching a design axis is refused at launch.
# Read from the train script (one definition), values validated against the IR
# at import (cells REFERENCE the gym block, never define — the sc pattern).
DESIGN_AXES: frozenset  # filled below, after the ast readers exist

# --- axis ownership (R1a), declared and then enforced ----------------------
# Which axis OWNS each training knob. This is the partition R1a describes, and
# declaring it is what lets R2 be checked rather than hoped for: every dest the
# train script exposes is owned by exactly one axis, excluded by R3, or is
# protocol — asserted at import, so a knob added to the CLI and to no axis is a
# registry failure reported before it can produce an inexpressible run.
G_OWNS = frozenset({"observation_mode", "action_mode", "reward_mode",
                    "norm_obs", "norm_reward", "vecnorm_clip_obs"})
H_OWNS = frozenset({"learning_rate", "lr_final", "n_envs", "n_steps",
                    "batch_size", "n_epochs", "gamma", "gae_lambda",
                    "clip_init", "clip_final", "ent_coef", "vf_coef",
                    "target_kl", "net_arch", "normalize_advantage"})
# `a` owns the policy-distribution parameterization. `order_head` is the ORDER
# decision's head (#E22): categorical = one logit per quantity (the origin),
# ordinal = trigger + discretized-Gaussian location/width. An arch knob like
# net_arch — except net_arch predates the layer split and stays `h` per §8.6's
# own row, so historical levels do not re-read (F42).
A_OWNS = frozenset({"order_head"})
R3_EXCLUDED = frozenset({"seed", "total_timesteps"})       # not config (R3)
PROTOCOL = frozenset({"scenario_name", "level", "tag", "outdir", "progress_bar",
                      "gym_log", "checkpoint_every_frac", "config"})
# --- instruments: not config, not protocol, and NOT a deviation ------------
# A research instrument REPLACES a decision rule wholesale rather than
# configuring the learner: `--fixed-order-ap` takes the order from the solved AP
# table and leaves only the protection to the agent (#E13), which is why its own
# module states it cannot be an action mode — schema v1 requires a mode to drive
# every declared decision, and this drives one. So it is neither `g` (it is not
# the env as PRESENTED, it is the env with a different actor inside) nor a
# deviation: F50's rule is that an address-forming coordinate is never a
# deviation, and two runs at identical knobs that differ in who chooses the
# order are not the same run at a moved knob. It forms the CELL, like a design
# axis, but it is not one — the IR does not declare it and never can.
#
# Consequence, and it is the intended one: an instrument run's cell has no `g`
# id, so R2 addresses it the way it addresses `vec_mip2` and the `_plus_*`
# probes — by its ledger entry and its `tag`, never by a tuple (F44). Adding
# this set is what un-breaks the audit: `fixed_order_ap` was owned by nothing
# from the moment A19 added the flag, so `_self_check` raised on import and R2
# went unverified across every run launched since.
INSTRUMENT = frozenset({"fixed_order_ap"})

# --- g: the env as presented to the algorithm ------------------------------
# observation_mode, action_mode, reward_mode and the §8.3 vec-env wrapper stack.
# g0's delta is COMPUTED, not written: see the origin block below.
G: dict[str, tuple[str | None, dict]] = {
    "g0": (None, {}),
    "g1": ("g0", {"observation_mode": "vec_mip"}),
    "g2": ("g0", {"norm_obs": False}),
    "g3": ("g1", {"norm_obs": False}),
    # THE ACTION AXIS FORKED, and as of #E17 it carries ids. F44's rule held it
    # empty until runs existed: `order_protection` now has 66 across two boards
    # (48 on sc3, 18 on sc4) and is the best encoding at every rung on both, so
    # the cells are promotions rather than a record that a mode exists. `g4`/`g5`
    # are minted as PARENTS — the registry's own rule that a base earns a label
    # by being made the parent of a later row — and `g6`/`g7` are the artifacts:
    # `g6` is A23's best cell and the first RL number in this campaign to beat a
    # published bar (#E17).
    # The three `target_*` modes stay UNMINTED. They ran the same factorial and
    # lost at every rung (#E11), so they are ledger-addressed by their `tag` —
    # a failed arm earns no id however fully it was replicated.
    "g4": ("g0", {"action_mode": "order_protection"}),
    "g5": ("g1", {"action_mode": "order_protection"}),
    "g6": ("g4", {"norm_obs": False}),
    "g7": ("g5", {"norm_obs": False}),
    # THE ACTION AXIS HAS FORKED AND CARRIES NO IDS YET. `action_mode` is a
    # design axis this campaign never moved — every id above is implicitly
    # `act=seq_mask` — and the IR now declares three more modes
    # (`order_protection`, `target_ip`, `target_mip`). Their cells are NOT
    # minted here, deliberately: R1's phantom-base check asserts that every
    # minted base was run, and F44's rule is that an id is a PROMOTION, not a
    # record that a configuration exists. Minting `g4..g7` against runs nobody
    # has launched is the same defect that got `sc3` withdrawn the first time
    # ("reserved for A1"), and it would fail the audit below rather than
    # quietly rot. The ids follow the runs.
    #
    # When they are minted, the pairing is a CROSS-AXIS CONSTRAINT and not a
    # free cross-product: `target_ip` is coherent only with `vec` and
    # `target_mip` only with the mip modes, because the mode hands each arm its
    # OWN sufficient statistic as the order-up-to reference. Pairing them the
    # other way is F30's defect — handing `vec` the transform RQ4 asks whether
    # a network can rebuild — so only the diagonal cells are runnable and the
    # constraint belongs in CONSTRAINTS below, declared on the `g` id.
}

# --- a: policy family, extractor, critic form ------------------------------
# Never moved. `algo_class` and `policy` are not CLI dests — they are facts the
# run records (§8.4 provenance) and R1c constrains against.
A: dict[str, tuple[str | None, dict]] = {
    "a0": (None, {"algo_class": "MaskablePPO", "policy": "MlpPolicy",
                  "order_head": "categorical"}),
    # A43/A44/A45 (#E22, #E23, #E24), adopted #E25 (2026-09-04): the ordinal
    # order head. The first time this axis moved. Six paired contrasts against
    # `a0` at identical hp and seeds, all negative: -8..-9 (sc3, L1 hp), -0.85
    # best-vs-best (sc3, tuned), -3.4/-3.6 (sc0, crowned hp), -1.13/-1.85 (sc4,
    # crowned hp). Same MultiDiscrete space and cascade; MaskablePPO with
    # `adi_flex_ordinal_head.OrdinalMaskablePolicy` (gate: the _probe, G1-G5).
    "a1": ("a0", {"order_head": "ordinal"}),
}
NON_CLI = frozenset({"algo_class", "policy"})

# --- h: optimizer, schedules, rollout geometry, loss weights, epochs -------
# h0's delta is NOT written here: it is read from the train script's
# `_L1_DERIVED` (§8.4), so the origin cannot drift from the derivation it is
# supposed to be. Read by ast, not by import — the train script pulls in torch.
H: dict[str, tuple[str | None, dict]] = {
    "h0": (None, {}),                       # computed at import — see below
    "h1": ("h0", {"learning_rate": 5.850233173311956e-04,
                  "lr_final": 5.850233173311956e-05, "batch_size": 256,
                  "gae_lambda": 0.9975850800083303,
                  "ent_coef": 1.0646629822331615e-08,
                  "vf_coef": 0.5172341531779983, "net_arch": [128, 128, 128]}),
    "h2": ("h0", {"learning_rate": 8.994440039003481e-05, "lr_final": 8.9944e-06,
                  "batch_size": 256, "n_epochs": 20, "gae_lambda": 0.814973,
                  "ent_coef": 5.6689805386315736e-05,
                  "vf_coef": 0.8183810076958199,
                  "net_arch": [256, 256, 256, 256]}),
    "h3": ("h0", {"learning_rate": 9.433832555898598e-04, "lr_final": 9.43383e-05,
                  "n_steps": 1024, "batch_size": 512, "n_epochs": 20,
                  "gae_lambda": 0.946323, "ent_coef": 1.3491582888841108e-04,
                  "vf_coef": 0.929347917477382, "net_arch": [64, 64, 64, 64]}),
    # A24/A25 (#E18) — the first CROWNED tuned configs in this campaign. Both
    # are `sc4`/`order_protection`/`norm_obs=false` (g6 and g7), re-scored on
    # the 8192 protocol block per §9.7's post-tuning rule, and both beat every
    # heuristic in the family including #E14's terminal-zero variant. They are
    # ids because they were crowned; #E16's tuned arm on sc3 still has none,
    # because it lost to its board's bar (F44).
    # The two agree on the shape the search found and the derivation missed:
    # n_steps 512 -> 2048 (4x the rollout), n_epochs 10 -> 20, lr ~5-7x, and a
    # DEEPER-not-wider net. gae_lambda drops in both, which is the credit
    # horizon shortening once the rollout lengthens.
    "h4": ("h0", {"learning_rate": 0.0005173509085450419,
                  "lr_final": 5.17351e-05, "n_steps": 2048, "batch_size": 256,
                  "n_epochs": 20, "gae_lambda": 0.850167,
                  "ent_coef": 0.0016711302478830468,
                  "vf_coef": 0.3375593387420656,
                  "net_arch": [64, 64, 64, 64]}),
    "h5": ("h0", {"learning_rate": 0.0006573978862390732,
                  "lr_final": 6.57398e-05, "n_steps": 2048,
                  "n_epochs": 20, "gae_lambda": 0.911785,
                  "ent_coef": 1.3344583398753121e-05,
                  "vf_coef": 0.3088508943791865,
                  "net_arch": [32, 32, 32]}),
    # A44 (#E23) -> crowned #E25: TUNE6's `vec` trial 87, tuned FOR the ordinal
    # head in (g6, a1) and confirmed top-3 on 6 seeds. -1.33 +-0.34 paired
    # against h2 under the same head. vs h2: rollout 512 -> 4096, lambda
    # 0.815 -> 0.995, advantage normalization OFF; width/depth unchanged. The
    # `vec_mip` study's top-3 did not beat h3 under the head, so that arm keeps
    # h3 and mints nothing (F44).
    "h6": ("h0", {"learning_rate": 6.184508588237097e-05, "lr_final": 6.1845e-06,
                  "n_steps": 4096, "batch_size": 256, "n_epochs": 20,
                  "gae_lambda": 0.99534, "ent_coef": 2.0913992797985608e-05,
                  "vf_coef": 0.9735708392469196,
                  "net_arch": [256, 256, 256, 256],
                  "normalize_advantage": False}),
}

# --- R1c: where one axis forces a value on another -------------------------
# Declared on the CONSTRAINING id, and a tuple that violates it is refused
# rather than resolved silently. Both of adi_flex's constraints point out of
# `g`/`sc` and into an axis that cannot see them.
REQUIRES: dict[str, dict[str, object]] = {
    # The sequential mask is not advisory: one shared Discrete space serves both
    # phases and the per-step mask selects the live range, so a policy that
    # cannot read `action_masks` emits illegal actions. The IR declares
    # `rl.algo: maskable_ppo` for this reason (F7).
    "g0": {"a.algo_class": "MaskablePPO"},
    # γ = β, and β is the IR's `objective.discount_factor` = 1.0 on every
    # instance here. γ > β is never legal (§8.6) and this domain derives
    # equality, so the scenario fixes an `h` knob. **This one binds the
    # sampler**: A16 passed `--fix gamma` by hand for exactly this, and a study
    # that forgets it optimizes a different objective than the one declared.
    "sc0": {"h.gamma": 1.0}, "sc1": {"h.gamma": 1.0}, "sc2": {"h.gamma": 1.0},
    "sc3": {"h.gamma": 1.0}, "sc4": {"h.gamma": 1.0},
}

AXES = {"sc": SC, "g": G, "a": A, "h": H}


def _from_train_source(name: str):
    """A module-level literal read out of the train script without importing it
    (it pulls in torch; the same reason `mdp_conformance` reads it by ast)."""
    src = (_HERE / "adi_flex_ppo_train.py").read_text()
    for node in ast.parse(src).body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            value = node.value
            # `frozenset({...})` / `set({...})` are calls, not literals
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id in ("frozenset", "set", "tuple", "list")
                    and len(value.args) == 1):
                return ast.literal_eval(value.args[0])
            return ast.literal_eval(value)
    raise RuntimeError(f"adi_flex_ppo_train.py declares no {name}")


# --- the origin rows, computed rather than written (R2 completeness) -------
# **An origin must be COMPLETE, or a citation cannot express a run.** §8.4's
# `_L1_DERIVED` holds only what the §8.6 table *derived*, and deliberately
# omits knobs no row derives (`vf_coef`) — that omission is right for the run
# name, which diffs the derivation, and wrong for a config, which must name
# every knob it governs or two runs differing on the omitted one resolve to the
# same tuple. That is the collision R2 exists to forbid, and it was live here:
# `sc0/g0/a0/h0` named no `vf_coef` until this was written.
#
# So an origin is: the derived value where §8.6 speaks, the train script's
# parser default where it does not. Both halves are READ from the script, so
# neither is a hand-copy that can drift.
#
# **The §8.6 derivation is not single-axis** either — it derives `norm_obs` and
# `norm_reward` alongside optimizer knobs, and guide §12's ladder says L1
# "defines the origin of each axis: `g0`, `a0`, `h0`", plural. Ownership above
# is what splits it.
_DERIVED = _from_train_source("_L1_DERIVED")


def _origin(owned: frozenset) -> dict:
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


# (the origins are computed just above _self_check, once the readers exist)


def _train_dests() -> set[str]:
    """Every dest the train script's parser exposes, by ast."""
    return _dests_of((_HERE / "adi_flex_ppo_train.py").read_text())


def _dests_of(src: str) -> set[str]:
    """The same, for any train-script source — used to test whether a sibling
    worktree's archive belongs to this campaign at all."""
    out = set()
    for node in ast.walk(ast.parse(src)):
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
    src = (_HERE / "adi_flex_ppo_train.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        this = None
        default = None
        for kw in node.keywords:
            if kw.arg == "dest":
                this = kw.value.value
            elif kw.arg == "default":
                default = ast.literal_eval(kw.value)
        if this is None:
            flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
            long = [f for f in flags if f.startswith("--")] or flags
            this = long[0].lstrip("-").replace("-", "_")
        if this == dest:
            return True, default
    return False, None


def _walk(axis: dict[str, tuple[str | None, dict]], cid: str) -> dict:
    """Resolve one id by walking to its origin and applying deltas outward."""
    chain, seen = [], set()
    while cid is not None:
        if cid in seen:
            raise ValueError(f"cycle in the registry at {cid!r}")
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
    """The knob values a cited tuple denotes. Order-independent by R1a: the
    axes' deltas are disjoint, asserted at import, so no key can be written
    twice and the merge order cannot matter."""
    if sc not in SC:
        raise KeyError(f"unknown scenario id {sc!r}")
    out = {"scenario_name": SC[sc]}
    for axis, cid in ((G, g), (A, a), (H, h)):
        out.update(_walk(axis, cid))
    return out


def check_requires(sc: str, g: str, a: str, h: str) -> list[str]:
    """R1c violations for a cited tuple — empty when the tuple is legal."""
    cited = {"sc": sc, "g": g, "a": a, "h": h}
    resolved = resolve(sc, g, a, h)
    bad = []
    for cid in cited.values():
        for key, want in REQUIRES.get(cid, {}).items():
            _, dest = key.split(".", 1)
            got = resolved.get(dest)
            if got != want:
                bad.append(f"{cid} requires {key} = {want!r}, tuple resolves "
                           f"{dest} = {got!r}")
    return bad


# --- the L0/L1/L2 ladder, as the proposal's own table states it ------------
# | L0  | not configured from the registry at all | `{sc}/L0` — the scenario and
# |     |                                         |  nothing else
# | L1  | the derivation's output; DEFINES the    | `{sc}/L1`, canonical for
# |     | origin of each axis: g0, a0, h0         |  `{sc}/g0/a0/h0`
# | L2+ | every escalation: a non-origin id on    | `{sc}/g0/a2/h0`,
# |     | any axis, or a declared deviation       |  `{sc}/g0/a0/h0` + deviations
# The ladder attaches at exactly one point — the origin of each axis — which is
# why every registry row above an origin is L2 permanently, and why the level
# stops being authored: origin ids everywhere with no deviations is L1, and
# which axes moved says which sub-layer.
_ORIGINS = {"g": "g0", "a": "a0", "h": "h0"}
_AXIS_LAYER = {"g": "gym", "a": "arch", "h": "hp"}
_LAYER_ORDER = ("hp", "gym", "arch")          # §8.6 writes L3(hp+arch)


def level_of(sc: str, g: str, a: str, h: str,
             deviations: tuple[str, ...] = ()) -> str:
    """The level a CITATION implies.

    **A non-origin id is not automatically an escalation** (F49). The
    proposal's ladder reads "L2+ — every escalation: a non-origin id on any
    axis, or a declared deviation", and that is too strong: `g` also carries the
    DESIGN axes (`observation_mode`, `action_mode`, `reward_mode`), which the
    §8.6 derivation does not derive and which name a different *cell*, not a
    different *level*. §8.6's own ladder is per-cell — "Δ(L1−L0) measures the
    configuration layer **per case**" — so `{sc}/g1/a0/h0`, this domain's
    `vec_mip` arm, is L1 in its own cell and calling it L2(gym) would relabel
    half of RQ4's declared instrument as an escalation.

    What makes a level is a move off a knob the derivation *names*. So the
    citation is resolved and diffed against `_L1_DERIVED`, which is the same
    question `adi_flex_ppo_train.derive_level` asks of a run's arguments — two
    implementations reading two different sources, cross-checked in the audit.
    """
    illegal = set(deviations) & DESIGN_AXES
    if illegal:
        raise ValueError(
            f"{sorted(illegal)} are design axes — fixed by the citation, never "
            f"deviations (F50). A different cell is a different citation")
    resolved = resolve(sc, g, a, h)
    origin = {**G["g0"][1], **H["h0"][1]}
    moved = set()
    # The baseline is the COMPLETE origin minus the design axes (F49/F50): a
    # design-axis value names a cell, never a level, while any other owned
    # knob — derived or not — makes a level when moved, since someone tried a
    # configuration the derivation did not produce (§8.6's L2 invariant).
    for dest, want in origin.items():
        if dest in DESIGN_AXES:
            continue
        owner = "g" if dest in G_OWNS else "h"
        if dest in deviations or _as_text(resolved.get(dest)) != _as_text(want):
            moved.add(_AXIS_LAYER[owner])
    layers = [x for x in _LAYER_ORDER if x in moved]
    if not layers:
        return "L1"
    return f"L{len(layers) + 1}({'+'.join(layers)})"


def parse_tuple(spec: str) -> tuple[str, str, str, str]:
    """`sc0/g2/a0/h2` -> the four ids, in any order within the string.

    `{sc}/L1` is the canonical spelling of `{sc}/g0/a0/h0`. `{sc}/L0` is
    refused: §8.6 *defines* L0, so there is no derived config to name and
    nothing for a citation to resolve — an L0 run passes L0's flags and records
    `{sc}/L0` as its address, which is not the same as configuring from here.
    """
    parts = [p for p in spec.replace(",", "/").split("/") if p]
    if any(p.upper() == "L0" for p in parts):
        raise ValueError(
            f"{spec!r}: L0 is not configured from the registry — §8.6 defines "
            f"it as the library's defaults plus what the problem forces, so no "
            f"axis id names it. Launch it with --level L0 and L0's flags")
    # cell-relative spelling: `sc0/vec_mip/L1` cites a cell by its IR-declared
    # NAME plus the wrapper/hp origins — that cell's own L1, since the ladder
    # is per-cell (§8.6, F50). Legal only for a REGISTERED cell: a probe cell
    # has no g id (F44) and is ledger-addressed, which the error says.
    if any(p.upper() == "L1" for p in parts):
        parts = [p for p in parts if p.upper() != "L1"]
        known = set(SC) | set(G) | set(A) | set(H)
        names = [p for p in parts if p not in known]
        if len(names) > 1:
            raise ValueError(f"{spec!r}: {names} are not registered ids, and a "
                             f"cell-relative citation may name at most one cell")
        if names:
            name = names[0]
            wrapper_origin = {k: v for k, v in G["g0"][1].items()
                              if k not in DESIGN_AXES}
            hit = [cid for cid in G
                   if _walk(G, cid).get("observation_mode") == name
                   and all(_as_text(_walk(G, cid).get(k)) == _as_text(v)
                           for k, v in wrapper_origin.items())]
            if not hit:
                raise ValueError(
                    f"{name!r} names no registered cell — it is either not a "
                    f"declared mode or an unregistered (probe) cell, which is "
                    f"ledger-addressed (F44); launch it without --config")
            parts = [p for p in parts if p != name] + [hit[0]]
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
            raise ValueError(f"{p!r} is not a registered config id")
    missing = [k for k in ("sc", "g", "a", "h") if k not in picked]
    if missing:
        raise ValueError(f"{spec!r} names no {'/'.join(missing)} id — a run is "
                         f"sc + g + a + h + deviations + seed (R2)")
    return picked["sc"], picked["g"], picked["a"], picked["h"]


def _self_check() -> None:
    """R1a and R1b, enforced at import rather than documented.

    An overlap between two axes' deltas is a registry bug — it means one knob
    has two homes and a resolution could depend on merge order — and it is
    reported before any run starts.
    """
    from adi_flex_scenarios import SCENARIOS
    dead = {k: v for k, v in SC.items() if v not in SCENARIOS}
    if dead:
        raise RuntimeError(f"sc alias(es) point at instances SCENARIOS does not "
                           f"have: {dead} — an alias may never outlive its key (R1b)")
    # the train script's `_GYM_KNOBS` (which `derive_level` reads, F42) is the
    # wrapper-stack subset of what `g` owns; two definitions of one boundary is
    # exactly the drift this module exists to make impossible
    stray = set(_from_train_source("_GYM_KNOBS")) - G_OWNS
    if stray:
        raise RuntimeError(f"the train script calls {sorted(stray)} gym knobs "
                           f"and G_OWNS does not — one boundary, two definitions")
    if not DESIGN_AXES <= G_OWNS:
        raise RuntimeError(f"design axes {sorted(DESIGN_AXES - G_OWNS)} not "
                           f"owned by g — the cell coordinates live in g")

    # cells REFERENCE the IR's gym block, never define (F50): every design-axis
    # value any g id sets must be a declared mode, the same rule that binds an
    # sc alias to SCENARIOS. A g delta citing a mode the IR does not declare is
    # a registry bug, reported before any run starts.
    from mdp_ir.schema import load_ir
    _gym = load_ir(_HERE / "adi_flex_schema.json").gym
    declared = {"observation_mode": {m.name for m in _gym.observation_modes},
                "action_mode": {m.name for m in _gym.action_modes},
                "reward_mode": {m.name for m in _gym.reward_modes}}
    for cid, (_, delta) in G.items():
        for axis in DESIGN_AXES & set(delta):
            if delta[axis] not in declared[axis]:
                raise RuntimeError(
                    f"{cid} sets {axis}={delta[axis]!r}, which the IR does not "
                    f"declare (gym block: {sorted(declared[axis])}) — cells "
                    f"reference the gym block, never define (F50)")

    # --- R1a + R2: the ownership partition is total and disjoint -----------
    # Every knob the train script exposes has exactly one home. A knob added to
    # the CLI and to no axis would make some run inexpressible as
    # `sc + g + a + h + deviations + seed`, which is the one thing R2 forbids —
    # so it is a registry failure, raised before any run starts.
    dests = _train_dests()
    groups = {"g": G_OWNS, "a": A_OWNS, "h": H_OWNS, "R3": R3_EXCLUDED,
              "protocol": PROTOCOL, "instrument": INSTRUMENT}
    for x, y in itertools.combinations(sorted(groups), 2):
        both = groups[x] & groups[y]
        if both:
            raise RuntimeError(f"{sorted(both)} claimed by both {x} and {y} — "
                               f"one knob, two homes (R1a)")
    claimed = G_OWNS | A_OWNS | H_OWNS | R3_EXCLUDED | PROTOCOL | INSTRUMENT
    orphan = dests - claimed
    if orphan:
        raise RuntimeError(
            f"{sorted(orphan)} are train-script arguments no axis owns and "
            f"neither R3, the protocol list nor INSTRUMENT excludes — one "
            f"could not be expressed as sc+g+a+h+deviations+seed (R2)")
    ghost = claimed - dests
    if ghost:
        raise RuntimeError(f"{sorted(ghost)} are owned but the train script "
                           f"exposes no such argument")
    # --- every delta stays inside its own axis, and every origin is complete
    for name, axis, owns in (("g", G, G_OWNS), ("a", A, A_OWNS | NON_CLI),
                             ("h", H, H_OWNS)):
        for cid, (_, delta) in axis.items():
            stray = set(delta) - owns
            if stray:
                raise RuntimeError(f"{cid} sets {sorted(stray)}, which {name} "
                                   f"does not own (R1a)")
        missing = owns - set(axis[f"{name}0"][1])
        if missing:
            raise RuntimeError(f"{name}0 names no value for {sorted(missing)} — "
                               f"an origin must be COMPLETE or a citation "
                               f"cannot express a run (R2)")
    # Ids are `{axis}{integer}` with an optional GENERATION letter — a
    # re-derivation may not reuse `g0` (R4), so it mints `g0b` under `L1b` and
    # restarts the ladder above it. Density is per generation, not across them:
    # a check that assumed one generation would refuse the first re-derivation,
    # which for this domain is one A1 away (T~30 -> 36).
    for name, axis in (("g", G), ("a", A), ("h", H)):
        gens: dict[str, list[int]] = {}
        for cid in axis:
            m = re.fullmatch(rf"{name}(\d+)([a-z]?)", cid)
            if not m:
                raise RuntimeError(f"{cid!r} is not `{name}{{int}}[{{gen}}]` (R1b)")
            gens.setdefault(m.group(2), []).append(int(m.group(1)))
        for gen, nums in gens.items():
            if sorted(nums) != list(range(len(nums))):
                raise RuntimeError(f"{name} ids in generation {gen or '(first)'} "
                                   f"are not dense from 0: {sorted(nums)} (R1b)")
        for cid in axis:
            _walk(axis, cid)
    for cid, req in REQUIRES.items():
        for key in req:
            if key.split(".", 1)[0] not in AXES:
                raise RuntimeError(f"{cid} constrains unknown axis in {key!r}")


DESIGN_AXES = frozenset(_from_train_source("_DESIGN_AXES"))
G["g0"] = (None, _origin(G_OWNS))
H["h0"] = (None, _origin(H_OWNS))
_self_check()


# ===========================================================================
# The archive audit  —  `python adi_flex_configs.py`
# ===========================================================================
# Four reports over `results/`, and they live beside the registry rather than in
# a file of their own because all four read the same thing: the partition of
# knobs by implementing layer that this module declares. A separate file would
# have been a second place to keep that partition in step.
#
#   1. `solve_level` against the level the knobs imply          (F42)
#   2. phantom bases — every minted row must match a run        (F44)
#   3. module <-> §CONFIG-REGISTRY agreement                    (R1d, F46)
#   4. R2 expressibility — every run is a tuple + deviations    (F47)
#
# Deliberately below `_self_check()` and behind `__main__`: the audit imports
# the train script, which pulls in torch, and the train script imports this
# module at launch. Importing `adi_flex_configs` must stay cheap and acyclic.

_ARGS_DEFAULTS = {"clip_init": "0.2", "vf_coef": "0.5",
                  "normalize_advantage": "True", "vecnorm_clip_obs": "10.0"}
# id -> how to find the run(s) that promoted it. "derived" means the h half sits
# at `_L1_DERIVED`, which is where the origin lives.
_PROMOTED_BY = {
    "g0": ("derived", "vec"),      "g1": ("derived", "vec_mip"),
    "g2": ("A13/#E7", "vec"),      "g3": ("A13/#E7", "vec_mip"),
    "g4": ("derived", "vec"),      "g5": ("derived", "vec_mip"),
    # the tag is the wave that PROMOTED the cell, not every wave that ran it:
    # A4c is #E11's L2(gym) order_protection pair on sc3, where the cell first
    # came out best; A23 then confirmed it on sc4 and is cited in the table
    "g6": ("A4c", "vec"),          "g7": ("A4c", "vec_mip"),
    "h0": ("derived", None),
    "h1": ("A9/#E5 stage2", "vec"),
    "h2": ("A16/#E8 stage2-sched", "vec"),
    "h3": ("A16/#E8 stage2-sched", "vec_mip"),
    # A CROWNED TRIAL is addressed by its path, not by a `tag`. F48 keeps trials
    # out of the archive because a sampled point is not a chosen run — but the
    # §9.7 post-tuning rule ships the winning TRIAL's artifact (re-scored on the
    # protocol block), with no retrain unless a claim is made about the tuning
    # PROCEDURE rather than the artifact. So the one run behind a crowned `h`
    # row legitimately lives under `results/tuning/`, and the phantom check has
    # to look there or it reports a base that demonstrably exists as missing.
    # Naming the exact trial is the point: it is the artifact, so the row says
    # which file it is.
    "h4": ("path:s4_prot_vec_obsF/trial_0185", "vec"),
    "h5": ("path:s4_prot_vecmip_obsF/trial_0321", "vec_mip"),
}
# --- the archive is SPLIT ACROSS WORKTREES, and the audit must read all of it
# F31 forced the split: a tuning trial re-execs the train script, so code
# changes had to be quarantined from a live study, and the campaign has run from
# two worktrees ever since. Their `results/` are physically separate — one tree
# symlinks the shared archive, the other keeps its own — so an audit rooted at
# `_HERE` reports on whichever half it happens to sit in and says nothing about
# the other. It reported GREEN over 144 runs while 120 in the sibling tree, every
# `order_protection`, `target_*` and het3 run in the campaign, had never been
# checked at all. Enumerating worktrees keeps this self-maintaining: when the
# branches merge and the second tree goes away, its root simply stops existing.
@functools.lru_cache(maxsize=1)
def _domain_dirs() -> tuple[Path, ...]:
    """This domain's folder in every worktree of the repo, `_HERE` first.

    Deduplication belongs to the CALLER, per artifact: the trees share some
    artifacts and not others — one tree's `results/` is a symlink into another's
    while its `ESCALATION.md` is its own file — so collapsing trees by any one
    artifact's identity drops a tree that is distinct in the rest. Doing exactly
    that is what first hid the stale log from R1d.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(_HERE), "worktree", "list",
                              "--porcelain"], capture_output=True, text=True,
                             timeout=20).stdout
        trees = [Path(l.split(" ", 1)[1]) / _HERE.name for l in out.splitlines()
                 if l.startswith("worktree ")]
    except Exception:
        trees = []
    seen, dirs = set(), []
    for d in [_HERE] + trees:
        if d.is_dir() and d not in seen:
            seen.add(d)
            dirs.append(d)
    return tuple(dirs)


@functools.lru_cache(maxsize=1)
def _result_roots() -> tuple[Path, ...]:
    """Archives belonging to THIS campaign.

    F52 made the audit enumerate every worktree, which fixed reporting green
    over half the archive — and then over-reached: a sibling worktree running a
    DIFFERENT experiment (an `order_head` action the train script here does not
    expose) had its runs pulled in and addressed against ids minted here, giving
    42 spurious level clashes. A run is only expressible in this registry if it
    was produced by a train script with the same argument surface, so that is
    the admission test. Rejected roots are NAMED, never dropped silently — the
    whole point of F52 is that silence and coverage look alike.
    """
    mine = _train_dests()
    roots, seen, rejected = [], set(), []
    for d in _domain_dirs():
        r = d / "results"
        if not r.is_dir():
            continue
        key = r.resolve()                 # one tree symlinks the other's archive
        if key in seen:
            continue
        seen.add(key)
        try:
            theirs = _dests_of((d / "adi_flex_ppo_train.py").read_text())
        except OSError:
            theirs = mine
        if theirs != mine:
            rejected.append((r, sorted(theirs ^ mine)))
            continue
        roots.append(r)
    for r, diff in rejected:
        print(f"  EXCLUDED {r}: its train script differs on {diff}")
    return tuple(roots) or (_HERE / "results",)


def _escalation_log() -> Path:
    """The ONE authoritative `ESCALATION.md`, across worktrees.

    Same split as the archive and worse, because the log is a single document
    rather than a union: it is only ever edited in the tree the campaign writes
    from, and the sibling's copy is an inert checkout that goes stale the moment
    the live one is touched. R1d compares the module against this file, so an
    audit reading the stale copy reports the module and the table disagreeing
    when they do not — a false failure — or, worse, agreeing on a row the live
    table has since moved. Resolve to the newest copy and SAY WHICH, so the
    reading can be checked; warn when the others differ rather than silently
    picking. When the branches merge there is one copy and the warning stops.
    """
    cands, seen = [], set()
    for d in _domain_dirs():
        c = (d / "ESCALATION.md").resolve()
        if c.is_file() and c not in seen:
            seen.add(c)
            cands.append(c)
    if not cands:
        return _HERE / "ESCALATION.md"
    live = max(cands, key=lambda c: c.stat().st_mtime)
    stale = [c for c in cands if c != live and c.read_text() != live.read_text()]
    if stale:
        print(f"  NOTE: reading {live}")
        for c in stale:
            print(f"        {c} differs and is IGNORED (stale worktree copy)")
    return live


def _load_runs() -> list[dict]:
    """Every run's args log, once, with the two legacy shapes normalized: a
    pre-flag log wrote `--no-vecnorm` instead of the split pair, and flags added
    later are absent rather than at their default."""
    out = []
    for root in _result_roots():
      for path in sorted(root.rglob("*_ppo_args.txt")):
        d = {"__path__": path, "__tuning__": path.parts[len(root.parts)] == "tuning"}
        for line in path.read_text().splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                d[k] = v
        if "norm_obs" not in d:
            on = d.get("no_vecnorm") != "True"
            d["norm_obs"] = d["norm_reward"] = str(on)
        for k, v in _ARGS_DEFAULTS.items():
            d.setdefault(k, v)
        out.append(d)
    return out


def _as_text(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)


def _ns_from_log(d: dict, train) -> "argparse.Namespace":
    """A namespace carrying every owned non-design dest, read from one args
    log — the input `derive_level` diffs against its complete baseline (F50).
    Missing keys (pre-flag logs) fall back to the baseline value, i.e. read as
    unmoved, which is what a knob that did not exist yet was."""
    import argparse
    baseline = {**G["g0"][1], **H["h0"][1]}
    cast = {"learning_rate": float, "lr_final": float, "n_envs": int,
            "n_steps": int, "batch_size": int, "n_epochs": int,
            "target_kl": float, "gamma": float, "gae_lambda": float,
            "clip_init": float, "clip_final": float, "ent_coef": float,
            "vf_coef": float, "vecnorm_clip_obs": float}
    ns = argparse.Namespace(level=d.get("solve_level", "?"))
    for dest, fallback in baseline.items():
        raw = d.get(dest)
        if raw is None:
            value = fallback
        elif dest == "net_arch":
            value = [int(x) for x in raw.strip("[]").replace(",", " ").split()]
        elif raw == "None":
            value = None
        elif dest in cast:
            value = cast[dest](raw)
        elif dest in DESIGN_AXES:
            value = raw
        else:
            value = raw == "True"
        setattr(ns, dest, value)
    return ns


def _report_levels(runs) -> None:
    """What `--level` claimed, against what the knobs say (F42).

    Training runs only. A tuning trial is a sampled point, not a run the
    campaign chose to make, and `mdp_tuning` stamps every one of them `L1` by
    driving the script with the flag at its default — so counting them here
    reports a driver artifact as an operator error (F48)."""
    import adi_flex_ppo_train as train
    table, mismatched = collections.Counter(), collections.Counter()
    for d in runs:
        ns = _ns_from_log(d, train)
        authored, derived = d.get("solve_level", "?"), train.derive_level(ns)
        table[(authored, derived)] += 1
        if authored != derived:
            mismatched[(d.get("tag", "") or "(untagged)", authored, derived)] += 1
    print(f"=== solve_level vs the knobs (F42) — {len(runs)} run(s) ===")
    print(f"  {'authored':10} {'derived':14} count   verdict")
    for (au, de), n in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f"  {au:10} {de:14} {n:5}   {'agree' if au == de else 'MISLABELLED'}")
    bad = sum(n for (au, de), n in table.items() if au != de)
    print(f"  mislabelled: {bad} of {len(runs)}")
    for (tag, au, de), n in sorted(mismatched.items(), key=lambda kv: (-kv[1], kv[0]))[:6]:
        print(f"    {tag:24} {au:10} -> {de:14} {n}")


def _at_derivation(d: dict) -> bool:
    return all(d.get(dest, "-") == _as_text(want) for dest, want in _DERIVED.items())


def _report_phantom(runs) -> None:
    """Every minted row must match a run that actually executed (F44). A base
    nothing ran is guide §12's motivating incident and this campaign's own
    (F41). The registry is NOT a partition of the archive — most configurations
    are one-time probes closed in the §LEDGER — so the check runs this way and
    not the other."""
    live = [d for d in runs if not d["__tuning__"]]
    pairs = {(tuple(d.get(k, "-") for k in sorted(G_OWNS)),
              tuple(d.get(k, "-") for k in sorted(H_OWNS))) for d in live}
    print(f"\n=== phantom-base check (F44) ===")
    phantom = 0
    for cid, (tag, obs) in _PROMOTED_BY.items():
        if tag.startswith("path:"):
            # a crowned tuning trial: search ALL runs, match the artifact's path
            hit = [d for d in runs if tag[5:] in str(d["__path__"])
                   and (obs is None or d.get("observation_mode") == obs)]
        else:
            hit = [d for d in live
                   if (obs is None or d.get("observation_mode") == obs)
                   and (_at_derivation(d) if tag == "derived" else d.get("tag") == tag)]
        where = f"[{tag}" + (f" / {obs}]" if obs else "]")
        if hit:
            print(f"  {cid:4} ok    {len(hit):3} run(s)   {where}")
        else:
            phantom += 1
            print(f"  {cid:4} PHANTOM — minted, but no run matches {where}")
    ng, nh = len({g for g, _ in pairs}), len({h for _, h in pairs})
    mg = len([c for c in _PROMOTED_BY if c[0] == "g"])
    mh = len([c for c in _PROMOTED_BY if c[0] == "h"])
    print(f"  archive: {len(pairs)} distinct (g,h) pairs over {ng} g and {nh} h")
    print(f"  minted : {mg} g, {mh} h   |   closed in the ledger: {ng - mg} g, {nh - mh} h")
    print("  " + ("all minted bases were run" if not phantom
                  else f"{phantom} PHANTOM BASE(S)"))


def _report_r1d() -> None:
    """The two artifacts must agree (R1d, F46). Deliberate redundancy is only
    safe when it is checked — guide §12 open point 3 is exactly that an
    unchecked second source of truth is what the issue-#11 disposition
    rejected. `sc` rows compare the alias, since `sc` references rather than
    derives and so has no parent column."""
    import re
    md = _escalation_log().read_text()
    body = md[md.index("## CONFIG-REGISTRY"):]
    body = body[:body.index("## LEDGER")]
    table = {}
    for line in body.splitlines():
        m = re.match(r'\|\s*<a id="([a-z]+\d+)"></a>`\1`\s*\|\s*([^|]*)\|', line)
        if m:
            parent = m.group(2).strip().strip("`").strip()
            table[m.group(1)] = None if parent in ("—", "-", "") else parent
    module = dict(SC)
    for axis in (G, A, H):
        module.update({cid: parent for cid, (parent, _) in axis.items()})
    only_mod = sorted(set(module) - set(table))
    only_tab = sorted(set(table) - set(module))
    disagree = sorted(k for k in set(module) & set(table) if module[k] != table[k])
    print("\n=== module <-> §CONFIG-REGISTRY agreement (R1d, F46) ===")
    for label, bad in (("in the module, not the table", only_mod),
                       ("in the table, not the module", only_tab),
                       ("parent disagrees", disagree)):
        print(f"  {label:30} {bad if bad else 'none'}")
    print(f"  {len(module)} id(s) registered in both" if not (only_mod or only_tab or disagree)
          else "  R1d VIOLATED")


def _report_r2(runs) -> None:
    """Every training run is expressible, in exactly one of three address forms
    (R2, F47/F50):

      registered cell    `sc/g/a/h` + deviations — and a deviation may never
                         touch a design axis (write-once, F50), so the tuple
                         search is restricted to tuples MATCHING the run's cell
      unregistered cell  a probe or an INSTRUMENT run: its cell has no g id
                         because nothing was promoted there (F44), or because
                         the cell is not one the registry can name at all —
                         ledger-addressed, by `tag`
      L0                 `{sc}/L0` — §8.6 defines it; no citation resolves it

    Training runs only (F48): a trial is the centre plus what the sampler drew.
    """
    tuples = [(g, a, h) for g in G for a in A for h in H]
    cells = {tuple(_as_text(_walk(G, cid).get(ax)) for ax in sorted(DESIGN_AXES))
             : cid for cid in G}
    knobs = sorted((G_OWNS | H_OWNS) - DESIGN_AXES)
    dist, probes, bad, clash, n_l0 = (collections.Counter(),
                                      collections.Counter(), [], [], 0)
    for d in runs:
        sc = next((k for k, v in SC.items() if v == d.get("scenario_name")), None)
        if sc is None:
            bad.append(f"{d['__path__'].parent.name[:44]}: no sc alias for "
                       f"{d.get('scenario_name')!r}")
            continue
        if d.get("solve_level") == "L0":
            n_l0 += 1
            continue
        instr = [k for k in sorted(INSTRUMENT)
                 if d.get(k) not in (None, "None", "-", "")]
        if instr:
            # checked BEFORE the cell lookup: an instrument run's knobs may sit
            # exactly on a registered cell, and letting it match would give it
            # the same address as the run it is the control FOR
            probes[f"{d.get('observation_mode', '?')}+{','.join(instr)}"] += 1
            continue
        cell = tuple(d.get(ax, "-") for ax in sorted(DESIGN_AXES))
        if cell not in cells:
            probes[d.get("observation_mode", "?")] += 1
            continue
        absent = [k for k in knobs if k not in d]
        if absent:
            bad.append(f"{d['__path__'].parent.name[:44]}: args log names no {absent}")
            continue
        # tuple search restricted to the run's cell: a design axis is fixed by
        # the citation, so a cross-cell tuple is not a candidate address (F50)
        in_cell = [(g, a, h) for g, a, h in tuples
                   if tuple(_as_text(_walk(G, g).get(ax))
                            for ax in sorted(DESIGN_AXES)) == cell]
        n_dev, (g, a, h) = min(
            ((sum(1 for k in knobs if d[k] != _as_text(resolve(sc, g, a, h)[k])),
              (g, a, h)) for g, a, h in in_cell), key=lambda t: t[0])
        dist[n_dev] += 1
        # two routes to one level — the citation's, and the knobs' (F49)
        dev_keys = tuple(k for k in knobs
                         if d[k] != _as_text(resolve(sc, g, a, h)[k]))
        want = level_of(sc, g, a, h, deviations=dev_keys)
        got = d.get("derived_level") or _knob_level(d)
        if want != got:
            clash.append(f"{d['__path__'].parent.name[:40]}: citation "
                         f"{sc}/{g}/{a}/{h}{'+dev' if n_dev else ''} says {want}, "
                         f"knobs say {got}")
    cited = sum(dist.values())
    print("\n=== R2 expressibility, three address forms (F47/F50) ===")
    print(f"  {n_l0} L0 run(s): `{{sc}}/L0`")
    print(f"  {sum(probes.values())} probe run(s) in unregistered cells, "
          f"ledger-addressed: "
          + ", ".join(f"{k}×{v}" for k, v in sorted(probes.items())))
    print(f"  {cited} tuple-cited run(s), search restricted to the run's cell:")
    for n in sorted(dist):
        note = ("  <- the bases, exactly" if n == 0 else
                "  <- a base plus the knob a control moved" if n <= 2 else
                "  <- an arm whose id was withdrawn (F44)")
        print(f"    {n:2} deviation(s)  {dist[n]:3} run(s){note}")
    print(f"  inexpressible : {len(bad)}")
    for line in bad[:5]:
        print(f"    {line}")
    print(f"  citation-level vs knob-level disagreements : {len(clash)}")
    for line in clash[:5]:
        print(f"    {line}")


def _knob_level(d: dict) -> str:
    """`derive_level` for a run whose args log predates `derived_level`."""
    import adi_flex_ppo_train as train
    return train.derive_level(_ns_from_log(d, train))


def main() -> None:
    runs = _load_runs()
    training = [d for d in runs if not d["__tuning__"]]
    trials = [d for d in runs if d["__tuning__"]]
    studies = sorted(db for r in _result_roots()
                     for db in r.joinpath("tuning").glob("*.db"))
    print(f"archive: {len(_result_roots())} results root(s)")
    for r in _result_roots():
        k = len([d for d in runs if not d["__tuning__"]
                 and str(d["__path__"]).startswith(str(r))])
        print(f"  {k:4} training run(s)  {r}")
    print(f"{len(training)} training run(s) under results/")
    print(f"{len(trials)} tuning trial(s) EXCLUDED — a trial is a sampled point, "
          f"not a chosen run, and its params are in {len(studies)} Optuna "
          f"stud(y/ies)")
    print("  (their args logs are still the only record of a trial's RESOLVED "
          "config — the DB holds the sampler's coordinates — which is where "
          "auto-mdp-solver#64 was found)\n")
    _report_levels(training)
    # ALL runs, not just training: a crowned `h` row's artifact is a tuning
    # trial (§9.7 ships the winner without a retrain), and the check filters
    # to `live` internally for every promoter that is not a path.
    _report_phantom(runs)
    _report_r1d()
    _report_r2(training)


if __name__ == "__main__":
    main()

"""Catalog ⊕ selection resolution for the MDP-IR (IR_LAYERING_PLAN.md §10).

A domain is authored as **one** ``{domain}_schema.json`` whose uncertainty
slots each declare a *pool of candidates* — the IR mirror of the generator
classes in ``_uncertainty.py``::

    "uncertainty_slots": [{
      "name": "demand", "interface": "DemandGenerator", "stream_id": 0,
      "stages": [...], "default": "discrete",
      "candidates": {
        "discrete": {"generator": "DiscreteDemand", "family": "categorical",
                     "settings": {...}},
        "poisson":  {"generator": "PoissonDemand", "family": "poisson",
                     "settings": {"rate": {"draw": {"family": "gamma", ...},
                                           "example": 30.0}}}
      }
    }]

A **selection** maps each slot to one candidate. It comes from the slots'
``default``s, overridden by the named instance's slot-valued keys (instances
carry constants *and* selections — the IR mirror of the ``SCENARIOS``
registry), overridden by an explicit ``--select slot=candidate``. Composing
families is a reference, never a file: declarations scale per-slot, and the
slot×family cross-product is never enumerated anywhere.

``resolve_catalog()`` merges catalog + selection into the ordinary
:class:`~mdp_ir.schema.MdpIR` document the interpreter, differential runner
and conformance harness already consume — the layering lives in the loader,
so nothing downstream learns a new shape:

- the selected candidate becomes the slot's ``UncertaintySource``;
- a setting whose value is a **draw spec** is the slot's world latent: it is
  desugared into a synthesized placeholder constant (``{slot}_{setting}``,
  value = the spec's ``example``) plus a ``ScenarioSampler`` at
  ``substream_id = slot.stream_id`` — one stream identity per source of
  randomness, on both seed branches, declared once in the structure. Draws
  run in settings-declaration order from one rng, so multi-draw recipes
  (a support then its weights) stay bit-reproducible;
- instances inconsistent with the active selection are dropped and slot keys
  stripped from the survivors, so the resolved IR is internally consistent
  (common random numbers across instances sharing a candidate fall out:
  they share the slot's meta stream);
- symbolic bounds (``"20 * demand.mean"``) resolve against a read-API
  namespace **derived** from :mod:`mdp_ir.families` (lazily, per attribute,
  composing through the latent hierarchy) — never hand-authored; a
  candidate's ``read_api`` block is the explicit override for underivable
  cases.

What is frozen is computed, not tagged: :func:`structural_fingerprint`
hashes the structural core plus the constant *names* its expressions
reference — shape and names frozen; values, candidates, instances, mixtures
free (hash-tracked, never re-confirmed).
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from mdp_ir import families

# read-API a slot exposes to symbolic bounds; derived via mdp_ir.families,
# overridable per candidate via its `read_api` block
READ_API = ("max", "mean", "min", "is_discrete")

# moment/envelope derivations per read-API attribute
_FAMILY_FNS = {"mean": families.mean, "max": families.max_value,
               "min": families.min_value}

_CANDIDATE_KEYS = {"generator", "family", "settings", "is_discrete", "read_api", "desc"}
_DRAW_KEYS = {"draw", "example", "hidden"}

# expressions are evaluated in a namespace of constants + slot read-APIs only
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "round": round,
    "int": int, "float": float, "sum": sum,
}


class LayeringError(ValueError):
    """Raised when a catalog schema is ill-formed or fails to resolve."""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def is_catalog(data: dict) -> bool:
    """True for the catalog form (``mdp.uncertainty_slots``); False for a
    legacy resolved single file (``mdp.uncertainty_sources``). The retired
    structure⊕binding kinds get a pointed error — they were never published."""
    kind = data.get("kind")
    if kind in ("structure", "binding"):
        raise LayeringError(
            f"kind={kind!r} is the retired structure⊕binding format "
            "(IR_LAYERING_PLAN.md §10): merge into a single catalog "
            "{domain}_schema.json with per-slot candidates"
        )
    return "uncertainty_slots" in (data.get("mdp") or {})


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------


def _eval_expr(expr: str, ns: dict, where: str) -> Any:
    try:
        return eval(  # noqa: S307 — namespace is restricted to constants + slots
            compile(expr, f"<{where}>", "eval"), {"__builtins__": _SAFE_BUILTINS}, ns
        )
    except LayeringError:
        raise
    except Exception as exc:
        raise LayeringError(f"{where}: cannot evaluate {expr!r} ({exc})") from exc


def _is_draw(value: Any) -> bool:
    return isinstance(value, dict) and "draw" in value


def _make_resolver(const_values: dict, where: str):
    """families.Resolver over this catalog's constants: literals pass, exprs
    evaluate over constants, draw specs recurse into their draw family."""

    def resolve(raw: Any, attr: str) -> Any:
        if _is_draw(raw):
            d = raw["draw"]
            return _FAMILY_FNS[attr](d.get("family"), d.get("settings", {}), resolve)
        if isinstance(raw, str):
            try:
                return _eval_expr(raw, dict(const_values), where)
            except LayeringError as exc:
                raise families.FamilyError(str(exc)) from exc
        return raw

    return resolve


def _slot_namespace(slot: dict, cand: dict, const_values: dict, where: str):
    """Lazy read-API object for one slot under its selected candidate:
    attribute access derives via the family registry (or the candidate's
    explicit ``read_api`` override); a slot nothing references never derives,
    so state-dependent settings are legal there."""
    explicit = cand.get("read_api") or {}
    family, settings = cand.get("family"), cand.get("settings", {})
    slot_name = slot["name"]
    resolve = _make_resolver(const_values, f"{where}:{slot_name}")

    class _Slot:
        def __getattr__(self, item: str) -> Any:
            if item not in READ_API:
                raise LayeringError(
                    f"{where}: slot {slot_name!r} has no read-API attribute "
                    f"{item!r}; available: {list(READ_API)}"
                )
            if item in explicit:
                raw = explicit[item]
                return (
                    _eval_expr(raw, dict(const_values), f"{where}:{slot_name}.{item}")
                    if isinstance(raw, str) else raw
                )
            try:
                if item in _FAMILY_FNS:
                    return _FAMILY_FNS[item](family, settings, resolve)
                if "is_discrete" in cand:
                    return cand["is_discrete"]
                derived = families.is_discrete(family)
                if derived is None:
                    raise families.FamilyError(f"unknown family {family!r}")
                return derived
            except families.FamilyError as exc:
                raise LayeringError(
                    f"{where}: cannot derive {slot_name}.{item} for family "
                    f"{family!r} ({exc}); add read_api.{item} to the candidate"
                ) from exc

    return _Slot()


def _mixture_namespace(slot: dict, comp_infos: list, where: str):
    """Envelope read-API for one slot under a mixture (spec §5.3): weighted
    mean of means, max of maxes, min of mins over the components' selected
    candidates (each with its own constant overrides); ``is_discrete`` must
    agree across components. Lazy like ``_slot_namespace``: an unreferenced
    attribute never derives."""
    slot_name = slot["name"]

    class _Mix:
        def __getattr__(self, item: str) -> Any:
            if item not in READ_API:
                raise LayeringError(
                    f"{where}: slot {slot_name!r} has no read-API attribute "
                    f"{item!r}; available: {list(READ_API)}"
                )
            weights, vals = [], []
            for w, comp_sel, cvals in comp_infos:
                cand = slot["candidates"][comp_sel[slot_name]]
                sub = _slot_namespace(slot, cand, cvals, where)
                weights.append(float(w))
                vals.append(getattr(sub, item))
            if item == "mean":
                return sum(w * v for w, v in zip(weights, vals)) / sum(weights)
            if item == "max":
                return max(vals)
            if item == "min":
                return min(vals)
            if len(set(vals)) > 1:      # is_discrete
                raise LayeringError(
                    f"{where}: mixture components disagree on "
                    f"{slot_name}.is_discrete: {vals}"
                )
            return vals[0]

    return _Mix()


def _resolve_bounds(
    value: Any, ns: dict, where: str, instance_consts: set[str]
) -> Any:
    """Resolve a bounds list whose entries may be numbers or expressions.

    A bare string that names an instance-overridden scenario constant is left
    untouched: that is the schema's existing *per-instance* bounds mechanism
    (``Decision.bounds``), and collapsing it here would freeze one instance's
    value for all of them.
    """
    if not isinstance(value, list):
        return value
    out = []
    for entry in value:
        if isinstance(entry, str):
            if entry in instance_consts:
                out.append(entry)          # leave to per-instance resolution
                continue
            out.append(_eval_expr(entry, ns, where))
        else:
            out.append(entry)
    return out


def _walk_bounds(node: Any, ns: dict, where: str, instance_consts: set[str]) -> None:
    """Resolve every ``bounds`` / ``element_bounds`` list in a nested document.

    Handles both the plain list form and the ``Confirmable`` form, whose
    ``value`` / ``suggested`` carry the lists.
    """
    if isinstance(node, dict):
        for key, val in node.items():
            if key in ("bounds", "element_bounds"):
                if isinstance(val, list):
                    node[key] = _resolve_bounds(
                        val, ns, f"{where}.{key}", instance_consts
                    )
                    continue
                if isinstance(val, dict):
                    for tag in ("value", "suggested"):
                        if isinstance(val.get(tag), list):
                            val[tag] = _resolve_bounds(
                                val[tag], ns, f"{where}.{key}.{tag}", instance_consts
                            )
                    continue
            _walk_bounds(val, ns, f"{where}.{key}", instance_consts)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk_bounds(item, ns, f"{where}[{i}]", instance_consts)


# ---------------------------------------------------------------------------
# Catalog validation helpers
# ---------------------------------------------------------------------------


def _validate_slots(slots: list, where: str) -> None:
    names, ids = set(), {}
    for slot in slots:
        for required in ("name", "interface", "stream_id", "stages", "default", "candidates"):
            if required not in slot:
                raise LayeringError(
                    f"{where}: slot {slot.get('name', '<unnamed>')!r} is missing {required!r}"
                )
        if slot["name"] in names:
            raise LayeringError(f"{where}: duplicate slot {slot['name']!r}")
        names.add(slot["name"])
        ids.setdefault(slot["stream_id"], []).append(slot["name"])
        if not slot["candidates"]:
            raise LayeringError(f"{where}: slot {slot['name']!r} declares no candidates")
        if slot["default"] not in slot["candidates"]:
            raise LayeringError(
                f"{where}: slot {slot['name']!r} default {slot['default']!r} is not "
                f"among its candidates {sorted(slot['candidates'])}"
            )
        for cname, cand in slot["candidates"].items():
            unknown = set(cand) - _CANDIDATE_KEYS
            if unknown:
                raise LayeringError(
                    f"{where}: candidate {slot['name']}.{cname} has unknown key(s) "
                    f"{sorted(unknown)}; allowed: {sorted(_CANDIDATE_KEYS)}"
                )
            for required in ("generator", "family"):
                if required not in cand:
                    raise LayeringError(
                        f"{where}: candidate {slot['name']}.{cname} must supply {required!r}"
                    )
            for key, raw in (cand.get("settings") or {}).items():
                if _is_draw(raw):
                    unknown = set(raw) - _DRAW_KEYS
                    if unknown:
                        raise LayeringError(
                            f"{where}: draw spec {slot['name']}.{cname}.{key} has "
                            f"unknown key(s) {sorted(unknown)}; allowed: {sorted(_DRAW_KEYS)}"
                        )
    shared = {k: v for k, v in ids.items() if len(v) > 1}
    if shared:
        raise LayeringError(f"{where}: slots share stream_id: {shared}")


def _build_selection(
    slots: list, instances_raw: dict, instance: str | None,
    select: dict[str, str] | None, where: str,
) -> dict[str, str]:
    by_name = {s["name"]: s for s in slots}
    selection = {s["name"]: s["default"] for s in slots}

    def apply(source: str, key: str, cand: str) -> None:
        if cand not in by_name[key]["candidates"]:
            raise LayeringError(
                f"{where}: {source} selects unknown candidate {key}={cand!r}; "
                f"slot {key!r} offers {sorted(by_name[key]['candidates'])}"
            )
        selection[key] = cand

    if instance is not None:
        for key, val in (instances_raw.get(instance) or {}).items():
            if key in by_name:
                apply(f"instance {instance!r}", key, val)
    for key, val in (select or {}).items():
        if key not in by_name:
            raise LayeringError(
                f"{where}: --select names unknown slot {key!r}; "
                f"slots: {sorted(by_name)}"
            )
        apply("--select", key, val)
    return selection


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_catalog(
    data: dict, instance: str | None = None, select: dict[str, str] | None = None,
    _expand_mixtures: bool = True,
) -> dict:
    """Resolve a catalog document under a selection into a full IR dict.

    ``instance`` may also name a mixture: the document resolves under the
    default selection (⊕ ``select``), components whose own selection differs
    get a per-component resolution stashed in ``mixture_resolutions`` (the
    interpreter swaps it in per episode — cross-family mixtures, §10f), and
    symbolic bounds resolve against the §5.3 component envelope (weighted
    mean of means, max of maxes). ``_expand_mixtures`` is internal recursion
    control: component resolutions never expand their own mixtures."""
    merged = copy.deepcopy(data)
    mdp = merged.get("mdp") or {}
    where = (merged.get("domain") or {}).get("name") or "catalog"

    slots = mdp.pop("uncertainty_slots", None)
    if slots is None:
        raise LayeringError(f"{where}: not a catalog (no mdp.uncertainty_slots)")
    _validate_slots(slots, where)
    slot_names = {s["name"] for s in slots}

    scenario = mdp.setdefault("scenario", {})
    if scenario.get("samplers"):
        raise LayeringError(
            f"{where}: the catalog form declares world latents as draw specs "
            "inside candidate settings, not scenario.samplers"
        )
    instances_raw = scenario.get("instances") or {}
    selection = _build_selection(slots, instances_raw, instance, select, where)

    # --- sources + world-latent desugar ------------------------------------
    consts = scenario.setdefault("constants", [])
    known = {c["name"] for c in consts}
    samplers: list[dict] = []
    sources: list[dict] = []
    for slot in slots:
        cand_name = selection[slot["name"]]
        cand = slot["candidates"][cand_name]
        settings: dict[str, Any] = {}
        draws: list[dict] = []
        hidden_flags: list[bool] = []
        for key, raw in (cand.get("settings") or {}).items():
            if not _is_draw(raw):
                settings[key] = copy.deepcopy(raw)
                continue
            synth = f"{slot['name']}_{key}"
            if synth in known:
                raise LayeringError(
                    f"{where}: draw for {slot['name']}.{key} would synthesize "
                    f"constant {synth!r}, which is already declared"
                )
            if "example" not in raw:
                raise LayeringError(
                    f"{where}: draw spec {slot['name']}.{cand_name}.{key} needs an "
                    f"'example' placeholder value (becomes constant {synth!r})"
                )
            consts.append({
                "name": synth, "value": copy.deepcopy(raw["example"]), "axis": "",
                "desc": f"world-latent placeholder for {slot['name']}.{key}; "
                        f"realized per episode by the {slot['name']}_latent sampler",
            })
            known.add(synth)
            draws.append({"name": synth, "distribution": copy.deepcopy(raw["draw"])})
            hidden_flags.append(bool(raw.get("hidden", True)))
            settings[key] = synth
        if draws:
            if len(set(hidden_flags)) > 1:
                raise LayeringError(
                    f"{where}: slot {slot['name']!r} mixes hidden and observed "
                    "draws; split them across slots or align the flags"
                )
            samplers.append({
                "name": f"{slot['name']}_latent",
                "substream_id": slot["stream_id"],
                "hidden": hidden_flags[0],
                "instances": [],
                "draws": draws,
                "desc": f"per-episode latents of the {slot['name']} slot "
                        f"({cand_name} candidate); substream = slot stream_id",
            })
        source: dict[str, Any] = {
            "name": slot["name"],
            "stream_id": slot["stream_id"],
            "latent": slot.get("latent", False),
            "stages": copy.deepcopy(slot.get("stages", [])),
            "generator": cand["generator"],
            "distribution": {"family": cand["family"], "settings": settings},
        }
        if cand.get("desc"):
            source["distribution"]["note"] = cand["desc"]
        if "is_discrete" in cand:
            source["is_discrete"] = cand["is_discrete"]
        else:
            derived = families.is_discrete(cand["family"])
            if derived is not None:
                source["is_discrete"] = derived
        sources.append(source)
    mdp["uncertainty_sources"] = sources
    scenario["samplers"] = samplers

    # --- instances: drop those inconsistent with the selection, strip keys --
    # a mixture component is kept regardless: its divergent selection gets a
    # per-component resolution below, and the interpreter still needs its
    # constant overrides when the mixture draws it
    mixtures = scenario.get("mixtures") or []
    mixture_comps = {
        comp for m in mixtures for _w, comp in m.get("components", []) if comp
    }
    kept: dict[str, dict] = {}
    for iname, overrides in instances_raw.items():
        sel_keys = {k: v for k, v in overrides.items() if k in slot_names}
        if any(selection[k] != v for k, v in sel_keys.items()) \
                and iname not in mixture_comps:
            continue
        kept[iname] = {k: v for k, v in overrides.items() if k not in slot_names}
    scenario["instances"] = kept

    # --- mixtures: composition-scoped drawers; component resolutions -------
    slot_ids = {s["stream_id"] for s in slots}
    for m in mixtures:
        if m.get("substream_id", 0) in slot_ids:
            raise LayeringError(
                f"{where}: mixture {m.get('name')!r} substream_id "
                f"{m.get('substream_id', 0)} collides with a slot stream_id; "
                "mixtures allocate above the slot range"
            )
    if _expand_mixtures:
        # cross-family mixtures (§10f): a component whose own selection
        # (defaults ⊕ its slot-valued keys — its standalone load, no --select)
        # differs from the active one is resolved separately; the interpreter
        # swaps in its sources/samplers per episode (standalone equivalence)
        top_const_names = {c["name"] for c in consts}
        resolutions: dict[str, dict[str, dict]] = {}
        for m in mixtures:
            comp_res: dict[str, dict] = {}
            for _w, comp in m.get("components", []):
                comp_sel = _build_selection(
                    slots, instances_raw, comp or None, None, where
                )
                if comp_sel == selection:
                    continue
                resolved = resolve_catalog(
                    data, instance=comp or None, _expand_mixtures=False
                )
                comp_res[comp] = {
                    "selection": resolved["selection"],
                    "sources": resolved["mdp"]["uncertainty_sources"],
                    "samplers": resolved["mdp"]["scenario"]["samplers"],
                    "constants": {
                        c["name"]: c["value"]
                        for c in resolved["mdp"]["scenario"]["constants"]
                        if c["name"] not in top_const_names
                    },
                }
            if comp_res:
                resolutions[m["name"]] = comp_res
        if resolutions:
            merged["mixture_resolutions"] = resolutions

    # --- symbolic bounds ----------------------------------------------------
    const_values = {c["name"]: c["value"] for c in consts}
    ns: dict[str, Any] = dict(const_values)
    by_name = {s["name"]: s for s in slots}
    active_mixture = next(
        (m for m in mixtures if m.get("name") == instance), None
    )
    if active_mixture is not None:
        # loading a mixture: bounds resolve against the §5.3 component
        # envelope — weighted mean of means, max of maxes, min of mins
        comp_infos = []
        for w, comp in active_mixture.get("components", []):
            comp_sel = _build_selection(
                slots, instances_raw, comp or None, None, where
            )
            over = {
                k: v for k, v in (instances_raw.get(comp) or {}).items()
                if k not in slot_names
            }
            comp_infos.append((w, comp_sel, {**const_values, **over}))
        for slot in slots:
            ns[slot["name"]] = _mixture_namespace(slot, comp_infos, where)
    else:
        for slot in slots:
            ns[slot["name"]] = _slot_namespace(
                slot, by_name[slot["name"]]["candidates"][selection[slot["name"]]],
                const_values, where,
            )
    # constants any instance overrides must stay symbolic (per-instance bounds);
    # union over the RAW pool so a dropped instance cannot collapse a bound
    instance_consts = {
        k for overrides in instances_raw.values() for k in overrides
        if k not in slot_names
    }
    for block in ("mdp", "gym"):
        if block in merged:
            _walk_bounds(merged[block], ns, f"{where}.{block}", instance_consts)

    merged["selection"] = selection
    return merged


# ---------------------------------------------------------------------------
# Structural fingerprint (what is frozen)
# ---------------------------------------------------------------------------


def structural_fingerprint(data: dict) -> str:
    """Hash of the structural core of a catalog document: everything in the
    mdp block except the scenario node and the candidate pools, plus the
    sorted constant *names* the core's expressions reference (computed with
    the schema's tokenizer, never hand-tagged). Shape and names frozen;
    values and the menu free — appending a candidate/instance or editing a
    constant value never moves this fingerprint."""
    from mdp_ir.schema import _identifiers

    if not is_catalog(data):
        raise LayeringError("structural_fingerprint takes a catalog document")
    core = copy.deepcopy(data["mdp"])
    scenario = core.pop("scenario", {}) or {}
    slots = core.pop("uncertainty_slots", [])
    core["uncertainty_slots"] = [
        {k: s.get(k) for k in ("name", "interface", "stream_id", "latent", "stages")}
        for s in slots
    ]

    exprs: list[str] = []
    for t in (core.get("dynamics") or {}).get("transitions", []):
        if t.get("guard"):
            exprs.append(t["guard"])
        exprs += t.get("updates", [])
    for c in (core.get("objective") or {}).get("per_step_components", []):
        exprs.append(c.get("expr", ""))
    # invariants ride along in `core` (never popped), so their name/expr/scope
    # already move the fingerprint; collect their expressions too, so a
    # constant referenced ONLY by a claim still freezes its name
    for iv in core.get("invariants", []):
        exprs.append(iv.get("expr", ""))
    exprs += [v for v in (core.get("initial_state") or {}).values() if isinstance(v, str)]
    t_ = (core.get("horizon") or {}).get("T")
    if isinstance(t_, str):
        exprs.append(t_)
    es = (core.get("dynamics") or {}).get("event_sequence")
    if isinstance(es, str):
        # event order in control position (§10d): the constant NAME freezes
        exprs.append(es)
    for d in core.get("decisions", []):
        b = d.get("bounds") or {}
        entries = b.get("value", []) + b.get("suggested", []) if isinstance(b, dict) else list(b)
        exprs += [x for x in entries if isinstance(x, str)]
    for s in core["uncertainty_slots"]:
        for st in s.get("stages") or []:
            if st.get("trigger"):
                exprs.append(st["trigger"])
            exprs += st.get("key_exprs") or []

    declared = {c["name"] for c in scenario.get("constants", [])}
    refs: set[str] = set()
    for e in exprs:
        refs |= _identifiers(e) & declared
    core["referenced_constants"] = sorted(refs)
    canonical = json.dumps(core, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]

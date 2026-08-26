"""The launch-time L1 check — spec §8.6, called by the train script on itself.

The pipeline had two executable gates and they sit at opposite ends of a run's
life: `mdp_conformance` asks *is this code spec-shaped?* before anything runs,
and `mdp_gates` asks *does this number clear its baselines?* after everything
has. The launch — *is this experiment well-posed?* — sat between them with
nothing to say, and that is where the expensive mistakes are made: a derivation
carried to a scale it was not measured at, a banner printing a rule the run
violates, a knob nobody chose. §8.6 already asks the script to *state* what it
derived and from what. This turns that statement into a check.

What it refuses is deliberately small — three rows §8.6 calls *never*:

* a **stale basis**, in the one form that is machine-decidable: the
  derivation's own rollout rule held at the T̄ it was measured on and does not
  hold at this run's. That is the script asserting a rule it is violating —
  not a prediction that the run will do badly — and the remedy is a
  re-derivation (a new L1 generation that re-bases the ladder above it), never
  a command-line override, which would record a broken derivation as an
  ordinary L2(hp) escalation and leave the campaign citing a ruler it had
  already replaced. A basis measured on a *different instance* is only a
  warning: a sibling scenario at the same scale leaves every row valid, and
  refusing it would make the check something to work around;
* **γ > β**: more bias against the objective *and* more variance;
* an **inverted schedule**: a final above its init, which §8.2's one-degree-of-
  freedom rule exists to make impossible.

Everything else reports. The rollout floors stay warnings because §8.6 measured
the ≥10-episode rule as a co-factor rather than a cliff, and the **prior** rows
— `norm_obs`, `normalize_advantage` — are never asserted at all: their §8.6
rows state a prior that only a run can settle, so a deviation there is an
experiment, not a violation.

Two further groups join it once a campaign keeps a §13 config registry, and
they answer the other half of "is this well-posed?" — not *are the knobs
derived* but *is this run the run it says it is*:

* **the configuration** — the args diff against the cited base must equal the
  declared deviation list exactly, the base must fix every knob an axis claims,
  the run must land in its own scenario's results tree, and every cited id must
  have a definition a reader can look up;
* **the comparator** — the run this arm will be read against must exist, share
  its scenario, and carry a comparable budget.

`assert_launch` runs whichever of the three the caller has inputs for.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["check_l1", "check_config", "check_comparator", "assert_launch",
           "assert_l1_current", "read_args_log", "registry_ids"]

# Derived rows whose value is a PRIOR the campaign may flip (§8.6): a run that
# deviates is doing the thing the row asks for, so it is reported, never judged.
PRIOR_ROWS: frozenset[str] = frozenset({"norm_obs", "normalize_advantage"})

# (init, final) schedule pairs — §8.2: the final is derived from the init, so a
# final above its init means the schedule inverted rather than decayed.
SCHEDULE_PAIRS: tuple[tuple[str, str], ...] = (
    ("learning_rate", "lr_final"), ("clip_init", "clip_final"))


def _get(args: Any, name: str, default=None):
    return getattr(args, name, default)


def _derived_rollout(derived: dict[str, object], args: Any) -> int | None:
    """n_steps x n_envs as the derivation set them, falling back to the run."""
    n_steps = derived.get("n_steps", _get(args, "n_steps"))
    n_envs = derived.get("n_envs", _get(args, "n_envs", 1))
    if isinstance(n_steps, int) and isinstance(n_envs, int):
        return n_steps * n_envs
    return None


def check_l1(args: Any, *, derived: dict[str, object],
             basis: dict[str, object] | None = None,
             beta: float | None = None,
             episode_len: float | None = None
             ) -> tuple[list[str], list[str], list[str]]:
    """``(errors, warnings, deviations)`` for one resolved argument namespace.

    ``derived`` is the script's ``_L1_DERIVED`` (§8.4); ``basis`` its
    ``_L1_BASIS`` — what the derivation was measured on. Both are read, never
    inferred: a check that re-derives the rules it is checking would agree with
    itself. ``episode_len`` is **this run's** T̄, which the caller knows from
    the scenario it just resolved; the basis carries the T̄ the derivation was
    made at, and comparing the two is the whole staleness question.
    """
    errors: list[str] = []
    warnings: list[str] = []
    deviations: list[str] = []

    basis = basis or {}
    basis_len = basis.get("episode_len")
    if episode_len is None:
        episode_len = basis_len           # no drift check available
    if beta is None:
        # β is a property of the problem the derivation was made against, so it
        # is basis rather than a derived value — and reading it from the
        # derived γ instead would make the γ ≤ β check tautological on a domain
        # that derives γ = β, and wrong on one that derives γ = 1 − 1/T̄.
        beta = basis.get("beta")

    # 1. the basis, before anything is measured against it
    want = basis.get("scenario_name")
    got = _get(args, "scenario_name")
    if want and got and want != got:
        warnings.append(
            f"the derivation was measured on {want!r}, this run is {got!r} — "
            f"same scale keeps every row valid, a moved T̄ does not (§8.6)")
    if basis_len and episode_len and abs(float(episode_len) / float(basis_len) - 1) > 0.2:
        warnings.append(
            f"T̄ has moved: derived at T~{basis_len}, this run is T~{episode_len} "
            f"— re-examine the rows that read it (gae_lambda's credit horizon, "
            f"the rollout composition, the budget in episodes)")
    # the one machine-decidable staleness failure: the derivation's OWN rollout
    # rule held where it was measured and does not hold here. Judged on the
    # DERIVED rollout, not this run's — a small rollout chosen on the command
    # line is the operator's L2 move and stays a warning below, while this is
    # the script printing a rule it is violating.
    d_rollout = _derived_rollout(derived, args)
    if d_rollout and basis_len and episode_len:
        was, now = d_rollout / float(basis_len), d_rollout / float(episode_len)
        if was >= 10 > now:
            errors.append(
                f"the derivation is STALE for this instance: its rollout of "
                f"{d_rollout} transitions held {was:.1f} episodes at the T~"
                f"{basis_len} it was measured on and holds {now:.1f} at this "
                f"run's T~{episode_len}, so the >=10-episode row it states is "
                f"one this run breaks. Re-derive (a new L1 generation, §8.6) "
                f"rather than override — a stale derivation recorded as an L2 "
                f"knob leaves the campaign citing a ruler it has replaced.")

    # 2. what the run resolved to, against what was derived
    for dest, value in sorted(derived.items()):
        if not hasattr(args, dest):
            continue
        actual = getattr(args, dest)
        actual = tuple(actual) if isinstance(actual, list) else actual
        want_v = tuple(value) if isinstance(value, list) else value
        if actual != want_v:
            deviations.append(f"{dest} {want_v!r}->{actual!r}"
                              + (" (prior row)" if dest in PRIOR_ROWS else ""))

    # 3. the three refusals
    gamma = _get(args, "gamma")
    if beta is not None and isinstance(gamma, (int, float)) and gamma - beta > 1e-9:
        errors.append(f"gamma={gamma} exceeds beta={beta} — §8.6 rules γ > β "
                      f"out entirely (more bias against the objective AND more "
                      f"variance)")
    for init, final in SCHEDULE_PAIRS:
        a, b = _get(args, init), _get(args, final)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > a:
            errors.append(f"{final}={b} is above {init}={a} — the schedule "
                          f"inverts (§8.2: the final is derived from the init)")

    # 4. the floors, which stay warnings on §8.6's own evidence
    n_steps, n_envs = _get(args, "n_steps"), _get(args, "n_envs", 1)
    if isinstance(n_steps, int) and isinstance(n_envs, int):
        rollout = n_steps * n_envs
        if rollout < 2048:
            warnings.append(f"rollout {n_steps}x{n_envs}={rollout} transitions "
                            f"< 2048 (§8.6 rollout row)")
        if episode_len:
            episodes = rollout / float(episode_len)
            if episodes < 10:
                warnings.append(
                    f"rollout holds {episodes:.1f} episodes at T~{episode_len} "
                    f"< the 10-episode floor (§8.6) — a co-factor, not a cliff, "
                    f"but re-derive if T̄ has moved")
    frac = _get(args, "checkpoint_every_frac")
    if isinstance(frac, (int, float)) and frac <= 0:
        warnings.append("checkpoint_every_frac=0 saves no checkpoints, so "
                        "§9.7's post-hoc selection screen cannot run and the "
                        "terminal checkpoint becomes the deliverable by default")
    elif frac is None:
        warnings.append("checkpoint_every_frac is unset — §8.2 defaults it to "
                        "0.05 (~20 checkpoints for §9.7's screen)")
    return errors, warnings, deviations


# ---------------------------------------------------------------------------
# The registry half (upstream #62 checks 4 and 5, deferred at v0.9.23)
#
# Both were held back for one reason: they resolve against the config
# registry's data half, `{domain}_configs.py`, which existed in no worktree.
# It does now, so they land — and the shape they land in is the one the L1
# check already uses. The harness is handed the RESOLVED base and the DECLARED
# deviations as data; it never imports the registry, never learns its module
# API, and never re-resolves a citation. That is not squeamishness about
# coupling: a check that re-derives what it is checking agrees with itself, and
# the registry's own resolver already computes the deviation list as the diff
# it is supposed to be verified against. What makes this a check rather than a
# restatement is WHEN it runs — after every derived-from-args adjustment the
# script makes, at the moment of use, against `args` as they will actually be
# handed to the learner. The resolver runs at the top of `main()`; anything
# that moves a knob afterwards silently invalidates the record it wrote.
# ---------------------------------------------------------------------------

_ID_ANCHOR = re.compile(r'<a id="([A-Za-z][A-Za-z0-9_.-]*)"></a>')


def read_args_log(path: str | Path) -> dict[str, str]:
    """The §8.4 args log as a mapping — one ``key: value`` per line."""
    out: dict[str, str] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def registry_ids(log: str | Path) -> set[str] | None:
    """Every id the log's §CONFIG-REGISTRY defines an anchor for, or None when
    the section is absent (a campaign that has not adopted §13 owes nothing).

    The guide fixes the anchor form — ``<a id="g1"></a>`g1``` — precisely so a
    ledger entry can link to a definition, which is also what makes the table
    machine-readable without parsing its columns.
    """
    text = Path(log).read_text(errors="replace")
    head = re.search(r"^##\s+CONFIG-REGISTRY.*$", text, re.M)
    if not head:
        return None
    rest = text[head.end():]
    nxt = re.search(r"^## ", rest, re.M)
    return set(_ID_ANCHOR.findall(rest[:nxt.start()] if nxt else rest))


def _norm(v: object) -> object:
    """Compare across sources: a list and a tuple of the same numbers are one
    value, and so are ``256`` and ``256.0`` — a registry literal, an argparse
    default and a JSON round-trip disagree on all three without disagreeing
    about anything."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, (int, float)):
        return float(v)
    return v


def _same(actual: object, declared: object) -> bool:
    """`actual` against a declared value that may have arrived as display text.

    A campaign records its deviations the way it prints them (``gamma
    1.0->0.99``), so the right-hand side reaches here as a string even when the
    knob is a float. Refusing that would make adoption mean rewriting the
    record rather than passing it in.
    """
    if _norm(actual) == _norm(declared):
        return True
    if isinstance(declared, str):
        return declared in {repr(actual), str(actual)}
    return False


def _declared_map(declared: object) -> dict[str, object | None]:
    """``{dest: value-or-None}`` from either form a campaign keeps.

    A mapping states what the run moved each knob *to*, so the value is checked
    as well as the fact; the string form (``"gamma 1.0->0.99"``, or a bare
    dest) states only the fact, and a dest whose value cannot be read is
    checked for presence alone rather than skipped.
    """
    if declared is None:
        return {}
    if isinstance(declared, Mapping):
        return dict(declared)
    out: dict[str, object | None] = {}
    for item in declared:
        text = str(item).strip()
        if not text or text == "-":
            continue
        dest = text.split()[0].split("=")[0]
        out[dest] = text.split("->", 1)[1].strip() if "->" in text else None
    return out


def check_config(args: Any, *, base: Mapping[str, object] | None,
                 declared: object = None, address: str = "",
                 owned: object = None, outdir: str | Path | None = None,
                 log: str | Path | None = None, ids: object = None,
                 ) -> tuple[list[str], list[str], list[str]]:
    """``(errors, warnings, notes)`` for check 4 — DECLARED == RESOLVED.

    ``base`` is the cited configuration as the registry resolves it (dest →
    value); ``declared`` this run's deviation list; ``address`` the citation
    itself (``"sc0/g2/a0/h2"``). An empty ``address`` is not a skip — it is a
    run that cites no base, which is the state §13 exists to end.

    ``owned`` is the set of dests the registry claims as *configuration*. Pass
    it and the base must be total over it: a knob an axis owns and no base
    fixes is one nobody can be shown to have chosen, which is the hole the
    check is for rather than an omission it should tolerate.

    ``log`` is the campaign's escalation log and ``ids`` the id set the data
    half actually defines. Given both, the two halves of §13 are asserted
    against each other — the reader-facing table cannot drift away from the
    module that configures runs, in either direction. Passing the id set keeps
    that decidable without this file learning any registry's module API, which
    at one implementation would be minting a contract from a sample of one.
    """
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    resolved = dict(base or {})
    dev = _declared_map(declared)

    if not address:
        errors.append(
            "this run cites no base — §13's whole subject is that an address "
            "says which question a run answers and never what it was "
            "configured from, so an uncited run is unreadable the moment its "
            "knobs stop matching the derivation")
        return errors, warnings, notes

    if owned:
        blind = sorted(d for d in owned
                       if d not in resolved and hasattr(args, d))
        if blind:
            errors.append(
                f"the base {address} resolves no value for {blind}, which an "
                f"axis owns — a run that moves one of those declares nothing "
                f"and diffs against nothing, so the knob is invisible in both "
                f"directions (R1a)")

    dests = set(owned) if owned else set(resolved) | set(dev)
    actual = {d: getattr(args, d) for d in dests if hasattr(args, d)}
    moved = {d: v for d, v in actual.items()
             if d in resolved and _norm(v) != _norm(resolved[d])}

    for d in sorted(set(moved) - set(dev)):
        errors.append(
            f"{d} resolves to {moved[d]!r} where {address} fixes "
            f"{resolved[d]!r}, and this run declares no deviation for it — an "
            f"undeclared second lever. Either cite a base that fixes it or "
            f"declare the move; a knob nobody chose is the incident class §13 "
            f"was built for")
    for d in sorted(set(dev) - set(moved)):
        errors.append(
            f"this run declares a deviation on {d} it does not have: "
            f"{address} fixes {resolved.get(d)!r} and the run resolves the "
            f"same — the banner describes a treatment the run is not running")
    for d in sorted(set(dev) & set(moved)):
        want = dev[d]
        if want is not None and not _same(moved[d], want):
            errors.append(
                f"{d} is declared as moving to {want!r} and resolves to "
                f"{moved[d]!r} — the record and the run disagree about the "
                f"same knob")

    scenario = _get(args, "scenario_name")
    if outdir is not None and scenario:
        parts = Path(outdir).resolve().parts
        if "results" in parts:
            i = len(parts) - 1 - parts[::-1].index("results")
            tree = parts[i + 1] if len(parts) > i + 1 else None
            if tree and tree != scenario:
                errors.append(
                    f"this run is scenario {scenario!r} and is writing into "
                    f"results/{tree}/ — spec §8.4 keys the tree by scenario, so "
                    f"the run lands where a reader will attribute it to "
                    f"{tree!r}")

    if log is not None:
        defined = registry_ids(log)
        if defined is None:
            notes.append(f"{Path(log).name} has no §CONFIG-REGISTRY section, so "
                         f"the cited ids are resolved but not readable")
        else:
            cited = [p for p in address.split("/")
                     if p and p.upper() != "L0"]
            unknown = [c for c in cited if c not in defined]
            if unknown:
                errors.append(
                    f"{address} cites {unknown}, which §CONFIG-REGISTRY defines "
                    f"no anchor for — the module that configures runs and the "
                    f"table a reader resolves them against have drifted, and "
                    f"the run would be addressed to a definition nobody can "
                    f"look up")
            if ids is not None:
                undocumented = sorted(set(ids) - defined)
                undefined = sorted(defined - set(ids))
                if undocumented:
                    errors.append(
                        f"the data half defines {undocumented}, which the log's "
                        f"§CONFIG-REGISTRY has no row for — a run can cite an id "
                        f"no reader can resolve")
                if undefined:
                    errors.append(
                        f"§CONFIG-REGISTRY documents {undefined}, which the data "
                        f"half does not define — the table describes bases that "
                        f"configure nothing (§13's ids are append-only, so a row "
                        f"is a promotion, not a plan)")
    if not errors:
        notes.append(f"config {address}: "
                     + (f"{len(moved)} declared deviation(s) — "
                        + "; ".join(f"{d}->{moved[d]!r}" for d in sorted(moved))
                        if moved else "at the cited base, no deviations"))
    return errors, warnings, notes


def _find_run(comparator: str | Path,
              results_root: str | Path | None) -> Path | None:
    """The run directory a comparator names, by path or by run name."""
    p = Path(comparator)
    if p.is_dir():
        return p
    if results_root is None:
        return None
    root = Path(results_root)
    if not root.is_dir():
        return None
    direct = [d for d in root.glob(f"*/{p.name}") if d.is_dir()]
    return direct[0] if direct else next(
        (d for d in sorted(root.rglob(p.name)) if d.is_dir()), None)


def check_comparator(args: Any, *, comparator: str | Path | None,
                     results_root: str | Path | None = None,
                     settling: bool = False, budget_tol: float = 0.1,
                     ) -> tuple[list[str], list[str], list[str]]:
    """``(errors, warnings, notes)`` for check 5 — COMPARATOR NAMED AND MATCHED.

    An arm names the run it will be read against, and the pairing has to hold
    at *launch*: the same scenario, a comparable budget, and a comparator that
    actually exists. All three failures are only visible here — at read time
    the numbers are in hand and the comparison looks like a result.

    ``settling`` is the deliberate exception: a base still being extended is a
    known budget mismatch a campaign has chosen to accept, so it downgrades to
    a warning instead of being silently tolerated.
    """
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    name = str(comparator).strip() if comparator is not None else ""
    if not name or name == "-":
        errors.append(
            "this arm names no comparator — the run it will be read against is "
            "chosen after the numbers are in, which is when a missing "
            "guardrail leg and a budget confound both stop being visible")
        return errors, warnings, notes

    run = _find_run(name, results_root)
    if run is None:
        errors.append(
            f"the comparator {name!r} names no run under "
            f"{results_root or '(no results root given)'} — a base that was "
            f"never run cannot be a base, and nothing else in the pipeline "
            f"looks")
        return errors, warnings, notes

    logs = sorted(run.glob("*_args.txt"))
    if not logs:
        errors.append(
            f"{run.name} carries no §8.4 args log, so the scenario and budget "
            f"this run will be compared against cannot be read at all")
        return errors, warnings, notes
    rec = read_args_log(logs[0])

    scenario = _get(args, "scenario_name")
    theirs = rec.get("scenario_name")
    if scenario and theirs and scenario != theirs:
        errors.append(
            f"the comparator {run.name} ran scenario {theirs!r} and this run "
            f"is {scenario!r} — two different problems, and the contrast would "
            f"read as a treatment effect")

    mine = _get(args, "total_timesteps")
    try:
        yours = float(str(rec.get("total_timesteps", "")).replace("_", ""))
    except ValueError:
        yours = None
    if isinstance(mine, (int, float)) and yours:
        ratio = float(mine) / yours
        if abs(ratio - 1.0) > budget_tol:
            said = (f"the comparator {run.name} trained {yours:.0f} steps and "
                    f"this run trains {float(mine):.0f} ({ratio:.2f}x)")
            if settling:
                warnings.append(
                    said + " — declared as a settling base, so the mismatch is "
                    "the campaign's call; the contrast still owes the caveat")
            else:
                errors.append(
                    said + " — a budget-confounded comparison reads as a "
                    "treatment effect. Match the budget, or declare the base "
                    "as settling")

    try:
        trained = float(str(rec.get("steps_trained", "")).replace("_", ""))
    except ValueError:
        trained = None
    if trained and yours and trained < yours * 0.99:
        warnings.append(
            f"the comparator {run.name} stopped at {trained:.0f} of "
            f"{yours:.0f} steps — a number from a run that stopped early is "
            f"not a worse result, it is not a result (§9.3)")
    if not errors:
        notes.append(f"comparator {run.name}: same scenario, "
                     + (f"{float(mine)/yours:.2f}x budget"
                        if isinstance(mine, (int, float)) and yours else
                        "budget unread"))
    return errors, warnings, notes


def assert_launch(args: Any, *, derived: dict[str, object],
                  basis: dict[str, object] | None = None,
                  beta: float | None = None,
                  episode_len: float | None = None,
                  base: Mapping[str, object] | None = None,
                  declared: object = None,
                  address: str | None = None,
                  owned: object = None,
                  outdir: str | Path | None = None,
                  log: str | Path | None = None, ids: object = None,
                  comparator: str | Path | None = None,
                  results_root: str | Path | None = None,
                  settling: bool = False,
                  stream=None) -> None:
    """Run every launch check the caller has the inputs for, print, exit on any
    error. Call it from ``main()`` once the args are fully resolved — *after*
    any knob the script derives from another, since checking before that is
    checking the resolver rather than the run.

    Three groups, each independently optional:

    * the **L1 derivation** (§8.6), always — `derived` is required;
    * the **configuration** (§13, upstream #62 check 4) when ``address`` is not
      None. ``None`` means this domain keeps no registry; ``""`` means it keeps
      one and this run cited nothing, which is an error rather than a skip;
    * the **comparator** (check 5) when ``comparator`` is not None, on the same
      rule — ``""`` is an arm that named none.

    An L0 run skips the first two: §8.6 defines L0 as the library's defaults
    plus what the problem forces, so neither the derivation's rows nor a
    registry base is in force, and checking a run against them would report the
    definition as a violation. The comparator check still runs, because an L0
    baseline is very often the thing another arm is read against.
    """
    out = stream or sys.stdout
    errors: list[str] = []
    is_l0 = str(_get(args, "level", "") or "").lower() == "l0"

    if is_l0:
        print("L1 check: skipped — L0 is faithful defaults (§8.6), so the "
              "derivation's rows are not in force", file=out)
    else:
        errs, warnings, deviations = check_l1(
            args, derived=derived, basis=basis, beta=beta,
            episode_len=episode_len)
        where = ""
        if basis and basis.get("scenario_name"):
            where = (f" (basis: {basis['scenario_name']}"
                     + (f", T~{basis['episode_len']}" if basis.get("episode_len") else "")
                     + ")")
        print(f"L1 check (spec §8.6): {len(derived)} derived row(s) checked{where}",
              file=out)
        print(f"  deviations: {'; '.join(deviations) if deviations else 'none — this run is at the L1 centre'}",
              file=out)
        for w in warnings:
            print(f"  WARNING: {w}", file=out)
        for e in errs:
            print(f"  FATAL: {e}", file=out)
        errors += errs

    if address is not None and not is_l0:
        errs, warns, notes = check_config(
            args, base=base, declared=declared, address=address, owned=owned,
            outdir=outdir, log=log, ids=ids)
        print("config check (§13, upstream #62):", file=out)
        for n in notes:
            print(f"  {n}", file=out)
        for w in warns:
            print(f"  WARNING: {w}", file=out)
        for e in errs:
            print(f"  FATAL: {e}", file=out)
        errors += errs
    elif address is not None:
        print("config check: skipped — L0 is not configured from the registry "
              "at all (guide §13.4), so a run cites {sc}/L0 and nothing resolves",
              file=out)

    if comparator is not None:
        errs, warns, notes = check_comparator(
            args, comparator=comparator, results_root=results_root,
            settling=settling)
        print("comparator check (§13, upstream #62):", file=out)
        for n in notes:
            print(f"  {n}", file=out)
        for w in warns:
            print(f"  WARNING: {w}", file=out)
        for e in errs:
            print(f"  FATAL: {e}", file=out)
        errors += errs

    if errors:
        raise SystemExit(
            "launch check failed — the run was not launched. Fix the "
            "derivation, the citation or the arguments; a check that can be "
            "argued past is a print statement.")


def assert_l1_current(args: Any, *, derived: dict[str, object],
                      basis: dict[str, object] | None = None,
                      beta: float | None = None,
                      episode_len: float | None = None,
                      stream=None) -> None:
    """The L1 rows alone — :func:`assert_launch` with no registry and no
    comparator. Kept as its own name because it is what every adopting script
    already calls, and because a domain with no config registry is not a domain
    that owes one: §13 graduated on two campaigns, not on all of them.
    """
    assert_launch(args, derived=derived, basis=basis, beta=beta,
                  episode_len=episode_len, stream=stream)

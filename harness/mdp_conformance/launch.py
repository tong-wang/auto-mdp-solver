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
"""

from __future__ import annotations

import sys
from typing import Any

__all__ = ["check_l1", "assert_l1_current"]

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


def assert_l1_current(args: Any, *, derived: dict[str, object],
                      basis: dict[str, object] | None = None,
                      beta: float | None = None,
                      episode_len: float | None = None,
                      stream=None) -> None:
    """Run :func:`check_l1` at startup, print the result, exit on an error.

    Call it from ``main()`` once the args are resolved. An L0 run is skipped
    outright: §8.6 defines L0 as the library's defaults plus what the problem
    forces, so the L1 rows are not in force and checking a run against them
    would report the definition as a violation.
    """
    out = stream or sys.stdout
    if str(_get(args, "level", "") or "").lower() == "l0":
        print("L1 check: skipped — L0 is faithful defaults (§8.6), so the "
              "derivation's rows are not in force", file=out)
        return
    errors, warnings, deviations = check_l1(
        args, derived=derived, basis=basis, beta=beta, episode_len=episode_len)
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
    if errors:
        for e in errors:
            print(f"  FATAL: {e}", file=out)
        raise SystemExit(
            "L1 check failed — the run was not launched. Fix the derivation or "
            "the arguments; a check that can be argued past is a print statement.")

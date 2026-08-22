"""Did the run that produced this number reach the budget it declared?

Spec §8.6 states the rule — "A training run runs to its budget — no early
stopping" — and nothing checked it at any stage. Conformance checks code shape,
the launch check (§8.6) verifies the derivation before step 0, and the eval gate
compares a finished number against its baselines. A run that stopped at 6% of
budget passes all three, and its artifacts actively disguise it: a model saved
at 0.6M of a 10M budget is a valid, loadable zip, and the eval it feeds produces
a well-formed TSV with correct standard errors. A short run and a bad arm are
then indistinguishable in the output — the gate reports the second correctly and
the first as a finding.

Two things this module deliberately does *not* ask, because both would fire on
healthy runs and a check campaigns route around is worse than none:

* **not "is the scored model at the budget?"** — under §9.7 the deliverable IS a
  mid-run checkpoint chosen post-hoc, so a scored model below the budget is the
  normal case. The question is where the *run* ended, not where its winner sat.
* **not "did the run finish?"** — a deliberately stopped run is a legitimate
  state, and a run directory that is a partial mirror of a completed one looks
  exactly like a truncated run from the outside. So the question is whether the
  step count is **known and declared**: reached budget passes, a short run whose
  operator declares it passes, an undeterminable one warns, and only a silently
  short run whose artifact is still scoreable fails.

Everything here reads SB3 zips with `zipfile` + `json` and never imports torch,
so the gate keeps working in the torch-free install.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Harvest", "model_steps", "run_terminus", "provenance_lines",
           "read_provenance", "harvest_verdict"]

# The keys an eval script stamps into its TSV's §9.3 provenance comments.
STEPS_SCORED = "steps_scored"    # num_timesteps of the model that produced it
STEPS_RUN = "steps_run"          # the furthest step count the run reached
STEPS_BUDGET = "steps_budget"    # the budget that run declared


@dataclass(frozen=True)
class Harvest:
    """One candidate's completion verdict."""

    status: str          # complete | declared | unknown | short
    detail: str

    @property
    def blocks(self) -> bool:
        return self.status == "short"


def model_steps(path: Path) -> tuple[int | None, int | None]:
    """``(num_timesteps, declared budget)`` from an SB3 zip.

    The budget comes from the artifact rather than the args log on purpose: it
    is `_total_timesteps`, recorded inside the zip by `learn()`, so a copied or
    stale args log cannot disagree with the model it claims to describe.
    """
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None, None
    steps = data.get("num_timesteps")
    budget = data.get("_total_timesteps")
    return (steps if isinstance(steps, int) else None,
            budget if isinstance(budget, int) else None)


def run_terminus(run_dir: Path) -> tuple[int | None, int | None]:
    """The furthest step count any artifact in a run dir reached, and its budget.

    Every zip is read — the terminal model, the final model, every checkpoint —
    because which of them exists is exactly what a partial copy changes, and the
    maximum is the only reading that is stable under one being absent.
    """
    run_dir = Path(run_dir)
    zips = sorted(run_dir.glob("*.zip")) + sorted(run_dir.glob("checkpoints/*.zip"))
    best = budget = None
    for z in zips:
        steps, declared = model_steps(z)
        if steps is not None and (best is None or steps > best):
            best = steps
        if declared is not None and (budget is None or declared > budget):
            budget = declared
    return best, budget


def provenance_lines(model_path: Path, run_dir: Path | None = None) -> list[str]:
    """The `#` lines an eval script stamps beside its TSV (spec §9.3).

    §9.3 already designates leading `#` lines as provenance that every §9 reader
    skips, and says to stamp them liberally — so the completion evidence travels
    with the number instead of being re-derived by whoever reads it later, from a
    run directory that may no longer be beside the TSV.
    """
    model_path = Path(model_path)
    scored, budget = model_steps(model_path)
    ran, run_budget = run_terminus(run_dir or model_path.parent)
    budget = run_budget or budget
    out = []
    if scored is not None:
        out.append(f"# {STEPS_SCORED}: {scored}")
    if ran is not None:
        out.append(f"# {STEPS_RUN}: {ran}")
    if budget is not None:
        out.append(f"# {STEPS_BUDGET}: {budget}")
    return out


def read_provenance(tsv_path: Path) -> dict[str, int]:
    """The integer `# key: value` provenance of one eval TSV.

    Tolerant by design: a domain's own free-form provenance line (`# ppo
    obs=stats | n_seeds=8192 ...`) is left alone, and a TSV written before this
    convention simply yields nothing — which the verdict reads as *unknown*
    rather than as a failure.
    """
    out: dict[str, int] = {}
    try:
        with open(tsv_path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                key, sep, value = line[1:].partition(":")
                if not sep:
                    continue
                try:
                    out[key.strip()] = int(value.strip())
                except ValueError:
                    continue
    except OSError:
        return {}
    return out


def harvest_verdict(provenance: dict[str, int], *, tolerance: float = 0.05,
                    declared: str | None = None) -> Harvest:
    """Read a TSV's completion provenance into a verdict.

    ``tolerance`` is one checkpoint cadence (§8.6 saves every ~5% of budget), so
    a completed run whose last artifact predates its final step still reads as
    complete. ``declared`` is the operator saying a short run was stopped on
    purpose — the one thing no artifact can carry, since a killed run and a
    silently dead one leave identical files.
    """
    ran = provenance.get(STEPS_RUN)
    budget = provenance.get(STEPS_BUDGET)
    if ran is None or not budget:
        return Harvest("unknown",
                       "completion is undeterminable — the TSV carries no "
                       f"{STEPS_RUN}/{STEPS_BUDGET} provenance (§9.3), so this "
                       "number may come from a run that stopped early")
    frac = ran / budget
    if frac >= 1.0 - tolerance:
        return Harvest("complete", f"ran to budget ({ran:,}/{budget:,})")
    short = (f"the run reached {ran:,} of the {budget:,} steps it declared "
             f"({frac:.1%})")
    if declared:
        return Harvest("declared", f"{short}, declared deliberate: {declared}")
    return Harvest("short", short + " and nothing declares that as intended — "
                   "§8.6: a training run runs to its budget. Re-run it, or pass "
                   "--short-ok '<reason>' to record the stop as deliberate")

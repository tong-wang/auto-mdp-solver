"""The harvest precondition — did the run that made this number reach budget?

Spec §8.6 says "a training run runs to its budget — no early stopping" and
nothing checked it. The artifacts disguise the violation rather than reveal it:
a model saved at 6% of budget is a valid, loadable zip, and the eval it feeds
produces a well-formed TSV with correct standard errors, so a short run and a
bad arm are indistinguishable in the output — the gate reports the second
correctly and the first as a finding.

The tests are mostly about what the check must NOT do. Two states look exactly
like truncation from outside and are legitimate: a run directory that is a
partial mirror of a completed one, and a run the operator killed on purpose. A
check that fails those is one campaigns route around, so the question is
whether the step count is known and declared, not whether the run finished.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from mdp_gates.compare import compare_evals
from mdp_gates.completion import (
    harvest_verdict, model_steps, provenance_lines, read_provenance,
    run_terminus,
)


def zip_model(path: Path, steps: int, budget: int) -> Path:
    """An SB3-shaped archive: `data` is the JSON member `learn()` writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", json.dumps({"num_timesteps": steps,
                                       "_total_timesteps": budget}))
        z.writestr("policy.pth", b"not read")
    return path


def tsv(path: Path, *, lines: list[str] = (), mean: float = 1.0) -> Path:
    body = "".join(f"{line}\n" for line in lines)
    path.write_text(f"{body}revenue_mean\trevenue_var\n{mean}\t0.01\n")
    return path


# --- reading the artifacts, without torch ---------------------------------

def test_the_budget_comes_from_the_zip_not_the_args_log(tmp_path):
    """`_total_timesteps` is recorded inside the archive by `learn()`, so a
    copied or stale args log cannot disagree with the model it describes."""
    assert model_steps(zip_model(tmp_path / "m.zip", 8_375_000, 10_000_000)) \
        == (8_375_000, 10_000_000)


def test_an_unreadable_file_is_unknown_rather_than_zero(tmp_path):
    (tmp_path / "broken.zip").write_bytes(b"not a zip")
    assert model_steps(tmp_path / "broken.zip") == (None, None)
    assert model_steps(tmp_path / "absent.zip") == (None, None)


def test_the_terminus_is_the_furthest_artifact_anywhere_in_the_run(tmp_path):
    zip_model(tmp_path / "checkpoints/ckpt_500_steps.zip", 500, 10_000)
    zip_model(tmp_path / "checkpoints/ckpt_9500_steps.zip", 9_500, 10_000)
    zip_model(tmp_path / "run_ppo.zip", 4_000, 10_000)   # a selected checkpoint
    assert run_terminus(tmp_path) == (9_500, 10_000)


def test_a_selected_checkpoint_below_budget_is_not_truncation(tmp_path):
    """§9.7 makes the deliverable a mid-run checkpoint chosen post-hoc, so the
    scored model sitting below the budget is the normal healthy case — the
    question is where the RUN ended, not where its winner sat."""
    zip_model(tmp_path / "best_model.zip", 8_375_000, 10_000_000)
    zip_model(tmp_path / "checkpoints/c.zip", 10_000_000, 10_000_000)
    lines = provenance_lines(tmp_path / "best_model.zip")
    assert lines == ["# steps_scored: 8375000", "# steps_run: 10000000",
                     "# steps_budget: 10000000"]
    assert harvest_verdict(read_provenance(tsv(tmp_path / "e.tsv", lines=lines))
                           ).status == "complete"


# --- the four verdicts ----------------------------------------------------

def test_reaching_the_budget_passes():
    assert harvest_verdict({"steps_run": 10_000_000,
                            "steps_budget": 10_000_000}).status == "complete"


def test_a_rollout_overshooting_the_budget_still_passes():
    """SB3 finishes the rollout it is in, so a run's last step count routinely
    exceeds the budget it declared. That is completion, not an anomaly."""
    assert harvest_verdict({"steps_run": 2048,
                            "steps_budget": 512}).status == "complete"


def test_a_silently_short_run_fails_with_its_fraction():
    v = harvest_verdict({"steps_run": 573_440, "steps_budget": 10_000_000})
    assert v.status == "short" and v.blocks
    assert "573,440" in v.detail and "5.7%" in v.detail


def test_a_deliberate_stop_passes_once_it_is_declared():
    """A killed run and a silently dead one leave identical artifacts, so the
    declaration is the one input no file can carry."""
    v = harvest_verdict({"steps_run": 573_440, "steps_budget": 10_000_000},
                        declared="killed at #E51")
    assert v.status == "declared" and not v.blocks and "#E51" in v.detail


def test_a_partial_mirror_is_undeterminable_not_short(tmp_path):
    """The case that fooled the proposal itself: mirroring copied only some
    artifacts, so `checkpoints/` stopped wherever the copy did and every local
    signal was indistinguishable from truncation."""
    v = harvest_verdict(read_provenance(tsv(tmp_path / "e.tsv")))
    assert v.status == "unknown" and not v.blocks


def test_one_cadence_short_still_reads_as_complete():
    """§8.6 saves a checkpoint every ~5% of budget, so a completed run whose
    last artifact predates its final step must not read as truncated."""
    assert harvest_verdict({"steps_run": 9_600_000,
                            "steps_budget": 10_000_000}).status == "complete"


# --- reading provenance off a TSV -----------------------------------------

def test_a_domains_own_provenance_line_is_left_alone(tmp_path):
    path = tsv(tmp_path / "e.tsv", lines=[
        "# ppo obs=stats policy=stochastic | n_seeds=8192 from 0",
        "# steps_run: 10000000", "# steps_budget: 10000000"])
    assert read_provenance(path) == {"steps_run": 10_000_000,
                                     "steps_budget": 10_000_000}


# --- the gate itself ------------------------------------------------------

def _gate(tmp_path, lines, **kw):
    cand = tsv(tmp_path / "cand.tsv", lines=lines, mean=10.0)
    base = tsv(tmp_path / "base.tsv", mean=1.0)
    return compare_evals(cand, [base], [], n_seeds=64,
                         metric="revenue_mean", **kw)


def test_a_short_run_fails_the_gate_even_when_it_beats_its_baseline(tmp_path):
    """The whole point: this arm wins on the numbers. A number from a run that
    stopped early is not a worse result, it is not a result."""
    report = _gate(tmp_path, ["# steps_run: 573440", "# steps_budget: 10000000"])
    assert all(c.passed for c in report.comparisons)
    assert not report.ok and report.harvest.blocks


def test_declaring_the_stop_restores_the_verdict(tmp_path):
    report = _gate(tmp_path, ["# steps_run: 573440", "# steps_budget: 10000000"],
                   short_ok="killed on purpose at #E51")
    assert report.ok and report.harvest.status == "declared"


def test_a_tsv_without_provenance_warns_and_still_gates(tmp_path):
    """Migration: every TSV written before this convention is undeterminable,
    and failing them all would make the check the first thing switched off."""
    report = _gate(tmp_path, [])
    assert report.ok and report.harvest.status == "unknown"

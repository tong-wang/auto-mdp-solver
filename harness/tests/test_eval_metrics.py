"""Root-level ``eval_metrics``: bystander report columns.

The contract under test (schema + interpreter):

- metrics are declared at the IR root, OUTSIDE the mdp block, so adding or
  editing one never moves ``mdp_fingerprint()`` (no Phase-A re-confirmation);
- each metric's expr is evaluated on the END_OF_PERIOD namespace every period
  and folded across the episode by its ``reduce``;
- metrics describe, never decide: ``total``/``reward`` are untouched;
- names must be unique and must not shadow the existing namespace.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mdp_ir.interpreter import IrInterpreter
from mdp_ir.schema import MdpIR


def _with_metrics(ir_doc: dict, metrics: list[dict]) -> dict:
    ir_doc["eval_metrics"] = metrics
    return ir_doc


def test_metrics_do_not_move_the_fingerprint(ir_doc):
    bare = MdpIR.model_validate(ir_doc)
    decorated = MdpIR.model_validate(_with_metrics(
        dict(ir_doc), [{"name": "peak_level", "expr": "level", "reduce": "max"}]
    ))
    assert bare.mdp_fingerprint() == decorated.mdp_fingerprint()


def test_reduce_semantics(ir_doc):
    ir = MdpIR.model_validate(_with_metrics(ir_doc, [
        {"name": "final_level", "expr": "level", "reduce": "last"},
        {"name": "peak_level", "expr": "level", "reduce": "max"},
        {"name": "trough_level", "expr": "level", "reduce": "min"},
        {"name": "total_outflow", "expr": "outflow", "reduce": "sum"},
    ]))
    traj = IrInterpreter(ir).run(episode_seed=7, decisions={"act": 2.0})

    levels = [row["level"] for row in traj.rows]
    outflows = [row["outflow"] for row in traj.rows]
    assert traj.metrics["final_level"] == pytest.approx(levels[-1])
    assert traj.metrics["peak_level"] == pytest.approx(max(levels))
    assert traj.metrics["trough_level"] == pytest.approx(min(levels))
    assert traj.metrics["total_outflow"] == pytest.approx(sum(outflows))


def test_metrics_never_touch_objective_or_reward(ir_doc):
    bare_traj = IrInterpreter(MdpIR.model_validate(dict(ir_doc))).run(
        episode_seed=3, decisions={"act": 1.0}
    )
    dec_traj = IrInterpreter(MdpIR.model_validate(_with_metrics(
        dict(ir_doc), [{"name": "peak_level", "expr": "level", "reduce": "max"}]
    ))).run(episode_seed=3, decisions={"act": 1.0})
    assert dec_traj.total == bare_traj.total
    assert dec_traj.total_reward == bare_traj.total_reward
    assert dec_traj.rows == bare_traj.rows
    assert not bare_traj.metrics and dec_traj.metrics


def test_metric_can_read_constants_and_components(ir_doc):
    # END_OF_PERIOD namespace: scenario constants and decomposition
    # components are legal references, same as objective components
    ir = MdpIR.model_validate(_with_metrics(ir_doc, [
        {"name": "worst_carry", "expr": "carry / h", "reduce": "max"},
    ]))
    traj = IrInterpreter(ir).run(episode_seed=11, decisions={"act": 4.0})
    assert traj.metrics["worst_carry"] == pytest.approx(
        max(row["carry"] for row in traj.rows)  # h == 1.0 in the base scenario
    )


def test_duplicate_names_rejected(ir_doc):
    with pytest.raises(ValidationError, match="declared twice"):
        MdpIR.model_validate(_with_metrics(ir_doc, [
            {"name": "m", "expr": "level"},
            {"name": "m", "expr": "outflow"},
        ]))


@pytest.mark.parametrize("taken", ["level", "outflow", "act", "h", "carry",
                                   "total", "reward"])
def test_shadowing_rejected(ir_doc, taken):
    with pytest.raises(ValidationError, match="shadows"):
        MdpIR.model_validate(_with_metrics(
            dict(ir_doc), [{"name": taken, "expr": "level"}]
        ))


def test_render_footer_mentions_metrics(ir_doc):
    ir = MdpIR.model_validate(_with_metrics(
        ir_doc, [{"name": "peak_level", "expr": "level", "reduce": "max"}]
    ))
    text = IrInterpreter(ir).run(episode_seed=1, decisions={"act": 1.0}).render()
    assert "eval metrics:" in text and "peak_level" in text

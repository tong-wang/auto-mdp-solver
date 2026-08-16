"""Run-log provenance, and the artifact-vs-declaration check it unblocks (#28, #26 item 4).

§8.6 already asked args.txt to carry the SB3 version and nothing checked it —
so the one provenance fact the spec required was missing from two of three
shipped examples. A spec line without a check is what produced that, which is
why the set and the check land together.
"""

from __future__ import annotations

import json

from mdp_conformance.checks import check_run_provenance
from mdp_conformance.loader import DomainHandle


def _domain(tmp_path, ir_doc, runs):
    (tmp_path / "d_schema.json").write_text(json.dumps(ir_doc))
    for name, lines in runs.items():
        run = tmp_path / "results" / "simple" / name
        run.mkdir(parents=True)
        (run / "simple_ppo_args.txt").write_text("".join(f"{k}: {v}\n" for k, v in lines.items()))
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


FULL = {"seed": 1, "algo_class": "PPO", "sb3_version": "2.6.0",
        "ir_mdp_fingerprint": "abc123"}


def test_no_run_directory_skips(tmp_path, ir_doc):
    assert check_run_provenance(_domain(tmp_path, ir_doc, {})).status == "SKIP"


def test_full_set_passes(tmp_path, ir_doc):
    h = _domain(tmp_path, ir_doc, {"PPO_1": FULL})
    assert check_run_provenance(h).status == "PASS"


def test_pre_convention_run_warns_rather_than_fails(tmp_path, ir_doc):
    """Adoption path: a run written before the convention is not a defect."""
    h = _domain(tmp_path, ir_doc, {"PPO_1": {"seed": 1, "observation_mode": "vec"}})
    result = check_run_provenance(h)
    assert result.status == "WARN" and "predate" in result.detail


def test_partial_adoption_fails(tmp_path, ir_doc):
    """Some keys but not all is a bug, not a legacy run."""
    h = _domain(tmp_path, ir_doc, {"PPO_1": {"seed": 1, "algo_class": "PPO"}})
    result = check_run_provenance(h)
    assert result.status == "FAIL" and "missing" in result.detail


def test_undeclared_algo_class_fails(tmp_path, ir_doc):
    """#26 item 4: the artifact's class must be one the IR declares."""
    rec = dict(FULL, algo_class="MaskablePPO")          # IR declares ppo only
    h = _domain(tmp_path, ir_doc, {"PPO_1": rec})
    result = check_run_provenance(h)
    assert result.status == "FAIL" and "not in declared" in result.detail


def test_declared_axis_admits_the_class(tmp_path, ir_doc):
    ir_doc["rl"]["algos"] = ["ppo", "maskable_ppo"]
    ir_doc["mdp"]["decisions"][0]["type"] = {"value": "discrete", "suggested": "discrete"}
    ir_doc["gym"]["action_modes"][0]["type"] = "discrete"
    h = _domain(tmp_path, ir_doc, {"PPO_1": dict(FULL, algo_class="MaskablePPO")})
    assert check_run_provenance(h).status == "PASS"


def test_unknown_class_name_fails(tmp_path, ir_doc):
    h = _domain(tmp_path, ir_doc, {"PPO_1": dict(FULL, algo_class="DreamerV3")})
    assert check_run_provenance(h).status == "FAIL"

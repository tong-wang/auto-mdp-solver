"""`mdp_conformance.launch.fit_run_name` — the run name fits one path component
(spec §8.4).

The defect (upstream #104): §8.4 encodes every knob off the derivation, and
`mdp_tuning --knobs all` sets every searched knob, so a knob-heavy draw grew
past NAME_MAX (255 bytes) and the run directory's `mkdir` raised ENAMETOOLONG
before training. The fixture is one of the four crashed names, verbatim.
"""

from __future__ import annotations

import pytest

from mdp_conformance.launch import NAME_MAX, fit_run_name

HEAD = "PPO_20261015_173158"
CRASHED = (
    "L1_obssingle_actprice-discrete_rewgap_steps4000000_seed718_lr3.54e-05_"
    "lrf3.54e-06_clip0.1_clipf0.025_nsteps1024_bs256_ep20_gamma0.998_"
    "gae0.996_ent2.35e-05_vf0.488_grad0.3_normadvFalse_arch128-128-128_"
    "normrewFalse_normobsFalse_vineTrue_vls0.03").split("_")


def test_the_crashed_name_was_past_the_limit():
    assert len("_".join([HEAD, *CRASHED]).encode()) == 262


def test_an_overlong_name_keeps_its_prefix_and_hashes_the_tail():
    name = fit_run_name(HEAD, CRASHED)
    assert len(name.encode()) <= NAME_MAX
    assert name == (
        "PPO_20261015_173158_L1_obssingle_actprice-discrete_rewgap_steps4000000_"
        "seed718_lr3.54e-05_lrf3.54e-06_clip0.1_clipf0.025_nsteps1024_bs256_"
        "ep20_gamma0.998_gae0.996_ent2.35e-05_vf0.488_grad0.3_normadvFalse_"
        "arch128-128-128_normrewFalse_h14db2ae738")   # tail normobsFalse_vineTrue_vls0.03
    assert fit_run_name(HEAD, CRASHED) == name        # deterministic


def test_a_name_that_fits_is_byte_identical():
    """No existing run directory or ledger join moves."""
    parts = ["obssales", "actdiscrete", "rewprofit", "steps5000000"]
    assert fit_run_name(HEAD, parts) == "_".join([HEAD, *parts])
    exact = ["x" * (NAME_MAX - len(HEAD) - 1)]
    assert fit_run_name(HEAD, exact) == f"{HEAD}_{exact[0]}"


def test_different_tails_get_different_markers():
    a = fit_run_name(HEAD, [*CRASHED[:-1], "vls0.03"])
    b = fit_run_name(HEAD, [*CRASHED[:-1], "vls0.05"])
    assert a != b and a[:-10] == b[:-10]


def test_bytes_not_characters_are_counted():
    parts = ["é" * 100, "ü" * 100]                  # 400 bytes, 200 characters
    assert len(fit_run_name(HEAD, parts).encode()) <= NAME_MAX


def test_a_head_with_no_room_is_refused():
    with pytest.raises(ValueError, match="no room"):
        fit_run_name("P" * 250, ["tag", "more"])

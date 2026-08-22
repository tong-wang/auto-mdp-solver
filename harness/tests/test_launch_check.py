"""`mdp_conformance.launch` — the derivation checked at launch (spec §8.6).

The gap it closes is one of timing rather than knowledge: conformance runs
before anything is launched and the eval gate after everything has, so the
question *is this experiment well-posed?* had nothing to answer it. The rule
the check enforces was already in the spec, already read, and read at the wrong
moment — one campaign printed a `>=10 episodes` banner unconditionally at a
scale where its runs held two.

What is tested here is mostly what the check REFUSES to refuse: a check that
fires on a legitimate run is one campaigns route around, so the refusals are
three rows §8.6 calls never, and everything else reports.
"""

from __future__ import annotations

import argparse

import pytest

from mdp_conformance.launch import assert_l1_current, check_l1

DERIVED = {"n_steps": 512, "n_envs": 4, "gamma": 1.0, "norm_obs": True}
BASIS = {"scenario_name": "3x3_20", "episode_len": 48, "beta": 1.0}


def args(**over):
    base = dict(scenario_name="3x3_20", gamma=1.0, n_steps=512, n_envs=4,
                checkpoint_every_frac=0.05, norm_obs=True)
    return argparse.Namespace(**{**base, **over})


def test_a_run_at_its_own_basis_is_clean():
    errors, warnings, deviations = check_l1(
        args(), derived=DERIVED, basis=BASIS, episode_len=48)
    assert not errors and not warnings and not deviations


def test_the_stale_derivation_is_refused_with_both_episode_counts():
    """The motivating incident: a rollout derived at T~48 carried to a scale
    where episodes run ~1000, with the banner still printing the rule."""
    errors, _, _ = check_l1(args(scenario_name="4x4_20"), derived=DERIVED,
                            basis=BASIS, episode_len=1030)
    assert len(errors) == 1
    assert "42.7 episodes" in errors[0] and "2.0" in errors[0]
    assert "Re-derive" in errors[0]


def test_a_sibling_at_the_same_scale_is_not_refused():
    """Scenario identity is the wrong test: a same-scale sibling leaves every
    row valid, and refusing it would make the check something to work around."""
    errors, warnings, _ = check_l1(args(scenario_name="3x3_zero"),
                                   derived=DERIVED, basis=BASIS, episode_len=48)
    assert not errors and len(warnings) == 1 and "3x3_zero" in warnings[0]


def test_a_smaller_scale_warns_but_does_not_refuse():
    """T̄ falling leaves the rollout holding MORE episodes, so the derivation's
    own rule still holds — the rows that read T̄ still want re-examining."""
    errors, warnings, _ = check_l1(args(), derived=DERIVED, basis=BASIS,
                                   episode_len=10)
    assert not errors and any("T̄ has moved" in w for w in warnings)


def test_a_small_rollout_chosen_on_the_command_line_only_warns():
    """The operator's own L2 move, not a stale derivation — §8.6 measured the
    floor as a co-factor, not a cliff, so it stays a warning."""
    errors, warnings, _ = check_l1(args(n_steps=64, n_envs=1), derived=DERIVED,
                                   basis=BASIS, episode_len=48)
    assert not errors
    assert any("< 2048" in w for w in warnings)


def test_gamma_above_beta_and_an_inverted_schedule_are_refused():
    errors, _, _ = check_l1(args(gamma=1.5), derived=DERIVED, basis=BASIS,
                            episode_len=48)
    assert len(errors) == 1 and "exceeds beta" in errors[0]
    errors, _, _ = check_l1(
        argparse.Namespace(learning_rate=1e-4, lr_final=1e-3),
        derived={}, basis=None)
    assert len(errors) == 1 and "inverts" in errors[0]


def test_a_prior_row_is_reported_never_refused():
    """§8.6's norm_obs and normalize_advantage rows state a prior only a run can
    settle (v0.9.20), so deviating there is the experiment the row asks for."""
    errors, warnings, deviations = check_l1(
        args(norm_obs=False), derived=DERIVED, basis=BASIS, episode_len=48)
    assert not errors and not warnings
    assert deviations == ["norm_obs True->False (prior row)"]


def test_no_checkpoints_warns_because_the_screen_cannot_run():
    _, warnings, _ = check_l1(args(checkpoint_every_frac=0), derived=DERIVED,
                              basis=BASIS, episode_len=48)
    assert any("post-hoc selection screen" in w for w in warnings)


def test_l0_is_skipped_because_the_rows_are_not_in_force(capsys):
    """L0 is the library's defaults plus what the problem forces, so checking a
    run against L1's rows would report the definition as a violation."""
    assert_l1_current(args(level="l0", gamma=99.0), derived=DERIVED, basis=BASIS)
    assert "skipped" in capsys.readouterr().out


def test_the_refusal_stops_the_run(capsys):
    with pytest.raises(SystemExit):
        assert_l1_current(args(gamma=1.5), derived=DERIVED, basis=BASIS,
                          episode_len=48)
    assert "FATAL" in capsys.readouterr().out


# --- the build-time half: what conformance can see without a launch --------

from mdp_conformance.checks import check_launch_check, check_selection_protocol
from mdp_conformance.loader import DomainHandle


class PlainEnv:
    def __init__(self, scenario, logger_filename=None): ...


def handle(tmp_path, source: str, scenarios=("simple",)) -> DomainHandle:
    (tmp_path / "d_ppo_train.py").write_text(source)
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={s: object() for s in scenarios},
        env_cls=PlainEnv, state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


CHECKED = ('_L1_BASIS = {"scenario_name": "simple", "episode_len": 30}\n'
           'def main():\n    assert_l1_current(args, derived={}, basis=_L1_BASIS)\n')


def test_a_script_that_checks_its_derivation_passes(tmp_path):
    assert check_launch_check(handle(tmp_path, CHECKED)).status == "PASS"


def test_a_script_that_only_prints_its_derivation_warns(tmp_path):
    r = check_launch_check(handle(tmp_path, 'def main():\n    print("L1: ...")\n'))
    assert r.status == "WARN" and "printed, not checked" in r.detail


def test_a_basis_naming_a_scenario_the_registry_lacks_fails(tmp_path):
    """Worse than no basis: the staleness comparison silently never fires."""
    r = check_launch_check(handle(tmp_path, CHECKED, scenarios=("other",)))
    assert r.status == "FAIL" and "SCENARIOS does not have" in r.detail


def test_a_live_selection_callback_is_reported_not_failed(tmp_path):
    """§9.7 excludes it outright, but a domain generated before the rule made
    every number it reports with the machinery it has — so the campaign is told
    at the moment of use rather than having its leaderboard invalidated."""
    r = check_selection_protocol(handle(
        tmp_path, "def main():\n    cb = MaskableEvalCallback(env)\n"))
    assert r.status == "WARN" and "MaskableEvalCallback" in r.detail


def test_a_checkpoint_flag_that_defaults_to_none_is_reported(tmp_path):
    """The capability/use distinction: the dest exists, so cli_contract is
    satisfied, and no run that forgets the flag saves a single checkpoint."""
    r = check_selection_protocol(handle(tmp_path, (
        "def _build_arg_parser():\n"
        "    p.add_argument('--checkpoint-every-frac', type=float, default=None)\n")))
    assert r.status == "WARN" and "no checkpoints" in r.detail


def test_a_conforming_selection_setup_passes(tmp_path):
    r = check_selection_protocol(handle(tmp_path, (
        "def _build_arg_parser():\n"
        "    p.add_argument('--checkpoint-every-frac', type=float, default=0.05)\n"
        "def main():\n    cb = CheckpointCallback(save_freq=1)\n")))
    assert r.status == "PASS"

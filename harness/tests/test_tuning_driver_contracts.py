"""Trial-scoring contracts: the artifact, the TSV reader, and boolean flags.

Each test here locks in a defect that was silent — the study still finished and
still printed a winner, but the ranking behind it was not the one the operator
asked for.
"""

from __future__ import annotations

import argparse
import os

import pytest

from mdp_tuning.driver import (
    ModelResolutionError, _parser_args, build_cmd, parse_overrides,
    resolve_model, tsv_column_means,
)


# --- the artifact each trial is scored on ---------------------------------

def _run_dir(trial_dir, scenario="simple", run="PPO_20260813_120000_default"):
    d = trial_dir / scenario / run
    d.mkdir(parents=True)
    # spec §8.4: the train script writes its args log into the run dir, which
    # is what marks this directory as the run rather than a subdirectory of it
    (d / f"{scenario}_ppo_args.txt").write_text("--scenario_name simple\n")
    return d


def test_canonical_prefers_the_run_dir_model_over_a_newer_checkpoint(tmp_path):
    """The defect: ranking by mtime let a periodic checkpoint win."""
    run = _run_dir(tmp_path)
    model = run / "simple_ppo.zip"
    model.write_bytes(b"final")
    ckpt = run / "checkpoints" / "rl_model_50000_steps.zip"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"checkpoint")
    # the checkpoint is the newest file on disk
    os.utime(ckpt, (model.stat().st_mtime + 100,) * 2)

    assert resolve_model(tmp_path, scenario="simple") == model
    assert resolve_model(tmp_path, scenario="simple", mode="final") == ckpt


def test_canonical_is_name_agnostic(tmp_path):
    """inv_single saves ppo_inv_single.zip, not {scenario}_ppo.zip — depth,
    not filename, has to be the discriminator."""
    run = _run_dir(tmp_path)
    model = run / "ppo_inv_single.zip"
    model.write_bytes(b"final")
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "rl_model_10_steps.zip").write_bytes(b"c")

    assert resolve_model(tmp_path, scenario="simple") == model


def test_two_top_level_models_raise_rather_than_guess(tmp_path):
    run = _run_dir(tmp_path)
    (run / "alpha.zip").write_bytes(b"a")
    (run / "beta.zip").write_bytes(b"b")

    with pytest.raises(ModelResolutionError):
        resolve_model(tmp_path, scenario="simple")


def test_ambiguity_is_broken_by_the_canonical_name(tmp_path):
    run = _run_dir(tmp_path)
    (run / "simple_ppo.zip").write_bytes(b"a")
    (run / "extra.zip").write_bytes(b"b")

    assert resolve_model(tmp_path, scenario="simple").name == "simple_ppo.zip"


def test_no_model_at_all_raises_file_not_found(tmp_path):
    _run_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path, scenario="simple")


def test_checkpoints_only_is_a_per_trial_failure_not_a_silent_score(tmp_path):
    """Training died before the final save. The lone checkpoint must NOT be
    scored as if it were canonical, and the failure must be FileNotFoundError
    — caught per trial — rather than aborting the whole study."""
    run = _run_dir(tmp_path)
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "rl_model_50000_steps.zip").write_bytes(b"c")

    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path, scenario="simple")
    # the operator can still opt into scoring it explicitly
    assert resolve_model(tmp_path, scenario="simple", mode="final").name \
        == "rl_model_50000_steps.zip"


def test_falls_back_to_depth_when_the_domain_writes_no_args_log(tmp_path):
    run = tmp_path / "simple" / "PPO_run"
    run.mkdir(parents=True)
    model = run / "ppo_x.zip"
    model.write_bytes(b"final")
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "rl_model_10_steps.zip").write_bytes(b"c")

    assert resolve_model(tmp_path, scenario="simple") == model


# --- the eval TSV reader ---------------------------------------------------

def test_comment_lines_are_skipped(tmp_path):
    """The defect: a leading '#' line became the header, so a completed trial
    was recorded as crashed. mdp_gates already tolerated these."""
    tsv = tmp_path / "eval.tsv"
    tsv.write_text(
        "# scenario: simple\n"
        "# model: simple_ppo.zip\n"
        "seed\tprofit_mean\n"
        "0\t10.0\n"
        "1\t20.0\n")

    cols, means = tsv_column_means(tsv)
    assert cols == ["seed", "profit_mean"]
    assert means["profit_mean"] == pytest.approx(15.0)


# --- boolean flags reach the domain script --------------------------------

def _args_of(*add):
    p = argparse.ArgumentParser()
    for fn in add:
        fn(p)
    return _parser_args(p)


def test_store_true_override_emits_a_bare_flag(tmp_path):
    """The defect: '--stochastic True' — argparse rejects it, so the flag
    silently kept the script's default."""
    args_map = _args_of(
        lambda p: p.add_argument("--stochastic", action="store_true"))
    assign = parse_overrides(["stochastic=true"], args_map, "eval")
    assert assign == {"stochastic": True}

    cmd = build_cmd(tmp_path / "eval.py", args_map, assign)
    assert cmd[-1] == "--stochastic"


def test_store_true_set_false_emits_nothing(tmp_path):
    args_map = _args_of(
        lambda p: p.add_argument("--stochastic", action="store_true"))
    cmd = build_cmd(tmp_path / "eval.py", args_map,
                    parse_overrides(["stochastic=false"], args_map, "eval"))
    assert "--stochastic" not in cmd


def test_store_false_dest_emits_its_negative_flag(tmp_path):
    args_map = _args_of(
        lambda p: p.add_argument("--no-warm-start", action="store_false",
                                 dest="warm_start", default=True))
    cmd = build_cmd(tmp_path / "train.py", args_map,
                    parse_overrides(["warm_start=false"], args_map, "train"))
    assert cmd[-1] == "--no-warm-start"


def test_boolean_optional_action_uses_both_forms(tmp_path):
    args_map = _args_of(
        lambda p: p.add_argument("--deterministic", default=True,
                                 action=argparse.BooleanOptionalAction))
    on = build_cmd(tmp_path / "eval.py", args_map,
                   parse_overrides(["deterministic=yes"], args_map, "eval"))
    off = build_cmd(tmp_path / "eval.py", args_map,
                    parse_overrides(["deterministic=no"], args_map, "eval"))
    assert on[-1] == "--deterministic"
    assert off[-1] == "--no-deterministic"


def test_non_boolean_word_for_a_flag_raises_with_the_spellings(tmp_path):
    args_map = _args_of(
        lambda p: p.add_argument("--stochastic", action="store_true"))
    with pytest.raises(ValueError, match="boolean flag"):
        parse_overrides(["stochastic=maybe"], args_map, "eval")


def test_unreachable_flag_value_raises(tmp_path):
    """store_true declared default=True: the bare flag can only store True, so
    False is unreachable — say so instead of emitting a no-op command."""
    args_map = _args_of(
        lambda p: p.add_argument("--strict", action="store_true", default=True))
    with pytest.raises(ValueError, match="BooleanOptionalAction"):
        build_cmd(tmp_path / "eval.py", args_map, {"strict": False})


def test_two_actions_on_one_dest_leave_one_value_unreachable(tmp_path):
    """The shape a runtime-derived default invites: `--norm-obs` (store_true)
    and `--no-norm-obs` (store_false) declared separately, both default None.
    A parser is read per *dest*, so the second action wins and the True arm is
    gone — a study opening the knob would die on its first True trial. The fix
    is one BooleanOptionalAction, which is what the error names."""
    args_map = _args_of(
        lambda p: p.add_argument("--norm-obs", action="store_true",
                                 dest="norm_obs", default=None),
        lambda p: p.add_argument("--no-norm-obs", action="store_false",
                                 dest="norm_obs", default=None))
    assert build_cmd(tmp_path / "train.py", args_map,
                     {"norm_obs": False})[-1] == "--no-norm-obs"
    with pytest.raises(ValueError, match="BooleanOptionalAction"):
        build_cmd(tmp_path / "train.py", args_map, {"norm_obs": True})


def test_a_none_default_reaches_the_space_as_a_refusal(tmp_path):
    """`l1_defaults` used to drop a None default, which made "the script
    derives this at runtime" indistinguishable from "the space encoded it" —
    trial 0 silently sampled the knob. Keeping it lets `encode` refuse it,
    which is what the launch banner reports."""
    from mdp_tuning.__main__ import l1_defaults
    from mdp_tuning.driver import ArgSpec, DomainScripts
    from mdp_tuning.spaces import encode_ppo

    scripts = DomainScripts(
        prefix="d", directory=tmp_path, algo="ppo",
        train_script=tmp_path / "d_ppo_train.py",
        eval_script=tmp_path / "d_ppo_eval.py",
        train_args={"norm_obs": ArgSpec(flag="--norm-obs", multi=False,
                                        default=None, takes_value=False,
                                        flag_const=True,
                                        negative_flag="--no-norm-obs")},
        eval_args={})
    defaults = l1_defaults(scripts, {"norm_obs"})
    assert defaults == {"norm_obs": None}
    assert encode_ppo(defaults)[1] == ["norm_obs"]


def test_value_taking_args_are_unaffected(tmp_path):
    args_map = _args_of(
        lambda p: p.add_argument("--learning_rate", type=float, default=3e-4),
        lambda p: p.add_argument("--net_arch", nargs="+", type=int))
    assign = parse_overrides(["learning_rate=0.001", "net_arch=64,64"],
                             args_map, "train")
    cmd = build_cmd(tmp_path / "train.py", args_map, assign)
    assert cmd[2:] == ["--learning_rate", "0.001", "--net_arch", "64", "64"]


def test_summary_labels_the_best_value_as_a_trial_layer_score(capsys):
    """Spec §9.7 (upstream #25): the number a human reads must carry its layer.

    A campaign compared a tuning best against a protocol leaderboard and closed
    the round as a null result; re-scored properly the same artifacts won. The
    spec now forbids the comparison — this puts the warning where the error is
    made.
    """
    import optuna

    from mdp_tuning.__main__ import print_summary

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=3)
    print_summary(study, eval_seeds=512)
    out = capsys.readouterr().out
    assert "TRIAL-LAYER score, 512 seeds" in out
    assert "not comparable to a protocol number" in out
    assert "maximum over 3 trials" in out       # the selection effect, named

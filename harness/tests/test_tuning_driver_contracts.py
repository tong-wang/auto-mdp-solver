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


def test_value_taking_args_are_unaffected(tmp_path):
    args_map = _args_of(
        lambda p: p.add_argument("--learning_rate", type=float, default=3e-4),
        lambda p: p.add_argument("--net_arch", nargs="+", type=int))
    assign = parse_overrides(["learning_rate=0.001", "net_arch=64,64"],
                             args_map, "train")
    cmd = build_cmd(tmp_path / "train.py", args_map, assign)
    assert cmd[2:] == ["--learning_rate", "0.001", "--net_arch", "64", "64"]

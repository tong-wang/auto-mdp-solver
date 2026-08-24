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


def test_the_warm_start_centre_is_the_derivation_not_the_default(tmp_path):
    """After a promotion the two part company, and every Δ(L2−L1) the study
    reports is measured from whichever one the warm start believed."""
    from mdp_tuning.__main__ import l1_defaults, promoted_knobs
    from mdp_tuning.driver import ArgSpec, DomainScripts

    scripts = DomainScripts(
        prefix="d", directory=tmp_path, algo="ppo",
        train_script=tmp_path / "d_ppo_train.py",
        eval_script=tmp_path / "d_ppo_eval.py",
        # gae_lambda has been promoted: the script now defaults to the
        # discovered 0.9, while the §8.6 derivation still says 0.95
        train_args={"gae_lambda": ArgSpec(flag="--gae-lambda", multi=False,
                                          default=0.9),
                    "ent_coef": ArgSpec(flag="--ent-coef", multi=False,
                                        default=0.005)},
        eval_args={},
        l1_derived={"gae_lambda": 0.95, "ent_coef": 0.005})
    assert l1_defaults(scripts, {"gae_lambda", "ent_coef"}) == {
        "gae_lambda": 0.95, "ent_coef": 0.005}
    assert promoted_knobs(scripts) == {"gae_lambda": (0.95, 0.9)}


def test_a_promotion_outside_the_searched_tier_is_still_reported(tmp_path):
    """The case that actually cost something: a trial used to carry a flag only
    for the knobs the study searched, so a promoted knob OUTSIDE the tier ran at
    the promoted default in every trial — trial 0 included — and Δ(L2−L1) was
    measured from a centre that is not L1 in a knob nobody asked it to move.
    Every trial now carries the derivation, and this stays reported because the
    study's runs still differ from a hand-run of the script. Scanning only the
    searched knobs would miss exactly this one."""
    from mdp_tuning.__main__ import promoted_knobs
    from mdp_tuning.driver import ArgSpec, DomainScripts

    scripts = DomainScripts(
        prefix="d", directory=tmp_path, algo="ppo",
        train_script=tmp_path / "d_ppo_train.py",
        eval_script=tmp_path / "d_ppo_eval.py",
        train_args={"max_grad_norm": ArgSpec(flag="--max-grad-norm",
                                             multi=False, default=1.0)},
        eval_args={},
        l1_derived={"max_grad_norm": 0.5})
    assert promoted_knobs(scripts) == {"max_grad_norm": (0.5, 1.0)}


def test_a_script_without_a_derivation_still_warm_starts_from_defaults(tmp_path):
    """§8.6's own statement — for a pre-convention script the defaults ARE the
    L1 centre — so nothing changes for it, and nothing is reported as promoted."""
    from mdp_tuning.__main__ import l1_defaults, promoted_knobs
    from mdp_tuning.driver import ArgSpec, DomainScripts

    scripts = DomainScripts(
        prefix="d", directory=tmp_path, algo="ppo",
        train_script=tmp_path / "d_ppo_train.py",
        eval_script=tmp_path / "d_ppo_eval.py",
        train_args={"gae_lambda": ArgSpec(flag="--gae-lambda", multi=False,
                                          default=0.9)},
        eval_args={})
    assert l1_defaults(scripts, {"gae_lambda"}) == {"gae_lambda": 0.9}
    assert promoted_knobs(scripts) == {}


# --- what a trial's command actually carries (upstream #64) ---------------

def _train_scripts(tmp_path, train_args: dict, derived: dict):
    """A domain whose train script exposes `train_args` on top of the three
    dests §8.2 requires of every train script, and derives `derived`."""
    from mdp_tuning.driver import ArgSpec, DomainScripts

    required = {
        "scenario_name": ArgSpec(flag="--scenario_name", multi=False,
                                 default="simple"),
        "total_timesteps": ArgSpec(flag="--total-timesteps", multi=False,
                                   default=1_000_000),
        "outdir": ArgSpec(flag="--outdir", multi=False, default="results"),
    }
    return DomainScripts(
        prefix="d", directory=tmp_path, algo="ppo",
        train_script=tmp_path / "d_ppo_train.py",
        eval_script=tmp_path / "d_ppo_eval.py",
        train_args={**required, **train_args}, eval_args={},
        l1_derived=derived)


def _adi_flex_like(tmp_path):
    """The motivating shape: the derivation says 0.02 / 0.05 and the parser
    defaults are null, because L0 must be able to have no KL valve and a
    constant schedule — so here the derived value *cannot* be the default."""
    from mdp_tuning.driver import ArgSpec

    return _train_scripts(
        tmp_path,
        {"target_kl": ArgSpec(flag="--target-kl", multi=False, default=None),
         "clip_final": ArgSpec(flag="--clip-final", multi=False, default=None),
         "learning_rate": ArgSpec(flag="--learning-rate", multi=False,
                                  default=3e-4)},
        {"target_kl": 0.02, "clip_final": 0.05, "learning_rate": 3e-4})


def _cmd(scripts, cfg, trial_dir, fixed_train=None, total_timesteps=50_000):
    from mdp_tuning.__main__ import train_assignment

    args = argparse.Namespace(total_timesteps=total_timesteps, seed=42)
    assign = train_assignment(scripts, args, "simple", trial_dir,
                              fixed_train or {}, cfg)
    return build_cmd(scripts.train_script, scripts.train_args, assign)


def test_every_trial_carries_the_derivation(tmp_path):
    """The defect: a trial's command set the searched knobs and let every other
    dest fall to the parser default, so a knob whose derived value is not its
    default ran off-derivation in every trial. The campaign that found this ran
    583 trials across two studies at target_kl=None against a derived 0.02."""
    cmd = _cmd(_adi_flex_like(tmp_path), {"learning_rate": 0.001}, tmp_path)
    assert cmd[cmd.index("--target-kl") + 1] == "0.02"
    assert cmd[cmd.index("--clip-final") + 1] == "0.05"


def test_the_sampler_wins_over_the_derivation(tmp_path):
    """Leaving the centre is what a searched knob is for."""
    cmd = _cmd(_adi_flex_like(tmp_path), {"target_kl": 0.05}, tmp_path)
    assert cmd.count("--target-kl") == 1
    assert cmd[cmd.index("--target-kl") + 1] == "0.05"


def test_a_train_arg_pin_wins_over_the_derivation(tmp_path):
    cmd = _cmd(_adi_flex_like(tmp_path), {}, tmp_path,
               fixed_train={"target_kl": "0.1"})
    assert cmd.count("--target-kl") == 1
    assert cmd[cmd.index("--target-kl") + 1] == "0.1"


def test_the_study_budget_wins_over_a_derived_one(tmp_path):
    """The three study-level dests stay the study's to set: a trial runs a
    trial budget into a trial directory, whatever the derivation says."""
    scripts = _train_scripts(tmp_path, {}, {"total_timesteps": 5_000_000})
    cmd = _cmd(scripts, {}, tmp_path, total_timesteps=50_000)
    assert cmd.count("--total-timesteps") == 1
    assert cmd[cmd.index("--total-timesteps") + 1] == "50000"


def test_a_runtime_derived_knob_is_left_off_the_command_line(tmp_path):
    """§8.4's carve-out: mab computes norm_obs inside parse_args, so there is no
    value to put on a command line and omitting the flag IS how the derivation
    is applied."""
    from mdp_tuning.driver import ArgSpec

    scripts = _train_scripts(
        tmp_path,
        {"norm_obs": ArgSpec(flag="--norm-obs", multi=False, default=None,
                             takes_value=False, flag_const=True,
                             negative_flag="--no-norm-obs")},
        {"norm_obs": None})
    cmd = _cmd(scripts, {}, tmp_path)
    assert "--norm-obs" not in cmd and "--no-norm-obs" not in cmd


def test_fix_holds_a_knob_at_its_derived_value_not_its_default(tmp_path):
    """Spec §8.6 says --fix "holds a knob at its derived value"; the flag's own
    help said "at the train script's default" and the code matched the help, so
    a study told to stay inside one layer left it whenever the two differed."""
    from mdp_tuning.__main__ import resolve_tunable
    from mdp_tuning.driver import ArgSpec

    scripts = _train_scripts(
        tmp_path,
        {"gae_lambda": ArgSpec(flag="--gae-lambda", multi=False, default=0.9)},
        {"gae_lambda": 0.95})
    args = argparse.Namespace(algo="ppo", knobs="core", fix=["gae_lambda"])
    assert "gae_lambda" not in resolve_tunable(scripts, args, {})
    # so it never reaches cfg — and what the command carries is the derivation
    cmd = _cmd(scripts, {}, tmp_path)
    assert cmd[cmd.index("--gae-lambda") + 1] == "0.95"


def test_a_derivation_the_cli_cannot_carry_fails_at_launch(tmp_path):
    """Seeding the derivation onto every trial makes one new way to fail, and
    it fails identically on all of them — so it fails once, before the study
    spends a budget proving it. The shape: a store_true declared default=True,
    where the bare flag can only store True and the derived False is
    unreachable."""
    from mdp_tuning.__main__ import make_objective
    from mdp_tuning.driver import ArgSpec

    scripts = _train_scripts(
        tmp_path,
        {"strict": ArgSpec(flag="--strict", multi=False, default=True,
                           takes_value=False, flag_const=True)},
        {"strict": False})
    args = argparse.Namespace(algo="ppo", knobs="core", fix=[], beta=1.0,
                              episode_len=None, min_rollout_episodes=10,
                              total_timesteps=50_000, seed=42)
    with pytest.raises(SystemExit, match="cannot carry"):
        make_objective(scripts, args, "simple", tmp_path, {}, {})


def test_a_null_default_is_reported_as_promoted(tmp_path):
    """The guard that hid all of the above: `default is not None` skipped
    exactly the case _L1_DERIVED exists to express — a derived value that
    cannot be the default. No banner fired for 583 trials."""
    from mdp_tuning.__main__ import promoted_knobs

    assert promoted_knobs(_adi_flex_like(tmp_path)) == {
        "clip_final": (0.05, None), "target_kl": (0.02, None)}


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

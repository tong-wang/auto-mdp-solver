"""`scripts.cli_contract` — the tier-1 CLI surface (spec §8.2, §9.1).

The convention had already converged: the four IR-era domains agreed on every
contested name. It was written down nowhere, so each new domain re-derived it,
and the misses were invisible until something downstream tried to join on a
name that was not there. These tests cover the three shapes that produced the
misses — a name the spec used but nothing checked, a name owed only where it
applies, and a script that does not exist yet — plus the dest rule itself,
since a check that misreads `dest=` would report flags that are right there.

The same failure in reverse gets its own coverage: a reader that stops at one
file, or that reads prose as code, reports names a domain already offers and
knobs it could not honour. Both cost the report its signal, since a warning
that cannot be cleared honestly is one nobody reads.

Synthetic scripts only: the check must not learn the shape of any shipped
domain.
"""

from __future__ import annotations

import pytest

from mdp_conformance.checks import (
    _add_argument_dests, check_schedule_pairs, check_script_cli_contract,
)
from mdp_conformance.loader import DomainHandle


class PlainEnv:
    def __init__(self, scenario, logger_filename=None): ...


class RenderingEnv:
    def __init__(self, scenario, observation_mode="vec", action_mode="box",
                 reward_mode="profit", logger_filename=None): ...


TRAIN = ["scenario_name", "total_timesteps", "outdir", "seed", "tag",
         "n_envs", "gym_log", "checkpoint_every_frac"]
EVAL = ["scenario_name", "model_path", "outfile", "n_seeds", "first_seed"]


def parser(dests, extra: str = "") -> str:
    """A script whose parser declares exactly `dests` (plus any raw `extra`)."""
    body = "\n".join(f'    p.add_argument("--{d.replace("_", "-")}")' for d in dests)
    return ("import argparse\n\n\ndef _build_arg_parser():\n"
            "    p = argparse.ArgumentParser()\n" + body + "\n" + extra + "    return p\n")


def domain(tmp_path, *, env_cls=PlainEnv, train=None, eval=None) -> DomainHandle:
    if train is not None:
        (tmp_path / "d_ppo_train.py").write_text(train)
    if eval is not None:
        (tmp_path / "d_ppo_eval.py").write_text(eval)
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=env_cls,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


# --- the dest rule --------------------------------------------------------

@pytest.mark.parametrize("call, dest", [
    ('p.add_argument("--first-seed", type=int)', "first_seed"),
    ('p.add_argument("-s", "--scenario_name")', "scenario_name"),
    ('p.add_argument("--no-norm-obs", action="store_false", dest="norm_obs")', "norm_obs"),
    ('p.add_argument("--learning_rate", "--lr-init", dest="learning_rate")', "learning_rate"),
    ('p.add_argument("-l", type=float)', "l"),
])
def test_the_dest_follows_argparse_not_the_flag_spelling(call, dest):
    assert dest in _add_argument_dests(call)


def test_an_unparseable_script_yields_no_dests_rather_than_raising():
    assert _add_argument_dests("def (:") == set()


# --- what is owed, and when -----------------------------------------------

def test_a_domain_with_no_train_script_skips(tmp_path):
    """A domain still in Phase A owes no scripts, so it owes no names."""
    assert check_script_cli_contract(domain(tmp_path)).status == "SKIP"


def test_the_full_surface_passes(tmp_path):
    h = domain(tmp_path, train=parser(TRAIN), eval=parser(EVAL))
    result = check_script_cli_contract(h)
    assert result.status == "PASS", result.detail
    assert "2 script(s)" in result.detail


def test_a_missing_name_is_reported_against_its_own_script(tmp_path):
    h = domain(tmp_path, train=parser([d for d in TRAIN if d != "tag"]),
               eval=parser([d for d in EVAL if d != "first_seed"]))
    result = check_script_cli_contract(h)
    assert result.status == "WARN"
    assert "d_ppo_train.py: missing tag" in result.detail
    assert "d_ppo_eval.py: missing first_seed" in result.detail


def test_an_absent_eval_script_is_not_a_violation(tmp_path):
    """The whole point of reading the parsers that *exist*: a Phase-B-stage-0
    gate must not report a stage the campaign has not claimed."""
    result = check_script_cli_contract(domain(tmp_path, train=parser(TRAIN)))
    assert result.status == "PASS"
    assert "no d_ppo_eval.py yet" in result.detail


# --- the two conditional groups -------------------------------------------

def test_a_rendering_mode_is_owed_only_when_the_env_takes_it(tmp_path):
    h = domain(tmp_path, env_cls=RenderingEnv, train=parser(TRAIN))
    assert check_script_cli_contract(h).status == "WARN"
    h = domain(tmp_path, env_cls=PlainEnv, train=parser(TRAIN))
    assert check_script_cli_contract(h).status == "PASS"


def test_the_rendering_modes_are_named_when_the_env_renders(tmp_path):
    h = domain(tmp_path, env_cls=RenderingEnv, train=parser(TRAIN))
    detail = check_script_cli_contract(h).detail
    assert "action_mode" in detail and "observation_mode" in detail and "reward_mode" in detail


def test_vecnormalize_names_are_owed_only_where_the_wrapper_is_built(tmp_path):
    """§8.3 is explicit that VecNormalize is SB3-only; a domain that omits it
    (drifting obs, §8.3) owes no flag for a wrapper it never constructs."""
    h = domain(tmp_path, train=parser(TRAIN))
    assert check_script_cli_contract(h).status == "PASS"
    h = domain(tmp_path, train=parser(TRAIN) + "\nenv = VecNormalize(env)\n")
    result = check_script_cli_contract(h)
    assert result.status == "WARN"
    assert "norm_obs" in result.detail and "vecnorm_clip_obs" in result.detail


def test_naming_the_wrapper_in_prose_owes_nothing(tmp_path):
    """A domain that implements its own reward scaling and mentions the wrapper
    only to say what it is an analog OF builds none: the three knobs would
    control nothing, and an args log recording a normalization the run never had
    is a worse artifact than the warning it silences."""
    prose = ('\n# Reward scale: the VecNormalize analog (gamma = 1)\n'
             '"""Own RunningReturnScale, not VecNormalize."""\n')
    h = domain(tmp_path, train=parser(TRAIN) + prose)
    assert check_script_cli_contract(h).status == "PASS"


def test_an_import_alone_does_not_oblige_the_eval(tmp_path):
    h = domain(tmp_path, train=parser(TRAIN),
               eval="from stable_baselines3.common.vec_env import VecNormalize\n"
                    + parser(EVAL))
    assert check_script_cli_contract(h).status == "PASS"


def test_the_eval_owes_vecnorm_path_when_it_loads_the_stats(tmp_path):
    h = domain(tmp_path, train=parser(TRAIN),
               eval=parser(EVAL) + "\nvenv = VecNormalize.load(p, venv)\n")
    result = check_script_cli_contract(h)
    assert result.status == "WARN"
    assert "d_ppo_eval.py: missing vecnorm_path" in result.detail


# --- a parser assembled elsewhere in the folder ---------------------------

def builder(dests, name: str = "build_arg_parser", extra: str = "") -> str:
    """A module exporting `name`, a builder declaring exactly `dests`."""
    body = "\n".join(f'    p.add_argument("--{d.replace("_", "-")}")' for d in dests)
    return (f"import argparse\n\n\ndef {name}():\n"
            "    p = argparse.ArgumentParser()\n" + body + "\n" + extra + "    return p\n")


def test_a_shared_builder_in_the_folder_offers_its_names(tmp_path):
    """The table is a contract over the names the eval surface offers. A domain
    that factors the common surface into one builder offers them from every
    script that calls it; a one-file reader would report flags in daily use as
    missing, clearable only by duplicating the builder into each script."""
    (tmp_path / "d_benchmark_common.py").write_text(builder(EVAL))
    h = domain(tmp_path, train=parser(TRAIN),
               eval="from d_benchmark_common import build_arg_parser, evaluate\n\n"
                    "def _build_arg_parser():\n    return build_arg_parser()\n")
    result = check_script_cli_contract(h)
    assert result.status == "PASS", result.detail


def test_the_reader_follows_the_calls_a_builder_makes(tmp_path):
    """Transitive — through a second local module, and only within the folder,
    so `argparse` and SB3 resolve to nothing and no training stack is ever
    imported."""
    (tmp_path / "d_cli_base.py").write_text(
        "def add_protocol(p):\n"
        '    p.add_argument("--n-seeds", type=int)\n'
        '    p.add_argument("--first-seed", type=int)\n')
    (tmp_path / "d_benchmark_common.py").write_text(
        "from d_cli_base import add_protocol\n"
        + builder([d for d in EVAL if d not in ("n_seeds", "first_seed")],
                  extra="    add_protocol(p)\n"))
    h = domain(tmp_path, train=parser(TRAIN),
               eval="from d_benchmark_common import build_arg_parser\n")
    assert check_script_cli_contract(h).status == "PASS"


def test_an_unrelated_module_s_own_cli_is_not_credited(tmp_path):
    """The reason the reach is per-symbol and not per-module: a domain's policy
    wrapper carries a `__main__` demo CLI of its own, and an eval that imports
    the policy class from it must not be credited with the demo's flags — that
    would hide a name the eval surface genuinely does not offer behind an
    unrelated file."""
    (tmp_path / "d_policy.py").write_text(
        "class DPolicy:\n    pass\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    import argparse\n"
        "    p = argparse.ArgumentParser()\n"
        '    p.add_argument("--n-seeds", type=int)\n')
    h = domain(tmp_path, train=parser(TRAIN),
               eval="from d_policy import DPolicy\n"
                    + parser([d for d in EVAL if d != "n_seeds"]))
    result = check_script_cli_contract(h)
    assert result.status == "WARN"
    assert "d_ppo_eval.py: missing n_seeds" in result.detail


def test_a_name_missing_from_the_whole_surface_is_still_reported(tmp_path):
    """Following imports widens where the check looks, not what it forgives."""
    (tmp_path / "d_benchmark_common.py").write_text(
        builder([d for d in EVAL if d != "n_seeds"]))
    h = domain(tmp_path, train=parser(TRAIN),
               eval="from d_benchmark_common import build_arg_parser\n")
    result = check_script_cli_contract(h)
    assert result.status == "WARN"
    assert "d_ppo_eval.py: missing n_seeds" in result.detail


def test_a_module_imported_whole_is_followed_through_its_attribute(tmp_path):
    (tmp_path / "d_benchmark_common.py").write_text(builder(EVAL))
    h = domain(tmp_path, train=parser(TRAIN),
               eval="import d_benchmark_common as common\n\n"
                    "def _build_arg_parser():\n"
                    "    return common.build_arg_parser()\n")
    assert check_script_cli_contract(h).status == "PASS"


def test_a_schedule_half_declared_in_a_shared_builder_is_not_a_half_pair(tmp_path):
    (tmp_path / "d_cli_base.py").write_text(builder(["lr_final"], name="add_schedule"))
    h = domain(tmp_path, train="from d_cli_base import add_schedule\n"
                               + parser(["learning_rate"]))
    assert check_schedule_pairs(h).status == "PASS"


# --- §8.2 schedule pairs --------------------------------------------------

def test_both_halves_of_a_schedule_pass(tmp_path):
    h = domain(tmp_path, train=parser(["learning_rate", "lr_final",
                                       "clip_init", "clip_final"]))
    assert check_schedule_pairs(h).status == "PASS"


def test_an_init_without_its_final_is_reported(tmp_path):
    """The derivation is guarded on the final's dest, so a half pair makes it a
    silent no-op — the run trains flat while its args log records a tuned init."""
    h = domain(tmp_path, train=parser(["learning_rate", "clip_init", "clip_final"]))
    result = check_schedule_pairs(h)
    assert result.status == "WARN"
    assert "--learning_rate without --lr_final" in result.detail


def test_a_final_without_its_init_is_reported(tmp_path):
    h = domain(tmp_path, train=parser(["clip_final"]))
    assert "--clip_final without --clip_init" in check_schedule_pairs(h).detail


def test_neither_half_is_out_of_scope(tmp_path):
    """Exposing no schedule at all is tier-2 completeness, which mdp_tuning
    owns — enumerating tuning knobs here would be the second source of truth
    the tiering exists to prevent."""
    h = domain(tmp_path, train=parser(["learning_rate", "lr_final"]))
    assert check_schedule_pairs(h).status == "PASS"


def test_no_train_script_skips_the_pair_check(tmp_path):
    assert check_schedule_pairs(domain(tmp_path)).status == "SKIP"

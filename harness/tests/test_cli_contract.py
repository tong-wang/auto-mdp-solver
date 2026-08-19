"""`scripts.cli_contract` — the tier-1 CLI surface (spec §8.2, §9.1).

The convention had already converged: the four IR-era domains agreed on every
contested name. It was written down nowhere, so each new domain re-derived it,
and the misses were invisible until something downstream tried to join on a
name that was not there. These tests cover the three shapes that produced the
misses — a name the spec used but nothing checked, a name owed only where it
applies, and a script that does not exist yet — plus the dest rule itself,
since a check that misreads `dest=` would report flags that are right there.

Synthetic scripts only: the check must not learn the shape of any shipped
domain.
"""

from __future__ import annotations

import pytest

from mdp_conformance.checks import _add_argument_dests, check_script_cli_contract
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


def test_the_eval_owes_vecnorm_path_when_it_loads_the_stats(tmp_path):
    h = domain(tmp_path, train=parser(TRAIN),
               eval=parser(EVAL) + "\nvenv = VecNormalize.load(p, venv)\n")
    result = check_script_cli_contract(h)
    assert result.status == "WARN"
    assert "d_ppo_eval.py: missing vecnorm_path" in result.detail

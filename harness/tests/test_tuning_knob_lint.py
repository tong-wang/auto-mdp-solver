"""The unmatched-knob lint, and why asking for a tier makes it fatal (§8.2 tier 2).

The failure this guards is silent and invisible in the result: a study accepts
`--knobs breadth`, finds one dest missing, warns once, and then searches seven
knobs while its own banner says eight. Nothing downstream of that log — not the
study, not the winner, not the eval TSV — records that the space was smaller
than the one requested. So the distinction that matters is between *requesting*
a tier and falling into the default one, which is what `knobs_explicit` carries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mdp_tuning.__main__ import parse_args, unmatched_knobs
from mdp_tuning.driver import ArgSpec, DomainScripts
from mdp_tuning.spaces import OPTIONAL_KNOBS, SPACES


def scripts(*dests: str) -> DomainScripts:
    return DomainScripts(
        prefix="d", directory=Path("."), algo="ppo",
        train_script=Path("d_ppo_train.py"), eval_script=Path("d_ppo_eval.py"),
        train_args={d: ArgSpec(flag=f"--{d}", multi=False, default=None) for d in dests},
        eval_args={},
    )


def test_a_fully_exposed_tier_has_nothing_unmatched():
    core = SPACES["ppo"].tiers["core"]
    assert unmatched_knobs(scripts(*core), "ppo", "core") == []


def test_one_missing_dest_is_named():
    core = list(SPACES["ppo"].tiers["core"])
    assert unmatched_knobs(scripts(*core[1:]), "ppo", "core") == [core[0]]


def test_an_out_of_tier_knob_is_not_unmatched():
    """`--knobs core` promising nothing about the breadth tier is the point of
    having tiers; only what the requested tier names can be betrayed."""
    core = SPACES["ppo"].tiers["core"]
    breadth_only = set(SPACES["ppo"].tiers["breadth"]) - set(core)
    assert breadth_only
    assert unmatched_knobs(scripts(*core), "ppo", "core") == []


def test_extractor_knobs_absent_from_an_mlp_domain_are_not_a_broken_contract():
    """OPTIONAL_KNOBS describe architectures only some domains have, so their
    absence is a fact about the domain rather than a missing flag."""
    optional_in_all = [k for k in SPACES["ppo"].knobs if k in OPTIONAL_KNOBS]
    assert optional_in_all
    exposed = [k for k in SPACES["ppo"].knobs if k not in OPTIONAL_KNOBS]
    assert unmatched_knobs(scripts(*exposed), "ppo", "all") == []


def test_the_vecnormalize_pair_is_optional_because_tier_1_owes_it_conditionally():
    """§8.2 owes `norm_obs`/`norm_reward` only where the script *builds* a
    VecNormalize, so a domain that normalizes its own inputs is missing them by
    design. The unconditional case has a better home: `scripts.cli_contract`
    fails a script that builds the wrapper and hides the flags, and checking it
    here too would give one violation two voices."""
    assert {"norm_obs", "norm_reward"} <= OPTIONAL_KNOBS
    exposed = [k for k in SPACES["ppo"].knobs if k not in OPTIONAL_KNOBS]
    assert unmatched_knobs(scripts(*exposed), "ppo", "all") == []


def test_a_missing_normalize_advantage_is_a_broken_contract():
    """It is PPO's own knob, not a wrapper's — every script in this family owes
    it, so its absence is the fatal that `--knobs breadth` exists to raise."""
    breadth = [k for k in SPACES["ppo"].tiers["breadth"]
               if k != "normalize_advantage"]
    assert unmatched_knobs(scripts(*breadth), "ppo", "breadth") == \
        ["normalize_advantage"]


# --- requesting a tier vs falling into one --------------------------------

def _args(monkeypatch, *argv: str):
    monkeypatch.setattr("sys.argv", ["mdp_tuning", "some_domain", *argv])
    return parse_args()


def test_the_default_tier_is_not_a_request(monkeypatch):
    args = _args(monkeypatch)
    assert args.knobs == "core" and args.knobs_explicit is False


def test_naming_the_default_tier_is_still_a_request(monkeypatch):
    """`--knobs core` and no flag resolve to the same space but not the same
    intent: only the first can be answered with a smaller space than it asked
    for."""
    args = _args(monkeypatch, "--knobs", "core")
    assert args.knobs == "core" and args.knobs_explicit is True


def test_strict_knobs_is_the_opt_in_for_the_default_tier(monkeypatch):
    args = _args(monkeypatch, "--strict-knobs")
    assert args.knobs_explicit is False and args.strict_knobs is True


def test_fix_is_the_deliberate_form_of_the_same_outcome(monkeypatch):
    """A locked knob and an unexposed one both end up at the script default;
    only one of them is a surprise, and `--fix` is what tells them apart."""
    args = _args(monkeypatch, "--knobs", "all", "--fix", "clip_init")
    core = list(SPACES["ppo"].tiers["all"])
    rot = unmatched_knobs(scripts(*[k for k in core if k != "clip_init"]),
                          "ppo", "all")
    assert rot == ["clip_init"]
    assert [k for k in rot if k not in set(args.fix)] == []


# --- what a study would find wrong, answered without launching one ---------

def _readiness(capsys, dests: dict, tier: str = "core") -> str:
    """--show-space's readiness report for a script with these dests/defaults."""
    import argparse
    from mdp_tuning.__main__ import report_readiness
    scripts = DomainScripts(
        prefix="d", directory=Path("."), algo="ppo",
        train_script=Path("d_ppo_train.py"), eval_script=Path("d_ppo_eval.py"),
        train_args={k: ArgSpec(flag=f"--{k}", multi=False, default=v)
                    for k, v in dests.items()},
        eval_args={})
    report_readiness(scripts, argparse.Namespace(
        algo="ppo", knobs=tier, fix=[], beta=1.0, episode_len=None,
        min_rollout_episodes=10))
    return capsys.readouterr().out


L1 = {"learning_rate": 3e-4, "net_arch": [64, 64], "n_steps": 2048,
      "ent_coef": 0.005, "gae_lambda": 0.95}


def test_a_ready_domain_reports_every_tier_open_and_trial_0_on_centre(capsys):
    out = _readiness(capsys, {**L1, "n_epochs": 10, "batch_size": 256,
                              "vf_coef": 0.5, "gamma": 1.0,
                              "clip_init": 0.2, "max_grad_norm": 0.5,
                              # the wrapper pair is optional; PPO's own
                              # boolean is not, so a ready script exposes it
                              "normalize_advantage": True})
    assert "core ok" in out and "breadth ok" in out and "all ok" in out
    assert "trial 0 = the L1 centre" in out


def test_a_tier_the_script_cannot_reach_is_named_before_launch(capsys):
    """Otherwise this is discoverable only as a fatal at study launch."""
    out = _readiness(capsys, L1)
    assert "core ok" in out
    assert "breadth BLOCKED" in out and "vf_coef" in out


def test_an_off_grid_default_is_reported_with_the_value_it_moved_to(capsys):
    """The mab case: a rollout floor derived to 2560 is not a power of two. It
    is rounded to the nearest one rather than dropped — but trial 0 is then L1
    rounded to the grid, not L1, and the study's Δ(L2−L1) rests on knowing so."""
    out = _readiness(capsys, {**L1, "n_steps": 2560})
    assert "L1 ROUNDED to the grid" in out
    assert "n_steps 2560->2048" in out


def test_a_refused_default_still_reads_as_not_the_centre(capsys):
    """§8.6 rules γ > β out entirely, so it is not rounded into range — the
    script is stating a spec violation, and the warm start says so."""
    out = _readiness(capsys, {**L1, "gamma": 1.5}, tier="breadth")
    assert "trial 0 is NOT the L1 centre" in out and "refuses" in out
    assert "gamma=1.5" in out


def test_a_default_outside_the_searched_tier_is_not_the_warm_start_s_problem(capsys):
    """The warm start only encodes knobs the tier actually searches, so a
    gamma the space would refuse is irrelevant to a core-tier study."""
    out = _readiness(capsys, {**L1, "gamma": 1.5}, tier="core")
    assert "trial 0 = the L1 centre" in out

"""The registry half of the launch gate — upstream #62 checks 4 and 5.

Both were accepted and deferred at v0.9.23 for one reason: they resolve against
the config registry's data half, `{domain}_configs.py`, which existed in no
worktree. It does now — the `adi_flex` campaign built one, with the log's
§CONFIG-REGISTRY asserted against it — so the tripwire has fired and the checks
land.

The subject is not "are the knobs derived" — that is check 1–3 next door — but
*is this run the run it says it is*. Every incident these close is one where
the numbers came out fine and meant something other than what they were read
as: a knob nobody chose, a banner describing a treatment the run did not have,
a base that had never been run, a contrast between two different budgets.

The design point under test is that the harness is handed the RESOLVED base and
the DECLARED deviations as data and never re-resolves a citation. That is what
makes it a check: the registry's own resolver computes the deviation list as
the diff it would be verified against, so re-deriving it here would agree with
itself. What this catches instead is drift between resolution and use — the
resolver runs at the top of `main()`, and anything that moves a knob afterwards
silently invalidates the record it wrote.
"""

from __future__ import annotations

import argparse

import pytest

from mdp_conformance.launch import (assert_launch, check_comparator,
                                    check_config, registry_ids)

# adi_flex's shape, minus the parts that are its own: an `h` axis owning the
# optimizer knobs, a `g` axis owning what the agent sees.
OWNED = ("learning_rate", "gamma", "batch_size", "n_epochs", "norm_obs",
         "observation_mode")
BASE = {"learning_rate": 3e-4, "gamma": 1.0, "batch_size": 256, "n_epochs": 10,
        "norm_obs": True, "observation_mode": "vec",
        "scenario_name": "homog_L0_T2"}


def args(**over):
    base = dict(scenario_name="homog_L0_T2", level="L2", **{
        k: v for k, v in BASE.items() if k != "scenario_name"})
    return argparse.Namespace(**{**base, **over})


# --- check 4: DECLARED == RESOLVED ------------------------------------------


def test_a_run_at_its_cited_base_is_clean():
    errors, warnings, notes = check_config(
        args(), base=BASE, declared=[], address="sc0/g0/a0/h0", owned=OWNED)
    assert not errors and not warnings
    assert "no deviations" in notes[0]


def test_a_declared_deviation_is_accepted_in_either_record_form():
    """A campaign records deviations the way it prints them, so the display
    string has to be readable — refusing it would make adoption mean rewriting
    the record rather than passing it in."""
    for declared in ({"gamma": 0.99}, ["gamma 1.0->0.99"], ("gamma",)):
        errors, _, notes = check_config(
            args(gamma=0.99), base=BASE, declared=declared,
            address="sc0/g0/a0/h1", owned=OWNED)
        assert not errors, (declared, errors)
        assert "gamma->0.99" in notes[0]


def test_an_undeclared_second_lever_is_refused():
    """The incident: a knob moved beside the one the arm is about, declared
    nowhere, and invisible in the run name because the name diffs the
    derivation rather than the base."""
    errors, _, _ = check_config(
        args(gamma=0.99, n_epochs=20), base=BASE, declared={"gamma": 0.99},
        address="sc0/g0/a0/h1", owned=OWNED)
    assert len(errors) == 1
    assert "n_epochs" in errors[0] and "undeclared second lever" in errors[0]


def test_a_deviation_the_run_does_not_have_is_refused():
    """Banner/run drift, the other direction: the record claims a treatment
    and the run is sitting at the base."""
    errors, _, _ = check_config(
        args(), base=BASE, declared={"gamma": 0.99}, address="sc0/g0/a0/h1",
        owned=OWNED)
    assert len(errors) == 1
    assert "does not have" in errors[0] and "gamma" in errors[0]


def test_a_declared_value_that_is_not_the_resolved_one_is_refused():
    errors, _, _ = check_config(
        args(gamma=0.95), base=BASE, declared={"gamma": 0.99},
        address="sc0/g0/a0/h1", owned=OWNED)
    assert len(errors) == 1 and "disagree about the same knob" in errors[0]


def test_a_knob_moved_after_resolution_is_caught():
    """The reason this is a check and not a restatement. The registry's own
    resolver computes the deviation list as the diff, so at the moment it runs
    the two agree by construction; a script that derives `batch_size` from the
    rollout AFTER applying the config invalidates the record it just wrote, and
    only a check at the point of use can see it."""
    recorded = {"gamma": 0.99}
    late = args(gamma=0.99)
    late.batch_size = 512                      # derived downstream of the config
    errors, _, _ = check_config(late, base=BASE, declared=recorded,
                                address="sc0/g0/a0/h1", owned=OWNED)
    assert len(errors) == 1 and "batch_size" in errors[0]


def test_a_base_that_fixes_less_than_its_axis_owns_is_refused():
    """R1a: a knob an axis owns and no base fixes is one nobody can be shown to
    have chosen — it declares nothing and diffs against nothing."""
    thin = {k: v for k, v in BASE.items() if k != "n_epochs"}
    errors, _, _ = check_config(args(), base=thin, declared=[],
                                address="sc0/g0/a0/h0", owned=OWNED)
    assert any("n_epochs" in e and "owns" in e for e in errors)


def test_an_uncited_run_is_an_error_not_a_skip():
    errors, _, _ = check_config(args(), base={}, declared=[], address="")
    assert len(errors) == 1 and "cites no base" in errors[0]


def test_a_run_landing_in_another_scenarios_tree_is_refused(tmp_path):
    out = tmp_path / "results" / "het_exp4" / "run_a"
    out.mkdir(parents=True)
    errors, _, _ = check_config(args(), base=BASE, declared=[],
                                address="sc0/g0/a0/h0", owned=OWNED, outdir=out)
    assert len(errors) == 1 and "results/het_exp4/" in errors[0]


def test_the_right_tree_passes(tmp_path):
    out = tmp_path / "results" / "homog_L0_T2" / "run_a"
    out.mkdir(parents=True)
    errors, _, _ = check_config(args(), base=BASE, declared=[],
                                address="sc0/g0/a0/h0", owned=OWNED, outdir=out)
    assert not errors


# --- check 4's other half: the data half and the log must agree -------------


LOG = """# adi_flex — escalation log

## CONFIG-REGISTRY   (living — ids append-only; §13)
### scenario
| id | key in SCENARIOS | note |
|---|---|---|
| <a id="sc0"></a>`sc0` | `homog_L0_T2` | the _L1_BASIS scenario |

### gym · `g`
| id | parent | delta | why |
|---|---|---|---|
| <a id="g0"></a>`g0` | — (L1 origin) | the derivation's gym output | — |

## LEDGER
<a id="E1"></a>
### #1 — a later section whose anchors are not registry ids
"""


def test_a_cited_id_with_no_definition_is_refused(tmp_path):
    log = tmp_path / "ESCALATION.md"
    log.write_text(LOG)
    assert registry_ids(log) == {"sc0", "g0"}
    errors, _, _ = check_config(args(), base=BASE, declared=[],
                                address="sc0/g0/a0/h9", owned=OWNED, log=log)
    assert len(errors) == 1
    assert "'a0'" in errors[0] and "'h9'" in errors[0] and "drifted" in errors[0]


def test_the_ledger_is_not_read_as_the_registry(tmp_path):
    """The section ends where the next `##` begins — a ledger anchor is not a
    config id, and reading one as a definition would make the check pass on a
    citation nobody defined."""
    log = tmp_path / "ESCALATION.md"
    log.write_text(LOG)
    assert "E1" not in (registry_ids(log) or set())


def test_a_campaign_with_no_registry_section_owes_nothing(tmp_path):
    log = tmp_path / "ESCALATION.md"
    log.write_text("## MAP\nnothing here\n")
    assert registry_ids(log) is None
    errors, _, notes = check_config(args(), base=BASE, declared=[],
                                    address="sc0/g0/a0/h0", owned=OWNED, log=log)
    assert not errors and any("no §CONFIG-REGISTRY" in n for n in notes)


def test_an_l0_citation_names_no_registry_id(tmp_path):
    """Guide §13.4: L0 is not in the registry at all, so `{sc}/L0` cites one id
    and a literal — refusing the literal would make the ladder's own spelling
    illegal."""
    log = tmp_path / "ESCALATION.md"
    log.write_text(LOG)
    errors, _, _ = check_config(args(), base=BASE, declared=[],
                                address="sc0/L0", owned=OWNED, log=log)
    assert not errors


def test_the_two_halves_of_the_registry_are_asserted_against_each_other(tmp_path):
    """#62 asks this of the same pass: the reader-facing table and the module
    that configures runs must not drift, in either direction."""
    log = tmp_path / "ESCALATION.md"
    log.write_text(LOG)
    errors, _, _ = check_config(args(), base=BASE, declared=[],
                                address="sc0/g0", owned=OWNED, log=log,
                                ids={"sc0", "g0", "h4"})
    assert len(errors) == 1
    assert "'h4'" in errors[0] and "no row for" in errors[0]

    errors, _, _ = check_config(args(), base=BASE, declared=[],
                                address="sc0/g0", owned=OWNED, log=log,
                                ids={"sc0"})
    assert len(errors) == 1
    assert "'g0'" in errors[0] and "configure nothing" in errors[0]

    errors, _, _ = check_config(args(), base=BASE, declared=[],
                                address="sc0/g0", owned=OWNED, log=log,
                                ids={"sc0", "g0"})
    assert not errors


# --- check 5: COMPARATOR NAMED AND MATCHED ----------------------------------


def run_dir(root, name, scenario="homog_L0_T2", steps=10_000_000, trained=None):
    d = root / "results" / scenario / name
    d.mkdir(parents=True)
    text = (f"scenario_name: {scenario}\ntotal_timesteps: {steps}\n"
            f"algo_class: MaskablePPO\n")
    if trained is not None:
        text += f"steps_trained: {trained}\n"
    (d / f"{name}_args.txt").write_text(text)
    return d


def test_a_matched_comparator_is_clean(tmp_path):
    run_dir(tmp_path, "base_a")
    errors, warnings, notes = check_comparator(
        args(total_timesteps=10_000_000), comparator="base_a",
        results_root=tmp_path / "results")
    assert not errors and not warnings and "same scenario" in notes[0]


def test_a_base_that_was_never_run_is_refused(tmp_path):
    (tmp_path / "results").mkdir()
    errors, _, _ = check_comparator(args(total_timesteps=10_000_000),
                                    comparator="base_that_never_ran",
                                    results_root=tmp_path / "results")
    assert len(errors) == 1 and "never run cannot be a base" in errors[0]


def test_an_arm_naming_no_comparator_is_refused(tmp_path):
    errors, _, _ = check_comparator(args(), comparator="",
                                    results_root=tmp_path)
    assert len(errors) == 1 and "names no comparator" in errors[0]
    # the dash a §8.4 log writes for "none" is the same statement
    assert check_comparator(args(), comparator="-", results_root=tmp_path)[0]


def test_a_comparator_from_another_scenario_is_refused(tmp_path):
    run_dir(tmp_path, "base_b", scenario="het_exp4")
    errors, _, _ = check_comparator(
        args(total_timesteps=10_000_000), comparator="base_b",
        results_root=tmp_path / "results")
    assert any("two different problems" in e for e in errors)


def test_a_budget_confound_is_refused_and_settling_downgrades_it(tmp_path):
    run_dir(tmp_path, "base_c", steps=2_000_000)
    a = args(total_timesteps=10_000_000)
    errors, _, _ = check_comparator(a, comparator="base_c",
                                    results_root=tmp_path / "results")
    assert len(errors) == 1 and "budget-confounded" in errors[0]
    errors, warnings, _ = check_comparator(
        a, comparator="base_c", results_root=tmp_path / "results",
        settling=True)
    assert not errors and len(warnings) == 1 and "settling base" in warnings[0]


def test_a_comparator_that_stopped_short_warns(tmp_path):
    run_dir(tmp_path, "base_d", trained=600_000)
    errors, warnings, _ = check_comparator(
        args(total_timesteps=10_000_000), comparator="base_d",
        results_root=tmp_path / "results")
    assert not errors and any("not a result" in w for w in warnings)


def test_a_comparator_given_as_a_path_needs_no_root(tmp_path):
    d = run_dir(tmp_path, "base_e")
    errors, _, _ = check_comparator(args(total_timesteps=10_000_000),
                                    comparator=d)
    assert not errors


# --- the entry point --------------------------------------------------------


DERIVED = {"gamma": 1.0, "n_steps": 512, "n_envs": 4}
BASIS = {"scenario_name": "homog_L0_T2", "episode_len": 48, "beta": 1.0}


def test_assert_launch_runs_every_group_it_has_inputs_for(tmp_path, capsys):
    run_dir(tmp_path, "base_a")
    a = args(total_timesteps=10_000_000, n_steps=512, n_envs=4,
             checkpoint_every_frac=0.05)
    assert_launch(a, derived=DERIVED, basis=BASIS, episode_len=48,
                  base=BASE, declared=[], address="sc0/g0/a0/h0", owned=OWNED,
                  comparator="base_a", results_root=tmp_path / "results")
    out = capsys.readouterr().out
    assert "L1 check" in out and "config check" in out and "comparator check" in out


def test_assert_launch_refuses_and_names_the_group(tmp_path, capsys):
    run_dir(tmp_path, "base_a", steps=1_000_000)
    a = args(total_timesteps=10_000_000, n_steps=512, n_envs=4,
             checkpoint_every_frac=0.05, n_epochs=20)
    with pytest.raises(SystemExit):
        assert_launch(a, derived=DERIVED, basis=BASIS, episode_len=48,
                      base=BASE, declared=[], address="sc0/g0/a0/h0",
                      owned=OWNED, comparator="base_a",
                      results_root=tmp_path / "results")
    out = capsys.readouterr().out
    assert "undeclared second lever" in out and "budget-confounded" in out


def test_an_l0_run_skips_the_config_group_and_keeps_the_comparator(tmp_path, capsys):
    """§8.6 defines L0 as faithful defaults, so no registry base is in force —
    but an L0 baseline is very often the run another arm is read against."""
    run_dir(tmp_path, "base_a")
    a = args(level="L0", total_timesteps=10_000_000)
    assert_launch(a, derived=DERIVED, basis=BASIS, base=BASE, declared=[],
                  address="sc0/L0", comparator="base_a",
                  results_root=tmp_path / "results")
    out = capsys.readouterr().out
    assert "config check: skipped" in out and "comparator check" in out


def test_the_l1_only_entry_point_is_unchanged(capsys):
    """Every adopting script calls this name; a domain with no registry is not
    a domain that owes one."""
    from mdp_conformance.launch import assert_l1_current
    assert_l1_current(args(n_steps=512, n_envs=4, checkpoint_every_frac=0.05),
                      derived=DERIVED, basis=BASIS, episode_len=48)
    out = capsys.readouterr().out
    assert "L1 check" in out and "config check" not in out

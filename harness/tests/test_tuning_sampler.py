"""The study's sampler: parallel workers must not collapse onto one draw.

The defect this locks in: workers of one study propose from the rows they can
see, and TPE without the constant liar sees COMPLETE rows only. A wave of
workers that draws against the same completed table then fits the same model
and proposes its mode — one configuration, trained once per worker, with the
per-worker seed moving only which candidates get scored. Observed on a
downstream campaign as twelve near-identical draws (lr within a factor 1.3,
same n_steps) out of a twelve-worker wave started two minutes apart.
"""

from __future__ import annotations

import optuna
import pytest

from mdp_tuning.__main__ import make_sampler

optuna.logging.set_verbosity(optuna.logging.ERROR)

SPACE = {
    "lr": optuna.distributions.FloatDistribution(1e-5, 3e-3, log=True),
    "log2_n_steps": optuna.distributions.IntDistribution(8, 12),
}


def _frozen_table(sampler) -> optuna.Study:
    """Twelve completed trials with one clear mode (lr ~3e-4, n_steps 2048)."""
    study = optuna.create_study(direction="maximize", sampler=sampler)
    rows = [
        (3.0e-4, 11, 100.0), (2.5e-4, 11, 97.0), (1.2e-3, 11, 90.0),
        (3.0e-4, 9, 88.0), (1.7e-4, 9, 80.0), (8.7e-4, 9, 70.0),
        (4.7e-4, 9, 68.0), (1.1e-5, 9, 60.0), (5.6e-5, 12, 55.0),
        (1.9e-5, 8, 50.0), (1.9e-5, 11, 40.0), (2.8e-3, 9, 5.0),
    ]
    for lr, k, value in rows:
        study.add_trial(optuna.trial.create_trial(
            params={"lr": lr, "log2_n_steps": k}, distributions=SPACE,
            value=value))
    return study


def _wave(study: optuna.Study, make, n: int = 8) -> list[tuple[float, int]]:
    """n workers drawing in turn, each earlier draw left RUNNING — the shape a
    staggered launch gives the shared storage."""
    draws = []
    for w in range(n):
        study.sampler = make(100 + w)
        trial = study.ask(SPACE)
        draws.append((trial.params["lr"], trial.params["log2_n_steps"]))
    return draws


def _distinct(draws) -> int:
    # lr binned to a factor 1.5, n_steps exact: "the same point" as an operator
    # reads a trial table, not float equality
    import math
    return len({(round(math.log(lr) / math.log(1.5)), k) for lr, k in draws})


def _draw_after_one_running(sampler):
    """One RUNNING row at the table's mode, then one ask: the only situation
    in which the liar changes a draw, so it is how the flag is observed
    without touching the sampler's private state."""
    study = _frozen_table(sampler)
    study.ask(SPACE)  # left RUNNING
    study.sampler = sampler
    return study.ask(SPACE).params


def test_the_liar_is_off_unless_asked():
    """The sampler is not persisted with the study, so the default decides
    whether a study's draws can be reproduced from its seed: the default
    draws exactly as optuna's own default sampler, the flag does not."""
    plain = _draw_after_one_running(optuna.samplers.TPESampler(seed=1))
    assert _draw_after_one_running(make_sampler(1)) == plain
    assert _draw_after_one_running(make_sampler(1, constant_liar=True)) != plain


def test_the_switch_is_on_the_cli_and_off_by_default():
    from mdp_tuning.__main__ import _build_arg_parser
    assert _build_arg_parser().parse_args(["d"]).constant_liar is False
    assert _build_arg_parser().parse_args(["d", "--constant-liar"]).constant_liar is True


def test_a_wave_without_the_liar_collapses_onto_the_mode():
    """The default, and the defect, reproduced on a synthetic table."""
    draws = _wave(_frozen_table(make_sampler(100)), make_sampler)
    assert _distinct(draws) <= 2, draws


def test_a_wave_with_the_liar_spreads():
    liar = lambda seed: make_sampler(seed, constant_liar=True)
    draws = _wave(_frozen_table(liar(100)), liar)
    assert _distinct(draws) >= 5, draws


def test_alone_the_liar_changes_nothing():
    """No RUNNING rows -> identical draws with and without the flag."""
    a = _frozen_table(make_sampler(7)).ask(SPACE).params
    b = _frozen_table(make_sampler(7, constant_liar=True)).ask(SPACE).params
    assert a == b

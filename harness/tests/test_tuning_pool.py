"""Study-wide trial budgets and worker pools: first come, first served.

The defect this locks in (upstream #102): ``--n-trials k`` is optuna's own
per-process count, so W workers on one study pre-assign k trials to each. A
worker that draws slow configurations still owes its k after the others have
exited — observed on a 2048 study as five of twelve workers idle on day 3
while the slow worker's last trial had not yet been asked for. And two studies
of one campaign idled each other's workers the same way: one study's workers
exited with nothing left while the other still had nine trials unasked.

The acceptance criterion is the first test: W workers launched with one
study-wide budget and no per-worker count, each taking the next trial whenever
it is free, until the study holds N. Early workers then run more than N/W
trials each, which no per-worker count summing to N can produce.

The integration tests drive real worker processes on a stub domain (a train
script that sleeps its ``cost`` and records what it received) over sqlite —
the lock, the record and the seed rule only exist across processes. The
in-process tests pin the loop's rules on an in-memory study.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import optuna
import pytest
from optuna.trial import TrialState

from mdp_tuning import pool
from mdp_tuning.__main__ import parse_args

optuna.logging.set_verbosity(optuna.logging.ERROR)

TRAIN = '''\
import argparse, json, os, sys, time
from pathlib import Path


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario_name", default="simple")
    p.add_argument("--total_timesteps", type=int, default=1000)
    p.add_argument("--outdir", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--cost", type=float, default=0.0)
    p.add_argument("--crash", type=int, default=0)
    return p


if __name__ == "__main__":
    a = _build_arg_parser().parse_args()
    device = os.environ.get("CUDA_VISIBLE_DEVICES")
    out = Path(a.outdir)
    (out / "received.json").write_text(json.dumps({**vars(a), "device": device}))
    if device == "broken":           # a broken device fails at once
        sys.exit("stub: no usable device")
    time.sleep(a.cost)
    if a.crash:
        sys.exit("stub: training crashed")
    run = out / a.scenario_name / "PPO_stub"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{a.scenario_name}_ppo_args.txt").write_text("stub\\n")
    (run / f"{a.scenario_name}_ppo.zip").write_text(json.dumps(vars(a)))
'''

EVAL = '''\
import argparse, json, math
from pathlib import Path


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", dest="model_path")
    p.add_argument("--outfile")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=10)
    p.add_argument("--scenario_name", default="simple")
    return p


if __name__ == "__main__":
    a = _build_arg_parser().parse_args()
    cfg = json.loads(Path(a.model_path).read_text())
    score = -abs(math.log10(cfg["learning_rate"]) + 3.5)
    Path(a.outfile).write_text(f"score_mean\\n{score}\\n")
'''


def _stub_domain(root: Path) -> Path:
    dom = root / "stub"
    dom.mkdir()
    for part in ("gym", "mdp", "scenarios"):
        (dom / f"stub_{part}.py").write_text("")
    (dom / "stub_ppo_train.py").write_text(TRAIN)
    (dom / "stub_ppo_eval.py").write_text(EVAL)
    return dom


class Fleet:
    """Worker processes of one stub domain on one sqlite storage."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.dom = _stub_domain(root)
        self.url = f"sqlite:///{root / 'studies.db'}"
        self.procs: list[tuple[subprocess.Popen, Path]] = []

    def launch(self, device: str, *argv: str) -> subprocess.Popen:
        log = self.root / f"worker_{len(self.procs)}_{device}.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": device,
               "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "mdp_tuning", str(self.dom),
             "--storage", self.url, "--total-timesteps", "10",
             "--eval-seeds", "1", *argv],
            stdout=log.open("w"), stderr=subprocess.STDOUT, env=env)
        self.procs.append((proc, log))
        return proc

    def run(self, device: str, *argv: str) -> tuple[int, str]:
        proc = self.launch(device, *argv)
        proc.wait(timeout=120)
        return proc.returncode, self.procs[-1][1].read_text()

    def wait_all(self) -> None:
        for proc, log in self.procs:
            assert proc.wait(timeout=180) == 0, log.read_text()[-3000:]

    def study(self, name: str) -> optuna.Study:
        return optuna.load_study(study_name=name, storage=self.url)

    def started(self, name: str, marker: str = "received.json") -> int:
        """Trials of a study that have reached ``marker`` on disk. Polled from
        the files rather than the database, so the test never opens the
        storage while workers may still be creating it."""
        return len(list((self.dom / "results" / "tuning" / name)
                        .glob(f"trial_*/{marker}")))

    def received(self, name: str, trial) -> dict:
        path = (self.dom / "results" / "tuning" / name
                / f"trial_{trial.number:04d}" / "received.json")
        return json.loads(path.read_text())


def _wait_until(cond, timeout: float = 90.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.1)


# --- the acceptance criterion ----------------------------------------------

def test_a_pool_of_workers_is_first_come_first_served(tmp_path):
    """Three workers, --max-trials 7, the third launched late: the study ends
    at exactly 7, and each early worker ran more than 7/3 of them."""
    fleet = Fleet(tmp_path)
    flags = ("--study-name", "s", "--max-trials", "7", "--constant-liar",
             "--train-arg", "cost=1.0")
    # launched together: the lock covers creation and the draw, so a budgeted
    # launch needs no stagger
    fleet.launch("w1", *flags, "--seed", "42")
    fleet.launch("w2", *flags, "--seed", "43")
    _wait_until(lambda: fleet.started("s") >= 6)
    fleet.launch("w3", *flags, "--seed", "44")
    fleet.wait_all()

    study = fleet.study("s")
    assert len(study.trials) == 7
    assert all(t.state == TrialState.COMPLETE for t in study.trials)
    ran = Counter(t.user_attrs["worker"]["cuda_visible_devices"]
                  for t in study.trials)
    assert ran["w1"] > 7 / 3 and ran["w2"] > 7 / 3, ran

    base = study.user_attrs[pool.RECORD_ATTR]["seed_base"]
    assert base in (42, 43)          # whichever early worker created it
    for t in study.trials:
        got = fleet.received("s", t)
        # trial n trains on base + n, whichever worker ran it
        assert got["seed"] == base + t.number == t.user_attrs["train_seed"]
        assert got["device"] == t.user_attrs["worker"]["cuda_visible_devices"]
    # trial 0 is the warm start, enqueued once although two workers raced to
    # create the study
    enqueued = [t.number for t in study.trials if "fixed_params" in t.system_attrs]
    assert enqueued == [0]


def test_a_worker_serves_every_study_of_its_pool(tmp_path):
    """Two studies with different fixed settings and different trial costs,
    three workers joining both: each study ends at its own N, with its own
    settings and seeds, and a worker crossed between them."""
    fleet = Fleet(tmp_path)
    studies = {"narrow": dict(n=5, width=64, cost=0.3, base=100),
               "wide": dict(n=4, width=128, cost=0.9, base=200)}
    for name, s in studies.items():
        code, out = fleet.run(
            "creator", "--study-name", name, "--max-trials", str(s["n"]),
            "--seed", str(s["base"]), "--constant-liar", "--create-only",
            "--train-arg", f"width={s['width']}",
            "--train-arg", f"cost={s['cost']}")
        assert code == 0, out
    for w, seed in (("g0", 7), ("g1", 8), ("g2", 9)):
        fleet.launch(w, "--study-name", "narrow", "--study-name", "wide",
                     "--seed", str(seed))
    fleet.wait_all()

    served = defaultdict(set)
    drawn = defaultdict(lambda: defaultdict(set))
    for name, s in studies.items():
        study = fleet.study(name)
        assert len(study.trials) == s["n"]
        assert all(t.state == TrialState.COMPLETE for t in study.trials)
        for t in study.trials:
            got = fleet.received(name, t)
            assert (got["width"], got["cost"]) == (s["width"], s["cost"])
            assert got["seed"] == s["base"] + t.number
            served[got["device"]].add(name)
            if "fixed_params" not in t.system_attrs:
                drawn[got["device"]][name].add(tuple(sorted(t.params.items())))
    assert any(len(names) == 2 for names in served.values()), dict(served)
    # one sampler seed per study: a worker does not replay its random-startup
    # draws in every study it serves
    for device, by_study in drawn.items():
        assert not (by_study["narrow"] & by_study["wide"]), device


# --- the stops ---------------------------------------------------------------

def test_a_script_that_always_crashes_closes_on_the_fail_cap(tmp_path):
    """FAIL does not count toward N, so without a cap it is redrawn forever."""
    fleet = Fleet(tmp_path)
    code, out = fleet.run("w1", "--study-name", "s", "--max-trials", "4",
                          "--max-fails", "2", "--train-arg", "crash=1")
    assert code == 0, out
    assert "FAIL cap" in out
    assert [t.state for t in fleet.study("s").trials] == [TrialState.FAIL] * 2


def test_a_broken_worker_stops_instead_of_eating_the_budget(tmp_path):
    """After the first completion a crash is a PRUNED row, which counts —
    and the worker that crashes in seconds is the one first come, first
    served hands the most trials."""
    fleet = Fleet(tmp_path)
    flags = ("--study-name", "s", "--max-trials", "8",
             "--train-arg", "cost=0.4")
    fleet.launch("good", *flags)
    _wait_until(lambda: fleet.started("s", "eval.tsv") >= 1)
    fleet.launch("broken", *flags, "--seed", "43")
    fleet.wait_all()

    study = fleet.study("s")
    assert len([t for t in study.trials if t.state != TrialState.FAIL]) == 8
    broken = [t for t in study.trials
              if t.user_attrs["worker"]["cuda_visible_devices"] == "broken"]
    assert len(broken) == 3 and all("failed" in t.user_attrs for t in broken)
    assert "crashed trials in a row" in fleet.procs[1][1].read_text()


# --- the refusals ------------------------------------------------------------

def test_a_study_launched_per_worker_is_not_adopted(tmp_path):
    """Its trials trained on per-worker seeds base + n cannot describe."""
    fleet = Fleet(tmp_path)
    old = optuna.create_study(study_name="s", storage=fleet.url)
    old.enqueue_trial({"learning_rate": 1e-3})
    code, out = fleet.run("w1", "--study-name", "s", "--max-trials", "4")
    assert code != 0 and "no study-wide budget record" in out


def test_a_pool_refuses_a_study_with_no_record(tmp_path):
    fleet = Fleet(tmp_path)
    code, out = fleet.run("w1", "--study-name", "a", "--study-name", "b")
    assert code != 0 and "has no record" in out


def test_a_recorded_study_refuses_contradicting_flags(tmp_path):
    fleet = Fleet(tmp_path)
    code, out = fleet.run("c", "--study-name", "s", "--max-trials", "4",
                          "--create-only")
    assert code == 0, out
    code, out = fleet.run("w", "--study-name", "s", "--total-timesteps", "99")
    assert code != 0 and "total_timesteps" in out
    # a recorded study counts study-wide whatever this launch says
    code, out = fleet.run("w", "--study-name", "s", "--n-trials", "3")
    assert code != 0 and "--n-trials" in out


def test_the_two_budgets_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["d", "--n-trials", "3", "--max-trials", "7"])


def test_only_given_flags_can_contradict_a_record():
    args = parse_args(["d", "--max-trials", "7", "-s", "big"])
    assert {"max_trials", "scenario_name"} <= args.explicit
    assert "n_trials" not in args.explicit and "seed" not in args.explicit
    record = pool.make_record(parse_args(["d", "--max-trials", "7"]), "d", "big")
    assert pool.record_mismatches(args, record) == []
    other = parse_args(["d", "--eval-seeds", "9"])
    assert [d.split(":")[0] for d in pool.record_mismatches(other, record)] \
        == ["eval_seeds"]


@pytest.mark.parametrize("url", ["postgresql://u@host/db", "sqlite:///:memory:",
                                 "mysql://u@host/db"])
def test_a_storage_with_no_shared_lock_is_refused(url):
    with pytest.raises(SystemExit, match="shares"):
        pool.storage_lock(url)


def test_a_filesystem_that_refuses_flock_is_named_not_a_traceback(
        tmp_path, monkeypatch):
    """Lustre without -o flock answers ENOSYS. A budgeted launch must say so
    and point at journal storage; the per-worker path catches this exception
    and opens its storage unlocked, as it always did."""
    import errno
    import fcntl

    def refuse(fd, op):
        raise OSError(errno.ENOSYS, "Function not implemented")

    monkeypatch.setattr(fcntl, "flock", refuse)
    lock = pool.storage_lock(f"sqlite:///{tmp_path / 's.db'}")
    with pytest.raises(pool.LockUnavailable, match="journal"):
        with lock:
            pass


# --- the loop's rules, in process --------------------------------------------

def _member(name="s", n=4, fails=5, run=None, draw=None, direction="maximize"):
    study = optuna.create_study(study_name=name, direction=direction)
    return pool.Member(name=name, study=study, max_trials=n, max_fails=fails,
                       draw=draw or (lambda t: {"x": t.suggest_float("x", 0, 1)}),
                       run=run or (lambda t, cfg: cfg["x"]))


def _add(member, *states):
    for state in states:
        member.study.add_trial(optuna.trial.create_trial(
            state=state, value=0.0 if state == TrialState.COMPLETE else None))


def _loop(*members, **kw):
    return pool.run_worker(list(members), contextlib.nullcontext(), **kw)


def test_the_largest_share_left_is_picked_ties_by_join_order():
    a, b = _member("a", n=5), _member("b", n=4)
    _add(a, TrialState.COMPLETE)                  # a: 4/5 left, b: 4/4
    assert pool.pick([a, b])[0] is b
    _add(b, TrialState.COMPLETE, TrialState.RUNNING)   # b: 2/4, a: 4/5
    assert pool.pick([a, b])[0] is a
    c, d = _member("c", n=4), _member("d", n=2)
    _add(c, TrialState.COMPLETE, TrialState.COMPLETE)
    _add(d, TrialState.WAITING)                    # 1/2 each: join order
    assert pool.pick([c, d])[0] is c and pool.pick([d, c])[0] is d


def test_fail_is_redrawn_and_the_study_ends_at_exactly_n():
    calls = iter([RuntimeError("diverged")] + [0.5] * 10)

    def run(trial, cfg):
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v

    m = _member(n=3, run=run)
    stop = _loop(m)
    assert "holds 3/3" in stop
    assert [t.state for t in m.study.trials] == \
        [TrialState.FAIL] + [TrialState.COMPLETE] * 3


def test_an_uncaught_exception_fails_the_trial_and_ends_the_worker():
    def run(trial, cfg):
        raise KeyError("not a trial crash")

    m = _member(run=run)
    with pytest.raises(KeyError):
        _loop(m)
    assert [t.state for t in m.study.trials] == [TrialState.FAIL]


def test_the_fail_cap_closes_a_study_and_the_streak_stops_a_worker():
    def crash(trial, cfg):
        raise RuntimeError("broken")

    capped = _member(n=10, fails=2, run=crash)
    assert "FAIL cap" in _loop(capped, max_consecutive_crashes=99)
    assert len(capped.study.trials) == 2

    streak = _member(n=10, fails=99, run=crash)
    assert "3 crashed trials in a row" in _loop(streak)
    assert len(streak.study.trials) == 3


def test_a_pruned_crash_counts_toward_the_streak_and_the_budget():
    def pruned(trial, cfg):
        trial.set_user_attr("failed", "RuntimeError: diverged")
        raise optuna.TrialPruned("training crashed (RuntimeError)")

    m = _member(n=10, run=pruned)
    assert "in a row" in _loop(m)
    assert [t.state for t in m.study.trials] == [TrialState.PRUNED] * 3
    assert pool.counts(m.study) == (3, 0)        # counted toward N, not FAIL


def test_the_timeout_is_per_worker_and_stops_only_new_trials():
    ticks = iter([0.0, 0.0, 5.0, 11.0])
    m = _member(n=10)
    stop = _loop(m, timeout=10.0, clock=lambda: next(ticks))
    assert "--timeout" in stop
    assert len(m.study.trials) == 2


def test_the_record_round_trips_through_the_study():
    args = parse_args(["d", "--max-trials", "7", "--fix", "gamma",
                       "--train-arg", "width=64", "--seed", "41"])
    record = pool.make_record(args, "d", "simple")
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(study_name="s", storage=storage)
    study.set_user_attr(pool.RECORD_ATTR, record)
    got, n = pool.read_record(storage, "s")
    assert n == 0 and got == record and got["seed_base"] == 41
    joined = pool.apply_record(parse_args(["d", "--seed", "99"]), got)
    assert (joined.fix, joined.train_arg, joined.max_trials, joined.seed) == \
        (["gamma"], ["width=64"], 7, 99)
    assert pool.read_record(storage, "absent") == (None, None)
    assert isinstance(joined, argparse.Namespace)

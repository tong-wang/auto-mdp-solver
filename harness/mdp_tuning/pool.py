"""Study-wide trial budgets, and worker pools across studies (upstream #102).

``--n-trials k`` is optuna's own ``optimize(n_trials=k)``: it counts the trials
*this process* runs. W workers on one study then pre-assign k trials to each,
and a worker that draws slow configurations still owes its k after the others
have exited — observed as five of twelve workers idle on day 3 of a 2048 study
while the slow worker's last trial waited to be asked for.

``--max-trials N`` makes the budget the study's. A worker, whenever it is free,
takes one lock, counts the study's trials that did not FAIL, and — if the study
holds fewer than N — asks for the next trial and draws its configuration,
all before releasing the lock. First come, first served:

- **The count and the ask sit under one lock.** optuna's ``MaxTrialsCallback``
  checks only at trial *end*, so two workers that finish at once both start a
  trial, and a worker launched after the study is full still runs one.
- **The draw sits under it too.** TPE samples in ``trial.suggest_*``, not in
  ``study.ask()``; released after the ask, the lock would let the next worker
  draw against a RUNNING row with no parameters, and the constant liar would
  have nothing to repel it from. Holding it through the draw is also why a
  budgeted launch needs no stagger between workers.
- **FAIL does not count**, so an unscored crash is redrawn — and a study takes
  no new trial once it holds ``max_fails`` FAIL rows, since a script that
  crashes every trial would otherwise be redrawn forever. FAIL is not confined
  to the first trials: a training crash is PRUNED once one trial has
  completed, but an eval or scoring crash is FAIL at any point.
- **A worker stops after K crashed trials in a row** (FAIL or PRUNED). First
  come, first served gives the most trials to the fastest worker, and a worker
  on a broken device is the fastest one: after the first completion its every
  crash is a PRUNED row that counts toward N.

A **pool** is one worker joining several studies: when free, it asks the one
with the largest share of its budget left (ties by join order), which keeps
paired studies at the same stage. Each trial runs with its own study's
settings, so those live *on the study* (``RECORD_ATTR``), written by the launch
that created it; a pool worker reads them and refuses a study that has none.
The record also fixes the study's seed base: trial n trains on ``base + n``,
whichever worker or device runs it.

The lock spans every joined study, so a pool's studies share one storage:

- ``sqlite:///…`` — an ``flock`` beside the database file: workers on ONE host
  (sqlite itself is unsafe across hosts on a shared filesystem).
- ``journal:///…`` — an O_EXCL lock file beside the journal, the protocol
  optuna's own journal lock uses on NFS: workers on several hosts sharing one
  filesystem.
- anything else is refused: without a lock every worker shares, the study
  overshoots N by up to W-1.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
import zlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import optuna
from optuna.trial import TrialState

# what a crashing trial raises and the worker survives — the same tuple the
# --n-trials path hands optimize(catch=...), so a crash means one thing on both
CATCH = (RuntimeError, FileNotFoundError, ValueError)

RECORD_ATTR = "mdp_tuning_record"
RECORD_VERSION = 1

# the per-study settings — what the creating launch records and every later
# worker takes from the study. What stays per worker: --seed (it seeds the
# worker's samplers), --timeout, --max-consecutive-crashes, --storage and the
# study names. The device is the worker's too: it comes from the environment
# (CUDA_VISIBLE_DEVICES) the train subprocess inherits, never from a flag.
RECORD_FIELDS = (
    "scenario_name", "algo", "knobs", "knobs_explicit", "strict_knobs", "fix",
    "beta", "episode_len", "min_rollout_episodes", "warm_start",
    "train_arg", "eval_arg", "total_timesteps", "eval_seeds", "metric",
    "minimize", "score_checkpoint", "constant_liar", "max_trials", "max_fails",
)

# a lock file older than this while a waiter waits is taken as abandoned by a
# dead holder; the locked section is a count, an ask and a draw — well under it
JOURNAL_LOCK_GRACE_S = 120


# --- the lock ---------------------------------------------------------------

class LockUnavailable(SystemExit):
    """The filesystem refuses the lock (``flock`` on a Lustre mount without
    ``-o flock`` is ENOSYS, on NFS without a lock daemon ENOLCK). A SystemExit,
    so a budgeted launch stops with the message rather than a traceback; the
    per-worker path catches it and opens its storage unlocked, as before."""

    def __init__(self, path: str, err: OSError, hint: str) -> None:
        super().__init__(f"FATAL: cannot take the pool lock {path}: {err}. "
                         f"{hint}")


class _FileLock:
    """``flock`` on a file beside the sqlite database; the kernel drops it
    when the holder dies, so a killed worker never wedges the pool."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fd: int | None = None

    def __enter__(self) -> "_FileLock":
        import fcntl
        try:
            self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except OSError as err:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            raise LockUnavailable(
                self.path, err,
                "This filesystem refuses flock; put the study on "
                "journal:///<file> instead (an O_EXCL lock file, which needs "
                "no flock)") from err
        return self

    def __exit__(self, *exc) -> None:
        import fcntl
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None


class _OpenLock:
    """optuna's NFS-safe O_EXCL lock file, beside the journal."""

    def __init__(self, path: str) -> None:
        from optuna.storages.journal import JournalFileOpenLock
        self.path = path
        self._lock = JournalFileOpenLock(path, grace_period=JOURNAL_LOCK_GRACE_S)

    def __enter__(self) -> "_OpenLock":
        try:
            self._lock.acquire()
        except OSError as err:
            raise LockUnavailable(
                self.path, err,
                "Every worker needs write access to the journal's "
                "directory") from err
        return self

    def __exit__(self, *exc) -> None:
        self._lock.release()


def lock_for(spec: str):
    """The lock beside a ``--storage`` URL, or ``None`` where none can be
    taken."""
    if spec.startswith("journal://"):
        # JournalFileOpenLock appends ".lock"; ".pool" keeps it apart from the
        # lock the journal storage itself takes on every write
        return _OpenLock(spec[len("journal://"):] + ".pool")
    if spec.startswith("sqlite:///"):
        path = spec[len("sqlite:///"):].split("?", 1)[0]
        if path and path != ":memory:":
            try:
                import fcntl  # noqa: F401 — POSIX only
            except ImportError:
                return None
            return _FileLock(path + ".pool.lock")
    return None


def storage_lock(spec: str):
    """The lock every worker of a budgeted study shares, from its ``--storage``
    URL — or a refusal, for a storage no shared lock can be taken beside."""
    lock = lock_for(spec)
    if lock is not None:
        return lock
    raise SystemExit(
        f"FATAL: a study-wide budget needs a lock that every worker of the "
        f"study shares, and mdp_tuning can take none beside {spec!r}. Use "
        f"sqlite:///<file> (workers on one host) or journal:///<file> "
        f"(workers on several hosts sharing a filesystem). Without a shared "
        f"lock two workers that finish at once both start a trial, and the "
        f"study overshoots its budget by up to one trial per worker")


# --- the study record -------------------------------------------------------

def _plain(value):
    """JSON's view of a value, so a recorded list compares equal to the tuple
    or list a launch parsed."""
    return list(value) if isinstance(value, tuple) else value


def make_record(args: argparse.Namespace, domain: str, scenario: str) -> dict:
    """The study's settings, as its creating launch gives them."""
    record = {f: _plain(getattr(args, f)) for f in RECORD_FIELDS}
    record["scenario_name"] = scenario
    record.update(version=RECORD_VERSION, domain=domain, seed_base=args.seed)
    return record


def read_record(storage, name: str) -> tuple[dict | None, int | None]:
    """``(record, trial count)`` of a stored study; ``(None, None)`` when the
    storage holds no study of that name."""
    try:
        study_id = storage.get_study_id_from_name(name)
    except KeyError:
        return None, None
    return (storage.get_study_user_attrs(study_id).get(RECORD_ATTR),
            storage.get_n_trials(study_id))


def record_mismatches(args: argparse.Namespace, record: dict) -> list[str]:
    """Per-study settings this launch was GIVEN that the record contradicts.

    Only a value on the command line can contradict a record: a default the
    launch fell into says nothing, and is what lets a pool worker carry no
    per-study flag at all."""
    diffs = []
    for f in RECORD_FIELDS:
        if f in args.explicit and _plain(getattr(args, f)) != record.get(f):
            diffs.append(f"{f}: this launch {getattr(args, f)!r}, "
                         f"recorded {record.get(f)!r}")
    return diffs


def apply_record(args: argparse.Namespace, record: dict) -> argparse.Namespace:
    """This worker's flags with the study's settings in place of its own."""
    joined = argparse.Namespace(**vars(args))
    for f in RECORD_FIELDS:
        setattr(joined, f, record[f])
    return joined


# --- counting, picking, running ---------------------------------------------

@dataclass
class Member:
    """One study a worker serves: its budget, and the trial as two steps."""

    name: str
    study: optuna.Study
    max_trials: int
    max_fails: int
    draw: Callable[[optuna.Trial], dict]
    run: Callable[[optuna.Trial, dict], float]


def counts(study: optuna.Study) -> tuple[int, int]:
    """``(trials that count toward the budget, FAIL rows)``.

    Everything but FAIL counts — WAITING (the enqueued warm start) and RUNNING
    included, since each of those becomes a scored row without another ask."""
    trials = study.get_trials(deepcopy=False)
    fails = sum(t.state == TrialState.FAIL for t in trials)
    return len(trials) - fails, fails


def pick(members: list[Member]) -> tuple[Member, int] | None:
    """The study to ask next, with its current count; ``None`` when none has
    budget left. Largest share of its budget left wins, ties by join order.

    Recomputed on every pick rather than cached: a RUNNING trial that ends as
    FAIL hands its study one trial back."""
    best: tuple[Fraction, Member, int] | None = None
    for m in members:
        counted, fails = counts(m.study)
        if counted >= m.max_trials or fails >= m.max_fails:
            continue
        share = Fraction(m.max_trials - counted, m.max_trials)
        if best is None or share > best[0]:
            best = (share, m, counted)
    return None if best is None else (best[1], best[2])


def describe(m: Member) -> str:
    counted, fails = counts(m.study)
    if fails >= m.max_fails:
        return (f"{m.name} CLOSED on its FAIL cap ({fails} unscored trials; "
                f"see their train.log / eval.log)")
    return f"{m.name} holds {counted}/{m.max_trials}"


def sampler_seed(seed: int, name: str) -> int:
    """A pool worker's sampler seed for one of its studies. One seed for all
    of them would replay the same random-startup draws in every study the
    worker serves (observed: one worker's first draw in each of two paired
    studies was the same configuration). crc32, not hash(), so the seed is the
    same in every process."""
    return zlib.crc32(f"{seed}/{name}".encode())


def worker_identity() -> dict:
    """Where a trial ran: host, process, and the GPU set the train subprocess
    inherits from this worker's environment."""
    return {"host": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def _fail(member: Member, trial: optuna.Trial, err: BaseException,
          step: str) -> None:
    member.study.tell(trial, state=TrialState.FAIL)
    if isinstance(err, CATCH):
        print(f"[{member.name}] trial {trial.number} FAILED while {step} "
              f"({err.__class__.__name__}: {err})", flush=True)


def _run(member: Member, trial: optuna.Trial, cfg: dict) -> bool:
    """Run and tell one drawn trial; ``True`` when it scored without a crash.
    A training crash after the first completion ends the trial as PRUNED and a
    caught exception as FAIL, and the worker continues; anything else ends the
    trial as FAIL and then the worker."""
    try:
        value = member.run(trial, cfg)
    except optuna.TrialPruned:
        member.study.tell(trial, state=TrialState.PRUNED)
        return False
    except BaseException as err:
        _fail(member, trial, err, "running")
        if not isinstance(err, CATCH):
            raise
        return False
    # a NaN value is told FAIL by optuna itself
    return member.study.tell(trial, value).state == TrialState.COMPLETE


def run_worker(members: list[Member], lock, *, timeout: float | None = None,
               max_consecutive_crashes: int = 3,
               clock: Callable[[], float] = time.monotonic) -> str:
    """Take trials first come, first served until no joined study has budget
    left; return why this worker stopped."""
    start = clock()
    streak = 0
    while True:
        if timeout is not None and clock() - start >= timeout:
            return (f"--timeout {timeout:g}s since this worker's launch; no "
                    f"new trial started")
        with lock:
            chosen = pick(members)
            if chosen is None:
                return ("no joined study has budget left — "
                        + "; ".join(describe(m) for m in members))
            member, counted = chosen
            trial = member.study.ask()
            print(f"[{member.name}] trial {trial.number}: "
                  f"{counted + 1}/{member.max_trials} of the study's budget",
                  flush=True)
            try:
                cfg = member.draw(trial)
            except BaseException as err:
                _fail(member, trial, err, "drawing")
                if not isinstance(err, CATCH):
                    raise
                cfg = None
        clean = cfg is not None and _run(member, trial, cfg)
        streak = 0 if clean else streak + 1
        if streak >= max_consecutive_crashes:
            return (f"{streak} crashed trials in a row on this worker — a "
                    f"broken device or environment is the usual cause; the "
                    f"study's other workers carry on (--max-consecutive-crashes)")

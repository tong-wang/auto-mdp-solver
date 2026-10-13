"""Prune cnv's results tree by the retention rule (human, 2026-10-11) — DRY RUN by default.

The rule, per run directory (a dir holding `*_ppo_args.txt`):
  ALWAYS KEPT  eval TSVs (summary + per-seed), args, select.log, TensorBoard events, the canonical model + its
               VecNormalize, every checkpoint that was evaluated (has a `ppo_eval_<cell>__<stem>.tsv` — the funnel's
               top-3 and any analysis read), and its VecNormalize.
  THINNED      other checkpoints of an ordinary run or a tuning study's top-3 trial: kept only at the 25 / 50 / 75 %
               grid of the run's budget (the nearest saved checkpoint), the rest removed — but only once the run has
               been screened (a `select.log`, or any evaluated checkpoint); an unscreened run keeps every checkpoint, the
               funnel's input.
  REMOVED      every checkpoint of a tuning trial OUTSIDE its study's top-3 (by the study db's values), and every
               checkpoint of a run marked ABORTED.txt (cut off / orphaned); the monitor CSVs of both (human,
               2026-10-11: no analysis reads a monitor CSV; TensorBoard carries the training curve).
  DEDUPED      a tuning trial's run-dir `train.log` when the trial dir's `train.log` ends with it byte for byte
               (`mdp_tuning` captures the same stream behind a 5-line `# cmd` / `# cwd` / L1-check header).
  GZIPPED      every other monitor CSV and every surviving `train.log` (run dir and trial dir): kept, compressed
               in place to `<name>.gz` (`zcat` / `gzip.open` read them).
  GATE         a run is touched only when its experiment's ledger entry is CLOSED — the run name ends in `_E<id>`;
               `<a id="E<id>">` (or a `### #Ea–#Eb` range entry, or a `(tags Ea–Eb)` header) must have no open
               `verdict: ▶` slot, annotated or not. Untagged runs and runs of open entries are never touched.

    python cnv_prune_results.py                    # dry run: summary + plan file scratch/prune_plan_<ts>.tsv
    python cnv_prune_results.py --apply PLAN.tsv   # carry out exactly the plan's rows, re-checking each gate
"""
from __future__ import annotations

import argparse
import gzip
import os
import random
import re
import shutil
import zlib
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
TAG = re.compile(r"_(E\d+[a-z]?)$")
STEP = re.compile(r"_(\d+)_steps")
MONITOR = re.compile(r"^monitor_\d+\.monitor\.csv$")


def ledger_status() -> tuple[dict[str, str], set[str]]:
    """tag -> entry anchor, and the set of anchors whose verdict slot is still open."""
    s = (D / "ESCALATION.md").read_text()
    anchors = [(m.start(), m.group(1)) for m in re.finditer(r'<a id="(E\d+[a-z]?)"></a>', s)]
    open_, tag2entry = set(), {}
    for i, (pos, eid) in enumerate(anchors):
        span = s[pos:(anchors[i + 1][0] if i + 1 < len(anchors) else len(s))]
        tag2entry[eid] = eid
        # ANY open slot, annotated or not ("verdict: ▶  (centres are in …)" is still open); the one exception is a
        # slot the entry itself marks "(superseded above)" under a verdict written higher up (#E94)
        if re.search(r"^verdict: ▶(?!\s*\(superseded above\))", span, re.M):
            open_.add(eid)
        m = (re.search(r"^### #E(\d+)[–-]#E(\d+)", span, re.M)          # a range entry covers its members,
             or re.search(r"^### #E\d+ .*\(tags E(\d+)[–-]E(\d+)\)", span, re.M))   # as does a "(tags Ea–Eb)" header
        if m:
            for n in range(int(m.group(1)), int(m.group(2)) + 1):
                tag2entry.setdefault(f"E{n}", eid)
    return tag2entry, open_


def study_top3(study_dir: Path, cache: dict) -> set[int] | None:
    if study_dir in cache:
        return cache[study_dir]
    db = study_dir.parent / f"{study_dir.name}.db"
    top = None
    if db.exists():
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = c.execute("select t.number from trials t join trial_values v on t.trial_id = v.trial_id "
                         "where t.state = 'COMPLETE' order by v.value desc limit 3").fetchall()
        top = {r[0] for r in rows}
    cache[study_dir] = top
    return top


def budget(run: Path) -> int | None:
    for f in run.glob("*_ppo_args.txt"):
        m = re.search(r"^total_timesteps: (\d+)", f.read_text(), re.M)
        if m:
            return int(m.group(1))
    return None


def plan_run(run: Path, cls: str) -> list[Path]:
    ck = run / "checkpoints"
    if not ck.is_dir():
        return []
    zips = sorted(ck.glob("ppo_cnv_*_steps.zip"), key=lambda p: int(STEP.search(p.name).group(1)))
    keep = {p for p in zips if any(run.glob(f"ppo_eval_*__{p.stem}.tsv"))}   # evaluated checkpoints: always kept
    if cls != "remove":
        T = budget(run)
        if T and zips:
            for frac in (0.25, 0.5, 0.75):
                keep.add(min(zips, key=lambda p: abs(int(STEP.search(p.name).group(1)) - frac * T)))
    gone = [p for p in zips if p not in keep]
    kept_steps = {STEP.search(p.name).group(1) for p in keep}
    gone += [p for p in ck.glob("ppo_cnv_vecnormalize_*_steps.pkl") if STEP.search(p.name).group(1) not in kept_steps]
    return gone


def trial_log_holds(run_log: Path, trial_log: Path) -> bool:
    """The trial dir's train.log ends with the run dir's train.log byte for byte."""
    a = run_log.read_bytes()
    return trial_log.stat().st_size >= len(a) and trial_log.read_bytes().endswith(a)


def touchable_bytes(run: Path) -> int:
    ck = sum(f.stat().st_size for f in (run / "checkpoints").glob("*") if f.is_file()) if (run / "checkpoints").is_dir() else 0
    return ck + sum(f.stat().st_size for f in run.iterdir() if MONITOR.match(f.name) or f.name == "train.log")


def make_plan(root: Path):
    tag2entry, open_ = ledger_status()
    cache, rows, blocked, skipped = {}, [], defaultdict(int), defaultdict(int)
    trial_logs, unfunnelled = set(), []
    for argf in root.rglob("*_ppo_args.txt"):
        run = argf.parent
        m = TAG.search(run.name)
        if not m or m.group(1) not in tag2entry:
            skipped["untagged / no entry"] += touchable_bytes(run); continue
        entry = tag2entry[m.group(1)]
        if entry in open_:
            blocked[entry] += touchable_bytes(run); continue
        tdir = next((p for p in run.parents if p.name.startswith("trial_")), None) if "tuning" in run.parts else None
        if (run / "ABORTED.txt").exists():
            cls = "remove"; why = "aborted"
        elif "tuning" in run.parts:
            top = study_top3(tdir.parent, cache) if tdir else None
            if top is not None and int(tdir.name.split("_")[1]) not in top:
                cls = "remove"; why = "tuning trial outside top-3"
            else:
                cls = "thin"; why = "tuning top-3 trial"
        else:
            cls = "thin"; why = "ordinary run"
        screened = (run / "select.log").exists() or any(run.glob("ppo_eval_*__ppo_cnv_*_steps.tsv"))
        if cls == "thin" and not screened:         # no screen yet: the selection funnel still needs every checkpoint
            unfunnelled.append(run)
        else:
            for f in plan_run(run, cls):
                rows.append((f, f.stat().st_size, entry, why, "rm"))
        for f in sorted(run.glob("monitor_*.monitor.csv")):
            rows.append((f, f.stat().st_size, entry, why if cls == "remove" else "monitor CSV kept",
                         "rm" if cls == "remove" else "gzip"))
        log = run / "train.log"
        if log.exists():
            tlog = tdir / "train.log" if tdir else None
            if tlog is not None and tlog.exists() and trial_log_holds(log, tlog):
                rows.append((log, log.stat().st_size, entry, "run log duplicated in the trial log", "rm"))
            else:
                rows.append((log, log.stat().st_size, entry, "train.log", "gzip"))
        if tdir is not None and (tdir / "train.log").exists() and tdir not in trial_logs:
            trial_logs.add(tdir)
            rows.append((tdir / "train.log", (tdir / "train.log").stat().st_size, entry, "trial train.log", "gzip"))
    return rows, blocked, skipped, unfunnelled


def gzip_in_place(f: Path) -> int:
    """f -> f.gz, verified by reading it back; returns the bytes saved. Leaves f untouched on any mismatch."""
    gz, tmp = f.with_name(f.name + ".gz"), f.with_name(f.name + ".gz.tmp")
    if gz.exists():
        return 0
    with open(f, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    with open(f, "rb") as a, gzip.open(tmp, "rb") as b:     # gzip checks its CRC at EOF
        while True:
            x, y = a.read(1 << 20), b.read(1 << 20)
            if x != y:
                tmp.unlink(); raise RuntimeError(f"gzip verify failed: {f}")
            if not x:
                break
    shutil.copystat(f, tmp); os.replace(tmp, gz)
    saved = f.stat().st_size - gz.stat().st_size
    f.unlink()
    return saved


def gzip_ratio(paths: list[Path], k: int = 40) -> float:
    """Compressed / raw on a random sample of whole files (the dry run's estimate)."""
    sample = random.Random(0).sample(paths, min(k, len(paths)))
    raw = sum(p.stat().st_size for p in sample)
    return sum(len(zlib.compress(p.read_bytes(), 6)) for p in sample) / raw if raw else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(D / "results"))
    ap.add_argument("--apply", default=None, help="a plan file from a dry run: delete exactly its files")
    a = ap.parse_args()
    if a.apply:
        tag2entry, open_ = ledger_status()
        rows = []
        for line in Path(a.apply).read_text().splitlines():
            if line and not line.startswith("#"):
                c = line.split("\t")
                rows.append((Path(c[0]), c[2], c[3], c[4] if len(c) > 4 else "rm"))
        n_rm = n_gz = freed = skipped = 0
        for p, entry, why, act in sorted(rows, key=lambda r: r[3] != "rm"):   # deletions first: a dedup check
            if entry in open_ or not p.exists():                               # must read the trial log unzipped
                skipped += 1; continue                                        # the gate is re-checked here
            if act == "rm":
                if "checkpoints" in p.parts or (MONITOR.match(p.name) and why != "monitor CSV kept"):
                    pass
                elif p.name == "train.log" and why.startswith("run log duplicated"):
                    tdir = next((q for q in p.parents if q.name.startswith("trial_")), None)
                    if not (tdir and (tdir / "train.log").exists() and trial_log_holds(p, tdir / "train.log")):
                        skipped += 1; continue
                else:
                    skipped += 1; continue
                freed += p.stat().st_size; p.unlink(); n_rm += 1
            elif act == "gzip" and (MONITOR.match(p.name) or p.name == "train.log"):
                freed += gzip_in_place(p); n_gz += 1
            else:
                skipped += 1
        print(f"deleted {n_rm} files, gzipped {n_gz}, skipped {skipped}; freed {freed / 1e9:.2f} GB"); return
    rows, blocked, skipped, unfunnelled = make_plan(Path(a.root))
    by = defaultdict(lambda: [0, 0])
    kind = lambda f: "checkpoint" if "checkpoints" in f.parts else ("monitor CSV" if MONITOR.match(f.name) else "train.log")
    for f, size, entry, why, act in rows:
        top = f.relative_to(D / "results").parts[0]
        by[(act, kind(f), top, why)][0] += 1; by[(act, kind(f), top, why)][1] += size
    ratio = {k: gzip_ratio([f for f, _, _, _, act in rows if act == "gzip" and kind(f) == k]) for k in ("monitor CSV", "train.log")}
    total_saved = 0.0
    for act in ("rm", "gzip"):
        print("would DELETE:" if act == "rm" else "would GZIP (kept, compressed in place):")
        for (a_, k, top, why), (n, size) in sorted(by.items(), key=lambda kv: -kv[1][1]):
            if a_ != act:
                continue
            saved = size if act == "rm" else size * (1 - ratio[k])
            total_saved += saved
            est = "" if act == "rm" else f"  -> saves ~{saved / 1e9:.2f} GB (sampled ratio {ratio[k]:.3f})"
            print(f"  {k:12s} {top:12s} {why:36s} {n:6d} files  {size / 1e9:7.2f} GB{est}")
    print(f"  TOTAL {len(rows)} files, {sum(r[1] for r in rows) / 1e9:.2f} GB touched, ~{total_saved / 1e9:.2f} GB freed")
    print("left untouched (entry still open — checkpoint + monitor + train.log bytes):")
    for e, size in sorted(blocked.items(), key=lambda kv: -kv[1]):
        print(f"  #{e:8s} {size / 1e9:7.2f} GB")
    for k, size in skipped.items():
        print(f"  {k}: {size / 1e9:.2f} GB")
    if unfunnelled:
        ck = lambda r: sum(f.stat().st_size for f in (r / "checkpoints").glob("*") if f.is_file()) if (r / "checkpoints").is_dir() else 0
        by_e = defaultdict(lambda: [0, 0])
        for r in unfunnelled:
            e = TAG.search(r.name).group(1); by_e[e][0] += 1; by_e[e][1] += ck(r)
        print("checkpoints kept whole — closed entry, but no screen yet, so the funnel is still owed:")
        for e, (n, size) in sorted(by_e.items(), key=lambda kv: -kv[1][1]):
            print(f"  {e:8s} {n:4d} runs  {size / 1e9:6.2f} GB")
    out = D / f"scratch/prune_plan_{time.strftime('%Y%m%d_%H%M')}.tsv"
    out.write_text("# path\tbytes\tentry\twhy\taction\n" + "".join(f"{f}\t{s}\t{e}\t{w}\t{a_}\n" for f, s, e, w, a_ in rows))
    print(f"plan: {out}")


if __name__ == "__main__":
    main()

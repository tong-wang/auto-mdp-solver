"""Index the mab results tree by escalation-campaign label.

The problem this solves: `ESCALATION.md`'s ledger entries name the actions
(`A6b`) and the log entries (`#E14`), while the results tree names runs by
their *config* (spec §8.4: `PPO_obsbayes_L1_policyindex_normobsFalse_seed1_…`).
Those are different namespaces, the relation between them is many-to-many (one
dir can serve several actions — `…_115417/` carries both #E7's crown eval and
#E13's frame-averaged one — and labels are often assigned after the run), and
when the linkage is hand-written into ledger prose it drifts: four of the
original nine citations were already dead by 2026-07-29.

So the linkage is stored **once**, at the source, and every view is derived:

  source   `run_status.json["campaign"]` — written by `mab_ppo_train.py --tag`,
           extended after the fact by `--annotate`. Co-located with the
           artifact, so a run dir always answers "what was this for?".
  view 1   the RUNS table spliced into `ESCALATION.md` between its
           `<!-- RUNS:BEGIN -->` / `<!-- RUNS:END -->` markers (`--write`).
  view 2   `by_action/{action}` symlinks to the TensorBoard leaf
           (`--symlinks`), so
               tensorboard --logdir results/by_action
           shows runs labelled `A6b`, `e35-T20000-scl`, … instead of 60-char
           config names. Both views span EVERY scenario dir that holds runs
           (`-s all`, the default); pass `-s {name}` for one, which puts the
           farm back under `results/{name}/by_action`.

Nothing is renamed: the run name stays a pure function of the config, and
`--tag` is excluded from it (see `_SKIP_KEYS` in `mab_ppo_train.py`).

Usage
    python mab_runs_index.py                       # print the table (all scenarios)
    python mab_runs_index.py --write --symlinks    # refresh both views
    python mab_runs_index.py -s gauss_K10_T1000    # one scenario only
    python mab_runs_index.py --annotate results/gauss_K10_T1000/PPO_… \
                            --add A6a,#E13
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ESCALATION = Path(__file__).with_name("ESCALATION.md")
BEGIN = "<!-- RUNS:BEGIN -->"
END = "<!-- RUNS:END -->"
#: dirs that are artifact collections rather than single training runs
NON_RUN_DIRS = {"benchmark", "by_action"}


# --------------------------------------------------------------------------
# source of truth
# --------------------------------------------------------------------------

def read_meta(run_dir: Path) -> dict:
    """Merge a run's status and campaign labels.

    `run_status.json` is written by the trainer; `campaign.json` is the
    fallback carrier for dirs that no trainer produced (`a5_probe/`, a
    benchmark bundle) so those can be labelled too.
    """
    meta: dict = {}
    status = run_dir / "run_status.json"
    if status.is_file():
        meta.update(json.loads(status.read_text()))
    sidecar = run_dir / "campaign.json"
    if sidecar.is_file():
        campaign = json.loads(sidecar.read_text())
        base = meta.setdefault("campaign", {"actions": [], "ledger": []})
        for field in ("actions", "ledger"):
            merged = list(dict.fromkeys(
                base.get(field, []) + campaign.get(field, [])))
            base[field] = merged
        if "note" in campaign:
            meta["note"] = campaign["note"]
    return meta


def annotate(run_dir: Path, add: str, note: str | None,
             set_level: str | None = None) -> dict:
    """Append campaign labels to an existing run, in place and idempotently.

    ``set_level`` corrects the recorded spec-§8.6 level. Needed for two cases
    the trainer cannot derive: runs written before the level was derived at
    all (they recorded ``--level``, so an L2(arch) escalation stored "l1"),
    and L2(hp) runs, which are indistinguishable from L1 by their args alone.
    Corrected here at the source — the §RUNS table is generated, never
    hand-edited, and run dirs are never renamed.
    """
    from mab_ppo_train import parse_tag

    new = parse_tag(add)
    target = run_dir / "run_status.json"
    if target.is_file():
        payload = json.loads(target.read_text())
        campaign = payload.setdefault("campaign", {"actions": [], "ledger": []})
    else:                                   # no trainer output — use a sidecar
        target = run_dir / "campaign.json"
        payload = json.loads(target.read_text()) if target.is_file() else {}
        campaign = payload
        campaign.setdefault("actions", [])
        campaign.setdefault("ledger", [])
    for field in ("actions", "ledger"):
        campaign[field] = list(dict.fromkeys(campaign[field] + new[field]))
    if note is not None:
        payload["note"] = note
    if set_level is not None:
        if payload.get("level") and "config_level" not in payload:
            # keep what the trainer actually selected, so the correction is
            # auditable rather than silently overwriting the original record
            payload["config_level"] = payload["level"]
        payload["level"] = set_level
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# --------------------------------------------------------------------------
# derived facts
# --------------------------------------------------------------------------

def read_evals(run_dir: Path, scenario: str) -> list[tuple[str, float]]:
    """Reported `reward_mean`s, one per spec-§9 eval TSV in the run dir.

    The variant label is the filename with the `ppo_eval_` prefix and the
    `_{scenario}` suffix stripped: `stoch`, `stoch_frameavg16`, …
    """
    out: list[tuple[str, float]] = []
    # no `_` before {scenario}: the default-mode eval writes
    # `ppo_eval_{scenario}.tsv` with no variant tag, which the stricter
    # pattern could never match — so an argmax/default eval sat in the run
    # dir unindexed while its `--stochastic` sibling showed up
    for tsv in sorted(run_dir.glob(f"ppo_eval_*{scenario}.tsv")):
        variant = tsv.stem[len("ppo_eval_"):-(len(scenario) + 1)] or "eval"
        with tsv.open() as fh:
            rows = list(csv.DictReader(
                (ln for ln in fh if not ln.startswith("#")), delimiter="\t"))
        if rows and rows[0].get("reward_mean"):
            out.append((variant, float(rows[0]["reward_mean"])))
    return out


def discover_scenarios(results: Path) -> list[str]:
    """Every scenario dir that holds at least one run.

    The index was scoped to a single scenario for as long as the domain had
    one. #E32/#E35 added 33 instances and #E36 a grid, and the table silently
    went on describing `gauss_K10_T1000` only — 44 of 79 runs — while the
    other 35 were annotated and invisible. A generated view that quietly stops
    covering the campaign is worse than no view, so the scenario set is now
    discovered rather than named.
    """
    return sorted(d.name for d in results.iterdir()
                  if d.is_dir() and any(d.glob("PPO_*")))


#: the campaign's main branch — listed first, everything else alphabetical
PRIMARY_SCENARIO = "gauss_K10_T1000"


def collect_many(results: Path, scenarios: list[str]) -> list[dict]:
    """Rows for every scenario, **grouped by scenario, ranked within it**.

    Deliberately not sorted globally on `best`: reward_mean is not comparable
    across horizons (#E28), so one ranking over all rows would put every
    T=20000 run (reward ~30,000) above the campaign's actual crown (1486.14)
    and read as a leaderboard. Grouping makes the incomparability structural
    rather than a footnote nobody reads.
    """
    rows = [r for s in scenarios for r in collect(results, s)]
    rows.sort(key=lambda r: (r["scenario"] != PRIMARY_SCENARIO, r["scenario"],
                             r["void"], r["best"] is None, -(r["best"] or 0)))
    return rows


def collect(results: Path, scenario: str) -> list[dict]:
    scen_dir = results / scenario
    rows: list[dict] = []
    for run_dir in sorted(p for p in scen_dir.iterdir() if p.is_dir()):
        if run_dir.name in NON_RUN_DIRS and not (run_dir / "campaign.json").is_file():
            continue
        meta = read_meta(run_dir)
        campaign = meta.get("campaign", {})
        evals = read_evals(run_dir, scenario)
        rows.append({
            "dir": run_dir,
            "scenario": scenario,
            "name": run_dir.name,
            "void": run_dir.name.startswith("VOID_"),
            "actions": campaign.get("actions", []),
            "ledger": campaign.get("ledger", []),
            # "l2(arch)" -> "L2(arch)": the level uppercases, the layer tag
            # does not — spec §8.6 writes them L2(hp) / L2(gym) / L2(arch)
            "level": re.sub(r"^l(\d)", lambda m: f"L{m.group(1)}",
                            (meta.get("level") or "").strip().lower()),
            "outcome": meta.get("outcome", ""),
            "timesteps": meta.get("timesteps"),
            "note": meta.get("note", ""),
            "evals": evals,
            "best": max((v for _, v in evals), default=None),
        })
    # crown-first for readability; unevaluated then voided runs sink
    rows.sort(key=lambda r: (r["void"], r["best"] is None, -(r["best"] or 0)))
    return rows


# --------------------------------------------------------------------------
# view 1 — the markdown table
# --------------------------------------------------------------------------

def _fmt_evals(row: dict) -> str:
    if not row["evals"]:
        return "—"
    return "<br>".join(
        f"{v:.2f} `{k}`" if len(row["evals"]) > 1 else f"{v:.2f}"
        for k, v in sorted(row["evals"], key=lambda kv: -kv[1]))


def render(rows: list[dict], scenario: str) -> str:
    steps = {None: "—"}
    scens = sorted({r["scenario"] for r in rows})
    span = len(scens) > 1
    scope = (f"{len(scens)} scenarios (`" + "`, `".join(scens) + "`)"
             if span else f"Scenario `{scenario}`")
    lines = [
        f"Generated by `mab_runs_index.py` from "
        f"`run_status.json['campaign']` — do not hand-edit between the "
        f"markers. {scope}. **reward_mean is NOT comparable across "
        f"scenarios** (different horizons, different oracles — #E28); the "
        f"sort is for readability only.",
        "",
        "| run dir |" + (" scenario |" if span else "")
        + " actions | ledger | level | steps | reward_mean |",
        "|---|" + ("---|" if span else "") + "---|---|---|---|---|",
    ]
    for row in rows:
        name = f"~~`{row['name']}`~~" if row["void"] else f"`{row['name']}`"
        ts = row["timesteps"]
        lines.append("| {} |".format(name)
                     + (f" `{row['scenario']}` |" if span else "")
                     + " {} | {} | {} | {} | {} |".format(
            ", ".join(f"**{a}**" for a in row["actions"]) or "—",
            ", ".join(row["ledger"]) or "—",
            row["level"] or "—",
            steps.get(ts, f"{ts:,}" if isinstance(ts, int) else "—"),
            _fmt_evals(row)))
    if any(r["note"] for r in rows):
        lines += [""] + [f"- `{r['name'].split('_seed')[0]}` — {r['note']}"
                         for r in rows if r["note"]]
    return "\n".join(lines)


def splice(table: str) -> bool:
    text = ESCALATION.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(
            f"{ESCALATION.name} has no {BEGIN} / {END} markers — add the RUNS "
            f"section first")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{BEGIN}\n{table}\n{END}{tail}"
    if new == text:
        return False
    ESCALATION.write_text(new)
    return True


# --------------------------------------------------------------------------
# view 2 — the TensorBoard symlink farm
# --------------------------------------------------------------------------

#: spec §8.4 run names end `_YYYYMMDD_HHMMSS`
_STAMP_RE = re.compile(r"_(\d{8})_(\d{6})$")


def _stamp(name: str) -> tuple[str, str]:
    """Sort key for "which run is newer".

    Probe bundles (`a5_probe`, `anneal_probe`) carry no stamp and sort oldest,
    which never matters: each is the only run under its label.
    """
    m = _STAMP_RE.search(name)
    return (m.group(1), m.group(2)) if m else ("", "")


def _suffix(name: str) -> str:
    """Disambiguator for a superseded run's label: `{action}__{suffix}`.

    The launch second alone is not a key. A seed sweep starts its runs in the
    same second (A8-a's three seeds all stamp `…_230542`), so two superseded
    siblings would claim one label and the second `symlink_to` would raise.
    The seed is exactly what differs there, so it joins the stamp.
    """
    stamp = _stamp(name)[1] or name
    m = re.search(r"_seed(\d+)_", name)
    return f"{stamp}_s{m.group(1)}" if m else stamp


def build_symlinks(rows: list[dict], results: Path, scenario: str) -> list[str]:
    """`by_action/{action}` -> the run's TensorBoard leaf.

    Rebuilt from scratch each time so a removed label leaves no stale link.
    `results/` is gitignored, so these cost nothing and stay local.

    When an action has been run more than once, the bare `{action}` label is
    the **newest** run, so `tensorboard --logdir by_action` opens on the live
    one; superseded runs stay reachable under `{action}__{HHMMSS}` instead of
    competing with it for the label. Nothing is renamed or removed — this is
    a view, and which run a label resolves to is derived, not stored.
    """
    # one farm for the whole campaign when the rows span scenarios, so a
    # single `tensorboard --logdir results/by_action` shows every round; the
    # per-scenario farm is kept for the single-scenario call.
    scens = {r["scenario"] for r in rows}
    farm = (results / "by_action") if len(scens) > 1 \
        else (results / scenario / "by_action")
    farm.mkdir(parents=True, exist_ok=True)
    for old in farm.iterdir():
        if old.is_symlink():
            old.unlink()
    current = {
        # name breaks stamp ties: a seed sweep launches its runs in the same
        # second, so the stamp alone leaves the label's owner undefined
        action: max((r for r in rows if action in r["actions"]),
                    key=lambda r: (_stamp(r["name"]), r["name"]))["name"]
        for action in {a for row in rows for a in row["actions"]}
    }
    made: list[str] = []
    for row in rows:
        # the trainer sets tb_log_name = run dir name, so the event files sit
        # in `{run}/{run}_1`; dirs with no TB leaf (a probe bundle) link to
        # the dir itself, so the target must not append a leaf name
        leaf = next(iter(sorted(row["dir"].glob(f"{row['name']}_*"))), None)
        target = Path("..") / row["dir"].name if len(scens) == 1 \
            else Path("..") / row["scenario"] / row["dir"].name
        if leaf is not None:
            target = target / leaf.name
        for action in row["actions"]:
            label = action if current[action] == row["name"] else \
                f"{action}__{_suffix(row['name'])}"
            link = farm / label
            link.symlink_to(target)
            made.append(label)
    return made


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-s", "--scenario_name", default="all",
                   help="a scenario name, or 'all' (default) to span every "
                        "scenario dir that holds runs")
    p.add_argument("--results", default="results")
    p.add_argument("--write", action="store_true",
                   help="splice the table into ESCALATION.md")
    p.add_argument("--symlinks", action="store_true",
                   help="rebuild by_action/ (results/by_action when spanning "
                        "scenarios, results/{scenario}/by_action for one)")
    p.add_argument("--annotate", metavar="RUN_DIR",
                   help="append campaign labels to an existing run")
    p.add_argument("--add", default="",
                   help="labels for --annotate, e.g. 'A6a,#E13'")
    p.add_argument("--note", help="one-line note for --annotate")
    p.add_argument("--set-level", metavar="LEVEL",
                   help="correct the recorded spec-§8.6 level of --annotate's "
                        "run, e.g. 'l2(arch)', 'l2(gym)', 'l2(hp)'. The "
                        "originally selected --level is preserved as "
                        "config_level")
    args = p.parse_args()

    results = Path(args.results)
    if args.annotate:
        payload = annotate(Path(args.annotate), args.add, args.note,
                           args.set_level)
        print(f"{args.annotate}: {json.dumps(payload.get('campaign', payload))}")
        return

    if args.scenario_name == "all":
        scenarios = discover_scenarios(results)
        rows = collect_many(results, scenarios)
    else:
        scenarios = [args.scenario_name]
        rows = collect(results, args.scenario_name)
    table = render(rows, args.scenario_name)
    if args.write:
        print("ESCALATION.md RUNS table: "
              + ("updated" if splice(table) else "already current"))
    else:
        print(table)
    if args.symlinks:
        made = build_symlinks(rows, results, args.scenario_name)
        print(f"by_action/: {len(made)} link(s) — {', '.join(sorted(made))}")


if __name__ == "__main__":
    main()

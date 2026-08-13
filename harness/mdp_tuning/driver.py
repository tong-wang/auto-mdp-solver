"""Domain discovery and trial execution for the MDP tuning harness.

A domain is tunable when it follows the spec's script conventions:
``{prefix}_{algo}_train.py`` and ``{prefix}_{algo}_eval.py``, each exposing
``_build_arg_parser()`` (spec §8.2 mandates the split precisely so other
scripts can reuse the parser). The driver imports both modules, introspects
their parsers, and adapts to what the domain actually exposes:

- only knobs present in the train script's CLI are tuned;
- fixed ``key=value`` overrides are validated against the real CLI;
- the eval episode-count flag is found by dest name (``n_seeds`` per spec
  §9.1, with fallbacks for legacy scripts);
- the metric is read from the eval TSV by column name.

Nothing in the domain is modified or imported beyond its two scripts.
"""

from __future__ import annotations

import csv
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mdp_conformance.loader import discover_prefix


REQUIRED_TRAIN_DESTS = ("scenario_name", "total_timesteps", "outdir")
EVAL_SEEDS_DESTS = ("n_seeds", "episodes", "n_episodes")


class ModelResolutionError(Exception):
    """The trial's artifact is ambiguous — a study-level misconfiguration.

    Deliberately outside the per-trial catch tuples: this fails identically on
    every trial, so it aborts the study instead of burning the whole budget on
    trials that would each be penalized as if training had diverged.
    """


@dataclass(frozen=True)
class ArgSpec:
    """One CLI argument of a domain script."""

    flag: str            # long option string, e.g. "--learning_rate"
    multi: bool          # nargs list-valued (e.g. --net_arch 64 64)
    default: object
    takes_value: bool = True     # False for store_true/store_false/BooleanOptionalAction
    flag_const: object = None    # what a bare `flag` stores when takes_value is False
    negative_flag: str | None = None  # BooleanOptionalAction's --no-* form


@dataclass(frozen=True)
class DomainScripts:
    """Normalized view of one domain's train/eval script pair."""

    prefix: str
    directory: Path
    algo: str
    train_script: Path
    eval_script: Path
    train_args: dict[str, ArgSpec]
    eval_args: dict[str, ArgSpec]


def _parser_args(parser) -> dict[str, ArgSpec]:
    out: dict[str, ArgSpec] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        flag = next((s for s in action.option_strings if s.startswith("--")),
                    action.option_strings[0])
        multi = action.nargs in ("+", "*") or (
            isinstance(action.nargs, int) and action.nargs > 1)
        # nargs == 0 marks the flag-shaped actions (store_true/store_false,
        # BooleanOptionalAction, count): they must be emitted bare, never as
        # "--flag value", which argparse rejects.
        takes_value = action.nargs != 0
        negative = None
        flag_const = None
        if not takes_value:
            negative = next((s for s in action.option_strings
                             if s.startswith("--no-") and s != flag), None)
            # BooleanOptionalAction carries both forms and no const; the bare
            # flag means True there. store_true/store_false state it as const.
            flag_const = True if negative else getattr(action, "const", None)
        out[action.dest] = ArgSpec(
            flag=flag, multi=multi, default=action.default,
            takes_value=takes_value, flag_const=flag_const,
            negative_flag=negative)
    return out


def load_domain(domain_dir: Path, algo: str = "ppo") -> DomainScripts:
    directory = Path(domain_dir).resolve()
    prefix = discover_prefix(directory)
    train_script = directory / f"{prefix}_{algo}_train.py"
    eval_script = directory / f"{prefix}_{algo}_eval.py"
    for script in (train_script, eval_script):
        if not script.exists():
            raise FileNotFoundError(
                f"{script.name} not found in {directory} — tuning needs both the "
                f"train and eval scripts (spec §8–§9)")

    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    train_mod = importlib.import_module(f"{prefix}_{algo}_train")
    eval_mod = importlib.import_module(f"{prefix}_{algo}_eval")
    train_args = _parser_args(train_mod._build_arg_parser())
    eval_args = _parser_args(eval_mod._build_arg_parser())

    missing = [d for d in REQUIRED_TRAIN_DESTS if d not in train_args]
    if missing:
        raise ValueError(
            f"{train_script.name} lacks required CLI dests {missing} (spec §8.2/§8.4)")
    if "model_path" not in eval_args or "outfile" not in eval_args:
        raise ValueError(
            f"{eval_script.name} must expose --model-path and --outfile (spec §9.1)")

    return DomainScripts(
        prefix=prefix, directory=directory, algo=algo,
        train_script=train_script, eval_script=eval_script,
        train_args=train_args, eval_args=eval_args,
    )


_BOOL_WORDS = {"1": True, "true": True, "yes": True, "on": True,
               "0": False, "false": False, "no": False, "off": False}


def parse_overrides(pairs: list[str], args_map: dict[str, ArgSpec],
                    label: str) -> dict[str, object]:
    """Parse repeatable KEY=VALUE overrides, validated against the script CLI.

    A flag-shaped dest (``store_true`` and friends) yields a real ``bool`` —
    passing its value through as the string ``"True"`` would make ``build_cmd``
    emit ``--flag True``, which argparse rejects, and the flag would silently
    keep the script's default.
    """
    out: dict[str, object] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--{label}-arg expects KEY=VALUE, got {pair!r}")
        if key not in args_map:
            raise ValueError(
                f"{label} script has no dest {key!r}; available: "
                f"{sorted(args_map)}")
        spec = args_map[key]
        if not spec.takes_value:
            parsed = _BOOL_WORDS.get(value.strip().lower())
            if parsed is None:
                raise ValueError(
                    f"{label} dest {key!r} is a boolean flag ({spec.flag}); "
                    f"pass {key}=true or {key}=false, got {value!r}")
            out[key] = parsed
        else:
            out[key] = value.split(",") if spec.multi else value
    return out


def build_cmd(script: Path, args_map: dict[str, ArgSpec],
              assignments: dict[str, object]) -> list[str]:
    cmd = [sys.executable, str(script)]
    for dest, value in assignments.items():
        spec = args_map[dest]
        if not spec.takes_value:
            want = bool(value)
            if spec.negative_flag is not None:
                cmd.append(spec.flag if want else spec.negative_flag)
            elif not isinstance(spec.flag_const, bool):
                raise ValueError(
                    f"{dest} ({spec.flag}) takes no value and stores "
                    f"{spec.flag_const!r} — mdp_tuning can only set boolean flags")
            elif want == spec.flag_const:
                cmd.append(spec.flag)
            elif want != bool(spec.default):
                raise ValueError(
                    f"cannot set {dest}={want} through {spec.flag}: the bare flag "
                    f"stores {spec.flag_const} and the script's default is "
                    f"{spec.default!r}, so the other value is unreachable — "
                    f"declare it with argparse.BooleanOptionalAction")
            # else: the script's default already is `want` — emit nothing
        elif spec.multi and isinstance(value, (list, tuple)):
            cmd += [spec.flag, *[str(v) for v in value]]
        else:
            cmd += [spec.flag, str(value)]
    return cmd


def run_logged(cmd: list[str], log_path: Path, cwd: Path) -> None:
    """Run a subprocess from the domain dir, tee-ing output to a log file.

    Pins BLAS/torch to one thread unless the caller set it explicitly: on the
    small models these domains use, thread contention on tiny tensor ops
    dominates the runtime (measured ~50x wall-clock slowdown on a small
    CNN-policy domain).
    """
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log,
                              stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"see {log_path}")


def resolve_model(trial_dir: Path, scenario: str | None = None,
                  algo: str = "ppo", mode: str = "canonical") -> Path:
    """The artifact a trial is scored on.

    ``canonical`` (default) takes the run directory's own saved model — the
    spec-§8.4 ``{run_dir}/{scenario}_{algo}.zip``. Selecting by mtime instead
    would let a periodic checkpoint under ``{run_dir}/checkpoints/`` outrank
    it, so trials in one study could be ranked on artifacts of different kinds
    — a silent reordering of the study. The final model sits at the run-dir top
    level and checkpoints sit below it, so depth, not name, is the reliable
    discriminator: shipped domains disagree on the filename
    (``ppo_inv_single.zip`` vs ``{scenario}_ppo.zip``), and the name is only a
    tiebreak here.

    ``final`` restores the newest-by-mtime rule, for a domain whose training
    genuinely ends on a checkpoint. Ambiguity raises rather than guesses.
    """
    hits = list(trial_dir.rglob("*.zip"))
    if not hits:
        raise FileNotFoundError(f"no saved model (*.zip) under {trial_dir}")
    if mode == "final":
        return max(hits, key=lambda p: p.stat().st_mtime)
    if mode != "canonical":
        raise ValueError(f"unknown score-checkpoint mode {mode!r}")

    # A run directory is the one carrying the spec-§8.4 args log: the final
    # model sits there, periodic checkpoints sit in a subdirectory below it.
    # Relative depth alone would not do — if training died before the final
    # save, the shallowest zip *is* a checkpoint, and scoring it silently is
    # the very thing this mode exists to prevent.
    run_dirs = {p.parent for p in trial_dir.rglob("*_args.txt")}
    if run_dirs:
        top = sorted(p for p in hits if p.parent in run_dirs)
    else:
        # domain writes no args log: fall back to the shallowest zips
        depth = min(len(p.relative_to(trial_dir).parts) for p in hits)
        top = sorted(p for p in hits
                     if len(p.relative_to(trial_dir).parts) == depth)
    if not top:
        raise FileNotFoundError(
            f"no model at a run-dir top level under {trial_dir}: {len(hits)} "
            f"zip(s) found, all inside subdirectories — training likely died "
            f"before the final save. Pass --score-checkpoint final to rank on "
            f"the newest file instead")
    if len(top) > 1 and scenario:
        named = [p for p in top
                 if p.name.lower() == f"{scenario}_{algo}.zip".lower()]
        if len(named) == 1:
            return named[0]
    if len(top) > 1:
        raise ModelResolutionError(
            f"ambiguous model under {trial_dir}: {[p.name for p in top]} all sit "
            f"at the run-dir top level. Name the final model "
            f"{scenario or '{scenario}'}_{algo}.zip (spec §8.4), or pass "
            f"--score-checkpoint final to rank on the newest file instead")
    return top[0]


def tsv_column_means(tsv: Path) -> tuple[list[str], dict[str, float]]:
    """Mean of every numeric column over the eval TSV's rows, in header order.

    Leading ``#`` comment lines are provenance, not data — an eval TSV may
    carry them, and ``mdp_gates`` already skips them. Feeding them to
    DictReader would make the first comment the header and turn a completed
    trial into a crashed one.
    """
    with tsv.open(newline="") as f:
        reader = csv.DictReader(
            (line for line in f if not line.startswith("#")), delimiter="\t")
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"no data rows in {tsv}")
    means: dict[str, float] = {}
    for col in cols:
        try:
            vals = [float(r[col]) for r in rows]
        except (TypeError, ValueError):
            continue
        means[col] = sum(vals) / len(vals)
    return cols, means


def resolve_metric(requested: str, cols: list[str],
                   means: dict[str, float]) -> str:
    """Pick the objective column: an explicit name, else the first ``*_mean``
    column in header order.

    The auto rule is domain-agnostic — no metric name is privileged. It relies
    on the spec-§9.3 convention that a domain's **primary objective column
    comes first** in the eval TSV (``profit_mean`` for a pricing domain,
    ``cost_total_mean`` for a cost domain, …). Pass ``--metric`` when a TSV
    carries several ``*_mean`` columns and the first is not the objective."""
    if requested != "auto":
        if requested in means:
            return requested
        raise ValueError(
            f"metric {requested!r} not in eval TSV; numeric columns: "
            f"{sorted(means)}")
    for col in cols:
        if col.endswith("_mean") and col in means:
            return col
    raise ValueError(
        f"could not auto-pick a metric; numeric columns: {sorted(means)} — "
        "pass --metric explicitly")

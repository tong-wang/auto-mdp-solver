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


@dataclass(frozen=True)
class ArgSpec:
    """One CLI argument of a domain script."""

    flag: str            # long option string, e.g. "--learning_rate"
    multi: bool          # nargs list-valued (e.g. --net_arch 64 64)
    default: object


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
        out[action.dest] = ArgSpec(flag=flag, multi=multi, default=action.default)
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


def parse_overrides(pairs: list[str], args_map: dict[str, ArgSpec],
                    label: str) -> dict[str, object]:
    """Parse repeatable KEY=VALUE overrides, validated against the script CLI."""
    out: dict[str, object] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--{label}-arg expects KEY=VALUE, got {pair!r}")
        if key not in args_map:
            raise ValueError(
                f"{label} script has no dest {key!r}; available: "
                f"{sorted(args_map)}")
        out[key] = value.split(",") if args_map[key].multi else value
    return out


def build_cmd(script: Path, args_map: dict[str, ArgSpec],
              assignments: dict[str, object]) -> list[str]:
    cmd = [sys.executable, str(script)]
    for dest, value in assignments.items():
        spec = args_map[dest]
        if isinstance(value, bool):
            if value:
                cmd.append(spec.flag)
        elif spec.multi and isinstance(value, (list, tuple)):
            cmd += [spec.flag, *[str(v) for v in value]]
        else:
            cmd += [spec.flag, str(value)]
    return cmd


def run_logged(cmd: list[str], log_path: Path, cwd: Path) -> None:
    """Run a subprocess from the domain dir, tee-ing output to a log file.

    Pins BLAS/torch to one thread unless the caller set it explicitly: on the
    small models these domains use, thread contention on tiny tensor ops
    dominates the runtime (measured ~50x wall-clock slowdown on sudoku 9x9).
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


def newest_model(trial_dir: Path) -> Path:
    """The saved model under the trial dir (run names are timestamped)."""
    hits = sorted(trial_dir.rglob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"no saved model (*.zip) under {trial_dir}")
    return hits[-1]


def tsv_column_means(tsv: Path) -> tuple[list[str], dict[str, float]]:
    """Mean of every numeric column over the eval TSV's rows, in header order."""
    with tsv.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
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
    """Pick the objective column: explicit name, else spec-canonical, else
    the first *_mean column in header order."""
    if requested != "auto":
        if requested in means:
            return requested
        raise ValueError(
            f"metric {requested!r} not in eval TSV; numeric columns: "
            f"{sorted(means)}")
    if "profit_mean" in means:
        return "profit_mean"
    for col in cols:
        if col.endswith("_mean") and col in means:
            return col
    raise ValueError(
        f"could not auto-pick a metric; numeric columns: {sorted(means)} — "
        "pass --metric explicitly")

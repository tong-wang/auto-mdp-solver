"""CLI for the domain-generic tuning harness.

    python -m mdp_tuning <domain_dir> [-s SCENARIO] [--metric NAME] \
        [--n-trials N] [--total-timesteps N] [--eval-seeds N] \
        [--train-arg KEY=VALUE ...] [--eval-arg KEY=VALUE ...]

Examples:
    python -m mdp_tuning sudoku -s 4x4_8 --metric solve_rate \
        --train-arg observation_mode=onehot --train-arg reward_mode=fill_solve
    python -m mdp_tuning sudoku --show-space        # what would be tuned
    python -m mdp_tuning sudoku -s 4x4_8 --summary-only

Optuna TPE proposes each trial from the algorithm-level space (spaces.py)
restricted to the knobs the domain's train script exposes. Each trial shells
out to the domain's own train + eval scripts; the study is stored in sqlite
under {domain}/results/tuning/, so interrupted sweeps resume with
the accumulated history intact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna

from mdp_tuning.driver import (
    DomainScripts, EVAL_SEEDS_DESTS, build_cmd, load_domain, newest_model,
    parse_overrides, resolve_metric, run_logged, tsv_column_means,
)
from mdp_tuning.spaces import SPACES


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mdp_tuning",
        description="Domain-generic hyperparameter tuning for spec-conformant "
                    "MDP domains (Optuna TPE over an algorithm-level space).",
    )
    p.add_argument("domain", type=str,
                   help="domain directory (path, or name relative to the repo root)")
    p.add_argument("-s", "--scenario_name", default=None, type=str,
                   help="scenario to tune on (default: the train script's own default)")
    p.add_argument("--algo", default="ppo", type=str, choices=sorted(SPACES))
    p.add_argument("--n-trials", default=25, type=int)
    p.add_argument("--total-timesteps", default=200_000, type=int,
                   help="training budget per trial")
    p.add_argument("--eval-seeds", default=500, type=int,
                   help="eval episodes used to score each trial")
    p.add_argument("--seed", default=42, type=int,
                   help="fixed training seed + TPE sampler seed")
    p.add_argument("--metric", default="auto", type=str,
                   help="objective column in the eval TSV (default: profit_mean "
                        "if present, else the first *_mean column)")
    p.add_argument("--minimize", action="store_true",
                   help="minimize the metric instead of maximizing")
    p.add_argument("--train-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="fixed override forwarded to the train script (repeatable); "
                        "comma-separate list values, e.g. net_arch=64,64")
    p.add_argument("--eval-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="fixed override forwarded to the eval script (repeatable)")
    p.add_argument("--study-name", default=None, type=str)
    p.add_argument("--storage", default=None, type=str,
                   help="optuna storage URL (default: sqlite under "
                        "{domain}/results/tuning/)")
    p.add_argument("--timeout", default=None, type=float,
                   help="wall-clock budget for the whole study, in seconds")
    p.add_argument("--show-space", action="store_true",
                   help="print which knobs would be tuned for this domain and exit")
    p.add_argument("--summary-only", action="store_true",
                   help="print the best trials of an existing study and exit")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def _resolve_domain_dir(name: str) -> Path:
    cand = Path(name)
    if cand.is_dir():
        return cand
    repo_root = Path(__file__).resolve().parent.parent
    cand = repo_root / name
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f"domain directory not found: {name}")


def show_space(scripts: DomainScripts, algo: str,
               pinned: set[str] = frozenset()) -> None:
    space = SPACES[algo]
    tunable = [k for k in space.knobs
               if k in scripts.train_args and k not in pinned]
    skipped = [k for k in space.knobs if k not in scripts.train_args]
    pinned_shown = [k for k in space.knobs if k in pinned]
    print(f"domain   : {scripts.prefix}  ({scripts.directory})")
    print(f"algo     : {algo}")
    print(f"train    : {scripts.train_script.name}")
    print(f"eval     : {scripts.eval_script.name}")
    seeds_dest = next((d for d in EVAL_SEEDS_DESTS if d in scripts.eval_args), None)
    print(f"eval episodes flag : {scripts.eval_args[seeds_dest].flag if seeds_dest else 'NOT FOUND'}")
    print(f"tunable  ({len(tunable)}): {', '.join(tunable)}")
    if pinned_shown:
        print(f"pinned   ({len(pinned_shown)}): {', '.join(pinned_shown)} "
              "(fixed via --train-arg)")
    if skipped:
        print(f"skipped  ({len(skipped)}): {', '.join(skipped)} "
              "(not exposed by the train script)")


def print_summary(study: optuna.Study, top: int = 5) -> None:
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        print("no completed trials yet.")
        return
    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE
    done.sort(key=lambda t: t.value, reverse=reverse)
    print(f"\n=== {study.study_name}: {len(done)} completed trials; top {top} ===")
    for t in done[:top]:
        print(f"  trial {t.number:3d}  value={t.value:.4f}  "
              f"cfg={t.user_attrs.get('cfg', '?')}")
    best = done[0]
    print(f"  -> best model: {best.user_attrs.get('model_path', '?')}")


def make_objective(scripts: DomainScripts, args: argparse.Namespace,
                   scenario: str, study_dir: Path,
                   fixed_train: dict, fixed_eval: dict):
    space = SPACES[args.algo]
    # a knob pinned via --train-arg is fixed, not tuned
    tunable = {k for k in space.knobs
               if k in scripts.train_args and k not in fixed_train}

    def _penalty(trial: optuna.Trial, err: Exception) -> float:
        """Value for a crashed trial (e.g. NaN divergence): the worst completed
        value so far. Returning a value — rather than failing the trial — is
        what teaches TPE to avoid divergent regions instead of resampling them."""
        trial.set_user_attr("failed", f"{err.__class__.__name__}: {err}"[:500])
        done = [t.value for t in trial.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None]
        if trial.study.direction == optuna.study.StudyDirection.MAXIMIZE:
            return min(done) if done else 0.0
        return max(done) if done else 0.0

    def objective(trial: optuna.Trial) -> float:
        cfg = space.sample(trial, tunable)
        trial.set_user_attr("cfg", str(cfg))
        trial_dir = study_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        train_assign: dict[str, object] = {
            "scenario_name": scenario,
            "total_timesteps": args.total_timesteps,
            "outdir": str(trial_dir),
        }
        if "seed" in scripts.train_args:
            train_assign["seed"] = args.seed
        train_assign.update(fixed_train)
        train_assign.update(cfg)
        try:
            run_logged(build_cmd(scripts.train_script, scripts.train_args, train_assign),
                       trial_dir / "train.log", scripts.directory)
            model = newest_model(trial_dir)
        except (RuntimeError, FileNotFoundError) as err:
            value = _penalty(trial, err)
            print(f"[trial {trial.number}] training FAILED "
                  f"({err.__class__.__name__}) -> penalized value {value:.4f}; "
                  f"see {trial_dir / 'train.log'}", flush=True)
            return value

        eval_tsv = trial_dir / "eval.tsv"
        eval_assign: dict[str, object] = {
            "model_path": str(model),
            "outfile": str(eval_tsv),
        }
        if "scenario_name" in scripts.eval_args:
            eval_assign["scenario_name"] = scenario
        seeds_dest = next((d for d in EVAL_SEEDS_DESTS
                           if d in scripts.eval_args), None)
        if seeds_dest:
            eval_assign[seeds_dest] = args.eval_seeds
        # forward shared fixed settings the eval script also understands
        # (e.g. observation_mode must match training)
        for k, v in fixed_train.items():
            if k in scripts.eval_args and k not in eval_assign:
                eval_assign[k] = v
        eval_assign.update(fixed_eval)
        run_logged(build_cmd(scripts.eval_script, scripts.eval_args, eval_assign),
                   trial_dir / "eval.log", scripts.directory)

        cols, means = tsv_column_means(eval_tsv)
        metric = resolve_metric(args.metric, cols, means)
        for k, v in means.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("model_path", str(model))
        print(f"[trial {trial.number}] {metric}={means[metric]:.4f}  cfg={cfg}",
              flush=True)
        return means[metric]

    return objective


def main() -> None:
    args = parse_args()

    domain_dir = _resolve_domain_dir(args.domain)
    scripts = load_domain(domain_dir, algo=args.algo)

    if args.show_space:
        show_space(scripts, args.algo)
        return

    scenario = args.scenario_name or str(scripts.train_args["scenario_name"].default)
    fixed_train = parse_overrides(args.train_arg, scripts.train_args, "train")
    fixed_eval = parse_overrides(args.eval_arg, scripts.eval_args, "eval")

    tuning_root = scripts.directory / "results" / "tuning"
    tuning_root.mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"{scripts.prefix}_{scenario}_{args.algo}"
    storage = args.storage or f"sqlite:///{tuning_root / 'optuna.db'}"
    if isinstance(storage, str) and storage.startswith("journal://"):
        # NFS-safe multi-node storage (cluster worker fleets): sqlite
        # corrupts under concurrent writers
        # on NFS; optuna's journal backend is append-only with advisory file
        # locks built for shared filesystems. URL form: journal:///abs/path.
        from optuna.storages import JournalStorage
        try:
            from optuna.storages.journal import (JournalFileBackend,
                                                 JournalFileOpenLock)
        except ImportError:  # optuna < 4 naming
            from optuna.storages import JournalFileOpenLock
            from optuna.storages import JournalFileStorage as JournalFileBackend
        jpath = storage[len("journal://"):]
        storage = JournalStorage(JournalFileBackend(
            jpath, lock_obj=JournalFileOpenLock(jpath)))
    study_dir = tuning_root / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize" if args.minimize else "maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

    if args.summary_only:
        print_summary(study)
        return

    print(f"study    : {study_name}  ({storage})")
    print(f"scenario : {scenario}")
    print(f"fixed    : train={fixed_train or '{}'}  eval={fixed_eval or '{}'}")
    print(f"budget   : {args.n_trials} trials x {args.total_timesteps} steps, "
          f"{args.eval_seeds} eval seeds"
          + (f", timeout {args.timeout}s" if args.timeout else ""))
    show_space(scripts, args.algo, pinned=set(fixed_train))

    study.optimize(
        make_objective(scripts, args, scenario, study_dir, fixed_train, fixed_eval),
        n_trials=args.n_trials,
        timeout=args.timeout,
        # a diverging config (e.g. NaN policy at high lr) fails its own trial,
        # not the study
        catch=(RuntimeError, FileNotFoundError, ValueError),
    )
    failed = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.FAIL
                 or "failed" in t.user_attrs)
    if failed:
        print(f"{failed} trial(s) crashed (typically NaN divergence) and were "
              f"penalized — see their train.log")
    print_summary(study)


if __name__ == "__main__":
    main()

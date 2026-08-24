"""CLI for the domain-generic tuning harness.

    python -m mdp_tuning <domain_dir> [-s SCENARIO] [--metric NAME] \
        [--n-trials N] [--total-timesteps N] [--eval-seeds N] \
        [--train-arg KEY=VALUE ...] [--eval-arg KEY=VALUE ...]

Examples:
    python -m mdp_tuning plugin/skills/mdp-solver/examples/dynamic_pricing -s simple \
        --metric revenue_mean --n-trials 25
    python -m mdp_tuning plugin/skills/mdp-solver/examples/dynamic_pricing --show-space  # what would be tuned
    python -m mdp_tuning plugin/skills/mdp-solver/examples/dynamic_pricing -s simple --summary-only

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
    DomainScripts, EVAL_SEEDS_DESTS, build_cmd, load_domain, parse_overrides,
    resolve_metric, resolve_model, run_logged, tsv_column_means,
)
from mdp_tuning.spaces import OPTIONAL_KNOBS, SPACES


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
    p.add_argument("--knobs", default=None, type=str,
                   choices=("core", "breadth", "all"),
                   help="knob tier to tune (SOLVE_LEVELS_PLAN §3.5): 'core' = "
                        "the high-impact knobs (default; right for ~25 trials); "
                        "'breadth' adds the second tier (use with ≥40 trials); "
                        "'all' also opens the frozen tier (clip, max_grad_norm, "
                        "gamma)")
    p.add_argument("--strict-knobs", action="store_true",
                   help="fail at launch when an in-tier knob has no matching "
                        "train-script dest (spec §8.2 tier 2). Implied whenever "
                        "--knobs is passed explicitly: asking for a tier and "
                        "silently getting a subset of it is the failure this "
                        "guards, and it is invisible in the result")
    p.add_argument("--fix", action="append", default=[], metavar="KNOB",
                   help="hold a knob at its L1-derived value instead of "
                        "tuning it (repeatable, spec §8.6) — the forced-move "
                        "lock; use --train-arg KEY=VALUE to pin at some other "
                        "value")
    p.add_argument("--beta", default=1.0, type=float,
                   help="problem discount factor (objective.discount_factor): "
                        "gamma is searched in (beta-0.05, beta], beta reachable; "
                        "gamma > beta is never sampled")
    p.add_argument("--episode-len", default=None, type=int,
                   help="mean episode length T̄; with --min-rollout-episodes it "
                        "raises the n_steps lower bound so every rollout spans "
                        "the episode floor")
    p.add_argument("--min-rollout-episodes", default=10, type=int,
                   help="episodes each rollout buffer must span when "
                        "--episode-len is given (rollout = n_steps × n_envs)")
    p.add_argument("--no-warm-start", action="store_false", dest="warm_start",
                   default=True,
                   help="do not enqueue trial 0 = the L1-derived centre "
                        "(the script's _L1_DERIVED, its defaults where it "
                        "declares none)")
    p.add_argument("--n-trials", default=25, type=int)
    p.add_argument("--total-timesteps", default=200_000, type=int,
                   help="training budget per trial")
    p.add_argument("--eval-seeds", default=500, type=int,
                   help="eval episodes used to score each trial")
    p.add_argument("--seed", default=42, type=int,
                   help="fixed training seed + TPE sampler seed")
    p.add_argument("--score-checkpoint", default="canonical", type=str,
                   choices=("canonical", "final"),
                   help="which artifact each trial is scored on: 'canonical' = "
                        "the run dir's own saved model (spec §8.4), so a "
                        "periodic checkpoint can never outrank it; 'final' = "
                        "the newest *.zip under the trial dir")
    p.add_argument("--metric", default="auto", type=str,
                   help="objective column in the eval TSV (default: the first "
                        "*_mean column in header order — the domain's primary "
                        "objective per spec §9.3)")
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
    args = _build_arg_parser().parse_args()
    # "core" is the default tier, but a *request* for a tier is a different
    # thing from falling into one, and only the request can be betrayed by a
    # missing dest — so the sentinel is resolved here rather than by argparse.
    args.knobs_explicit = args.knobs is not None
    if args.knobs is None:
        args.knobs = "core"
    return args


def _resolve_domain_dir(name: str) -> Path:
    cand = Path(name)
    if cand.is_dir():
        return cand
    repo_root = Path(__file__).resolve().parent.parent
    cand = repo_root / name
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f"domain directory not found: {name}")


def l1_defaults(scripts: DomainScripts, tunable: set[str]) -> dict[str, object]:
    """The L1-**derived** value of each tunable knob — the centre trial 0 claims.

    Read from the script's ``_L1_DERIVED`` (spec §8.4) where it declares one,
    and from the parser default otherwise. The two differ only after a
    promotion: §8.6 says the script's defaults *are* the L1 centre, which stops
    being true the moment a campaign adopts a discovered value as a default —
    and a warm start reading defaults would then enqueue the promoted value
    while calling it L1, so every Δ(L2−L1) the study reports would be measured
    from the wrong place. A script with no ``_L1_DERIVED`` is pre-convention,
    and for it the two are the same thing by §8.6's own statement.

    A ``None`` value is **kept**, not dropped. It means the script derives that
    knob at runtime (mab's ``norm_obs`` reads the observation mode inside
    ``parse_args``), so there is no single centre for the warm start to enqueue
    — and dropping it made that indistinguishable from a knob the space encoded
    successfully: trial 0 silently sampled it. Passing it on lets ``encode``
    refuse it, which is what the launch banner reports.
    """
    out: dict[str, object] = {}
    for k in tunable:
        if k in scripts.l1_derived:
            out[k] = scripts.l1_derived[k]
            continue
        v = scripts.train_args[k].default
        out[k] = tuple(v) if isinstance(v, list) else v
    return out


def derived_assignment(scripts: DomainScripts) -> dict[str, object]:
    """The derivation as train-script assignments — what every trial carries
    before the study's own deltas go on top.

    A trial's command sets the knobs the study *searches*, and every other dest
    falls to the parser default. That was harmless while §8.6's identity held —
    the defaults *were* the L1 centre — and stopped being harmless when §8.4
    separated the two: a promoted default, or one a script deliberately leaves
    null so L0 can exist at all, then rides on every trial in a knob nobody
    asked the study to move. Seeding from ``_L1_DERIVED`` makes a trial *be*
    the derivation plus the delta that was searched, which is the property
    Δ(L2−L1) is read off.

    Two omissions. A dest the train script does not expose is skipped —
    conformance's ``scripts.l1_derived`` check is what reports that. And a
    ``None`` derived value is skipped because there is nothing to put on a
    command line: it means the script computes that knob at runtime (§8.4's
    mab carve-out), so leaving the flag off *is* how the derivation is applied.
    """
    return {k: v for k, v in scripts.l1_derived.items()
            if k in scripts.train_args and v is not None}


def promoted_knobs(scripts: DomainScripts) -> dict[str, tuple]:
    """Knobs whose parser default has moved off the derived value — ``(derived,
    default)`` each.

    That is a *promotion*: a campaign adopting a discovered value, which §8.4
    makes safe for run names (they diff the derivation, so the tag survives).
    Every trial carries the derivation explicitly (``derived_assignment``), in
    the tier and out of it, so what this reports is provenance rather than a
    defect: on these knobs the study's trials and a hand-run of the train
    script are different configurations, and the difference appears in no trial
    run name for exactly the reason §8.4 gives. It scans every derived knob
    rather than only the searched ones; the caller marks which is which.

    A ``None`` default **counts**. It is what a script writes when the derived
    value cannot be its default — adi_flex keeps ``target_kl`` and
    ``clip_final`` null so L0 can have no KL valve and a constant schedule —
    which is the strongest form of the divergence, not an absent one. Skipping
    it is what let 583 trials run at ``target_kl=None`` against a derived 0.02
    with no banner anywhere (upstream #64).
    """
    out: dict[str, tuple] = {}
    for k, derived in sorted(scripts.l1_derived.items()):
        spec = scripts.train_args.get(k)
        if spec is None:
            continue                      # conformance reports this one
        default = tuple(spec.default) if isinstance(spec.default, list) else spec.default
        if default != derived:
            out[k] = (derived, default)
    return out


def unmatched_knobs(scripts: DomainScripts, algo: str, tier: str) -> list[str]:
    """In-tier knobs the train script exposes no matching dest for (§8.2 tier 2).

    ``OPTIONAL_KNOBS`` are the extractor-shaped ones only some architectures
    have; their absence is a fact about the domain, not a broken contract.
    """
    space = SPACES[algo]
    tier_knobs = space.tiers.get(tier, space.knobs)
    return [k for k in tier_knobs
            if k not in scripts.train_args and k not in OPTIONAL_KNOBS]


def show_space(scripts: DomainScripts, algo: str, tier: str = "all",
               pinned: set[str] = frozenset(),
               locked: set[str] = frozenset()) -> None:
    space = SPACES[algo]
    tier_knobs = space.tiers.get(tier, space.knobs)
    tunable = [k for k in tier_knobs
               if k in scripts.train_args and k not in pinned and k not in locked]
    outside = [k for k in space.knobs if k not in tier_knobs]
    skipped = [k for k in tier_knobs if k not in scripts.train_args]
    pinned_shown = [k for k in space.knobs if k in pinned]
    locked_shown = [k for k in tier_knobs if k in locked]
    print(f"domain   : {scripts.prefix}  ({scripts.directory})")
    print(f"algo     : {algo}  (tier: {tier})")
    print(f"train    : {scripts.train_script.name}")
    print(f"eval     : {scripts.eval_script.name}")
    seeds_dest = next((d for d in EVAL_SEEDS_DESTS if d in scripts.eval_args), None)
    print(f"eval episodes flag : {scripts.eval_args[seeds_dest].flag if seeds_dest else 'NOT FOUND'}")
    print(f"tunable  ({len(tunable)}): {', '.join(tunable)}")
    if pinned_shown:
        print(f"pinned   ({len(pinned_shown)}): {', '.join(pinned_shown)} "
              "(fixed via --train-arg)")
    if locked_shown:
        print(f"locked   ({len(locked_shown)}): {', '.join(locked_shown)} "
              "(held at the derived value via --fix)")
    if outside:
        opens = "all" if tier == "breadth" else "breadth|all"
        print(f"out-of-tier ({len(outside)}): {', '.join(outside)} "
              f"(open with --knobs {opens})")
    if skipped:
        print(f"skipped  ({len(skipped)}): {', '.join(skipped)} "
              "(not exposed by the train script)")
    rot = unmatched_knobs(scripts, algo, tier)
    if rot:
        print(f"WARNING: in-tier knob(s) with no matching train-script flag: "
              f"{', '.join(rot)} — the script predates the spec's required "
              f"dests, or a dest was renamed (unmatched-knob lint)")


def report_readiness(scripts: DomainScripts, args: argparse.Namespace) -> None:
    """What a study would find wrong, answered without launching one.

    Both facts below are otherwise discoverable only at launch — one as a fatal,
    one as a banner warning — and both are properties of the *script*, so a
    maintainer should be able to ask them of a domain directly.
    """
    space = SPACES[args.algo]
    reach = []
    for tier in ("core", "breadth", "all"):
        rot = [k for k in unmatched_knobs(scripts, args.algo, tier)
               if k not in set(args.fix)]
        reach.append(f"{tier} " + ("ok" if not rot else f"BLOCKED({', '.join(rot)})"))
    print(f"tier reach : {'  |  '.join(reach)}")

    if space.encode is None:
        return
    tunable = resolve_tunable(scripts, args, {})
    promoted = promoted_knobs(scripts)
    if promoted:
        detail = "; ".join(
            f"{k}: derived {d!r}, default {f!r}"
            + ("" if k in tunable else " (outside this tier — held at the "
                                       "derivation in every trial)")
            for k, (d, f) in promoted.items())
        print(f"promoted   : {len(promoted)} knob(s) whose default has moved "
              f"off the derivation — {detail}")
        print(f"             every trial command carries the derived value, so "
              f"a hand-run of {scripts.train_script.name} without those flags "
              f"is a different configuration (§8.4)")
    enq, skipped, snapped = space.encode(l1_defaults(scripts, tunable),
                                         **sample_kwargs_for(scripts, args))
    if not skipped and not snapped:
        print(f"warm start : trial 0 = the L1 centre ({len(enq)} knobs encoded)")
        return
    # spec §8.6: trial 0 carries the study's whole claim to measuring Δ(L2−L1),
    # so both ways it can fall short of L1 are answerable before launch
    if snapped:
        moved = "; ".join(f"{k} {a!r}->{u!r}" for k, (a, u) in sorted(snapped.items()))
        print(f"warm start : trial 0 = L1 ROUNDED to the grid — {moved}")
    if skipped:
        detail = ", ".join(f"{k}={scripts.train_args[k].default!r}"
                           for k in sorted(skipped))
        print(f"warm start : trial 0 is NOT the L1 centre — the space refuses "
              f"{detail} outright, so it is sampled instead")


def print_summary(study: optuna.Study, top: int = 5,
                  eval_seeds: int | None = None) -> None:
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
    # the winner's full recorded eval row, name-blind: the study's objective
    # is one column of it, and a block-portable column (a regret, a ratio)
    # beside the raw one is what makes the number readable across studies
    means = {k: v for k, v in best.user_attrs.items()
             if k.endswith("_mean") and isinstance(v, (int, float))}
    if means:
        row = "  ".join(f"{k}={v:.4f}" for k, v in means.items())
        print(f"  -> winner's eval row: {row}")
    print(f"  -> best model: {best.user_attrs.get('model_path', '?')}"
          + (f"  (scored on {best.user_attrs['score_checkpoint']} checkpoint)"
             if "score_checkpoint" in best.user_attrs else ""))
    # spec §9.7: the trial layer is a different instrument from the protocol
    # layer, and this is the number a human reads — so the warning belongs at
    # the moment it is read, not only in the spec. A campaign that compared
    # these two layers directly closed a tuning round as a null result on an
    # inverted reading (upstream #25).
    block = f", {eval_seeds} seeds" if eval_seeds else ""
    print(f"  -> best trial value {best.value:.4f} is a TRIAL-LAYER score"
          f"{block} — not comparable to a protocol number (§9.7). Re-score the "
          f"winner with {{domain}}_ppo_eval.py before quoting it anywhere; it "
          f"is also a maximum over {len(done)} trials, so it flatters the "
          f"procedure even when it is honest about the artifact.")


def resolve_tunable(scripts: DomainScripts, args: argparse.Namespace,
                    fixed_train: dict) -> set[str]:
    """Tier knobs ∩ exposed dests, minus --train-arg pins and --fix locks."""
    space = SPACES[args.algo]
    tier_knobs = space.tiers.get(args.knobs, space.knobs)
    bad = [k for k in args.fix if k not in space.knobs]
    if bad:
        raise ValueError(f"--fix knob(s) not in the {args.algo} space: {bad}; "
                         f"known: {sorted(space.knobs)}")
    return {k for k in tier_knobs
            if k in scripts.train_args
            and k not in fixed_train and k not in set(args.fix)}


def sample_kwargs_for(scripts: DomainScripts,
                      args: argparse.Namespace) -> dict[str, object]:
    """Domain-context kwargs for the sampler (SOLVE_LEVELS_PLAN §3.4):
    structural n_envs from the script default, and the rollout-episode floor
    translated into a per-env n_steps lower bound."""
    n_envs_spec = scripts.train_args.get("n_envs")
    n_envs = int(n_envs_spec.default) if n_envs_spec and n_envs_spec.default else 1
    min_n_steps = None
    if args.episode_len:
        min_n_steps = -(-args.min_rollout_episodes * args.episode_len // n_envs)
    # the derivation's own depth: core searches net_arch's width and holds the
    # layer count the train script derived (spaces.NET_DEPTH_RANGE)
    arch = scripts.train_args.get("net_arch")
    default_arch = arch.default if arch else None
    net_depth_default = (len(default_arch)
                         if isinstance(default_arch, (list, tuple)) and default_arch
                         else 2)
    return {"n_envs": n_envs, "min_n_steps": min_n_steps, "beta": args.beta,
            "tier": args.knobs, "net_depth_default": net_depth_default}


def train_assignment(scripts: DomainScripts, args: argparse.Namespace,
                     scenario: str, trial_dir: Path,
                     fixed_train: dict, cfg: dict) -> dict[str, object]:
    """One trial's train-script assignment, in precedence order.

    *derivation → study-level → ``--train-arg`` → sampled.* The study level
    wins over the derivation because its three dests are the study's to set —
    a trial runs a trial budget, not the derived one — the operator wins over
    both, and the sampler wins last, since leaving the centre is what a
    searched knob is for. Trial 0 still sits *at* the centre: the warm start
    puts the derived values into ``cfg`` itself.
    """
    assign: dict[str, object] = dict(derived_assignment(scripts))
    assign.update({
        "scenario_name": scenario,
        "total_timesteps": args.total_timesteps,
        "outdir": str(trial_dir),
    })
    if "seed" in scripts.train_args:
        assign["seed"] = args.seed
    assign.update(fixed_train)
    assign.update(cfg)
    return assign


def make_objective(scripts: DomainScripts, args: argparse.Namespace,
                   scenario: str, study_dir: Path,
                   fixed_train: dict, fixed_eval: dict):
    space = SPACES[args.algo]
    tunable = resolve_tunable(scripts, args, fixed_train)
    sample_kw = sample_kwargs_for(scripts, args)
    # a derivation the CLI cannot express fails identically on every trial, so
    # it fails here rather than spending the whole budget proving it
    try:
        build_cmd(scripts.train_script, scripts.train_args,
                  derived_assignment(scripts))
    except ValueError as err:
        raise SystemExit(
            f"FATAL: {scripts.train_script.name} derives a value its own CLI "
            f"cannot carry, so no trial can run the derivation — {err}")

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
        cfg = space.sample(trial, tunable, **sample_kw)
        # one-DOF schedule pairs: finals derived from the tuned init, so a
        # schedule can never invert (lr_final=lr/10, clip_final=clip_init/4)
        for dest, (src, factor) in space.derived.items():
            if src in cfg and dest in scripts.train_args and dest not in fixed_train:
                cfg[dest] = round(cfg[src] * factor, 10)
        trial.set_user_attr("cfg", str(cfg))
        trial_dir = study_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        train_assign = train_assignment(scripts, args, scenario, trial_dir,
                                        fixed_train, cfg)
        try:
            run_logged(build_cmd(scripts.train_script, scripts.train_args, train_assign),
                       trial_dir / "train.log", scripts.directory)
            model = resolve_model(trial_dir, scenario=scenario, algo=args.algo,
                                  mode=args.score_checkpoint)
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
        trial.set_user_attr("score_checkpoint", args.score_checkpoint)
        print(f"[trial {trial.number}] {metric}={means[metric]:.4f}  cfg={cfg}",
              flush=True)
        return means[metric]

    return objective


def main() -> None:
    args = parse_args()

    domain_dir = _resolve_domain_dir(args.domain)
    scripts = load_domain(domain_dir, algo=args.algo)

    if args.show_space:
        show_space(scripts, args.algo, tier=args.knobs, locked=set(args.fix))
        report_readiness(scripts, args)
        return

    # spec §8.2 tier 2 — before the study directory exists, because a study that
    # searches 7 knobs while its banner says 8 leaves no trace of the
    # discrepancy in any artifact downstream of this launch.
    # --fix is the operator saying "hold this at its derived value" — a knob
    # deliberately outside the search, which is what a missing dest produces
    # too, so it is asked-for rather than a silent shrink.
    rot = [k for k in unmatched_knobs(scripts, args.algo, args.knobs)
           if k not in set(args.fix)]
    if rot and not args.summary_only and (args.knobs_explicit or args.strict_knobs):
        raise SystemExit(
            f"FATAL: --knobs {args.knobs} asks for {', '.join(rot)}, but "
            f"{scripts.train_script.name} exposes no matching dest — the study "
            f"would search a smaller space than it reports. Expose the flag(s), "
            f"or --fix {rot[0]} to lock it deliberately, or drop to a tier that "
            f"does not include them. (--show-space lists the resolved space "
            f"without launching.)")

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
        print_summary(study, eval_seeds=args.eval_seeds)
        return

    print(f"study    : {study_name}  ({storage})")
    print(f"scenario : {scenario}")
    print(f"fixed    : train={fixed_train or '{}'}  eval={fixed_eval or '{}'}")
    print(f"budget   : {args.n_trials} trials x {args.total_timesteps} steps, "
          f"{args.eval_seeds} eval seeds"
          + (f", timeout {args.timeout}s" if args.timeout else ""))
    print(f"scored on: {args.score_checkpoint} checkpoint, metric={args.metric}")
    show_space(scripts, args.algo, tier=args.knobs,
               pinned=set(fixed_train), locked=set(args.fix))

    # warm start (SOLVE_LEVELS_PLAN §3.6): trial 0 = the derived value of every
    # tunable knob (§8.4's _L1_DERIVED, the script's defaults where it declares
    # none). The study then directly measures what tuning adds over L1.
    space = SPACES[args.algo]
    if args.warm_start and space.encode is not None and not study.trials:
        tunable = resolve_tunable(scripts, args, fixed_train)
        sample_kw = sample_kwargs_for(scripts, args)
        # one reader for the centre, shared with --show-space: a readiness
        # report that answers a different question from the launch it predicts
        # is worse than no report
        defaults = l1_defaults(scripts, tunable)
        promoted = promoted_knobs(scripts)
        if promoted:
            print(f"warm start: {len(promoted)} knob(s) PROMOTED — the script "
                  f"default has moved off the derived value (§8.4); every trial "
                  f"command carries the derivation, not the default:")
            for dest, (derived, default) in promoted.items():
                mark = ("searched, and trial 0 sits at the derivation"
                        if dest in tunable
                        else "outside the searched tier, held at the derivation "
                             "in every trial")
                print(f"         {dest}: derived {derived!r}, "
                      f"default {default!r} — {mark}")
            study.set_user_attr("l1_promoted",
                                {k: [repr(d), repr(f)] for k, (d, f) in promoted.items()})
        enq, skipped, snapped = space.encode(defaults, **sample_kw)
        if enq:
            source = ("_L1_DERIVED" if scripts.l1_derived
                      else "train-script defaults")
            study.enqueue_trial(enq)
            print(f"warm start: enqueued trial 0 = the L1 centre from {source} "
                  f"({', '.join(sorted(enq))})")
        # trial 0 is supposed to BE the L1 centre, which is what makes Δ(L2−L1)
        # readable off the study — so any distance from it is stated, and
        # recorded on the study so the record outlives this console.
        if snapped:
            print(f"WARNING: warm start: {len(snapped)} L1 default(s) are not "
                  f"exactly representable and were moved to the nearest value "
                  f"the space has — trial 0 is L1 rounded to the grid, not L1:")
            for dest, (asked, used) in sorted(snapped.items()):
                print(f"         {dest}: {asked!r} -> {used!r}")
            study.set_user_attr("warm_start_snapped",
                                {k: [repr(a), repr(u)] for k, (a, u) in snapped.items()})
        if skipped:
            print(f"WARNING: warm start: the {args.algo} space refuses "
                  f"{sorted(skipped)} outright, so trial 0 SAMPLES it — trial 0 "
                  f"is then not the L1 centre and the study does not measure "
                  f"what tuning adds over L1. A refusal is a signal, not a "
                  f"rounding error (§8.6): γ > β, an n_steps under the rollout "
                  f"floor, or a structure with no nearest neighbour.")
            for dest in sorted(skipped):
                print(f"         {dest}={defaults.get(dest)!r}")
            study.set_user_attr("warm_start_skipped", sorted(skipped))

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
    print_summary(study, eval_seeds=args.eval_seeds)
    report_importances(study, study_dir)


def report_importances(study: optuna.Study, study_dir: Path) -> None:
    """Per-study parameter importances (fANOVA) — the empirical feedback that
    turns the tier assignment from a literature prior into a measured one
    (SOLVE_LEVELS_PLAN §3.8). Accretes into the escalation playbook."""
    try:
        from optuna.importance import get_param_importances
        imp = get_param_importances(study)
    except Exception as err:  # <2 completed trials, missing sklearn, …
        print(f"(param importances unavailable: "
              f"{err.__class__.__name__}: {err})")
        return
    print("\nparam importances (fANOVA):")
    for k, v in imp.items():
        print(f"  {k:24s} {v:.4f}")
    out = study_dir / "param_importances.txt"
    out.write_text("".join(f"{k}\t{v:.6f}\n" for k, v in imp.items()))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

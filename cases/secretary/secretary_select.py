"""Post-hoc secretary checkpoint selection and local artifact promotion.

The screen uses seeds disjoint from the protocol block.  Every checkpoint is
scored with the VecNormalize snapshot saved at the same training step.  The
top-k checkpoints are confirmed on the protocol block, after which the winner
and its matching normalizer are copied to the canonical names inside the
ignored run directory.  No model artifact belongs in version control.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SCREEN_SEEDS = 2_048
DEFAULT_SCREEN_FIRST = 1_000_000
DEFAULT_PROTOCOL_SEEDS = 8_192
DEFAULT_PROTOCOL_FIRST = 0
DEFAULT_TOP_K = 5


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Completed PPO run containing checkpoints/")
    parser.add_argument("-s", "--scenario_name", default="standard")
    parser.add_argument("--screen-seeds", type=int, default=DEFAULT_SCREEN_SEEDS)
    parser.add_argument("--screen-first", type=int, default=DEFAULT_SCREEN_FIRST)
    parser.add_argument("--protocol-seeds", type=int, default=DEFAULT_PROTOCOL_SEEDS)
    parser.add_argument("--protocol-first", type=int, default=DEFAULT_PROTOCOL_FIRST)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-envs", type=int, default=64)
    parser.add_argument(
        "--outdir",
        default=None,
        help="Evaluation records (default: <run_dir>/selection)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Selection ledger (default: <run_dir>/selection.tsv)",
    )
    parser.add_argument(
        "--promote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Promote the confirmed winner to canonical local artifact names.",
    )
    return parser


def _step_of(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if not match:
        raise ValueError(f"checkpoint name has no step count: {path}")
    return int(match.group(1))


def _checkpoints(run_dir: Path) -> list[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(
        checkpoint_dir.glob("*_ppo_*_steps.zip"),
        key=_step_of,
    )
    if not checkpoints:
        raise SystemExit(f"no PPO checkpoints found under {checkpoint_dir}")
    return checkpoints


def _vecnorm_for(checkpoint: Path) -> Path:
    step = _step_of(checkpoint)
    matches = sorted(checkpoint.parent.glob(f"*_ppo_vecnormalize_{step}_steps.pkl"))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one VecNormalize sidecar for {checkpoint.name}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _read_eval(path: Path) -> dict[str, float]:
    lines = [
        line for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    row = next(csv.DictReader(lines, delimiter="\t"))
    return {
        "success_mean": float(row["success_mean"]),
        "success_var": float(row["success_var"]),
        "rank_mean": float(row["expected_selected_rank_mean"]),
    }


def _evaluate(
    domain_dir: Path,
    checkpoint: Path,
    vecnorm: Path,
    outfile: Path,
    scenario_name: str,
    n_seeds: int,
    first_seed: int,
    batch_envs: int,
) -> dict[str, float]:
    command = [
        sys.executable,
        str(domain_dir / "secretary_ppo_eval.py"),
        "--model-path",
        str(checkpoint),
        "--vecnorm-path",
        str(vecnorm),
        "--outfile",
        str(outfile),
        "-s",
        scenario_name,
        "--n-seeds",
        str(n_seeds),
        "--first-seed",
        str(first_seed),
        "--batch-envs",
        str(batch_envs),
    ]
    subprocess.run(command, check=True)
    return _read_eval(outfile)


def _blocks_overlap(first_a: int, size_a: int, first_b: int, size_b: int) -> bool:
    return max(first_a, first_b) < min(first_a + size_a, first_b + size_b)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if _blocks_overlap(
        args.screen_first,
        args.screen_seeds,
        args.protocol_first,
        args.protocol_seeds,
    ):
        raise SystemExit("selection and protocol seed blocks must be disjoint")

    domain_dir = Path(__file__).resolve().parent
    run_dir = Path(args.run_dir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else run_dir / "selection"
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoints(run_dir)

    print(f"run={run_dir}")
    print(
        f"screen={args.screen_first}.."
        f"{args.screen_first + args.screen_seeds - 1} "
        f"checkpoints={len(checkpoints)}"
    )
    screened: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        step = _step_of(checkpoint)
        vecnorm = _vecnorm_for(checkpoint)
        metrics = _evaluate(
            domain_dir,
            checkpoint,
            vecnorm,
            outdir / f"screen_{step}.tsv",
            args.scenario_name,
            args.screen_seeds,
            args.screen_first,
            args.batch_envs,
        )
        screened.append({
            "checkpoint": checkpoint,
            "vecnorm": vecnorm,
            "step": step,
            "screen": metrics,
        })
        print(f"screen step={step}: success={metrics['success_mean']:.10f}")

    screened.sort(
        key=lambda item: (-item["screen"]["success_mean"], item["step"]),
    )
    finalists = screened[: min(args.top_k, len(screened))]
    for item in finalists:
        step = item["step"]
        metrics = _evaluate(
            domain_dir,
            item["checkpoint"],
            item["vecnorm"],
            outdir / f"confirm_{step}.tsv",
            args.scenario_name,
            args.protocol_seeds,
            args.protocol_first,
            args.batch_envs,
        )
        item["confirm"] = metrics
        print(f"confirm step={step}: success={metrics['success_mean']:.10f}")

    finalists.sort(
        key=lambda item: (
            -item["confirm"]["success_mean"],
            -item["screen"]["success_mean"],
            item["step"],
        ),
    )
    winner = finalists[0]

    summary = Path(args.summary).resolve() if args.summary else run_dir / "selection.tsv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w") as stream:
        stream.write(
            "layer\tcheckpoint\tsteps\tn_seeds\tfirst_seed\t"
            "success_mean\tsuccess_var\texpected_selected_rank_mean\n"
        )
        for item in screened:
            metric = item["screen"]
            stream.write(
                f"screen\t{item['checkpoint'].name}\t{item['step']}\t"
                f"{args.screen_seeds}\t{args.screen_first}\t"
                f"{metric['success_mean']:.10f}\t{metric['success_var']:.10f}\t"
                f"{metric['rank_mean']:.10f}\n"
            )
        for item in finalists:
            metric = item["confirm"]
            stream.write(
                f"confirm\t{item['checkpoint'].name}\t{item['step']}\t"
                f"{args.protocol_seeds}\t{args.protocol_first}\t"
                f"{metric['success_mean']:.10f}\t{metric['success_var']:.10f}\t"
                f"{metric['rank_mean']:.10f}\n"
            )

    if args.promote:
        model_destination = run_dir / f"{args.scenario_name}_ppo.zip"
        vecnorm_destination = run_dir / "vecnormalize.pkl"
        _atomic_copy(winner["checkpoint"], model_destination)
        _atomic_copy(winner["vecnorm"], vecnorm_destination)
        _evaluate(
            domain_dir,
            model_destination,
            vecnorm_destination,
            run_dir / f"ppo_eval_{args.scenario_name}.tsv",
            args.scenario_name,
            args.protocol_seeds,
            args.protocol_first,
            args.batch_envs,
        )
        print(
            f"promoted step={winner['step']} -> {model_destination.name} + "
            f"{vecnorm_destination.name}"
        )
    else:
        print(f"winner step={winner['step']} (promotion disabled)")
    print(f"selection record -> {summary}")


if __name__ == "__main__":
    main()

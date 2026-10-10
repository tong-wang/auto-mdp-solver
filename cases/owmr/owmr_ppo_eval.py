"""Protocol-tier evaluation of a trained PPO model on owmr (spec 9).

Scores one model on one scenario over the CRN block, writes the spec-9.3 record
with the harvest provenance lines (steps_scored / steps_run / steps_budget) plus
the per-seed sidecar, and optionally reports the paired delta against another
arm's sidecar.

Usage:
    python owmr_ppo_eval.py -s base --model-path results/base/PPO_.../checkpoints/ppo_owmr_2500000_steps.zip \\
        --vecnorm-path .../checkpoints/ppo_owmr_vecnormalize_2500000_steps.pkl --outfile .../confirm_2500000.tsv \\
        --paired-with results/base/benchmark/benchmark_lb_heuristic_eval_base.seeds.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from mdp_gates.completion import provenance_lines
from owmr_eval_common import SELECT_SEED_OFFSET, eval_seed_block, paired_report, rollout_seeds, write_record
from owmr_gym import OwmrEnv
from owmr_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model on owmr.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None,
                   help="defaults to <model_dir>/vecnormalize.pkl when present")
    p.add_argument("-s", "--scenario_name", type=str, default="base", choices=list(SCENARIOS))
    p.add_argument("-o", "--observation_mode", type=str, default="raw", choices=list(OwmrEnv.OBS_MODES))
    p.add_argument("-a", "--action_mode", type=str, default="order_frac", choices=list(OwmrEnv.ACTION_MODES))
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0,
                   help="first seed of the block (0 = the protocol block, the only quotable one)")
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=False,
                   help="sample actions instead of the deterministic mean (a labelled figure, never the record)")
    p.add_argument("--arm", type=str, default=None, help="label for the record's `arm` column")
    p.add_argument("--outfile", type=str, default=None)
    p.add_argument("--paired-with", type=str, default=None, help="another arm's .seeds.tsv")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    model_dir = model_path.parent
    vecnorm = Path(args.vecnorm_path) if args.vecnorm_path else model_dir / "vecnormalize.pkl"
    outfile = Path(args.outfile) if args.outfile else model_dir / f"ppo_eval_{args.scenario_name}.tsv"
    arm = args.arm or model_dir.name
    scenario = SCENARIOS[args.scenario_name]
    model = PPO.load(str(model_path), device="cpu")
    deterministic = not args.stochastic
    block = ("protocol" if args.first_seed == 0 else "selection" if args.first_seed == SELECT_SEED_OFFSET
             else "non-standard")
    print(f"[eval] model    : {model_path}")
    print(f"[eval] vecnorm  : {vecnorm if vecnorm.exists() else '(none)'}")
    print(f"[eval] scenario : {args.scenario_name}   obs={args.observation_mode}   act={args.action_mode}")
    print(f"[eval] seeds    : {args.first_seed}..{args.first_seed + args.n_seeds - 1}   ({block} block)   "
          f"deterministic={deterministic}")
    if args.first_seed != 0:
        print("[eval] WARNING: not the protocol block — ranking only, never quote this number (9.7).")

    def act_fn(obs: np.ndarray, raw_envs) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)
    per_seed = rollout_seeds(scenario=scenario, observation_mode=args.observation_mode,
                             action_mode=args.action_mode, seeds=seeds, act_fn=act_fn,
                             batch=args.batch_envs, vecnorm_path=vecnorm)
    run_dir = model_dir.parent if model_dir.name == "checkpoints" else model_dir
    prov = [f"# arm: {arm}   model: {model_path}   vecnorm: {vecnorm if vecnorm.exists() else '(none)'}",
            f"# scenario: {args.scenario_name}   obs: {args.observation_mode}   act: {args.action_mode}   "
            f"seeds: {args.first_seed}..{args.first_seed + args.n_seeds - 1} ({block})   "
            f"eval: {'stochastic' if args.stochastic else 'deterministic'}"]
    prov += provenance_lines(model_path, run_dir)
    write_record(outfile, args.scenario_name, arm, per_seed, seeds=seeds, provenance=prov)
    if args.paired_with:
        paired_report(per_seed["cost_total"], Path(args.paired_with), Path(args.paired_with).stem)


if __name__ == "__main__":
    main()

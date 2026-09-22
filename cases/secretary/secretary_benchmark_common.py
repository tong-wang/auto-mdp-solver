"""Shared spec-§9 evaluation harness for secretary benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from secretary_gym import SecretaryEnv
from secretary_mdp import SecretaryState
from secretary_scenarios import SCENARIOS, SecretaryScenario

DEFAULT_N_SEEDS = 8192


class Policy(Protocol):
    def act(self, state: SecretaryState) -> int: ...


PolicyFactory = Callable[[SecretaryScenario, int], Policy]


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-s", "--scenario_name", default="standard", choices=SCENARIOS)
    parser.add_argument("-o", "--observation_mode", default="relative", choices=["relative"])
    parser.add_argument("-a", "--action_mode", default="accept", choices=["accept"])
    parser.add_argument("--reward_mode", default="success", choices=["success"])
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--outfile", type=str, default=None)
    return parser


def _stats(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(values.mean())
    if values.size < 2:
        return mean, 0.0, 0.0, 0.0
    variance = float(values.var(ddof=1))
    downside = float(np.square(np.maximum(mean - values, 0.0)).sum() / (values.size - 1))
    upside = float(np.square(np.maximum(values - mean, 0.0)).sum() / (values.size - 1))
    return mean, variance, downside, upside


def _output_path(method: str, args: argparse.Namespace) -> Path:
    domain_dir = Path(__file__).resolve().parent
    if args.outfile:
        supplied = Path(args.outfile)
        return supplied if supplied.is_absolute() else domain_dir / supplied
    return (domain_dir / "results" / args.scenario_name / "benchmark"
            / f"benchmark_{method}_eval_{args.scenario_name}.tsv")


def evaluate(method: str, make_policy: PolicyFactory, args: argparse.Namespace) -> Path:
    """Evaluate one observable policy on the shared CRN episode-seed block."""
    if args.n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    scenario = SCENARIOS[args.scenario_name]
    env = SecretaryEnv(
        scenario=scenario,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        reward_mode=args.reward_mode,
    )
    successes = np.empty(args.n_seeds, dtype=np.float64)
    selected_ranks = np.empty(args.n_seeds, dtype=np.float64)
    episode_seeds = range(args.first_seed, args.first_seed + args.n_seeds)
    for index, episode_seed in enumerate(episode_seeds):
        if index % 1000 == 0:
            print(f"  seed {index}/{args.n_seeds}", flush=True)
        env.reset(seed=episode_seed)
        policy = make_policy(scenario, episode_seed)
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(policy.act(env._state))
        successes[index] = float(info["outcome"]["success"])
        selected_ranks[index] = float(info["selected_rank"])
    env.close()

    success_mean, success_var, semivar_d, semivar_u = _stats(successes)
    rank_mean, rank_var, _, _ = _stats(selected_ranks)
    outfile = _output_path(method, args)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    header = ("scenario\tn_candidates\tn_seeds\tsuccess_mean\tsuccess_var\t"
              "semivar_d\tsemivar_u\texpected_selected_rank_mean\t"
              "expected_selected_rank_var")
    row = (f"{args.scenario_name}\t{scenario.n_candidates}\t{args.n_seeds}\t"
           f"{success_mean:.10f}\t{success_var:.10f}\t{semivar_d:.10f}\t"
           f"{semivar_u:.10f}\t{rank_mean:.10f}\t{rank_var:.10f}")
    outfile.write_text(
        f"# method: {method}\n"
        f"# seed_block: {args.first_seed}..{args.first_seed + args.n_seeds - 1}\n"
        f"{header}\n{row}\n"
    )
    sidecar = outfile.with_suffix(".seeds.tsv")
    with sidecar.open("w") as stream:
        stream.write("seed\tsuccess\texpected_selected_rank\n")
        for seed, success, rank in zip(episode_seeds, successes, selected_ranks):
            stream.write(f"{seed}\t{success:.0f}\t{rank:.0f}\n")
    se = float(np.sqrt(success_var / args.n_seeds))
    print(f"{method}: success={success_mean:.10f} ± {se:.10f} SE; "
          f"selected_rank={rank_mean:.6f}")
    print(f"results saved -> {outfile}")
    return outfile

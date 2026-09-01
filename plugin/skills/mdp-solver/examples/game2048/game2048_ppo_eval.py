"""Evaluate a trained game2048 policy over the spec-§9 seed protocol.

Runs episode seeds 0..n_seeds-1 on the named scenario through
``game2048_benchmark_common.evaluate`` — the *same* seed loop, env and TSV
writer the benchmarks use — and writes

    results/{scenario}/{run}/ppo_eval_{scenario}.tsv

with columns identical to the benchmark TSVs, so ``mdp_gates`` compares them
directly:

    python -m mdp_gates \\
      --candidate results/3x3_20/<run>/ppo_eval_3x3_20.tsv \\
      --baseline  results/3x3_20/benchmark/benchmark_random_eval_3x3_20.tsv \\
      --baseline  results/3x3_20/benchmark/benchmark_greedy_eval_3x3_20.tsv \\
      --reference results/3x3_20/benchmark/benchmark_expectimax_d2_eval_3x3_20.tsv \\
      --n-seeds 8192

Two deliberate choices:

**What is scored** is total merge score — the objective — whatever reward mode
the policy was TRAINED with. A run trained under ``score_penalty`` is still
judged on the game, which is why ``invalid_penalty`` is kept out of
``objective.per_step_components``.

**What is loaded** is ``Game2048Policy``, the deployable Stage-5 artifact,
rather than the raw SB3 model. The thing on the leaderboard is then the thing
that ships — encoding, obs stats (§9.5) and action masking all exercised on
the eval path instead of only in a smoke test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from game2048_benchmark_common import build_arg_parser, evaluate
from game2048_gym import ACTION_MODES, OBSERVATION_MODES, REWARD_MODES
from game2048_policy import Game2048Policy
from game2048_scenarios import SCENARIOS


class _PolicyAdapter:
    """Give Game2048Policy the ``act(board, mask)`` shape the harness calls.

    The mask is recomputed inside the policy from the same ``valid_moves`` the
    MDP layer uses, so it is ignored here rather than threaded through — the
    deployed policy must not depend on a caller handing it a correct mask.
    """

    def __init__(self, policy: Game2048Policy) -> None:
        self.policy = policy

    def act(self, board, mask) -> int:
        return self.policy.act(board)


def _build_parser() -> argparse.ArgumentParser:
    p = build_arg_parser(__doc__)
    p.add_argument("--model-path", required=True,
                   help="path to the saved .zip; its directory is the run dir")
    p.add_argument("-o", "--observation_mode", default="vec",
                   choices=list(OBSERVATION_MODES))
    p.add_argument("-a", "--action_mode", default="masked",
                   choices=list(ACTION_MODES))
    p.add_argument("-r", "--reward_mode", default="score",
                   choices=list(REWARD_MODES),
                   help="recorded for provenance; scoring is always merge points")
    p.add_argument("--stochastic", dest="deterministic", action="store_false",
                   default=True)
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    model_path = Path(args.model_path).resolve()
    run_dir = model_path.parent

    template = SCENARIOS[args.scenario]
    scenario = template(0) if callable(template) else template

    policy = Game2048Policy(
        model_path,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        grid_size=scenario.grid_size,
        deterministic=args.deterministic,
    )
    adapter = _PolicyAdapter(policy)

    evaluate(
        f"ppo[{model_path.stem}]",
        lambda env: adapter,
        args,
        out_path=run_dir / f"ppo_eval_{args.scenario}.tsv",
        note=(f" | model={model_path.name} obs={args.observation_mode} "
              f"act={args.action_mode} trained_rew={args.reward_mode} "
              f"deterministic={args.deterministic}"),
    )


if __name__ == "__main__":
    main()

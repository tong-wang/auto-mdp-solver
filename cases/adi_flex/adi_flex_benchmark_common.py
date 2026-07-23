"""Shared evaluation machinery for the AdiFlex benchmarks.

Every benchmark eval — and the PPO eval — scores policies on the *same* seed
protocol (episode seeds 0..n-1, spec section 9.2) over the *same* simulator, so
the numbers are directly comparable and `mdp_gates` can build an honest
standard error from them.

Benchmarks drive the MDP layer directly rather than going through the gym. The
gym's reset(seed=k) forwards k as the episode seed unchanged, so the demand
draws are identical either way; driving the MDP directly just avoids coupling a
benchmark to an observation encoding it does not use.

Metric convention: the reported metric is `reward` = -(total cost), because the
eval gate passes when the candidate's mean is *higher*. Cost is carried
alongside for human reading only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from adi_flex_scenarios import AdiFlexScenario
import adi_flex_mdp as mdp


TSV_HEADER = (
    "lambda_now\tlambda_next\tlambda_later\t"
    "reward_mean\treward_var\tsemivar_d\tsemivar_u"
)


class BenchmarkPolicy(Protocol):
    """A non-RL policy over the true (unrelaxed) AdiFlex dynamics."""

    def reset(self) -> None:
        """Called once at the start of each episode."""

    def act(
        self,
        scenario: AdiFlexScenario,
        state: "mdp.AdiFlexState",
        vhat: int,
    ) -> tuple[int, int]:
        """Return (order_quantity, hold_back) at the pre-order state.

        `vhat` is the *total* advance demand already known to be due next
        period — the paper's v-hat. It is not the same as `state.due_next`,
        which is the *unsatisfied* remainder after any early shipments, and the
        AP ordering policy is indexed by the former.
        """


def rollout(
    scenario: AdiFlexScenario,
    policy: BenchmarkPolicy,
    episode_seed: int,
) -> float:
    """Run one episode; return total cost."""
    policy.reset()
    # stochastic policies (the random benchmark) re-seed per episode so the
    # whole benchmark is reproducible from the episode seed alone
    seed_episode = getattr(policy, "seed_episode", None)
    if seed_episode is not None:
        seed_episode(episode_seed)
    state, _ = mdp.init_state(scenario=scenario, episode_seed=episode_seed)
    total_cost = 0.0
    # at period 0 nothing has arrived yet, so nothing is known to be due next
    vhat = 0
    while not state.terminated:
        state1 = mdp.advance1(scenario, state)
        order_quantity, hold_back = policy.act(scenario, state1, vhat)
        state, info = mdp.advance2(
            scenario, state1,
            order_quantity=order_quantity, hold_back=hold_back,
        )
        total_cost += info["cost"]["total"]
        # what this period's arrivals contribute to *next* period's known
        # due-next total: orders that arrived now and are due two periods out
        vhat = info["demand_later"]
    return total_cost


def evaluate(
    scenario: AdiFlexScenario,
    make_policy: Callable[[], BenchmarkPolicy],
    n_seeds: int,
    progress_every: int = 2048,
) -> dict:
    """Seed loop over episode seeds 0..n_seeds-1 (spec section 9.2)."""
    costs = np.zeros(n_seeds)
    policy = make_policy()
    for ep_seed in range(n_seeds):
        if progress_every and ep_seed % progress_every == 0:
            print(f"  seed {ep_seed}/{n_seeds}", flush=True)
        costs[ep_seed] = rollout(scenario, policy, ep_seed)

    rewards = -costs
    mu = float(rewards.mean())
    n = len(rewards)
    return {
        "reward_mean": mu,
        "reward_var":  float(rewards.var(ddof=1)),
        "semivar_d":   float(np.sum(np.maximum(mu - rewards, 0.0) ** 2) / (n - 1)),
        "semivar_u":   float(np.sum(np.maximum(rewards - mu, 0.0) ** 2) / (n - 1)),
        "cost_mean":   float(costs.mean()),
    }


def format_row(scenario: AdiFlexScenario, stats: dict) -> str:
    return (
        f"{scenario.demand_now.mean():g}\t"
        f"{scenario.demand_next.mean():g}\t"
        f"{scenario.demand_later.mean():g}\t"
        f"{stats['reward_mean']:.6f}\t{stats['reward_var']:.6f}\t"
        f"{stats['semivar_d']:.6f}\t{stats['semivar_u']:.6f}"
    )


def write_tsv(outfile: Path, scenario: AdiFlexScenario, stats: dict) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(TSV_HEADER + "\n")
        f.flush()
        row = format_row(scenario, stats)
        print(row, flush=True)
        f.write(row + "\n")
        f.flush()
    print(f"wrote {outfile}")


def default_outfile(scenario_name: str, method: str) -> Path:
    """Spec section 9.6: results/{scenario}/benchmark/benchmark_{method}_eval_{scenario}.tsv"""
    return (
        Path(__file__).resolve().parent
        / "results" / scenario_name / "benchmark"
        / f"benchmark_{method}_eval_{scenario_name}.tsv"
    )


def solutions_path(scenario_name: str, method: str) -> Path:
    """Spec section 9.6: results/{scenario}/benchmark/{method}/{scenario}.txt"""
    return (
        Path(__file__).resolve().parent
        / "results" / scenario_name / "benchmark" / method
        / f"{scenario_name}.txt"
    )

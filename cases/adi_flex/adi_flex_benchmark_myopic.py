"""Myopic benchmark: single-period order-up-to, no hold-back.

The paper supplies no myopic ordering rule — its heuristics all inherit AP's
optimised policy — so this one is synthesised, and it is deliberately the
naive thing: minimise *this* period's expected cost and ignore everything
after it.

Concretely it drops the continuation term C_i(y) from the AP recurrence and
orders up to the minimiser of what remains,

    y* = argmin_y  h * E[(y - D)^+] + p * E[(d0 - y - v-hat)^+]

which is a newsvendor at the ratio p / (p + h) on total per-period demand. It
holds back nothing, so its allocation is myopic too — ship as early as stock
allows.

Its known weakness is the fixed ordering cost: with no lookahead there is
nothing to trade K off against, so it re-orders almost every period and pays K
almost every period. That is exactly the behaviour a good policy has to
improve on, which is what makes it a useful floor rather than a strawman.
"""

from __future__ import annotations

import argparse

import numpy as np

from adi_flex_scenarios import SCENARIOS, AdiFlexScenario
from adi_flex_mdp import AdiFlexState
from adi_flex_benchmark_ap import poisson_pmf, truncation_point


class MyopicPolicy:
    """Order up to the single-period newsvendor level; hold back nothing."""

    def __init__(self, scenario: AdiFlexScenario) -> None:
        l0 = scenario.demand_now.mean()
        total = scenario.total_rate
        h, p = scenario.holding_cost, scenario.backorder_cost

        k_total = truncation_point(total)
        k_now = truncation_point(l0)
        p_total = poisson_pmf(total, k_total)
        p_now = poisson_pmf(l0, k_now)
        sup_total = np.arange(k_total + 1)
        sup_now = np.arange(k_now + 1)

        # y* is v-hat dependent in principle; tabulate it over the plausible range
        self._target: dict[int, int] = {}
        for vhat in range(0, 21):
            best_val, best_y = None, 0
            for y in range(0, k_total + 1):
                val = (h * float(np.dot(p_total, np.maximum(y - sup_total, 0)))
                       + p * float(np.dot(p_now, np.maximum(sup_now - y - vhat, 0))))
                if best_val is None or val < best_val - 1e-12:
                    best_val, best_y = val, y
            self._target[vhat] = best_y

    def reset(self) -> None:
        pass

    def act(
        self,
        scenario: AdiFlexScenario,
        state: AdiFlexState,
        vhat: int,
    ) -> tuple[int, int]:
        u = state.inventory - state.due_now - state.due_next
        target = self._target[min(max(vhat, 0), 20)]
        return max(0, target - u), 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Myopic single-period benchmark.")
    p.add_argument("-s", "--scenario_name", type=str, default="exp4")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


if __name__ == "__main__":
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    pol = MyopicPolicy(scenario)
    print(f"scenario : {args.scenario_name}  ({scenario.desc})")
    print(f"order-up-to target by vhat: "
          f"{ {v: pol._target[v] for v in range(0, 6)} } ...")

"""PL benchmark: the paper's three protection-level heuristics.

Wang & Toktay (2008) section 4.2. Unlike AP these are *implementable* — they
run against the true dynamics, crossover and all — so they give upper bounds on
the optimal cost, and they are the real thing a learned policy has to beat.

All three share the same ordering rule (the AP relaxation's state-dependent
(s(v-hat), S(v-hat)) policy, read from the AP solutions file) and differ only in
how much stock they hold back before shipping against not-yet-due orders:

  PL(0)      hold back nothing — myopic allocation, ship as early as possible.
             Minimises holding cost while ignoring future shortages entirely.
  PL(sigma)  hold back the newsvendor quantity of eq (20), balancing the
             holding cost of the reserve against the shortage cost of not
             having it. The paper's best heuristic: 2.08% average optimality
             gap over its 540-instance experiment.
  PL(Sigma)  hold back enough to cover essentially the whole support of next
             period's urgent demand, which removes crossover outright at the
             cost of carrying a lot of stock.

Reusing AP's ordering policy for all three is the paper's own choice, and it
justifies it empirically: section 4.3.1 reports that PL(Sigma)'s independently
optimised ordering policy comes out "exactly the same" as AP's, supporting the
conjecture that ordering is insensitive to the allocation rule.

The protection level is a *constant* in all three — which is precisely the
opening the paper leaves and this domain exists to explore. Section 4.2 notes
eq (20) "could be further refined by incorporating the effect of the
fixed-ordering cost K, ..., inventory level x_i, or even the whole system
state" — i.e. exactly the state-dependent hold-back a learned policy can
express and these heuristics cannot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adi_flex_scenarios import SCENARIOS, AdiFlexScenario
from adi_flex_mdp import AdiFlexState
from adi_flex_benchmark_ap import (
    ApSolution,
    protection_level_max,
    protection_level_sigma,
    solve_ap,
)
from adi_flex_benchmark_common import solutions_path

POLICIES = ("pl0", "plsigma", "plmax")


class ProtectionLevelPolicy:
    """AP's (s, S) ordering rule plus a constant hold-back."""

    def __init__(self, solution: ApSolution, hold_back: int) -> None:
        self.solution = solution
        self.hold_back = hold_back

    def reset(self) -> None:
        pass

    def act(
        self,
        scenario: AdiFlexScenario,
        state: AdiFlexState,
        vhat: int,
    ) -> tuple[int, int]:
        # modified inventory position: net inventory less ALL outstanding
        # advance demand. With L=0 there is no pipeline term. This is the
        # statistic the paper's propositions make the policy a function of.
        u = state.inventory - state.due_now - state.due_next
        return self.solution.order_up_to(state.period, u, vhat), self.hold_back


def make_policy(
    scenario: AdiFlexScenario,
    policy: str,
    solution: ApSolution,
) -> ProtectionLevelPolicy:
    if policy == "pl0":
        hold_back = 0
    elif policy == "plsigma":
        hold_back = protection_level_sigma(scenario)
    elif policy == "plmax":
        hold_back = protection_level_max(scenario)
    else:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    return ProtectionLevelPolicy(solution=solution, hold_back=hold_back)


def load_or_solve(scenario_name: str, solutions: str | None) -> ApSolution:
    """Read the AP policy table, solving and caching it if absent."""
    path = Path(solutions) if solutions else solutions_path(scenario_name, "ap")
    if not path.exists():
        print(f"no AP solution at {path}; solving now", flush=True)
        sol = solve_ap(SCENARIOS[scenario_name])
        sol.write(path)
        return sol
    return ApSolution.read(path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Protection-level heuristics (PL).")
    p.add_argument("-s", "--scenario_name", type=str, default="exp4")
    p.add_argument("--policy", type=str, default="plsigma", choices=POLICIES)
    p.add_argument("--solutions", type=str, default=None,
                   help="AP policy table; defaults to the scenario's results path")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


if __name__ == "__main__":
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    solution = load_or_solve(args.scenario_name, args.solutions)
    print(f"scenario  : {args.scenario_name}  ({scenario.desc})")
    print(f"AP bound  : {solution.lower_bound:.4f}")
    for name in POLICIES:
        pol = make_policy(scenario, name, solution)
        print(f"  {name:8s} hold_back={pol.hold_back}")

"""The paper's LB heuristic policy for owmr (spec 9, role `feasible`).

Doğru, de Kok & van Houtum (2010) §3.4: implement the relaxed model's solution
with the non-negativity of allocations restored — every period order the
warehouse echelon position up to y0*, then allocate on-hand by the myopic
problem (eqs. 3-5) with a Lagrangian search. Feasible under the real
information set, so its cost is an upper bound on the optimum; together with
the lower bound it brackets it.

The policy reads the simulator's post-receipt state and emits actions in the
`order_ship` encoding (quantities), so the MDP's proportional scale-down never
fires. `owmr_benchmark_lb_heuristic_eval.py` scores it on the shared seed block.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import owmr_mdp as mdp
from owmr_benchmark_lb import BalanceBound, load_or_solve
from owmr_scenarios import OwmrScenario


class LbHeuristicPolicy:
    """order = max(0, y0* − IP_0); shipments = myopic allocation of on-hand."""

    def __init__(self, scenario: OwmrScenario, bound: BalanceBound, y0_star: float) -> None:
        self.scenario = scenario
        self.bound = bound
        self.y0 = float(y0_star)

    def decide(self, state: mdp.OwmrState) -> tuple[float, np.ndarray]:
        order = max(0.0, self.y0 - mdp.echelon_position(state))
        ip = np.asarray(mdp.inventory_position(state), dtype=float)
        ship = self.bound.allocate_myopic(float(max(0.0, state.wh_stock)), ip)
        return order, ship

    def act_fn(self):
        """Batched action callback for `owmr_eval_common.rollout_seeds`
        (action_mode = order_ship)."""
        def act(obs, envs):
            rows = []
            for e in envs:
                order, ship = self.decide(e._state)
                rows.append(np.concatenate([[order], ship]))
            return np.asarray(rows, dtype=np.float32)
        return act


def build(scenario_name: str, here: Path | None = None, outdir: str = "results") -> LbHeuristicPolicy:
    here = here or Path(__file__).resolve().parent
    bound, sol = load_or_solve(scenario_name, here, outdir)
    return LbHeuristicPolicy(bound.scenario, bound, sol["y0_star"])


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "base"
    pol = build(name)
    sc = pol.scenario
    state, _ = mdp.init_state(sc, 0)
    total = 0.0
    while not state.terminated:
        s1 = mdp.advance1(sc, state)
        order, ship = pol.decide(s1)
        state, info = mdp.advance2(sc, s1, order, list(ship))
        total += info["cost"]["total"]
    print(f"{name}: y0*={pol.y0:.3f}  one episode (seed 0) total cost {total:.2f}  per period {total / sc.horizon:.3f}")

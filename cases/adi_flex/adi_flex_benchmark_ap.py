"""AP relaxation for the ADI-flex heterogeneous branch — role `relaxed` (§9.9).

On a heterogeneous customer base demand arrives across several due-date classes
and crossover exists, so the allocation decision genuinely binds. The recursion
in `adi_flex_benchmark_common.py` costs early delivery as though that decision
could be POSTPONED until the due date — Wang & Toktay (2008) §4.1's
allocation-postponement relaxation. Dropping that constraint can only help, so
the value is a LOWER bound on optimal cost and is **not attainable** by any
policy the simulator can run.

Two consequences, both of which the IR's `role: relaxed` makes checkable:

- It must never be passed as `--baseline` (must-beat): asking a candidate to
  beat a bound it cannot beat is a category error, and `mdp_gates --ir` exits 2.
- A `feasible` arm — a PL heuristic, or the RL artifact — scoring strictly
  better than this bound indicts the eval, the bound, or the simulator, not the
  policy. `mdp_gates --ir` fails the gate on that, judged against a band.

Paired with the `feasible` arms this brackets the het optimum without an exact
solver, which is the whole reason the het branch can quote an optimality gap.
The stored y* table is also what the PL(0)/PL(sigma)/PL(Sigma) heuristics order
against (`adi_flex_benchmark_rule.py --dp-solutions`).

Requires a heterogeneous instance, and a reduced state at most one v-hat
component wide — `dim(V-hat) = T - L - 1 <= 1`, paper eq. (11). That covers
(T=2, L=0) and (T=3, L=1) alike; it is the DIMENSION that binds, not L. The
recursion carries the protection window of eq. (14): with a supply lead time
the order lands at i+L, so the loss is charged on the inventory at i+L+1 and
the two demand aggregates span i..i+L rather than the period, discounted by
alpha^L per eq. (12). At L=0 every one of those collapses to what the L=0
path always computed, which is asserted rather than assumed.

Example usage:
    python adi_flex_benchmark_ap.py -s het_exp4
"""

from __future__ import annotations

from adi_flex_benchmark_common import build_arg_parser, run_arm


def main() -> None:
    args = build_arg_parser(
        "AP-relaxation lower bound for the ADI-flex heterogeneous branch.",
        default_scenario="het_exp4",
        arm="ap",
    ).parse_args()
    run_arm("ap", args, homogeneous=False)


if __name__ == "__main__":
    main()

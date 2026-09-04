"""Exact DP for the ADI-flex homogeneous branch — benchmark role `exact` (§9.9).

On a homogeneous customer base every customer shares one demand lead time, so
there is a single due-date class and the allocation-postponement relaxation is
TIGHT: the backward induction in `adi_flex_benchmark_common.py` is then the
paper's own §3.2 recursion and its value is the optimum. Wang & Toktay (2008)
Prop 2 gives the optimal policy the state-dependent form (s(V), S(V)); the full
argmin table y*(i, u, vhat) is stored so evaluation never leans on that result.

Exact up to two stated numerics, which is what `basis` in the IR records:
Poisson tails truncated at mass < 1e-12 and renormalized, and the u-grid clamp
at U_MIN (below any state an optimal policy visits at these costs).

Requires L = 0 (the (u, vhat) reduction) and a homogeneous instance; running it
on a het_* instance is refused rather than silently relabelled.

Example usage:
    python adi_flex_benchmark_dp.py -s homog_L0_T2
"""

from __future__ import annotations

from adi_flex_benchmark_common import build_arg_parser, run_arm


def main() -> None:
    args = build_arg_parser(
        "Exact DP for the ADI-flex homogeneous L=0,T=2 case.",
        default_scenario="homog_L0_T2",
        arm="dp",
    ).parse_args()
    run_arm("dp", args, homogeneous=True)


if __name__ == "__main__":
    main()

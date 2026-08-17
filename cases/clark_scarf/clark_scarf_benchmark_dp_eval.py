"""Evaluate the exact Clark-Scarf DP benchmark (spec §9).

Replays the decomposition's critical-number table through ClarkScarfEnv — the
same wrapper RL trains on — over the same CRN seed block, so the RL arm can be
compared to it pairwise.

This arm is the paper's optimum (Theorems 1-2), subject to the one documented
approximation: the DP truncates the Poisson pmf at a high quantile while the
simulator does not (mass < 1e-9). Quote "% of optimal" against it with that
caveat attached.

Usage:
    python clark_scarf_benchmark_dp_eval.py -s n3_l2_p09 --n-seeds 8192
"""

from __future__ import annotations

from clark_scarf_benchmark_dp import ClarkScarfDP
from clark_scarf_eval_common import benchmark_main

if __name__ == "__main__":
    benchmark_main(
        "dp",
        ClarkScarfDP,
        "Evaluate the exact Clark-Scarf DP benchmark on clark_scarf.",
    )

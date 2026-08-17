"""Evaluate the echelon base-stock benchmark, no decomposition (spec §9).

Each echelon covers its own lead-time demand to a newsvendor fractile and
ignores the level below. The gap to the DP arm prices exactly what Clark &
Scarf's induced penalty (Theorem 2) buys.

Usage:
    python clark_scarf_benchmark_myopic_eval.py -s n3_l2_p09 --n-seeds 8192
"""

from __future__ import annotations

from clark_scarf_benchmark_echelon import EchelonBaseStock
from clark_scarf_eval_common import benchmark_main

if __name__ == "__main__":
    benchmark_main(
        "echelon_bs",
        EchelonBaseStock,
        "Evaluate the myopic base-stock benchmark on clark_scarf.",
    )

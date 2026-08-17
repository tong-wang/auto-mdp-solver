"""Evaluate the random benchmark (spec §9).

The floor of the leaderboard. Shares the CRN seed block and record format with
every other arm, so all differences are pairwise.

Usage:
    python clark_scarf_benchmark_random_eval.py -s n3_l2_p09 --n-seeds 8192
"""

from __future__ import annotations

from clark_scarf_benchmark_random import RandomShipper
from clark_scarf_eval_common import benchmark_main

if __name__ == "__main__":
    benchmark_main(
        "random",
        RandomShipper,
        "Evaluate the random benchmark on clark_scarf.",
    )

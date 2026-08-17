"""Evaluate the local (installation) base-stock benchmark (spec §9).

The traditional pre-Clark-Scarf rung: each installation runs its own
single-location base-stock on its LOCAL inventory position. The gap between this
and the echelon arm prices the echelon idea itself.

Usage:
    python clark_scarf_benchmark_local_eval.py -s n3_l2_p09 --n-seeds 8192
"""

from __future__ import annotations

from clark_scarf_benchmark_local import LocalBaseStock
from clark_scarf_eval_common import benchmark_main

if __name__ == "__main__":
    benchmark_main("local", LocalBaseStock,
                   "Evaluate the local base-stock benchmark on clark_scarf.")

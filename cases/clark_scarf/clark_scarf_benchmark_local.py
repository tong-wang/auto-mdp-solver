"""Local (installation) base-stock — the genuinely traditional benchmark.

This is what an inventory manager would do **without** Clark & Scarf. §1 of the
paper describes the prior literature as "devoted to the determination of optimal
purchasing quantities at a *single installation* faced with some pattern of
demand" — so the pre-1960 practice is to run each stocking point as its own
independent single-location problem.

Concretely, installation *k* watches its **own local inventory position**

    IP_k = on-hand at k  +  everything in transit toward k

and orders up to a local target `S_k`, set by a newsvendor over **its own lead
time** — `leadtime + 1` periods — at a common service fractile.

**The one thing it gets wrong is the one thing the paper is about.** Each level
covers only its *local* replenishment lead time, as if it were the last stop.
It has no notion that echelon k must actually cover its **cumulative** distance
to the customer, because it has no notion of echelon stock at all. The result is
independent safety stock at every level — the classic over-stocking that the
echelon formulation removes.

This is the honest naive rung. The arm above it
(`clark_scarf_benchmark_echelon.py`) is already theory-informed, and treating
*that* as the naive baseline would understate what the echelon idea is worth and
would unfairly penalize an RL agent that has to discover the structure from raw
stock.

Usage:
    python clark_scarf_benchmark_local.py -s n3_l2_p09
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clark_scarf_benchmark_dp import _poisson_pmf
from clark_scarf_mdp import inventory_position, ship_capacity
from clark_scarf_scenarios import SCENARIOS, ClarkScarfScenario


@dataclass(slots=True)
class LocalBaseStock:
    """Independent single-installation base-stock at every level."""

    scenario: ClarkScarfScenario
    ybar: np.ndarray = None      # (N,) local targets, stationary

    def __post_init__(self) -> None:
        sc = self.scenario
        mean = float(sc.demand.mean())
        b = sc.p_short + sc.h_install[0]
        # LOCAL lead time only — the level does not know it sits behind others
        vals, p = _poisson_pmf(mean * (sc.leadtime + 1))
        cdf = np.cumsum(p)
        targets = []
        for k in range(sc.n_echelons):
            ratio = b / (b + sc.h_install[k])
            targets.append(int(vals[int(np.searchsorted(cdf, ratio))]))
        self.ybar = np.asarray(targets, dtype=np.int64)

    def actions(self, state) -> np.ndarray:
        sc = self.scenario
        pos = inventory_position(state)          # per-level LOCAL position
        caps = ship_capacity(sc, state)
        q = np.zeros(sc.n_echelons)
        for k in range(sc.n_echelons):
            want = max(0.0, float(self.ybar[k]) - pos[k])
            q[k] = max(0.0, round(min(want, caps[k])))
        return q


def _cli() -> None:
    p = argparse.ArgumentParser(description="Solve the local base-stock targets.")
    p.add_argument("-s", "--scenario_name", default="n3_l2_p09",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    sc = SCENARIOS[args.scenario_name]
    pol = LocalBaseStock(sc)
    root = Path(__file__).resolve().parent
    out = root / args.outdir / args.scenario_name / "benchmark" / "local"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.scenario_name}.txt"
    np.savetxt(path, pol.ybar[None, :], fmt="%d",
               header=f"local installation base-stock targets S_k; "
                      f"N={sc.n_echelons} L={sc.leadtime} (stationary, LOCAL lead time)")
    print(f"[local] targets -> {path}")
    print(f"  S = {pol.ybar.tolist()}  (on LOCAL inventory position)")


if __name__ == "__main__":
    _cli()

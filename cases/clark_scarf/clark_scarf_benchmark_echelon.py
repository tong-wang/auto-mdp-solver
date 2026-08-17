"""Echelon base-stock WITHOUT the decomposition — a theory-informed benchmark.

**Read the name carefully. This is NOT a naive baseline.** It is handed the
paper's central insight — that the right state variable is *echelon* inventory
position, and that echelon k must cover its **cumulative** distance to the
customer — and lacks only Theorem 2's induced penalty. Nobody arrives at echelon
base-stock without Clark & Scarf's analysis; §1 of the paper describes the prior
literature as "devoted to the determination of optimal purchasing quantities at a
**single installation**". For the genuinely traditional rung, see
``clark_scarf_benchmark_local.py``.

So the ladder is:

    random  <  local base-stock  <  THIS  <  Clark-Scarf DP
              (pre-paper practice)  (echelon idea,   (+ induced
                                     no decomposition) penalty)

and the DP-minus-this gap prices Theorem 2 specifically, while
this-minus-local prices the echelon idea itself.

Each echelon covers its own lead-time demand to a newsvendor fractile and
ignores the level below entirely.

Echelon *i* is `i` links from the customer, so a unit released there takes
``i * L`` periods to become available, and its cost is then assessed at the end
of that period — the periodic-review "lead time **plus one**". Treat the echelon
as a standalone newsvendor over ``D ~ Poisson((i*L + 1) * mu)``, trading its own
cumulative holding rate ``H_i`` against the backorder penalty ``b``:

    ybar_i = quantile of Poisson((i*L + 1) * mu) at  b / (b + H_i)

Then run the same echelon base-stock policy the DP runs — ship up to the
target, clipped by what the level above holds. Only the *targets* differ.

**What it deliberately gets wrong.** It prices only its own holding-vs-shortage
trade-off. It never accounts for the extra cost it inflicts on the level below
by being short — the induced penalty ``Lambda`` of Eq. (10)/(25). So the
DP-minus-myopic gap is a direct price tag on the paper's central contribution:
everything Theorem 2 buys over treating each echelon in isolation.

Note this is *not* the same as deleting ``Lambda`` from the DP. Do that and an
upper echelon's cost becomes purely linear-increasing in ``y``, its target runs
off to minus infinity, and it stocks nothing at all — degenerate, not myopic.
Coverage-based targets are what a person would actually do.

Usage:
    python clark_scarf_benchmark_myopic.py -s n3_l2_p09
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clark_scarf_benchmark_dp import _poisson_pmf
from clark_scarf_mdp import echelon_position, echelon_stock
from clark_scarf_scenarios import SCENARIOS, ClarkScarfScenario


@dataclass(slots=True)
class EchelonBaseStock:
    """Per-echelon newsvendor targets from CUMULATIVE lead-time coverage."""

    scenario: ClarkScarfScenario
    ybar: np.ndarray = None      # (N,) — stationary, unlike the DP's (T, N)

    def __post_init__(self) -> None:
        sc = self.scenario
        mean = float(sc.demand.mean())
        b = sc.p_short + sc.h_install[0]
        targets = []
        for i in range(1, sc.n_echelons + 1):
            ratio = b / (b + sc.h_install[i - 1])
            vals, p = _poisson_pmf(mean * (i * sc.leadtime + 1))
            cdf = np.cumsum(p)
            targets.append(int(vals[int(np.searchsorted(cdf, ratio))]))
        # targets must not decrease up the chain: echelon i+1 contains echelon i,
        # so a lower target upstream is not implementable and would only starve
        # the level below
        self.ybar = np.maximum.accumulate(np.asarray(targets, dtype=np.int64))

    def actions(self, state) -> np.ndarray:
        """Same base-stock form as the DP; only the targets differ."""
        sc = self.scenario
        N = sc.n_echelons
        u = echelon_position(sc, state)
        x = echelon_stock(sc, state)
        q = np.zeros(N)
        for k in range(N):
            hi = x[k + 1] if k + 1 < N else float(sc.ship_max) + u[k]
            y = min(max(float(self.ybar[k]), u[k]), hi)
            q[k] = max(0.0, round(y - u[k]))
        return q


def _cli() -> None:
    p = argparse.ArgumentParser(description="Solve the myopic base-stock targets.")
    p.add_argument("-s", "--scenario_name", default="n3_l2_p09",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    sc = SCENARIOS[args.scenario_name]
    pol = EchelonBaseStock(sc)
    root = Path(__file__).resolve().parent
    out = root / args.outdir / args.scenario_name / "benchmark" / "echelon_bs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.scenario_name}.txt"
    np.savetxt(path, pol.ybar[None, :], fmt="%d",
               header=f"echelon base-stock targets (no decomposition) ybar[i]; N={sc.n_echelons} "
                      f"L={sc.leadtime} (stationary)")
    print(f"[echelon_bs] targets -> {path}")
    print(f"  ybar = {pol.ybar.tolist()}")


if __name__ == "__main__":
    _cli()

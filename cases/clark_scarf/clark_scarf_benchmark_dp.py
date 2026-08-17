"""Exact Clark–Scarf benchmark for the serial multi-echelon chain.

This is the paper's own solution (§2–§3, Theorems 1–2): the N-echelon problem
is solved as **N one-dimensional problems**, never one N-dimensional one.

Derivation for this domain's timing (A -> S -> D)
-------------------------------------------------
Write echelon *i*'s **position** ``u_i`` = everything at installations 1..i or
in transit toward them, and echelon *i*'s **stock** ``x_i`` = the same minus
what is still in transit *into* installation i (Assumption 3). Then:

* a shipment on link *i* raises ``u_i`` by ``q_i`` and leaves every other
  ``u_m`` untouched — so the decision is exactly ``y_i = u_i + q_i``;
* the availability constraint ``q_i <= on-hand at installation i+1`` is exactly
  **``y_i <= x_{i+1}``** — the paper's Eq. (14) constraint ``y <= x_2``;
* positions evolve as ``u_i(t+1) = y_i(t) - d_t``;
* and therefore the *decision-point* echelon stock is ``x_i(t) = y_i(t-L) - D_L``,
  demand over the L periods in transit. **Each decision is charged its
  consequence L periods later** — the paper's ``alpha^lambda`` term in Eq. (4).

Two different risk periods, and conflating them is a real bug (it cost this
module one wrong first draft, caught by ``clark_scarf_dp_exactness.py``):

* **cost** is assessed on the END-OF-PERIOD state, one further demand draw on
  from the decision point — so the newsvendor term runs on ``D_{L+1}``, the
  classic periodic-review "lead time **plus one**";
* **availability** binds at the DECISION point of period ``s+L``, so the
  induced-penalty term runs on ``D_L``.

Re-index the discounted total by the decision that determines it and the
one-period cost becomes, in echelon variables,

    sum_i h_i * x_i  +  b * max(0, -x_1),      b = p_short + H_1

which **separates additively across echelons**: echelon 1 contributes a convex
newsvendor term ``h_1*y + b*E[(D_{L+1} - y)^+]`` and echelon i>=2 contributes
the linear ``h_i*y``. The only coupling left is the constraint ``y_i <= x_{i+1}``.

Clark & Scarf's Theorem 1 is what lets that coupling be priced instead of
enumerated: solve echelon i **unconstrained**, get its critical number
``ybar_i``, and charge echelon i+1 the *induced penalty*

    Lambda_i(x) = C_i(min(x, ybar_i)) - C_i(ybar_i)

— the paper's Expression (10)/(25), the extra cost of echelon i being unable to
reach its target because the level above was short. Echelon i+1's own problem
is then a single-variable problem with ``Lambda_i`` added to its costs
(Eq. (26)), and the recursion repeats up the chain — exactly what §3 says
happens beyond two echelons.

The resulting policy is echelon base-stock, clipped by availability:

    y_i = median(u_i, ybar_i(t), x_{i+1})        i.e. the paper's min(xbar_n, x_2)

so the whole solution is an ``N x T`` table of critical numbers. Those numbers
are also what Stage 5 reads the trained policy back against.

Optimality is the paper's Theorems 1–2, not an assumption of this module —
and ``clark_scarf_test.py`` checks it against a brute-force DP over the full
joint state on small instances rather than taking it on faith.

Caveat, deliberate and documented: the Poisson pmf is truncated at a high
quantile and renormalized while the simulator is not truncated, so this is
optimal for a distribution differing from the simulated one by mass < 1e-9.

Usage:
    python clark_scarf_benchmark_dp.py -s n3_l2_p09
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clark_scarf_mdp import echelon_position, echelon_stock
from clark_scarf_scenarios import SCENARIOS, ClarkScarfScenario

BETA = 0.95
_TAIL = 1e-12


def _poisson_pmf(rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Truncated, renormalized Poisson pmf. Support 0..k where the tail < 1e-12."""
    k = 0
    logp, acc, vals = -rate, 0.0, []
    while True:
        vals.append(np.exp(logp))
        acc += vals[-1]
        if k > rate and 1.0 - acc < _TAIL:
            break
        k += 1
        logp += np.log(rate) - np.log(k)
        if k > 100000:  # pragma: no cover - guards a pathological rate
            break
    p = np.asarray(vals, dtype=np.float64)
    return np.arange(len(p)), p / p.sum()


@dataclass(slots=True)
class ClarkScarfDP:
    """Solve the chain by the Clark–Scarf decomposition.

    Why the induced penalty is load-bearing, not a refinement: drop the
    ``Lambda`` terms and an upper echelon's stage cost becomes purely
    ``h_i * y``, linear and increasing — its unconstrained minimizer runs off
    to the bottom of the grid and it never stocks anything at all. The penalty
    is the *only* thing giving an upper echelon a reason to hold inventory,
    since nothing else in its own costs rewards it. That is why the myopic
    baseline (``clark_scarf_benchmark_myopic.py``) sets targets from
    lead-time-demand coverage instead of by deleting the penalty.
    """

    scenario: ClarkScarfScenario
    grid_lo: int | None = None
    grid_hi: int | None = None

    # solved tables
    ybar: np.ndarray = None      # (T, N) critical numbers
    grid: np.ndarray = None

    def __post_init__(self) -> None:
        sc = self.scenario
        mean = float(sc.demand.mean())
        if self.grid_lo is None:
            self.grid_lo = int(-6 * mean - 10)
        if self.grid_hi is None:
            self.grid_hi = int((sc.n_echelons + sc.leadtime + 4) * 3 * mean + 20)
        self.grid = np.arange(self.grid_lo, self.grid_hi + 1)
        self._solve()

    # -- economics in echelon variables --------------------------------------

    def _echelon_rates(self) -> np.ndarray:
        """h_i = H_i - H_{i+1}: each echelon's own holding increment."""
        h = self.scenario.h_install
        return np.array(
            [h[i] - (h[i + 1] if i + 1 < len(h) else 0.0)
             for i in range(self.scenario.n_echelons)]
        )

    def _solve(self) -> None:
        sc = self.scenario
        N, T, L = sc.n_echelons, sc.horizon, sc.leadtime
        g = self.grid
        n_g = len(g)
        h_i = self._echelon_rates()
        b = sc.p_short + sc.h_install[0]     # effective echelon backorder penalty
        mean = float(sc.demand.mean())

        d_vals, d_p = _poisson_pmf(mean)                   # one period
        # availability binds at the DECISION point L periods on -> L draws
        DL_vals, DL_p = _poisson_pmf(mean * L)
        # cost is assessed END-OF-PERIOD, one draw further on -> L+1 draws
        DR_vals, DR_p = _poisson_pmf(mean * (L + 1))

        def shift_expect(f: np.ndarray, vals: np.ndarray, p: np.ndarray) -> np.ndarray:
            """E[f(y - X)] on the grid, clamping below the grid's floor.

            Clamping is an edge approximation only: the grid runs several mean
            demands below any state a sensible policy reaches.
            """
            out = np.zeros(n_g)
            for v, pv in zip(vals, p):
                idx = np.clip(np.arange(n_g) - int(v), 0, n_g - 1)
                out += pv * f[idx]
            return out

        # newsvendor term for echelon 1: h_1*y + b*E[(D_{L+1} - y)^+]
        short = np.zeros(n_g)
        for v, pv in zip(DR_vals, DR_p):
            short += pv * np.maximum(v - g, 0.0)
        stage1 = h_i[0] * (g - mean * (L + 1)) + b * short

        ybar = np.zeros((T, N), dtype=np.int64)
        # Lambda[i][s] — induced penalty echelon i imposes on echelon i+1 at epoch s
        Lam = [[np.zeros(n_g) for _ in range(T + L + 1)] for _ in range(N)]
        V_next = [np.zeros(n_g) for _ in range(N)]   # V_i^{s+1}

        for s in range(T - 1, -1, -1):
            V_cur = []
            for i in range(N):
                charged = (s + L) <= (T - 1)      # decision affects a period in-horizon
                if not charged:
                    stage = np.zeros(n_g)
                elif i == 0:
                    stage = stage1.copy()
                else:
                    # holding on the end-of-period state (L+1 draws); the
                    # induced penalty on the decision-point state (L draws)
                    stage = h_i[i] * (g - mean * (L + 1)) + shift_expect(
                        Lam[i - 1][s + L], DL_vals, DL_p
                    )

                C = (BETA**L) * stage + BETA * shift_expect(V_next[i], d_vals, d_p)

                j = int(np.argmin(C))
                ybar[s, i] = int(g[j])
                # V_i^s(u) = min over y >= u  ->  suffix minimum of C
                V_cur.append(np.minimum.accumulate(C[::-1])[::-1])
                # Lambda_i^s(x) = C(min(x, ybar)) - C(ybar); zero once x >= ybar
                lam = C - C[j]
                lam[j:] = 0.0
                Lam[i][s] = np.maximum(lam, 0.0)
            V_next = V_cur

        self.ybar = ybar

    # -- policy ---------------------------------------------------------------

    def actions(self, state) -> np.ndarray:
        """Shipments for the LIVE links: y_i = median(u_i, ybar_i(t), x_{i+1}).

        The clip to ``x_{i+1}`` is the paper's ``min(xbar_n, x_2)``: ship up to
        the target, but never more than the level above actually holds.
        """
        sc = self.scenario
        N = sc.n_echelons
        t = min(int(state.period), sc.horizon - 1)
        u = echelon_position(sc, state)
        x = echelon_stock(sc, state)
        q = np.zeros(N)
        for k in range(N):
            target = float(self.ybar[t, k])
            hi = x[k + 1] if k + 1 < N else float(sc.ship_max) + u[k]
            y = min(max(target, u[k]), hi)
            q[k] = max(0.0, round(y - u[k]))
        return q

    def critical_numbers(self) -> np.ndarray:
        """The (T, N) table of echelon critical numbers — the paper's xbar_n."""
        return self.ybar


def _cli() -> None:
    p = argparse.ArgumentParser(description="Solve the Clark-Scarf DP.")
    p.add_argument("-s", "--scenario_name", default="n3_l2_p09",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    sc = SCENARIOS[args.scenario_name]
    dp = ClarkScarfDP(sc)
    method = "dp"
    root = Path(__file__).resolve().parent
    out = root / args.outdir / args.scenario_name / "benchmark" / method
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.scenario_name}.txt"
    np.savetxt(path, dp.ybar, fmt="%d",
               header=f"echelon critical numbers ybar[t, i]; "
                      f"N={sc.n_echelons} L={sc.leadtime} T={sc.horizon}")
    print(f"[{method}] critical numbers -> {path}")
    print(f"  t=0   {dp.ybar[0].tolist()}")
    print(f"  t=T/2 {dp.ybar[sc.horizon // 2].tolist()}")
    print(f"  t=T-1 {dp.ybar[-1].tolist()}")


if __name__ == "__main__":
    _cli()

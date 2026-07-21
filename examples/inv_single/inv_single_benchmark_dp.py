"""Finite-horizon DP benchmark for single-echelon inventory.

Solves by backward induction and caches V / pi tables to
  results/<scenario_name>/benchmark/dp/
so repeated runs load from disk rather than recomputing.
Pass no_cache=True to force a full re-solve and overwrite the cache.

State variable:
  - inventory              when leadtime.max() == 0  (exact)
  - inventory position     when leadtime.max() >  0  (exact for deterministic
    (IP = inv + pipeline)  LT=1; approximation for LT>1 or stochastic LT)

Usage:
    dp = FiniteHorizonDP(scenario)
    order = dp.act(t, x)   # x = inventory (LT=0) or IP (LT>0)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inv_single_scenarios import InvSingleScenario


class FiniteHorizonDP:

    def __init__(
        self,
        scenario: InvSingleScenario,
        x_min: int | None = None,
        x_max: int | None = None,
        q_max: int | None = None,
        d_max: int | None = None,
        results_dir: str = "results",
        no_cache: bool = False,
    ) -> None:
        assert scenario.demand.is_discrete, (
            f"FiniteHorizonDP requires discrete demand; got {type(scenario.demand).__name__}."
        )
        assert scenario.scenario_name is not None, (
            "scenario.scenario_name must be set for cache file naming."
        )

        self.scenario = scenario
        T  = scenario.horizon
        LT = scenario.leadtime.max()

        span = int(np.ceil(scenario.demand.max())) * (LT + 2)
        self.x_min = x_min if x_min is not None else -span
        self.x_max = x_max if x_max is not None else  2 * span
        self.q_max = q_max if q_max is not None else  2 * span
        self.d_max = d_max if d_max is not None else int(np.ceil(
            scenario.demand.mean() + 6.0 * np.sqrt(max(1.0, scenario.demand.mean()))
        ))

        # Demand PMF, truncated and renormalized
        self.d_vals  = np.arange(0, self.d_max + 1, dtype=float)
        raw_probs    = np.array([scenario.demand.phi(d) for d in self.d_vals])
        self.d_probs = raw_probs / raw_probs.sum()

        self.x_grid = np.arange(self.x_min, self.x_max + 1)
        self.q_vals = np.arange(0, self.q_max + 1)
        self.n_x    = len(self.x_grid)

        self.V  = np.zeros((T + 1, self.n_x))
        self.pi = np.zeros((T,     self.n_x), dtype=np.int32)

        self._cache_dir = (
            Path(__file__).resolve().parent
            / results_dir
            / scenario.scenario_name
            / "benchmark"
            / "dp"
        )

        if not no_cache and self._cache_valid():
            self._load()
        else:
            self._solve()
            self._save()

    # ------------------------------------------------------------------
    # Grid helper
    # ------------------------------------------------------------------

    def _idx(self, x: np.ndarray) -> np.ndarray:
        """Map state values to grid indices, clamping at boundaries."""
        return np.clip(np.round(x).astype(int) - self.x_min, 0, self.n_x - 1)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _meta(self) -> dict:
        return {
            "version":       2,
            "scenario_name": self.scenario.scenario_name,
            "horizon":       self.scenario.horizon,
            "x_min":         self.x_min,
            "x_max":         self.x_max,
            "q_max":         self.q_max,
            "d_max":         self.d_max,
        }

    def _cache_valid(self) -> bool:
        meta_path = self._cache_dir / "meta.json"
        if not all((self._cache_dir / f).exists() for f in ("meta.json", "V.csv", "pi.csv")):
            return False
        with open(meta_path) as f:
            return json.load(f) == self._meta()

    def _save(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(self._cache_dir / "V.csv",  self.V,  delimiter=",", fmt="%.6f")
        np.savetxt(self._cache_dir / "pi.csv", self.pi, delimiter=",", fmt="%d")
        with open(self._cache_dir / "meta.json", "w") as f:
            json.dump(self._meta(), f, indent=2)
        print(f"[DP] saved  → {self._cache_dir}")

    def _load(self) -> None:
        T = self.scenario.horizon
        self.V  = np.loadtxt(self._cache_dir / "V.csv",  delimiter=",").reshape(T + 1, self.n_x)
        self.pi = np.loadtxt(self._cache_dir / "pi.csv", delimiter=",", dtype=np.int32).reshape(T, self.n_x)
        print(f"[DP] loaded ← {self._cache_dir}")

    # ------------------------------------------------------------------
    # Solver helpers
    # ------------------------------------------------------------------

    def _compute_L_period_phi(self, L: int) -> np.ndarray:
        """PMF of the sum of L i.i.d. demand periods via convolution.

        L=0 returns [1.0] (Poisson(0) = always 0), which makes g_0(x)
        degenerate to the direct holding/shortage cost — identical to the
        original single-period cost for LT=0.
        """
        if L == 0:
            return np.array([1.0])
        phi_L = self.d_probs.copy()
        for _ in range(L - 1):
            phi_L = np.convolve(phi_L, self.d_probs)
        return phi_L / phi_L.sum()

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def _solve(self) -> None:
        print(f"[DP] solving {self.scenario.scenario_name} ...")
        h  = self.scenario.holding_cost
        b  = self.scenario.shortage_cost
        c  = self.scenario.order_cost_linear
        K  = self.scenario.order_cost_fixed
        T  = self.scenario.horizon
        LT = self.scenario.leadtime.max()

        # Post-order state: y[i,j] = x_grid[i] + q_vals[j],  (n_x, n_q)
        y = self.x_grid[:, None] + self.q_vals[None, :]

        # Next state after single-period demand: next_x[i,j,k] = y[i,j] - d_vals[k],  (n_x, n_q, n_d)
        next_x = y[:, :, None] - self.d_vals[None, None, :]

        # Ordering cost: variable + fixed,  (n_q,)
        ord_cost = c * self.q_vals + K * (self.q_vals > 0).astype(float)

        # Next-state indices — precomputed once, reused every period,  (n_x, n_q, n_d)
        next_idx = self._idx(next_x)

        # g_L(x): expected holding/shortage cost from inventory x facing L demand periods.
        # For LT=0: phi_L = [1.0], so g_0(x) = h*x^+ + b*(-x)^+  (direct cost, same as before).
        # For LT>0: phi_L is the L-period demand convolution; cost is deferred to period t+LT.
        phi_L   = self._compute_L_period_phi(LT)
        dL_vals = np.arange(len(phi_L), dtype=float)
        gL_table = np.array([                                      # (n_x,)
            float(np.dot(phi_L, h * np.maximum(0.0, x - dL_vals)
                              + b * np.maximum(0.0, dL_vals - x)))
            for x in self.x_grid
        ])

        # exp_gL[i,j] = E_{d0}[ g_L(y[i,j] - d0) ],  (n_x, n_q)
        gL_next = gL_table[next_idx]                               # (n_x, n_q, n_d)
        exp_gL  = np.einsum('ijk,k->ij', gL_next, self.d_probs)   # (n_x, n_q)

        # Backward induction.
        # Orders placed at t >= T-LT arrive after the horizon → pi[t]=0, V[t]=0 (already init).
        for t in range(T - LT - 1, -1, -1):
            exp_cont = np.einsum(
                'ijk,k->ij', self.V[t + 1][next_idx], self.d_probs
            )                                                      # (n_x, n_q)

            total = ord_cost[None, :] + exp_gL + exp_cont         # (n_x, n_q)

            best       = np.argmin(total, axis=1)
            self.V[t]  = total[np.arange(self.n_x), best]
            self.pi[t] = self.q_vals[best]

        print(f"[DP] done.")

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def act(self, t: int, x: int) -> int:
        """Optimal order at period t with state x (inventory or IP)."""
        if t >= self.scenario.horizon:
            return 0
        return int(self.pi[t, self._idx(np.array([x]))[0]])

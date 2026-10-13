"""Exact dynamic programs for the censored newsvendor at a short horizon —
benchmark ``dp``, role ``exact`` (Jain, Rudi & Wang 2015, eq. 3).

Two policies (``--policy``), one per observation type that admits a
fixed-dimensional belief state (Proposition 2):

- ``full``   — belief (n, cum_demand): posterior Gamma(alpha + cum_demand,
  beta + n). The continuation value does not depend on today's stocking
  level (full observation learns the same whatever is stocked), so the
  optimal order IS the myopic newsvendor quantity (Proposition 1(b),
  y^F = y^M); the recursion here computes V^F for the record.
- ``timing`` — belief (n, S, T): posterior Gamma(alpha + S, beta + T). The
  stocking level now shapes what is learned, so the DP trades today's profit
  against the information a stock-out buys. Backward induction over the
  integer S lattice and a T grid of step ``1 / grid_m``; the stock-out
  branch integrates the y-th arrival epoch's law — T_y / (T_y + beta') ~
  Beta(y, alpha') under the posterior — over the grid's bins by exact CDF
  differences, and evaluates V_{n+1} at the bin's right edge. The played
  policy reads the table at the nearest T grid point.

Numerical basis (recorded in the IR's ``benchmarks[dp].basis``): demand and
orders truncated at the prior-predictive (1 - ``tail``) quantile; T on a
grid of step 1/grid_m (default 1/40); bin right-edge evaluation; nearest-grid
policy lookup. Exact for the full type up to the truncation; for the timing
type exact up to the T discretization, which the eval's paired comparison
against the myopic timing policy bounds empirically.

Cost: the timing recursion is O(N * S_max * T_pts * y_max * grid_m) — seconds
for the concentrated cells (prior mean 10, 1/beta <= 1), minutes to hours for
the widest cells of ``paper510`` (prior mean 50, 1/beta = 16). Needs scipy
for the vectorised incomplete beta; the eval wrapper replays a saved table
without it.

Tables are cached at ``results/{target}/benchmark/dp/{row}_{policy}.npz``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from cnv_bayes import gamma_poisson_quantile
from cnv_benchmark_myopic import myopic_full

POLICIES = ("timing", "full")


def _nb_pmf(d: np.ndarray, shape: float, rate: float) -> np.ndarray:
    """Posterior-predictive demand pmf: NB(r = shape, p = 1 / (rate + 1))."""
    p = 1.0 / (rate + 1.0)
    logpmf = (np.array([math.lgamma(k + shape) - math.lgamma(k + 1) for k in d])
              - math.lgamma(shape) + d * math.log(p) + shape * math.log(1.0 - p))
    return np.exp(logpmf)


def _expected_min(pmf: np.ndarray, y_max: int) -> np.ndarray:
    """E[min(D, y)] for y = 0..y_max from the pmf: sum_{k<y} P(D > k).

    Beyond the pmf's truncation P(D > k) is 0 (the tail mass is below the
    truncation tolerance), so the table is padded with zeros: a stocking
    level above the cap sells exactly what the cap sells. This is the case
    a late posterior reaches — its myopic level can exceed the cap derived
    from the prior predictive."""
    surv = 1.0 - np.cumsum(pmf)                     # P(D > k), k = 0..d_cap
    surv = np.maximum(surv, 0.0)
    if len(surv) < y_max:
        surv = np.concatenate([surv, np.zeros(y_max - len(surv))])
    emin = np.concatenate([[0.0], np.cumsum(surv[:y_max])])
    return emin


class DpTable:
    """A solved policy table for one scenario (the solutions artifact)."""

    def __init__(self, policy: str, orders: np.ndarray, values: np.ndarray, meta: dict):
        self.policy, self.orders, self.values, self.meta = policy, orders, values, meta

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, orders=self.orders, values=self.values,
                            policy=self.policy, **{k: np.asarray(v) for k, v in self.meta.items()})

    @classmethod
    def load(cls, path: Path) -> "DpTable":
        z = np.load(path, allow_pickle=False)
        meta = {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files
                if k not in ("orders", "values", "policy")}
        return cls(str(z["policy"]), z["orders"], z["values"], meta)

    def act(self, env, obs=None) -> int:
        st = env._state
        n = st.period
        if self.policy == "full":
            return myopic_full(env)                  # y^F = y^M (Proposition 1(b))
        S = min(st.cum_sales, self.orders.shape[1] - 1)
        j = int(round(st.cum_duration * self.meta["grid_m"]))
        j = min(j, self.orders.shape[2] - 1)
        return int(self.orders[n, S, j])


def solve_full(sc, tail: float = 1e-9) -> DpTable:
    """V^F_n(k) by backward induction over cumulative demand k."""
    N, r, c = sc.horizon, sc.price, sc.unit_cost
    a0, b0, ratio = sc.prior_shape, sc.prior_rate, sc.critical_ratio
    d_cap = min(int(sc.arrivals_buffer), _predictive_quantile(a0, b0, 1.0 - tail))
    d = np.arange(d_cap + 1)
    k_max = N * d_cap
    v_next = np.zeros(k_max + 1)
    values = np.zeros((N, k_max + 1))
    orders = np.zeros((N, k_max + 1), dtype=np.int64)
    for n in reversed(range(N)):
        v = np.zeros(k_max + 1)
        for k in range(min(k_max, n * d_cap) + 1):
            a, b = a0 + k, b0 + n
            pmf = _nb_pmf(d, a, b)
            y = gamma_poisson_quantile(a, b, ratio)
            emin = _expected_min(pmf, y)[y]
            cont = float(pmf @ v_next[k:k + d_cap + 1]) if k + d_cap <= k_max else \
                float(pmf[: k_max - k + 1] @ v_next[k:k_max + 1])
            v[k] = r * emin - c * y + cont
            orders[n, k] = y
        values[n] = v
        v_next = v
    return DpTable("full", orders, values, {"d_cap": d_cap, "tail": tail, "horizon": N})


def solve_timing(sc, grid_m: int = 40, tail: float = 1e-9, verbose: bool = True) -> DpTable:
    """V^T_n(S, T) by backward induction over the sales lattice x a T grid."""
    from scipy.special import betainc          # vectorised incomplete beta

    N, r, c = sc.horizon, sc.price, sc.unit_cost
    a0, b0 = sc.prior_shape, sc.prior_rate
    d_cap = min(int(sc.arrivals_buffer), _predictive_quantile(a0, b0, 1.0 - tail))
    y_max = d_cap
    s_max = N * y_max
    h = 1.0 / grid_m
    t_pts = N * grid_m + 1
    t_grid = np.arange(t_pts) * h
    d = np.arange(d_cap + 1)
    y = np.arange(y_max + 1)
    edges = np.arange(grid_m + 1) * h            # t bins over one period: [0, 1]

    v_next = np.zeros((s_max + 1, t_pts))
    values = np.zeros((N, s_max + 1, t_pts))
    orders = np.zeros((N, s_max + 1, t_pts), dtype=np.int64)
    for n in reversed(range(N)):
        v = np.zeros((s_max + 1, t_pts))
        for S in range(n * y_max + 1):
            a = a0 + S
            for j in range(n * grid_m + 1):
                b = b0 + t_grid[j]
                pmf = _nb_pmf(d, a, b)
                w = r * _expected_min(pmf, y_max) - c * y
                # no stock-out: D = d < y, T advances a full period (j + grid_m)
                vn = v_next[S:S + d_cap + 1, j + grid_m] if n < N - 1 else np.zeros(d_cap + 1)
                cont_ns = np.concatenate([[0.0], np.cumsum(pmf * vn)[:y_max]])
                # stock-out: y >= 1, T_y's CDF on the bin edges; y = 0 leaves T unchanged
                if n < N - 1:
                    x = edges[None, 1:] / (b + edges[None, 1:])          # (1, grid_m)
                    cdf = betainc(y[1:, None].astype(float), a, x)        # (y_max, grid_m)
                    mass = np.diff(np.concatenate([np.zeros((y_max, 1)), cdf], axis=1), axis=1)
                    v_so = v_next[S + 1:S + y_max + 1, j + 1:j + grid_m + 1]   # (y_max, grid_m)
                    cont_so = np.concatenate([[v_next[S, j]], (mass * v_so).sum(axis=1)])
                else:
                    cont_so = np.zeros(y_max + 1)
                total = w + cont_ns + cont_so
                k = int(np.argmax(total))
                orders[n, S, j] = k
                v[S, j] = total[k]
        values[n] = v
        v_next = v
        if verbose:
            print(f"  dp timing: period {n} solved (S <= {n * y_max}, T pts {n * grid_m + 1})", flush=True)
    return DpTable("timing", orders, values,
                   {"d_cap": d_cap, "y_max": y_max, "grid_m": grid_m, "tail": tail, "horizon": N})


def _predictive_quantile(shape: float, rate: float, q: float) -> int:
    return gamma_poisson_quantile(shape, rate, q)


def solve(sc, policy: str, **kw) -> DpTable:
    if policy == "full":
        return solve_full(sc, **{k: v for k, v in kw.items() if k in ("tail",)})
    if policy == "timing":
        return solve_timing(sc, **kw)
    raise ValueError(f"unknown dp policy {policy!r}; choose from {POLICIES}")


def table_path(domain_dir: Path, outdir: str, target: str, row_id: str, policy: str) -> Path:
    from cnv_grids import cell_slug
    return domain_dir / outdir / target / "benchmark" / "dp" / f"{cell_slug(row_id)}_{policy}.npz"


if __name__ == "__main__":
    import time

    from cnv_gym import CnvEnv
    from cnv_scenarios import SCENARIOS

    src = SCENARIOS["base"]
    for policy in POLICIES:
        t0 = time.time()
        table = solve(src, policy, verbose=False) if policy == "timing" else solve(src, policy)
        print(f"{policy}: solved in {time.time() - t0:.1f}s; V_0 at the prior = "
              f"{table.values[0, 0, 0] if policy == 'timing' else table.values[0, 0]:.4f}; "
              f"first-period order {table.orders[0, 0, 0] if policy == 'timing' else table.orders[0, 0]}")
        env = CnvEnv(scenario=src)
        totals = []
        for seed in range(200):
            env.reset(seed=seed)
            done, total = False, 0.0
            while not done:
                _, rew, done, _, _ = env.step(table.act(env, None))
                total += rew
            totals.append(total)
        print(f"   200-episode mean profit {np.mean(totals):.3f}")

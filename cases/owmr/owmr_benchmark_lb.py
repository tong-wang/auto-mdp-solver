"""The balance-assumption lower bound for owmr (spec 9, role `relaxed`).

Doğru, de Kok & van Houtum (2010) §3.3: relax the non-negativity of the
allocations (the *balance assumption*) and the problem decomposes. The optimal
relaxed policy is a warehouse echelon order-up-to level y0 with newsvendor
targets z_i for the retailers, and the relaxed system's optimum is a **lower
bound** on the true optimum — attained by no deployable policy. §3.4's *LB
heuristic* (order up to y0*, allocate myopically with non-negativity restored)
is the feasible sibling, implemented in `owmr_benchmark_lb_heuristic.py`.

**The record is the relaxed optimum on THIS problem's criterion** — the
T-period total from the mean-cover start (`owmr_mdp.init_state`) — computed
by `BalanceBound.finite_horizon_bound` (ESCALATION #E26). The paper's long-run
average per period is solved alongside for the paper check only: it is a
different criterion (no start state, no horizon), and the mean-cover start
over-stocks the warehouse for the first l0 periods, which that average never
charges (+7.4 per episode in `base`, paid by every policy).

Rendered in THIS domain's cost accounting (h0 on warehouse on-hand, h0 + h_i on
retailer on-hand, p_i on backlog, nothing on transit). The paper charges h0 on
retailer-bound transit too; the two differ by the constant h0 · Σ l_i μ_i per
period, so the fractiles are identical and only the level shifts.

One cycle, started by the order of period t, under the balance assumption:

    X  = y0 − D0(l0 periods)                      echelon stock at allocation
    w  = argmin_w  h0 (X − Σw) + Σ R_i(w_i)  s.t. Σw ≤ X   (w free in sign)
    R_i(w) = (h0 + h_i) E(w − D_i)^+ + p_i E(D_i − w)^+,   D_i over l_i + 1 periods
    LB(y0) = E_X[ h0 (X − Σw(X))^+ + Σ R_i(w_i(X)) ]      per cycle; y0* = argmin

The finite-horizon record sums the cycles the horizon charges from the start
state instead of taking the stationary average (see `finite_horizon_bound`).

With gamma demand every convolution is another gamma (same scale), so the
bound is exact to quadrature accuracy: `scipy.special.gammainc` /
`gammaincinv` supply the CDF and quantile that numpy's sampler does not.

Usage:
    python owmr_benchmark_lb.py -s base          # prints y0*, z*, the finite-horizon bound; writes the solution table
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, special

from owmr_scenarios import SCENARIOS, OwmrScenario

QUAD_POINTS = 4000   # midpoint rule in quantile space of D0


def gamma_cdf(x, shape, scale):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, special.gammainc(shape, np.maximum(x, 0.0) / scale), 0.0)


def gamma_ppf(q, shape, scale):
    return scale * special.gammaincinv(shape, np.clip(q, 0.0, 1.0))


def loss_above(w, shape, scale):
    """E(D − w)^+ for D ~ gamma(shape, scale): mean(1 − F_{k+1}(w)) − w(1 − F_k(w))."""
    w = np.asarray(w, dtype=float)
    mean = shape * scale
    pos = np.maximum(w, 0.0)
    val = mean * (1.0 - gamma_cdf(pos, shape + 1.0, scale)) - pos * (1.0 - gamma_cdf(pos, shape, scale))
    return np.where(w > 0, val, mean - w)   # w <= 0: E(D - w) = mean - w


@dataclass
class BalanceBound:
    """The relaxed model's solution for one scenario (identical scale across
    retailers, which gamma at one (mu, cv) per retailer with equal cv gives;
    unequal scales would need a numerical convolution and are refused)."""

    scenario: OwmrScenario

    def __post_init__(self) -> None:
        sc = self.scenario
        g = sc.demand
        assert hasattr(g, "shape") and hasattr(g, "scale"), (
            "the lower bound needs the gamma candidate (per-retailer shape/scale)")
        self.n = sc.n_retailers
        self.a = np.asarray(g.shape, dtype=float)          # per-period shape per retailer
        self.theta = np.asarray(g.scale, dtype=float)
        assert np.allclose(self.theta, self.theta[0]), (
            "unequal gamma scales across retailers: the system lead-time demand is "
            "no longer a gamma; numerical convolution not implemented")
        self.h0 = float(sc.h0)
        self.h = np.asarray(sc.h_rt, dtype=float)
        self.p = np.asarray(sc.p_rt, dtype=float)
        self.l0, self.lr = int(sc.l0), int(sc.l_rt)
        # retailer protection-interval demand: l_rt + 1 periods
        self.a_r = self.a * (self.lr + 1)
        # system demand over the warehouse lead time: gamma(l0 * sum a, theta)
        self.a_0 = self.l0 * float(self.a.sum())
        self.theta0 = float(self.theta[0])
        # unconstrained newsvendor targets (paper eq. 7)
        self.frac = (self.h0 + self.p) / (self.h0 + self.h + self.p)
        self.z = gamma_ppf(self.frac, self.a_r, self.theta)
        self.identical = bool(np.allclose(self.a, self.a[0]) and np.allclose(self.h, self.h[0])
                              and np.allclose(self.p, self.p[0]))

    # -- retailer cost pieces --------------------------------------------------

    def retailer_cost(self, w: np.ndarray) -> np.ndarray:
        """R_i(w_i): expected end-of-protection-interval holding + backorder cost."""
        w = np.asarray(w, dtype=float)
        mean = self.a_r * self.theta
        above = loss_above(w, self.a_r, self.theta)         # E(D - w)^+
        below = w - mean + above                            # E(w - D)^+
        return (self.h0 + self.h) * below + self.p * above

    def marginal(self, w: np.ndarray) -> np.ndarray:
        """d/dw of [R_i(w) − h0 w]: (h0+h+p) F(w) − (h0+p); increasing in w."""
        return (self.h0 + self.h + self.p) * gamma_cdf(w, self.a_r, self.theta) - (self.h0 + self.p)

    def allocate_balance(self, x: float) -> np.ndarray:
        """The relaxed allocation of echelon stock x (w may be negative)."""
        if x >= self.z.sum():
            return self.z.copy()
        if self.identical:
            return np.full(self.n, x / self.n)
        # equalize marginals at -lam with sum w = x, over w >= 0 first
        def total(lam):
            q = (self.h0 + self.p - lam) / (self.h0 + self.h + self.p)
            return gamma_ppf(np.clip(q, 0.0, 1.0), self.a_r, self.theta).sum() - x
        if x <= 0:  # every marginal saturates; split the deficit by mean demand
            return x * (self.a_r * self.theta) / (self.a_r * self.theta).sum()
        lam = optimize.brentq(total, 0.0, float((self.h0 + self.p).max()), xtol=1e-10)
        q = (self.h0 + self.p - lam) / (self.h0 + self.h + self.p)
        return gamma_ppf(np.clip(q, 0.0, 1.0), self.a_r, self.theta)

    def cycle_cost(self, x: float) -> float:
        w = self.allocate_balance(x)
        kept = max(0.0, x - float(w.sum()))
        return self.h0 * kept + float(self.retailer_cost(w).sum())

    # -- the bound -------------------------------------------------------------

    def _grid(self):
        u = (np.arange(QUAD_POINTS) + 0.5) / QUAD_POINTS
        return gamma_ppf(u, self.a_0, self.theta0)          # D0 quantiles

    def average_cost(self, y0: float) -> float:
        d0 = self._grid()
        return float(np.mean([self.cycle_cost(y0 - d) for d in d0]))

    def solve(self) -> dict:
        mean0 = self.a_0 * self.theta0
        hi = float(self.z.sum() + mean0 + 6.0 * np.sqrt(self.a_0) * self.theta0)
        res = optimize.minimize_scalar(self.average_cost, bounds=(0.0, hi), method="bounded",
                                       options={"xatol": 1e-4})
        y0 = float(res.x)
        per_period = float(res.fun)
        fh = self.finite_horizon_bound(y0, per_period)
        return {
            "y0_star": y0,
            "z_star": [float(v) for v in self.z],
            "horizon": self.scenario.horizon,
            "lb_finite_T": fh["total"],
            "lb_finite_components": {k: v for k, v in fh.items() if k != "total"},
            "lb_per_period": per_period,
            "lb_longrun_x_T": per_period * self.scenario.horizon,
            "transit_constant_per_period": float(self.h0 * (self.lr * self.a * self.theta).sum()),
            "note": ("lb_finite_T is the record: the relaxed system's optimum on this "
                     "problem's criterion (horizon periods from the mean-cover start, this "
                     "domain's accounting, no transit charge). lb_per_period is the paper's "
                     "long-run average per period, kept for the paper check only (their "
                     "figure = lb_per_period + transit_constant_per_period); it is a "
                     "different criterion, not a bound on the finite-horizon total "
                     "(ESCALATION #E26)."),
        }

    # -- the bound on the finite-horizon criterion (the record; ESCALATION #E26) --

    def _nv_cost(self, w, shape: float, h_i: float, p_i: float):
        """Newsvendor cost at position w against gamma(shape, theta) demand:
        (h0 + h_i) E(w − D)^+ + p_i E(D − w)^+, elementwise in w."""
        w = np.asarray(w, dtype=float)
        mean = shape * self.theta0
        above = loss_above(w, shape, self.theta0)
        return (self.h0 + h_i) * (w - mean + above) + p_i * above

    def cycle_cost_vec(self, x):
        """`cycle_cost` over an array of echelon stocks — identical retailers only,
        where the relaxed allocation is z* above sum z* and an equal split below."""
        assert self.identical, "the finite-horizon bound needs identical retailers (equal-split closed form)"
        x = np.asarray(x, dtype=float)
        zsum = float(self.z.sum())
        w = np.where(x >= zsum, float(self.z[0]), x / self.n)
        kept = np.maximum(0.0, x - zsum)
        return self.h0 * kept + self.n * self._nv_cost(w, float(self.a_r[0]), float(self.h[0]), float(self.p[0]))

    def average_cost_vec(self, y):
        """`average_cost` over an array of echelon order-up-to positions (same D0 grid)."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        d0 = self._grid()
        out = np.empty(len(y))
        for s in range(0, len(y), 256):
            blk = y[s:s + 256]
            out[s:s + 256] = self.cycle_cost_vec(blk[:, None] - d0[None, :]).mean(axis=1)
        return out

    def finite_horizon_bound(self, y0: float, lb_per_period: float) -> dict:
        """The relaxed system's optimum on THIS problem's criterion: the T-period
        total from the mean-cover start (`owmr_mdp.init_state`), in this domain's
        accounting. Exact (to quadrature) because the relaxed problem separates by
        cycle — the cycle started by period t's order charges h0 on what the
        warehouse keeps at t and the retailers' newsvendor cost at t + l_rt — and
        each cycle's minimum over every policy is known:

          start    periods k < l_rt charge the retailers' own start stock,
                   (k + 2) mu_i on hand against k + 1 periods of demand;
          fixed    cycles t < l0 see echelon stock X0 + t mu_sys − D(t periods),
                   fixed by the start state (no order has arrived yet);
          ordered  cycles t >= l0: the echelon position before ordering is at least
                   IP0 − D(t − l0 periods) under any policy (orders are >= 0), so the
                   cycle costs at least average_cost(max(y0*, IP0 − D)) and the
                   base-stock policy attains it; once that equals the long-run
                   minimum, the remaining cycles are counted at the minimum;
          end      the last l_rt cycles' retailer costs fall outside the horizon
                   and their warehouse term is >= 0.

        X0 = 2 mu_sys at the warehouse after the first receipt + (1 + l_rt) mu_i per
        retailer; IP0 = X0 + (l0 − 1) mu_sys. With mu_i = 1 everywhere, X0 = 20 and
        IP0 = 30 in every cell — above y0* in `base`, hence the start transient."""
        T = int(self.scenario.horizon)
        mu = self.a * self.theta
        mu_sys = float(mu.sum())
        a_sys = float(self.a.sum())
        u = (np.arange(QUAD_POINTS) + 0.5) / QUAD_POINTS

        def expect_sys(k: int, f):                     # E f(D), D = system demand over k periods
            d = np.zeros(1) if k == 0 else gamma_ppf(u, k * a_sys, self.theta0)
            return float(np.mean(f(d)))

        start = sum(float(self._nv_cost((k + 2) * mu[i], float(self.a[i]) * (k + 1),
                                        float(self.h[i]), float(self.p[i])))
                    for k in range(self.lr) for i in range(self.n))
        X0 = (3 + self.lr) * mu_sys
        IP0 = X0 + (self.l0 - 1) * mu_sys
        fixed = [expect_sys(t, lambda d, t=t: self.cycle_cost_vec(X0 + t * mu_sys - d))
                 for t in range(self.l0)]
        n_ordered = T - self.lr - self.l0               # cycles l0 .. T − 1 − l_rt
        ordered: list[float] = []
        for k in range(n_ordered):
            v = expect_sys(k, lambda d: self.average_cost_vec(np.maximum(y0, IP0 - d)))
            ordered.append(v)
            if v - lb_per_period < 1e-9:
                break
        rest = n_ordered - len(ordered)
        return {
            "total": start + sum(fixed) + sum(ordered) + rest * lb_per_period,
            "start_retailer_periods": start,
            "fixed_cycles": fixed,
            "ordered_cycles_exact": ordered,
            "ordered_cycles_at_longrun_min": rest,
            "X0": X0,
            "IP0": IP0,
        }

    # -- the feasible myopic allocation the LB heuristic uses ------------------

    def allocate_myopic(self, onhand: float, ip_before: np.ndarray) -> np.ndarray:
        """Shipments s_i >= 0 with sum s <= onhand minimizing the same myopic
        objective (paper eqs. 3-5): w_i = max(ip_i, w_i(lam)), lam >= 0 such that
        the request fits on-hand."""
        ip = np.asarray(ip_before, dtype=float)
        want = np.maximum(0.0, self.z - ip)
        if want.sum() <= onhand:
            return want
        def total(lam):
            q = (self.h0 + self.p - lam) / (self.h0 + self.h + self.p)
            w = gamma_ppf(np.clip(q, 0.0, 1.0), self.a_r, self.theta)
            return np.maximum(0.0, w - ip).sum() - onhand
        lam_hi = float((self.h0 + self.p).max())
        if total(lam_hi) > 0:
            # Even w = 0 everywhere asks for more than is on hand: the retailers'
            # backlogs alone exceed it. Below zero position a retailer's marginal
            # myopic cost is the constant -(h0 + p_i) (its demand CDF is 0 there),
            # so the optimum gives on-hand to backlogged retailers by descending
            # p_i, splitting ties in proportion to backlog. (Reached only on states
            # the heuristic's own trajectories never produce — e.g. a swap probe
            # running another policy's allocation — so no recorded number moves.)
            backlog = np.maximum(0.0, -ip)
            s = np.zeros_like(backlog)
            left = float(onhand)
            for pv in sorted(set(self.p.tolist()), reverse=True):
                grp = np.isclose(self.p, pv) & (backlog > 0)
                need = float(backlog[grp].sum())
                if need <= 0 or left <= 0:
                    continue
                take = min(left, need)
                s[grp] = backlog[grp] * (take / need)
                left -= take
            return s
        lam = optimize.brentq(total, 0.0, lam_hi, xtol=1e-9)
        q = (self.h0 + self.p - lam) / (self.h0 + self.h + self.p)
        w = gamma_ppf(np.clip(q, 0.0, 1.0), self.a_r, self.theta)
        s = np.maximum(0.0, w - ip)
        if s.sum() > onhand:           # bisection residue
            s *= onhand / s.sum()
        return s


def solution_path(here: Path, outdir: str, scenario_name: str) -> Path:
    return here / outdir / scenario_name / "benchmark" / "lb" / f"{scenario_name}.txt"


def load_or_solve(scenario_name: str, here: Path, outdir: str = "results") -> tuple[BalanceBound, dict]:
    sc = SCENARIOS[scenario_name]
    bound = BalanceBound(sc)
    path = solution_path(here, outdir, scenario_name)
    if path.exists():
        return bound, json.loads(path.read_text())
    sol = bound.solve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sol, indent=2) + "\n")
    return bound, sol


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Balance-assumption lower bound for owmr (relaxed).")
    p.add_argument("-s", "--scenario_name", type=str, default="base", choices=list(SCENARIOS))
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--force", action="store_true", help="re-solve even if the table exists")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    path = solution_path(here, args.outdir, args.scenario_name)
    if args.force and path.exists():
        path.unlink()
    bound, sol = load_or_solve(args.scenario_name, here, args.outdir)
    print(SCENARIOS[args.scenario_name])
    print(json.dumps(sol, indent=2))
    print(f"solution table -> {path}")


if __name__ == "__main__":
    main()

"""The paper's myopic policies (Jain, Rudi & Wang 2015, §5.3) — benchmark
``myopic``, role ``feasible``.

Each period the newsvendor updates her belief about lambda from what she
observed and stocks the single-period newsvendor quantity under that belief,
ignoring the effect of today's stocking level on tomorrow's learning. Three
policies, one per observation type (``--policy``):

- ``timing`` — conjugate update on (sales, stock-out time): posterior
  Gamma(alpha + S, beta + T) (eq. 7); order = the smallest y with
  P(D <= y) >= (r - c) / r under the negative-binomial predictive.
- ``full``   — conjugate update on the true demand: Gamma(alpha + sum D,
  beta + n). Under full observation the myopic quantity is also the
  optimal one (Proposition 1(b): y^F = y^M).
- ``event``  — no conjugate prior exists (§3): the posterior over lambda is
  carried on a grid and updated by the event likelihood (4) — Poisson pmf at
  the sales when no stock-out occurred, the Poisson tail P(D >= y) when one
  did. The predictive CDF is the posterior average of Poisson CDFs.

All three read only the paper's observation for their type off the env's
state / last info — never the hidden rate. ``timing`` and ``full`` read the
running totals the simulator carries; ``event`` keeps its own grid posterior
across the episode, so its factory is called once per episode.
"""

from __future__ import annotations

import math

import numpy as np

from cnv_bayes import gamma_cdf, gamma_poisson_quantile

POLICIES = ("timing", "full", "event")

# event policy: lambda grid resolution and prior tail covered
_EVENT_GRID_POINTS = 2000
_EVENT_PRIOR_TAIL = 1e-9


def myopic_timing(env, obs=None) -> int:
    sc, st = env._scenario_ep, env._state
    return gamma_poisson_quantile(sc.prior_shape + st.cum_sales,
                                  sc.prior_rate + st.cum_duration, sc.critical_ratio)


def myopic_full(env, obs=None) -> int:
    sc, st = env._scenario_ep, env._state
    return gamma_poisson_quantile(sc.prior_shape + st.cum_demand,
                                  sc.prior_rate + st.period, sc.critical_ratio)


class MyopicEvent:
    """Grid posterior over lambda under event observations (sales, stock-out flag).

    The grid spans [0, lambda_hi], lambda_hi the prior's (1 - tail) quantile
    found by bisection on the Gamma CDF; cell masses are exact prior CDF
    differences, so the singular shape < 1 prior is handled without a
    density evaluation at 0. The likelihood of each period's observation is
    applied to the masses; the myopic order is read off the posterior
    predictive CDF, a mass-weighted average of Poisson CDFs.
    """

    def __init__(self, env, n_points: int = _EVENT_GRID_POINTS, tail: float = _EVENT_PRIOR_TAIL):
        sc = env._scenario_ep
        self.shape, self.rate, self.ratio = sc.prior_shape, sc.prior_rate, sc.critical_ratio
        self.d_cap = int(sc.arrivals_buffer)
        lam_hi = self._prior_quantile(1.0 - tail)
        edges = np.linspace(0.0, lam_hi, n_points + 1)
        cdf = np.array([gamma_cdf(e, self.shape, self.rate) for e in edges])
        self.lam = 0.5 * (edges[1:] + edges[:-1])
        self.log_w = np.log(np.maximum(np.diff(cdf), 1e-300))
        # log Poisson pmf table: rows = grid points, cols = d in 0..d_cap
        d = np.arange(self.d_cap + 1)
        self.log_pmf = (d[None, :] * np.log(np.maximum(self.lam[:, None], 1e-300))
                        - self.lam[:, None] - np.array([math.lgamma(k + 1) for k in d])[None, :])
        self.cdf_table = np.cumsum(np.exp(self.log_pmf), axis=1)   # P(D <= d | lambda)
        self._seen_period = -1

    def _prior_quantile(self, q: float) -> float:
        lo, hi = 0.0, max(1.0, self.shape / self.rate)
        while gamma_cdf(hi, self.shape, self.rate) < q:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if gamma_cdf(mid, self.shape, self.rate) < q:
                lo = mid
            else:
                hi = mid
        return hi

    def _update(self, sales: int, stockout: int, order: int) -> None:
        if order == 0:
            return                                   # y = 0 observes nothing
        if stockout:                                 # D >= y: the Poisson tail
            tail = 1.0 - (self.cdf_table[:, order - 1] if order >= 1 else 0.0)
            self.log_w += np.log(np.maximum(tail, 1e-300))
        else:                                        # D = sales exactly
            self.log_w += self.log_pmf[:, sales]
        self.log_w -= self.log_w.max()

    def __call__(self, env, obs=None) -> int:
        st, info = env._state, env._info
        if st.period > 0 and st.period != self._seen_period:
            # absorb the last period's observation exactly once
            self._update(int(info["sales"]), int(info["stockout_event"]), int(info["order"]))
        self._seen_period = st.period
        w = np.exp(self.log_w)
        w /= w.sum()
        pred_cdf = w @ self.cdf_table                # P(D <= d) under the posterior
        return int(np.searchsorted(pred_cdf, self.ratio, side="left"))


def make_policy(policy: str):
    """Policy factory for ``cnv_benchmark_common.evaluate``: called with a
    probe env once per target; the returned callable is applied per step.
    The event policy keeps per-episode state, so it re-creates itself when
    the env's episode seed changes."""
    if policy == "timing":
        return lambda env_probe: myopic_timing
    if policy == "full":
        return lambda env_probe: myopic_full
    if policy == "event":
        def factory(env_probe):
            holder = {"seed": None, "pol": None}

            def act(env, obs=None) -> int:
                if holder["seed"] != env._episode_seed:
                    holder["seed"], holder["pol"] = env._episode_seed, MyopicEvent(env)
                return holder["pol"](env)
            return act
        return factory
    raise ValueError(f"unknown myopic policy {policy!r}; choose from {POLICIES}")


if __name__ == "__main__":
    from cnv_gym import CnvEnv
    from cnv_scenarios import SCENARIOS

    for name in ("base", "fig3_c10", "fig3_c15"):
        env = CnvEnv(scenario=SCENARIOS[name])
        line = []
        for policy in POLICIES:
            act = make_policy(policy)(env)
            totals = []
            for seed in range(50):
                env.reset(seed=seed)
                done, total = False, 0.0
                while not done:
                    _, r, done, _, _ = env.step(act(env, None))
                    total += r
                totals.append(total)
            line.append(f"{policy}={np.mean(totals):7.2f}")
        print(f"{name:10s} 50-episode mean profit: " + "  ".join(line))

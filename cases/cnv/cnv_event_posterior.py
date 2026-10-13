"""The exact posterior over the hidden rate under EVENT observations (sales and a stock-out
flag per period), carried on a grid — the belief the `event_belief_log` gym mode renders (#E107).

Under event observation no conjugate form exists (the paper's §3): an uncensored period
contributes the Poisson pmf at the observed sales, a censored one the Poisson tail at the placed
order, and the posterior is the prior times that product, with no fixed-size sufficient statistic
(the censored order LEVELS matter one by one). The benchmark's `MyopicEvent` carries the same
object on a LINEAR grid (2000 cells to the prior's 1 - 1e-9 quantile, ~0.15 apart), which is fine
for an integer quantile but coarse for a LOG posterior mean below a rate of 1 — the zero band. This
grid is LOG-SPACED (default 400 cells from 1e-4 to the same upper quantile, plus one cell for
[0, 1e-4], which holds 0.06% of the fig6a prior's mass), with EXACT prior cell masses (CDF
differences, so the singular shape < 1 prior needs no density at 0) and the geometric midpoint as
each cell's node. Tables are cached per (prior, cap), so a grid scenario pays once per cell.

Dependency: numpy and the dependency-free `cnv_bayes.gamma_cdf`; no scipy, so the domain's
import graph stays what the differential needs.
"""
from __future__ import annotations

import math

import numpy as np

from cnv_bayes import gamma_cdf

_TABLES: dict[tuple, tuple] = {}


def _prior_quantile(q: float, shape: float, rate: float) -> float:
    lo, hi = 0.0, max(1.0, shape / rate)
    while gamma_cdf(hi, shape, rate) < q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gamma_cdf(mid, shape, rate) < q:
            lo = mid
        else:
            hi = mid
    return hi


def _tables(shape: float, rate: float, d_cap: int, n_cells: int, lam_lo: float, tail: float):
    key = (float(shape), float(rate), int(d_cap), int(n_cells), float(lam_lo), float(tail))
    if key not in _TABLES:
        lam_hi = _prior_quantile(1.0 - tail, shape, rate)
        edges = np.concatenate([[0.0], np.exp(np.linspace(np.log(lam_lo), np.log(lam_hi), n_cells))])
        nodes = np.concatenate([[0.5 * lam_lo], np.sqrt(edges[1:-1] * edges[2:])])
        cdf = np.array([gamma_cdf(float(e), shape, rate) for e in edges])
        mass = np.maximum(np.diff(cdf), 1e-300)
        d = np.arange(d_cap + 1)
        lgam = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, d_cap + 1)))])        # log d!
        log_pmf = d[None, :] * np.log(nodes)[:, None] - nodes[:, None] - lgam[None, :]  # (cells, d_cap+1)
        pmf = np.exp(log_pmf)
        cdf_tab = np.cumsum(pmf, axis=1)                                                 # P(D <= d | node)
        sf = np.flip(np.cumsum(np.flip(pmf, 1), 1), 1)                                   # P(D >= d | node), summed from the top (tail-stable)
        _TABLES[key] = (nodes, np.log(mass), log_pmf, np.log(np.maximum(sf, 1e-300)), cdf_tab)
    return _TABLES[key]


class EventGridPosterior:
    """One episode's belief. `update` absorbs one period's event observation; `stats` returns
    (posterior mean, posterior CV); `myopic_order` the smallest y with predictive P(D <= y) >= q,
    the benchmark `myopic_event` rule on this grid."""

    def __init__(self, prior_shape: float, prior_rate: float, d_cap: int,
                 n_cells: int = 400, lam_lo: float = 1e-4, tail: float = 1e-9) -> None:
        self.lam, self.log_prior, self.log_pmf, self.log_sf, self.cdf_tab = _tables(prior_shape, prior_rate, d_cap, n_cells, lam_lo, tail)
        self.log_w = self.log_prior.copy()
        self._w = None

    def update(self, sales: int, stockout: int, order: int) -> None:
        if order <= 0:
            return                                    # y = 0 observes nothing
        self.log_w = self.log_w + (self.log_sf[:, order] if stockout else self.log_pmf[:, sales])
        self.log_w -= self.log_w.max()
        self._w = None

    @property
    def w(self) -> np.ndarray:
        if self._w is None:
            w = np.exp(self.log_w)
            self._w = w / w.sum()
        return self._w

    def stats(self) -> tuple[float, float]:
        w = self.w
        m = float(w @ self.lam)
        v = max(float(w @ (self.lam * self.lam)) - m * m, 0.0)
        return m, math.sqrt(v) / m

    def myopic_order(self, ratio: float) -> int:
        pred_cdf = self.w @ self.cdf_tab
        return int(np.searchsorted(pred_cdf, ratio, side="left"))

"""CNV expression builtins declared in ``cnv_schema.json`` (``mdp.expr_builtins``).

Pure functions over floats and ints, imported lazily by the IR interpreter on
first call and usable by the domain's own benchmarks. No numpy dependency so
the interpreter side stays exact and portable.
"""

from __future__ import annotations

import math


def poisson_quantile(rate: float, q: float) -> int:
    """Smallest integer y with P(Poisson(rate) <= y) >= q.

    The newsvendor quantity under a known rate at critical ratio ``q``; the
    paper's perfect-knowledge model P stocks this every period. ``rate <= 0``
    or ``q <= 0`` give 0; ``q >= 1`` is not a valid critical ratio.
    """
    if rate <= 0.0 or q <= 0.0:
        return 0
    if q >= 1.0:
        raise ValueError(f"critical ratio must be < 1, got {q}")
    pmf = math.exp(-rate)
    cdf = pmf
    y = 0
    while cdf < q:
        y += 1
        pmf *= rate / y
        cdf += pmf
    return y


def gamma_poisson_quantile(shape: float, rate: float, q: float) -> int:
    """Smallest integer y with P(D <= y) >= q where D is the predictive demand
    under a Gamma(shape, rate) posterior on the Poisson rate: negative
    binomial with r = shape and success probability p = 1 / (rate + 1).

    The myopic newsvendor quantity y^M of the paper (§2, §5.3).
    """
    if shape <= 0.0 or rate <= 0.0 or q <= 0.0:
        return 0
    if q >= 1.0:
        raise ValueError(f"critical ratio must be < 1, got {q}")
    p = 1.0 / (rate + 1.0)          # P(one more arrival before the period ends)
    pmf = (1.0 - p) ** shape        # P(D = 0)
    if pmf == 0.0:
        # P(D = 0) underflows for a large shape on a small rate (a belief no episode reaches, but a probe's
        # sweep can ask for): the recursion below would stay at 0 forever, so run it in log space. Only
        # inputs that used to hang take this branch; every other input keeps the exact path below.
        log_pmf, cdf, y = shape * math.log1p(-p), 0.0, 0
        while cdf < q:
            y += 1
            log_pmf += math.log((shape + y - 1.0) / y * p)
            cdf += math.exp(log_pmf)
        return y
    cdf = pmf
    y = 0
    while cdf < q:
        y += 1
        pmf *= (shape + y - 1.0) / y * p
        cdf += pmf
    return y


# ---------------------------------------------------------------------------
# Special functions the benchmarks need, dependency-free (Numerical Recipes
# forms). scipy is not part of the harness's install, and the differential
# side of the domain must stay importable without it.
# ---------------------------------------------------------------------------


def regularized_gamma_p(a: float, x: float) -> float:
    """P(a, x) = gamma(a, x) / Gamma(a): the CDF of Gamma(shape a, scale 1) at x."""
    if x <= 0.0:
        return 0.0
    if a <= 0.0:
        raise ValueError("shape must be positive")
    gln = math.lgamma(a)
    if x < a + 1.0:                      # series
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(10_000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return s * math.exp(-x + a * math.log(x) - gln)
    # continued fraction (Lentz)
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


def gamma_cdf(x: float, shape: float, rate: float) -> float:
    """CDF of Gamma(shape, rate) at x (shape-rate form, as the paper writes it)."""
    return regularized_gamma_p(shape, rate * x)


def regularized_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b): the CDF of Beta(a, b) at x."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 10_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def stockout_time_cdf(t: float, order: int, shape: float, rate: float) -> float:
    """P(T_y <= t): the y-th arrival epoch under a Gamma(shape, rate) posterior
    on the Poisson rate. T_y | lambda ~ Gamma(y, lambda), so T_y / (T_y + rate)
    ~ Beta(y, shape) and P(T_y <= t) = I_{t/(rate+t)}(y, shape). At t = 1 this
    is the posterior-predictive stock-out probability P(D >= y).
    """
    if order <= 0:
        return 1.0
    if t <= 0.0:
        return 0.0
    return regularized_beta(t / (rate + t), float(order), shape)

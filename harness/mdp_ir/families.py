"""Distribution-family registry: harness-owned knowledge about families.

The catalog model (IR_LAYERING_PLAN.md §10) removes the hand-authored
family read-API blocks from the IR: ``mean`` / ``max`` / ``is_discrete`` of a
slot's selected candidate are *derived* here from the distribution family and
its settings, composing through the latent hierarchy — a setting that is
itself a draw spec contributes its draw-family's mean/envelope. Example::

    poisson(rate ~ gamma(shape=9, scale=1/0.3))
    mean = E[rate] = 30
    max  = envelope(poisson, envelope(gamma)) = 70 + 4*sqrt(70)

which reproduces exactly the formula domains used to hand-write.

Sampling itself stays in ``interpreter._sample_family`` (plus the generic
numpy dispatch); this module is the *moments/envelope* side. Envelopes use a
4-sigma convention for unbounded families — they are space-sizing bounds,
not hard supports, matching what domain gym wrappers have always done.

The ``resolve(raw_value, attr)`` callback is supplied by the caller
(``layering``): it turns a raw setting value — literal, expr string over
scenario constants, or nested draw spec — into a number/list for the
requested attr (``"mean"`` or ``"max"``). A setting this module cannot
resolve (e.g. an expr over *state*, like a price-dependent rate) surfaces as
a :class:`FamilyError`; the caller reports it only if the structure actually
references the attribute (lazy derivation), with the ``read_api`` candidate
override as the escape hatch.
"""

from __future__ import annotations

from typing import Any, Callable

# resolve(raw_setting_value, attr) -> number | list, attr in {"mean", "max"}
Resolver = Callable[[Any, str], Any]

_SIGMAS = 4.0  # envelope convention for unbounded families


class FamilyError(ValueError):
    """A moment/envelope this registry cannot derive for a family/settings."""


_DISCRETE = frozenset({
    "categorical", "poisson", "bernoulli", "binomial", "geometric",
    "hypergeometric", "negative_binomial", "logseries", "zipf",
    "choice_without_replacement",
})
_CONTINUOUS = frozenset({
    "normal", "lognormal", "uniform", "gamma", "exponential", "beta",
    "chisquare", "f", "gumbel", "laplace", "logistic", "pareto", "power",
    "rayleigh", "standard_cauchy", "standard_exponential", "standard_gamma",
    "standard_normal", "standard_t", "triangular", "vonmises", "wald",
    "weibull", "normalized_uniform_weights",
})


def is_discrete(family: str) -> bool | None:
    """Integer-valued (True), continuous (False), or unknown (None)."""
    if family in _DISCRETE:
        return True
    if family in _CONTINUOUS:
        return False
    return None


def _num(resolved: Any, family: str, key: str, attr: str) -> float:
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        raise FamilyError(
            f"{family}.{key} resolved to non-numeric {resolved!r} for {attr}"
        )
    return float(resolved)


def _get(settings: dict, family: str, key: str, resolve: Resolver, attr: str) -> float:
    if key not in settings:
        raise FamilyError(f"{family} needs setting {key!r} to derive {attr}")
    return _num(resolve(settings[key], attr), family, key, attr)


def mean(family: str, settings: dict, resolve: Resolver) -> float:
    """E[X] of one draw, composed through latent settings via ``resolve``."""
    g = lambda key: _get(settings, family, key, resolve, "mean")  # noqa: E731
    if family == "deterministic":
        return g("value")
    if family == "categorical":
        vals = resolve(settings.get("values"), "mean")
        probs = resolve(settings.get("probabilities"), "mean")
        if isinstance(vals, list) and isinstance(probs, list):
            if len(vals) != len(probs):
                raise FamilyError("categorical values/probabilities length mismatch")
            return float(sum(float(v) * float(p) for v, p in zip(vals, probs)))
        if isinstance(vals, list):
            # latent weights: positions exchangeable => uniform in expectation
            return float(sum(float(v) for v in vals) / len(vals))
        # latent values: exchangeable positions => E[dot] = E[value] under any
        # independent weights
        return _num(vals, family, "values", "mean")
    if family == "poisson":
        return g("rate")
    if family == "normal":
        return g("mean")
    if family == "lognormal":
        m, s = g("mean"), g("sigma")
        return float(__import__("math").exp(m + s * s / 2.0))
    if family == "uniform":
        return (g("low") + g("high")) / 2.0
    if family == "gamma":
        return g("shape") * g("scale")
    if family == "exponential":
        return g("scale")
    if family == "beta":
        a, b = g("a"), g("b")
        return a / (a + b)
    if family == "bernoulli":
        (p,) = [resolve(v, "mean") for v in settings.values()]
        return _num(p, family, "p", "mean")
    if family == "binomial":
        return g("n") * g("p")
    if family == "geometric":
        return 1.0 / g("p")
    if family == "negative_binomial":
        n, p = g("n"), g("p")
        return n * (1.0 - p) / p
    if family == "choice_without_replacement":
        return (g("low") + g("high")) / 2.0
    if family == "normalized_uniform_weights":
        return 1.0 / g("size")
    raise FamilyError(f"no mean derivation for family {family!r}")


def min_value(family: str, settings: dict, resolve: Resolver) -> float:
    """Envelope lower bound of one draw (exact support min where bounded
    below, mean − 4·sd for unbounded-below families), composed through
    latent settings. Mirrors :func:`max_value`; leadtime-style interfaces
    read it (e.g. an R-O-D validity check needs ``leadtime.min``)."""
    g = lambda key: _get(settings, family, key, resolve, "min")  # noqa: E731
    if family == "deterministic":
        return g("value")
    if family == "categorical":
        vals = resolve(settings.get("values"), "min")
        if isinstance(vals, list):
            return float(min(float(v) for v in vals))
        return _num(vals, family, "values", "min")
    if family in ("poisson", "binomial", "bernoulli", "hypergeometric",
                  "negative_binomial", "beta", "gamma", "exponential",
                  "lognormal", "chisquare", "f", "rayleigh", "power", "wald",
                  "weibull", "standard_exponential", "standard_gamma",
                  "normalized_uniform_weights"):
        return 0.0  # support bounded below at 0
    if family in ("geometric", "logseries", "zipf"):
        return 1.0  # support starts at 1
    if family == "normal":
        return g("mean") - _SIGMAS * g("std")
    if family == "uniform":
        return g("low")
    if family == "choice_without_replacement":
        return g("low")
    raise FamilyError(f"no min derivation for family {family!r}")


def max_value(family: str, settings: dict, resolve: Resolver) -> float:
    """Envelope upper bound of one draw (exact support max where bounded,
    mean + 4·sd for unbounded families), composed through latent settings."""
    g = lambda key: _get(settings, family, key, resolve, "max")  # noqa: E731
    if family == "deterministic":
        return g("value")
    if family == "categorical":
        vals = resolve(settings.get("values"), "max")
        if isinstance(vals, list):
            return float(max(float(v) for v in vals))
        return _num(vals, family, "values", "max")
    if family == "poisson":
        lam = g("rate")
        return lam + _SIGMAS * lam ** 0.5
    if family == "normal":
        return g("mean") + _SIGMAS * g("std")
    if family == "lognormal":
        return float(__import__("math").exp(g("mean") + _SIGMAS * g("sigma")))
    if family == "uniform":
        return g("high")
    if family == "gamma":
        shape, scale = g("shape"), g("scale")
        return shape * scale + _SIGMAS * (shape ** 0.5) * scale
    if family == "exponential":
        return (1.0 + _SIGMAS) * g("scale")
    if family == "beta":
        return 1.0
    if family == "bernoulli":
        return 1.0
    if family == "binomial":
        return g("n")
    if family == "geometric":
        p = g("p")
        return 1.0 / p + _SIGMAS * (1.0 - p) ** 0.5 / p
    if family == "negative_binomial":
        n, p = g("n"), g("p")
        return n * (1.0 - p) / p + _SIGMAS * (n * (1.0 - p)) ** 0.5 / p
    if family == "choice_without_replacement":
        return g("high")
    if family == "normalized_uniform_weights":
        return 1.0
    raise FamilyError(f"no max derivation for family {family!r}")

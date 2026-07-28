"""Generic exogenous-generator runtime (IR_LAYERING_PLAN §10f, stage 3).

The catalog model derives a slot's read-API from :mod:`mdp_ir.families` and
samples any registered family in the interpreter; this module closes the
remaining gap on the *domain* side: a :class:`FamilyGenerator` is a
spec-§4.2-shaped generator (``sample`` / ``realize`` / ``mean`` / ``max`` /
``min`` / ``is_discrete`` / ``latent`` / ``source_id``) over **any**
registered distribution family, with the latent recipe nested in its
settings exactly as a catalog candidate declares it. A domain bridges it
into its own generator interface once per *slot*::

    # {domain}_uncertainty.py
    from mdp_ir.runtime import FamilyGenerator

    class FamilyDemand(FamilyGenerator, DemandGenerator):
        \"\"\"Any registered family as demand — new families need no code.\"\"\"

after which a freshly appended catalog candidate (a new family, a new latent
recipe) trains and differentially verifies with **zero new domain Python** —
``FamilyGenerator.from_ir`` rebuilds the generator from the resolved IR, so
the hand-written per-family classes remain only where they add domain value
(an exact ``phi()``, state-dependent settings).

Seed contract (spec §6.3, scheme v2 only): intrinsic draws key
``[(draw,) period, source_id, 1, episode_seed, seed_salt]``; a source's
episode latents key on the meta branch at the same ``source_id`` —
``[source_id, 0, episode_seed, seed_salt]`` — with draws executed in
settings-declaration order from one rng, bit-identical to the interpreter's
desugared ``ScenarioSampler``.

Sampling dispatch is shared: :func:`sample_family` is the single
family→numpy mapping, used both here and by ``interpreter._sample_family``,
so a family cannot drift between the twin's two sides.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from mdp_ir import families

SEED_SCHEME = "v2"

_INTRINSIC_BRANCH = 1
_META_BRANCH = 0

# numpy Generator scalar-distribution methods callable by name: a family not
# handled explicitly in `sample_family` is dispatched to `rng.<family>(...)`
# with its settings passed straight through as numpy's own keyword arguments
# (e.g. gamma -> {shape, scale}; binomial -> {n, p}; beta -> {a, b}). Curated
# once from numpy's univariate distributions, so a new distribution needs no
# code here — just name it in the IR. Array-valued draws (dirichlet,
# multinomial, multivariate_*) are intentionally excluded: this samples one
# scalar. Families with friendlier aliases (poisson.rate, normal.mean/std)
# are handled explicitly and take precedence over this generic path.
NUMPY_SCALAR_DISTS: frozenset[str] = frozenset({
    "beta", "binomial", "chisquare", "exponential", "f", "gamma",
    "geometric", "gumbel", "hypergeometric", "laplace", "logistic",
    "lognormal", "logseries", "negative_binomial", "noncentral_chisquare",
    "noncentral_f", "normal", "pareto", "poisson", "power", "rayleigh",
    "standard_cauchy", "standard_exponential", "standard_gamma",
    "standard_normal", "standard_t", "triangular", "uniform", "vonmises",
    "wald", "weibull", "zipf",
})


class SamplingContext(Protocol):
    """What ``sample()`` reads off the state (spec §4.1)."""

    period: int
    episode_seed: int
    seed_salt: int


# ---------------------------------------------------------------------------
# Seed-key helpers (spec §6.3, scheme v2 — the canonical template)
# ---------------------------------------------------------------------------


def intrinsic_key(
    source_id: int, ctx: SamplingContext, draw: int | None = None
) -> list[int]:
    """v2 intrinsic seed key: [(draw,) period, source_id, 1, episode_seed,
    seed_salt], leaf-first so the trailing word is always seed_salt (>= 1)."""
    key = [ctx.period, source_id, _INTRINSIC_BRANCH, ctx.episode_seed, ctx.seed_salt]
    if draw is not None:
        key.insert(0, draw)
    return key


def meta_key(substream_id: int, episode_seed: int, seed_salt: int) -> list[int]:
    """v2 meta seed key: [substream_id, 0, episode_seed, seed_salt] —
    for a source's episode latents ``substream_id`` IS the ``source_id``."""
    return [substream_id, _META_BRANCH, episode_seed, seed_salt]


# ---------------------------------------------------------------------------
# Family sampling — the one dispatch (shared with the interpreter)
# ---------------------------------------------------------------------------


def _concrete_setting(value: Any) -> Any:
    if isinstance(value, str):
        raise TypeError(
            f"unresolved expression setting {value!r}: resolve settings to "
            "numbers at construction (FamilyGenerator.from_ir does), or pass "
            "a `resolve` callback"
        )
    return value


def sample_family(
    rng: np.random.Generator, family: str, settings: dict, resolve=None
) -> Any:
    """Sample one value of ``family`` from ``rng``. ``resolve(value)`` turns
    a raw setting into a number/list — defaults to concrete-only (str raises);
    the interpreter passes its namespace evaluator. This is the single
    family→numpy mapping both twin sides draw through."""
    get = _concrete_setting if resolve is None else resolve
    if family == "categorical":
        vals = get(settings["values"])
        probs = get(settings["probabilities"])
        return int(rng.choice(np.asarray(vals), p=np.asarray(probs, dtype=float)))
    if family == "poisson":
        return int(rng.poisson(float(get(settings["rate"]))))
    if family == "normal":
        return float(rng.normal(float(get(settings["mean"])),
                                float(get(settings["std"]))))
    if family == "lognormal":
        # settings are the underlying normal's (mean, sigma)
        return float(rng.lognormal(float(get(settings["mean"])),
                                   float(get(settings["sigma"]))))
    if family == "uniform":
        return float(rng.uniform(float(get(settings["low"])),
                                 float(get(settings["high"]))))
    if family == "bernoulli":
        # keyed on `p` like every other aliased family. It cannot infer the
        # parameter from "whichever setting is the only one": a candidate that
        # owns a world latent carries it as an extra settings key (a draw spec
        # desugars to `{slot}_{setting}`), which is how a per-component `iid`
        # latent is consumed — `p: "payout_arm_p[int(arm)]"`.
        return int(rng.random() < float(get(settings["p"])))
    if family == "deterministic":
        # degenerate (Dirac): always the constant `value`, no rng draw —
        # mirrors a fixed generator (e.g. DeterministicLeadtime.sample returns
        # its value without consuming the stream)
        return get(settings["value"])
    # sampler-recipe families (scenario samplers, spec §5.2): list-valued
    if family == "choice_without_replacement":
        lo = int(get(settings["low"]))
        hi = int(get(settings["high"]))
        size = int(get(settings["size"]))
        return [int(v) for v in
                rng.choice(np.arange(lo, hi + 1), size=size, replace=False)]
    if family == "normalized_uniform_weights":
        size = int(get(settings["size"]))
        lo = float(get(settings.get("low", 1.0)))
        hi = float(get(settings.get("high", 10.0)))
        raw = rng.uniform(lo, hi, size=size)
        return [float(v) for v in raw / raw.sum()]
    if family == "iid":
        # `size` independent draws of the `of` base family, in order from this
        # rng — the latent-vector recipe (e.g. N bandit arm means). `of` is a
        # structural family name like `family` itself, never resolved as an
        # expression; the remaining settings are the base family's own.
        of = settings["of"]
        if not isinstance(of, str) or of == "iid":
            raise ValueError(f"iid setting 'of' must name a base family, got {of!r}")
        size = int(get(settings["size"]))
        sub = {k: v for k, v in settings.items() if k not in ("of", "size")}
        return [sample_family(rng, of, sub, resolve) for _ in range(size)]
    # any other numpy Generator scalar distribution, by name: settings map
    # straight to numpy's own parameters (see NUMPY_SCALAR_DISTS). `.item()`
    # surfaces numpy's native int/float, so discrete families coerce to int
    # and continuous to float.
    if family in NUMPY_SCALAR_DISTS:
        draw = getattr(rng, family)(**{k: get(v) for k, v in settings.items()})
        return draw.item() if hasattr(draw, "item") else draw
    raise NotImplementedError(f"distribution family {family!r}")


# ---------------------------------------------------------------------------
# FamilyGenerator
# ---------------------------------------------------------------------------


def is_draw_spec(value: Any) -> bool:
    """A settings value declaring the slot's world latent: {"draw": {...}}."""
    return isinstance(value, dict) and "draw" in value


class FamilyGenerator:
    """A spec-§4.2 generator over any registered distribution family.

    ``settings`` values are numbers/lists, or draw specs
    ``{"draw": {"family": ..., "settings": {...}}}`` for per-episode latents
    (the catalog candidate shape, settings pre-resolved to numbers).
    A latent-bearing instance realizes on the meta branch at its own
    ``source_id`` (draws in declaration order from one rng — bit-identical
    to the interpreter's desugared sampler) and returns the realized
    concrete generator; a concrete instance's ``realize`` returns itself.
    Moments (``mean``/``max``/``min``) derive from :mod:`mdp_ir.families`,
    composing through the latent hierarchy — the family-level envelopes gym
    wrappers size spaces from.
    """

    def __init__(
        self,
        family: str,
        settings: dict[str, Any],
        source_id: int = 0,
        *,
        is_discrete: bool | None = None,
    ) -> None:
        self.family = family
        self.settings = dict(settings)
        self.source_id = source_id
        self._draw_keys = [k for k, v in self.settings.items() if is_draw_spec(v)]
        self.latent = bool(self._draw_keys)
        if is_discrete is None:
            is_discrete = families.is_discrete(family)
            if is_discrete is None:
                raise ValueError(
                    f"family {family!r} is not in the registry; pass is_discrete="
                )
        self.is_discrete = is_discrete

    # -- realize / sample (the two seed branches) ---------------------------

    def realize(self, episode_seed: int, seed_salt: int) -> "FamilyGenerator":
        if not self.latent:
            return self
        rng = np.random.default_rng(np.random.SeedSequence(
            meta_key(self.source_id, episode_seed, seed_salt)))
        realized = dict(self.settings)
        for key in self._draw_keys:   # declaration order == sampler draw order
            d = self.settings[key]["draw"]
            realized[key] = sample_family(rng, d["family"], d.get("settings", {}))
        return type(self)(
            self.family, realized, self.source_id, is_discrete=self.is_discrete
        )

    def sample(self, ctx: SamplingContext):
        if self.latent:
            raise RuntimeError(
                "unrealized latent generator — realize(episode_seed, seed_salt) "
                "first (a scenario source does this per episode)"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence(intrinsic_key(self.source_id, ctx))
        )
        return sample_family(rng, self.family, self.settings)

    # -- derived read-API (family-level envelopes over the latent prior) ----

    def _resolve(self, raw: Any, attr: str) -> Any:
        if is_draw_spec(raw):
            d = raw["draw"]
            fn = {"mean": families.mean, "max": families.max_value,
                  "min": families.min_value}[attr]
            return fn(d["family"], d.get("settings", {}), self._resolve)
        if isinstance(raw, str):
            raise families.FamilyError(
                f"unresolved expression setting {raw!r}; FamilyGenerator "
                "settings must be numbers (from_ir resolves them)"
            )
        return raw

    def mean(self) -> float:
        return float(families.mean(self.family, self.settings, self._resolve))

    def max(self) -> float:
        return float(families.max_value(self.family, self.settings, self._resolve))

    def min(self) -> float:
        return float(families.min_value(self.family, self.settings, self._resolve))

    def __repr__(self) -> str:
        tag = " latent" if self.latent else ""
        return (
            f"{type(self).__name__}({self.family},{tag} "
            f"settings={self.settings}, source_id={self.source_id})"
        )

    # -- construction from a resolved IR ------------------------------------

    @classmethod
    def from_parts(
        cls,
        source,
        sampler,
        consts: dict[str, Any],
    ) -> "FamilyGenerator":
        """Build from a resolved ``UncertaintySource`` + its slot's desugared
        ``ScenarioSampler`` (or None) + merged constant values. The low-level
        constructor ``from_ir`` and mixture-component adapters share."""
        for st in source.stages:
            if st.sub_stream is not None or st.key_exprs:
                raise ValueError(
                    f"source {source.name!r}: sub-streams / keyed stages need a "
                    "hand-authored generator (FamilyGenerator keys the plain "
                    "v2 intrinsic template)"
                )
        draw_specs = {d.name: d.distribution for d in sampler.draws} if sampler else {}
        where = f"FamilyGenerator[{source.name}]"
        settings: dict[str, Any] = {}
        for key, raw in source.distribution.settings.items():
            if isinstance(raw, str) and raw in draw_specs:
                d = draw_specs[raw]
                settings[key] = {"draw": {
                    "family": d.family,
                    "settings": {
                        k: _const_eval(v, consts, f"{where}.{key}")
                        for k, v in d.settings.items()
                    },
                }}
            elif isinstance(raw, str):
                settings[key] = _const_eval(raw, consts, f"{where}.{key}")
            else:
                settings[key] = raw
        # the interpreter draws in the sampler's order; realize() draws in
        # settings order — require them to agree rather than silently differ
        if sampler:
            by_target = {v: k for k, v in source.distribution.settings.items()
                         if isinstance(v, str) and v in draw_specs}
            sampler_order = [by_target[d.name] for d in sampler.draws
                             if d.name in by_target]
            settings_order = [k for k in settings if is_draw_spec(settings[k])]
            if sampler_order != settings_order:
                raise ValueError(
                    f"{where}: sampler draw order {sampler_order} disagrees "
                    f"with settings declaration order {settings_order}"
                )
        return cls(
            source.distribution.family,
            settings,
            source.stream_id,
            is_discrete=source.is_discrete,
        )

    @classmethod
    def from_ir(cls, ir, slot: str, instance: str | None = None) -> "FamilyGenerator":
        """Build the ``slot``'s generator from a resolved ``MdpIR`` — the
        zero-code path: a freshly appended catalog candidate needs no domain
        class, this reconstructs family + settings + latent recipe from the
        resolved source, its sampler, and the (instance-merged) constants."""
        if ir.seed_scheme != "v2":
            raise ValueError(
                "FamilyGenerator implements the v2 seed template; v1 domains "
                "keep their hand-written generators (frozen legacy keys)"
            )
        if ir.mdp.entity_structure.entity_id_in_seed:
            raise ValueError(
                "entity-keyed sources need a hand-authored generator"
            )
        source = next(
            (s for s in ir.mdp.uncertainty_sources if s.name == slot), None
        )
        if source is None:
            raise KeyError(
                f"no uncertainty source {slot!r}; have "
                f"{[s.name for s in ir.mdp.uncertainty_sources]}"
            )
        consts = {c.name: c.value for c in ir.mdp.scenario.constants}
        if instance is not None:
            consts.update(ir.mdp.scenario.instances[instance])
        sampler = next(
            (s for s in ir.mdp.scenario.samplers
             if s.substream_id == source.stream_id
             and (not s.instances or (instance or "") in s.instances)),
            None,
        )
        return cls.from_parts(source, sampler, consts)


def _const_eval(value: Any, consts: dict[str, Any], where: str) -> Any:
    """Resolve a settings value against constants: literals pass, expr
    strings evaluate over constants only — a state/decision-dependent
    setting (legal in the IR) has no constant value, so it surfaces here as
    the hand-author hint."""
    if not isinstance(value, str):
        return value
    from mdp_ir.layering import LayeringError, _eval_expr

    try:
        return _eval_expr(value, dict(consts), where)
    except LayeringError as exc:
        raise ValueError(
            f"{where}: setting {value!r} does not resolve over scenario "
            f"constants ({exc}); state-dependent settings need a "
            "hand-authored generator"
        ) from exc

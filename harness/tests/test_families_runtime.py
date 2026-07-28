"""Distribution families: one dispatch, two consumers.

``mdp_ir.runtime.sample_family`` is the single family→numpy mapping shared by
the interpreter and by the domain-side ``FamilyGenerator``, so the twin's two
sides cannot drift. ``mdp_ir.families`` derives ``mean``/``min``/``max`` that
symbolic bounds resolve against. Both are registries: adding a family must
need no per-domain code, which is what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdp_ir import families
from mdp_ir.runtime import NUMPY_SCALAR_DISTS, sample_family

# runtime resolves settings with a 1-arg callable; families passes the
# attribute it is deriving as a second argument
IDENTITY = (lambda v: v)
ENVELOPE = (lambda v, _attr=None: v)


def draw(family: str, settings: dict, seed: int = 7):
    return sample_family(np.random.default_rng(seed), family, settings,
                         resolve=IDENTITY)


# -- generic numpy dispatch --------------------------------------------------


@pytest.mark.parametrize("family,kw", [
    ("gamma", {"shape": 9.0, "scale": 3.3333}),
    ("binomial", {"n": 20, "p": 0.3}),
    ("exponential", {"scale": 2.0}),
    ("negative_binomial", {"n": 5, "p": 0.4}),
])
def test_dispatch_reproduces_numpy_bit_for_bit(family, kw):
    assert family in NUMPY_SCALAR_DISTS
    want = getattr(np.random.default_rng(7), family)(**kw)
    want = want.item() if hasattr(want, "item") else want
    assert draw(family, dict(kw)) == want


def test_integer_valued_families_coerce_to_int():
    assert isinstance(draw("binomial", {"n": 5, "p": 0.5}, seed=1), int)


def test_settings_evaluate_as_expressions():
    """Settings are exprs over the namespace, so a latent can be composed in
    without the family knowing anything about it."""
    got = sample_family(np.random.default_rng(2), "gamma",
                        {"shape": "3 + 6", "scale": 3.3333},
                        resolve=lambda v: eval(v) if isinstance(v, str) else v)
    assert got == np.random.default_rng(2).gamma(shape=9, scale=3.3333)


def test_unknown_family_raises():
    with pytest.raises(NotImplementedError):
        draw("no_such_dist", {})


# -- the degenerate family ---------------------------------------------------


def test_deterministic_family_returns_its_value():
    assert draw("deterministic", {"value": 3.0}) == 3.0


def test_deterministic_family_consumes_no_randomness():
    """A fixed generator must not advance the stream, or swapping a constant
    lead time for a stochastic one would shift every later draw."""
    rng = np.random.default_rng(5)
    sample_family(rng, "deterministic", {"value": 2.0}, resolve=IDENTITY)
    untouched = np.random.default_rng(5)
    assert rng.random() == untouched.random()


@pytest.mark.parametrize("attr", ["mean", "min_value", "max_value"])
def test_deterministic_envelope_is_the_value(attr):
    assert getattr(families, attr)("deterministic", {"value": 4.0}, ENVELOPE) == 4.0


# -- the iid vector recipe ---------------------------------------------------


def test_iid_is_size_base_draws_in_order():
    """Bit-identical to drawing the base family `size` times from one rng —
    the sampler contract (draws in order), so a latent vector costs no new
    seed-grammar machinery."""
    rng = np.random.default_rng(7)
    want = [float(rng.beta(a=2.0, b=5.0)) for _ in range(3)]
    assert draw("iid", {"of": "beta", "size": 3, "a": 2.0, "b": 5.0}) == want


def test_iid_resolves_settings_but_never_of():
    """`of` is structural, like `family` itself: with an eval-based resolver,
    resolving it would raise NameError on the bare family name."""
    got = sample_family(np.random.default_rng(2), "iid",
                        {"of": "poisson", "size": "1 + 2", "rate": "2.0 * 5"},
                        resolve=lambda v: eval(v) if isinstance(v, str) else v)
    rng = np.random.default_rng(2)
    assert got == [int(rng.poisson(10.0)) for _ in range(3)]


def test_iid_components_keep_the_base_type():
    vals = draw("iid", {"of": "bernoulli", "size": 4, "p": 0.5})
    assert len(vals) == 4
    assert all(isinstance(v, int) and v in (0, 1) for v in vals)


def test_iid_requires_a_base_family_name():
    with pytest.raises(ValueError, match="of"):
        draw("iid", {"of": 3, "size": 2})
    with pytest.raises(families.FamilyError, match="of"):
        families.mean("iid", {"size": 2}, ENVELOPE)


def test_iid_moments_are_per_component():
    """The list-valued convention (cf. normalized_uniform_weights): mean/min/
    max describe one component, so symbolic bounds stay scalar."""
    settings = {"of": "beta", "size": 6, "a": 2.0, "b": 2.0}
    assert families.mean("iid", settings, ENVELOPE) == pytest.approx(0.5)
    assert families.min_value("iid", settings, ENVELOPE) == 0.0
    assert families.max_value("iid", settings, ENVELOPE) == 1.0


# -- parameters are read BY KEY, so a latent may sit beside them -------------


def test_a_family_reads_its_parameter_beside_a_latent_setting():
    """A candidate that owns a world latent carries it as an extra settings key
    (a draw spec desugars to ``{slot}_{setting}``), which is how an ``iid``
    latent is consumed: ``p: "payout_arm_p[int(arm)]"`` sits next to the
    resolved ``arm_p`` list. A family that inferred its parameter from
    "whichever setting is the only one" broke on exactly that shape."""
    latent = {"arm_p": [0.5] * 10}          # the desugared iid latent key
    assert draw("bernoulli", {"p": 1.0, **latent}, seed=3) == 1
    assert draw("bernoulli", {"p": 0.0, **latent}, seed=3) == 0
    # the value used is `p`, not the other key that happens to be present
    assert draw("bernoulli", {"p": 1.0}, seed=3) == \
        draw("bernoulli", {"p": 1.0, **latent}, seed=3)


def test_bernoulli_without_its_parameter_key_raises():
    """Strict: the parameter is named `p`. A missing one is a broken IR, not a
    silent fall back to whichever setting is present."""
    with pytest.raises(KeyError):
        draw("bernoulli", {"prob": 0.5})


def test_bernoulli_mean_ignores_a_latent_setting():
    """``families.mean`` carried the same inference, so symbolic bounds over a
    latent-bearing bernoulli candidate broke the same way."""
    assert families.mean(
        "bernoulli", {"p": 0.25, "arm_p": [0.5] * 10}, ENVELOPE) == 0.25


# -- derived envelopes -------------------------------------------------------


def test_categorical_envelope_spans_its_support():
    settings = {"values": [8, 10, 12], "probabilities": [0.25, 0.5, 0.25]}
    assert families.min_value("categorical", settings, ENVELOPE) == 8
    assert families.max_value("categorical", settings, ENVELOPE) == 12
    assert families.mean("categorical", settings, ENVELOPE) == pytest.approx(10.0)


def test_poisson_mean_is_its_rate():
    assert families.mean("poisson", {"rate": 30.0}, ENVELOPE) == pytest.approx(30.0)


def test_unbounded_family_max_exceeds_its_mean():
    """An unbounded support has no exact max, so the envelope is mean + 4·sd —
    what a symbolic bound needs in order to be finite."""
    settings = {"rate": 30.0}
    assert (families.max_value("poisson", settings, ENVELOPE)
            > families.mean("poisson", settings, ENVELOPE))

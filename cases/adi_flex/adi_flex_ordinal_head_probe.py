"""Acceptance gate for the ordinal order head (DESIGN_ORDINAL_HEAD.md §4).

The decode is untouched by the head, so the new failure surface is the
DISTRIBUTION — and nothing in conformance/laws/differential can see a policy
parameterization (the F30 limitation, third occurrence). This probe is the
executable gate: run it green before trusting any training.

    python adi_flex_ordinal_head_probe.py
"""

from __future__ import annotations

import numpy as np
import torch as th

from sb3_contrib.common.maskable.distributions import (
    MaskableMultiCategoricalDistribution,
)

from adi_flex_ordinal_head import (
    N_ORDER_PARAMS, TAU_MIN, TAU_SCALE,
    OrdinalMultiCategoricalDistribution, expand_order_logits,
)

N_ORD, N_SIG = 108, 9        # het_exp4: order 0..107, sigma 0..8


def _params(w: float, mu: float, tau: float) -> th.Tensor:
    """Invert the transforms to hit (w, mu, tau) exactly (clamped at edges)."""
    t = np.log(w / (1.0 - w))
    x = np.clip((mu - 1.0) / (N_ORD - 2), 1e-9, 1 - 1e-9)
    m = np.log(x / (1.0 - x))
    y = max((tau - TAU_MIN) / TAU_SCALE, 1e-12)
    s = np.log(np.expm1(y)) if y < 30 else y
    return th.tensor([[t, m, s]], dtype=th.float64)


def _reference(w: float, mu: float, tau: float) -> np.ndarray:
    return np.exp(_reference_log(w, mu, tau))


def _reference_log(w: float, mu: float, tau: float) -> np.ndarray:
    """Log-probabilities, computed stably (tail terms underflow in linear space)."""
    k = np.arange(1, N_ORD)
    quad = -((k - mu) ** 2) / (2 * tau * tau)
    log_z = quad.max() + np.log(np.exp(quad - quad.max()).sum())
    return np.concatenate([[np.log(w)], np.log1p(-w) + quad - log_z])


def g1_exactness() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = float(rng.uniform(0.01, 0.99))
        mu = float(rng.uniform(1.5, N_ORD - 2.5))
        tau = float(rng.uniform(TAU_MIN + 0.05, 60.0))
        probs = th.softmax(expand_order_logits(_params(w, mu, tau), N_ORD),
                           dim=1).numpy()[0]
        ref = _reference(w, mu, tau)
        assert np.abs(probs - ref).max() < 1e-9, (w, mu, tau)
        ent = -(probs * np.log(probs + 1e-300)).sum()
        d = OrdinalMultiCategoricalDistribution([N_ORD])
        d.proba_distribution(_params(w, mu, tau))
        assert abs(float(d.entropy()[0]) - ent) < 1e-6
        a = th.tensor([[int(rng.integers(0, N_ORD))]])
        ref_lp = _reference_log(w, mu, tau)[int(a[0, 0])]
        got_lp = float(d.log_prob(a)[0])
        # exact in log space, RELATIVE for deep-tail actions (both are huge
        # negative numbers there; the distributions agree, floats differ late)
        assert abs(got_lp - ref_lp) < 1e-6 * max(1.0, abs(ref_lp)), \
            (w, mu, tau, got_lp, ref_lp)
    print("G1 exactness (probs / entropy / log_prob vs reference)   PASS")


def g2_point_masses() -> None:
    d = OrdinalMultiCategoricalDistribution([N_ORD])
    d.proba_distribution(_params(1 - 1e-4, 50.0, 5.0))
    assert float(d.distributions[0].probs[0, 0]) >= 0.999
    worst = 1.0
    for k in range(1, N_ORD):
        d.proba_distribution(_params(1e-4, float(k), TAU_MIN))
        p = d.distributions[0].probs[0]
        assert int(p.argmax()) == k, k
        worst = min(worst, float(p[k]))
    assert worst >= 0.99, worst
    print(f"G2 point masses (P(0)>=0.999; argmax=k all k, min P {worst:.4f})  PASS")


def g3_gradient_separation() -> None:
    p = _params(0.4, 30.0, 3.0).requires_grad_(True)
    logits = expand_order_logits(p, N_ORD)
    g0 = th.autograd.grad(logits[0, 0], p, retain_graph=True)[0][0]
    assert g0[1] == 0 and g0[2] == 0, "logit(0) must not depend on mu/tau"
    gk = th.autograd.grad(logits[0, 30], p, retain_graph=True)[0][0]
    gj = th.autograd.grad(logits[0, 60], p)[0][0]
    assert abs(float(gk[0]) - float(gj[0])) < 1e-9, \
        "d logit(k>0)/dt must be the shared log(1-w) term"
    print("G3 trigger/magnitude gradient separation                 PASS")


def g4_masking() -> None:
    d = OrdinalMultiCategoricalDistribution([N_ORD])
    d.proba_distribution(_params(0.3, 40.0, 8.0))
    before = d.distributions[0].probs.clone()
    d.apply_masking([np.ones(N_ORD, dtype=bool)])
    assert th.allclose(d.distributions[0].probs, before, atol=1e-7)
    rng = np.random.default_rng(1)
    mask = rng.random(N_ORD) < 0.5
    mask[0] = True
    d.proba_distribution(_params(0.3, 40.0, 8.0))
    d.apply_masking([mask])
    p = d.distributions[0].probs[0].numpy()
    ref = before[0].numpy() * mask
    ref = ref / ref.sum()
    assert np.abs(p - ref).max() < 1e-6
    print("G4 masking (all-true no-op; partial renormalizes)        PASS")


def g5_sigma_passthrough() -> None:
    rng = np.random.default_rng(2)
    sig_logits = th.tensor(rng.normal(size=(4, N_SIG)))
    ordinal = OrdinalMultiCategoricalDistribution([N_ORD, N_SIG])
    plain = MaskableMultiCategoricalDistribution([N_ORD, N_SIG])
    op = _params(0.3, 40.0, 8.0).repeat(4, 1)
    ordinal.proba_distribution(th.cat([op, sig_logits], dim=1))
    plain.proba_distribution(
        th.cat([th.zeros(4, N_ORD, dtype=th.float64), sig_logits], dim=1))
    assert th.allclose(ordinal.distributions[1].probs,
                       plain.distributions[1].probs, atol=1e-9)
    net = ordinal.proba_distribution_net(latent_dim=16)
    assert net.out_features == N_ORDER_PARAMS + N_SIG
    print("G5 sigma-head passthrough + net width                    PASS")


if __name__ == "__main__":
    th.set_default_dtype(th.float64)
    g1_exactness()
    g2_point_masses()
    g3_gradient_separation()
    g4_masking()
    g5_sigma_passthrough()
    print("gate G1-G5: ALL PASS (G6 smoke = a short training run)")

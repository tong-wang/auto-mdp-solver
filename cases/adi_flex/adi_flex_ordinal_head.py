"""Hurdle-discretized-Gaussian ("ordinal") order head for `order_protection`
(design and gate: ESCALATION.md #E22; evidence base: #E21, the order-gap
dissection).

The categorical order head splits its gradient signal across 108 independent
logits with no notion of adjacency, which is why a ±1-unit magnitude error
(measured +1.18 of the het_exp4 gap) sits below PPO's resolution. This head
replaces the parameterization ONLY: the policy still emits a categorical
distribution over the same MultiDiscrete([order_max+1, sigma_max+1, ...])
space, but the ORDER head's 108 logits are induced from 3 numbers —

    w   = sigmoid(t)                       P(order = 0)   (the (s,S) trigger)
    mu  = 1 + (n_ord - 2) * sigmoid(m)     location on 1..n_ord-1
    tau = TAU_MIN + TAU_SCALE*softplus(s)  width, floored

    logit(0) = log(w)
    logit(k) = log(1-w) - (k-mu)^2/(2 tau^2) - logZ ,  k >= 1
               logZ = logsumexp_j [-(j-mu)^2/(2 tau^2)]

a HURDLE construction: a zero gate w over a discretized Gaussian on the
positive quantities. The positive branch is zero-truncated, so the gate
exclusively owns P(0) — there is no second route to zero through mu. (The
trigger keeps its own degree of freedom — near the (s,S) boundary the target
distribution is bimodal, "0 or ~S-u", which a plain unimodal head cannot
express.) The body's location gradient  d log pi / d mu ~ (a-mu)/tau^2 pools
every sample into one estimate instead of one category's logit — adjacency
pooling, the mechanism the flat head lacks. Gate and body are SEPARABLE
mechanisms: a domain without a genuine "do nothing" mass wants the body
without the gate (PLAYBOOK.md LV1 SCOPE (ii)).

The sigma head(s) stay plain categorical: only +0.10 at stake there, and its
residual is partly the terminal-boundary effect (#E14), where a smoothness
prior points the wrong way.

Everything downstream — log_prob, entropy, sampling, argmax, MASKING — is the
inherited MaskableMultiCategorical machinery over the expanded logits, so
MaskablePPO and the `a0` algorithm axis do not move. Gate:
`adi_flex_ordinal_head_probe.py` (run it before trusting any training).

At init (ortho gain 0.01, zero bias): w = 0.5, mu = mid-range, tau ~ 28. The
positive branch is smooth and wide (max/min ~ 6 over 1..107), but the head as
a whole starts nothing like a zero-init categorical: the gate owns P(0)
outright, so P(order = 0) begins at 0.5 against a flat categorical's 1/108
(54x), and the deterministic argmax at init is "order nothing" until t has
travelled ~log(107) ~ 4.7 logit-units. Inside this domain's scope the fixed
cost makes that a sane prior; on a domain without a "do nothing" mass it
makes every slow-learning configuration evaluate as the do-nothing constant
(measured downstream — PLAYBOOK.md LV1 SCOPE (ii)).
"""

from __future__ import annotations

import torch as th
from torch import nn
from torch.nn import functional as F

from sb3_contrib.common.maskable.distributions import (
    MaskableMultiCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

TAU_MIN = 0.25     # width floor: the head may approach but never reach a
                   # degenerate point mass (deterministic play is eval argmax)
TAU_SCALE = 40.0   # softplus scale: init tau ~ 28, tau = 0.5 at s ~ -5
N_ORDER_PARAMS = 3


def expand_order_logits(raw: th.Tensor, n_ord: int) -> th.Tensor:
    """(batch, 3) raw head outputs -> (batch, n_ord) exact log-probabilities."""
    t, m, s = raw[:, 0], raw[:, 1], raw[:, 2]
    log_w = F.logsigmoid(t)                    # log P(order = 0)
    log_nw = F.logsigmoid(-t)                  # log P(order > 0)
    mu = 1.0 + (n_ord - 2) * th.sigmoid(m)
    tau = TAU_MIN + TAU_SCALE * F.softplus(s)
    k = th.arange(1, n_ord, dtype=raw.dtype, device=raw.device)
    quad = -((k[None, :] - mu[:, None]) ** 2) / (2.0 * tau[:, None] ** 2)
    log_z = th.logsumexp(quad, dim=1, keepdim=True)
    return th.cat(
        [log_w[:, None], log_nw[:, None] + quad - log_z], dim=1)


class OrdinalMultiCategoricalDistribution(MaskableMultiCategoricalDistribution):
    """First head a hurdle-discretized-Gaussian ("ordinal"), remaining heads
    plain categorical.

    The action layer outputs `3 + sum(action_dims[1:])` numbers; this class
    expands them to the full `sum(action_dims)` logits and delegates to the
    parent, so masking / log_prob / entropy / mode are all inherited.
    """

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(
            latent_dim, N_ORDER_PARAMS + sum(self.action_dims[1:]))

    def proba_distribution(self, action_logits: th.Tensor):
        p = action_logits.view(-1, N_ORDER_PARAMS + sum(self.action_dims[1:]))
        order_logits = expand_order_logits(
            p[:, :N_ORDER_PARAMS], self.action_dims[0])
        return super().proba_distribution(
            th.cat([order_logits, p[:, N_ORDER_PARAMS:]], dim=1))


class OrdinalMaskablePolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy with the ordinal order head installed.

    `_build` swaps the distribution object before the parent constructs the
    action net and optimizer, so the swap is complete before any parameter
    exists; everything else (masking path, evaluate_actions, save/load) is the
    parent's. Saved models reference this class by module path, so eval /
    select must run where `adi_flex_ordinal_head` is importable (the domain
    dir, as always).
    """

    def _build(self, lr_schedule) -> None:
        self.action_dist = OrdinalMultiCategoricalDistribution(
            list(self.action_space.nvec))
        super()._build(lr_schedule)

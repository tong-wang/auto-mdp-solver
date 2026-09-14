"""Ordinal (mixture-at-zero) order head — ported from the `adi_flex` case
(`adi_flex_ordinal_head.py`), whose diagnosis is the same one measured here.

The plain categorical head spreads its gradient over `q_high + 1` independent
logits with **no notion of adjacency**, so learning that 14 is right teaches the
head nothing about 13 or 15. #E2/#E6 measured the consequence directly: the
learned policy emits a coarse staircase (10 distinct `q` over 28 states where
the optimum needs 16) and its order-up-to level jitters +-1, which on this
domain's razor-sharp cost surface is 2.7-6.3% per unit.

This head replaces the PARAMETERIZATION only. The policy still emits a
categorical over the same `Discrete(q_high + 1)` space, so the action encoding
-- the design axis -- does not move; what changes is the policy class, which is
the `a` axis (guide 13.1: "the policy/extractor code implements it -- a custom
class, not a constructor argument"). The logits are induced from three numbers:

    w   = sigmoid(t)                        P(order = 0)  -- the (s,S) trigger
    mu  = 1 + (n - 2) * sigmoid(m)          location on 1..n-1
    tau = TAU_MIN + TAU_SCALE * softplus(s) width, floored

    logit(0) = log w
    logit(k) = log(1 - w) - (k - mu)^2 / (2 tau^2) - logZ,   k >= 1

a point mass at zero mixed with a discretized Gaussian over the positive
quantities. The zero atom is exactly what the `hurdle` encoding tried to buy by
splitting the action space (#E6, worse on both targets) -- here it costs one
parameter instead of a second head, and keeps a single categorical so log_prob,
entropy, sampling and mode are all inherited unchanged.

The location gradient d log pi / d mu ~ (a - mu) / tau^2 pools EVERY sample into
one estimate of where the level is, instead of nudging one category's logit.
That is the property the flat head lacks and the one the diagnosis says is
missing.

Gate: `inv_single_test.py::test_ordinal_head_is_a_proper_distribution` and
`::test_ordinal_head_can_represent_the_optimal_policy`. Saved models reference
this class by module path, so eval/select must run where it is importable (the
domain dir, as always).

At init (SB3's ortho gain 0.01, zero bias) w = 0.5, mu = mid-range and tau is
large, so the induced distribution is near-uniform -- the same exploration a
zero-init categorical starts from.
"""

from __future__ import annotations

import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

TAU_MIN = 0.25      # width floor: never a degenerate point mass
TAU_SCALE = 40.0    # softplus scale: init tau is wide, tau = 0.5 at s ~ -5
N_ORDER_PARAMS = 3


def expand_order_logits(params: th.Tensor, n: int) -> th.Tensor:
    """(batch, 3) raw params -> (batch, n) exact log-probabilities."""
    t, m, s = params[:, 0], params[:, 1], params[:, 2]
    log_w = F.logsigmoid(t)                     # log P(order = 0)
    log_nw = F.logsigmoid(-t)                   # log P(order > 0)
    mu = 1.0 + (n - 2) * th.sigmoid(m)
    tau = TAU_MIN + TAU_SCALE * F.softplus(s)
    k = th.arange(1, n, dtype=params.dtype, device=params.device)
    quad = -((k[None, :] - mu[:, None]) ** 2) / (2.0 * tau[:, None] ** 2)
    log_z = th.logsumexp(quad, dim=1, keepdim=True)
    return th.cat([log_w[:, None], log_nw[:, None] + quad - log_z], dim=1)


class OrdinalCategoricalDistribution(CategoricalDistribution):
    """Categorical over `n` order quantities whose logits are induced from 3
    numbers. Everything but the parameterization is the parent's."""

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, N_ORDER_PARAMS)

    def proba_distribution(self, action_logits: th.Tensor):
        p = action_logits.view(-1, N_ORDER_PARAMS)
        return super().proba_distribution(expand_order_logits(p, self.action_dim))


class OrdinalPolicy(ActorCriticPolicy):
    """`ActorCriticPolicy` with the ordinal order head installed.

    `_build` swaps the distribution before the parent constructs the action net
    and the optimizer, so the swap is complete before any parameter exists.
    """

    def _build(self, lr_schedule) -> None:
        self.action_dist = OrdinalCategoricalDistribution(int(self.action_space.n))
        super()._build(lr_schedule)

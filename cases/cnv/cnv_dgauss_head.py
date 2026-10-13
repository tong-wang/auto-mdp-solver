"""Discretized-Gaussian head for the `order_discrete` action (Discrete(403)).

WHY, on this domain (#E44). The order distribution under a diffuse prior is
heavy-tailed: at `fig6a` (prior shape alpha = 0.625) the myopic order has median
5, p99 64 and max 107, so the orders that carry the profit occur in ~2% of
states. A flat categorical must learn class 80 from the 2% of samples where 80
is correct, and it does not: a supervised clone of the myopic rule matches it
96.5% of the time and still plays at 328 against 793, under-ordering by ~7 units
exactly on that tail. This head replaces 403 logits with two numbers:

    mu  = location over bins 0..n-1
    tau = tau_min + tau_scale * softplus(s)     per-state width, floored

    logit(k) = -(k - mu)^2 / (2 tau^2) - logZ,  k = 0..n-1

The location gradient d log pi / d mu ~ (a - mu) / tau^2 pools EVERY sample into
one estimate of where the order should sit, so a large order is an extrapolation
of a smooth function rather than a rarely-visited class. That pooling is the one
mechanism the categorical lacks and the whole reason for the swap
(`clark_scarf_dgauss_head.py`, whose construction and hard-won defaults this
file follows).

THE ADAPTATION this domain needs: a LOG location, `mu_param="log"` (the default).
It carries NO knowledge of the solution — the head starts FLAT, and the only
problem information in it is the PRIOR, which the agent holds before it acts
(human's instruction, 2026-09-25: "do not inject your solution into the head,
start with flat; at most, use the prior info at the very beginning").

clark_scarf's affine location spans the support, mu = (n-1)/2 * (1 + m), which
over Discrete(403) is ~201 order-units per unit of the raw output — far too
coarse where almost all the mass lives (median order 5). The fix is to make the
location multiplicative:

    mu = exp(log_center + LOG_SPAN * m)        (mu = exp(log_center) at m = 0)

so d mu / d m = LOG_SPAN * mu: the step scales with the current location, giving
CONSTANT RELATIVE resolution. That is what a target spanning two orders of
magnitude needs, and it reaches the tail by extrapolating a smooth function
rather than by learning a rare class.

`log_center` is NOT tuned to any measured optimum. It defaults to 0 (mu = 1 at
m = 0); the train script passes **log(E[lambda]) of the scenario's own prior** —
information the agent has before acting, which the *_ctx modes already render.

THE INIT IS FLAT, BY THE WIDTH. `tau_scale` is set so that at raw = 0 the width
spans the support and the distribution is near-uniform over all 403 levels (the
max/min ratio is printed by this file's smoke, measured, never asserted). The
location only starts to matter once the net has narrowed tau — i.e. once it has
learned something. An undertrained policy is a near-uniform orderer, not one
pre-loaded with an answer.

WHY NOT ANCHOR ON THE POSTERIOR MEAN (the first design here, discarded — a real
bug, kept in the record). The obvious move is mu = (post_shape / post_rate) *
multiplier, reading the anchor out of the observation. It fails twice:
  * `norm_obs` is ON in the derivation (§8.3), so the policy sees STANDARDISED
    features — the ratio of the two posterior columns is not the posterior mean,
    and a smoke run built this way scored -15.5 at `base`, below the 25.72
    do-nothing floor;
  * the *_raw modes carry NO posterior at all (their vector is the prior's
    parameters plus the last period's observation; reconstructing the statistic
    from the frame stack is the whole point of the mode), so there is nothing to
    anchor on there even in principle.
The log location needs neither the posterior nor unnormalised inputs, so it
works identically for the belief and the raw modes.

tau is floored in ORDER UNITS, which here equals bins (the encoding is the
identity over integer stocking levels); a future binned action mode would break
that equivalence.

NO ZERO ATOM, deliberately, per clark_scarf's scope clause: zero-inflate only
when the problem has a genuine "do nothing" mass from a fixed cost / (s,S)
trigger. cnv's cost is per-unit linear with no setup, so zero is just the
boundary bin. It is not an idle bin — at the two highest-cost grid cells
(c = 1.7, 1.8 at the most diffuse prior) the myopic policy stocks nothing and
earns exactly 0.000 — but a boundary mode is what a half-Gaussian at 0 IS, and
those cells are the ones to watch for a regression.

Saved models reference this class by module path: eval / select must run where
`cnv_dgauss_head` is importable (it lives beside the train script, so running
from the domain directory per spec 8.4 is enough).
"""

from __future__ import annotations

import math
from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from torch.nn import functional as F

TAU_MIN = 0.5      # width floor in order units: never a degenerate point mass
TAU_SCALE = 290.0  # softplus scale: tau ~ 201 at raw = 0 -> near-uniform over 403 levels
N_PARAMS = 2       # (m, s) -> (mu, tau); a third raw output z when zero_atom=True
# zero atom (#E57): an ADDITIVE logit bump on bin 0, `logit(0) += z`, renormalised. The first
# build was a sigmoid mixture weight started at sigmoid(-6) so the init stayed flat — and that
# saturated sigmoid gave z a gradient ~200x smaller than the location's; both 4M runs ended
# with w ~ 0.000 everywhere (a dead atom, same class of bug as #E55's clamp). The additive
# form starts flat at z = 0 with d logp(0)/dz = 1 - p(0): never saturated, never a gate at 0.5.
LOG_CENTER = 0.0   # mu = 1 at m = 0; the train script passes log(prior mean) instead
LOG_SPAN = 2.0     # d mu / d m = 2 mu: |m| <= 1 is a ~7.4x swing either side of the centre
SQRT_SPAN = 3.0    # sqrt location: mu = (sqrt(centre) + 3 m)^2; |m| <= 1 spans ~0 .. 38 around a centre of 10
MU_CAP = 1e4       # keep exp() finite for a diverging head


def expand_dgauss_logits(
    raw: th.Tensor,
    n: int,
    tau_min: float = TAU_MIN,
    tau_scale: float = TAU_SCALE,
    mu_param: str = "log",
    log_center: float = LOG_CENTER,
    zero_atom: bool = False,
) -> th.Tensor:
    """``(batch, 2)`` raw head outputs -> ``(batch, n)`` exact log-probabilities.

    With ``zero_atom`` the head has a third output ``z`` added to bin 0's logit
    before normalisation: order 0 gets its OWN parameter, so the zero band no
    longer has to be carved out of the shared location (#E54: under GAE noise
    that bend never forms).
    """
    m, s = raw[:, 0], raw[:, 1]
    if mu_param == "log":                   # multiplicative: constant RELATIVE resolution
        mu = th.clamp(th.exp(log_center + LOG_SPAN * m), max=MU_CAP)
    elif mu_param == "sqrt":                # resolution ~ sqrt(mu): the Poisson spread and the
        root = math.exp(0.5 * log_center) + SQRT_SPAN * m      # profit peak's width both scale that way
        # softplus, NOT clamp: a hard clamp at 0 has zero gradient below it, and #E55's first sqrt
        # run drove root to -4 in the first 100k steps and could never come back (an absorbing
        # order-nothing state, scored 0.13). softplus keeps the gradient alive at the floor.
        mu = F.softplus(root) ** 2          # (#E54: log's unit error grows like mu, affine's is flat)
    else:                                   # clark_scarf's form: |m| <= 1 spans the support
        mu = 0.5 * (n - 1) * (1.0 + m)
    tau = tau_min + tau_scale * F.softplus(s)
    k = th.arange(n, dtype=raw.dtype, device=raw.device)
    quad = -((k[None, :] - mu[:, None]) ** 2) / (2.0 * tau[:, None] ** 2)
    logq = quad - th.logsumexp(quad, dim=1, keepdim=True)
    if not zero_atom:
        return logq
    bump = th.cat([raw[:, 2:3], th.zeros_like(logq[:, 1:])], dim=1)   # z on bin 0 only
    logit = logq + bump
    return logit - th.logsumexp(logit, dim=1, keepdim=True)


class DGaussCategoricalDistribution(CategoricalDistribution):
    """A single discretized Gaussian; log_prob / entropy / sample / mode are the
    inherited categorical machinery over the expanded logits."""

    def __init__(self, n: int, *, tau_min: float = TAU_MIN, tau_scale: float = TAU_SCALE,
                 mu_param: str = "log", log_center: float = LOG_CENTER, zero_atom: bool = False) -> None:
        super().__init__(int(n))
        self.tau_min, self.tau_scale, self.mu_param = float(tau_min), float(tau_scale), str(mu_param)
        self.log_center, self.zero_atom = float(log_center), bool(zero_atom)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, N_PARAMS + (1 if self.zero_atom else 0))

    def proba_distribution(self, action_logits: th.Tensor):
        return super().proba_distribution(
            expand_dgauss_logits(action_logits, self.action_dim, self.tau_min,
                                 self.tau_scale, self.mu_param, self.log_center, self.zero_atom))


class DGaussActorCriticPolicy(ActorCriticPolicy):
    """``ActorCriticPolicy`` with the discretized-Gaussian head installed.

    ``_build`` swaps the distribution BEFORE the parent constructs the action net
    and optimizer, so the swap is complete before any parameter exists. Nothing
    else is overridden: the location needs no state-dependent anchor, which is
    what makes it safe under ``norm_obs`` and identical for the raw modes.
    """

    def __init__(self, *args: Any, tau_min: float = TAU_MIN, tau_scale: float = TAU_SCALE,
                 mu_param: str = "log", log_center: float = LOG_CENTER, zero_atom: bool = False,
                 **kwargs: Any) -> None:
        self._tau_min, self._tau_scale = float(tau_min), float(tau_scale)
        self._mu_param, self._log_center, self._zero_atom = str(mu_param), float(log_center), bool(zero_atom)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict:
        d = super()._get_constructor_parameters()
        d.update(tau_min=self._tau_min, tau_scale=self._tau_scale,
                 mu_param=self._mu_param, log_center=self._log_center, zero_atom=self._zero_atom)
        return d

    def _build(self, lr_schedule) -> None:
        space = self.action_space
        assert isinstance(space, spaces.Discrete), (
            f"the dgauss head needs a Discrete action space, got {space}; use -a order_discrete")
        self.action_dist = DGaussCategoricalDistribution(
            int(space.n), tau_min=self._tau_min, tau_scale=self._tau_scale,
            mu_param=self._mu_param, log_center=self._log_center, zero_atom=self._zero_atom)
        super()._build(lr_schedule)



if __name__ == "__main__":   # smoke: the init, MEASURED (never asserted)
    import math
    n = 403
    for lab, lc in (("log_center 0 (bare default)", 0.0),
                    ("log_center log(10) (a base-like prior mean)", math.log(10.0))):
        q = expand_dgauss_logits(th.zeros(1, N_PARAMS), n, mu_param="log", log_center=lc).exp().detach().numpy()[0]
        print(lab)
        print(f"  raw=0: argmax {q.argmax():3d}  P(argmax) {q.max():.5f}  uniform {1/n:.5f}"
              f"  max/min {q.max()/q.min():.2f}  mass<=20 {q[:21].sum():.3f}")
    print("  once the net narrows tau (raw s = -3):")
    for m_ in (-0.5, 0.0, 0.5, 1.0):
        q = expand_dgauss_logits(th.tensor([[m_, -3.0]]), n, mu_param="log",
                                 log_center=math.log(10.0)).exp().detach().numpy()[0]
        print(f"    m={m_:+.1f}: argmax {q.argmax():3d}  P {q.max():.3f}")

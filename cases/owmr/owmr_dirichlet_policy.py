"""Dirichlet-allocation actor-critic for owmr (ESCALATION #E23) — an ARCHITECTURE lever.

The continuous counterpart of the categorical retention head (#E21). The
policy outputs the ALLOCATION ITSELF as a random point on the simplex: the
network emits concentration parameters alpha(obs) for a Dirichlet over the
N + 1 shares [kept, retailer_1..retailer_N]; the action is a sample from it
(always feasible, no decode), the deterministic action is its mode. What this
changes relative to every "point + Gaussian noise" head:

- a share of exactly 0 is the MODE whenever alpha_i < 1 — a corner reached at
  a finite parameter value, with a finite, informative score function
  (d log p / d alpha_i = log s_i - psi(alpha_i) + psi(sum alpha)); no asymptote
  (softmax, #E18), no flat region (relu, #E15/#E17);
- the spread is the concentration sum alpha_0(obs), a function of the state:
  sharp where the policy is sure, wide where it is not — the state-dependent
  exploration SB3's Gaussian std lacks (#E19/#E20) and the categorical has;
- exploration lives on the simplex: near a corner the mass sits on the faces.

Action (gym mode `order_dirichlet`, width N + 2): [order, s_0, s_1..s_N] with
sum_j s_j = 1; the gym ships s_i * W and keeps s_0 * W. The order stays a
Gaussian component. Joint log-prob and entropy are the sums.

Numerics: alpha = softplus(z) + ALPHA_FLOOR; samples are floored at SHARE_EPS
and renormalized BEFORE being stored as the action, so the log-prob PPO
re-scores is the log-prob of the action the environment saw. The concentration
head's bias is initialized so alpha starts near ALPHA_INIT per share (interior
mode, gentle exploration); corners are learned, not imposed. Mode convention:
components with alpha_i <= 1 get 0 and the rest split in proportion to
(alpha_i - 1); if no component exceeds 1, the largest alpha takes everything.
"""

from __future__ import annotations

import math

import torch as th
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

ALPHA_FLOOR = 1e-3
SHARE_EPS = 1e-6
GAMMA_FLOOR = 1e-30     # a Gamma variate under a tiny alpha can underflow to 0; the log-density is scored at this floor
ALPHA_INIT = 2.0


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def dirichlet_mode(alpha: th.Tensor) -> th.Tensor:
    """Mode with the boundary convention described in the module docstring."""
    above = alpha > 1.0
    w = th.where(above, alpha - 1.0, th.zeros_like(alpha))
    tot = w.sum(dim=1, keepdim=True)
    mode = th.where(tot > 0, w / tot.clamp(min=1e-12), th.zeros_like(w))
    none = (tot <= 0).squeeze(1)
    if none.any():
        one_hot = th.nn.functional.one_hot(alpha.argmax(dim=1), alpha.shape[1]).to(alpha.dtype)
        mode = th.where(none.unsqueeze(1), one_hot, mode)
    return mode


class GaussianDirichlet(DiagGaussianDistribution):
    """Gaussian over the order (component 0) x Dirichlet over the N + 1 shares.

    Subclasses DiagGaussianDistribution (action_dim = 1) so SB3's `_build`
    wires the head and the order's log-std through `proba_distribution_net`.
    """

    def __init__(self, n_shares: int) -> None:
        super().__init__(1)
        self.n_shares = int(n_shares)
        self.gauss: th.distributions.Normal | None = None
        self.dir: th.distributions.Dirichlet | None = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        head = nn.Linear(latent_dim, 1 + self.n_shares)         # order mean | concentration logits
        with th.no_grad():
            head.bias[1:] = _inv_softplus(ALPHA_INIT - ALPHA_FLOOR)
        log_std = nn.Parameter(th.ones(1) * log_std_init, requires_grad=True)
        return head, log_std

    def proba_distribution(self, head_out: th.Tensor, log_std: th.Tensor) -> "GaussianDirichlet":
        mean = head_out[:, :1]
        alpha = th.nn.functional.softplus(head_out[:, 1:]) + ALPHA_FLOOR
        self.gauss = th.distributions.Normal(mean, th.ones_like(mean) * log_std.exp())
        self.dir = th.distributions.Dirichlet(alpha)
        return self

    @property
    def alpha(self) -> th.Tensor:
        return self.dir.concentration

    @staticmethod
    def _clean(shares: th.Tensor) -> th.Tensor:
        s = shares.clamp(min=SHARE_EPS)
        return s / s.sum(dim=1, keepdim=True)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.gauss.log_prob(actions[:, :1]).sum(dim=1) + self.dir.log_prob(self._clean(actions[:, 1:]))

    def entropy(self) -> th.Tensor:
        return self.gauss.entropy().sum(dim=1) + self.dir.entropy()

    def coord_log_prob(self, actions: th.Tensor, gammas: th.Tensor | None = None) -> th.Tensor:
        """Per-COORDINATE log-probs for the vine (#E32, `owmr_vine_ppo`).

        With `gammas` ([n, N + 1], the Gamma variates behind the shares: a
        Dirichlet(alpha) draw is G_i / sum G with independent G_i ~ Gamma(alpha_i, 1)):
        [n, 1 + N + 1] = the order's Gaussian log-prob, then log Gamma(G_i; alpha_i)
        per share — the PER-SHARE form: in Gamma space the density factorises, so a
        sibling that redrew one G_i is scored on that one factor,
        d/d alpha_i = log G_i − psi(alpha_i). Without `gammas`: [n, 2], the order
        then the shares' Dirichlet log-prob as one block (the block form)."""
        order = self.gauss.log_prob(actions[:, :1]).sum(dim=1, keepdim=True)
        if gammas is None:
            return th.cat([order, self.dir.log_prob(self._clean(actions[:, 1:])).unsqueeze(1)], dim=1)
        g = gammas.clamp(min=GAMMA_FLOOR)
        a = self.alpha
        return th.cat([order, (a - 1.0) * th.log(g) - g - th.lgamma(a)], dim=1)

    def sample(self) -> th.Tensor:
        return th.cat([self.gauss.rsample(), self._clean(self.dir.rsample())], dim=1)

    def mode(self) -> th.Tensor:
        return th.cat([self.gauss.mean, dirichlet_mode(self.alpha)], dim=1)

    def actions_from_params(self, head_out, log_std, deterministic: bool = False):
        self.proba_distribution(head_out, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, head_out, log_std):
        actions = self.actions_from_params(head_out, log_std)
        return actions, self.log_prob(actions)


class DirichletPolicy(ActorCriticPolicy):
    """MlpPolicy whose action distribution is GaussianDirichlet."""

    def _build(self, lr_schedule) -> None:
        n_shares = int(self.action_space.shape[0]) - 1
        self.action_dist = GaussianDirichlet(n_shares)
        super()._build(lr_schedule)
        # SB3's ortho_init re-initialized the head (gain 0.01, zero bias): restore
        # the concentration bias so alpha starts near ALPHA_INIT
        with th.no_grad():
            self.action_net.bias[1:] = _inv_softplus(ALPHA_INIT - ALPHA_FLOOR)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        return self.action_dist.proba_distribution(self.action_net(latent_pi), self.log_std)

    def evaluate_coord_log_prob(self, obs: th.Tensor, actions: th.Tensor, aux: th.Tensor | None = None) -> th.Tensor:
        """Per-coordinate log-probs of `actions` under the current policy (#E32);
        `aux` = the Gamma variates behind the shares (per-share form)."""
        return self.get_distribution(obs).coord_log_prob(actions, aux)

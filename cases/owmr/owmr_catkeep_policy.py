"""Categorical-retention actor-critic for owmr (ESCALATION #E21) — an ARCHITECTURE lever.

Retention is the one decision no continuous head has learned (#E14–#E20): the
softmax reaches "keep nothing" only asymptotically (#E18), the relu heads reach
it as a clip and then sit in the flat region behind it (#E15, #E17), and PPO's
state-independent Gaussian std makes exploring retention costly in the 72% of
periods where stock is short, so the mean is driven away (#E19, #E20). Here the
retention decision is DISCRETE — the standard categorical policy: logits ->
softmax -> a sampled OPTION from a menu of kept fractions of on-hand,
"keep nothing" being one option among others. A categorical has no asymptote
and no flat region, every option keeps positive probability while it is being
learned, and its probabilities are a function of the state, so the policy can
explore retention in surplus periods and not in short ones.

Action (gym mode `order_catkeep`, width N + 2): [order, k_index, x_1..x_N]
  - order:   Gaussian, as always
  - k_index: the categorical's choice, written into the action as a float index
             into KEEP_MENU (the gym rounds it); sampled during training,
             argmax when deterministic
  - x_i:     Gaussian, decoded by the relu split of the shipped (1 - k) * W
             (exact zeros; equal split if all <= 0), as in order_nested

The joint distribution is Gaussian(order, x) x Categorical(k): log-prob and
entropy are the sums. Everything else is SB3's flat MlpPolicy: one policy
trunk on the observation, one value trunk, nothing shared across retailers.
"""

from __future__ import annotations

import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

# kept fraction of on-hand per option; the heuristic's kept fraction in its
# surplus periods on base: mean 0.28, p90 0.25, p99 0.55, max 0.81 (#E17)
KEEP_MENU = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)


class GaussianCategorical(DiagGaussianDistribution):
    """Gaussian over the continuous components + a categorical over the menu.

    Action layout: [cont_0 (order), k_index, cont_1..cont_{n_cont-1} (x_i)].
    The categorical's index sits at position 1 so the layout matches the gym.
    Subclasses DiagGaussianDistribution so SB3's `_build` wires the head and
    the log-std through `proba_distribution_net` unchanged.
    """

    def __init__(self, n_cont: int, n_options: int) -> None:
        super().__init__(int(n_cont))
        self.n_cont, self.n_options = int(n_cont), int(n_options)
        self.gauss: th.distributions.Normal | None = None
        self.cat: th.distributions.Categorical | None = None

    # -- SB3 Distribution API -------------------------------------------------

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        head = nn.Linear(latent_dim, self.n_cont + self.n_options)   # means | logits
        log_std = nn.Parameter(th.ones(self.n_cont) * log_std_init, requires_grad=True)
        return head, log_std

    def proba_distribution(self, head_out: th.Tensor, log_std: th.Tensor) -> "GaussianCategorical":
        means, logits = head_out[:, :self.n_cont], head_out[:, self.n_cont:]
        self.gauss = th.distributions.Normal(means, th.ones_like(means) * log_std.exp())
        self.cat = th.distributions.Categorical(logits=logits)
        return self

    def _split(self, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        cont = th.cat([actions[:, :1], actions[:, 2:]], dim=1)
        idx = th.round(actions[:, 1]).long().clamp(0, self.n_options - 1)
        return cont, idx

    def _join(self, cont: th.Tensor, idx: th.Tensor) -> th.Tensor:
        return th.cat([cont[:, :1], idx.to(cont.dtype).unsqueeze(1), cont[:, 1:]], dim=1)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        cont, idx = self._split(actions)
        return self.gauss.log_prob(cont).sum(dim=1) + self.cat.log_prob(idx)

    def entropy(self) -> th.Tensor:
        return self.gauss.entropy().sum(dim=1) + self.cat.entropy()

    def coord_log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Per-COORDINATE log-probs, [n, n_cont + 1]: the Gaussian's for
        [order, x_1..x_N] then the categorical's — `log_prob` is their row
        sum. The vine (#E30, `owmr_vine_ppo`) scores a sibling on the one
        coordinate it changed, so the parent's window cost stays a valid
        baseline for it."""
        cont, idx = self._split(actions)
        return th.cat([self.gauss.log_prob(cont), self.cat.log_prob(idx).unsqueeze(1)], dim=1)

    def sample(self) -> th.Tensor:
        return self._join(self.gauss.rsample(), self.cat.sample())

    def mode(self) -> th.Tensor:
        return self._join(self.gauss.mean, th.argmax(self.cat.probs, dim=1))

    def actions_from_params(self, head_out, log_std, deterministic: bool = False):
        self.proba_distribution(head_out, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, head_out, log_std):
        actions = self.actions_from_params(head_out, log_std)
        return actions, self.log_prob(actions)


class CatKeepPolicy(ActorCriticPolicy):
    """MlpPolicy whose action distribution is GaussianCategorical."""

    def __init__(self, *args, n_options: int = len(KEEP_MENU), **kwargs) -> None:
        self.n_options = int(n_options)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        # replace the distribution BEFORE SB3 builds the action net from it
        n_cont = int(self.action_space.shape[0]) - 1
        self.action_dist = GaussianCategorical(n_cont, self.n_options)
        super()._build(lr_schedule)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        return self.action_dist.proba_distribution(self.action_net(latent_pi), self.log_std)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data["n_options"] = self.n_options
        return data

    def evaluate_coord_log_prob(self, obs: th.Tensor, actions: th.Tensor, aux: th.Tensor | None = None) -> th.Tensor:
        """Per-coordinate log-probs of `actions` under the current policy (#E30); `aux` unused here."""
        return self.get_distribution(obs).coord_log_prob(actions)

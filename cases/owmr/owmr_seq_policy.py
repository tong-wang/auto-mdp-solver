"""Phase-masked actor-critic for the two-phase `seq_order_softmax` mode (ESCALATION #E10).

SB3 needs ONE action space for both phases, so every step emits the full
[order, logit_1..logit_N] vector and the gym ignores the components that are
not live: the logits at the order step, the order at the allocation step. Left
in PPO's objective, the ignored components would add pure noise to the
probability ratio and collect entropy bonus for nothing. This policy masks them
out of the log-probability and the entropy: at the order step only component 0
counts, at the allocation step only components 1..N.

The phase is the LAST observation feature (`raw_seq`). The policy reads it
AFTER VecNormalize, where the running mean sits between the two phase values
(the phases alternate exactly, so the mean tends to 1/2; at initialization it
is 0 with unit variance). Phase 1 therefore always normalizes to a positive
number and phase 0 to a non-positive one, so the test is `> 0`.

Everything else is SB3's standard MlpPolicy: one shared (64, 64) policy trunk
reading the phase feature, a separate value trunk. The value function is what
separates the credit: it is learned at the order-step state AND at the
allocation-step interstate, so each decision's advantage is measured against
its own baseline.
"""

from __future__ import annotations

import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy


class PhaseMaskedPolicy(ActorCriticPolicy):
    """MlpPolicy whose log-probability and entropy count only the live components."""

    def _live_mask(self, obs: th.Tensor) -> th.Tensor:
        """(B, D) float mask: component 0 at the order step, 1..N at the allocation step."""
        alloc = (obs[:, -1] > 0).unsqueeze(1)                       # (B, 1)
        d = int(self.action_space.shape[0])
        first = th.zeros(1, d, dtype=th.bool, device=obs.device)
        first[0, 0] = True
        return th.where(alloc, ~first, first).to(th.float32)       # (B, D)

    def _masked_log_prob(self, distribution, actions: th.Tensor, mask: th.Tensor) -> th.Tensor:
        per_dim = distribution.distribution.log_prob(actions)        # (B, D), Normal
        return (per_dim * mask).sum(dim=1)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = self._masked_log_prob(distribution, actions, self._live_mask(obs))
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        mask = self._live_mask(obs)
        log_prob = self._masked_log_prob(distribution, actions, mask)
        values = self.value_net(latent_vf)
        entropy = (distribution.distribution.entropy() * mask).sum(dim=1)
        return values, log_prob, entropy

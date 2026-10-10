"""Split actor for owmr (ESCALATION #E9) — an ARCHITECTURE lever.

The IR and the gym are untouched: the policy emits the `order_softmax` action
[order, logit_1..logit_N] on the `raw` observation, normalized by VecNormalize
as in #E4. What changes is the actor:

- SB3's standard policy computes the order and the N allocation logits from ONE
  shared trunk, with only the last linear layer per output. Here the order and
  the allocation each get their OWN trunk, both reading the full global
  observation (the warehouse's view), with no shared weights, so neither
  decision's gradient can reshape the features the other relies on.
- The value network is separate, as in SB3.
- Trunks are either plain tanh MLPs (``split``, SB3's own layer shape) or
  deeper residual MLPs with LayerNorm (``splitres``: an input projection to
  the width, then one pre-norm residual block per ``net_arch`` entry), the
  form that keeps a deep policy network trainable.

What a split cannot change: PPO still scores the whole action with ONE
advantage per period, so the two decisions share their credit signal.

The actor's heads emit the action means directly; ``action_net`` is replaced
by a block-diagonal map (order head on the order trunk, allocation head on the
allocation trunk), initialized at SB3's action-net gain 0.01.
"""

from __future__ import annotations

from typing import Sequence

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class ResBlock(nn.Module):
    """Pre-norm residual block: x + W2 tanh(W1 LN(x))."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x + self.fc2(th.tanh(self.fc1(self.norm(x))))


def trunk(n_in: int, net_arch: Sequence[int], residual: bool) -> tuple[nn.Module, int]:
    """(module, output width). Plain: tanh MLP over net_arch. Residual: project to
    net_arch[0], one ResBlock per entry, final LayerNorm + tanh."""
    if not residual:
        layers: list[nn.Module] = []
        d = n_in
        for w in net_arch:
            layers += [nn.Linear(d, w), nn.Tanh()]
            d = w
        return nn.Sequential(*layers), d
    width = int(net_arch[0])
    assert all(int(w) == width for w in net_arch), "residual trunks need a constant width"
    mods: list[nn.Module] = [nn.Linear(n_in, width)]
    mods += [ResBlock(width) for _ in net_arch]
    mods += [nn.LayerNorm(width), nn.Tanh()]
    return nn.Sequential(*mods), width


class SplitExtractor(nn.Module):
    """latent_pi = [order trunk features, allocation trunk features]; latent_vf = value trunk."""

    def __init__(self, feature_dim: int, net_arch: Sequence[int], residual: bool) -> None:
        super().__init__()
        self.order_trunk, d_o = trunk(feature_dim, net_arch, residual)
        self.alloc_trunk, d_a = trunk(feature_dim, net_arch, residual)
        self.value_trunk, d_v = trunk(feature_dim, net_arch, residual)
        self.d_order, self.d_alloc = d_o, d_a
        self.latent_dim_pi = d_o + d_a
        self.latent_dim_vf = d_v

    def forward_actor(self, x: th.Tensor) -> th.Tensor:
        return th.cat([self.order_trunk(x), self.alloc_trunk(x)], dim=1)

    def forward_critic(self, x: th.Tensor) -> th.Tensor:
        return self.value_trunk(x)

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(x), self.forward_critic(x)


class SplitHead(nn.Module):
    """Block-diagonal action map: order <- order trunk, logits <- allocation trunk."""

    def __init__(self, d_order: int, d_alloc: int, n_logits: int) -> None:
        super().__init__()
        self.d_order = d_order
        self.order = nn.Linear(d_order, 1)
        self.alloc = nn.Linear(d_alloc, n_logits)
        for lin in (self.order, self.alloc):
            nn.init.orthogonal_(lin.weight, gain=0.01)
            nn.init.zeros_(lin.bias)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        return th.cat([self.order(latent[:, :self.d_order]),
                       self.alloc(latent[:, self.d_order:])], dim=1)


class SplitActorCriticPolicy(ActorCriticPolicy):
    """Separate order / allocation / value trunks on the global observation."""

    def __init__(self, *args, residual: bool = False, **kwargs) -> None:
        self.residual = bool(residual)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SplitExtractor(self.features_dim, list(self.net_arch), self.residual)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        n_act = int(self.action_space.shape[0])
        ex = self.mlp_extractor
        # replace SB3's full linear map (which would mix the trunks) by the block-diagonal one
        self.action_net = SplitHead(ex.d_order, ex.d_alloc, n_act - 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data["residual"] = self.residual
        return data

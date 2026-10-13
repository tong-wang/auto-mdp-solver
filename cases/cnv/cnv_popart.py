"""PPO with a PopArt-normalised value head (van Hasselt et al. 2016), for #E57.

WHY (#E54's split): the top band's loss is the critic's — the value at a rate of
100 with 100 periods to go is ~10,000, learned from ~1% of episodes, and a 64-64
critic regresses those states toward typical values; under PPO's clip that
offset survives into the policy update. PopArt keeps the value head's OUTPUT at
unit scale: the network predicts a normalised value, the last layer is rescaled
whenever the running statistics move so its de-normalised prediction is
preserved, and the value loss is a unit-scale regression.

HOW, inside SB3 without copying ``train``: the policy's ``value_net`` is wrapped.
During rollouts the wrapper returns ``v_hat * sigma + mu`` (raw units, so GAE and
the advantages are exactly PPO's). At the top of ``train`` the statistics are
updated from the buffer's returns, the last layer is rescaled, the buffer's
``returns`` are normalised in place and the wrapper switches to returning
``v_hat``; SB3's ``F.mse_loss(returns, values_pred)`` is then PopArt's loss.
Afterwards the wrapper switches back. ``clip_range_vf`` must stay ``None``.
Everything else in PPO is untouched.
"""
from __future__ import annotations

import torch as th
from stable_baselines3 import PPO
from torch import nn


class _PopArtValueNet(nn.Linear):
    """A Linear value head whose state_dict keys are exactly a plain Linear's
    (``weight`` / ``bias``), so a saved model loads into the stock policy for
    eval; the running statistics are non-persistent buffers (eval only needs
    the actor)."""

    def __init__(self, inner: nn.Linear, beta: float) -> None:
        assert isinstance(inner, nn.Linear) and inner.out_features == 1, inner
        super().__init__(inner.in_features, 1)
        with th.no_grad():
            self.weight.copy_(inner.weight); self.bias.copy_(inner.bias)
        self.beta = float(beta)
        self.register_buffer("mu", th.zeros(()), persistent=False)
        self.register_buffer("nu", th.ones(()), persistent=False)      # second moment
        self.normalised_output = False

    @property
    def sigma(self) -> th.Tensor:
        return th.sqrt(th.clamp(self.nu - self.mu ** 2, min=1e-4))

    def forward(self, latent: th.Tensor) -> th.Tensor:
        v = super().forward(latent)
        return v if self.normalised_output else v * self.sigma + self.mu

    @th.no_grad()
    def update(self, returns: th.Tensor) -> None:
        mu_old, sigma_old = self.mu.clone(), self.sigma.clone()
        self.mu.mul_(1 - self.beta).add_(self.beta * returns.mean())
        self.nu.mul_(1 - self.beta).add_(self.beta * (returns ** 2).mean())
        sigma_new = self.sigma
        # preserve the de-normalised prediction: W <- W s_old/s_new, b <- (s_old b + mu_old - mu_new)/s_new
        self.weight.mul_(sigma_old / sigma_new)
        self.bias.mul_(sigma_old).add_(mu_old - self.mu).div_(sigma_new)


class PopArtPPO(PPO):
    def __init__(self, *args, popart_beta: float = 0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.clip_range_vf is None, "PopArt here needs clip_range_vf=None"
        old = self.policy.value_net
        self.policy.value_net = _PopArtValueNet(old, popart_beta)
        # the optimizer holds the OLD Linear's tensors: swap them for the new head's
        for g in self.policy.optimizer.param_groups:
            g["params"] = [self.policy.value_net.weight if q is old.weight else
                           self.policy.value_net.bias if q is old.bias else q for q in g["params"]]

    def train(self) -> None:
        vn: _PopArtValueNet = self.policy.value_net  # type: ignore[assignment]
        buf = self.rollout_buffer
        returns = th.as_tensor(buf.returns, device=self.device).flatten()
        vn.update(returns)
        raw_returns = buf.returns.copy()
        buf.returns = ((returns - vn.mu) / vn.sigma).cpu().numpy().reshape(buf.returns.shape)
        vn.normalised_output = True
        try:
            super().train()
        finally:
            vn.normalised_output = False
            buf.returns = raw_returns
        self.logger.record("train/popart_mu", float(vn.mu))
        self.logger.record("train/popart_sigma", float(vn.sigma))

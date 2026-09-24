"""Discretized-Gaussian head for the `ship_discrete` MultiDiscrete action.

THE PLAIN BODY — adjacency pooling with NO zero atom. Per link, two numbers:

    mu  = (n-1)/2 * (1 + m)                  location over bins 0..n-1, AFFINE
    tau = tau_min + tau_scale * softplus(s)  per-state width, floored

    logit(k) = -(k - mu)^2 / (2 tau^2) - logZ,   k = 0..n-1

Six numbers replace 123 logits at n_echelons = 3, n_bins = 41. The location
gradient d log pi / d mu ~ (a - mu) / tau^2 pools every sample into one
estimate of where the level is — the property a flat categorical lacks, and
the ONLY mechanism this head keeps from `clark_scarf_ordinal_head.py`.

WHY THE ATOM IS GONE (#E6). The sibling head is a HURDLE construction: a
sigmoid gate exclusively owns P(ship = 0) and the positive branch is
zero-truncated. That gate is the (s,S) trigger — a fixed-cost mechanism, per
adi_flex LV1's own scope clause ("zero-inflate only when the problem has a
genuine 'do nothing' mass (a fixed cost => an (s,S) trigger); otherwise the
plain ordinal body"). This domain has NO fixed cost: shipping is per-unit
linear, the DP ships ~ demand nearly every period (P(d=0) = e^-10), and
zero-ship is ~16% on one link, endgame-concentrated. Importing the composite
anyway cost this board a measured pathology: the gate starts at sigmoid(0) =
0.5, the deterministic-eval argmax is "ship nothing" until that one scalar
travels ~3.7 logit-units, and every slow-learning configuration scored as a
constant 22501.15 — 28 of 57 tuning trials. Here, zero is just the boundary
bin: reachable, never privileged.

WHY AFFINE mu, not the sibling's sigmoid squashing. There is no
parameter-identical "sibling minus atom" — their mu lived on [1, n-1], so
removing the atom re-ranges it regardless. Affine is chosen because a sigmoid
mu re-creates the travel problem in miniature at the SUPPORT boundaries: a
mode on bin 0 would again need ~|logit(1/n)| = 3.7 units of a saturating
input. Affine reaches every bin at |m| <= 1 with constant sensitivity, and a
mu wandering past the support is not an error — the renormalized quadratic
piles mass on the boundary bin, which is exactly what a boundary mode IS (a
half-Gaussian at 0, the endgame's shape).

At init (SB3 ortho gain 0.01, zero bias): mu = mid-range (bin 20), tau ~ 28,
the distribution GENUINELY near-flat (max/min ~ 1.27 — measured, not
asserted; contrast the hurdle head's 20x atom), and the deterministic argmax
is a live mid-range action that moves continuously with mu. An undertrained
policy evaluates as a mediocre shipper, never as a catastrophe — the property
whose absence produced #E6.

`tau_min` / `tau_scale` travel through `policy_kwargs` into
`_get_constructor_parameters` (the Beta head's recorded persistence bug, not
repeated). Saved models reference this class by module path: eval / select
must run where `clark_scarf_dgauss_head` is importable.
"""

from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from torch.nn import functional as F

TAU_MIN = 0.25     # width floor: never a degenerate point mass
TAU_SCALE = 40.0   # softplus scale: tau ~ 28 at init, near-flat over 41 bins
N_PARAMS_PER_COMPONENT = 2


def expand_dgauss_logits(
    raw: th.Tensor,
    action_dims: list[int],
    tau_min: float = TAU_MIN,
    tau_scale: float = TAU_SCALE,
    mu_param: str = "affine",
) -> th.Tensor:
    """``(batch, 2*C)`` raw head outputs -> ``(batch, sum(action_dims))`` exact
    log-probabilities, laid out as ``MultiCategoricalDistribution`` splits them."""
    p = raw.view(-1, len(action_dims), N_PARAMS_PER_COMPONENT)
    out: list[th.Tensor] = []
    for j, n in enumerate(action_dims):
        if n == 1:
            out.append(th.zeros_like(p[:, j, :1]))
            continue
        m, s = p[:, j, 0], p[:, j, 1]
        if mu_param == "affine":
            mu = 0.5 * (n - 1) * (1.0 + m)          # |m| <= 1 spans the support
        else:                                        # "sigmoid": the SIBLING form,
            # atomless twin of the hurdle body (a3). mu = (n-1)*sigmoid(m) spans
            # (0, n-1); boundary bins need a saturating |m| ~ 4 -- that edge
            # cliff is the TWINNED property, kept deliberately so a1-vs-a3
            # differs in exactly one thing: the atom.
            mu = (n - 1) * th.sigmoid(m)
        tau = tau_min + tau_scale * F.softplus(s)
        k = th.arange(n, dtype=raw.dtype, device=raw.device)
        quad = -((k[None, :] - mu[:, None]) ** 2) / (2.0 * tau[:, None] ** 2)
        out.append(quad - th.logsumexp(quad, dim=1, keepdim=True))
    return th.cat(out, dim=1)


class DGaussMultiCategoricalDistribution(MultiCategoricalDistribution):
    """Every component a discretized Gaussian; log_prob / entropy / sample /
    mode are the inherited machinery over the expanded logits."""

    def __init__(self, action_dims: list[int], *,
                 tau_min: float = TAU_MIN, tau_scale: float = TAU_SCALE,
                 mu_param: str = "affine") -> None:
        super().__init__(action_dims)
        self.tau_min = float(tau_min)
        self.tau_scale = float(tau_scale)
        assert mu_param in ("affine", "sigmoid"), mu_param
        self.mu_param = mu_param

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, N_PARAMS_PER_COMPONENT * len(self.action_dims))

    def proba_distribution(self, action_logits: th.Tensor):
        return super().proba_distribution(
            expand_dgauss_logits(action_logits, list(self.action_dims),
                                 self.tau_min, self.tau_scale, self.mu_param))


class DGaussActorCriticPolicy(ActorCriticPolicy):
    """``ActorCriticPolicy`` with the discretized-Gaussian head installed.

    ``_build`` swaps the distribution BEFORE the parent constructs the action
    net and optimizer, so the swap is complete before any parameter exists.
    Non-maskable base, deliberately: masking is rejected three times on this
    domain and a masked run FAILs ``run.provenance``.
    """

    def __init__(self, *args: Any, tau_min: float = TAU_MIN,
                 tau_scale: float = TAU_SCALE, mu_param: str = "affine",
                 **kwargs: Any) -> None:
        self._tau_min = float(tau_min)
        self._tau_scale = float(tau_scale)
        self._mu_param = str(mu_param)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict:
        d = super()._get_constructor_parameters()
        d.update(tau_min=self._tau_min, tau_scale=self._tau_scale,
                 mu_param=self._mu_param)
        return d

    def _build(self, lr_schedule) -> None:
        space = self.action_space
        assert isinstance(space, spaces.MultiDiscrete), (
            f"the dgauss head needs a MultiDiscrete action space, got {space}; "
            f"use -a ship_discrete"
        )
        self.action_dist = DGaussMultiCategoricalDistribution(
            list(int(n) for n in space.nvec),
            tau_min=self._tau_min, tau_scale=self._tau_scale,
            mu_param=self._mu_param)
        super()._build(lr_schedule)


if __name__ == "__main__":   # smoke: the init, measured
    dims = [41, 41, 41]
    raw = th.zeros(1, N_PARAMS_PER_COMPONENT * len(dims))
    probs = expand_dgauss_logits(raw, dims).exp().detach().numpy()[0][:41]
    print(f"init (raw = 0): P(0)={probs[0]:.4f}  P(20)={probs[20]:.4f}  "
          f"P(40)={probs[40]:.4f}  flat={1/41:.4f}")
    print(f"  max/min ratio {probs.max()/probs.min():.3f}   argmax bin {probs.argmax()}")
    print(f"  sums to {probs.sum():.6f}")

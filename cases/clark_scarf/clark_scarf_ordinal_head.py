"""HURDLE-DISCRETIZED-GAUSSIAN head for `ship_discrete` ("ordinal" is a
MISNOMER kept for artifact compatibility — saved models reference this class by
module path, run names embed `distordinal`, and the registry's `a1` delta is
`policy_dist=ordinal`, so the identifiers are frozen and the correction lives
here and in the record).

Two mechanisms under the one name (#E6): ADJACENCY POOLING (the discretized-
Gaussian body — the general part) and a ZERO ATOM (the hurdle gate, a
fixed-cost mechanism). The positive branch is zero-truncated, so the gate
EXCLUSIVELY owns P(ship = 0) — `P(0) = w`, one scalar, no second route via
mu -> 0. That exclusivity is a measured hazard on this domain: at zero-init
w = sigmoid(0) = 0.5, the deterministic-eval argmax is "ship nothing" until t
travels ~3.7 logit-units, and every slow-learning configuration scores as a
constant 22501.15 (28 of 57 tuning trials; see #E6). This domain has NO fixed
cost, so the atom is unwarranted here by adi_flex LV1's own scope clause; the
domain-appropriate head is `clark_scarf_dgauss_head.py` (a2).

WHY THIS EXISTS
---------------
`ship_discrete` renders each link's shipment as a categorical over whole units,
`MultiDiscrete([n_bins] * n_echelons)`. A plain categorical spends one free
logit per bin with **no notion of adjacency**: learning that 13 is right teaches
the head nothing about 12 or 14, and the gradient that arrives for the sampled
bin nudges that bin's logit alone.

RETRACTED (#E6): this docstring originally justified the atom with "at
target, ship nothing is the modal action by construction". That was wrong
arithmetic — under base-stock, demand knocks the echelon position below target
essentially every period (P(d=0) = e^-10), so the DP ships ~ demand nearly
always; zero-ship is ~16% on one link, endgame-concentrated. There is no fixed
cost and no step in the cost surface, so PLAYBOOK LV7's own trigger test says
no atom was warranted. The head remains in the codebase as the a1 record and
for the atom-vs-no-atom decomposition against a2.

This head replaces the PARAMETERIZATION only. The action space, the encoding
and the IR are untouched — `gym.action_modes` does not move and no fingerprint
changes. What moves is the policy class, i.e. the `a` axis (guide §13.1: "the
policy/extractor code implements it — a custom class, not a constructor
argument"). Per link, every logit is induced from three numbers:

    w   = sigmoid(t)                          P(ship = 0)      -- the trigger
    mu  = 1 + (n-2)*sigmoid(m)                location on 1..n-1
    tau = tau_min + tau_scale*softplus(s)     width, floored

    logit(0) = log w
    logit(k) = log(1-w) - (k-mu)^2/(2 tau^2) - logZ,   k >= 1
               logZ = logsumexp_j[ -(j-mu)^2/(2 tau^2) ]

a point mass at zero mixed with a discretized Gaussian over the positive
quantities. **9 parameters replace 123 logits** at `n_echelons = 3`,
`n_bins = 41`.

The property the flat head lacks: the location gradient

    d log pi / d mu  ~  (a - mu) / tau^2

pools EVERY sample into one estimate of where the level is, instead of moving
one category's logit. That is the mechanism, and it is why the head is expected
to buy variance before it buys a mean.

PROVENANCE AND WHAT CHANGED IN THE PORT
---------------------------------------
Ported from `inv_single_ordinal_head.py` and `adi_flex_ordinal_head.py`, which
are the same design. Neither was a drop-in and the differences are deliberate:

* `inv_single`'s is a single `Discrete` on a plain `ActorCriticPolicy`;
  `adi_flex`'s is `MultiDiscrete` but ordinalizes **only component 0** (its
  other head is a protection level, where a smoothness prior points the wrong
  way) and inherits `MaskableActorCriticPolicy`.
* Here **every component is a quantity**, so every component is ordinal.
* Here the base is the **non-maskable** `ActorCriticPolicy`. Masking is
  rejected three times in this campaign (#E5/#E6/#E15) and a masked run now
  FAILs `run.provenance`, the IR declaring `ppo` alone.
* **`tau_min` / `tau_scale` travel through `policy_kwargs`, not module
  globals.** Both siblings keep them as module constants, which carries the
  persistence bug this campaign has already paid for once on the Beta head: a
  class attribute does not travel, so `PPO.load` would rebuild the head with
  whatever default was in force and score a policy that was never trained.
  `_get_constructor_parameters` puts them in the saved model.

Saved models reference this class by module path, so eval / select must run
where `clark_scarf_ordinal_head` is importable (the domain dir, as always).

INITIALIZATION, STATED RATHER THAN ASSUMED
------------------------------------------
SB3 inits the action layer with ortho gain 0.01 and zero bias, so the raw
outputs start near 0, giving `w = 0.5`, `mu` mid-range and `tau = tau_min +
tau_scale*log(2)`. At the defaults and `n_bins = 41` that is `tau ~ 28`, wide
enough that the positive branch is near-flat.

**The zero atom starts at 0.5, and that is NOT a flat categorical's start.** A
zero-init categorical over 41 bins gives every action ~0.024; this head gives
action 0 half the mass and spreads the rest. That is a real prior toward not
shipping. It is kept because it is what both siblings ran and what their
evidence was measured under — changing it here would make this port an
unrecorded second experiment. `test_ordinal_head_init_is_the_documented_prior`
pins the numbers so the choice is visible rather than inherited.
"""

from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from torch.nn import functional as F

# width floor: the head may approach but never reach a degenerate point mass.
# Deterministic play is the eval argmax, not a collapsed training distribution.
TAU_MIN = 0.25
# softplus scale: sets how wide the positive branch starts. 40 gives tau ~ 28
# at init, i.e. near-flat over a 41-bin range.
TAU_SCALE = 40.0
N_PARAMS_PER_COMPONENT = 3


def expand_ordinal_logits(
    raw: th.Tensor,
    action_dims: list[int],
    tau_min: float = TAU_MIN,
    tau_scale: float = TAU_SCALE,
) -> th.Tensor:
    """``(batch, 3*C)`` raw head outputs -> ``(batch, sum(action_dims))`` exact
    log-probabilities, laid out as ``MultiCategoricalDistribution`` splits them.

    Components are handled in a loop rather than vectorized: `action_dims` may
    differ per component (nothing here forbids it), and C is 2-4.
    """
    p = raw.view(-1, len(action_dims), N_PARAMS_PER_COMPONENT)
    out: list[th.Tensor] = []
    for j, n in enumerate(action_dims):
        if n == 1:
            # Discrete(1): one legal value, log-prob 0. Upstream v0.9.30 made a
            # degenerate discrete bound legal, so this is reachable in principle
            # and the arange below would be empty.
            out.append(th.zeros_like(p[:, j, :1]))
            continue
        t, m, s = p[:, j, 0], p[:, j, 1], p[:, j, 2]
        log_w = F.logsigmoid(t)                      # log P(ship = 0)
        log_nw = F.logsigmoid(-t)                    # log P(ship > 0)
        mu = 1.0 + (n - 2) * th.sigmoid(m)           # location on 1..n-1
        tau = tau_min + tau_scale * F.softplus(s)
        k = th.arange(1, n, dtype=raw.dtype, device=raw.device)
        quad = -((k[None, :] - mu[:, None]) ** 2) / (2.0 * tau[:, None] ** 2)
        log_z = th.logsumexp(quad, dim=1, keepdim=True)
        out.append(th.cat([log_w[:, None], log_nw[:, None] + quad - log_z], dim=1))
    return th.cat(out, dim=1)


class OrdinalMultiCategoricalDistribution(MultiCategoricalDistribution):
    """Every component ordinal (mixture-at-zero).

    The action layer emits ``3*C`` numbers; this class expands them to the full
    ``sum(action_dims)`` logits and delegates to the parent, so ``log_prob``,
    ``entropy``, ``sample``, ``mode`` and ``actions_from_params`` are all the
    inherited machinery over the expanded logits.
    """

    def __init__(self, action_dims: list[int], *,
                 tau_min: float = TAU_MIN, tau_scale: float = TAU_SCALE) -> None:
        super().__init__(action_dims)
        self.tau_min = float(tau_min)
        self.tau_scale = float(tau_scale)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, N_PARAMS_PER_COMPONENT * len(self.action_dims))

    def proba_distribution(self, action_logits: th.Tensor):
        return super().proba_distribution(
            expand_ordinal_logits(action_logits, list(self.action_dims),
                                  self.tau_min, self.tau_scale))


class OrdinalActorCriticPolicy(ActorCriticPolicy):
    """``ActorCriticPolicy`` with the ordinal head installed.

    ``_build`` swaps the distribution BEFORE the parent constructs the action
    net and the optimizer, so the swap is complete before any parameter exists
    — no rebuild, and the optimizer never sees an orphaned tensor.
    """

    def __init__(self, *args: Any, tau_min: float = TAU_MIN,
                 tau_scale: float = TAU_SCALE, **kwargs: Any) -> None:
        # set BEFORE super().__init__, which calls _build
        self._tau_min = float(tau_min)
        self._tau_scale = float(tau_scale)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict:
        # saved with the model: a class attribute does not travel, and a head
        # rebuilt at load time with a different width is a policy that was
        # never trained (the Beta head's recorded bug)
        d = super()._get_constructor_parameters()
        d.update(tau_min=self._tau_min, tau_scale=self._tau_scale)
        return d

    def _build(self, lr_schedule) -> None:
        space = self.action_space
        assert isinstance(space, spaces.MultiDiscrete), (
            f"the ordinal head needs a MultiDiscrete action space, got {space}; "
            f"use -a ship_discrete"
        )
        self.action_dist = OrdinalMultiCategoricalDistribution(
            list(int(n) for n in space.nvec),
            tau_min=self._tau_min, tau_scale=self._tau_scale)
        super()._build(lr_schedule)


if __name__ == "__main__":   # smoke: what the head looks like at init
    import numpy as np

    dims = [41, 41, 41]
    raw = th.zeros(1, N_PARAMS_PER_COMPONENT * len(dims))
    probs = expand_ordinal_logits(raw, dims).exp().detach().numpy()[0][:41]
    tau = TAU_MIN + TAU_SCALE * float(F.softplus(th.tensor(0.0)))
    print(f"init (raw params = 0), n_bins=41: tau={tau:.2f}  mu={1 + 39*0.5:.1f}")
    print(f"  P(ship=0)            = {probs[0]:.4f}   (flat categorical: {1/41:.4f})")
    print(f"  P(k) for k=1..40     = {probs[1]:.5f} .. {probs[20]:.5f} .. {probs[-1]:.5f}")
    print(f"  sums to {probs.sum():.6f}")

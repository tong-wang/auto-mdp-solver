"""A Beta-distributed policy head for PPO on the `[0,1]` fraction action.

WHY THIS EXISTS
---------------
SB3 gives every ``Box`` action space a diagonal Gaussian and clips the sample
into the box. On this domain that is not a mild approximation, it is the
operating regime. Measured on the L1 `ship_fraction` artifact
(`g1_n3_l2_p09`, echelon, seed 2):

    pre-clip Gaussian mean per link :  [ 0.499,  -0.007,  -2.394 ]   on a [0,1] action
    learned std                     :  [ 0.341,   0.248,   0.178 ]
    share of sampled actions clipped:  58.8% / 52.3% / 88.6%

A Gaussian can only concentrate at a boundary by pushing its mean OUTSIDE the
support and letting the clip pile mass there — hence the −2.394. PPO's ratio
then scores ``log pi(a_unclipped)``, the density of a sample the environment
never executed, so the majority of the gradient signal describes actions that
did not happen. Chou, Maturana & Scherer (ICML 2017) make exactly this
argument for the Beta distribution; Fujita & Maeda (ICML 2018) instead debias
the Gaussian estimator. We take the first route because the Beta's support IS
this action space.

The measured cost of the Gaussian here, holding encoding, cap and rounding
fixed and changing only the head (2 seeds, spread ~60):

    Gaussian    1570.77      categorical  1144.95      -> head effect -425.8
    DP bar 993.30, ship_discrete 1015.57

TWO DESIGN POINTS
-----------------
*State-dependent concentration.* SB3's ``log_std`` is a global ``nn.Parameter``
— one sigma per dimension, shared across every state, so the policy cannot be
sharp where it knows the answer and broad where it does not. Beta's alpha and
beta are network OUTPUTS, which is the capability the Gaussian was missing.

*Bimodality is available if wanted.* ``alpha, beta > 1`` is unimodal and
numerically comfortable; below 1 the Beta becomes U-shaped, with real mass at
both ends. `MIN_CONCENTRATION` selects between them. The default is 1.0 (safe,
unimodal). Note the per-state optimal action here is a point mass —
``clip(ybar - u, 0, cap)`` — so the bimodality visible in the ACTION MARGINAL
is a fact about the mix of states, not about any single conditional, and a
unimodal-per-state head reproduces it. Lower this only if that assumption is
being tested.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution, sum_independent_dims
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from torch.distributions import Beta

# Beta's support is the OPEN interval (0,1): the log-density diverges at the
# ends, and float32 rsample() can land exactly on them. Actions are pulled
# inside by this much before any log_prob is taken.
_EPS = 1e-6

# both concentrations within this of 1 => the density is flat to within
# numerical noise and its argmax is meaningless; fall back to the mean
_FLAT_TOL = 1e-3


class BetaDistribution(Distribution):
    """Independent Beta per action dimension, on ``[0,1]``."""

    #: alpha, beta = softplus(x) + MIN_CONCENTRATION.
    #:
    #: A Beta needs only alpha, beta > 0, and softplus already gives (0, inf),
    #: so this exists for NUMERICAL validity and nothing else — hence 1e-3.
    #:
    #: It was 1.0 first, copied from Chou et al.'s recommendation to hold
    #: alpha,beta >= 1 so every conditional stays unimodal and concave. That is
    #: reasonable where the optimum is interior, and wrong here: with alpha > 1
    #: the density at 0 is exactly ZERO, and with beta > 1 the density at 1 is
    #: zero, so the head could place no mass on either wall. Measured on the DP
    #: optimal policy at `g1_n3_l2_p09`, the walls are where the policy lives —
    #: link 0 ships its full available quantity 51.1% of the time (median
    #: fraction 1.000) and link 2 ships exactly nothing 16.2% of the time. The
    #: floor forbade precisely the behaviour the domain requires, and to
    #: approach a wall the network had to drive a logit toward -inf, which is
    #: the Gaussian's own failure mode relocated into parameter space.
    #:
    #: Not a safety cost: with actions clamped to [_EPS, 1-_EPS], |log_prob|
    #: peaks near 10 even at alpha = 0.01, no sample lands on an endpoint in
    #: 200k draws, and PPO's ratio clip bounds the update regardless.
    #:
    #: `--beta-min-conc 1.0` restores the old family. That is a COMPARISON
    #: instrument — every result recorded before 2026-08-23 used it — not a
    #: tuning knob, and it should be retired once the comparison is written up.
    MIN_CONCENTRATION: float = 1e-3

    def __init__(self, action_dim: int, min_concentration: float | None = None):
        super().__init__()
        self.action_dim = action_dim
        # INSTANCE attribute, defaulting to the class one. A class attribute
        # alone does NOT travel with a saved model: `PPO.load` rebuilt the head
        # with whatever default was in force and scored a policy that was never
        # trained -- measured, a Gamma arm went 1004 -> 13000 that way.
        self.min_concentration = float(
            self.MIN_CONCENTRATION if min_concentration is None else min_concentration)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        # 2 outputs per dimension; both concentrations are state-dependent,
        # which is the point (contrast SB3's global log_std parameter)
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, action_params: th.Tensor) -> "BetaDistribution":
        a_raw, b_raw = th.chunk(action_params, 2, dim=-1)
        alpha = nn.functional.softplus(a_raw) + self.min_concentration
        beta = nn.functional.softplus(b_raw) + self.min_concentration
        self.distribution = Beta(alpha, beta)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        actions = actions.clamp(_EPS, 1.0 - _EPS)
        return sum_independent_dims(self.distribution.log_prob(actions))

    def entropy(self) -> Optional[th.Tensor]:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        """The deterministic action. ``CLARK_BETA_DET`` selects which summary.

        SB3 calls this for ``deterministic=True``, and for a Gaussian the
        question is empty because mode == mean. For a skewed Beta it is not:
        measured on the L1 `om40 raw` artifact the two differ by −0.062,
        −0.059, +0.047 in fraction units per link, and those biases do NOT
        cancel — the first two ship less downstream, the third orders more from
        the supplier, so both push inventory up and a 50-period system
        integrates the drift. That is the whole of the deterministic-vs-
        stochastic gap (holding 974.77 -> 2379.04 under sampling).

        MODE is the most likely action and the SB3 convention. MEAN is what the
        sampled policy realizes on average, so it is closer to the quantity PPO
        actually maximizes. Set ``CLARK_BETA_DET=mean`` to compare; `mode` is
        the default because it is what every recorded number here used.
        """
        a, b = self.distribution.concentration1, self.distribution.concentration0
        if os.environ.get("CLARK_BETA_DET", "mean" if False else "mode") == "mean":
            return (a / (a + b)).clamp(0.0, 1.0)

        # Beta's mode is only interior when BOTH concentrations exceed 1. Once
        # the floor is numerical (1e-3) rather than 1.0 every other branch is
        # reachable, and each needs its own answer — an earlier version fell
        # back to the mean everywhere, which for the U-shaped case returns the
        # point of MINIMUM density, i.e. the anti-mode:
        #
        #   a>1, b>1   unimodal, interior      (a-1)/(a+b-2)
        #   a<1, b<1   U-shaped, TIE at 0 and 1 -> the end holding more mass
        #   a<1, b>=1  spike at 0              -> 0
        #   a>=1, b<1  spike at 1              -> 1
        #   a==1,b==1  UNIFORM, every point ties -> 0.5, arbitrary but central
        #   a==1, b>1  monotone decreasing     -> 0
        #   a>1, b==1  monotone increasing     -> 1
        #
        # The ties are real, not numerical: a flat density has no argmax, and a
        # symmetric U has two. Breaking them by MASS (which side carries more
        # probability) is the choice that degrades gracefully as the shape moves
        # off the tie, where picking arbitrarily would jump.
        zero, one, half = th.zeros_like(a), th.ones_like(a), th.full_like(a, 0.5)
        interior = (a - 1.0) / (a + b - 2.0).clamp(min=_EPS)
        # for a U-shape the heavier side is the one with the SMALLER
        # concentration (a<b puts more mass near 0); exact tie -> 0
        u_end = th.where(a <= b, zero, one)
        out = th.where(b < 1.0, one, zero)          # spike at 1, else at 0
        out = th.where((a < 1.0) & (b < 1.0), u_end, out)
        out = th.where((a > 1.0) & (b > 1.0), interior, out)
        # NEAR-flat, by tolerance and not by `== 1.0`: exact float equality
        # essentially never fires, and without this a policy that has expressed
        # no preference at all falls into the U-shaped branch and commits to an
        # ENDPOINT. Ship-nothing is a strong action to take on the strength of
        # numerical noise; the mean (0.5 when a == b) is the honest answer when
        # the density has no argmax.
        flat = ((a - 1.0).abs() < _FLAT_TOL) & ((b - 1.0).abs() < _FLAT_TOL)
        out = th.where(flat, a / (a + b), out)
        return out.clamp(0.0, 1.0)

    def actions_from_params(self, action_params: th.Tensor,
                            deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(action_params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_params: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_params)
        return actions, self.log_prob(actions)


class BetaActorCriticPolicy(ActorCriticPolicy):
    """``ActorCriticPolicy`` with the action head replaced by a Beta.

    Only the two hooks that name a distribution class are overridden; the rest
    of SB3's policy — feature extractor, value net, orthogonal init, optimizer
    — is inherited untouched, so the arm differs from the Gaussian one in the
    action distribution and in nothing else.
    """

    def __init__(self, *args: Any, min_concentration: float | None = None,
                 **kwargs: Any) -> None:
        # set BEFORE super().__init__, which calls _build
        self._min_conc = min_concentration
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict:
        d = super()._get_constructor_parameters()
        d.update(min_concentration=self.action_dist.min_concentration)
        return d

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        space = self.action_space
        assert isinstance(space, spaces.Box), "Beta head needs a Box action space"
        assert np.allclose(space.low, 0.0) and np.allclose(space.high, 1.0), (
            f"Beta's support is [0,1]; this action space is "
            f"[{space.low.min()}, {space.high.max()}]. Rescaling is deliberately "
            f"NOT done here — a fraction action already lives on [0,1], and an "
            f"affine remap would hide a mismatch rather than surface it."
        )
        self.action_dist = BetaDistribution(int(np.prod(space.shape)),
                                            min_concentration=self._min_conc)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        # A global log_std is meaningless for a Beta, and it must be DELETED
        # rather than set to None: `PPO.train` does
        #     if hasattr(self.policy, "log_std"): th.exp(self.policy.log_std)
        # to log train/std, so a None attribute passes the guard and then
        # raises. Removing it makes hasattr False, which is the intent anyway —
        # this head has no single global scale to report.
        if hasattr(self, "log_std"):
            del self.log_std
        self.action_net.to(self.device)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        return self.action_dist.proba_distribution(self.action_net(latent_pi))


# ---------------------------------------------------------------------------
# Gamma head: the decision's own domain is [0, inf)
# ---------------------------------------------------------------------------


class GammaDistribution(Distribution):
    """Independent Gamma per action dimension, on ``[0, inf)``.

    WHY, over Beta. Once the availability constraint is enforced by the env's
    clip rather than encoded into the action (`ship_scaled`), the decision has
    **no right-hand wall**: the model bounds a shipment only below, at zero,
    and the IR is explicit that ``ship_max`` is "an ACTION-SCALE cap, not a
    physical limit". Putting a Beta on it forces an upper bound to be invented
    purely so the support fits. Gamma needs no range chosen, and its only
    boundary is the one the model actually declares.

    The clipping that remains is at *availability*, which is the genuine
    physical constraint and the same projection Clark & Scarf's median form
    applies -- categorically different from a Gaussian parking its mean outside
    its own support.

    PARAMETERIZED BY (shape, MEAN) rather than (shape, rate), for one practical
    reason: a shipment here is ~10 units, and with a rate parameterization the
    initial distribution sits near 1 and has to crawl two orders of magnitude.
    ``mean = softplus(x) * scale_hint`` with ``scale_hint`` = mean demand puts
    the initial mean at ~0.7x mean demand, i.e. in the useful range from the
    first update -- the same hazard the `ship` mode failed on (22560 vs 982),
    handled at initialization instead of by rescaling the action.
    """

    MIN_CONCENTRATION: float = 1e-3

    def __init__(self, action_dim: int, scale_hint: float = 1.0,
                 min_concentration: float | None = None):
        super().__init__()
        self.action_dim = action_dim
        self.scale_hint = float(scale_hint)
        self.min_concentration = float(
            self.MIN_CONCENTRATION if min_concentration is None else min_concentration)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, action_params: th.Tensor) -> "GammaDistribution":
        shape_raw, mean_raw = th.chunk(action_params, 2, dim=-1)
        conc = nn.functional.softplus(shape_raw) + self.min_concentration
        mean = nn.functional.softplus(mean_raw) * self.scale_hint + _EPS
        self.distribution = th.distributions.Gamma(conc, conc / mean)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return sum_independent_dims(self.distribution.log_prob(actions.clamp(min=_EPS)))

    def entropy(self) -> Optional[th.Tensor]:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        """(conc-1)/rate where conc > 1; 0 otherwise (the density is monotone
        decreasing there, so its argmax is the left boundary -- which IS in the
        support, unlike Beta's, so no tie-breaking is needed)."""
        c, r = self.distribution.concentration, self.distribution.rate
        if os.environ.get("CLARK_BETA_DET", "mode") == "mean":
            return c / r
        return th.where(c > 1.0, (c - 1.0) / r, th.zeros_like(c))

    def actions_from_params(self, action_params: th.Tensor,
                            deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(action_params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_params: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_params)
        return actions, self.log_prob(actions)


class GammaActorCriticPolicy(ActorCriticPolicy):
    """``ActorCriticPolicy`` with a Gamma action head."""

    def __init__(self, *args: Any, scale_hint: float | None = None,
                 min_concentration: float | None = None, **kwargs: Any) -> None:
        # set BEFORE super().__init__, which calls _build
        self._scale_hint = scale_hint
        self._min_conc = min_concentration
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict:
        d = super()._get_constructor_parameters()
        d.update(scale_hint=self.action_dist.scale_hint,
                 min_concentration=self.action_dist.min_concentration)
        return d

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        space = self.action_space
        assert isinstance(space, spaces.Box), "Gamma head needs a Box action space"
        assert np.allclose(space.low, 0.0), "Gamma's support starts at 0"
        hint = _GAMMA_SCALE_HINT[0] if self._scale_hint is None else self._scale_hint
        self.action_dist = GammaDistribution(int(np.prod(space.shape)),
                                             scale_hint=hint,
                                             min_concentration=self._min_conc)
        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.mlp_extractor.latent_dim_pi)
        if hasattr(self, "log_std"):
            del self.log_std          # see BetaActorCriticPolicy for why
        self.action_net.to(self.device)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        return self.action_dist.proba_distribution(self.action_net(latent_pi))


#: LEGACY FALLBACK ONLY. The scale hint now travels inside the model via
#: ``policy_kwargs``, which is what `PPO.load` restores. This global is kept so
#: artifacts saved before that fix -- which carry no such kwarg -- can still be
#: scored, with the eval script setting it from the run's args log.
_GAMMA_SCALE_HINT = [1.0]

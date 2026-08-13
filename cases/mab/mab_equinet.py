"""Equivariant policies for the mab domain (ESCALATION.md A6, L2(arch)).

Rung b — ``IndexPolicy``: the pure index-policy class. One scoring MLP
phi(x_i, y_i, ttg) is shared across arms; its per-arm outputs ARE the action
logits (the softmax over them is the only cross-arm interaction — comparison
as action selection). UCB1 (mean + c*sd) is a member of this class, and its
Thompson distillation reaches 99.5% of Thompson at 4.5k params (#E11), so the
class provably contains a near-reference policy; this run asks whether PPO can
*find* one when the search space is only that big.

Rung c — ``DeepSetsPolicy``: the DeepSets tier. Same depth and width as rung
b's scorer, but every hidden layer also sees the *mean-pooled* representation
of all arms: h_i <- tanh(h_i W + mean_j(h_j) U + b) (the Sn-equivariant layer
of Ravanbakhsh et al. 2017 — two free weight blocks per feature pair —
implemented structurally, so it is K-agnostic). Arm i's logit may now depend
on how good the *other* arms look, which is exactly what rung b forbids: an
index rule is optimal for the infinite-horizon discounted problem, provably
not for the finite-horizon undiscounted one this domain poses (README §1).
The rung isolates that finite-horizon-vs-index delta and nothing else — the
value head is byte-identical to rung b's, so the only b->c change is cross-arm
context inside the scorer. Note the final scoring layer is deliberately a
plain per-arm ``Linear``: a pooled term there adds the *same* constant to
every logit, which the softmax cancels, so it would be a literal no-op.

Equivariance by construction: relabel the arms and the logits relabel with
them, exactly — the property the trained MLP measurably lacks (TV 0.298,
34.5% argmax flips under relabeling, #E11).

The value head is permutation-INVARIANT (a scalar must not depend on arm
labels): mean-pooled per-arm embeddings + time-to-go -> MLP.

Integration contract (mab_ppo_train enforces it): ``norm_obs`` must be OFF —
VecNormalize keeps per-slot stats, and an equivariant net behind a slotwise
normalizer is not an equivariant policy. The extractor scales its own inputs
(ttg / horizon; the bayes features are already O(1)).

Works for either observation mode — both are [per-arm block x2, ttg] — but
the scaling differs (ESCALATION A11, operator design 2026-08-04). ``bayes``
features pass through raw. ``stats`` features (pull counts up to T, unbounded
payout totals) would saturate the tanh scorer on contact, and the tied
normalizer they would need is exactly what the contract forbids, so the
extractor rescales them losslessly instead: count -> **pull share**
n_i / max(t, 1) with t = T - ttg, total -> **running average** with the prior
mean filled for unpulled arms (unambiguous: share = 0 identifies them).
(share, avg, ttg) <-> (count, total, ttg) is a bijection, so this is still
the stats *encoding* — the swap hands the arithmetic, never the posterior;
what stays for the net to construct is the uncertainty feature
sd = 1/sqrt(1+n) that the bayes mode hands over. t = sum_j n_j is
permutation-invariant (= T - ttg), so equivariance is exact.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class _IndexExtractor(nn.Module):
    """Splits the flat obs [x_0..x_{K-1}, y_0..y_{K-1}, ttg] into per-arm
    features, scores each arm with the shared ``phi``, and pools an invariant
    representation for the value head."""

    def __init__(self, n_arms: int, horizon: int, obs_mode: str = "bayes",
                 prior_mean: float = 0.0) -> None:
        super().__init__()
        self.n_arms = n_arms
        self.horizon = float(horizon)
        self.obs_mode = obs_mode
        self.prior_mean = float(prior_mean)
        self.latent_dim_pi = n_arms          # per-arm logits, used verbatim
        self.latent_dim_vf = 64
        self._build_scorer()                 # registered first — see the note
        self.psi = nn.Sequential(nn.Linear(2, 32), nn.Tanh())
        self.vhead = nn.Sequential(
            nn.Linear(33, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )

    def _build_scorer(self) -> None:
        """Rung b's scorer: a shared per-arm MLP with no cross-arm context.

        A hook, not inlined, so rung c can swap the stack while registering
        it at the same point in ``__init__``. Submodule registration order
        drives SB3's orthogonal-init traversal, hence its RNG draws, so
        keeping it fixed is what makes rung b's crowned run reproducible.
        """
        self.phi = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def _split(self, obs: torch.Tensor):
        K = self.n_arms
        x, y, ttg = obs[:, :K], obs[:, K:2 * K], obs[:, 2 * K:]
        if self.obs_mode == "stats":
            # A11: x = pull counts, y = payout totals — lossless rescale to
            # pull share + running average (module docstring). max-guards:
            # t=0 only on the first step (all shares 0), and y/max(x,1) is
            # exact wherever the where() takes it (x >= 1 there).
            t = (self.horizon - ttg).clamp(min=1.0)
            share = x / t
            avg = torch.where(x > 0, y / x.clamp(min=1.0),
                              torch.full_like(y, self.prior_mean))
            x, y = share, avg
        return x, y, ttg / self.horizon

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        x, y, ttg = self._split(obs)
        feats = torch.stack([x, y, ttg.expand(-1, self.n_arms)], dim=2)
        return self.phi(feats).squeeze(-1)                     # (N, K) logits

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        x, y, ttg = self._split(obs)
        pooled = self.psi(torch.stack([x, y], dim=2)).mean(dim=1)
        return self.vhead(torch.cat([pooled, ttg], dim=1))     # (N, 64)

    def forward(self, obs: torch.Tensor):
        return self.forward_actor(obs), self.forward_critic(obs)


class _DeepSetsLayer(nn.Module):
    """One Sn-equivariant layer: h'_i = h_i W + mean_j(h_j) U + b.

    Two free weight blocks per feature pair (Ravanbakhsh et al. 2017): the
    diagonal ``W`` acting on the arm itself and the off-diagonal ``U`` acting
    on the pooled context. Mean pooling (not sum) keeps the scale independent
    of K.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_w = nn.Linear(in_dim, out_dim)                  # W, b
        self.ctx_w = nn.Linear(in_dim, out_dim, bias=False)       # U

    def _pool(self, h: torch.Tensor) -> torch.Tensor:             # (N, K, in)
        return h.mean(dim=1, keepdim=True)                        # (N, 1, in)

    def forward(self, h: torch.Tensor) -> torch.Tensor:           # (N, K, in)
        return self.self_w(h) + self.ctx_w(self._pool(h))


class _LooMaxLayer(_DeepSetsLayer):
    """Same layer, pooling by LEAVE-ONE-OUT MAX: ctx_i = max_{j != i} h_j.

    A regret-minimizing rule compares arm i against the current *leader*, not
    against the average, so this hands the scorer the decision-relevant
    statistic directly. Measured on Thompson-visited states (4096 episodes),
    variance explained in max_{j!=i} pm_j: mean-pool 0.33, plain max-pool
    0.82, this 1.00 by construction — plain max caps out because for the
    leading arm the pool contains itself, which is exactly the arm whose
    comparison you wanted.

    Still exactly Sn-equivariant (relabel the arms and the contexts relabel
    with them) and O(K) via top-2. Ties are correct without a special case:
    when two arms share the max, both match ``top1`` and both receive
    ``top2``, which equals that same value.
    """

    def _pool(self, h: torch.Tensor) -> torch.Tensor:             # (N, K, in)
        top2 = h.topk(2, dim=1).values                            # (N, 2, in)
        return torch.where(h == top2[:, :1], top2[:, 1:2], top2[:, :1])


class _AttnLayer(_DeepSetsLayer):
    """Rung d. Same layer again, pooling by single-head self-attention over
    the arms: ctx_i = sum_j alpha_ij h_j with alpha learned, masked at j = i.

    This makes the three cross-arm rungs one family that differs *only* in the
    pooling weights alpha over the same h:

        rung c      alpha_ij = 1/K            (uniform)
        rung c-max  alpha_ij = 1[j = argmax]  (one-hot on the best other arm)
        rung d      alpha_ij = learned        (anything in between, per state)

    which is the point of the rung: softmax attention IS a learned pooling
    statistic, uniform at low logit scale and max-like at high, so it spans
    the c/c-max endpoints and picks a position per feature and per state
    instead of having one baked in.

    No value projection on purpose — pooling the same ``h`` the siblings pool
    is what keeps the comparison about alpha alone. No positional encoding
    either: attention keyed on the arm *index* would attend to slot identity
    and destroy the exact equivariance this whole ladder buys.

    The diagonal is masked because ``self_w`` already carries arm i's own
    features. For the mean this is immaterial — pooling over all j versus over
    j != i differ by (K-1)/K plus an h_i/K term that ``self_w`` absorbs, so
    rung c's canonical DeepSets form spans the same functions either way. For
    attention it is not: alpha_ii would be state-dependent and therefore not
    absorbable into a fixed W, which would quietly turn "context" into a
    learned gate on the arm's own features.
    """

    D_K = 16                                   # attention key/query width

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(in_dim, out_dim)
        self.q_w = nn.Linear(in_dim, self.D_K, bias=False)
        self.k_w = nn.Linear(in_dim, self.D_K, bias=False)

    def _pool(self, h: torch.Tensor) -> torch.Tensor:             # (N, K, in)
        att = self.q_w(h) @ self.k_w(h).transpose(1, 2)           # (N, K, K)
        att = att / self.D_K ** 0.5
        eye = torch.eye(h.shape[1], dtype=torch.bool, device=h.device)
        att = att.masked_fill(eye, float("-inf"))                 # no self
        return torch.softmax(att, dim=-1) @ h                     # (N, K, in)


class _DeepSetsExtractor(_IndexExtractor):
    """Rung c: rung b's extractor with the scorer's hidden layers made
    context-aware. Depth, width and the whole critic are inherited unchanged,
    so the parameter delta (8.8k vs 4.5k) is exactly the pooled-context
    weights ``U`` and nothing else."""

    #: the pooling layer this variant uses; the ONLY difference between the
    #: rung-c siblings, so their comparison isolates the pooling statistic
    _LAYER = _DeepSetsLayer

    def _build_scorer(self) -> None:
        self.phi = nn.Sequential(
            self._LAYER(3, 64), nn.Tanh(),
            self._LAYER(64, 64), nn.Tanh(),
            nn.Linear(64, 1),                 # plain: pooled here would cancel
        )
        # forward_actor is inherited verbatim: it feeds phi an (N, K, 3)
        # tensor, and a _DeepSetsLayer consumes that shape exactly as the
        # nn.Linear it replaces does.


class _DeepSetsMaxExtractor(_DeepSetsExtractor):
    """Rung c-max: identical to rung c but pooling by leave-one-out max.
    Same parameter count — only the pooling statistic changes."""

    _LAYER = _LooMaxLayer


class _AttnExtractor(_DeepSetsExtractor):
    """Rung d: the pooling weights become learned attention. Adds only the
    query/key projections (10,913 scorer par vs the siblings' 8,769)."""

    _LAYER = _AttnLayer


class IndexPolicy(ActorCriticPolicy):
    """ActorCriticPolicy whose actor is the shared per-arm scorer.

    ``action_net`` is replaced by Identity — SB3's default Linear(K, K) on
    top of the per-arm logits would mix arm slots and silently destroy the
    equivariance the extractor provides.
    """

    #: read by mab_policy to skip frame averaging (an exact no-op here) and
    #: by mab_ppo_train to force norm_obs off. Set on the class, not matched
    #: by name, so rung c inherits it correctly.
    IS_EQUIVARIANT = True

    #: the extractor this rung installs; rungs differ only in this
    _EXTRACTOR = _IndexExtractor

    def __init__(self, *args, horizon: int = 1000, obs_mode: str = "bayes",
                 prior_mean: float = 0.0, **kwargs):
        self._horizon = int(horizon)
        self._obs_mode = obs_mode
        self._prior_mean = float(prior_mean)
        kwargs["net_arch"] = []              # the extractor IS the network
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = self._EXTRACTOR(
            n_arms=int(self.action_space.n), horizon=self._horizon,
            obs_mode=self._obs_mode, prior_mean=self._prior_mean)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()      # logits pass through untouched
        # the optimizer was created over the old parameter set (including the
        # discarded Linear); rebuild it exactly as SB3's _build does
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


class DeepSetsPolicy(IndexPolicy):
    """Rung c. Identical integration to rung b (Identity ``action_net``,
    ``net_arch=[]``, self-scaled inputs, equivariant); only the scorer's
    hidden layers gain the mean-pooled cross-arm context."""

    _EXTRACTOR = _DeepSetsExtractor


class DeepSetsMaxPolicy(DeepSetsPolicy):
    """Rung c-max. Same class size and shape as rung c; the pooled context is
    the leave-one-out max instead of the mean, i.e. "the best arm other than
    me" — see _LooMaxLayer for why that is the statistic the decision needs."""

    _EXTRACTOR = _DeepSetsMaxExtractor


class AttentionPolicy(DeepSetsPolicy):
    """Rung d. The pooling weights are learned per state by single-head
    self-attention over the arms, so the fixed statistic of rungs c/c-max
    becomes a learned one spanning both. See _AttnLayer.

    Scope: this is the MINIMAL form of the rung — attention as the pooling
    operator, in the same slot the siblings use, so the ladder keeps changing
    one lever at a time. The full attention tier (multi-head, LayerNorm, and PMA
    learned-query readout) would be a further rung and is not run here."""

    _EXTRACTOR = _AttnExtractor


# ---------------------------------------------------------------------------
# A7 anneal prescriptions (#E17): the crown provably over-explores late —
# +69-ish reward from just forcing argmax after t0 (anneal probe, commit
# part). The defect is the training objective: the entropy bonus is FLAT in
# episode time while the option value of randomness falls toward zero at
# small ttg, so the optimizer prefers entropy exactly where it only costs.
# Two general fixes, neither of which encodes anything Bayes-specific:
# t2 fixes the objective, t3 fixes the parametrization.
# ---------------------------------------------------------------------------

class _TtgEntropyMixin:
    """Weight each sample's entropy by remaining horizon before SB3 averages
    it into the entropy loss, so the bonus pays for randomness in proportion
    to its remaining option value — full price at t=0, ~nothing at t=T.

    Why ttg/T and not some other schedule (the beta-probe finding, 2026-08-01):
    with gamma=1 over a T-step horizon the MAGNITUDE of the advantage signal
    scales with return-to-go — getting the arm right at t=50 affects ~950
    remaining pulls, at t=950 only ~50 — while the entropy bonus is flat. Their
    ratio therefore goes as 1/ttg and entropy dominates exactly late. Weighting
    entropy by ttg/T is the unique weighting that holds the entropy-to-advantage
    exchange rate CONSTANT in episode time. It is a cancellation of a scaling
    the objective already has, not a hand-picked anneal.

    Mixin, not a subclass, so the weighting composes with the t3 temperature
    head (which lives in the extractor) instead of multiplying policy classes.
    It adds no modules, so init fingerprints are untouched — asserted below.

    ``_MEAN_ONE`` rescales the weight to E_t[w] = 1, matching the flat
    baseline's AVERAGE entropy coefficient so that shape and level are
    orthogonal knobs. Without it, mean(w) = E[ttg/T] = (T+1)/2T ~ 1/2 and the
    run confounds the schedule with a halved ent_coef (the confound
    ``TtgEntropyIndexPolicy`` discloses and A7-t2n removes). Note this matches
    the average COEFFICIENT, not the average entropy — the policy's entropy
    responds nonlinearly to its price — so the contrast it buys is
    matched-price, not matched-behavior.
    """

    #: rescale to E_t[w] == 1 over t = 0..T-1, i.e. divide by (T+1)/(2T)
    _MEAN_ONE = False

    def evaluate_actions(self, obs, actions):
        values, log_prob, entropy = super().evaluate_actions(obs, actions)
        if entropy is not None:
            # bayes obs layout: [post_mean x K, post_sd x K, ttg]; raw scale
            # (norm_obs is forced off for every policy in this module)
            w = obs[..., -1] / self._horizon
            if self._MEAN_ONE:
                w = w * (2.0 * self._horizon / (self._horizon + 1.0))
            entropy = entropy * w
        return values, log_prob, entropy


class TtgEntropyIndexPolicy(_TtgEntropyMixin, IndexPolicy):
    """A7-t2, L2(arch+algo): ttg-weighted entropy on the index class.

    Run of record (2026-08-01, 20M). Carries the disclosed level confound:
    its average entropy weight is ~1/2, so its read must be made against
    A3's ent_coef marginal. A7-t2n is the same schedule at matched level.
    """


class NormTtgEntropyIndexPolicy(_TtgEntropyMixin, IndexPolicy):
    """A7-t2n, L2(arch+algo): t2's schedule at the flat baseline's level.

    The de-confounded t2: same shape, E[w] = 1. Paired with A7-t2 it
    decomposes the effect — (t2n - crown) is the schedule at matched price,
    (t2 - t2n) is the price at matched shape. Both halves are needed; t2n
    does not supersede t2.
    """

    _MEAN_ONE = True


_BETA0 = 0.5413248546129181          # softplus(_BETA0) == 1.0 exactly


class _TempIndexExtractor(_IndexExtractor):
    """Rung-b scorer + an invariant inverse-temperature head:
    logits = beta(s) * z, beta = softplus(g(pooled, ttg) + _BETA0).

    beta is a symmetric function of the arm set, so scaling the equivariant
    per-arm scores z by it preserves exact equivariance. The head gets its
    own tiny pooling encoder rather than borrowing the critic's ``psi`` —
    the base class keeps actor and critic parameters disjoint, and routing
    value-loss gradients into the actor's temperature would blur t3's read.
    """

    def _build_scorer(self) -> None:
        super()._build_scorer()
        self.tau_enc = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.tau_out = nn.Linear(17, 1)

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        z = super().forward_actor(obs)                          # (N, K)
        x, y, ttg = self._split(obs)
        pooled = self.tau_enc(torch.stack([x, y], dim=2)).mean(dim=1)
        beta = nn.functional.softplus(
            self.tau_out(torch.cat([pooled, ttg], dim=1)) + _BETA0)
        return beta * z


class TempIndexPolicy(IndexPolicy):
    """A7-t3, L2(arch): the learned temperature head.

    Factors the policy into "which arm" (the equivariant scores z) times
    "how sure" (one invariant multiplicative scale beta(s)). The softmax's
    sharpness is modulated by the absolute scale of its scores, so beta is
    a single degree of freedom gradient can move to sharpen late without
    re-coordinating the scorer's output weights — the parametrization fix
    for the same defect t2 attacks in the objective. Generalizes to any
    set-scoring softmax policy.

    ``tau_out`` is zero-initialized AFTER SB3's orthogonal init pass (which
    would otherwise overwrite it), so beta(s) == 1 for every state at init:
    t3 starts exactly on the index-policy manifold — it IS an index policy
    until gradient moves the head. Not at seed-0 IndexPolicy's particular
    point, though: constructing the tau modules consumes global-RNG draws
    before the ortho-init pass, shifting the shared layers' init. The smoke
    test asserts the real property (beta == 1, tau zeroed), not fingerprint
    equality.
    """

    _EXTRACTOR = _TempIndexExtractor

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        nn.init.zeros_(self.mlp_extractor.tau_out.weight)
        nn.init.zeros_(self.mlp_extractor.tau_out.bias)


class _IdTempIndexExtractor(_TempIndexExtractor):
    """t3's extractor with the factorization IDENTIFIED (#E22): z is
    standardized across arms before beta scales it, so the scorer emits only
    ranking + relative gaps and beta is the sole owner of scale. In t3 only
    the product beta*z entered the softmax and nothing pinned the scale of z,
    so temperature leaked into the scorer (measured: the scorer's own entropy
    profile co-inverted, H 0.878 -> 1.208 at 20M). Here it cannot.

    mean/var are symmetric functions of the arm set, so exact equivariance is
    preserved. At the symmetric t=0 belief z is exactly constant, zhat == 0,
    and the policy is uniform for ANY beta — the correct equivariant answer;
    the 1/sqrt(var+eps) backward factors there are bounded by _EPS and the
    config's max_grad_norm clips the rest.
    """

    _EPS = 1e-5          # inside the sqrt (LayerNorm convention)

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        z = _IndexExtractor.forward_actor(self, obs)             # (N, K)
        zhat = (z - z.mean(dim=1, keepdim=True)) / torch.sqrt(
            z.var(dim=1, unbiased=False, keepdim=True) + self._EPS)
        x, y, ttg = self._split(obs)
        pooled = self.tau_enc(torch.stack([x, y], dim=2)).mean(dim=1)
        beta = nn.functional.softplus(
            self.tau_out(torch.cat([pooled, ttg], dim=1)) + _BETA0)
        return beta * zhat


class IdTempIndexPolicy(TempIndexPolicy):
    """A8 rungs b/c, L2(arch): the identified temperature head (#E22).

    logits = beta(s) * zhat with zhat unit-spread by construction, so beta is
    the ONLY sharpness dof — the repair for t3's unidentified beta<->||z||
    null direction. tau_out zero-init (inherited) keeps beta == 1 at init.
    Under ent_coef ~ 0 (A8 rung b) the trained beta(s) is a price-free,
    return-driven confidence schedule and doubles as the instrument for the
    t4 pegging question: regress it on ttg vs H2(top-two gap).
    """

    _EXTRACTOR = _IdTempIndexExtractor


# --- the composed rungs (2026-08-01) ---------------------------------------
# The beta probe (results/{scenario}/beta_probe) read t3's learned beta back
# off both its checkpoints: beta > 1 everywhere and dH < 0, so beta SHARPENS —
# there is no entropy hijack. What it learned is a near-perfectly monotone
# anneal with the sign INVERTED (corr(beta, t) = -0.995 at 20M): sharp when
# the posterior is wide, soft when it is sharp — the opposite of Thompson,
# and a steeper version of the crown's pathology (explore 10% -> 40%).
#
# That is exactly what the ttg-scaling of the advantage magnitude predicts, so
# t3 alone was never going to work: beta is optimized by the same mis-scaled
# objective and a dedicated dof merely lets it express the mis-scaling more
# cleanly. But it also proves beta CAN learn a sharp uncertainty-keyed
# schedule — it has the mechanism and converges on it decisively; it just
# points downhill. Fix the exchange rate (t2) and the dof should reverse.

class TtgEntropyTempIndexPolicy(_TtgEntropyMixin, TempIndexPolicy):
    """A7-t2-t3: t2's objective + t3's temperature head, level unmatched."""


class NormTtgEntropyTempIndexPolicy(_TtgEntropyMixin, TempIndexPolicy):
    """A7-t2n-t3: t2's schedule at matched level + t3's temperature head.

    The cell the beta probe actually predicts: a correct entropy-to-advantage
    exchange rate, and a dedicated dof free to express the schedule it implies.
    """

    _MEAN_ONE = True


class _IndexHExtractor(_IndexExtractor):
    """#E36 generalist extractor: per-arm phi over (pm, psd, ttg, T).

    Two differences from `_IndexExtractor`, both load-bearing for the readback:

    1. **The horizon is an INPUT, not a constant.** Under a horizon-varying
       sampler the episode's T changes per episode, so it arrives in the
       observation (`bayes_h`) instead of being baked in at construction.

    2. **ttg is NOT divided by the episode's horizon.** The parent returns
       `ttg / self.horizon`, which hands phi the ratio ttg/T — precisely the
       form #E26 fitted and #E36 exists to TEST, so passing it would beg the
       question. Both time features are scaled by the CONSTANT `t_max`
       instead: a change of units, which fixes conditioning (T ~ 1e4 into an
       unnormalised MLP) without manufacturing the ratio.

    `norm_obs` stays off, as for every equivariant policy — VecNormalize
    normalises each observation DIMENSION independently, so arm slots i and j
    would acquire different running stats and the permutation equivariance
    that is this class's entire point would break.
    """

    def __init__(self, n_arms: int, horizon: int, obs_mode: str = "bayes_h",
                 prior_mean: float = 0.0) -> None:
        assert obs_mode == "bayes_h", (
            "_IndexHExtractor reads the episode horizon from the observation, "
            f"so it requires obs_mode='bayes_h'; got {obs_mode!r}")
        super().__init__(n_arms, horizon, obs_mode, prior_mean)

    def _build_scorer(self) -> None:
        # 4 features per arm: pm, psd, ttg/t_max, T/t_max
        self.phi = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def _split(self, obs: torch.Tensor):
        """[pm x K, psd x K, ttg, T] -> (pm, psd, ttg/t_max, T/t_max)."""
        K = self.n_arms
        x, y = obs[:, :K], obs[:, K:2 * K]
        ttg = obs[:, 2 * K:2 * K + 1]
        T = obs[:, 2 * K + 1:2 * K + 2]
        t_max = self.horizon              # the grid's family maximum: a constant
        return x, y, ttg / t_max, T / t_max

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        x, y, ttg, T = self._split(obs)
        feats = torch.stack([x, y, ttg.expand(-1, self.n_arms),
                             T.expand(-1, self.n_arms)], dim=2)
        return self.phi(feats).squeeze(-1)                     # (N, K) logits

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        x, y, ttg, T = self._split(obs)
        pooled = self.psi(torch.stack([x, y], dim=2)).mean(dim=1)
        return self.vhead(torch.cat([pooled, ttg], dim=1))     # (N, 64)

    def forward(self, obs: torch.Tensor):
        return self.forward_actor(obs), self.forward_critic(obs)


class IndexHPolicy(IndexPolicy):
    """IndexPolicy over `bayes_h` — the #E36 generalist over varying T."""

    _EXTRACTOR = _IndexHExtractor


if __name__ == "__main__":
    # smoke: equivariance of the untrained policy, exact by construction
    import numpy as np
    from stable_baselines3 import PPO
    from mab_gym import MabEnv
    from mab_scenarios import SCENARIOS

    env = MabEnv(scenario=SCENARIOS["gauss_K10_T1000"], observation_mode="bayes")
    # Probe an ASYMMETRIC state. At t=0 every arm carries the same (mean=0,
    # sd=1) prior, so permuting the obs is the identity there: the check
    # degenerates to "identical arms are treated identically", and an
    # equivariant policy is exactly uniform. Ten mixed pulls give arms
    # distinct posteriors, so the permutation actually rearranges the input.
    obs, _ = env.reset(seed=3)
    for a in (0, 0, 1, 2, 2, 2, 3, 5, 5, 7):
        obs, *_ = env.step(a)
    obs = obs[np.newaxis]

    # unit: the top-2 trick must equal a brute-force leave-one-out max. This
    # is not implied by the equivariance check below — plain max is equivariant
    # too, so a degraded pool would pass it silently.
    torch.manual_seed(0)
    pool = _LooMaxLayer(1, 1)._pool
    for h in (torch.randn(4, 10, 3),
              torch.zeros(4, 10, 3),                       # every arm tied
              torch.cat([torch.zeros(4, 6, 3),             # ties + distinct
                         torch.randn(4, 4, 3)], dim=1)):
        want = torch.stack([torch.cat([h[:, :i], h[:, i + 1:]], dim=1)
                            .max(dim=1).values
                            for i in range(h.shape[1])], dim=1)
        assert torch.allclose(pool(h), want), "leave-one-out max is wrong"
    print("smoke OK: leave-one-out pooling matches brute force (incl. ties)")

    # rung d's premise is that learned alpha SPANS the sibling statistics.
    # Check both endpoints are actually reachable: zero query/key scale must
    # give uniform-over-others (rung c), large scale must give a one-hot pick
    # (the rung c-max shape, though selected by a learned score, not by value).
    lay, h = _AttnLayer(4, 4), torch.randn(2, 10, 4)
    with torch.no_grad():
        lay.q_w.weight.zero_(), lay.k_w.weight.zero_()
        others = (h.sum(dim=1, keepdim=True) - h) / (h.shape[1] - 1)
        assert torch.allclose(lay._pool(h), others, atol=1e-6), \
            "alpha is not uniform-over-others at zero query/key scale"
        lay.q_w.weight.normal_(0, 8), lay.k_w.weight.normal_(0, 8)
        att = lay.q_w(h) @ lay.k_w(h).transpose(1, 2) / lay.D_K ** 0.5
        att = att.masked_fill(torch.eye(10, dtype=torch.bool), float("-inf"))
        peak = torch.softmax(att, dim=-1).max(dim=-1).values.mean()
    assert peak > 0.95, f"alpha cannot sharpen toward one-hot: {peak:.3f}"
    print(f"smoke OK: attention alpha spans uniform -> one-hot (peak {peak:.3f})")

    fingerprints, scorer_sizes = {}, {}
    for cls in (IndexPolicy, DeepSetsPolicy, DeepSetsMaxPolicy, AttentionPolicy,
                TtgEntropyIndexPolicy, TempIndexPolicy,
                NormTtgEntropyIndexPolicy, TtgEntropyTempIndexPolicy,
                NormTtgEntropyTempIndexPolicy, IdTempIndexPolicy):
        model = PPO(cls, env, policy_kwargs={"horizon": 1000},
                    n_steps=64, batch_size=32, device="cpu", seed=0)
        rng = np.random.default_rng(0)
        with torch.no_grad():
            base = model.policy.get_distribution(
                torch.as_tensor(obs)).distribution.probs.numpy()[0]
            for _ in range(5):
                perm = rng.permutation(10)
                po = np.concatenate([obs[:, :10][:, perm], obs[:, 10:20][:, perm],
                                     obs[:, 20:]], axis=1)
                pp = model.policy.get_distribution(
                    torch.as_tensor(po)).distribution.probs.numpy()[0]
                assert np.allclose(pp, base[perm], atol=1e-6), \
                    f"{cls.__name__} is not equivariant!"
        n_par = sum(p.numel() for p in model.policy.parameters())
        n_act = sum(p.numel() for p in model.policy.mlp_extractor.phi.parameters())
        fingerprints[cls.__name__] = float(base[0])
        scorer_sizes[cls.__name__] = n_act
        model.learn(total_timesteps=256)
        print(f"smoke OK: {cls.__name__:15s} exactly equivariant, trains  "
              f"(scorer {n_act:,} par, policy {n_par:,} par)")

    # rung b's crowned run must stay reproducible across the rung-c refactor.
    # Factoring the scorer out into _build_scorer() keeps submodule
    # registration order — hence SB3's orthogonal-init RNG draws — identical;
    # 0.10072249 is what HEAD~ produced at this state under seed 0.
    assert abs(fingerprints["IndexPolicy"] - 0.10072249) < 1e-7, \
        f"IndexPolicy init drifted: {fingerprints['IndexPolicy']!r}"
    # the four ARCH rungs must differ at init; t2 must NOT — it is the same
    # network under a different objective, so its init is byte-identical
    arch = {k: fingerprints[k] for k in
            ("IndexPolicy", "DeepSetsPolicy", "DeepSetsMaxPolicy",
             "AttentionPolicy")}
    assert len(set(arch.values())) == len(arch), \
        f"arch rungs cannot share an init fingerprint: {arch}"
    assert fingerprints["TtgEntropyIndexPolicy"] == fingerprints["IndexPolicy"], \
        "t2 is the same network — init must be byte-identical"
    # the mixin refactor must not have moved t2, and t2n is the same net again:
    # _TtgEntropyMixin registers no modules, so it cannot consume RNG draws
    assert fingerprints["NormTtgEntropyIndexPolicy"] == fingerprints["IndexPolicy"], \
        "t2n is the same network as the index class — init must be identical"

    # the entropy weight is the ONLY thing separating t2 from t2n, and their
    # ratio must be exactly the mean-one rescale 2T/(T+1) at every t
    T_ = 1000.0
    probe = torch.as_tensor(
        np.stack([np.concatenate([obs[0, :20], [ttg]]).astype(np.float32)
                  for ttg in (1.0, 250.0, 500.0, 1000.0)]))
    acts = torch.zeros(len(probe), dtype=torch.long)
    # The weight must be read against the UNWEIGHTED twin, not against this
    # policy's own ttg=T row: ttg is an input to the scorer, so H varies with
    # ttg on its own. Each pair below is the same network at seed 0 (the mixin
    # registers no modules), so the ratio isolates the weight exactly.
    for flat_cls, base_cls, norm_cls in (
            (IndexPolicy, TtgEntropyIndexPolicy, NormTtgEntropyIndexPolicy),
            (TempIndexPolicy, TtgEntropyTempIndexPolicy,
             NormTtgEntropyTempIndexPolicy)):
        ent = {}
        for cls in (flat_cls, base_cls, norm_cls):
            m = PPO(cls, env, policy_kwargs={"horizon": 1000},
                    n_steps=64, batch_size=32, device="cpu", seed=0)
            with torch.no_grad():
                ent[cls] = m.policy.evaluate_actions(probe, acts)[2]
        w = (ent[base_cls] / ent[flat_cls]).numpy()
        assert np.allclose(w, [1 / T_, 250 / T_, 500 / T_, 1.0], atol=1e-6), \
            f"{base_cls.__name__} weight is not ttg/T: {w}"
        wn = (ent[norm_cls] / ent[flat_cls]).numpy()
        assert np.allclose(wn, w * 2.0 * T_ / (T_ + 1.0), atol=1e-6), \
            f"{norm_cls.__name__} rescale is not 2T/(T+1): {wn / w}"
        assert abs(wn.mean() - np.mean([w_ / T_ for w_ in range(1, 1001)])
                   * 2 * T_ / (T_ + 1)) < 0.5, "mean-one rescale sanity"
    print("smoke OK: ttg weight is ttg/T; the norm variants rescale by "
          f"2T/(T+1) = {2 * T_ / (T_ + 1):.6f} (mean weight 1)")

    # the composed rungs must still start on the index manifold (beta == 1):
    # t2 changes only the objective, so it cannot move t3's init property
    for cls in (TtgEntropyTempIndexPolicy, NormTtgEntropyTempIndexPolicy):
        m = PPO(cls, env, policy_kwargs={"horizon": 1000},
                n_steps=64, batch_size=32, device="cpu", seed=0)
        ex_ = m.policy.mlp_extractor
        with torch.no_grad():
            x_, y_, ttg_ = ex_._split(torch.as_tensor(obs))
            pooled_ = ex_.tau_enc(torch.stack([x_, y_], dim=2)).mean(dim=1)
            b_ = nn.functional.softplus(
                ex_.tau_out(torch.cat([pooled_, ttg_], dim=1)) + _BETA0)
        assert torch.allclose(b_, torch.ones_like(b_)), \
            f"{cls.__name__} must start at beta == 1: {b_}"
    print("smoke OK: both composed rungs start at beta == 1 (index manifold)")

    # A8's identified head: beta == 1 at init, zhat unit-spread on asymmetric
    # states, and exactly uniform at the symmetric t=0 belief for any beta
    m = PPO(IdTempIndexPolicy, env, policy_kwargs={"horizon": 1000},
            n_steps=64, batch_size=32, device="cpu", seed=0)
    ex_ = m.policy.mlp_extractor
    assert ex_.tau_out.weight.abs().max().item() == 0.0
    with torch.no_grad():
        z_ = _IndexExtractor.forward_actor(ex_, torch.as_tensor(obs))
        zh_ = (z_ - z_.mean(dim=1, keepdim=True)) / torch.sqrt(
            z_.var(dim=1, unbiased=False, keepdim=True) + ex_._EPS)
        assert abs(zh_.var(dim=1, unbiased=False).item() - 1.0) < 1e-3, \
            f"zhat not unit-var: {zh_.var(dim=1, unbiased=False)}"
        obs0, _ = env.reset(seed=5)                   # symmetric t=0 belief
        p0 = m.policy.get_distribution(
            torch.as_tensor(obs0[np.newaxis].astype(np.float32))
        ).distribution.probs
        assert torch.allclose(p0, torch.full_like(p0, 0.1), atol=1e-6), \
            f"t=0 not uniform: {p0}"
    print("smoke OK: IdTempIndexPolicy — beta==1 at init, zhat unit-spread, "
          "uniform at the symmetric t=0 state")

    # t3's manifold property is beta == 1 everywhere at init (tau zeroed after
    # the ortho pass), NOT fingerprint equality with IndexPolicy: building the
    # tau modules consumes global-RNG draws before ortho init, so the shared
    # layers draw differently. beta == 1 is what makes it an index policy.
    model = PPO(TempIndexPolicy, env, policy_kwargs={"horizon": 1000},
                n_steps=64, batch_size=32, device="cpu", seed=0)
    ex = model.policy.mlp_extractor
    assert ex.tau_out.weight.abs().max().item() == 0.0
    assert ex.tau_out.bias.abs().max().item() == 0.0
    with torch.no_grad():
        x, y, ttg = ex._split(torch.as_tensor(obs))
        pooled = ex.tau_enc(torch.stack([x, y], dim=2)).mean(dim=1)
        beta = nn.functional.softplus(
            ex.tau_out(torch.cat([pooled, ttg], dim=1)) + _BETA0)
    assert torch.allclose(beta, torch.ones_like(beta)), f"beta != 1: {beta}"
    print("smoke OK: IndexPolicy init unchanged by the rung-c refactor; "
          "t2 byte-identical; t3 starts at beta == 1 (index manifold)")
    assert scorer_sizes["DeepSetsPolicy"] == scorer_sizes["DeepSetsMaxPolicy"], \
        "the rung-c siblings must differ only in the pooling statistic"
    print("smoke OK: both rung-c siblings are the same size "
          f"({scorer_sizes['DeepSetsPolicy']:,} scorer par) — pooling differs")

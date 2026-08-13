"""#E31 rung 5 — forward-KL-to-Thompson anchor for PPO (COVERAGE_PLAN.md).

The coverage round's verdict: the return landscape prefers near-full
coverage (P3: +23/seed at c ≈ 2.2–2.5, +251/episode on the starved 12%),
but PPO cannot climb there because the on-policy gradient for revisit
actions dies within ~20 rounds of every episode (P2: eventually-untouched
arms get R0 median 0.5 expected tries, half spent by t = 0), leaving the
policy's floor wherever across-training entropy decay sits at the
selection snapshot (P4: corr(entropy, starvation) = −0.92).

The anchor adds ``+ anchor_coef * KL(pi_TS || pi)`` to the PPO loss, where
``pi_TS(a | belief)`` is Thompson sampling's action law — the probability
of maximality of each arm under the exact conjugate posterior the
observation encodes. The FORWARD direction is load-bearing: it diverges as
pi(a) -> 0 on any arm pi_TS still gives mass to, so it maintains exactly
the two things the round showed die together — coverage of
wide-posterior arms AND the gradient channel through them. Unlike the
entropy bonus (= KL to uniform), the pressure is posterior-targeted and
self-annealing: as posteriors sharpen, pi_TS itself concentrates and the
penalty stops fighting exploitation, no schedule needed. Unlike the old
crown's ent 0.01 subsidy, it does not pay the uniform-floor premium
(#E31 rung 0: oldcrown bought 8x coverage for +71/seed of diffuse regret).

pi_TS is computed exactly per minibatch by deterministic quadrature — NOT
Monte Carlo: MC's 1/draws granularity floors out exactly the small tail
probabilities the anchor exists to protect —

    pi_TS(a) = int phi_a(x) * prod_{j != a} Phi((x - pm_j)/psd_j) dx

on a per-state union grid (every arm contributes nodes at pm_j + psd_j *
{-5 .. +5, step 1/4}, sorted; trapezoid in between), evaluated in log
space (torch.special.log_ndtr). The union grid is the load-bearing
choice: a heavily-pulled arm's cdf is a near-step, and any fixed rule
centered on the candidate arm (Gauss–Hermite included — tried first,
halved the tail mass in the gate) integrates through the step blind;
here every transition region is resolved by that arm's own nodes.
Gaussian branch only (Beta posteriors would need different quadrature);
asserted at construction.

Trains-only, like the A7 shaping wrapper: selection and eval score the
unmodified stochastic policy on the faithful payout.
"""

from __future__ import annotations

import numpy as np
import torch as th
from torch.nn import functional as F  # noqa: N812 — SB3's import style
from gymnasium import spaces

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

#: train() below is a copy of this SB3 version's PPO.train with the anchor
#: block inserted; a version bump means re-diffing, not trusting.
_SB3_COPIED_FROM = "2.6.0"
assert stable_baselines3.__version__ == _SB3_COPIED_FROM, (
    f"mab_kl_anchor copied PPO.train from SB3 {_SB3_COPIED_FROM}, found "
    f"{stable_baselines3.__version__}; re-diff train() before using."
)

#: per-arm node ladder: pm + psd * _OFFSETS, pooled over arms and sorted.
#: 1/2-sigma spacing over +-5 sigma puts ~20 trapezoid intervals through
#: every arm's density spike and cdf transition, however sharp — gate
#: accuracy ~2e-3 (mab_kl_anchor gate block in COVERAGE_PLAN.md), an
#: order below anything the anchor coefficient can resolve.
_OFFSETS = th.arange(-5.0, 5.001, 0.5, dtype=th.float32)
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def thompson_probs(pm: th.Tensor, psd: th.Tensor) -> th.Tensor:
    """P(arm = argmax of independent posterior draws), (B, K) -> (B, K).

    Exact conjugate law by union-grid trapezoid quadrature in log space;
    renormalized on the way out. No gradients flow (stop-grad target by
    construction — callers need no no_grad())."""
    with th.no_grad():
        pm64 = pm.to(th.float32)
        psd64 = psd.to(th.float32).clamp_min(1e-9)
        b = pm64.shape[0]
        nodes = (pm64.unsqueeze(-1) + psd64.unsqueeze(-1) * _OFFSETS)
        nodes, _ = th.sort(nodes.reshape(b, -1), dim=1)          # (B, N)
        z = ((nodes.unsqueeze(1) - pm64.unsqueeze(-1))
             / psd64.unsqueeze(-1))                              # (B, K, N)
        log_cdf = th.special.log_ndtr(z)
        log_pdf = (-0.5 * z * z - _LOG_SQRT_2PI
                   - th.log(psd64).unsqueeze(-1))
        # integrand for arm a: pdf_a * prod_{j != a} cdf_j
        g = th.exp(log_pdf + log_cdf.sum(dim=1, keepdim=True) - log_cdf)
        dx = (nodes[:, 1:] - nodes[:, :-1]).unsqueeze(1)         # (B, 1, N-1)
        p = 0.5 * ((g[:, :, 1:] + g[:, :, :-1]) * dx).sum(dim=-1)
        return (p / p.sum(dim=1, keepdim=True)).to(pm.dtype)


def belief_from_obs(obs: th.Tensor, obs_mode: str) -> tuple[th.Tensor, th.Tensor]:
    """(pm, psd) from a mab_gym observation batch, gaussian branch.

    bayes: [pm x K, psd x K, ttg] verbatim. stats: [pulls x K, payouts x K,
    ttg] -> the conjugate transform (prior N(0,1), sigma = 1)."""
    n_arms = (obs.shape[1] - 1) // 2
    a, b = obs[:, :n_arms], obs[:, n_arms:2 * n_arms]
    if obs_mode == "bayes":
        return a, b
    return b / (1.0 + a), th.sqrt(1.0 / (1.0 + a))


class KlAnchorPPO(PPO):
    """PPO + anchor_coef * KL(pi_TS || pi). Construction args beyond PPO's:
    anchor_coef (the beta) and anchor_obs_mode ('bayes' | 'stats')."""

    def __init__(self, *sb3_args, anchor_coef: float = 0.0,
                 anchor_obs_mode: str = "bayes", **sb3_kwargs):
        self.anchor_coef = float(anchor_coef)
        self.anchor_obs_mode = str(anchor_obs_mode)
        assert self.anchor_obs_mode in ("bayes", "stats")
        super().__init__(*sb3_args, **sb3_kwargs)

    def train(self) -> None:
        # ---- verbatim SB3 2.6.0 PPO.train, + the [anchor] blocks ----
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        anchor_kls, anchor_ces = [], []                    # [anchor]

        # [anchor] pi_TS depends only on the observation and the buffer is
        # fixed for all epochs, so the quadrature runs ONCE per rollout
        # (5120 per-minibatch calls would double the wall time). Minibatches
        # are drawn by an explicit index loop replicating stock get() —
        # fresh full permutation per epoch, same _get_samples — so ts rows
        # stay aligned. The first next() forces the storage flattening
        # stock get() performs; its full-batch draw is discarded.
        buf = self.rollout_buffer
        total = buf.buffer_size * buf.n_envs
        gen = buf.get(total)
        next(gen)
        del gen
        if self.anchor_coef > 0:
            obs_all = th.as_tensor(buf.observations, device=self.device)
            pm_all, psd_all = belief_from_obs(obs_all, self.anchor_obs_mode)
            ts_all = th.cat([thompson_probs(pm_all[i:i + 2048],
                                            psd_all[i:i + 2048])
                             for i in range(0, total, 2048)])
            with th.no_grad():
                ts_ent_all = -(ts_all * th.log(ts_all.clamp_min(1e-12))).sum(dim=1)

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            epoch_indices = np.random.permutation(total)
            for start in range(0, total, self.batch_size):
                minibatch_inds = epoch_indices[start:start + self.batch_size]
                rollout_data = buf._get_samples(minibatch_inds)
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # [anchor] forward KL to the Thompson action law. The
                # distribution forward pass carries gradients; pi_TS is the
                # per-rollout stop-grad quadrature target, index-aligned.
                if self.anchor_coef > 0:
                    ts = ts_all[minibatch_inds]
                    dist = self.policy.get_distribution(rollout_data.observations)
                    logp_all = dist.distribution.logits    # log-normalized
                    anchor_ce = -(ts * logp_all).sum(dim=1).mean()
                    anchor_loss = anchor_ce
                    anchor_kls.append(
                        (anchor_ce - ts_ent_all[minibatch_inds].mean()).item())
                    anchor_ces.append(anchor_ce.item())
                else:
                    anchor_loss = th.zeros((), device=policy_loss.device)

                loss = (policy_loss + self.ent_coef * entropy_loss
                        + self.anchor_coef * anchor_loss              # [anchor]
                        + self.vf_coef * value_loss)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if anchor_kls:                                     # [anchor]
            self.logger.record("train/anchor_kl", np.mean(anchor_kls))
            self.logger.record("train/anchor_ce", np.mean(anchor_ces))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

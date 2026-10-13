"""A learned compression of the raw frame stack, then the spread treatment (#E68).

The raw modes hand the policy 100 frames x 5 numbers and no statistic; the network
must build one. Standardising the frames per position (VecNormalize) squashes the
low counts (0, 1, 2 sit 0.10 apart) and clips the high ones, the same input-resolution
failure the belief modes had (#E56) and the oracle had (#E60). The human's design
(2026-09-28) splits the job in two, assuming nothing about what the statistic is:

  1. COMPRESSION on the raw, un-normalised frames: an MLP with a NON-SATURATING
     activation (a tanh first layer saturates on raw counts — the raw-input oracle's
     failure, #E61) and no normalisation anywhere inside it. A sum over frames is one
     function it can represent; nothing makes it choose that.
  2. THE SPREAD: the compressed features are widely spread (running totals live here),
     so they get the treatment the oracle validated (#E60): a log-like compression —
     asinh, since a learned feature can be negative — then per-feature RUNNING
     standardisation (VecNormalize's rule, moved inside the network; never a per-sample
     LayerNorm, which would erase the magnitude that is the signal).
  3. The existing tanh body and the dgauss head, unchanged.

Run with `--no-norm-obs`: the standardisation now lives after the compression.
Shared by actor and critic (SB3's default for the features extractor). The running
statistics are persistent buffers, so a saved model evaluates with the statistics it
trained with; they update only in training mode (SB3 sets that during `train()`).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class RunningStandardize(nn.Module):
    """Per-feature running mean / variance (EMA), applied as (x - mean) / sd, clipped.

    VecNormalize's TIMING (fixes of 2026-09-29, #E71 and its review): the statistics are
    refreshed ONCE PER ITERATION by `RawStatsCallback`, and held fixed through BOTH the
    rollout collection and the PPO update that consumes it. The first build updated them on
    every minibatch during the update — 160 times per rollout — so the same observation was
    standardised differently in epoch 1 and epoch 10 (a policy-ratio drift PPO's clip then
    fought) and the critic's input drifted under it with no trust region: the periodic value
    collapses and crashes of #E68–#E70. The second build (#E71–#E73) updated them at rollout
    END, before the update: the stored log-probabilities were computed under the old
    statistics and the update recomputed them under the new — a ratio that does not start at
    one (measured on the #E72 model: mean |Δ log p| 0.024, approx KL 7e-4 per iteration).
    Now the update is COMPUTED at rollout end and APPLIED at the next rollout start; the
    stats a rollout is collected under are the ones its update sees."""

    def __init__(self, n: int, momentum: float = 0.01, clip: float = 10.0, freeze: bool = False, freeze_at: int = 1) -> None:
        super().__init__()
        # freeze (#E77): the statistics stop after `freeze_at` refreshes — 1 = the first rollout sets them (E77);
        # K > 1 (#E78): running through the first K rollouts (the EMA's ~100-rollout memory then holds the regime the
        # policy has reached by then), frozen after.
        self.momentum, self.clip, self.freeze, self.freeze_at = float(momentum), float(clip), bool(freeze), int(freeze_at)
        self.register_buffer("mean", th.zeros(n))
        self.register_buffer("var", th.ones(n))
        self.register_buffer("count", th.zeros(()))

    @th.no_grad()
    def update(self, x: th.Tensor) -> None:
        if self.freeze and self.count >= self.freeze_at:
            return
        m = 1.0 if self.count == 0 else self.momentum   # the first rollout sets the statistics
        self.mean.mul_(1 - m).add_(m * x.mean(0))
        self.var.mul_(1 - m).add_(m * x.var(0, unbiased=False))
        self.count += 1

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.clamp((x - self.mean) / th.sqrt(self.var + 1e-8), -self.clip, self.clip)


class RawCompressExtractor(BaseFeaturesExtractor):
    """``mix=True`` (#E69, iteration 2): a linear layer AFTER the asinh. #E68's probe showed the
    compression rediscovers the Bayesian SUMS (cumulative demand, periods played; R² 0.997 /
    0.999) — the right statistics in the wrong coordinates: the policy needs their RATIO, and
    log(sum) − log(count) is a DIFFERENCE of asinh'd features, i.e. linear after the asinh. The
    mixing layer gives the network that arithmetic where it is linear; nothing makes it use it."""

    def __init__(self, observation_space: gym.Space, hidden: int = 128, features: int = 32,
                 momentum: float = 0.01, mix: bool = False, asinh: bool = True, depth: int = 1) -> None:
        # asinh=False (#E70): the human's split with nothing added — compression, standardisation,
        # body — so the network must learn its own scale; depth=3: three GELU layers, capacity for it.
        super().__init__(observation_space, features_dim=features)
        n_in = int(th.tensor(observation_space.shape).prod())
        layers = [nn.Flatten(), nn.Linear(n_in, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, features)]
        self.compress = nn.Sequential(*layers)
        self.use_asinh = bool(asinh)
        self.mix = nn.Linear(features, features) if mix else None
        self.standardize = RunningStandardize(features, momentum)

    def pre(self, obs: th.Tensor) -> th.Tensor:
        """The features BEFORE standardisation — what the running statistics are computed on."""
        z = self.compress(obs)
        if self.use_asinh:
            z = th.asinh(z)
        if self.mix is not None:
            z = self.mix(z)
        return z

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.standardize(self.pre(obs))


class RawStatsCallback:
    """SB3 callback with VecNormalize's timing for the extractor's running statistics: at the
    end of every rollout, COMPUTE the refresh from that rollout's observations; at the start of
    the next rollout, APPLY it. The statistics are therefore fixed across a rollout's collection
    AND the PPO update that consumes it (the stored log-probabilities and the recomputed ones
    agree; the probability ratio starts at one), and change only between an update and the next
    collection. The first iteration trains under the identity statistics (mean 0, var 1).

    Check (logged as `train/rawstats_logp_maxdiff`, asserted on the first `check_rollouts`
    rollouts): recomputing the rollout's log-probabilities before the update reproduces the
    stored ones. Built lazily as a BaseCallback so this module stays importable without SB3's
    callback machinery at eval time."""

    def __new__(cls, check_rollouts: int = 3, tol: float = 1e-4):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Cb(BaseCallback):
            def __init__(self):
                super().__init__()
                self._pending = None
                self._checked = 0

            def _on_step(self) -> bool:
                return True

            def _on_rollout_start(self) -> None:
                if getattr(self.model.policy.features_extractor, "standardize", None) is None:
                    return   # #E76: no statistics to refresh
                if self._pending is not None:
                    fx = self.model.policy.features_extractor
                    with th.no_grad():   # the refresh uses the CURRENT extractor weights on the last rollout's observations
                        z = fx.pre(self._pending)
                        fx.standardize.update(z[:, : fx.standardize.mean.numel()] if getattr(fx, "stats_freeze", False) else z)
                    self._pending = None
                    self.logger.record("train/rawstats_updates", int(fx.standardize.count))

            def _on_rollout_end(self) -> None:
                fx = self.model.policy.features_extractor
                if getattr(fx, "standardize", None) is None:
                    return
                buf = self.model.rollout_buffer
                obs = th.as_tensor(buf.observations, device=self.model.device)
                obs = obs.reshape(-1, *obs.shape[2:]) if obs.dim() > 2 else obs
                obs = obs.float()
                if self._checked < check_rollouts:
                    # under the statistics in force, the stored log-probabilities must reproduce
                    acts = th.as_tensor(buf.actions, device=self.model.device).reshape(obs.shape[0], -1)
                    if isinstance(self.model.action_space, gym.spaces.Discrete):
                        acts = acts.long().flatten()
                    with th.no_grad():
                        _, logp, _ = self.model.policy.evaluate_actions(obs, acts)
                    diff = float((logp - th.as_tensor(buf.log_probs, device=self.model.device).flatten()).abs().max())
                    self.logger.record("train/rawstats_logp_maxdiff", diff)
                    self._checked += 1
                    assert diff < tol, f"raw statistics changed inside a rollout: max |dlogp| {diff:.2e}"
                self._pending = obs   # the refresh is applied at the next rollout start, not now

        return _Cb()


class FrameSumExtractor(BaseFeaturesExtractor):
    """The per-frame log-likelihood form (#E72): a transform shared across frames, then a sum
    over the real frames of the stack.

    The human's framing (2026-09-29): the exact log-likelihood of an episode's observations is
    ADDITIVE over independent periods — for full observation a Poisson term per period, under
    censoring a survival term per censored period at the order actually placed — so the
    sufficient statistic is a sum over frames of a per-frame function. It has no closed form
    under event observation (the censored terms depend on each period's level), which is why
    no two-number compression exists there; this network learns the per-frame function as a
    bank of k channels and sums them, so it can approximate the exact additive statistic as
    closely as k allows, assuming nothing about what it is.

    phi:  a 1-by-F map (one frame's features) -> k channels with a GELU, the SAME weights on
          every frame — 6*k weights instead of the MLP's 500*128.
    mask: a frame is real iff its time-to-go is >= 1 (the stack zero-pads before the episode
          starts) — derived from the input, not from the domain; padded frames contribute 0.
    pool: "sum" over real frames (the log-likelihood's own aggregation); "mean" = sum / count
          with the count as an extra channel (stationary features); "kernel" = the human's
          column kernel — a learned length-T weight per channel down the time axis,
          initialised to ones (a sum), free to learn a recency weighting.
    Then the running standardisation (once per rollout, fixed through the epochs) and the body.

    v2 (the corrected form, review of 2026-09-29, #E74 — Deep Sets done properly):
      - the RESET frame is excluded from the pool. The stack's oldest real frame is the reset
        observation (time-to-go = N, zero sales / demand, a full period's duration): pooled as
        a likelihood term it is a phantom zero-demand period, which against a diffuse prior is
        not small. Real history = 1 <= time-to-go <= N - 1.
      - phi sees the OBSERVATION features only (columns 4..: demand / sales and the stock-out
        flag or time), not the frame's time index — the sum assumes exchangeability given the
        rate, so the per-frame map has no use for time.
      - the present state (the newest frame's time-to-go, critical ratio, prior rate and mean)
        is concatenated to the pooled statistic and reaches the body directly, instead of as
        one summand among N (what the #E72 kernel's gain on the newest frame compensated for).
    """

    N_STATE = 4   # leading frame columns: time-to-go, critical ratio, prior rate, prior mean

    def __init__(self, observation_space: gym.Space, channels: int = 32, hidden: int = 64,
                 pool: str = "sum", momentum: float = 0.01, v2: bool = False, phi_time: bool = False,
                 standardize: bool = True, units: bool = True, stats_freeze: bool = False, freeze_at: int = 1) -> None:
        T, F = observation_space.shape                      # (frames, features)
        n_out = channels + (1 if pool == "mean" else 0) + (self.N_STATE if v2 else 0)
        super().__init__(observation_space, features_dim=n_out)
        self.T, self.F, self.pool, self.v2, self.phi_time = int(T), int(F), pool, bool(v2), bool(phi_time)
        self.horizon = float(np.asarray(observation_space.high)[..., 0].max())   # the reset frame's time-to-go
        f_in = (F - self.N_STATE + (1 if phi_time else 0)) if v2 else F   # phi_time: the frame's time-to-go stays in phi (v2b)
        self.phi = nn.Sequential(nn.Linear(f_in, hidden), nn.GELU(), nn.Linear(hidden, channels))
        self.kernel = nn.Parameter(th.ones(T, channels)) if pool == "kernel" else None
        # standardize=False (#E76): NO normalisation after the pool — pooled MEANS are bounded and n-independent, the
        # body's first linear layer learns the scale, and no statistic follows the policy-dependent observation
        # distribution of the censored modes (the suspect for #E75's decline after 1-1.6M). The callback is a no-op.
        # stats_freeze (#E77 = E76B + one change): the 32 pooled channels are standardised by the FIRST rollout's
        # mean / sd, frozen (the prior's spread of phi-means, in phi's units); count and state columns stay in fixed
        # units as in E76B. standardize=True (E75) standardises everything with running statistics.
        self.stats_freeze = bool(stats_freeze) and not standardize
        self.standardize = RunningStandardize(n_out, momentum) if standardize else (RunningStandardize(channels, momentum, freeze=True, freeze_at=freeze_at) if self.stats_freeze else None)
        # #E76 option B — fixed UNITS, no statistics: without the standardisation the columns beside the pooled means
        # (time-to-go 0..N, the count 0..N, the prior mean) saturate the tanh body at init (89% of first-layer
        # pre-activations beyond |2| measured), so they are expressed in units the frame defines: time and count in
        # horizons (/N), the prior mean in horizons of demand (/N); ratio and prior rate are already O(1).
        self.units = (not standardize) and bool(units)   # units=False with standardize=False: pure identity (#E76 arm A); with stats_freeze the units still apply to count/state
        self.register_buffer("state_scale", th.tensor([self.horizon, 1.0, 1.0, self.horizon]))

    def pre(self, obs: th.Tensor) -> th.Tensor:
        ttg = obs[..., 0]
        if self.v2:
            real = ((ttg >= 1.0) & (ttg < self.horizon)).float()   # (B, T): history only, reset frame out
            x = th.cat([obs[..., :1], obs[..., self.N_STATE:]], dim=-1) if self.phi_time else obs[..., self.N_STATE:]
            z = self.phi(x) * real[..., None]
            state = obs[:, -1, : self.N_STATE]                        # the present frame's state columns
            pooled = self._pool(z, real)
            if self.units:
                state = state / self.state_scale
                if self.pool == "mean":                                # the count channel, in horizons
                    pooled = th.cat([pooled[:, :-1], pooled[:, -1:] / self.horizon], dim=1)
            return th.cat([pooled, state], dim=1)
        real = (ttg >= 1.0).float()                          # (B, T): time-to-go >= 1 marks a real frame
        z = self.phi(obs) * real[..., None]                  # (B, T, k), padding zeroed
        return self._pool(z, real)

    def _pool(self, z: th.Tensor, real: th.Tensor) -> th.Tensor:
        if self.pool == "kernel":
            return (z * self.kernel[None]).sum(1)
        s = z.sum(1)
        if self.pool == "mean":
            n = real.sum(1, keepdim=True).clamp(min=1.0)
            return th.cat([s / n, n], dim=1)
        return s

    def forward(self, obs: th.Tensor) -> th.Tensor:
        z = self.pre(obs)
        if self.standardize is None:
            return z
        if self.stats_freeze:   # pooled channels only; the rest already in fixed units
            k = self.standardize.mean.numel()
            return th.cat([self.standardize(z[:, :k]), z[:, k:]], dim=1)
        return self.standardize(z)

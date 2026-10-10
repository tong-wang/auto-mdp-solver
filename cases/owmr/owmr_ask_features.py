"""Ask features for owmr (ESCALATION #E7) — an ARCHITECTURE lever.

The IR and the gym are untouched: the policy still emits the `order_softmax`
action [order, logit_1..logit_N] on the `raw` observation, and the network is
SB3's standard flat actor-critic — one global policy MLP for the order and the
allocation (whose anchored softmax also sets the reserve), one global value
MLP. The only change is what that network reads:

- **ask (local, fixed, not learned).** Each retailer's standard single-location
  need, ``a_i = max(0, z*_i - (stock_i + in_transit_i))`` with the balance
  bound's newsvendor target ``z*`` (owmr_benchmark_lb.BalanceBound). The
  campaign does not claim the retailer decision; the ask is handed in as a
  FEATURE so that no gradient can bend it toward the allocation (a jointly
  trained ask would co-adapt with the allocator and stop meaning "need").
- **everything else is global**, from the warehouse's point of view: the full
  raw observation plus every retailer's ask and the asks relative to on-hand.

The ask needs stock in physical units, so this extractor must read an
UN-normalized observation (``norm_obs`` off in VecNormalize). It then
normalizes the finished feature vector itself, with the same running
mean/variance and clip VecNormalize applies (#E8: without it the network sees
backlogs in the hundreds during the start-up collapse and stalls for ~2.5M
steps, #E7). The statistics are buffers, so every checkpoint carries its own;
they update only while ``update_stats`` is on, which the training script
switches on for rollout collection alone (VecNormalize's own schedule) —
gradient passes and every evaluation read them frozen.
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# floor on on-hand in the ratio features, in units of mean retailer demand
_EPS = 0.01
# ratio features (ask / on-hand) are clipped here: W can reach 0
_RATIO_CAP = 10.0
# VecNormalize's constants (clip_obs, epsilon, initial count)
_CLIP = 10.0
_NORM_EPS = 1e-8
_COUNT0 = 1e-4


class AskFeatureExtractor(BaseFeaturesExtractor):
    """raw obs -> [scaled raw obs, a (N), a / W (N), sum a, sum a / W]. No weights."""

    def __init__(self, observation_space: gym.spaces.Box, *, n: int, l0: int, l_rt: int,
                 horizon: int, z_star: Sequence[float], mu_rt: Sequence[float],
                 running_norm: bool = False) -> None:
        n_raw = int(observation_space.shape[0])
        assert n_raw == 2 + l0 + n + n * l_rt, "ask features read the `raw` observation layout"
        dim = n_raw + 2 * n + 2
        super().__init__(observation_space, features_dim=dim)
        self.n, self.l0, self.l_rt = int(n), int(l0), int(l_rt)
        self.horizon = float(horizon)
        self.unit = float(sum(mu_rt) / len(mu_rt))   # mean retailer demand
        self.register_buffer("z_star", th.as_tensor(list(z_star), dtype=th.float32))
        self.running_norm = bool(running_norm)
        self.update_stats = False                    # a switch, never saved: loads frozen
        self.register_buffer("rms_mean", th.zeros(dim, dtype=th.float64))
        self.register_buffer("rms_var", th.ones(dim, dtype=th.float64))
        self.register_buffer("rms_count", th.tensor(_COUNT0, dtype=th.float64))

    @th.no_grad()
    def _update(self, x: th.Tensor) -> None:
        """RunningMeanStd.update (Chan's parallel moments), as VecNormalize."""
        x = x.to(th.float64)
        b_mean, b_var, b_n = x.mean(dim=0), x.var(dim=0, unbiased=False), float(x.shape[0])
        delta = b_mean - self.rms_mean
        tot = self.rms_count + b_n
        m2 = self.rms_var * self.rms_count + b_var * b_n + delta.pow(2) * self.rms_count * b_n / tot
        self.rms_mean += delta * b_n / tot
        self.rms_var.copy_(m2 / tot)
        self.rms_count.copy_(tot)

    def parse(self, obs: th.Tensor) -> dict[str, th.Tensor]:
        """The observation contract (owmr_gym.observation, mode "raw"):
        [time_to_go, wh_stock, wh_pipe (l0), rt_stock (N), rt_pipe (N x l_rt)]."""
        n, l0, lr = self.n, self.l0, self.l_rt
        i = 2 + l0
        rt = obs[:, i:i + n]
        transit = obs[:, i + n:i + n + n * lr].reshape(-1, n, lr).sum(dim=2)
        ask = th.clamp(self.z_star - (rt + transit), min=0.0)
        return {"ttg": obs[:, 0], "W": th.clamp(obs[:, 1], min=0.0), "wh_pipe": obs[:, 2:i],
                "rt": rt, "transit": transit, "ask": ask}

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = self.features(obs)
        if not self.running_norm:
            return x
        if self.update_stats:
            self._update(x)
        z = (x.to(th.float64) - self.rms_mean) / th.sqrt(self.rms_var + _NORM_EPS)
        return th.clamp(z, -_CLIP, _CLIP).to(x.dtype)

    def features(self, obs: th.Tensor) -> th.Tensor:
        """[scaled raw obs, a, a / W, sum a, sum a / W] in demand units, before normalization."""
        p = self.parse(obs)
        u = self.unit
        W, ask = p["W"] / u, p["ask"] / u
        ask_sum = ask.sum(dim=1, keepdim=True)
        raw = th.cat([(p["ttg"] / self.horizon).unsqueeze(1), obs[:, 1:] / u], dim=1)
        return th.cat([
            raw,
            ask,
            th.clamp(ask / (W.unsqueeze(1) + _EPS), max=_RATIO_CAP),
            ask_sum,
            th.clamp(ask_sum / (W.unsqueeze(1) + _EPS), max=_RATIO_CAP),
        ], dim=1)


def ask_policy_kwargs(scenario, net_arch: Sequence[int], running_norm: bool = False) -> dict:
    """policy_kwargs for PPO("MlpPolicy", ...) on a fixed scenario."""
    from owmr_benchmark_lb import BalanceBound

    bound = BalanceBound(scenario)
    return {
        "features_extractor_class": AskFeatureExtractor,
        "features_extractor_kwargs": {
            "n": int(scenario.n_retailers), "l0": int(scenario.l0), "l_rt": int(scenario.l_rt),
            "horizon": int(scenario.horizon),
            "z_star": [float(v) for v in bound.z],
            "mu_rt": [float(m) for m in scenario.mu_rt],
            "running_norm": bool(running_norm),
        },
        "net_arch": list(net_arch),
    }


def rollout_stats_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class _Switch(BaseCallback):
        def _extractor(self):
            ext = getattr(self.model.policy, "features_extractor", None)
            return ext if isinstance(ext, AskFeatureExtractor) else None

        def _on_rollout_start(self) -> None:
            if (ext := self._extractor()) is not None:
                ext.update_stats = True

        def _on_rollout_end(self) -> None:
            if (ext := self._extractor()) is not None:
                ext.update_stats = False

        def _on_step(self) -> bool:
            return True

    return _Switch()

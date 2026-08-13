"""Deployable policy wrapper for the mab domain (spec §12).

Bundles the trained SB3 PPO model, its VecNormalize observation statistics,
and the action contract behind a single ``act(obs)`` call — no SB3/Gym
knowledge needed by the caller.

Observation contract (must match the model's training observation_mode):

- ``stats`` (21 floats, in this order):
    pulls[0..9]    — times each arm has been pulled so far (ints as floats)
    payouts[0..9]  — total payout collected from each arm so far
    time_to_go     — remaining rounds (T - t)
- ``bayes`` (21 floats):
    post_mean[0..9] — per-arm conjugate posterior mean of the true arm mean
    post_sd[0..9]   — per-arm conjugate posterior sd
    time_to_go      — remaining rounds
  (Use mab_bayes.bayes_post_mean / bayes_post_sd to build these from raw
  counts — the same functions the gym and the IR use.)

Action: an int arm index in [0, 9] — the arm to pull this round.

``deterministic`` defaults to **False**: in this domain the policy's entropy
IS its exploration mechanism, and argmax evaluation collapses the policy
into an under-explorer (measured on gauss_K10_T1000 @ 8192 seeds:
stochastic ≈ 1135 vs deterministic ≈ 684 mean reward for the same model).
Pass deterministic=True only if you specifically want the mode of the
policy distribution.

``frame_avg`` defaults to **16**: the deployed action distribution is the
permutation average pi_bar(s) = mean_sigma sigma^-1(pi(sigma(s))) over a
fixed sample of arm relabelings (identity included). The problem is exactly
arm-exchangeable but the trained MLP is not (measured TV 0.298 between
pi(perm(s)) and perm(pi(s))); averaging removes that slot noise and is worth
+38.9 mean reward at the 8192-seed protocol (1353.35 vs 1314.46, ESCALATION
#E11/A6-a0) for 16x inference cost on a tiny net. Pass frame_avg=0 for the
raw single-pass policy.

It is **automatically disabled for an equivariant policy** (``IndexPolicy``,
ESCALATION A6-rung-b): such a policy is exactly symmetric by construction, so
the average equals the single pass and the 16x cost buys nothing. Frame
averaging compensates a *non*-equivariant artifact; it is not a standing
component of the deliverable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class MabPolicy:
    """Trained PPO policy for the multi-armed bandit domain."""

    def __init__(
        self,
        model_path: str | Path,
        vecnorm_path: str | Path | None = None,
        observation_mode: str = "bayes",
        action_mode: str = "arm",
        deterministic: bool = False,
        seed: int | None = None,
        frame_avg: int = 16,
    ) -> None:
        assert action_mode == "arm", "only the 'arm' action mode is trained."
        model_path = Path(model_path)
        if vecnorm_path is None:
            vecnorm_path = model_path.parent / "vecnormalize.pkl"
        self.model = PPO.load(str(model_path), device="cpu")
        if seed is not None:
            self.model.set_random_seed(seed)
        self.observation_mode = observation_mode
        self.deterministic = deterministic
        # fixed permutation frame (identity first) + private action rng;
        # rng seed 5 matches the scored frame-averaged eval (ESCALATION #E11)
        self.frame_avg = int(frame_avg)
        # an equivariant policy is already symmetric by construction (measured
        # TV 2.3e-8 vs the MLP's 0.298), so averaging is an exact no-op that
        # would cost 16x inference for nothing — disable it silently. The flag
        # is a class attribute on mab_equinet's policies (IndexPolicy and its
        # A6-rung-c subclass), so this needs no import and no name matching.
        if getattr(type(self.model.policy), "IS_EQUIVARIANT", False):
            self.frame_avg = 0
        if self.frame_avg:
            import torch  # noqa: F401  (SB3 dependency, used via the policy)
            prng = np.random.default_rng(5)
            self._perms = [np.arange(10)] + [prng.permutation(10)
                                             for _ in range(self.frame_avg - 1)]
            self._invs = [np.argsort(pm) for pm in self._perms]
        self._act_rng = np.random.default_rng(seed)

        # VecNormalize stats are part of the model's I/O contract (§9.5)
        self._obs_rms = None
        self._clip_obs = 10.0
        vecnorm_path = Path(vecnorm_path)
        if not vecnorm_path.exists():
            raise FileNotFoundError(
                f"{vecnorm_path}: the saved VecNormalize is part of the "
                "model's I/O contract (§9.5) and must travel with it."
            )
        import pickle
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        # read the state dict directly: an unpickled VecNormalize has no
        # `venv`, so any attribute *miss* recurses inside SB3's __getattr__.
        # A run trained with norm_obs=False (e.g. an equivariant policy, which
        # must not sit behind slotwise stats) legitimately has no obs_rms.
        vn_state = vn.__dict__
        if vn_state.get("norm_obs", True):
            self._obs_rms = vn_state["obs_rms"]
            self._clip_obs = vn_state.get("clip_obs", 10.0)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        rms = self._obs_rms
        if rms is None:                 # trained on raw observations
            return obs
        return np.clip(
            (obs - rms.mean) / np.sqrt(rms.var + 1e-8),
            -self._clip_obs, self._clip_obs,
        )

    def act(self, obs) -> int:
        """Map one raw observation vector (see module docstring) to an arm."""
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if not self.frame_avg:
            nobs = self._normalize(obs).astype(np.float32)
            action, _ = self.model.predict(nobs, deterministic=self.deterministic)
            return int(action[0])

        import torch
        stacked = np.concatenate([
            np.concatenate([obs[:, :10][:, pm], obs[:, 10:20][:, pm],
                            obs[:, 20:]], axis=1) for pm in self._perms])
        nobs = self._normalize(stacked).astype(np.float32)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(torch.as_tensor(nobs))
            probs = dist.distribution.probs.numpy()
        pbar = np.mean([probs[m][self._invs[m]]
                        for m in range(self.frame_avg)], axis=0)
        if self.deterministic:
            return int(pbar.argmax())
        return int((pbar.cumsum() > self._act_rng.random()).argmax())


# ---------------------------------------------------------------------------
# Module smoke test: replay against the raw _mdp loop (no gym stack)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, ".")
    from mab_bayes import bayes_post_mean, bayes_post_sd
    from mab_mdp import advance, init_state
    from mab_scenarios import SCENARIOS

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("-s", "--scenario_name", type=str, default="gauss_K10_T1000")
    ap.add_argument("-o", "--observation_mode", type=str, default="bayes",
                    choices=["stats", "bayes"])
    ap.add_argument("--n-episodes", type=int, default=5)
    smoke = ap.parse_args()

    policy = MabPolicy(
        smoke.model_path, observation_mode=smoke.observation_mode, seed=0
    )
    source = SCENARIOS[smoke.scenario_name]
    is_gauss = not source.payout.is_discrete

    for seed in range(smoke.n_episodes):
        scenario = source(seed)
        state, _ = init_state(scenario, seed)
        total = 0.0
        while not state.terminated:
            ttg = float(scenario.horizon - state.period)
            if smoke.observation_mode == "stats":
                obs = ([float(n) for n in state.pulls]
                       + [float(x) for x in state.payouts] + [ttg])
            else:
                obs = (bayes_post_mean(state.pulls, state.payouts, is_gauss)
                       + bayes_post_sd(state.pulls, state.payouts, is_gauss)
                       + [ttg])
            state, info = advance(scenario, state, policy.act(obs))
            total += info["payoff"]["total"]
        print(f"episode_seed={seed}  reward_total={total:9.2f}  "
              f"oracle={scenario.horizon * info['opt_mean']:9.2f}")
    print("smoke OK: policy wrapper drives the raw _mdp loop without the gym")

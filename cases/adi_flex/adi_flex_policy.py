"""Deployable policy wrapper for a trained ADI-flex MaskablePPO model.

Bundles the trained model's I/O contract behind a small interface with no
SB3/Gym knowledge required of the caller:

1. the MaskablePPO model (.zip), loaded on CPU;
2. the VecNormalize observation stats (vecnormalize.pkl next to the model by
   default) — mandatory at inference, the policy was trained on normalized
   inputs;
3. the per-step action mask — under ``seq_mask`` the caller passes the
   feasible range's upper bound (from ``adi_flex_mdp.valid_allocation`` at the
   allocation phase, or None at the order phase for the full range); under the
   protection modes every point of the action box is feasible and no mask
   information is needed.

Action contract — read off the loaded model's action space, so a wrapper can
never decode a model in the wrong encoding:

- ``seq_mask`` (Discrete): one call per AGENT step, two per period on the
  heterogeneous branch; ``act`` returns an int (the order quantity, then the
  allocation).
- ``order_protection`` and the other protection modes (MultiDiscrete): ONE call
  per PERIOD at the pre-demand information set; ``act`` returns an integer
  vector ``[order, sigma_1, ..., sigma_{n_sigma}]`` — the cascade in
  ``adi_flex_gym`` turns the protection levels into the allocation. Every
  crowned artifact since #E18 is in this encoding (#E25).

Observation contract (must match the model's observation_mode at training) —
the F7 two-phase layout; see adi_flex_gym.AdiFlexEnv:

- "vec" (default):
    [inv, pipe(L), adv(2), time_to_go, phase, d(3), surplus, outstanding]
  where phase is 0 at the order step, 1 at the allocation step; `d` is the
  most recently realized demand vector; surplus/outstanding are the
  allocation's feasible-set coordinates (0 at the order phase).
- "vec_mip": [mip, adv(2), time_to_go, phase, d(3), surplus, outstanding],
  mip = inv + sum(pipe) - sum(adv).

Usage:
    policy = AdiFlexPolicy(model_path=".../het3_exp2_ppo_final.zip",
                           scenario_name="het3_exp2")      # mode inferred
    order, *sigmas = policy.act(obs)          # order_protection: one call/period

    policy = AdiFlexPolicy(model_path=".../seq_mask_model.zip")
    order    = policy.act(order_obs)                    # order phase
    allocate = policy.act(alloc_obs, feasible_max=fmax) # allocation phase
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize

from adi_flex_scenarios import SCENARIOS


class AdiFlexPolicy:
    """Trained ADI-flex ordering + allocation policy behind act(obs)."""

    def __init__(
        self,
        model_path: str,
        vecnorm_path: str | None = None,
        action_mode: str | None = None,
        scenario_name: str = "homog_L0_T2",
    ) -> None:
        """``action_mode`` may be omitted: the encoding is read off the model's
        action space (Discrete -> ``seq_mask``, MultiDiscrete ->
        ``order_protection``). When given, it is checked against that space —
        a model cannot be decoded in an encoding it was not trained in."""
        from adi_flex_gym import ACTION_MODES, PROTECT_MODES
        model_path_p = Path(model_path)
        self.model = MaskablePPO.load(str(model_path_p), device="cpu")
        space = self.model.action_space
        self._multi = isinstance(space, gym.spaces.MultiDiscrete)
        if action_mode is None:
            action_mode = "order_protection" if self._multi else "seq_mask"
        assert action_mode in ACTION_MODES, f"unknown action_mode {action_mode!r}"
        if (action_mode in PROTECT_MODES) != self._multi:
            raise ValueError(
                f"action_mode {action_mode!r} does not match the model's action "
                f"space {space}: protection modes are MultiDiscrete, the others "
                f"a shared Discrete")
        self.action_mode = action_mode
        self.alloc_enabled = SCENARIOS[scenario_name].alloc_enabled
        # the ORDER head's width, whichever encoding: nvec[0] under a
        # MultiDiscrete box, n under the shared Discrete
        self._order_head = int(space.nvec[0]) if self._multi else int(space.n)

        vp = Path(vecnorm_path) if vecnorm_path else model_path_p.parent / "vecnormalize.pkl"
        self._obs_rms = None
        self._clip_obs = 10.0
        if vp.exists():
            # the .pkl is a pickled VecNormalize; unpickle for the stats only
            # (no venv attach needed — act() applies the normalization itself)
            import pickle

            with open(vp, "rb") as fh:
                vn: VecNormalize = pickle.load(fh)
            # read the pickled fields directly: an unpickled VecNormalize has
            # no venv attached, and attribute access on it falls into SB3's
            # VecEnvWrapper.__getattr__ forwarding, which recurses without end
            vn_fields = vars(vn)
            # norm_obs=False (every crowned cell since #E8) pickles no obs
            # statistics; the raw observation is then what the policy saw
            self._obs_rms  = vn_fields.get("obs_rms") if vn_fields.get("norm_obs", True) else None
            self._clip_obs = vn_fields.get("clip_obs", 10.0)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_rms is None:
            return obs
        rms = self._obs_rms
        return np.clip(
            (obs - rms.mean) / np.sqrt(rms.var + 1e-8),
            -self._clip_obs, self._clip_obs,
        )

    def act(self, obs, feasible_max: int | None = None, order_max: int | None = None):
        """Deterministic action for one raw observation vector.

        Returns an ``int`` under ``seq_mask`` and an integer vector
        ``[order, sigma_1, ..., sigma_{n_sigma}]`` under the protection modes —
        in both cases exactly what ``AdiFlexEnv.step`` takes.

        ``feasible_max`` (``seq_mask`` only): at the allocation phase, the
        largest feasible allocation (``adi_flex_mdp.valid_allocation``); None
        (order phase, or the single-phase homogeneous env) leaves the full
        range live. Under a protection mode every point of the box is feasible
        (IR feasibility_strategy="none"), so the argument must be None.
        ``order_max``: the order head's width (``scenario.order_max``); defaults
        to the loaded policy's own, which is what it was trained on.
        """
        if order_max is None:
            order_max = self._order_head - 1
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        x = self._normalize(x)
        if self._multi:
            if feasible_max is not None:
                raise ValueError("feasible_max applies to seq_mask only: under a "
                                 "protection mode every action is feasible")
            # MaskablePPO's MultiDiscrete mask is the concatenation of one
            # boolean vector per component — all live, as the gym emits it
            mask = np.ones(int(sum(self.model.action_space.nvec)), dtype=bool)
            action, _ = self.model.predict(x, action_masks=mask[np.newaxis],
                                           deterministic=True)
            return np.asarray(action, dtype=np.int64).reshape(-1)
        mask = np.zeros(order_max + 1, dtype=bool)
        if feasible_max is None:
            mask[:] = True
        else:
            mask[: min(int(feasible_max), order_max) + 1] = True
        action, _ = self.model.predict(x, action_masks=mask[np.newaxis], deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Module self-test: replay through the gym (the obs contract's owner)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from adi_flex_gym import AdiFlexEnv

    ap = argparse.ArgumentParser(description="Smoke-test a trained ADI-flex policy.")
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--vecnorm-path", type=str, default=None)
    ap.add_argument("-s", "--scenario_name", type=str, default="homog_L0_T2",
                    choices=list(SCENARIOS.keys()))
    ap.add_argument("-o", "--observation_mode", type=str, default="vec",
                    help="the model's observation contract (vec / vec_mip / ...)")
    ap.add_argument("-a", "--action_mode", type=str, default=None,
                    help="default: inferred from the model's action space")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--first-seed", type=int, default=0)
    args = ap.parse_args()

    policy = AdiFlexPolicy(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        action_mode=args.action_mode,
        scenario_name=args.scenario_name,
    )
    env = AdiFlexEnv(scenario=SCENARIOS[args.scenario_name],
                     action_mode=policy.action_mode,
                     observation_mode=args.observation_mode)
    print(f"action_mode={policy.action_mode}  action_space={env.action_space}")

    costs = []
    for ep_seed in range(args.first_seed, args.first_seed + args.episodes):
        obs, _ = env.reset(seed=ep_seed)
        terminated = False
        while not terminated:
            mask = env.action_masks()
            fmax = (None if mask.all()
                    else int(np.flatnonzero(mask)[-1]))
            obs, _, terminated, _, _ = env.step(policy.act(obs, feasible_max=fmax))
        costs.append(-env.total_reward)
        print(f"episode_seed={ep_seed}  total cost={-env.total_reward:.2f}")
    print(f"mean cost over {len(costs)} episodes: {np.mean(costs):.4f}")

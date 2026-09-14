"""Deployable dynamic pricing policy: a thin, stable wrapper over a trained model.

Bundles everything the trained artifact needs at inference time:
- the SB3 PPO model (.zip),
- the VecNormalize observation stats (vecnormalize.pkl) — part of the model's
  input contract: the policy was trained on normalized observations,
- the action transform for the 'intensity' action mode (reparametrized
  continuous action, imported from dynamic_pricing_gym — not duplicated here).

Observation contract (what the caller must supply to ``act``):
    observation_mode "vec":   [inventory, time_to_go]
    observation_mode "vec_d": [inventory, time_to_go, units_sold_last_period]
where inventory is the remaining stock (units), time_to_go = horizon - period
(periods), and units_sold_last_period is the previous period's sales (units).

``act`` returns the price to charge this period (float, non-negative) —
already transformed from the raw model action, whatever the action mode.

Example:
    from dynamic_pricing_policy import DynamicPricingPolicy
    policy = DynamicPricingPolicy("results/simple/PPO_.../simple_ppo.zip",
                                  action_mode="intensity")
    price = policy.act([25, 50])   # full stock, full horizon ahead
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from dynamic_pricing_gym import intensity_to_price
from dynamic_pricing_scenarios import DynamicPricingScenario, SCENARIOS


class DynamicPricingPolicy:
    """Loads a trained PPO model and exposes ``act(obs) -> price``.

    vecnorm_path defaults to ``vecnormalize.pkl`` next to the model; pass
    ``vecnorm_path=""`` to disable observation normalization explicitly.
    ``action_mode`` and ``scenario`` must match what the model was trained
    with (the run directory's ``*_args.txt`` records them).
    """

    def __init__(
        self,
        model_path: str | Path,
        vecnorm_path: str | Path | None = None,
        action_mode: str = "price",
        scenario: DynamicPricingScenario | str = "simple",
    ) -> None:
        assert action_mode in {"price", "intensity"}, \
            f"action_mode must be 'price' or 'intensity', got '{action_mode}'."
        model_path = Path(model_path)
        self.model = PPO.load(str(model_path), device="cpu")
        self.action_mode = action_mode
        self.scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario

        if vecnorm_path is None:
            vecnorm_path = model_path.parent / "vecnormalize.pkl"
        self._obs_rms = None
        self._clip_obs = 10.0
        if vecnorm_path and Path(vecnorm_path).exists():
            with open(vecnorm_path, "rb") as f:
                vecnorm = pickle.load(f)
            self._obs_rms  = vecnorm.obs_rms
            self._clip_obs = vecnorm.clip_obs

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_rms is None:
            return obs
        rms = self._obs_rms
        return np.clip(
            (obs - rms.mean) / np.sqrt(rms.var + 1e-8),
            -self._clip_obs, self._clip_obs,
        )

    def act(self, obs) -> float:
        """Deterministic price for one observation (see the contract above)."""
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        obs = self._normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        raw = float(np.asarray(action).ravel()[0])
        if self.action_mode == "intensity":
            return intensity_to_price(self.scenario, raw)
        return max(0.0, raw)


# ---------------------------------------------------------------------------
# Module self-test: replay the policy against the simulator
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import dynamic_pricing_mdp as mdp

    ap = argparse.ArgumentParser(description="Smoke-test a deployed policy on a few episodes.")
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("-s", "--scenario_name", type=str, default="simple",
                    choices=list(SCENARIOS.keys()))
    ap.add_argument("-a", "--action_mode", type=str, default="price",
                    choices=["price", "intensity"])
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    scenario = SCENARIOS[args.scenario_name]
    policy = DynamicPricingPolicy(args.model_path, action_mode=args.action_mode,
                                  scenario=scenario)

    for ep_seed in range(args.episodes):
        state, info = mdp.init_state(scenario, ep_seed)
        total = 0.0
        first_price = None
        while not state.terminated:
            price = policy.act([state.inventory, scenario.horizon - state.period])
            first_price = price if first_price is None else first_price
            state, info = mdp.advance(scenario, state, price)
            total += info["revenue"]["total"]
        print(f"seed={ep_seed}  first_price={first_price:.3f}  "
              f"periods={state.period}  unsold={state.inventory}  revenue={total:.2f}")

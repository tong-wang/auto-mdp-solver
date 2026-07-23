"""Deployable AdiFlex policy: a trained model behind a small, stable interface.

A caller needs no SB3 or Gymnasium knowledge — construct it with the paths from
a training run and call `act(obs)`.

OBSERVATION CONTRACT
--------------------
`act()` takes a sequence of floats in exactly this order. The observation is
read at the *pre-order* point of the period: after the previous period's
fulfilment, before this period's order is placed and before this period's
demand arrives.

observation_mode="vec"  (4 features, the default)
    0  period      current period index, 0-based, 0..11
    1  inventory   net inventory in units; NEGATIVE means overdue backlog
    2  due_now     unsatisfied advance orders due by the current period
    3  due_next    unsatisfied advance orders due by the next period

observation_mode="vec_mip"  (3 features)
    0  period      as above
    1  mip         modified inventory position = inventory - due_now - due_next
    2  due_next    as above

All quantities are whole units. `due_now` / `due_next` are the *unsatisfied*
remainders carried in the state, not the totals ever ordered.

ACTION
------
`act()` returns `(order_quantity, hold_back)` as ints:
    order_quantity  units to order now; arrives immediately (supply lead time 0)
    hold_back       units of stock to reserve before shipping against orders
                    that are not yet due, protecting against next period's
                    urgent arrivals

Both are committed before demand is observed, matching the model that was
trained. Bounds come from adi_flex_gym (ORDER_MAX, HOLD_BACK_MAX) and are not
duplicated here.

DEPENDENCIES
------------
This file is intentionally not standalone: it imports the action bounds from
`adi_flex_gym` so the decode cannot drift from the trained action space. The
VecNormalize observation statistics are part of the model's input contract, not
an optimization — the policy was trained on normalized inputs and will behave
incorrectly without them.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from adi_flex_gym import HOLD_BACK_MAX, ORDER_MAX


class AdiFlexPolicy:
    """Trained AdiFlex policy: observation in, (order, hold-back) out."""

    def __init__(
        self,
        model_path: str | Path,
        vecnorm_path: str | Path | None = None,
        observation_mode: str = "vec",
        action_mode: str = "joint",
        scenario_name: str = "exp4",
    ) -> None:
        model_path = Path(model_path)
        self.model = PPO.load(model_path, device="cpu")
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.scenario_name = scenario_name

        path = Path(vecnorm_path) if vecnorm_path else model_path.parent / "vecnormalize.pkl"
        self._obs_rms = None
        self._clip_obs = 10.0
        self._epsilon = 1e-8
        if path.exists():
            # The saved artifact is a pickled VecNormalize. Unpickle it directly
            # rather than VecNormalize.load(), which requires a live venv — this
            # wrapper deliberately has no Gym env, only the normalization stats.
            with open(path, "rb") as fh:
                saved: VecNormalize = pickle.load(fh)
            self._obs_rms = saved.obs_rms
            self._clip_obs = float(saved.clip_obs)
            self._epsilon = float(saved.epsilon)
        else:
            raise FileNotFoundError(
                f"VecNormalize stats not found at {path}. They are part of the "
                f"model's input contract, not optional — pass vecnorm_path "
                f"explicitly if they live elsewhere."
            )

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        mean, var = self._obs_rms.mean, self._obs_rms.var
        return np.clip(
            (obs - mean) / np.sqrt(var + self._epsilon),
            -self._clip_obs, self._clip_obs,
        ).astype(np.float32)

    def act(self, obs: Sequence[float], deterministic: bool = True) -> tuple[int, int]:
        """Map one observation to (order_quantity, hold_back)."""
        raw = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(self._normalize(raw), deterministic=deterministic)
        flat = np.asarray(action).reshape(-1)
        return (int(np.clip(flat[0], 0, ORDER_MAX)),
                int(np.clip(flat[1], 0, HOLD_BACK_MAX)))


# ---------------------------------------------------------------------------
# Module smoke test — replays against the raw MDP loop, not the gym
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import adi_flex_mdp as mdp
    from adi_flex_scenarios import SCENARIOS

    ap = argparse.ArgumentParser(description="Smoke-test the deployable policy.")
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--vecnorm-path", type=str, default=None)
    ap.add_argument("-s", "--scenario_name", type=str, default="exp4")
    ap.add_argument("-o", "--observation_mode", type=str, default="vec")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    scenario = SCENARIOS[args.scenario_name]
    policy = AdiFlexPolicy(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        observation_mode=args.observation_mode,
        scenario_name=args.scenario_name,
    )

    def observe(state) -> list[float]:
        """Build the documented observation straight off the MDP state."""
        if args.observation_mode == "vec_mip":
            return [float(state.period),
                    float(state.inventory - state.due_now - state.due_next),
                    float(state.due_next)]
        return [float(state.period), float(state.inventory),
                float(state.due_now), float(state.due_next)]

    print(f"scenario: {args.scenario_name}  ({scenario.desc})")
    totals = []
    for ep_seed in range(args.episodes):
        state, _ = mdp.init_state(scenario=scenario, episode_seed=ep_seed)
        total, orders, holds = 0.0, [], []
        while not state.terminated:
            state1 = mdp.advance1(scenario, state)
            order_quantity, hold_back = policy.act(observe(state1))
            state, info = mdp.advance2(
                scenario, state1,
                order_quantity=order_quantity, hold_back=hold_back,
            )
            total += info["cost"]["total"]
            orders.append(order_quantity)
            holds.append(hold_back)
        totals.append(total)
        print(f"  seed {ep_seed}: cost={total:8.2f}  orders={orders}  hold_back={holds}")
    print(f"mean cost over {args.episodes} episodes: {np.mean(totals):.2f}")

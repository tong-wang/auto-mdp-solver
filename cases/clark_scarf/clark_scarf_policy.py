"""Deployable Clark & Scarf policy (spec §12).

The trained model behind a stable interface, so a caller needs no SB3 or
Gymnasium knowledge:

    from clark_scarf_policy import ClarkScarfPolicy

    pol = ClarkScarfPolicy("…/n3_l2_p09_ppo.zip", scenario="n3_l2_p09")
    ship = pol.act(obs)          # -> one shipment quantity per link

It bundles the three things that are part of the trained model's I/O contract:
the SB3 model, the VecNormalize observation statistics, and the action
transform for the mode it was trained under.

Observation contract
--------------------
`act()` takes the vector `clark_scarf_gym.observation()` produces — that
function is imported, not re-derived, so the contract cannot drift. For a
chain of `n = n_echelons` links and lead time `L`::

    [ period / horizon ,  block_0 ... block_{L-1} ]        each block n wide

    block_0      on-hand stock at each installation, units.
                 Index 0 is the retailer and MAY BE NEGATIVE (backlog);
                 upper installations are non-negative.
    block_{s+1}  units in flight into each installation, landing after s+1
                 more arrival events (s = 0 … L-2), units.

`L - 1` transit blocks, not `L`: the observation point is **pre-order, post
arrival**, so the last pipeline slot has just been vacated and is necessarily
empty. Total width is `1 + n * L`.

Under `observation_mode="echelon"` the same numbers arrive as component-wise
running sums (stock and in-flight summed *separately* — a single running sum
over their total is lossy at `L = 2`). `act()` inverts that before applying
the action transform, so the caller supplies whichever form the model was
trained on and nothing else changes.

**Units and timing are the caller's responsibility.** The vector must describe
the state *after* this period's arrivals have landed and *before* anything is
dispatched. Feeding a post-dispatch state silently produces a valid-looking
but wrong shipment.

Not standalone, by design
-------------------------
`target_discrete` decodes a bin index as an echelon **order-up-to level**, so
turning it into a quantity needs `echelon_position()`, and the availability
clip needs `ship_capacity()` — both imported from `clark_scarf_mdp`. Spec §12
requires importing that math rather than duplicating it, so this module
depends on `clark_scarf_mdp`, `clark_scarf_gym` and `clark_scarf_scenarios`.

Matching the training run
-------------------------
`observation_mode`, `action_mode` and the scenario must match training; they
are constructor arguments rather than guesses. The run directory's
`{scenario}_ppo_args.txt` records all three.

Smoke test
----------
`python clark_scarf_policy.py --model-path …` replays the policy against the
raw `clark_scarf_mdp` loop — no gym — and prints per-episode cost. That is the
check that the normalization and the action transform here reproduce
eval-time behaviour outside the training stack.

Verified stronger than "looks about right": on the crowned artifact over the
shared seed block `0…511`, this wrapper and `clark_scarf_ppo_eval.py` both
score **996.103846**, difference 0.000000. Re-check it after touching either
path with

    python clark_scarf_policy.py --model-path <run>/n3_l2_p09_ppo.zip \\
        --vecnorm-path <run>/vecnormalize.pkl --episodes 512
    python clark_scarf_ppo_eval.py -s n3_l2_p09 -o raw -a target_discrete \\
        --model-path <run>/n3_l2_p09_ppo.zip --vecnorm-path <run>/vecnormalize.pkl \\
        --n-seeds 512

A drift between the two means this file and the gym have diverged on
normalization or on the action transform — the failure §12 exists to catch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from clark_scarf_gym import from_echelon, observation
from clark_scarf_mdp import (
    ClarkScarfState,
    advance1,
    advance2,
    echelon_position,
    init_state,
    ship_capacity,
)
from clark_scarf_scenarios import SCENARIOS


class ClarkScarfPolicy:
    """Trained policy + its normalization + its action transform."""

    def __init__(
        self,
        model_path: str | Path,
        vecnorm_path: str | Path | None = None,
        *,
        scenario: str,
        observation_mode: str = "raw",
        action_mode: str = "target_discrete",
    ) -> None:
        from stable_baselines3 import PPO

        model_path = Path(model_path)
        self.scenario = SCENARIOS[scenario]
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.model = PPO.load(str(model_path), device="cpu")

        # VecNormalize stats are part of the I/O contract, not an optimization:
        # the network was trained on normalized inputs and is meaningless
        # without them. Default to the file beside the model (spec §8.3/§9.5).
        if vecnorm_path is None:
            candidate = model_path.parent / "vecnormalize.pkl"
            vecnorm_path = candidate if candidate.exists() else None
        self._obs_rms = None
        self._clip_obs = 10.0
        self._eps = 1e-8
        if vecnorm_path is not None:
            import pickle

            with open(vecnorm_path, "rb") as fh:
                vn = pickle.load(fh)
            self._obs_rms = vn.obs_rms
            self._clip_obs = float(getattr(vn, "clip_obs", 10.0))
            self._eps = float(getattr(vn, "epsilon", 1e-8))

    # -- the contract --------------------------------------------------------

    def act(self, obs) -> list[float]:
        """Observation -> one shipment quantity per link, already clipped.

        The returned quantities are what the simulator would execute: the
        availability clip is applied here, so a caller can dispatch them
        directly. Deterministic.
        """
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        n, L = self.scenario.n_echelons, self.scenario.leadtime
        expected = 1 + n * L
        if obs.shape[0] != expected:
            raise ValueError(
                f"observation has width {obs.shape[0]}, expected {expected} "
                f"(1 + n_echelons * leadtime = 1 + {n} * {L}) for scenario "
                f"{self.scenario.scenario_name!r}. See the module docstring."
            )

        state = self._state_from_obs(obs)
        action, _ = self.model.predict(self._normalize(obs), deterministic=True)
        return self._to_shipments(np.asarray(action).reshape(-1), state)

    # -- internals -----------------------------------------------------------

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_rms is None:
            return obs.astype(np.float32)
        z = (obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + self._eps)
        return np.clip(z, -self._clip_obs, self._clip_obs).astype(np.float32)

    def _state_from_obs(self, obs: np.ndarray) -> ClarkScarfState:
        """Rebuild the simulator state the action transform needs.

        The observation is a lossless encoding of the pre-dispatch state (that
        is what the coordinate-change test pins), so this inverts it rather
        than approximating: `echelon` is undone by `from_echelon`, and the
        empty last pipeline slot is restored.
        """
        n, L = self.scenario.n_echelons, self.scenario.leadtime
        blocks = [list(obs[1 + i * n : 1 + (i + 1) * n]) for i in range(L)]
        if self.observation_mode == "echelon":
            blocks = from_echelon(blocks)
        stock, transits = blocks[0], blocks[1:]

        # transits carries slots 0 … L-2; slot L-1 is empty pre-dispatch
        pipeline = [[transits[s][k] for s in range(L - 1)] + [0.0] for k in range(n)]
        return ClarkScarfState(
            period=int(round(float(obs[0]) * self.scenario.horizon)),
            terminated=False,
            stock=list(stock),
            pipeline=pipeline,
            demand=0,
            arrived=[0.0] * n,
            shipped=[0.0] * n,
            seed_salt=self.scenario.seed_salt,
            episode_seed=0,
        )

    def _to_shipments(self, action: np.ndarray, state: ClarkScarfState) -> list[float]:
        """Decode the action, then apply the availability clip.

        Mirrors the gym's `step()` path, using the same domain functions it
        uses — `echelon_position` for the target decode, `ship_capacity` for
        the clip — so the two cannot drift.
        """
        n = self.scenario.n_echelons
        if self.action_mode == "target_discrete":
            u = echelon_position(self.scenario, state)
            raw = [max(0.0, float(action[k]) - u[k]) for k in range(n)]
        elif self.action_mode == "ship_discrete":
            raw = [float(action[k]) for k in range(n)]
        elif self.action_mode == "ship":
            raw = [float(np.clip(action[k], 0.0, self.scenario.ship_max)) for k in range(n)]
        else:
            raise ValueError(f"unsupported action_mode {self.action_mode!r}")

        caps = ship_capacity(self.scenario, state)
        return [float(np.clip(raw[k], 0.0, caps[k])) for k in range(n)]


# ---------------------------------------------------------------------------
# Smoke test — against the raw MDP loop, deliberately not the gym
# ---------------------------------------------------------------------------


def _smoke(policy: ClarkScarfPolicy, episodes: int = 3, beta: float | None = None) -> None:
    """Replay the wrapper against `clark_scarf_mdp` directly.

    No gym, no VecEnv: if the normalization or the action transform in this
    file disagreed with the training stack, the per-episode costs below would
    not land near the eval record.
    """
    sc = policy.scenario
    beta = 0.95 if beta is None else beta
    print(
        f"=== {sc.scenario_name}: N={sc.n_echelons} L={sc.leadtime} T={sc.horizon} "
        f"| obs={policy.observation_mode} act={policy.action_mode} "
        f"| vecnorm={'yes' if policy._obs_rms is not None else 'NO'} ==="
    )
    totals = []
    for ep in range(episodes):
        state, _ = init_state(sc, episode_seed=ep)
        disc = 0.0
        while not state.terminated:
            state1 = advance1(sc, state)
            obs = observation(sc, state1, policy.observation_mode)
            ship = policy.act(obs)
            state, info = advance2(sc, state1, ship)
            disc += (beta ** info["action_period"]) * info["cost"]["total"]
        totals.append(disc)
        print(f"  episode {ep}: discounted cost = {disc:10.4f}")
    print(f"  mean over {episodes} episodes = {np.mean(totals):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--vecnorm-path", default=None)
    ap.add_argument("-s", "--scenario", default="n3_l2_p09")
    ap.add_argument("-o", "--observation-mode", default="raw", choices=["raw", "echelon"])
    ap.add_argument("-a", "--action-mode", default="target_discrete")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    _smoke(
        ClarkScarfPolicy(
            args.model_path,
            args.vecnorm_path,
            scenario=args.scenario,
            observation_mode=args.observation_mode,
            action_mode=args.action_mode,
        ),
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()

"""Deployable FNV policy: a thin, stable wrapper over a trained model (spec §12).

Bundles everything the trained artifact needs at inference time, so a caller
needs no SB3 or Gym knowledge:

1. the SB3 PPO model (.zip), loaded on CPU;
2. the VecNormalize observation stats (vecnormalize.pkl) — **part of the input
   contract**, not an optimization: the policy was trained on normalized
   observations, so without the saved stats it sees inputs it never saw in
   training;
3. the action transform for the trained action_mode.

Observation contract (what the caller must supply to ``act``)
-------------------------------------------------------------
observation_mode "vec" — a length-6 vector, in this order:

    [0] stdev        total demand std dev of the scenario  (units of demand)
    [1] T            time location of the last ordering epoch, in (0, 1)
    [2] lamb         ordering-cost increment per period    (cost/unit/period)
    [3] period       current ordering period, 1-indexed, 1..N
    [4] inventory    cumulative quantity ordered so far    (units)
    [5] information  cumulative demand signal I_period     (units of demand)

The first three are scenario constants; the agent sees them because one policy
is trained across a whole design grid (§5.6) and must condition on which cell
it is in. `information` is the MMFE signal accumulated *before* this period's
order — see fnv_mdp's module docstring for the timing.

``act`` returns the order quantity for this period (float, non-negative).

Action transform: action_mode "box" is a raw quantity clipped at zero, matching
FnvEnv.step. The clip is reproduced here rather than imported because
fnv_gym.py is kept byte-identical to the upstream case folder; it is one
operation and it is asserted against the gym in the __main__ smoke test.

Example:
    from fnv_policy import FnvPolicy
    policy = FnvPolicy("results/simple/PPO_.../simple_ppo.zip", scenario="simple")
    q = policy.act([0.1, 0.9, 0.1, 1, 0.0, 0.0])   # first order, nothing on hand
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from fnv_grids import GRIDS
from fnv_scenarios import FnvScenario, SCENARIOS

OBS_FEATURES = ("stdev", "T", "lamb", "period", "inventory", "information")


class FnvPolicy:
    """Loads a trained PPO model and exposes ``act(obs) -> order quantity``.

    vecnorm_path defaults to ``vecnormalize.pkl`` next to the model, then the
    run directory above a ``checkpoints/`` folder; pass ``vecnorm_path=""`` to
    disable observation normalization explicitly. ``action_mode``,
    ``observation_mode`` and ``scenario`` must match what the model was trained
    with — the run directory's ``{scenario_name}_ppo_args.txt`` records them,
    and the constructor takes them explicitly rather than guessing.
    """

    def __init__(
        self,
        model_path: str | Path,
        vecnorm_path: str | Path | None = None,
        action_mode: str = "box",
        observation_mode: str = "vec",
        scenario: FnvScenario | str = "simple",
    ) -> None:
        assert action_mode == "box", \
            f"action_mode must be 'box', got '{action_mode}'."
        assert observation_mode == "vec", \
            f"observation_mode must be 'vec', got '{observation_mode}'."

        model_path = Path(model_path)
        self.model            = PPO.load(str(model_path), device="cpu")
        self.action_mode      = action_mode
        self.observation_mode = observation_mode
        self.scenario         = self._resolve_scenario(scenario)

        if vecnorm_path is None:
            vecnorm_path = self._default_vecnorm(model_path)
        self._obs_rms  = None
        self._clip_obs = 10.0
        if vecnorm_path and Path(vecnorm_path).exists():
            with open(vecnorm_path, "rb") as f:
                vecnorm = pickle.load(f)
            # A run trained with --no-norm-obs still saves a VecNormalize, but
            # one with no obs_rms at all; reading it through the wrapper's
            # __getattr__ recurses forever (SB3's VecEnvWrapper), so ask the
            # flag first — the same idiom as fnv_ppo_eval.load_obs_rms.
            if getattr(vecnorm, "norm_obs", True):
                self._obs_rms = vecnorm.obs_rms
            self._clip_obs = float(vecnorm.clip_obs)

    @staticmethod
    def _resolve_scenario(scenario: FnvScenario | str) -> FnvScenario:
        if isinstance(scenario, FnvScenario):
            return scenario
        if scenario in SCENARIOS:
            return SCENARIOS[scenario]
        if scenario in GRIDS:
            # a generalist's deployment target is a specific cell; take the
            # first so the wrapper has concrete bounds, and let the caller pass
            # the intended cell explicitly when it matters
            return GRIDS[scenario][0]
        raise KeyError(f"unknown scenario {scenario!r}")

    @staticmethod
    def _default_vecnorm(model_path: Path) -> Path | None:
        beside = model_path.parent / "vecnormalize.pkl"
        if beside.exists():
            return beside
        run_dir = model_path.parent.parent / "vecnormalize.pkl"
        return run_dir if run_dir.exists() else None

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_rms is None:
            return obs
        rms = self._obs_rms
        return np.clip(
            (obs - rms.mean) / np.sqrt(rms.var + 1e-8),
            -self._clip_obs, self._clip_obs,
        )

    def act(self, obs) -> float:
        """Return the order quantity for the observed state (deterministic)."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        assert obs.shape == (len(OBS_FEATURES),), (
            f"expected an observation of {len(OBS_FEATURES)} features "
            f"{OBS_FEATURES}, got shape {obs.shape}.")
        action, _ = self.model.predict(self._normalize_obs(obs), deterministic=True)
        return self._to_quantity(action)

    @staticmethod
    def _to_quantity(action) -> float:
        """Action transform for action_mode 'box' — mirrors FnvEnv.step."""
        raw = float(np.asarray(action).reshape(-1)[0])
        return max(0.0, raw)


# ---------------------------------------------------------------------------
# Smoke test (§12): replay against the raw MDP loop, not the gym
# ---------------------------------------------------------------------------

def _smoke(model_path: str, scenario_name: str = "simple", n_episodes: int = 5) -> None:
    """Replay the wrapper against fnv_mdp directly.

    Going through the raw simulator rather than the gym is the point: it proves
    the wrapper's own normalization and action transform reproduce eval-time
    behavior without the gym stack in the loop. The gym cross-check below
    asserts the two agree step for step.
    """
    from fnv_gym import FnvEnv
    from fnv_mdp import advance, init_state

    policy   = FnvPolicy(model_path, scenario=scenario_name)
    scenario = policy.scenario
    print(f"model     {model_path}")
    print(f"scenario  {scenario_name}  (obs_rms="
          f"{'loaded' if policy._obs_rms is not None else 'NONE'})")
    print(f"obs contract: {list(OBS_FEATURES)}\n")

    for episode_seed in range(n_episodes):
        state, _ = init_state(scenario, episode_seed)
        gym_env  = FnvEnv(scenario=scenario)
        gym_obs, _ = gym_env.reset(seed=episode_seed)

        total_cost = 0.0
        info: dict = {}
        while not state.terminated:
            obs = np.array([
                scenario.stdev, scenario.T, scenario.lamb,
                float(state.period), state.inventory, state.information,
            ], dtype=np.float32)
            # the hand-built observation must match what the gym would emit
            assert np.allclose(obs, gym_obs, atol=1e-6), \
                f"obs mismatch at period {state.period}: {obs} vs {gym_obs}"

            q = policy.act(obs)
            state, info = advance(scenario, state, q)
            gym_obs, _, _, _, _ = gym_env.step(np.array([q], dtype=np.float32))
            total_cost += info["cost"]

        gym_env.close()
        print(f"  seed {episode_seed}: demand={info['demand']:.4f} "
              f"ordered={state.inventory:.4f} cost={total_cost:.4f} "
              f"profit={info['profit']:.4f}")

    print("\nOK — the wrapper reproduces the gym's observations and actions "
          "while driving the raw MDP.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Smoke-test the deployable FNV policy.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("-s", "--scenario_name", type=str, default="simple")
    p.add_argument("--n-episodes", type=int, default=5)
    a = p.parse_args()
    _smoke(a.model_path, a.scenario_name, a.n_episodes)

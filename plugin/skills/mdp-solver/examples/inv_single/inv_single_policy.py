"""Deployable single-item inventory policy (spec §12).

The trained model behind a stable interface, so a caller needs no SB3 or
Gymnasium knowledge:

    from inv_single_policy import InvSinglePolicy

    pol = InvSinglePolicy("…/ppo_inv_single.zip", scenario="simple")
    q   = pol.act(obs)            # -> one order quantity, units

It bundles the three things that are part of the trained model's I/O contract:
the SB3 model, the VecNormalize observation statistics, and the action
transform for the mode it was trained under.

Observation contract
--------------------
`act()` takes the vector `InvSingleEnv._get_obs()` produces. The leading
feature is **time-to-go**, not the period index (F5, 2026-08-28): `horizon -
period`, counting DOWN to the terminal boundary. For pipeline length
`P = leadtime.max() + 1`:

    observation_mode="vec"        [ ttg, inventory, pipe_0 … pipe_{P-1} ]
    observation_mode="vec_ip"     [ ttg, inventory_position ]
    observation_mode="vec_ctx"    vec    + [ b/(b+h), K ]
    observation_mode="vec_ctx_slt"    vec    + [ leadtime law (3) , b/(b+h), K ]
    observation_mode="vec_ip_ctx_slt" vec_ip + [ leadtime law (3) , b/(b+h), K ]

  ttg          periods remaining, `horizon - period`.
  inventory    on-hand, units. NEGATIVE under backlog; never negative under
               lost sales.
  pipe_k       units arriving at the k-th future receipt event, units.
  inventory_position   `inventory + sum(pipe)`. NOT a sufficient statistic
               under stochastic lead time or lost sales — see README.
  b/(b+h), K   the cost regime, for the grid generalist only.
  leadtime law the pmf of one lead-time draw over the categorical support
               {1,2,3} (IR feature `leadtime.probs`, `dim` 3) — the regime the
               `lt_variance_*` generalist conditions on. Never a realized
               draw. `InvSingleEnv.leadtime_law()` renders it.

**Timing is the caller's responsibility.** The vector must describe the
**pre-order** state: with event sequence `O-R-D` that is before this period's
receipt, so `pipe_0` has not yet landed. Feeding a post-receipt state produces
a valid-looking but wrong order.

Not standalone, by design
-------------------------
The action transform is imported from `inv_single_gym.InvSingleEnv.decode_action`
rather than re-derived — that method's own docstring says a second copy of the
arithmetic "is a drift waiting to happen". `policy="ordinal"` artifacts also
need `inv_single_ordinal_head` importable at load time, because SB3 stores the
custom policy class by module path; this module imports it for that reason.

Which artifact
--------------
`observation_mode`, `action_mode` and the scenario **must match training** and
are taken explicitly, never guessed. The run directory's
`{scenario}_ppo_args.txt` records all three. The campaign's crowned artifacts
(README's headline board) are:

    simple          sc0/g4/a1/h2   vec     discrete   176.11  (99.92% of exact)
    simple_k        sc1/g4/a1/h3   vec     discrete   618.96  (99.96%)
    lt_lost_sales   unregistered   vec_ip  discrete   416.01  (99.95%)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# imported for its side effect: SB3 stores a custom policy class by module
# path, so `policy=ordinal` artifacts fail to load unless this is importable.
import inv_single_ordinal_head  # noqa: F401
from inv_single_gym import InvSingleEnv
from inv_single_scenarios import SCENARIOS


class InvSinglePolicy:
    """A trained ordering policy: observation vector in, order quantity out."""

    def __init__(
        self,
        model_path: str | Path,
        vecnorm_path: str | Path | None = None,
        *,
        scenario: str = "simple",
        observation_mode: str = "vec",
        action_mode: str = "discrete",
        deterministic: bool = True,
    ) -> None:
        from stable_baselines3 import PPO

        model_path = Path(model_path)
        self.model = PPO.load(str(model_path), device="cpu")
        self.scenario_name = scenario
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.deterministic = deterministic

        sc = SCENARIOS[scenario]
        if callable(sc):                       # a latent-demand source
            sc = sc(0)
        self.scenario = sc

        # the decoder lives on the env so there is exactly one copy of it
        self._env = InvSingleEnv(
            scenario=sc,
            observation_mode=observation_mode,
            action_mode=action_mode,
        )

        # VecNormalize stats are part of the I/O contract, not an optimization:
        # a model trained on normalized inputs is wrong without them. `g4`
        # (norm_obs=False) is the campaign operating point, and there the saved
        # object reports norm_obs=False -- so this reads the FLAG rather than
        # assuming, and a mismatch would otherwise be silent.
        if vecnorm_path is None:
            cand = model_path.parent / "vecnormalize.pkl"
            vecnorm_path = cand if cand.exists() else None
        self._obs_rms = None
        self._clip_obs = 10.0
        self._eps = 1e-8
        if vecnorm_path is not None:
            from stable_baselines3.common.vec_env import VecNormalize

            vn = VecNormalize.load(str(vecnorm_path), _DummyVenv(self._env))
            if getattr(vn, "norm_obs", False):
                self._obs_rms = vn.obs_rms
                self._clip_obs = float(vn.clip_obs)
                self._eps = float(vn.epsilon)

    # -- inference -------------------------------------------------------
    def normalize(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self._obs_rms is None:
            return obs
        z = (obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + self._eps)
        return np.clip(z, -self._clip_obs, self._clip_obs).astype(np.float32)

    def act(self, obs) -> float:
        """The order quantity for one pre-order observation, in units."""
        action, _ = self.model.predict(
            self.normalize(obs), deterministic=self.deterministic
        )
        return float(self._env.decode_action(action))


class _DummyVenv:
    """Minimal VecEnv surface VecNormalize.load needs to rebind its spaces."""

    def __init__(self, env):
        self.num_envs = 1
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render_mode = None

    def __getattr__(self, name):        # pragma: no cover - never driven
        raise AttributeError(name)


# -- smoke test: replay through the RAW mdp loop, never the gym -------------

def _smoke(policy: InvSinglePolicy, episodes: int = 3) -> None:
    """Spec §12: prove normalization + transform reproduce eval-time behavior
    WITHOUT the gym stack, by driving `inv_single_mdp` directly."""
    from inv_single_mdp import advance1, advance2, init_state

    sc = policy.scenario
    print(f"[smoke] scenario={policy.scenario_name}  obs={policy.observation_mode}"
          f"  act={policy.action_mode}  (raw mdp loop, no gym)")
    for ep in range(episodes):
        state, _ = init_state(scenario=sc, episode_seed=ep)
        total, orders = 0.0, 0
        while not state.terminated:
            state1 = advance1(sc, state)
            # build the observation exactly as the gym would, from the raw state
            policy._env._state = state1
            policy._env._scenario_ep = sc
            obs = policy._env._get_obs()
            q = policy.act(obs)
            state, info = advance2(sc, state1, order=q)
            total += info["cost"]["total"]
            orders += int(q > 0)
        print(f"  episode {ep}: cost {total:9.2f}   orders {orders:2d}/{sc.horizon}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--vecnorm-path", default=None)
    p.add_argument("-s", "--scenario", default="simple")
    p.add_argument("-o", "--observation_mode", default="vec")
    p.add_argument("-a", "--action_mode", default="discrete")
    p.add_argument("--episodes", type=int, default=3)
    a = p.parse_args()
    pol = InvSinglePolicy(a.model_path, a.vecnorm_path, scenario=a.scenario,
                          observation_mode=a.observation_mode,
                          action_mode=a.action_mode)
    _smoke(pol, a.episodes)


if __name__ == "__main__":
    main()

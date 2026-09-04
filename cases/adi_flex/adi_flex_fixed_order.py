"""Order fixed at the AP policy; the protection levels left to the agent.

RQ2 asks whether a better protection-level policy exists than the paper's
constant-sigma heuristics. Every arm measured so far answers it only
indirectly, because the agent chooses the ORDER as well: an RL number below
PL(sigma) could come from better protection, better ordering, or both, and a
number above it says nothing about which half is at fault. #E11 sharpened the
question without settling it — the protection ENCODING helps at every rung
while the order encoding does not — but even there the agent still orders.

This wrapper removes the confound by construction. The order is taken from the
AP y* table, which is exactly what `adi_flex_benchmark_rule.py` plays for the
PL arms, so the ONLY thing that differs from PL(sigma) is who chooses sigma:

    PL(sigma)   AP order  +  CONSTANT sigma        -> 341.26 (#E1)
    this        AP order  +  sigma chosen by PPO   -> ?

The baseline is therefore not a bar the agent has to be compared against by
protocol — it IS the same policy with one function replaced, so any difference
is attributable to the protection rule alone, and a difference in either
direction is a finding. Beating it answers RQ2's `discover` stance in the
affirmative; failing to beat it is evidence that the constant-sigma heuristic
is at or near the best available protection policy and the residual gap to the
AP bound lives in the ordering.

NOT AN ACTION MODE, and it cannot be one. Schema v1 requires every action mode
to drive every declared decision ("action mode {m} leaves decision(s) {...}
unencoded; every mode must drive all N decisions"), and this drives only
`allocate`. It also depends on a SOLVED artifact — the AP table under
`results/` — which the IR cannot declare and which would make the domain
unportable. It is a research instrument over the domain, in the same category
as a probe, and is recorded in the escalation log rather than in the schema.

    from adi_flex_fixed_order import FixedOrderProtectEnv
    env = FixedOrderProtectEnv(scenario=SCENARIOS["het_exp4"],
                               ap_table="results/het_exp4/ap/het_exp4_policy.npz")
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from adi_flex_gym import AdiFlexEnv
from adi_flex_scenarios import SCENARIOS, AdiFlexScenario, AdiFlexScenarioSampler

U_MIN = -150          # the AP/DP grid's lower edge; mirrors adi_flex_benchmark_rule


class FixedOrderProtectEnv(gym.Env):
    """`order_protection` with the order decision taken by the AP policy.

    The action is the protection vector alone — `MultiDiscrete([sigma_max + 1]
    * n_sigma)`, or `Discrete(sigma_max + 1)` where a single level is needed.
    Observations, dynamics, costs, seeding and the §9.7 artifacts are the inner
    env's, unchanged.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: AdiFlexScenario | AdiFlexScenarioSampler,
        ap_table: str | Path,
        observation_mode: str = "vec",
        reward_mode: str = "neg_cost",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()
        self.env = AdiFlexEnv(
            scenario=scenario, action_mode="order_protection",
            observation_mode=observation_mode, reward_mode=reward_mode,
            logger_filename=logger_filename,
        )
        assert self.env.alloc_enabled and self.env.n_sigma > 0, (
            "the protection decision is not live on this scenario "
            f"(alloc_enabled={self.env.alloc_enabled}, n_sigma={self.env.n_sigma}); "
            "there would be nothing for the agent to choose"
        )
        z = np.load(Path(ap_table))
        self._y = z["y"] if "y" in z.files else z[z.files[0]]
        self._u_min = int(z["u_min"]) if "u_min" in z.files else U_MIN
        self._d_max = int(z["d_max"]) if "d_max" in z.files else self._y.shape[-1] - 1

        self.observation_space = self.env.observation_space
        n = self.env.n_sigma
        self.action_space = (
            gym.spaces.Discrete(self.env.sigma_max + 1) if n == 1
            else gym.spaces.MultiDiscrete([self.env.sigma_max + 1] * n)
        )
        self.n_sigma = n

    # -- the AP order, computed exactly as the PL arms compute it -------------

    def _ap_order(self) -> int:
        """`clip(y*(period, u, vhat) - u, 0, order_max)`.

        `vhat` is LAST PERIOD'S GROSS FAR DRAW, `d[T_dl]`, read from the info
        dict — NOT `adv[T_dl - 1]` off the state. The two are interchangeable
        only on the homogeneous branch, where the forced maximal fill empties
        the near class first so `v < vhat` implies `adv[0] == 0`; on an
        allocating branch the policy may withhold a reserve and that property
        fails in 37.1% of states (F25/F26). Reading the profile here would
        therefore drive the table off a coordinate it was not solved for, and
        would do it silently — the run would simply score worse.
        """
        e = self.env
        s = e._state
        sc = e._scenario_ep
        u = int(s.inv + sum(s.pipe) - sum(s.adv))
        d = (e._info or {}).get("d", ())
        vhat = int(d[sc.T_dl]) if len(d) > sc.T_dl else 0      # 0 at period 0
        n_u = self._y.shape[1] if self._y.ndim == 3 else self._y.shape[0]
        ui = int(np.clip(u - self._u_min, 0, n_u - 1))
        vi = int(np.clip(vhat, 0, self._d_max))
        y = int(self._y[s.period, ui, vi] if self._y.ndim == 3 else self._y[ui, vi])
        return int(np.clip(y - u, 0, sc.order_max))

    # -- gym API --------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """All-true: every protection level in the box is feasible, because
        pre-demand the components share no budget (the IR's
        feasibility_strategy="none"). Kept so the algorithm axis does not move
        with the instrument — `a0` requires MaskablePPO.
        """
        n = (int(self.action_space.n) if self.n_sigma == 1
             else int(sum(self.action_space.nvec)))
        return np.ones(n, dtype=bool)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        sig = np.asarray(action).reshape(-1).astype(int).tolist()
        assert len(sig) == self.n_sigma, f"expected {self.n_sigma} levels, got {sig}"
        return self.env.step([self._ap_order(), *sig])

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # total_reward, _state, n_sigma-adjacent internals, the logger handles…
        return getattr(self.__dict__["env"], name)


if __name__ == "__main__":
    import argparse

    from adi_flex_benchmark_rule import protection_level

    p = argparse.ArgumentParser(description="constant-sigma check on the wrapper")
    p.add_argument("-s", "--scenario_name", default="het_exp4",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--ap-table", default=None)
    p.add_argument("--n-seeds", type=int, default=512)
    a = p.parse_args()

    sc = SCENARIOS[a.scenario_name]
    tbl = a.ap_table or f"results/{a.scenario_name}/ap/{a.scenario_name}_policy.npz"
    env = FixedOrderProtectEnv(scenario=sc, ap_table=tbl)
    for name in ("pl0", "plsigma", "plmax"):
        sig = protection_level(sc, name)
        tot = np.zeros(a.n_seeds)
        for ep in range(a.n_seeds):
            env.reset(seed=ep)
            done = False
            while not done:
                _, _, t, tr, _ = env.step([sig] * env.n_sigma)
                done = t or tr
            tot[ep] = env.total_reward
        print(f"  {name:<9} sigma={sig:<3} cost {-tot.mean():10.4f}")

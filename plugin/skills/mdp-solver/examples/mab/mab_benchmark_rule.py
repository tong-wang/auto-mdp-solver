"""The distilled two-constant index rule (#E26) as a first-class benchmark.

    argmax_i( pm_i + c(ttg) * psd_i ),   c(ttg) = c0 * (ttg/T)^p

with the campaign's fitted `c0 = 2.5`, `p = 0.15`. `pm`/`psd` are the exact
conjugate posterior mean and sd — the same `mab_bayes` builtins the gym's
`bayes` observation is built from — so this policy sees exactly what the
trained net saw and differs from it only in how it turns that into a choice.

**Why this exists as a benchmark rather than only inside a probe.** The rule is
this campaign's shipped deliverable: it outscores both the trained network and
thompson (#E26). It was originally produced and scored inside
`mab_policy_probe.py --part csweep`, which is a readback instrument — it needs
a fitted-c artifact and a trained run on disk before it will start. That made
the headline number reproducible only from a machine that already had the
campaign's run directories, which is not reproducible at all. Here it is a
policy like any other: no artifact, no network, no torch, scored by the same
spec-§9 loop as random/greedy/ucb1/thompson and writing the same TSV.

Deterministic — there is no policy RNG, hence no `POLICY_STREAM_*` id. Two
episodes with the same seed give identical trajectories, and the eval's
`--stochastic` distinction does not apply.

Scope, because a fitted constant is easy to over-trust (see PLAYBOOK LV8/LV9):

  * `c0 = 2.5` is fitted on GAUSSIAN payouts at K=10, T=1000. #E32 measured the
    level's dependence on pulls-per-arm, `c* = 1.204 + 0.286*ln n`, so 2.5 is
    cell-local; the SHAPE `(ttg/T)^p` survives the grid, the level does not.
  * A fixed quantile is inconsistent (#E34): this rule's regret is linear in T
    against thompson's log, so it crosses back at T ~ 14,000 total rounds.
    Above that horizon thompson is the better policy and this benchmark is
    the weaker one — which is the point of shipping it with its envelope.

Usage (from the domain folder):
    python mab_benchmark_rule_eval.py -s gauss_K10_T1000 --n-seeds 8192
    python mab_benchmark_rule.py            # 5-episode smoke over every scenario
"""

from __future__ import annotations

import numpy as np

from mab_bayes import bayes_post_mean, bayes_post_sd

#: the #E26 fit — see the module docstring for the scope these carry
C0 = 2.5
P = 0.15


class RulePolicy:
    """`argmax(pm + c0*(ttg/T)^p * psd)` — deterministic, no network."""

    def __init__(self, n_arms: int, is_gauss: bool, horizon: int,
                 c0: float = C0, p: float = P) -> None:
        self.n_arms = n_arms
        self.is_gauss = is_gauss
        self.horizon = int(horizon)
        self.c0 = float(c0)
        self.p = float(p)

    def act(self, pulls: list[int], payouts: list[float], period: int) -> int:
        # time-to-go as a FRACTION of this episode's horizon: the rule anneals
        # on relative time, which is what lets one pair of constants serve a
        # whole episode (and what #E37 found the trained generalist did NOT do)
        ttg = max(self.horizon - int(period), 1)
        c = self.c0 * (ttg / self.horizon) ** self.p
        pm = np.asarray(bayes_post_mean(pulls, payouts, self.is_gauss), dtype=float)
        psd = np.asarray(bayes_post_sd(pulls, payouts, self.is_gauss), dtype=float)
        return int(np.argmax(pm + c * psd))


def make_policy(env) -> RulePolicy:
    return RulePolicy(env.n_arms, env._is_gauss, env.horizon)


if __name__ == "__main__":
    from mab_scenarios import SCENARIOS
    from mab_gym import MabEnv

    for name in SCENARIOS:
        env = MabEnv(scenario=SCENARIOS[name])
        totals = []
        for seed in range(5):
            env.reset(seed=seed)
            pol = make_policy(env)
            done, total = False, 0.0
            while not done:
                _, r, done, _, _ = env.step(
                    pol.act(env._state.pulls, env._state.payouts, env._state.period)
                )
                total += r
            totals.append(total)
        print(f"{name:18s} rule 5-episode mean reward: {np.mean(totals):9.2f}")

"""Mab core MDP.

This module contains the domain-level simulator only.
It does not depend on Gym/Gymnasium and computes no reward (spec §6.4);
the realized payout travels in the info dict's ``payoff`` decomposition.

Key components:
1. MabState: simulation state (period, per-arm pull counts and payout totals).
2. State transition functions:
   a. init_state(scenario, episode_seed) — start an episode (all counts zero).
   b. advance(scenario, state, arm) — pull one arm: draw its payout, update
      the sufficient statistics, advance the period counter.

Event sequence per period (mab_schema.json dynamics):
    P — pull: the chosen arm's payout is drawn (branch-guarded source)
    U — update: pulls[arm] += 1; payouts[arm] += payout

The hidden arm means live on scenario.payout.means (realized per episode by
MabScenarioSampler); they are never part of MabState and surface only in the
diagnostic info fields mean_pulled / opt_mean for eval-time regret accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mab_scenarios import MabScenario


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MabState:
    """Full internal state of the simulator.

    This is the MDP state from the *simulator's* perspective; what the agent
    observes is decided by the gym wrapper (mab_gym.py) via its
    observation_mode. (pulls, payouts) is the exact sufficient statistic of
    the pull history for the arm means under both branch priors.
    """

    # current round (0-indexed; incremented after each advance())
    period: int

    # True once period == scenario.horizon
    terminated: bool

    # pulls[i] = number of times arm i has been pulled so far
    pulls: list[int]

    # payouts[i] = total payout collected from arm i so far
    payouts: list[float]

    seed_salt:    int = field(repr=False)
    episode_seed: int

    def copy(self) -> MabState:
        return MabState(
            period=self.period,
            terminated=self.terminated,
            pulls=list(self.pulls),
            payouts=list(self.payouts),
            seed_salt=self.seed_salt,
            episode_seed=self.episode_seed,
        )


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------


def init_state(
    scenario: MabScenario,
    episode_seed: int,
) -> tuple[MabState, dict]:
    """Initialize a fresh episode: no pulls yet, all statistics zero."""
    state = MabState(
        seed_salt=scenario.seed_salt,
        episode_seed=episode_seed,
        period=0,
        terminated=False,
        pulls=[0] * scenario.n_arms,
        payouts=[0] * scenario.n_arms,
    )
    info = {
        "action_period": state.period - 1,  # no action taken yet
        "arm":           -1,
        "payout":        0.0,
        "mean_pulled":   0.0,
        "opt_mean":      scenario.payout.opt_mean(),
        "payoff": {
            "pull":  0.0,
            "total": 0.0,
        },
    }
    return state, info


def advance(
    scenario: MabScenario,
    state:    MabState,
    arm:      int,
) -> tuple[MabState, dict]:
    """Pull ``arm``: draw its payout, update the statistics, advance time."""
    assert not state.terminated, (
        "Episode is already finished. Call init_state() to start a new one."
    )
    arm = int(arm)
    assert 0 <= arm < scenario.n_arms, \
        f"arm must be in [0, {scenario.n_arms - 1}], got {arm}."

    state2 = state.copy()

    # P event: draw the pulled arm's payout (decision-conditioned generator;
    # seed key contains only (period, episode_seed, seed_salt) — spec §6.3)
    payout = scenario.payout.sample(state2, arm)

    # U event: update the sufficient statistics
    state2.pulls[arm]   += 1
    state2.payouts[arm] += payout

    state2.period    += 1
    state2.terminated = state2.period >= scenario.horizon

    info = {
        # action_period = input state.period (state2.period has advanced)
        "action_period": state.period,
        "arm":           arm,
        "payout":        payout,
        "mean_pulled":   scenario.payout.mean(arm),
        "opt_mean":      scenario.payout.opt_mean(),
        "payoff": {
            "pull":  float(payout),
            "total": float(payout),
        },
    }
    return state2, info


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

def _run_test(scenario_name: str, episode_seed: int = 42, periods: int = 12) -> None:
    from mab_scenarios import SCENARIOS

    source = SCENARIOS[scenario_name]
    scenario = source(episode_seed) if callable(source) else source
    print(f"\n=== {scenario_name} | {scenario.payout!r} ===")
    state, _ = init_state(scenario, episode_seed)
    total = 0.0
    for t in range(periods):
        arm = t % scenario.n_arms  # round-robin smoke policy
        state, info = advance(scenario, state, arm)
        total += info["payoff"]["total"]
        print(
            f"t={info['action_period']:3d}  arm={info['arm']}  "
            f"payout={info['payout']:6.2f}  mean={info['mean_pulled']:5.2f}  "
            f"opt={info['opt_mean']:5.2f}  pulls={state.pulls}"
        )
    print(f"partial total after {periods} pulls: {total:.2f}")

    # full-horizon termination check
    state, _ = init_state(scenario, episode_seed)
    steps = 0
    while not state.terminated:
        state, info = advance(scenario, state, steps % scenario.n_arms)
        steps += 1
    assert steps == scenario.horizon, (steps, scenario.horizon)
    print(f"full episode: {steps} pulls, terminated correctly")


if __name__ == "__main__":
    _run_test("bern_K10_T1000")
    _run_test("gauss_K10_T1000")

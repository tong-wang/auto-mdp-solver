"""Episode-seed provenance across resets (spec §7, upstream #18).

The deviation under test is a two-line omission from the mandated ``reset()``
pattern: no ``else`` branch, so ``_episode_seed`` survives an unseeded reset and
every auto-reset episode replays the first one. It is invisible to the rest of
the suite — ``behavior.determinism`` asserts the mirror property (same explicit
seed ⇒ identical trajectories), which the deviation satisfies perfectly.

The fixtures are synthetic gyms rather than a shipped example: these tests must
not depend on ``plugin/``, and the property is about ``reset()`` alone, so a
minimal env expresses it exactly.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from gymnasium import Env, spaces

from mdp_conformance.checks import check_determinism, check_gym_reseed
from mdp_conformance.loader import DomainHandle


class _Demand:
    """An exogenous generator keyed the v2 way: (period, episode_seed, salt)."""

    def sample(self, ctx):
        seq = np.random.SeedSequence([ctx.period, ctx.episode_seed, ctx.seed_salt])
        return int(np.random.default_rng(seq).integers(0, 100))


@dataclasses.dataclass
class _Scenario:
    demand: _Demand = dataclasses.field(default_factory=_Demand)
    horizon: int = 4
    seed_salt: int = 1


@dataclasses.dataclass
class _ConstantScenario:
    """No episode-seed sensitivity anywhere — the legitimate SKIP case."""

    demand: object = dataclasses.field(default_factory=lambda: _Constant())
    horizon: int = 4
    seed_salt: int = 1


class _Constant:
    def sample(self, ctx):
        return 7


class _CompliantEnv(Env):
    """Implements the §7 pattern, ``else`` branch included."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.observation_space = spaces.Box(low=0.0, high=1e6, shape=(2,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._episode_seed = 0
        self._period = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._episode_seed = seed
        else:
            self._episode_seed = int(self.np_random.integers(0, 2_147_483_647))
        self._period = 0
        return self._obs(), {}

    def _obs(self):
        draw = self.scenario.demand.sample(
            _Ctx(self._period, self._episode_seed, self.scenario.seed_salt))
        return np.array([self._period, draw], dtype=np.float32)

    def step(self, action):
        self._period += 1
        terminated = self._period >= self.scenario.horizon
        return self._obs(), 0.0, terminated, False, {}


class _StaleSeedEnv(_CompliantEnv):
    """The deviation: no ``else``, so an unseeded reset keeps the old seed."""

    def reset(self, *, seed=None, options=None):
        Env.reset(self, seed=seed, options=options)
        if seed is not None:
            self._episode_seed = seed
        self._period = 0
        return self._obs(), {}


class _Ctx:
    def __init__(self, period, episode_seed, seed_salt):
        self.period = period
        self.episode_seed = episode_seed
        self.seed_salt = seed_salt


def _handle(env_cls, scenario_cls=_Scenario) -> DomainHandle:
    """A DomainHandle carrying only what the gym-level checks read."""
    return DomainHandle(
        name="synthetic",
        directory=None,
        files={},
        modules={},
        mdp_module_name="synthetic_mdp",
        SCENARIOS={"simple": scenario_cls()},
        env_cls=env_cls,
        state_cls=None,
        init_state=lambda scenario, seed: (None, {}),
    )


def test_compliant_domain_passes():
    result = check_gym_reseed(_handle(_CompliantEnv))
    assert result.status == "PASS", result.detail


def test_missing_else_branch_fails():
    result = check_gym_reseed(_handle(_StaleSeedEnv))
    assert result.status == "FAIL"
    assert "unseeded" in result.detail


def test_determinism_cannot_see_the_deviation():
    """Why the new check is needed: the existing gate passes on the broken env."""
    assert check_determinism(_handle(_StaleSeedEnv)).status == "PASS"
    assert check_determinism(_handle(_CompliantEnv)).status == "PASS"


def test_deterministic_domain_skips():
    """No episode-seed-sensitive generator ⇒ identical episodes are legitimate."""
    result = check_gym_reseed(_handle(_CompliantEnv, _ConstantScenario))
    assert result.status == "SKIP"

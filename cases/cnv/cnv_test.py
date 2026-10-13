"""cnv's own tests — the claims only this domain can make.

Run from anywhere (the portable-domain contract):

    pytest cnv
    python cnv_test.py

Everything true of *any* IR — determinism, decision-path independence,
termination, declared invariants — is checked by ``mdp_ir.laws`` and only
invoked here. What lives in this file is cnv-specific: the bit-exact
differential over its covering set, the equivalence between the family-generic
runtime and the hand-written latent generator, the vectorised epoch draw's
equality with the interpreter's sequential draw, the grids' agreement with the
IR's own enumeration, the rendered observation vectors against the declared
ones, and the negative controls that prove the gates can fail.

The conservation claims live in the IR as ``mdp.invariants`` (the running
totals' balances, the censoring identities), so every instance inherits them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mdp_ir import runtime
from mdp_ir.interpreter import IrInterpreter
from mdp_ir.schema import load_ir
from mdp_ir.testing import (
    assert_diverges,
    assert_laws,
    assert_match,
    mutated,
    schema_beside,
)

SCHEMA = schema_beside(__file__)
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

EPISODES = 8


def _declared_compositions() -> list[str | None]:
    """Base + every declared instance, read from the schema so the sweep
    cannot drift from the catalog it is meant to cover."""
    raw = json.loads(SCHEMA.read_text())
    scenario = raw["mdp"]["scenario"]
    return [None] + sorted(scenario.get("instances") or {})


COMPOSITIONS = _declared_compositions()


# -- engine laws -------------------------------------------------------------


def test_engine_laws():
    assert_laws(SCHEMA)


# -- the covering set: one node per declared composition ---------------------


@pytest.mark.parametrize("instance", COMPOSITIONS, ids=lambda i: i or "base")
def test_differential_matches_the_domain(instance):
    """The interpreter and the hand-written domain must agree bit-for-bit on
    every declared composition — both N = 4 priors, both unit costs, and the
    N = 100 long-horizon pair."""
    assert_match(SCHEMA, instance=instance, episodes=EPISODES)


def test_the_covering_set_is_not_silently_empty():
    assert len(COMPOSITIONS) >= 5, COMPOSITIONS
    assert "fig6a" in COMPOSITIONS


# -- negative controls: the gates must be able to fail -----------------------


def test_a_corrupted_dynamics_update_diverges():
    """Drop the sales total's update from the U event: the domain still
    accumulates, the interpreter no longer does, so the rows must part."""
    def drop_total(doc):
        doc["mdp"]["dynamics"]["transitions"][1]["updates"] = [
            "cum_duration += stockout_time", "cum_demand += demand",
        ]
    assert_diverges(SCHEMA, mutated(load_ir(SCHEMA), drop_total), episodes=2)


def test_a_mis_stated_claim_is_caught_while_the_sides_still_agree():
    """The reason invariants exist: claim the duration total always grows by
    a full period and the claim breaks on the first stock-out, even though
    the interpreter and the domain remain bit-identical."""
    def weaken(doc):
        for inv in doc["mdp"]["invariants"]:
            if inv["name"] == "cum_duration_balance":
                inv["expr"] = "close(cum_duration, prev.cum_duration + 1.0)"
    ir = mutated(load_ir(SCHEMA), weaken)
    traj = IrInterpreter(ir, seed_salt=1).run(3, decisions={"order": 3})   # y=3 stocks out
    assert traj.violations, "a wrong balance claim went unnoticed"


# -- the hand-written generators equal the generic runtime -------------------


@pytest.fixture(scope="module")
def uncertainty():
    import cnv_uncertainty
    return cnv_uncertainty


def test_generic_runtime_matches_the_latent_poisson_generator(uncertainty):
    """``FamilyDemand.from_ir`` (the zero-code path) and ``LatentPoissonDemand``
    realize the same lambda and draw the same demand on the same keys."""
    ir = load_ir(SCHEMA, instance="fig3_c10")
    generic = uncertainty.FamilyDemand.from_ir(ir, "demand", instance="fig3_c10")
    hand = uncertainty.LatentPoissonDemand(alpha=0.0625 * 10.0, beta=0.0625, source_id=0)

    class Ctx:
        def __init__(self, period, seed):
            self.period, self.episode_seed, self.seed_salt = period, seed, 1

    for seed in range(5):
        g, h = generic.realize(seed, 1), hand.realize(seed, 1)
        assert g.settings["rate"] == h.rate
        for t in range(4):
            assert g.sample(Ctx(t, seed)) == h.sample(Ctx(t, seed))


def test_vectorised_epochs_equal_the_interpreters_sequential_draw(uncertainty):
    """``UniformEpochs`` draws its vector in one call; the interpreter draws
    ``size`` scalars from the same stream. numpy consumes the stream
    identically either way — the claim the differential's MATCH rests on."""
    epochs = uncertainty.UniformEpochs(size=402, source_id=1)

    class Ctx:
        period, episode_seed, seed_salt = 2, 17, 1

    rng = np.random.default_rng(np.random.SeedSequence(
        uncertainty.intrinsic_key(1, Ctx)))
    sequential = [runtime.sample_family(rng, "uniform", {"low": 0.0, "high": 1.0})
                  for _ in range(402)]
    assert np.array_equal(epochs.sample(Ctx), np.asarray(sequential))


def test_stockout_time_is_the_order_statistic():
    """t^s is the y-th smallest of the first D epochs; y = 0 gives 0 and no
    stock-out gives 1 (the paper's eq. 7 edge cases)."""
    import cnv_mdp
    u = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2])
    assert cnv_mdp.stockout_time_of(u, demand=4, order=2) == 0.3   # 2nd smallest of [0.9,0.1,0.5,0.3]
    assert cnv_mdp.stockout_time_of(u, demand=4, order=4) == 0.9
    assert cnv_mdp.stockout_time_of(u, demand=4, order=5) == 1.0
    assert cnv_mdp.stockout_time_of(u, demand=4, order=0) == 0.0
    assert cnv_mdp.stockout_time_of(u, demand=0, order=0) == 0.0


# -- the grids materialize the IR's declaration ------------------------------


@pytest.mark.parametrize("grid_name", ["paper510", "fig3"])
def test_grids_match_the_ir_declaration(grid_name):
    """Cell ids, order and per-cell constants agree with the schema's
    ``ScenarioGrid.cells()`` — the leaderboard join key."""
    import cnv_grids
    ir = load_ir(SCHEMA)
    declared = next(g for g in ir.grids if g.name == grid_name)
    base_over = dict(ir.mdp.scenario.instances.get(declared.base_instance, {}))
    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    consts.update(base_over)
    grid = cnv_grids.GRIDS[grid_name]
    cells = list(declared.cells())
    assert [cid for cid, _ in grid] == [cid for cid, _ in cells]
    for (cid, over), (_, src) in zip(cells, grid):
        cell = {**consts, **over}
        assert src.unit_cost == cell["unit_cost"], cid
        assert src.prior_rate == cell["prior_rate"], cid
        assert src.prior_mean == cell["prior_mean"], cid
        assert src.horizon == ir.mdp.horizon_T(declared.base_instance or None)
        assert src.demand.alpha == cell["prior_rate"] * cell["prior_mean"]


def test_grid_sampler_is_pure_and_realizes_the_cell_standalone():
    """A drawn cell's episode is bit-identical to that cell run alone on the
    same seed (standalone equivalence), and the sampler is a pure function."""
    import cnv_grids
    sampler = cnv_grids.GRIDS["fig3"].as_sampler()
    for seed in (0, 7, 123):
        a, b = sampler(seed), sampler(seed)
        assert a == b
        cell_name = a.scenario_name
        standalone = next(src for cid, src in cnv_grids.GRIDS["fig3"]
                          if src.scenario_name == cell_name)(seed)
        assert standalone.demand.rate == a.demand.rate


# -- the rendered vector is the declared vector -------------------------------


def test_the_rendered_vector_is_the_declared_vector():
    """Each gym mode renders exactly the IR mode's features, in order:
    evaluated on the interpreter's own END_OF_PERIOD namespace and compared
    with the gym after the same actions (spec §7)."""
    import cnv_gym
    from cnv_scenarios import SCENARIOS
    ir = load_ir(SCHEMA, instance="fig3_c10")
    source = SCENARIOS["fig3_c10"]
    modes = {m.name: m for m in ir.gym.observation_modes}
    assert set(cnv_gym.OBSERVATION_MODES) == set(modes)
    interp = IrInterpreter(ir, instance="fig3_c10", seed_salt=source.seed_salt)
    orders = [10, 3, 0, 25]
    traj = interp.run(5, decisions=lambda ns: {"order": orders[int(ns["period"])]})
    for name, mode in modes.items():
        env = cnv_gym.CnvEnv(source, observation_mode=name)
        obs, _ = env.reset(seed=5)
        rendered = [obs]
        for y in orders:
            obs, *_ = env.step(y)
            rendered.append(obs)
        # the interpreter's rows are END_OF_PERIOD namespaces; row k is the
        # state the gym renders after step k
        for k, row in enumerate(traj.rows):
            declared = list(interp.observe(name, {**row, "period": row["t"] + 1}).values())
            assert np.allclose(np.asarray(declared, dtype=float), rendered[k + 1]), (name, k)


def test_event_belief_log_agrees_with_the_benchmark_grid():
    """The gym's event posterior (log-spaced grid, cnv_event_posterior) and the benchmark's
    MyopicEvent (linear grid) carry the same belief: along random play on the event information set
    they place the same myopic order and agree on the posterior mean to the grids' resolution."""
    import cnv_gym
    from cnv_benchmark_myopic import MyopicEvent
    from cnv_scenarios import SCENARIOS
    source = SCENARIOS["fig6a"]
    rng = np.random.default_rng(7)
    env = cnv_gym.CnvEnv(source, observation_mode="event_belief_log")
    n_steps = same = 0
    for ep in range(6):
        obs, _ = env.reset(seed=100 + ep)
        ref = MyopicEvent(env)
        for _ in range(int(source.horizon)):
            y_ref = ref(env)                      # absorbs the last period's observation, returns y_myo on its grid
            y_gym = env.myopic_order()
            w = np.exp(ref.log_w); w /= w.sum()
            m_ref = float(w @ ref.lam)
            m_gym = float(np.exp(obs[2]))
            assert abs(m_gym - m_ref) <= 0.02 * m_ref + 0.05, (ep, m_gym, m_ref)
            same += int(y_gym == y_ref); n_steps += 1
            y = int(rng.integers(0, 3)) * int(rng.integers(0, 12)) if rng.random() < 0.5 else y_gym
            obs, *_ = env.step(y)
    assert same / n_steps >= 0.98, same / n_steps


def test_no_mode_exposes_the_hidden_rate():
    import cnv_gym
    from cnv_scenarios import SCENARIOS
    source = SCENARIOS["base"]
    for name in cnv_gym.OBSERVATION_MODES + cnv_gym.HISTORY_BELIEF_MODES:
        env = cnv_gym.CnvEnv(source, observation_mode=name)
        obs, _ = env.reset(seed=11)
        lam = env._scenario_ep.demand.rate
        assert not np.any(np.isclose(obs, lam)), (name, obs, lam)


# -- the expression builtins ----------------------------------------------------


def test_poisson_quantile_is_the_smallest_integer_reaching_q():
    import math
    from cnv_bayes import gamma_poisson_quantile, poisson_quantile
    for rate in (0.5, 3.0, 10.0, 57.3):
        for q in (0.1, 0.5, 0.9, 0.99):
            y = poisson_quantile(rate, q)
            cdf = lambda k: sum(math.exp(j * math.log(rate) - rate - math.lgamma(j + 1))
                                for j in range(k + 1))
            assert cdf(y) >= q and (y == 0 or cdf(y - 1) < q)
    # the predictive is negative binomial NB(r = shape, p = 1 / (rate + 1)):
    # the same smallest-integer property against a brute-force cdf
    for shape, rate in ((10.0, 1.0), (0.625, 0.0625), (3.125, 0.0625)):
        p = 1.0 / (rate + 1.0)
        pmf = lambda k: math.exp(math.lgamma(k + shape) - math.lgamma(shape) - math.lgamma(k + 1)
                                 + k * math.log(p) + shape * math.log(1 - p))
        cdf = lambda k: sum(pmf(j) for j in range(k + 1))
        for q in (0.1, 0.5, 0.9):
            y = gamma_poisson_quantile(shape, rate, q)
            assert cdf(y) >= q and (y == 0 or cdf(y - 1) < q), (shape, rate, q, y)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# -- the batched rollout is the sequential loop, per seed --------------------


def test_batched_rollout_equals_the_sequential_loop_per_seed():
    """``rollout_batched`` (the protocol eval's path for trained policies)
    must reproduce ``rollout`` seed by seed. Run with a fixed decision rule
    that reads the observation — the batched path's only new ingredient is
    the lockstep loop, so any divergence is in the loop, not the policy."""
    import numpy as np
    from cnv_benchmark_common import eval_seed_block, rollout, rollout_batched
    from cnv_scenarios import SCENARIOS
    from cnv_grids import GRIDS
    def rule_single(env, obs):
        return int(round(float(np.asarray(obs).reshape(-1)[-2]))) % 30 + 3   # a rule of the vector
    def rule_batch(obs):
        return np.array([int(round(float(np.asarray(o).reshape(-1)[-2]))) % 30 + 3 for o in obs])
    for source, mode, fs in ((SCENARIOS["base"], "timing_belief", 1), (GRIDS["fig3"].as_sampler(), "timing_raw", 4)):
        seeds = eval_seed_block(0, 70)
        a = rollout(source, seeds, rule_single, observation_mode=mode, frame_stack=fs, progress_every=0)
        b = rollout_batched(source, seeds, rule_batch, observation_mode=mode, frame_stack=fs, batch=16, progress_every=0)
        for m in a:
            assert np.array_equal(a[m], b[m]), (mode, m)


# -- the probe's instruments read their reference correctly (spec §14.1) -----

def test_gamma_poisson_quantile_survives_an_underflowing_zero_term():
    """A large shape on a small rate underflows P(D = 0) to 0.0; the recursion used to stay at 0 forever
    (a probe's belief sweep hung on it, 2026-10-13). The log-space branch must still return the smallest
    integer reaching q."""
    import math
    from cnv_bayes import gamma_poisson_quantile
    shape, rate = 720.0, 0.5
    p = 1.0 / (rate + 1.0)
    assert (1.0 - p) ** shape == 0.0                       # the case the branch exists for
    log_pmf = lambda k: (math.lgamma(k + shape) - math.lgamma(shape) - math.lgamma(k + 1)
                         + k * math.log(p) + shape * math.log1p(-p))
    for q in (0.1, 0.5, 0.9):
        y = gamma_poisson_quantile(shape, rate, q)
        cdf = lambda k: sum(math.exp(log_pmf(j)) for j in range(k + 1))
        assert cdf(y) >= q and cdf(y - 1) < q, (q, y)


@pytest.mark.parametrize("source_name,mode,rule", [
    ("base", "timing_belief", "myopic_timing"), ("base", "full_raw", "myopic_full"), ("base", "event_raw", "myopic_event"),
    ("paper510", "timing_belief_ctx", "myopic_timing"), ("paper510", "full_belief_ctx", "myopic_full")])
def test_probe_self_check_agrees_on_every_observation_type(source_name, mode, rule):
    """The probe's on-policy reference is the myopic level of the mode's own TYPE (the _ctx and raw modes
    included — the probe once compared timing_belief_ctx against myopic_full); the event self-check plays an
    independent grid posterior, so agreement also validates the probe's copied prior grid."""
    from cnv_benchmark_common import eval_seed_block
    from cnv_grids import GRIDS
    from cnv_policy_probe import on_policy, summarize
    from cnv_scenarios import SCENARIOS
    source = SCENARIOS[source_name] if source_name in SCENARIOS else next(iter(GRIDS[source_name]))[1]
    fs = 4 if mode.endswith("_raw") else 1
    rows = on_policy(None, rule, source, eval_seed_block(1_000_000, 16), mode, "order_discrete", fs)
    assert summarize(rows)["agree_myopic"] == 1.0

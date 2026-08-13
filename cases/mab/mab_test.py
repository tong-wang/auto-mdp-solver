"""mab's own tests — the bandit exemplar.

    pytest mab
    python mab_test.py

The generic guarantees live in ``mdp_ir.laws``; the ledger and regret-sign
laws live in the IR as ``mdp.invariants`` (so both branches and any appended
candidate inherit them). What is here is what only this domain can state: that
the payout is a function of the round and the arm rather than of the pull
history, that the censoring actually holds — no observation or reward mode
reads the hidden arm means — and that the conjugate belief features the
``bayes`` observation is built from are the real posteriors.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

from mdp_ir.interpreter import IrInterpreter, simulate
from mdp_ir.schema import load_ir
from mdp_ir.testing import (
    assert_diverges,
    assert_laws,
    assert_match,
    mutated,
    schema_beside,
)

from mab_bayes import bayes_post_mean, bayes_post_sd

SCHEMA = schema_beside(__file__)
EPISODES = 4        # x T=1000 periods x every composition
RAW = json.loads(SCHEMA.read_text())


def _compositions() -> list[str | None]:
    """Base + every declared instance + every mixture, read from the schema so
    the sweep cannot drift from the catalog it is meant to cover."""
    scenario = RAW["mdp"]["scenario"]
    return ([None] + sorted(scenario.get("instances") or {})
            + sorted(m["name"] for m in scenario.get("mixtures") or []))


COMPOSITIONS = _compositions()


def test_engine_laws():
    assert_laws(SCHEMA)


def _episodes_for(instance: str | None) -> int:
    """Episode budget for the differential, scaled so total PERIODS stay
    bounded. #E35 registered cells up to T=40,000; at a flat 4 episodes the
    suite went 31s -> 347s, all of it the pure-Python IR interpreter replaying
    long episodes. A mismatch shows in the first periods, so trading episodes
    for horizon costs no coverage — every instance is still checked."""
    scen = RAW["mdp"]["scenario"]
    consts = {c["name"]: c["value"] for c in scen["constants"]}
    if instance:
        consts.update(scen["instances"].get(instance, {}))
    T = RAW["mdp"]["horizon"]["T"]
    T = consts[T] if isinstance(T, str) else T
    return max(1, round(EPISODES * 1000 / T))


@pytest.mark.parametrize("instance", COMPOSITIONS, ids=lambda i: i or "base")
def test_differential_matches_the_domain(instance):
    """Base is the Bernoulli branch, `gaussian` is the Gaussian one."""
    assert_match(SCHEMA, instance=instance, episodes=_episodes_for(instance))


def test_the_covering_set_holds_both_branches():
    """Guards the parametrization: an empty derived list would let every case
    above 'pass' by not existing."""
    assert None in COMPOSITIONS, "the base (Bernoulli) branch must be covered"
    assert "gaussian" in COMPOSITIONS
    assert len(COMPOSITIONS) == len(set(COMPOSITIONS))


# -- the censoring: hidden means reach info, never the policy -----------------


def _hidden_latents() -> set[str]:
    """The per-seed world latents the agent is meant to infer, as the constant
    names they desugar to. In the catalog form a latent is a hidden draw spec
    inside a candidate's settings, synthesized at load time into
    ``{slot}_{setting}`` — derived here, so a new latent is covered on sight."""
    return {
        f"{slot['name']}_{key}"
        for slot in RAW["mdp"]["uncertainty_slots"]
        for cand in slot["candidates"].values()
        for key, raw in (cand.get("settings") or {}).items()
        if isinstance(raw, dict) and "draw" in raw and raw.get("hidden")
    }


# the two info fields that deliberately expose the latents for eval-time regret
_LATENT_INFO = {"mean_pulled", "opt_mean"}


def test_the_hidden_latents_and_their_diagnostics_exist():
    """Non-vacuity guard for the leakage test below: if these names vanished,
    the scan would trivially find nothing to leak."""
    latents = _hidden_latents()
    assert latents, "no hidden candidate latents found — the scan would be vacuous"
    # both recipes name their latent identically, so the synthesized constant
    # exists under either selection and the dynamics can reference it once
    assert latents == {"payout_arm_means"}, latents
    assert _LATENT_INFO <= {f["name"] for f in RAW["mdp"]["info_fields"]}


def test_every_candidate_draws_its_arm_means_as_one_iid_vector():
    """K independent components from one recipe, rather than K hand-listed
    scalar draws: the arms are i.i.d., so the latent is one `iid` draw whose
    size is the arm count.

    #E32 promoted K to the axis constant `n_arms`, so the linkage is now by
    NAME and must hold for every cell at once. Decision bounds resolve constant
    names and not expressions, so the index ceiling `arm_max` is hand-carried;
    that it equals `n_arms - 1` in the base and in every instance is exactly
    the coupling nothing else would catch."""
    scen = RAW["mdp"]["scenario"]
    consts = {c["name"]: c["value"] for c in scen["constants"]}
    assert RAW["mdp"]["decisions"][0]["bounds"]["value"][1] == "arm_max"
    cands = RAW["mdp"]["uncertainty_slots"][0]["candidates"]
    assert cands, "no candidates — the assertion below would be vacuous"
    for name, cand in cands.items():
        draw = cand["settings"]["arm_means"]["draw"]
        assert draw["family"] == "iid", name
        assert draw["settings"]["size"] == "n_arms", name
        assert draw["settings"]["of"] in ("uniform", "normal"), name

    compositions = [("base", {})] + sorted((scen.get("instances") or {}).items())
    assert len(compositions) > 1, "no instances — the sweep below would be thin"
    for inst, overrides in compositions:
        c = {**consts, **overrides}
        assert c["arm_max"] == c["n_arms"] - 1, (
            f"{inst}: arm_max={c['arm_max']} must be n_arms-1={c['n_arms'] - 1}")


def test_no_observation_or_reward_mode_reads_the_hidden_means():
    """Spec: latent is safe in an eval-only metric, a terminal reward, or an
    action-independent normalizer — never in a per-step reward the policy
    consumes, because a memory policy fed r_{t-1} can back the latent out and
    the censoring is gone. Here the per-step reward is the *realized* payout,
    an observable, and the true means appear only in `info`."""
    forbidden = _hidden_latents() | _LATENT_INFO
    exprs: list[tuple[str, str]] = []
    for mode in RAW["gym"]["observation_modes"]:
        for feature in mode["features"]:
            exprs.append((f"obs:{mode['name']}",
                          feature.get("expr") or feature["ref"]))
    for mode in RAW["gym"]["reward_modes"]:
        exprs.append((f"reward:{mode['name']}", mode["expr"]))

    assert exprs, "no gym modes found — the scan would be vacuous"
    for where, expr in exprs:
        leaked = set(re.findall(r"[A-Za-z_]\w*", expr)) & forbidden
        assert not leaked, f"{where} expr {expr!r} leaks hidden state {sorted(leaked)}"


# -- the payout process ------------------------------------------------------


def test_the_payout_depends_on_the_round_and_arm_not_the_pull_history():
    """The seed key is (arm, period, source, episode, salt) — no pull counts in
    it — so the payout of a given arm in a given round is fixed by the episode
    seed whatever route the policy took to get there. ``mdp_ir.laws`` skips
    this one — it cannot assert
    path-independence for a state/decision-conditioned draw, which is exactly
    what a bandit payout is — so the bandit-shaped form is stated here: two
    histories that diverge early but agree on the arm played now must see the
    same payout now."""
    ir = load_ir(SCHEMA)
    always_0 = simulate(ir, 11, decisions={"arm": 0})
    switch = simulate(
        ir, 11, decisions=lambda ns: {"arm": 3 if ns["period"] < 500 else 0})
    tail_a = [r["reward"] for r in always_0.rows[500:]]
    tail_b = [r["reward"] for r in switch.rows[500:]]
    assert tail_a == tail_b
    assert [r["reward"] for r in always_0.rows[:500]] != \
           [r["reward"] for r in switch.rows[:500]], \
        "the two histories must actually differ before they re-converge"


def _stream_compositions() -> list[str | None]:
    """Representative subset for the per-arm stream test below. The property is
    about the SEED SCHEME — that every (arm, period) owns its own stream — so it
    varies with the arm count and the payout family, and not at all with T.
    Running it on all 34 compositions cost 100s on gauss_K20_T40000 alone; the
    smallest cell at each K covers exactly the same ground. Both families are
    kept (base = Bernoulli, gaussian = Normal), and the guard below fails if a
    named cell disappears."""
    scen = RAW["mdp"]["scenario"]
    by_k: dict[int, tuple[str, int]] = {}
    for name, ov in (scen.get("instances") or {}).items():
        if "n_arms" not in ov:
            continue
        K, T = ov["n_arms"], ov["horizon_T"]
        if K not in by_k or T < by_k[K][1]:
            by_k[K] = (name, T)
    return [None, "gaussian"] + sorted(n for n, _ in by_k.values())


STREAM_COMPOSITIONS = _stream_compositions()


def test_the_stream_subset_covers_every_arm_count():
    """Guards the subset: it must still exercise every registered K, or the
    test below silently stops covering an arm count."""
    scen = RAW["mdp"]["scenario"]
    consts = {c["name"]: c["value"] for c in scen["constants"]}
    all_k = {consts["n_arms"]} | {ov["n_arms"] for ov in
                                  (scen.get("instances") or {}).values()
                                  if "n_arms" in ov}
    covered = set()
    for inst in STREAM_COMPOSITIONS:
        c = dict(consts)
        if inst:
            c.update(scen["instances"][inst])
        covered.add(c["n_arms"])
    assert covered == all_k, f"arm counts not covered: {sorted(all_k - covered)}"


@pytest.mark.parametrize("instance", STREAM_COMPOSITIONS, ids=lambda i: i or "base")
def test_the_arms_are_independent_streams_within_a_round(instance):
    """The `pull` stage is keyed on the chosen arm (`key_exprs`, solver >=
    v0.5.9), so every (arm, period) pair owns its own stream and the round
    reads one of them. Before that key existed a single primitive variate per
    period was shared by all ten arms, so the round's luck was fixed before the
    choice was made — leaving two decisive fingerprints, both asserted absent
    here:

    * Gaussian — one shared z_t made ``r_i(t) - r_j(t)`` exactly ``mu_i - mu_j``
      at *every* t, i.e. a difference sequence with a single distinct value.
    * Bernoulli — one shared u_t made the arms comonotone in p: a win at the
      lowest-p arm forced a win at every higher-p arm, so an inversion was
      impossible.

    This is the regression test for that coupling, which the other tests in
    this file passed straight through — the realized law the agent sees was
    correct, because exactly one arm is revealed per round.

    K and the payout family are both read from the *resolved* composition
    (#E32 made n_arms an axis constant, so the cells vary in arm count and the
    branch can no longer be inferred from the instance name)."""
    ir = load_ir(SCHEMA, instance=instance) if instance else load_ir(SCHEMA)
    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance:
        consts.update(ir.mdp.scenario.instances[instance])
    n_arms = int(consts["n_arms"])
    payout_src = next(s for s in ir.mdp.uncertainty_sources if s.name == "payout")
    is_gauss = payout_src.distribution.family != "bernoulli"

    runs = {arm: IrInterpreter(ir, instance=instance, seed_salt=1)
                 .run(5, decisions={"arm": arm}).rows for arm in range(n_arms)}
    rewards = {arm: [r["reward"] for r in rows] for arm, rows in runs.items()}
    means = {arm: rows[0]["mean_pulled"] for arm, rows in runs.items()}
    assert len(set(means.values())) == n_arms, "arms must have distinct means here"

    if is_gauss:
        # spread-out index pairs, valid for any K >= 2
        pairs = {(0, n_arms - 1), (1 % n_arms, max(0, n_arms - 2)), (0, n_arms // 2)}
        for i, j in sorted(p for p in pairs if p[0] != p[1]):
            spread = {round(x - y, 9) for x, y in zip(rewards[i], rewards[j])}
            assert len(spread) > 1, (
                f"arms {i},{j} differ by a constant {spread} over the whole "
                "episode — they are sharing one primitive draw per period")
    else:
        inversions = sum(
            1
            for i, j in ((a, b) for a in range(n_arms) for b in range(n_arms)
                         if means[a] < means[b])
            for x, y in zip(rewards[i], rewards[j])
            if x == 1.0 and y == 0.0
        )
        assert inversions > 0, (
            "no round where a lower-p arm paid while a higher-p arm did not — "
            "the arms are comonotone, i.e. sharing one uniform per period")


def test_pulling_one_arm_forever_learns_only_about_that_arm():
    """The identification problem: information is arm-local, so a policy that
    never explores keeps the prior on the other nine arms."""
    rows = simulate(load_ir(SCHEMA), 5, decisions={"arm": 4}).rows
    pulls, payouts = rows[-1]["pulls"], rows[-1]["payouts"]
    assert pulls[4] == len(rows)
    assert sum(pulls) == pulls[4]
    assert all(payouts[i] == 0.0 for i in range(10) if i != 4)


def test_only_the_selected_candidate_resolves():
    """The two recipes are alternatives for one source, not two coexisting
    sources: whichever is selected, the IR resolves to a single `payout` source
    at a single stream id, with one synthesized latent sampler on the same id
    (one stream identity per source, both seed branches). Nothing is guarded,
    because there is nothing to guard against."""
    slot = RAW["mdp"]["uncertainty_slots"][0]
    for instance in COMPOSITIONS:
        ir = load_ir(SCHEMA, instance=instance)
        sources = ir.mdp.uncertainty_sources
        assert [s.name for s in sources] == ["payout"], instance
        samplers = ir.mdp.scenario.samplers
        assert len(samplers) == 1, instance
        assert samplers[0].substream_id == sources[0].stream_id == slot["stream_id"]
        assert samplers[0].hidden, instance
    assert not any(t.get("guard") for t in RAW["mdp"]["dynamics"]["transitions"])


def test_the_bernoulli_recipe_pays_only_zero_or_one():
    """Stated here rather than in `mdp.invariants` because the claim is about
    one *candidate*: an expression cannot name the selection, and the branch
    selector left the IR when the slot took over.

    The instance is named EXPLICITLY. This test used to rely on `bernoulli`
    being the slot default, so it silently changed meaning when the default
    moved to `gaussian` (F5) — it began asserting that Gaussian payouts are
    0/1, and failed. A test about one candidate must select that candidate.
    """
    rows = simulate(load_ir(SCHEMA, instance="bernoulli"), 3,
                    instance="bernoulli", decisions={"arm": 1}).rows
    assert rows, "no rows — the assertion would be vacuous"
    assert all(r["reward"] in (0.0, 1.0) for r in rows)


def test_the_gaussian_branch_pays_off_the_binary_support():
    """The branches are separate leaderboards because the payout scales differ;
    a Gaussian run that only ever returned 0/1 would mean the mode selector
    never took effect."""
    ir = load_ir(SCHEMA, instance="gaussian")
    rows = IrInterpreter(ir, instance="gaussian", seed_salt=1).run(
        7, decisions={"arm": 2}).rows
    assert any(r["reward"] not in (0.0, 1.0) for r in rows)
    assert any(r["reward"] < 0.0 for r in rows), \
        "N(mu, 1) payouts must be able to go negative"


# -- the belief features the `bayes` observation is built from ----------------


def _numeric_posterior(grid, log_weight):
    """Posterior mean/sd by quadrature — an independent derivation, so this
    catches a wrong conjugate update instead of restating it."""
    logw = log_weight(grid)
    w = np.exp(logw - logw.max())
    z = np.trapezoid(w, grid)
    mean = np.trapezoid(grid * w, grid) / z
    var = np.trapezoid((grid - mean) ** 2 * w, grid) / z
    return float(mean), math.sqrt(float(var))


@pytest.mark.parametrize("n,s", [(0, 0), (1, 0), (1, 1), (7, 3), (40, 31)])
def test_the_bernoulli_belief_is_the_beta_posterior(n, s):
    """Uniform(0,1) prior, n pulls, s successes — checked against quadrature
    over the exact likelihood rather than against the closed form."""
    grid = np.linspace(1e-9, 1 - 1e-9, 400_001)
    mean, sd = _numeric_posterior(
        grid, lambda p: s * np.log(p) + (n - s) * np.log1p(-p))
    pulls = [n] + [0] * 9
    payouts = [float(s)] + [0.0] * 9
    assert bayes_post_mean(pulls, payouts, False)[0] == pytest.approx(mean, rel=1e-6)
    assert bayes_post_sd(pulls, payouts, False)[0] == pytest.approx(sd, rel=1e-6)


@pytest.mark.parametrize("n,total", [(0, 0.0), (1, 2.5), (5, -3.0), (30, 11.0)])
def test_the_gaussian_belief_is_the_normal_posterior(n, total):
    """N(0,1) prior with known sigma=1: only (n, sum of payouts) enter the
    likelihood, which is the sufficiency claim the `stats` observation rests
    on. Checked by quadrature over mu."""
    sigma = next(c["value"] for c in RAW["mdp"]["scenario"]["constants"]
                 if c["name"] == "sigma")
    grid = np.linspace(-12.0, 12.0, 400_001)
    mean, sd = _numeric_posterior(
        grid,
        lambda mu: -0.5 * mu ** 2
        - (n * mu ** 2 - 2.0 * mu * total) / (2.0 * sigma ** 2),
    )
    pulls = [n] + [0] * 9
    payouts = [total] + [0.0] * 9
    assert bayes_post_mean(pulls, payouts, True)[0] == pytest.approx(mean, rel=1e-6)
    assert bayes_post_sd(pulls, payouts, True)[0] == pytest.approx(sd, rel=1e-6)


def test_an_unpulled_arm_sits_at_the_prior():
    """Prior means 0.5 (Beta(1,1)) and 0.0 (N(0,1)) — the value the `bayes`
    observation must show for an arm nothing is known about."""
    zeros = [0] * 10
    assert bayes_post_mean(zeros, [0.0] * 10, False) == pytest.approx([0.5] * 10)
    assert bayes_post_sd(zeros, [0.0] * 10, False) == \
        pytest.approx([math.sqrt(1.0 / 12.0)] * 10)
    assert bayes_post_mean(zeros, [0.0] * 10, True) == pytest.approx([0.0] * 10)
    assert bayes_post_sd(zeros, [0.0] * 10, True) == pytest.approx([1.0] * 10)


def test_the_belief_features_are_declared_as_ir_builtins():
    """The gym and the interpreter must read the same conjugate code, or the
    differential would be comparing two different observations."""
    declared = {b["name"]: b["module"] for b in RAW["mdp"]["expr_builtins"]}
    assert declared == {"bayes_post_mean": "mab_bayes",
                        "bayes_post_sd": "mab_bayes"}


# -- negative controls: the gates must be able to fail -----------------------


def test_a_corrupted_ledger_update_diverges():
    """Drop the payout accrual from the U event: the domain still credits the
    arm, the interpreter no longer does, so the trajectories must part."""
    def drop_accrual(doc):
        for transition in doc["mdp"]["dynamics"]["transitions"]:
            if transition["event"] == "U":
                transition["updates"] = ["pulls[int(arm)] += 1"]

    assert_diverges(SCHEMA, mutated(load_ir(SCHEMA), drop_accrual), episodes=1)


def test_a_mis_stated_regret_claim_is_caught_while_the_sides_still_agree():
    """The reason invariants exist: flip the regret sign and the claim breaks
    even though the interpreter and the domain remain bit-identical, because a
    shared mis-formalization is invisible to the differential."""
    def flip(doc):
        for inv in doc["mdp"]["invariants"]:
            if inv["name"] == "regret_is_nonnegative":
                inv["expr"] = "opt_mean <= mean_pulled"

    traj = IrInterpreter(mutated(load_ir(SCHEMA), flip), seed_salt=1).run(0)
    assert traj.violations, "a wrong regret-sign claim went unnoticed"


def test_a_mis_stated_pull_count_claim_is_caught():
    """`sum(pulls) == t + 1` pins the one-pull-per-round clock; off-by-one it
    and every row must complain."""
    def off_by_one(doc):
        for inv in doc["mdp"]["invariants"]:
            if inv["name"] == "one_pull_per_round":
                inv["expr"] = "sum(pulls) == t"

    traj = IrInterpreter(mutated(load_ir(SCHEMA), off_by_one), seed_salt=1).run(0)
    assert traj.violations, "an off-by-one clock claim went unnoticed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# -- second simulators: the licence is a gate, not an argument ----------------
#
# The domain permits fast re-implementations of the MDP core (spec doctrine:
# the SCHEMA is the source of truth, not any one implementation), but each owes
# a runnable proof of the equivalence it claims. These are those proofs; they
# live here so they run on every change rather than from a manual `--part`.
# See README.md §Simulators for which simulator to use when.


def test_vec_sim_constants_still_match_the_base_instance():
    """DRIFT GUARD. `mab_a5_probe.VecSim` is a bit-exact replica of the base
    Gaussian cell, but it hard-codes K, T and the N(0,1)/sigma=1 belief. #E32
    made K and T axis constants, so an instance edit can silently invalidate
    it — with no failure anywhere, because the probe simply keeps simulating
    the old problem. This test is the tripwire that edit must trip."""
    import mab_a5_probe as a5

    consts = {c["name"]: c["value"] for c in RAW["mdp"]["scenario"]["constants"]}
    resolved = {**consts, **RAW["mdp"]["scenario"]["instances"]["gaussian"]}
    assert a5.K == resolved["n_arms"], (
        f"VecSim K={a5.K} but the gaussian instance has n_arms="
        f"{resolved['n_arms']} — the fast path is simulating a different problem")
    assert a5.T == resolved["horizon_T"], (
        f"VecSim T={a5.T} but the gaussian instance has horizon_T="
        f"{resolved['horizon_T']}")
    assert resolved["sigma"] == 1.0, (
        "VecSim hard-codes sigma=1 in realize_means/payout_draws")
    draw = (RAW["mdp"]["uncertainty_slots"][0]["candidates"]["gaussian"]
            ["settings"]["arm_means"]["draw"]["settings"])
    assert (draw["mean"], draw["std"]) == (0.0, 1.0), (
        "VecSim hard-codes the N(0,1) arm-mean prior")
    assert a5.SEED_SALT == RAW["mdp"].get("seed_salt", a5.SEED_SALT)


def test_vec_sim_is_bit_exact_against_the_gym():
    """The replica's claim is bit-exactness, so assert exactly that: identical
    payouts and observations against MabEnv over the same actions and seeds.
    Was a manual `--part verify`; a gate that runs when someone remembers it
    is not a gate."""
    pytest.importorskip("gymnasium")
    from mab_a5_probe import part_verify

    res = part_verify()
    assert res["bit_exact"] and res["payout_err"] == 0.0 and res["obs_err"] == 0.0


def test_grid_sim_posterior_matches_the_domain_belief():
    """`GridSim` claims distributional, not bit-exact, equivalence — it drops
    the domain's seed scheme for speed, so it cannot be diffed episode by
    episode. Its deterministic core still can be: the posterior it scores
    every policy from must equal `mab_bayes`, the IR's own `expr_builtin`."""
    from mab_grid_probe import gate_posterior_matches_domain

    res = gate_posterior_matches_domain()
    assert res["max_pm_err"] < 1e-12 and res["max_psd_err"] < 1e-12


def test_grid_sim_posterior_is_calibrated_off_the_base_cell():
    """Away from K=10/T=1000/sigma=1 the domain has no reference to diff
    against, so the general posterior is checked by calibration: mu drawn from
    the prior must fall within psd of pm at the right rate. Catches a sigma
    mis-scaling, which no formula-vs-formula check can see."""
    from mab_grid_probe import gate_posterior_calibration

    res = gate_posterior_calibration()
    assert res["max_dev"] < 0.03

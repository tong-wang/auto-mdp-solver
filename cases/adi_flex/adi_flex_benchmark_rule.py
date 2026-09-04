"""Heuristic baselines for the ADI-flex domain — role `feasible` (§9.9).

Every policy here runs under the real information set, so all of them sit at or
below the optimum and any of them is a legitimate `--baseline` (must-beat).
Against the `ap` arm they form the other half of the het bracket. Multi-policy
solver per §9.8: the arms below are selected with `--policy` and declared as
`benchmarks[rule].policies` in the IR.

Rule policies, evaluated with the spec §9 seed protocol through the gym:

All are two-phase policies since F7 (the env takes an order step, then an
allocation step; the PL heuristics are what the paper says they are —
POLICIES over the allocation decision, parameterized by sigma):

- random (DIAGNOSTIC, not a reported arm — see below): order ~
  Uniform{0..scenario.order_max}; allocation ~ Uniform over the FEASIBLE
  range {0..min(surplus, outstanding)} (mask-uniform — the natural random
  policy in the masked action space; pre-F7 this arm drew sigma instead).
- myopic: single-period newsvendor with fixed cost — at state (u, vhat) choose
  y minimizing K*1{y>u} + E[h*(y-Dtot)^+] + E_d0[p*(y+vhat-d0)^-], ignoring
  all future periods; allocation = maximal fill (sigma = 0).
- pl0 / plsigma / plmax: the paper's protection-level heuristics PL(0), PL(s),
  PL(S) (§4.2): ordering follows the AP-relaxation optimal policy (the y*
  table from adi_flex_benchmark_ap.py, --dp-solutions); the allocation is
  a = clip(surplus - sigma, 0, outstanding) with a constant sigma —
      pl0:     sigma = 0                     (maximal fill, FCFS)
      plsigma: sigma = argmin h*s + p*E[(d0-s)^+]   (newsvendor fractile)
      plmax:   sigma = min {s : P(d0 > s) < 0.001}  (full protection)

All TSVs share the eval format (reward = -cost, higher is better).

Example usage:
    python adi_flex_benchmark_rule.py -s het_exp4 --policy plsigma \
        --dp-solutions results/het_exp4/ap/het_exp4_policy.npz

**`random` is a diagnostic, never a leaderboard arm.** It draws uniformly
over the action space, so its value measures `order_max` rather than the
problem: it moved twice on 2026-08-18 (F11 widened the allocation, F14 derived
the order cap) for reasons that had nothing to do with policy quality. A
number that reports the size of the space it is drawn from is not a floor.
Kept runnable because an env that a uniform policy cannot step is broken, and
that is worth being able to check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adi_flex_benchmark_common import (U_MAX, U_MIN, ObsView, _truncated_poisson, EVAL_HEADER,
                                       eval_row, poisson_cdf, poisson_ppf, poisson_sf)
from adi_flex_gym import AdiFlexEnv
from adi_flex_scenarios import SCENARIOS, AdiFlexScenario

_PL_POLICIES = {"pl0", "plsigma", "plmax"}


def myopic_policy_table(scenario: AdiFlexScenario) -> tuple[np.ndarray, int, int]:
    """y*(u, vhat) minimizing the single-period cost; returns (table, u_min, d_max)."""
    # same reduction shape as solve_dp: dim(V-hat) = T - L - 1, so 1 at
    # L=0/T_dl=2 and 0 at L=0/T_dl<=1, where the single vhat level is pinned
    # at 0 (F26). Anything wider needs a wider table, not a wider loop.
    n_vhat = max(0, scenario.T_dl - scenario.L - 1)
    assert scenario.L == 0 and scenario.T_dl - scenario.L - 1 <= 1, (
        f"myopic table implements dim(V-hat) in {{0, 1}} at L=0; got "
        f"L={scenario.L}, T_dl={scenario.T_dl} -> {n_vhat} v-hat components."
    )
    # near/far as in solve_dp: at T_dl = 0 the near part is empty, so it must
    # start from a point mass and not from component 0
    rates = scenario.demand.rates
    pmf0  = _truncated_poisson(rates[0])
    pmf01 = np.array([1.0])
    for r in rates[0:scenario.T_dl]:
        pmf01 = np.convolve(pmf01, _truncated_poisson(r))
    pmf2  = _truncated_poisson(rates[scenario.T_dl])
    pmf_tot = np.convolve(pmf01, pmf2)
    d2_max, d0_max = len(pmf2) - 1, len(pmf0) - 1

    u_grid = np.arange(U_MIN, U_MAX + 1)
    n_v    = d2_max + 1 if n_vhat else 1
    K, h, p = scenario.K, scenario.h, scenario.p

    tvals  = np.arange(len(pmf_tot))
    d0vals = np.arange(d0_max + 1)
    eh  = h * (np.maximum(u_grid[:, None] - tvals[None, :], 0) @ pmf_tot)
    pen = p * (np.maximum(
        d0vals[None, None, :] - (u_grid[:, None, None] + np.arange(n_v)[None, :, None]), 0
    ) @ pmf0)
    G = eh[:, None] + pen                                     # (y, vhat)

    ystar = np.empty((len(u_grid), n_v), dtype=np.int32)
    for k, u in enumerate(u_grid):
        hi = min(u + scenario.order_max, U_MAX)
        stay = G[k]
        if hi > u:
            win   = G[k + 1: k + 1 + (hi - u)]
            jbest = win.argmin(axis=0)
            order = K + win[jbest, np.arange(n_v)]
            take  = order < stay - 1e-12
            ystar[k] = np.where(take, u + 1 + jbest, u)
        else:
            ystar[k] = u
    return ystar, U_MIN, n_v - 1     # the TABLE's last vhat index, not the draw support


def protection_level(scenario: AdiFlexScenario, policy: str) -> int:
    """The constant protection level of a PL heuristic (paper §4.2).

    ONE level, because at T_dl = 2 there is one crossover-eligible class. The
    paper states the generalization explicitly (§4.2, after eq. 20): "When
    T > 2, more than one protection level is needed ... when T = 3, we need
    protection stock sigma^{i+1}, covering future demand d_{i+1}^{i+1}, and
    protection stock sigma^{i+2}, covering d_{i+1}^{i+2}" — so a wider instance
    needs a LADDER of levels, one per eligible class, not this scalar reused.
    Refuse rather than protect every class by the same nearest-class number
    (F10).
    """
    if policy in {"pl0", "myopic"}:
        return 0
    assert scenario.T_dl == 2, (
        f"the PL heuristics here carry one protection level (T_dl = 2); "
        f"T_dl={scenario.T_dl} needs {scenario.n_alloc} of them (paper §4.2, "
        f"after eq. 20)."
    )
    if policy == "plsigma":
        # eq. (20)'s minimizer: smallest s with P(d0 <= s) >= 1 - h/p. NOT
        # p/(p+h) — that is the textbook ratio for h*E[(s-X)^+], and (20)
        # charges h*s outright. Differs only at lambda_0 = 4 (het_exp2).
        return protection_ladder(scenario, "plsigma")[0] if scenario.lambda0 > 0 else 0
    if policy == "plmax":
        # smallest s with P(d0 > s) < 0.001
        s = poisson_ppf(1.0 - 0.001, scenario.lambda0) if scenario.lambda0 > 0 else 0
        while poisson_sf(s, scenario.lambda0) >= 0.001:
            s += 1
        return s
    raise ValueError(f"no protection level for policy {policy!r}")


def _pl_fractile(scenario: AdiFlexScenario) -> float:
    """The critical ratio of paper eq. (20), which is NOT the textbook one.

        H(sigma) = h*sigma + p*E[(X - sigma)^+]                          (20)

    The holding term is `h*sigma`, DETERMINISTIC — §4.2 says so outright:
    "holding cost h*sigma_i is incurred for sure". A textbook newsvendor has
    `h*E[(sigma - X)^+]` and critical ratio `p/(p+h)`; this one differs:

        dH/dsigma = h - p*P(X > sigma) = 0   ->   P(X <= sigma) >= 1 - h/p

    At h=1, p=9 that is 0.8889 against the textbook 0.9000. They land on the
    same integer for every rate this domain declares EXCEPT lambda_0 = 4
    (het_exp2), where the textbook ratio gives 7 and eq. (20) gives 6.
    """
    return 1.0 - scenario.h / scenario.p


def protection_ladder(scenario: AdiFlexScenario, policy: str) -> tuple[int, ...]:
    """The PL heuristic's protection levels, paper §4.2 — `T_dl - 1` of them.

    §4.2 after eq. (20), verbatim: "when T = 3, we need protection stock
    sigma_i^{i+1}, covering future demand d_{i+1}^{i+1}, and protection stock,
    sigma_i^{i+2}, covering d_{i+1}^{i+2}. sigma_i^{i+1} can still be defined as
    the minimizer of (20), while sigma_i^{i+2} can be defined as the minimizer
    of h*sigma_i^{i+2} + p*E[(d_{i+1}^{i+1} + d_{i+1}^{i+2} - sigma_i^{i+1} -
    sigma_i^{i+2})^+]."

    CUMULATIVE ON BOTH SIDES, and over ONE period's arrivals:

        sum_{k<=t} sigma_k  =  quantile_{1-h/p} ( sum_{j<t} D[j] )

    so `sigma_t` is the increment of that. Two readings this is NOT, both of
    which were implemented here first and are wrong:

      * per-class — `sigma_t` from `D[t-1]` alone. Ignores that the levels
        share the protection and that the threshold accumulates.
      * `sum_{j<t} (t-j)*D[j]` — the envelope of everything that could EVER
        preempt `adv[t]`, over several future periods. That IS the right
        quantity for `AdiFlexScenario.sigma_max`, which must not bind; it is
        not this heuristic, whose `H(.)` is a deliberately myopic ONE-period
        tradeoff that conservatively takes the arriving replenishment
        `z_{i+1}` as zero (§4.2: "z_{i+1} is difficult to estimate or
        predict, so we conservatively take it as zero").
        Cap and ladder are different objects; agreement between them is a
        coincidence, never a check.

    PL(Sigma) takes the same cumulative form at a near-certain quantile —
    "protection level high enough to cover the whole support" — so crossover is
    avoided rather than balanced.

    At T_dl = 2 the ladder is one level and reduces to `protection_level`.
    """
    n = scenario.n_sigma
    if policy in {"pl0", "myopic"}:
        return (0,) * n
    if policy == "plsigma":
        q = _pl_fractile(scenario)
    elif policy == "plmax":
        q = 1.0 - 0.001
    else:
        raise ValueError(f"no protection ladder for policy {policy!r}")
    lam, out, prev = scenario.lambda_seg, [], 0
    for t in range(1, scenario.T_dl):
        rate = float(sum(lam[:t]))                 # D[0] + ... + D[t-1]
        if rate <= 0.0:
            out.append(0)
            continue
        cum = poisson_ppf(q, rate)
        while poisson_cdf(cum, rate) < q:          # guard against float slop
            cum += 1
        out.append(max(0, cum - prev))
        prev = cum
    return tuple(out)


def evaluate_policy(
    scenario: AdiFlexScenario, policy: str, n_seeds: int,
    ap_table: np.ndarray | None = None, ap_u_min: int = U_MIN, ap_d_max: int = 0,
) -> dict:
    """Play the rule policy for seeds 0..n_seeds-1; return reward stats."""
    env = AdiFlexEnv(scenario=scenario, observation_mode="vec")

    if policy == "myopic":
        table, u_min, d_max = myopic_policy_table(scenario)
    elif policy in _PL_POLICIES:
        assert ap_table is not None, f"policy {policy!r} needs --dp-solutions (AP y* table)"
        table, u_min, d_max = ap_table, ap_u_min, ap_d_max
    else:
        table, u_min, d_max = None, U_MIN, 0

    # a LADDER, `T_dl - 1` wide, not a scalar reused across classes (F10): the
    # level protecting `adv[k]` is its own `sigma_k`, and at T_dl = 2 the ladder
    # is one element and this is exactly what the scalar did.
    ladder = protection_ladder(scenario, policy) if policy != "random" else ()
    if policy != "random":
        print(f"  policy={policy}  sigma={ladder}")

    rewards = np.zeros(n_seeds)

    for ep_seed in range(n_seeds):
        if ep_seed % 10000 == 0:
            print(f"  seed {ep_seed}/{n_seeds}", flush=True)
        obs, _ = env.reset(seed=ep_seed)
        rng = np.random.default_rng(np.random.SeedSequence([9999, ep_seed]))
        terminated = False
        comp = 0            # which allocation component the env is asking for
        while not terminated:
            v = ObsView(obs, env.pipe_slots, scenario.N, scenario.T_dl,
                        alloc_enabled=env.alloc_enabled)
            if v.phase == 0:
                comp = 0
                # ORDER phase; vhat = d[2] here is last period's far draw (0 at t=0)
                if policy == "random":
                    order = int(rng.integers(0, scenario.order_max + 1))
                else:
                    n_u = table.shape[1] if table.ndim == 3 else table.shape[0]
                    ui = int(np.clip(v.u - u_min, 0, n_u - 1))
                    vi = int(np.clip(v.vhat, 0, d_max))
                    if table.ndim == 3:      # nonstationary AP table: (period, u, vhat)
                        y = table[v.period, ui, vi]
                    else:                    # stationary myopic table: (u, vhat)
                        y = table[ui, vi]
                    order = int(np.clip(y - v.u, 0, scenario.order_max))
                obs, _, terminated, _, _ = env.step(order)
            else:
                # ALLOCATION phase (heterogeneous branch only). Components run
                # over due-offsets 1..T_dl. Component 0 is the DUE-NEXT class:
                # §4.2's heuristic serves it in full before protecting anything
                # ("neither of these will be crossed over by future demands"),
                # and since F10 that is the policy's job — the simulator no
                # longer forces it. The protection level applies from the first
                # crossover-eligible class onward.
                feasible = min(v.surplus, v.outstanding)
                if policy == "random":
                    a = int(rng.integers(0, feasible + 1))
                elif comp == 0:
                    # §4.2 serves the due-next class in full before protecting
                    # anything; since F10 that is the policy's job, not forced
                    a = feasible
                else:
                    # CUMULATIVE, sum(ladder[:comp]) — not this component's own
                    # level. `v.surplus` is the remainder after previous
                    # ALLOCATIONS only; it does not carry the earlier RESERVES,
                    # which stay held back. §4.2's form is cumulative on both
                    # sides (sigma_1 + sigma_2 protects D[0] + D[1]), and the
                    # cascade in adi_flex_gym reproduces that because its `rem`
                    # loses each reserve as it goes.
                    #
                    # At T_dl = 2 there is one level and the two readings
                    # coincide, which is why every T=2 number here is unaffected
                    # and why the per-component version survived a 200k-state
                    # equivalence check before failing at T=3 by 14 cost units.
                    a = int(np.clip(v.surplus - sum(ladder[:comp]), 0, v.outstanding))
                comp += 1
                obs, _, terminated, _, _ = env.step(a)
        rewards[ep_seed] = env.total_reward

    env.close()
    mu = rewards.mean()
    return {
        "reward_mean": float(mu),
        "reward_var":  float(rewards.var(ddof=1)),
        "semivar_d":   float(np.sum(np.maximum(mu - rewards, 0.0) ** 2) / (n_seeds - 1)),
        "semivar_u":   float(np.sum(np.maximum(rewards - mu, 0.0) ** 2) / (n_seeds - 1)),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rule-policy baselines for the ADI-flex domain.")
    p.add_argument("-s", "--scenario_name", type=str, default="homog_L0_T2",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--policy", type=str, default="myopic",
                   choices=["random", "myopic", "pl0", "plsigma", "plmax"])
    p.add_argument("--dp-solutions", type=str, default=None,
                   help="AP y* table (.npz from adi_flex_benchmark_ap.py); required for pl0/plsigma/plmax")
    p.add_argument("--n-seeds", type=int, default=8192,
                   help="Number of episode seeds (seeds 0..n-1)")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV output (defaults to results/<scenario>/benchmark/"
                        "benchmark_rule_eval_<scenario>_<policy>.tsv)")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    ap_table, ap_u_min, ap_d_max = None, U_MIN, 0
    if args.policy in _PL_POLICIES:
        assert args.dp_solutions, f"--dp-solutions is required for --policy {args.policy}"
        sol = np.load(args.dp_solutions)
        ap_table, ap_u_min, ap_d_max = sol["ystar"], int(sol["u_min"]), int(sol["d_max"])
        assert ap_table.shape[0] == scenario.N, \
            f"AP policy horizon {ap_table.shape[0]} != scenario N {scenario.N}"

    outfile = (
        Path(args.outfile) if args.outfile
        # `benchmark_{method}_eval` marker: what lets `mdp_gates --ir` resolve
        # this arm's declared role=feasible (so it is a legal must-beat baseline)
        else Path(__file__).resolve().parent / "results" / args.scenario_name / "benchmark"
        / f"benchmark_rule_eval_{args.scenario_name}_{args.policy}.tsv"
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)

    header = EVAL_HEADER
    print(header)
    with open(outfile, "w") as f:
        f.write(f"# policy={args.policy}\n")
        f.write(header + "\n")
        f.flush()
        stats_out = evaluate_policy(
            scenario, args.policy, args.n_seeds,
            ap_table=ap_table, ap_u_min=ap_u_min, ap_d_max=ap_d_max,
        )
        row = (
            eval_row(scenario, stats_out)
        )
        print(row, flush=True)
        f.write(row + "\n")

    print(f"\nresults saved -> {outfile}")


if __name__ == "__main__":
    main()

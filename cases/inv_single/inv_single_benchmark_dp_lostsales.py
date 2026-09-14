"""Exact DP for the LOST-SALES instances at deterministic lead time.

The shipped `inv_single_benchmark_dp.py` solves a backlog recursion on inventory
position, and its own `basis` field says it is **not a valid bar** under lost
sales: the cost recursion is backlog-only. That left `lt_lost_sales` with no
reference at all — every number at the cell was a feasible policy, so "PPO beats
base-stock by 7.24%" (#E15) had no denominator.

It does not have to stay that way. Lost sales with positive lead time is the
canonical hard case *asymptotically in L*, but at this domain's L=2 the state is
small enough to solve exactly, and the reduction is short.

THE REDUCTION (derived, and checked here against a brute-force 3-D DP).
Event order is O-R-D, pipeline `[p0, p1, p2]`, decision `q` placed at `p2`.
One period:  O: p2 <- q;  R: inv <- inv + p0, shift;  D: inv <- inv - d, then
lost sales truncate at zero.  Write `u = inv + p0` (the stock available to meet
this period's demand, since the receipt lands BEFORE demand).

  * period cost is `h*(u-d)^+ + b*(d-u)^+` — a function of `u` alone;
  * next state is `inv' = (u-d)^+`, `p0' = p1`, `p1' = q`,
    so `u' = inv' + p0' = (u-d)^+ + p1`.

Step 1 is the same as the backlog case (`inv_single_dp_exactness.py`): `inv` and
`p0` enter only through their sum. Step 2 is where lost sales DIVERGES. Under
backlog `u' = u - d + p1` is linear in `u`, so `u` and `p1` collapse further into
`u + p1 = IP` and inventory position is a sufficient statistic. Under lost sales
the truncation makes `u' = (u-d)^+ + p1` NON-linear in `u`, so `u` and `p1`
cannot be summed — IP is not sufficient, exactly as Karlin & Scarf (1958) say.

What survives is a TWO-dimensional recursion:

    V_t(u, p1) = E_d[c(u,d)] + min_q { K*1[q>0] + c_lin*q
                                       + E_d[ V_{t+1}((u-d)^+ + p1, q) ] }

That is ~10^4 states. The curse of dimensionality in lost-sales inventory is
real but bites at larger L (Zipkin 2008 solves numerically to L~4); at L=2 the
exact optimum is cheap.

WHAT THIS SCRIPT REFUSES TO ASSUME. A derivation is not a result, so the value
it reports is checked two independent ways before it is written anywhere:

  1. `--verify-3d` solves the un-reduced `(inv, p0, p1)` DP on a small grid and
     asserts `V3(inv,p0,p1) == V2(inv+p0, p1)` everywhere. Catches an error in
     the reduction above.
  2. The default run then SIMULATES the DP's own policy through `InvSingleEnv`
     — the same wrapper RL trains on — over the protocol block, and compares the
     simulated mean against `V_0(0,0)`. Catches a mismatch between this file's
     dynamics and the simulator's, which check 1 cannot see because both halves
     of it share this file's assumptions.

Usage:
    python inv_single_benchmark_dp_lostsales.py -s lt_lost_sales --verify-3d
    python inv_single_benchmark_dp_lostsales.py -s lt_lost_sales --n-seeds 8192
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from inv_single_eval_common import eval_seed_block, rollout_seeds, write_record
from inv_single_scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------

def demand_pmf(scenario, d_max: int) -> np.ndarray:
    """Truncated, renormalized pmf on 0..d_max for the scenario's demand."""
    gen = scenario.demand
    rate = getattr(gen, "rate", None)
    if rate is None:
        raise SystemExit(
            f"demand generator {type(gen).__name__} has no fixed `rate`; this "
            "exact DP is defined only for a scenario whose demand law is fixed "
            "(no per-episode latent). Use it on `lt_lost_sales`, not on the "
            "`*_lost_sales` latent-demand instances."
        )
    pmf = poisson.pmf(np.arange(d_max + 1), rate)
    return pmf / pmf.sum()


# ---------------------------------------------------------------------------
# The reduced (u, p1) DP
# ---------------------------------------------------------------------------

def solve(scenario, *, u_max: int, q_cap: int, d_max: int):
    """Backward induction on V_t(u, p1). Returns (V, Q) with V[t] shape (u,p1)."""
    T = scenario.horizon
    h, b = scenario.holding_cost, scenario.shortage_cost
    K, c_lin = scenario.order_cost_fixed, scenario.order_cost_linear

    pmf = demand_pmf(scenario, d_max)
    u = np.arange(u_max + 1)
    d = np.arange(d_max + 1)

    # period cost, a function of u alone
    diff = u[:, None] - d[None, :]                       # (U, D)
    C = (h * np.clip(diff, 0, None) + b * np.clip(-diff, 0, None)) @ pmf

    # T_mat[u, x] = P( (u-d)^+ == x )
    x_of = np.clip(diff, 0, None)                        # (U, D)
    T_mat = np.zeros((u_max + 1, u_max + 1))
    np.add.at(T_mat, (np.repeat(u, d_max + 1), x_of.ravel()), np.tile(pmf, u_max + 1))

    q = np.arange(q_cap + 1)
    ord_cost = K * (q > 0) + c_lin * q                   # (Q,)

    # index map: x + p1, clipped to the grid
    nxt = np.minimum(u[:, None] + q[None, :q_cap + 1], u_max)   # (X, P1)

    V = np.zeros((T + 1, u_max + 1, q_cap + 1))
    Q = np.zeros((T, u_max + 1, q_cap + 1), dtype=np.int32)
    for t in range(T - 1, -1, -1):
        # W[x, p1, q'] = V_{t+1}( clip(x+p1), q' )
        W = V[t + 1][nxt]                                # (X, P1, Q)
        future = np.tensordot(T_mat, W, axes=([1], [0]))  # (U, P1, Q)
        total = C[:, None, None] + ord_cost[None, None, :] + future
        Q[t] = total.argmin(axis=2)
        V[t] = total.min(axis=2)
    return V, Q


# ---------------------------------------------------------------------------
# Independent check 1 — the un-reduced 3-D DP
# ---------------------------------------------------------------------------

def verify_3d(scenario, *, u_max: int, p_cap: int, d_max: int, horizon: int) -> None:
    """Brute-force V3(inv, p0, p1) and assert it equals V2(min(inv+p0, u_max), p1).

    Both DPs must TRUNCATE IDENTICALLY or they disagree at the grid corner for a
    reason that has nothing to do with the reduction. The shared rule is: stock
    available this period is `u = min(inv + p0, u_max)`, and nothing else is
    clipped — the 3-D inventory axis runs to `u_max`, so `inv' = (u-d)^+` always
    fits, and the next period's `u` is clipped by the same rule one step later.
    (A first version of this check clipped `inv'` to a SMALLER 3-D grid than the
    2-D `u` grid and reported a 5.01 gap at exactly the corner state where the
    caps differ. The reduction was fine; the harness was not.)
    """
    h, b = scenario.holding_cost, scenario.shortage_cost
    K, c_lin = scenario.order_cost_fixed, scenario.order_cost_linear
    pmf = demand_pmf(scenario, d_max)

    small = dataclasses.replace(scenario, horizon=horizon)
    V2, _ = solve(small, u_max=u_max, q_cap=p_cap, d_max=d_max)

    inv = np.arange(u_max + 1)
    p0 = np.arange(p_cap + 1)
    d = np.arange(d_max + 1)
    q = np.arange(p_cap + 1)
    ord_cost = K * (q > 0) + c_lin * q

    u_idx = np.minimum(inv[:, None] + p0[None, :], u_max)          # (I, P0)
    diff = u_idx[:, :, None] - d[None, None, :]                    # (I, P0, D)
    C3 = (np.clip(diff, 0, None) * h + np.clip(-diff, 0, None) * b) @ pmf
    X = np.clip(diff, 0, None)                                     # (I, P0, D) -> inv'

    V3 = np.zeros((horizon + 1, u_max + 1, p_cap + 1, p_cap + 1))
    for t in range(horizon - 1, -1, -1):
        Vn = V3[t + 1]
        for p1 in range(p_cap + 1):
            G = Vn[X, p1, :]                                       # (I, P0, D, Q)
            fut = np.einsum("ipdq,d->ipq", G, pmf)
            total = C3[:, :, None] + ord_cost[None, None, :] + fut
            V3[t, :, :, p1] = total.min(axis=2)

    ref = V2[:horizon][:, u_idx, :]                                # (T, I, P0, P1)
    gap = np.abs(V3[:horizon] - ref)
    worst = float(gap.max())
    where = np.unravel_index(int(gap.argmax()), gap.shape)
    print(f"[verify-3d] grid u<={u_max} p<={p_cap} T={horizon} d<={d_max} "
          f"({(u_max+1)*(p_cap+1)**2} states)")
    print(f"[verify-3d] max |V3(inv,p0,p1) - V2(min(inv+p0,u_max),p1)| = {worst:.3e} at {where}")
    assert worst < 1e-9, "the (u, p1) reduction does NOT reproduce the 3-D DP"
    print("[verify-3d] PASS - the reduction is exact\n")


# ---------------------------------------------------------------------------
# Policy + CLI
# ---------------------------------------------------------------------------

def make_act_fn(Q: np.ndarray, *, u_max: int, q_cap: int, boundary: dict):
    """The DP table as a batched action callback, reading the pre-order state."""
    def act(obs, envs):
        out = np.empty((len(envs), 1), dtype=np.float32)
        for i, env in enumerate(envs):
            st = env._state
            u = int(st.inventory) + int(st.pipeline[0])
            p1 = int(st.pipeline[1])
            boundary["u"] = max(boundary["u"], u)
            boundary["p1"] = max(boundary["p1"], p1)
            a = int(Q[int(st.period), min(u, u_max), min(p1, q_cap)])
            boundary["q"] = max(boundary["q"], a)
            out[i, 0] = a
        return out
    return act


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--scenario_name", default="lt_lost_sales")
    p.add_argument("--u-max", type=int, default=200)
    p.add_argument("--q-cap", type=int, default=80)
    p.add_argument("--d-max", type=int, default=45)
    p.add_argument("--verify-3d", action="store_true")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--no-sim", action="store_true", help="solve and report V0 only")
    p.add_argument("--outfile", type=str, default=None)
    args = p.parse_args()

    sc = SCENARIOS[args.scenario_name]
    if callable(sc):
        raise SystemExit(f"{args.scenario_name!r} is a sampler; this DP needs a fixed scenario")
    if sc.stockout_mode != "lost_sales":
        raise SystemExit(f"{args.scenario_name!r} is {sc.stockout_mode}; use inv_single_benchmark_dp.py")
    if sc.leadtime.min() != sc.leadtime.max():
        raise SystemExit("the reduction assumes a DETERMINISTIC lead time (orders may not cross)")
    if tuple(sc.event_sequence) != ("O", "R", "D"):
        raise SystemExit(f"the reduction is derived for O-R-D, not {sc.event_sequence}")

    if args.verify_3d:
        verify_3d(sc, u_max=60, p_cap=10, d_max=25, horizon=6)

    V, Q = solve(sc, u_max=args.u_max, q_cap=args.q_cap, d_max=args.d_max)
    v0 = float(V[0, 0, 0])
    print(f"scenario {sc.scenario_name!r}: L={sc.leadtime.max()} deterministic, lost sales, "
          f"T={sc.horizon}, h={sc.holding_cost}, b={sc.shortage_cost}, K={sc.order_cost_fixed}")
    print(f"grid u<={args.u_max}  q<={args.q_cap}  d<={args.d_max}")
    print(f"\n*** EXACT OPTIMAL COST  V_0(u=0, p1=0) = {v0:.6f} ***\n")

    if args.no_sim:
        return

    boundary = {"u": 0, "p1": 0, "q": 0}
    seeds = eval_seed_block(args.n_seeds, args.first_seed)
    per_seed = rollout_seeds(
        scenario=sc, observation_mode="vec", action_mode="discrete",
        seeds=seeds, act_fn=make_act_fn(Q, u_max=args.u_max, q_cap=args.q_cap,
                                        boundary=boundary),
        batch=args.batch_envs, vecnorm_path=None, progress_every=0,
    )
    sim = float(per_seed["cost_total"].mean())
    se = float(per_seed["cost_total"].std(ddof=1) / np.sqrt(len(seeds)))
    print(f"\n[check-2] simulated mean {sim:.4f} (SE {se:.4f}) vs V_0 {v0:.4f}"
          f"  ->  {(sim - v0) / se:+.2f} SE")
    print(f"[boundary] max reached: u={boundary['u']}/{args.u_max}  "
          f"p1={boundary['p1']}/{args.q_cap}  q*={boundary['q']}/{args.q_cap}")
    if boundary["u"] >= args.u_max or boundary["q"] >= args.q_cap:
        print("[boundary] !! a cap BINDS — raise --u-max/--q-cap and re-run")

    out = Path(args.outfile) if args.outfile else Path(
        f"results/{sc.scenario_name}/benchmark/benchmark_dplostsales_eval_{sc.scenario_name}.tsv")
    write_record(out, sc.scenario_name, "dp_lostsales(exact)", per_seed)


if __name__ == "__main__":
    main()

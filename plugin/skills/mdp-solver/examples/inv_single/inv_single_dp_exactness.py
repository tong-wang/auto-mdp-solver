"""Settle whether the shipped DP-on-IP benchmark is exact at deterministic L > 1.

Campaign item A7a. `inv_single_benchmark_dp.py`'s docstring and the README both
call the inventory-position DP an "approximation" for LT > 1. Classical theory
says otherwise for backlog + linear costs + K = 0, and S3's bar is only
quotable as *optimal* if the claim is settled — so this script computes an
independent exact DP and compares.

The reduction it exploits (derived, not assumed). With event order O-R-D and
deterministic L, write the pre-order state as `(inv, p0, p1)` where `pk` is the
quantity arriving at the k-th future R event. One period does
O: `p2 <- q`; R: `inv <- inv + p0`, shift; D: `inv <- inv - d`, and the period
cost depends only on the post-demand inventory. So with `u = inv + p0`:

    V_t(inv, p0, p1) = G(u) + min_q E_d[ V_{t+1}(u - d, p1, q) ]

- the cost term `G(u) = E_d[h(u-d)^+ + b(u-d)^-]` does not involve `q`;
- the next state's first two slots do not involve `q` either.

Hence `V_t` depends on `(inv, p0)` only through their **sum**, i.e. the exact
state is two-dimensional: `Vt(u, p1)`. Substituting once more,

    Vt(u, p1) = G(u) + M_{t+1}(u + p1),    M_{t+1}(z) = min_q E_d[ V_{t+1}(z - d, q) ]

and the minimizer of the `t`-period problem depends on the state **only through
`u + p1 = inv + p0 + p1 = IP`**. So inventory position is a sufficient
statistic for the *decision* at any deterministic L — the docstring's L > 1
caveat has no basis in the structure. What remains genuinely open is whether
the shipped *implementation* realizes that optimum, which is what this script
measures: it compares the exact minimizer `q*(t, IP)` against `dp.act(t, IP)`
action by action.

Usage:
    python inv_single_dp_exactness.py -s lt
    python inv_single_dp_exactness.py -s lt --u-min -80 --u-max 90
"""
from __future__ import annotations

import argparse

import numpy as np

from inv_single_benchmark_dp import FiniteHorizonDP
from inv_single_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exactness check for the DP-on-IP benchmark.")
    p.add_argument("-s", "--scenario_name", type=str, default="lt",
                   choices=list(SCENARIOS.keys()))
    # grids must be wide enough that order-up-to is reachable from the most
    # negative IP compared; a saturated cell is a grid artifact, not a defect,
    # so saturated cells are excluded from the agreement statistic below
    p.add_argument("--u-min", type=int, default=-120)
    p.add_argument("--u-max", type=int, default=140)
    p.add_argument("--p-max", type=int, default=140)
    p.add_argument("--q-max", type=int, default=140)
    p.add_argument("--d-max", type=int, default=32)
    p.add_argument("--report-periods", type=int, default=6)
    return p


def compare_exact_vs_shipped(
    scenario_name: str = "lt",
    *,
    u_min: int = -120,
    u_max: int = 140,
    p_max: int = 140,
    q_max: int = 140,
    d_max: int = 32,
    no_cache: bool = False,
    results_dir: str = "results",
) -> dict:
    """Compare the shipped DP-on-IP table against the independently derived
    exact DP, action by action.

    Pure and printing-free so the claim can be GATED rather than eyeballed
    (spec 1.2: gate every second implementation; guide rule 9). `main` renders
    what this returns, and `inv_single_test.py` asserts on it. The claim class
    is bit-exact, so the gate is exact equality: `n_bad == 0` and `max_dq == 0`.

    Returns the verdict plus the geometry needed to render it.
    """
    sc = SCENARIOS[scenario_name]
    assert not callable(sc), "needs a fixed scenario"
    L = sc.leadtime.max()
    assert sc.leadtime.min() == L, (
        f"{scenario_name!r} has a stochastic lead time; this reduction is "
        "for deterministic L (orders cannot cross)"
    )
    assert sc.stockout_mode == "backlog", "the reduction assumes backlog"
    assert sc.event_sequence == ("O", "R", "D"), "derived for the O-R-D order"
    assert L <= 2, (
        f"the (u, p1) collapse as coded covers L <= 2; L={L} needs one extra "
        "pipeline slot in the state"
    )

    T = sc.horizon
    h, b, K, c = (sc.holding_cost, sc.shortage_cost,
                  sc.order_cost_fixed, sc.order_cost_linear)
    assert K == 0.0, "a fixed cost breaks the order-up-to reduction (that is S2)"

    u_grid = np.arange(u_min, u_max + 1)
    n_u = len(u_grid)
    n_p = p_max + 1
    q_vals = np.arange(0, q_max + 1)
    d_vals = np.arange(0, d_max + 1)
    phi = np.array([sc.demand.phi(int(d)) for d in d_vals], dtype=float)
    phi /= phi.sum()

    def uidx(x: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x).astype(int) - u_min, 0, n_u - 1)

    # G(u) = E_d[ h (u-d)^+ + b (u-d)^- ]
    diff = u_grid[:, None] - d_vals[None, :]
    G = (h * np.maximum(diff, 0.0) + b * np.maximum(-diff, 0.0)) @ phi

    # exact backward induction on the 2-dim state (u, p1)
    z_vals = np.arange(u_min, u_max + n_p)
    z_shift = u_min
    V = np.zeros((n_u, n_p))                            # V_T = 0
    pol = np.zeros((T, len(z_vals)), dtype=np.int32)    # q*(t, z), z = u + p1 = IP
    ord_cost = c * q_vals + K * (q_vals > 0)
    nxt = uidx(z_vals[:, None] - d_vals[None, :])       # (n_z, n_d)

    for t in range(T - 1, -1, -1):
        # V[nxt] is (n_z, n_d, n_p); contracting d gives A[z, q] = E_d[V(z-d, q)]
        # because the next period's p1 slot IS this period's order q
        A = np.tensordot(V[nxt], phi, axes=([1], [0]))  # (n_z, n_p)
        A = A[:, : len(q_vals)] + ord_cost[None, :]
        best = np.argmin(A, axis=1)
        M = A[np.arange(len(z_vals)), best]
        pol[t] = q_vals[best]
        # V_t(u, p1) = G(u) + M(u + p1)
        z_of = (u_grid[:, None] + np.arange(n_p)[None, :]) - z_shift
        z_of = np.clip(z_of, 0, len(z_vals) - 1)
        V = G[:, None] + M[z_of]

    # `no_cache=True` is what the gate uses: a test that reads a cached table
    # gates the cache, not the implementation it is supposed to check.
    dp = FiniteHorizonDP(sc, results_dir=results_dir, no_cache=no_cache)

    # compare on the IP range both tables cover
    ip_lo = max(u_min, dp.x_min)
    ip_hi = min(u_max + p_max, dp.x_max)
    ip_grid = np.arange(ip_lo, ip_hi + 1)

    n_bad = 0
    max_dq_all = 0
    n_sat = 0
    rows = []
    for t in range(T):
        q_exact = pol[t][np.clip(ip_grid - z_shift, 0, len(z_vals) - 1)]
        q_ship = np.array([dp.act(t, int(x)) for x in ip_grid])
        # a cell where either table is pinned at its own action ceiling says
        # nothing about exactness — it is a grid edge
        free = (q_exact < q_vals[-1]) & (q_ship < dp.q_vals[-1])
        n_sat += int((~free).sum())
        dq = np.abs(q_exact - q_ship)[free]
        dq = dq if dq.size else np.zeros(1, dtype=int)
        agree = float(np.mean((q_exact == q_ship)[free])) if free.any() else float("nan")

        # order-up-to level: modal (IP + q) where an order is placed
        def level(qs):
            m = qs > 0
            if not m.any():
                return float("nan")
            y = ip_grid[m] + qs[m]
            v, ct = np.unique(y, return_counts=True)
            return float(v[int(np.argmax(ct))])

        rows.append((t, level(q_exact), level(q_ship), int(dq.max()), agree))
        max_dq_all = max(max_dq_all, int(dq.max()))
        if dq.max() > 0:
            n_bad += 1

    return {
        "scenario_name": scenario_name,
        "L": L, "K": K, "T": T,
        "n_u": n_u, "n_p": n_p, "dp_n_x": dp.n_x,
        "ip_grid": ip_grid,
        "rows": rows,
        "n_bad": n_bad,
        "max_dq": max_dq_all,
        "n_sat": n_sat,
        "exact": n_bad == 0 and max_dq_all == 0,
    }


def main() -> None:
    args = _build_arg_parser().parse_args()
    r = compare_exact_vs_shipped(
        args.scenario_name,
        u_min=args.u_min, u_max=args.u_max, p_max=args.p_max,
        q_max=args.q_max, d_max=args.d_max,
    )

    print(f"\nscenario {r['scenario_name']!r}: L={r['L']} (deterministic), backlog, "
          f"K={r['K']}, T={r['T']}")
    print(f"exact DP state (u, p1) = ({r['n_u']} x {r['n_p']}); shipped DP state = IP "
          f"({r['dp_n_x']} points)")
    print("\nDerivation: q* depends on the state only through IP = inv+p0+p1, so "
          "IP is\nsufficient for the DECISION at any deterministic L. Comparing "
          "the shipped\ntable's action against the exact minimizer:\n")

    print(f"{'t':>3} {'exact S*':>9} {'ship S*':>8} {'max |dq|':>9} {'agree':>7}")
    T = r["T"]
    show = list(range(min(args.report_periods, T))) + list(
        range(max(0, T - args.report_periods), T)
    )
    for t, se, ss, mdq, ag in r["rows"]:
        if t in show:
            print(f"{t:3d} {se:9.1f} {ss:8.1f} {mdq:9d} {ag:7.3f}")

    print(f"\nperiods where the shipped table differs from exact: {r['n_bad']}/{T}"
          f"   worst |dq| = {r['max_dq']}")
    print(f"IP cells excluded as action-ceiling-saturated: {r['n_sat']} of "
          f"{T * len(r['ip_grid'])}")
    if r["exact"]:
        print("VERDICT: the shipped DP-on-IP is EXACT here — the docstring's "
              "'approximation for LT>1' claim is wrong and should be fixed.")
    else:
        print("VERDICT: the shipped DP-on-IP is NOT exact here; the periods "
              "above localize where it departs.")


if __name__ == "__main__":
    main()

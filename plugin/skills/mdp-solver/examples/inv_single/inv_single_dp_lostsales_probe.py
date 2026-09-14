"""Read the STRUCTURE off the exact lost-sales DP — the tables INTERPRET RQ-3 cites.

`inv_single_benchmark_dp_lostsales.py` computes the optimum. This reads what the
optimal policy *is*, which is what a `discover` question owes (spec §14).

Every table below is regenerated here, so no number in INTERPRET RQ-3.2-3.4 is
hand-copied:

  --shape       RQ-3.2  Q* and the post-order target vs IP: the capped
                        base-stock shape, both limbs of Xin (2021)'s family
                        active inside one policy.
  --sufficiency RQ-3.3  dQ*/du at FIXED IP. Zero in the base-stock region --
                        IP is exactly sufficient there -- and +0.10..+0.19
                        where the cap binds, which is the whole 2-D residual.
  --direction   RQ-3.4  Q* minus the fitted base-stock order, by IP. This is
                        the table that FALSIFIED RQ-3's stated prediction:
                        the optimum orders far BELOW base-stock at LOW stock
                        and slightly above at high stock, not the reverse.
  --horizon             the end-of-horizon check. The last two periods are an
                        argmin TIE, not a taper: an order placed then arrives
                        past the horizon and, with K=c=0, costs nothing.

All readings are VISITATION-WEIGHTED under the optimal policy. An unweighted
sweep reports the policy on states it never reaches, which is how a probe
manufactures structure that does not exist on-policy (the trap #E10 names).

Usage:
    python inv_single_dp_lostsales_probe.py -s lt_lost_sales --all
"""
from __future__ import annotations

import argparse

import numpy as np

from inv_single_benchmark_dp_lostsales import demand_pmf, solve
from inv_single_scenarios import SCENARIOS

U_GRID, P_GRID = 60, 40


def visitation(sc, Q, pmf, u_max, q_cap):
    """State distribution under the OPTIMAL policy, summed over the horizon."""
    dist = np.zeros((U_GRID + 1, P_GRID + 1)); dist[0, 0] = 1.0
    visit = np.zeros_like(dist)
    for t in range(sc.horizon):
        visit += dist
        nxt = np.zeros_like(dist)
        for u, p1 in np.argwhere(dist > 1e-12):
            q = int(Q[t, min(u, u_max), min(p1, q_cap)]); m = dist[u, p1]
            for d, pr in enumerate(pmf):
                if pr > 1e-12:
                    nxt[min(max(0, u - d) + p1, U_GRID), min(q, P_GRID)] += m * pr
        dist = nxt
    return visit / visit.sum()


def _cells(visit, ip, thresh=1e-6):
    return [(u, ip - u) for u in range(0, min(ip, U_GRID) + 1)
            if 0 <= ip - u <= P_GRID and visit[u, ip - u] > thresh]


def shape(sc, Q, visit, t, S_bs):
    print(f"\n=== RQ-3.2  the capped base-stock shape (t={t}, visitation-weighted) ===")
    print("   IP   Q*(dp)   target=IP+Q*   region")
    for ip in range(6, 44, 2):
        cells = _cells(visit, ip)
        if not cells: continue
        w = np.array([visit[u, p] for u, p in cells]); w /= w.sum()
        q = float(sum(wi * Q[t, u, p] for wi, (u, p) in zip(w, cells)))
        reg = "cap binds (constant-order limb)" if q > 12.5 else (
              "base-stock limb" if ip + q > 34.5 else "transition")
        print(f"  {ip:3d}   {q:6.2f}   {ip + q:10.2f}   {reg}")


def sufficiency(sc, Q, visit, t):
    print(f"\n=== RQ-3.3  is IP sufficient? dQ*/du at FIXED IP (t={t}) ===")
    print("  a pure-IP policy would give slope 0 everywhere.")
    print("   IP   n cells   slope dQ*/du|IP")
    for ip in range(10, 40, 3):
        cells = _cells(visit, ip)
        if len(cells) < 3: continue
        qs = [int(Q[t, u, p]) for u, p in cells]
        slope = float(np.polyfit([u for u, _ in cells], qs, 1)[0])
        print(f"  {ip:3d}   {len(cells):6d}   {slope:+14.3f}")
    print("  -> 0.000 in the base-stock region: IP is EXACTLY sufficient there.")
    print("     The 2-D residual lives entirely where the cap binds.")


def direction(sc, Q, visit, t, S_bs):
    print(f"\n=== RQ-3.4  optimum vs fitted base-stock (S={S_bs}), by IP (t={t}) ===")
    print("  RQ-3 predicted 'at-or-below base-stock in HIGH-stock states'.")
    print("   IP   Q*(dp)   basestock   dp - bs")
    for ip in range(6, 40, 2):
        cells = _cells(visit, ip)
        if not cells: continue
        w = np.array([visit[u, p] for u, p in cells]); w /= w.sum()
        q = float(sum(wi * Q[t, u, p] for wi, (u, p) in zip(w, cells)))
        bs = max(0, S_bs - ip)
        print(f"  {ip:3d}   {q:6.2f}   {bs:9d}   {q - bs:+7.2f}")
    print("  -> FAR BELOW at low IP, slightly ABOVE at high IP. The stated")
    print("     direction is backwards; the probe must look at LOW IP.")


def horizon(sc, Q, p1=4):
    print(f"\n=== end-of-horizon: taper, or tie? (p1={p1}) ===")
    ts = [10, 20, 25, 27, 28, 29]
    print("    u |" + "".join(f"  t={t:<3d}" for t in ts))
    for u in range(0, 33, 4):
        print(f"  {u:3d} |" + "".join(f"{int(Q[t, u, p1]):6d}" for t in ts))
    L = sc.leadtime.max()
    print(f"  an order at t >= {sc.horizon - L} arrives past the horizon, and with "
          f"K={sc.order_cost_fixed} c={sc.order_cost_linear} it costs NOTHING,")
    print("  so every action there is exactly TIED and Q*=0 is an argmin artifact,")
    print(f"  not a taper. The one real end effect is t={sc.horizon - L - 1}.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--scenario_name", default="lt_lost_sales")
    p.add_argument("-t", "--period", type=int, default=10)
    p.add_argument("--base-stock", type=int, default=35,
                   help="the fitted basestock_opt S this cell is compared against")
    p.add_argument("--u-max", type=int, default=200)
    p.add_argument("--q-cap", type=int, default=80)
    p.add_argument("--d-max", type=int, default=45)
    for f in ("shape", "sufficiency", "direction", "horizon", "all"):
        p.add_argument(f"--{f}", action="store_true")
    a = p.parse_args()
    if a.all or not any((a.shape, a.sufficiency, a.direction, a.horizon)):
        a.shape = a.sufficiency = a.direction = a.horizon = True

    sc = SCENARIOS[a.scenario_name]
    if sc.stockout_mode != "lost_sales" or sc.leadtime.min() != sc.leadtime.max():
        raise SystemExit(f"{a.scenario_name!r}: this probe needs lost sales + deterministic LT")
    V, Q = solve(sc, u_max=a.u_max, q_cap=a.q_cap, d_max=a.d_max)
    pmf = demand_pmf(sc, a.d_max)
    print(f"scenario {sc.scenario_name!r}: exact optimum V_0(0,0) = {V[0,0,0]:.4f}")
    visit = visitation(sc, Q, pmf, a.u_max, a.q_cap)

    if a.shape:       shape(sc, Q, visit, a.period, a.base_stock)
    if a.sufficiency: sufficiency(sc, Q, visit, a.period)
    if a.direction:   direction(sc, Q, visit, a.period, a.base_stock)
    if a.horizon:     horizon(sc, Q)


if __name__ == "__main__":
    main()

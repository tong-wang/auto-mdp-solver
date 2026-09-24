"""Is the Clark-Scarf decomposition actually optimal *here*?

The shipped DP benchmark solves the chain as N one-dimensional problems, on the
strength of Theorems 1-2. That is a theorem about the paper's model; this module
refuses to take it on faith for *our* implementation of that model, and checks
it against a **brute-force DP over the full joint state**.

This is a spec-§1.2 second-implementation gate. The equivalence claimed:

    on an instance small enough to enumerate, the expected discounted cost of
    the Clark-Scarf policy equals the true optimum, to within the demand
    truncation and grid resolution.

Why it is tractable only here: at ``leadtime == 1`` the arrival event empties
the pipeline before the decision, so the joint state collapses to on-hand stock
per installation — 2 numbers at N=2. At the campaign's target (N=3, L=2) the
same state is 6-dimensional, which is exactly why the decomposition is needed.

The brute force is a SECOND implementation of the dynamics, written from the
cost expression rather than by calling the simulator, so agreement is evidence
about the model and not just about one code path. ``clark_scarf_test.py`` runs
it as a test; ``python clark_scarf_dp_exactness.py`` prints the comparison.
"""

from __future__ import annotations

import numpy as np

from clark_scarf_benchmark_dp import ClarkScarfDP, _poisson_pmf
from clark_scarf_scenarios import SCENARIOS, ClarkScarfScenario


class JointDP:
    """Exact DP over the full joint state (s0, s1) for an N=2, L=1 chain.

    ``s0`` = on-hand at the retailer (may be negative: backlog), ``s1`` = on-hand
    at the depot, both read at the decision point (after arrivals land).

    Transition, derived independently of the simulator: shipping ``q0`` down and
    buying ``q1`` in leaves, after demand ``d`` and next period's arrivals,

        s0' = s0 - d + q0        s1' = s1 - q0 + q1

    One period's cost, with ``h = h_install``:

        h[0]*max(0, s0-d)  +  h[1]*(s1-q0)  +  h[1]*q0  +  h[2]*q1
                                                   +  p*max(0, d-s0)

    Note ``q0`` cancels: stock in transit to the retailer is charged the same
    rate as stock sitting at the depot, which is precisely Assumption 3's
    "at a lower level **or in transit to a lower level**". And ``h[2] == 0`` at
    N=2, so goods in transit from the outside supplier cost nothing — they are
    on order, not yet in the system. Both are consequences of the corrected
    echelon assignment, and if either were wrong this DP would disagree with
    the simulator.
    """

    def __init__(self, scenario: ClarkScarfScenario, s0_lo=-34, s0_hi=30,
                 s1_hi=44, q_max=22) -> None:
        assert scenario.n_echelons == 2 and scenario.leadtime == 1, (
            "the joint DP is only tractable for N=2, L=1"
        )
        self.sc = scenario
        self.s0 = np.arange(s0_lo, s0_hi + 1)
        self.s1 = np.arange(0, s1_hi + 1)
        self.q_max = q_max
        self._solve()

    def _period_cost(self) -> np.ndarray:
        """E_d[cost] on the (s0, s1) grid — independent of the decision."""
        sc = self.sc
        d_vals, d_p = _poisson_pmf(float(sc.demand.mean()))
        S0 = self.s0[:, None]
        ec = np.zeros((len(self.s0), 1))
        for d, p in zip(d_vals, d_p):
            ec = ec + p * (
                sc.h_install[0] * np.maximum(S0 - d, 0.0)
                + sc.p_short * np.maximum(d - S0, 0.0)
            )
        return ec + sc.h_install[1] * self.s1[None, :]

    def _solve(self) -> None:
        sc = self.sc
        d_vals, d_p = _poisson_pmf(float(sc.demand.mean()))
        n0, n1 = len(self.s0), len(self.s1)
        ec = self._period_cost()

        self.V = [None] * (sc.horizon + 1)
        self.V[sc.horizon] = np.zeros((n0, n1))
        for t in range(sc.horizon - 1, -1, -1):
            Vn = self.V[t + 1]
            # G[a, b] = E_d[ V_{t+1}(a - d, b) ]  where a = s0+q0, b = s1-q0+q1
            G = np.zeros((n0, n1))
            for d, p in zip(d_vals, d_p):
                idx = np.clip(np.arange(n0) - int(d), 0, n0 - 1)
                G += p * Vn[idx, :]
            best = np.full((n0, n1), np.inf)
            I0 = np.arange(n0)[:, None]
            I1 = np.arange(n1)[None, :]
            for q0 in range(self.q_max + 1):
                feasible = self.s1[None, :] >= q0        # cannot ship what depot lacks
                a = np.clip(I0 + q0, 0, n0 - 1)
                for q1 in range(self.q_max + 1):
                    b = np.clip(I1 - q0 + q1, 0, n1 - 1)
                    cand = np.where(feasible, G[a, b], np.inf)
                    best = np.minimum(best, cand)
            self.V[t] = ec + self.sc.beta * best

    def value_at(self, s0: int, s1: int, t: int = 0) -> float:
        return float(self.V[t][int(s0) - self.s0[0], int(s1) - self.s1[0]])

    def evaluate_policy(self, action_fn) -> np.ndarray:
        """Policy evaluation on the same grid — the cost of a GIVEN rule.

        ``action_fn(t, s0, s1) -> (q0, q1)``. Returns V_0, comparable term for
        term with ``self.V[0]``.
        """
        sc = self.sc
        d_vals, d_p = _poisson_pmf(float(sc.demand.mean()))
        n0, n1 = len(self.s0), len(self.s1)
        ec = self._period_cost()
        V = np.zeros((n0, n1))
        for t in range(sc.horizon - 1, -1, -1):
            G = np.zeros((n0, n1))
            for d, p in zip(d_vals, d_p):
                idx = np.clip(np.arange(n0) - int(d), 0, n0 - 1)
                G += p * V[idx, :]
            nxt = np.zeros((n0, n1))
            for i, s0 in enumerate(self.s0):
                for j, s1 in enumerate(self.s1):
                    q0, q1 = action_fn(t, int(s0), int(s1))
                    q0 = int(min(max(q0, 0), s1))
                    q1 = int(max(q1, 0))
                    a = min(max(i + q0, 0), n0 - 1)
                    b = min(max(j - q0 + q1, 0), n1 - 1)
                    nxt[i, j] = G[a, b]
            V = ec + sc.beta * nxt
        return V


def clark_scarf_action_fn(dp: ClarkScarfDP):
    """The shipped policy as ``(t, s0, s1) -> (q0, q1)``.

    At L=1 nothing is in transit at the decision point, so echelon position and
    echelon stock coincide: ``u_1 = x_1 = s0`` and ``u_2 = x_2 = s0 + s1``.
    """
    def fn(t: int, s0: int, s1: int) -> tuple[int, int]:
        tt = min(t, dp.scenario.horizon - 1)
        y1 = min(max(float(dp.ybar[tt, 0]), s0), s0 + s1)   # clipped by availability
        y2 = max(float(dp.ybar[tt, 1]), s0 + s1)            # outside supplier
        return int(round(y1 - s0)), int(round(y2 - (s0 + s1)))
    return fn


def compare(scenario_name: str = "verify_tiny") -> dict:
    sc = SCENARIOS[scenario_name]
    joint = JointDP(sc)
    dp = ClarkScarfDP(sc)
    V_cs = joint.evaluate_policy(clark_scarf_action_fn(dp))

    cover = int(sc.demand.mean())
    s0 = s1 = 2 * cover          # init_state gives cover; the A event adds cover again
    opt = joint.value_at(s0, s1, 0)
    cs = float(V_cs[s0 - joint.s0[0], s1 - joint.s1[0]])
    return {
        "scenario": scenario_name,
        "start": (s0, s1),
        "joint_optimal": opt,
        "clark_scarf": cs,
        "gap": cs - opt,
        "gap_pct": 100.0 * (cs - opt) / abs(opt) if opt else float("nan"),
        "ybar": dp.ybar.tolist(),
    }


if __name__ == "__main__":
    r = compare()
    print(f"=== Clark-Scarf exactness on {r['scenario']} ===")
    print(f"  start state (s0, s1) = {r['start']}")
    print(f"  brute-force joint DP optimum : {r['joint_optimal']:.6f}")
    print(f"  Clark-Scarf decomposition    : {r['clark_scarf']:.6f}")
    print(f"  gap                          : {r['gap']:+.6f}  ({r['gap_pct']:+.4f}%)")
    print(f"  critical numbers ybar[t]     : {r['ybar']}")

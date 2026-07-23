"""AP benchmark: the allocation-assumption relaxation, solved exactly by DP.

This is Wang & Toktay (2008) section 4.1. Relaxing the non-negativity
constraint on deliveries lets "misallocated" units be taken back, which makes
myopic earliest-due-date allocation optimal and removes the allocation decision
from the problem entirely. What remains is a pure ordering problem in the
two-dimensional state (u, v-hat), solvable by backward induction.

Because it is a relaxation, its optimal cost is a LOWER BOUND on the true
problem's optimal cost. It is not an implementable policy — no simulator can
realise it — so it plays the role of a `--reference` in the eval gate, never a
`--baseline`. Its ordering policy, however, *is* implementable, and the paper
adopts it for the protection-level heuristics (section 4.2); this module writes
that policy table out for adi_flex_benchmark_pl.py to consume.

Derivation, for T=2, L=0, alpha=1 (the frozen IR's structural setting):

  state      (u, v-hat)   u = modified inventory position before ordering,
                          v-hat = TOTAL advance demand known to be due next period
  decision   y = u + z    the position after ordering
  eq (14)    x_{i+1} = (y - D)^+ - (y + v-hat - d0)^-,   D = d0 + d1 + d2

  Since y - D <= y + v-hat - d0 always, eq (14) is a deadband and the loss
  separates into two independent one-dimensional expectations:

      E[L(x_{i+1})] = h * E[(y - D)^+]  +  p * E[(d0 - y - v-hat)^+]

  eq (15)    u_{i+1} = y - D            eq (17)    v-hat_{i+1} = d2

  so, with A(y) = E[(y-D)^+], B(w) = E[(d0-w)^+] and
  C_i(y) = E[f_{i+1}(y - m - d2, d2)] where m = d0 + d1 is independent of d2,

      G_i(y, v-hat) = h*A(y) + p*B(y + v-hat) + C_i(y)
      f_i(u, v-hat) = min_{y >= u} { K * 1{y > u} + G_i(y, v-hat) }      f_N = 0

  The v-hat dependence enters only through B(y + v-hat). That is exactly why
  the paper's Proposition 3 holds: at the optimum y is far above the support of
  d0, so B is flat there and S(v-hat) comes out independent of v-hat, while the
  reorder point s(v-hat) still decreases in v-hat.

Validation: `--check-tables` reproduces the paper's Table 2 and Table 3 for all
eight registered instances. Both match exactly, which is the evidence that this
recurrence is the paper's and not merely a plausible one.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from adi_flex_scenarios import SCENARIOS, AdiFlexScenario
from adi_flex_benchmark_common import solutions_path


# state-grid extent; asserted non-binding after solving
U_LO, U_HI = -120, 120
VHAT_MAX = 20
TAIL = 1e-12


# ---------------------------------------------------------------------------
# Poisson helpers (kept local: the repo is deliberately scipy-free)
# ---------------------------------------------------------------------------


def poisson_pmf(rate: float, kmax: int) -> np.ndarray:
    """pmf over 0..kmax, with the tail beyond kmax folded into the last cell."""
    if rate == 0.0:
        out = np.zeros(kmax + 1)
        out[0] = 1.0
        return out
    k = np.arange(kmax + 1)
    logp = k * math.log(rate) - rate - np.array([math.lgamma(i + 1) for i in k])
    out = np.exp(logp)
    out[-1] += max(0.0, 1.0 - out.sum())
    return out / out.sum()


def truncation_point(rate: float, tail: float = TAIL) -> int:
    """Smallest k with P(X > k) <= tail."""
    if rate == 0.0:
        return 0
    pr = poisson_pmf(rate, 400)
    cum, k = 0.0, 0
    while k < 400 and cum < 1.0 - tail:
        cum += pr[k]
        k += 1
    return k


# ---------------------------------------------------------------------------
# The DP
# ---------------------------------------------------------------------------


class ApSolution:
    """Solved AP relaxation: the full (s, S) policy table and the bound.

    The policy is NON-STATIONARY — a finite horizon with terminal value zero
    means the order-up-to level falls as the horizon approaches, steeply in the
    last few periods where there is almost no future left to stock for. The
    table is therefore indexed by (period, v-hat), not v-hat alone. The paper's
    Table 3 reports period 1 only, "for ease of comparison"; using that row at
    every period would badly over-order near the end.
    """

    def __init__(
        self,
        scenario_name: str,
        lower_bound: float,
        s_table: dict[int, dict[int, int]],
        big_s_table: dict[int, dict[int, int]],
    ) -> None:
        self.scenario_name = scenario_name
        self.lower_bound = lower_bound
        self.s_table = s_table              # period -> vhat -> s
        self.big_s_table = big_s_table      # period -> vhat -> S

    @property
    def horizon(self) -> int:
        return len(self.s_table)

    def s_by_vhat(self, period: int = 0) -> dict[int, int]:
        return self.s_table[min(period, self.horizon - 1)]

    def big_s_by_vhat(self, period: int = 0) -> dict[int, int]:
        return self.big_s_table[min(period, self.horizon - 1)]

    def order_up_to(self, period: int, u: int, vhat: int) -> int:
        """Order quantity under the (s_i(v-hat), S_i(v-hat)) rule."""
        i = min(max(period, 0), self.horizon - 1)
        s_row, big_s_row = self.s_table[i], self.big_s_table[i]
        v = min(max(vhat, 0), max(s_row))
        return max(0, big_s_row[v] - u) if u <= s_row[v] else 0

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# AdiFlex AP (allocation-assumption relaxation) solution\n")
            f.write(f"# scenario: {self.scenario_name}\n")
            f.write(f"# lower_bound_cost: {self.lower_bound:.6f}\n")
            f.write(f"# horizon: {self.horizon}\n")
            f.write("# state-dependent (s, S) policy in the modified inventory\n")
            f.write("# position, per period and per total advance demand due next period\n")
            f.write("period\tvhat\ts\tS\n")
            for i in sorted(self.s_table):
                for v in sorted(self.s_table[i]):
                    f.write(f"{i}\t{v}\t{self.s_table[i][v]}\t{self.big_s_table[i][v]}\n")

    @classmethod
    def read(cls, path: Path) -> "ApSolution":
        s_table: dict[int, dict[int, int]] = {}
        big_s_table: dict[int, dict[int, int]] = {}
        bound, name = float("nan"), "?"
        for line in Path(path).read_text().splitlines():
            if line.startswith("# lower_bound_cost:"):
                bound = float(line.split(":")[1])
            elif line.startswith("# scenario:"):
                name = line.split(":")[1].strip()
            elif line.startswith("#") or line.startswith("period"):
                continue
            elif line.strip():
                i, v, s, big_s = (int(x) for x in line.split("\t"))
                s_table.setdefault(i, {})[v] = s
                big_s_table.setdefault(i, {})[v] = big_s
        return cls(name, bound, s_table, big_s_table)


def solve_ap(scenario: AdiFlexScenario) -> ApSolution:
    """Backward induction over (u, v-hat); see the module docstring."""
    l0 = scenario.demand_now.mean()
    l1 = scenario.demand_next.mean()
    l2 = scenario.demand_later.mean()
    h, p, K = (scenario.holding_cost, scenario.backorder_cost,
               scenario.order_cost_fixed)

    us = np.arange(U_LO, U_HI + 1)
    n_u = len(us)

    k_total, k_now, k_m, k_later = (truncation_point(r) for r in
                                    (l0 + l1 + l2, l0, l0 + l1, l2))
    p_total = poisson_pmf(l0 + l1 + l2, k_total)
    p_now   = poisson_pmf(l0, k_now)
    p_m     = poisson_pmf(l0 + l1, k_m)
    p_later = poisson_pmf(l2, k_later)

    # A(y) = E[(y - D)^+] over the y grid
    A = np.array([
        float(np.dot(p_total, np.maximum(y - np.arange(k_total + 1), 0)))
        for y in us
    ])
    # B(w) = E[(d0 - w)^+], tabulated over w = y + v-hat
    w_lo, w_hi = U_LO, U_HI + VHAT_MAX
    B_tab = np.array([
        float(np.dot(p_now, np.maximum(np.arange(k_now + 1) - w, 0)))
        for w in range(w_lo, w_hi + 1)
    ])

    f_next = np.zeros((n_u, VHAT_MAX + 1))          # f_N == 0
    s_table: dict[int, dict[int, int]] = {}
    big_s_table: dict[int, dict[int, int]] = {}

    for i in range(scenario.horizon - 1, -1, -1):
        # C_i(y) = sum_{m, d2} P(m) P(d2) f_{i+1}(y - m - d2, d2)
        C = np.zeros(n_u)
        for d2 in range(k_later + 1):
            if p_later[d2] == 0.0:
                continue
            col = f_next[:, min(d2, VHAT_MAX)]
            acc = np.zeros(n_u)
            for m in range(k_m + 1):
                if p_m[m] == 0.0:
                    continue
                acc += p_m[m] * col[np.clip(np.arange(n_u) - (m + d2), 0, n_u - 1)]
            C += p_later[d2] * acc

        f_cur = np.empty((n_u, VHAT_MAX + 1))
        for vhat in range(VHAT_MAX + 1):
            idx = np.clip((us + vhat) - w_lo, 0, len(B_tab) - 1)
            G = h * A + p * B_tab[idx] + C

            suffix_min = np.minimum.accumulate(G[::-1])[::-1]   # min_{y >= u} G
            best_strictly_after = np.empty(n_u)
            best_strictly_after[:-1] = suffix_min[1:]
            best_strictly_after[-1] = np.inf
            f_cur[:, vhat] = np.minimum(G, K + best_strictly_after)

            # record the policy for EVERY period: it is non-stationary, and
            # applying period 0's row at period 11 would badly over-order
            big_s_idx = int(np.argmin(G))
            threshold = K + G[big_s_idx]
            below = np.where(G[:big_s_idx] > threshold)[0]
            s_idx = int(below[-1]) if len(below) else big_s_idx
            s_table.setdefault(i, {})[vhat] = int(us[s_idx])
            big_s_table.setdefault(i, {})[vhat] = int(us[big_s_idx])
        f_next = f_cur

    # the initial state is (x, v^i, v^{i+1}) = (0,0,0), so u = 0 and v-hat = 0
    u0 = int(np.where(us == 0)[0][0])
    lower_bound = float(f_next[u0, 0])

    lo = min(v for row in s_table.values() for v in row.values())
    hi = max(v for row in big_s_table.values() for v in row.values())
    assert lo > U_LO + 5 and hi < U_HI - 5, (
        f"state grid [{U_LO}, {U_HI}] is binding on the policy ([{lo}, {hi}]) — widen it"
    )
    return ApSolution(scenario.scenario_name or "?", lower_bound,
                      s_table, big_s_table)


# ---------------------------------------------------------------------------
# Protection levels (Table 2)
# ---------------------------------------------------------------------------


def protection_level_sigma(scenario: AdiFlexScenario) -> int:
    """PL(sigma): argmin_s h*s + p*E[(d0 - s)^+]  — eq (20), a newsvendor at
    the ratio p / (p + h) on the urgent-demand distribution."""
    l0 = scenario.demand_now.mean()
    k = truncation_point(l0)
    pr = poisson_pmf(l0, k)
    support = np.arange(k + 1)
    best_val, best_s = None, 0
    for s in range(k + 1):
        val = (scenario.holding_cost * s
               + scenario.backorder_cost * float(np.dot(pr, np.maximum(support - s, 0))))
        if best_val is None or val < best_val - 1e-12:
            best_val, best_s = val, s
    return best_s


def protection_level_max(scenario: AdiFlexScenario, tol: float = 0.001) -> int:
    """PL(Sigma): smallest Sigma with P(d0 > Sigma) < tol — 'large enough' to
    cover the urgent demand's whole support, which is what removes crossover."""
    l0 = scenario.demand_now.mean()
    k = truncation_point(l0)
    pr = poisson_pmf(l0, k)
    cum = 0.0
    for s in range(k + 1):
        cum += pr[s]
        if 1.0 - cum < tol:
            return s
    return k


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _check_tables() -> int:
    """Reproduce the paper's Table 2 and Table 3 across the whole ladder."""
    paper_big_s = {"exp0": 36, "exp1": 35, "exp2": 33, "exp3": 32,
                   "exp4": 30, "exp5": 28, "exp6": 26, "exp7": 26}
    paper_s0 = {"exp0": 2, "exp1": 1, "exp2": 0, "exp3": -1,
                "exp4": -1, "exp5": -2, "exp6": -3, "exp7": -3}
    # Table 2 covers only the instances where crossover exists
    paper_sigma = {"exp2": 6, "exp3": 5, "exp4": 4, "exp5": 2, "exp6": 0}
    paper_max   = {"exp2": 11, "exp3": 10, "exp4": 8, "exp5": 5, "exp6": 0}

    failures = []
    print("=== Table 3: period-1 (s(vhat), S(vhat)) ===")
    print(f"{'inst':6s} {'S':>4s} {'paper':>6s} {'s(0)':>6s} {'paper':>6s} "
          f"{'S const':>8s} {'s decr':>7s} {'AP bound':>10s}")
    for name, scenario in SCENARIOS.items():
        sol = solve_ap(scenario)
        # the paper's Table 3 is period 1, i.e. our period 0
        s_row, big_s_row = sol.s_by_vhat(0), sol.big_s_by_vhat(0)
        big_s = big_s_row[0]
        s0 = s_row[0]
        s_const = all(big_s_row[v] == big_s for v in big_s_row)
        s_decr = all(s_row[v + 1] <= s_row[v] for v in range(VHAT_MAX))
        ok = big_s == paper_big_s[name] and s0 == paper_s0[name]
        if not (ok and s_const and s_decr):
            failures.append(name)
        print(f"{name:6s} {big_s:4d} {paper_big_s[name]:6d} {s0:6d} "
              f"{paper_s0[name]:6d} {str(s_const):>8s} {str(s_decr):>7s} "
              f"{sol.lower_bound:10.2f}  {'OK' if ok else 'MISMATCH'}")

    print("\n=== Table 2: protection levels ===")
    print(f"{'inst':6s} {'PL(sigma)':>10s} {'paper':>6s} {'PL(Sigma)':>10s} {'paper':>6s}")
    for name, scenario in SCENARIOS.items():
        sig, mx = protection_level_sigma(scenario), protection_level_max(scenario)
        exp_sig, exp_mx = paper_sigma.get(name), paper_max.get(name)
        mark = ""
        if exp_sig is not None:
            if sig != exp_sig or mx != exp_mx:
                failures.append(name)
                mark = "  MISMATCH"
            else:
                mark = "  OK"
        # note: `exp or '-'` would misreport a legitimate 0 (exp6) as absent
        show_sig = "-" if exp_sig is None else str(exp_sig)
        show_mx  = "-" if exp_mx  is None else str(exp_mx)
        print(f"{name:6s} {sig:10d} {show_sig:>6s} "
              f"{mx:10d} {show_mx:>6s}{mark}")

    if failures:
        print(f"\nFAIL: {sorted(set(failures))}")
        return 1
    print("\nall paper tables reproduced exactly")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Solve the AP relaxation and write its (s,S) policy table."
    )
    p.add_argument("-s", "--scenario_name", type=str, default="exp4")
    p.add_argument("--outfile", type=str, default=None)
    p.add_argument("--check-tables", action="store_true",
                   help="reproduce the paper's Tables 2 and 3 and exit")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> int:
    args = parse_args()
    if args.check_tables:
        return _check_tables()

    scenario = SCENARIOS[args.scenario_name]
    sol = solve_ap(scenario)
    out = Path(args.outfile) if args.outfile else solutions_path(args.scenario_name, "ap")
    sol.write(out)
    print(f"scenario   : {args.scenario_name}  ({scenario.desc})")
    print(f"AP bound   : {sol.lower_bound:.4f}   (lower bound on the optimal cost)")
    print(f"policy(t=0): S={sol.big_s_by_vhat(0)[0]} (constant in vhat), "
          f"s(0)={sol.s_by_vhat(0)[0]} .. s({VHAT_MAX})={sol.s_by_vhat(0)[VHAT_MAX]}")
    tail = [sol.big_s_by_vhat(i)[0] for i in range(sol.horizon)]
    print(f"S(vhat=0) by period: {tail}   <- non-stationary; falls toward the horizon")
    print(f"PL(sigma)  : {protection_level_sigma(scenario)}")
    print(f"PL(Sigma)  : {protection_level_max(scenario)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

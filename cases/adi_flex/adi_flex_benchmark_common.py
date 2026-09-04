"""Shared backward induction behind the `dp` and `ap` benchmark arms.

**One recursion, two roles (spec §9.9).** This module owns the math; the role
it is read under belongs to the arm that calls it, and the two arms are
separate files because their positions relative to the optimum differ:

- ``adi_flex_benchmark_dp.py`` — role ``exact``. On a *homogeneous* customer
  base (one demand lead time, `alloc_enabled=false`) a single due-date class
  makes the allocation relaxation tight, so this recursion is the paper's own
  §3.2 DP and its value *is* the optimum.
- ``adi_flex_benchmark_ap.py`` — role ``relaxed``. On a *heterogeneous* base
  the same recursion is the §4.1 allocation-postponement relaxation: it costs
  early delivery as though the allocation could be deferred, which drops a
  constraint the deployed policy must respect. Its value is a LOWER bound on
  optimal cost and is not attainable.

Declaring these as one arm would either promise exactness the het branch does
not have, or give up the `relaxed` + `feasible` bracket that certifies the het
optimality gap without an exact solver. Hence the split.

Implements the state-space reduction of Wang & Toktay (2008) §3.2. The system
state collapses to (u, vhat):

    u    = modified inventory position before ordering
         = inv - adv[0] - adv[1]            (L=0: pipeline is empty at review)
    vhat = last period's far-class DRAW, d[2] — the gross amount due next
           period, NOT the profile still outstanding after early fulfilment.
           The two differ in 70% of visited states (measured, #E4); the
           paragraph below is why the reduction is still exact

with linear evolution u' = y - d (y = u + z is the post-order MIP) and
vhat' = d. The per-period cost in reduced coordinates,

    K*1{z>0} + E_d[ h*(y - d)^+ ] + p*(y + vhat)^-,

equals the simulator's physical cost along any consistent trajectory: whenever
early fulfillment has reduced the unsatisfied profile below the total (v <
vhat), the due-now slot is provably zero, so the backorder terms agree.
Checked, not assumed: over 18000 states drawn under both the exact and a
random policy, `v < vhat` held in 71.8% and `adv[0] > 0` co-occurred with it
**zero** times (#E4).

Backward induction over periods 1..N with boundary f_{N+1} = 0 yields the
optimal state-dependent policy, which is (s_i(vhat), S_i(vhat)) by Prop 2;
the full argmin table y*(i, u, vhat) is stored so evaluation never relies on
the structural result. Structural checks (Prop 3: S independent of vhat, s
decreasing in vhat) are printed.

Outputs (spec §9.6), written by whichever arm ran, under its own directory:
    results/{scenario}/{arm}/{scenario}.txt         (s/S table + value)
    results/{scenario}/{arm}/{scenario}_policy.npz  (y* table)

This module has no CLI — run an arm, not the shared solver.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from adi_flex_scenarios import SCENARIOS, AdiFlexScenario

# u grid: wide enough that the clamp at the bottom only affects states an
# optimal policy never visits (optimal s stays above ~-40 at these costs)
U_MIN = -150
U_MAX = 100          # the DP's own grid ceiling on the post-order position y
_TAIL = 1e-12        # Poisson truncation tail mass


class ObsView:
    """Decoded `vec` observation — the one decode both eval arms play on.

    Layout (F7 two-phase gym):
    [inv, pipe(pipe_slots), adv(T_dl), time_to_go, phase, d(T_dl+1), surplus,
     outstanding].
    TWO widths vary — `pipe_slots` is the instance's own supply lead time
    (paper eq. 1, zero at L=0) and `T_dl` its demand lead time (F10) — so every
    offset after the pipeline must be computed, never written out. Both eval
    arms held those offsets as literals against a fixed 5-slot register until
    F6; dropping the register broke both at once, which is why the decode lives
    here instead of twice at the call sites.

    `u` is the paper's modified inventory position inv + sum(pipe) - sum(adv);
    `d` is the full realized demand vector, and `surplus`/`outstanding` are the
    feasible-set coordinates of whichever allocation component the env is
    currently asking about.
    """

    __slots__ = ("u", "period", "phase", "adv", "vhat", "d", "surplus", "outstanding")

    def __init__(
        self, obs: np.ndarray, pipe_slots: int, horizon_n: int, T_dl: int = 2,
        alloc_enabled: bool = True,
    ) -> None:
        n, a = pipe_slots, T_dl
        r = lambda x: int(round(float(x)))  # noqa: E731
        adv_lo = 1 + n                       # first advance-profile slot
        tail   = adv_lo + a                  # time_to_go, then d, then the alloc trio
        self.adv         = [r(x) for x in obs[adv_lo:tail]]
        self.u           = r(obs[0] + obs[1:adv_lo].sum()) - sum(self.adv)
        self.period      = horizon_n - r(obs[tail])
        # the DP's reduced coordinate: advance demand due NEXT period. Read off
        # the STATE (`adv[1]`), never off last period's draw `d[2]` — the two
        # differ in 70% of visited states, and the state one is what the
        # optimal policy actually needs: the same y* table played with
        # `vhat = adv[1]` costs 685.263 against `d[2]`'s 685.263, identical
        # over 61440 decisions, while `vhat = 0` costs 791.486 (F25, #E4).
        # Reading it from the state is also what keeps this decoder working
        # when the observation carries no history at all.
        # placeholder; set below, once we know whether history is rendered
        self.vhat        = 0
        # the within-period block exists only when there IS an allocation (F23).
        # The caller passes the flag rather than this inferring it from a width:
        # inferring would silently mis-decode the day a width moves for some
        # other reason, which is exactly how F23 broke this decoder.
        if alloc_enabled:
            self.d           = [r(x) for x in obs[tail + 1: tail + 2 + a]]
            self.phase       = r(obs[tail + 2 + a])       # 0 = order, 1 = allocate
            self.surplus     = r(obs[tail + 3 + a])
            self.outstanding = r(obs[tail + 4 + a])
            # the AP/DP table's coordinate is the GROSS far draw
            self.vhat        = self.d[a] if len(self.d) > a else 0
        else:
            self.d = []                       # no history rendered on this branch
            self.phase, self.surplus, self.outstanding = 0, max(0, r(obs[0])), 0
            # ...and here `adv[1]` is an EXACT substitute for it, which is what
            # lets the table be driven from the state alone. Sound only because
            # the forced maximal fill empties the near class first, so
            # `v < vhat` implies `adv[0] == 0` — measured 0 violations in 18000
            # states here, against 37.1% on the allocating branch, where the
            # policy may withhold a reserve and the property fails (F25/F26)
            self.vhat        = self.adv[1] if len(self.adv) > 1 else 0
        expect = tail + 1 + ((a + 1) + 3 if alloc_enabled else 0)
        assert len(obs) == expect, (
            f"ObsView decoded {expect} slots but the observation has {len(obs)} — "
            f"the layout moved and this decoder did not (F24)")


# --------------------------------------------------------------------------
# The shared eval row (spec §9.1) — ONE definition, four arms
# --------------------------------------------------------------------------

EVAL_HEADER = ("lambda_seg\tL\tK\th\tp\tN\t"
               "reward_mean\treward_var\tsemivar_d\tsemivar_u")


def eval_row(scenario, stats: dict) -> str:
    """One TSV row, with the rate vector DECLARED rather than enumerated.

    Was `lambda0\tlambda1\tlambda2` in all four arms — three literal columns
    for a vector whose length is `T_dl + 1` and varies by instance. That is
    F10's defect one layer out of the IR, and it had both failure modes live at
    once: `T_dl = 1` raised `IndexError: tuple index out of range` on
    `scenario.lambda2`, and `T_dl = 3` silently dropped its fourth rate, so
    every het3 row on disk described a mix it was not run on. One
    comma-joined column carries any width.
    """
    seg = ",".join(f"{r:g}" for r in scenario.demand.rates)
    return (f"{seg}\t{scenario.L}\t{scenario.K}\t{scenario.h}\t"
            f"{scenario.p}\t{scenario.N}\t"
            f"{stats['reward_mean']:.6f}\t{stats['reward_var']:.6f}\t"
            f"{stats['semivar_d']:.6f}\t{stats['semivar_u']:.6f}")


def _truncated_poisson(rate: float) -> np.ndarray:
    """pmf over 0..d_max with cdf tail below _TAIL, renormalized."""
    if rate == 0:
        return np.array([1.0])
    d_max = int(stats.poisson.ppf(1.0 - _TAIL, rate)) + 1
    pmf = stats.poisson.pmf(np.arange(d_max + 1), rate)
    return pmf / pmf.sum()


def solve_dp(scenario: AdiFlexScenario) -> dict:
    """Backward induction; returns the policy table and value function.

    General (lambda0, lambda1, lambda2) with L=0, T=2: the AP-relaxation DP of
    paper §4.1 in reduced state (u, vhat), vhat = total advance demand due next
    period (= last period's d2 draw). Per-period cost at post-order position y:

        E[ h*(y - d0-d1-d2)^+ ] + E_d0[ p*(y + vhat - d0)^- ]

    (holding on stock left after delivering everything outstanding — early
    delivery is free under the allocation relaxation; backorders only on the
    due-now shortfall). For the homogeneous branch (lambda0=lambda1=0) the
    relaxation is tight and this is the exact DP of §3.2; for heterogeneous
    mixes with crossover it is a lower bound on the optimal cost.
    """
    # The reduced state is (u, V-hat) with dim(V-hat) = T - L - 1, so the DP's
    # shape follows the INSTANCE, not a constant (F10's lesson one layer up).
    # Two cases are implemented, and they are the two the paper's own numerics
    # cover (§3.3):
    #
    #   dim 1  — Case 2 at L=0, T=2: V-hat is the scalar vhat = the far-class
    #            draw, and next period's vhat IS this period's realized d[T].
    #   dim 0  — Case 1 at L=0, T<=1: V-hat is EMPTY and Prop. 1's scalar u is
    #            the entire state. The recursion is the same one with a single
    #            vhat level pinned at 0, which is exact rather than a
    #            degeneration: with the profile one class wide, the demand due
    #            next period is drawn AFTER the order and so cannot be in the
    #            pre-order state at all.
    #
    # Anything wider needs a wider DP, not a wider loop — refuse rather than
    # silently solve the wrong reduction.
    n_vhat = max(0, scenario.T_dl - scenario.L - 1)
    assert n_vhat <= 1, (
        f"this DP implements dim(V-hat) in {{0, 1}}; got L={scenario.L}, "
        f"T_dl={scenario.T_dl}, so the reduced state carries {n_vhat} v-hat "
        f"components. A wider instance needs a wider DP."
    )

    # components 0..T_dl-1 are the NEAR classes (settled or shipped within the
    # period); component T_dl is the FAR class, which is what becomes next
    # period's vhat when there is one
    # NEAR = components 0..T_dl-1, FAR = component T_dl. At T_dl = 0 there is
    # no near part at all (the whole draw is the far class, due on arrival),
    # so it starts from a point mass rather than from component 0 — starting
    # from component 0 would convolve the single component with itself and
    # solve a DP for twice the demand.
    rates   = scenario.demand.rates
    pmf0    = _truncated_poisson(rates[0])
    pmf_near = np.array([1.0])
    for r in rates[0:scenario.T_dl]:
        pmf_near = np.convolve(pmf_near, _truncated_poisson(r))
    pmf_far = _truncated_poisson(rates[scenario.T_dl])
    d2_max, d01_max, d0_max = len(pmf_far) - 1, len(pmf_near) - 1, len(pmf0) - 1
    pmf2, pmf01 = pmf_far, pmf_near
    pmf_tot = np.convolve(pmf_near, pmf_far)          # |D_i|, ONE period's arrival

    # --- the protection window, paper eq. (14) -----------------------------
    # With a supply lead time the order placed now lands at i+L, so the loss is
    # charged on the inventory at i+L+1 and the two demand aggregates span the
    # window rather than the period:
    #
    #   A = sum_{l=i..i+L} sum_j d_l^j          everything that ARRIVES in it
    #   B = sum_{l=i..i+L} sum_{j=l..i+L} d_l^j  ... of which is DUE by i+L
    #
    #   x_{i+L+1} = (y - A)^+ - (y + vhat - B)^-        (14)
    #   G_i(y, Vhat) = alpha^L E[L(x_{i+L+1})] + alpha E f_{i+1}(...)   (12)
    #
    # Both are plain sums of INDEPENDENT Poissons (unlike the protection
    # ladder's nested quantity, which is weighted and therefore not Poisson),
    # so closed forms exist: rate(A) = (L+1)*sum(lam), rate(B) =
    # sum_k (L+1-k)*lam_k. They are built here by CONVOLUTION anyway, so that
    # at L=0 they reduce to exactly the arrays the L=0 path already used —
    # a single Poisson of the same rate would differ in the truncation tail and
    # move numbers on boards that are already published.
    pmf_A = pmf_tot
    for _ in range(scenario.L):
        pmf_A = np.convolve(pmf_A, pmf_tot)
    pmf_B = np.array([1.0])
    for sft in range(scenario.L + 1):                 # arrival period i+sft
        for k in range(0, scenario.L - sft + 1):      # due-offsets 0..L-sft
            pmf_B = np.convolve(pmf_B, _truncated_poisson(rates[k]))
    disc_L = scenario.discount ** scenario.L          # alpha^L on the loss (12)

    u_grid = np.arange(U_MIN, U_MAX + 1)
    n_u    = len(u_grid)
    n_v    = d2_max + 1 if n_vhat else 1                      # vhat' = realized far draw
    K, h, p, alpha, N = scenario.K, scenario.h, scenario.p, scenario.discount, scenario.N

    def uidx(u: np.ndarray) -> np.ndarray:
        return np.clip(u - U_MIN, 0, n_u - 1)

    # E[h*(y - d_total)^+] for every grid level y (independent of vhat and period)
    tvals = np.arange(len(pmf_A))
    eh = disc_L * h * (np.maximum(u_grid[:, None] - tvals[None, :], 0) @ pmf_A)
    # E_B[p*(y + vhat - B)^-] for every (y, vhat)
    d0vals = np.arange(len(pmf_B))
    pen = disc_L * p * (np.maximum(
        d0vals[None, None, :] - (u_grid[:, None, None] + np.arange(n_v)[None, :, None]), 0
    ) @ pmf_B)

    d01vals = np.arange(d01_max + 1)
    f_next = np.zeros((n_u, n_v))
    ystar  = np.zeros((N, n_u, n_v), dtype=np.int32)
    s_tab  = np.zeros((N, n_v), dtype=np.int32)
    S_tab  = np.zeros((N, n_v), dtype=np.int32)
    f_first: np.ndarray | None = None

    for i in range(N - 1, -1, -1):
        # E_{d01,d2}[f_{i+1}(y - d01 - d2, d2)]: next vhat IS the realized d2
        ef = np.zeros(n_u)
        for d2 in range(d2_max + 1):
            # with no vhat coordinate every successor lands in the single
            # level 0; with one, it lands in the level the far draw realized
            nxt = d2 if n_vhat else 0
            ef += pmf2[d2] * (
                f_next[uidx(u_grid[:, None] - d01vals[None, :] - d2), nxt] @ pmf01
            )
        G = eh[:, None] + pen + alpha * ef[:, None]           # (y, vhat)

        f_i  = np.empty((n_u, n_v))
        y_i  = np.empty((n_u, n_v), dtype=np.int32)
        for k, u in enumerate(u_grid):
            hi = min(u + scenario.order_max, U_MAX)           # mirrors the action space
            stay = G[k]
            if hi > u:
                win   = G[k + 1: k + 1 + (hi - u)]            # y in (u .. hi]
                jbest = win.argmin(axis=0)
                order = K + win[jbest, np.arange(n_v)]
                take  = order < stay - 1e-12                  # strict: ties keep z=0
                f_i[k] = np.where(take, order, stay)
                y_i[k] = np.where(take, u + 1 + jbest, u)
            else:
                f_i[k], y_i[k] = stay, u
        ystar[i] = y_i
        f_next   = f_i
        if i == 0:
            f_first = f_i

        # (s, S) extraction for reporting: s = largest u that still orders
        for v in range(n_v):
            orders = np.nonzero(y_i[:, v] > u_grid)[0]
            if len(orders):
                s_tab[i, v] = u_grid[orders[-1]]
                S_tab[i, v] = y_i[orders[-1], v]
            else:
                s_tab[i, v] = U_MIN - 1
                S_tab[i, v] = U_MIN - 1

    assert f_first is not None
    return {
        "value": float(f_first[uidx(np.array([0]))[0], 0]),
        "ystar": ystar,
        "u_min": U_MIN,
        # the clip bound the eval applies to vhat, so it must be the TABLE's
        # last vhat index, not the far draw's support. They coincide at
        # dim(V-hat)=1 and differ at 0, where the table has a single level
        "d_max": n_v - 1,
        "s_tab": s_tab,
        "S_tab": S_tab,
    }


def build_arg_parser(description: str, default_scenario: str, arm: str) -> argparse.ArgumentParser:
    """Shared CLI for both arms; `arm` only names the default output dir."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("-s", "--scenario_name", type=str, default=default_scenario,
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--outdir", type=str, default=None,
                   help=f"Output dir (defaults to results/<scenario>/{arm}/)")
    return p


def run_arm(arm: str, args: argparse.Namespace, *, homogeneous: bool) -> dict:
    """Solve, write the arm's artifacts, and report.

    `homogeneous` is the branch this arm claims, and it is asserted rather than
    inferred: running the `exact` arm on a heterogeneous instance would report a
    relaxation bound under a label that promises the optimum, which is exactly
    the confusion §9.9's roles exist to prevent.
    """
    scenario = SCENARIOS[args.scenario_name]
    print(scenario)

    if homogeneous and scenario.alloc_enabled:
        raise SystemExit(
            f"{args.scenario_name!r} is a heterogeneous instance (alloc_enabled=true): "
            f"the recursion is the AP RELAXATION there, not the exact optimum. "
            f"Run adi_flex_benchmark_ap.py instead."
        )
    if not homogeneous and not scenario.alloc_enabled:
        raise SystemExit(
            f"{args.scenario_name!r} is a homogeneous instance (alloc_enabled=false): "
            f"the relaxation is tight there, so it is the EXACT optimum and belongs to "
            f"the dp arm. Run adi_flex_benchmark_dp.py instead."
        )

    sol = solve_dp(scenario)
    N = scenario.N

    outdir = (
        Path(args.outdir) if args.outdir
        else Path(__file__).resolve().parent / "results" / args.scenario_name / arm
    )
    outdir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        outdir / f"{args.scenario_name}_policy.npz",
        ystar=sol["ystar"], u_min=sol["u_min"], d_max=sol["d_max"],
    )

    label = "exact DP" if homogeneous else "AP relaxation (LOWER bound)"
    s_tab, S_tab = sol["s_tab"], sol["S_tab"]
    with open(outdir / f"{args.scenario_name}.txt", "w") as f:
        f.write(f"# {label}, scenario={args.scenario_name}, C_1(0,0)={sol['value']:.6f}\n")
        f.write("i\tvhat\ts\tS\n")
        for i in range(N):
            for v in range(sol["d_max"] + 1):
                f.write(f"{i}\t{v}\t{s_tab[i, v]}\t{S_tab[i, v]}\n")

    meaning = ("expected optimal cost, empty start" if homogeneous
               else "LOWER BOUND on optimal cost, empty start - not attainable")
    print(f"\nC_1(u=0, vhat=0) = {sol['value']:.4f}   ({meaning})")

    v_show = min(15, sol["d_max"])
    print(f"\nperiod-0 policy (vhat = 0..{v_show}):")
    print("  vhat:", "  ".join(f"{v:4d}" for v in range(v_show + 1)))
    print("  S   :", "  ".join(f"{S_tab[0, v]:4d}" for v in range(v_show + 1)))
    print("  s   :", "  ".join(f"{s_tab[0, v]:4d}" for v in range(v_show + 1)))

    s0, S0 = s_tab[0, : v_show + 1], S_tab[0, : v_show + 1]
    print("\nstructural checks (paper Prop 3, period 0):")
    print(f"  S(vhat) independent of vhat : {'OK' if len(set(S0.tolist())) == 1 else 'VIOLATED'}")
    print(f"  s(vhat) decreasing in vhat  : {'OK' if all(s0[j] >= s0[j+1] for j in range(len(s0)-1)) else 'VIOLATED'}")
    print(f"\nsolutions saved -> {outdir}")
    return sol

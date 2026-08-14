"""Paper-exact benchmark solver: the Proposition 2 recursion (spec §9).

Implements the optimal multiordering policy of

    Wang, Atasu & Kurtuluş (2012), "A Multiordering Newsvendor Model with
    Dynamic Forecast Evolution", MSOM 14(3) 472-484

directly from the paper's own characterization, rather than by solving a value
function numerically as fnv_benchmark_dp.py does. Both produce the same policy;
having two independent derivations of it is the point (spec §1.2's
second-implementation gate) — run --cross-check to compare them.

The algorithm (Proposition 2)
-----------------------------
The optimal base-stock level in period n is

    a-MMFE:  S_n(I_n) = mu + I_n + b_n
    m-MMFE:  S_n(I_n) = exp(mu + I_n + b_n)

where the safety-stock terms b_n are **independent of the information state and
of the forecast evolution**, and solve a set of one-dimensional recursive
equations that can be computed off-line — which is what makes the policy cheap
to evaluate. With y = x - mu - I (a-MMFE) or log x - mu - I (m-MMFE), b_n is the
root of g_n(b_n) = 0, where

    g_N(y) = r * Phibar(y / sigma_{N+1}) - c_N                             (7)

    g_n(y) = int_{-inf}^{(y - b_{n+1})/sigma_{n+1}}
                 g_{n+1}(y - sigma_{n+1} * zeta) dPhi(zeta) + c_{n+1} - c_n (6)

Each g_n decreases monotonically from c_{n+1} - c_n to -c_n, so the root exists
and is unique. Equations (6) and (7) are stated identically for both MMFE
modes, so **b_n does not depend on the mode** — only the mapping from b_n to
the stock level does. `--cross-check` asserts that.

Index mapping to this codebase: the paper's sigma_{n+1} (the std of the forecast
adjustment revealed between the period-n and period-(n+1) decisions) is this
domain's `scenario.signal.stdevs[n]`; in particular the paper's sigma_{N+1} is
`stdevs[N]`, the residual uncertainty the last order faces.

Output is the same offsets TSV that fnv_benchmark_dp_eval.py consumes, so the
paper's policy is evaluated by the existing benchmark eval unchanged.

Example usage:
    python fnv_benchmark_prop2.py -s FNV-aMMFE --cross-check
    python fnv_benchmark_prop2.py -s simple
    python fnv_benchmark_dp_eval.py --dp-solutions results/FNV-aMMFE/prop2/FNV-aMMFE.txt -s FNV-aMMFE
"""

import argparse
import math
from pathlib import Path

import numpy as np

from fnv_benchmark_myopic import myopic_offsets
from fnv_grids import GRIDS
from fnv_scenarios import FnvScenario, SCENARIOS

# Deliberately self-contained: a second implementation that imports the first's
# numerics is not an independent check. Only stdlib erf is shared.
_erf = np.vectorize(math.erf)

# Standard normal tail truncation for the quadrature; beyond 8 sigma the mass is
# ~1e-15 and every integrand here is bounded.
Z_MAX = 8.0


def _Phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _Phi_bar(x):
    return 1.0 - _Phi(x)


def _phi(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi_inv(p: float) -> float:
    """Inverse standard normal CDF by bisection."""
    assert 0.0 < p < 1.0, f"p must be in (0,1), got {p}."
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(_Phi(mid)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# g_n, tabulated
# ---------------------------------------------------------------------------

class _G:
    """A tabulated g_n with its exact asymptotes.

    g_n is BOUNDED — it runs monotonically from `left` = c_{n+1} - c_n down to
    `right` = -c_n — so evaluating outside the tabulated range is not
    extrapolation guesswork: the asymptotes are the analytic limits and clamping
    to them is exact in the tail.
    """

    def __init__(self, y_grid: np.ndarray, values: np.ndarray, left: float, right: float):
        self.y_grid = y_grid
        self.values = values
        self.left = left
        self.right = right

    def __call__(self, y):
        y = np.asarray(y, dtype=float)
        out = np.interp(y, self.y_grid, self.values)
        out = np.where(y < self.y_grid[0], self.left, out)
        out = np.where(y > self.y_grid[-1], self.right, out)
        return out

    def root(self) -> float:
        """The unique b with g(b) = 0, by bisection on the interpolant."""
        lo, hi = float(self.y_grid[0]), float(self.y_grid[-1])
        g_lo, g_hi = float(self(lo)), float(self(hi))
        if g_lo < 0.0 or g_hi > 0.0:
            raise SystemExit(
                f"g does not bracket a root on [{lo:.4f}, {hi:.4f}] "
                f"(g_lo={g_lo:.6f}, g_hi={g_hi:.6f}) — widen --span.")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(self(mid)) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def _g_last(scenario: FnvScenario, y_grid: np.ndarray) -> _G:
    """Equation (7): g_N(y) = r * Phibar(y / sigma_{N+1}) - c_N."""
    N       = scenario.N
    r       = scenario.r
    c_N     = scenario.ordering_cost_rate(N)
    sigma   = scenario.signal.stdevs[N]           # paper's sigma_{N+1}
    if sigma <= 0.0:
        values = np.where(y_grid < 0.0, r - c_N, -c_N)
    else:
        values = r * _Phi_bar(y_grid / sigma) - c_N
    return _G(y_grid, values, left=r - c_N, right=-c_N)


def _g_step(
    scenario: FnvScenario, n: int, g_next: _G, b_next: float,
    y_grid: np.ndarray, n_quad: int,
) -> _G:
    """Equation (6), one step back from n+1 to n.

    g_n(y) = int_{-inf}^{u(y)} g_{n+1}(y - sigma * zeta) dPhi(zeta) + c_{n+1} - c_n
    with u(y) = (y - b_{n+1}) / sigma.

    The upper limit moves with y, so this is a truncated expectation rather than
    a plain one: Gauss-Legendre over [-Z_MAX, u(y)] against the normal density,
    not Gauss-Hermite.
    """
    c_n     = scenario.ordering_cost_rate(n)
    c_next  = scenario.ordering_cost_rate(n + 1)
    sigma   = scenario.signal.stdevs[n]            # paper's sigma_{n+1}
    drift   = c_next - c_n

    if sigma <= 0.0:
        # degenerate: no adjustment revealed, the indicator collapses to y >= b_next
        values = np.where(y_grid >= b_next, np.asarray(g_next(y_grid)), 0.0) + drift
        return _G(y_grid, values, left=drift, right=-c_n)

    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    upper = (y_grid - b_next) / sigma                       # (M,)
    upper = np.minimum(upper, Z_MAX)
    lower = -Z_MAX

    # map the Legendre nodes onto [lower, upper(y)] per grid point
    half   = (upper - lower) / 2.0                          # (M,)
    mid    = (upper + lower) / 2.0                          # (M,)
    zeta   = mid[:, None] + half[:, None] * nodes[None, :]  # (M, n_quad)
    integrand = np.asarray(g_next(y_grid[:, None] - sigma * zeta)) * _phi(zeta)
    integral  = (integrand * weights[None, :]).sum(axis=1) * half
    # below the truncation point there is no mass worth integrating
    integral = np.where(upper <= lower, 0.0, integral)

    return _G(y_grid, integral + drift, left=drift, right=-c_n)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

def solve_offsets(
    scenario: FnvScenario,
    n_grid: int = 4001,
    n_quad: int = 96,
    span: float = 10.0,
) -> tuple[float, ...]:
    """Return (b_1, ..., b_N) by the Proposition 2 recursion.

    Mode-independent by construction: equations (6) and (7) never reference the
    MMFE mode, which is why one recursion serves both branches.
    """
    N     = scenario.N
    total = max(float(scenario.stdev), 1e-9)
    y_grid = np.linspace(-span * total, span * total, n_grid)

    g = _g_last(scenario, y_grid)
    b = [0.0] * (N + 1)
    b[N] = g.root()

    for n in range(N - 1, 0, -1):
        g = _g_step(scenario, n, g, b[n + 1], y_grid, n_quad)
        b[n] = g.root()

    return tuple(b[1:])


# ---------------------------------------------------------------------------
# Gates: the paper's own invariants, plus the second-implementation check
# ---------------------------------------------------------------------------

def check_invariants(cells: list[tuple[str, FnvScenario]], n_grid: int, n_quad: int) -> None:
    """Corollary 1 (b_n <= b_hat_n) and mode-independence of b_n."""
    sample = cells[:: max(1, len(cells) // 25)]
    worst_corollary = -math.inf
    worst_mode = 0.0

    for _, sc in sample:
        b = solve_offsets(sc, n_grid=n_grid, n_quad=n_quad)
        worst_corollary = max(worst_corollary,
                              max(bi - bh for bi, bh in zip(b, myopic_offsets(sc))))
        # same scenario, opposite MMFE mode: (6)-(7) never see the mode
        flipped = FnvScenario(
            stdev=sc.stdev, T=sc.T, lamb=sc.lamb, N=sc.N, r=sc.r, c1=sc.c1, mu=sc.mu,
            mmfe_mode=("multiplicative" if sc.mmfe_mode == "additive" else "additive"),
            seed_salt=sc.seed_salt,
        )
        b_flip = solve_offsets(flipped, n_grid=n_grid, n_quad=n_quad)
        worst_mode = max(worst_mode, max(abs(x - y) for x, y in zip(b, b_flip)))

    tol = 20.0 * span_step(sample, n_grid)
    print(f"Corollary 1 (b_n <= b_hat_n): worst margin {worst_corollary:+.3e} "
          f"over {len(sample)} cells")
    if worst_corollary > tol:
        raise SystemExit(
            f"COROLLARY 1 VIOLATED by {worst_corollary:.3e}: a computed safety "
            f"stock exceeds the myopic one, which the paper proves impossible.")
    print(f"mode-independence of b_n: worst |Δ| {worst_mode:.3e} (eq 6-7 do not "
          f"reference the MMFE mode)")
    if worst_mode > tol:
        raise SystemExit(
            f"MODE-INDEPENDENCE VIOLATED by {worst_mode:.3e}: the recursion "
            f"should not depend on the MMFE mode.")
    print("paper invariants PASSED\n")


def span_step(cells: list[tuple[str, FnvScenario]], n_grid: int, span: float = 10.0) -> float:
    return 2.0 * span * max(sc.stdev for _, sc in cells) / (n_grid - 1)


def cross_check(cells: list[tuple[str, FnvScenario]], n_grid: int, n_quad: int) -> None:
    """Spec §1.2 second-implementation gate, against the value-function solver.

    fnv_benchmark_dp.py maximizes a reduced value function on a grid; this file
    solves the paper's first-order conditions for the thresholds. Independent
    routes to one policy — they must land in the same place.
    """
    from fnv_benchmark_dp import solve_offsets as dp_solve

    sample = cells[:: max(1, len(cells) // 25)]
    worst, worst_cell = 0.0, None
    for cid, sc in sample:
        a = np.array(solve_offsets(sc, n_grid=n_grid, n_quad=n_quad))
        d = np.array(dp_solve(sc))
        gap = float(np.max(np.abs(a - d)))
        if gap > worst:
            worst, worst_cell = gap, cid

    tol = 20.0 * span_step(sample, n_grid)
    print(f"second-implementation gate (§1.2): worst |b_prop2 - b_dp| = "
          f"{worst:.3e} at cell {worst_cell} over {len(sample)} cells "
          f"(tolerance {tol:.3e})")
    if worst > tol:
        raise SystemExit(
            f"CROSS-CHECK FAILED: the paper recursion and the value-function DP "
            f"disagree by {worst:.3e}. One of them is wrong; do not ship either "
            f"until it is resolved.")
    print("cross-check PASSED — two independent derivations agree.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Solve FNV offsets by the paper's Proposition 2 recursion.")
    p.add_argument("-s", "--scenario_name", type=str, default="FNV-aMMFE",
                   choices=list(SCENARIOS) + list(GRIDS),
                   help="Scenario (§5.4) or grid (§5.6); a grid solves every cell")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV of offsets (defaults to results/<scenario>/prop2/<scenario>.txt)")
    p.add_argument("--n-grid", type=int, default=4001)
    p.add_argument("--n-quad", type=int, default=96)
    p.add_argument("--check-invariants", action="store_true",
                   help="Check Corollary 1 and mode-independence before solving")
    p.add_argument("--cross-check", action="store_true",
                   help="Compare against fnv_benchmark_dp.py (§1.2 gate)")
    return p


def resolve_cells(scenario_name: str) -> list[tuple[str, FnvScenario]]:
    if scenario_name in SCENARIOS:
        return [(scenario_name, SCENARIOS[scenario_name])]
    if scenario_name in GRIDS:
        return list(GRIDS[scenario_name])
    raise SystemExit(f"unknown scenario/grid {scenario_name!r}")


def main() -> None:
    args  = _build_arg_parser().parse_args()
    cells = resolve_cells(args.scenario_name)

    # §8.4: benchmark solution tables live at
    # results/{scenario}/benchmark/{method}/{scenario}.txt
    outfile = (Path(args.outfile) if args.outfile else
               Path(__file__).resolve().parent / "results" / args.scenario_name
               / "benchmark" / "prop2" / f"{args.scenario_name}.txt")

    print(f"scenario  {args.scenario_name}  ({len(cells)} cell(s))")
    print(f"output    {outfile}")
    print("method    Wang, Atasu & Kurtuluş (2012) Proposition 2, eq (6)-(7)\n")

    if args.check_invariants:
        check_invariants(cells, args.n_grid, args.n_quad)
    if args.cross_check:
        cross_check(cells, args.n_grid, args.n_quad)

    N = cells[0][1].N
    header = "stdev\tT\tlambda\t" + "\t".join(f"b{n}" for n in range(1, N + 1))
    print(header)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()
        for _, sc in cells:
            b = solve_offsets(sc, n_grid=args.n_grid, n_quad=args.n_quad)
            row = (f"{sc.stdev}\t{sc.T}\t{sc.lamb}\t"
                   + "\t".join(f"{x:.6f}" for x in b))
            print(row, flush=True)
            f.write(row + "\n")
            f.flush()

    print(f"\noffsets saved → {outfile}")
    print(f"next: python fnv_benchmark_dp_eval.py --dp-solutions {outfile} "
          f"-s {args.scenario_name}")


if __name__ == "__main__":
    main()

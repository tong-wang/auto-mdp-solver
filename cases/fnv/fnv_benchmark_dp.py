"""DP benchmark solver for FNV: optimal base-stock offsets b_1..b_N (spec §9).

Computes the optimal ordering policy by backward recursion and writes one row
per scenario/grid cell, in the format fnv_benchmark_dp_eval.py consumes.

The 2-D state (inventory x, cumulative information I) collapses to 1-D
------------------------------------------------------------------------
Demand depends on the information only through its terminal value, so shifting
inventory and information together leaves the problem unchanged. Work in the
**reduced coordinate**

    additive:        z = x - (mu + I_n)          D = mu + I_{N+1}
    multiplicative:  z = x / exp(mu + I_n)       D = exp(mu + I_{N+1})

Writing the post-order level as u (so the order is q = u - z in reduced units),
the value function separates as V_n(x, I) = A_n(I) + W_n(z) for the additive
case and V_n(x, I) = exp(mu + I) * W_n(z) for the multiplicative one, with

    W_N(z) = c_N z + max_{u >= z} [ -c_N u + r E min(eps_N, u) ]          (add)
    W_n(z) = c_n z + max_{u >= z} [ -c_n u + E W_{n+1}(u - eps_n) ]       (add)
    W_N(z) = c_N z + max_{u >= z} [ -c_N u + r E min(e^eps_N, u) ]        (mul)
    W_n(z) = c_n z + max_{u >= z} [ -c_n u + E e^eps_n W_{n+1}(u e^-eps_n) ]

because E[eps_n] = 0 kills the cross term. The unconstrained maximizer of each
bracket is the offset: order up to `mu + I + b_n` (additive) or
`exp(mu + I + b_n)` (multiplicative), which is exactly the policy the eval
script runs. b_n is a pure function of the residual uncertainty and the cost
ladder — it does not depend on x or I, which is why one number per period is a
complete policy.

Instrument validation (spec §14): the last period is an exact newsvendor with a
closed form, b_N = sigma_N * Phi^-1(1 - c_N/r) in both modes. --self-test checks
the numerical recursion against it before any solution is trusted.

Example usage:
    python fnv_benchmark_dp.py -s FNV-aMMFE
    python fnv_benchmark_dp.py -s simple --self-test
"""

import argparse
import math
from pathlib import Path

import numpy as np

from fnv_grids import GRIDS
from fnv_scenarios import FnvScenario, SCENARIOS

# Vectorized standard normal, stdlib-only: scipy is installed in this venv but
# is not a declared dependency, and the DP is small enough not to need it.
_erf = np.vectorize(math.erf)


def _Phi(x: np.ndarray | float) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _phi(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation that EXTRAPOLATES rather than clamping.

    This is load-bearing, not a nicety. Below the maximizer W_n(z) = c_n z +
    const exactly, so a clamped lookup reports a value that is too high for
    small z; the inflated continuation then drags the next period's argmax onto
    the grid's left edge and the solver silently emits "never order" offsets.
    Linear extrapolation is exact in that region and asymptotically right above
    the grid, where W flattens.
    """
    y = np.interp(x, xp, fp)
    lo = (fp[1] - fp[0]) / (xp[1] - xp[0])
    hi = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    y = np.where(x < xp[0],  fp[0]  + lo * (x - xp[0]),  y)
    y = np.where(x > xp[-1], fp[-1] + hi * (x - xp[-1]), y)
    return y


def _Phi_inv(p: float) -> float:
    """Inverse standard normal CDF by bisection — used once per solve."""
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
# Terminal-period expectations (exact — no quadrature at the kink)
# ---------------------------------------------------------------------------

def _E_min_normal(u: np.ndarray, sigma: float) -> np.ndarray:
    """E[min(eps, u)] for eps ~ N(0, sigma^2)."""
    u = np.asarray(u, dtype=float)
    if sigma <= 0.0:
        return np.minimum(0.0, u)
    z = u / sigma
    return u * (1.0 - _Phi(z)) - sigma * _phi(z)


def _E_min_lognormal(u: np.ndarray, sigma: float) -> np.ndarray:
    """E[min(exp(eps), u)] for eps ~ N(0, sigma^2)."""
    u = np.asarray(u, dtype=float)
    if sigma <= 0.0:
        return np.minimum(1.0, u)
    out = np.zeros_like(u)
    pos = u > 0.0
    lu  = np.log(np.where(pos, u, 1.0))
    # E[e^eps 1{eps < ln u}] + u P(eps >= ln u)
    out[pos] = (
        math.exp(0.5 * sigma * sigma) * _Phi((lu[pos] - sigma * sigma) / sigma)
        + u[pos] * (1.0 - _Phi(lu[pos] / sigma))
    )
    return out


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

def solve_offsets(
    scenario: FnvScenario,
    n_grid: int = 4001,
    n_quad: int = 121,
    span: float = 8.0,
) -> tuple[float, ...]:
    """Return the optimal offsets (b_1, ..., b_N) for one scenario."""
    return _solve(scenario, n_grid, n_quad, span)[0]


def expected_profit(
    scenario: FnvScenario,
    n_grid: int = 4001,
    n_quad: int = 121,
    span: float = 8.0,
) -> float:
    """The DP's own prediction of the optimal expected profit.

    A second reading of the same recursion, from the initial state (x = 0,
    I = 0): a simulation of the fitted offsets must reproduce it to within
    Monte-Carlo error, or the recursion and the policy it emits disagree.
    """
    return _solve(scenario, n_grid, n_quad, span)[1]


def _argmax_interior(bracket: np.ndarray, period: int) -> int:
    """argmax of the bracket, refusing a maximizer pinned to a grid edge.

    An edge maximizer means the grid does not contain the optimum, so the
    offset it reports is an artifact of where the grid was cut — the failure
    that produced silent "never order" offsets before the extrapolating
    interpolation went in. Fail loudly rather than emit a plausible number.
    """
    k = int(np.argmax(bracket))
    if k in (0, len(bracket) - 1):
        raise SystemExit(
            f"DP grid too narrow: the period-{period} maximizer sits on the "
            f"{'left' if k == 0 else 'right'} grid edge. Widen --span.")
    return k


def _solve(
    scenario: FnvScenario,
    n_grid: int = 4001,
    n_quad: int = 121,
    span: float = 8.0,
) -> tuple[tuple[float, ...], float]:
    """Backward recursion; returns (offsets, predicted expected profit).

    n_grid  points on the reduced post-order level u
    n_quad  Gauss-Hermite nodes for the inter-period information increments
    span    grid half-width in units of the total demand sigma
    """
    N     = scenario.N
    r     = scenario.r
    sigma = scenario.signal.stdevs      # sigma[n] is revealed at period n
    additive = scenario.mmfe_mode == "additive"

    # u lives on the same scale as the demand uncertainty (additive) or its
    # exponential (multiplicative); span it generously so the unconstrained
    # maximizer is interior.
    # The grid must cover both the maximizers (O(sigma) around 0) and the
    # initial reduced state z_1 = -mu (additive) / 0 (multiplicative), so the
    # value readout does not fall off the left end.
    total_sigma = max(float(scenario.stdev), 1e-9)
    if additive:
        lo = min(-span * total_sigma, -scenario.mu - 2.0 * total_sigma)
        u_grid = np.linspace(lo, span * total_sigma, n_grid)
    else:
        u_grid = np.exp(np.linspace(-span * total_sigma, span * total_sigma, n_grid))

    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    weights = weights / math.sqrt(math.pi)          # normalize to a probability
    b: list[float] = [0.0] * (N + 1)                # 1-indexed

    # --- period N: exact newsvendor -------------------------------------
    c_N = scenario.ordering_cost_rate(N)
    if additive:
        bracket = -c_N * u_grid + r * _E_min_normal(u_grid, sigma[N])
    else:
        bracket = -c_N * u_grid + r * _E_min_lognormal(u_grid, sigma[N])
    k = _argmax_interior(bracket, N)
    b[N] = float(u_grid[k]) if additive else float(math.log(max(u_grid[k], 1e-300)))

    # W_N(z) = c_N z + max_{u >= z} bracket(u); the reverse cumulative max makes
    # no concavity assumption.
    sup_from = np.maximum.accumulate(bracket[::-1])[::-1]
    W_next   = c_N * u_grid + sup_from            # evaluated on z = u_grid

    # --- periods N-1 .. 1 ------------------------------------------------
    for n in range(N - 1, 0, -1):
        c_n = scenario.ordering_cost_rate(n)
        s   = sigma[n]
        eps = math.sqrt(2.0) * s * nodes          # Gauss-Hermite change of var

        if additive:
            # E[W_{n+1}(u - eps)]
            z_of = u_grid[:, None] - eps[None, :]
            cont = _interp(z_of, u_grid, W_next) @ weights
        else:
            # E[e^eps W_{n+1}(u e^-eps)]
            z_of = u_grid[:, None] * np.exp(-eps)[None, :]
            cont = (_interp(z_of, u_grid, W_next) * np.exp(eps)[None, :]) @ weights

        bracket = -c_n * u_grid + cont
        k = _argmax_interior(bracket, n)
        b[n] = float(u_grid[k]) if additive else float(math.log(max(u_grid[k], 1e-300)))

        sup_from = np.maximum.accumulate(bracket[::-1])[::-1]
        W_next   = c_n * u_grid + sup_from

    # Value at the initial state: x = 0, I = 0 (init_state), so the reduced
    # coordinate is z_1 = -mu (additive) or z_1 = 0 (multiplicative), and the
    # separated constant is r*mu resp. exp(mu).
    if additive:
        z1    = -scenario.mu
        value = r * scenario.mu + float(np.interp(z1, u_grid, W_next))
    else:
        z1    = 0.0
        value = math.exp(scenario.mu) * float(np.interp(z1, u_grid, W_next))

    return tuple(b[1:]), value


def closed_form_last_offset(scenario: FnvScenario) -> float:
    """b_N = sigma_N * Phi^-1(1 - c_N/r) — exact in both MMFE modes.

    The last order faces a plain newsvendor: buy another unit while the chance
    it sells, P(eps_N > u), exceeds the cost ratio c_N/r.
    """
    c_N = scenario.ordering_cost_rate(scenario.N)
    ratio = c_N / scenario.r
    if ratio >= 1.0:
        return -math.inf                      # never worth ordering
    return scenario.signal.stdevs[scenario.N] * _Phi_inv(1.0 - ratio)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Solve FNV base-stock offsets by backward recursion.")
    p.add_argument("-s", "--scenario_name", type=str, default="FNV-aMMFE",
                   choices=list(SCENARIOS) + list(GRIDS),
                   help="Scenario (§5.4) or grid (§5.6); a grid solves every cell")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV of offsets (defaults to results/<scenario>/dp/<scenario>.txt)")
    p.add_argument("--n-grid", type=int, default=4001)
    p.add_argument("--n-quad", type=int, default=121)
    p.add_argument("--self-test", action="store_true",
                   help="Check the recursion's last period against the closed "
                        "form before solving (§14 instrument validation)")
    return p


def resolve_cells(scenario_name: str) -> list[tuple[str, FnvScenario]]:
    if scenario_name in SCENARIOS:
        return [(scenario_name, SCENARIOS[scenario_name])]
    if scenario_name in GRIDS:
        return list(GRIDS[scenario_name])
    raise SystemExit(f"unknown scenario/grid {scenario_name!r}")


def run_self_test(cells: list[tuple[str, FnvScenario]], n_grid: int, n_quad: int) -> None:
    """The recursion must recover the closed form where one exists."""
    worst = 0.0
    checked = cells[:: max(1, len(cells) // 25)]
    for _, sc in checked:
        b = solve_offsets(sc, n_grid=n_grid, n_quad=n_quad)
        worst = max(worst, abs(b[sc.N - 1] - closed_form_last_offset(sc)))
    tol = 8.0 * max(c.stdev for _, c in checked) / n_grid * 10
    print(f"self-test: |b_N(numeric) - b_N(closed form)| <= {worst:.3e} "
          f"over {len(checked)} cells (grid step tolerance {tol:.3e})")
    if worst > tol:
        raise SystemExit(
            f"SELF-TEST FAILED: last-period offset off by {worst:.3e} — the "
            f"recursion does not reproduce the exact newsvendor, so its earlier "
            f"periods cannot be trusted either.")
    print("self-test PASSED — the instrument recovers a known answer.\n")


def main() -> None:
    args  = parse_args = _build_arg_parser().parse_args()
    cells = resolve_cells(args.scenario_name)

    # §8.4: benchmark solution tables live at
    # results/{scenario}/benchmark/{method}/{scenario}.txt
    outfile = (Path(args.outfile) if args.outfile else
               Path(__file__).resolve().parent / "results" / args.scenario_name
               / "benchmark" / "dp" / f"{args.scenario_name}.txt")

    print(f"scenario  {args.scenario_name}  ({len(cells)} cell(s))")
    print(f"output    {outfile}")

    if args.self_test:
        run_self_test(cells, args.n_grid, args.n_quad)

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

"""#E32 robustness grid: does the tuned two-constant rule survive off its
tuning cell? (`mab/ROBUSTNESS_PLAN.md`)

Subject: `argmax_i(pm_i + 2.5*(ttg/T)^0.15 * psd_i)` (#E26, 1486.14 +/- 6.47 at
K=10, T=1000, sigma=1) — two constants, both fitted at that one cell.

Parts (results accumulate under results/{base scenario}/grid_probe/):

  sweep   every policy on every cell, report block; per-episode arrays kept
  tune    per-cell (c*, p*) fit on a DISJOINT block, then scored on the report
          block — so the reported number is never selected on its own set
  report  aggregate the JSONs into the cell tables + the B-collapse view

The grid deliberately does NOT reproduce the domain's seed scheme. We measure
policy behavior on fresh instances rather than reproducing a shipped artifact,
so payouts come from a vectorized per-step draw (~1000x faster than one
SeedSequence per episode per period). What IS load-bearing is common random
numbers, and they are stronger here than the domain gives:

  * arm means are drawn once at KMAX and sliced to K, so a K=5 cell's means are
    a prefix of a K=20 cell's — exactly the domain's own behavior (one iid draw
    of `size` components off one meta-keyed rng);
  * per-period noise is drawn at KMAX and sliced, keyed [stream, K, t, block],
    so a short-horizon cell's noise is a PREFIX of the long-horizon cell's at
    the same K, and the sigma cells share z with their sigma=1 partners.

So every comparison — across policies, along the n ladder, and between a sigma
cell and its budget-matched partner — is paired. `--part sweep` gates itself at
the tuned cell against the recorded 8192-seed numbers (distributional
agreement, not bit-exactness: the seed scheme differs by design).

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_grid_probe.py --part sweep
    OMP_NUM_THREADS=1 python mab_grid_probe.py --part tune
    OMP_NUM_THREADS=1 python mab_grid_probe.py --part report
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

SCENARIO = "gauss_K10_T1000"                     # the base cell, for the outdir
OUTDIR = Path(__file__).resolve().parent / "results" / SCENARIO / "grid_probe"

# seed-stream ids and blocks (blocks are disjoint by construction: the block id
# enters every key, so tuning never sees the reporting draws)
_SALT = 90325
_MEANS_STREAM, _NOISE_STREAM = 11, 12
BLOCK_REPORT, BLOCK_TUNE = 0, 1

N_REPORT = 8192                                  # evidence-grade (spec 9.7)
N_TUNE = 2048

PRIOR_MEAN, PRIOR_SD = 0.0, 1.0

# --- the grid (mirrors mab_scenarios._K_LADDER / _N_LADDER / _SIGMA_CELLS) ---
K_LADDER = (5, 10, 20)
N_LADDER = (2, 5, 10, 20, 50, 100)               # pulls per arm; T = K * n
SIGMA_CELLS = ((10, 200, 2.0), (10, 400, 2.0), (10, 320, 4.0))   # (K, T, sigma)
KMAX = max(K_LADDER)

# recorded 8192-seed numbers at the tuned cell (#E26 / #E31 rung 0), the gate
GATE_TUNED = {"thompson": 1463.19, "rule_frozen": 1486.14}
GATE_TOL = 20.0            # ~3x the +/-6.5 eval SE; the seed scheme differs


#: #E32 round 2 — the long-horizon extension. Theory says a FIXED-quantile
#: bonus is not asymptotically consistent: it never grows with t, so with
#: constant probability it abandons the best arm forever and that tail costs
#: regret linear in T, while thompson's grows like log T. There must therefore
#: be a crossover. The core ladder already shows its onset — the share of
#: thompson's gap that the rule closes peaks and TURNS at K=20 (33.5% at n=50
#: -> 31.5% at n=100) while K=5 is still climbing — so n* should fall as K
#: rises, and this ladder is where the turn becomes a reversal.
EXT_N_LADDER = (200, 500, 1000, 2000)


def cells(ladder: str = "core") -> list[dict]:
    """Canonical cell order: the K x n ladder row-major, then the sigma cells."""
    out = []
    ns = N_LADDER if ladder == "core" else EXT_N_LADDER
    for k in K_LADDER:
        for n in ns:
            out.append({"name": f"gauss_K{k}_T{k * n}", "K": k, "T": k * n,
                        "sigma": 1.0, "n": n, "B": float(n)})
    if ladder == "core":
        for k, t, sg in SIGMA_CELLS:
            out.append({"name": f"gauss_K{k}_T{t}_s{sg:g}", "K": k, "T": t,
                        "sigma": sg, "n": t / k, "B": (t / k) / sg ** 2})
    return out


CELLS = {c["name"]: c for c in cells("core") + cells("ext")}


# ---------------------------------------------------------------------------
# Generalized simulator
# ---------------------------------------------------------------------------

class GridSim:
    """All episodes of one cell advance in lockstep.

    The conjugate posterior for a Normal prior N(mu0, sd0^2) with known noise
    sd sigma, after n pulls totalling `s`:

        1/psd^2 = 1/sd0^2 + n/sigma^2
        pm      = psd^2 * (mu0/sd0^2 + s/sigma^2)

    which reduces to the domain's `1/(1+n)`, `sum/(1+n)` at mu0=0, sd0=sigma=1
    (asserted by `--part sweep`'s reduction check).
    """

    def __init__(self, cell: dict, n_seeds: int, block: int = BLOCK_REPORT):
        self.K, self.T, self.sigma = cell["K"], cell["T"], float(cell["sigma"])
        self.n_seeds, self.block = n_seeds, block
        # means at KMAX then sliced: a smaller-K cell's means are a prefix
        rng = np.random.default_rng(
            np.random.SeedSequence([_MEANS_STREAM, self.K, block, _SALT]))
        self.means = rng.normal(PRIOR_MEAN, PRIOR_SD, (n_seeds, KMAX))[:, :self.K]
        self.opt = self.means.max(axis=1)
        self.best = self.means.argmax(axis=1)

        self.pulls = np.zeros((n_seeds, self.K))
        self.sums = np.zeros((n_seeds, self.K))
        self.t = 0
        self.total = np.zeros(n_seeds)           # realized payout
        self.mean_total = np.zeros(n_seeds)      # sum of TRUE means pulled
        self._prior_prec = 1.0 / PRIOR_SD ** 2
        self._noise_prec = 1.0 / self.sigma ** 2

    def post(self) -> tuple[np.ndarray, np.ndarray]:
        prec = self._prior_prec + self.pulls * self._noise_prec
        psd = np.sqrt(1.0 / prec)
        pm = (PRIOR_MEAN * self._prior_prec + self.sums * self._noise_prec) / prec
        return pm, psd

    def n_of(self, psd: np.ndarray) -> np.ndarray:
        """Posterior sd -> pull count (the general inverse of `post`)."""
        return np.maximum(
            (1.0 / np.square(psd) - self._prior_prec) / self._noise_prec, 0.0)

    def step(self, arms: np.ndarray) -> None:
        # noise drawn at KMAX and sliced, keyed on (K, t, block): a short cell's
        # stream is a prefix of a long cell's at the same K, and sigma cells
        # share z with their sigma=1 partners
        rng = np.random.default_rng(np.random.SeedSequence(
            [_NOISE_STREAM, self.K, self.t, self.block, _SALT]))
        z = rng.standard_normal((self.n_seeds, KMAX))[:, :self.K]
        idx = np.arange(self.n_seeds)
        mu = self.means[idx, arms]
        pay = mu + self.sigma * z[idx, arms]
        self.pulls[idx, arms] += 1.0
        self.sums[idx, arms] += pay
        self.total += pay
        self.mean_total += mu
        self.t += 1

    # -- outcome accessors ---------------------------------------------------

    def pseudo_regret(self) -> np.ndarray:
        """T*max_i mu_i - sum_t mu_{a_t}: the true-mean regret (lower variance
        than realized, and the statistic that is comparable across cells)."""
        return self.T * self.opt - self.mean_total

    def best_untouched(self) -> np.ndarray:
        return self.pulls[np.arange(self.n_seeds), self.best] == 0.0


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def _norm_ppf(p):
    """Standard normal quantile (Acklam-free: via erfinv)."""
    from scipy.special import erfinv          # noqa: PLC0415 - optional dep
    return math.sqrt(2.0) * erfinv(2.0 * np.asarray(p, float) - 1.0)


def _ppf(p):
    """_norm_ppf without scipy: bisection on the erf-based cdf (scalars only)."""
    try:
        return _norm_ppf(p)
    except ImportError:
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def index_rules(T: int) -> dict:
    """c(ttg, psd) multipliers on the posterior sd — the #E27 comparison set.

    Every rule is `argmax(pm + c(ttg, psd) * psd)`. Multiplying psd (rather
    than sigma*sqrt(./n)) is what makes these scale-correct off the tuned cell:
    psd already carries sigma and the prior.
    """
    return {
        "rule_frozen":  lambda ttg, s, sim: 2.5 * (ttg / T) ** 0.15,
        "ucb1_anytime": lambda ttg, s, sim: np.sqrt(2.0 * np.log(T - ttg + 2.0)),
        "lai_T_over_n": lambda ttg, s, sim: np.sqrt(
            2.0 * np.log(np.maximum(T / np.maximum(sim.n_of(s), 1.0), 1.0))),
        "bayes_ucb":    lambda ttg, s, sim: _ppf(
            1.0 - 1.0 / (max(T - ttg + 1.0, 2.0) * math.log(max(T, 3)))),
        "greedy":       lambda ttg, s, sim: 0.0,
        # --- #E31's insurance variants, applied to the TUNED rule ----------
        # Same semantics as mab_coverage_probe.sweep_rule (max, not additive),
        # there applied to the 12-knot readback. `floor` bounds the bonus below;
        # `log` makes it GROW with t, which is the term that restores
        # asymptotic consistency — so if the crossover is the fixed-quantile
        # failure, log:0.7 should push it out and floor:2.2 should not.
        "rule_floor2.2": lambda ttg, s, sim: max(2.5 * (ttg / T) ** 0.15, 2.2),
        "rule_log0.7":   lambda ttg, s, sim: max(
            2.5 * (ttg / T) ** 0.15,
            0.7 * math.sqrt(2.0 * math.log(max(T - ttg + 1.0, 2.0)))),
    }


def tuned_rule(c: float, p: float, T: int):
    return lambda ttg, s, sim: c * (ttg / T) ** p


def play_index(cell: dict, cfn, n_seeds: int, block: int) -> GridSim:
    sim = GridSim(cell, n_seeds, block)
    for _ in range(cell["T"]):
        pm, psd = sim.post()
        ttg = float(cell["T"] - sim.t)
        sim.step((pm + cfn(ttg, psd, sim) * psd).argmax(axis=1))
    return sim


def play_thompson(cell: dict, n_seeds: int, block: int,
                  rng_seed: int = 7) -> GridSim:
    """Exact conjugate Thompson. Its own free rng stream (the randomization is
    the policy, not the instance); the payout stream stays CRN."""
    sim = GridSim(cell, n_seeds, block)
    rng = np.random.default_rng(rng_seed)
    for _ in range(cell["T"]):
        pm, psd = sim.post()
        sim.step((pm + psd * rng.standard_normal(pm.shape)).argmax(axis=1))
    return sim


def outcome(sim: GridSim) -> dict:
    reg = sim.pseudo_regret()
    return {
        "reward_mean": float(sim.total.mean()),
        "regret_mean": float(reg.mean()),
        "regret_se": float(reg.std(ddof=1) / math.sqrt(len(reg))),
        "regret_p90": float(np.percentile(reg, 90)),
        "regret_p99": float(np.percentile(reg, 99)),
        "regret_norm": float(reg.mean() / math.sqrt(sim.K * sim.T)),
        "p_best_untouched": float(sim.best_untouched().mean()),
        "_regret": reg,
    }


# ---------------------------------------------------------------------------
# Equivalence gates — the licence for a second simulator (README §Simulators)
#
# `GridSim` is a REIMPLEMENTATION, not a replica: it deliberately abandons the
# domain's seed scheme for speed, so it can never be diffed episode-by-episode
# against `mab_mdp`. That buys ~1000x and costs the bit-exactness gate
# `mab_a5_probe.VecSim` enjoys, so the equivalence has to be established
# where it still can be — on the deterministic core, against the domain's own
# functions, and by calibration where the domain has no reference at all.
#
# Both gates run in `mab_test.py`, not only from `--part sweep`.
# ---------------------------------------------------------------------------

def gate_posterior_matches_domain(n_states: int = 64, seed: int = 5) -> dict:
    """GridSim's general posterior vs the DOMAIN's canonical belief.

    `mab_bayes.bayes_post_mean/sd` is what the gym's `bayes` observation is
    built from and what the IR names in `expr_builtins`, so it — not a
    re-derivation — is the thing to agree with.

    This replaces an earlier "reduction check" that asserted the general
    formula against a hard-coded special case of *itself*. That was circular:
    it checked the algebra against the same algebra, and would have passed
    with the identical bug on both sides.
    """
    from mab_bayes import bayes_post_mean, bayes_post_sd

    cell = CELLS["gauss_K10_T1000"]              # the cell the domain implements
    rng = np.random.default_rng(seed)
    sim = GridSim(cell, n_states, BLOCK_REPORT)
    # arbitrary reachable states, including unpulled arms (pulls = 0)
    sim.pulls = rng.integers(0, 50, size=(n_states, sim.K)).astype(float)
    sim.sums = rng.normal(0.0, 5.0, size=(n_states, sim.K))
    pm, psd = sim.post()

    max_pm = max_psd = 0.0
    for i in range(n_states):
        ref_m = np.asarray(bayes_post_mean(sim.pulls[i], sim.sums[i], True))
        ref_s = np.asarray(bayes_post_sd(sim.pulls[i], sim.sums[i], True))
        max_pm = max(max_pm, float(np.abs(pm[i] - ref_m).max()))
        max_psd = max(max_psd, float(np.abs(psd[i] - ref_s).max()))
    assert max_pm < 1e-12, f"posterior mean disagrees with mab_bayes: {max_pm}"
    assert max_psd < 1e-12, f"posterior sd disagrees with mab_bayes: {max_psd}"
    # and the inverse used by the Lai-style rules must round-trip
    inv = float(np.abs(sim.n_of(psd) - sim.pulls).max())
    assert inv < 1e-9, f"n_of does not invert post: {inv}"
    return {"n_states": n_states, "max_pm_err": max_pm,
            "max_psd_err": max_psd, "max_n_of_err": inv}


def gate_posterior_calibration(n_draws: int = 20000, seed: int = 11) -> dict:
    """Off the base cell the domain has NO reference implementation, so check
    the posterior is *calibrated* instead.

    Draw mu from the prior, observe n payouts at noise sd sigma, and the
    standardized residual (mu - pm) / psd must have unit variance. This is
    independent of how pm/psd are computed, so it catches a sigma mis-scaling
    — exactly the error a formula-vs-formula check cannot see, and exactly the
    thing the sigma cells depend on.
    """
    rng = np.random.default_rng(seed)
    out, worst = {}, 0.0
    for n, sigma in ((1, 0.5), (3, 1.0), (10, 2.0), (40, 4.0)):
        mu = rng.normal(PRIOR_MEAN, PRIOR_SD, n_draws)
        # sum of n observations, each mu + sigma * N(0,1)
        sums = n * mu + sigma * np.sqrt(n) * rng.standard_normal(n_draws)
        sim = GridSim({"K": 1, "T": 1, "sigma": sigma}, n_draws, BLOCK_REPORT)
        sim.pulls[:, 0] = float(n)
        sim.sums[:, 0] = sums
        pm, psd = sim.post()
        sd_z = float(((mu - pm[:, 0]) / psd[:, 0]).std())
        out[f"n={n},sigma={sigma:g}"] = sd_z
        worst = max(worst, abs(sd_z - 1.0))
    # sd(z) has SE ~ 1/sqrt(2 n_draws) ~ 0.005 here; 0.03 is ~6 SE
    assert worst < 0.03, f"posterior not calibrated: max |sd(z)-1| = {worst}"
    out["max_dev"] = worst
    return out


# ---------------------------------------------------------------------------
# part: sweep
# ---------------------------------------------------------------------------

def part_sweep(n_seeds: int, only: str | None,
               ladder: str = "core") -> dict:
    g1 = gate_posterior_matches_domain()
    g2 = gate_posterior_calibration()
    print(f"  GATE posterior vs mab_bayes: pm {g1['max_pm_err']:.2e} "
          f"psd {g1['max_psd_err']:.2e}", flush=True)
    print(f"  GATE calibration |sd(z)-1| max {g2['max_dev']:.4f}", flush=True)

    rows: dict[str, dict] = {}
    targets = [only] if only else [c["name"] for c in cells(ladder)]
    for name in targets:
        cell = CELLS[name]
        t0 = time.time()
        res: dict[str, dict] = {}
        ts = play_thompson(cell, n_seeds, BLOCK_REPORT)
        res["thompson"] = outcome(ts)
        ref = res["thompson"]["_regret"]
        for pname, cfn in index_rules(cell["T"]).items():
            res[pname] = outcome(play_index(cell, cfn, n_seeds, BLOCK_REPORT))
        # paired CRN differences vs thompson: NEGATIVE regret diff = better
        for pname, r in res.items():
            d = r.pop("_regret") - ref
            r["paired_vs_thompson_mean"] = float(d.mean())
            r["paired_vs_thompson_se"] = float(d.std(ddof=1) / math.sqrt(len(d)))
        rows[name] = {**{k: cell[k] for k in ("K", "T", "sigma", "n", "B")},
                      "policies": res, "wall_seconds": round(time.time() - t0, 1)}
        d = res["rule_frozen"]
        print(f"  {name:22s} B={cell['B']:6.2f}  rule regret "
              f"{d['regret_mean']:8.2f}  vs thompson "
              f"{d['paired_vs_thompson_mean']:+7.2f} "
              f"(+/-{2 * d['paired_vs_thompson_se']:.2f})  "
              f"unt {d['p_best_untouched']:.4f}  [{rows[name]['wall_seconds']}s]",
              flush=True)

    # gate at the tuned cell against the recorded 8192-seed numbers
    if "gauss_K10_T1000" in rows and n_seeds >= 4096:
        got = rows["gauss_K10_T1000"]["policies"]
        for pname, recorded in GATE_TUNED.items():
            delta = got[pname]["reward_mean"] - recorded
            ok = abs(delta) <= GATE_TOL
            print(f"  GATE {pname:12s} {got[pname]['reward_mean']:8.2f} vs "
                  f"recorded {recorded:8.2f}  delta {delta:+6.2f}  "
                  f"{'OK' if ok else 'FAIL'}")
            assert ok, f"{pname} off the recorded bar by {delta:+.2f}"
    return rows


# ---------------------------------------------------------------------------
# part: tune  (two-stage — fit on BLOCK_TUNE, score on BLOCK_REPORT)
# ---------------------------------------------------------------------------

C_GRID = tuple(round(0.5 + 0.25 * i, 2) for i in range(15))     # 0.50 .. 4.00
P_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


def part_tune(n_tune: int, n_report: int, only: str | None) -> dict:
    rows: dict[str, dict] = {}
    targets = [only] if only else list(CELLS)
    for name in targets:
        cell = CELLS[name]
        t0 = time.time()
        best = None
        surface = []
        for c in C_GRID:
            for p in P_GRID:
                sim = play_index(cell, tuned_rule(c, p, cell["T"]),
                                 n_tune, BLOCK_TUNE)
                r = float(sim.pseudo_regret().mean())
                surface.append({"c": c, "p": p, "regret": r})
                if best is None or r < best[2]:
                    best = (c, p, r)
        c_star, p_star, r_tune = best
        # score the winner on the disjoint reporting block, next to the frozen
        # rule and thompson on the same block
        scored = outcome(play_index(cell, tuned_rule(c_star, p_star, cell["T"]),
                                    n_report, BLOCK_REPORT))
        ts = play_thompson(cell, n_report, BLOCK_REPORT)
        d = scored.pop("_regret") - ts.pseudo_regret()
        scored["paired_vs_thompson_mean"] = float(d.mean())
        scored["paired_vs_thompson_se"] = float(d.std(ddof=1) / math.sqrt(len(d)))
        rows[name] = {**{k: cell[k] for k in ("K", "T", "sigma", "n", "B")},
                      "c_star": c_star, "p_star": p_star,
                      "regret_tune_block": r_tune, "scored": scored,
                      "surface": surface,
                      "wall_seconds": round(time.time() - t0, 1)}
        print(f"  {name:22s} B={cell['B']:6.2f}  c*={c_star:4.2f} p*={p_star:4.2f}"
              f"  regret {scored['regret_mean']:8.2f}  vs thompson "
              f"{scored['paired_vs_thompson_mean']:+7.2f}  "
              f"[{rows[name]['wall_seconds']}s]", flush=True)
    return rows


# ---------------------------------------------------------------------------
# part: report
# ---------------------------------------------------------------------------

def part_report() -> None:
    sw = json.loads((OUTDIR / "grid_sweep.json").read_text())
    tu_path = OUTDIR / "grid_tune.json"
    tu = json.loads(tu_path.read_text()) if tu_path.exists() else {}

    print("\n## Head-to-head vs Thompson (paired CRN pseudo-regret; "
          "NEGATIVE = the policy is better)\n")
    print(f"{'cell':22s} {'B':>7s} {'frozen':>9s} {'retuned':>9s} "
          f"{'c*':>5s} {'p*':>5s} {'ucb1':>9s} {'bayesucb':>9s} {'unt(f)':>8s}")
    for name, r in sw.items():
        f = r["policies"]["rule_frozen"]
        t = tu.get(name)
        rt = f"{t['scored']['paired_vs_thompson_mean']:+9.2f}" if t else "        -"
        cs = f"{t['c_star']:5.2f}" if t else "    -"
        ps = f"{t['p_star']:5.2f}" if t else "    -"
        print(f"{name:22s} {r['B']:7.2f} "
              f"{f['paired_vs_thompson_mean']:+9.2f} {rt} {cs} {ps} "
              f"{r['policies']['ucb1_anytime']['paired_vs_thompson_mean']:+9.2f} "
              f"{r['policies']['bayes_ucb']['paired_vs_thompson_mean']:+9.2f} "
              f"{f['p_best_untouched']:8.4f}")

    print("\n## Tails (frozen rule pseudo-regret)\n")
    print(f"{'cell':22s} {'mean':>9s} {'p90':>9s} {'p99':>9s} {'norm':>8s}")
    for name, r in sw.items():
        f = r["policies"]["rule_frozen"]
        print(f"{name:22s} {f['regret_mean']:9.2f} {f['regret_p90']:9.2f} "
              f"{f['regret_p99']:9.2f} {f['regret_norm']:8.3f}")

    if tu:
        print("\n## B-collapse: cells sorted by information budget\n")
        print(f"{'B':>7s} {'cell':22s} {'frozen':>9s} {'retuned':>9s} {'c*':>5s}")
        for name, r in sorted(sw.items(), key=lambda kv: kv[1]["B"]):
            t = tu.get(name)
            if not t:
                continue
            print(f"{r['B']:7.2f} {name:22s} "
                  f"{r['policies']['rule_frozen']['paired_vs_thompson_mean']:+9.2f} "
                  f"{t['scored']['paired_vs_thompson_mean']:+9.2f} {t['c_star']:5.2f}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", required=True,
                    choices=["sweep", "tune", "report", "all"])
    ap.add_argument("--n-seeds", type=int, default=N_REPORT)
    ap.add_argument("--n-tune", type=int, default=N_TUNE)
    ap.add_argument("--cell", default=None, help="run one cell only")
    ap.add_argument("--ladder", default="core", choices=["core", "ext"],
                    help="core = the 21-cell #E32 grid; ext = the long-horizon "
                         "n in (200,500,1000,2000) extension, written to its "
                         "own artifact so the core census is never rewritten")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.ladder == "core" else "_ext"

    if args.part in ("sweep", "all"):
        print(f"=== sweep ({args.ladder}) ===", flush=True)
        rows = part_sweep(args.n_seeds, args.cell, args.ladder)
        out = OUTDIR / f"grid_sweep{suffix}.json"
        out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {out}")
    if args.part in ("tune", "all"):
        print("=== tune ===", flush=True)
        rows = part_tune(args.n_tune, args.n_seeds, args.cell)
        (OUTDIR / "grid_tune.json").write_text(json.dumps(rows, indent=2))
        print(f"wrote {OUTDIR / 'grid_tune.json'}")
    if args.part in ("report", "all"):
        part_report()


if __name__ == "__main__":
    main()

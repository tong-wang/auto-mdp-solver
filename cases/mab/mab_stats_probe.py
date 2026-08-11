"""Spec-§14 readback of the A11-a stats scorer (ESCALATION #E29 follow-up).

The question #E29 licensed: the index net trained on the RAW sufficient
statistics — (pull share, running average, ttg), never handed the posterior —
lands 45.02 below its bayes twin. Did it *recover* the conjugate transform
(pm = shrunk avg, sd = 1/sqrt(1+n)) and lose those points elsewhere, or did
it build something else? Concretely:

  1. is the learned index affine in the BELIEF coordinates (pm, psd) —
     i.e. did the net internally construct the nonlinear map the bayes obs
     hands over — and is it better-described there than in its own raw
     input coordinates?
  2. what quantile schedule c(ttg) does it trace, vs the bayes crown's
     c = 1.51*(ttg/T)^0.079 (#E24) and the optimal 2.5*(ttg/T)^0.15 (#E26)?
  3. does the 12-knot fitted rule reproduce the net (98% agreement was the
     bayes crown's figure), and what does the rule score on the protocol?

Coordinates. The gauss belief map per arm is a bijection (mab_bayes):
n = 1/psd^2 - 1, pm = avg * n/(1+n); inverse share = n/t (t = T - ttg),
avg = pm * (1+n)/n. The scorer's own inputs are (share, avg, ttg/T)
(mab_equinet._split, the A11 share/avg map), so every probe of phi in
belief coordinates runs the forward transform inside torch and autograds
through it — the exchange rate c = (dphi/dpsd)/(dphi/dpm) needs no manual
Jacobian. Grid probes are built ON-SUPPORT (n <= t, n >= 1): unlike the
bayes readback's free (m, s) grid, a stats state must be realizable — a
share > 1 or an avg for an unpulled arm is off the data manifold and the
net's behavior there is unconstrained.

Parts (artifacts under results/{scenario}/interpret/, stats_ prefix):

  validate  §14.1 gate — run the full machinery (transform, autograd chain,
            per-slice linear fits) on a SYNTHETIC scorer built from the
            known index pm + c*psd re-expressed in stats coordinates; it
            must recover c pointwise and R^2 ~ 1 before anything else is
            trusted.
  surface   c-exchange surface + linear R^2 per ttg slice, in belief
            coords AND in raw (share, avg) coords — the coordinate contest
            is question 1.
  fitc      c(ttg) trajectory-weighted on the stats net's own play
            (256 seeds), knots + power law -> stats_fitc.json.
  score     the fitted rules as deterministic policies on the 8192
            protocol (CRN), + argmax agreement vs the stats net and vs the
            bayes crown on the same visited states.

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_stats_probe.py --part validate
    OMP_NUM_THREADS=1 python mab_stats_probe.py --part all --n-seeds 8192
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mab_a5_probe import K, T, PROTO_SEEDS, SCENARIO, VecSim
from mab_anneal_probe import _sample
from mab_interpret import RESULTS, RUNS, _HP

OUTDIR = Path("results") / SCENARIO / "interpret"

#: A11-a artifacts (#E29). Deep dive on the best seed (s2, 1417.69),
#: s1/s3 as robustness — the #E24 convention.
STATS_RUNS = {
    f"a11_s{s}": RESULTS / (f"PPO_obsstats_L1_policyindex_{_HP}"
                            f"_normobsFalse_seed{s}_20260804_174359")
    for s in (1, 2, 3)
}
CANON = "a11_s2"
BAYES_CANON = "crown_s3"                 # the #E24 deep-dive artifact

TTG_SLICES = (1000, 900, 750, 500, 250, 100, 10)
M_GRID = np.linspace(-2.5, 2.5, 41)
S_GRID = np.linspace(0.03, 1.0, 40)


# ---------------------------------------------------------------------------
# the learned stats index, probed in belief coordinates
# ---------------------------------------------------------------------------

def _stats_coords(pm, psd, ttg):
    """Belief -> the scorer's own inputs. numpy, for grid construction."""
    n = 1.0 / np.square(psd) - 1.0
    t = np.maximum(np.asarray(T, dtype=float) - ttg, 1.0)
    share = n / t
    avg = np.where(n > 0, pm * (1.0 + n) / np.maximum(n, 1e-12), 0.0)
    return share, avg


class StatsIndex:
    """phi as a scalar function on belief triples (pm, psd, ttg_raw).

    The forward transform belief -> (share, avg) runs inside torch, so
    grad=True returns d(phi)/d(pm), d(phi)/d(psd) by autograd through it.
    """

    def __init__(self, run_name: str = CANON):
        import torch
        from stable_baselines3 import PPO
        self.torch = torch
        run = STATS_RUNS[run_name]
        self.model = PPO.load(str(run / f"{SCENARIO}_ppo.zip"), device="cpu")
        self.model.policy.set_training_mode(False)
        self.ex = self.model.policy.mlp_extractor
        assert type(self.ex).__name__ == "_IndexExtractor", \
            f"probe expects the plain index class, got {type(self.ex).__name__}"
        assert self.ex.obs_mode == "stats", \
            f"probe expects a stats-mode scorer, got {self.ex.obs_mode!r}"

    def _phi_from_belief(self, pm_t, psd_t, ttg_t):
        t = self.torch
        n = 1.0 / psd_t**2 - 1.0
        tt = t.clamp(float(T) - ttg_t, min=1.0)
        share = n / tt
        avg = pm_t * (1.0 + n) / t.clamp(n, min=1e-12)
        feats = t.stack([share, avg, ttg_t / self.ex.horizon], dim=-1)
        return self.ex.phi(feats).squeeze(-1)

    def phi(self, pm, psd, ttg, grad: bool = False):
        t = self.torch
        pm_t = t.as_tensor(np.asarray(pm), dtype=t.float32)
        psd_t = t.as_tensor(np.asarray(psd), dtype=t.float32)
        ttg_t = t.as_tensor(np.asarray(ttg), dtype=t.float32)
        if not grad:
            with t.no_grad():
                return self._phi_from_belief(pm_t, psd_t, ttg_t).numpy()
        pm_t = pm_t.clone().requires_grad_(True)
        psd_t = psd_t.clone().requires_grad_(True)
        out = self._phi_from_belief(pm_t, psd_t, ttg_t)
        out.sum().backward()
        return (out.detach().numpy(), pm_t.grad.numpy(), psd_t.grad.numpy())

    def phi_raw(self, share, avg, ttg):
        """phi on its own input coordinates (for the coordinate contest)."""
        t = self.torch
        feats = t.stack([t.as_tensor(np.asarray(share), dtype=t.float32),
                         t.as_tensor(np.asarray(avg), dtype=t.float32),
                         t.as_tensor(np.asarray(ttg), dtype=t.float32)
                         / self.ex.horizon], dim=-1)
        with t.no_grad():
            return self.ex.phi(feats).squeeze(-1).numpy()

    def probs(self, obs: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            return self.model.policy.get_distribution(
                t.as_tensor(obs.astype(np.float32))).distribution.probs.numpy()


def _stats_obs(sim: VecSim) -> np.ndarray:
    """[pulls x K, payouts x K, ttg] — mab_gym's stats mode, batched."""
    ttg = np.full((len(sim.seeds), 1), float(T - sim.t))
    return np.concatenate([sim.pulls, sim.payouts, ttg],
                          axis=1).astype(np.float32)


def _support_grid(ttg: float):
    """(pm, psd) grid restricted to realizable states at this ttg:
    n = 1/psd^2 - 1 must satisfy 1 <= n <= t. Returns flat arrays."""
    t = max(T - ttg, 1.0)
    mm, ss = np.meshgrid(M_GRID, S_GRID, indexing="ij")
    n = 1.0 / ss**2 - 1.0
    ok = (n >= 1.0) & (n <= t)
    return mm[ok], ss[ok]


def _linfit_r2(x1, x2, y):
    """R^2 and coef ratio of y ~ a*x1 + b*x2 + const."""
    X = np.column_stack([x1, x2, np.ones(len(x1))])
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(res[0]) / sst if len(res) and sst > 0 else float("nan")
    return r2, float(coef[1] / coef[0])


def _behavior_c(net_a, pm_rows, sd_rows, lo=-4.0, hi=4.0):
    """The c whose rule argmax(pm + c*psd) best matches the net's argmax.

    An ACTION-space fit, deliberately distinct from the phi regression: on
    visited states share and psd are deterministically tied (psd =
    1/sqrt(1+share*t)), so a value-space linear fit can load the scorer's
    share-dependence onto psd with either sign, while what #E24-style
    "the policy IS a quantile index" claims is about the argmax. Coarse
    grid then one refinement; returns (c*, agreement at c*)."""
    def agree(c):
        return float(((pm_rows + c * sd_rows).argmax(axis=1) == net_a).mean())
    cs = np.arange(lo, hi + 1e-9, 0.1)
    ag = np.array([agree(c) for c in cs])
    c0 = cs[ag.argmax()]
    cs2 = np.arange(c0 - 0.09, c0 + 0.09 + 1e-9, 0.02)
    ag2 = np.array([agree(c) for c in cs2])
    return float(cs2[ag2.argmax()]), float(ag2.max())


# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------

def part_validate(n_seeds: int) -> dict:
    """§14.1 recover-a-known-answer: a synthetic scorer that IS
    pm + c_true*psd, written in the net's own stats coordinates, run through
    the identical transform + autograd + fit machinery."""
    import torch

    C_TRUE = 1.7

    class _SyntheticIndex(StatsIndex):
        def __init__(self):                     # no artifact loaded
            self.torch = torch

            class _Ex:                          # duck-typed extractor
                horizon = float(T)
                obs_mode = "stats"

                @staticmethod
                def phi(feats):
                    share, avg = feats[..., 0], feats[..., 1]
                    ttg = feats[..., 2] * T
                    n = share * torch.clamp(T - ttg, min=1.0)
                    pm = avg * n / (1.0 + n)
                    psd = 1.0 / torch.sqrt(1.0 + n)
                    return (pm + C_TRUE * psd).unsqueeze(-1)
            self.ex = _Ex()

    net = _SyntheticIndex()
    report = {}
    for ttg in (900, 500, 100):
        m, s = _support_grid(ttg)
        phi, gm, gs = net.phi(m, s, np.full(len(m), float(ttg)), grad=True)
        c_auto = gs / gm
        r2, c_fit = _linfit_r2(m, s, phi)
        rel = np.abs(c_auto - C_TRUE) / C_TRUE
        report[f"ttg{ttg}"] = {
            "median_rel_err_autograd": float(np.median(rel)),
            "p90_rel_err_autograd": float(np.quantile(rel, 0.9)),
            "linear_r2": round(r2, 6),
            "c_linfit": round(c_fit, 4),
        }
    ok = all(v["median_rel_err_autograd"] < 0.05
             and abs(v["c_linfit"] - C_TRUE) / C_TRUE < 0.05
             and v["linear_r2"] > 0.999 for v in report.values())

    # the behavioral instrument on the same known answer: random realizable
    # states, the synthetic scorer's argmax must be matched at c ~= C_TRUE
    # with agreement ~1
    rng = np.random.default_rng(7)
    rows = 4000
    n = rng.integers(1, 60, size=(rows, K)).astype(float)
    pm = rng.normal(0.0, 0.7, size=(rows, K)) * np.sqrt(1.0 / (1.0 + n)) \
        + rng.normal(0.0, 0.5, size=(rows, K))
    psd = np.sqrt(1.0 / (1.0 + n))
    ttg = np.full(rows * K, 500.0)
    phi = net.phi(pm.ravel(), psd.ravel(), ttg).reshape(rows, K)
    c_beh, ag_beh = _behavior_c(phi.argmax(axis=1), pm, psd)
    report["behavior_fit"] = {"c": c_beh, "agreement": ag_beh}
    ok = ok and abs(c_beh - C_TRUE) < 0.05 and ag_beh > 0.995

    report["c_true"] = C_TRUE
    report["verdict"] = "PROBE VALIDATED" if ok else "PROBE FAILED"
    return report


def part_surface(n_seeds: int, run_name: str = CANON) -> dict:
    """Question 1: per-slice linear R^2 in belief coords vs raw coords,
    + the autograd exchange rate c."""
    net = StatsIndex(run_name)
    out = {"run": run_name, "slices": {}}
    for ttg in TTG_SLICES:
        m, s = _support_grid(ttg)
        if len(m) < 100:                      # ttg=1000: t=0, nothing on-support
            continue
        tt = np.full(len(m), float(ttg))
        phi, gm, gs = net.phi(m, s, tt, grad=True)
        c = gs / np.where(np.abs(gm) > 1e-9, gm, np.nan)
        r2_belief, c_fit = _linfit_r2(m, s, phi)
        share, avg = _stats_coords(m, s, tt)
        r2_raw, _ = _linfit_r2(share, avg, phi)
        # full-grid phi (None off-support) for the contour figure
        mm, ss = np.meshgrid(M_GRID, S_GRID, indexing="ij")
        n_g = 1.0 / ss**2 - 1.0
        ok = (n_g >= 1.0) & (n_g <= max(T - ttg, 1.0))
        grid = np.full(mm.shape, np.nan)
        grid[ok] = phi
        out["m_grid"] = M_GRID.tolist()
        out["s_grid"] = S_GRID.tolist()
        out["slices"][str(int(ttg))] = {
            "phi_grid": [[None if np.isnan(v) else round(float(v), 4)
                          for v in row] for row in grid],
            "n_points": int(len(m)),
            "linear_r2_belief": round(r2_belief, 5),
            "linear_r2_raw": round(r2_raw, 5),
            "c_linfit": round(c_fit, 4),
            "c_median": float(np.nanmedian(c)),
            "c_iqr": [float(np.nanquantile(c, q)) for q in (0.25, 0.75)],
            "monotone_in_pm_frac": float((gm > 0).mean()),
        }
    (OUTDIR / f"stats_surface_{run_name}.json").write_text(json.dumps(out))
    return out["slices"]


def _trajectory_states(net: StatsIndex, n_seeds: int, stride: int = 5):
    """(pm, psd, ttg) triples + per-state stats obs from the stats net's own
    stochastic play on the faithful streams."""
    rng = np.random.default_rng(31)
    sim = VecSim(PROTO_SEEDS[:n_seeds])
    PM, SD, TTg = [], [], []
    for t in range(T):
        if t % stride == 0:
            pm = sim.payouts / (1.0 + sim.pulls)
            psd = np.sqrt(1.0 / (1.0 + sim.pulls))
            PM.append(pm.copy()); SD.append(psd.copy())
            TTg.append(np.full((len(sim.seeds), K), float(T - t)))
        sim.step(_sample(net.probs(_stats_obs(sim)), rng))
    return (np.concatenate([x.ravel() for x in PM]),
            np.concatenate([x.ravel() for x in SD]),
            np.concatenate([x.ravel() for x in TTg]))


def part_fitc(n_seeds: int) -> dict:
    """Question 2, two instruments per ttg bucket on visited states:

    - regression knots: phi ~ a*pm + b*psd + const (value-space,
      DESCRIPTIVE — see _behavior_c's caveat on the share/psd tie);
    - behavioral knots: the c maximizing argmax agreement with the net
      (action-space — the #E24-style "is it a quantile index" statistic).
    The scored rule and the power law are built from the behavioral knots.
    """
    net = StatsIndex(CANON)
    pm, sd, tt = _trajectory_states(net, min(n_seeds, 256))
    pm_r = pm.reshape(-1, K); sd_r = sd.reshape(-1, K)
    tt_r = tt.reshape(-1, K)[:, 0]
    phi_r = net.phi(pm.ravel(), sd.ravel(), tt.ravel()).reshape(-1, K)
    net_a = phi_r.argmax(axis=1)

    knots = np.array([950, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 10],
                     dtype=float)
    c_beh, ag_beh, c_reg, r2_reg = {}, {}, {}, {}
    for k in knots:
        rows = (tt_r >= k - 50) & (tt_r < k + 50)
        if rows.sum() < 200:
            continue
        c, ag = _behavior_c(net_a[rows], pm_r[rows], sd_r[rows])
        c_beh[int(k)] = c
        ag_beh[int(k)] = round(ag, 4)
        # regression on the same bucket, pulled arms only (n >= 1)
        on = rows[:, None] & (sd_r < 0.72)
        sub = np.random.default_rng(5).choice(
            np.flatnonzero(on.ravel()), min(50_000, int(on.sum())), False)
        r2, c_r = _linfit_r2(pm.ravel()[sub], sd.ravel()[sub],
                             net.phi(pm.ravel()[sub], sd.ravel()[sub],
                                     tt.ravel()[sub]))
        c_reg[int(k)] = round(c_r, 4)
        r2_reg[int(k)] = round(r2, 5)

    kk = np.array(sorted(c_beh)); cc = np.array([c_beh[k] for k in kk])
    pos = cc > 0
    if pos.sum() >= 2:                       # a c<=0 knot has no log
        p, logc0 = np.polyfit(np.log(kk[pos] / T), np.log(cc[pos]), 1)
        power = {"c0": float(np.exp(logc0)), "p": float(p),
                 "n_pos_knots": int(pos.sum())}
    else:
        power = {"c0": None, "p": None, "n_pos_knots": int(pos.sum()),
                 "note": "no positive-c behavioral knots"}
    # overall best single c (flat), for the one-number comparison, plus the
    # agreement-vs-c curve so the peak's sharpness is on the record (a high
    # peak on a flat curve would mean the statistic barely discriminates)
    c_flat, ag_flat = _behavior_c(net_a, pm_r, sd_r)
    curve = {str(round(c, 2)):
             float(((pm_r + c * sd_r).argmax(axis=1) == net_a).mean())
             for c in [*np.arange(-3.0, 3.01, 0.25), -1.09, 1.51, 2.5]}
    fit = {"run": CANON,
           "knots_behavior": c_beh, "agreement_behavior": ag_beh,
           "flat_behavior": {"c": c_flat, "agreement": ag_flat,
                             "agreement_curve": curve},
           "knots_regression": c_reg, "r2_regression": r2_reg,
           "powerlaw": power,
           "bayes_crown_powerlaw_E24": {"c0": 1.51, "p": 0.079},
           "optimal_E26": {"c0": 2.5, "p": 0.15}}
    (OUTDIR / "stats_fitc.json").write_text(json.dumps(fit, indent=2))
    return {k: fit[k] for k in ("knots_behavior", "agreement_behavior",
                                "flat_behavior", "knots_regression",
                                "r2_regression", "powerlaw")}


class _FittedRule:
    """argmax(pm_i + c(ttg) * psd_i), deterministic, from stats_fitc.json."""

    def __init__(self, mode: str):
        fit = json.loads((OUTDIR / "stats_fitc.json").read_text())
        if mode == "powerlaw":
            if fit["powerlaw"]["c0"] is None:
                raise ValueError("no power law fitted (no positive knots)")
            c0, p = fit["powerlaw"]["c0"], fit["powerlaw"]["p"]
            self.c = lambda ttg: c0 * (ttg / T) ** p
        else:
            kk = np.array(sorted(int(x) for x in fit["knots_behavior"]))
            cc = np.array([fit["knots_behavior"][str(x)] for x in kk])
            self.c = lambda ttg: np.interp(ttg, kk, cc)

    def act(self, sim: VecSim) -> np.ndarray:
        pm = sim.payouts / (1.0 + sim.pulls)
        psd = np.sqrt(1.0 / (1.0 + sim.pulls))
        return (pm + self.c(float(T - sim.t)) * psd).argmax(axis=1)


def part_score(n_seeds: int) -> dict:
    """Question 3: the fitted rules scored on the protocol + agreement of
    the stats net with its own fit and with the bayes crown."""
    out = {}
    for mode in ("knots", "powerlaw"):
        pol = _FittedRule(mode)
        sim = VecSim(PROTO_SEEDS[:n_seeds])
        for _ in range(T):
            sim.step(pol.act(sim))
        out[mode] = {
            "reward_mean": float(sim.total.mean()),
            "reward_se": float(sim.total.std(ddof=1) / np.sqrt(len(sim.total))),
            "n_seeds": int(n_seeds)}

    # agreement on the stats net's own visited states (64-seed replay)
    from mab_policy_probe import LearnedIndex
    net = StatsIndex(CANON)
    bayes = LearnedIndex(BAYES_CANON)
    pm, sd, tt = _trajectory_states(net, 64)
    pm = pm.reshape(-1, K); sd = sd.reshape(-1, K)
    tt0 = tt.reshape(-1, K)[:, 0]
    sub = np.random.default_rng(9).choice(len(pm), min(40_000, len(pm)), False)
    phi_s = net.phi(pm[sub].ravel(), sd[sub].ravel(),
                    np.repeat(tt0[sub], K)).reshape(-1, K)
    phi_b = bayes.phi(pm[sub].ravel(), sd[sub].ravel(),
                      np.repeat(tt0[sub], K)).reshape(-1, K)
    a_stats = phi_s.argmax(axis=1)
    out["argmax_agreement"] = {"stats_vs_bayescrown":
                               float((a_stats == phi_b.argmax(axis=1)).mean())}
    for mode in ("knots", "powerlaw"):
        pol = _FittedRule(mode)
        c = pol.c(tt0[sub])[:, None]
        out["argmax_agreement"][f"stats_vs_fit_{mode}"] = float(
            ((pm[sub] + c * sd[sub]).argmax(axis=1) == a_stats).mean())
    (OUTDIR / "stats_fitted_score.json").write_text(json.dumps(out, indent=2))
    return out


def part_gap(n_seeds: int) -> dict:
    """The #E23 diagnostic, on the stats net: stochastic vs argmax play on
    the CRN streams. A8-a's gap was 37 (exploration in the index); a large
    gap here means the stats net kept its exploration in the SAMPLING —
    which is what a near-greedy behavioral c with a 1417 stochastic score
    already implies. final_id = P(the end-of-episode posterior argmax is the
    true best arm)."""
    return _gap_for(CANON, n_seeds, write=True)


def _gap_for(run_name: str, n_seeds: int, write: bool = False) -> dict:
    net = StatsIndex(run_name)
    rng = np.random.default_rng(123)
    out = {"run": run_name}
    for leg in ("stochastic", "argmax"):
        sim = VecSim(PROTO_SEEDS[:n_seeds])
        for _ in range(T):
            p = net.probs(_stats_obs(sim))
            sim.step(_sample(p, rng) if leg == "stochastic"
                     else p.argmax(axis=1))
        pm = sim.payouts / (1.0 + sim.pulls)
        out[leg] = {
            "reward_mean": float(sim.total.mean()),
            "reward_se": float(sim.total.std(ddof=1) / np.sqrt(len(sim.total))),
            "final_id": float((pm.argmax(axis=1) == sim.best).mean())}
    out["gap"] = out["stochastic"]["reward_mean"] - out["argmax"]["reward_mean"]
    out["a8a_gap_reference"] = 37.0
    if write:
        (OUTDIR / "stats_gap.json").write_text(json.dumps(out, indent=2))
    return out


def part_gaprobust(n_seeds: int) -> dict:
    """Direction check of the gap on the other two seeds (2048 is plenty:
    the s2 gap is ~120x its SE)."""
    out = {r: _gap_for(r, min(n_seeds, 2048)) for r in ("a11_s1", "a11_s3")}
    (OUTDIR / "stats_gap_robust.json").write_text(json.dumps(out, indent=2))
    return {r: {"gap": v["gap"],
                "argmax_reward": v["argmax"]["reward_mean"],
                "argmax_final_id": v["argmax"]["final_id"]}
            for r, v in out.items()}


def part_profile(n_seeds: int) -> dict:
    """#E17's explore-rate statistic on the stats net's stochastic play:
    P(sampled pull != current posterior argmax) by phase, + mean policy
    entropy. References: the old bayes crown ROSE 0.22 -> 0.305 (#E17,
    dithering); thompson anneals 0.35 -> 0.05; A8-a's argmax gap 37 means
    its stochastic profile ~= its index."""
    net = StatsIndex(CANON)
    rng = np.random.default_rng(123)
    n = min(n_seeds, 2048)
    sim = VecSim(PROTO_SEEDS[:n])
    xr = np.zeros(T); ent = np.zeros(T)
    for t in range(T):
        p = net.probs(_stats_obs(sim))
        pm = sim.payouts / (1.0 + sim.pulls)
        a = _sample(p, rng)
        xr[t] = float((a != pm.argmax(axis=1)).mean())
        ent[t] = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1).mean())
        sim.step(a)
    out = {"n_seeds": n,
           "explore_rate": {"early": float(xr[:100].mean()),
                            "mid": float(xr[100:900].mean()),
                            "late": float(xr[900:].mean()),
                            "final50": float(xr[-50:].mean())},
           "entropy_nats": {"early": float(ent[:100].mean()),
                            "mid": float(ent[100:900].mean()),
                            "late": float(ent[900:].mean())},
           "explore_curve_50": np.round(xr.reshape(-1, 50).mean(axis=1),
                                        4).tolist(),
           "explore_curve": np.round(xr, 4).tolist(),
           "entropy_curve": np.round(ent, 4).tolist(),
           "references": {"oldcrown_E17": "0.22 -> 0.305 (rises)",
                          "thompson_E17": "0.35 -> 0.05 (anneals)"}}
    (OUTDIR / "stats_profile.json").write_text(json.dumps(out, indent=2))
    return {k: v for k, v in out.items() if k != "explore_curve_50"}


# ---------------------------------------------------------------------------
# rung 0 — the #E30 story as a replay GIF: the SAME net, sampled vs argmax
# ---------------------------------------------------------------------------

def _replay_episode(net: StatsIndex, seed: int, rng_seed: int):
    """One seed, two legs of the SAME policy on shared latents: the deployed
    stochastic sampling vs the deterministic argmax mode. Pseudo-regret
    sum(opt - mu[chosen]) per the Part-I convention (mab_interpret)."""
    rng = np.random.default_rng(rng_seed)
    sims = {"sampled": VecSim(np.array([seed])),
            "argmax": VecSim(np.array([seed]))}
    preg = {k: 0.0 for k in sims}
    snaps = []
    for t in range(T):
        frame = {"t": t}
        for name, sim in sims.items():
            p = net.probs(_stats_obs(sim))
            a = int(_sample(p, rng)[0]) if name == "sampled" \
                else int(p.argmax(axis=1)[0])
            pm = sim.payouts[0] / (1.0 + sim.pulls[0])
            psd = np.sqrt(1.0 / (1.0 + sim.pulls[0]))
            preg[name] += float(sim.opt[0] - sim.means[0, a])
            frame[name] = {
                "pm": pm, "psd": psd, "a": a,
                "explore": bool(a != int(pm.argmax())),
                "entropy": float(-(p * np.log(np.clip(p, 1e-12, 1))).sum()),
                "regret": preg[name],
            }
            sim.step(np.array([a]))
        snaps.append(frame)
    means = sims["sampled"].means[0]
    return snaps, means, {n: float(s.total[0]) for n, s in sims.items()}


def _render_gif(net: StatsIndex, seed: int, tag: str, stride: int = 5,
                rng_seed: int = 900):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    snaps, means, totals = _replay_episode(net, seed, rng_seed)
    best = int(means.argmax())
    frames = snaps[::stride] + [snaps[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5),
                             gridspec_kw={"height_ratios": [3, 1.6]})
    (ax_s, ax_a), (ax_r, ax_e) = axes
    fig.suptitle(f"stats twin, seed {seed} — sampled "
                 f"{totals['sampled']:.0f} vs its own argmax mode "
                 f"{totals['argmax']:.0f}  (true best arm {best})")
    LEG = {"sampled": "#e87ba4", "argmax": "#77766f"}

    def draw(i):
        f = frames[i]
        for ax, name in ((ax_s, "sampled"), (ax_a, "argmax")):
            ax.clear()
            d = f[name]
            colors = ["#bbbbbb"] * K
            colors[best] = "#88bb88"
            colors[d["a"]] = "#cc4444" if d["explore"] else "#2255cc"
            ax.bar(range(K), d["pm"], yerr=d["psd"], color=colors, capsize=2)
            ax.plot(range(K), means, "k*", ms=7)
            ax.set_ylim(-2.6, 2.9)
            ax.set_title(f"{name}  t={f['t']}  H={d['entropy']:.2f} nats")
            ax.axhline(0, color="k", lw=0.4)
        ts = [fr["t"] for fr in frames[:i + 1]]
        ax_r.clear()
        for name in ("sampled", "argmax"):
            ax_r.plot(ts, [fr[name]["regret"] for fr in frames[:i + 1]],
                      label=name, color=LEG[name])
        ax_r.set_title("cumulative pseudo-regret  Σ(μ* − μ chosen)")
        ax_r.legend(loc="upper left"); ax_r.set_xlim(0, T)
        ax_e.clear()
        w = 20
        for name in ("sampled", "argmax"):
            ex = [np.mean([fr[name]["explore"]
                           for fr in frames[max(0, j - w):j + 1]])
                  for j in range(i + 1)]
            ax_e.plot(ts, ex, color=LEG[name])
        ax_e.set_ylim(0, 1); ax_e.set_title("explore rate (rolling)")
        ax_e.set_xlim(0, T)

    ani = animation.FuncAnimation(fig, draw, frames=len(frames))
    out = OUTDIR / f"stats_replay_{tag}_seed{seed}.gif"
    ani.save(out, writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    dpr = snaps[-1]["argmax"]["regret"] - snaps[-1]["sampled"]["regret"]
    return out, totals, dpr


def part_gif(n_seeds: int) -> dict:
    """Case GIFs of the #E30 story. Scan a CRN block for the per-seed
    sampled-vs-argmax reward delta, then render: `collapse` (a
    representative large argmax loss — the 90th-percentile delta, not the
    max, which is an outlier by construction), `median`, and `survive`
    (the argmax leg's best case — honesty panel: the mode is not broken on
    every seed). Part-I acceptance discipline: the sampled leg is
    stochastic, so a case GIF must REPRODUCE its claimed effect in
    pseudo-regret on replay or the next candidate seed is tried."""
    net = StatsIndex(CANON)
    n = min(n_seeds, 256)
    totals = {}
    for leg in ("sampled", "argmax"):
        sim = VecSim(PROTO_SEEDS[:n])
        rng = np.random.default_rng(123)
        for _ in range(T):
            p = net.probs(_stats_obs(sim))
            sim.step(_sample(p, rng) if leg == "sampled"
                     else p.argmax(axis=1))
        totals[leg] = sim.total.copy()
    delta = totals["sampled"] - totals["argmax"]

    cases = {
        "collapse": (np.argsort(-delta)[int(0.10 * n)], lambda d: d > 400.0),
        "median": (np.argsort(np.abs(delta - np.median(delta)))[0],
                   lambda d: d > 100.0),
        "survive": (np.argsort(delta)[0], lambda d: abs(d) < 200.0),
    }
    out = {}
    for tag, (idx, accept) in cases.items():
        order = {"collapse": np.argsort(-delta)[int(0.10 * n):],
                 "median": np.argsort(np.abs(delta - np.median(delta))),
                 "survive": np.argsort(delta)}[tag]
        for cand in order[:12]:
            seed = int(PROTO_SEEDS[cand])
            path, tot, dpr = _render_gif(net, seed, tag)
            if accept(dpr):
                out[tag] = {"seed": seed,
                            "scanned_delta": float(delta[cand]),
                            "replayed_pseudoregret_delta": float(dpr),
                            "totals": tot, "gif": str(path)}
                break
            path.unlink()               # effect not reproduced: next seed
    (OUTDIR / "stats_replay_cases.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# rung 0b — the three-way race: thompson vs the bayes twin vs the stats twin
# ---------------------------------------------------------------------------

#: entity colors, shared with mab_stats_plot / mab_plot_policy
RACE_C = {"thompson": "#eb6834", "bayes": "#2a78d6", "stats": "#e87ba4"}


def _race_episode(seed: int, rng_seed: int, pols):
    """One seed, three policies on shared latents, each played the way it is
    DEPLOYED (thompson samples its posterior; both nets sample their softmax
    — D5/D7). Same VecSim seed ⇒ identical arm means and identical payout
    draws for any (arm, t) either policy chooses: a CRN race, not three
    independent episodes."""
    from mab_a5_probe import VecThompson
    # PER-LEG streams. A leg's trajectory must not depend on which OTHER
    # legs share the render: with one shared generator the legs interleave
    # their draws, so adding a policy to the panel silently rewrites the
    # others' episodes (it did — see INTERPRET.md's note on seed 7003).
    # The bayes leg inherits Part I's exact stream (mab_interpret.
    # _replay_episode: default_rng(rng_seed), one draw per step) and
    # thompson keeps rng_seed+1, so this race REPRODUCES that round's
    # episode bit-for-bit and the two can be laid side by side.
    rngs = {"bayes": np.random.default_rng(rng_seed),
            "stats": np.random.default_rng(rng_seed + 2)}
    th = VecThompson(rng_seed=rng_seed + 1)
    sims = {n: VecSim(np.array([seed])) for n in ("thompson", "bayes", "stats")}
    preg = {n: 0.0 for n in sims}
    snaps = []
    for t in range(T):
        frame = {"t": t}
        for name, sim in sims.items():
            pm = sim.payouts[0] / (1.0 + sim.pulls[0])
            psd = np.sqrt(1.0 / (1.0 + sim.pulls[0]))
            if name == "thompson":
                a = int(th.act(sim)[0])
                ent = float("nan")
            else:
                obs = sim.obs() if name == "bayes" else _stats_obs(sim)
                p = pols[name].probs(obs)
                a = int(_sample(p, rngs[name])[0])
                ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
            preg[name] += float(sim.opt[0] - sim.means[0, a])
            frame[name] = {"pm": pm, "psd": psd, "a": a,
                           "explore": bool(a != int(pm.argmax())),
                           "entropy": ent, "regret": preg[name],
                           "total": float(sim.total[0])}
            sim.step(np.array([a]))
        snaps.append(frame)
    return snaps, sims["stats"].means[0], \
        {n: float(s.total[0]) for n, s in sims.items()}


def _render_race(seed: int, tag: str, pols, stride: int = 5,
                 rng_seed: int = 900):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    snaps, means, totals = _race_episode(seed, rng_seed, pols)
    best = int(means.argmax())
    frames = snaps[::stride] + [snaps[-1]]
    names = ("thompson", "bayes", "stats")
    titles = {"thompson": "thompson (reference)",
              "bayes": "bayes twin (A8-a)", "stats": "stats twin (A11-a)"}

    fig = plt.figure(figsize=(12.6, 6.8))
    gs = fig.add_gridspec(2, 6, height_ratios=[3, 1.7], hspace=0.32,
                          wspace=0.55)
    tops = {n: fig.add_subplot(gs[0, 2 * i:2 * i + 2])
            for i, n in enumerate(names)}
    ax_r = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])
    fig.suptitle(f"same seed {seed}, same latents — "
                 + "  ·  ".join(f"{titles[n]} {totals[n]:.0f}" for n in names)
                 + f"   (true best arm {best})")
    # the y-range must contain every true mean: whether a policy found the
    # best arm is the thing being watched, so a clipped star hides the point
    ylim = (min(-2.6, float(means.min()) - 0.8),
            max(2.9, float(means.max()) + 0.7))

    def draw(i):
        f = frames[i]
        for n in names:
            ax = tops[n]; ax.clear()
            d = f[n]
            colors = ["#bbbbbb"] * K
            colors[best] = "#88bb88"
            colors[d["a"]] = "#cc4444" if d["explore"] else RACE_C[n]
            ax.bar(range(K), d["pm"], yerr=d["psd"], color=colors, capsize=2)
            ax.plot(range(K), means, "k*", ms=7)
            ent = "" if np.isnan(d["entropy"]) else f"  H={d['entropy']:.2f}"
            ax.set_title(f"{titles[n]}   t={f['t']}{ent}", fontsize=10,
                         color=RACE_C[n])
            ax.set_ylim(*ylim); ax.axhline(0, color="k", lw=0.4)
        ts = [fr["t"] for fr in frames[:i + 1]]
        ax_r.clear(); ax_e.clear()
        w = 20
        for n in names:
            ax_r.plot(ts, [fr[n]["regret"] for fr in frames[:i + 1]],
                      color=RACE_C[n], lw=1.8, label=titles[n])
            ex = [np.mean([fr[n]["explore"]
                           for fr in frames[max(0, j - w):j + 1]])
                  for j in range(i + 1)]
            ax_e.plot(ts, ex, color=RACE_C[n], lw=1.8)
        ax_r.set_title("cumulative pseudo-regret  Σ(μ* − μ chosen)",
                       fontsize=10)
        ax_r.legend(loc="lower right", fontsize=8, frameon=False)
        ax_r.set_xlim(0, T)
        ax_e.set_title("explore rate (rolling)", fontsize=10)
        ax_e.set_ylim(0, 1); ax_e.set_xlim(0, T)

    ani = animation.FuncAnimation(fig, draw, frames=len(frames))
    out = OUTDIR / f"race_{tag}_seed{seed}.gif"
    ani.save(out, writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    regrets = {n: snaps[-1][n]["regret"] for n in names}
    return out, totals, regrets


def part_race(n_seeds: int, seeds: list[int] | None = None) -> dict:
    """Three-way replay: thompson, the bayes twin (A8-a s3) and the stats
    twin (A11-a s2) on ONE seed's latents, each deployed-stochastic.

    Seed choice is a scan over a CRN block for two illustrative cases —
    `typical` (the three finish close: the everyday picture, three routes to
    the same place) and `spread` (where the three visibly part). Both are
    SINGLE EPISODES of stochastic policies: illustrations, not evidence.
    The quantitative claims live in the battery (#E24) and #E29/#E30's
    committed evals — a replay's own numbers are one draw, labelled as such.

    Selection and gate are both **pseudo-regret**, the plotted quantity and
    the means-based one (Part I's argument: realized payout is a random walk
    once a policy locks, so a payout-spread scan selects luck). The gate is
    Part I's discipline and it is not decoration: the first `spread`
    candidate this scan proposed (payout spread 1569) replayed to a
    three-way tie — a stochastic policy does not replay its scanned episode,
    so a case that cannot reproduce its effect is discarded and the next
    candidate tried.
    """
    from mab_a5_probe import VecThompson
    from mab_anneal_probe import CrownPolicy
    pols = {"bayes": CrownPolicy(RUNS[BAYES_CANON]), "stats": StatsIndex(CANON)}
    names = ("thompson", "bayes", "stats")

    if seeds:
        # Prescribed seeds — no scan, no gate: the episodes are named in
        # advance (Part I's #E24 replay cases, so the two rounds can be
        # watched on the same latents), which is exactly the situation a
        # selection gate exists to protect against and therefore does not
        # apply. Part I's win/loss/median labels describe the CROWN-vs-
        # THOMPSON delta that chose them; they are carried here as
        # provenance, not as predictions about this three-way draw.
        roles = {}
        cases_f = OUTDIR / "replay_cases.json"
        if cases_f.is_file():
            roles = {v["seed"]: k for k, v in
                     json.loads(cases_f.read_text()).items()}
        out = {}
        for seed in seeds:
            role = roles.get(seed)
            tag = f"E24{role}" if role else "seed"
            path, totals, regrets = _render_race(int(seed), tag, pols)
            out[str(seed)] = {"seed": int(seed), "E24_role": role,
                              "replayed_regrets": regrets,
                              "replayed_regret_spread":
                                  float(max(regrets.values())
                                        - min(regrets.values())),
                              "replayed_totals": totals, "gif": str(path)}
        f = OUTDIR / "race_cases_e24seeds.json"
        f.write_text(json.dumps(out, indent=2))
        return out

    n = min(n_seeds, 256)
    preg = {}
    for i, name in enumerate(names):
        sim = VecSim(PROTO_SEEDS[:n])
        th = VecThompson(rng_seed=8)
        rng = np.random.default_rng(123 + i)      # per-policy, as above
        acc = np.zeros(n)
        for _ in range(T):
            if name == "thompson":
                a = th.act(sim)
            else:
                obs = sim.obs() if name == "bayes" else _stats_obs(sim)
                a = _sample(pols[name].probs(obs), rng)
            acc += sim.opt - sim.means[np.arange(n), a]
            sim.step(a)
        preg[name] = acc
    stack = np.stack([preg[n_] for n_ in names])
    spread = stack.max(axis=0) - stack.min(axis=0)

    cases = (("typical", np.argsort(spread), lambda s: s <= 120.0),
             ("spread", np.argsort(-spread), lambda s: s >= 300.0))
    out = {}
    for tag, order, accept in cases:
        for cand in order[:12]:
            seed = int(PROTO_SEEDS[cand])
            path, totals, regrets = _render_race(seed, tag, pols)
            rspread = max(regrets.values()) - min(regrets.values())
            if accept(rspread):
                out[tag] = {"seed": seed,
                            "scan_regret_spread": float(spread[cand]),
                            "replayed_regret_spread": float(rspread),
                            "replayed_regrets": regrets,
                            "replayed_totals": totals, "gif": str(path)}
                break
            path.unlink()              # effect not reproduced: next seed
    (OUTDIR / "race_cases.json").write_text(json.dumps(out, indent=2))
    return out


PARTS = {"validate": part_validate, "surface": part_surface,
         "fitc": part_fitc, "score": part_score, "gap": part_gap,
         "gaprobust": part_gaprobust, "profile": part_profile,
         "gif": part_gif, "race": part_race}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", required=True, choices=[*PARTS, "all"])
    ap.add_argument("--n-seeds", type=int, default=len(PROTO_SEEDS))
    ap.add_argument("--run", default=CANON, choices=list(STATS_RUNS),
                    help="artifact for --part surface (fitc/score fix CANON)")
    ap.add_argument("--seeds", default="",
                    help="--part race: render these episode seeds instead of "
                         "scanning for cases, e.g. '7003,3310,4891' (the #E24 "
                         "replay seeds)")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    names = list(PARTS) if args.part == "all" else [args.part]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for name in names:
        t0 = time.time()
        kw = {"run_name": args.run} if name == "surface" else {}
        if name == "race" and seeds:
            kw = {"seeds": seeds}
        res = PARTS[name](args.n_seeds, **kw)
        print(f"--- {name} ({time.time() - t0:.1f}s) ---")
        print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()

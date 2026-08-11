"""Spec-§14 policy-interpretation probe for the mab crown (INTERPRET_PLAN.md).

Anchored form: the crown (A8-a, #E23) is architecturally an index policy —
the actor IS phi(post_mean_i, post_sd_i, ttg/T) -> logit_i, shared across
arms — so the readback extracts the learned index directly and puts its
exploration bonus on the same axes as UCB1's sigma*sqrt(2 ln t / n).

Parts (artifacts under results/{scenario}/interpret/):

  ucbcheck  §14.1 validation gate — run the c-extraction machinery on the
            ANALYTIC UCB1 index, where the answer is known in closed form:
            in posterior coords n = 1/s^2 - 1, the bonus is
            sigma*sqrt(2 ln t)*s/sqrt(1-s^2), so c = d(bonus)/ds is known
            pointwise. The probe is trusted for discovery only if it
            recovers this.
  surface   the raw index surface phi over an (m, s) grid at ttg slices,
            + per-slice linear-fit R^2 (straight-indifference-lines
            statistic: the UCB class predicts phi ~ a*(m + c*s) + b), and
            the autograd exchange rate c = (dphi/ds)/(dphi/dm) per point.
  fitc      c(ttg): (a) grid-fit at each ttg knot; (b) trajectory-weighted
            fit on states the policy actually visits (256-seed replay);
            (c) a two-parameter smooth form c(ttg) = c0 * (ttg/T)^p.
  score     §14.2 — the fitted rules ARE policies: argmax(m_i + c(ttg)*s_i),
            deterministic; scored on the 8192 protocol block (CRN). The
            published trio is net / fitted-knots / fitted-powerlaw.
  critic    2048's lesson — read the other network. V calibration vs
            realized return-to-go (overall AND within phase — the Simpson
            split), and dV/d(sd) by phase: does the critic price
            information (positive early, ~0 late — the #E22 mechanism)?

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_policy_probe.py --part ucbcheck
    OMP_NUM_THREADS=1 python mab_policy_probe.py --part all
    OMP_NUM_THREADS=1 python mab_policy_probe.py --part score --n-seeds 8192
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mab_a5_probe import K, T, PROTO_SEEDS, SCENARIO, VecSim
from mab_anneal_probe import _sample
from mab_interpret import RUNS, SIGMA

OUTDIR = Path("results") / SCENARIO / "interpret"
CANON  = "crown_s3"                      # deep-dive artifact; s1/s2 = robustness

TTG_SLICES = (1000, 900, 750, 500, 250, 100, 10)
M_GRID = np.linspace(-2.5, 2.5, 41)
S_GRID = np.linspace(0.03, 1.0, 40)


# ---------------------------------------------------------------------------
# the learned index, called directly
# ---------------------------------------------------------------------------

class LearnedIndex:
    """phi as a scalar function on (m, s, ttg_raw) triples, plus V."""

    def __init__(self, run_name: str = CANON):
        import torch
        from stable_baselines3 import PPO
        self.torch = torch
        run = RUNS[run_name]
        self.model = PPO.load(str(run / f"{SCENARIO}_ppo.zip"), device="cpu")
        self.model.policy.set_training_mode(False)
        self.ex = self.model.policy.mlp_extractor
        assert type(self.ex).__name__ == "_IndexExtractor", \
            f"probe expects the plain index class, got {type(self.ex).__name__}"

    def phi(self, m, s, ttg, grad: bool = False):
        """Index values (and optionally d(phi)/dm, d(phi)/ds) on triples."""
        t = self.torch
        feats = t.stack([t.as_tensor(m, dtype=t.float32),
                         t.as_tensor(s, dtype=t.float32),
                         t.as_tensor(ttg, dtype=t.float32) / self.ex.horizon],
                        dim=-1)
        if not grad:
            with t.no_grad():
                return self.ex.phi(feats).squeeze(-1).numpy()
        feats = feats.clone().requires_grad_(True)
        out = self.ex.phi(feats).squeeze(-1)
        out.sum().backward()
        g = feats.grad.numpy()
        return out.detach().numpy(), g[..., 0], g[..., 1]

    def values(self, obs: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            return self.model.policy.predict_values(
                t.as_tensor(obs.astype(np.float32))).numpy().ravel()

    def probs(self, obs: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            return self.model.policy.get_distribution(
                t.as_tensor(obs.astype(np.float32))).distribution.probs.numpy()


def ucb_index(m_post, s, ttg):
    """The analytic UCB1 index mapped into posterior coordinates.

    n = 1/s^2 - 1 (gauss posterior sd -> pull count); t = T - ttg + 1.
    Empirical mean ~= posterior mean for the mapping's purpose (they differ
    by the prior shrink factor (n+1)/n, noted in the artifact)."""
    n = np.maximum(1.0 / np.square(s) - 1.0, 1e-9)
    t = np.maximum(T - np.asarray(ttg, dtype=float) + 1.0, 2.0)
    return m_post + SIGMA * np.sqrt(2.0 * np.log(t) / n)


# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------

def part_ucbcheck(n_seeds: int) -> dict:
    """Fit-c machinery on the analytic UCB index: recovered c must match the
    closed-form derivative d(bonus)/ds pointwise."""
    mm, ss = np.meshgrid(M_GRID, S_GRID, indexing="ij")
    report = {}
    for ttg in (900, 500, 100):
        idx = ucb_index(mm, ss, ttg)
        # numeric dphi/ds and dphi/dm on the grid
        dm = np.gradient(idx, M_GRID, axis=0)
        ds = np.gradient(idx, S_GRID, axis=1)
        c_num = ds / dm
        n = 1.0 / ss**2 - 1.0
        t = T - ttg + 1.0
        c_true = SIGMA * np.sqrt(2.0 * np.log(t)) / (1.0 - ss**2)**1.5 \
            * np.ones_like(mm)
        # interior points only (gradient endpoints are one-sided)
        sl = (slice(1, -1), slice(1, -1))
        err = np.abs(c_num[sl] - c_true[sl]) / np.maximum(c_true[sl], 1e-9)
        report[f"ttg{ttg}"] = {
            "median_rel_err": float(np.median(err)),
            "p90_rel_err": float(np.quantile(err, 0.9)),
        }
    ok = all(v["median_rel_err"] < 0.05 for v in report.values())
    report["verdict"] = "PROBE VALIDATED" if ok else "PROBE FAILED"
    return report


def part_surface(n_seeds: int, run_name: str = CANON) -> dict:
    net = LearnedIndex(run_name)
    mm, ss = np.meshgrid(M_GRID, S_GRID, indexing="ij")
    out = {"run": run_name, "m_grid": M_GRID.tolist(), "s_grid": S_GRID.tolist(),
           "slices": {}}
    for ttg in TTG_SLICES:
        tt = np.full_like(mm, float(ttg))
        phi, gm, gs = net.phi(mm, ss, tt, grad=True)
        c = gs / np.where(np.abs(gm) > 1e-9, gm, np.nan)
        # straight-indifference statistic: R^2 of phi ~ a*m + b*s + const
        X = np.column_stack([mm.ravel(), ss.ravel(), np.ones(mm.size)])
        coef, res, *_ = np.linalg.lstsq(X, phi.ravel(), rcond=None)
        sst = float(((phi - phi.mean()) ** 2).sum())
        r2 = 1.0 - float(res[0]) / sst if len(res) else float("nan")
        out["slices"][str(ttg)] = {
            "phi": np.round(phi, 4).tolist(),
            "c_surface": np.round(c, 4).tolist(),
            "linear_r2": round(r2, 5),
            "c_linfit": round(float(coef[1] / coef[0]), 4),
            "monotone_in_m_frac": float((gm > 0).mean()),
            "c_median": float(np.nanmedian(c)),
            "c_iqr": [float(np.nanquantile(c, q)) for q in (0.25, 0.75)],
        }
    (OUTDIR / f"index_surface_{run_name}.json").write_text(json.dumps(out))
    return {k: {kk: vv for kk, vv in v.items()
                if kk in ("linear_r2", "c_linfit", "c_median",
                          "monotone_in_m_frac")}
            for k, v in out["slices"].items()}


def _trajectory_states(net: LearnedIndex, n_seeds: int, stride: int = 5):
    """(m, s, ttg, visit-time) triples from the crown's own play."""
    from mab_anneal_probe import CrownPolicy
    pol = CrownPolicy(RUNS[CANON])
    rng = np.random.default_rng(31)
    sim = VecSim(PROTO_SEEDS[:n_seeds])
    M, S, TT = [], [], []
    for t in range(T):
        obs = sim.obs()
        if t % stride == 0:
            M.append(obs[:, :K].copy()); S.append(obs[:, K:2 * K].copy())
            TT.append(np.full((len(obs), K), float(T - t)))
        sim.step(_sample(pol.probs(obs), rng))
    return (np.concatenate([x.ravel() for x in M]),
            np.concatenate([x.ravel() for x in S]),
            np.concatenate([x.ravel() for x in TT]))


def part_fitc(n_seeds: int) -> dict:
    net = LearnedIndex(CANON)
    m, s, tt = _trajectory_states(net, min(n_seeds, 256))
    # bucket by ttg; weighted LS phi ~ a*m + b*s + const within bucket
    knots = np.array([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 10],
                     dtype=float)
    c_traj, c_grid = {}, {}
    for k in knots:
        lo, hi = k - 50, k + 50
        mask = (tt >= lo) & (tt < hi)
        if mask.sum() < 500:
            continue
        sub = np.random.default_rng(5).choice(np.flatnonzero(mask),
                                              min(50_000, mask.sum()),
                                              replace=False)
        phi = net.phi(m[sub], s[sub], tt[sub])
        X = np.column_stack([m[sub], s[sub], np.ones(len(sub))])
        coef, *_ = np.linalg.lstsq(X, phi, rcond=None)
        c_traj[int(k)] = float(coef[1] / coef[0])
        mm, ss = np.meshgrid(M_GRID, S_GRID, indexing="ij")
        phi_g = net.phi(mm.ravel(), ss.ravel(), np.full(mm.size, k))
        Xg = np.column_stack([mm.ravel(), ss.ravel(), np.ones(mm.size)])
        cg, *_ = np.linalg.lstsq(Xg, phi_g, rcond=None)
        c_grid[int(k)] = float(cg[1] / cg[0])
    # smooth form c(ttg) = c0 * (ttg/T)^p on the trajectory knots
    kk = np.array(sorted(c_traj)); cc = np.array([c_traj[k] for k in kk])
    pos = cc > 0
    p, logc0 = np.polyfit(np.log(kk[pos] / T), np.log(cc[pos]), 1)
    fit = {"knots_traj": c_traj, "knots_grid": c_grid,
           "powerlaw": {"c0": float(np.exp(logc0)), "p": float(p)},
           "ucb_reference_c_at": {str(int(k)):
               float(SIGMA * np.sqrt(2 * np.log(T - k + 1)))
               for k in kk}}
    (OUTDIR / "fitc.json").write_text(json.dumps(fit, indent=2))
    return {"powerlaw": fit["powerlaw"],
            "c_traj_ends": {str(int(kk[0])): cc[0], str(int(kk[-1])): cc[-1]}}


class FittedIndexPolicy:
    """argmax(m_i + c(ttg) * s_i), deterministic — the §14.2 scored rule."""

    def __init__(self, mode: str):
        fit = json.loads((OUTDIR / "fitc.json").read_text())
        if mode == "powerlaw":
            c0, p = fit["powerlaw"]["c0"], fit["powerlaw"]["p"]
            self.c = lambda ttg: c0 * (ttg / T) ** p
        else:
            kk = np.array(sorted(int(x) for x in fit["knots_traj"]))
            cc = np.array([fit["knots_traj"][str(x)] for x in kk])
            self.c = lambda ttg: np.interp(ttg, kk, cc)

    def act(self, sim: VecSim) -> np.ndarray:
        pm = sim.payouts / (1.0 + sim.pulls)
        psd = np.sqrt(1.0 / (1.0 + sim.pulls))
        return (pm + self.c(float(T - sim.t)) * psd).argmax(axis=1)


def part_score(n_seeds: int) -> dict:
    out = {}
    for mode in ("knots", "powerlaw"):
        pol = FittedIndexPolicy(mode)
        sim = VecSim(PROTO_SEEDS[:n_seeds])
        for _ in range(T):
            sim.step(pol.act(sim))
        out[mode] = {"reward_mean": float(sim.total.mean()),
                     "reward_se": float(sim.total.std(ddof=1) / np.sqrt(len(sim.total))),
                     "n_seeds": int(n_seeds)}
    # agreement of the fitted rules with the net's argmax on trajectory states
    net = LearnedIndex(CANON)
    m, s, tt = _trajectory_states(net, 64)
    m = m.reshape(-1, K); s = s.reshape(-1, K); tt0 = tt.reshape(-1, K)[:, 0]
    sub = np.random.default_rng(9).choice(len(m), min(40_000, len(m)), False)
    phi = net.phi(m[sub].ravel(), s[sub].ravel(),
                  np.repeat(tt0[sub], K)).reshape(-1, K)
    net_a = phi.argmax(axis=1)
    for mode in ("knots", "powerlaw"):
        pol = FittedIndexPolicy(mode)
        c = pol.c(tt0[sub])[:, None]
        out[mode]["argmax_agreement_vs_net"] = float(
            ((m[sub] + c * s[sub]).argmax(axis=1) == net_a).mean())
    (OUTDIR / "fitted_score.json").write_text(json.dumps(out, indent=2))
    return out


def part_critic(n_seeds: int) -> dict:
    from mab_anneal_probe import CrownPolicy
    net = LearnedIndex(CANON)
    pol = CrownPolicy(RUNS[CANON])
    rng = np.random.default_rng(41)
    n = min(n_seeds, 512)
    sim = VecSim(PROTO_SEEDS[:n])
    V = np.zeros((n, T)); R = np.zeros((n, T))
    dVds = np.zeros(T)
    tch = net.torch
    for t in range(T):
        obs = sim.obs()
        V[:, t] = net.values(obs)
        # dV/d(sd_i), summed over arms, autograd on a 64-episode slice
        ot = tch.as_tensor(obs[:64].astype(np.float32)).requires_grad_(True)
        net.model.policy.predict_values(ot).sum().backward()
        dVds[t] = float(ot.grad[:, K:2 * K].sum(dim=1).mean())
        R[:, t] = sim.total
        sim.step(_sample(pol.probs(obs), rng))
    rtg = sim.total[:, None] - R                     # realized return-to-go
    phases = {"early": EARLY_SL, "mid": MID_SL, "late": LATE_SL}
    def corr(sl):
        v, r = V[:, sl].ravel(), rtg[:, sl].ravel()
        return float(np.corrcoef(v, r)[0, 1])
    out = {"n_seeds": n,
           "pearson_overall": float(np.corrcoef(V.ravel(), rtg.ravel())[0, 1]),
           **{f"pearson_{k}": corr(sl) for k, sl in phases.items()},
           "dVds_early": float(dVds[:100].mean()),
           "dVds_mid": float(dVds[100:900].mean()),
           "dVds_late": float(dVds[900:].mean()),
           "dVds_curve": np.round(dVds, 5).tolist()}
    (OUTDIR / "critic.json").write_text(json.dumps(out))
    return {k: v for k, v in out.items() if k != "dVds_curve"}


EARLY_SL, MID_SL, LATE_SL = slice(0, 100), slice(100, 900), slice(900, 1000)


def part_csweep(n_seeds: int) -> dict:
    """A9 (#E25): the quantile-schedule sweep, on the fitted-rule harness.

    Every candidate is a c-function inside argmax(pm + c*psd) — deterministic,
    no network. Two-block discipline (the t1-temper precedent, #E17):
    candidates are RANKED on the selection block (seeds 1e6.., 2048, the
    training callback's convention) and only the top few are scored ONCE on
    the 0..8191 protocol, so the record block is never max-picked over.

    Families (#E25 pre-registration): flat levels (quantile level), power
    laws (anneal shape), late floors (partial UCB insurance), UCB anchor
    (full growing-quantile insurance), discounted c_inf*(1-gamma^ttg)
    (operator's discounted-mass VOI form — endgame collapse), and joint
    a + b*ln(1+s^2*ttg) (Gittins-style non-separability: the curvature the
    affine readback discarded)."""
    fit = json.loads((OUTDIR / "fitc.json").read_text())
    kk = np.array(sorted(int(x) for x in fit["knots_traj"]))
    cc = np.array([fit["knots_traj"][str(x)] for x in kk])

    cands = {"fit_knots": lambda ttg, s: np.interp(ttg, kk, cc)}
    for f in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        cands[f"flat_{f}"] = (lambda f: lambda ttg, s: f)(f)
    for c0 in (1.5, 2.0, 2.5):
        for p in (0.15, 0.3, 0.6, 1.0):
            cands[f"pow_{c0}_{p}"] = (
                lambda c0, p: lambda ttg, s: c0 * (ttg / T) ** p)(c0, p)
    for fl in (1.5, 2.0, 2.5):
        cands[f"floor_{fl}"] = (
            lambda fl: lambda ttg, s: np.maximum(np.interp(ttg, kk, cc), fl))(fl)
    for a in (0.5, 0.75, 1.0):
        cands[f"ucb_{a}"] = (
            lambda a: lambda ttg, s: a * np.sqrt(2.0 * np.log(T - ttg + 2)))(a)
    for ci in (1.25, 1.5, 1.75, 2.0):
        for g in (0.9, 0.95, 0.99, 0.995):
            cands[f"disc_{ci}_{g}"] = (
                lambda ci, g: lambda ttg, s: ci * (1.0 - g ** ttg))(ci, g)
    for a in (0.5, 1.0, 1.5):
        for b in (0.1, 0.25, 0.5):
            cands[f"joint_{a}_{b}"] = (
                lambda a, b: lambda ttg, s: a + b * np.log1p(s * s * ttg))(a, b)

    def play(cfn, seeds):
        sim = VecSim(seeds)
        for _ in range(T):
            pm = sim.payouts / (1.0 + sim.pulls)
            psd = np.sqrt(1.0 / (1.0 + sim.pulls))
            sim.step((pm + cfn(float(T - sim.t), psd) * psd).argmax(axis=1))
        return sim.total

    SEL = np.arange(1_000_000, 1_000_000 + 2048)
    sel = {}
    for name, cfn in cands.items():
        sel[name] = float(play(cfn, SEL).mean())
        print(f"  sel {name:18s} {sel[name]:9.2f}", flush=True)
    ranked = sorted(sel, key=sel.get, reverse=True)
    proto = {}
    for name in dict.fromkeys(ranked[:3] + ["fit_knots"]):
        tot = play(cands[name], PROTO_SEEDS[:n_seeds])
        proto[name] = {"reward_mean": float(tot.mean()),
                       "reward_se": float(tot.std(ddof=1) / np.sqrt(len(tot)))}
    out = {"selection_n": int(len(SEL)), "selection": sel,
           "ranked_top10": ranked[:10], "protocol": proto}
    (OUTDIR / "csweep.json").write_text(json.dumps(out, indent=2))
    return {"top5_selection": {k: round(sel[k], 2) for k in ranked[:5]},
            "protocol": proto}


def part_freesweep(n_seeds: int) -> dict:
    """A10 (#E27): are there PARAMETER-FREE index rules that match the tuned
    constant? #E26's winner is a policy *instance* (c retuned per K/T/prior);
    Thompson and UCB1 are rules. A zero-parameter form landing at the tuned
    level would mean the learned policy rediscovered a classical
    finite-horizon index — transferable, not fitted.

    Gap this closes: A9's `ucb` family used sqrt(2 ln t) — the ANYTIME form.
    The finite-horizon family keys on the horizon and the arm's own pull
    count (Lai-style sqrt(2 ln(T/n_i))), is per-arm, and was never tested.

    Scored on the protocol block directly: these have no free parameters, so
    there is nothing to select over and no two-block discipline needed. The
    tuned incumbent is re-scored alongside as the yardstick.
    """
    def n_of(psd):                       # posterior sd -> pull count
        return np.maximum(1.0 / np.square(psd) - 1.0, 0.0)

    cands = {
        # --- zero-parameter, horizon-aware, per-arm --------------------------
        "lai_T_over_n":   lambda ttg, s: np.sqrt(
            2.0 * np.log(np.maximum(T / np.maximum(n_of(s), 1.0), 1.0))),
        "lai_ttg_over_n": lambda ttg, s: np.sqrt(
            2.0 * np.log(np.maximum(ttg / np.maximum(n_of(s), 1.0), 1.0))),
        # --- zero-parameter, horizon-aware, arm-blind ------------------------
        "sqrt2ln_ttg":    lambda ttg, s: np.sqrt(2.0 * np.log(max(ttg, 1.0))),
        # --- zero-parameter, anytime (the A9 anchor, a=1) --------------------
        "ucb1_anytime":   lambda ttg, s: np.sqrt(2.0 * np.log(T - ttg + 2.0)),
        # --- zero-parameter, bayes-ucb quantile 1 - 1/(t ln T) ---------------
        "bayes_ucb":      lambda ttg, s: _norm_ppf(
            1.0 - 1.0 / (np.maximum(T - ttg + 1.0, 2.0) * np.log(T))),
        # --- the tuned incumbent, as yardstick (2 params) --------------------
        "tuned_2.5_0.15": lambda ttg, s: 2.5 * (ttg / T) ** 0.15,
    }

    def play(cfn, seeds):
        sim = VecSim(seeds)
        for _ in range(T):
            pm = sim.payouts / (1.0 + sim.pulls)
            psd = np.sqrt(1.0 / (1.0 + sim.pulls))
            sim.step((pm + cfn(float(T - sim.t), psd) * psd).argmax(axis=1))
        return sim.total, sim

    from mab_a5_probe import realize_means
    means = realize_means(PROTO_SEEDS[:n_seeds])
    best = means.argmax(axis=1)
    out = {}
    for name, cfn in cands.items():
        tot, sim = play(cfn, PROTO_SEEDS[:n_seeds])
        fid = (sim.payouts / (1.0 + sim.pulls)).argmax(axis=1) == best
        out[name] = {
            "reward_mean": float(tot.mean()),
            "reward_se": float(tot.std(ddof=1) / np.sqrt(len(tot))),
            "final_id_rate": float(fid.mean()),
            "free_params": 0 if name != "tuned_2.5_0.15" else 2,
        }
        print(f"  {name:18s} {out[name]['reward_mean']:9.2f} "
              f"± {out[name]['reward_se']:.2f}   ID {fid.mean():.4f}",
              flush=True)
    (OUTDIR / "freesweep.json").write_text(json.dumps(out, indent=2))
    return out


def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam), vectorized — avoids a scipy dep."""
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(p)
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(np.where(lo, p, plow)))
    out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])[lo] / \
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)[lo]
    q = np.sqrt(-2 * np.log(np.where(hi, 1 - p, 1 - phigh)))
    out[hi] = -((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])[hi] /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)[hi])
    q = np.where(mid, p, 0.5) - 0.5
    r = q * q
    out[mid] = ((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q)[mid] / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)[mid]
    return out


PARTS = {"ucbcheck": part_ucbcheck, "surface": part_surface,
         "fitc": part_fitc, "score": part_score, "critic": part_critic,
         "csweep": part_csweep, "freesweep": part_freesweep}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=[*PARTS, "all"])
    ap.add_argument("--n-seeds", type=int, default=len(PROTO_SEEDS))
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name in (list(PARTS) if args.part == "all" else [args.part]):
        t0 = time.time()
        print(f"=== {name} ===", flush=True)
        res = PARTS[name](args.n_seeds)
        res["wall_seconds"] = round(time.time() - t0, 1)
        print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()

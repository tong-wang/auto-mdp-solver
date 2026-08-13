"""#E37 — spec-§14 readback of the #E36 GENERALIST: does phi see (ttg, T) only
through their ratio?

The question this answers
------------------------
#E26 distilled the crown into `argmax_i(pm_i + c(ttg/T)*psd_i)` with
`c = 2.5*(ttg/T)^0.15` — a **ratio** form, fitted at one horizon (T=1000).
#E32 then found that off that cell the LEVEL must move: the best constant is
not 2.5 but `c* = 1.204 + 0.286*ln n` with `n = T/K`. Those two facts are in
tension. If exploration really depends only on the fraction of the episode
remaining, `c*` should not depend on `n` at all.

The #E36 generalist is the instrument that can separate them: it is one
network trained across a 20x range of horizons, and — by construction — it is
handed `ttg` and `T` **raw and separately**, never their ratio
(`_IndexHExtractor._split` scales both by the constant `t_max`; see its
docstring). So its learned index is free to depend on `T` at fixed `ttg/T`,
and whether it does is a measurement rather than an assumption.

Three pre-registered outcomes:

  H-ratio   c depends on (ttg, T) only through r = ttg/T. #E26's form
            generalizes; the extra freedom bought nothing.
  H-scale   c grows with T at fixed r, tracking #E32's c*(n) = 1.204 +
            0.286*ln(T/K). The net learned the scaling law the fixed rule
            lacks — and #E26's ratio form is then an artifact of being fitted
            at a single horizon.
  H-neither c varies with T at fixed r but not as c*(n) predicts.

Method
------
`c` is the exchange rate between posterior mean and posterior sd — how much
`sd` one unit of `mean` is worth to the index, i.e. the size of the
exploration bonus. Two estimators, reported side by side:

  * **LS**: least squares `phi ~ a*m + b*s + const` over an (m, s) grid at a
    fixed (ttg, T); `c = b/a`. This is the statistic #E24 published.
  * **autograd**: median of `(dphi/ds)/(dphi/dm)` pointwise on the same grid.

They agree only to the extent the index really is in the `m + c*s` class, so
their spread is itself a diagnostic.

**The gate is two-sided, and that is the point.** A one-sided validation
("does the probe recover a known c?") cannot show the probe is able to *detect*
T-dependence, which is the whole question. So the same extraction is run on two
analytic indices whose answers are known and opposite:

  * `rule`  = m + 2.5*(ttg/T)^0.15 * s  -- ratio-only BY CONSTRUCTION.
             The test must report H-ratio on it.
  * `ucb1`  = m + sigma*sqrt(2 ln t / n), t = T - ttg + 1, n = 1/s^2 - 1 --
             depends on elapsed time, hence on T at fixed r.
             The test must report NOT-H-ratio on it, with q > 0.

If the probe cannot separate those two, nothing it says about the net counts.

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_generalist_readback.py --part gate
    OMP_NUM_THREADS=1 python mab_generalist_readback.py --part ratio
    OMP_NUM_THREADS=1 python mab_generalist_readback.py --part traj
    OMP_NUM_THREADS=1 python mab_generalist_readback.py --part explore
    OMP_NUM_THREADS=1 python mab_generalist_readback.py --part all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRID = "gauss_K10_Tlog"
RESULTS = Path(__file__).resolve().parent / "results"
OUTDIR = RESULTS / GRID / "interpret"
SIGMA = 1.0
K = 10

# The (m, s) box the fit is taken over. s = posterior sd; s -> 0 is a
# well-pulled arm, s -> 1 an untouched one (prior sd = 1).
M_GRID = np.linspace(-2.5, 2.5, 41)
S_GRID = np.linspace(0.05, 1.0, 40)

# The test design: the same set of ratios at every horizon, so `r` and `T` are
# CROSSED rather than confounded. Any dependence on T at fixed r is the signal.
RATIOS = (0.9, 0.7, 0.5, 0.3, 0.1)
HORIZONS = (500, 1000, 2000, 5000, 10000)


# ---------------------------------------------------------------------------
# indices: two analytic controls and the learned one
# ---------------------------------------------------------------------------

def idx_rule(m, s, ttg, T):
    """#E26's distilled rule — ratio-only by construction (H-ratio control)."""
    return m + 2.5 * np.power(np.asarray(ttg, float) / T, 0.15) * s


def idx_ucb1(m, s, ttg, T):
    """UCB1 in posterior coordinates — depends on elapsed time t = T-ttg+1,
    so at fixed r = ttg/T its bonus still grows with T (not-H-ratio control).

    `n` is clipped at 1, not at 1e-9. The map n = 1/s^2 - 1 sends s -> 1 (an
    untouched arm, posterior = prior) to n -> 0, where UCB1's bonus diverges;
    with a 1e-9 floor the fitted c came out ~1.6e4 and the gate "passed" on a
    singularity rather than on the T-dependence it is meant to certify. One
    pull is the smallest state UCB1 is defined on, so that is the floor.
    """
    n = np.maximum(1.0 / np.square(s) - 1.0, 1.0)
    t = np.maximum(float(T) - np.asarray(ttg, float) + 1.0, 2.0)
    return m + SIGMA * np.sqrt(2.0 * np.log(t) / n)


class LearnedIndexH:
    """phi(m, s, ttg, T) for a #E36 generalist, plus its gradients.

    Mirrors `_IndexHExtractor._split` exactly: both time features are divided
    by the CONSTANT `t_max = ex.horizon` (the grid's family maximum), never by
    the episode's own T. Getting this wrong would manufacture the very ratio
    the probe is testing for.
    """

    def __init__(self, run_dir: Path):
        import torch
        from stable_baselines3 import PPO
        self.torch = torch
        self.model = PPO.load(str(run_dir / f"{GRID}_ppo.zip"), device="cpu")
        self.model.policy.set_training_mode(False)
        self.ex = self.model.policy.mlp_extractor
        assert type(self.ex).__name__ == "_IndexHExtractor", \
            f"expected _IndexHExtractor, got {type(self.ex).__name__}"
        self.t_max = float(self.ex.horizon)

    def phi(self, m, s, ttg, T, grad: bool = False):
        t = self.torch
        m_ = t.as_tensor(np.asarray(m, float), dtype=t.float32)
        s_ = t.as_tensor(np.asarray(s, float), dtype=t.float32)
        ttg_ = t.as_tensor(np.broadcast_to(np.asarray(ttg, float), m_.shape).copy(),
                           dtype=t.float32) / self.t_max
        T_ = t.as_tensor(np.broadcast_to(np.asarray(T, float), m_.shape).copy(),
                         dtype=t.float32) / self.t_max
        feats = t.stack([m_, s_, ttg_, T_], dim=-1)
        if not grad:
            with t.no_grad():
                return self.ex.phi(feats).squeeze(-1).numpy()
        feats = feats.clone().requires_grad_(True)
        out = self.ex.phi(feats).squeeze(-1)
        out.sum().backward()
        g = feats.grad.numpy()
        return out.detach().numpy(), g[..., 0], g[..., 1]


# ---------------------------------------------------------------------------
# c extraction
# ---------------------------------------------------------------------------

def extract_c(fn, ttg: float, T: float, net: LearnedIndexH | None = None) -> dict:
    """c at one (ttg, T) cell, by least squares and (for the net) autograd."""
    mm, ss = np.meshgrid(M_GRID, S_GRID, indexing="ij")
    if net is not None:
        phi, gm, gs = net.phi(mm, ss, ttg, T, grad=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            c_auto = gs / np.where(np.abs(gm) > 1e-9, gm, np.nan)
        c_auto_med = float(np.nanmedian(c_auto))
        mono = float((gm > 0).mean())
    else:
        phi = fn(mm, ss, ttg, T)
        c_auto_med, mono = float("nan"), float("nan")

    X = np.column_stack([mm.ravel(), ss.ravel(), np.ones(mm.size)])
    coef, res, *_ = np.linalg.lstsq(X, phi.ravel(), rcond=None)
    sst = float(((phi - phi.mean()) ** 2).sum())
    r2 = 1.0 - float(res[0]) / sst if len(res) else float("nan")
    return {"c_ls": float(coef[1] / coef[0]), "c_auto": c_auto_med,
            "linear_r2": float(r2), "monotone_in_m": mono}


def ratio_table(fn, net=None) -> dict:
    """c on the crossed (r, T) design."""
    tab = {}
    for r in RATIOS:
        for T in HORIZONS:
            ttg = r * T
            tab[f"{r}|{T}"] = extract_c(fn, ttg, T, net)
    return tab


def analyze(tab: dict, key: str = "c_ls") -> dict:
    """Is c a function of r alone?

    Two statistics:
      * `q`  -- the T exponent in log c = log c0 + p*log r + q*log T. H-ratio
                predicts q = 0.
      * `beta` -- the slope in c = a(r) + beta*ln T, pooled across r with a
                per-r intercept. #E32's c*(n) = 1.204 + 0.286*ln(T/K) predicts
                beta = 0.286 (ln K is absorbed by the intercept).
    Plus the raw spread of c within each r row, which needs no model.
    """
    rows, R, TT, C = [], [], [], []
    for r in RATIOS:
        vals = [tab[f"{r}|{T}"][key] for T in HORIZONS]
        rows.append({"r": r, "c_by_T": [round(v, 4) for v in vals],
                     "spread": round(max(vals) - min(vals), 4),
                     "rel_spread": round((max(vals) - min(vals))
                                         / max(abs(np.mean(vals)), 1e-9), 4)})
        for T, v in zip(HORIZONS, vals):
            R.append(r); TT.append(T); C.append(v)
    R, TT, C = np.array(R), np.array(TT), np.array(C)

    out = {"rows": rows,
           "mean_rel_spread_within_r": round(float(np.mean(
               [x["rel_spread"] for x in rows])), 4)}

    ok = C > 0
    if ok.sum() >= 6:
        X = np.column_stack([np.ones(ok.sum()), np.log(R[ok]), np.log(TT[ok])])
        b, *_ = np.linalg.lstsq(X, np.log(C[ok]), rcond=None)
        out["p_logr"] = round(float(b[1]), 4)
        out["q_logT"] = round(float(b[2]), 4)

    # per-r intercepts + one shared ln T slope
    D = np.zeros((len(C), len(RATIOS) + 1))
    for i, r in enumerate(R):
        D[i, RATIOS.index(r)] = 1.0
    D[:, -1] = np.log(TT)
    b2, *_ = np.linalg.lstsq(D, C, rcond=None)
    out["beta_lnT"] = round(float(b2[-1]), 4)
    out["e32_beta_prediction"] = 0.286
    return out


# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------

def runs() -> list[Path]:
    return sorted((RESULTS / GRID).glob("PPO_*/"))


def part_gate() -> dict:
    """Two analytic controls with opposite known answers. The probe is trusted
    only if it separates them."""
    rep = {}
    for name, fn in (("rule_ratio_only", idx_rule), ("ucb1_T_dependent", idx_ucb1)):
        tab = ratio_table(fn)
        a = analyze(tab)
        rep[name] = {"q_logT": a["q_logT"], "beta_lnT": a["beta_lnT"],
                     "mean_rel_spread_within_r": a["mean_rel_spread_within_r"],
                     "linear_r2_median": round(float(np.median(
                         [v["linear_r2"] for v in tab.values()])), 4),
                     "c_range": [round(min(v["c_ls"] for v in tab.values()), 3),
                                 round(max(v["c_ls"] for v in tab.values()), 3)],
                     "rows": a["rows"]}
    q_rule = abs(rep["rule_ratio_only"]["q_logT"])
    q_ucb = rep["ucb1_T_dependent"]["q_logT"]
    passed = q_rule < 0.02 and q_ucb > 0.05
    rep["verdict"] = "PROBE VALIDATED" if passed else "PROBE FAILED"
    rep["criterion"] = ("|q| < 0.02 on the ratio-only control AND q > 0.05 on "
                        "the T-dependent control")
    return rep


def part_ratio() -> dict:
    rep = {}
    for rd in runs():
        seed = rd.name.split("_seed")[-1][:1]
        net = LearnedIndexH(rd)
        tab = ratio_table(None, net)
        a_ls = analyze(tab, "c_ls")
        a_au = analyze(tab, "c_auto")
        r2 = [tab[k]["linear_r2"] for k in tab]
        mono = [tab[k]["monotone_in_m"] for k in tab]
        rep[f"seed{seed}"] = {
            "t_max": net.t_max,
            "linear_r2_min": round(float(np.min(r2)), 4),
            "linear_r2_median": round(float(np.median(r2)), 4),
            "monotone_in_m_min": round(float(np.min(mono)), 4),
            "ls": a_ls, "autograd": a_au,
        }
        (OUTDIR / f"generalist_c_seed{seed}.json").write_text(json.dumps(tab, indent=1))
    return rep


def part_traj(n_seeds: int = 48) -> dict:
    """Reachability check: the (m, s) box includes states the policy may never
    occupy at a given T. Re-fit c on states it ACTUALLY visits, through the
    gym (the source of truth), at three horizons.

    **Read the diagnostics before the number.** The visited cloud here is
    bimodal — one heavily-pulled arm at s -> 0 and a tail of near-untouched
    arms at s in {1.0, 0.707} (n = 0 or 1), with more than half of all
    arm-states at n <= 1. A least-squares slope over two far-apart clusters is
    a chord between them, not a local exchange rate, so `c_ls` here is NOT
    comparable to the grid fit and must not be quoted as "c on visited
    states". `frac_s_gt_0.6` and `linear_r2` are reported so the degeneracy is
    visible in the artifact; when `frac_s_gt_0.6` is large the fit is
    describing starvation, which is #E31's finding, not an index shape.
    """
    from mab_gym import MabEnv
    from mab_scenarios import SCENARIOS
    rep = {}
    for rd in runs()[:1]:                      # one net; this is a cross-check
        seed = rd.name.split("_seed")[-1][:1]
        net = LearnedIndexH(rd)
        model = net.model
        for T in (500, 1000, 10000):
            env = MabEnv(scenario=SCENARIOS[f"gauss_K10_T{T}"],
                         observation_mode="bayes_h")
            M, S, TT = [], [], []
            stride = max(1, T // 100)
            for sd in range(n_seeds):
                obs, _ = env.reset(seed=3_000_000 + sd)
                done, step = False, 0
                while not done:
                    if step % stride == 0:
                        M.append(obs[:K].copy()); S.append(obs[K:2 * K].copy())
                        TT.append(float(T - step))
                    a, _ = model.predict(obs[np.newaxis], deterministic=False)
                    obs, _, done, _, _ = env.step(int(a[0]))
                    step += 1
            m = np.concatenate(M); s = np.concatenate(S)
            tt = np.repeat(np.array(TT), K)
            # bucket by ratio, weighted LS within bucket
            out = {}
            for r in RATIOS:
                lo, hi = (r - 0.1) * T, (r + 0.1) * T
                sel = (tt >= lo) & (tt < hi)
                if sel.sum() < 200:
                    out[str(r)] = None
                    continue
                phi = net.phi(m[sel], s[sel], tt[sel], T)
                X = np.column_stack([m[sel], s[sel], np.ones(sel.sum())])
                coef, res, *_ = np.linalg.lstsq(X, phi, rcond=None)
                sst = float(((phi - phi.mean()) ** 2).sum())
                r2 = 1.0 - float(res[0]) / sst if len(res) else float("nan")
                out[str(r)] = {"c_ls": round(float(coef[1] / coef[0]), 4),
                               "linear_r2": round(float(r2), 4),
                               "n_states": int(sel.sum()),
                               "s_median": round(float(np.median(s[sel])), 4),
                               "frac_s_gt_0.6": round(float((s[sel] > 0.6).mean()), 4),
                               "frac_s_lt_0.1": round(float((s[sel] < 0.1).mean()), 4)}
            rep[f"seed{seed}_T{T}"] = out
    return rep


def part_explore(n_seeds: int = 48) -> dict:
    """Behavioural face of a constant c: does the policy still COVER the arms?

    A fixed-quantile index never grows its bonus, so an abandoned arm is never
    revisited (#E34) — the prediction is that coverage collapses as the horizon
    grows. Measured directly: distinct arms pulled, the top arm's share of
    pulls, and how many pulls the runner-up gets.

    The torch RNG is seeded because `predict(deterministic=False)` samples: an
    unseeded 3-episode read of this said "one arm, 10,000/10,000, zero
    exploration", which did not survive replication (#E37 method note).
    """
    import torch
    from mab_gym import MabEnv
    from mab_scenarios import SCENARIOS
    rd = runs()[0]
    model = LearnedIndexH(rd).model
    out = {}
    for T, n_ep in ((500, n_seeds), (1000, n_seeds),
                    (2000, n_seeds), (10000, max(8, n_seeds // 2))):
        env = MabEnv(scenario=SCENARIOS[f"gauss_K10_T{T}"],
                     observation_mode="bayes_h")
        torch.manual_seed(12345)
        touched, top, second = [], [], []
        for sd in range(n_ep):
            obs, _ = env.reset(seed=3_000_000 + sd)
            done, acts = False, []
            while not done:
                a, _ = model.predict(obs[np.newaxis], deterministic=False)
                acts.append(int(a[0]))
                obs, _, done, _, _ = env.step(int(a[0]))
            c = np.bincount(acts, minlength=K)
            c.sort()
            touched.append(int((c > 0).sum()))
            top.append(float(c[-1] / T))
            second.append(int(c[-2]))
        out[f"T{T}"] = {"n_episodes": n_ep,
                        "arms_touched_mean": round(float(np.mean(touched)), 2),
                        "arms_touched_median": float(np.median(touched)),
                        "top_arm_pull_share": round(float(np.mean(top)), 4),
                        "second_arm_pulls_median": float(np.median(second))}
        print(f"  T={T:<6} {out[f'T{T}']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="all",
                    choices=["gate", "ratio", "traj", "explore", "all"])
    ap.add_argument("--n-seeds", type=int, default=48)
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    parts = (["gate", "ratio", "traj", "explore"]
             if a.part == "all" else [a.part])
    res = {}
    for p in parts:
        res[p] = (part_gate() if p == "gate" else
                  part_ratio() if p == "ratio" else
                  part_traj(a.n_seeds) if p == "traj" else
                  part_explore(a.n_seeds))
        print(f"\n===== {p} =====")
        print(json.dumps(res[p], indent=1))
    (OUTDIR / "generalist_readback.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

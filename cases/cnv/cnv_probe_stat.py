"""Readback v2 for the frame-sum extractor (#E74, after the review of probe_loglik.py):
do the pooled channels carry the exact posterior — on episodes the fit never saw, over the
policy's own orders AND orders up to the cap, at every rate band?

For each raw mode the exact per-period log-likelihood term is known:
    full_raw    log Poisson(demand | rate)
    timing_raw  sales * log(rate) - rate * duration          (a Poisson process observed for `duration`)
    event_raw   log Poisson(sales | rate) if no stock-out, else log P(D >= sales | rate)
Summed along the history and combined with the Gamma prior on a fine log-spaced rate grid this
gives the EXACT posterior by quadrature (no closed form is needed, event_raw included), hence
three scalar targets per state: the posterior mean, its log, and the myopic order (the smallest
y with posterior-predictive P(D <= y) >= critical ratio). The log-likelihood vector on a coarse
rate grid is kept as the fourth target (what probe_loglik measured, now out of sample).

Readouts are LINEAR (standardised float64 ridge, lambda 1e-6, on the fit episodes; R² on the score episodes) — the question
is whether the statistic is present in coordinates a linear map can read, which is the #E59 /
#E73 question. An UNTRAINED extractor of the same shape is scored beside the trained one.

    python cnv_probe_stat.py MODEL.zip --mode timing_raw [--n-fit 80 --n-score 80]
"""
import argparse, sys
import numpy as np, torch as th
from scipy.special import gammaln
from scipy.stats import poisson
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from cnv_gym import CnvEnv
from cnv_scenarios import SCENARIOS
from cnv_raw_extractor import FrameSumExtractor

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--mode", required=True, choices=["full_raw", "timing_raw", "event_raw"])
ap.add_argument("--n-fit", type=int, default=80); ap.add_argument("--n-score", type=int, default=80)
ap.add_argument("--fit-seed0", type=int, default=3_000_000); ap.add_argument("--score-seed0", type=int, default=3_100_000)
ap.add_argument("--p-policy", type=float, default=0.5, help="share of steps ordered by the trained policy (stochastic); the rest log-uniform on [0, cap]")
ap.add_argument("--coarse-rates", default="0.3,1,3,10,30,80")
a = ap.parse_args()
sc = SCENARIOS["fig6a"]; N = int(sc.horizon); alpha, beta = float(sc.prior_shape), float(sc.prior_rate)
coarse = np.array([float(r) for r in a.coarse_rates.split(",")])
grid = np.exp(np.linspace(np.log(0.01), np.log(400.0), 500))                      # quadrature nodes for the rate
log_prior = (alpha - 1) * np.log(grid) + (-beta * grid) + np.log(np.gradient(grid))  # Gamma density x node width (up to a constant)
Y = np.arange(0, 403)
log_cdf_grid = poisson.logcdf(Y[:, None], grid[None, :])                            # (403, G): log P(D <= y | rate)

model = PPO.load(a.model, device="cpu", custom_objects={"lr_schedule": lambda _: 0, "learning_rate": 0, "clip_range": lambda _: 0})
fx_trained = model.policy.features_extractor
cap = int(model.action_space.n) - 1

def term(info, rates):
    s = int(info["sales"])
    if a.mode == "full_raw":
        return poisson.logpmf(int(info["demand"]), rates)
    if a.mode == "timing_raw":
        return s * np.log(rates) - rates * float(info["stockout_time"])
    return np.log(np.clip(poisson.sf(s - 1, rates), 1e-300, None)) if int(info["stockout_event"]) else poisson.logpmf(s, rates)

def collect(seed0, n):
    env = FrameStackObservation(CnvEnv(scenario=sc, observation_mode=a.mode), stack_size=N, padding_type="zero"); base = env.unwrapped
    rng = np.random.default_rng(seed0)
    O, LLc, LLg, LAM, CR, T = [], [], [], [], [], []
    for s in range(seed0, seed0 + n):
        o, _ = env.reset(seed=s); llc = np.zeros(len(coarse)); llg = np.zeros(len(grid))
        lam = float(base._state.rate) if hasattr(base._state, "rate") else float(base._rate) if hasattr(base, "_rate") else np.nan
        for t in range(N):
            O.append(o.copy()); LLc.append(llc.copy()); LLg.append(llg.copy()); LAM.append(lam); CR.append(float(o[-1, 1])); T.append(t)
            if rng.random() < a.p_policy:
                y = int(model.predict(o, deterministic=False)[0])
            else:
                y = int(np.exp(rng.uniform(0.0, np.log(cap + 1.0)))) - 1
            o, r, done, _, info = env.step(y)
            llc = llc + term(info, coarse); llg = llg + term(info, grid)
            if done: break
    return np.array(O, np.float32), np.array(LLc), np.array(LLg), np.array(LAM), np.array(CR), np.array(T)

def targets(LLg, CR):
    lp = LLg + log_prior[None]; lp -= lp.max(1, keepdims=True); w = np.exp(lp); w /= w.sum(1, keepdims=True)   # exact posterior on the grid
    pm = w @ grid                                                                                              # posterior mean
    pred_cdf = np.exp(log_cdf_grid) @ w.T                                                                     # (403, rows): posterior-predictive P(D <= y)
    ymyo = (pred_cdf >= CR[None, :]).argmax(0)                                                                 # the myopic order under the exact posterior
    return pm, np.log(pm), ymyo.astype(float)

def channels(fx, O):
    with th.no_grad():
        return np.concatenate([fx.pre(th.as_tensor(O[i:i + 2048])).numpy() for i in range(0, len(O), 2048)])

def r2(Xf, yf, Xs, ys, lam=1e-6):
    # standardised float64 ridge (tiny lambda): float32 lstsq on near-collinear pooled channels extrapolated wildly on
    # rare high-rate episodes (full_raw mean-pooled read 0.40 +/- 0.79 that way, 0.977 +/- 0.002 this way; #E75)
    Xf, Xs = Xf.astype(np.float64), Xs.astype(np.float64); mu, sd = Xf.mean(0), Xf.std(0) + 1e-8; A, B = (Xf - mu) / sd, (Xs - mu) / sd
    b = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (yf - yf.mean())); p = B @ b + yf.mean()
    return 1 - ((ys - p) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-12), p

Of, LLcf, LLgf, LAMf, CRf, Tf = collect(a.fit_seed0, a.n_fit)
Os, LLcs, LLgs, LAMs, CRs, Ts = collect(a.score_seed0, a.n_score)
pm_f, lpm_f, ym_f = targets(LLgf, CRf); pm_s, lpm_s, ym_s = targets(LLgs, CRs)
if np.isnan(LAMs).all():   # the true rate is not exposed by the state; band by the exact posterior mean at the END of the episode instead
    LAMs = np.repeat([pm_s[Ts == Ts.max()].mean()], len(pm_s))
bands = [(0.0, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, np.inf)]
print(f"{a.mode} · {type(fx_trained).__name__} v2={getattr(fx_trained, 'v2', False)} pool={getattr(fx_trained, 'pool', '-')} · fit {a.n_fit} episodes (seeds from {a.fit_seed0}), score {a.n_score} (from {a.score_seed0}) · orders: {a.p_policy:.0%} policy, rest log-uniform on [0, {cap}]")
print(f"score rows {len(Os)}; orders placed: median {np.median(np.r_[Of[:, -1, 4], Os[:, -1, 4]]):.0f}, share > 24: {(Os[1:, -1, 4] > 24).mean():.0%} (sales/demand column of the newest frame, a lower bound on the order)")
th.manual_seed(0)
fx_untrained = FrameSumExtractor(model.observation_space, pool=getattr(fx_trained, "pool", "sum"), v2=bool(getattr(fx_trained, "v2", False)))
for lab, fx in (("TRAINED  ", fx_trained), ("UNTRAINED", fx_untrained)):
    Zf, Zs = channels(fx, Of), channels(fx, Os)
    out = []
    for name, yf, ys in (("post. mean", pm_f, pm_s), ("log post. mean", lpm_f, lpm_s), ("myopic order", ym_f, ym_s)):
        R, p = r2(Zf, yf, Zs, ys); out.append(f"{name} {R:.3f}")
        if name == "myopic order": out.append(f"(MAE {np.abs(p - ys).mean():.2f}, exact {np.mean(np.round(p) == ys):.0%})")
    llR = np.mean([r2(Zf, LLcf[:, i], Zs, LLcs[:, i])[0] for i in range(len(coarse))])
    print(f"  {lab}  held-out linear R²:  " + "  ".join(out) + f"  |  log-lik vector (mean over {len(coarse)} rates) {llR:.3f}")
    # the log posterior mean by rate band (posterior mean of the episode's own state, score rows)
    parts = []
    for lo, hi in bands:
        m = (pm_s >= lo) & (pm_s < hi)
        if m.sum() < 50: parts.append(f"[{lo:g},{hi:g}) n={m.sum()} -"); continue
        R, _ = r2(Zf, lpm_f, Zs[m], lpm_s[m]); parts.append(f"[{lo:g},{hi:g}) n={m.sum()} {R:.3f}")
    print(f"             log post. mean by posterior-mean band:  " + "  ".join(parts))
    # and by period: early states are where the coordinates problem bites
    parts = []
    for lo, hi in ((0, 5), (5, 20), (20, 60), (60, 100)):
        m = (Ts >= lo) & (Ts < hi); R, _ = r2(Zf, lpm_f, Zs[m], lpm_s[m]); parts.append(f"t[{lo},{hi}) {R:.3f}")
    print(f"             log post. mean by period:  " + "  ".join(parts))

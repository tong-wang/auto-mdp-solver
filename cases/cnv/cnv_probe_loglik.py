"""Readback for the per-frame log-likelihood extractor (#E72): do the pooled channels span the
exact per-period log-likelihood terms?

Under event observation the exact log-likelihood of an episode's history, as a function of the
rate, is a sum over periods of
    uncensored (no stock-out):   log Poisson(sales | rate)
    censored   (stock-out):      log P(D >= sales | rate)        (sales == the order placed)
Evaluated on a grid of rates it is a vector-valued sufficient statistic. The extractor's pooled
channels are a learned sum of per-frame functions; if they span the log-likelihood they should
predict this vector linearly. Reported: R² of a linear fit from the 32 (or 33) pooled channels
to the summed exact term at each grid rate, on held-out episodes at every period.

    python cnv_probe_loglik.py MODEL.zip [--mode event_raw] [--n-episodes 60]
"""
import argparse, sys
import numpy as np, torch as th
from scipy.stats import poisson
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from cnv_gym import CnvEnv
from cnv_scenarios import SCENARIOS

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--mode", default="event_raw"); ap.add_argument("--n-episodes", type=int, default=60)
ap.add_argument("--rates", default="0.3,1,3,10,30,80")
a = ap.parse_args()
sc = SCENARIOS["fig6a"]; N = int(sc.horizon); rates = np.array([float(r) for r in a.rates.split(",")])
fx = PPO.load(a.model, device="cpu", custom_objects={"lr_schedule": lambda _: 0, "learning_rate": 0, "clip_range": lambda _: 0}).policy.features_extractor
env = FrameStackObservation(CnvEnv(scenario=sc, observation_mode=a.mode), stack_size=N, padding_type="zero"); base = env.unwrapped
rng = np.random.default_rng(0)
Z, LL, T, CUM, PER, CUMS, FLAGS = [], [], [], [], [], [], []
cs = fl = 0
for s in range(2000000, 2000000 + a.n_episodes):
    o, _ = env.reset(seed=s); ll = np.zeros(len(rates)); cs = fl = 0
    for t in range(N):
        with th.no_grad(): Z.append(fx.pre(th.as_tensor(o[None], dtype=th.float32))[0].numpy())
        LL.append(ll.copy()); T.append(t); CUM.append(base._state.cum_demand); PER.append(t + 1); CUMS.append(cs); FLAGS.append(fl)
        y = int(rng.integers(0, 25)); o, r, done, _, info = env.step(y)
        sales, stockout = int(info["sales"]), int(info["stockout_event"])
        # this period's exact log-likelihood term at each grid rate
        ll = ll + (np.log(poisson.sf(sales - 1, rates)) if stockout else poisson.logpmf(sales, rates)); cs += sales; fl += stockout
        if done: break
Z, LL, T = np.array(Z), np.array(LL), np.array(T); CUM = np.array(CUM, float); PER = np.array(PER, float); CUMS = np.array(CUMS, float); FLAGS = np.array(FLAGS, float)
def r2(X, y):
    A = np.c_[X, np.ones(len(X))]; b, *_ = np.linalg.lstsq(A, y, rcond=None); p = A @ b
    return 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
print(f"{type(fx).__name__} pool={getattr(fx, 'pool', '-')}  channels {Z.shape[1]}  |  {a.n_episodes} episodes x {N} periods, rates {rates.tolist()}")
print("linear R² of the pooled channels for the exact summed log-likelihood at each rate:")
print("   " + "  ".join(f"rate {r:g}: {r2(Z, LL[:, i]):.3f}" for i, r in enumerate(rates)))
# the log-likelihood DIFFERENCES between rates are what the posterior actually uses
d = LL[:, 1:] - LL[:, :-1]
print("   differences between adjacent rates: " + "  ".join(f"{r2(Z, d[:, i]):.3f}" for i in range(d.shape[1])))
print(f"for reference: R² for cumulative sales-as-demand proxy {r2(Z, CUM):.3f}, periods played {r2(Z, PER):.3f}")
# The discriminating read: 32 summed random per-frame functions of a ~50-valued frame span nearly any
# additive statistic (an untrained extractor already reads R² 1.000 above), so ask instead how
# CONCENTRATED the statistic is — the R² reached by the leading principal components of the channels.
Zc = Z - Z.mean(0); U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
print("R² for the log-likelihood vector (mean over rates) from the leading principal components of the channels:")
print("   " + "  ".join(f"{k} PCs: {np.mean([r2(Zc @ Vt[:k].T, LL[:, i]) for i in range(len(rates))]):.3f}" for k in (1, 2, 3, 4, 8, 16, Z.shape[1])))
print("   variance explained by the leading PCs: " + "  ".join(f"{k}: {(S[:k]**2).sum()/(S**2).sum():.3f}" for k in (1, 2, 3, 4, 8)))
# The part that is NOT the two sums: regress the count, the summed sales and the summed stock-out
# flags out of the log-likelihood vector, then ask how much of the RESIDUAL the channels explain.
# This is the "beyond the two-number statistic" content — the censored levels — and a linear
# per-frame map cannot reach it, so an untrained extractor should read ~0 here.
SUMS = np.c_[PER, CUMS, FLAGS]
def resid(y):
    A = np.c_[SUMS, np.ones(len(y))]; b, *_ = np.linalg.lstsq(A, y, rcond=None); return y - A @ b
R = np.stack([resid(LL[:, i]) for i in range(len(rates))], 1)
print("the residual beyond (count, sum of sales, sum of stock-outs): its share of the log-likelihood's variance, and the channels' R² for it")
print("   " + "  ".join(f"rate {r:g}: share {R[:, i].var()/max(LL[:, i].var(),1e-12):.3f} R² {r2(Z, R[:, i]):.3f}" for i, r in enumerate(rates)))
if getattr(fx, "kernel", None) is not None:
    k = fx.kernel.detach().numpy()
    print(f"learned temporal kernel: mean {k.mean():.3f}, sd across positions {k.std(0).mean():.3f}, per-position mean over channels first/mid/last {k.mean(1)[0]:.3f} / {k.mean(1)[N//2]:.3f} / {k.mean(1)[-1]:.3f}")

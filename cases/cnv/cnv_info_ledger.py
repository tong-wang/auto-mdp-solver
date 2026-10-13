"""Information ledger: what a policy PAYS to deviate from the myopic order at its own belief,
and what it GETS for it in later beliefs (the user's #E83 critique: under censoring the optimum
over-orders to buy information, and the cost cell moves both the price and the value of it).

On the policy's own trajectory, per period t, with the posterior b_t the policy's observation
type supports (timing: Gamma(a0 + S, b0 + T); full: Gamma(a0 + cum demand, b0 + n)):

  C_t = L(y_myo(b_t) | b_t) - L(y_pol | b_t)   the immediate expected profit given up (>= 0)
  B_t = L(y_myo(b_t) | b_t)                     what the myopic rule would earn on THIS belief
  L(y | b) = r E[min(D, y)] - c y, D ~ the negative-binomial predictive of b

and the same B on the myopic policy's own path (B^M). Summed over the episode:
  G = sum B_t - sum B^M_t        information value of the policy's belief path (0 under full obs:
                                  the beliefs do not depend on the orders)
  E[pol] - E[myopic] = G - C     (seeds drawn from the prior, so predictive sums are unbiased
                                  for realised profit; the realised paired gap is printed as a check)

  python cnv_info_ledger.py --model-path M -s fig6b -o timing_belief_log [--n-seeds 2048]
  python cnv_info_ledger.py --dp-sweep            # exact N=4 timing DP over the fig3 cost axis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import betainc
from scipy.stats import nbinom

from cnv_bayes import gamma_poisson_quantile
from cnv_benchmark_myopic import MyopicEvent
from cnv_benchmark_common import eval_seed_block
from cnv_gym import CnvEnv

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_GL_X, _GL_W = 0.5 * (_GL_X + 1.0), 0.5 * _GL_W          # on [0, 1]


def expected_profit(a, b, y, r, c, chunk=20000):
    """L(y | Gamma(a, b)) = r E[min(D, y)] - c y, vectorised over rows."""
    a, b, y = (np.asarray(v, dtype=float) for v in (a, b, y))
    out = np.empty(len(a))
    for s in range(0, len(a), chunk):
        aa, bb, yy = a[s:s + chunk], b[s:s + chunk], y[s:s + chunk]
        K = int(yy.max()) if len(yy) else 0
        if K == 0:
            out[s:s + chunk] = 0.0
            continue
        k = np.arange(K)[None, :]
        sf = nbinom.sf(k, aa[:, None], (bb / (bb + 1.0))[:, None])
        emin = np.where(k < yy[:, None], sf, 0.0).sum(axis=1)
        out[s:s + chunk] = r * emin - c * yy
    return out


def expected_obs_time(a, b, y):
    """E[min(1, tau_y)] under the predictive: the period's observation time when y is stocked."""
    a, b, y = (np.asarray(v, dtype=float) for v in (a, b, y))
    t = _GL_X[None, :]
    x = t / (b[:, None] + t)
    cdf = betainc(np.maximum(y, 1.0)[:, None], a[:, None], x)
    et = ((1.0 - cdf) * _GL_W[None, :]).sum(axis=1)
    return np.where(y >= 1, et, 0.0)


def rollout(sc, act, seeds, info):
    """Run `act(env, obs, base)` on seeds; record the belief the info type supports, orders, rewards."""
    N = int(sc.horizon)
    env = act.env
    base = env.unwrapped
    A = np.zeros((len(seeds), N)); B = np.zeros_like(A); Y = np.zeros_like(A); R = np.zeros_like(A)
    LAM = np.zeros(len(seeds))
    for i, sd in enumerate(seeds):
        obs, _ = env.reset(seed=int(sd))
        LAM[i] = base._scenario_ep.demand.rate        # the hidden rate: for stratification only
        for t in range(N):
            st = base._state
            if info == "timing":
                A[i, t], B[i, t] = sc.prior_shape + st.cum_sales, sc.prior_rate + st.cum_duration
            else:
                A[i, t], B[i, t] = sc.prior_shape + st.cum_demand, sc.prior_rate + st.period
            a = int(act(base, obs))
            Y[i, t] = base.decode_action(a)        # the ORDER (premium_discrete: y_myo + a); step gets the action
            obs, r, done, _, _ = env.step(a)
            R[i, t] = r
            if done:
                break
    return A, B, Y, R, LAM


class _Myopic:
    def __init__(self, sc, mode, info):
        self.env = CnvEnv(scenario=sc, observation_mode=mode)
        self.sc, self.info = sc, info

    def __call__(self, base, obs):
        st, sc = base._state, self.sc
        if self.info == "timing":
            return gamma_poisson_quantile(sc.prior_shape + st.cum_sales, sc.prior_rate + st.cum_duration, sc.critical_ratio)
        return gamma_poisson_quantile(sc.prior_shape + st.cum_demand, sc.prior_rate + st.period, sc.critical_ratio)


class _Model:
    def __init__(self, sc, mode, path):
        from gymnasium.wrappers import FrameStackObservation
        from cnv_ppo_eval import RAW_MODES, _run_action_mode, _vecnorm_for, load_policy
        N = int(sc.horizon)
        self.action_mode = _run_action_mode(Path(path).resolve()) or "order_discrete"

        def env_fn():
            e = CnvEnv(scenario=sc, observation_mode=mode, action_mode=self.action_mode)
            return FrameStackObservation(e, stack_size=N, padding_type="zero") if mode in RAW_MODES else e
        mp = Path(path).resolve()
        self.policy, _ = load_policy(mp, _vecnorm_for(mp, None), True, env_fn)
        self.env = env_fn()

    def __call__(self, base, obs):
        return self.policy(base, obs)


class _Dp:
    def __init__(self, sc, table):
        self.env = CnvEnv(scenario=sc, observation_mode="timing_belief")
        self.table = table

    def __call__(self, base, obs):
        return self.table.act(base)


def ledger(sc, A, B, Y, R, LAM, myo_paths, buckets):
    """Per-bucket and total C, G, deviation, observation time."""
    r, c, q = float(sc.price), float(sc.unit_cost), float(sc.critical_ratio)
    n, N = A.shape
    a, b, y = A.ravel(), B.ravel(), Y.ravel()
    ym = np.array([gamma_poisson_quantile(ai, bi, q) for ai, bi in zip(a, b)], dtype=float)
    Lp = expected_profit(a, b, y, r, c).reshape(n, N)
    Lm = expected_profit(a, b, ym, r, c).reshape(n, N)
    Tp = expected_obs_time(a, b, y).reshape(n, N)
    Tm = expected_obs_time(a, b, ym).reshape(n, N)
    Am, Bm, Ym, Rm, _ = myo_paths
    LM = expected_profit(Am.ravel(), Bm.ravel(), Ym.ravel(), r, c).reshape(n, N)
    TM = expected_obs_time(Am.ravel(), Bm.ravel(), Ym.ravel()).reshape(n, N)
    ym = ym.reshape(n, N); dev = Y - ym
    C = Lm - Lp; G = Lm - LM
    Cov, Cun = np.where(dev > 0, C, 0.0), np.where(dev < 0, C, 0.0)   # over-ordering = buying info; under = never
    rows = []
    for lo, hi in buckets:
        sl = slice(lo, hi)
        rows.append(dict(
            periods=f"{lo}-{hi - 1}", ttg=f"{N - lo}-{N - hi + 1}",
            dev=dev[:, sl].mean(), over=(dev[:, sl] > 0).mean(), under=(dev[:, sl] < 0).mean(),
            C=C[:, sl].sum(1).mean(), Cov=Cov[:, sl].sum(1).mean(), Cun=Cun[:, sl].sum(1).mean(), G=G[:, sl].sum(1).mean(),
            T_pol=Tp[:, sl].mean(), T_myo_here=Tm[:, sl].mean(), T_myo_own=TM[:, sl].mean()))
    tot = dict(C=C.sum(1).mean(), C_se=C.sum(1).std(ddof=1) / np.sqrt(n), Cov=Cov.sum(1).mean(), Cun=Cun.sum(1).mean(),
               G=G.sum(1).mean(), G_se=G.sum(1).std(ddof=1) / np.sqrt(n),
               pred_gap=(Lp.sum(1) - LM.sum(1)).mean(),
               real_gap=(R.sum(1) - Rm.sum(1)).mean(),
               real_gap_se=(R.sum(1) - Rm.sum(1)).std(ddof=1) / np.sqrt(n),
               real_pol=R.sum(1).mean(), real_myo=Rm.sum(1).mean())
    return rows, tot


LAM_BANDS = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 25), (25, np.inf)]


# ---------------------------------------------------------------------------------------------------
class EventBelief:
    """The exact event-observation posterior as the benchmark carries it (MyopicEvent's grid), exposed as a
    per-episode tracker with the statistics the ledger needs: posterior mean / CV / log mean, the myopic
    order (the bar's order at the same history) and L(y | w) = r E[min(D, y)] - c y under its predictive."""
    def __init__(self, base_env):
        self.mb = MyopicEvent(base_env)
        sc = base_env._scenario_ep
        self.r, self.c, self.q = float(sc.price), float(sc.unit_cost), float(sc.critical_ratio)
        self._w = None; self._pred_cdf = None

    def observe(self, base_env) -> int:
        """Absorb the last period's observation (once) and return y_myo at the current belief."""
        y = self.mb(base_env)
        w = np.exp(self.mb.log_w); w /= w.sum()
        self._w = w; self._pred_cdf = w @ self.mb.cdf_table
        return int(y)

    @property
    def w(self): return self._w

    def stats(self):
        w, lam = self._w, self.mb.lam
        m = float(w @ lam); v = float(w @ (lam * lam)) - m * m
        return m, float(np.sqrt(max(v, 0.0)) / max(m, 1e-12)), float(np.log(max(m, 1e-12)))

    def L(self, y: int) -> float:
        y = int(y)
        if y <= 0:
            return 0.0
        emin = float((1.0 - self._pred_cdf[:y]).sum())
        return self.r * emin - self.c * y


# EVENT observation (2026-10-06): the belief is the benchmark's grid posterior (``EventBelief`` below — the same object the myopic_event bar plays),
# not a Gamma, so the per-step quantities L(y | b_t), y_myo(b_t) are computed on the fly and only scalars
# are kept. Same identities: E[pol] - E[myopic] = G - C in expectation; G == 0 only under full observation.
# No observation TIME exists under event; the stock-out share replaces the E[T] columns.
# ---------------------------------------------------------------------------------------------------
def rollout_event(sc, act, seeds):
    """act(base, obs) on seeds; per step: the order, reward, y_myo at the policy's own grid belief, L(y_pol | b), L(y_myo | b)."""
    N = int(sc.horizon); env = act.env; base = env.unwrapped
    Y = np.zeros((len(seeds), N)); R = np.zeros_like(Y); YM = np.zeros_like(Y); LP = np.zeros_like(Y); LM = np.zeros_like(Y); SO = np.zeros_like(Y)
    LAM = np.zeros(len(seeds))
    for i, sd in enumerate(seeds):
        obs, _ = env.reset(seed=int(sd)); eb = EventBelief(base)
        LAM[i] = base._scenario_ep.demand.rate
        for t in range(N):
            ym = eb.observe(base)
            a = int(act(base, obs)); y = int(base.decode_action(a))
            YM[i, t], Y[i, t], LP[i, t], LM[i, t] = ym, y, eb.L(y), eb.L(ym)
            obs, r, done, _, info = env.step(a)
            R[i, t] = r; SO[i, t] = int(info["stockout_event"])
            if done:
                break
    return Y, R, LAM, YM, LP, LM, SO


class _MyopicEvent:
    """The bar's own policy (cnv_benchmark_myopic.MyopicEvent), re-created per episode as the benchmark does."""
    def __init__(self, sc):
        self.env = CnvEnv(scenario=sc, observation_mode="event_raw"); self._eb = None; self._ep = None

    def __call__(self, base, obs):
        if self._eb is None or self._ep != base._episode_seed:   # a new episode: a fresh grid belief, as the benchmark re-creates it
            self._eb = EventBelief(base); self._ep = base._episode_seed
        return self._eb.observe(base)


def ledger_event(sc, pol, myo, buckets):
    Y, R, LAM, YM, LP, LM, SO = pol; Ym, Rm, _, _, _, LMm, SOm = myo
    n, N = Y.shape; dev = Y - YM; C = LM - LP; G = LM - LMm
    Cov, Cun = np.where(dev > 0, C, 0.0), np.where(dev < 0, C, 0.0)
    rows = []
    for lo, hi in buckets:
        sl = slice(lo, hi)
        rows.append(dict(periods=f"{lo}-{hi - 1}", ttg=f"{N - lo}-{N - hi + 1}",
                         dev=dev[:, sl].mean(), over=(dev[:, sl] > 0).mean(), under=(dev[:, sl] < 0).mean(),
                         C=C[:, sl].sum(1).mean(), Cov=Cov[:, sl].sum(1).mean(), Cun=Cun[:, sl].sum(1).mean(), G=G[:, sl].sum(1).mean(),
                         T_pol=SO[:, sl].mean(), T_myo_here=np.nan, T_myo_own=SOm[:, sl].mean()))
    tot = dict(C=C.sum(1).mean(), C_se=C.sum(1).std(ddof=1) / np.sqrt(n), Cov=Cov.sum(1).mean(), Cun=Cun.sum(1).mean(),
               G=G.sum(1).mean(), G_se=G.sum(1).std(ddof=1) / np.sqrt(n),
               pred_gap=(LP.sum(1) - LMm.sum(1)).mean(),
               real_gap=(R.sum(1) - Rm.sum(1)).mean(), real_gap_se=(R.sum(1) - Rm.sum(1)).std(ddof=1) / np.sqrt(n),
               real_pol=R.sum(1).mean(), real_myo=Rm.sum(1).mean())
    return rows, tot


def stratified_event(sc, pol, myo, title):
    """As `stratified`, with y_myo from the grid belief; every term scored at the TRUE rate."""
    from scipy.stats import poisson
    r, c = float(sc.price), float(sc.unit_cost)
    Y, R, LAM, YM, _, _, SO = pol; Ym, Rm, LAMm, _, _, _, SOm = myo
    assert np.allclose(LAM, LAMm)
    n, N = Y.shape; lam_steps = np.repeat(LAM, N)
    Lp = expected_profit_true(lam_steps, Y.ravel(), r, c).reshape(n, N)
    Lm = expected_profit_true(lam_steps, YM.ravel(), r, c).reshape(n, N)
    LM = expected_profit_true(lam_steps, Ym.ravel(), r, c).reshape(n, N)
    dev = Y - YM; C = Lm - Lp; G = Lm - LM
    Cov, Cun = np.where(dev > 0, C, 0.0), np.where(dev < 0, C, 0.0)
    print(title)
    print(f"  {'lambda':>9s} {'n':>5s} {'myopic':>8s} {'gap real':>9s} {'G-C':>7s} {'C_over':>7s} {'C_under':>8s} {'G':>7s}"
          f" {'dev t0':>7s} {'dev 1-4':>8s} {'dev 5-19':>8s} {'dev 20+':>8s} {'so% 0-4 pol/myo':>16s} {'locked0% pol/myo':>17s}")
    for lo, hi in LAM_BANDS:
        m = (LAM >= lo) & (LAM < hi)
        if not m.any():
            continue
        gap = (R[m].sum(1) - Rm[m].sum(1))
        print(f"  {f'[{lo:g},{hi:g})':>9s} {m.sum():5d} {Rm[m].sum(1).mean():8.2f} {gap.mean():+9.2f} "
              f"{(G[m].sum(1) - C[m].sum(1)).mean():+7.2f} {Cov[m].sum(1).mean():7.2f} {Cun[m].sum(1).mean():8.2f} {G[m].sum(1).mean():+7.2f}"
              f" {dev[m, 0].mean():+7.2f} {dev[m, 1:5].mean():+8.2f} {dev[m, 5:20].mean():+8.2f} {dev[m, 20:].mean():+8.2f}"
              f" {100 * SO[m, :5].mean():7.0f}/{100 * SOm[m, :5].mean():<7.0f}"
              f" {100 * (Y[m, 20:] == 0).all(1).mean():8.1f}/{100 * (Ym[m, 20:] == 0).all(1).mean():<7.1f}")
    gap = R.sum(1) - Rm.sum(1)
    print(f"  {'all':>9s} {n:5d} {Rm.sum(1).mean():8.2f} {gap.mean():+9.2f} {(G.sum(1) - C.sum(1)).mean():+7.2f} "
          f"{Cov.sum(1).mean():7.2f} {Cun.sum(1).mean():8.2f} {G.sum(1).mean():+7.2f}")


def expected_profit_true(lam, y, r, c, chunk=20000):
    """r E[min(D, y)] - c y with D ~ Poisson(lam): the period's expected profit at the TRUE rate."""
    from scipy.stats import poisson
    lam, y = np.asarray(lam, dtype=float), np.asarray(y, dtype=float)
    out = np.empty(len(lam))
    for s in range(0, len(lam), chunk):
        ll, yy = lam[s:s + chunk], y[s:s + chunk]
        K = int(yy.max()) if len(yy) else 0
        if K == 0:
            out[s:s + chunk] = 0.0
            continue
        k = np.arange(K)[None, :]
        emin = np.where(k < yy[:, None], poisson.sf(k, ll[:, None]), 0.0).sum(axis=1)
        out[s:s + chunk] = r * emin - c * yy
    return out


def stratified(sc, pol, myo, title):
    """The ledger per stratum of the hidden rate, every term scored at the TRUE rate (the predictive is not
    unbiased given lambda). C_lam can be < 0 (an over-order that paid immediately); G_lam - C_lam is the
    expected realised gap given lambda and the two belief paths."""
    r, c, q = float(sc.price), float(sc.unit_cost), float(sc.critical_ratio)
    A, B, Y, R, LAM = pol; Am, Bm, Ym, Rm, LAMm = myo
    assert np.allclose(LAM, LAMm), "policy and myopic paths must share the seeds"
    n, N = A.shape
    lam_steps = np.repeat(LAM, N)
    ym = np.array([gamma_poisson_quantile(x, z, q) for x, z in zip(A.ravel(), B.ravel())], dtype=float)
    Lp = expected_profit_true(lam_steps, Y.ravel(), r, c).reshape(n, N)
    Lm = expected_profit_true(lam_steps, ym, r, c).reshape(n, N)
    LM = expected_profit_true(lam_steps, Ym.ravel(), r, c).reshape(n, N)
    dev = Y - ym.reshape(n, N); C = Lm - Lp; G = Lm - LM
    Cov, Cun = np.where(dev > 0, C, 0.0), np.where(dev < 0, C, 0.0)
    # stock-out share per period at the true rate: P(D >= y) — how often the order was censored
    from scipy.stats import poisson
    so_p = np.where(Y > 0, poisson.sf(Y - 1, LAM[:, None]), 1.0); so_m = np.where(Ym > 0, poisson.sf(Ym - 1, LAM[:, None]), 1.0)
    print(title)
    print(f"  {'lambda':>9s} {'n':>5s} {'myopic':>8s} {'gap real':>9s} {'G-C':>7s} {'C_over':>7s} {'C_under':>8s} {'G':>7s}"
          f" {'dev t0':>7s} {'dev 1-4':>8s} {'dev 5-19':>8s} {'dev 20+':>8s} {'so% 0-4 pol/myo':>16s} {'locked0% pol/myo':>17s}")
    for lo, hi in LAM_BANDS:
        m = (LAM >= lo) & (LAM < hi)
        if not m.any():
            continue
        gap = (R[m].sum(1) - Rm[m].sum(1))
        print(f"  {f'[{lo:g},{hi:g})':>9s} {m.sum():5d} {Rm[m].sum(1).mean():8.2f} {gap.mean():+9.2f} "
              f"{(G[m].sum(1) - C[m].sum(1)).mean():+7.2f} {Cov[m].sum(1).mean():7.2f} {Cun[m].sum(1).mean():8.2f} {G[m].sum(1).mean():+7.2f}"
              f" {dev[m, 0].mean():+7.2f} {dev[m, 1:5].mean():+8.2f} {dev[m, 5:20].mean():+8.2f} {dev[m, 20:].mean():+8.2f}"
              f" {100 * so_p[m, :5].mean():7.0f}/{100 * so_m[m, :5].mean():<7.0f}"
              f" {100 * (Y[m, 20:] == 0).all(1).mean():8.1f}/{100 * (Ym[m, 20:] == 0).all(1).mean():<7.1f}")
    gap = R.sum(1) - Rm.sum(1)
    print(f"  {'all':>9s} {n:5d} {Rm.sum(1).mean():8.2f} {gap.mean():+9.2f} {(G.sum(1) - C.sum(1)).mean():+7.2f} "
          f"{Cov.sum(1).mean():7.2f} {Cun.sum(1).mean():8.2f} {G.sum(1).mean():+7.2f}")


def print_ledger(title, rows, tot):
    print(title)
    print(f"  realised: policy {tot['real_pol']:.2f}  myopic {tot['real_myo']:.2f}  paired {tot['real_gap']:+.2f} ± {tot['real_gap_se']:.2f}")
    print(f"  ledger:   C (paid) {tot['C']:.2f} ± {tot['C_se']:.2f} [over {tot['Cov']:.2f} / under {tot['Cun']:.2f}]   G (information got) {tot['G']:+.2f} ± {tot['G_se']:.2f}   "
          f"G - C {tot['G'] - tot['C']:+.2f}   (predictive gap {tot['pred_gap']:+.2f})")
    print(f"  {'periods':>8s} {'ttg':>8s} {'y-y_myo':>8s} {'over%':>6s} {'under%':>7s} {'C':>7s} {'C_over':>7s} {'C_under':>7s} {'G':>7s} "
          f"{'E[T] pol':>9s} {'myo@here':>9s} {'myo own':>8s}")
    for x in rows:
        print(f"  {x['periods']:>8s} {x['ttg']:>8s} {x['dev']:+8.2f} {100 * x['over']:6.1f} {100 * x['under']:7.1f} "
              f"{x['C']:7.2f} {x['Cov']:7.2f} {x['Cun']:7.2f} {x['G']:+7.2f} {x['T_pol']:9.3f} {x['T_myo_here']:9.3f} {x['T_myo_own']:8.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path")
    p.add_argument("-s", "--scenario_name", default="fig6b")
    p.add_argument("-o", "--observation_mode", default="timing_belief_log")
    p.add_argument("--n-seeds", type=int, default=2048)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--dp-sweep", action="store_true")
    p.add_argument("--by-lambda", action="store_true", help="the ledger per stratum of the hidden rate, at the true rate")
    p.add_argument("--cells", type=str, default=None, help="dp sweep: comma-separated unit costs to keep")
    a = p.parse_args()
    seeds = eval_seed_block(a.first_seed, a.n_seeds)

    if a.dp_sweep:
        from cnv_benchmark_dp import DpTable
        from cnv_grids import GRIDS, cell_slug
        print("N=4 exact timing DP vs myopic_timing at the fig6 prior Gamma(10/16, 1/16), "
              f"{len(seeds)} seeds per cell; myopic_full - myopic_timing = the censoring loss (an upper bound on G)")
        print(f"  {'c':>4s} {'ratio':>5s} {'y0 dp':>5s} {'y0 myo':>6s} {'myo_T':>7s} {'dp_T':>7s} {'myo_F':>7s} "
              f"{'censor':>7s} {'C_dp':>6s} {'G_dp':>6s} {'G/cens':>6s}  y-y_myo by period")
        keep = None if a.cells is None else {float(x) for x in a.cells.split(",")}
        for cid, src in GRIDS["fig3"].cells:
            sc = src
            if keep is not None and float(sc.unit_cost) not in keep:
                continue
            tp = Path("results/paper510/benchmark/dp") / f"c{float(sc.unit_cost):g}_b{float(sc.prior_rate):g}_m{float(sc.prior_mean):g}_timing.npz"
            tab = DpTable.load(tp) if tp.exists() else None
            if tab is None:
                print(f"  {cid}: no table"); continue
            q = float(sc.critical_ratio)
            myoT = rollout(sc, _Myopic(sc, "timing_belief", "timing"), seeds, "timing")
            myoF = rollout(sc, _Myopic(sc, "full_belief", "full"), seeds, "full")
            dp = rollout(sc, _Dp(sc, tab), seeds, "timing")
            rows, tot = ledger(sc, *dp, myoT, [(t, t + 1) for t in range(int(sc.horizon))])
            y0d = int(tab.orders[0, 0, 0]); y0m = gamma_poisson_quantile(sc.prior_shape, sc.prior_rate, q)
            cens = myoF[3].sum(1).mean() - myoT[3].sum(1).mean()
            print(f"  {sc.unit_cost:4.2f} {q:5.3f} {y0d:5d} {y0m:6d} {myoT[3].sum(1).mean():7.3f} {dp[3].sum(1).mean():7.3f} "
                  f"{myoF[3].sum(1).mean():7.3f} {cens:7.3f} {tot['C']:6.3f} {tot['G']:+6.3f} "
                  f"{(tot['G'] / cens if cens > 1e-9 else float('nan')):6.2f}  " + " ".join(f"{x['dev']:+.2f}" for x in rows))
            if a.by_lambda:
                stratified(sc, dp, myoT, f"  -- N=4 DP, c={sc.unit_cost:g}, by the hidden rate (true-rate scoring)")
        return

    from cnv_scenarios import SCENARIOS
    sc = SCENARIOS[a.scenario_name]
    info = "timing" if a.observation_mode.startswith("timing") else "event" if a.observation_mode.startswith("event") else "full"
    N = int(sc.horizon)
    model = _Model(sc, a.observation_mode, a.model_path)
    if info == "event":
        buckets = [(0, 1), (1, 5), (5, 20), (20, 40), (40, 60), (60, 80), (80, 99), (99, 100)]
        pol = rollout_event(sc, model, seeds)
        myo = rollout_event(sc, _MyopicEvent(sc), seeds)
        title = (f"{a.scenario_name} · {a.observation_mode} · {Path(a.model_path).parent.name[:60]}/{Path(a.model_path).name} · "
                 f"{len(seeds)} seeds from {a.first_seed} · info=event (grid posterior; E[T] columns = stock-out share)")
        if a.by_lambda:
            stratified_event(sc, pol, myo, title + " · by the hidden rate (true-rate scoring)")
            return
        rows, tot = ledger_event(sc, pol, myo, buckets)
        print_ledger(title, rows, tot)
        return
    pol = rollout(sc, model, seeds, info)
    if model.action_mode == "premium_discrete":
        from cnv_gym import PREMIUM_MAX
        A_, B_, Y_ = pol[0], pol[1], pol[2]
        ym_ = np.vectorize(lambda x, z: gamma_poisson_quantile(x, z, float(sc.critical_ratio)))(A_, B_)
        prem = Y_ - ym_
        print(f"  premium_discrete: mean premium {prem.mean():.3f}, share of periods at the cap a = {PREMIUM_MAX}: "
              f"{100 * (prem >= PREMIUM_MAX).mean():.2f}%  (min premium {prem.min():.0f}: must be >= 0)")
    myo = rollout(sc, _Myopic(sc, "timing_belief" if info == "timing" else "full_belief", info), seeds, info)
    buckets = ([(t, t + 1) for t in range(N)] if N <= 10 else
               [(0, 1), (1, 5), (5, 20), (20, 40), (40, 60), (60, 80), (80, 99), (99, 100)])
    if a.by_lambda:
        stratified(sc, pol, myo, f"{a.scenario_name} · {a.observation_mode} · {Path(a.model_path).parent.name[:50]}/{Path(a.model_path).name}"
                   f" · {len(seeds)} seeds · by the hidden rate (true-rate scoring)")
        return
    rows, tot = ledger(sc, *pol, myo, buckets)
    print_ledger(f"{a.scenario_name} · {a.observation_mode} · {Path(a.model_path).parent.name[:60]}/{Path(a.model_path).name} · "
                 f"{len(seeds)} seeds from {a.first_seed} · info={info}", rows, tot)
    # by belief: where (in posterior mean) the deviations sit, all periods pooled
    A, B, Y, R, LAM = pol
    m = (A / B).ravel(); dv = (Y.ravel() - np.array([gamma_poisson_quantile(x, z, float(sc.critical_ratio)) for x, z in zip(A.ravel(), B.ravel())]))
    cv = (1.0 / np.sqrt(A)).ravel()
    print(f"  by posterior mean (pooled periods): " + "  ".join(
        f"[{lo:g},{hi:g}) n={((m >= lo) & (m < hi)).mean() * 100:.0f}% dev {dv[(m >= lo) & (m < hi)].mean():+.2f}"
        for lo, hi in [(0, 1), (1, 3), (3, 8), (8, 20), (20, 1e9)] if ((m >= lo) & (m < hi)).any()))
    print(f"  by posterior CV (pooled periods):   " + "  ".join(
        f"[{lo:g},{hi:g}) n={((cv >= lo) & (cv < hi)).mean() * 100:.0f}% dev {dv[(cv >= lo) & (cv < hi)].mean():+.2f}"
        for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 10)] if ((cv >= lo) & (cv < hi)).any()))


if __name__ == "__main__":
    main()

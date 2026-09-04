"""Every scored number in `INTERPRET.md`, reproducible in one place.

The §14 readback quotes ~40 costs across four comparisons. They were produced by
one-off scripts in `scratch/`, which is gitignored — so the document cited
evidence a reader could not re-run, which this folder's own file-hygiene rule
forbids ("an escalation entry cites numbers that only exist if the code behind
them can be re-run"). This is that code.

Every arm holds the seed block and the harness fixed and moves ONE thing, so
the deltas are paired and attributable:

    --factorial   order x protection 2x2            INTERPRET §8
    --protection  sigma rules, order held at AP     INTERPRET §5, §6
    --order       order rules, sigma held at PL     INTERPRET §8b
    --structure   (s,S) and Prop 3 readback         INTERPRET §8b
    --costs       setup/holding/backorder split     INTERPRET §8b

    python adi_flex_readback_score.py --all --n-seeds 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adi_flex_benchmark_rule import protection_ladder
from adi_flex_fixed_order import U_MIN, FixedOrderProtectEnv
from adi_flex_gym import AdiFlexEnv
from adi_flex_policy_probe import CROWN, CROWNED, _load, _resolve
from adi_flex_scenarios import SCENARIOS

HERE = Path(__file__).resolve().parent
SCEN = "het3_exp2"
CFG = CROWN                     # overridden by --config; a0 ids reproduce the 2026-08-31 readback


def _setup(cfg=None, scen=SCEN):
    cfg = cfg or CFG
    sc = SCENARIOS[scen]
    _, obs, mp, vp = _resolve(cfg)
    pol = _load(mp, vp)
    ap = HERE / f"results/{scen}/ap/{scen}_policy.npz"
    z = np.load(ap)
    Y = z["y"] if "y" in z.files else z[z.files[0]]
    umin = int(z["u_min"]) if "u_min" in z.files else U_MIN
    dmax = int(z["d_max"]) if "d_max" in z.files else Y.shape[-1] - 1
    return sc, obs, pol, str(ap), Y, umin, dmax


def _apq(Y, umin, dmax, order_max, period, u, vhat, delta=0):
    nu = Y.shape[1] if Y.ndim == 3 else Y.shape[0]
    ui = int(np.clip(u - umin, 0, nu - 1)); vi = int(np.clip(vhat, 0, dmax))
    y = int(Y[period, ui, vi] if Y.ndim == 3 else Y[ui, vi]) - delta
    return int(np.clip(y - u, 0, order_max))


def _roll(sc, obs, pol, order_fn, sigma_fn, n, collect=None):
    """One env, both decisions supplied by callables. `order_fn=None` means the
    AP-fixed wrapper drives the order (the #E13 instrument)."""
    env = (AdiFlexEnv(scenario=sc, action_mode="order_protection", observation_mode=obs)
           if order_fn is not None else None)
    tot = np.zeros(n)
    for ep in range(n):
        o, _ = env.reset(seed=ep)
        done = False
        while not done:
            st = env._state
            u = int(st.inv + sum(st.pipe) - sum(st.adv))
            d = (env._info or {}).get("d", ())
            vh = int(d[sc.T_dl]) if len(d) > sc.T_dl else 0
            q = order_fn(o, st, u, vh)
            sg = sigma_fn(o, st, u, vh)
            if collect is not None:
                collect.append(dict(period=st.period, u=u, vhat=vh, q=q, y=u + q))
            o, _, t, tr, _ = env.step([q, *sg])
            done = t or tr
        tot[ep] = env.total_reward
    env.close()
    return -tot.mean()


def _net(pol, which):
    def f(o, st, u, vh):
        a, _ = pol.predict(np.asarray(o)[None, :], deterministic=True)
        a = np.asarray(a).reshape(-1)
        return int(a[0]) if which == "order" else [int(x) for x in a[1:]]
    return f


def _table(rows, base_key, target_key=None, label="of the gap"):
    base = rows[base_key]
    tgt = rows.get(target_key)
    w = max(len(k) for k in rows)
    head = f"  {'arm':<{w}}{'cost':>11}{'vs base':>10}"
    if tgt is not None:
        head += f"  {label:>11}"
    print(head)
    for k, v in rows.items():
        line = f"  {k:<{w}}{v:11.4f}{v-base:+10.4f}"
        if tgt is not None:
            line += f"  {(base-v)/(base-tgt)*100:10.0f}%"
        print(line)


def factorial(n):
    sc, obs, pol, ap, Y, umin, dmax = _setup()
    lad = list(protection_ladder(sc, "plsigma"))
    pl = lambda o, st, u, vh: list(lad)
    apf = lambda o, st, u, vh: _apq(Y, umin, dmax, sc.order_max, st.period, u, vh)
    print(f"\n=== order x protection 2x2 (INTERPRET §8) · {n} CRN ===")
    r = {"AP order   + PL(4,1)": _roll(sc, obs, pol, apf, pl, n),
         "AP order   + net σ":   _roll(sc, obs, pol, apf, _net(pol, "sigma"), n),
         "net order  + PL(4,1)": _roll(sc, obs, pol, _net(pol, "order"), pl, n),
         "net order  + net σ":   _roll(sc, obs, pol, _net(pol, "order"), _net(pol, "sigma"), n)}
    _table(r, "AP order   + PL(4,1)", "net order  + net σ")
    mo = r["net order  + PL(4,1)"] - r["AP order   + PL(4,1)"]
    mp_ = r["AP order   + net σ"] - r["AP order   + PL(4,1)"]
    both = r["net order  + net σ"] - r["AP order   + PL(4,1)"]
    print(f"\n  ORDER alone {mo:+8.4f}   PROTECTION alone {mp_:+8.4f}"
          f"   interaction {both-mo-mp_:+.4f}")


def protection(n):
    sc, obs, pol, ap, Y, umin, dmax = _setup()
    lad = list(protection_ladder(sc, "plsigma"))
    apf = lambda o, st, u, vh: _apq(Y, umin, dmax, sc.order_max, st.period, u, vh)
    last = sc.N - 1
    k = lambda f: (lambda o, st, u, vh: f(st.period))
    print(f"\n=== σ rules, order held at AP (INTERPRET §5, §6) · {n} CRN ===")
    r = {"PL(σ) ladder (4,1) — the paper": _roll(sc, obs, pol, apf, k(lambda i: list(lad)), n),
         "best constant (4,2)":            _roll(sc, obs, pol, apf, k(lambda i: [4, 2]), n),
         "(4,1) + zero in the last period": _roll(sc, obs, pol, apf, k(lambda i: [0,0] if i==last else list(lad)), n),
         "(4,2) + zero in the last period": _roll(sc, obs, pol, apf, k(lambda i: [0,0] if i==last else [4,2]), n),
         "the net's own σ":                 _roll(sc, obs, pol, apf, _net(pol, "sigma"), n)}
    _table(r, "PL(σ) ladder (4,1) — the paper")


def order(n):
    sc, obs, pol, ap, Y, umin, dmax = _setup()
    lad = list(protection_ladder(sc, "plsigma"))
    pl = lambda o, st, u, vh: list(lad)
    def rule(cut, dl):
        return lambda o, st, u, vh: (0 if st.period >= sc.N - cut
                                     else _apq(Y, umin, dmax, sc.order_max, st.period, u, vh, dl))
    print(f"\n=== order rules, σ held at PL(4,1) (INTERPRET §8b) · {n} CRN ===")
    r = {"AP as published":                  _roll(sc, obs, pol, rule(0,0), pl, n),
         "AP, y* − 2":                       _roll(sc, obs, pol, rule(0,2), pl, n),
         "AP, no order in the last 2":       _roll(sc, obs, pol, rule(2,0), pl, n),
         "AP, no order last 3 + y* − 4":     _roll(sc, obs, pol, rule(3,4), pl, n),
         "the net's own order":              _roll(sc, obs, pol, _net(pol,"order"), pl, n)}
    _table(r, "AP as published", "the net's own order", "of the order gap")


def structure(n):
    sc, obs, pol, ap, Y, umin, dmax = _setup()
    lad = list(protection_ladder(sc, "plsigma"))
    rows = []
    _roll(sc, obs, pol, _net(pol, "order"), lambda o, st, u, vh: list(lad),
          max(n // 8, 200), collect=rows)
    P = np.array([r["period"] for r in rows]); U = np.array([r["u"] for r in rows])
    V = np.array([r["vhat"] for r in rows]); Q = np.array([r["q"] for r in rows])
    Yv = np.array([r["y"] for r in rows])
    print(f"\n=== (s, S) readback (INTERPRET §8b) · {len(rows)} decisions ===")
    print("\n  by PERIOD — the horizon taper")
    print(f"    {'period':>7}{'ttg':>5}{'orders':>9}{'s':>6}{'S':>9}")
    for i in sorted(set(P)):
        m = P == i; on = m & (Q > 0)
        if not on.sum():
            print(f"    {i:>7}{sc.N-i:>5}{0:>9}{'—':>6}{'never':>9}"); continue
        print(f"    {i:>7}{sc.N-i:>5}{on.sum()/m.sum():>8.1%}{U[on].max():>6}{Yv[on].mean():>9.2f}")
    print("\n  by V̂, pooled over the periods that order — Wang & Toktay Prop 3")
    print("  (S constant in V̂, s decreasing in V̂)")
    print(f"    {'V̂':>4}{'n':>7}{'orders':>8}{'s':>6}{'S':>9}")
    mid = (P >= 5) & (P <= 7)
    for v in sorted(set(V[mid])):
        m = mid & (V == v); on = m & (Q > 0)
        if m.sum() < 30 or on.sum() < 5: continue
        print(f"    {v:>4}{m.sum():>7}{on.sum():>8}{U[on].max():>6}{Yv[on].mean():>9.2f}")


def costs(n):
    import adi_flex_mdp as M
    sc, obs, pol, ap, Y, umin, dmax = _setup()
    lad = list(protection_ladder(sc, "plsigma"))
    apf = lambda o, st, u, vh: _apq(Y, umin, dmax, sc.order_max, st.period, u, vh)
    pl = lambda o, st, u, vh: list(lad)
    orig1, orig2 = M.advance1, M.advance2
    acc = {}
    def wrap(fn):
        def g(*a, **k):
            out = fn(*a, **k)
            info = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
            c = info.get("cost", {}) if isinstance(info, dict) else {}
            for key in ("order_fixed", "holding", "backorder"):
                if key in c: acc[key] = acc.get(key, 0.0) + float(c[key])
            return out
        return g
    M.advance1, M.advance2 = wrap(orig1), wrap(orig2)
    print(f"\n=== per-episode cost decomposition (INTERPRET §8b) · {n} CRN ===")
    print(f"  {'policy':<26}{'total':>9}{'setup':>9}{'holding':>9}{'backorder':>11}")
    try:
        for nm, of, sf in (("AP order + PL(4,1)", apf, pl),
                           ("net order + PL(4,1)", _net(pol,"order"), pl)):
            acc.clear()
            t = _roll(sc, obs, pol, of, sf, n)
            print(f"  {nm:<26}{t:9.2f}{acc.get('order_fixed',0)/n:9.2f}"
                  f"{acc.get('holding',0)/n:9.2f}{acc.get('backorder',0)/n:11.2f}")
    finally:
        M.advance1, M.advance2 = orig1, orig2


def fig4(n, path="figures/fig4_order_sS.svg"):
    """INTERPRET §8b's figure: u AFTER ordering against u BEFORE, the net's order
    (its own σ held at PL(4,1) so the state distribution is the same for both)
    and AP's y* on the identical states. An (s,S) policy draws the 45° no-order
    locus for u ≥ s and a horizontal run at S for u < s; what differs between the
    two is where the lines sit. Committed generator for a figure that was first
    produced by a scratch script (F-note in #E26)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sc, obs, pol, ap, Y, umin, dmax = _setup()
    lad = list(protection_ladder(sc, "plsigma"))
    rows = []
    _roll(sc, obs, pol, _net(pol, "order"), lambda o, st, u, vh: list(lad),
          max(n // 8, 200), collect=rows)
    U = np.array([r["u"] for r in rows]); P = np.array([r["period"] for r in rows])
    V = np.array([r["vhat"] for r in rows]); Ynet = np.array([r["y"] for r in rows])
    Yap = np.array([u + _apq(Y, umin, dmax, sc.order_max, int(p_), u, v)
                    for u, p_, v in zip(U, P, V)])
    f, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    lo, hi = int(U.min()) - 1, int(max(Ynet.max(), Yap.max())) + 1
    for ax, y, name, col in ((axes[0], Ynet, f"the net's order · {CFG}", "#1f6feb"),
                             (axes[1], Yap, "AP's y* on the same states", "#d1622b")):
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, ls="--", label="no order (45°)")
        # the cloud is rasterized (an SVG of ~10k path elements is megabytes);
        # the per-u mean is the vector layer the reader measures s and S from
        ax.scatter(U, y, s=5, alpha=0.2, color=col, rasterized=True)
        # the vector layer: the mean order-up-to level among decisions that DID
        # order, per u (the horizontal run is S; where the curve ends on the
        # right is s). Pooling ordering and no-order decisions would pull the
        # line to the diagonal wherever the late horizon stops ordering.
        on = y > U
        us = np.array(sorted(set(U[on].tolist())))
        ax.plot(us, [y[on & (U == u)].mean() for u in us], color=col, lw=1.8,
                label="mean y among decisions that ordered")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlabel("u before ordering (modified inventory position)")
        ax.set_title(name, fontsize=10); ax.grid(alpha=0.2)
    axes[0].set_ylabel("u after ordering (= y)")
    f.suptitle(f"§8b — the order decision as (s, S) · {len(rows)} on-policy decisions · "
               "the horizontal run is S, its left end is s", fontsize=10)
    f.tight_layout(); (HERE / path).parent.mkdir(exist_ok=True)
    f.savefig(HERE / path, dpi=110); plt.close(f)
    print(f"  {path}: net S≈{np.median(Ynet[Ynet > U]):.1f} (median y where an order happened, n={(Ynet > U).sum()}), "
          f"AP S≈{np.median(Yap[Yap > U]):.1f} (n={(Yap > U).sum()})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for f in ("factorial", "protection", "order", "structure", "costs", "all"):
        p.add_argument(f"--{f}", action="store_true")
    p.add_argument("--n-seeds", type=int, default=2048)
    p.add_argument("--config", default=CROWN, choices=list(CROWNED),
                   help="the artifact to read back (default: the crown)")
    p.add_argument("--fig4", default=None, metavar="PATH",
                   help="also draw INTERPRET §8b's (s,S) figure to PATH (e.g. figures/fig4_order_sS.svg)")
    a = p.parse_args()
    global CFG
    CFG = a.config
    todo = [f for f in ("factorial","protection","order","structure","costs")
            if getattr(a, f) or a.all]
    if not todo and not a.fig4:
        p.error("choose at least one section, --all, or --fig4")
    if a.fig4:
        fig4(a.n_seeds, a.fig4)
    print(f"adi_flex §14 readback scoring · {SCEN} · {CFG} · {a.n_seeds} CRN seeds")
    for f in todo:
        globals()[f](a.n_seeds)


if __name__ == "__main__":
    main()

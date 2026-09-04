"""Figures for the §14 readback: what the crowned policies' protection levels do.

STRUCTURE. Every quantity here is an INTEGER — sigma is a MultiDiscrete
component, and inv/adv/period are counts — so a plain scatter draws a few
hundred points on top of each other and shows density as solid black. Three
choices follow from that:

  * marker AREA encodes how many on-policy states sit at that (x, sigma) cell,
    so the eye reads mass rather than extent. A lone outlier is a dot; the mode
    is a disc.
  * a conditional-mean line (and its IQR band) is overlaid, because the mean is
    the thing the text quotes and it should be visible on the same axes as the
    cloud it summarizes.
  * SMALL MULTIPLES with a shared sigma axis: rows are the two components, and
    columns are either the two arms (fig 1-2) or the state features (fig 3).
    Sharing the y axis is what makes "sigma_2 varies more than sigma_1"
    readable across panels instead of a caption claim.

Plotted against the PHYSICAL state, not the observation vector, so the two arms
are comparable: `vec_mip` never sees adv0/adv1, but the underlying state has
them, and the question is what the policy DOES, not what it is shown.

`--drop-last K` removes the final K periods. The last period is a different
decision problem — nothing ordered can arrive and nothing protected can be used
— so its sigma collapses toward 0 and, plotted with the rest, it reads as
spread when it is really a boundary.

    python adi_flex_plot_policy.py                 # all three, to figures/
    python adi_flex_plot_policy.py --drop-last 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adi_flex_benchmark_rule import protection_ladder
from adi_flex_policy_probe import CROWNED, _load, _resolve
from adi_flex_ppo_eval import build_env
from adi_flex_scenarios import SCENARIOS

HERE = Path(__file__).resolve().parent
LADDER = list(protection_ladder(SCENARIOS["het3_exp2"], "plsigma"))   # (4, 1)
ARMS_BY_HEAD = {   # a1 = the ordinal order head, the crown since #E25; a0 = #E18's categorical crowns
    "a1": [("vec", "sc4/g6/a1/h4", "#1f6feb"), ("vec_mip", "sc4/g7/a1/h5", "#d1622b")],
    "a0": [("vec", "sc4/g6/a0/h4", "#1f6feb"), ("vec_mip", "sc4/g7/a0/h5", "#d1622b")],
}
ARMS = ARMS_BY_HEAD["a1"]
# The PL family's own protection levels, as REFERENCE LINES. These are the
# constants the learned sigma is a departure from, so a panel without them
# shows variation with nothing to judge it against. PL(sigma) is the bar
# (#E15); PL(0) and PL(Sigma) bracket the family and are drawn faint, so the
# span the heuristics occupy is visible without competing with the data.
PL_REFS = [("plsigma", "PL(σ) — the bar", "0.25", "--", 1.4, 1.0),
           ("plmax", "PL(Σ)", "0.55", ":", 1.0, 0.75),
           ("pl0", "PL(0)", "0.55", ":", 1.0, 0.75)]


def _pl_lines(ax, k: int, sc, label: bool = False, cumulative: bool = False):
    """Horizontal references for protection component k (0-based).

    `cumulative` switches both sides to the running total. The ACTION
    components are increments — the cascade subtracts each sigma in turn, so by
    the time it reaches adv[k] it has withheld sigma_1 + ... + sigma_k — and
    `protection_ladder` returns increments to match (`out.append(cum - prev)`).
    The two are therefore directly comparable as drawn. But the increment view
    EXAGGERATES: on the flush regime the far component reads 2.3 against
    PL(sigma)'s 1, which is +130%, while the quantity that actually governs
    behaviour — total stock withheld before adv[2] is served — is 6.3 against
    5, or +26%. Both views are drawn because each is misleading alone.
    """
    for pol, lab, col, ls, lw, alpha in PL_REFS:
        lad = protection_ladder(sc, pol)
        if k >= len(lad):
            continue
        lad = np.cumsum(lad) if cumulative else lad
        ax.axhline(lad[k], color=col, ls=ls, lw=lw, alpha=alpha, zorder=2,
                   label=(lab if label else None))
        # the VALUE is annotated per panel, never in the legend: the ladder
        # differs by component (PL(sigma) is 4 for sigma_1 and 1 for sigma_2),
        # so one legend entry carrying a number would be wrong in half the rows
        ax.annotate(f"{lad[k]:g}", xy=(1.0, lad[k]), xycoords=("axes fraction", "data"),
                    xytext=(2, 0), textcoords="offset points", va="center",
                    fontsize=6, color=col, clip_on=False)


def collect_physical(cfg: str, episodes: int):
    """On-policy sigma, the PHYSICAL state, and PER-COMPONENT identification.

    `id1`/`id2` record whether that component actually changed the allocation
    at that event — replayed exactly, by forcing the component to each end of
    its range and comparing. Every figure filters on these, because a sigma
    that reaches an empty box has no effect on the trajectory and the network
    is free to emit anything there (PLAYBOOK P1). Plotting those events is what
    produced the retracted "scarcity regime" in INTERPRET §4c: they are not
    just noise, they are noise biased UPWARD by about four units.

    Identification is PER COMPONENT, so the filter is per panel — sigma_1 can
    bite at an event where sigma_2 cannot. The cumulative panel needs both.
    """
    scen, obs, mp, vp = _resolve(cfg)
    sc = SCENARIOS[scen]
    env = build_env(sc, obs, "order_protection")
    pol = _load(mp, vp)
    casc = env._sigma_cascade
    rows, pending = [], {}

    def spy(mid, sigmas):
        sg = [int(x) for x in sigmas]
        base = casc(mid, sg)
        flags, eff = [], []
        # BINDING, not merely range-identified: the CHOSEN value must change the
        # allocation against no protection at all. The weaker test ("sigma=0 or
        # sigma=max differs") passes on events where the chosen value does
        # nothing but some other value would, which is not the same claim.
        for k in range(sc.n_sigma):
            zero = list(sg)
            zero[k] = 0
            flags.append(casc(mid, zero) != base)
        # EFFECTIVE withholding: min(stock reaching box k, sigma_k). This is the
        # quantity the environment sees. Nominal sigma is unbounded above it —
        # a saturated 17 against 3 units of stock withholds 3 — which is why
        # nominal values near sigma_max survived the earlier filter.
        rem = max(mid.inv, 0)
        for j in range(sc.n_alloc):
            if j >= 1:
                held = min(rem, sg[j - 1])
                eff.append(held)
                rem -= held
            rem -= min(rem, int(mid.adv[j]) if j < len(mid.adv) else 0)
        # what the PAPER'S ladder would have withheld AT THIS SAME STATE.
        # A nominal reference line is not comparable to effective withholding —
        # PL saturates too, and min(rem, 4) is not 4 when rem is 3. Replaying
        # the ladder on the identical mid-state is the matched counterfactual.
        pl, rem = [], max(mid.inv, 0)
        for j in range(sc.n_alloc):
            if j >= 1:
                held = min(rem, LADDER[j - 1])
                pl.append(held)
                rem -= held
            rem -= min(rem, int(mid.adv[j]) if j < len(mid.adv) else 0)
        pending["id"], pending["eff"], pending["pl"] = flags, eff, pl
        return base

    env._sigma_cascade = spy
    for ep in range(episodes):
        o, _ = env.reset(seed=ep)
        done = False
        while not done:
            a, _ = pol.predict(np.asarray(o)[None, :], deterministic=True)
            a = np.asarray(a).reshape(-1)
            s = env._state
            row = dict(period=s.period, ttg=sc.N - s.period,
                       inv=s.inv, pipe=sum(s.pipe), adv=tuple(s.adv),
                       s1=int(a[1]), s2=int(a[2]))
            pending["id"] = pending["eff"] = pending["pl"] = None
            o, _, t, tr, _ = env.step(a)
            flags = pending.get("id") or [False] * sc.n_sigma
            eff = pending.get("eff") or [0] * sc.n_sigma
            row["id1"], row["id2"] = bool(flags[0]), bool(flags[1])
            row["e1"], row["e2"] = int(eff[0]), int(eff[1])
            pl = pending.get("pl") or [0] * sc.n_sigma
            row["p1"], row["p2"] = int(pl[0]), int(pl[1])
            rows.append(row)
            done = t or tr
    env._sigma_cascade = casc
    env.close()
    return rows


def _cloud(ax, x, y, colour, xlabel, ylabel=None, title=None):
    """Count-weighted scatter + conditional mean and IQR band."""
    pts = {}
    for xi, yi in zip(x, y):
        pts[(xi, yi)] = pts.get((xi, yi), 0) + 1
    px = np.array([k[0] for k in pts])
    py = np.array([k[1] for k in pts])
    pn = np.array(list(pts.values()), dtype=float)
    ax.scatter(px, py, s=6 + 220 * np.sqrt(pn / pn.max()), alpha=0.42,
               color=colour, linewidths=0)
    xs = np.unique(x)
    if len(xs) > 18:                      # bin a wide axis so the line is stable
        edges = np.linspace(x.min(), x.max(), 13)
        centres, mu, q1, q3 = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum() >= 12:
                centres.append((lo + hi) / 2); mu.append(y[m].mean())
                q1.append(np.percentile(y[m], 25)); q3.append(np.percentile(y[m], 75))
        xs = np.array(centres)
    else:
        mu, q1, q3 = [], [], []
        keep = []
        for v in xs:
            m = x == v
            # a column holding ONE distinct state repeated once per episode has
            # no dispersion to summarize — period 0 is the deterministic start,
            # so a line through it reads as a trend that is not there
            if m.sum() >= 8 and len(np.unique(y[m])) > 1:
                keep.append(v); mu.append(y[m].mean())
                q1.append(np.percentile(y[m], 25)); q3.append(np.percentile(y[m], 75))
        xs = np.array(keep)
    if len(xs):
        ax.fill_between(xs, q1, q3, color=colour, alpha=0.13, linewidth=0)
        ax.plot(xs, mu, color=colour, lw=2.0, zorder=5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.tick_params(labelsize=7)


def _pl_curve(ax, x, y, label: bool = False):
    """PL(σ)'s EFFECTIVE withholding on the same states, as a conditional mean."""
    xs, mu = [], []
    for v in np.unique(x):
        m = x == v
        if m.sum() >= 8:
            xs.append(v); mu.append(y[m].mean())
    if len(xs) > 18:
        edges = np.linspace(min(xs), max(xs), 13)
        xs2, mu2 = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum() >= 12:
                xs2.append((lo + hi) / 2); mu2.append(y[m].mean())
        xs, mu = xs2, mu2
    if xs:
        ax.plot(xs, mu, color="0.25", ls="--", lw=1.5, zorder=4,
                label=("PL(σ) on the same states" if label else None))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--drop-last", type=int, default=0,
                   help="exclude the final K periods (end-of-horizon effect)")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--head", default="a1", choices=list(ARMS_BY_HEAD),
                   help="which crowns to draw: a1 (ordinal head, ships) or a0 "
                        "(the 2026-08-31 categorical crowns, kept under figures/a0)")
    p.add_argument("--effective", action="store_true",
                   help="plot min(stock, sigma) on all BINDING events, with "
                        "PL(sigma) replayed on the same states. Answers 'how "
                        "much was actually withheld', but BOTH curves then bend "
                        "with available stock, so a constant policy does not "
                        "look constant — which is why it is not the default")
    p.add_argument("--all-events", action="store_true",
                   help="do NOT filter to identified events — reproduces the "
                        "view retracted in INTERPRET §4c. Off by default: an "
                        "action with no effect carries no information")
    a = p.parse_args()
    global ARMS
    ARMS = ARMS_BY_HEAD[a.head]

    out = HERE / a.outdir
    out.mkdir(exist_ok=True)
    data = {nm: collect_physical(cfg, a.episodes) for nm, cfg, _ in ARMS}
    sc = SCENARIOS["het3_exp2"]
    tag = f"last {a.drop_last} period(s) dropped · " if a.drop_last else ""
    tag += ("all events · " if a.all_events else
            "binding events · " if a.effective else "COMPARABLE events · ")
    tag += "σ actually withheld" if a.effective else "nominal σ (fully expressed)"
    tag = "\n" + tag

    def arr(rows, drop, need=None):
        """`need` selects the identification flag(s) this panel requires."""
        rows = [r for r in rows if r["ttg"] > drop]
        if need and not a.all_events:
            # COMPARABLE events. Three conditions, and the last two are what
            # keep a constant policy looking constant:
            #   binding      the chosen sigma changed the allocation
            #   RL expressed sigma <= the stock reaching its box, so the
            #                nominal action IS what was withheld
            #   PL expressed the reference's own level also fits, so its line
            #                is a genuine horizontal constant rather than a
            #                curve bent by the resource
            # Dropping the last two is what made PL(sigma) plot as non-constant.
            for f in need:
                k = f[-1]
                rows = [r for r in rows if r[f"id{k}"]]
                if not a.effective:
                    rows = [r for r in rows
                            if r[f"e{k}"] == r[f"s{k}"] and r[f"p{k}"] == LADDER[int(k) - 1]]
        g = lambda k: np.array([r[k] for r in rows], dtype=float)
        adv = np.array([r["adv"] for r in rows], dtype=float)
        ip = g("inv") + g("pipe")
        d = dict(ttg=g("ttg"), inv=g("inv"), pipe=g("pipe"), ip=ip,
                 u=ip - adv.sum(axis=1), adv=adv, s1=g("s1"), s2=g("s2"),
                 e1=g("e1"), e2=g("e2"), p1=g("p1"), p2=g("p2"))
        if a.effective:              # what was WITHHELD, on every binding event
            d["s1"], d["s2"] = d["e1"], d["e2"]
        return d

    # --- fig 1: sigma vs time_to_go, per arm. NO drop — this is where the
    #     end-of-horizon effect is the subject rather than a nuisance.
    f, axes = plt.subplots(3, 2, figsize=(9.0, 8.4), sharex=True)
    for c, (nm, _, col) in enumerate(ARMS):
        series = [("s1", "σ_1", 0, False, ("id1",)),
                  ("s2", "σ_2", 1, False, ("id2",)),
                  ("cum", "σ_1 + σ_2\n(withheld before adv[2])", 1, True,
                   ("id1", "id2"))]
        for r, (k, ylab, ref, cum, need) in enumerate(series):
            d = arr(data[nm], 0, need)
            y = d["s1"] + d["s2"] if k == "cum" else d[k]
            _cloud(axes[r][c], d["ttg"], y, col,
                   "time to go  (N − period)" if r == 2 else None,
                   ylab if c == 0 else None, f"{nm}" if r == 0 else None)
            if a.effective:
                yp = d["p1"] + d["p2"] if k == "cum" else d[f"p{k[-1]}"]
                _pl_curve(axes[r][c], d["ttg"], yp, label=(r == 0 and c == 1))
            else:
                _pl_lines(axes[r][c], ref, sc, label=(r == 0 and c == 1), cumulative=cum)
    for r in range(3):
        lo = min(ax.get_ylim()[0] for ax in axes[r])
        hi = max(ax.get_ylim()[1] for ax in axes[r])
        for ax in axes[r]:
            ax.set_ylim(lo, hi)
    axes[0][1].legend(fontsize=7, loc="upper left", framealpha=0.9)
    # ONCE — the axes are shared, so inverting each of the four flips it an
    # even number of times and lands back where it started
    axes[0][0].invert_xaxis()          # time flows left -> right
    f.suptitle(f"§2  protection over the horizon{tag}", fontsize=10)
    f.tight_layout(); f.savefig(out / "fig1_sigma_vs_time.svg"); plt.close(f)

    # --- fig 2: sigma vs u, per arm, tail dropped
    f, axes = plt.subplots(3, 2, figsize=(9.0, 8.4), sharex=True)
    for c, (nm, _, col) in enumerate(ARMS):
        series = [("s1", "σ_1", 0, False, ("id1",)),
                  ("s2", "σ_2", 1, False, ("id2",)),
                  ("cum", "σ_1 + σ_2\n(withheld before adv[2])", 1, True,
                   ("id1", "id2"))]
        for r, (k, ylab, ref, cum, need) in enumerate(series):
            d = arr(data[nm], a.drop_last, need)
            y = d["s1"] + d["s2"] if k == "cum" else d[k]
            _cloud(axes[r][c], d["u"], y, col,
                   "u = inv + Σpipe − Σadv" if r == 2 else None,
                   ylab if c == 0 else None, f"{nm}" if r == 0 else None)
            if a.effective:
                yp = d["p1"] + d["p2"] if k == "cum" else d[f"p{k[-1]}"]
                _pl_curve(axes[r][c], d["u"], yp, label=(r == 0 and c == 1))
            else:
                _pl_lines(axes[r][c], ref, sc, label=(r == 0 and c == 1), cumulative=cum)
    for r in range(3):
        lo = min(ax.get_ylim()[0] for ax in axes[r])
        hi = max(ax.get_ylim()[1] for ax in axes[r])
        for ax in axes[r]:
            ax.set_ylim(lo, hi)
    axes[0][1].legend(fontsize=7, loc="upper right", framealpha=0.9)
    f.suptitle(f"§3  protection against the modified inventory position{tag}",
               fontsize=10)
    f.tight_layout(); f.savefig(out / "fig2_sigma_vs_mip.svg"); plt.close(f)

    # --- fig 3: open up u — ip and each adv component, both arms overlaid
    cols = [("ip", "ip = inv + Σpipe"), ("adv0", "adv[0]  (due next)"),
            ("adv1", "adv[1]"), ("adv2", "adv[2]  (furthest out)")]
    def resid_on_u(u, y):
        out = y.astype(float).copy()
        for lo in range(int(u.min()) - 1, int(u.max()) + 2, 2):
            m = (u >= lo) & (u < lo + 2)
            if m.sum() >= 10:
                out[m] = y[m] - y[m].mean()
        return out

    f, axes = plt.subplots(4, len(cols), figsize=(14.0, 10.5))
    panes = [("s1", False), ("s1", True), ("s2", False), ("s2", True)]
    for ci, (key, lab) in enumerate(cols):
        for nm, _, col in ARMS:
            for r, (k, resid) in enumerate(panes):
                d = arr(data[nm], a.drop_last, (f"id{k[-1]}",))
                x = d["ip"] if key == "ip" else d["adv"][:, int(key[-1])]
                y = resid_on_u(d["u"], d[k]) if resid else d[k]
                ylab = None
                if ci == 0:
                    ylab = f"σ_{k[-1]} − E[σ|u]" if resid else f"σ_{k[-1]}"
                _cloud(axes[r][ci], x, y, col, lab if r == 3 else None, ylab,
                       lab if r == 0 else None)
                if resid:
                    axes[r][ci].axhline(0, color="0.35", lw=0.9, ls="--", zorder=1)
                else:
                    # marginal rows only: PL's sigma is a constant in LEVEL, so
                    # it has no counterpart on a residual axis
                    _pl_lines(axes[r][ci], int(k[-1]) - 1, sc)
    for r in range(4):
        lo = min(ax.get_ylim()[0] for ax in axes[r])
        hi = max(ax.get_ylim()[1] for ax in axes[r])
        for ax in axes[r]:
            ax.set_ylim(lo, hi)
    handles = [plt.Line2D([], [], color=c, lw=2.5, label=n) for n, _, c in ARMS]
    axes[0][0].legend(handles=handles, fontsize=8, loc="upper right")
    f.suptitle("§4  opening up u — rows 1, 3 MARGINAL (confounded); rows 2, 4 "
               f"remove u first, flat = adds nothing{tag}", fontsize=10)
    f.tight_layout(); f.savefig(out / "fig3_sigma_components.svg"); plt.close(f)

    for nm in ("fig1_sigma_vs_time", "fig2_sigma_vs_mip", "fig3_sigma_components"):
        print(f"  {out / (nm + '.svg')}")


if __name__ == "__main__":
    main()

"""Spec-§14.3 figures for the STATS readback (#E30, INTERPRET.md Part II).

Reads the `mab_stats_probe.py` artifacts under results/{scenario}/interpret/
and writes committed SVGs to mab/figures/ (same figure contract as
`mab_plot_policy.py`: committed SVG for posting, everything exploratory
stays gitignored under results/).

Figures:
  fig_stats_mode.svg     argmax agreement vs the quantile constant c — the
                         flat left shoulder (any c <= 0.5 fits ~98%) and the
                         collapse at the bayes crown's c and the optimal c:
                         the mode is not a positive-quantile index.
  fig_stats_gap.svg      stochastic vs argmax play, per seed, against the
                         bayes twin — the round's money figure: all of the
                         stats policy's performance lives in the sampling.
  fig_stats_explore.svg  explore-rate over episode time — the stats twin's
                         sampling anneals, Thompson's shape, at ent ~= 0.
  fig_stats_index.svg    indifference contours of phi in belief coordinates
                         (on-support region) — visibly not the straight
                         lines of the m + c*s class (contrast fig_index.svg).

Colors: dataviz reference palette; the stats twin is a NEW entity and takes
categorical slot 5 (magenta) in every figure; crown/thompson/oldcrown keep
their Part-I hues (color follows the entity). 5-slot order re-validated
(adjacent CVD DeltaE >= 9.1, normal >= 19.6; the sub-3:1 contrast WARN is
relieved by the direct ink-text labels every figure carries).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mab_a5_probe import SCENARIO, T

IN  = Path("results") / SCENARIO / "interpret"
OUT = Path("figures")

C = {"stats": "#e87ba4", "crown": "#2a78d6", "thompson": "#eb6834",
     "ucb1": "#1baf7a", "oldcrown": "#eda100", "ref": "#8a8a85",
     "ink": "#33322e", "muted": "#77766f", "grid": "#e8e7e0"}

GREEDY, RANDOM, THOMPSON = 1015.69, -2.51, 1462.38


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=C["ink"], fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=C["muted"], fontsize=9)
    ax.set_ylabel(ylabel, color=C["muted"], fontsize=9)
    ax.tick_params(colors=C["muted"], labelsize=8)
    for s in ax.spines.values():
        s.set_color(C["grid"])
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_facecolor("white")


def fig_stats_mode():
    fit = json.loads((IN / "stats_fitc.json").read_text())
    curve = fit["flat_behavior"]["agreement_curve"]
    pts = sorted((float(k), v) for k, v in curve.items())
    cc = np.array([p[0] for p in pts]); ag = np.array([p[1] for p in pts])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor("white")
    ax.plot(cc, ag, "o-", color=C["stats"], lw=2, ms=4)
    best_c = fit["flat_behavior"]["c"]
    ax.axvline(1.51, color=C["crown"], lw=1.4, ls=":")
    ax.text(1.51, 0.30, "bayes crown\nc = 1.51 (#E24)", color=C["crown"],
            fontsize=8, ha="center")
    ax.axvline(2.5, color=C["ref"], lw=1.4, ls=":")
    ax.text(2.5, 0.55, "optimal\nc = 2.5 (#E26)", color=C["ref"],
            fontsize=8, ha="center")
    ax.annotate(f"best flat c = {best_c:.2f}  (98.0%)",
                xy=(best_c, 0.98), xytext=(best_c + 0.4, 0.80),
                color=C["ink"], fontsize=8,
                arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.8))
    ax.text(-2.9, 0.905, "the flat shoulder: any c ≤ 0.5\ndescribes the mode "
            "almost equally well", color=C["muted"], fontsize=8, va="top")
    ax.set_xlim(-3.1, 3.1); ax.set_ylim(0, 1.02)
    _style(ax, "The stats twin's mode is not a positive-quantile index",
           "quantile constant c in argmax(pm + c·psd)",
           "argmax agreement with the stats net")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stats_mode.svg"); plt.close(fig)


def fig_stats_gap():
    gap = json.loads((IN / "stats_gap.json").read_text())
    rob = json.loads((IN / "stats_gap_robust.json").read_text())
    # (label, stoch, argmax, color) — bayes twin from #E23 (gap 37)
    rows = [
        ("A8-a  (bayes twin)", 1453.57, 1453.57 - 37.0, C["crown"]),
        ("A11-a s1", rob["a11_s1"]["argmax"]["reward_mean"]
         + rob["a11_s1"]["gap"], rob["a11_s1"]["argmax"]["reward_mean"],
         C["stats"]),
        ("A11-a s2", gap["stochastic"]["reward_mean"],
         gap["argmax"]["reward_mean"], C["stats"]),
        ("A11-a s3", rob["a11_s3"]["argmax"]["reward_mean"]
         + rob["a11_s3"]["gap"], rob["a11_s3"]["argmax"]["reward_mean"],
         C["stats"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.patch.set_facecolor("white")
    for ref, name in ((THOMPSON, "thompson"), (GREEDY, "greedy"),
                      (RANDOM, "random")):
        ax.axvline(ref, color=C["grid"], lw=1.2)
        ax.text(ref, -0.52, name, color=C["muted"], fontsize=7.5,
                ha="center", va="top")
    for i, (label, st, am, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([am, st], [y, y], color=col, lw=2, zorder=2)
        ax.plot([st], [y], "o", color=col, ms=8, zorder=3)
        ax.plot([am], [y], "o", ms=8, mfc="white", mec=col, mew=1.8, zorder=3)
        if label.startswith("A8"):
            ax.annotate(f"gap {st - am:.0f}", xy=((st + am) / 2, y),
                        xytext=((st + am) / 2 - 60, y + 0.42),
                        color=C["muted"], fontsize=8,
                        arrowprops=dict(arrowstyle="-", color=C["muted"],
                                        lw=0.8))
        else:
            ax.text(st + 30, y, f"{st:.0f}", color=C["ink"], fontsize=8,
                    va="center")
            ax.text(am, y - 0.30, f"{am:.0f}", color=C["ink"], fontsize=8,
                    va="top", ha="center")
    ax.plot([], [], "o", color=C["muted"], label="stochastic (deployed)")
    ax.plot([], [], "o", mfc="white", mec=C["muted"], mew=1.8,
            label="argmax (mode)")
    ax.legend(loc="upper left", fontsize=8, frameon=False,
              labelcolor=C["muted"], borderaxespad=0.2)
    ax.set_xlim(-180, 1760)
    ax.set_ylim(-0.85, 3.75)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8.5,
                       color=C["ink"])
    _style(ax, "Argmax play collapses — the stats twin's performance "
               "lives in the sampling",
           "mean reward @ CRN protocol streams", "")
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stats_gap.svg"); plt.close(fig)


def fig_stats_explore():
    prof = json.loads((IN / "stats_profile.json").read_text())
    series = [(np.array(prof["explore_curve"]), "stats", "stats twin (A11-a)"),
              (None, "thompson", "thompson"),
              (None, "crown", "crown (A8-a)"),
              (None, "oldcrown", "old crown (ent 0.01)")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor("white")
    w = 25
    nudge = {"stats": 0.010, "thompson": -0.010, "crown": 0.004,
             "oldcrown": 0.0}
    for cur, ck, label in series:
        if cur is None:
            key = {"thompson": "thompson", "crown": "crown_s3",
                   "oldcrown": "oldcrown"}[ck]
            cur = np.array(json.loads(
                (IN / f"metrics_{key}.json").read_text())
                ["curves"]["explore_rate"])
        smooth = np.convolve(cur, np.ones(w) / w, mode="valid")
        lw = 2.4 if ck == "stats" else 1.6
        ax.plot(np.arange(len(smooth)) + w // 2, smooth, color=C[ck], lw=lw)
        ax.text(1005, smooth[-1] + nudge[ck], label, color=C[ck], fontsize=8,
                va="center")
    ax.set_xlim(25, 1230); ax.set_ylim(0, 0.55)
    _style(ax, "The stats twin's sampling anneals — Thompson's shape, "
               "learned at ent ≈ 0",
           "episode step t", "explore rate  P(aₜ ≠ argmax posterior mean)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stats_explore.svg"); plt.close(fig)


def fig_stats_index():
    surf = json.loads((IN / "stats_surface_a11_s2.json").read_text())
    m = np.array(surf["m_grid"]); s = np.array(surf["s_grid"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, ttg in zip(axes, ("900", "100")):
        sl = surf["slices"][ttg]
        phi = np.array([[np.nan if v is None else v for v in row]
                        for row in sl["phi_grid"]])
        cs = ax.contour(m, s, phi.T, levels=12, colors=C["stats"],
                        linewidths=1.2)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
        r2 = sl["linear_r2_belief"]
        _style(ax, f"ttg = {ttg}   (linear R² = {r2:.3f})",
               "posterior mean m", "posterior sd s" if ttg == "900" else "")
        ax.text(0.02, 0.94, "gray = off-support (n > t or n < 1)",
                color=C["muted"], fontsize=7, transform=ax.transAxes)
        ax.set_facecolor("#f4f3ef")
        mask = ~np.isnan(phi)
        ax.contourf(m, s, mask.T.astype(float), levels=[0.5, 1.5],
                    colors=["white"], zorder=0)
    fig.suptitle("The stats scorer's indifference curves in belief "
                 "coordinates — not the m + c·s straight lines",
                 color=C["ink"], fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_stats_index.svg"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (fig_stats_mode, fig_stats_gap, fig_stats_explore,
              fig_stats_index):
        f()
        print(f"wrote figures/{f.__name__[4:]}.svg")

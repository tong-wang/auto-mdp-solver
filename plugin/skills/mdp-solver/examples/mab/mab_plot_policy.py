"""Spec-§14.3 figures for the mab interpretation round (INTERPRET_PLAN.md).

Reads the probe/battery artifacts under results/{scenario}/interpret/ and
writes committed SVGs to mab/figures/ (figure contract: committed SVG for
posting; everything interactive/exploratory stays gitignored in results/).

Figures:
  fig_bonus.svg     c(ttg) — the learned exploration bonus vs UCB1's
                    sqrt(2 ln t), + the E[max of 10 normals] reference:
                    the round's money figure.
  fig_explore.svg   explore-rate over episode time, crown vs the references
                    and the old (subsidized) crown.
  fig_index.svg     indifference contours of the learned index phi(m, s) at
                    an early and a late ttg slice — the UCB class predicts
                    straight lines.
  fig_critic.svg    dV/d(sum sd) over episode time — the critic's price of
                    information (#E22's mechanism, measured).

Colors: dataviz reference palette, categorical slots 1-4 in the documented
adjacent order; color follows the entity across every figure
(crown=blue, thompson=orange, ucb1=aqua, oldcrown=yellow).
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

C = {"crown": "#2a78d6", "thompson": "#eb6834", "ucb1": "#1baf7a",
     "oldcrown": "#eda100", "ref": "#8a8a85", "ink": "#33322e",
     "muted": "#77766f", "grid": "#e8e7e0"}

EMAX10 = 1.5388                       # E[max of 10 iid standard normals]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=C["ink"], fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=C["muted"], fontsize=9)
    ax.set_ylabel(ylabel, color=C["muted"], fontsize=9)
    ax.tick_params(colors=C["muted"], labelsize=8)
    for s in ax.spines.values():
        s.set_color(C["grid"])
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_facecolor("white")


def fig_bonus():
    fit = json.loads((IN / "fitc.json").read_text())
    kk = np.array(sorted(int(k) for k in fit["knots_traj"]))
    cc = np.array([fit["knots_traj"][str(k)] for k in kk])
    c0, p = fit["powerlaw"]["c0"], fit["powerlaw"]["p"]
    tt = np.linspace(10, 1000, 200)
    ucb = np.sqrt(2.0 * np.log(T - tt + 1.0))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor("white")
    ax.plot(tt, ucb, color=C["ucb1"], lw=2)
    ax.plot(tt, c0 * (tt / T) ** p, color=C["crown"], lw=2, ls="--")
    ax.plot(kk, cc, "o-", color=C["crown"], lw=2, ms=5)
    ax.axhline(EMAX10, color=C["ref"], lw=1.4, ls=":")
    ax.text(310, 3.45, "UCB1  σ√(2 ln t)", color=C["ucb1"], fontsize=9)
    ax.text(500, 1.78, "learned c(ttg)", color=C["crown"], fontsize=9,
            ha="center")
    ax.text(60, 1.28, "E[max of 10 N(0,1)] = 1.54", color=C["ref"], fontsize=8,
            ha="right" if False else "left")
    ax.set_ylim(0, 4.2)
    ax.set_xlim(1000, 0)          # episode time flows left -> right
    _style(ax, "The learned bonus is Thompson-sized and nearly flat, "
               "not UCB-sized",
           "time-to-go (episode runs left → right)",
           "bonus multiplier c  (index = m + c·s)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_bonus.svg"); plt.close(fig)


def fig_explore():
    series = [("crown_s3", "crown", "crown (A8-a)"),
              ("thompson", "thompson", "thompson"),
              ("ucb1", "ucb1", "ucb1"),
              ("oldcrown", "oldcrown", "old crown (ent 0.01)")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor("white")
    w = 25
    nudge = {"crown": 0.012, "ucb1": -0.012, "thompson": -0.004,
             "oldcrown": 0.0}
    for key, ck, label in series:
        cur = np.array(json.loads(
            (IN / f"metrics_{key}.json").read_text())["curves"]["explore_rate"])
        smooth = np.convolve(cur, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(smooth)) + w // 2, smooth, color=C[ck], lw=2)
        ax.text(1005, smooth[-1] + nudge[ck], label, color=C[ck], fontsize=8,
                va="center")
    ax.set_xlim(25, 1210); ax.set_ylim(0, 0.55)
    _style(ax, "Exploration anneals — except under the flat entropy subsidy",
           "episode step t", "explore rate  P(aₜ ≠ argmax posterior mean)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_explore.svg"); plt.close(fig)


def fig_index():
    surf = json.loads((IN / "index_surface_crown_s3.json").read_text())
    m = np.array(surf["m_grid"]); s = np.array(surf["s_grid"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, ttg in zip(axes, ("900", "100")):
        phi = np.array(surf["slices"][ttg]["phi"])
        cs = ax.contour(m, s, phi.T, levels=12, colors=C["crown"],
                        linewidths=1.2)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
        r2 = surf["slices"][ttg]["linear_r2"]
        _style(ax, f"ttg = {ttg}   (linear R² = {r2:.3f})",
               "posterior mean m", "posterior sd s" if ttg == "900" else "")
    fig.suptitle("Indifference curves of the learned index — near-straight "
                 "lines = the m + c·s class", color=C["ink"], fontsize=11,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_index.svg"); plt.close(fig)


def fig_critic():
    crit = json.loads((IN / "critic.json").read_text())
    dv = np.array(crit["dVds_curve"])
    w = 25
    smooth = np.convolve(dv, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fig.patch.set_facecolor("white")
    ax.plot(np.arange(len(smooth)) + w // 2, smooth, color=C["crown"], lw=2)
    ax.axhline(0, color=C["ref"], lw=1, ls=":")
    ax.set_xlim(0, 1000)
    _style(ax, "The critic prices information: ∂V/∂(Σ sd) anneals ~9× "
               "across the episode", "episode step t",
           "∂V/∂(Σᵢ sdᵢ)  (normalized-return units)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_critic.svg"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (fig_bonus, fig_explore, fig_index, fig_critic):
        f()
        print(f"wrote figures/{f.__name__[4:]}.svg" .replace("fig_", "fig_"))

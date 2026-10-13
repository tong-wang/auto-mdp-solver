"""Overlay the learned stocking rules on Proposition 1(b)'s, in premium coordinates (spec §14.3).

The confirm stance (``research_questions`` in ``cnv_schema.json``) is that the
optimal order sits ABOVE the myopic newsvendor level y^M of its own
observation type — an information premium — and that timing's premium is
smaller than event's over most of the grid. The coordinate in which that
structure is a canonical shape is the **premium**, order − y^M: the myopic
rule is the zero line, the DP's over-order is a positive bar that decays to
zero by the last period (no information is worth buying then), and under full
observation the optimum IS the zero line (y^F = y^M).

Two rows, one column per observation type (timing / event / full):

- **premium by period** — each net's mean premium over its own visited states
  (the probe's on-policy dump, screen block), whiskers = the inter-quartile
  range of the per-cell means; underneath, the reference: the timing DP's
  premium at the same states (the probe's ``y_dp`` column), the paper's exact
  event DP at the prior (period 0 only), the zero line for full;
- **premium at the prior, cell by cell** — every episode of a cell starts at
  the prior, so the net's period-0 order is one number per cell; plotted
  against the paper's own exact first-period premium (github.com/tong-wang/
  NVLearning, ``Q_T − Q_Tm`` and ``Q_E − Q_Em``; #E43). On the diagonal the
  net buys exactly the DP's information premium.

The script also writes the confirm statistics to
``results/{target}/interpret/confirm_stats_{target}.tsv`` and prints them.

**One figure spec, two files** (the same split as the other domains'
``*_plot_policy.py``): the static ``.svg`` → ``figures/`` (committed, inlined
by markdown) and the interactive ``.html`` → ``results/{target}/figures/``
(gitignored, beside the runs it came from).

    python cnv_plot_policy.py -g paper510 \\
        --arm timing_belief_ctx=results/tuning/cnv_paper510_ppo_timing_belief_ctx_all/trial_0062/paper510/PPO_*/probe/onpolicy_paper510.tsv \\
        --arm timing_raw=... --arm event_raw=... --arm full_belief_ctx=... --arm full_raw=... \\
        --paper-dir results/paper510/reference/NVLearning
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cnv_grids import GRIDS

HERE = Path(__file__).resolve().parent
TYPES = ("timing", "event", "full")
COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def read_onpolicy(path: str) -> tuple[str, pd.DataFrame]:
    path = sorted(glob.glob(path))[0] if any(ch in path for ch in "*?[") else path
    kind = None
    for ln in Path(path).read_text().splitlines()[:3]:
        if " type=" in ln:
            kind = ln.split(" type=")[1].split()[0]
    df = pd.read_csv(path, sep="\t", comment="#")
    df["excess"] = df["y_net"] - df["y_myopic"]
    if df["y_dp"].notna().any():
        df["dp_excess"] = df["y_dp"] - df["y_myopic"]
    return kind, df


def paper_premiums(paper_dir: Path, target: str) -> pd.DataFrame:
    """Per cell: the paper's exact first-period premium for timing and event (and 0 for full)."""
    rd = lambda n: pd.read_csv(paper_dir / f"NVLearning_{n}.txt", sep="\t")
    ref = (rd("Timing")[["c", "alpha", "beta", "Q_T"]].merge(rd("Timing_myopic")[["c", "alpha", "beta", "Q_Tm"]])
           .merge(rd("Event")[["c", "alpha", "beta", "Q_E"]]).merge(rd("Event_myopic")[["c", "alpha", "beta", "Q_Em"]]))
    key = {(round(r.c, 4), round(r.beta, 6), round(r.alpha / r.beta, 4)): r for r in ref.itertuples()}
    rows = []
    for cid, src in GRIDS[target]:
        sc = src(0)
        r = key[(round(sc.unit_cost, 4), round(sc.prior_rate, 6), round(sc.prior_mean, 4))]
        rows.append({"target": cid, "timing": r.Q_T - r.Q_Tm, "event": r.Q_E - r.Q_Em, "full": 0})
    return pd.DataFrame(rows).set_index("target")


def build(arms: dict, paper: pd.DataFrame | None, horizon: int, title: str) -> tuple[go.Figure, pd.DataFrame]:
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{t}: premium by period" for t in TYPES]
                        + [f"{t}: premium at the prior, per cell" for t in TYPES],
                        vertical_spacing=0.14, horizontal_spacing=0.07)
    stats = []
    rng = np.random.default_rng(0)
    for col, kind in enumerate(TYPES, start=1):
        mine = [(lab, df) for lab, (k, df) in arms.items() if k == kind]
        periods = np.arange(horizon)
        # reference underneath
        if kind == "timing" and mine and "dp_excess" in mine[0][1]:
            ref_df = mine[0][1]
            fig.add_trace(go.Bar(x=periods, y=[ref_df.loc[ref_df.period == p, "dp_excess"].mean() for p in periods],
                                 name="timing DP (same states)", marker_color="#cbd5e1", showlegend=True), 1, col)
        elif kind == "event" and paper is not None:
            fig.add_trace(go.Bar(x=[0], y=[paper["event"].mean()], name="event DP at the prior (paper)",
                                 marker_color="#e2e8f0", showlegend=True), 1, col)
        fig.add_hline(y=0, line=dict(color="#64748b", width=1), row=1, col=col)
        for lab, df in mine:
            c = COLORS[list(arms).index(lab) % len(COLORS)]
            per_cell = df.groupby(["target", "period"])["excess"].mean().unstack()
            mean = [df.loc[df.period == p, "excess"].mean() for p in periods]
            q1, q3 = per_cell.quantile(0.25).values, per_cell.quantile(0.75).values
            fig.add_trace(go.Scatter(x=periods, y=mean, mode="lines+markers", name=lab, line=dict(color=c),
                                     error_y=dict(type="data", symmetric=False, array=q3 - mean, arrayminus=mean - q1)),
                          1, col)
            prem0 = df[df.period == 0].groupby("target")["excess"].first()
            row = {"arm": lab, "type": kind, "states": len(df),
                   "frac_at_or_above_myopic": float((df.excess >= 0).mean()),
                   **{f"mean_premium_t{p}": float(mean[p]) for p in periods},
                   "cells_premium_t0_pos": int((prem0 > 0).sum()), "cells": int(len(prem0))}
            if paper is not None:
                ref0 = paper.loc[prem0.index, kind]
                row["t0_match_paper_dp"] = int((prem0 == ref0).sum())
                row["t0_mean_paper_dp"] = float(ref0.mean())
                fig.add_trace(go.Scatter(x=ref0 + rng.uniform(-0.18, 0.18, len(ref0)),
                                         y=prem0 + rng.uniform(-0.18, 0.18, len(prem0)), mode="markers",
                                         name=f"{lab} (cells)", marker=dict(color=c, size=4, opacity=0.5),
                                         text=list(prem0.index), showlegend=False), 2, col)
            if "dp_excess" in df:
                row["agree_dp"] = float((df.y_net == df.y_dp).mean())
            stats.append(row)
        if paper is not None:
            lo, hi = -3, max(4, int(paper[kind].max()) + 1)
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#94a3b8", dash="dash"),
                                     name="net = DP", showlegend=(col == 1)), 2, col)
            fig.update_xaxes(title_text="exact DP premium at the prior", range=[lo, hi], row=2, col=col)
            # a few cells sit far off the rule; a fixed window keeps the 500 that do not readable, and says so
            y_lo, y_hi = -6, hi + 6
            fig.update_yaxes(range=[y_lo, y_hi], row=2, col=col)
            notes = []
            for lab, df in mine:
                prem0 = df[df.period == 0].groupby("target")["excess"].first()
                n_off = int(((prem0 < y_lo) | (prem0 > y_hi)).sum())
                next(s for s in stats if s["arm"] == lab)["t0_cells_off_window"] = n_off
                notes.append(f"{lab}: {n_off} cells off-window")
            fig.add_annotation(text="<br>".join(notes), xref=f"x{3 + col} domain", yref=f"y{3 + col} domain",
                               x=0.02, y=0.98, showarrow=False, align="left", font=dict(size=10, color="#475569"))
        fig.update_xaxes(title_text="period", dtick=1, row=1, col=col)
    fig.update_yaxes(title_text="order − y^M (mean over visited states)", row=1, col=1)
    fig.update_yaxes(title_text="net's order − y^M at the prior", row=2, col=1)
    fig.update_layout(title=title, template="plotly_white", height=820, width=1250, barmode="overlay",
                      legend=dict(orientation="h", y=-0.12))
    return fig, pd.DataFrame(stats)


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay the learned premium on Proposition 1(b)'s (spec §14.3).")
    p.add_argument("-g", "--grid_name", default="paper510")
    p.add_argument("--arm", action="append", required=True, metavar="LABEL=ONPOLICY_TSV")
    p.add_argument("--paper-dir", default=None, help="the paper's NVLearning output/ (per-cell exact first-period orders)")
    p.add_argument("--inline-js", action="store_true", help="embed plotly.js in the HTML (offline viewing)")
    args = p.parse_args()
    target = args.grid_name
    arms = {}
    for a in args.arm:
        lab, path = a.split("=", 1)
        arms[lab] = read_onpolicy(str(HERE / path) if not Path(path).is_absolute() else path)
    horizon = int(next(iter(GRIDS[target]))[1](0).horizon)
    paper = paper_premiums(HERE / args.paper_dir if not Path(args.paper_dir).is_absolute() else Path(args.paper_dir),
                           target) if args.paper_dir else None
    fig, stats = build(arms, paper, horizon, f"cnv {target}: the learned information premium vs Proposition 1(b)")
    svg = HERE / "figures" / f"cnv_policy_{target}.svg"
    html = HERE / "results" / target / "figures" / f"cnv_policy_{target}.html"
    out = HERE / "results" / target / "interpret" / f"confirm_stats_{target}.tsv"
    for f in (svg, html, out):
        f.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(svg))
    fig.write_html(str(html), include_plotlyjs=("inline" if args.inline_js else "cdn"))
    stats.to_csv(out, sep="\t", index=False, float_format="%.4f")
    if paper is not None:
        tp, ev = paper["timing"], paper["event"]
        print(f"paper DP at the prior: timing premium > 0 on {(tp > 0).sum()} cells, event on {(ev > 0).sum()}; "
              f"timing < event on {(tp < ev).sum()}, equal on {(tp == ev).sum()}, timing > event on {(tp > ev).sum()}")
        prem0 = {lab: df[df.period == 0].groupby("target")["excess"].first() for lab, (k, df) in arms.items()}
        ev_labs = [l for l, (k, _) in arms.items() if k == "event"]
        for tl in [l for l, (k, _) in arms.items() if k == "timing"]:
            for el in ev_labs:
                a, b = prem0[tl], prem0[el].reindex(prem0[tl].index)
                print(f"nets at the prior: {tl} < {el} on {(a < b).sum()} cells, equal {(a == b).sum()}, greater {(a > b).sum()}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(stats.round(3).to_string(index=False))
    print(f"[plot] {svg}\n[plot] {html}\n[plot] {out}")


if __name__ == "__main__":
    main()

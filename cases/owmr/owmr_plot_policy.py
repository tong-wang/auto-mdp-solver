"""Figures for the readback (spec §14.3, the figure contract): one plot spec, two renders.

`figures/hicv_policy_curves.svg` (committed; what INTERPRET.md inlines) and
`results/hicv/figures/policy_curves_hicv.html` (interactive, gitignored, beside the runs it derives from)
are written from ONE spec in ONE run, so they cannot drift. The spec: what the policy does, read off the
Stage-5 probe's synthetic states at mid-horizon (`owmr_policy_probe.py`, period 50, pipelines at mean
cover) — three panels, each a decision against one state coordinate, for the #E33 winner (catkeep + vine,
seed 2, ckpt 10M), its matched control (same recipe, no vine) and the LB heuristic computed on the same
states — the reference underneath the trained arms, overlaid, never side by side:
  (a) retained fraction of warehouse on-hand vs on-hand   (retailers at mean cover)
  (b) order vs warehouse on-hand                          (the heuristic: order up to y0*)
  (c) shipment to retailer 0 vs its net stock             (warehouse at 4 periods of demand)
The network curves are read from each run's probe JSON; the heuristic's from `LbHeuristicPolicy.decide`
on `make_state` (the probe's own state builder), so the three series see identical states.

Usage (from owmr/):  python owmr_plot_policy.py        # the SVG always; the HTML when plotly is installed; needs the two probe JSONs
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from owmr_benchmark_lb_heuristic import build as build_heuristic
from owmr_policy_probe import make_state
from owmr_scenarios import SCENARIOS

HERE = Path(__file__).resolve().parent
SCENARIO = "hicv"
# dataviz palette (reference instance, light surface): categorical slots 1-3 in fixed order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = {"vine": ("#2a78d6", "catkeep + vine (winner, seed 2)"),
          "ctl": ("#eb6834", "catkeep, no vine (matched control)"),
          "heur": ("#1baf7a", "LB heuristic")}


def load_probe(pattern: str) -> dict:
    d = sorted(glob.glob(str(HERE / pattern)))[0]
    r = json.load(open(Path(d) / "checkpoints" / "probe" / "probe_hicv_raw_order_catkeep.json"))
    return {s["name"]: s for s in r["sweeps"]} | {"period": r["period"]}


def heuristic_curves(sc, heur, period: int, wh_grid, rt_grid, ret_grid):
    order = [heur.decide(make_state(sc, period, x))[0] for x in wh_grid]
    kept = []
    for W in ret_grid:
        _, ship = heur.decide(make_state(sc, period, W))
        kept.append(1.0 - float(np.sum(ship)) / W)
    ship0 = []
    W = 4.0 * sc.mu_sys
    for z in rt_grid:
        rt = list(sc.mu_rt); rt[0] = z
        ship0.append(float(heur.decide(make_state(sc, period, W, rt_stock=rt))[1][0]))
    return order, kept, ship0


def render_html(spec: dict) -> Path:
    """The interactive render of the same spec (plotly), beside the runs it derives from (gitignored)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=[p["title"] for p in spec["panels"]], horizontal_spacing=0.08)
    for c, p in enumerate(spec["panels"], start=1):
        for ser in p["series"]:
            fig.add_trace(go.Scatter(x=p["x"], y=ser["y"], mode="lines", name=ser["name"], legendgroup=ser["name"],
                                     showlegend=(c == 1), line={"color": ser["color"], "width": 2},
                                     hovertemplate=f"{ser['name']}<br>%{{x:.2f}} → %{{y:.3f}}<extra></extra>"), row=1, col=c)
        fig.update_xaxes(title_text=p["xlabel"], row=1, col=c); fig.update_yaxes(title_text=p["ylabel"], row=1, col=c)
    fig.add_vline(x=spec["z_star"], line={"dash": "dot", "color": INK2}, row=1, col=3,
                  annotation_text=f"z* = {spec['z_star']:.2f}", annotation_position="top left")
    fig.update_layout(title=spec["title"], hovermode="x unified", template="plotly_white", height=420,
                      legend={"orientation": "h", "y": -0.25}, font={"size": 12})
    out = HERE / "results" / SCENARIO / "figures"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"policy_curves_{SCENARIO}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def style(ax, xlabel, ylabel):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)


def main() -> None:
    sc = SCENARIOS["hicv"]
    heur = build_heuristic("hicv", HERE)
    vine = load_probe("results/hicv/PPO_*seed2*E33catkeepvine")
    ctl = load_probe("results/hicv/PPO_*seed2*E33catkeepctl")
    period = vine["period"]
    wh_grid = vine["order_vs_wh_stock"]["grid"]
    ret_grid = vine["retention_vs_wh_stock"]["grid"]
    rt_grid = vine["allocation_vs_rt0_stock"]["grid"]
    h_order, h_kept, h_ship0 = heuristic_curves(sc, heur, period, wh_grid, rt_grid, ret_grid)
    z_star = float(heur.bound.z[0])

    # ---- the one spec both renders read ----------------------------------------------------------
    panels = [
        ("retention_vs_wh_stock", ret_grid, lambda p: p["retention_vs_wh_stock"]["derived"]["retained_fraction"], h_kept,
         "warehouse on-hand (units)", "fraction of on-hand kept", "(a) retention"),
        ("order_vs_wh_stock", wh_grid, lambda p: p["order_vs_wh_stock"]["order"], h_order,
         "warehouse on-hand (units)", "order (units)", "(b) order"),
        ("allocation_vs_rt0_stock", rt_grid, lambda p: [row[0] for row in p["allocation_vs_rt0_stock"]["shipped"]], h_ship0,
         "retailer 0 net stock (units)", "shipped to retailer 0 (units)", "(c) allocation"),
    ]
    spec = {"title": f"hicv — what the policy does, read at period {period} on synthetic states (pipelines at mean cover)",
            "panels": [{"title": t, "x": list(map(float, grid)), "xlabel": xl, "ylabel": yl,
                        "series": [{"name": SERIES[k][1], "color": SERIES[k][0], "y": list(map(float, ys))}
                                   for k, ys in (("heur", hcurve), ("ctl", get(ctl)), ("vine", get(vine)))]}
                       for (key, grid, get, hcurve, xl, yl, t) in panels],
            "z_star": z_star, "y0_star": float(heur.y0)}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=100)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, panel in zip(axes, spec["panels"]):
        ax.set_facecolor("#fcfcfb")
        for ser in panel["series"]:
            ax.plot(panel["x"], ser["y"], color=ser["color"], linewidth=2.0, label=ser["name"], solid_capstyle="round")
        style(ax, panel["xlabel"], panel["ylabel"])
        ax.set_title(panel["title"], loc="left", color=INK, fontsize=10.5, pad=6)
    # references, in ink, never in a series colour
    axes[0].set_ylim(-0.02, 0.36)
    axes[0].annotate("menu rung 0.2", xy=(ret_grid[0], 0.2), xytext=(4, 3), textcoords="offset points",
                     ha="left", color=INK2, fontsize=8)
    axes[0].annotate("menu rung 0.3", xy=(ret_grid[0], 0.3), xytext=(4, 3), textcoords="offset points",
                     ha="left", color=INK2, fontsize=8)
    axes[2].axvline(z_star, color=INK2, linewidth=1.0, linestyle=(0, (3, 3)))
    axes[2].annotate(f"newsvendor target z* = {z_star:.2f}", xy=(z_star, max(h_ship0) * 0.92), xytext=(5, 0),
                     textcoords="offset points", color=INK2, fontsize=8)
    k = int(len(wh_grid) * 0.3)
    axes[1].annotate(f"heuristic:\norder up to y0* = {heur.y0:.1f}", xy=(wh_grid[k], h_order[k]),
                     xytext=(8, 8), textcoords="offset points", ha="left", color=INK2, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(spec["title"],
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / "hicv_policy_curves.svg", format="svg", facecolor=fig.get_facecolor())
    print("wrote", out / "hicv_policy_curves.svg")
    try:
        import plotly  # noqa: F401  the interactive render needs it; the committed SVG does not
    except ImportError:
        print("plotly not installed: the interactive HTML render is skipped (pip install plotly)")
        return
    print("wrote", render_html(spec))


if __name__ == "__main__":
    main()

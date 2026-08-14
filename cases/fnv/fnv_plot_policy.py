"""Policy-overlay figure for FNV — one plot spec, two renders (spec §14.3).

Plots the paper's structural claim directly, in its own coordinates:

    Proposition 2:  x_(n-1) + q_n = S_n(I_n) = mu + I_n + b_n

so post-order inventory against cumulative information is a straight line of
**slope 1** with intercept `mu + b_n`. The trained policy's realized
(I_n, x_(n-1)+q_n) pairs are scattered over that predicted line, one period per
colour.

**One panel per scenario, never pooled.** `b_n` differs cell to cell, so the
predicted line has a different intercept in every cell; overlaying cells in one
panel smears those intercepts together and blurs exactly the structure the
figure exists to show.

Points where the policy did not order are drawn faintly and excluded from the
fit: there `q_n = 0` and the post-order position is just `x_(n-1)`, which is
censored at the inventory already held, not the order-up-to level.

Where the renders land (§14.3): committed static PNG in `fnv/figures/`
(raster, because a 27k-point scatter as SVG is several MB), the
interactive HTML beside its runs in `results/{scenario}/figures/` (gitignored).
Committed markdown cites the regenerating command, never the HTML.

Example usage:
    python fnv_plot_policy.py --model-path results/FNV-aMMFE/PPO_.../checkpoints/x.zip
    python fnv_plot_policy.py --model-path <ckpt> --cell "stdev=0.3,T=0.5,lamb=0.1"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO

from fnv_benchmark_prop2 import solve_offsets
from fnv_mdp import advance, init_state
from fnv_policy_probe import build_obs
from fnv_ppo_eval import (load_obs_rms, normalize_obs, resolve_cells,
                          resolve_vecnorm_path)

PLOT_POINTS = 400   # markers drawn per period per panel (fits use all)
PERIOD_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
ACT_TOL = 1e-4

def default_cells(cells: dict) -> list[str]:
    """Six representative cells, derived from whichever grid was passed.

    The two grids do not share axis values (aMMFE sweeps stdev 0.05–0.3,
    mMMFE 0.1–0.6), so a hardcoded list resolves on one board and silently
    collapses to a single panel on the other.
    """
    def axes(cid):
        return {k: float(v) for k, v in (p.split("=") for p in cid.split(","))}

    parsed = {cid: axes(cid) for cid in cells}
    vals = {k: sorted({a[k] for a in parsed.values()}) for k in ("stdev", "T", "lamb")}
    mid = {k: v[len(v) // 2] for k, v in vals.items()}

    def pick(**over):
        want = dict(mid, **over)
        for cid, a in parsed.items():
            if all(abs(a[k] - want[k]) < 1e-9 for k in want):
                return cid
        return None

    out = [pick(stdev=vals["stdev"][0]), pick(), pick(stdev=vals["stdev"][-1]),
           pick(T=vals["T"][0]), pick(T=vals["T"][-1]), pick(lamb=vals["lamb"][0])]
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot FNV policy structure (§14.3).")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None)
    p.add_argument("-s", "--scenario_name", type=str, default="FNV-aMMFE")
    p.add_argument("--cell", type=str, action="append", default=None,
                   help="Scenario cell to panel (repeatable; default: a spread of six)")
    p.add_argument("--n-episodes", type=int, default=1500)
    p.add_argument("--static-out", type=str, default=None)
    p.add_argument("--interactive-out", type=str, default=None)
    return p


def trajectories(sc, policy, n_ep: int) -> pd.DataFrame:
    """Realized (period, I_n, post-order inventory) along the policy's own path."""
    rows = []
    for ep in range(n_ep):
        st, _ = init_state(sc, ep)
        while not st.terminated:
            I, x, n = st.information, st.inventory, st.period
            q = policy(x, I, n)
            rows.append({"period": n, "I": I, "post": x + q, "acted": q > ACT_TOL})
            st, _ = advance(sc, st, q)
    return pd.DataFrame(rows)


def structural_y(post: np.ndarray, additive: bool) -> np.ndarray:
    """The coordinate the paper's structure is LINEAR in.

        a-MMFE   S_n(I) = mu + I + b_n        -> y = S
        m-MMFE   S_n(I) = exp(mu + I + b_n)   -> y = log S

    Both then predict slope 1 against I with intercept mu + b_n. Fitting the
    multiplicative branch on the raw level instead would find a curve, not a
    line, and report a meaningless slope.
    """
    post = np.asarray(post, dtype=float)
    return post if additive else np.log(np.maximum(post, 1e-12))


def fit_slope(df: pd.DataFrame, period: int,
              additive: bool = True) -> tuple[float, float, float, int]:
    """Slope/intercept/r2 of the structural coordinate on I, ACTING points only."""
    d = df[(df["period"] == period) & df["acted"]]
    if len(d) < 10 or d["I"].std(ddof=1) < 1e-9:
        return float("nan"), float("nan"), float("nan"), len(d)
    y = structural_y(d["post"].to_numpy(), additive)
    sl, ic = np.polyfit(d["I"].to_numpy(), y, 1)
    res = y - (sl * d["I"].to_numpy() + ic)
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((res ** 2).sum()) / ss if ss > 0 else float("nan")
    return float(sl), float(ic), r2, len(d)


def build_figure(panels: list[tuple[str, object, pd.DataFrame]]) -> go.Figure:
    n = len(panels)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    titles = []
    for cid, sc, df in panels:
        add = sc.mmfe_mode == "additive"
        parts = []
        for p in range(1, sc.N + 1):
            sl, _, _, k = fit_slope(df, p, add)
            if sl == sl:
                parts.append(f"n{p}: {sl:.2f}")
        titles.append(f"{cid}<br><sub>slope {', '.join(parts) or 'n/a'}</sub>")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.07, vertical_spacing=0.16)

    for i, (cid, sc, df) in enumerate(panels):
        r, c = divmod(i, cols)
        r += 1; c += 1
        b = np.array(solve_offsets(sc))
        add = sc.mmfe_mode == "additive"
        first = (i == 0)

        for p in range(1, sc.N + 1):
            col = PERIOD_COLORS[(p - 1) % len(PERIOD_COLORS)]
            d = df[df["period"] == p]
            act, idle = d[d["acted"]], d[~d["acted"]]
            # Fits above use every episode; the markers are thinned so the
            # committed SVG stays a few hundred KB rather than several MB.
            # Deterministic head-slice, not a random sample, so the figure is
            # reproducible from the same run.
            act, idle = act.iloc[:PLOT_POINTS], idle.iloc[:PLOT_POINTS]

            # censored points: q = 0, so post is just the stock already held
            if len(idle):
                fig.add_trace(go.Scatter(
                    x=idle["I"], y=structural_y(idle["post"], add), mode="markers",
                    name=f"period {p} — did not order",
                    marker=dict(color=col, size=3, opacity=0.13, symbol="x"),
                    legendgroup=f"idle{p}", showlegend=first,
                ), row=r, col=c)
            if len(act):
                fig.add_trace(go.Scatter(
                    x=act["I"], y=structural_y(act["post"], add), mode="markers",
                    name=f"period {p} — ordered",
                    marker=dict(color=col, size=4, opacity=0.5),
                    legendgroup=f"act{p}", showlegend=first,
                ), row=r, col=c)

            # the prediction: slope exactly 1, intercept mu + b_n -- in the
            # structural coordinate this is one straight line for BOTH modes
            span = d["I"]
            if len(span):
                xs = np.linspace(float(span.min()), float(span.max()), 2)
                fig.add_trace(go.Scatter(
                    x=xs, y=sc.mu + xs + b[p - 1], mode="lines",
                    name=(f"period {p} — optimal "
                          + ("S" if add else "log S") + " = mu + I + b (slope 1)"),
                    line=dict(color=col, width=2, dash="dash"),
                    legendgroup=f"opt{p}", showlegend=first,
                ), row=r, col=c)

        fig.update_xaxes(title_text="cumulative information  I", row=r, col=c)
        if c == 1:
            fig.update_yaxes(
                title_text=("post-order inventory  x + q" if add
                            else "log post-order inventory  log(x + q)"),
                row=r, col=c)

    # subplot titles are annotations; nudge them clear of the figure title
    for ann in fig.layout.annotations:
        ann.font = dict(size=12)

    fig.update_layout(
        title=dict(
            text=("<b>FNV policy readback — does the net recover "
                  + ("S" if panels[0][1].mmfe_mode == "additive" else "log S")
                  + " = mu + I + b?</b>"
                  "<br><sub>dashed = Proposition 2 (slope 1) · dots = realized "
                  "orders · faint × = no order that period (post-order position "
                  "is censored at the stock already held)<br>one panel per "
                  "scenario: b differs by cell, so pooling would blur the lines"
                  "</sub>"),
            x=0.5, xanchor="center", y=0.975, yanchor="top",
        ),
        template="plotly_white",
        height=400 * rows + 190, width=470 * cols,
        legend=dict(orientation="h", yanchor="top", y=-0.10, x=0.5,
                    xanchor="center", font=dict(size=10)),
        margin=dict(t=170, b=150, l=70, r=40),
    )
    return fig


def main() -> None:
    args = _build_arg_parser().parse_args()
    mp = Path(args.model_path)
    model = PPO.load(str(mp), device="cpu")
    obs_rms, clip = load_obs_rms(resolve_vecnorm_path(mp, args.vecnorm_path))
    cells = dict(resolve_cells(args.scenario_name))
    wanted = args.cell or default_cells(cells) or list(cells)[:6]

    panels = []
    for cid in wanted:
        sc = cells[cid]

        def net(x, I, n, sc=sc):
            a, _ = model.predict(
                normalize_obs(build_obs(sc, n, [x], I), obs_rms, clip),
                deterministic=True)
            return max(0.0, float(np.asarray(a).reshape(-1)[0]))

        df = trajectories(sc, net, args.n_episodes)
        panels.append((cid, sc, df))
        add = sc.mmfe_mode == "additive"
        for p in range(1, sc.N + 1):
            sl, ic, r2, k = fit_slope(df, p, add)
            print(f"{cid:>26} period {p}: n_act={k:>5}  slope={sl:7.3f}  "
                  f"r2={r2:7.4f}  b_hat={ic - sc.mu if ic == ic else float('nan'):+.4f}")

    fig = build_figure(panels)
    here = Path(__file__).resolve().parent
    static = Path(args.static_out) if args.static_out else (
        here / "figures" / f"policy_structure_{args.scenario_name}.png")
    inter = Path(args.interactive_out) if args.interactive_out else (
        here / "results" / args.scenario_name / "figures" / "policy_structure.html")
    static.parent.mkdir(parents=True, exist_ok=True)
    inter.parent.mkdir(parents=True, exist_ok=True)
    # both renders from the one spec, in one run, so they cannot drift
    fig.write_image(str(static))
    fig.write_html(str(inter), include_plotlyjs="cdn")
    print(f"\nstatic (committed)    → {static}")
    print(f"interactive (ignored) → {inter}")


if __name__ == "__main__":
    main()

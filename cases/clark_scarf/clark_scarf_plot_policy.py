"""Overlay the learned shipping rule on Clark & Scarf's, in base-stock coordinates.

Campaign deliverable for A2 (the tier-2 *confirm* half), one panel per echelon.
Plots **echelon position after ordering** against **echelon position before
ordering**, which is the coordinate system a base-stock policy is legible in:

    y_i = ybar_i     (a flat ceiling)      while u_i <= ybar_i  — order up to it
    y_i = u_i        (the 45-degree line)  while u_i >  ybar_i  — order nothing

so the critical number is the height of the plateau, and the vertical distance
above the diagonal IS the shipped quantity. Plotting q directly hides ybar; this
does not.

The paper's optimum is a *median*, not a max (`clark_scarf_benchmark_dp.py`):

    y_i = median(u_i, ybar_i(t), x_{i+1})

— the third argument is the availability clip, what the level above actually
holds. A point sitting below the plateau is therefore not necessarily a
disagreement with the rule: it may be the clip binding. The panels mark those
separately, because conflating them would read a supply constraint as a policy
error.

**Why both a sweep and a scatter.** The scatter is the policy's own decisions,
on its own trajectories — the honest sample, and the only one that shows where
the policy actually lives. A grid sweep would probe states no policy visits
(`clark_scarf_policy_probe.py` declines to do this for exactly that reason).
But a scatter alone cannot show the *rule* where the policy rarely goes, so the
DP reference is drawn as a line across the full axis and the scatter is laid
over it. Where they coincide, the learned rule is the paper's.

**One figure spec, two files** — the same split `inv_single_plot_policy.py`
uses, so the static and interactive views cannot drift:
- `.svg`  — the posting surface, written to `figures/` and committed, for
            inline embedding in markdown. GitHub and VS Code markdown do not
            execute JS, so an inline figure has to be a real image.
- `.html` — the local-analysis surface, under `results/{scenario}/figures/`:
            interactive, hover reporting u, y and the implied q.

The split of locations IS the gitignore boundary: `results/` is already
ignored, so a regenerable figure belongs there beside the runs it came from,
while anything committed markdown references must live in `figures/`.

Both rules are step functions on integer states, so the reference traces use
mid-step shape; straight interpolation would imply behaviour neither policy has.

Usage:
    python clark_scarf_plot_policy.py -s n3_l2_p09 \
        --model-path results/tuning/hp_target_raw/trial_0211/**/n3_l2_p09_ppo.zip
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from clark_scarf_benchmark_dp import ClarkScarfDP
from clark_scarf_policy_probe import _load_policy, collect, sweep_policy
from clark_scarf_scenarios import SCENARIOS

# dataviz categorical slots 1 (blue) and 2 (orange) on a light surface — the
# same pair inv_single validated for CVD separation.
BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "#9aa0a6"
STEP = "hvh"  # mid-step: the state is integral, so the rule is piecewise flat


def _stair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Explicit mid-step staircase vertices, so a filled edge steps too."""
    xs = np.repeat(x, 2)[1:]
    ys = np.repeat(y, 2)[:-1]
    return xs, ys


def build_figure(D: dict, title: str, static: bool = False) -> go.Figure:
    N = D["n_echelons"]
    fig = make_subplots(
        rows=1, cols=N, shared_yaxes=False,
        subplot_titles=[f"echelon {k + 1} — ȳ = {D['ybar'][k]}" for k in range(N)],
        horizontal_spacing=0.06,
    )
    for k in range(N):
        u, y, clipped = D["u"][:, k], D["y"][:, k], D["clipped"][:, k]
        lo = float(min(u.min(), y.min())) - 2
        hi = float(max(u.max(), y.max(), D["ybar"][k])) + 4
        grid = np.arange(np.floor(lo), np.ceil(hi) + 1)

        # y = u — "order nothing"
        fig.add_trace(go.Scatter(
            x=grid, y=grid, mode="lines", name="order nothing (y = u)",
            line=dict(color=GREY, width=1, dash="dot"),
            showlegend=(k == 0), hoverinfo="skip"), row=1, col=k + 1)

        # the paper's rule, ignoring the availability clip: y = max(u, ybar)
        ref = np.maximum(grid, D["ybar"][k])
        sx, sy = _stair(grid, ref)
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="lines", name="Clark & Scarf: y = max(u, ȳ)",
            line=dict(color=ORANGE, width=2.5, shape=STEP),
            showlegend=(k == 0),
            hovertemplate="u=%{x:.0f}<br>DP y=%{y:.0f}<extra></extra>"),
            row=1, col=k + 1)

        # the learned rule, split by whether the availability clip binds
        for mask, nm, col, sym in ((~clipped, "PPO (clip slack)", BLUE, "circle"),
                                   (clipped, "PPO (clip binding)", GREY, "x")):
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=u[mask], y=y[mask], mode="markers", name=nm,
                marker=dict(color=col, size=5, symbol=sym,
                            opacity=0.45 if not static else 0.6,
                            line=dict(width=0)),
                showlegend=(k == 0),
                customdata=(y[mask] - u[mask]),
                hovertemplate="u=%{x:.0f}<br>y=%{y:.0f}<br>q=%{customdata:.0f}"
                              "<extra></extra>"), row=1, col=k + 1)

        fig.add_hline(y=D["ybar"][k], line=dict(color=ORANGE, width=1, dash="dash"),
                      row=1, col=k + 1)
        fig.update_xaxes(title_text="echelon position before ordering  u",
                         row=1, col=k + 1)
        fig.update_yaxes(title_text="after ordering  y" if k == 0 else None,
                         row=1, col=k + 1)

    fig.update_layout(
        title=title, template="plotly_white",
        width=D["width"], height=D["height"],
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0),
        margin=dict(l=70, r=30, t=110, b=60),
    )
    return fig


def build_sweep_figure(D: dict, title: str, static: bool = False) -> go.Figure:
    """The policy as a FUNCTION: one input, one output, over an enumerated grid.

    Not a sample of anything. Each point is a synthetic state queried
    deterministically, so the curve is the rule itself — including where it
    stops ordering, which no trajectory sample reaches because a good policy
    does not visit deeply overstocked states.
    """
    N = D["n_echelons"]
    fig = make_subplots(
        rows=1, cols=N, shared_yaxes=False,
        subplot_titles=[f"echelon {k + 1} — ȳ = {D['ybar'][k]}" for k in range(N)],
        horizontal_spacing=0.06)
    for k in range(N):
        u, y = D["u"][k], D["y"][k]
        grid = np.arange(np.floor(u.min()), np.ceil(u.max()) + 1)
        fig.add_trace(go.Scatter(
            x=grid, y=grid, mode="lines", name="order nothing (y = u)",
            line=dict(color=GREY, width=1, dash="dot"),
            showlegend=(k == 0), hoverinfo="skip"), row=1, col=k + 1)
        sx, sy = _stair(grid, np.maximum(grid, D["ybar"][k]))
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="lines", name="Clark & Scarf: y = max(u, ȳ)",
            line=dict(color=ORANGE, width=2.5, shape=STEP), showlegend=(k == 0),
            hovertemplate="u=%{x:.0f}<br>DP y=%{y:.0f}<extra></extra>"),
            row=1, col=k + 1)
        px, py = _stair(u, y)
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="lines", name="the learned rule",
            line=dict(color=BLUE, width=2.5, shape=STEP), showlegend=(k == 0),
            hovertemplate="u=%{x:.0f}<br>y=%{y:.0f}<extra></extra>"),
            row=1, col=k + 1)
        fig.add_hline(y=D["ybar"][k], line=dict(color=ORANGE, width=1, dash="dash"),
                      row=1, col=k + 1)
        fig.update_xaxes(title_text="echelon position before ordering  u",
                         row=1, col=k + 1)
        fig.update_yaxes(title_text="after ordering  y" if k == 0 else None,
                         row=1, col=k + 1)
    fig.update_layout(
        title=title, template="plotly_white",
        width=D["width"], height=D["height"],
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0),
        margin=dict(l=70, r=30, t=110, b=60))
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay the PPO and DP shipping rules.")
    p.add_argument("-s", "--scenario_name", default="n3_l2_p09",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--model-path", required=True)
    p.add_argument("-o", "--observation_mode", default=None)
    p.add_argument("-a", "--action_mode", default=None)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--period", type=int, default=None,
                   help="which period to plot; ȳ is time-varying on a finite "
                        "horizon, so mixing periods would blur the plateau. "
                        "Default: mid-horizon.")
    p.add_argument("--outstem", default=None)
    p.add_argument("--width", type=int, default=1180)
    p.add_argument("--height", type=int, default=460)
    p.add_argument("--inline-js", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="ENUMERATE the input and read the output: the policy as "
                        "a deterministic function over a grid of echelon "
                        "positions, with the rest of the chain at the DP "
                        "optimum and the source freed. Swept past every "
                        "critical number, so the plot shows where ordering "
                        "STOPS. Nothing to do with training or eval draws.")
    p.add_argument("--span", type=float, default=1.5,
                   help="sweep width per echelon, as a multiple of the distance "
                        "from its floor to its critical number. 1.5 puts the "
                        "kink at the right third of the axis.")
    args = p.parse_args()

    sc = SCENARIOS[args.scenario_name]
    assert not callable(sc), "needs a fixed scenario"
    model_path = Path(args.model_path).resolve()
    run_dir = model_path.parent

    ra = {}
    f = run_dir / f"{args.scenario_name}_ppo_args.txt"
    if f.exists():
        ra = dict(l.split(": ", 1) for l in f.read_text().splitlines() if ": " in l)
    obs_mode = args.observation_mode or ra.get("observation_mode", "raw").strip()
    act_mode = args.action_mode or ra.get("action_mode", "ship_discrete").strip()
    period = args.period if args.period is not None else sc.horizon // 2

    dp = ClarkScarfDP(sc)
    act = _load_policy(model_path, sc, obs_mode, act_mode)
    if args.sweep:
        S = sweep_policy(sc, act, obs_mode, act_mode, period, args.span)
        N = sc.n_echelons
        D = {"n_echelons": N, "u": S["u"], "y": S["y"],
             "ybar": dp.ybar[period][:N].astype(int),
             "width": args.width, "height": args.height}
        print(f"[plot] SWEEP (enumerated states)  {obs_mode}/{act_mode}  "
              f"period={period}  "
              f"{[len(a) for a in S['u']]} inputs per echelon")
        print(f"[plot] DP critical numbers at t={period}: {D['ybar'].tolist()}")
        for k in range(N):
            q, u = S["q"][k], S["u"][k]
            off = u[q <= 0]
            print(f"[plot]   echelon {k+1}: DP ȳ={D['ybar'][k]:3d} | u "
                  f"{u.min():.0f}..{u.max():.0f} | stops ordering at "
                  f"u={off[0]:.0f} ({100*(off[0]-u.min())/(u.max()-u.min()):.0f}% "
                  f"across the axis)" if len(off) else
                  f"[plot]   echelon {k+1}: DP ȳ={D['ybar'][k]:3d} | policy NEVER "
                  f"stops ordering over the swept range")
        stem = args.outstem or f"policy_sweep_{args.scenario_name}_{obs_mode}_t{period}"
        title = (f"The learned rule as a function — {args.scenario_name}, "
                 f"obs={obs_mode}, period {period}")
        figures = Path("figures"); figures.mkdir(exist_ok=True)
        svg = figures / f"{stem}.svg"
        build_sweep_figure(D, title, static=True).write_image(str(svg))
        print(f"[plot] static  -> {svg}")
        out = Path("results") / args.scenario_name / "figures"
        out.mkdir(parents=True, exist_ok=True)
        build_sweep_figure(D, title).write_html(
            str(out / f"{stem}.html"),
            include_plotlyjs=True if args.inline_js else "cdn")
        print(f"[plot] interactive -> {out / (stem + '.html')}")
        return

    d = collect(sc, act, obs_mode, act_mode, args.episodes)

    sel = d["t"] == period
    assert sel.any(), f"no decisions recorded at period {period}"
    N = sc.n_echelons
    u, q, cap = d["u"][sel][:, :N], d["q"][sel][:, :N], d["cap"][sel][:, :N]
    D = {
        "n_echelons": N,
        "u": u,
        "y": u + q,
        # the clip binds when the policy shipped everything available
        "clipped": q >= cap - 1e-9,
        "ybar": dp.ybar[period][:N].astype(int),
        "width": args.width, "height": args.height,
    }
    print(f"[plot] {obs_mode}/{act_mode}  period={period}  n={int(sel.sum())} decisions")
    print(f"[plot] DP critical numbers at t={period}: {D['ybar'].tolist()}")
    for k in range(N):
        print(f"[plot]   echelon {k+1}: clip binds on "
              f"{100 * D['clipped'][:, k].mean():.1f}% of decisions")

    stem = args.outstem or f"policy_{args.scenario_name}_{obs_mode}_t{period}"
    title = (f"Learned shipping rule vs Clark & Scarf — {args.scenario_name}, "
             f"obs={obs_mode}, period {period}")

    figures = Path("figures")
    figures.mkdir(exist_ok=True)
    svg = figures / f"{stem}.svg"
    build_figure(D, title, static=True).write_image(str(svg))
    print(f"[plot] static  -> {svg}")

    hdir = Path("results") / args.scenario_name / "figures"
    hdir.mkdir(parents=True, exist_ok=True)
    html = hdir / f"{stem}.html"
    build_figure(D, title).write_html(
        str(html), include_plotlyjs=("inline" if args.inline_js else "cdn"))
    print(f"[plot] interactive -> {html}")


if __name__ == "__main__":
    main()

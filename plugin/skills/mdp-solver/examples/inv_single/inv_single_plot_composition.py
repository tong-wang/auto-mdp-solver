"""Is inventory position a sufficient statistic? (campaign question 4, §14.3)

The claim is a DIFFERENTIAL, not a shape, so this figure does not use the
canonical (x = position before, y = after) coordinates that
`inv_single_plot_policy.py` uses. It tried: in the band where the policy
actually operates it is near enough order-up-to that every curve is flat near
y = S, and a one-unit dependence on the pipeline is invisible against an axis
the y = x diagonal has already stretched over twenty units. The effect is small
in LEVEL and large in COST, so the level plot is the wrong instrument.

Here the deviation itself is the y-axis:

    x = lateness = sum(k * pipe[k]) / sum(pipe),  how far out the stock sits
    y = order  MINUS  the mean order at that same inventory position

Subtracting the per-IP mean is what makes the figure a test. The best possible
rule that reads only inventory position must emit ONE number per IP, so in
these coordinates it is the flat line y = 0 — by construction, with no fitting.
Any slope is behaviour no IP-only rule can reproduce, and the panel title
carries what that slope is worth in cost (#E12).

One faint line per inventory-position level, one bold line for their mean.
Compositions come from an on-policy roll, so only states the net actually
visits are drawn — synthesizing (inventory, pipeline) splits would measure
extrapolation, not policy (#E10). The roll is seeded, so a redraw reproduces.

Read the two panels as SHAPES, never as scores: `lt` and `slt` are separate
leaderboards and their costs are not comparable. What is comparable is slope.

One spec, two renders, written in the same run so they cannot drift:
  * committed static SVG -> inv_single/figures/
  * interactive HTML     -> results/{scenario}/figures/  (gitignored)

    python inv_single_plot_composition.py \
        --arm lt=<crowned_lt>/ppo_inv_single.zip \
        --arm slt=<crowned_slt>/ppo_inv_single.zip
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_LATENESS_BINS = 7


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="SCENARIO=MODEL",
                   help="one panel per arm; repeat. vecnormalize.pkl is taken "
                        "from the model's own directory")
    p.add_argument("-o", "--observation_mode", type=str, default="vec")
    p.add_argument("-a", "--action_mode", type=str, default="discrete")
    p.add_argument("--period", type=int, default=10,
                   help="period held fixed while composition varies")
    p.add_argument("--comp-seed", type=int, default=0)
    p.add_argument("--comp-steps", type=int, default=900)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--min-comps", type=int, default=9,
                   help="skip an IP level with fewer visited compositions")
    p.add_argument("--ip-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                   help="inventory-position band to draw. Default: the visited "
                        "IQR, which is the band the probe reports its gradient "
                        "over — so the figure and #E12's numbers agree. Widen it "
                        "to see the saturated tails, where the fan necessarily "
                        "closes because the net is ordering nothing or the max")
    p.add_argument("--outdir", type=str, default="results")
    return p


def _price(model_dir: Path, scenario: str):
    """The cost this structure is worth, if the probe has already scored it.

    Read rather than recomputed: the figure must quote the same number the
    ledger does, and re-deriving it here would be a second source of truth.
    """
    def mean(path):
        if not path.exists():
            return None
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                return float(row["cost_total_mean"])
        return None

    net = mean(model_dir / f"ppo_eval_{scenario}.tsv")
    fit = mean(model_dir / f"fitted_iponly_eval_{scenario}.tsv")
    return None if net is None or fit is None else fit - net


def lateness_response(model, normalizer, scenario, obs_mode, act_mode, states,
                      period, min_comps: int, ip_range=None):
    """Order deviation from the per-IP mean, as a function of lateness.

    Every point is compared only against other states at the SAME inventory
    position, so inventory position is differenced out rather than controlled
    for statistically. What survives is the part of the action that depends on
    when the outstanding stock lands — precisely what an IP-only rule discards.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = InvSingleEnv(scenario, observation_mode=obs_mode, action_mode=act_mode)
    env.reset(seed=0)
    base = env._state

    def order_at(inv, pipe):
        env._state = dataclasses.replace(base, period=int(period), inventory=int(inv),
                                         pipeline=list(pipe), terminated=False)
        obs = env._get_obs().reshape(1, -1)
        if normalizer is not None:
            obs = normalizer.normalize_obs(obs)
        act, _ = model.predict(obs, deterministic=True)
        return float(env.decode_action(np.asarray(act).reshape(-1)))

    by_ip = {}
    for (_, ip), comps in states.items():
        by_ip.setdefault(ip, set()).update(comps)
    if ip_range is None:
        visited = np.array([ip for (_, ip) in states])
        lo, hi = int(np.percentile(visited, 25)), int(np.percentile(visited, 75))
    else:
        lo, hi = ip_range

    per_ip = {}
    for ip in sorted(k for k in by_ip if lo <= k <= hi):
        pts = []
        for inv, pipe in sorted(by_ip[ip]):
            tot = sum(pipe)
            if tot == 0:
                continue  # nothing outstanding: no composition to vary
            pts.append((sum(k * v for k, v in enumerate(pipe)) / tot, order_at(inv, pipe)))
        if len(pts) >= min_comps:
            per_ip[ip] = np.asarray(pts)

    if not per_ip:
        return np.array([]), {}, np.array([])
    lat_all = np.concatenate([v[:, 0] for v in per_ip.values()])
    edges = np.linspace(lat_all.min(), lat_all.max(), N_LATENESS_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    lines = {}
    for ip, v in per_ip.items():
        resid = v[:, 1] - v[:, 1].mean()          # <- the per-IP mean removed
        idx = np.clip(np.digitize(v[:, 0], edges) - 1, 0, N_LATENESS_BINS - 1)
        lines[ip] = np.array([resid[idx == b].mean() if (idx == b).any() else np.nan
                              for b in range(N_LATENESS_BINS)])
    stack = np.vstack(list(lines.values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # all-NaN bins are legitimate
        mean = np.nanmean(stack, axis=0)
    return centres, lines, mean


def main() -> None:
    args = _build_arg_parser().parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import inv_single_ordinal_head  # noqa: F401  (policy class must be importable)
    from inv_single_policy_probe import _roll_visited

    global InvSingleEnv
    from inv_single_gym import InvSingleEnv  # noqa: F811
    from inv_single_scenarios import SCENARIOS

    spec = {"period": args.period, "panels": []}
    for arm in args.arm:
        name, _, model_path = arm.partition("=")
        model_dir = Path(model_path).resolve().parent
        sc = SCENARIOS[name]
        model = PPO.load(model_path, device="cpu")
        vecnorm = model_dir / "vecnormalize.pkl"

        normalizer = None
        if vecnorm.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dummy = DummyVecEnv([lambda: InvSingleEnv(
                    sc, observation_mode=args.observation_mode,
                    action_mode=args.action_mode)])
            normalizer = VecNormalize.load(str(vecnorm), dummy)
            normalizer.training = False

        states, _ = _roll_visited(model, vecnorm if vecnorm.exists() else None, sc,
                                  args.observation_mode, args.action_mode,
                                  n_envs=args.batch_envs, n_steps=args.comp_steps,
                                  seed=args.comp_seed)
        centres, lines, mean = lateness_response(
            model, normalizer, sc, args.observation_mode, args.action_mode,
            states, args.period, args.min_comps, args.ip_range)
        lt = sc.leadtime
        sub = (f"L = {lt.value} fixed" if hasattr(lt, "value")
               else f"L ~ uniform{{{', '.join(str(v) for v in lt.values)}}}")
        # slope per UNIT of lateness, not the raw end-to-end rise: a stochastic
        # lead time reaches a wider spread of pipeline shapes, so the two panels
        # do not span the same x-range and the raw rise would flatter `slt`
        ok = ~np.isnan(mean)
        slope = float(np.polyfit(centres[ok], mean[ok], 1)[0]) if ok.sum() > 1 else float("nan")
        spec["panels"].append({
            "scenario": name, "subtitle": sub, "x": centres.tolist(),
            "lines": {str(k): v.tolist() for k, v in lines.items()},
            "mean": mean.tolist(), "slope": slope,
            "price": _price(model_dir, name),
        })
        print(f"[plot] {name}: {len(lines)} IP levels, "
              f"slope {slope:+.2f} per unit lateness "
              f"(raw rise {mean[ok][-1] - mean[ok][0]:+.2f} over "
              f"{centres[ok][-1] - centres[ok][0]:.2f})")

    _render_static(spec, HERE / "figures")
    for panel in spec["panels"]:
        _render_interactive(spec, HERE / args.outdir / panel["scenario"] / "figures")


def _panel_title(panel):
    t = f"{panel['scenario']}  —  {panel['subtitle']}"
    t += f"\nslope {panel['slope']:+.2f} per unit lateness"
    if panel["price"] is not None:
        t += f"  ·  restricting to inventory position costs {panel['price']:+.2f}"
    return t


def _render_static(spec, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = spec["panels"]
    lim = max(np.nanmax(np.abs(np.asarray(list(p["lines"].values())))) for p in panels)
    # x shared as well as y: the panels are only comparable if the same
    # lateness means the same place on both. `lt` then visibly stops short —
    # a fixed lead time cannot produce the far-out pipelines `slt` reaches,
    # and that is a fact about the scenarios, not an empty axis
    fig, axes = plt.subplots(1, len(panels), figsize=(6.1 * len(panels), 4.9),
                             sharey=True, sharex=True)
    for ax, panel in zip(np.atleast_1d(axes), panels):
        xs = np.asarray(panel["x"])
        for v in panel["lines"].values():
            ax.plot(xs, v, color="#d73027", lw=0.9, alpha=0.28)
        ax.axhline(0, color="#1f77b4", lw=5, alpha=0.35, solid_capstyle="round",
                   label="any rule reading only inventory position")
        ax.plot(xs, panel["mean"], color="#d73027", lw=2.6, marker="o", ms=5,
                label="the trained policy (mean over IP levels)")
        ax.set_title(_panel_title(panel), fontsize=9.5)
        ax.set_xlabel("lateness of the outstanding stock\n"
                      "(0 = all landing this period  →  higher = further out)")
        ax.set_ylim(-1.15 * lim, 1.15 * lim)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel(
        "order MINUS the mean order at the same\ninventory position   (units)")
    fig.suptitle(
        "A policy reading only inventory position is the flat line y = 0 here, by construction.\n"
        f"Slope ⇒ the same inventory position is treated differently depending on WHEN the stock lands "
        f"(t = {spec['period']}, visited states only)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "composition_lateness.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] static      → {out}")


def _render_interactive(spec, outdir: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = spec["panels"]
    fig = make_subplots(rows=1, cols=len(panels), shared_yaxes=True,
                        subplot_titles=[_panel_title(p).replace("\n", "<br>")
                                        for p in panels])
    for i, panel in enumerate(panels, start=1):
        xs = panel["x"]
        for ip, v in panel["lines"].items():
            fig.add_trace(go.Scatter(x=xs, y=v, mode="lines", name=f"IP {ip}",
                                     line=dict(color="#d73027", width=1),
                                     opacity=0.3, showlegend=False,
                                     hovertemplate=f"IP {ip}: %{{y:.2f}}<extra></extra>"),
                          row=1, col=i)
        fig.add_trace(go.Scatter(x=xs, y=[0] * len(xs), mode="lines",
                                 name="any IP-only rule",
                                 line=dict(color="#1f77b4", width=9), opacity=0.35,
                                 showlegend=(i == 1)), row=1, col=i)
        fig.add_trace(go.Scatter(x=xs, y=panel["mean"], mode="lines+markers",
                                 name="the trained policy",
                                 line=dict(color="#d73027", width=3),
                                 marker=dict(size=7), showlegend=(i == 1)),
                      row=1, col=i)
        fig.update_xaxes(title_text="lateness (0 = landing now)", row=1, col=i)
    fig.update_yaxes(title_text="order − mean order at same IP", row=1, col=1)
    fig.update_layout(
        title="Is inventory position sufficient? Flat = yes, sloped = no "
              f"(t = {spec['period']}, visited states only)",
        height=560, hovermode="x unified")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "composition_lateness.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[plot] interactive → {out}")


if __name__ == "__main__":
    main()

"""Policy figures in canonical coordinates (spec §14.3).

x = inventory position BEFORE ordering, y = inventory position AFTER ordering.
In these coordinates the predicted structures are canonical shapes:

    base-stock  ->  a flat line at S, meeting the y = x no-act diagonal at S
    (s, S)      ->  a flat plateau at S that JOINS the diagonal at s, and
                    follows the diagonal (order nothing) for every x > s

so a deviation from the form is visible as raggedness rather than something you
have to compute. The reference (exact DP) is drawn UNDERNEATH the trained
policy, never side by side, so agreement and deviation regions read directly
(§14.3).

One spec, two renders, written in the same run so they cannot drift:
  * committed static SVG -> inv_single/figures/          (README inlines this)
  * interactive HTML     -> results/{scenario}/figures/  (gitignored, beside
                            the runs it derives from; the location split IS the
                            gitignore boundary)

Committed markdown cites the regenerating command below, never the HTML path —
that file does not exist in a fresh clone.

    python inv_single_plot_policy.py \
        --model-path <crowned>/ppo_inv_single.zip \
        --vecnorm-path <crowned>/vecnormalize.pkl -s simple_k
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None)
    p.add_argument("-s", "--scenario_name", type=str, default="simple")
    p.add_argument("-o", "--observation_mode", type=str, default="vec")
    p.add_argument("-a", "--action_mode", type=str, default="discrete")
    p.add_argument("--x-min", type=int, default=-6)
    p.add_argument("--x-max", type=int, default=34)
    p.add_argument("--regular-period", type=int, default=10,
                   help="a period in the stationary region")
    p.add_argument("--outdir", type=str, default="results")
    return p


def policy_curve(env, model, venv, scenario, xs, period):
    """y = x + q(x, t), read through the gym's own decoder so the figure can
    never disagree with what `step` would do."""
    ys = []
    for x in xs:
        env._state = dataclasses.replace(env._state, period=period, inventory=int(x))
        obs = env._get_obs().reshape(1, -1)
        if venv is not None:
            obs = venv.normalize_obs(obs)
        act, _ = model.predict(obs, deterministic=True)
        ys.append(x + env.decode_action(act))
    return np.asarray(ys, dtype=float)


def dp_curve(scenario_name, xs, period):
    meta = json.load(open(HERE / f"results/{scenario_name}/benchmark/dp/meta.json"))
    pi = np.loadtxt(HERE / f"results/{scenario_name}/benchmark/dp/pi.csv", delimiter=",")
    grid = np.arange(meta["x_min"], meta["x_max"] + 1)
    idx = {int(v): j for j, v in enumerate(grid)}
    out = []
    for x in xs:
        j = idx.get(int(np.clip(x, meta["x_min"], meta["x_max"])))
        out.append(x + float(pi[period, j]))
    return np.asarray(out, dtype=float)


def main() -> None:
    args = _build_arg_parser().parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import inv_single_ordinal_head  # noqa: F401  (policy class must be importable)
    from inv_single_gym import InvSingleEnv
    from inv_single_scenarios import SCENARIOS

    sc = SCENARIOS[args.scenario_name]
    horizon = sc.horizon
    periods = [(args.regular_period, f"regular period (t = {args.regular_period})"),
               (horizon - 1, f"last period (t = {horizon - 1})")]

    env = InvSingleEnv(sc, observation_mode=args.observation_mode,
                       action_mode=args.action_mode)
    env.reset(seed=0)
    venv = None
    if args.vecnorm_path:
        venv = VecNormalize.load(args.vecnorm_path, DummyVecEnv(
            [lambda: InvSingleEnv(sc, observation_mode=args.observation_mode,
                                  action_mode=args.action_mode)]))
        venv.training = False
    model = PPO.load(args.model_path, device="cpu")

    xs = np.arange(args.x_min, args.x_max + 1)
    spec = {"scenario": args.scenario_name, "x": xs.tolist(), "panels": []}
    for t, title in periods:
        spec["panels"].append({
            "title": title, "period": t,
            "ppo": policy_curve(env, model, venv, sc, xs, t).tolist(),
            "dp": dp_curve(args.scenario_name, xs, t).tolist(),
        })

    _render_static(spec, HERE / "figures")
    _render_interactive(spec, HERE / args.outdir / args.scenario_name / "figures")


def _render_static(spec, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.asarray(spec["x"])
    fig, axes = plt.subplots(1, len(spec["panels"]), figsize=(11, 4.4), sharey=True)
    for ax, panel in zip(np.atleast_1d(axes), spec["panels"]):
        ax.plot(xs, xs, color="0.75", lw=1, ls=":", label="y = x  (order nothing)")
        # reference UNDERNEATH the trained policy (§14.3)
        ax.plot(xs, panel["dp"], color="#1f77b4", lw=5, alpha=0.35, solid_capstyle="round",
                label="exact DP")
        ax.plot(xs, panel["ppo"], color="#d62728", lw=1.6, label="PPO (crowned)")
        ax.set_title(panel["title"], fontsize=10)
        ax.set_xlabel("inventory position BEFORE ordering  (x)")
        ax.grid(alpha=0.25, lw=0.5)
    np.atleast_1d(axes)[0].set_ylabel("inventory position AFTER ordering  (y = x + q)")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{spec['scenario']} — policy in canonical coordinates "
                 f"(base-stock = flat line; (s,S) = plateau joining the diagonal at s)",
                 fontsize=11)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"policy_overlay_{spec['scenario']}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] static      → {out}")


def _render_interactive(spec, outdir: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    xs = spec["x"]
    fig = make_subplots(rows=1, cols=len(spec["panels"]), shared_yaxes=True,
                        subplot_titles=[p["title"] for p in spec["panels"]])
    for i, panel in enumerate(spec["panels"], start=1):
        fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="y = x",
                                 line=dict(color="lightgray", dash="dot"),
                                 showlegend=(i == 1)), row=1, col=i)
        fig.add_trace(go.Scatter(x=xs, y=panel["dp"], mode="lines", name="exact DP",
                                 line=dict(color="#1f77b4", width=9), opacity=0.35,
                                 showlegend=(i == 1)), row=1, col=i)
        fig.add_trace(go.Scatter(x=xs, y=panel["ppo"], mode="lines+markers",
                                 name="PPO (crowned)", line=dict(color="#d62728", width=2),
                                 marker=dict(size=4), showlegend=(i == 1)), row=1, col=i)
        fig.update_xaxes(title_text="x (before ordering)", row=1, col=i)
    fig.update_yaxes(title_text="y = x + q (after ordering)", row=1, col=1)
    fig.update_layout(title=f"{spec['scenario']} — policy in canonical coordinates",
                      height=480, hovermode="x unified")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"policy_overlay_{spec['scenario']}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[plot] interactive → {out}")


if __name__ == "__main__":
    main()

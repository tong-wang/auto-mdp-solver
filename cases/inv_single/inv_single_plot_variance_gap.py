"""What the pipeline is worth, against how variable the lead time is (§14.3).

y is the paired difference `vec_ctx_slt − vec_ip_ctx_slt`: two policies on the
same 45 cells, differing only in whether they see the pipeline or its sum. x is
`Var(L)`, swept at MATCHED MEAN so nothing here is a lead-time-length effect.

**The y = 0 line is a control, not a decoration.** At Var(L) = 0 the lead time
is deterministic, inventory position is provably sufficient (Karlin-Scarf; the
d=1 distillation of #E13 agrees), so the two modes carry IDENTICAL information
and their difference must be zero. Any offset there is learnability — one
observation being harder to fit than another — and it has to be read out before
the rest of a curve means anything.

Three rungs are drawn because that offset is what changed between them: +75.01
at the §8.6-derived config, +23.30 once this campaign's own levers were carried,
+0.54 tuned. Only the last curve starts at the control, which is what licenses
reading its slope as information.

One spec, two renders:
  * committed static SVG -> inv_single/figures/
  * interactive HTML     -> results/lt_variance_k0/figures/  (gitignored)

    python inv_single_plot_variance_gap.py
"""
from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VAR = {"p00": 0.0, "p17": 1 / 3, "p25": 0.5, "p33": 2 / 3, "p50": 1.0}
# Each rung is summarised by the MEAN over its arms, one arm per run (its best
# checkpoint) — never best-arm-vs-best-arm. Best-vs-best is a maximum over
# replicates on both sides, so it inherits whichever mode got luckier: at the
# derived rung one `vec_ip` seed landed at 601.5 against its siblings' 750–906
# and inflates the gap to +205, while at the carried rung a lucky `vec` seed
# shrinks it to +4.5. The rung mean is the honest statistic for a rung; a single
# crowned artifact is a different object and is quoted as such in the ledger.
RUNGS = [
    ("L1, §8.6-derived", ["results/lt_variance_k0/PPO_*k0L1v/checkpoints/ppo_eval_*.tsv"],
                         ["results/lt_variance_k0/PPO_*k0L1p/checkpoints/ppo_eval_*.tsv"], "#bbbbbb"),
    ("L1, carried levers", ["results/lt_variance_k0/PPO_*k0Cv/checkpoints/ppo_eval_*.tsv"],
                           ["results/lt_variance_k0/PPO_*k0Cp/checkpoints/ppo_eval_*.tsv"], "#4575b4"),
    ("tuned", None, None, "#d73027"),      # resolved from the confirmed trials
]


def _arms(patterns: list) -> list:
    """One arm per RUN (its best checkpoint), each as {(p, b): cost}.

    Only complete arms count: a partial 45-cell set would silently average over
    a different set of cells than its siblings.
    """
    byrun: dict = {}
    for pattern in patterns:
        for f in glob.glob(pattern, recursive=True):
            m = re.search(r"__(p\d\d)_b(\d)_K0__(\S+?)\.tsv$", f)
            if not m:
                continue
            run = f.split("/checkpoints")[0] if "/checkpoints" in f else f.rsplit("/", 1)[0]
            with open(f) as fh:
                row = list(csv.DictReader(fh, delimiter="\t"))[-1]
            byrun.setdefault((run, m.group(3)), {})[(m.group(1), int(m.group(2)))] = \
                float(row["cost_total_mean"])
    best: dict = {}
    for (run, _ck), c in byrun.items():
        if len(c) != 45:
            continue
        a = float(np.mean(list(c.values())))
        if run not in best or a < best[run][0]:
            best[run] = (a, c)
    return [c for _, c in best.values()]


def _confirmed_tuned() -> tuple:
    """The tuned rung is its CONFIRMED trials, not all 120 searched ones.

    Sweeping every trial would average the search itself — including its
    failures, which score ~23000 — and report a rung nobody ran.
    """
    dirs = [d.rsplit("/", 1)[0]
            for d in glob.glob("results/tuning/k0_*/trial_*/confirm2048.tsv")]
    return ([f"{d}/**/ppo_eval_*.tsv" for d in dirs if "_vec_" in d],
            [f"{d}/**/ppo_eval_*.tsv" for d in dirs if "_ip_" in d])


def build_spec() -> dict:
    series = []
    for label, gv, gi, colour in RUNGS:
        if gv is None:
            gv, gi = _confirmed_tuned()
        V, I = _arms(gv), _arms(gi)
        if not V or not I:
            continue
        pts = []
        for p, var in VAR.items():
            a = float(np.mean([np.mean([c[(p, b)] for b in range(1, 10)]) for c in V]))
            c_ = float(np.mean([np.mean([c[(p, b)] for b in range(1, 10)]) for c in I]))
            pts.append((var, a - c_))
        series.append({"label": f"{label}  (n={len(V)}/{len(I)} arms)",
                       "colour": colour,
                       "x": [x for x, _ in pts], "y": [y for _, y in pts]})
    return {"series": series}


def _render_static(spec: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.axhline(0, color="#333333", lw=2.5, alpha=0.55, zorder=1)
    ax.text(1.005, 0.4, "y = 0: what two modes carrying the SAME\ninformation must score",
            fontsize=8, color="#333333", va="bottom")
    for s in spec["series"]:
        ax.plot(s["x"], s["y"], marker="o", ms=6, lw=2.2,
                color=s["colour"], label=s["label"], zorder=3)
    ax.set_xlabel("Var(L)   —   lead-time variability at MATCHED MEAN E[L] = 2")
    ax.set_ylabel("cost(sees pipeline) − cost(sees only its sum)\n"
                  "negative = the pipeline is worth something")
    ax.set_title("What the pipeline is worth, and how much of that was ever real\n"
                 "45 cells, protocol block; the y=0 intercept is the learnability control",
                 fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, loc="upper right", title="rung")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "variance_gap.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] static      → {out}")


def _render_interactive(spec: dict, outdir: Path) -> None:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#333333", width=3), opacity=0.55)
    for s in spec["series"]:
        fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="lines+markers",
                                 name=s["label"], line=dict(color=s["colour"], width=3),
                                 marker=dict(size=8)))
    fig.update_layout(
        title="What the pipeline is worth vs lead-time variance (y=0 is the control)",
        xaxis_title="Var(L) at matched mean E[L]=2",
        yaxis_title="vec − vec_ip  (negative = pipeline pays)",
        height=520, hovermode="x unified")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "variance_gap.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[plot] interactive → {out}")


if __name__ == "__main__":
    spec = build_spec()
    for s in spec["series"]:
        print(f"  {s['label']:<22} " + "  ".join(f"{y:+7.2f}" for y in s["y"]))
    _render_static(spec, HERE / "figures")
    _render_interactive(spec, HERE / "results/lt_variance_k0/figures")

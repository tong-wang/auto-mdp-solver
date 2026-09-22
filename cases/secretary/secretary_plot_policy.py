"""Render the PPO/reference secretary action overlay from probe output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot secretary policy readback.")
    parser.add_argument("--model-path", required=True)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    model_path = Path(args.model_path).resolve()
    run_dir = model_path.parent
    domain_dir = Path(__file__).resolve().parent
    with (run_dir / "probe/action_surface.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    records = [row for row in rows if int(row["relative_rank"]) == 1 and int(row["period"]) < 99]
    periods = [int(row["period"]) for row in records]
    probabilities = [float(row["accept_probability"]) for row in records]
    net_actions = [int(row["net_action"]) for row in records]
    reference = [int(row["reference_action"]) for row in records]

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.step(periods, reference, where="post", linewidth=4, alpha=0.35, label="Exact cutoff 37")
    axis.plot(periods, probabilities, color="#d95f02", linewidth=1.5, label="PPO accept probability")
    axis.scatter(periods, net_actions, s=10, color="#1b9e77", label="PPO deterministic action")
    axis.axvline(37, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Candidates already rejected (period)", ylabel="Accept current record", ylim=(-0.05, 1.05))
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    figures = domain_dir / "figures"
    figures.mkdir(exist_ok=True)
    static_path = figures / "secretary_policy_overlay.svg"
    fig.savefig(static_path)
    plt.close(fig)

    interactive_dir = domain_dir / "results" / "standard" / "figures"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([
        {"period": p, "probability": q, "action": a, "reference": r}
        for p, q, a, r in zip(periods, probabilities, net_actions, reference)
    ])
    html_path = interactive_dir / "secretary_policy_overlay.html"
    html_path.write_text(f"""<!doctype html><meta charset=\"utf-8\"><title>Secretary policy overlay</title>
<h1>Secretary PPO vs exact cutoff</h1><p>Hover over a point for exact values.</p>
<svg id=\"plot\" width=\"900\" height=\"460\" viewBox=\"0 0 900 460\"></svg>
<script>const data={payload}; const s=document.getElementById('plot');
const x=p=>60+p*8, y=v=>410-v*360;
s.innerHTML='<line x1=\"60\" y1=\"410\" x2=\"860\" y2=\"410\" stroke=\"black\"/><line x1=\"60\" y1=\"50\" x2=\"60\" y2=\"410\" stroke=\"black\"/>';
for(const d of data){{const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',x(d.period));c.setAttribute('cy',y(d.probability));c.setAttribute('r',4);c.setAttribute('fill',d.action?'#1b9e77':'#d95f02');const t=document.createElementNS('http://www.w3.org/2000/svg','title');t.textContent=`period=${{d.period}}, p(accept)=${{d.probability.toFixed(6)}}, action=${{d.action}}, reference=${{d.reference}}`;c.appendChild(t);s.appendChild(c);}}</script>""")
    print(f"static -> {static_path}")
    print(f"interactive -> {html_path}")


if __name__ == "__main__":
    main()

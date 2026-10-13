#!/usr/bin/env python
"""Per-cell paired gate for a grid target (spec §9.6 / §9.9, #E27 protocol).

    python cnv_grid_gate.py ARM.seeds.tsv --bar BAR.seeds.tsv --floor FLOOR.seeds.tsv [--ref REF.seeds.tsv] [--out gate.tsv]

Joins the arm's per-seed rows to each benchmark's on (cell, seed) — the shared
prefix of the CRN block — and reports, per cell, the paired Δ ± SE against the
bar (feasible role), the floor (must-beat) and the reference (exact/relaxed,
judged as a band). Cells are separate leaderboards; the aggregate lines are
summaries. Writes one row per cell to --out (default: next to the arm file).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=["target", "seed", "profit"])
    return df.set_index(["target", "seed"])["profit"]


def paired(arm: pd.Series, other: pd.Series) -> pd.DataFrame:
    j = pd.concat({"a": arm, "b": other}, axis=1, join="inner")
    d = (j["a"] - j["b"]).groupby(level="target")
    n = d.count()
    out = pd.DataFrame({"n": n, "delta": d.mean(), "se": d.std(ddof=1) / np.sqrt(n)})
    out["z"] = out["delta"] / out["se"].replace(0, np.nan)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("arm")
    p.add_argument("--bar", required=True, help="feasible bar per cell (e.g. myopic_*)")
    p.add_argument("--floor", required=True, help="must-beat floor per cell (random_prior_mean)")
    p.add_argument("--ref", default=None, help="exact/relaxed reference per cell (dp_*), judged as a band")
    p.add_argument("--band-se", type=float, default=2.0, help="band half-width in SE units (§9.9)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    arm = load(args.arm)
    cells = arm.groupby(level="target").agg(["mean", "count"])
    res = pd.DataFrame({"arm_mean": cells["mean"], "n_seeds": cells["count"]})
    for name, path in (("bar", args.bar), ("floor", args.floor), ("ref", args.ref)):
        if not path:
            continue
        pr = paired(arm, load(path))
        res[f"{name}_delta"] = pr["delta"]; res[f"{name}_se"] = pr["se"]; res[f"{name}_z"] = pr["z"]
    res["above_floor"] = res["floor_z"] >= 2.0
    res["beats_bar"] = res["bar_z"] >= 2.0
    res["within_bar_band"] = res["bar_delta"] >= -args.band_se * res["bar_se"]
    if args.ref:
        res["within_ref_band"] = res["ref_delta"] >= -args.band_se * res["ref_se"]
    out = Path(args.out) if args.out else Path(args.arm).with_name(Path(args.arm).name.replace(".seeds.tsv", ".gate.tsv"))
    res.to_csv(out, sep="\t", float_format="%.4f")

    k = len(res)
    print(f"cells {k}  (paired on {int(res['n_seeds'].min())}..{int(res['n_seeds'].max())} shared seeds)")
    print(f"mean over cells: arm {res['arm_mean'].mean():.3f}   Δ vs bar {res['bar_delta'].mean():+.3f}   "
          f"Δ vs floor {res['floor_delta'].mean():+.3f}" + (f"   Δ vs ref {res['ref_delta'].mean():+.3f}" if args.ref else ""))
    print(f"above floor (z≥2): {int(res['above_floor'].sum())}/{k}   beats bar (z≥2): {int(res['beats_bar'].sum())}/{k}   "
          f"within {args.band_se:g} SE of bar: {int(res['within_bar_band'].sum())}/{k}"
          + (f"   within {args.band_se:g} SE of ref: {int(res['within_ref_band'].sum())}/{k}" if args.ref else ""))
    worst = res["bar_delta"].idxmin()
    print(f"worst cell vs bar: {worst}  Δ {res.loc[worst, 'bar_delta']:+.3f} ± {res.loc[worst, 'bar_se']:.3f}")
    # by axis: parse "unit_cost=0.2,prior_rate=2,prior_mean=10"
    ax = res.index.to_series().str.extract(r"unit_cost=([\d.]+),prior_rate=([\d.]+),prior_mean=([\d.]+)").astype(float)
    ax.columns = ["unit_cost", "prior_rate", "prior_mean"]
    res2 = res.join(ax)
    for axis in ("prior_mean", "prior_rate", "unit_cost"):
        g = res2.groupby(axis)["bar_delta"].agg(["mean", "min"])
        print(f"Δ vs bar by {axis}: " + "  ".join(f"{v:g}:{m:+.2f}(min {mn:+.2f})" for v, (m, mn) in g.iterrows()))
    print(f"[gate] per-cell -> {out}")


if __name__ == "__main__":
    sys.exit(main())

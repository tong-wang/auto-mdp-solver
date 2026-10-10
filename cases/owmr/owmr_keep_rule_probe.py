"""Retention-step rule probe for owmr (ESCALATION #E33 addendum 2) — the fitted rule of the discover stance.

Question: is the hicv winner's structure (#E33 readback) worth its gain on its own? The probe
scores the LB HEURISTIC with one change — a step retention rule at the warehouse:

    step  = k * on-hand  if on-hand >= theta  else 0
    ship  = min(sum of the heuristic's myopic shipments, on-hand - step), in the heuristic's
            proportions (the heuristic itself leaves stock behind only when every retailer is
            above target, so kept = max(step, its own leftover); equal shares if it ships nothing)

The order is the heuristic's (order up to y0*), so the rule isolates the retention decision the
readback found (keep 0.2 above ~15 units). k = 0 reproduces the heuristic (validation). Sweeping k
in 0.05 steps answers whether the categorical head's menu (0, .1, .2, .3, .5, .8) is fine enough on
this cell: a flat cost in k around the policy's rung means the menu loses nothing.

Scored like the swap probe: closed loop on a CRN block, paired per seed against the heuristic record.
One (k, theta) per invocation (one core each; a launcher runs a grid in parallel); `--summary`
reads every record in the outdir and prints the paired table.

Second form (`--alpha`, #E33 addendum 3 — the first form lost everywhere): SCALED TARGETS. The
heuristic's order; each retailer is shipped up to alpha * z*_i (want_i = max(0, alpha z*_i - IP_i));
when the wants fit on-hand they are shipped and the rest is kept, when they do not the heuristic's
own myopic split of on-hand applies (the shortage regime is unchanged). alpha = 1 is the heuristic.

Usage:
    python owmr_keep_rule_probe.py -s hicv --k 0.2 --theta 15 --n-seeds 8192
    python owmr_keep_rule_probe.py -s hicv --alpha 0.8 --n-seeds 8192
    python owmr_keep_rule_probe.py -s hicv --summary
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

import owmr_mdp as mdp
from owmr_benchmark_lb_heuristic import build as build_heuristic
from owmr_eval_common import eval_seed_block, rollout_seeds, write_record
from owmr_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LB heuristic + a step retention rule (ESCALATION #E33 addendum 2).")
    p.add_argument("-s", "--scenario_name", type=str, default="hicv", choices=list(SCENARIOS))
    p.add_argument("--k", type=float, default=0.2, help="kept fraction of warehouse on-hand above the threshold")
    p.add_argument("--theta", type=float, default=15.0, help="on-hand threshold below which nothing is kept")
    p.add_argument("--alpha", type=float, default=None, help="second form: ship up to alpha * z*_i, keep the rest (k, theta ignored)")
    p.add_argument("--y0", type=float, default=None, help="with --alpha: order up to this echelon level instead of y0* (the order clause, #E33 addendum 4)")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--outdir", type=str, default=None, help="default results/<scenario>/benchmark/keep_rule/")
    p.add_argument("--summary", action="store_true", help="tabulate every record in outdir, paired vs the heuristic")
    return p


def make_act_fn(heuristic, k: float, theta: float):
    def act(obs, envs):
        rows = []
        for e in envs:
            st = e._state
            onhand = max(0.0, float(st.wh_stock))
            h_order, h_ship = heuristic.decide(st)
            hs = np.asarray(h_ship, dtype=np.float64)
            split = hs / hs.sum() if hs.sum() > 0 else np.full(len(hs), 1.0 / len(hs))
            step = k * onhand if onhand >= theta else 0.0
            shipped = min(float(hs.sum()), onhand - step)      # k = 0 -> exactly the heuristic's shipments
            rows.append(np.concatenate([[float(h_order)], shipped * split]))
        return np.asarray(rows, dtype=np.float64)
    return act


def scaled_target_decision(heuristic, state, alpha: float, y0: float | None = None) -> tuple[float, np.ndarray]:
    """The second form's decision on one post-receipt state — the fitted rule of #E33 addenda 3–4,
    also the deployable's `FittedRulePolicy` (owmr_policy.py imports this; one copy of the arithmetic).
    Order up to `y0` (default y0*); ship each retailer up to alpha * z*_i, keep the leftover; the
    heuristic's own myopic split when the wants exceed on-hand."""
    bound = heuristic.bound
    level = heuristic.y0 if y0 is None else float(y0)
    onhand = max(0.0, float(state.wh_stock))
    order = max(0.0, level - mdp.echelon_position(state))
    ip = np.asarray(mdp.inventory_position(state), dtype=float)
    want = np.maximum(0.0, alpha * bound.z - ip)
    ship = want if want.sum() <= onhand else bound.allocate_myopic(onhand, ip)
    return float(order), np.asarray(ship, dtype=np.float64)


def make_alpha_act_fn(heuristic, alpha: float, y0: float | None = None):
    def act(obs, envs):
        rows = []
        for e in envs:
            order, ship = scaled_target_decision(heuristic, e._state, alpha, y0)
            rows.append(np.concatenate([[order], ship]))
        return np.asarray(rows, dtype=np.float64)
    return act


def summary(scenario_name: str, outdir: Path) -> None:
    import pandas as pd
    bar_path = outdir.parent / f"benchmark_lb_heuristic_eval_{scenario_name}.seeds.tsv"
    bar = pd.read_csv(bar_path, sep="\t", comment="#").set_index("seed")
    print(f"bar {scenario_name} lb_heuristic: {bar['cost_total'].mean():.2f}   (paired per seed, n = {len(bar)})")
    print("(second form rows: the k column is alpha; theta = -1 means order up to y0*, otherwise theta is the y0 level)")
    print(f"{'k':>5s} {'theta':>6s} {'cost':>9s} {'d_bar':>9s} {'se':>5s} {'wh_share':>8s} {'d_wh_hold':>9s} {'d_rt_hold':>9s} {'d_short':>8s}")
    rows = []
    for f in sorted(glob.glob(str(outdir / "keep_rule_k*_t*.seeds.tsv")) + glob.glob(str(outdir / "keep_rule_a*.seeds.tsv"))):
        stem = Path(f).name
        if "_a" in stem:    # second form: k column carries alpha; theta column = -1, or the y0 level when one was set
            body = stem.split("_a")[1].split(".seeds")[0]
            k = float(body.split("_y")[0]); th = -1.0 if "_y" not in body else float(body.split("_y")[1])
        else:
            k = float(stem.split("_k")[1].split("_t")[0]); th = float(stem.split("_t")[1].split(".seeds")[0])
        df = pd.read_csv(f, sep="\t", comment="#").set_index("seed")
        m = df.join(bar, rsuffix="_b", how="inner")
        d = m["cost_total"] - m["cost_total_b"]
        rows.append((k, th, m["cost_total"].mean(), d.mean(), d.std(ddof=1) / np.sqrt(len(d)), m["wh_share"].mean(),
                     (m["cost_wh_holding"] - m["cost_wh_holding_b"]).mean(), (m["cost_rt_holding"] - m["cost_rt_holding_b"]).mean(),
                     (m["cost_rt_shortage"] - m["cost_rt_shortage_b"]).mean()))
    for r in sorted(rows, key=lambda r: (r[1], r[0])):
        print(f"{r[0]:5.2f} {r[1]:6.1f} {r[2]:9.2f} {r[3]:+9.2f} {r[4]:5.2f} {r[5]:8.3f} {r[6]:+9.2f} {r[7]:+9.2f} {r[8]:+8.2f}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    here = Path(__file__).resolve().parent
    outdir = Path(args.outdir) if args.outdir else here / "results" / args.scenario_name / "benchmark" / "keep_rule"
    if args.summary:
        summary(args.scenario_name, outdir); return
    outdir.mkdir(parents=True, exist_ok=True)
    heuristic = build_heuristic(args.scenario_name, here)
    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)
    if args.alpha is not None:
        ysfx = "" if args.y0 is None else f"_y{args.y0:g}"
        tag, act_fn = f"keep_rule_a{args.alpha:g}{ysfx}", make_alpha_act_fn(heuristic, args.alpha, args.y0)
        form = (f"scaled targets alpha={args.alpha:g} (ship up to alpha z*, keep the rest; heuristic split when short) — #E33 addendum 3"
                + ("" if args.y0 is None else f"; order up to y0={args.y0:g} instead of y0*={heuristic.y0:.4f} — #E33 addendum 4"))
    else:
        tag, act_fn = f"keep_rule_k{args.k:g}_t{args.theta:g}", make_act_fn(heuristic, args.k, args.theta)
        form = f"step retention k={args.k:g} above theta={args.theta:g} — #E33 addendum 2"
    print(f"[keep_rule] {args.scenario_name}  {form}  y0*={heuristic.y0:.4f}  seeds {args.first_seed}..{args.first_seed + args.n_seeds - 1}")
    per_seed = rollout_seeds(scenario=SCENARIOS[args.scenario_name], observation_mode="raw", action_mode="order_ship",
                             seeds=seeds, act_fn=act_fn, batch=args.batch_envs, vecnorm_path=None)
    prov = [f"# DIAGNOSTIC (ESCALATION #E33), a fitted rule, not an RL record: LB heuristic order + {form}",
            f"# heuristic y0*: {heuristic.y0:.4f}   scenario: {args.scenario_name}   seeds: {args.first_seed}..{args.first_seed + args.n_seeds - 1}"]
    write_record(outdir / f"{tag}.tsv", args.scenario_name, tag, per_seed, seeds=seeds, provenance=prov)


if __name__ == "__main__":
    main()

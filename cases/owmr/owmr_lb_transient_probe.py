"""Per-period cost profile of the LB heuristic from the mean-cover start (ESCALATION #E26).

Checks the finite-horizon bound's cycle accounting against the simulator. The
heuristic is a stationary policy, so its per-period profile shows exactly the
start transient that the finite-horizon criterion charges and the paper's
long-run average never did: in `base`, 5.29 units of surplus sit at the
warehouse for periods 0-2 (echelon stock 20 at the first allocation against
targets summing to 14.71). Its first cycles — the warehouse term at t plus the
retailers' term at t + l_rt — must match the bound's fixed cycles up to the
balance gap (`BalanceBound.finite_horizon_bound`: 8.531 each in `base`), and
its late periods must sit just above the long-run minimum per cycle.

Usage (from owmr/):
    python owmr_lb_transient_probe.py -s base --n-seeds 2048
writes results/{scenario}/benchmark/lb/transient_{scenario}.tsv (per-period
means of the cost components, the SE of the total) and prints the profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from owmr_benchmark_lb_heuristic import build
from owmr_eval_common import eval_seed_block
from owmr_gym import OwmrEnv
from owmr_scenarios import SCENARIOS

COMPONENTS = ("total", "wh_holding", "rt_holding", "rt_shortage")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-period cost profile of the LB heuristic from the start state.")
    p.add_argument("-s", "--scenario_name", type=str, default="base", choices=list(SCENARIOS))
    p.add_argument("--n-seeds", type=int, default=2048)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def trace(scenario_name: str, seeds: np.ndarray, here: Path, outdir: str) -> np.ndarray:
    """(n_seeds, horizon, 4) per-period costs of the heuristic under the shared seed protocol."""
    pol = build(scenario_name, here, outdir)
    sc = pol.scenario
    env = OwmrEnv(sc, observation_mode="raw", action_mode="order_ship")
    out = np.zeros((len(seeds), sc.horizon, len(COMPONENTS)))
    for k, s in enumerate(seeds):
        env.reset(seed=int(s))
        for t in range(sc.horizon):
            order, ship = pol.decide(env._state)
            _, _, term, trunc, info = env.step(np.concatenate([[order], ship]).astype(np.float32))
            out[k, t] = [info["cost"][c] for c in COMPONENTS]
            if term or trunc:
                assert t == sc.horizon - 1, (t, sc.horizon)
                break
    return out


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)
    tr = trace(args.scenario_name, seeds, here, args.outdir)
    n, T = tr.shape[0], tr.shape[1]
    m = tr.mean(axis=0)
    se = tr[:, :, 0].std(axis=0) / np.sqrt(n)
    sol = json.loads((here / args.outdir / args.scenario_name / "benchmark" / "lb" / f"{args.scenario_name}.txt").read_text())
    lr = SCENARIOS[args.scenario_name].l_rt
    ep = tr[:, :, 0].sum(axis=1)
    print(f"[probe] {args.scenario_name}: {n} seeds from {args.first_seed}; episode cost {ep.mean():.2f} ± {ep.std() / np.sqrt(n):.2f}")
    print(f"[probe] long-run minimum per cycle {sol['lb_per_period']:.4f}; heuristic late periods (t = 20..{T - 6}) {m[20:T - 5, 0].mean():.4f}")
    print("  t   total (±se)   wh     rt_hold  rt_short")
    for t in list(range(0, 6)) + [10, 50, T - 2, T - 1]:
        print(f"  {t:3d} {m[t, 0]:7.3f} (±{se[t]:.3f}) {m[t, 1]:6.3f} {m[t, 2]:7.3f} {m[t, 3]:7.3f}")
    cycles = [m[t, 1] + m[t + lr, 2] + m[t + lr, 3] for t in range(4)]
    print("[probe] simulated cycles 0-3 (warehouse at t + retailers at t + l_rt): "
          + " ".join(f"{c:.3f}" for c in cycles)
          + f"   vs the bound's fixed cycles {[round(v, 3) for v in sol['lb_finite_components']['fixed_cycles']]}"
          + f" and ordered {[round(v, 3) for v in sol['lb_finite_components']['ordered_cycles_exact']]}")
    outfile = here / args.outdir / args.scenario_name / "benchmark" / "lb" / f"transient_{args.scenario_name}.tsv"
    with outfile.open("w") as f:
        f.write(f"# lb transient probe: lb_heuristic per-period means, {n} seeds from {args.first_seed}, scenario {args.scenario_name}\n")
        f.write("t\t" + "\t".join(COMPONENTS) + "\tse_total\n")
        for t in range(T):
            f.write(f"{t}\t" + "\t".join(f"{m[t, j]:.6f}" for j in range(len(COMPONENTS))) + f"\t{se[t]:.6f}\n")
    print(f"[probe] -> {outfile}")


if __name__ == "__main__":
    main()

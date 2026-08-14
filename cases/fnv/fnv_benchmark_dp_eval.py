"""Protocol-tier evaluation of the DP benchmark policy on an FNV scenario or grid (spec §9).

Runs the optimal base-stock policy from fnv_benchmark_dp.py:

    additive:        S = mu + I + b_t        multiplicative: S = exp(mu + I + b_t)
    order            q = max(0, S - x)

and reports the same columns, on the same CRN seed block, as fnv_ppo_eval.py —
so the two are directly comparable and gate-referenceable (§13). In a gate this
arm is normally the `--reference` (the optimum a learned policy is measured
against) rather than a must-beat `--baseline`.

Example usage:
    python fnv_benchmark_dp.py -s FNV-aMMFE                       # solve first
    python fnv_benchmark_dp_eval.py --dp-solutions results/FNV-aMMFE/benchmark/dp/FNV-aMMFE.txt -s FNV-aMMFE
    python fnv_benchmark_dp_eval.py --dp-solutions ... -s simple --n-seeds 8192
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fnv_grids import GRIDS
from fnv_ppo_eval import (
    COLUMNS,
    DEFAULT_N_SEEDS,
    PROTOCOL_SEED_BASE,
    resolve_cells,
    rollout_seeds,
    summarize,
    tee_console,
)
from fnv_scenarios import FnvScenario, SCENARIOS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the DP benchmark on an FNV scenario or grid.")
    p.add_argument("--dp-solutions", type=str, required=True,
                   help="TSV of offsets from fnv_benchmark_dp.py (stdev, T, lambda, b1..bN)")
    p.add_argument("-s", "--scenario_name", type=str, default="FNV-aMMFE",
                   choices=list(SCENARIOS) + list(GRIDS),
                   help="Scenario (§5.4) or grid (§5.6); a grid enumerates to "
                        "one row per cell")
    p.add_argument("-o", "--observation_mode", type=str, default="vec", choices=["vec"])
    p.add_argument("-a", "--action_mode",      type=str, default="box", choices=["box"])
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS,
                   help=f"Episode seeds per cell (default {DEFAULT_N_SEEDS}, §9.7 evidence grade)")
    p.add_argument("--seed-base", type=int, default=PROTOCOL_SEED_BASE,
                   help="First seed of the CRN block — must match the arm being "
                        "compared against, or the comparison is not paired")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--method", type=str, default=None,
                   help="Method name for the output filename (§9.8); defaults "
                        "to the solutions file's parent directory, e.g. "
                        "results/<s>/benchmark/prop2/<s>.txt -> 'prop2'")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV output (defaults to results/<scenario>/benchmark/"
                        "benchmark_<method>_eval_<scenario>.tsv)")
    p.add_argument("--per-seed-out", type=str, default=None,
                   help="Optional TSV of per-seed profits, for paired Δ vs another arm")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# The DP policy
# ---------------------------------------------------------------------------

def make_dp_act_batch(offsets: np.ndarray, mu: float, additive: bool):
    """Batched base-stock policy: order up to S, given raw observations.

    Observation layout (mode 'vec'): [stdev, T, lamb, period, inventory, information].
    `offsets[t-1]` is the period-t offset b_t.
    """
    def act_batch(obs: np.ndarray) -> np.ndarray:
        period = obs[:, 3].astype(int)
        x      = obs[:, 4]
        info   = obs[:, 5]
        b      = offsets[np.clip(period, 1, len(offsets)) - 1]
        target = mu + info + b if additive else np.exp(mu + info + b)
        return np.maximum(0.0, target - x).astype(np.float32)[:, None]
    return act_batch


def lookup_offsets(dp_solutions: pd.DataFrame, scenario: FnvScenario) -> np.ndarray | None:
    """Row matching this cell's (stdev, T, lambda), as a b_1..b_N array."""
    match = dp_solutions[
        np.isclose(dp_solutions["stdev"],   scenario.stdev, atol=1e-6)
        & np.isclose(dp_solutions["T"],     scenario.T,     atol=1e-6)
        & np.isclose(dp_solutions["lambda"], scenario.lamb, atol=1e-6)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return np.array([float(row[f"b{n}"]) for n in range(1, scenario.N + 1)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    dp_path = Path(args.dp_solutions)
    # §9.8: the method name rides in the filename so each benchmark run writes a
    # distinct, gate-referenceable file; §8.4 puts them in benchmark/, one
    # level above the per-method solution dirs.
    method  = args.method or dp_path.parent.name
    bench   = (Path(__file__).resolve().parent / "results"
               / args.scenario_name / "benchmark")
    outfile = (Path(args.outfile) if args.outfile else
               bench / f"benchmark_{method}_eval_{args.scenario_name}.tsv")
    tee_console(outfile.parent, f"benchmark_{method}.log")

    cells        = resolve_cells(args.scenario_name)
    dp_solutions = pd.read_csv(str(dp_path), delimiter="\t")
    seeds        = range(args.seed_base, args.seed_base + args.n_seeds)

    print(f"dp_solutions  {dp_path}  ({len(dp_solutions)} rows)")
    print(f"scenario      {args.scenario_name}  ({len(cells)} cell(s))")
    print(f"seeds         {args.seed_base}..{args.seed_base + args.n_seeds - 1}")
    print(f"output        {outfile}")

    header = "\t".join(COLUMNS)
    print(header)

    per_seed_fh = None
    if args.per_seed_out:
        Path(args.per_seed_out).parent.mkdir(parents=True, exist_ok=True)
        per_seed_fh = open(args.per_seed_out, "w")
        per_seed_fh.write("cell_id\tepisode_seed\tprofit\tregret\n")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()

        for cell_id, scenario in cells:
            offsets = lookup_offsets(dp_solutions, scenario)
            if offsets is None:
                print(f"WARNING: no DP solution for stdev={scenario.stdev} "
                      f"T={scenario.T} lamb={scenario.lamb}, skipping", flush=True)
                continue

            profits, regrets = rollout_seeds(
                make_dp_act_batch(offsets, scenario.mu,
                                  scenario.mmfe_mode == "additive"),
                scenario, seeds,
                observation_mode=args.observation_mode,
                action_mode=args.action_mode,
                reward_mode="profit",
                batch_size=args.batch_size,
            )
            stats = summarize(profits, regrets)
            row = (
                f"{scenario.stdev}\t{scenario.T}\t{scenario.lamb}\t{args.n_seeds}\t"
                f"{stats['profit_mean']:.6f}\t{stats['profit_se']:.6f}\t"
                f"{stats['regret_mean']:.6f}\t{stats['regret_se']:.6f}\t"
                f"{stats['profit_var']:.6f}\t{stats['semivar_d']:.6f}\t{stats['semivar_u']:.6f}"
            )
            print(row, flush=True)
            # incremental (§9.4): a killed run keeps every finished cell
            f.write(row + "\n")
            f.flush()

            if per_seed_fh is not None:
                for s, p, rg in zip(seeds, profits, regrets):
                    per_seed_fh.write(f"{cell_id}\t{s}\t{p:.6f}\t{rg:.6f}\n")
                per_seed_fh.flush()

    if per_seed_fh is not None:
        per_seed_fh.close()
        print(f"per-seed profits → {args.per_seed_out}")
    print(f"\nresults saved → {outfile}")


if __name__ == "__main__":
    main()

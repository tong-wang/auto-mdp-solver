"""Heuristic benchmark policies for inv_single (spec §9, §9.8).

One script, several closely-related policies selected by `--policy`; each run
writes its own §9.3 record on the campaign's CRN seed block, so any of them can
serve as a `--baseline` or `--reference` in a gate.

| `--policy`  | rule |
|---|---|
| `zero`      | never order — the do-nothing floor |
| `random`    | uniform on the gym's action space — the "did it learn anything" floor |
| `constant`  | order the demand mean every period (a flow-matching rule) |
| `myopic`    | order up to the newsvendor fractile of demand over the protection interval L+1, measured on inventory position |
| `basestock` | order up to a fixed S on inventory position; `--optimize` grid-searches S by simulation |

The two bars that matter for the unsolved rungs are `myopic` and an optimized
`basestock`; `zero` and `random` exist because spec §8.6 makes "L1 ≤ random"
a *build* diagnosis rather than an escalation trigger.

`--optimize` searches S on the **selection** seed block (`SELECT_SEED_OFFSET`),
never on the reporting block — a bar tuned on the seeds that report it is as
selection-biased as an RL arm tuned that way.

Usage:
    python inv_single_benchmark_heuristic.py -s simple --policy random --n-seeds 8192
    python inv_single_benchmark_heuristic.py -s slt --policy basestock --optimize
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inv_single_eval_common import (
    SELECT_SEED_OFFSET,
    eval_seed_block,
    paired_report,
    rollout_seeds,
    write_record,
)
from inv_single_scenarios import SCENARIOS

POLICIES = ["zero", "random", "constant", "myopic", "basestock",
            "capped_basestock"]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Heuristic benchmarks for inv_single.")
    p.add_argument("-g", "--grid_name", type=str, default=None,
                   help="run the policy over every cell of this GRIDS entry "
                        "(de-padded twins; one record per cell, spec 9.6). "
                        "Overrides -s. --optimize re-searches S per cell.")
    p.add_argument("--cell", type=str, default=None,
                   help="with -g: restrict to one cell id (as the grid "
                        "enumerates it, e.g. 'b=9,K=0,leadtime_value=2')")
    p.add_argument("-s", "--scenario_name", type=str, default="simple",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--policy", type=str, default="myopic", choices=POLICIES)
    p.add_argument("-o", "--observation_mode", type=str, default="vec",
                   choices=["vec", "vec_ip"])
    p.add_argument("-a", "--action_mode", type=str, default="discrete",
                   choices=["continuous", "discrete", "hurdle"])
    p.add_argument("--n-seeds", type=int, default=65536)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("-S", "--base-stock", type=float, default=None,
                   help="S for --policy basestock (ignored with --optimize).")
    p.add_argument("--const-order", type=float, default=None,
                   help="Order quantity for --policy constant (default: demand mean).")
    p.add_argument("--optimize", action="store_true",
                   help="Grid-search S on the selection seed block before reporting.")
    p.add_argument("--cap", type=float, default=None,
                   help="order cap for --policy capped_basestock (Xin 2021); "
                        "with --optimize the (S, cap) pair is grid-searched")
    p.add_argument("--opt-cap-lo", type=int, default=1)
    p.add_argument("--opt-cap-hi", type=int, default=None)
    p.add_argument("--opt-lo", type=int, default=0)
    p.add_argument("--opt-hi", type=int, default=None)
    p.add_argument("--opt-n-seeds", type=int, default=512)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--outfile", type=str, default=None)
    p.add_argument("--paired-with", type=str, default=None)
    return p


def _ip(env, lt_max: int) -> float:
    st = env._state
    return float(st.inventory if lt_max == 0 else st.inventory + sum(st.pipeline))


def make_act_fn(policy: str, scenario, *, S: float | None, const: float | None,
                cap: float | None = None):
    """The chosen rule as a batched action callback."""
    lt_max = scenario.leadtime.max()

    if policy == "zero":
        def act(obs, envs):
            return np.zeros((len(envs), 1), dtype=np.float32)
        return act

    if policy == "random":
        def act(obs, envs):
            return np.stack([e.action_space.sample() for e in envs]).astype(np.float32)
        return act

    if policy == "constant":
        q = float(const if const is not None else scenario.demand.mean())
        def act(obs, envs):
            return np.full((len(envs), 1), q, dtype=np.float32)
        return act

    if policy == "myopic":
        # newsvendor fractile of demand over the protection interval L+1: the
        # critical ratio b/(b+h) applied to the (L+1)-fold demand convolution.
        # Uses the scenario's declared costs and demand read-API only.
        h, b = scenario.holding_cost, scenario.shortage_cost
        crit = b / (b + h) if (b + h) > 0 else 0.5
        d_max = int(np.ceil(scenario.demand.mean() + 8 * np.sqrt(max(1.0, scenario.demand.mean()))))
        pmf = np.array([scenario.demand.phi(d) for d in range(d_max + 1)], dtype=float)
        pmf /= pmf.sum()
        conv = np.array([1.0])
        for _ in range(lt_max + 1):
            conv = np.convolve(conv, pmf)
        S_star = float(np.searchsorted(np.cumsum(conv), crit))
        print(f"[heuristic] myopic: critical ratio {crit:.3f}, protection interval "
              f"{lt_max + 1} periods ⇒ order up to {S_star:.0f} on IP")

        def act(obs, envs):
            return np.array(
                [[max(0.0, S_star - _ip(e, lt_max))] for e in envs], dtype=np.float32
            )
        return act

    if policy == "capped_basestock":
        # Xin (2021), "Understanding the Performance of Capped Base-Stock
        # Policies in Lost-Sales Inventory Models", Operations Research: a
        # two-parameter family q = min(cap, (S - IP)+) whose limits are pure
        # base-stock (cap -> inf) and pure constant-order (S -> inf). It is the
        # shape the exact lost-sales DP takes at `lt_lost_sales` (#E17): the
        # constant-order limb below IP~14 and the base-stock limb above IP~26,
        # both active inside one policy. Worth 33 cost units over uncapped
        # base-stock there, which is 91.7% of the gap base-stock leaves.
        assert S is not None and cap is not None, "capped_basestock needs S and cap"
        S_f, cap_f = float(S), float(cap)

        def act(obs, envs):
            return np.array(
                [[min(cap_f, max(0.0, S_f - _ip(e, lt_max)))] for e in envs],
                dtype=np.float32,
            )
        return act

    assert policy == "basestock" and S is not None, "basestock needs S"
    S_f = float(S)

    def act(obs, envs):
        return np.array(
            [[max(0.0, S_f - _ip(e, lt_max))] for e in envs], dtype=np.float32
        )
    return act


def optimize_S(args, scenario, *, with_cap: bool = False):
    """Grid-search on the SELECTION seed block (never the reporting block).

    Returns `S` for `basestock`, or `(S, cap)` for `capped_basestock`. The cap
    is the second parameter of Xin (2021)'s family; searching it jointly with S
    matters because the two levers substitute — at `lt_lost_sales` each is worth
    ~180 alone and only 20-33 once the other is present (#E17)."""
    hi = args.opt_hi
    if hi is None:
        hi = int(np.ceil(scenario.demand.mean() * (scenario.leadtime.max() + 1) * 2.5))
    seeds = eval_seed_block(args.opt_n_seeds, offset=SELECT_SEED_OFFSET)
    print(f"[heuristic] optimizing S over {args.opt_lo}…{hi} on "
          f"{args.opt_n_seeds} selection seeds")
    cap_hi = args.opt_cap_hi
    if cap_hi is None:
        cap_hi = int(np.ceil(scenario.demand.mean() * 2))
    caps = range(args.opt_cap_lo, cap_hi + 1) if with_cap else [None]
    pol = "capped_basestock" if with_cap else "basestock"
    best_S, best_cap, best_cost = None, None, float("inf")
    for S in range(args.opt_lo, hi + 1):
      for cap in caps:
        per_seed = rollout_seeds(
            scenario=scenario,
            observation_mode=args.observation_mode,
            action_mode=args.action_mode,
            seeds=seeds,
            act_fn=make_act_fn(pol, scenario, S=S, const=None, cap=cap),
            batch=min(args.batch_envs, args.opt_n_seeds),
            vecnorm_path=None,
            progress_every=0,
        )
        cost = float(per_seed["cost_total"].mean())
        flag = ""
        if cost < best_cost:
            best_S, best_cap, best_cost, flag = float(S), cap, cost, "  *"
        capstr = "" if cap is None else f" cap={cap:3d}"
        print(f"  S={S:4d}{capstr}  cost={cost:10.4f}{flag}", flush=True)
    if with_cap:
        print(f"[heuristic] optimized (S*, cap*) = ({best_S:.0f}, {best_cap:.0f}) "
              f"(selection cost {best_cost:.4f})")
        return best_S, best_cap
    print(f"[heuristic] optimized S* = {best_S:.0f} (selection cost {best_cost:.4f})")
    return best_S


def _run_one(args, source, label: str, outfile: Path) -> None:
    """One (scenario, policy) record on the reporting block."""
    S, cap = args.base_stock, args.cap
    arm = args.policy
    if args.policy == "capped_basestock":
        if args.optimize:
            S, cap = optimize_S(args, source, with_cap=True)
            arm = f"capped_basestock_opt(S={S:.0f},cap={cap:.0f})"
        else:
            assert S is not None and cap is not None, \
                "--policy capped_basestock needs -S and --cap, or --optimize"
            arm = f"capped_basestock(S={S:.0f},cap={cap:.0f})"
    if args.policy == "basestock":
        if args.optimize:
            S = optimize_S(args, source)
            arm = f"basestock_opt(S={S:.0f})"
        else:
            assert S is not None, "--policy basestock needs -S or --optimize"
            arm = f"basestock(S={S:.0f})"

    print(f"[eval] scenario : {label}   policy={arm}")
    print(f"[eval] seeds    : 0…{args.n_seeds - 1}")

    per_seed = rollout_seeds(
        scenario=source,
        observation_mode=args.observation_mode,
        action_mode=args.action_mode,
        seeds=eval_seed_block(args.n_seeds),
        act_fn=make_act_fn(args.policy, source, S=S, const=args.const_order, cap=cap),
        batch=args.batch_envs,
        vecnorm_path=None,
    )

    write_record(outfile, label, arm, per_seed)

    if args.paired_with:
        paired_report(
            per_seed["cost_total"], Path(args.paired_with),
            bar_name=Path(args.paired_with).stem,
        )


def main() -> None:
    args = _build_arg_parser().parse_args()
    here = Path(__file__).resolve().parent

    if args.grid_name:
        # per-cell records over the grid's DE-PADDED twins (see
        # inv_single_grids.cells_depadded for why mean(), not max())
        from inv_single_grids import cells_depadded
        assert args.outfile is None, "-g writes one file per cell; drop --outfile"
        cells = cells_depadded(args.grid_name)
        if args.cell is not None:
            cells = [c for c in cells if c[0] == args.cell]
            assert cells, f"unknown cell {args.cell!r}"
        for cid, slug, sc in cells:
            outfile = (here / args.outdir / f"{args.grid_name}__{slug}" / "benchmark"
                       / f"benchmark_{args.policy}_eval_{args.grid_name}__{slug}.tsv")
            _run_one(args, sc, f"{args.grid_name}[{cid}]", outfile)
        return

    source = SCENARIOS[args.scenario_name]
    assert not callable(source), (
        f"scenario {args.scenario_name!r} is a world sampler; these heuristics "
        "need a fixed scenario."
    )
    outfile = Path(
        args.outfile if args.outfile
        else here / args.outdir / args.scenario_name / "benchmark"
        / f"benchmark_{args.policy}_eval_{args.scenario_name}.tsv"
    )
    _run_one(args, source, args.scenario_name, outfile)


if __name__ == "__main__":
    main()

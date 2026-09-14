"""Order-up-to on a WEIGHTED inventory position -- the rule the bottleneck found.

    wp = inv + sum_k w_k * pipe_k        order q = max(0, S - wp)

w = (1,1,...,1) is textbook inventory position, so `basestock_opt` is the
w-frozen member of this family and the comparison is like-for-like. Search
protocol is `optimize_S`'s: grid on the SELECTION block, report on 0..n-1.

w_0 is FIXED at 1.0 on theory, not fitted: under the O->R->D sequence pipe_0 is
received before this period's demand, so it is on-hand stock. The distillation
agreed without being told (1.018 / 1.045).
"""
import argparse, itertools
from pathlib import Path

import numpy as np
from inv_single_scenarios import SCENARIOS
from inv_single_eval_common import SELECT_SEED_OFFSET, eval_seed_block, rollout_seeds

def act_fn_for(w, S, lt_max):
    wv = np.asarray(w, dtype=float)
    def act(obs, envs):
        out = np.empty((len(envs), 1), dtype=np.float32)
        for i, e in enumerate(envs):
            st = e._state
            wp = st.inventory + float(np.dot(wv, st.pipeline[:len(wv)]))
            out[i, 0] = max(0.0, S - wp)
        return out
    return act

def score(sc, w, S, seeds, batch):
    r = rollout_seeds(scenario=sc, observation_mode="vec", action_mode="continuous",
                      seeds=seeds, act_fn=act_fn_for(w, S, sc.leadtime.max()),
                      batch=batch, vecnorm_path=None, progress_every=0)
    return float(r["cost_total"].mean()), r

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--scenario_name", default="slt")
    p.add_argument("--mode", choices=["net", "search", "ones"], required=True,
                   help="net = weights recovered from the distilled policy; "
                        "search = grid-search the weights by COST; ones = plain IP")
    p.add_argument("--w", type=float, nargs="+", default=None)
    p.add_argument("--opt-n-seeds", type=int, default=512)
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--batch", type=int, default=256)
    a = p.parse_args()
    sc = SCENARIOS[a.scenario_name]
    n_live = sc.leadtime.max()          # last slot is structurally zero
    sel = eval_seed_block(a.opt_n_seeds, offset=SELECT_SEED_OFFSET)

    def best_S(w, seeds, batch):
        """Coarse-to-fine scan; cost is unimodal in S so this finds the same
        optimum as the exhaustive scan optimize_S runs, at a fifth the cost."""
        lo, hi = 20, int(np.ceil(sc.demand.mean() * (sc.leadtime.max() + 1) * 2.5))
        cand = list(range(lo, hi + 1, 3))
        vals = {S: score(sc, w, S, seeds, batch)[0] for S in cand}
        c0 = min(vals, key=vals.get)
        for S in range(max(lo, c0 - 3), min(hi, c0 + 3) + 1):
            if S not in vals:
                vals[S] = score(sc, w, S, seeds, batch)[0]
        bS = min(vals, key=vals.get)
        return bS, vals[bS]

    if a.mode == "ones":
        cands = [tuple([1.0] * n_live)]
    elif a.mode == "net":
        assert a.w is not None, "--mode net needs --w"
        cands = [tuple(a.w)]
    else:
        # the one-parameter deviation from inventory position: w = (1, 1, alpha).
        # w_0 = 1 is theory; w_1 came back at 0.996 +/- 0.001 from distillation,
        # so alpha on the FAR slot is the only coefficient in question.
        cands = [tuple([1.0] * (n_live - 1) + [float(x)])
                 for x in np.round(np.arange(0.60, 1.101, 0.05), 3)]
    print(f"[wbs] {a.scenario_name}  mode={a.mode}  {len(cands)} weight vector(s)")

    best = (None, None, float("inf"))
    for w in cands:
        bS, bc = best_S(w, sel, min(a.batch, a.opt_n_seeds))
        print(f"  w={np.array2string(np.asarray(w), precision=3)} S*={bS} "
              f"selection={bc:.3f}", flush=True)
        if bc < best[2]:
            best = (w, bS, bc)
    w, S, sel_cost = best
    print(f"\n[wbs] chosen w={np.array2string(np.asarray(w), precision=3)} S*={S} "
          f"(selection {sel_cost:.3f})")
    outdir = Path("results") / a.scenario_name / "benchmark"
    outdir.mkdir(parents=True, exist_ok=True)
    cost, r = score(sc, w, S, eval_seed_block(a.n_seeds), a.batch)
    np.save(str(outdir / f"weighted_basestock_{a.mode}_perseed.npy"), r["cost_total"])
    print(f"REPORT {a.scenario_name} mode={a.mode} "
          f"w={np.array2string(np.asarray(w), precision=3)} S={S} cost={cost:.2f}")

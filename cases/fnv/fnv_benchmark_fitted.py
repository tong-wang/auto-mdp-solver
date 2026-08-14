"""Fitted-rule benchmark: read the net back into offsets, then SCORE them (spec §14.2).

The readback (`INTERPRET.md`) finds that the trained policy implements the
paper's structure — `S_n(I) = mu + I + b_n`, log-linear on the multiplicative
branch. §14.2 says the fitted rule is then **itself a policy** and must be
evaluated under the same §9 protocol as every other arm, because it can *beat*
the net it came from: fitting a constant offset deletes the raggedness the
network carries.

So this file distils a trained model into `b̂_1..b̂_N` per scenario cell and
emits the same offsets TSV the other benchmark solvers write. It is then scored
by `fnv_benchmark_dp_eval.py` on the identical CRN block, which puts the trio
reference / fitted rule / raw net on one paired footing.

How each offset is recovered, from the policy's own trajectories:

    period 1   I_1 = 0 and x = 0 identically, so the order IS the level:
               b̂_1 = structural(q_1) - mu          (no regression possible)
    period n>1 regress structural(x+q) on I over ACTING points; the intercept
               is mu + b_n, so b̂_n = intercept - mu

where `structural` is the identity on a-MMFE and `log` on m-MMFE.

**Full-horizon assertion (§14.2).** A rule fitted on only some periods acts
arbitrarily in the rest and returns a plausible-looking garbage score. A cell is
emitted only if every period yielded an offset; cells that fail are dropped and
counted, never silently defaulted.

Example usage:
    python fnv_benchmark_fitted.py --model-path <ckpt> -s FNV-aMMFE
    python fnv_benchmark_dp_eval.py --dp-solutions results/FNV-aMMFE/benchmark/fitted/FNV-aMMFE.txt \\
        -s FNV-aMMFE --n-seeds 2048
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from fnv_gym import FnvEnv
from fnv_plot_policy import structural_y
from fnv_ppo_eval import (load_obs_rms, normalize_obs, resolve_cells,
                          resolve_vecnorm_path)

ACT_TOL = 1e-4
MIN_ACTING = 25          # per period, below this the intercept is not trustworthy


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distil a trained FNV policy into offsets (§14.2).")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None)
    p.add_argument("-s", "--scenario_name", type=str, default="FNV-aMMFE")
    p.add_argument("--n-episodes", type=int, default=400,
                   help="episodes per cell used to fit the offsets")
    p.add_argument("--outfile", type=str, default=None)
    return p


def rollout_cell(act_batch, sc, n_ep: int) -> list[dict]:
    """Batched lockstep rollout: every episode of a cell advances together.

    FNV episodes are exactly N steps and share no cross-episode state, so the
    policy sees one (n_ep, obs_dim) array per period instead of n_ep scalar
    calls — fitting all 540 cells stays minutes, not hours.
    """
    envs = [FnvEnv(scenario=sc) for _ in range(n_ep)]
    obs = np.stack([e.reset(seed=s)[0] for e, s in zip(envs, range(n_ep))])
    rows = []
    for _ in range(sc.N):
        acts = np.asarray(act_batch(obs)).reshape(-1)
        rows.append({
            "period": int(obs[0, 3]),
            "I": obs[:, 5].copy(),
            "post": obs[:, 4] + np.maximum(0.0, acts),
            "acted": np.maximum(0.0, acts) > ACT_TOL,
        })
        for i, e in enumerate(envs):
            o, _, _, _, _ = e.step(np.array([acts[i]], dtype=np.float32))
            obs[i] = o
    for e in envs:
        e.close()
    return rows


def fit_offsets(rows: list[dict], sc) -> tuple[np.ndarray | None, str]:
    """(b̂_1..b̂_N, "") or (None, reason) — the full-horizon assertion."""
    add = sc.mmfe_mode == "additive"
    b = np.full(sc.N, np.nan)
    for r in rows:
        n = r["period"]
        act = r["acted"]
        if act.sum() < MIN_ACTING:
            return None, f"period {n}: only {int(act.sum())} acting episodes"
        y = structural_y(r["post"][act], add)
        I = r["I"][act]
        if I.std(ddof=1) < 1e-9:
            # period 1: I is degenerate, so the order IS the level
            b[n - 1] = float(np.median(y)) - sc.mu
        else:
            _, intercept = np.polyfit(I, y, 1)
            b[n - 1] = float(intercept) - sc.mu
    if np.isnan(b).any():
        return None, f"periods {list(np.where(np.isnan(b))[0] + 1)} unfitted"
    return b, ""


def main() -> None:
    args = _build_arg_parser().parse_args()
    mp = Path(args.model_path)
    model = PPO.load(str(mp), device="cpu")
    rms, clip = load_obs_rms(resolve_vecnorm_path(mp, args.vecnorm_path))

    def act_batch(o):
        a, _ = model.predict(normalize_obs(o, rms, clip), deterministic=True)
        return a

    cells = resolve_cells(args.scenario_name)
    outfile = (Path(args.outfile) if args.outfile else
               Path(__file__).resolve().parent / "results" / args.scenario_name
               / "benchmark" / "fitted" / f"{args.scenario_name}.txt")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    N = cells[0][1].N
    header = "stdev\tT\tlambda\t" + "\t".join(f"b{n}" for n in range(1, N + 1))
    kept, dropped = 0, []
    with open(outfile, "w") as f:
        f.write(header + "\n")
        for cid, sc in cells:
            b, why = fit_offsets(rollout_cell(act_batch, sc, args.n_episodes), sc)
            if b is None:
                dropped.append((cid, why))
                continue
            f.write(f"{sc.stdev}\t{sc.T}\t{sc.lamb}\t"
                    + "\t".join(f"{x:.6f}" for x in b) + "\n")
            kept += 1

    print(f"model    {mp.parent.parent.name}/{mp.name}")
    print(f"scenario {args.scenario_name}")
    print(f"fitted   {kept}/{len(cells)} cells")
    if dropped:
        # never silently defaulted: an unfitted cell is reported, not guessed
        print(f"DROPPED  {len(dropped)} cells failing the full-horizon assertion:")
        for cid, why in dropped[:8]:
            print(f"           {cid}: {why}")
        if len(dropped) > 8:
            print(f"           … and {len(dropped) - 8} more")
    print(f"\noffsets → {outfile}")
    print(f"next: python fnv_benchmark_dp_eval.py --dp-solutions {outfile} "
          f"-s {args.scenario_name} --n-seeds 2048")


if __name__ == "__main__":
    main()

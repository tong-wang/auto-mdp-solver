"""Shared evaluation core for the cnv campaign (spec §9).

Driver-level module: sits above the gym, imported by every
``cnv_benchmark_{method}_eval.py`` and by ``cnv_ppo_eval.py``, so the RL arm
and every benchmark are scored by *identical* code on an identical seed
block. The campaign's verdicts are **paired** differences under common random
numbers, which is only valid if every arm sees the same episode seeds in the
same objective — so the seed block, the accumulators and the TSV record are
defined once here.

Key components:
- ``build_arg_parser`` — the spec-§9.1 CLI surface (tier-1 names); a
  benchmark eval adds ``--policy`` where its solver exposes several.
- ``eval_targets`` — §9.6 dispatch: a ``SCENARIOS`` entry is one row, a
  ``GRIDS`` entry one row per cell, all on ONE seed block.
- ``rollout`` — the seed loop through ``CnvEnv``: per-seed profit (the
  objective) and the IR's two bystander metrics, ``hindsight_profit`` and
  ``perfect_profit`` (the latter reads the realized hidden rate, which a
  metric may — spec §9.3).
- ``write_record`` — the §9.3 TSV row(s), objective column first, plus the
  per-seed ``.seeds.tsv`` sidecar that makes pairing possible later.

A policy is a callable ``policy(env, obs) -> order``: benchmark policies read
the base env's public state (``env._state``, ``env._scenario_ep``'s prior and
cost constants, ``env._info`` for the last observation) and ignore ``obs``; a
trained policy reads ``obs``, the vector the (possibly frame-stacked) wrapper
rendered. Neither sees the hidden rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from cnv_bayes import poisson_quantile
from cnv_grids import GRIDS
from cnv_gym import CnvEnv
from cnv_scenarios import SCENARIOS

METRICS = ("profit", "hindsight_profit", "perfect_profit")

Policy = Callable[[CnvEnv, np.ndarray], int]


def build_arg_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("-s", "--scenario_name", type=str, default="base",
                   choices=list(SCENARIOS.keys()),
                   help="a SCENARIOS entry: one row over the seed block")
    p.add_argument("-g", "--grid_name", type=str, default=None,
                   choices=list(GRIDS.keys()),
                   help="a GRIDS entry: one row per cell, all cells on the same seed block")
    p.add_argument("-o", "--observation_mode", type=str, default="timing_belief",
                   help="gym mode the episodes are rendered in (benchmarks read state, not obs)")
    p.add_argument("-a", "--action_mode", type=str, default="order_discrete")
    p.add_argument("-r", "--reward_mode", type=str, default="profit")
    p.add_argument("--n-seeds", type=int, default=8192,
                   help="episodes per target: seeds first_seed .. first_seed + n_seeds - 1")
    p.add_argument("--first-seed", type=int, default=0,
                   help="protocol block starts at 0; the screen block at 1_000_000 (spec §9.7)")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV path (default: results/{target}/benchmark/benchmark_{name}_eval_{target}.tsv)")
    p.add_argument("--shard", type=str, default=None, metavar="I/N",
                   help="grid targets only: evaluate every N-th cell starting at I (1-based) into a per-shard "
                        "TSV; `cnv_benchmark_common.py --merge` joins the shards in grid order")
    return p


def eval_seed_block(first_seed: int, n_seeds: int) -> np.ndarray:
    """Episode seeds ``first_seed .. first_seed + n_seeds - 1`` (spec §9.2)."""
    return np.arange(first_seed, first_seed + n_seeds, dtype=np.int64)


def eval_targets(args: argparse.Namespace) -> tuple[str, list[tuple[str, object]]]:
    """§9.6 dispatch: ``(target_name, [(row_id, source), ...])``; a shard I/N
    keeps every N-th cell from I-1 (grid order), the join is by cell id."""
    if args.grid_name:
        grid = GRIDS[args.grid_name]
        cells = [(cid, src) for cid, src in grid]
        shard = getattr(args, "shard", None)
        if shard:
            i, n = (int(x) for x in shard.split("/"))
            cells = cells[i - 1::n]
        return args.grid_name, cells
    return args.scenario_name, [(args.scenario_name, SCENARIOS[args.scenario_name])]


def rollout(source, seeds: np.ndarray, policy: Policy, *,
            observation_mode: str = "timing_belief",
            action_mode: str = "order_discrete",
            frame_stack: int = 1,
            progress_every: int = 2000) -> dict[str, np.ndarray]:
    """One episode per seed through ``CnvEnv``; per-seed metric arrays.

    ``frame_stack > 1`` wraps the env in gymnasium's ``FrameStackObservation``
    (zero padding) exactly as the train script does, so a raw-mode policy sees
    the vector it was trained on; ``policy`` still receives the base env.
    """
    env = CnvEnv(scenario=source, observation_mode=observation_mode, action_mode=action_mode)
    stepper = env
    if frame_stack > 1:
        from gymnasium.wrappers import FrameStackObservation
        stepper = FrameStackObservation(env, stack_size=frame_stack, padding_type="zero")
    out = {m: np.zeros(len(seeds), dtype=np.float64) for m in METRICS}
    for i, seed in enumerate(seeds):
        if progress_every and i % progress_every == 0:
            print(f"  seed {i}/{len(seeds)}", flush=True)
        obs, _ = stepper.reset(seed=int(seed))
        sc = env._scenario_ep
        ratio = sc.critical_ratio
        y_perfect = poisson_quantile(sc.demand.rate, ratio)   # the paper's model P
        done = False
        while not done:
            order = int(policy(env, obs))
            obs, reward, done, _, info = stepper.step(order)
            d = info["demand"]
            out["profit"][i] += reward
            out["hindsight_profit"][i] += (sc.price - sc.unit_cost) * d
            out["perfect_profit"][i] += sc.price * min(d, y_perfect) - sc.unit_cost * y_perfect
    env.close()
    return out


def rollout_batched(source, seeds: np.ndarray, batch_act, *,
                    observation_mode: str = "timing_belief",
                    action_mode: str = "order_discrete",
                    frame_stack: int = 1,
                    batch: int = 64,
                    progress_every: int = 0) -> dict[str, np.ndarray]:
    """``rollout`` for a trained policy: ``batch`` episodes in lockstep.

    Every episode runs exactly N periods (``horizon_end: terminated``, no
    early exit), so a block of seeds starts and finishes together and the
    policy's forward pass is amortised over the block — ``batch_act(obs)``
    maps a (K, ...) observation batch to K actions. Each slot is seeded
    explicitly, the dynamics are seed-keyed, and the per-episode metrics are
    accumulated exactly as ``rollout`` does, so the per-seed outputs are the
    sequential loop's to the bit (gated in ``cnv_test.py``: the only
    possible divergence is an argmax flip on a numerically tied logit pair).
    """
    from gymnasium.wrappers import FrameStackObservation

    n = len(seeds)
    batch = max(1, min(batch, n))
    envs, bases = [], []
    for _ in range(batch):
        e = CnvEnv(scenario=source, observation_mode=observation_mode, action_mode=action_mode)
        bases.append(e)
        envs.append(FrameStackObservation(e, stack_size=frame_stack, padding_type="zero") if frame_stack > 1 else e)
    out = {m: np.zeros(n, dtype=np.float64) for m in METRICS}
    for start in range(0, n, batch):
        block = seeds[start:start + batch]
        k = len(block)
        if progress_every and start % progress_every == 0:
            print(f"  seed {start}/{n}", flush=True)
        obs = np.stack([envs[i].reset(seed=int(block[i]))[0] for i in range(k)])
        y_perfect = [poisson_quantile(bases[i]._scenario_ep.demand.rate, bases[i]._scenario_ep.critical_ratio)
                     for i in range(k)]
        acc = {m: np.zeros(k) for m in METRICS}
        done = np.zeros(k, dtype=bool)
        while not done.all():
            actions = np.asarray(batch_act(obs)).reshape(-1)
            nxt = []
            for i in range(k):
                if done[i]:
                    nxt.append(obs[i]); continue
                o, r, term, _, info = envs[i].step(int(actions[i]))
                sc = bases[i]._scenario_ep
                d = info["demand"]
                acc["profit"][i] += r
                acc["hindsight_profit"][i] += (sc.price - sc.unit_cost) * d
                acc["perfect_profit"][i] += sc.price * min(d, y_perfect[i]) - sc.unit_cost * y_perfect[i]
                done[i] = term
                nxt.append(o)
            obs = np.stack(nxt)
        for m in METRICS:
            out[m][start:start + k] = acc[m]
    for e in bases:
        e.close()
    return out


def summarize(per_seed: dict[str, np.ndarray]) -> dict[str, float]:
    x = per_seed["profit"]
    n = len(x)
    mu = float(x.mean())
    var = float(x.var(ddof=1)) if n > 1 else 0.0
    denom = max(1, n - 1)
    stats = {
        "profit_mean": mu,
        "profit_var": var,
        "semivar_d": float((np.maximum(mu - x, 0.0) ** 2).sum() / denom),
        "semivar_u": float((np.maximum(x - mu, 0.0) ** 2).sum() / denom),
    }
    for m in METRICS[1:]:
        stats[f"{m}_mean"] = float(per_seed[m].mean())
    return stats


# varied scenario parameters first (E_lambda, not prior_mean: a column ending in _mean
# would be mistaken for the objective by the first-*_mean contract) (spec §9.3), then the objective's columns,
# then the bystander metrics — profit_mean is the first *_mean column, which is
# the contract mdp_gates / mdp_tuning auto-pick the metric by
HEADER = ["target", "horizon_N", "unit_cost", "prior_rate", "E_lambda", "n_seeds",
          "profit_mean", "profit_var", "semivar_d", "semivar_u",
          "hindsight_profit_mean", "perfect_profit_mean"]


def format_row(row_id: str, source, n_seeds: int, stats: dict[str, float]) -> str:
    cells = [row_id, str(source.horizon), f"{source.unit_cost:g}",
             f"{source.prior_rate:g}", f"{source.prior_mean:g}", str(n_seeds)]
    cells += [f"{stats[c]:.6f}" for c in HEADER[6:]]
    return "\t".join(cells)


def default_outfile(domain_dir: Path, args: argparse.Namespace, name: str, target: str) -> Path:
    if args.outfile:
        return Path(args.outfile)
    # anchored to the domain folder, never the CWD (spec §8.4)
    return domain_dir / args.outdir / target / "benchmark" / f"benchmark_{name}_eval_{target}.tsv"


def evaluate(name: str, make_policy: Callable[[CnvEnv], Policy], args: argparse.Namespace,
             domain_dir: Path, provenance: list[str] | None = None) -> list[dict[str, float]]:
    """Run the seed loop for one policy factory over the target(s) and write
    the spec-§9 TSV incrementally (§9.4), plus the per-seed sidecar."""
    target, rows = eval_targets(args)
    seeds = eval_seed_block(args.first_seed, args.n_seeds)
    outfile = default_outfile(domain_dir, args, name, target)
    if getattr(args, "shard", None) and not args.outfile:
        outfile = outfile.with_name(outfile.stem + f".shard{args.shard.replace('/', 'of')}.tsv")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    seeds_file = outfile.with_name(outfile.stem + ".seeds.tsv")

    header = "\t".join(HEADER)
    stamps = [f"# benchmark: {name}", f"# target: {target}",
              f"# seed block: first_seed={args.first_seed} n_seeds={args.n_seeds}",
              f"# observation_mode={args.observation_mode} action_mode={args.action_mode}"]
    stamps += provenance or []
    print("\n".join(stamps)); print(header)
    results = []
    with open(outfile, "w") as f, open(seeds_file, "w") as fs:
        f.write("\n".join(stamps) + "\n" + header + "\n"); f.flush()
        fs.write("target\tseed\t" + "\t".join(METRICS) + "\n")
        for row_id, source in rows:
            env_probe = CnvEnv(scenario=source, observation_mode=args.observation_mode,
                               action_mode=args.action_mode)
            policy = make_policy(env_probe)
            per_seed = rollout(source, seeds, policy,
                               observation_mode=args.observation_mode,
                               action_mode=args.action_mode,
                               frame_stack=getattr(args, "frame_stack", 1) or 1)
            stats = summarize(per_seed)
            row = format_row(row_id, source, len(seeds), stats)
            print(row, flush=True)
            f.write(row + "\n"); f.flush()
            for k, seed in enumerate(seeds):
                fs.write(f"{row_id}\t{seed}\t" + "\t".join(f"{per_seed[m][k]:.6f}" for m in METRICS) + "\n")
            fs.flush()
            results.append({"target": row_id, **stats})
    print(f"\n[eval] record   -> {outfile}")
    print(f"[eval] per-seed -> {seeds_file}")
    return results


def merge_shards(domain_dir: Path, outdir: str, name: str, target: str) -> Path:
    """Join per-shard TSVs (and sidecars) into the canonical files, in grid order."""
    from cnv_grids import GRIDS as _G
    base = default_outfile(domain_dir, argparse.Namespace(outfile=None, outdir=outdir), name, target)
    shards = sorted(base.parent.glob(base.stem + ".shard*.tsv"))
    shards = [p for p in shards if not p.name.endswith(".seeds.tsv")]
    order = [cid for cid, _ in _G[target]]
    stamps, header, rows, srows = [], None, {}, {}
    for sh in shards:
        for line in sh.read_text().splitlines():
            if line.startswith("#"):
                if line not in stamps: stamps.append(line)
            elif header is None or line.startswith("target\t"):
                header = line
            else:
                rows[line.split("\t")[0]] = line
        sidecar = sh.with_name(sh.stem + ".seeds.tsv")
        if sidecar.exists():
            for line in sidecar.read_text().splitlines()[1:]:
                srows.setdefault(line.split("\t")[0], []).append(line)
    missing = [c for c in order if c not in rows]
    with open(base, "w") as f:
        f.write("\n".join(stamps) + f"\n# merged from {len(shards)} shard(s); missing cells: {len(missing)}\n" + header + "\n")
        for c in order:
            if c in rows: f.write(rows[c] + "\n")
    with open(base.with_name(base.stem + ".seeds.tsv"), "w") as fs:
        fs.write("target\tseed\t" + "\t".join(METRICS) + "\n")
        for c in order:
            for line in srows.get(c, []): fs.write(line + "\n")
    print(f"[merge] {base}: {len(rows)}/{len(order)} cells" + (f", MISSING {missing[:3]}..." if missing else ""))
    return base


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "--merge":
        merge_shards(Path(__file__).resolve().parent, "results", sys.argv[2], sys.argv[3])

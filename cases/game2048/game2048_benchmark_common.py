"""Shared benchmark-eval harness for the game2048 domain (spec §9).

Every benchmark eval (``game2048_benchmark_{method}_eval.py``) runs the same
seed protocol — episode seeds 0..n_seeds-1 on the named scenario — through the
same ``Game2048Env`` the RL agent is evaluated in, and writes one spec-§9 TSV
row to

    results/{scenario}/benchmark/benchmark_{method}_eval_{scenario}.tsv

Columns: scenario  grid_size  prob_4  score_mean  score_var  semivar_d
semivar_u  moves_mean  max_tile_mean  reach_rate  reach_2..reach_{2^kmax} —
``score_mean`` is the gate metric (total merge points, sense=maximize) and is
deliberately the first ``*_mean`` column, which is what ``mdp_gates`` picks up
by default. Everything after it is a **bystander metric**: it describes, it
never decides — gates, crowns, and model selection stay on the objective.

The ``reach_{tile}`` block is the max-tile CDF, P(max_tile >= 2^k), enumerated
over the scenario's full support rather than hand-picked (``reach_curve_tiles``
below). Each eval also writes a per-seed sidecar (``<name>.seeds.tsv``:
seed/score/moves/max_tile) — the ground truth any post-hoc distribution or
paired per-seed comparison can be rebuilt from, since every eval runs the same
seed list.

All benchmarks run under ``action_mode="masked"``: they are legal-move
policies by construction, so masking is the encoding that matches them. The
scored quantity is pure merge score either way — ``invalid_penalty`` is a
training-time shaping term and never reaches evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from game2048_gym import Game2048Env
from game2048_scenarios import SCENARIOS

# policy stream, kept clear of the model streams and the interpreter's 9999
POLICY_STREAM_RANDOM = 9996

# a run "reaches" this tile — the headline milestone, scaled to the board so
# it actually discriminates (a 3x3 board rarely passes 128; 256 read 0.000 for
# every policy including expectimax, which measures nothing). Kept as the
# summary `reach_rate` column for continuity; the reach_{tile} CDF block is
# the curated-free replacement.
REACH_TILE = {2: 16, 3: 64, 4: 1024, 5: 2048}


def reach_curve_tiles(template) -> list[int]:
    """Full log2 support of the max-tile distribution for this scenario.

    The theoretical max tile on n_cells cells is 2**(n_cells + 1) when 4-tiles
    can spawn (the last cell is a fresh 4) and 2**n_cells with 2s only, so the
    column set is derived, never curated: degenerate ends read 1.000 / 0.000
    and are self-evident, while trimming is left to the presentation layer.
    """
    n_cells = template.grid_size ** 2
    k_max = n_cells + (1 if template.spawn_value.prob_4 > 0 else 0)
    return [2 ** k for k in range(1, k_max + 1)]


def build_arg_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("-s", "--scenario", type=str, default="3x3_20",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--n-seeds", type=int, default=8192,
                   help="episode seeds 0..n-1 (shared across all evals)")
    p.add_argument("--seed-start", dest="seed_start", type=int, default=0,
                   help="first episode seed; a nonzero start marks this run "
                        "as a CHUNK of the seed protocol: outputs get a "
                        ".part{start} suffix and the summary row describes "
                        "only the chunk — the protocol row is rebuilt from "
                        "the per-seed sidecars at merge time")
    p.add_argument("--outdir", type=str, default="results")
    return p


def score_stats(scores: np.ndarray) -> dict:
    n = scores.size
    mu = scores.mean()
    return {
        "score_mean": float(mu),
        "score_var": float(scores.var(ddof=1)),
        "semivar_d": float(np.sum(np.maximum(mu - scores, 0.0) ** 2) / (n - 1)),
        "semivar_u": float(np.sum(np.maximum(scores - mu, 0.0) ** 2) / (n - 1)),
    }


def evaluate(method: str, make_policy, args: argparse.Namespace,
             out_path: Path | None = None, note: str = "") -> dict:
    """Run the seed loop for one policy factory and write the spec-§9 TSV.

    ``make_policy(env)`` is called once after each reset and must return an
    object with ``act(board, mask) -> int``. Policies see only the board — the
    whole state is observable in this domain, so there is nothing to withhold.

    The trained policy is scored through this same function (see
    ``game2048_ppo_eval.py``), so RL and benchmarks share one seed loop, one
    env, and one TSV writer by construction rather than by convention. Note
    that the env here is the RAW ``Game2048Env`` — wrapping it in a VecEnv
    would auto-reset on termination and the end-of-episode board would be the
    NEXT episode's.

    ``out_path`` overrides the benchmark destination (the RL eval writes into
    its own run directory instead).
    """
    source = SCENARIOS[args.scenario]
    env = Game2048Env(source, observation_mode="vec",
                      action_mode="masked", reward_mode="score")

    seed_start = getattr(args, "seed_start", 0)
    seed_list = range(seed_start, seed_start + args.n_seeds)
    scores = np.zeros(args.n_seeds)
    moves = np.zeros(args.n_seeds)
    max_tiles = np.zeros(args.n_seeds)

    for i, seed in enumerate(seed_list):
        if i % 500 == 0:
            print(f"  seed {seed} ({i}/{args.n_seeds})", flush=True)
        env.reset(seed=seed)
        policy = make_policy(env)
        done = False
        while not done:
            mask = env.action_masks()
            action = policy.act(list(env._state.board), mask)
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        scores[i] = env.total_score
        moves[i] = env._state.period
        max_tiles[i] = max(env._state.board)
    env.close()

    template = source(0) if callable(source) else source
    reach = REACH_TILE.get(template.grid_size, 2048)
    curve_tiles = reach_curve_tiles(template)

    stats = score_stats(scores)
    stats["moves_mean"] = float(moves.mean())
    stats["max_tile_mean"] = float(max_tiles.mean())
    stats["reach_rate"] = float((max_tiles >= reach).mean())
    for tile in curve_tiles:
        stats[f"reach_{tile}"] = float((max_tiles >= tile).mean())

    # anchored to this file, never the CWD (spec §8.4)
    if out_path is None:
        out_dir = (Path(__file__).resolve().parent / args.outdir
                   / args.scenario / "benchmark")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"benchmark_{method}_eval_{args.scenario}.tsv"
    else:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
    if seed_start:
        # a chunk never claims the protocol filename
        out = out.with_name(f"{out.stem}.part{seed_start}{out.suffix}")
    header = ("scenario\tgrid_size\tprob_4\tscore_mean\tscore_var\t"
              "semivar_d\tsemivar_u\tmoves_mean\tmax_tile_mean\treach_rate"
              + "".join(f"\treach_{tile}" for tile in curve_tiles))
    row = (f"{args.scenario}\t{template.grid_size}\t"
           f"{template.spawn_value.prob_4}\t"
           f"{stats['score_mean']:.6f}\t{stats['score_var']:.6f}\t"
           f"{stats['semivar_d']:.6f}\t{stats['semivar_u']:.6f}\t"
           f"{stats['moves_mean']:.6f}\t{stats['max_tile_mean']:.6f}\t"
           f"{stats['reach_rate']:.6f}"
           + "".join(f"\t{stats[f'reach_{tile}']:.6f}" for tile in curve_tiles))
    range_note = (f" seeds={seed_start}..{seed_start + args.n_seeds - 1}"
                  if seed_start else "")
    out.write_text(
        f"# {method} | n_seeds={args.n_seeds}{range_note}{note}\n"
        f"{header}\n{row}\n"
    )

    # per-seed sidecar: the raw distribution every summary above derives from
    seeds_out = out.with_name(out.stem + ".seeds.tsv")
    seed_lines = [f"# {method} | n_seeds={args.n_seeds}{range_note}{note}",
                  "seed\tscore\tmoves\tmax_tile"]
    seed_lines += [f"{seed}\t{scores[i]:.0f}\t{moves[i]:.0f}\t{max_tiles[i]:.0f}"
                   for i, seed in enumerate(seed_list)]
    seeds_out.write_text("\n".join(seed_lines) + "\n")

    se = np.sqrt(stats["score_var"] / args.n_seeds)
    print(f"\n{'─' * 60}")
    print(f"  method        : {method}")
    print(f"  scenario      : {args.scenario}  (n_seeds={args.n_seeds})")
    print(f"  score_mean    : {stats['score_mean']:10.2f}  ± {se:.2f} (SE)")
    print(f"  moves_mean    : {stats['moves_mean']:10.2f}")
    print(f"  max_tile_mean : {stats['max_tile_mean']:10.2f}")
    print(f"  reach {reach:<4d}    : {stats['reach_rate']:10.4f}")
    curve = "  ".join(f"{t}:{stats[f'reach_{t}']:.3f}" for t in curve_tiles)
    print(f"  reach curve   : {curve}")
    print(f"  TSV           : {out}  (+ {seeds_out.name})")
    print(f"{'─' * 60}\n")
    return stats

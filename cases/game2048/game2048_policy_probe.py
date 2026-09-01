"""Spec-§14.1 policy probe for game2048 — the canonical entry point.

The interpretation ladder itself lives in ``game2048_interpret.py`` (replay,
structure, probes, counterfactual, symmetry, trunkprobe). That module is cited
by name in ~30 ledger entries, so it is **not** renamed; this file is the
§14.1 name the pipeline and ``mdp_conformance research.deliverables`` look for,
and it adds the one thing the ladder does not: a per-stance readout tied to the
tier-2 questions the IR actually declares.

``game2048_schema.json`` declares two (spec §14.0):

  discover  the crowned policy's own decision pattern
  confirm   snake ordering — the monotone boustrophedon chain

The ``stances`` stage reports the structural-form statistic for each, **as a
number**, measured through the same rollout instrument for the policy and for
two references. Any other stage name forwards to the ladder unchanged.

Per §14.1's "validate the instrument where the answer is known": the greedy row
is the control whose answer is known in advance — a myopic point-grabber builds
no chain — and the snake row is the reference the ``confirm`` stance is defined
against, since ``SnakeProcPolicy`` plays the ordering by construction. A
policy row is only readable once those two bracket it as expected.

Usage:
    python game2048_policy_probe.py stances -s 3x3_20 -o onehot \
        --model-path results/3x3_20/<run>/best_model.zip
    python game2048_policy_probe.py structure --model-path <...>   # -> ladder
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from game2048_board import valid_moves
from game2048_benchmark_snake import SnakeProcPolicy
from game2048_interpret import PolicyLens, _GreedyLens, _structure_row
from game2048_scenarios import SCENARIOS

LADDER_STAGES = ("replay", "structure", "probes", "counterfactual",
                 "symmetry", "trunkprobe", "frameavg")


class _SnakeLens:
    """SnakeProcPolicy through the rollout instrument the policy rows use.

    The reference for the `confirm` stance: it builds the boustrophedon chain
    by construction, so its snake statistic is the value a policy that had
    recovered the structure would have to approach.
    """

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self._pi = SnakeProcPolicy(grid_size)

    def probs(self, board: list[int]) -> np.ndarray:
        mask = valid_moves(board)
        p = np.zeros(4)
        p[self._pi.act(board, mask)] = 1.0
        return p

    def act(self, board: list[int]) -> int:
        return int(np.argmax(self.probs(board)))


def _fmt(rows: list[tuple[str, dict]], n: int) -> str:
    out = ["| row | score | moves | corner occ | home | "
           f"snake ep-max /{n * n} | mono life /{2 * n} | mono@death |",
           "|---|---|---|---|---|---|---|---|"]
    for tag, r in rows:
        home = max(r["by_corner"], key=r["by_corner"].get)
        out.append(f"| {tag} | {r['score']:.1f} ± {r['score_se']:.1f} "
                   f"| {r['moves']:.1f} | {r['corner_occ']:.3f} "
                   f"| {home} {r['by_corner'][home]:.3f} | {r['snake']:.2f} "
                   f"| {r['mono_life']:.2f} | {r['mono_death']:.2f} |")
    return "\n".join(out)


def stage_stances(lens, scenario, n_seeds: int) -> str:
    """Report each declared tier-2 stance as a number (spec §14.0/§14.1)."""
    n = lens.grid_size
    pol = _structure_row(lens, scenario, n_seeds)
    grd = _structure_row(_GreedyLens(n), scenario, n_seeds)
    snk = _structure_row(_SnakeLens(n), scenario, n_seeds)
    rows = [("policy", pol), ("greedy (control)", grd), ("snake (reference)", snk)]

    # instrument validation: the two references must bracket on the statistic
    # the confirm stance turns on, or nothing below is readable.
    ok = snk["snake"] > grd["snake"]
    span = snk["snake"] - grd["snake"]
    recovery = ((pol["snake"] - grd["snake"]) / span) if span > 0 else float("nan")

    lines = [f"# Declared tier-2 stances — {n}x{n}, {n_seeds} seeds, deterministic",
             "", _fmt(rows, n), "",
             "## instrument validation (§14.1: validate where the answer is known)",
             f"snake reference {snk['snake']:.2f} > greedy control "
             f"{grd['snake']:.2f}: {'PASS' if ok else 'FAIL'} "
             f"(span {span:.2f})",
             "",
             "## discover — the crowned policy's own decision pattern",
             f"corner occupancy {pol['corner_occ']:.3f} "
             f"(greedy {grd['corner_occ']:.3f}); home corner "
             f"{max(pol['by_corner'], key=pol['by_corner'].get)} at "
             f"{max(pol['by_corner'].values()):.3f} of max-tile states",
             f"monotone lines {pol['mono_life']:.2f}/{2 * n} during life, "
             f"{pol['mono_death']:.2f} at death "
             f"(ordering {'held to death' if pol['mono_death'] >= pol['mono_life'] else 'lost before death'})",
             "",
             "## confirm — snake ordering",
             f"policy snake ep-max {pol['snake']:.2f}/{n * n}; "
             f"recovery vs the greedy->snake span = {recovery:.2f} "
             f"(1.00 = reference structure recovered, 0.00 = no more chain "
             f"than a myopic control)"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", nargs="?", default="stances",
                    choices=("stances",) + LADDER_STAGES)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("-s", "--scenario", default="3x3_20")
    ap.add_argument("-o", "--obs-mode", dest="obs_mode", default="onehot")
    ap.add_argument("--n-seeds", type=int, default=64)
    args, rest = ap.parse_known_args(argv)

    if args.stage != "stances":
        from game2048_interpret import main as ladder_main
        fwd = [args.stage, "--model-path", args.model_path,
               "-s", args.scenario, "--obs-mode", args.obs_mode,
               "--n-seeds", str(args.n_seeds), *rest]
        return ladder_main(fwd)

    if rest:
        ap.error(f"unrecognized arguments for 'stances': {' '.join(rest)}")
    sc = SCENARIOS[args.scenario]
    sc = sc(0) if callable(sc) else sc
    lens = PolicyLens(args.model_path, grid_size=sc.grid_size,
                      observation_mode=args.obs_mode)
    report = stage_stances(lens, sc, args.n_seeds)
    print(report)
    from pathlib import Path
    out = Path(args.model_path).resolve().parent / "interpret"
    out.mkdir(exist_ok=True)
    (out / "stances.md").write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

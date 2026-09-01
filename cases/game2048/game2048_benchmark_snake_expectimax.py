"""Snake-leaf expectimax — the strategy WITH eyes (#E31 carry (i)).

The last open question of the snake instrument suite: #E31 measured the
NO-DEPTH ceiling of disciplined snake play (greedy-class at both scales,
~3.3-3.7x under the learned corner-pushers) and deliberately could not say
whether the strategy fails HERE or merely fails BLIND. This module answers
that: the same depth-limited expectimax as the reference rows — identical
MAX/CHANCE structure over the true spawn model — with the snake potential
as the LEAF EVALUATION:

    leaf(board) = max over the 8 D4 orientations of  sum_i W_o[i] * b[i],
    W = 4 ** boustrophedon_rank        (game2048_benchmark_snake geometry)

so search maximizes the expected snake-ness of the board at the horizon.
In-tree merge score is kept in the recursion for structural identity with
the reference implementation, but Phi differences dwarf it by orders of
magnitude — the objective IS the strategy; score arrives as a consequence
(in value space, every merge moves mass up-path and raises Phi).

Comparisons this row licenses, all same-depth same-chance-model:
  vs expectimax_d2 (score leaf)  — what the STRATEGY adds on top of depth;
  vs snake v3-proc (no depth)    — what DEPTH adds on top of the strategy;
  vs the learned arms            — whether snake+eyes reaches learned play.

Usage: python game2048_benchmark_snake_expectimax.py     # self-checks
"""

from __future__ import annotations

from functools import lru_cache

from game2048_benchmark_snake import orientation_weights
from game2048_board import MOVES, apply_move, empty_cells, side

DEFAULT_DEPTH = 2

_WEIGHTS: dict[int, list] = {}     # grid size -> 8 weight rows (pure python)


def _weights(n: int) -> list:
    # PURE PYTHON rows, deliberately: the leaf is a (8 x n^2) dot with tiny
    # operands, where numpy buys nothing locally and on a shared cluster its
    # BLAS threading hung a chunked job for 20h in the foreground self-check
    # (2026-08-10) — zero chunks ever started.
    if n not in _WEIGHTS:
        _WEIGHTS[n] = [list(row) for row in orientation_weights(n, 4.0)[0]]
    return _WEIGHTS[n]


@lru_cache(maxsize=1 << 18)
def _slide(board: tuple[int, ...], move: int) -> tuple[tuple[int, ...], int, bool]:
    out, gained, changed = apply_move(list(board), move)
    return tuple(out), gained, changed


@lru_cache(maxsize=1 << 18)
def _leaf_value(board: tuple[int, ...]) -> float:
    """The snake potential, orientation-maxed (pure python — see _weights)."""
    rows = _weights(side(board))
    return max(sum(w * v for w, v in zip(row, board)) for row in rows)


def _max_value(board: tuple[int, ...], depth: int, prob_4: float) -> float:
    # <= 0, not == 0: act() at depth d calls _chance_value(d), whose children
    # are _max_value(d-1) — at depth 0 that is -1, and an == guard recursed
    # into a ~112^8 tree. THIS was the hang that ate a 20h cluster job and two
    # smoke timeouts before being found (2026-08-10/11).
    if depth <= 0:
        return _leaf_value(board)
    best = None
    for move in MOVES:
        nxt, gained, changed = _slide(board, move)
        if not changed:
            continue
        v = gained + _chance_value(nxt, depth, prob_4)
        if best is None or v > best:
            best = v
    return _leaf_value(board) if best is None else best      # None => game over


def _chance_value(board: tuple[int, ...], depth: int, prob_4: float) -> float:
    cells = empty_cells(board)
    if not cells:
        return _leaf_value(board)
    per_cell = 1.0 / len(cells)
    total = 0.0
    for cell in cells:
        for tile, p in ((2, 1.0 - prob_4), (4, prob_4)):
            if p == 0.0:
                continue
            child = list(board)
            child[cell] = tile
            total += per_cell * p * _max_value(tuple(child), depth - 1, prob_4)
    return total


class SnakeExpectimaxPolicy:
    """Depth-limited expectimax with the snake potential at the leaves."""

    def __init__(self, prob_4: float, depth: int = DEFAULT_DEPTH) -> None:
        self.prob_4 = float(prob_4)
        self.depth = int(depth)

    def act(self, board: list[int], mask) -> int:
        key = tuple(board)
        best, best_value = None, None
        for action, ok in enumerate(mask):
            if not ok:
                continue
            nxt, gained, changed = _slide(key, action)
            if not changed:
                continue
            value = gained + _chance_value(nxt, self.depth, self.prob_4)
            if best_value is None or value > best_value:
                best, best_value = action, value
        if best is None:                      # only on a game-over board
            return next(a for a, ok in enumerate(mask) if ok)
        return best


def make_policy(env, depth: int = DEFAULT_DEPTH):
    return SnakeExpectimaxPolicy(env._current_scenario.spawn_value.prob_4,
                                 depth)


if __name__ == "__main__":
    from game2048_benchmark_snake import SnakePolicy
    from game2048_board import valid_moves
    from game2048_mdp import advance, init_state, open_episode
    from game2048_scenarios import SCENARIOS

    # 1. the leaf IS the v1 potential — one geometry, one implementation
    ref = SnakePolicy(3)
    for b in ([0, 2, 128, 2, 8, 16, 4, 16, 2], [8, 16, 32, 4, 0, 2, 2, 0, 0]):
        assert abs(_leaf_value(tuple(b)) - ref.phi(b)) < 1e-9, b
    # 2. depth-0 sanity: terminates (the negative-depth recursion trap) and
    #    returns a legal move. Semantic equivalence to v1's argmax is NOT
    #    asserted — at depth 0 act() is a spawn-averaged leaf, which can
    #    legitimately break leaf-argmax ties differently.
    pol0 = SnakeExpectimaxPolicy(0.2, depth=0)
    b = [8, 16, 32, 4, 0, 2, 2, 0, 0]
    a0 = pol0.act(b, valid_moves(b))
    assert valid_moves(b)[a0]
    # 3. determinism on a replayed episode
    def rollout(seed, depth, cap=120):
        # CAPPED at `cap` moves: good d2 play means LONG episodes, and an
        # uncapped double-rollout determinism check once blew a 300s smoke
        # timeout (2026-08-10). 120 deterministic moves are ample evidence.
        sc = SCENARIOS["3x3_20"]
        st, _ = init_state(sc, seed)
        st = open_episode(sc, st)
        p, acts, score = SnakeExpectimaxPolicy(sc.spawn_value.prob_4, depth), [], 0.0
        while not st.terminated and st.period < sc.T_cap and len(acts) < cap:
            a = p.act(list(st.board), valid_moves(st.board))
            acts.append(a)
            st, info = advance(sc, st, a)
            score += info["gained"]
        return acts, score
    a1, s1 = rollout(0, 2)
    a2, s2 = rollout(0, 2)
    assert a1 == a2 and s1 == s2
    print("self-checks pass", flush=True)
    print(f"\n{'seed':>5} {'score@120':>10} {'moves':>6}   (d2, capped)")
    for seed in range(3):
        acts, score = rollout(seed, 2)
        print(f"{seed:>5} {score:>10.0f} {len(acts):>6}", flush=True)

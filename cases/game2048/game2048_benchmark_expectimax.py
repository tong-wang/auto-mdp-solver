"""Expectimax baseline — the strong reference row.

2048 admits no exact DP: tile values are unbounded, so the state space cannot
be enumerated and there is no closed-form optimum to compare against. The
honest stand-in is a depth-limited **expectimax** search over the true chance
model — uniform over the empty cells, a 4 with probability ``prob_4`` — which
is what the rest of the leaderboard is measured against.

It is a *reference*, not a free upper bound: it pays real compute at every
decision (thousands of board evaluations per move), where a trained policy
pays one forward pass. Beating random and greedy is the gate; the gap to this
row is the interesting number.

Search shape, alternating layers:

    MAX    our four slides
    CHANCE the spawn: every empty cell x {2, 4}, weighted exactly
    ...    repeated `depth` times
    LEAF   `_leaf_value`

The leaf estimates *future* score, which is what a depth cut-off throws away.
Both terms are about being able to keep merging: empty cells are room to
manoeuvre, and adjacent equal tiles are merges already lined up. The weights
are hand-set, and deliberately coarse — this is a strong reference player, not
a tuned solver.
"""

from __future__ import annotations

from functools import lru_cache

from game2048_board import MOVES, apply_move, empty_cells, side

# leaf weights: roughly "an empty cell is worth about a small merge"
W_EMPTY = 8.0
W_PAIR = 4.0

DEFAULT_DEPTH = 2


@lru_cache(maxsize=1 << 18)
def _slide(board: tuple[int, ...], move: int) -> tuple[tuple[int, ...], int, bool]:
    """Cached slide. Boards repeat heavily across a search, and this is the
    single hottest call in the whole benchmark."""
    out, gained, changed = apply_move(list(board), move)
    return tuple(out), gained, changed


@lru_cache(maxsize=1 << 18)
def _leaf_value(board: tuple[int, ...]) -> float:
    """Static estimate of the score still available from this board."""
    n = side(board)
    empties = sum(1 for v in board if v == 0)
    pairs = 0
    for r in range(n):
        for c in range(n):
            v = board[r * n + c]
            if v == 0:
                continue
            if c + 1 < n and board[r * n + c + 1] == v:
                pairs += 1
            if r + 1 < n and board[(r + 1) * n + c] == v:
                pairs += 1
    return W_EMPTY * empties + W_PAIR * pairs


def _max_value(board: tuple[int, ...], depth: int, prob_4: float) -> float:
    """Our turn: the best slide, counting the points it earns."""
    if depth == 0:
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
    """Nature's turn: expectation over where the tile lands and what it is."""
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


class ExpectimaxPolicy:
    """Depth-limited expectimax over the true spawn model."""

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
    return ExpectimaxPolicy(env._current_scenario.spawn_value.prob_4, depth)

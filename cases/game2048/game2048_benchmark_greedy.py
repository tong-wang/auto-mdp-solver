"""Greedy (myopic) baseline: take the slide that scores most right now.

The mandatory myopic heuristic (skill Stage 3). One-step lookahead only — no
model of the spawn, no thought about what the board looks like afterwards
beyond a tie-break. It is a genuinely decent 2048 player and the bar an RL
policy has to clear to be interesting.

Ties on immediate score are broken by the slide that leaves the most empty
cells (room to keep playing), then by direction index so the policy is
deterministic given the board.
"""

from __future__ import annotations

from game2048_board import apply_move


class GreedyPolicy:
    """Maximize immediate merge score; tie-break on empty cells."""

    def act(self, board: list[int], mask) -> int:
        best, best_key = None, None
        for action, ok in enumerate(mask):
            if not ok:
                continue
            new_board, gained, _ = apply_move(board, action)
            key = (gained, sum(1 for v in new_board if v == 0), -action)
            if best_key is None or key > best_key:
                best, best_key = action, key
        if best is None:                      # only on a game-over board
            return next(a for a, ok in enumerate(mask) if ok)
        return best


def make_policy(env):
    return GreedyPolicy()

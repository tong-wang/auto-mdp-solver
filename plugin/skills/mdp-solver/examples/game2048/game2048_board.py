"""game2048 board mechanics — the IR's expression builtins.

Declared in ``game2048_schema.json``'s ``mdp.expr_builtins`` and resolved next
to the IR (portable-domain contract). The same functions back ``_mdp``, so the
board rules have exactly one implementation on both sides of the twin.

**Board representation.** Boards travel through the IR as a FLAT list of length
``n_cells = grid_size**2``, row-major: cell ``(r, c)`` is index
``r * grid_size + c``. The IR has no 2-D state variable, and the flat form is
what ``zeros(n_cells)`` and ``board[i] = v`` need. ``to_grid`` / ``from_grid``
convert for the gym's CNN observation modes.

**Move encoding** (unchanged from the pre-IR domain): 0=Up, 1=Down, 2=Right,
3=Left.

**On what the differential can and cannot prove.** The Stage-1 differential
proves the interpreter and the generated domain *agree*; because both call
these functions, it cannot prove the merge rule itself is right. That load is
carried elsewhere, deliberately:

* ``game2048_test.py`` checks hand-worked merge cases against this module;
* the schema's ``sum_conserved`` invariant asserts a slide never changes the
  board total — a claim taken from the rules of the game, independent of this
  code, and therefore able to catch an error both sides share.
"""

from __future__ import annotations

from math import isqrt

UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
MOVES = (UP, DOWN, RIGHT, LEFT)
MOVE_NAMES = {UP: "up", DOWN: "down", RIGHT: "right", LEFT: "left"}


# ---------------------------------------------------------------------------
# Orientation: every move becomes a compress-toward-index-0 on each line
# ---------------------------------------------------------------------------

def _to_lines(grid: list[list[int]], move: int, n: int) -> list[list[int]]:
    """Split the grid into the lines this move compresses, each oriented so
    index 0 is the wall the tiles travel toward."""
    if move == LEFT:
        return [list(row) for row in grid]
    if move == RIGHT:
        return [list(reversed(row)) for row in grid]
    if move == UP:
        return [[grid[r][c] for r in range(n)] for c in range(n)]
    if move == DOWN:
        return [[grid[r][c] for r in reversed(range(n))] for c in range(n)]
    raise ValueError(f"move must be one of {MOVES}, got {move!r}")


def _from_lines(lines: list[list[int]], move: int, n: int) -> list[list[int]]:
    """Inverse of :func:`_to_lines`."""
    grid = [[0] * n for _ in range(n)]
    if move == LEFT:
        for r, line in enumerate(lines):
            grid[r] = list(line)
    elif move == RIGHT:
        for r, line in enumerate(lines):
            grid[r] = list(reversed(line))
    elif move == UP:
        for c, line in enumerate(lines):
            for r in range(n):
                grid[r][c] = line[r]
    elif move == DOWN:
        for c, line in enumerate(lines):
            for i, r in enumerate(reversed(range(n))):
                grid[r][c] = line[i]
    else:
        raise ValueError(f"move must be one of {MOVES}, got {move!r}")
    return grid


def _merge_line(line: list[int]) -> tuple[list[int], int]:
    """Compress one line toward index 0, merging equal neighbours once each.

    Returns ``(new_line, gained)`` where ``gained`` is the 2048 score earned:
    every merge that produces a tile of value v pays v. A tile formed by a
    merge cannot merge again in the same move — the classic rule — which is
    why the scan steps by two on a match.
    """
    vals = [v for v in line if v]
    out: list[int] = []
    gained = 0
    i = 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            merged = vals[i] * 2
            out.append(merged)
            gained += merged
            i += 2
        else:
            out.append(vals[i])
            i += 1
    out.extend([0] * (len(line) - len(out)))
    return out, gained


def _apply(board, move) -> tuple[list[int], int, bool]:
    """The one slide implementation: ``(new_board, gained, changed)``.

    Inputs are normalised to ``int`` so a float leaking in from either side of
    the twin (numpy scalars, the interpreter's arithmetic) cannot make two
    otherwise-identical boards compare unequal.
    """
    flat = [int(v) for v in board]
    n = isqrt(len(flat))
    if n * n != len(flat):
        raise ValueError(f"board length {len(flat)} is not a perfect square")
    grid = [flat[r * n:(r + 1) * n] for r in range(n)]

    new_lines: list[list[int]] = []
    gained = 0
    for line in _to_lines(grid, int(move), n):
        merged, g = _merge_line(line)
        new_lines.append(merged)
        gained += g

    out_grid = _from_lines(new_lines, int(move), n)
    out = [v for row in out_grid for v in row]
    return out, gained, out != flat


# ---------------------------------------------------------------------------
# IR expression builtins (declared in mdp.expr_builtins)
# ---------------------------------------------------------------------------

def slide_board(board, move) -> list[int]:
    """The board after sliding `move`; unchanged if the move is illegal."""
    return _apply(board, move)[0]


def slide_score(board, move) -> float:
    """2048 score earned by sliding `move` — the sum of the tiles created."""
    return float(_apply(board, move)[1])


def board_changed(board, move) -> int:
    """1 if sliding `move` would move or merge anything, else 0."""
    return int(_apply(board, move)[2])


def empty_cells(board) -> list[int]:
    """Flat indices of the empty cells — the support of the spawn draw."""
    return [i for i, v in enumerate(board) if int(v) == 0]


def uniform_probs(board) -> list[float]:
    """Uniform weights over the empty cells, aligned with `empty_cells`."""
    k = len(empty_cells(board))
    if k == 0:
        raise ValueError("no empty cell to spawn into: board is full")
    return [1.0 / k] * k


def is_game_over(board) -> int:
    """1 if no legal slide remains. Checked cheaply: an empty cell always
    admits one, so the four-way scan only runs on a full board."""
    if any(int(v) == 0 for v in board):
        return 0
    return int(not any(_apply(board, m)[2] for m in MOVES))


# ---------------------------------------------------------------------------
# Helpers for the _mdp / _gym / benchmark layers (not IR builtins)
# ---------------------------------------------------------------------------

def valid_moves(board) -> list[bool]:
    """Boolean list over MOVES marking slides that change the board (spec
    §7.1). The MDP layer owns this; the gym's `action_masks()` delegates."""
    return [bool(_apply(board, m)[2]) for m in MOVES]


def apply_move(board, move) -> tuple[list[int], int, bool]:
    """Public alias of the single slide implementation, for the _mdp layer."""
    return _apply(board, move)


def side(board) -> int:
    """Board dimension n for a flat length-n² board."""
    return isqrt(len(board))


def to_grid(board) -> list[list[int]]:
    """Flat board -> n×n row-major nested list (gym observation modes)."""
    n = side(board)
    return [[int(v) for v in board[r * n:(r + 1) * n]] for r in range(n)]


def from_grid(grid) -> list[int]:
    """n×n nested list -> flat board."""
    return [int(v) for row in grid for v in row]


def d4_transforms(n: int) -> list[tuple[str, list[int], list[int]]]:
    """The 8 dihedral symmetries of the n×n board, with the induced move map.

    Returns ``(name, cell_perm, move_perm)`` triples where
    ``transformed[i] = board[cell_perm[i]]`` and a slide ``m`` on the original
    corresponds to ``move_perm[m]`` on the transformed board:

        apply_move(transform(b), move_perm[m]) == transform(apply_move(b, m))

    — the equivariance law that ``game2048_test.py`` certifies (the doubling
    law's sibling). ``move_perm`` is never hand-tabulated: it is derived from
    the transform's action on direction vectors, so the table cannot silently
    disagree with the cell permutation it ships beside.
    """
    sources = {
        "identity":       lambda i, j: (i, j),
        "rot90":          lambda i, j: (j, n - 1 - i),
        "rot180":         lambda i, j: (n - 1 - i, n - 1 - j),
        "rot270":         lambda i, j: (n - 1 - j, i),
        "flip_h":         lambda i, j: (i, n - 1 - j),
        "flip_v":         lambda i, j: (n - 1 - i, j),
        "transpose":      lambda i, j: (j, i),
        "anti_transpose": lambda i, j: (n - 1 - j, n - 1 - i),
    }
    deltas = {UP: (-1, 0), DOWN: (1, 0), RIGHT: (0, 1), LEFT: (0, -1)}
    out = []
    for name, src in sources.items():
        perm = [src(i, j)[0] * n + src(i, j)[1]
                for i in range(n) for j in range(n)]
        # where each original cell LANDS (inverse of src), for the linear part
        land = {src(i, j): (i, j) for i in range(n) for j in range(n)}
        q0, qr, qc = land[(0, 0)], land[(1, 0)], land[(0, 1)]
        lin_r = (qr[0] - q0[0], qr[1] - q0[1])   # image of delta (1, 0)
        lin_c = (qc[0] - q0[0], qc[1] - q0[1])   # image of delta (0, 1)
        move_perm = []
        for m in MOVES:
            dr, dc = deltas[m]
            image = (dr * lin_r[0] + dc * lin_c[0],
                     dr * lin_r[1] + dc * lin_c[1])
            move_perm.append(next(mm for mm, dd in deltas.items()
                                  if dd == image))
        out.append((name, perm, move_perm))
    return out


def render(board) -> str:
    """Human-readable board, for __main__ smoke tests and debugging."""
    n = side(board)
    width = max(len(str(int(v))) for v in board) if board else 1
    rows = to_grid(board)
    return "\n".join(" ".join(f"{v:>{width}}" if v else "." * width for v in row)
                     for row in rows)


if __name__ == "__main__":
    # hand-worked cases: the merge rule is the one thing the differential
    # cannot check, so it gets checked here and in game2048_test.py
    line, gained = _merge_line([2, 2, 4, 4])
    assert (line, gained) == ([4, 8, 0, 0], 12), (line, gained)
    line, gained = _merge_line([2, 2, 2, 0])
    assert (line, gained) == ([4, 2, 0, 0], 4), (line, gained)   # leftmost pair only
    line, gained = _merge_line([4, 4, 4, 4])
    assert (line, gained) == ([8, 8, 0, 0], 16), (line, gained)  # no chain merge
    line, gained = _merge_line([0, 0, 2, 0])
    assert (line, gained) == ([2, 0, 0, 0], 0), (line, gained)   # pure shift, no score

    b = [2, 2, 0,
         4, 0, 0,
         0, 0, 2]
    out, g, changed = _apply(b, LEFT)
    assert out == [4, 0, 0, 4, 0, 0, 2, 0, 0] and g == 4 and changed
    assert sum(out) == sum(b), "a slide must conserve the board total"
    assert board_changed(b, LEFT) == 1
    assert empty_cells([2, 0, 0, 4]) == [1, 2]
    assert is_game_over([2, 2, 4, 8]) == 0        # top row 2|2 can still merge
    assert is_game_over([2, 4, 4, 2]) == 1        # "2 4 / 4 2": no equal neighbours
    assert is_game_over([2, 4, 8, 16]) == 1       # 2x2, all distinct -> stuck
    print(render(b))
    print("game2048_board: all checks passed")

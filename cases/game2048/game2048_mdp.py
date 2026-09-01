"""game2048 core MDP — the simulator, with no Gym dependency and no reward.

State is the simulator's full view. What the *agent* sees is decided in
``game2048_gym.py`` by its ``observation_mode``; whether the agent may pick a
slide that changes nothing is decided there too, by its ``action_mode``. At
this layer every direction in {0,1,2,3} is accepted and an illegal slide is a
legal no-op.

**One period = one attempted slide**, running the IR's event sequence:

    I  (period 0 only)  the opening tile spawns
    M                   slide and merge; `gained` is the 2048 score earned
    S  (if the slide changed the board)  a new tile spawns
    END_OF_PERIOD       game-over check, period += 1

``init_state`` returns an **empty** board, because the IR's ``initial_state``
cannot draw: the opening spawn has to be the guarded ``I`` event inside period
0. That would leave a caller choosing its first slide before seeing any tile,
so ``I`` is guarded here on ``spawn_count == 0`` rather than ``period == 0``
— equivalent whenever an episode runs from ``init_state``, which is the path
the differential takes, but it also lets a caller perform the opening spawn
*early* via :func:`open_episode` and get a state ``advance`` will then leave
alone. The gym's ``reset`` does exactly that, so the agent sees the board its
first move lands on and its action mask means something. Either route
produces bit-identical trajectories: the same spawn_count keys the same draws.

Board mechanics live in ``game2048_board.py``, shared with the IR
interpreter, so the rules have one implementation.

Dependency order: game2048_uncertainty  <-  game2048_scenarios  <-  game2048_mdp
"""

from __future__ import annotations

from dataclasses import dataclass, field

from game2048_board import (
    MOVES,
    board_changed,
    is_game_over,
    render,
    slide_board,
    slide_score,
    valid_moves,
)
from game2048_exceptions import Game2048Error, InvalidActionError
from game2048_scenarios import Game2048Scenario


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Game2048State:
    """Full internal state of the simulator.

    ``board`` is the grid flattened row-major (cell (r,c) at r*n + c, 0 =
    empty) — the same representation the IR carries, so the two sides of the
    twin compare directly.
    """

    period: int          # attempted slides so far, illegal ones included
    terminated: bool     # game over: board full and no slide legal
    spawn_count: int     # tiles spawned so far; keys the spawn draws
    board: list[int]
    episode_seed: int
    seed_salt: int = field(repr=False)


@dataclass(slots=True)
class _SpawnCtx:
    """The SamplingContext the spawn generators read (spec §4.2)."""

    spawn_count: int
    board: list[int]
    episode_seed: int
    seed_salt: int


# ---------------------------------------------------------------------------
# Transition functions
# ---------------------------------------------------------------------------


def _spawn(
    scenario: Game2048Scenario,
    board: list[int],
    spawn_count: int,
    episode_seed: int,
    seed_salt: int,
) -> tuple[int, int, int]:
    """One spawn (the I or S event): bump the counter, then draw.

    The counter is incremented *before* the draw because it keys it — the
    IR's transitions do ``spawn_count = spawn_count + 1`` ahead of
    ``cell ~ spawn_cell.draw`` for exactly this reason. Returns
    ``(new_spawn_count, cell, tile_value)``.
    """
    ctx = _SpawnCtx(
        spawn_count=spawn_count + 1,
        board=board,
        episode_seed=episode_seed,
        seed_salt=seed_salt,
    )
    cell = scenario.spawn_cell.sample(ctx)
    tile = 2 + 2 * scenario.spawn_value.sample(ctx)
    return ctx.spawn_count, cell, tile


def init_state(
    scenario: Game2048Scenario, episode_seed: int
) -> tuple[Game2048State, dict]:
    """Start an episode on an empty board.

    The opening tile is NOT placed here — it is the ``I`` event, which
    ``advance`` runs on the first call (see the module docstring). Call
    :func:`open_episode` to place it up front instead.
    """
    state = Game2048State(
        period=0,
        terminated=False,
        spawn_count=0,
        board=[0] * scenario.n_cells,
        episode_seed=episode_seed,
        seed_salt=scenario.seed_salt,
    )
    return state, {"action_period": -1}


def open_episode(
    scenario: Game2048Scenario, state: Game2048State
) -> Game2048State:
    """Run the ``I`` event now: place the opening tile, still at period 0.

    Pure. ``advance`` skips ``I`` on the returned state (its guard is
    ``spawn_count == 0``), so every board, score and spawn draw downstream is
    identical to letting ``advance`` do it — the same spawn_count keys the
    same draw. Callers that must show or plan against the first board (the
    gym's ``reset``, the expectimax baseline) use this.

    The one visible difference is bookkeeping: ``info["spawned"]`` counts
    tiles that appeared *during* a period, so opening early leaves the
    opening tile out of period 0's count (2 rather than 4 on a 2-spawn
    opening). Nothing reads ``spawned`` but diagnostics and the
    ``sum_conserved`` invariant, and the invariant is checked on the
    interpreter's trajectory, which always opens inside period 0.
    """
    if state.spawn_count != 0:
        raise Game2048Error(
            f"episode already opened (spawn_count={state.spawn_count})"
        )
    board = list(state.board)
    spawn_count, cell, tile = _spawn(
        scenario, board, 0, state.episode_seed, state.seed_salt
    )
    board[cell] = tile
    return Game2048State(
        period=state.period,
        terminated=False,
        spawn_count=spawn_count,
        board=board,
        episode_seed=state.episode_seed,
        seed_salt=state.seed_salt,
    )


def opening_board(scenario: Game2048Scenario, episode_seed: int) -> list[int]:
    """Convenience: the board the first slide will be applied to."""
    state, _ = init_state(scenario, episode_seed)
    return open_episode(scenario, state).board


def advance(
    scenario: Game2048Scenario,
    state: Game2048State,
    decision: int,
) -> tuple[Game2048State, dict]:
    """Apply one slide and return ``(next_state, info)``. Pure — never
    mutates *state*.

    info:
        gained (float)     2048 points earned by this slide
        move_valid (int)   1 if the board moved or merged, else 0
        spawned (int)      total value of tiles spawned this period
        game_over (int)    1 if no slide is legal on the resulting board
        action_period (int) the period this slide was applied at
        score (dict)       the objective decomposition {merge, total}
    """
    move = int(decision)
    if move not in MOVES:
        raise InvalidActionError(
            f"decision must be one of {MOVES} (0=Up, 1=Down, 2=Right, 3=Left), "
            f"got {decision!r}"
        )

    board = list(state.board)
    spawn_count = state.spawn_count
    spawned = 0

    # I — the opening tile, unless a caller already placed it via open_episode
    if state.spawn_count == 0:
        spawn_count, cell, tile = _spawn(
            scenario, board, spawn_count, state.episode_seed, state.seed_salt
        )
        board[cell] = tile
        spawned += tile

    # M — slide and merge
    move_valid = board_changed(board, move)
    gained = slide_score(board, move)
    board = slide_board(board, move)

    # S — a new tile after any board-changing slide
    if move_valid:
        spawn_count, cell, tile = _spawn(
            scenario, board, spawn_count, state.episode_seed, state.seed_salt
        )
        board[cell] = tile
        spawned += tile

    # END_OF_PERIOD
    game_over = is_game_over(board)
    next_state = Game2048State(
        period=state.period + 1,
        terminated=bool(game_over),
        spawn_count=spawn_count,
        board=board,
        episode_seed=state.episode_seed,
        seed_salt=state.seed_salt,
    )
    info = {
        "gained": gained,
        "move_valid": int(move_valid),
        "spawned": int(spawned),
        "game_over": int(game_over),
        "action_period": state.period,
        # objective components travel in info; the gym builds reward from them
        "score": {"merge": gained, "total": gained},
    }
    return next_state, info


def valid_actions(state: Game2048State) -> list[bool]:
    """Slides that would change the board (spec §7.1).

    Lives here rather than in the gym because it is a property of the
    transition function; ``action_masks()`` delegates to it.
    """
    return valid_moves(state.board)


# ---------------------------------------------------------------------------
# Smoke episode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    from game2048_scenarios import SCENARIOS

    sc = SCENARIOS["3x3_20"]
    assert opening_board(sc, 42).count(0) == sc.n_cells - 1, \
        "the opening board carries exactly one tile"

    # both routes through the opening spawn must agree, step for step —
    # everything except period 0's `spawned` tally (see open_episode)
    a, _ = init_state(sc, episode_seed=42)
    b = open_episode(sc, a)
    for mv in (0, 3, 1, 2, 0, 3):
        a, ia = advance(sc, a, mv)
        b, ib = advance(sc, b, mv)
        assert a.board == b.board and a.spawn_count == b.spawn_count, \
            "open_episode must not change the trajectory"
        assert all(ia[k] == ib[k] for k in
                   ("gained", "move_valid", "game_over", "action_period")), \
            "open_episode must not change the per-period outcomes"

    state, _ = init_state(sc, episode_seed=42)
    state = open_episode(sc, state)
    rng = np.random.default_rng(0)
    total = 0.0
    while not state.terminated and state.period < sc.T_cap:
        mask = valid_actions(state)
        if not any(mask):                       # only reachable at game over
            break
        move = int(rng.choice([a for a, ok in enumerate(mask) if ok]))
        state, info = advance(sc, state, move)
        total += info["gained"]

    print(render(state.board))
    print(f"\nperiods={state.period} spawns={state.spawn_count} "
          f"score={total:.0f} max_tile={max(state.board)} "
          f"terminated={state.terminated}")
    assert sum(state.board) > 0
    print("game2048_mdp: smoke episode ok")

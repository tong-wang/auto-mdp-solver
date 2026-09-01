"""game2048's own tests — the spatial-board / action-masking exemplar.

    pytest game2048
    python game2048_test.py

The generic guarantees live in ``mdp_ir.laws``; conservation and the
illegal-slide laws live in the IR as ``mdp.invariants``. What is here is what
only this domain can state.

Two clusters carry most of the weight:

**The merge rule.** The differential proves the interpreter and the domain
agree, but both call ``game2048_board``, so the slide itself has no
independent checker there. These hand-worked cases are that checker.

**The keyed spawn stream.** Spawns are keyed on ``spawn_count``, not
``period``, which is what makes the masked and free action modes comparable on
shared seeds. The tests below assert that property directly, and a negative
control re-keys the IR on ``period`` to prove the choice is load-bearing
rather than decorative.
"""

from __future__ import annotations

import json

import pytest

from mdp_ir.schema import load_ir
from mdp_ir.testing import assert_diverges, assert_laws, assert_match, mutated, schema_beside

import numpy as np

from game2048_board import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    _merge_line,
    apply_move,
    board_changed,
    d4_transforms,
    empty_cells,
    is_game_over,
    slide_board,
    slide_score,
    uniform_probs,
    valid_moves,
)
from game2048_mdp import advance, init_state, open_episode, opening_board, valid_actions
from game2048_scenarios import SCENARIOS

SCHEMA = schema_beside(__file__)
EPISODES = 8
SALT = 1


def _compositions() -> list[str | None]:
    scenario = json.loads(SCHEMA.read_text())["mdp"]["scenario"]
    return ([None] + sorted(scenario.get("instances") or {})
            + sorted(m["name"] for m in scenario.get("mixtures") or []))


COMPOSITIONS = _compositions()


# -- engine + twin ------------------------------------------------------------


def test_engine_laws():
    assert_laws(SCHEMA)


@pytest.mark.parametrize("instance", COMPOSITIONS, ids=lambda i: i or "base")
def test_differential_matches_the_domain(instance):
    assert_match(SCHEMA, instance=instance, episodes=EPISODES, seed_salt=SALT)


def test_the_covering_set_spans_every_board_size_and_both_spawn_regimes():
    named = [c for c in COMPOSITIONS if c]
    assert {c.split("_")[0] for c in named} == {"2x2", "3x3", "4x4", "5x5"}
    # base is 3x3_20, so the 3x3 20% cell is covered without being named
    assert {c.split("_")[1] for c in named} == {"0", "20"}


# -- negative controls: the differential must have teeth ----------------------


"""A note on what a negative control can be here.

The adapter builds the scenario *from the IR's constants*, so corrupting a
constant (prob_4, grid_size, ...) corrupts both sides of the twin at once and
still matches — correctly so. A control has to change something the
interpreter reads but the domain hard-codes: a dynamics expression, an
objective expression, or a seed key.
"""


def test_a_wrong_spawn_tile_formula_is_caught():
    """`tile = 2 + 2 * four` is written into the dynamics AND hard-coded in
    game2048_mdp._spawn. Change the IR's copy and the twin must part company
    the first time a 4 spawns."""
    def bend(doc):
        for tr in doc["mdp"]["dynamics"]["transitions"]:
            tr["updates"] = [u.replace("2 + 2 * four", "2 + 4 * four")
                             for u in tr["updates"]]
    assert_diverges(SCHEMA, mutated(load_ir(SCHEMA), bend),
                    episodes=EPISODES, seed_salt=SALT)


def test_a_wrong_score_expression_is_caught():
    """The objective reads `gained`; doubling it in the IR must not match the
    domain's own merge accounting."""
    def bend(doc):
        for comp in doc["mdp"]["objective"]["per_step_components"]:
            comp["expr"] = f"2 * ({comp['expr']})"
    assert_diverges(SCHEMA, mutated(load_ir(SCHEMA), bend),
                    episodes=EPISODES, seed_salt=SALT)


def test_rekeying_the_spawn_on_period_is_caught():
    """The load-bearing modelling choice, stated as a falsifiable claim: keying
    the spawn draws on `period` instead of `spawn_count` changes the tiles as
    soon as any illegal slide is attempted (the random policy attempts plenty),
    so the twin must diverge. `mutated` edits the RESOLVED IR, hence
    `uncertainty_sources` rather than the authored `uncertainty_slots`."""
    def rekey(doc):
        for src in doc["mdp"]["uncertainty_sources"]:
            src["stages"][0]["key_exprs"] = ["period"]
    assert_diverges(SCHEMA, mutated(load_ir(SCHEMA), rekey),
                    episodes=EPISODES, seed_salt=SALT)


# -- the merge rule (the differential's blind spot) ---------------------------


@pytest.mark.parametrize("line,expected,gained", [
    ([2, 2, 4, 4], [4, 8, 0, 0], 12),      # two independent merges
    ([4, 4, 4, 4], [8, 8, 0, 0], 16),      # NOT [16]: a merged tile cannot re-merge
    ([2, 2, 2, 0], [4, 2, 0, 0], 4),       # the pair nearest the wall resolves
    ([0, 0, 2, 0], [2, 0, 0, 0], 0),       # pure shift earns nothing
    ([2, 4, 2, 4], [2, 4, 2, 4], 0),       # nothing adjacent is equal
    ([0, 0, 0, 0], [0, 0, 0, 0], 0),       # empty line
    ([2, 0, 2, 4], [4, 4, 0, 0], 4),       # a gap does not block a merge
])
def test_merge_line(line, expected, gained):
    assert _merge_line(line) == (expected, gained)


def test_merges_resolve_from_the_wall_the_tiles_travel_to():
    """[2,2,2] is ambiguous until you say which end the tiles pack against."""
    row = [2, 2, 2,
           0, 0, 0,
           0, 0, 0]
    assert slide_board(row, LEFT)[:3] == [4, 2, 0]
    assert slide_board(row, RIGHT)[:3] == [0, 2, 4]


def test_a_slide_conserves_the_board_total():
    """The claim the sum_conserved invariant rests on, checked directly on the
    mechanics rather than through a trajectory."""
    board = [2, 2, 4,
             4, 8, 8,
             2, 0, 2]
    for move in (UP, DOWN, LEFT, RIGHT):
        out, _, _ = apply_move(board, move)
        assert sum(out) == sum(board)


def test_score_is_the_sum_of_the_tiles_created():
    board = [2, 2, 0,
             4, 4, 0,
             0, 0, 0]
    # left: 2|2 -> 4 (pays 4), 4|4 -> 8 (pays 8)
    assert slide_score(board, LEFT) == 12.0
    assert slide_board(board, LEFT)[:6] == [4, 0, 0, 8, 0, 0]


def test_vertical_and_horizontal_orientations_are_consistent():
    """A column merge under UP must mirror the row merge under LEFT."""
    col = [2, 0, 0,
           2, 0, 0,
           4, 0, 0]
    assert slide_board(col, UP) == [4, 0, 0, 4, 0, 0, 0, 0, 0]
    assert slide_score(col, UP) == 4.0
    assert slide_board(col, DOWN) == [0, 0, 0, 4, 0, 0, 4, 0, 0]


def test_an_illegal_slide_is_detected_and_changes_nothing():
    packed = [4, 2, 0,
              0, 0, 0,
              0, 0, 0]
    assert board_changed(packed, LEFT) == 0
    assert slide_board(packed, LEFT) == packed
    assert slide_score(packed, LEFT) == 0.0
    assert board_changed(packed, RIGHT) == 1


def test_game_over_is_exactly_a_full_board_with_no_legal_slide():
    stuck = [2, 4, 2,
             4, 2, 4,
             2, 4, 2]
    assert is_game_over(stuck) == 1
    assert not any(valid_moves(stuck))

    mergeable = [2, 2, 4,
                 4, 8, 16,
                 2, 4, 2]
    assert is_game_over(mergeable) == 0
    assert any(valid_moves(mergeable))

    has_space = [2, 4, 0,
                 4, 2, 4,
                 2, 4, 2]
    assert is_game_over(has_space) == 0, "an empty cell always admits a slide"


def test_valid_moves_agrees_with_board_changed():
    board = [2, 0, 4,
             0, 4, 0,
             8, 0, 2]
    assert valid_moves(board) == [bool(board_changed(board, m))
                                  for m in (UP, DOWN, RIGHT, LEFT)]


def test_the_spawn_support_is_the_empty_cells():
    board = [2, 0, 4, 0, 0, 8, 0, 0, 0]
    cells = empty_cells(board)
    assert cells == [1, 3, 4, 6, 7, 8]
    probs = uniform_probs(board)
    assert len(probs) == len(cells)
    assert probs == pytest.approx([1 / 6] * 6)


def test_a_full_board_has_no_spawn_support():
    with pytest.raises(ValueError):
        uniform_probs([2, 4, 2, 4])


# -- the keyed spawn stream ---------------------------------------------------


class _Ctx:
    """A bare SamplingContext, for exercising a generator without a state."""

    def __init__(self, spawn_count: int, seed_salt: int, episode_seed: int = 77):
        self.spawn_count = spawn_count
        self.board = [0] * 9
        self.episode_seed = episode_seed
        self.seed_salt = seed_salt


def _play(scenario, seed, moves, *, open_early=False):
    """Run a move list, returning (boards, spawn_counts) per period."""
    state, _ = init_state(scenario, seed)
    if open_early:
        state = open_episode(scenario, state)
    boards, counts = [], []
    for mv in moves:
        if state.terminated:
            break
        state, _ = advance(scenario, state, mv)
        boards.append(list(state.board))
        counts.append(state.spawn_count)
    return boards, counts


def test_illegal_slides_do_not_disturb_the_tile_sequence():
    """The property that makes masked and free comparable on shared seeds.

    Two runs play the same real moves; one of them first burns every illegal
    slide available at that board. Because spawn_count — not period — keys the
    draws, the boards must stay identical move for move, even though the
    padded run has consumed far more periods.
    """
    sc = SCENARIOS["3x3_20"]
    moves = [LEFT, UP, RIGHT, DOWN] * 3

    plain, _ = init_state(sc, 5)
    plain = open_episode(sc, plain)
    padded, _ = init_state(sc, 5)
    padded = open_episode(sc, padded)

    injected = 0
    for mv in moves:
        if plain.terminated:
            break
        for bad in (UP, DOWN, RIGHT, LEFT):          # no-ops first
            if not board_changed(padded.board, bad):
                padded, info = advance(sc, padded, bad)
                assert info["move_valid"] == 0 and info["spawned"] == 0
                injected += 1
        plain, _ = advance(sc, plain, mv)
        padded, _ = advance(sc, padded, mv)
        assert padded.board == plain.board
        assert padded.spawn_count == plain.spawn_count

    assert injected >= 3, "no illegal slides were injected — test is vacuous"
    assert padded.period > plain.period, "the padded run burned extra moves"


def test_the_same_spawn_count_always_yields_the_same_tile_value():
    """Keyed semantics: the value drawn at spawn_count k depends only on
    (k, episode_seed, seed_salt) — never on when it was drawn."""
    sc = SCENARIOS["3x3_20"]
    first = [sc.spawn_value.sample(_Ctx(k, sc.seed_salt)) for k in range(1, 12)]
    again = [sc.spawn_value.sample(_Ctx(k, sc.seed_salt))
             for k in reversed(range(1, 12))][::-1]
    assert first == again
    assert set(first) == {0, 1}, "prob_4=0.2 should produce both tile values"


def test_opening_early_does_not_change_the_trajectory():
    sc = SCENARIOS["3x3_20"]
    moves = [UP, LEFT, DOWN, RIGHT, UP, LEFT]
    late, c_late = _play(sc, 21, moves)
    early, c_early = _play(sc, 21, moves, open_early=True)
    assert late == early and c_late == c_early


def test_the_opening_board_carries_exactly_one_tile():
    for name, sc in SCENARIOS.items():
        board = opening_board(sc, 9)
        assert board.count(0) == sc.n_cells - 1, name
        assert max(board) in (2, 4), name


def test_prob_4_zero_never_spawns_a_four():
    sc = SCENARIOS["3x3_0"]
    assert all(sc.spawn_value.tile_value(_Ctx(k, sc.seed_salt)) == 2
               for k in range(1, 60)), "prob_4=0 must never draw a 4"

    # and in play. The episode is opened early so period 0 carries only its own
    # spawn — opening inside period 0 would tally both tiles as spawned=4.
    state, _ = init_state(sc, 4)
    state = open_episode(sc, state)
    for mv in (UP, LEFT, DOWN, RIGHT) * 6:
        if state.terminated:
            break
        state, info = advance(sc, state, mv)
        assert info["spawned"] in (0, 2), "prob_4=0 must only ever spawn 2s"


# -- the mdp/gym boundary -----------------------------------------------------


def test_valid_actions_is_owned_by_the_mdp_layer():
    """Spec §7.1: the mask is a property of the transition function, so it is
    computed here and merely delegated to by the gym."""
    sc = SCENARIOS["3x3_20"]
    state, _ = init_state(sc, 13)
    state = open_episode(sc, state)
    assert valid_actions(state) == valid_moves(state.board)
    assert any(valid_actions(state)), "the opening board always admits a slide"


def test_an_illegal_slide_is_a_legal_no_op_at_the_mdp_layer():
    """The mdp layer never refuses a direction — masking is the gym's choice."""
    sc = SCENARIOS["3x3_20"]
    state, _ = init_state(sc, 8)
    state = open_episode(sc, state)
    before = list(state.board)
    illegal = [m for m in (UP, DOWN, RIGHT, LEFT) if not board_changed(before, m)]
    assert illegal, "the opening board should pin the tile against two walls"
    state, info = advance(sc, state, illegal[0])
    assert info["move_valid"] == 0
    assert state.board == before
    assert info["gained"] == 0.0 and info["spawned"] == 0
    assert state.period == 1, "an illegal slide still consumes a move"


def test_all_4s_is_all_2s_doubled():
    """Metamorphic scale law: with spawn *values* pinned (prob_4 = 0 vs 1) and
    the same seed, the all-4s game is the all-2s game with every tile doubled
    — same valid-move sets, same move count, score exactly x2 (every merge
    payment doubles). A strong engine check the differential cannot make
    (both twins would share a scale bug identically), and the mechanical
    content of the log2 lever: in exponent space, prob_4=1 is prob_4=0
    shifted up one plane."""
    from game2048_scenarios import _scenario

    s0 = SCENARIOS["3x3_0"]
    s0 = s0(0) if callable(s0) else s0
    s1 = _scenario("3x3_100", 3, 1.0, s0.T_cap, "all-4s scale probe")
    for seed in range(100):
        st0 = open_episode(s0, init_state(s0, seed)[0])
        st1 = open_episode(s1, init_state(s1, seed)[0])
        score0 = score1 = 0.0
        for _ in range(s0.T_cap):
            va = valid_actions(st0)
            assert va == valid_actions(st1)
            if not any(va):
                break
            move = next(i for i, ok in enumerate(va) if ok)
            st0, i0 = advance(s0, st0, move)
            st1, i1 = advance(s1, st1, move)
            score0 += i0["gained"]
            score1 += i1["gained"]
            assert [v * 2 for v in st0.board] == list(st1.board)
        assert score1 == score0 * 2
        assert st0.period == st1.period


def test_the_dynamics_are_d4_equivariant():
    """Metamorphic symmetry law (the doubling law's sibling): 2048 has no
    preferred direction. For every dihedral transform g of the board and its
    induced move relabelling σ_g (both from ``d4_transforms``), sliding σ_g(m)
    on g·b is g applied to sliding m on b — same score, same validity, bit for
    bit. Generalizes ``test_vertical_and_horizontal_orientations_are_
    consistent`` (one hand-picked element, one board) to the full 8-element
    group over random boards and every size.

    The spawn stage cannot be path-exact under g — the keyed draw picks the
    k-th empty cell in row-major order, and g reorders that enumeration — so
    its law-equality is asserted via the coupling instead: the same context
    draws the same *rank* k into either support, and the supports biject
    through g. Uniform-over-support + equal rank ⇒ equal law.

    Beyond licensing any equivariance lever (the campaign principle's
    IR-readable trigger, certified as an engine fact), this catches any
    directionally-asymmetric merge bug — a class the differential cannot see,
    since both twins share ``game2048_board``."""
    rng = np.random.default_rng(39)
    for n in (2, 3, 4):
        elems = d4_transforms(n)
        assert len({tuple(p) for _, p, _ in elems}) == 8, "D4 acts faithfully"
        for _, perm, mp in elems:
            assert sorted(perm) == list(range(n * n))
            assert sorted(mp) == [0, 1, 2, 3]

        boards = [[int(v) for v in rng.choice([0, 0, 2, 2, 4, 8, 16, 32],
                                              size=n * n)]
                  for _ in range(30)]
        boards.append([2 * (1 + (i % 2) + 2 * ((i // n) % 2))
                       for i in range(n * n)])          # full, checkerboard
        for b in boards:
            base = {m: apply_move(b, m) for m in (UP, DOWN, RIGHT, LEFT)}
            over = is_game_over(b)
            for name, perm, mp in elems:
                tb = [b[p] for p in perm]
                assert is_game_over(tb) == over, name
                assert sorted(perm[i] for i in empty_cells(tb)) \
                    == empty_cells(b), name
                for m in (UP, DOWN, RIGHT, LEFT):
                    out, gained, changed = base[m]
                    t_out, t_gained, t_changed = apply_move(tb, mp[m])
                    assert t_out == [out[p] for p in perm], (name, m)
                    assert t_gained == gained and t_changed == changed, (name, m)

    # the spawn coupling: same context => same rank into either support
    sc = SCENARIOS["3x3_20"]
    board = [2, 0, 4, 0, 0, 8, 0, 2, 0]
    for name, perm, _ in d4_transforms(3):
        tb = [board[p] for p in perm]
        for k in range(1, 9):
            ctx, tctx = _Ctx(k, sc.seed_salt), _Ctx(k, sc.seed_salt)
            ctx.board, tctx.board = board, tb
            rank = empty_cells(board).index(sc.spawn_cell.sample(ctx))
            t_rank = empty_cells(tb).index(sc.spawn_cell.sample(tctx))
            assert rank == t_rank, (name, k)


def _log2_vec(board):
    from game2048_gym import _log2_tiles
    return _log2_tiles(board)


def test_the_objective_never_reads_the_penalty():
    """invalid_penalty is gym shaping; the scored objective is merge points
    alone, so a run trained with the penalty is still scored on the game."""
    ir = load_ir(SCHEMA)
    exprs = " ".join(c.expr for c in ir.mdp.objective.per_step_components)
    assert "invalid_penalty" not in exprs
    modes = {m.name: m.expr for m in ir.gym.reward_modes}
    assert "invalid_penalty" not in modes["score"]
    assert "invalid_penalty" in modes["score_penalty"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

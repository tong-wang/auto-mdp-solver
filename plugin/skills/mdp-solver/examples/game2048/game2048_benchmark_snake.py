"""Snake-heuristic baseline: argmax over afterstates of a monotone-path weight.

INSTRUMENTATION, not a lever (2026-08-10, the S-snake discussion): this is a
benchmark beside random/greedy/expectimax that measures what disciplined
snake play is WORTH in these worlds, so the learned policies' corner-pushing
can be judged against a quantified alternative rather than an intuition. It
never feeds the learned system (the n-tuple prune stands: shape priors are
the human insight the solver exists to not need — but a baseline is a ruler,
and the interpret ladder's snake_len metric set the precedent).

The strategy, as formulated in discussion: fix a target corner; keep the
anchor row full and monotone toward it (then horizontal slides act on it as
the IDENTITY — no gaps, no equal adjacents — and only one move direction can
destroy it); build subsequent rows in alternating direction. Implemented as
an EVALUATION rather than a procedure: the boustrophedon rank grid

    6 7 8        weight W[i] = base ** rank[i]     (3x3, TR anchor;
    5 4 3         generalizes to any n: row bands top-down, even rows
    0 1 2         ascending left->right, odd rows descending)

and the policy takes the slide whose AFTERSTATE maximizes

    Phi(b) = max over the 8 D4 orientations of  sum_i W_o[i] * b[i].

A tile 2^k at rank r contributes 2^(k + 2r) at base 4, so position strictly
dominates value: any out-of-rank-order pair loses potential. base must
exceed 2 — at base 2 a tile one rank down exactly cancels one exponent up
and the ordering degenerates. The orientation max implements the dynamic
corner choice (and mid-game corner switching) with no logic; the
never-move-toward-the-anchor rule is EMERGENT — collapsing the anchor row
never wins the argmax while an alternative is legal. Phi scores the
post-merge pre-spawn afterstate — the deterministic half of the transition,
the same object a lookahead observation presents — so this baseline scores
that object with W in place of a learned trunk.

Two instruments from the discussion ride along as counters:
  forced breaks  — steps where EVERY legal move has Phi(after) < Phi(board)
                   (the hazard the move-budget argument is about);
  regime share   — fraction of steps where the active orientation's anchor
                   row is full and strictly monotone (the 3-safe-move regime).

Usage: python game2048_benchmark_snake.py          # self-checks + demo
"""

from __future__ import annotations

import numpy as np

from game2048_board import MOVES, apply_move, d4_transforms, valid_moves

DOWN = 1                       # MOVES order: 0=Up, 1=Down, 2=Right, 3=Left


def snake_ranks(n: int) -> np.ndarray:
    """Boustrophedon ranks, TR anchor: row i holds band [n²-(i+1)n, n²-in-1],
    even rows ascending left->right, odd rows descending."""
    ranks = np.empty(n * n, dtype=np.int64)
    for i in range(n):
        lo = n * n - (i + 1) * n
        band = np.arange(lo, lo + n)
        ranks[i * n:(i + 1) * n] = band if i % 2 == 0 else band[::-1]
    return ranks


def orientation_weights(n: int, base: float) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray]:
    """Per-orientation geometry: (8, n²) weight matrix M with
    Phi(b) = (M @ b).max(); (8, n) anchor-cell index rows (each
    orientation's anchor row, rank order); (8, n²) PATHS — board positions
    in DESCENDING rank order, so ``tuple(b[j] for j in path)`` is the
    board read along the snake, anchor first (the lex mode's sequence)."""
    ranks = snake_ranks(n)
    w = np.power(float(base), ranks).astype(np.float64)
    desc = np.argsort(-ranks)                      # canonical cells, rank desc
    mats, anchors, paths = [], [], []
    canon_anchor = np.arange(n)                    # row 0, ranks ascending
    for _, perm, _ in d4_transforms(n):
        perm = np.asarray(perm)
        w_o = np.empty_like(w)
        w_o[perm] = w             # dot(w_o, b) == dot(w, transformed(b))
        mats.append(w_o)
        anchors.append(perm[canon_anchor])
        paths.append(perm[desc])
    return np.stack(mats), np.stack(anchors), np.stack(paths)


class SnakePolicy:
    """Maximize the snake potential of the afterstate.

    mode="weight" (v1, KEPT for the record): Phi = orientation-maxed
    exponential dot. Two defects, diagnosed 2026-08-10 from per-move traces
    and deliberately preserved here: (a) the every-step orientation max
    THRASHES (8 corner switches per 75-move episode observed); (b) base**rank
    lets the top tile outweigh the entire rest of the chain ~30x, so the
    argmax is near-blind below the biggest one or two tiles.

    mode="lex" (v2, the fix — the discussion's definition taken literally):
    the afterstate is read ALONG THE PATH in rank order and compared
    lexicographically — anchor value first, ties broken by the next cell
    down the chain, and so on. Every cell of the chain is decisive at its
    own level; no base, no near-tie mush. Orientation is STICKY
    (``orient_lock``): the current orientation is kept unless a challenger
    strictly beats it at the ANCHOR cell, and once the anchor tile reaches
    the lock value the orientation is frozen — "dynamically adjusted in the
    early stage", operationalized.
    """

    def __init__(self, n: int, base: float = 4.0, stats: dict | None = None,
                 mode: str = "weight", orient_lock: int = 0):
        assert mode in ("weight", "lex"), mode
        self.weights, self.anchors, self.paths = orientation_weights(n, base)
        self.mode = mode
        self.orient_lock = orient_lock
        self.cur_o: int | None = None
        self.stats = stats if stats is not None else {}
        for k in ("steps", "forced_breaks", "regime_steps",
                  "orient_switches", "episodes"):
            self.stats.setdefault(k, 0)
        self.stats["episodes"] += 1

    # -- potentials ---------------------------------------------------------

    def phi(self, board) -> float:
        return float((self.weights @ np.asarray(board, dtype=np.float64)).max())

    def _active(self, board) -> int:
        return int((self.weights @ np.asarray(board, dtype=np.float64)).argmax())

    def seq(self, board, o: int) -> tuple:
        return tuple(board[j] for j in self.paths[o])

    # -- orientation (lex mode): sticky with anchor-only challenge ----------

    def _update_orientation(self, board) -> int:
        seqs = [self.seq(board, o) for o in range(len(self.paths))]
        if self.cur_o is None:
            self.cur_o = max(range(len(seqs)), key=lambda o: seqs[o])
            return self.cur_o
        cur_anchor = seqs[self.cur_o][0]
        if self.orient_lock and cur_anchor >= self.orient_lock:
            return self.cur_o                      # frozen for the episode
        best = max(range(len(seqs)), key=lambda o: seqs[o])
        if seqs[best][0] > cur_anchor:             # anchor strictly beaten
            self.cur_o = best
            self.stats["orient_switches"] += 1
        return self.cur_o

    # -- the decision -------------------------------------------------------

    def act(self, board: list[int], mask) -> int:
        if self.mode == "lex":
            o = self._update_orientation(board)
            pot_now = self.seq(board, o)
            pot = lambda after: self.seq(after, o)
        elif self.orient_lock:
            # v2w, sticky-weight: the thrash fix (hysteresis + lock) applied
            # to the SOFT potential — dominance moderate enough to trade a
            # prefix dip for structure (which pure lex can never do), but one
            # orientation served at a time (which pure argmax never did)
            o = self._update_orientation(board)
            w = self.weights[o]
            pot_now = float(w @ np.asarray(board, dtype=np.float64))
            pot = lambda after: float(w @ np.asarray(after, dtype=np.float64))
        else:
            o = self._active(board)
            pot_now = self.phi(board)
            pot = self.phi
        # regime: the serving orientation's anchor row is full & strictly
        # monotone in rank order (then only one move can destroy it)
        row = [board[j] for j in self.anchors[o]]
        if all(v > 0 for v in row) and all(a < b for a, b in zip(row, row[1:])):
            self.stats["regime_steps"] += 1
        best, best_key = None, None
        for a, ok in enumerate(mask):
            if not ok:
                continue
            after, gained, _ = apply_move(board, a)
            key = (pot(after), gained,
                   sum(1 for v in after if v == 0), -a)
            if best_key is None or key > best_key:
                best, best_key = a, key
        if best is None:                       # only on a game-over board
            return next(a for a, ok in enumerate(mask) if ok)
        self.stats["steps"] += 1
        if best_key[0] < pot_now:
            self.stats["forced_breaks"] += 1
        return best


class SnakeProcPolicy:
    """v3 — the PROCEDURAL snake: the discussion's steps (1)-(4) taken as
    CONSTRAINTS, not preferences (2026-08-10, "think harder" pass).

    Three corrections over v1/v2, each traceable to the original definition:

    1. "We never move Down" is a HARD constraint: the in-frame forbidden
       move is excluded from the candidate set and played only when forced
       (which is what a forced break IS). v1/v2 merely dis-preferred it.
    2. The invariant obliges one-spawn lookahead, and a small theorem makes
       it exact and ~free: L and R both illegal forces every row full with
       no horizontal merge, i.e. a FULL board — so only a ONE-EMPTY
       afterstate can be locked by the next spawn. Safety check: for
       1-empty afterstates, try both spawn values in that cell and require
       some non-forbidden move to survive. This is the "never LET yourself
       be forced Down" half of the strategy, not search.
    3. The objective is the monotone PREFIX of the path — (P, prefix
       values) compared lexicographically, then merge score, then empties.
       The built chain is decisive at every level (no exponential-weight
       mush); everything PAST the prefix is unconstrained, so tail
       restructuring stays free (the rigidity that sank full-sequence lex).
       P extending past the anchor row down the reversed second row IS
       steps (3)-(4).

    Orientation: dynamic while the anchor tile < lock (step (1)'s "early
    stage"), frozen after, unfrozen by a forced break (structure gone —
    re-choose). ``safety=False`` ablates correction 2 only, so the sweep
    can attribute any gain between EXECUTION (1+3) and the EYES (2).
    """

    def __init__(self, n: int, stats: dict | None = None, lock: int = 32,
                 safety: bool = True, prob_4: float = 0.2):
        _, self.anchors, self.paths = orientation_weights(n, 4.0)  # geometry
        self.n = n
        self.lock = lock
        self.safety = safety
        self.spawn_vals = (2, 4) if prob_4 > 0 else (2,)
        self.cur_o: int | None = None
        self.frozen = False
        # forbidden[o]: the real-board move that acts as canonical Down on
        # the o-oriented picture — apply_move(T(b), move_perm[m]) ==
        # T(apply_move(b, m)), so forbidden = the m with move_perm[m] == DOWN
        self.forbidden = []
        for _, _, move_perm in d4_transforms(n):
            self.forbidden.append(move_perm.index(DOWN))
        self.stats = stats if stats is not None else {}
        for k in ("steps", "forced_breaks", "regime_steps",
                  "orient_switches", "safety_saves", "episodes"):
            self.stats.setdefault(k, 0)
        self.stats["episodes"] += 1

    # -- the objective: monotone prefix along the path ----------------------

    def pkey(self, board, o: int) -> tuple:
        """(P, prefix values): longest nonempty non-increasing path prefix.
        Equal neighbours stay in the prefix — an adjacent mergeable pair is
        a cascade about to happen, not disorder."""
        vals = [board[j] for j in self.paths[o]]
        p = 0
        for i, v in enumerate(vals):
            if v == 0 or (i > 0 and v > vals[i - 1]):
                break
            p += 1
        return (p, tuple(vals[:p]))

    # -- orientation (step 1): dynamic early, frozen at lock ----------------

    def _orient(self, board) -> int:
        if self.cur_o is None:
            self.cur_o = max(range(len(self.paths)),
                             key=lambda o: self.pkey(board, o))
        elif not self.frozen:
            best = max(range(len(self.paths)),
                       key=lambda o: self.pkey(board, o))
            if best != self.cur_o and (self.pkey(board, best)
                                       > self.pkey(board, self.cur_o)):
                self.cur_o = best
                self.stats["orient_switches"] += 1
        if board[self.paths[self.cur_o][0]] >= self.lock:
            self.frozen = True
        return self.cur_o

    # -- the safety obligation (exact 1-empty trigger) ----------------------

    def _unsafe(self, after, forb: int) -> bool:
        empt = [i for i, v in enumerate(after) if v == 0]
        if len(empt) != 1:
            return False                       # a forced state needs a FULL board
        for v in self.spawn_vals:
            b2 = list(after)
            b2[empt[0]] = v
            if not any(ok for a, ok in enumerate(valid_moves(b2))
                       if a != forb):
                return True
        return False

    # -- the decision -------------------------------------------------------

    def act(self, board: list[int], mask) -> int:
        o = self._orient(board)
        forb = self.forbidden[o]
        self.stats["steps"] += 1
        if self.pkey(board, o)[0] >= self.n:
            self.stats["regime_steps"] += 1
        cands = []
        for a, ok in enumerate(mask):
            if not ok or a == forb:
                continue
            after, gained, _ = apply_move(board, a)
            p, pref = self.pkey(after, o)
            safe = 0 if (self.safety and self._unsafe(after, forb)) else 1
            cands.append(((safe, p, pref, gained, after.count(0), -a), a))
        if not cands:                          # the invariant cannot be kept
            self.stats["forced_breaks"] += 1
            self.frozen = False                # structure broken: re-choose
            if mask[forb]:
                return forb
            return next(a for a, ok in enumerate(mask) if ok)
        best = max(cands)[1]
        if self.safety and len(cands) > 1:
            no_safety = max(((k[1:], a) for k, a in cands))[1]
            if no_safety != best:
                self.stats["safety_saves"] += 1
        return best


def make_policy_factory(stats: dict, base: float = 4.0,
                        mode: str = "weight", orient_lock: int = 0,
                        safety: bool = True):
    def make_policy(env):
        if mode == "proc":
            sc = getattr(env, "_current_scenario", None)
            # prob_4 lives on the spawn_value stage, not on the scenario
            prob_4 = sc.spawn_value.prob_4
            return SnakeProcPolicy(env.n, stats=stats,
                                   lock=orient_lock or 32,
                                   safety=safety, prob_4=prob_4)
        return SnakePolicy(env.n, base=base, stats=stats,
                           mode=mode, orient_lock=orient_lock)
    return make_policy


# ---------------------------------------------------------------------------
# Self-checks (executable claims from the discussion) + demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from game2048_mdp import advance, init_state, open_episode
    from game2048_scenarios import SCENARIOS
    from game2048_board import valid_moves

    # 1. the 8 orientation vectors are distinct permutations of one another,
    #    and each path visits every cell exactly once
    m, _, paths = orientation_weights(3, 4.0)
    assert all(sorted(r) == sorted(m[0].tolist()) for r in m.tolist())
    assert len({tuple(r) for r in m.tolist()}) == 8
    assert all(sorted(p) == list(range(9)) for p in paths.tolist())
    # 2. full strictly-monotone anchor row: horizontal slides are the
    #    identity on it, and NEITHER mode chooses Down while an
    #    alternative is legal (the emergent never-move-Down invariant)
    for kw in ({"mode": "weight"}, {"mode": "lex", "orient_lock": 32}):
        pol = SnakePolicy(3, **kw)
        for board in ([8, 16, 32, 4, 0, 2, 2, 0, 0],
                      [4, 8, 64, 2, 2, 0, 0, 0, 2],
                      [16, 32, 128, 0, 4, 8, 2, 0, 0]):
            for mv in (2, 3):                         # Right, Left
                after, _, _ = apply_move(board, mv)
                assert after[:3] == board[:3], (board, MOVES[mv])
            mask = valid_moves(board)
            if any(ok for a, ok in enumerate(mask) if a != DOWN):
                assert pol.act(board, mask) != DOWN, (kw, board)
    # 3. lex sequences: anchor first, descending-rank read; the TR-canonical
    #    orientation reads the top row right-to-left first
    pol = SnakePolicy(3, mode="lex")
    b = [8, 16, 32, 4, 2, 0, 0, 0, 2]
    assert any(pol.seq(b, o)[:3] == (32, 16, 8) for o in range(8))
    # 4. hysteresis: a locked orientation does not switch even when another
    #    orientation's full sequence reads higher (anchor not beaten)
    pol = SnakePolicy(3, mode="lex", orient_lock=32)
    b0 = [2, 4, 64, 0, 8, 4, 0, 0, 2]      # anchor 64 -> locks immediately
    pol.act(b0, valid_moves(b0))
    o_locked = pol.cur_o
    b1 = [2, 4, 64, 0, 8, 16, 2, 0, 32]    # other corners improve; 64 stays
    pol.act(b1, valid_moves(b1))
    assert pol.cur_o == o_locked
    assert pol.stats["orient_switches"] == 0
    # 5. v3 forbidden-move table: identity orientation forbids DOWN, and
    #    across the 8 orientations every direction is forbidden exactly twice
    p3 = SnakeProcPolicy(3)
    assert p3.forbidden[0] == DOWN
    from collections import Counter
    assert set(Counter(p3.forbidden).values()) == {2}
    # 6. v3 pkey: monotone path prefix, equal neighbours kept
    #    (identity path reads TR->TL then the reversed second row)
    assert p3.pkey([8, 16, 32, 4, 2, 0, 0, 0, 2], 0) == \
        (5, (32, 16, 8, 4, 2))
    assert p3.pkey([8, 16, 32, 4, 4, 0, 0, 0, 0], 0)[0] == 5
    assert p3.pkey([0, 0, 0, 0, 0, 0, 0, 0, 2], 0)[0] == 0
    # 7. v3 safety theorem in action: a 1-empty afterstate whose both spawn
    #    fills leave no non-forbidden move is flagged; roomy boards are not
    assert p3._unsafe([4, 8, 16, 2, 4, 64, 8, 32, 0], DOWN) is True
    assert p3._unsafe([4, 8, 16, 2, 4, 64, 8, 0, 0], DOWN) is False
    # 8. v3 hard constraint: never the in-frame forbidden move while an
    #    alternative is legal (structural, not preferential)
    for board in ([8, 16, 32, 4, 0, 2, 2, 0, 0],
                  [4, 8, 64, 2, 2, 0, 0, 0, 2]):
        pp = SnakeProcPolicy(3)
        mask = valid_moves(board)
        if any(ok for a, ok in enumerate(mask) if a != DOWN):
            assert pp.act(board, mask) != DOWN, board
    # 9. determinism: identical action sequence on a replayed episode
    def rollout(seed, cls=SnakePolicy, **kw):
        sc = SCENARIOS["3x3_20"]
        st, _ = init_state(sc, seed)
        st = open_episode(sc, st)
        p, acts, score = cls(3, **kw), [], 0.0
        while not st.terminated and st.period < sc.T_cap:
            a = p.act(list(st.board), valid_moves(st.board))
            acts.append(a)
            st, info = advance(sc, st, a)
            score += info["gained"]
        return acts, score, p
    for cls, kw in ((SnakePolicy, {"mode": "weight"}),
                    (SnakePolicy, {"mode": "lex", "orient_lock": 32}),
                    (SnakeProcPolicy, {}),
                    (SnakeProcPolicy, {"safety": False})):
        a1, s1, _ = rollout(0, cls, **kw)
        a2, s2, _ = rollout(0, cls, **kw)
        assert a1 == a2 and s1 == s2, (cls.__name__, kw)
    print("self-checks pass")

    print(f"\n{'seed':>5} {'mode':>9} {'score':>8} {'moves':>6} "
          f"{'switch':>7} {'breaks':>7} {'saves':>6}")
    for seed in range(5):
        for cls, kw, name in ((SnakePolicy, {"mode": "weight"}, "weight"),
                              (SnakeProcPolicy, {}, "proc"),
                              (SnakeProcPolicy, {"safety": False}, "proc-ns")):
            acts, score, p = rollout(seed, cls, **kw)
            print(f"{seed:>5} {name:>9} {score:>8.0f} {len(acts):>6} "
                  f"{p.stats['orient_switches']:>7} "
                  f"{p.stats['forced_breaks']:>7} "
                  f"{p.stats.get('safety_saves', 0):>6}")

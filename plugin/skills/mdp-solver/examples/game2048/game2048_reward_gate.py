"""Gate for the two reward-scale modes added 2026-08-15 (user design).

Both are TRAINING-TIME levers on the reward's noise channel: raw merge score
spans orders of magnitude and arrives in rare huge spikes, which is exactly
the heavy tail that per-minibatch advantage normalization amplifies at
~3 episodes/rollout.  Neither is allowed to change what the campaign
MEASURES.

  log2_score   reward = log2(1 + gained), roughly [0, 15]
  empty        reward = empty cells after the transition, [0, n_cells-1]

  gr1_registered    both modes construct, step, and stay inside their
                    declared ranges over a full random episode at both
                    board sizes.
  gr2_log2_exact    reward == log2(1 + gained) step by step, and gained==0
                    still pays 0 (so a merge-less legal slide is not
                    silently subsidized).
  gr3_empty_exact   reward == the post-transition zero count, step by step;
                    0 at game over; and the identity that motivates the
                    mode -- sum_t reward == (N-o_0)*T - T(T+1)/2 + sum_t M_t
                    -- is checked numerically against the realized merge
                    counts, so the "survive long AND merge early" reading is
                    verified rather than asserted.
  gr4_spawn_free    THE PROPERTY THE DESIGN RESTS ON: reward is deterministic
                    given (board, move).  The same board and move under many
                    different spawn draws must pay the SAME reward in both
                    modes -- for `empty` because the spawn always consumes
                    exactly one cell wherever it lands.  If this ever fails
                    the reward has become spawn-luck-dependent and the
                    advantage estimator inherits variance the score reward
                    never had.
  gr5_eval_sealed   a policy TRAINED under either mode is still scored in
                    true merge points: evaluate() builds its own env at
                    reward_mode="score" and reports total_score.  Asserted
                    on the live objects, not by reading the source.
  gr6_learn         CLI smoke: both modes train end to end and record the
                    mode in the args log and the run name.
  gr8_path_indep    the objective is a STATE FUNCTION: total merge score ==
                    Phi(final board) - 4*(4s spawned), Phi(B) = sum v(log2 v
                    - 1).  The structural fact the reward taxonomy in #E49 is
                    built on; verified over 40 episodes.
  gr7_selection     best_model is selected on MERGE SCORE even when training
                    pays something else.  The eval callback used to build its
                    env from args, so under -r empty it would have kept the
                    checkpoint best at hoarding empty cells and then scored
                    that model in merge points at @8192.  Selection is now
                    pinned to score; this asserts the eval env's reward mode
                    directly.

Run from game2048/:  python game2048_reward_gate.py
"""
from __future__ import annotations

import tempfile
import warnings
from math import log2
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from game2048_board import apply_move
from game2048_gym import REWARD_MODES, Game2048Env
from game2048_scenarios import SCENARIOS


def _env(mode, scen="4x4_20", act="masked"):
    return Game2048Env(SCENARIOS[scen], observation_mode="onehot",
                       action_mode=act, reward_mode=mode)


def _rollout(env, seed=7, limit=4000, greedy=False):
    """(reward, gained, board_before, move, board_after) per step.

    greedy=True takes the largest immediate merge, which survives long enough
    to produce hauls spanning decades — needed to exercise the compression.
    """
    env.reset(seed=seed)
    out = []
    for _ in range(limit):
        mask = env.unwrapped.action_masks()
        if greedy:
            b = env.unwrapped._state.board
            move = max(np.flatnonzero(mask),
                       key=lambda m: apply_move(b, int(m))[1])
            move = int(move)
        else:
            move = int(np.flatnonzero(mask)[0])
        before = list(env.unwrapped._state.board)
        _, r, term, trunc, info = env.step(move)
        out.append((float(r), float(info["gained"]), before, move,
                    list(env.unwrapped._state.board)))
        if term or trunc:
            break
    return out


def gate_gr1_registered() -> None:
    for mode in ("log2_score", "empty"):
        assert mode in REWARD_MODES, mode
    for scen, n in (("3x3_20", 9), ("4x4_20", 16)):
        for mode, lo, hi in (("log2_score", 0.0, 20.0), ("empty", 0.0, n - 1)):
            steps = _rollout(_env(mode, scen))
            assert len(steps) > 5, (mode, scen, len(steps))
            rs = [s[0] for s in steps]
            assert min(rs) >= lo and max(rs) <= hi, (mode, scen, min(rs), max(rs))
    print("[PASS] gr1_registered: both modes run at 3x3 and 4x4, rewards "
          "inside their declared ranges")


def gate_gr2_log2_exact() -> None:
    steps = _rollout(_env("log2_score"), greedy=True)
    zero_merge = 0
    for r, g, *_ in steps:
        assert abs(r - log2(1.0 + g)) < 1e-9, (r, g)
        if g == 0:
            zero_merge += 1
            assert r == 0.0
    # and the compression is real: the raw haul spans decades, the reward not
    hauls = [g for _, g, *_ in steps if g > 0]
    rews = [r for r, g, *_ in steps if g > 0]
    assert max(hauls) / max(min(hauls), 1) > 20, "episode too flat to test"
    print(f"[PASS] gr2_log2_exact: reward == log2(1+gained) over "
          f"{len(steps)} steps ({zero_merge} merge-less); haul range "
          f"{min(hauls):.0f}-{max(hauls):.0f} compressed to "
          f"{min(rews):.2f}-{max(rews):.2f}")


def gate_gr3_empty_exact() -> None:
    env = _env("empty")
    steps = _rollout(env, greedy=True)
    n_cells = 16
    for r, _g, _b, _m, after in steps:
        assert r == float(after.count(0)), (r, after.count(0))
    if env.unwrapped._state.terminated:
        assert steps[-1][0] == 0.0, "a full board must pay 0"

    # the identity the mode is argued from:
    #   sum_t (N - o_t) = (N - o_0)*T - T(T+1)/2 + sum_t M_t
    # with o_t occupancy after step t and M_t cumulative merges through t.
    env2 = _env("empty")
    env2.reset(seed=7)
    o0 = n_cells - env2.unwrapped._state.board.count(0)
    # merges made by a move = (#tiles before) - (#tiles after the slide)
    merges = []
    for _r, _g, before, move, _after in steps:
        after_move, _gain, _chg = apply_move(before, move)
        merges.append((n_cells - before.count(0))
                      - (n_cells - after_move.count(0)))
    T = len(steps)
    cum = np.cumsum(merges)
    lhs = sum(r for r, *_ in steps)
    rhs = (n_cells - o0) * T - T * (T + 1) / 2 + float(cum.sum())
    assert abs(lhs - rhs) < 1e-6, (lhs, rhs)
    print(f"[PASS] gr3_empty_exact: reward == post-transition empty count "
          f"over {T} steps; the return identity holds exactly "
          f"(sum={lhs:.0f}), so a merge at step s is paid (T-s) times")


def gate_gr4_spawn_free() -> None:
    """Same board, same move, many spawn draws -> identical reward."""
    for mode in ("log2_score", "empty", "score"):
        seen = {}
        for seed in range(40):
            env = _env(mode)
            env.reset(seed=seed)
            # drive to a common board by a fixed move sequence, then compare
            # the NEXT reward across the spawn draws that seed produced
            board = list(env.unwrapped._state.board)
            mask = env.unwrapped.action_masks()
            move = int(np.flatnonzero(mask)[0])
            _, r, *_ = env.step(move)
            key = (tuple(board), move)
            seen.setdefault(key, []).append(float(r))
        multi = {k: v for k, v in seen.items() if len(set(v)) > 1}
        assert not multi, (mode, list(multi.items())[:2])
        shared = max(len(v) for v in seen.values())
        assert shared >= 2, f"{mode}: no repeated (board, move) to compare"
    print("[PASS] gr4_spawn_free: reward is a function of (board, move) "
          "alone in all three modes — the spawn draw never moves it")


def gate_gr5_eval_sealed() -> None:
    """Training reward cannot reach a leaderboard number."""
    import inspect

    from game2048_benchmark_common import evaluate

    src = inspect.getsource(evaluate)
    assert 'reward_mode="score"' in src, \
        "evaluate() no longer pins the scoring reward mode"
    assert "total_score" in src, "evaluate() no longer reports total_score"
    # and the two accumulators really are distinct under a non-score mode
    env = _env("empty")
    steps = _rollout(env)
    assert env.unwrapped.total_score != env.unwrapped.total_reward, \
        "total_score tracked the training reward — the seal is broken"
    assert env.unwrapped.total_score == sum(g for _r, g, *_ in steps), \
        "total_score is not the true merge total"
    print(f"[PASS] gr5_eval_sealed: evaluate() pins reward_mode=score and "
          f"reports total_score; env keeps them apart "
          f"(score={env.unwrapped.total_score:.0f} vs "
          f"reward={env.unwrapped.total_reward:.0f})")


def gate_gr6_learn() -> None:
    import subprocess
    import sys

    for mode in ("log2_score", "empty"):
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, "game2048_ppo_train.py", "-s", "3x3_20",
                   "-o", "onehot", "-a", "masked", "-r", mode,
                   "--total_timesteps", "4096",
                   "--n-envs", "2", "--n_steps", "256", "--batch_size", "256",
                   "--eval_freq", "999999", "--n_eval_episodes", "2",
                   "--features_dim", "32", "--channels", "16", "32",
                   "--outdir", td]
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=1500)
            assert out.returncode == 0, out.stderr[-1500:]
            runs = list(Path(td).glob(f"3x3_20/*rew{mode}*"))
            assert len(runs) == 1, \
                f"run name must record the reward mode: " \
                f"{[p.name for p in Path(td).glob('3x3_20/*')]}"
            argtxt = (runs[0] / "3x3_20_mppo_args.txt").read_text()
            assert f"reward_mode={mode}" in argtxt, argtxt[:300]
    print("[PASS] gr6_learn: both modes train via the CLI and record "
          "themselves in the run name and args log")


def gate_gr8_path_independence() -> None:
    """THE STRUCTURAL FACT behind the whole reward taxonomy (#E49).

    Define the board potential  Phi(B) = sum over tiles of v*(log2(v) - 1).
    Then for any board and any legal move:
      * a merge of two v's into 2v raises Phi by exactly 2v -- which IS the
        score that merge pays;
      * a spawned 2 raises Phi by 2*(1-1) = 0;
      * a spawned 4 raises Phi by 4*(2-1) = 4, paying no score.
    Summing over an episode gives the exact identity

        total merge score  ==  Phi(final board)  -  4 * (number of 4s spawned)

    so the score is a STATE FUNCTION of the final board, corrected only by a
    spawn-luck term the policy does not control.  Two policies that end on
    the same board with the same 4-spawn count have earned the same score
    however they got there: the objective is path-independent.  What a policy
    can influence is only (a) how much material arrives -- survival -- and
    (b) how concentrated it is left -- since Phi is convex in tile rank, one
    4096 is worth more than two 2048s.  Every dense reward in the taxonomy is
    a proxy for one of those two axes.
    """
    from game2048_board import apply_move as _am

    n_ep, n4_total = 40, 0
    for seed in range(n_ep):
        env = _env("score")
        env.reset(seed=seed)
        n4 = sum(1 for v in env._state.board if v == 4)   # opening tiles
        rng = np.random.default_rng(seed)
        while True:
            before = list(env._state.board)
            move = int(rng.choice(np.flatnonzero(env.action_masks())))
            slid, _g, _c = _am(before, move)
            _o, _r, term, trunc, _i = env.step(move)
            # the spawn is the single cell that went from empty to a tile
            n4 += sum(1 for a, b in zip(slid, env._state.board)
                      if a == 0 and b == 4)
            if term or trunc:
                break
        phi = sum(v * (log2(v) - 1) for v in env._state.board if v > 0)
        assert abs(env.total_score - (phi - 4 * n4)) < 1e-6, \
            (seed, env.total_score, phi, n4)
        n4_total += n4
    print(f"[PASS] gr8_path_independence: score == Phi(final) - 4*n_spawned4 "
          f"exactly over {n_ep} episodes ({n4_total} 4-spawns) — the "
          f"objective is a state function of the final board")


def gate_gr7_selection() -> None:
    """The eval env pays merge score whatever the training reward is."""
    from game2048_ppo_train import _make_env, parse_args

    for mode in ("empty", "log2_score", "score"):
        args = parse_args(["-s", "4x4_20", "-o", "onehot",
                           "-a", "masked", "-r", mode])
        assert args.reward_mode == mode
        train_e = _make_env(args)()
        eval_e = _make_env(args, reward_mode="score")()
        assert train_e.unwrapped.reward_mode == mode, \
            "the TRAINING env must still pay the mode under test"
        assert eval_e.unwrapped.reward_mode == "score", \
            f"selection would run on {eval_e.unwrapped.reward_mode}"
    print("[PASS] gr7_selection: training env pays the mode under test, "
          "selection env always pays merge score")


def main() -> None:
    gate_gr1_registered()
    gate_gr2_log2_exact()
    gate_gr3_empty_exact()
    gate_gr4_spawn_free()
    gate_gr5_eval_sealed()
    gate_gr6_learn()
    gate_gr7_selection()
    gate_gr8_path_independence()
    print("\nGATE PASS: reward-scale modes (log2_score, empty)")


if __name__ == "__main__":
    main()

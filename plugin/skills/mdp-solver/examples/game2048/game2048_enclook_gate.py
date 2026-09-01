"""Executable gate for the ENC/LOOK + AX pools (#E24/#E25).

Torch-dependent, so it lives OUTSIDE ``game2048_test.py`` (the domain test
file stays runnable under the torch-free harness ``[dev]`` extra). Gates:

  1. THERMOMETER correctness (gym `thermo` mode): decode(planes) recovers
     every exponent on random boards; plane 0 marks exactly the empty
     cells; nesting is monotone (plane_{k+1} <= plane_k for k >= 1)
  2. LOOKAHEAD planes (gym `onehot_la` / `thermo_la`): block 0 equals the
     base encoding of the board; blocks 1..4 equal the base encoding of
     apply_move(board, m) in MOVES order; the onehot_la blocks are
     bit-exact against the ``afterstate_candidates`` path (same
     encoder, no drift); an illegal slide contributes the unchanged board
  3. trainable counts, per arm, against the 4x4 control's 261,984
     (SmallBoardCnn (64,96) fd128) — a lever's delta must never be a
     silent capacity delta. T is an exact-shape swap (0.00%)
  4. AXIAL reach (the load-bearing structural gate): with 1 round, the
     trunk grid at cell (k,l) has NONZERO input-gradient support exactly
     on row k + column l (the "cross") and ZERO elsewhere — structural
     zeros hold exactly; with 2 rounds the support is the whole board.
     Proves the broadcast-mix rounds are what carry cross-axis
     composition, i.e. that ``axial - rowcol`` isolates composition
  5. determinism: two seeded forward/backward passes bit-identical, for
     every (obs mode, extractor) cell the pool runs

Usage: python game2048_enclook_gate.py
"""

from __future__ import annotations

import numpy as np
import torch

from game2048_board import MOVES, apply_move
from game2048_gym import Game2048Env
from game2048_ppo_train import EXTRACTORS
from game2048_scenarios import SCENARIOS

BATCH = 16
CONTROL_TRAINABLES = 261_984  # SmallBoardCnn (64,96) fd128 @ 17 planes (#E17)

# the arms: (name, observation_mode, extractor, extractor kwargs)
ARMS = [
    ("T          (thermo, crown)", "thermo", "small",
     {"features_dim": 128, "channels": (64, 96), "kernel_size": 3}),
    ("A          (onehot_la, crown-shaved)", "onehot_la", "small",
     {"features_dim": 128, "channels": (40, 96), "kernel_size": 3}),
    ("TA         (thermo_la, crown-shaved)", "thermo_la", "small",
     {"features_dim": 128, "channels": (40, 96), "kernel_size": 3}),
    ("rowcol4    (onehot, rowcol)", "onehot", "rowcol",
     {"features_dim": 128, "channels": (225,)}),
    ("axial      (onehot, axial)", "onehot", "axial",
     {"features_dim": 128, "channels": (72, 71)}),
    ("axial-th   (thermo, axial)", "thermo", "axial",
     {"features_dim": 128, "channels": (72, 71)}),
    # S arms (2026-08-08): the augment-not-substitute answer to T's failure
    ("S          (onehot_s, crown)", "onehot_s", "small",
     {"features_dim": 128, "channels": (64, 96), "kernel_size": 3}),
    ("axial-S    (onehot_s, axial)", "onehot_s", "axial",
     {"features_dim": 128, "channels": (72, 71)}),
    ("SA         (onehot_s_la, crown-shaved)", "onehot_s_la", "small",
     {"features_dim": 128, "channels": (39, 96), "kernel_size": 3}),
]


def _env(obs_mode: str) -> Game2048Env:
    return Game2048Env(SCENARIOS["4x4_20"], observation_mode=obs_mode,
                       action_mode="masked", reward_mode="score")


def _random_boards(rng: np.random.Generator, n_cells: int, count: int = 64):
    """Random boards over the full exponent range, empties included."""
    for _ in range(count):
        exps = rng.integers(0, n_cells + 1, size=n_cells)  # 0 = empty
        yield [0 if e == 0 else 2 ** int(e) for e in exps]


def gate_thermometer() -> None:
    env = _env("thermo")
    rng = np.random.default_rng(0)
    for board in _random_boards(rng, env.n_cells):
        planes = env._encode_board(board)
        assert planes.shape == (env.n_planes, env.n, env.n)
        for i, v in enumerate(board):
            r, c = i // env.n, i % env.n
            col = planes[:, r, c]
            k = 0 if v == 0 else min(int(np.log2(v)), env.n_planes - 1)
            # decode: exponent = number of set thermometer planes
            assert int(col[1:].sum()) == (k if v else 0), (board, i)
            # plane 0 is exactly the empty indicator
            assert col[0] == (1.0 if v == 0 else 0.0), (board, i)
            # monotone nesting
            assert all(col[j + 1] <= col[j] for j in range(1, env.n_planes - 1))
    print("thermo : decode/empty-plane/nesting on 64 random boards    OK")


def gate_lookahead() -> None:
    for mode in ("onehot_la", "thermo_la", "onehot_s_la"):
        env = _env(mode)
        base_env = _env(mode.removesuffix("_la"))
        rng = np.random.default_rng(1)
        block = base_env.observation_space.shape[0]
        for board in _random_boards(rng, env.n_cells):
            obs = env._encode_board(board)
            assert obs.shape == ((1 + len(MOVES)) * block, env.n, env.n)
            blocks = obs.reshape(1 + len(MOVES), block, env.n, env.n)
            assert np.array_equal(blocks[0], base_env._encode_board(board))
            for a, m in enumerate(MOVES):
                after, _, moved = apply_move(board, m)
                assert np.array_equal(
                    blocks[1 + a], base_env._encode_board(after)), (m,)
                if not moved:  # illegal slide -> identity block
                    assert np.array_equal(blocks[1 + a], blocks[0])
        print(f"{mode:9s}: 5 blocks vs apply_move on 64 random boards"
              "     OK")
    # cross-check against afterstate_candidates: same boards, same encoder
    env = _env("onehot_la")
    base_env = _env("onehot")
    env.reset(seed=7)
    base_env.reset(seed=7)
    cand, _ = base_env.afterstate_candidates()
    blocks = env._get_obs().reshape(1 + len(MOVES), env.n_planes, env.n, env.n)
    assert np.array_equal(blocks[1:], cand), "la blocks != afterstate_candidates"
    print("onehot_la: blocks 1..4 bit-exact vs afterstate_candidates    OK")


def gate_onehot_s() -> None:
    env = _env("onehot_s")
    base = _env("onehot")
    rng = np.random.default_rng(3)
    for board in _random_boards(rng, env.n_cells):
        obs = env._encode_board(board)
        assert obs.shape == (env.n_planes + 1, env.n, env.n)
        # planes 0..16 ARE the onehot encoding, untouched
        assert np.array_equal(obs[:env.n_planes], base._encode_board(board))
        # plane 17 is log2(v)/n_cells, empty = 0, max tile = 1.0
        for i, v in enumerate(board):
            want = 0.0 if v == 0 else np.log2(v) / env.n_cells
            assert abs(obs[env.n_planes, i // env.n, i % env.n] - want) < 1e-6
    print("onehot_s: onehot planes untouched + scalar = log2/n_cells    OK")


def gate_policy_mirror() -> None:
    """Game2048Policy._encode 'must mirror game2048_gym._get_obs' (its own
    contract) — assert it actually does, for every new mode, so the
    deployable artifact and the training env can never drift apart."""
    from game2048_policy import Game2048Policy
    for mode in ("thermo", "onehot_s", "onehot_la", "thermo_la",
                 "onehot_s_la"):
        env = _env(mode)
        # white-box: _encode needs only these attrs, not a loaded model
        pol = object.__new__(Game2048Policy)
        pol.observation_mode = mode
        pol.grid_size = env.n
        pol.n_cells = env.n_cells
        rng = np.random.default_rng(2)
        for board in _random_boards(rng, env.n_cells):
            assert np.array_equal(pol._encode(board), env._encode_board(board))
    print("mirror : Game2048Policy._encode == gym encoding, new modes  OK")


def _build(arm) -> tuple[torch.nn.Module, Game2048Env]:
    name, obs_mode, extractor, kwargs = arm
    env = _env(obs_mode)
    torch.manual_seed(0)
    return EXTRACTORS[extractor](env.observation_space, **kwargs), env


def gate_counts() -> None:
    print(f"control: SmallBoardCnn (64,96) fd128 @ onehot = "
          f"{CONTROL_TRAINABLES:,} trainables")
    for arm in ARMS:
        model, _ = _build(arm)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        delta = 100.0 * (n - CONTROL_TRAINABLES) / CONTROL_TRAINABLES
        print(f"  {arm[0]:38s} {n:>9,}  ({delta:+.2f}%)")
        assert abs(delta) < 1.0, f"{arm[0]} outside the 1% matching band"
    print(f"counts : all {len(ARMS)} arms within 1% of the control            OK")


def gate_forward_grads() -> None:
    for arm in ARMS:
        model, env = _build(arm)
        obs, _ = env.reset(seed=3)
        x = torch.tensor(obs[None].repeat(BATCH, 0))
        out = model(x)
        assert out.shape == (BATCH, arm[3]["features_dim"])
        assert torch.isfinite(out).all()
        out.sum().backward()
        missing = [n for n, p in model.named_parameters()
                   if p.requires_grad and (p.grad is None or
                                           not torch.isfinite(p.grad).all())]
        assert not missing, (arm[0], missing)
    print("fwd/bwd: (B, fd) finite, grads reach every trainable        OK")


def gate_axial_reach() -> None:
    env = _env("onehot")
    obs, _ = env.reset(seed=5)
    for n_rounds, expect in ((1, "cross"), (2, "board")):
        torch.manual_seed(0)
        model = EXTRACTORS["axial"](env.observation_space, features_dim=128,
                                    channels=(72, 71), n_rounds=n_rounds)
        x = torch.tensor(obs[None], requires_grad=True)
        grid = model.trunk(x)
        k, l = 1, 2  # probe cell, off both diagonals
        grid[0, :, k, l].sum().backward()
        support = x.grad[0].abs().sum(dim=0) > 0  # (n, n) over planes
        cross = torch.zeros_like(support)
        cross[k, :] = True
        cross[:, l] = True
        if expect == "cross":
            # structural zeros are exact: nothing outside the cross may leak
            assert not support[~cross].any(), "reach leaked past the cross"
            assert support[k, l], "probe cell not in its own support"
        else:
            assert support.all(), "2 rounds should reach the whole board"
    print("axial  : reach = cross @1 round, whole board @2 rounds      OK")


def gate_determinism() -> None:
    for arm in ARMS:
        outs = []
        for _ in range(2):
            model, env = _build(arm)  # re-seeded identically in _build
            obs, _ = env.reset(seed=11)
            x = torch.tensor(obs[None].repeat(BATCH, 0))
            out = model(x)
            out.sum().backward()
            g = torch.cat([p.grad.flatten() for _, p in
                           sorted(model.named_parameters())])
            outs.append((out.detach().clone(), g.clone()))
        assert torch.equal(outs[0][0], outs[1][0]), arm[0]
        assert torch.equal(outs[0][1], outs[1][1]), arm[0]
    print("determ : seeded forward/backward bit-identical, all arms    OK")


if __name__ == "__main__":
    gate_thermometer()
    gate_lookahead()
    gate_onehot_s()
    gate_policy_mirror()
    gate_counts()
    gate_forward_grads()
    gate_axial_reach()
    gate_determinism()
    print("\ngame2048_enclook_gate: ALL GATES PASS")

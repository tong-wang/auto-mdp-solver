"""Executable gate for the ATT pool (#E21/#E22).

Torch-dependent, so it lives OUTSIDE ``game2048_test.py`` — the domain test
file must stay runnable under the torch-free harness ``[dev]`` extra. Gates,
per pool arm x board size (3x3: 10 planes, 4x4: 17 planes):

  1. forward returns (B, features_dim), all finite, gradients reach every
     trainable tensor; trainable counts printed against the scale's control
     so a positional-scheme delta is never silently a capacity delta
  2. embedding-is-a-lookup, inherited from the EMB gate: the bias-free embed
     of a one-hot equals column selection from the weight matrix
  3. PERMUTATION TEST (the load-bearing one): with the positional path off,
     the trunk is permutation-EQUIVARIANT over tokens -- trunk(Px) = P
     trunk(x) -- so the board is only a multiset; with it on, that breaks.
     This is what proves the positional term, not the attention, is the arm's
     actual lever. (Equivariance, not invariance: the head then flattens
     tokens in a fixed order, which is where absolute position re-enters.)
  4. declared sharing structure of the bias tensors -- rel: depends only on
     (drow, dcol); axis: exactly two learned off-diagonal values (same-row,
     same-col) with everything else pinned to 0
  5. determinism: two seeded forward/backward passes bit-identical

Usage: python game2048_attn_gate.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from game2048_ppo_train import EXTRACTORS, SmallBoardCnn

BATCH = 16
ARMS = ["attn-abs", "attn-rel", "attn-axis"]

# (tag, n_planes, n, control, attn dials) -- the control is the scale's crown
# of record: 3x3 the capacity CNN (#E14), 4x4 the PM4-resolved 2D-wide (#E17).
# Dials are trainable-matched per scale. Depth
# and FFN mult are held EQUAL across scales so a scale-dependent result is not
# also an architecture change; only d_model (and the head count, which costs
# no parameters) adapt. Head dim stays ~24-26 at both scales.
BOARDS = [
    ("3x3", 10, 3, {"channels": (64, 128), "features_dim": 256},
     {"features_dim": 256, "d_model": 96, "n_layers": 2, "n_heads": 4,
      "ffn_mult": 2}),
    ("4x4", 17, 4, {"channels": (64, 96), "features_dim": 128},
     {"features_dim": 128, "d_model": 78, "n_layers": 2, "n_heads": 3,
      "ffn_mult": 2}),
]


def _space(n_planes: int, n: int) -> gym.spaces.Box:
    return gym.spaces.Box(low=0.0, high=1.0, shape=(n_planes, n, n),
                          dtype=np.float32)


def _onehot(rng, n_planes: int, n: int, batch: int = BATCH) -> torch.Tensor:
    """Random one-hot boards: every cell has exactly one active plane."""
    obs = np.zeros((batch, n_planes, n, n), dtype=np.float32)
    ks = rng.integers(0, n_planes, size=(batch, 1, n, n))
    np.put_along_axis(obs, ks, 1.0, axis=1)
    return torch.as_tensor(obs)


def _n_trainable(module: nn.Module) -> int:
    return sum(t.numel() for t in module.parameters() if t.requires_grad)


def gate_forward_backward() -> None:
    rng = np.random.default_rng(0)
    print(f"{'arm':10s} {'board':6s} {'out':12s} {'trainable':>10s} "
          f"{'vs control':>11s}")
    for board, n_planes, n, ctrl_kw, kw in BOARDS:
        ctrl = _n_trainable(SmallBoardCnn(_space(n_planes, n), **ctrl_kw))
        print(f"{'control':10s} {board:6s} {'-':12s} {ctrl:>10,} {'':>11s}")
        for name in ARMS:
            ex = EXTRACTORS[name](_space(n_planes, n), **kw)
            out = ex(_onehot(rng, n_planes, n))
            assert out.shape == (BATCH, kw["features_dim"]), (name, out.shape)
            assert torch.isfinite(out).all(), (name, board, "non-finite out")
            out.sum().backward()
            for tag, t in ex.named_parameters():
                assert t.grad is not None and torch.isfinite(t.grad).all(), \
                    (name, board, tag, "no finite gradient")
            k = _n_trainable(ex)
            print(f"{name:10s} {board:6s} {str(tuple(out.shape)):12s} "
                  f"{k:>10,} {100 * (k / ctrl - 1):>+10.2f}%")


def gate_embed_is_lookup() -> None:
    rng = np.random.default_rng(1)
    for board, n_planes, n, _, kw in BOARDS:
        for name in ARMS:
            ex = EXTRACTORS[name](_space(n_planes, n), **kw)
            assert ex.embed.bias is None, (name, "embed must be bias-free")
            obs = _onehot(rng, n_planes, n)
            with torch.no_grad():
                y = ex.embed(obs.flatten(2).transpose(1, 2))   # (B, cells, d)
            ks = obs.argmax(dim=1).flatten(1)                  # (B, cells)
            expect = ex.embed.weight.detach().T[ks]            # (B, cells, d)
            assert torch.allclose(y, expect, atol=1e-6), (board, name,
                                                          "not a lookup")
    print("embed : bias-free embed == embedding lookup            OK")


def gate_permutation() -> None:
    """Positional path off => trunk permutation-equivariant; on => not."""
    rng = np.random.default_rng(2)
    g = torch.Generator().manual_seed(7)
    for board, n_planes, n, _, kw in BOARDS:
        for name in ARMS:
            ex = EXTRACTORS[name](_space(n_planes, n), **kw)
            obs = _onehot(rng, n_planes, n)
            cells = n * n
            perm = torch.randperm(cells, generator=g)
            # permute cells, keeping each cell's one-hot intact
            flat = obs.flatten(2)                              # (B, planes, C)
            obs_p = flat[:, :, perm].reshape(obs.shape)
            with torch.no_grad():
                off, off_p = (ex.trunk(obs, positional=False),
                              ex.trunk(obs_p, positional=False))
                on, on_p = ex.trunk(obs), ex.trunk(obs_p)
            assert torch.allclose(off_p, off[:, perm], atol=1e-5), (
                board, name, "trunk is NOT permutation-equivariant with the "
                             "positional path off — something else is "
                             "carrying position")
            assert not torch.allclose(on_p, on[:, perm], atol=1e-4), (
                board, name, "positional path is ACTIVE but changes nothing — "
                             "the arm's lever is dead")
    print("perm  : positional path off => equivariant, on => broken  OK")


def gate_bias_structure() -> None:
    """rel depends only on (drow, dcol); axis has exactly two learned values."""
    for board, n_planes, n, _, kw in BOARDS:
        rel = EXTRACTORS["attn-rel"](_space(n_planes, n), **kw)
        with torch.no_grad():
            b = rel.attn_bias()                                # (H, C, C)
        idx = torch.arange(n * n)
        dr = (idx[:, None] // n - idx[None, :] // n)
        dc = (idx[:, None] % n - idx[None, :] % n)
        key = (dr + n) * (2 * n + 1) + (dc + n)
        for h in range(b.shape[0]):
            for k in key.unique():
                vals = b[h][key == k]
                assert torch.allclose(vals, vals[0], atol=1e-7), (
                    board, "rel bias varies within one (drow,dcol) offset")

        ax = EXTRACTORS["attn-axis"](_space(n_planes, n), **kw)
        with torch.no_grad():
            b = ax.attn_bias()
            row_w, col_w = ax.axis_bias[:, 0], ax.axis_bias[:, 1]
        eye = torch.eye(n * n, dtype=torch.bool)
        same_r = ax.row_mask.bool() & ~eye
        same_c = ax.col_mask.bool() & ~eye
        neither = ~(same_r | same_c) & ~eye
        for h in range(b.shape[0]):
            assert torch.allclose(b[h][same_r], row_w[h].expand(
                int(same_r.sum())), atol=1e-7), (board, "same-row not shared")
            assert torch.allclose(b[h][same_c], col_w[h].expand(
                int(same_c.sum())), atol=1e-7), (board, "same-col not shared")
            assert torch.allclose(b[h][neither], torch.zeros(
                int(neither.sum())), atol=1e-7), (board, "off-axis not pinned")
            assert torch.allclose(b[h].diagonal(), (row_w[h] + col_w[h]).expand(
                n * n), atol=1e-7), (board, "diagonal not row+col")
        n_off = len({round(float(v), 7) for v in b[0][~eye].unique()})
        assert n_off <= 3, (board, f"{n_off} distinct off-diagonal values, "
                                   "expected {row, col, 0}")
    print("bias  : rel shared by (drow,dcol); axis = 2 learned values  OK")


def gate_determinism() -> None:
    for board, n_planes, n, _, kw in BOARDS:
        for name in ARMS:
            outs, grads = [], []
            for _ in range(2):
                torch.manual_seed(11)
                ex = EXTRACTORS[name](_space(n_planes, n), **kw)
                obs = _onehot(np.random.default_rng(3), n_planes, n)
                out = ex(obs)
                out.sum().backward()
                outs.append(out.detach().clone())
                grads.append(ex.embed.weight.grad.clone())
            assert torch.equal(*outs), (board, name, "forward not bit-exact")
            assert torch.equal(*grads), (board, name, "grad not bit-exact")
    print("det   : seeded forward/backward bit-identical            OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    gate_forward_backward()
    gate_embed_is_lookup()
    gate_permutation()
    gate_bias_structure()
    gate_determinism()
    print("\ngame2048_attn_gate: ALL GATES PASS")

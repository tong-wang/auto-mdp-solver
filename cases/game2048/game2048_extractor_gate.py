"""Executable gate for the lever-matrix feature extractors (#E14).

Torch-dependent, so it lives OUTSIDE ``game2048_test.py`` — the domain test
file must stay runnable under the torch-free harness ``[dev]`` extra. Gates,
per pool arm x board size (3x3: 10 planes, 4x4: 17 planes):

  1. forward returns (B, features_dim), all finite, and gradients reach every
     trainable tensor
  2. trainable count printed — the pool's ledger row records it, so a lever's
     delta is never silently a capacity delta
  3. embed: the bias-free 1x1 conv IS an embedding table — its output at
     every cell equals column-selection from the weight matrix
  4. conv3d: EXACT exponent-shift equivariance of the trunk away from the
     axis padding (the coded form of domain test 38's licence); the empty
     channel is broadcast constant along the axis, so it commutes with the
     shift by construction

Usage: python game2048_extractor_gate.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from game2048_ppo_train import EXTRACTORS, Conv3dExponentCnn, EmbedBoardCnn

BATCH = 16

# the six pool arms exactly as launched (ESCALATION.md #E14)
ARMS = [
    ("control",  "small",  {}),
    ("embed",    "embed",  {"embed_dim": 8}),
    ("conv3d",   "conv3d", {"channels": (8, 8)}),
    ("rowcol",   "rowcol", {"channels": (128,)}),
    ("k2",       "small",  {"kernel_size": 2}),
    ("capacity", "small",  {"channels": (64, 128), "features_dim": 256}),
]
BOARDS = [("3x3", 10, 3), ("4x4", 17, 4)]


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
    print(f"{'arm':10s} {'board':6s} {'out':12s} {'trainable':>10s}")
    for board, n_planes, n in BOARDS:
        for name, key, kw in ARMS:
            ex = EXTRACTORS[key](_space(n_planes, n), **kw)
            out = ex(_onehot(rng, n_planes, n))
            fdim = kw.get("features_dim", 128)
            assert out.shape == (BATCH, fdim), (name, board, out.shape)
            assert torch.isfinite(out).all(), (name, board, "non-finite out")
            out.sum().backward()
            for tag, t in ex.named_parameters():
                assert t.grad is not None and torch.isfinite(t.grad).all(), \
                    (name, board, tag, "no finite gradient")
            print(f"{name:10s} {board:6s} {str(tuple(out.shape)):12s} "
                  f"{_n_trainable(ex):>10,}")


def gate_embed_is_lookup() -> None:
    rng = np.random.default_rng(1)
    for board, n_planes, n in BOARDS:
        ex = EmbedBoardCnn(_space(n_planes, n), embed_dim=8)
        emb = ex.cnn[0]
        assert isinstance(emb, nn.Conv2d) and emb.bias is None
        obs = _onehot(rng, n_planes, n)
        with torch.no_grad():
            y = emb(obs)                                    # (B, d, n, n)
        w = emb.weight.detach().reshape(8, n_planes)        # (d, K)
        ks = obs.argmax(dim=1)                              # (B, n, n)
        expect = w[:, ks].permute(1, 0, 2, 3)               # (B, d, n, n)
        assert torch.allclose(y, expect, atol=1e-6), (board, "not a lookup")
    print("embed : 1x1 conv == embedding lookup                    OK")


def gate_conv3d_shift_equivariance() -> None:
    rng = np.random.default_rng(2)
    for board, n_planes, n in BOARDS:
        chans = (8, 8)
        ex = Conv3dExponentCnn(_space(n_planes, n), channels=chans)
        trunk = nn.Sequential(*list(ex.cnn.children())[:-1])   # drop Flatten
        depth = len(chans)              # receptive field +-depth along the axis
        obs = _onehot(rng, n_planes, n)
        # the doubling map: tile 2^j -> 2^(j+1) shifts planes 1..K-2 up one,
        # zero-fills plane 1, drops the old top plane; empties (plane 0) fixed
        shifted = obs.clone()
        shifted[:, 1:] = 0.0
        shifted[:, 2:] = obs[:, 1:-1]
        with torch.no_grad():
            y = trunk(ex._volume(obs))
            ys = trunk(ex._volume(shifted))
        ke = n_planes - 1               # exponent-axis length
        lo, hi = depth + 1, ke - depth  # both receptive fields off the padding
        assert lo < hi, (board, "axis too short for an interior check")
        assert torch.allclose(ys[:, :, lo:hi], y[:, :, lo - 1:hi - 1],
                              atol=1e-5), (board, "shift equivariance broken")
    print("conv3d: exact exponent-shift equivariance of the trunk  OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    gate_forward_backward()
    gate_embed_is_lookup()
    gate_conv3d_shift_equivariance()
    print("\ngame2048_extractor_gate: ALL GATES PASS")

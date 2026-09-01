"""game2048 stochastic primitives (seed scheme v2).

One source of randomness per class: where a tile lands (``spawn_cell``,
source 0) and whether it is a 4 (``spawn_value``, source 1). Both are drawn
once per board-changing slide.

**Keyed on spawn_count, not period.** These slots declare a ``keyed`` stage
(``key_exprs: ["spawn_count"]``), so the seed key is

    [spawn_count, source_id, 1, episode_seed, seed_salt]

— the v2 intrinsic template with ``spawn_count`` in the leaf slot where a
per-period source would put ``period``. Spawns are counted in slides that
actually changed the board, so an episode seed yields the same tile sequence
however many illegal slides the agent attempted. That is what makes the gym's
``masked`` and ``free`` action modes comparable on shared seeds.

Because the stages are keyed, ``mdp_ir.runtime.FamilyGenerator`` cannot back
them (``from_parts`` rejects keyed stages — it keys the plain v2 template), so
both generators are hand-written. They still delegate the *draw itself* to
``mdp_ir.runtime.sample_family``, the one family->numpy dispatch the
interpreter also draws through, so the two sides of the twin cannot drift in
how they consume the stream.

The cell draw is a categorical whose **support depends on the state**: the
empty cells of the current board. It is therefore not decision-path
independent, and cannot be — which cells are free is exactly what the agent's
moves determine. Only the seed key is path-independent.

Dependency order: game2048_uncertainty  <-  game2048_scenarios  <-  game2048_mdp
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from mdp_ir.runtime import sample_family

from game2048_board import empty_cells, uniform_probs

SEED_SCHEME = "v2"

_INTRINSIC_BRANCH = 1
_META_BRANCH = 0

# source_id range of this domain's slots (spawn_cell=0, spawn_value=1);
# composition-scoped meta drawers would allocate at or above this
N_SOURCE_IDS = 2


# ---------------------------------------------------------------------------
# Sampling context protocol
# ---------------------------------------------------------------------------


class SamplingContext(Protocol):
    """What a state must expose for ``generator.sample()``.

    ``Game2048State`` satisfies this structurally, so no import of the
    concrete type is needed here and the dependency stays acyclic.
    """

    spawn_count: int
    episode_seed: int
    seed_salt: int
    board: list[int]


def intrinsic_key(source_id: int, ctx: SamplingContext) -> list[int]:
    """v2 intrinsic key for a KEYED stage:
    ``[spawn_count, source_id, 1, episode_seed, seed_salt]``.

    Still the intrinsic branch (word ``1``) — a keyed draw is an intrinsic
    draw — but ``spawn_count`` occupies the leaf slot where a per-period
    source would put ``period``, which is what ``UncertaintyStage.seed_key``
    emits for a stage whose single ``key_exprs`` entry is ``spawn_count``.
    Leaf-first, so the trailing word is always seed_salt (>= 1): SeedSequence
    zero-pads short entropy lists, and a routinely-zero leaf must never sit in
    trailing position.

    Named ``intrinsic_key`` deliberately — the conformance harness requires
    every raw ``SeedSequence`` to be built inside ``intrinsic_key()`` or
    ``meta_key()``, and this is genuinely the former.
    """
    return [
        int(ctx.spawn_count),
        int(source_id),
        _INTRINSIC_BRANCH,
        int(ctx.episode_seed),
        int(ctx.seed_salt),
    ]


def meta_key(substream_id: int, episode_seed: int, seed_salt: int) -> list[int]:
    """v2 meta seed key: [substream_id, 0, episode_seed, seed_salt].

    Unused by the shipped candidates — game2048 has no world latents — and
    kept so a latent-bearing candidate appended to either slot has the branch
    it belongs on already defined.
    """
    return [int(substream_id), _META_BRANCH, int(episode_seed), int(seed_salt)]


def _rng(source_id: int, ctx: SamplingContext) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(intrinsic_key(source_id, ctx)))


# ---------------------------------------------------------------------------
# Where the tile lands
# ---------------------------------------------------------------------------


class SpawnCellGenerator:
    """Abstract: which cell a newly spawned tile occupies."""

    is_discrete: bool = True
    latent: bool = False
    source_id: int

    def realize(self, episode_seed: int, seed_salt: int) -> "SpawnCellGenerator":
        return self

    def sample(self, state: SamplingContext) -> int:
        raise NotImplementedError


class UniformEmptyCell(SpawnCellGenerator):
    """Uniform over the currently empty cells.

    The caller must only sample when an empty cell exists. In the dynamics
    that is guaranteed by the ``move_valid`` guard: a board-changing slide
    either merged (freeing a cell) or slid a tile into one that was already
    free, so at least one cell is empty afterwards.
    """

    def __init__(self, source_id: int = 0) -> None:
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> int:
        return int(sample_family(
            _rng(self.source_id, state),
            "categorical",
            {
                "values": empty_cells(state.board),
                "probabilities": uniform_probs(state.board),
            },
        ))

    def __repr__(self) -> str:
        return f"UniformEmptyCell(source_id={self.source_id})"


# ---------------------------------------------------------------------------
# What the tile is worth
# ---------------------------------------------------------------------------


class SpawnValueGenerator:
    """Abstract: whether a newly spawned tile is a 4 (1) or a 2 (0)."""

    is_discrete: bool = True
    latent: bool = False
    source_id: int

    def realize(self, episode_seed: int, seed_salt: int) -> "SpawnValueGenerator":
        return self

    def sample(self, state: SamplingContext) -> int:
        raise NotImplementedError

    def tile_value(self, state: SamplingContext) -> int:
        """The spawned tile's value, mirroring the dynamics' 2 + 2*four."""
        return 2 + 2 * self.sample(state)


class Bernoulli4(SpawnValueGenerator):
    """The standard 2-or-4 spawn: a 4 with probability ``prob_4``."""

    def __init__(self, prob_4: float = 0.2, source_id: int = 1) -> None:
        assert 0.0 <= prob_4 <= 1.0, "prob_4 must be in [0, 1]."
        self.prob_4 = float(prob_4)
        self.source_id = source_id

    def sample(self, state: SamplingContext) -> int:
        return int(sample_family(
            _rng(self.source_id, state), "bernoulli", {"p": self.prob_4}
        ))

    def mean(self) -> float:
        """Expected tile value — what gym wrappers size tile scales from."""
        return 2.0 + 2.0 * self.prob_4

    def __repr__(self) -> str:
        return f"Bernoulli4(prob_4={self.prob_4}, source_id={self.source_id})"

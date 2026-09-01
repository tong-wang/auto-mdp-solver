"""game2048 scenario configuration and the named registry.

A scenario is one concrete world: a board size, how often a spawned tile is a
4, and the move cap. Look one up by name:

    from game2048_scenarios import SCENARIOS
    scenario = SCENARIOS["3x3_20"]

Naming follows the pre-IR domain: ``{n}x{n}_{prob_4 x 100}``. The registry
mirrors the IR catalog's instances one-for-one (``3x3_20`` is the IR's base
composition), which is what lets the differential sweep the whole set.

Dependency order: game2048_uncertainty  <-  game2048_scenarios  <-  game2048_mdp
"""

from __future__ import annotations

from dataclasses import dataclass, field

from game2048_uncertainty import Bernoulli4, SpawnCellGenerator, SpawnValueGenerator, UniformEmptyCell


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Game2048Scenario:
    """One game2048 world.

    grid_size:        board dimension n; the board is n x n
    n_cells:          grid_size**2, the flat board length (kept explicit to
                      mirror the IR constant of the same name)
    spawn_cell:       generator for where a new tile lands
    spawn_value:      generator for whether it is a 4
    T_cap:            move cap = max(200, 2^(n_cells+2)) — always >= 2x the
                      masked-play theoretical limit (moves <= max terminal
                      board sum / 2 <= 2^(n_cells+1)), so it can never distort
                      masked play; it exists to bound the FREE action mode's
                      no-op stalling (IR-CHANGELOG F2, 2026-08-10; before
                      that, 3x3's cap sat BELOW its 1023-move natural bound
                      and 4x4_0 was amputated to ~6% of its world).
                      The episode truncates here if game over has
                      not arrived first
    invalid_penalty:  points the gym's score_penalty reward mode deducts per
                      illegal slide. Shaping only — never part of the scored
                      objective.
    seed_salt:        reproducibility knob; must be >= 1 under seed scheme v2
    """

    grid_size: int
    spawn_cell: SpawnCellGenerator
    spawn_value: SpawnValueGenerator
    n_cells: int = 0
    T_cap: int = 1000
    invalid_penalty: float = 4.0
    seed_salt: int = 421
    scenario_name: str | None = None
    desc: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        assert self.grid_size >= 2, "grid_size must be >= 2."
        if not self.n_cells:
            self.n_cells = self.grid_size ** 2
        assert self.n_cells == self.grid_size ** 2, (
            f"n_cells ({self.n_cells}) must equal grid_size^2 "
            f"({self.grid_size ** 2})."
        )
        assert self.T_cap >= 1, "T_cap must be >= 1."
        assert self.seed_salt >= 1, "seed_salt must be >= 1 under seed scheme v2."
        ids = [self.spawn_cell.source_id, self.spawn_value.source_id]
        assert len(set(ids)) == len(ids), (
            f"generators composed into one scenario need distinct source_ids, got {ids}"
        )

    @property
    def has_latents(self) -> bool:
        """No shipped candidate carries a world latent; kept so the adapter's
        ``source(episode_seed) if callable(source)`` path stays uniform."""
        return bool(self.spawn_cell.latent or self.spawn_value.latent)


def _scenario(name: str, grid_size: int, prob_4: float, T_cap: int, desc: str) -> Game2048Scenario:
    return Game2048Scenario(
        grid_size=grid_size,
        spawn_cell=UniformEmptyCell(source_id=0),
        spawn_value=Bernoulli4(prob_4=prob_4, source_id=1),
        T_cap=T_cap,
        scenario_name=name,
        desc=desc,
    )


# ---------------------------------------------------------------------------
# Named scenarios — one per IR catalog instance
# ---------------------------------------------------------------------------

scenario_2x2_0 = _scenario(
    "2x2_0", 2, 0.0, 200,
    "2x2 board, only 2-tiles spawn: tile values deterministic, the smallest world")
scenario_2x2_20 = _scenario(
    "2x2_20", 2, 0.2, 200,
    "2x2 board, 20% chance of a 4-tile")
scenario_3x3_0 = _scenario(
    "3x3_0", 3, 0.0, 2048,
    "3x3 board, only 2-tiles spawn")
scenario_3x3_20 = _scenario(
    "3x3_20", 3, 0.2, 2048,
    "3x3 board, 20% chance of a 4-tile - the IR base composition and the training target")
scenario_4x4_0 = _scenario(
    "4x4_0", 4, 0.0, 262144,
    "4x4 board (the standard game), only 2-tiles spawn")
scenario_4x4_20 = _scenario(
    "4x4_20", 4, 0.2, 262144,
    "4x4 board (the standard game), 20% chance of a 4-tile - the canonical 2048")
scenario_5x5_0 = _scenario(
    "5x5_0", 5, 0.0, 134217728,
    "5x5 board, only 2-tiles spawn: more room to manoeuvre")
scenario_5x5_20 = _scenario(
    "5x5_20", 5, 0.2, 134217728,
    "5x5 board, 20% chance of a 4-tile")

SCENARIOS: dict[str, Game2048Scenario] = {
    "2x2_0": scenario_2x2_0,
    "2x2_20": scenario_2x2_20,
    "3x3_0": scenario_3x3_0,
    "3x3_20": scenario_3x3_20,
    "4x4_0": scenario_4x4_0,
    "4x4_20": scenario_4x4_20,
    "5x5_0": scenario_5x5_0,
    "5x5_20": scenario_5x5_20,
}


if __name__ == "__main__":
    for name, s in SCENARIOS.items():
        print(f"{name:8s} n={s.grid_size} cells={s.n_cells:2d} "
              f"prob_4={s.spawn_value.prob_4:.1f} T_cap={s.T_cap:5d}  {s.desc}")

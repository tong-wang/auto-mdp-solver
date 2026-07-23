"""Design-layer grids for the FNV domain (spec §5.6).

An FnvScenarioGrid is a finite set of complete scenario sources — the
generality target of a generalist policy. It is deliberately NOT callable (a
grid must never pass where a ScenarioSource belongs): training samples over the
cells via a derived sampler (as_sampler()); evaluation enumerates the cells,
one leaderboard row per cell.

Seed scheme v2 (spec §6.3): the derived sampler's per-episode cell draw keys
the meta branch via meta_key() — [substream_id, 0, episode_seed, seed_salt].

Layering (spec §1.1, §5.6): this module imports from fnv_scenarios; nothing in
fnv_uncertainty, fnv_scenarios, fnv_mdp, or fnv_gym may import it. Only
training / eval / tuning drivers do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools

import numpy as np

from fnv_scenarios import FnvScenario

SEED_SCHEME = "v2"

_META_BRANCH = 0


def meta_key(substream_id: int, episode_seed: int, seed_salt: int) -> list[int]:
    """v2 meta seed key: [substream_id, 0, episode_seed, seed_salt] (leaf-first)."""
    return [substream_id, _META_BRANCH, episode_seed, seed_salt]


def _cell_id(overrides: dict[str, float]) -> str:
    """Canonical cell id: pure function of axis names/values in declaration order."""
    return ",".join(f"{k}={v:g}" for k, v in overrides.items())


# ---------------------------------------------------------------------------
# Derived training sampler
# ---------------------------------------------------------------------------


@dataclass
class FnvGridTrainingSampler:
    """Ordinary sampler over a grid's cells, derived via grid.as_sampler().

    The gym receives this, never the grid. Pure function
    `episode_seed -> FnvScenario` (spec §5.2, §5.6): the cell index is drawn on
    the v2 meta key; uniform over cells unless per-cell weights are given.
    Defined at module level so it pickles into SubprocVecEnv workers.
    """

    grid: "FnvScenarioGrid"
    probs: np.ndarray | None = None          # None = uniform over cells
    substream_id: int = 0
    seed_salt: int = field(default=6409, repr=False)
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        if self.probs is not None:
            assert len(self.probs) == len(self.grid), "one weight per cell."
            assert np.all(self.probs > 0), "weights must be positive."
            self.probs = np.asarray(self.probs, float) / np.sum(self.probs)

    # -- family-level attributes: delegate to the grid ----------------------
    def __getattr__(self, name):
        return getattr(self.grid, name)

    def __call__(self, episode_seed: int) -> FnvScenario:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                meta_key(self.substream_id, episode_seed, self.seed_salt)
            )
        )
        k = int(rng.choice(len(self.grid), p=self.probs))
        return self.grid[k]


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


@dataclass
class FnvScenarioGrid:
    """A finite set of complete scenario sources; see module docstring."""

    # (cell_id, source) in canonical order — row-major over axes in
    # declaration order (from_axes), or definition order (from_sources)
    cells: list[tuple[str, FnvScenario]]

    # axis name -> value tuple, declaration-ordered; None for from_sources
    axes: dict[str, tuple] | None = None

    grid_name: str | None = None
    desc: str = ""
    seed_salt: int = field(default=6409, repr=False)

    def __post_init__(self) -> None:
        ids = [cid for cid, _ in self.cells]
        assert len(self.cells) >= 1, "a grid needs at least one cell."
        assert len(ids) == len(set(ids)), "cell ids must be unique."
        # family agreement: one gym must serve every cell (mirroring rule, §5.2)
        for attr in ("N", "mmfe_mode"):
            vals = {getattr(sc, attr) for _, sc in self.cells}
            assert len(vals) == 1, f"cells disagree on {attr}: {vals}."
        self._by_id = dict(self.cells)

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_axes(
        cls,
        axes: dict[str, tuple],
        grid_name: str | None = None,
        desc: str = "",
        seed_salt: int = 6409,
        **base_kwargs,
    ) -> "FnvScenarioGrid":
        """Row-major cross product of constant overrides, axes in declaration order."""
        names = list(axes)
        cells = []
        for combo in itertools.product(*axes.values()):
            overrides = dict(zip(names, combo))
            cells.append((
                _cell_id(overrides),
                FnvScenario(
                    **overrides, **base_kwargs,
                    seed_salt=seed_salt, scenario_name=_cell_id(overrides),
                ),
            ))
        return cls(
            cells=cells, axes={k: tuple(v) for k, v in axes.items()},
            grid_name=grid_name, desc=desc, seed_salt=seed_salt,
        )

    @classmethod
    def from_sources(
        cls,
        sources: list[FnvScenario],
        grid_name: str | None = None,
        desc: str = "",
        seed_salt: int = 6409,
    ) -> "FnvScenarioGrid":
        """Explicit list — a grid with one categorical axis; definition order
        preserved; cell ids default to the sources' scenario_names."""
        for s in sources:
            assert s.scenario_name, "from_sources cells need scenario_name as cell id."
        return cls(
            cells=[(s.scenario_name, s) for s in sources],
            axes=None, grid_name=grid_name, desc=desc, seed_salt=seed_salt,
        )

    # -- family-level attributes (same names as FnvScenario fields) ---------

    @property
    def N(self) -> int:
        return self.cells[0][1].N

    @property
    def r(self) -> float:
        return self.cells[0][1].r

    @property
    def c1(self) -> float:
        return self.cells[0][1].c1

    @property
    def mu(self) -> float:
        return self.cells[0][1].mu

    @property
    def mmfe_mode(self) -> str:
        return self.cells[0][1].mmfe_mode

    @property
    def stdev(self) -> float:
        """Bound-relevant aggregate (not an agreed family value): the widest
        cell's stdev, so gym action/observation scales cover every cell."""
        return max(sc.stdev for _, sc in self.cells)

    # -- enumeration is data access, not behavior ----------------------------

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self):
        return iter(self.cells)

    def __getitem__(self, key: str | int) -> FnvScenario:
        if isinstance(key, int):
            return self.cells[key][1]
        return self._by_id[key]

    # -- the only verb: sampling is derived behavior -------------------------

    def as_sampler(
        self, weights=None, substream_id: int = 0,
    ) -> FnvGridTrainingSampler:
        return FnvGridTrainingSampler(
            grid=self,
            probs=None if weights is None else np.asarray(weights, float),
            substream_id=substream_id,
            seed_salt=self.seed_salt,
            scenario_name=self.grid_name,
            desc=self.desc,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

grid_ammfe = FnvScenarioGrid.from_axes(
    axes=dict(
        stdev=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
        T=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        lamb=(0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2),
    ),
    N=3, r=2.0, c1=1.0, mu=1.0,
    mmfe_mode="additive",
    grid_name="FNV-aMMFE",
    desc="additive MMFE; generalist over stdev x T x lamb (6x9x10 = 540 cells)",
)

grid_mmmfe = FnvScenarioGrid.from_axes(
    axes=dict(
        stdev=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        T=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        lamb=(0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2),
    ),
    N=3, r=2.0, c1=1.0, mu=1.0,
    mmfe_mode="multiplicative",
    grid_name="FNV-mMMFE",
    desc="multiplicative MMFE; generalist over stdev x T x lamb (6x9x10 = 540 cells)",
)

GRIDS: dict[str, FnvScenarioGrid] = {
    "FNV-aMMFE": grid_ammfe,
    "FNV-mMMFE": grid_mmmfe,
}


if __name__ == "__main__":
    for name, g in GRIDS.items():
        first, last = g.cells[0][0], g.cells[-1][0]
        print(f"{name:10s}  {len(g)} cells  [{first} ... {last}]  N={g.N}  mode={g.mmfe_mode}")

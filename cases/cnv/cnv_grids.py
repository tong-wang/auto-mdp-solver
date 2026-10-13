"""Design-layer grids for the cnv domain (spec §5.6).

A ``CnvScenarioGrid`` is a finite set of complete scenario sources — the
generality target of a **generalist** policy. It is deliberately NOT
callable (a grid must never pass where a ScenarioSource belongs): training
samples over the cells via ``as_sampler()``; evaluation enumerates them, one
leaderboard row per cell, all cells on one shared seed block (§9.6).

Layering (§1.1, §5.6): imports from cnv_scenarios; nothing in
cnv_uncertainty, _scenarios, _mdp or _gym may import it. Only training /
eval drivers do.

The two grids are the IR's declared ``grids`` (schema root), the paper's §5
study:

  * ``paper510`` — unit_cost (17: newsvendor ratio 0.1..0.9) x prior_rate
    (6: 1/beta in 0.5..16) x prior_mean (5: 10..50), crossed on ``base``
    (N = 4). Every axis is tier-3 (reward or distribution parameters): the
    action space and the observation width are the same in every cell, so
    one policy serves the grid without padding. The primary generalist target.
  * ``fig3`` — the unit_cost sweep alone at the Fig. 3(a)/4(a) prior
    (alpha = 10/16, beta = 1/16), crossed on ``fig3_c10``.

Cell ids follow the IR ``ScenarioGrid.cells()`` spelling
(``"unit_cost=0.2,prior_rate=2,prior_mean=10"``: axis order as declared,
values in ``%g`` form) so leaderboard rows join the schema's enumeration 1:1.
Every cell shares the demand slot's latent stream, so the same seed draws the
same lambda in every cell whose prior agrees (common random numbers).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from cnv_scenarios import (
    CnvScenario,
    CnvScenarioSource,
    make_source,
    source_base,
    source_fig3_c10,
)
from cnv_uncertainty import N_SOURCE_IDS, meta_key

SEED_SCHEME = "v2"

# the grid sampler's meta substream: composition-scoped, so at or above the
# sources' latent range (demand=0, epochs=1)
_GRID_SUBSTREAM_ID = N_SOURCE_IDS


@dataclass
class CnvGridTrainingSampler:
    """Ordinary sampler over a grid's cells, derived via ``grid.as_sampler()``.

    The gym receives this, never the grid. A pure function
    ``episode_seed -> CnvScenario`` (§5.2, §5.6): the cell index is drawn on
    the v2 meta key, then that cell's source realizes the SAME episode seed —
    so the drawn cell's episode is bit-identical to the cell run standalone.
    Module-level so it pickles into SubprocVecEnv workers.
    """

    grid: "CnvScenarioGrid"
    probs: np.ndarray | None = None          # None = uniform over cells
    substream_id: int = _GRID_SUBSTREAM_ID
    seed_salt: int = field(default=4729, repr=False)
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        assert self.substream_id >= N_SOURCE_IDS, (
            "a grid sampler is composition-scoped: substream_id >= N_SOURCE_IDS")
        if self.probs is not None:
            assert len(self.probs) == len(self.grid), "one weight per cell."
            assert np.all(np.asarray(self.probs) > 0), "weights must be positive."
            self.probs = np.asarray(self.probs, float) / np.sum(self.probs)

    # family-level attributes delegate to the grid (mirroring rule, §5.2)
    def __getattr__(self, name):
        return getattr(self.grid, name)

    def __call__(self, episode_seed: int) -> CnvScenario:
        rng = np.random.default_rng(np.random.SeedSequence(
            meta_key(self.substream_id, episode_seed, self.seed_salt)))
        k = int(rng.choice(len(self.grid), p=self.probs))
        source = self.grid[k]
        return source(episode_seed)

    def __repr__(self) -> str:
        return (f"CnvGridTrainingSampler({self.grid.grid_name!r}, "
                f"{len(self.grid)} cells, "
                f"{'uniform' if self.probs is None else 'weighted'})")


@dataclass
class CnvScenarioGrid:
    """A finite set of complete scenario sources; see the module docstring."""

    cells: list[tuple[str, CnvScenarioSource]]
    axes: dict[str, tuple] | None = None
    grid_name: str | None = None
    desc: str = ""
    seed_salt: int = field(default=4729, repr=False)

    def __post_init__(self) -> None:
        ids = [cid for cid, _ in self.cells]
        assert len(self.cells) >= 1, "a grid needs at least one cell."
        assert len(ids) == len(set(ids)), "cell ids must be unique."
        # family agreement: one gym must serve every cell (§5.2 mirroring).
        # Costs and priors are the axes; everything that sizes a space agrees.
        for attr in ("horizon", "price", "arrivals_buffer", "seed_salt"):
            vals = {getattr(src, attr) for _, src in self.cells}
            assert len(vals) == 1, f"cells disagree on {attr}: {vals}."
        self._by_id = dict(self.cells)

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_axes(cls, base: CnvScenarioSource, axes: dict[str, tuple],
                  grid_name: str, desc: str = "") -> "CnvScenarioGrid":
        """Row-major cross product of constant overrides on ``base``, axes in
        declaration order — the IR's ``ScenarioGrid.cells()`` enumeration."""
        names = list(axes)
        cells = []
        for values in product(*(axes[n] for n in names)):
            overrides = dict(zip(names, values))
            cid = ",".join(f"{n}={v:g}" for n, v in overrides.items())
            cells.append((cid, make_source(
                f"{grid_name}[{cid}]",
                f"grid cell of {grid_name}: " + ", ".join(
                    f"{n}={v:g}" for n, v in overrides.items()),
                horizon=base.horizon,
                price=overrides.get("price", base.price),
                unit_cost=overrides.get("unit_cost", base.unit_cost),
                prior_rate=overrides.get("prior_rate", base.prior_rate),
                prior_mean=overrides.get("prior_mean", base.prior_mean),
                arrivals_buffer=base.arrivals_buffer,
                seed_salt=base.seed_salt,
            )))
        return cls(cells=cells, axes={n: tuple(axes[n]) for n in names},
                   grid_name=grid_name, desc=desc, seed_salt=base.seed_salt)

    # -- family-level attributes --------------------------------------------

    @property
    def horizon(self) -> int:
        return self.cells[0][1].horizon

    @property
    def price(self) -> float:
        return self.cells[0][1].price

    @property
    def arrivals_buffer(self) -> int:
        return self.cells[0][1].arrivals_buffer

    @property
    def demand(self):
        """Bound-relevant envelope: the widest cell's demand generator."""
        return max((src.demand for _, src in self.cells), key=lambda g: g.max())

    @property
    def epochs(self):
        return self.cells[0][1].epochs

    @property
    def unit_cost_max(self) -> float:
        return max(src.unit_cost for _, src in self.cells)

    @property
    def prior_rate_max(self) -> float:
        return max(src.prior_rate for _, src in self.cells)

    @property
    def prior_mean_max(self) -> float:
        return max(src.prior_mean for _, src in self.cells)

    # -- enumeration is data access, not behavior ---------------------------

    def __repr__(self) -> str:
        return (f"CnvScenarioGrid({self.grid_name!r}, "
                f"{len(self.cells)} cells, axes={self.axes})")

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self):
        return iter(self.cells)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.cells[key][1]
        return self._by_id[key]

    # -- the only verb: sampling is derived behavior ------------------------

    def as_sampler(self, weights=None) -> CnvGridTrainingSampler:
        return CnvGridTrainingSampler(
            grid=self,
            probs=None if weights is None else np.asarray(weights, float),
            seed_salt=self.seed_salt,
            scenario_name=self.grid_name, desc=self.desc)


def cell_slug(cell_id: str) -> str:
    """Filesystem-safe cell id: ``"unit_cost=0.2,prior_rate=2,prior_mean=10"``
    -> ``"c0.2_b2_m10"``. Deterministic and invertible against the grid's own
    enumeration; used to suffix per-cell benchmark/eval artifacts."""
    return (cell_id.replace("unit_cost=", "c").replace("prior_rate=", "b")
                   .replace("prior_mean=", "m").replace(",", "_"))


# ---------------------------------------------------------------------------
# Registry — the IR's `grids` node, materialized
# ---------------------------------------------------------------------------

_UNIT_COSTS = tuple(round(0.2 + 0.1 * i, 1) for i in range(17))   # ratio 0.9 .. 0.1
_PRIOR_RATES = (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625)                # 1/beta = 0.5 .. 16
_PRIOR_MEANS = (10.0, 20.0, 30.0, 40.0, 50.0)

paper510 = CnvScenarioGrid.from_axes(
    base=source_base,
    axes={"unit_cost": _UNIT_COSTS, "prior_rate": _PRIOR_RATES, "prior_mean": _PRIOR_MEANS},
    grid_name="paper510",
    desc="the paper's §5.1 study (IR grids[paper510]): 17 unit costs x 6 prior "
         "scales x 5 prior means = 510 cells at N = 4; the primary generalist "
         "target, one policy per observation type, evaluated per cell against "
         "the cell's own myopic and DP bars.")

fig3 = CnvScenarioGrid.from_axes(
    base=source_fig3_c10,
    axes={"unit_cost": _UNIT_COSTS},
    grid_name="fig3",
    desc="the paper's Figures 3(a)/4(a) (IR grids[fig3]): the unit-cost sweep "
         "at alpha = 10/16, beta = 1/16, 17 cells at N = 4.")

GRIDS: dict[str, CnvScenarioGrid] = {
    "paper510": paper510,
    "fig3": fig3,
}


if __name__ == "__main__":
    for name, grid in GRIDS.items():
        print(f"{name}: {grid}")
        sampler = grid.as_sampler()
        counts: dict[str, int] = {}
        for seed in range(2000):
            sc = sampler(seed)
            counts[sc.scenario_name] = counts.get(sc.scenario_name, 0) + 1
        assert sampler(7) == sampler(7), "sampler must be pure"
        print(f"  cells drawn: {len(counts)}/{len(grid)}, "
              f"min/max draws {min(counts.values())}/{max(counts.values())}")
        for cid, src in list(grid)[:3]:
            print(f"  {cid:45s} c={src.unit_cost:g} prior=Gamma({src.prior_shape:g}, "
                  f"{src.prior_rate:g}) demand.max={src.demand.max():.1f}")

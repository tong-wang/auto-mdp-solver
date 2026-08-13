"""Design-layer grids for the MAB domain (spec §5.6).

A MabScenarioGrid is a finite set of complete scenario sources — the generality
target of a **generalist** policy. It is deliberately NOT callable (a grid must
never pass where a ScenarioSource belongs): training samples over the cells via
`as_sampler()`; evaluation enumerates them, one leaderboard row per cell, all
cells on one shared seed block (§9.6).

Contrast with the #E32/#E35 cells, which are `mdp.scenario.instances`: those
are SPECIALISTS, one policy trained per cell. The distinction is the training
target, not the values (spec §5.5) — and it is why the instances came first and
this file only appears now that a policy is meant to span cells.

Layering (§1.1, §5.6): imports from mab_scenarios; nothing in mab_uncertainty,
mab_scenarios, mab_mdp or mab_gym may import it. Only training / eval drivers do.

#E36 grid `T-loguniform`: horizon varying, K fixed.

  * **Log-spaced**, not linear. The quantity a generalist must represent is
    linear in ln T — #E32 fitted c* = 1.204 + 0.286*ln n, and n = T/K at fixed
    K — so log spacing gives uniform statistical resolution in the coordinate
    that actually varies. It is also ~40% cheaper (E[T] 3170 vs 5250 on the
    same interval), and uniform weights over log-spaced cells ARE log-uniform
    in T, so the §5.5 weights knob stays at its default.
  * **Capped at T=10000**, below the T ~ 14,000 crossover #E34 measured, so the
    generalist lives entirely inside the regime where the rule beats thompson.
  * **Dense (40 cells), not a handful.** With a few coarse cells a policy can
    memorise regimes instead of learning a function of T, and the readback
    could not tell those apart — which would defeat the round's whole purpose.
    §5.5 pins this: continuous generality ranges are discretized at grid
    construction time.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from mab_scenarios import MabScenario, MabScenarioSource
from mab_uncertainty import LatentGaussianPayout

SEED_SCHEME = "v2"
_META_BRANCH = 0


def meta_key(substream_id: int, episode_seed: int, seed_salt: int) -> list[int]:
    """v2 meta seed key: [substream_id, 0, episode_seed, seed_salt] (leaf-first)."""
    return [substream_id, _META_BRANCH, episode_seed, seed_salt]


@dataclass
class MabGridTrainingSampler:
    """Ordinary sampler over a grid's cells, derived via `grid.as_sampler()`.

    The gym receives this, never the grid. A pure function
    `episode_seed -> MabScenario` (§5.2, §5.6): the cell index is drawn on the
    v2 meta key, then that cell's own source realizes the episode's arm means.
    Module-level so it pickles into SubprocVecEnv workers.
    """

    grid: "MabScenarioGrid"
    probs: np.ndarray | None = None          # None = uniform over cells
    substream_id: int = 3                    # distinct from the payout slot (0)
    seed_salt: int = field(default=4729, repr=False)
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        if self.probs is not None:
            assert len(self.probs) == len(self.grid), "one weight per cell."
            assert np.all(self.probs > 0), "weights must be positive."
            self.probs = np.asarray(self.probs, float) / np.sum(self.probs)

    # family-level attributes delegate to the grid (mirroring rule, §5.2)
    def __getattr__(self, name):
        return getattr(self.grid, name)

    def __repr__(self) -> str:
        w = "uniform" if self.probs is None else "weighted"
        return f"MabGridTrainingSampler({self.grid!r}, {w})"

    def __call__(self, episode_seed: int) -> MabScenario:
        rng = np.random.default_rng(np.random.SeedSequence(
            meta_key(self.substream_id, episode_seed, self.seed_salt)))
        k = int(rng.choice(len(self.grid), p=self.probs))
        source = self.grid[k]
        return source(episode_seed) if callable(source) else source


@dataclass
class MabScenarioGrid:
    """A finite set of complete scenario sources; see the module docstring."""

    cells: list[tuple[str, MabScenarioSource]]
    axes: dict[str, tuple] | None = None
    grid_name: str | None = None
    desc: str = ""
    seed_salt: int = field(default=4729, repr=False)

    def __post_init__(self) -> None:
        ids = [cid for cid, _ in self.cells]
        assert len(self.cells) >= 1, "a grid needs at least one cell."
        assert len(ids) == len(set(ids)), "cell ids must be unique."
        # family agreement: one gym must serve every cell (§5.2 mirroring).
        # `horizon` is deliberately NOT asserted — it is the axis, and is
        # exposed below as a bound-relevant aggregate instead (the FNV `stdev`
        # pattern), so the observation space is sized for the widest cell.
        for attr in ("n_arms",):
            vals = {getattr(sc, attr) for _, sc in self.cells}
            assert len(vals) == 1, f"cells disagree on {attr}: {vals}."
        self._by_id = dict(self.cells)

    # -- constructors --------------------------------------------------------

    @classmethod
    def log_horizon(cls, n_arms: int, t_lo: int, t_hi: int, n_cells: int,
                    grid_name: str, desc: str = "",
                    seed_salt: int = 4729) -> "MabScenarioGrid":
        """Log-spaced horizons on [t_lo, t_hi] at fixed K — the #E36 ladder."""
        ts = sorted({int(round(t)) for t in
                     np.geomspace(t_lo, t_hi, n_cells)})
        cells = []
        for T in ts:
            cells.append((f"T={T}", MabScenarioSource(MabScenario(
                scenario_name=f"gauss_K{n_arms}_T{T}",
                desc=f"grid cell: {n_arms} arms, T={T}",
                horizon=T, seed_salt=seed_salt,
                payout=LatentGaussianPayout(n_arms=n_arms, prior_mean=0.0,
                                            prior_sd=1.0, sigma=1.0)))))
        return cls(cells=cells, axes={"horizon_T": tuple(ts)},
                   grid_name=grid_name, desc=desc, seed_salt=seed_salt)

    # -- family-level attributes --------------------------------------------

    @property
    def n_arms(self) -> int:
        return self.cells[0][1].n_arms

    @property
    def horizon(self) -> int:
        """Bound-relevant aggregate, NOT an agreed family value: the widest
        cell, so the gym's observation space covers every cell. The extractor
        also uses it as the constant time scale (#E36)."""
        return max(sc.horizon for _, sc in self.cells)

    @property
    def payout(self):
        return self.cells[0][1].payout

    def min_payout(self) -> float:
        return min(sc.min_payout() for _, sc in self.cells)

    def max_payout(self) -> float:
        return max(sc.max_payout() for _, sc in self.cells)

    # -- enumeration is data access, not behavior ---------------------------

    def __repr__(self) -> str:
        # compact: the default dataclass repr dumps all 40 cells
        return (f"MabScenarioGrid({self.grid_name!r}, {len(self.cells)} cells, "
                f"K={self.n_arms}, T={self.cells[0][1].horizon}"
                f"..{self.horizon})")

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self):
        return iter(self.cells)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.cells[key][1]
        return self._by_id[key]

    # -- the only verb: sampling is derived behavior ------------------------

    def as_sampler(self, weights=None,
                   substream_id: int = 3) -> MabGridTrainingSampler:
        return MabGridTrainingSampler(
            grid=self,
            probs=None if weights is None else np.asarray(weights, float),
            substream_id=substream_id, seed_salt=self.seed_salt,
            scenario_name=self.grid_name, desc=self.desc)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

grid_t_loguniform = MabScenarioGrid.log_horizon(
    n_arms=10, t_lo=500, t_hi=10000, n_cells=40,
    grid_name="gauss_K10_Tlog",
    desc="#E36 generalist target: K=10, T log-uniform on [500, 10000] over 40 "
         "cells. Log-spaced because c* is linear in ln n; capped below #E34's "
         "T ~ 14,000 crossover; dense so the policy must interpolate in T "
         "rather than memorise regimes.")

GRIDS: dict[str, MabScenarioGrid] = {
    "gauss_K10_Tlog": grid_t_loguniform,
}


if __name__ == "__main__":
    for name, g in GRIDS.items():
        first, last = g.cells[0][0], g.cells[-1][0]
        print(f"{name}: {len(g)} cells [{first} ... {last}]  "
              f"K={g.n_arms}  T_max={g.horizon}")
        s = g.as_sampler()
        Ts = [s(seed).horizon for seed in range(4000)]
        print(f"  sampler: mean T = {np.mean(Ts):.0f} "
              f"(log-uniform predicts {(10000 - 500) / np.log(20):.0f}), "
              f"min {min(Ts)}, max {max(Ts)}")
        assert not callable(g), "a grid must never be callable"
        print(f"  purity: sampler(7) twice -> "
              f"{s(7).horizon == s(7).horizon}")

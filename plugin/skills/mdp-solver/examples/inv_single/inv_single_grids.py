"""Design-layer grids for the inv_single domain (spec §5.6).

An ``InvSingleScenarioGrid`` is a finite set of complete scenarios — the
generality target of a **generalist** policy (RQ-6). It is deliberately NOT
callable (a grid must never pass where a ScenarioSource belongs): training
samples over the cells via ``as_sampler()``; evaluation enumerates them, one
leaderboard row per cell, all cells on one shared seed block (§9.6).

Layering (§1.1, §5.6): imports from inv_single_scenarios; nothing in
inv_single_uncertainty, _scenarios, _mdp or _gym may import it. Only
training / eval drivers do.

``grid16`` — one of the IR's declared grids (schema root ``grids``), crossed on the
``lt`` instance:

  * axes: b ∈ {1, 4, 9, 19} (critical fractile 0.5/0.8/0.9/0.95),
          K ∈ {0, 20} (base-stock vs (s,S) regime),
          leadtime ∈ {0, 2} (deterministic).
  * every cell is backlog + deterministic lead time, so the ``dp``
    benchmark's ``exact`` role holds at each cell.
  * **obs-dim padding (§5.6 tier-2 obs-dim axis):** one policy must serve
    LT=0 and LT=2 cells, so every cell's pipeline is sized to the family
    maximum (len 3). LT=0 cells carry ``DiscreteLeadtime([0, 2], [1, 0])`` —
    ``max()`` = 2 sizes the pipeline, the zero-probability tail is never
    drawn, and the extra slots hold 0 forever, so the dynamics are exactly
    the unpadded cell's with two constant-zero observation entries. A padded
    cell's *effective* lead time is therefore ``leadtime.mean()``, which a
    per-cell DP must read instead of ``max()``.
  * cell ids follow the IR ``ScenarioGrid.cells()`` spelling
    (``"b=1,K=0,leadtime_value=0"``) so leaderboard rows join the schema's
    enumeration 1:1.
  * three cells coincide with specialist targets: (b=9,K=0,LT=0) = simple,
    (b=9,K=20,LT=0) = simple_k, (b=9,K=0,LT=2) = lt — same economics; the
    grid cell differs from the first two only by the padded pipeline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from fractions import Fraction as _Fraction
from itertools import product

import numpy as np

from inv_single_scenarios import InvSingleScenario, scenario_lt, scenario_slt
from inv_single_uncertainty import (
    DeterministicLeadtime,
    DiscreteLeadtime,
    meta_key,
)

SEED_SCHEME = "v2"

# the grid sampler's meta substream: distinct from the demand slot (0), the
# leadtime slot (1) and the mix_demand mixture sampler (2)
_GRID_SUBSTREAM_ID = 3


@dataclass
class InvSingleGridTrainingSampler:
    """Ordinary sampler over a grid's cells, derived via ``grid.as_sampler()``.

    The gym receives this, never the grid. A pure function
    ``episode_seed -> InvSingleScenario`` (§5.2, §5.6): the cell index is
    drawn on the v2 meta key, then that cell's (concrete) scenario is
    returned. Module-level so it pickles into SubprocVecEnv workers.
    """

    grid: "InvSingleScenarioGrid"
    probs: np.ndarray | None = None          # None = uniform over cells
    substream_id: int = _GRID_SUBSTREAM_ID
    seed_salt: int = field(default=1, repr=False)
    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        if self.probs is not None:
            assert len(self.probs) == len(self.grid), "one weight per cell."
            assert np.all(np.asarray(self.probs) > 0), "weights must be positive."
            self.probs = np.asarray(self.probs, float) / np.sum(self.probs)

    # family-level attributes delegate to the grid (mirroring rule, §5.2)
    def __getattr__(self, name):
        return getattr(self.grid, name)

    def __call__(self, episode_seed: int) -> InvSingleScenario:
        rng = np.random.default_rng(np.random.SeedSequence(
            meta_key(self.substream_id, episode_seed, self.seed_salt)))
        k = int(rng.choice(len(self.grid), p=self.probs))
        return self.grid[k]

    def __repr__(self) -> str:
        return (f"InvSingleGridTrainingSampler({self.grid.grid_name!r}, "
                f"{len(self.grid)} cells, "
                f"{'uniform' if self.probs is None else 'weighted'})")


@dataclass
class InvSingleScenarioGrid:
    """A finite set of complete scenarios; see the module docstring."""

    cells: list[tuple[str, InvSingleScenario]]
    axes: dict[str, tuple] | None = None
    grid_name: str | None = None
    desc: str = ""
    seed_salt: int = field(default=1, repr=False)

    def __post_init__(self) -> None:
        ids = [cid for cid, _ in self.cells]
        assert len(self.cells) >= 1, "a grid needs at least one cell."
        assert len(ids) == len(set(ids)), "cell ids must be unique."
        # family agreement: one gym must serve every cell (§5.2 mirroring).
        # Costs are the axes; everything that sizes a space must agree.
        for attr in ("horizon", "stockout_mode", "event_sequence"):
            vals = {getattr(sc, attr) for _, sc in self.cells}
            assert len(vals) == 1, f"cells disagree on {attr}: {vals}."
        pipe = {sc.leadtime.max() for _, sc in self.cells}
        assert len(pipe) == 1, (
            f"cells disagree on pipeline length (leadtime.max()): {pipe} — "
            f"pad short-leadtime cells to the family maximum (§5.6).")
        dmax = {sc.demand.max() for _, sc in self.cells}
        assert len(dmax) == 1, f"cells disagree on demand.max(): {dmax}."
        self._by_id = dict(self.cells)

    # -- constructors --------------------------------------------------------

    @classmethod
    def cost_leadtime_cross(cls, base: InvSingleScenario,
                            b_values: tuple[float, ...],
                            k_values: tuple[float, ...],
                            lt_values: tuple[int, ...],
                            grid_name: str, desc: str = "",
                            ) -> "InvSingleScenarioGrid":
        """b × K × leadtime crossed on ``base``, pipelines padded to max lt."""
        lt_max = max(lt_values)
        cells = []
        for b, K, lt in product(b_values, k_values, lt_values):
            cid = f"b={b:g},K={K:g},leadtime_value={lt:g}"
            leadtime = (DeterministicLeadtime(value=lt) if lt == lt_max
                        else DiscreteLeadtime(values=[lt, lt_max],
                                              probabilities=[1, 0]))
            cells.append((cid, dataclasses.replace(
                base,
                scenario_name=f"{grid_name}[{cid}]",
                desc=f"grid cell: b={b:g}, K={K:g}, deterministic LT={lt:g} "
                     f"(pipeline padded to {lt_max + 1})",
                shortage_cost=float(b),
                order_cost_fixed=float(K),
                leadtime=leadtime,
            )))
        return cls(cells=cells,
                   axes={"b": tuple(b_values), "K": tuple(k_values),
                         "leadtime_value": tuple(lt_values)},
                   grid_name=grid_name, desc=desc, seed_salt=base.seed_salt)

    @classmethod
    def cost_variance_cross(cls, base: InvSingleScenario,
                            prob_vectors: tuple[tuple[float, ...], ...],
                            b_values: tuple[float, ...],
                            k_fixed: float,
                            grid_name: str, desc: str = "",
                            ) -> "InvSingleScenarioGrid":
        """Lead-time law x b x K crossed on ``base``, at MATCHED MEAN.

        The lead-time axis carries whole probability vectors over a support
        that never changes, so every cell has the same ``leadtime.max()`` and
        the same pipeline length — no padding is needed and the observation
        vector never changes shape. A degenerate vector (one 1.0, rest 0.0) is
        a deterministic lead time expressed in the same family, which is what
        makes the p=0 endpoint comparable to the rest rather than a different
        kind of object.

        Cell ids follow the IR's ``ScenarioGrid.cells()`` spelling so
        leaderboard rows join the schema's enumeration 1:1 — axis order and
        float repr included, which is why the vectors are written out rather
        than derived from a scalar here.
        """
        values = list(base.leadtime.values)
        cells = []
        # K is FIXED, not crossed: K=0 and K>0 are different policy structures
        # (order-up-to vs (s,S)), so they are separate grids and separate
        # leaderboards. It still appears in the cell id -- a one-value axis in
        # the IR -- so a row always carries the regime it was scored under.
        for probs, b, K in product(prob_vectors, b_values, [k_fixed]):
            assert len(probs) == len(values), "probability vector must match the support"
            cid = f"leadtime_probs={list(probs)},b={b:g},K={K:g}"
            cells.append((cid, dataclasses.replace(
                base,
                scenario_name=f"{grid_name}[{cid}]",
                desc=f"grid cell: b={b:g}, K={K:g}, leadtime law {list(probs)} "
                     f"on {values}",
                shortage_cost=float(b),
                order_cost_fixed=float(K),
                leadtime=DiscreteLeadtime(values=list(values),
                                          probabilities=list(probs)),
            )))
        return cls(cells=cells,
                   axes={"leadtime_probs": tuple(tuple(v) for v in prob_vectors),
                         "b": tuple(b_values), "K": (k_fixed,)},
                   grid_name=grid_name, desc=desc, seed_salt=base.seed_salt)

    # -- family-level attributes --------------------------------------------

    @property
    def horizon(self) -> int:
        return self.cells[0][1].horizon

    @property
    def demand(self):
        return self.cells[0][1].demand

    @property
    def leadtime(self):
        """Bound-relevant aggregate: any cell's generator sizes the family
        pipeline (all cells agree on ``max()`` by construction)."""
        return self.cells[0][1].leadtime

    @property
    def stockout_mode(self) -> str:
        return self.cells[0][1].stockout_mode

    # -- enumeration is data access, not behavior ---------------------------

    def __repr__(self) -> str:
        return (f"InvSingleScenarioGrid({self.grid_name!r}, "
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

    def as_sampler(self, weights=None) -> InvSingleGridTrainingSampler:
        return InvSingleGridTrainingSampler(
            grid=self,
            probs=None if weights is None else np.asarray(weights, float),
            seed_salt=self.seed_salt,
            scenario_name=self.grid_name, desc=self.desc)


def cell_slug(cell_id: str) -> str:
    """Filesystem-safe cell id: ``"b=1,K=0,leadtime_value=0"`` -> ``"b1_K0_lt0"``.

    Deterministic and invertible against the grid's own enumeration; used to
    suffix per-cell benchmark/eval artifacts (one file per cell, §9.6)."""
    import re

    # a vector axis would otherwise put a float list in a path; name the law by
    # its tail mass instead ("leadtime_probs=[0.25, 0.5, 0.25]" -> "p25"), which
    # is compact, stable, and still 1:1 with the cell
    cell_id = re.sub(r"leadtime_probs=\[([0-9.eE+-]+),[^\]]*\]",
                     lambda m: f"p{round(float(m.group(1)) * 100):02d}", cell_id)
    return (cell_id.replace("leadtime_value=", "lt")
                   .replace("=", "").replace(",", "_"))


def cells_depadded(grid_name: str):
    """``[(cell_id, slug, depadded_scenario)]`` for per-cell benchmarks.

    A padded cell's dynamics equal the unpadded cell's exactly (gated by
    ``test_grid16_padded_cell_is_the_unpadded_cell_plus_zeros``); only the
    observation length differs, which no benchmark policy reads. Per-cell
    tools must see the EFFECTIVE lead time (``leadtime.mean()``, never
    ``max()`` — see CLAUDE.md's padded-cells trap): the myopic protection
    interval, the DP's state choice and the base-stock IP all key on it, so
    they get the deterministic de-padded twin. The slug names the twin for
    cache/artifact paths."""
    out = []
    for cid, sc in GRIDS[grid_name]:
        # collapsing to the mean is right for a PADDED deterministic cell and
        # destructive for a genuinely stochastic one: it is the lead-time
        # VARIANCE that `lt_variance` sweeps, and a deterministic twin has none
        probs = getattr(sc.leadtime, "probabilities", None)
        assert probs is None or max(probs) == 1.0, (
            f"{grid_name}[{cid}]: cells_depadded() replaces the lead time with a "
            f"deterministic twin at its mean, which is only valid where the law "
            f"is degenerate (a padded cell). This cell is genuinely stochastic "
            f"(probabilities={probs}), so the twin would erase the variance the "
            f"grid exists to sweep. Use the cell itself.")
        lt_eff = int(round(sc.leadtime.mean()))
        slug = cell_slug(cid)
        out.append((cid, slug, dataclasses.replace(
            sc,
            scenario_name=f"{grid_name}__{slug}",
            leadtime=DeterministicLeadtime(value=lt_eff),
        )))
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

grid16 = InvSingleScenarioGrid.cost_leadtime_cross(
    base=scenario_lt,
    b_values=(1.0, 4.0, 9.0, 19.0),
    k_values=(0.0, 20.0),
    lt_values=(0, 2),
    grid_name="grid16",
    desc="RQ-6 generality target (IR grids[grid16]): shortage cost b x fixed "
         "cost K x deterministic lead time, 16 cells, every cell with an "
         "exact DP bar; pipelines padded to the family maximum (len 3).")

# Built through Fraction, then converted -- NOT `1.0 - 2.0 * p`. The two differ
# in the last bit at p = 1/6 (0.6666666666666666 vs ...67), and the cell id
# embeds the float's repr, so the arithmetic path is part of the join key
# against the IR's enumeration. Gated by
# `test_lt_variance_matches_the_ir_declaration`.
# p <= 1/2 is forced: beyond it 1-2p is a negative probability, which
# DiscreteLeadtime rejects (and would NOT be caught by its normalisation, since
# [2/3, -1/3, 2/3] already sums to 1). So 0 and 1/2 are the family's own
# extremes, and Var(L) = 2p spans its full range 0 -> 1.
_SPREADS = (_Fraction(0), _Fraction(1, 6), _Fraction(1, 4),
            _Fraction(1, 3), _Fraction(1, 2))       # Var(L) = 2p: 0, 1/3, 1/2, 2/3, 1
_LT_LAWS = tuple((float(p), float(1 - 2 * p), float(p))
                 for p in _SPREADS)                 # E[L] = 2 at every cell

_B_VALUES = tuple(float(b) for b in range(1, 10))    # fractile b/(b+1): 0.50 -> 0.90


def _lt_variance(k_fixed: float, name: str) -> InvSingleScenarioGrid:
    return InvSingleScenarioGrid.cost_variance_cross(
        base=scenario_slt,
        prob_vectors=_LT_LAWS,
        b_values=_B_VALUES,
        k_fixed=k_fixed,
        grid_name=name,
        desc=f"RQ-4 generality target (IR grids[{name}]): does ONE linear "
             "statistic of (inventory, pipeline) serve a whole family? "
             "Lead-time variability swept at matched mean — [p, 1-2p, p] on "
             "support {1,2,3}, E[L]=2 everywhere, Var(L)=2p spanning its full "
             "range 0 -> 1; crossed with shortage cost b uniform on 1..9 "
             "(fractile 0.50 -> 0.90). p=0 reproduces `lt`'s dynamics and "
             "p=1/3 is `slt`, both measured (#E13); p=1/2 is the bimodal "
             "maximum-variance endpoint, so a jump there is not evidence about "
             "variance alone. Length is held fixed on purpose: moving E[L] "
             "would move the order-up-to level and confound the weight "
             f"question. K={k_fixed:g} throughout (the same fixed cost `simple_k` and "
             "`grid16` use) — K=0 and K>0 are different "
             "policy structures, so they are separate grids and never compared. "
             "45 cells, one pipeline length (4), no padding.")


lt_variance_k0 = _lt_variance(0.0, "lt_variance_k0")
lt_variance_k20 = _lt_variance(20.0, "lt_variance_k20")

# RQ-7's interpolation target: the same family as `grid16`, finer. `grid16` is
# the TRAINING grid; this is the EVALUATION grid. `lt_max` is 2 in both, so both
# render pipeline_len = 3 and one policy can be scored on either without the
# observation width moving — gated by
# `test_cost_leadtime_overlaps_grid16_bit_exactly`.
# b=19 is deliberately absent: it stays in the training grid to anchor the top
# of the range, so b in 5..8 INTERPOLATE here rather than extrapolate.
cost_leadtime = InvSingleScenarioGrid.cost_leadtime_cross(
    base=scenario_lt,
    b_values=tuple(float(b) for b in range(1, 10)),
    k_values=(0.0, 20.0),
    lt_values=(0, 1, 2),
    grid_name="cost_leadtime",
    desc="RQ-7 generality target (IR grids[cost_leadtime]): does a policy "
         "trained on grid16's 16 cells realize the DP's (s,S) map at regimes "
         "it never saw? 54 cells; 12 overlap grid16 (the memorization "
         "control) and 42 are held out in three strata -- b-interpolation "
         "only (24), lead-time-interpolation only (6), and both (12).")


GRIDS: dict[str, InvSingleScenarioGrid] = {
    "grid16": grid16,
    "cost_leadtime": cost_leadtime,
    "lt_variance_k0": lt_variance_k0,
    "lt_variance_k20": lt_variance_k20,
}


if __name__ == "__main__":
    for name, grid in GRIDS.items():
        print(f"{name}: {grid}")
        sampler = grid.as_sampler()
        counts: dict[str, int] = {}
        for seed in range(2000):
            sc = sampler(seed)
            counts[sc.scenario_name] = counts.get(sc.scenario_name, 0) + 1
        assert sampler(7) is sampler(7), "sampler must be pure"
        print(f"  cells drawn: {len(counts)}/{len(grid)}, "
              f"min/max draws {min(counts.values())}/{max(counts.values())}")
        for cid, sc in grid:
            print(f"  {cid:38s} b={sc.shortage_cost:>4g} K={sc.order_cost_fixed:>3g} "
                  f"E[L]={sc.leadtime.mean():g} pipe={sc.leadtime.max() + 1}")

# fnv — MMFE Sequential Ordering (the grid exemplar)

A "fresh-newsvendor" ordering problem under a **martingale demand signal**
(Heath & Jackson 1994, MMFE): place `N` sequential orders while demand
information is progressively revealed, realize demand at the horizon, and sell
against cumulative inventory to maximize expected profit.

This is the reference example for the **design layer** (spec §5.5–§5.6): FNV's
parameter variety is an *experiment-design grid* — the generality target of a
generalist policy — not a world latent. It is the only shipped example with a
`{domain}_grids.py`.

## Files

| File | Layer | Purpose |
|---|---|---|
| `fnv_uncertainty.py` | uncertainty | `NormalDemandSignal` — the MMFE increment generator (seed scheme v2) |
| `fnv_scenarios.py` | scenario | `FnvScenario` + the `simple` registry entry |
| `fnv_grids.py` | **design (§5.6)** | `FnvScenarioGrid`, `GRIDS = {FNV-aMMFE, FNV-mMMFE}` — driver-only |
| `fnv_mdp.py` | core | `FnvState`, `init_state`, `advance` (Gym-free simulator) |
| `fnv_gym.py` | gym | Gymnasium wrapper; accepts a scenario **or** a grid-derived sampler |
| `fnv_schema.json` | IR | the MDP-IR (seed scheme v2, additive MMFE) |
| `fnv_ir_adapter.py` | IR | differential adapter (portable-domain contract) |

## The MMFE process

`N` ordering epochs sit at times `tau = [0, linspace(0, T, N), 1]`. The demand
signal is a martingale: at period `n` an independent mean-zero normal increment
of std `sqrt(tau[n+1]-tau[n]) * stdev` is revealed and added to the cumulative
information `I`. Demand realizes at the horizon as `D = mu + I` (additive MMFE)
or `D = exp(mu + I)` (multiplicative). Each order costs
`(c1 + (n-1)*lamb)` per unit; profit is `r * min(D, inventory) - total_cost`.

Because the signal is **observed** (it enters `information`), the problem is
fully observed — no hidden latent, no memory required.

## World layer vs. design layer

A concrete `FnvScenario` (`stdev`, `T`, `lamb`, …) plus an `episode_seed` pins
one episode completely. FNV has **no world-latent sampler** — nothing is drawn
per episode to *become* the problem. What varies across a study is the
`stdev × T × lamb` sweep, and a finite sweep a policy is trained to generalize
over is a **grid**, not a sampler (spec §5.5). The `simple` scenario is the
single fixed instance; the grids hold the sweeps.

## Grids (`fnv_grids.py`, spec §5.6)

```python
from fnv_grids import GRIDS
grid = GRIDS["FNV-aMMFE"]        # 6×9×10 = 540 cells

len(grid)                         # 540
grid["stdev=0.1,T=0.9,lamb=0.1"]  # a cell by id -> FnvScenario
grid[0]                           # ... or by index
for cell_id, scenario in grid: ...# enumeration (canonical, row-major order)
grid.N, grid.mmfe_mode            # family attrs, no cell resolved
```

A grid is deliberately **not callable** — it can never pass where a
`ScenarioSource` (a scenario or sampler) belongs. Its one verb is
`as_sampler()`, which derives an ordinary per-episode sampler over the cells:

```python
sampler = grid.as_sampler()               # uniform over cells
sampler = grid.as_sampler(weights=w)      # per-cell training weights (§5.5)
scenario = sampler(episode_seed)          # pure: episode_seed -> FnvScenario
```

The cell index is drawn on the **v2 meta seed key**
`[substream_id, 0, episode_seed, seed_salt]`; every cell shares the same
intrinsic seed protocol, so eval reuses one block of episode seeds across all
cells for **paired (common-random-number)** comparisons.

### Generalist training (targets a grid)

The gym receives the derived sampler, never the grid (spec §8, §9.6):

```python
from fnv_grids import GRIDS
from fnv_gym import FnvEnv

grid = GRIDS[args.grid_name]
sampler = grid.as_sampler()                        # optionally weights=...
env = FnvEnv(scenario=sampler)                     # re-draws a cell each reset
# ... SB3 PPO over `env` (SubprocVecEnv-safe: the sampler is a module-level class)
```

### Generalist evaluation (enumerates the grid)

```python
grid = GRIDS[args.grid_name]
seeds = eval_seed_block(args)                       # ONE block, reused per cell (CRN)
for cell_id, scenario in grid:                      # one leaderboard row per cell
    rows = evaluate(policy, scenario, seeds)        # {grid_name}/{cell_id}
```

## Regression gates

Run from the repo root (harness installed via `pip install -e ./harness`):

```bash
E=plugin/skills/mdp-solver/examples
python -m mdp_conformance $E/fnv
python -m mdp_ir $E/fnv/fnv_schema.json
python -m mdp_ir.differential $E/fnv/fnv_schema.json --episodes 40
```

The differential vouches for `t, order, cost, inventory, information,
total_cost` every period, plus terminal `demand, sales, revenue`. The IR bakes
the per-period signal-volatility schedule as exact float constants so the
interpreter's normal draw matches the domain's `FnvScenario.signal`
bit-for-bit; the domain re-derives the same schedule from `stdev/T/N`.

## Modeling notes

- **1-indexed clock**: periods run `1..N`; the interpreter terminates when the
  post-increment clock reaches `horizon.T`, so `horizon.T = N+1` (see the IR's
  `assumptions_log`).
- **Additive MMFE only** in the IR (`D = mu + I`). The multiplicative mode is a
  domain/grid-layer variant (`mmfe_mode`), out of IR scope.
- The design grids are **out of IR scope** — the IR formalizes one concrete
  scenario; grid coverage is verified structurally by `mdp_conformance`.

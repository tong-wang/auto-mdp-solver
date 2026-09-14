# dynamic_pricing — plain-English restatement

Round-trip artifact for the Phase-A gate. Source: Guillermo Gallego and
Garrett van Ryzin, *Optimal Dynamic Pricing of Inventories with Stochastic
Demand over Finite Horizons*, Management Science 40(8), 1994, pp. 999–1020.
The model below is the paper's single-product, regular-demand case
(`lambda(p) = a * exp(-alpha * p)`), discretized in time.

IR: `dynamic_pricing_schema.json` · `mdp` fingerprint `a1bf3475a7da` ·
structural `5b5d30fa24b8` (v0.10.11 pin).

**Provenance.** Backfilled on 2026-09-14 from the frozen IR — the domain was
formalized in July 2026 before the restatement artifact existed, and no
interview record survives. Everything here is rendered from the schema; the
`source` tags are the schema's own. Two `Confirmable`s are still `derived`
(the decision's type and bounds), so this document is not yet the signed-off
result of a Phase-A gate; see `ESCALATION.md` §Frontier A1.

---

## The problem in prose

A seller starts the season with `n0` units of one product and has `T`
periods to sell them. At the start of each period the seller sets a price.
Customers then arrive during the period as a Poisson stream whose mean count
is `a * exp(-alpha * price) * dt` — the higher the price, the fewer arrive —
and each arrival buys one unit if any stock remains. An arrival that finds
the shelf empty is lost; nothing is backlogged and nothing is reordered. The
season ends when the periods run out, or earlier the moment the last unit
sells, since a seller with no stock has nothing left to decide. Unsold units
at the horizon are worth a salvage value `q` each, zero in the paper's
worked case.

### What is being decided

One number per period: the **price** charged this period, a non-negative
real. The paper's control is the demand *intensity*; the IR makes the price
the canonical decision consumed by the transition and keeps the intensity as
a second agent-facing encoding in the gym layer (`gym.action_modes`), so the
two are one problem rendered two ways, never two formalizations.

### Objective and mode stance

Maximize the **expected total revenue** of the season: per period
`price * units_sold`, plus `q * inventory` once at the horizon. Undiscounted
(`discount_factor` 1.0). Purchase and production costs are sunk and do not
appear.

Mode stance: there are no scenario modes. The two structural flags
(`allow_backlog`, `allow_reorder`) are held false — the paper's case — and
the scenario class asserts them; they are not compared. The two action
encodings are gym-layer selections on one decision (above).

No tier-2 research question is declared (`research_questions` absent): the
campaign asks the standing tier-1 question only.

### Randomness classification

| source | kind | where it lives |
|---|---|---|
| customer arrivals | **intrinsic** — realized every period, keyed on `period` (v2 seed tree, stream `demand`, source id 0) | `uncertainty_slots.demand`, candidate `poisson`, rate `a * exp(-alpha * price) * dt` |

The arrival count is conditioned on the **current** price through its rate
only; the seed key carries no decision, so realized randomness is independent
of the price *path* (a thinning of one underlying Poisson stream). There is
no world latent, no mixture, no per-episode draw: a scenario plus an episode
seed pins the episode completely.

### The scenario set

Constants, with their design axes:

| constant | value | axis | meaning |
|---|---|---|---|
| `a` | 100 | demand | demand scale; 100 is "demand high relative to stock" (the paper's Fig. 1, bottom) |
| `alpha` | 1 | demand | price sensitivity |
| `dt` | 0.02 | — | period length = 1 / `T` (unit horizon); a discretization granularity, not a design axis |
| `q` | 0 | salvage | salvage value per unsold unit at the horizon (0 without loss of generality per the paper) |
| `n0` | 25 | sizing | initial stock |
| `allow_backlog`, `allow_reorder` | false | — | structural flags; referenced by nothing in the rendering (`schema.no_enumeration` WARNs on them) |

Instances: `ample_stock` overrides `a` = 20 — demand low relative to stock,
where run-out is unlikely and the myopic price is optimal. No grid, no
sampler. Train/eval strategy: **specialist** on `simple` (the base
instance); `ample_stock` is a separate board opened for its benchmarks.

### Decision, objective, horizon

- **Decision** `price`: continuous, scalar, bounds `[0, 8]`, feasibility
  non-negative. Type and bounds are `derived` (rationale: with `a` 100 and
  `alpha` 1 the optimal price is near 1 and run-out prices stay well under
  6; at 8 the intensity is about 0.03, effectively the null price).
  **Unconfirmed.**
- **State**: `period` (time index, 0-based, ends at `T`) and `inventory`
  (core, integer in `[0, n0]`, non-increasing). Both observable.
- **Info fields** per period: `arrivals`, `units_sold`, `lost_demand`,
  `intensity`, and the revenue decomposition `sales` / `salvage` / `total`.
- **Event sequence** `P, D`: the price is set (`intensity` computed), then
  demand arrives (`arrivals ~ demand.sample`, `units_sold = min(arrivals,
  inventory)`, `inventory -= units_sold`, `lost_demand = arrivals -
  units_sold`), then `period += 1`. Observation point: pre-price.
- **Horizon** `T` = 50, `horizon_end` = `terminated` (the horizon is the
  problem's boundary, so the gym renders `time_to_go = T - period` as a
  declared feature); early termination when `inventory == 0`.
- **Gym**: observation modes `vec` = `[inventory, time_to_go]` (default) and
  `vec_d` = `[inventory, time_to_go, units_sold]`; action modes `price`
  (identity, clip to `[0, 8]`, default) and `intensity` (box `[0, 36.8]` =
  `[0, a/e]`, reparametrized by `price = log(a / intensity) / alpha`);
  reward mode `revenue` = `total`.
- **RL**: PPO; `requires_memory` false (`human_confirmed`: inventory and
  time-to-go are the sufficient statistic); `frame_stack` 1; observation
  normalization on (`rationale`: stationary, modestly heterogeneous scales).
- **Benchmarks** (`human_confirmed`): `dp` — role `exact`, backward
  induction over `(period, inventory)` with the price on a fine grid;
  `fluid` — role `feasible`, the paper's static policies `fixed` and
  `myopic` plus a `random` floor. The fluid relaxation's own value is a
  `relaxed` quantity, not emitted as an eval column, so it has no entry.

### Assumptions filled in without being told

From the IR's `assumptions_log`, each with its resolved source:

1. **Continuous → discrete time** (modeling call): `T` = 50 equal periods
   of length `dt` = 0.02; per-period demand Poisson(`lambda(price) * dt`).
   Larger `T` approaches the paper's point process. — `derived`
2. **Action encoding**: `price` (default) and `intensity` are gym-layer
   selections on one canonical decision. — `derived`
3. **Seed key** derived from `realization = period`; the primitive is
   transformed by the current price through its rate, so realized sales are
   independent of past actions. — `derived`
4. **Scenario axes** are the axis-tagged constants: demand (`a`, `alpha`),
   sizing (`n0`), salvage (`q`). — `derived`
5. **No backlog**: demand at zero stock is lost. — `derived`
6. **Early termination** at `inventory == 0` drops no salvage value, since
   `q * 0 = 0` under any `q`. — `derived`
7. **Costs sunk**: the objective is revenue only. — `derived`
8. **Out of scope** (the paper's §5 extensions): compound-Poisson demand,
   time-varying seasonality, holding costs and discounting, initial stock as
   a decision, overbooking and cancellations. — `derived`

### Invariants (`mdp.invariants`)

Claims independent of the model, checked by the laws gate and the
differential on every instance:

| name | claim, in the problem's words | expression |
|---|---|---|
| `sales_capped_by_stock` | a period sells the lesser of arrivals and the stock it opened with | `close(units_sold, min(arrivals, prev.inventory))` |
| `arrivals_split` | every arrival either buys or is lost — no third outcome | `close(lost_demand, arrivals - units_sold)` |
| `inventory_balance` | stock is depleted only by sales; there is no replenishment | `close(inventory, prev.inventory - units_sold)` |
| `no_backlog` | unsold demand is lost, never carried as negative stock | `inventory >= 0` |

---

## Sample trajectory (step 7b)

One episode of the base instance at a constant price of 1.0 — the myopic
price `1/alpha` — for episode seed 3. Read the columns as: the price set,
the arrivals the period drew at that price, how many bought, how many were
lost, the intensity `lambda(1.0) = 100/e`, the stock after the period, and
the revenue split. Stock runs out in period 40 of 50 and the episode
terminates early; the last arrival there finds one unit and one is lost.

```step7b
python -m mdp_ir.interpreter dynamic_pricing/dynamic_pricing_schema.json \
  --decision price=1.0 --episode-seed 3 --max-rows 60
```

```
dynamic_pricing v0.4  episode_seed=3
 t  price  arrivals  units_sold  lost_demand  intensity  inventory  sales  salvage  total  reward
 0   1.00         0           0            0      36.79         25   0.00     0.00   0.00    0.00
 1   1.00         1           1            0      36.79         24   1.00     0.00   1.00    1.00
 2   1.00         2           2            0      36.79         22   2.00     0.00   2.00    2.00
 3   1.00         2           2            0      36.79         20   2.00     0.00   2.00    2.00
 4   1.00         1           1            0      36.79         19   1.00     0.00   1.00    1.00
 5   1.00         2           2            0      36.79         17   2.00     0.00   2.00    2.00
 6   1.00         1           1            0      36.79         16   1.00     0.00   1.00    1.00
 7   1.00         0           0            0      36.79         16   0.00     0.00   0.00    0.00
 8   1.00         0           0            0      36.79         16   0.00     0.00   0.00    0.00
 9   1.00         2           2            0      36.79         14   2.00     0.00   2.00    2.00
10   1.00         2           2            0      36.79         12   2.00     0.00   2.00    2.00
11   1.00         0           0            0      36.79         12   0.00     0.00   0.00    0.00
12   1.00         1           1            0      36.79         11   1.00     0.00   1.00    1.00
13   1.00         0           0            0      36.79         11   0.00     0.00   0.00    0.00
14   1.00         0           0            0      36.79         11   0.00     0.00   0.00    0.00
15   1.00         0           0            0      36.79         11   0.00     0.00   0.00    0.00
16   1.00         0           0            0      36.79         11   0.00     0.00   0.00    0.00
17   1.00         0           0            0      36.79         11   0.00     0.00   0.00    0.00
18   1.00         0           0            0      36.79         11   0.00     0.00   0.00    0.00
19   1.00         0           0            0      36.79         11   0.00     0.00   0.00    0.00
20   1.00         0           0            0      36.79         11   0.00     0.00   0.00    0.00
21   1.00         1           1            0      36.79         10   1.00     0.00   1.00    1.00
22   1.00         1           1            0      36.79          9   1.00     0.00   1.00    1.00
23   1.00         1           1            0      36.79          8   1.00     0.00   1.00    1.00
24   1.00         1           1            0      36.79          7   1.00     0.00   1.00    1.00
25   1.00         0           0            0      36.79          7   0.00     0.00   0.00    0.00
26   1.00         0           0            0      36.79          7   0.00     0.00   0.00    0.00
27   1.00         1           1            0      36.79          6   1.00     0.00   1.00    1.00
28   1.00         0           0            0      36.79          6   0.00     0.00   0.00    0.00
29   1.00         0           0            0      36.79          6   0.00     0.00   0.00    0.00
30   1.00         2           2            0      36.79          4   2.00     0.00   2.00    2.00
31   1.00         0           0            0      36.79          4   0.00     0.00   0.00    0.00
32   1.00         0           0            0      36.79          4   0.00     0.00   0.00    0.00
33   1.00         0           0            0      36.79          4   0.00     0.00   0.00    0.00
34   1.00         0           0            0      36.79          4   0.00     0.00   0.00    0.00
35   1.00         0           0            0      36.79          4   0.00     0.00   0.00    0.00
36   1.00         0           0            0      36.79          4   0.00     0.00   0.00    0.00
37   1.00         1           1            0      36.79          3   1.00     0.00   1.00    1.00
38   1.00         1           1            0      36.79          2   1.00     0.00   1.00    1.00
39   1.00         1           1            0      36.79          1   1.00     0.00   1.00    1.00
40   1.00         2           1            1      36.79          0   1.00     0.00   1.00    1.00
episode total = 25.00   reward total = 25.00   periods = 41 (terminated early)
```

The same command with `--instance ample_stock` renders the low-demand
instance: at `a` = 20 the intensity at price 1.0 is 7.36, five units sell in
fifty periods, and the episode runs the full horizon with 20 units unsold.

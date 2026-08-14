# fnv — plain-English restatement (Phase A round-trip artifact)

IR: `fnv_schema.json` · ir_version 0.4 · `mdp` fingerprint `40110193eba5` ·
structural fingerprint `36f5c3aaf7a6` · seed scheme v2

*(Fingerprints moved 2026-08-13 from `b992f2adead2` / `b1a471107c9d` when the
two Phase-A confirmables were resolved and the `order` bound was re-derived —
see "Decision, horizon, objective". Entries written before that date quote the
old pair. Owes a §IR-CHANGELOG entry once a campaign log exists.)*

> **Provenance: this IR was PORTED, not elicited here.** It arrived with the
> upstream case folder (`mdp_solver` `cases/fnv/`, demoted from `examples/` in
> v0.8.0) and its eight IR-half files are byte-identical to that source. No
> Phase-A elicitation conversation happened in this repo, so this document is
> **reverse-engineered from the frozen IR and the code it generates** — it
> records what the model *says*, not what a user was asked. Every assumption
> below therefore carries the `source` the IR itself records, and where that
> source is `derived` it means derived by the original author, not confirmed
> by this repo's user.
>
> **The Phase-A confirmables are now resolved (2026-08-13).** They were not:
> `mdp.decisions[0].type` and `.bounds` were both `source: derived`, so
> `unconfirmed()` was non-empty and the `mdp` block had never been formally
> frozen. Resolving them **forked `fnv_schema.json` from upstream** — the
> other seven IR-half files remain byte-identical, and the patch is owed back
> to `cases/fnv/`. See **Open items**.

## The problem

You are a newsvendor with **one selling season** and **several chances to buy
before it**. Demand for the season is unknown, but your forecast of it keeps
improving: at each ordering date a piece of market information arrives and
updates the forecast, and the updates form a **martingale** — today's forecast
is your best guess of tomorrow's (Heath & Jackson 1994, the MMFE model).

You place **N = 3 orders** at fixed dates before the season. Buying early is
**cheaper** (`c_n = c1 + (n-1)·λ`, rising with each period) but you know less;
buying late costs more but you have seen more of the forecast. Nothing is
delivered late, nothing is cancelled, and there is no salvage value and no
penalty for unmet demand. At the horizon the true demand `D` realizes, you
sell `min(D, total ordered)` at price `r`, and your profit is that revenue
minus everything you paid.

The whole problem is the tension between those two effects: **order early and
cheap under a vague forecast, or late and dear under a sharp one.**

## Mode stance (source: derived, from the IR's `mmfe_mode` constant)

Two **independent branches**, each its own scenario instance and its own
leaderboard — no head-to-head comparison (the demand scales differ, so the
profits are not commensurable):

| branch | instance | demand link | base `stdev` |
|---|---|---|---|
| additive (a-MMFE) | base (`fnv`) | `D = mu + I` | 0.1 |
| multiplicative (m-MMFE) | `mmmfe` | `D = exp(mu + I)` | 0.2 |

The branch is **not** an uncertainty candidate. Both branches draw the *same*
mean-zero normal signal from the *same* generator; they diverge only in the
deterministic terminal link applied once at the horizon. The IR therefore
encodes the choice as a **categorical scenario constant** (`mmfe_mode`), read
by a constant-conditioned dynamics expression — not as a second candidate of
the uncertainty slot. The `mmmfe` instance additionally widens the signal
(`stdev` 0.1 → 0.2) with its `signal_stdevs` schedule re-baked to match, so
one instance exercises both a mode selector and a derived-constant re-bake.

## Randomness classification

There is exactly **one source of randomness**: the MMFE forecast increment.

- **Per-period signal increment** — the `signal` slot, `stream_id` 0, one
  stage `increment` with `realization: period`, single candidate `ammfe`
  (`NormalDemandSignal`, normal, mean 0, std `signal_stdevs[period]`).
  It is **observed, not latent** (`latent: false`): the increment is revealed
  into the state variable `information` immediately after the period's order,
  so the agent sees the whole information process as it unfolds. Nothing about
  demand is hidden behind a per-episode world latent.
- **No world latents.** Unlike a bandit, nature fixes nothing secret at the
  start of the episode. Demand is a deterministic function of the accumulated,
  fully-observed signal: `D = mu + I[N+1]` or `exp(mu + I[N+1])`.
- **No training-only distributions.** The per-episode signal path *is* the
  problem's own story.

The volatility schedule is **pre-resolved data, not a formula**:
`signal_stdevs[n] = sqrt(tau[n+1] − tau[n]) · stdev` with
`tau = concat([0], linspace(0, t_last, N), [1])`, carried in the IR as exact
float reprs so the interpreter's draw matches `FnvScenario.signal`
bit-for-bit. `signal_stdevs[0] = 0` because `tau[0] = tau[1] = 0`, so the
first order is placed under the prior with no information yet. The variances
sum to `stdev²` — total uncertainty is conserved however the epochs are
spaced, which is what makes `t_last` a pure *timing* lever.

## Decision, horizon, objective

- **Decision**: `order` — continuous (`human_confirmed`), `dim` 1, feasibility
  `non_negative`, bounds **[0, `order_max`]** (`human_override`). One order
  quantity per period, added to cumulative inventory.

  `order_max` is a scenario constant carrying the **~5-sigma demand ceiling**,
  re-baked per instance:

  | | formula | base | `mmmfe` |
  |---|---|---|---|
  | additive | `mu + 5*stdev` | **1.5** | — |
  | multiplicative | `exp(mu + 5*stdev)` | — | **7.389056…** |

  With no salvage value, an order beyond the highest plausible demand is
  strictly dominated — the marginal unit cannot be sold — so this is the
  useful action scale rather than a safety margin. It is also exactly
  `FnvEnv`'s `action_space` high, so the model bound and the agent-facing
  encoding now derive from one argument instead of two. The bound is
  non-binding for the optimum: every computed `b_n` is negative, so optimal
  order-up-to sits below `mu + I`.

  It replaced the literal **[0, 3]**, which was reasoned from the additive
  branch alone. That literal was simultaneously too loose on base (real
  ceiling 1.5) and too tight on `mmmfe` (real ceiling 7.39) — so the
  differential drew decisions from [0, 3] while the multiplicative domain
  reached 7.39, leaving that range unverified.
- **Horizon**: `T = 4`, 1-based indexing, `horizon_end: terminated`. Note the
  off-by-one: there are **N = 3 ordering periods** (t = 1, 2, 3); the
  interpreter terminates when the post-increment clock reaches 4, so
  `horizon.T = N + 1`. The seed key uses the raw `period` (1..N), matching
  `state.period`, which is what makes the trajectories diff bit-exactly.
- **Objective**: `maximize`, two per-step components —
  `ordering = -cost` (paid every period) and `revenue = r * sales`
  (zero until the horizon, since `sales == 0` before it).
  Episode total = `-total_cost + revenue` = **profit**.
- **Terminal realization**: demand and sales fire under a `period == N` guard,
  *before* the period increment, so every earlier row reports demand 0.

## Claims that must hold (declared as `mdp.invariants`)

Independent of the model — the differential only proves the interpreter and
the generated domain *agree*, so a shared mis-formalization needs a claim taken
from the problem statement to catch it. Checked on every row of every
trajectory, on both branches:

| claim | expression | what it pins |
|---|---|---|
| `inventory_accumulates_orders` | `close(inventory, prev.inventory + order)` | "the only inflow is the period's order; nothing leaves before the horizon" |
| `cost_accumulates` | `close(total_cost, prev.total_cost + cost)` | the running bill is exactly the sum of per-period order costs |
| `demand_is_terminal_only` | `t == N or close(demand, 0)` | demand realizes **once**, at the horizon — not period by period |
| `sales_capped_by_stock` (terminal) | `close(sales, min(demand, inventory))` | you sell the lesser of demand and what you accumulated — no backorders, no salvage |

Claims an expression cannot state live in `fnv_test.py` instead: that the
signal increments are path-independent (a claim about counterfactual decision
paths, which no single trajectory can express), and that the variance schedule
sums to `stdev²`.

## What the policy observes

Observation mode `vec` (the only one), **6 features in this order**:

| # | feature | form | units |
|---|---|---|---|
| 0 | `stdev` | derived (scenario constant) | demand units |
| 1 | `t_last` | derived (scenario constant) | fraction of the season, (0,1) |
| 2 | `lamb` | derived (scenario constant) | cost/unit/period |
| 3 | `period` | state variable | 1..N |
| 4 | `inventory` | state variable | units ordered so far |
| 5 | `information` | state variable | demand units |

The three **scenario constants** are in the observation because one policy is
trained across a whole design grid (§5.6) and must condition on which cell it
is in. That is load-bearing, not decoration: the optimal offsets move
substantially with all three — `b_1` roughly doubles from `stdev` 0.15→0.30,
and again from `lamb` 0.10→0.02 — so a policy blind to them cannot be optimal
grid-wide, only a single compromise across all 540 cells.

`period` is carried raw rather than as `orders_to_go = N − period + 1`; the two
are bijective for fixed `N`, so this is an encoding choice.

> **Record updated 2026-08-13**, from an older 3-feature encoding
> (`inventory`, `information`, `orders_to_go`) to the 6 the gym emits.
>
> This is bookkeeping, not a correctness fix. **The observation is a solution
> lever, not part of the problem**: only the `mdp` block freezes, `gym`/`rl`
> are Phase-B mutable, and the skill states that observation levers are *not*
> IR changes — they live in the gym with their own executable gate. So
> `gym.observation_modes` records an encoding without binding `_get_obs`, and
> no gate compares them, which is correct rather than a blind spot. The gym
> was never "wrong" to have moved on. Moved **no** fingerprint.

`rl.requires_memory` = **false** (`human_confirmed`): the signal is revealed
into `information`, so `(inventory, information, period)` is a sufficient
statistic and the MDP is fully observed. `rl.obs_normalization` enabled
(`derived`): the features sit on modestly heterogeneous scales.

## Scenario set & training target

The IR formalizes **concrete scenarios only** — the base plus one instance.
FNV's parameter variety is *experiment design*, not world uncertainty, so it
lives outside the IR in the design layer (spec §5.5–§5.6):

| | where | contents |
|---|---|---|
| IR scenarios | `fnv_schema.json` | base (additive, stdev 0.1) + `mmmfe` (multiplicative, stdev 0.2) |
| `SCENARIOS` | `fnv_scenarios.py` | `simple` — one concrete additive scenario |
| `GRIDS` | `fnv_grids.py` | `FNV-aMMFE`, `FNV-mMMFE` — 540 cells each, `stdev × t_last × lamb` |

Training samples a grid via `grid.as_sampler()` (one cell drawn per episode on
the v2 meta key); evaluation enumerates cells, one leaderboard row each, all
cells on the same eval seed block. The two grids are **separate leaderboards**.
Grid coverage is verified structurally by the conformance harness; the
differential covers base + `mmmfe` only.

## Key assumptions (with the `source` the IR records)

- MMFE information process, martingale forecast updates — **derived** (the
  problem's defining structure; Heath & Jackson 1994).
- `mmfe_mode` as a categorical *constant* selecting a deterministic terminal
  link, rather than a second uncertainty candidate — **derived** (both modes
  share one generator and diverge only at the horizon).
- Volatility schedule baked as resolved float data, not computed in-IR —
  **derived** (bit-exactness with `FnvScenario.signal`; the cost is that an
  instance changing `stdev`/`t_last`/`N` must re-bake it, as `mmmfe` does).
- `horizon.T = N + 1 = 4` with 1-based indexing — **derived** (terminal clock
  convention; the seed key still uses raw `period`).
- Design grids out of IR scope — **derived** (they are the experimenter's
  choice, not the world's).
- `requires_memory` = false — **human_confirmed**.
- Obs normalization enabled — **derived** (§8.3).
- Decision type `continuous` — **human_confirmed** (2026-08-13).
- Decision bounds `[0, order_max]` — **human_override** (2026-08-13),
  replacing the derived literal `[0, 3]`. `order_max` = the ~5σ demand
  ceiling, baked per instance; see "Decision, horizon, objective".

## Annotated sample trajectory (interpreter output, episode_seed=3, fixed `order=0.3`)

Base = **additive**. Three orders of 0.3 at rising cost (0.30, 0.33, 0.36);
demand stays 0 until the horizon, then realizes as `mu + I = 1 + 0.06 = 1.06`.
Inventory reached only 0.90, so sales are capped at 0.90 — the
`sales_capped_by_stock` invariant biting — and revenue is `2.0 × 0.90 = 1.80`:

```
fnv v0.4  episode_seed=3
t  order  cost  demand  sales  inventory  information  total_cost  ordering  revenue  total  reward
1   0.30  0.30       0      0       0.30         0.09        0.30     -0.30     0.00  -0.30   -0.30
2   0.30  0.33       0      0       0.60         0.04        0.63     -0.33     0.00  -0.33   -0.33
3   0.30  0.36    1.06   0.90       0.90         0.06        0.99     -0.36     1.80   1.44    1.44
episode total = 0.81   reward total = 0.81   periods = 3
```

`mmmfe` = **multiplicative**, same seed. The cost/inventory columns are
identical — the branch changes nothing before the horizon — but the wider
signal (`stdev` 0.2) gives a different information path, and the terminal link
exponentiates it: `D = exp(mu + I) = exp(1 + 0.12) = 3.05`. Demand far exceeds
the 0.90 on hand, so sales are again stock-capped and the profit is the same
0.81; the branches differ in *how badly* the fixed policy under-orders:

```
fnv v0.4  episode_seed=3 instance=mmmfe
t  order  cost  demand  sales  inventory  information  total_cost  ordering  revenue  total  reward
1   0.30  0.30       0      0       0.30         0.18        0.30     -0.30     0.00  -0.30   -0.30
2   0.30  0.33       0      0       0.60         0.08        0.63     -0.33     0.00  -0.33   -0.33
3   0.30  0.36    3.05   0.90       0.90         0.12        0.99     -0.36     1.80   1.44    1.44
episode total = 0.81   reward total = 0.81   periods = 3
```

Commands to reproduce (from the repo root):

```
python -m mdp_ir.interpreter fnv/fnv_schema.json --decision order=0.3 --episode-seed 3
python -m mdp_ir.interpreter fnv/fnv_schema.json --instance mmmfe --decision order=0.3 --episode-seed 3
```

## Open items

1. ~~**Two confirmables unresolved**~~ — **resolved 2026-08-13**;
   `unconfirmed()` is now empty and all four gates pass. The cost is that
   **`fnv_schema.json` is forked from upstream** `cases/fnv/`: the patch
   (`fnv_order_max.patch`, verified to apply cleanly there) is owed back, and
   until it lands a naive re-port would silently revert the bound and re-open
   the confirmables.
2. ~~**IR `gym` block is stale**~~ — **fixed 2026-08-13**; the IR now declares
   the 6 features the gym emits. Part of the same owed-upstream fork as (1);
   moved no fingerprint.
3. **No `ESCALATION.md` / `INTERPRET.md`** yet — the solve leg has only just
   been built and its first ladder is running.

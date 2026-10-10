# owmr — Phase-A restatement

The problem as `owmr_schema.json` states it, in prose, for the human to check
before the `mdp` block freezes. Rendered 2026-09-22 at mdp fingerprint
`ad3be8d58d0d`, structural fingerprint `4b1b412fb64a`. Source: Doğru, de Kok & van Houtum
(2009), "A numerical study on the effect of the balance assumption in
one-warehouse multi-retailer inventory systems", *Flexible Services and
Manufacturing Journal* 21(3–4):114–147 — the model of its §3 and the
identical-retailer test bed of its §4.1.

## The problem

One warehouse supplies five retailers. Every period, in this order: the
warehouse **receives** what the outside supplier shipped three periods ago;
the warehouse **orders** any non-negative quantity from the supplier, who has
unlimited stock and delivers after a fixed three-period lead time; each
retailer **receives** what the warehouse shipped it one period ago; the
warehouse **ships** any non-negative quantities to the retailers, in total no
more than it has on hand (a request beyond on-hand is scaled down
proportionally, never refused and never negative — the "balance assumption"
that lets the literature pretend otherwise is *not* in the model); then
**demand** hits every retailer. Demand at each retailer is independent across
retailers and periods, gamma-distributed with mean 1 per period and a
coefficient of variation of 0.5 or 2 depending on the cell. Unmet demand is
**backordered** and waits.

At the end of each period the system pays 0.5 per unit on hand at the
warehouse, 1.0 (= 0.5 + 0.5) per unit on hand at each retailer, and a
backorder penalty per unit of retailer backlog of 4 or 19 depending on the
cell. Stock in transit is not charged (it is the same under every policy in
the paper's long-run model). The objective is to **minimize total cost over
100 periods**, undiscounted; results are also quoted per period (total / 100)
beside the paper's per-period lower bound.

Both decisions are taken at the start of the period, with this period's
supplier delivery already known from the pipeline — exactly the information
the paper's decision maker has under fixed lead times.

## Objective and stances

- **Objective:** total holding + backorder cost over the horizon, β = 1. No
  rival objective was declined; three **report-only columns** ride along in
  every evaluation and never decide anything: the warehouse's share of system
  on-hand (mean over the episode), retailer-periods ending in backlog, and
  mean system stock (on-hand plus in transit).
- **Mode stance:** the paper's model has no categorical mode (one review
  discipline, complete backordering, fixed lead times, free ordering), so
  there is nothing to compare head-to-head or branch.
- **Tier-2 stance: discover.** In the cells where the paper shows the
  balance-assumption bound to be loose (high cv), RL is claimed to find an
  ordering/allocation structure the LB heuristic lacks — retaining stock at
  the warehouse, allocating non-myopically — and a rule fitted to the trained
  policy is scored as its own arm. The tight cells validate the readback
  instrument first. A probe and an INTERPRET.md are owed.

## Randomness

One source: per-period, per-retailer demand — intrinsic, drawn every period
on the seed tree's period branch (one stream, five independent components).
Lead times are fixed integers (`stochastic: false`). Nothing is drawn once
per episode: no world latents, no hidden quantities, so the current stocks
and pipelines are a sufficient statistic and no memory is required.

## The scenario set

Fixed for the campaign (human, 2026-09-22): N = 5 identical retailers,
warehouse lead time 3, retailer lead time 1, added holding value 0.5 at the
warehouse and 0.5 at each retailer, mean demand 1 per retailer, horizon 100,
mean-cover initial state (every pipeline slot holds one period of mean
demand; warehouse 5 and each retailer 1 on hand). Four cells, a 2 × 2:

| cell | penalty p | cv | paper's regime |
|---|---|---|---|
| `base` (the IR's base) | 4 | 0.5 | bound tight (gap ≈ 0.05 %) — instrument validation |
| `hicv` | 4 | 2 | bound loose (ave gap ≈ 5 %, max ≈ 17 %) |
| `hipen` | 19 | 0.5 | bound tight, 95 % service target |
| `hipen_hicv` | 19 | 2 | bound loose with the high service target |

No grids; the training target is a **specialist per cell**. Cells share one
observation and action shape but differ in cost scale, so each is scored
against its own bracket and never against another cell.

## Decisions, horizon, benchmarks

- **Decisions:** `order` (one continuous quantity) and `ship` (five continuous
  quantities), both in [0, 10 × (l0 + 1) × Σμ] = [0, 200] here — an
  action-scale cap justified from the lead time and the demand rate, not a
  physical limit. Default gym encoding `order_frac`: the order plus five
  fractions of warehouse on-hand; retention is implicit (1 − Σf). Also
  authored: `order_ship` (raw quantities) and `target_frac` (an echelon
  order-up-to level — a lever that hands over the bound's structure, never the
  discover arm).
- **Horizon:** 100 periods, 0-based, `terminated` at the end; the countdown
  `time_to_go` leads every observation mode.
- **Benchmarks (declared roles):** `lb` — the balance-assumption lower bound,
  **relaxed**, computed analytically per cell (paper eqs. 7–8 with gamma
  convolutions) as a long-run per-period number × 100; `lb_heuristic` — the
  paper's LB heuristic (order up to y0*, myopic allocation with
  non-negativity), **feasible**, simulated; `random` — the floor. There is no
  exact optimum: the bracket is the reference.

## Assumptions filled in without being told

| assumption | source |
|---|---|
| Event order rendered receipt-before-dispatch (RW, O, RR, S, D): identical trajectories to the paper's order for all lead times ≥ 1; l_i = 0 is out of scope | derived, stated in `out_of_scope` |
| Shipments are quantities in the MDP with proportional scale-down when they exceed on-hand (spec §7.2 option 2) | derived; the default encoding never triggers it |
| Demand family = gamma at the paper's (μ, cv) | human_confirmed |
| Demand realization kept in `info`, not state | human_confirmed |
| Decisions continuous; cap 10 × (l0 + 1) × Σμ | human_confirmed |
| No memory; plain PPO | human_confirmed |
| Finite horizon 100, mean-cover start, β = 1 | human_confirmed |
| Retailer lead time shared by all retailers (narrowing of the paper's per-retailer l_i) | derived, recorded on `l_rt` |
| Obs normalization on (a prior, checked at the breadth tier) | derived |

## Invariants the paper's statement implies (checked by the laws gate)

warehouse balance; retailer balance; conservation (only the supplier order
enters, only demand leaves); the warehouse never goes negative; shipments fit
within post-receipt on-hand.

## Sample trajectory

A fixed policy — order 5 every period (mean system demand), ship 1 to each
retailer — on the base cell, seed 3. Re-render with the command below
whenever the model changes; `docs.restatement_current` diffs it.

```step7b
python -m mdp_ir.interpreter owmr/owmr_schema.json --decision order=5 --decision ship=1 --episode-seed 3
```

```
owmr v0.4  episode_seed=3
 t  order                        ship  wh_arrival                  rt_arrival                     shipped  ship_scale                      demand  action_period  wh_stock           wh_pipe                      rt_stock                               rt_pipe  wh_holding  rt_holding  rt_shortage  total  reward
 0   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [0.41,1.49,0.67,0.42,1.63]              0      5.00  [5.00,5.00,5.00]    [1.59,0.51,1.33,1.58,0.37]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        5.37         0.00   7.87   -7.87
 1   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [1.74,1.30,1.44,0.82,1.54]              1      5.00  [5.00,5.00,5.00]   [0.85,0.21,0.88,1.75,-0.17]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        3.70         0.70   6.90   -6.90
 2   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [0.60,1.18,0.84,0.57,1.44]              2      5.00  [5.00,5.00,5.00]   [1.25,0.03,1.05,2.18,-0.61]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        4.51         2.45   9.46   -9.46
 3   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [1.18,1.23,0.43,1.43,1.13]              3      5.00  [5.00,5.00,5.00]  [1.07,-0.20,1.62,1.76,-0.74]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        4.45         3.78  10.72  -10.72
 4   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [0.68,0.30,1.36,0.69,0.57]              4      5.00  [5.00,5.00,5.00]   [1.39,0.50,1.26,2.06,-0.31]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        5.21         1.26   8.97   -8.97
 5   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [0.43,0.57,1.30,1.48,1.25]              5      5.00  [5.00,5.00,5.00]   [1.97,0.92,0.96,1.58,-0.56]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        5.43         2.24  10.17  -10.17
 6   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [1.45,0.94,0.93,0.86,0.58]              6      5.00  [5.00,5.00,5.00]   [1.51,0.98,1.03,1.73,-0.14]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        5.25         0.56   8.31   -8.31
 7   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [1.60,0.21,0.57,1.23,0.85]              7      5.00  [5.00,5.00,5.00]    [0.92,1.77,1.46,1.49,0.01]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        5.65         0.00   8.15   -8.15
 8   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [0.39,0.55,0.75,1.53,0.58]              8      5.00  [5.00,5.00,5.00]    [1.52,2.22,1.71,0.97,0.42]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        6.85         0.00   9.35   -9.35
 9   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [1.45,1.30,0.67,0.50,0.84]              9      5.00  [5.00,5.00,5.00]    [1.07,1.92,2.04,1.47,0.58]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        7.09         0.00   9.59   -9.59
10   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [0.49,1.31,1.42,0.84,1.39]             10      5.00  [5.00,5.00,5.00]    [1.59,1.61,1.62,1.64,0.20]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        6.65         0.00   9.15   -9.15
11   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [0.38,1.60,0.52,1.16,1.05]             11      5.00  [5.00,5.00,5.00]    [2.21,1.01,2.09,1.48,0.15]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        6.94         0.00   9.44   -9.44
12   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [1.16,0.61,1.92,0.28,1.62]             12      5.00  [5.00,5.00,5.00]   [2.04,1.41,1.17,2.19,-0.46]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        6.82         1.85  11.17  -11.17
13   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [2.43,0.30,0.38,0.67,1.56]             13      5.00  [5.00,5.00,5.00]   [0.61,2.11,1.79,2.52,-1.02]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        7.03         4.09  13.62  -13.62
14   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [0.92,1.00,0.51,0.54,1.01]             14      5.00  [5.00,5.00,5.00]   [0.69,2.11,2.29,2.98,-1.04]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        8.07         4.15  14.71  -14.71
15   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [0.57,1.85,1.18,1.56,1.57]             15      5.00  [5.00,5.00,5.00]   [1.12,1.26,2.11,2.42,-1.61]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        6.92         6.44  15.86  -15.86
16   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [1.23,1.21,0.68,0.64,0.58]             16      5.00  [5.00,5.00,5.00]   [0.90,1.05,2.43,2.78,-1.19]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        7.16         4.75  14.41  -14.41
17   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [0.77,1.09,1.22,1.55,1.05]             17      5.00  [5.00,5.00,5.00]   [1.13,0.95,2.21,2.23,-1.24]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        6.53         4.96  13.99  -13.99
18   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [1.15,0.76,1.06,2.87,0.51]             18      5.00  [5.00,5.00,5.00]   [0.98,1.19,2.15,0.36,-0.75]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        4.68         3.01  10.19  -10.19
19   5.00  [1.00,1.00,1.00,1.00,1.00]        5.00  [1.00,1.00,1.00,1.00,1.00]  [1.00,1.00,1.00,1.00,1.00]        1.00  [2.01,1.73,0.46,0.17,0.58]             19      5.00  [5.00,5.00,5.00]  [-0.03,0.46,2.69,1.20,-0.34]  [[1.00],[1.00],[1.00],[1.00],[1.00]]        2.50        4.35         1.45   8.30   -8.30
... (80 more periods)
episode total = 2559.93   reward total = -2559.93   periods = 100
eval metrics: wh_share = 0.37   stockout_periods = 195.00   system_stock = 33.80
```

# cnv — plain-English restatement (Phase A round-trip artifact)

IR: `cnv/cnv_schema.json` (v0.4, seed scheme v2) · mdp fingerprint `6afcb1d28ddc` ·
drafted 2026-09-17 as a faithful translation of Jain, Rudi & Wang (2015),
"Demand Estimation and Ordering Under Censoring: Stock-Out Timing Is (Almost)
All You Need", *Operations Research* 63(1), 134–150, with the legacy `cnv/`
simulator consulted for its observation and reward menus only. Every
modelling choice below carries its resolved `source`.

## The problem

A newsvendor sells a perishable product over N periods. At the start of each
period she chooses how many units to stock (an integer y ≥ 0, delivered at
once, cost c per unit, revenue r per unit sold). Customers arrive during the
period as a Poisson process whose rate λ she does not know; she holds a
Gamma(α, β) prior on it and the same λ drives every period of the horizon.
Demand D is the number of arrivals; sales are min(D, y); unmet demand is lost
and never seen; leftovers are worthless. Learning λ is the whole difficulty:
when she runs out she does not see how many more would have bought, so
observations are censored, and the paper's point is that *when* she ran out
carries most of what the censoring hid.

**Objective (maximize):** the undiscounted expected sum over the horizon of
r·sales − c·y. No salvage, no holding cost, no discounting. [paper §2]

**Mode stance:** the three observation types the paper defines are rival
designs judged on **one head-to-head board** per scenario, with the paper's
loss-recovery measure as the headline. [human_confirmed]

**Tier-2 stances:** [human_confirmed]
- **confirm** — over-ordering: the optimal stocking level under censored
  observations exceeds the myopic newsvendor level (Proposition 1(b)), and
  by less under timing than under event observations across most of the
  grid (§5.1). Owes a policy probe and INTERPRET.md.
- **bypass** — the two-number sufficient statistic (cumulative sales,
  cumulative sales duration; Proposition 2): a policy fed only raw
  per-period observations with a frame stack should match the policy fed
  the statistic, so the conjugate machinery need not be handed over.

## Randomness classification

Two sources, one world latent:

1. **λ, the arrival rate** — a **hidden world latent**: drawn once per
   episode by nature from Gamma(α = prior_rate·prior_mean, rate = prior_rate),
   never revealed to the agent. In the IR it is the `demand` slot's draw
   spec on the Poisson rate, desugared to the placeholder constant
   `demand_rate`, barred from every observation mode and readable only by
   the eval metrics. [human_confirmed]
2. **demand** (slot `demand`, stream 0) — intrinsic, per period:
   Poisson(λ). [paper §3]
3. **arrival epochs** (slot `epochs`, stream 1) — intrinsic, per period:
   402 i.i.d. uniforms on [0, 1], of which the first D are the arrival
   epochs. Given the count, this is the exact law of a Poisson process on
   the unit interval, so the stock-out time — the y-th smallest of those D
   epochs when D ≥ y ≥ 1 — has the paper's distribution. Chosen over summed
   exponential gaps (which the IR's catalog form cannot parameterise by
   rate) and over the Beta(y, D−y+1) shortcut. [human_confirmed]

The 510-cell design grid is **neither**: it is the experimenter's choice of
which problem instances to train and evaluate on (design layer, `grids`).

## What the simulator carries between periods

Besides the clock, three running totals: cumulative sales S, cumulative
sales duration T (a full period when no stock-out occurred, the stock-out
time otherwise), and cumulative demand. S and T are the paper's sufficient
statistic under timing observations (posterior Gamma(α+S, β+T), eq. 7);
cumulative demand with the period count is the full-observation statistic.
No transition reads them; they exist so the belief observation modes can
render the paper's state. [human_confirmed; the shipped mab example's
pulls/payouts precedent]

Edge cases fixed by the paper's eq. (7): stocking nothing (y = 0) gives
sales 0, a stock-out flag of 1 and a stock-out time of 0, so the posterior
does not move; no stock-out gives a stock-out time of 1.

## Observation menu (gym layer, mutable)

| mode | features, in render order | memory |
|---|---|---|
| `timing_belief` (default) | time_to_go, critical ratio (r−c)/r, posterior shape α+S, posterior rate β+T | none |
| `full_belief` | time_to_go, critical ratio, α + cum_demand, β + periods played | none |
| `event_raw` | time_to_go, critical ratio, prior rate, prior mean, last sales, last stock-out flag | frame stack over the horizon |
| `timing_raw` | time_to_go, critical ratio, prior rate, prior mean, last sales, last stock-out time | frame stack |
| `full_raw` | time_to_go, critical ratio, prior rate, prior mean, last demand | frame stack |

Event has no fixed-size sufficient statistic (paper §3), so it has only the
raw encoding. The legacy "sales" mode is dropped: sales plus the stocking
level already implies the stock-out flag.

**Decision:** `order`, integer in [0, arrivals_buffer = 402]; one discrete
action mode. The cap is the simulator's own arrival cap, derived from the
marginal demand law of the widest grid cell, never from a solved policy.
[human_confirmed]

**Reward:** `profit` = the per-period objective. The legacy per-step regret
mode is dropped (it would leak demand to a memory policy); a terminal gap
mode may return later as a gym-layer lever.

## The scenario set [human_confirmed, composition B of `cnv.scenarios.md`]

Constants: horizon_N, price r = 2, unit_cost c, prior_rate β, prior_mean
E[λ], arrivals_buffer 402.

| instance | N | c | 1/β | E[λ] | paper role |
|---|---|---|---|---|---|
| base | 4 | 1.0 | 1 | 10 | Fig. 5(a) cell (α = 10, β = 1) |
| fig3_c10 | 4 | 1.0 | 16 | 10 | Fig. 3(a)/4(a) family at ratio 0.5 |
| fig3_c15 | 4 | 1.5 | 16 | 10 | same family at ratio 0.25 |
| fig6a | 100 | 1.0 | 16 | 10 | Fig. 6(a), long-horizon myopic simulation |
| fig6b | 100 | 1.5 | 16 | 10 | Fig. 6(b) |

Grids: `paper510` = unit_cost (17) × prior_rate (6) × prior_mean (5) at
N = 4, the §5.1 study; `fig3` = the 17-point unit-cost sweep at
(1/β = 16, E[λ] = 10).

**Training target:** a **generalist over `paper510`**, one policy per
observation type, trained on cells sampled uniformly, evaluated cell by
cell on shared seeds against each cell's own bars. The N = 100 instances are
separate leaderboards, never compared with the N = 4 boards.
[human_confirmed]

**Benchmarks declared:** `myopic` (feasible; the paper's myopic timing /
full / event policies), `dp` (exact at N = 4 for timing and full; the
timing DP discretizes the continuous rate coordinate, basis recorded at
build), `perfect` (relaxed; the paper's perfect-knowledge model P, emitted
by the eval as the `perfect_profit` column). **Report-only metrics:**
`hindsight_profit` (clairvoyant on this period's demand, the legacy
regret denominator) and `perfect_profit`.

## Assumptions filled in without being told

- arrivals_buffer = 402 caps demand at the 1e-8 marginal quantile of the
  widest cell (1/β = 16, E[λ] = 50); the legacy buffer of 26–86 capped
  demand materially in the diffuse-prior cells. [derived]
- Period indexing 0-based; horizon end is `terminated`. [derived]
- Seed scheme v2: demand on stream 0, epochs on stream 1, λ on meta
  substream 0. [derived]
- `frame_stack` = 4 = N for the primary target; the belief modes run
  without a stack. [human_confirmed]

## Invariants (transcribed claims; advisory here, fatal at the Stage-1 gate)

sales are censored demand · demand within the buffer · stock-out iff all
sold · stock-out time inside the period · no stock-out means full duration ·
zero stock means zero duration · the three running-total balances · profit
identity. All ten pass the IR laws over 4 policies.

## Sample trajectories

The base cell (α = 10, β = 1, c = 1), stocking 10 every period:

```step7b
python -m mdp_ir.interpreter cnv/cnv_schema.json --decision order=10 --episode-seed 3
```

```
cnv v0.4  episode_seed=3
t  order  demand  sales  stockout_event  stockout_time  action_period  cum_sales  cum_duration  cum_demand  revenue  stocking  total  reward
0  10.00      10     10               1           0.97              0         10          0.97          10    20.00    -10.00  10.00   10.00
1  10.00       6      6               0           1.00              1         16          1.97          16    12.00    -10.00   2.00    2.00
2  10.00      13     10               1           0.70              2         26          2.67          29    20.00    -10.00  10.00   10.00
3  10.00       8      8               0           1.00              3         34          3.67          37    16.00    -10.00   6.00    6.00
episode total = 28.00   reward total = 28.00   periods = 4
eval metrics: hindsight_profit = 37.00   perfect_profit = 28.00
```

Reading it: period 0 sold out (10 of 10) at time 0.97, so the timing
statistic moved to (10, 0.97); period 1 did not sell out, so a full period
of duration was added; period 2 had 13 arrivals against 10 stocked and the
tenth customer arrived at 0.70. Profit is r·sales − c·y each period; the
hindsight column stocks each period's demand exactly, the perfect column
knows λ and stocks the Poisson newsvendor quantity.

The paper's Fig. 3 cell (α = 10/16, β = 1/16, c = 1), stocking nothing:

```step7b
python -m mdp_ir.interpreter cnv/cnv_schema.json --instance fig3_c10 --decision order=0 --episode-seed 3
```

```
cnv v0.4  episode_seed=3 instance=fig3_c10
t  order  demand  sales  stockout_event  stockout_time  action_period  cum_sales  cum_duration  cum_demand  revenue  stocking  total  reward
0   0.00      13      0               1           0.00              0          0          0.00          13     0.00     -0.00   0.00    0.00
1   0.00      20      0               1           0.00              1          0          0.00          33     0.00     -0.00   0.00    0.00
2   0.00      20      0               1           0.00              2          0          0.00          53     0.00     -0.00   0.00    0.00
3   0.00       9      0               1           0.00              3          0          0.00          62     0.00     -0.00   0.00    0.00
episode total = 0.00   reward total = 0.00   periods = 4
eval metrics: hindsight_profit = 62.00   perfect_profit = 44.00
```

Reading it: with nothing stocked every period is a stock-out at time 0,
sales and duration never move, and the posterior stays at the prior — the
paper's eq. (7) edge case.

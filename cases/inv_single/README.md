# inv_single — single-item periodic-review inventory control

Finite-horizon single-item inventory — Scarf's textbook (s, S) setting, with
lead-time, stochastic-lead-time and lost-sales variants as separate instances —
generated from [`inv_single_schema.json`](inv_single_schema.json) by
**auto-mdp-solver**. It is the pipeline's **correctness fixture**: exact DP
optima exist on both sides of the fixed-cost line and, since this campaign
built one, on both sides of the lost-sales line too, so a policy can be scored
against a known optimum instead of against another arm.

**TL;DR**

- **Where an exact optimum exists, tuned PPO reaches it.** 99.86–99.96% of the
  DP on all four such cells, from 93.1% before tuning.
- **One of those optima did not exist before this campaign.** Lost sales under
  a two-period lead time was on record as having no compact solution; against
  the exact DP built here, textbook base-stock is 8.66% above optimal and PPO
  closes 99.4% of that gap.
- **Where no optimum exists, the learned policy is the bar.** Under stochastic
  lead time PPO beats every position-based rule, and its structure compresses
  to one weighted inventory number that beats the crossover literature's
  whole correction.
- **A generalist transfers along the axes it reads, not the ones that change
  the state.** Over a 16-cell cost × lead-time grid it holds 96.5% of each
  cell's own DP, interpolates unseen cost regimes at no loss, and fails at an
  unseen lead time (55.1%).

## The problem

A single stocking point holds one item over a finite season of **T = 30**
periods. Each period the manager observes the stock on hand and the orders
still in transit, then places a replenishment order. That order arrives after a
**lead time** L — immediately at L = 0, otherwise L periods later — after which
demand for the period arrives and is served from whatever is on the shelf.
Demand that cannot be served is either **backlogged** (served later, at a
penalty per unit per period) or **lost** (gone, at a penalty per unit),
depending on the instance.

Three costs accrue: **h** per unit of leftover stock per period, **b** per unit
short, and a **fixed cost K** charged on any period in which an order is placed
at all, regardless of size. The objective is to minimize the expected
undiscounted sum of the three over the horizon (β = 1.0). Holding and shortage
pull in opposite directions and the fixed cost pulls against ordering at all,
so the manager is trading a known carrying cost against an unknown shortage.

What makes it hard is the interaction of three things. **Lead time** means today's
order is aimed at demand L periods out, so the decision is made on a forecast
and the relevant state is not stock but *inventory position* (stock plus
everything in transit). **The fixed cost** destroys the smoothness of the
problem: with K > 0 the optimal policy is no longer "top up every period" but
"wait until stock falls far enough, then batch" — a discrete trigger that a
smooth order-up-to rule cannot express. And **the finite horizon** makes the
optimal rule non-stationary: near period T there is no future left to hold
stock for, so the right order-up-to level falls away as the season ends.

What is known analytically is unusually complete, which is why this domain is
the pipeline's correctness fixture. With **K = 0** an order-up-to (base-stock)
policy is optimal, and its level is a newsvendor fractile. With **K > 0**
Scarf's classical result makes an **(s, S)** policy optimal — order up to S when
inventory position falls to or below s, otherwise do not order — and for a
finite horizon both thresholds are functions of the period. Under **backlogging
with deterministic lead time** the whole problem collapses onto inventory
position as a sufficient statistic and a backward-induction DP solves it
exactly. That exact DP is the bar every leaderboard below is measured against.
Two regimes break it and are genuinely open: **lost sales with positive lead
time** (no compact sufficient statistic is known; base-stock is a heuristic, not
an optimum) and **stochastic lead time**, where orders can cross and inventory
position stops being sufficient at all.

**Source.** A textbook periodic-review inventory problem — Scarf's (s, S)
setting with the finite-horizon, lead-time and lost-sales variants added as
separate instances. The authoritative statement of the problem is
`inv_single_schema.json`; `inv_single.restatement.md` renders it in prose.

## Layout

Documents — where to read about the case:

| doc | what it holds |
|---|---|
| `inv_single.restatement.md` | the Phase-A restatement: the problem as the IR states it, re-rendered whenever the model moves (step 7b) |
| `ESCALATION.md` | the campaign — design tree (§MAP), changelogs, §CONFIG-REGISTRY, numbered findings (#E1–#E9) with verdicts. Campaign 1's retired log is at `git show 27e0546:inv_single/ESCALATION.md` |
| `INTERPRET.md` | the policy readback (spec §14): what the trained nets actually do. **One section per research question** — RQ-1 and RQ-4 are answered there; `bypass` questions owe no section. Written 2026-08-31 |
| `figures/` | the committed policy-overlay SVGs the readback cites |
| `PLAYBOOK.md` | the guide-§10 digest, written 2026-09-12. One entry — FM1, the generalist protocol: sample to train, enumerate to score, hold out a finer grid stratified by axis. Methodology only; every empirical finding stays here (RQ-7's is in `2-structural`, Q-A/Q-B in `3-methodology`) and upstream prior art is cited, not restated |
| `CLAUDE.md` | operating brief for an agent changing this folder |

Code — where things are implemented:

| module | role |
|---|---|
| `inv_single_schema.json` | **the IR — authoritative** for the problem definition |
| `inv_single_uncertainty.py` | stochastic primitives: `SamplingContext`, the demand and leadtime generators, including the latent-bearing ones |
| `inv_single_scenarios.py` | world layer: `InvSingleScenario`, `InvSingleScenarioSource`, the `SCENARIOS` registry |
| `inv_single_grids.py` | design layer: `GRIDS` — the `grid16`, `lt_variance_k0` and `lt_variance_k20` cell samplers |
| `inv_single_mdp.py` | pure simulator: `InvSingleState`, `init_state`, `advance` — no Gym, no reward |
| `inv_single_gym.py` | the Gymnasium wrapper (`InvSingleEnv`): spaces, observation modes, reward |
| `inv_single_ordinal_head.py` | the mixture-at-zero order head — the arch lever both crowns use |
| `inv_single_configs.py` | the §CONFIG-REGISTRY data half; `--config sc0/g4/a1/h2` resolves through it |
| `inv_single_ppo_train.py` | PPO training (spec §8) |
| `inv_single_select.py` | post-hoc checkpoint screen — the §9.7 screen layer |
| `inv_single_ppo_eval.py` | PPO evaluation on the protocol block (spec §9) |
| `inv_single_eval_common.py` | batched CRN rollout + record writing, shared by every eval |
| `inv_single_benchmark_dp.py` | finite-horizon DP solver (backward induction), cached per scenario |
| `inv_single_benchmark_dp_eval.py` | DP evaluation + TensorBoard logging |
| `inv_single_benchmark_heuristic.py` | the rule arms: `myopic`, `basestock_opt`, `constant`, `zero`, `random` |
| `inv_single_benchmark_dp_lostsales.py` | the **exact lost-sales DP** (#E17): backward induction on `(u, p1)` under `O-R-D`. The bar for `lt_lost_sales`, verified two ways |
| `inv_single_dp_lostsales_probe.py` | reads structure off that DP — `--shape`, `--sufficiency`, `--direction`, `--horizon`; every reading visitation-weighted |
| `inv_single_dp_exactness.py` | second implementation: exact `(u, p1)` DP, gated by the test suite |
| `inv_single_policy_probe.py` | the §14 readback probe — action surface, structural fit, fitted-rule scoring, and `--composition` for whether inventory position is a sufficient statistic (#E12) |
| `inv_single_distill_bottleneck.py` | distils a trained policy through a linear information bottleneck on `(inv, pipe)` — how many numbers the state compresses to, and the weights (#E13) |
| `inv_single_distill_generalist.py` | distils a crowned **grid** generalist through a context-aware bottleneck — the three-number family rule (#E20) |
| `inv_single_benchmark_weighted.py` | order-up-to on a *weighted* inventory position. `basestock_opt` is its `w=(1,1,1)` member, reproduced to the cent |
| `inv_single_plot_variance_gap.py` | RQ-4's family figure: what the pipeline is worth against lead-time variance, with the `Var(L)=0` learnability control as the zero line. Summarises each rung by its MEAN over arms, never best-vs-best |
| `inv_single_plot_composition.py` | RQ-4's figure: order deviation at fixed inventory position against how late the outstanding stock lands. Harvests states through the probe's own roll, so figure and ledger cannot drift |
| `inv_single_plot_policy.py` | the policy figures in canonical coordinates (§14.3) |
| `inv_single_ir_adapter.py` | differential adapter (portable-domain contract) |
| `inv_single_test.py` | the domain's own tests (spec §1.2) — **60** at the last gate run (v0.10.4, 2026-09-12) |
| `inv_single_policy.py` | the deployable wrapper (spec §12): observation vector in, order quantity out. Imports `decode_action` from the gym rather than re-deriving it, and `inv_single_ordinal_head` for the custom policy class. `__main__` replays through the raw `inv_single_mdp` loop and reproduces the eval per-seed costs exactly |

## Results

### Protocol

Every number below is quoted at: **8192 seeds from `--first-seed 0`**, **CRN
across arms** (the same seed block drives every arm and every benchmark, so
per-seed differences are paired), **mean total cost, lower is better**. The
reference bar is named per board. Record eval = **deterministic argmax**;
`--stochastic` exists but appears only as a labelled figure, never as a record.
Screen scores (`inv_single_select.py`, seeds from 1 000 000) are **never**
quoted — the blocks are disjoint (spec §9.7).

Symbols used below:

| symbol | meaning |
|---|---|
| **IP** | inventory position = on-hand stock + sum of the in-transit pipeline |
| **(s, S)** | reorder point `s` and order-up-to level `S`: order up to `S` when IP ≤ `s`, else do not order |
| **`% of bar`** | bar ÷ arm, in percent. Cost is minimized, so 100% means the arm equals the bar and lower is worse |
| **`sc/g/a/h`** | the §CONFIG-REGISTRY id grammar — scenario / gym / arch / hyperparameter axis, resolved by `inv_single_configs.py` |
| **`L0…L4`** | escalation level: `L0` no-tuning control, `L1` the §8.6 derivation, `L2(gym)` + gym levers, `L4(hp+gym+arch)` + tuned hyperparameters and the ordinal head |
| **SE** | standard error of the mean over the seed block |

### `1-comparative` — how does RL compare with the existing solutions?

The scenarios, and the role each plays in the study. All have T = 30 and
β = 1.0; h = 1.0 and b = 9.0 except where the grid varies them.

| scenario | config | what it is for |
|---|---|---|
| `simple` | Poisson(10), L=0, K=0, backlog | the base case: base-stock is provably optimal, so it measures whether RL can *reach* a known bar |
| `simple_k` | Poisson(10), L=0, K=20, backlog | RQ-1's target: the fixed cost opens an (s, S) gap that order-up-to cannot close |
| `lt` | Poisson(10), L=2, K=0, backlog | positive lead time; IP is the sufficient statistic — RQ-2's target |
| `rdo` / `rod` | Poisson(10), L=2, K=0, backlog | the two alternative within-period event orders (R-D-O, R-O-D) |
| `lost_sales` | Poisson(10), L=0, K=0, lost | RQ-3's **control**: at L=0 it coincides with `simple` at the optimum |
| `lt_lost_sales` | Poisson(10), L=2, K=0, lost | RQ-3's target: base-stock is **8.66% above optimal** here — the exact bar is `dp_lostsales` (#E17) |
| `slt` | Poisson(10), L∈{1,2,3} stochastic, K=0, backlog | RQ-4's target: orders cross, so IP is not sufficient and the DP's own state is misspecified |
| `discrete` / `discrete_lost_sales` | sampled(3, 8–12), L∈2–5 | latent-bearing demand: 3 integer values drawn per episode with random weights |
| `poisson` / `poisson_lost_sales` | Poisson(λ), λ~Gamma(9,0.3), L∈2–5 | latent-bearing demand: a hidden rate drawn per episode |
| `mix_demand` | 0.5 discrete + 0.5 poisson, L∈2–5 | the mixture source |
| `grid16` | 16 cells over (b ∈ {1,4,9,19}) × (K ∈ {0,20}) × (L ∈ {0,2}) | RQ-6's target: one generalist against 16 per-cell exact DPs |
| `cost_leadtime` | 54 cells over (b ∈ 1..9) × (K ∈ {0,20}) × (L ∈ {0,1,2}) | RQ-7's target: the **evaluation** grid for a `grid16`-trained generalist. 12 cells overlap `grid16` (memorization control), 42 are held out in strata that separate `b` from `LT` |
| `lt_variance_k0` / `lt_variance_k20` | 45 cells each: lead-time **law** × (b ∈ 1…9), K fixed per grid | RQ-4's generality target: does one linear statistic serve a whole family? |

**Every scenario is a separate leaderboard, and so is every `grid16`,
`lt_variance_k0` and `lt_variance_k20` cell.**
They differ in lead time, stockout mode, demand family and fixed cost, so a
score is never comparable across rows and an aggregate over them is not a
number. `simple` and `slt` are different problems, not two settings of one.

The latent-bearing sources (`discrete*`, `poisson*`, `mix_demand`) put the world
latent on the demand generator itself (spec §5.2), realized per episode on the
demand slot's own stream — so instances sharing a demand model see identical
latent draws per episode seed.

#### Headline — every cell with an exact bar

Bar = the exact DP. All rows are the tuned artifact re-scored on the protocol
block. Three of the four are **crowned** — `lt_lost_sales` was crowned by #E22,
which cleared RQ-3's last deliverable (the probe and the `INTERPRET.md`
readback). `lt` is **covered** rather than crowned (#E10) — its stance, RQ-2,
is `bypass`, which owes an outcome comparison and not an optimized artifact.

| scenario | exact DP | artifact | cost | `% of bar` | best heuristic | status |
|---|---:|---|---:|---:|---:|---|
| `simple` | 175.97 | `sc0/g4/a1/h2` | 176.11 | **99.92%** | 175.97 | ★ crowned |
| `simple_k` | 618.71 | `sc1/g4/a1/h3` | 618.96 | **99.96%** | 775.94 | ★ crowned |
| `lt` | 549.88 | *unregistered* | 550.63 | **99.86%** | 549.88 | ✓ covered |
| `lt_lost_sales` | 415.79 | *unregistered* | **416.01** | **99.95%** | 418.78 | ★ crowned |

`slt` has no exact bar (the shipped DP is mis-specified there) and is listed on
its own board; `lost_sales` is covered **by collapse** (#E14) rather than by an
arm, since at L=0 it is the same problem as `simple` on-policy.

RQ-1 opened at 93.1% / 97.7% (#E2). **Every artifact above runs the ordinal
(mixture-at-zero) order head** at `L4(hp+gym+arch)` — `a1`, adopted #E9 and
**re-confirmed at a tuned centre in #E21**, where a 213-trial head-to-head
against the flat head at `lt_lost_sales` gave +0.560 (t=4.9) at `vec` and
+0.407 (t=3.7) at `vec_ip`. Its value is **robustness across hyperparameter
draws** — the flat head's median is 1.70 worse at `vec` — not peak cost, where
the difference is half a unit on a 416 bar.

**Shipped: the tuned artifacts themselves** — `sc0/g4/a1/h2` and
`sc1/g4/a1/h3` — not retrains of their configs. Spec §8.6 says re-evaluate the
artifact and prefer shipping it; if one is ever retrained, its config must be
read off the trial's own args log, never a printed cfg dict. The two cells share
`g4`/`a1` and differ only in `h`: `simple` wants `ent_coef` 5.78e-6, `simple_k`
4.57e-8 — **opposite exploration settings**, because one optimum is a single
sharp order-up-to level where entropy is pure cost, and the other has a discrete
order/don't trigger worth exploring for (#E8).

Two arms are absent from the boards below and are absent deliberately.
**`action_mode=hurdle`** was measured and lost on both targets — 180.81 and
626.44, i.e. +1.62 and +2.03 against the flat head at the same recipe (#E7) —
and is *reported rather than pruned*, because action encoding is a `discover`
stance (the `3-methodology` section's Q-B) and every encoding drawn owes a number permanently.
**`action_mode=continuous`** lost heavily in #E4, but every number was measured
before F5 changed the observation, so it is not comparable to the current frame
and is parked until re-run.

#### `simple` — L=0, K=0, backlog

Bar = `benchmark_dp` (exact). Sorted by mean cost, ascending.

| arm | mean cost | SE | `% of bar` |
|---|---:|---:|---:|
| `benchmark_dp` | 175.97 | 0.39 | 100.0% |
| `basestock_opt(S=14)` | 175.97 | 0.39 | 100.0% |
| `myopic` | 175.97 | 0.39 | 100.0% |
| **PPO** `sc0/g4/a1/h2` (shipped) | 176.11 | — | 99.92% |
| PPO L2(gym), median of 4 seeds | 182.94 | 0.39 | 96.2% |
| PPO L1, median of 4 seeds | 188.92 | 0.42 | 93.1% |
| PPO L0 control, 1 seed | 209.83 | 0.39 | 83.9% |
| `constant` | 1,442.66 | 17.04 | 12.2% |
| `random` | 5,845.49 | 14.50 | 3.0% |
| `zero` | 41,912.59 | 30.64 | 0.4% |

`L2(gym)` is `g4` — `g0` with observation normalization off. Every seed
improves paired against L1 (−4.20 to −23.54, all |t| > 15) and the seed spread
collapses from 19.06 to 1.33: obs-norm was preventing runs from training at all,
not costing a uniform few percent (#E2, #E3).

#### `simple_k` — L=0, K=20, backlog

Bar = `benchmark_dp` (exact). Sorted by mean cost, ascending.

| arm | mean cost | SE | `% of bar` |
|---|---:|---:|---:|
| `benchmark_dp` | 618.71 | 0.47 | 100.0% |
| **PPO** `sc1/g4/a1/h3` (shipped) | 618.96 | — | 99.96% |
| PPO L2(gym), median of 4 seeds | 625.95 | 0.48 | 98.8% |
| PPO L1, median of 4 seeds | 633.22 | 0.54 | 97.7% |
| PPO L0 control, 1 seed | 664.92 | 0.67 | 93.0% |
| `basestock_opt(S=14)` | 775.94 | 0.39 | 79.7% |
| `myopic` | 775.94 | 0.39 | 79.7% |
| `constant` | 2,042.66 | 17.04 | 30.3% |
| `random` | 6,407.03 | 14.39 | 9.7% |
| `zero` | 41,912.59 | 30.64 | 1.5% |

This is the board RQ-1 exists for. The best fitted base-stock leaves 157 units
on the table; the shipped artifact leaves 0.25. **Order-up-to cannot batch;
PPO evidently can** (#E2, #E3, #E9).

#### `lt` — L=2, K=0, backlog

Bar = `benchmark_dp` (exact — deterministic lead time, backlog). Sorted by mean
cost, ascending.

| arm | mean cost | SE | `% of bar` |
|---|---:|---:|---:|
| `basestock_opt(S=37)` | 549.88 | 1.10 | 100.0% |
| `benchmark_dp` | 549.88 | 1.10 | 100.0% |
| `myopic` | 549.88 | 1.10 | 100.0% |
| **PPO** (`vec_ip`, tuned) | **550.63** | 1.10 | **99.86%** |
| **PPO** (`vec`, tuned) | 551.57 | 1.10 | 99.69% |
| `constant` | 5,467.75 | 28.83 | 10.1% |
| `random` | 13,999.00 | 25.22 | 3.9% |
| `zero` | 41,912.59 | 30.64 | 1.3% |

Both PPO rows are tuned **per observation mode** (151 trials, #E10) — the
contrast is RQ-2's, and tuning each mode separately is what makes it valid.

#### `lost_sales` — L=0, K=0, lost

Bar = `basestock_opt`, and here it **is** the optimum: at L=0 the state is
scalar on-hand and the one-period cost is convex in the order-up-to level, so
order-up-to is optimal — and #E1 measured the fitted `S=14` landing on the same
175.97 the backlog DP computes. The *shipped* `dp` is still not the route to it
(its recursion is backlog-only); the bar is the independent fit.
Sorted by mean cost, ascending. **PPO covered by collapse (#E14).**

| arm | mean cost | SE | `% of bar` |
|---|---:|---:|---:|
| `basestock_opt(S=14)` | 175.97 | 0.39 | 100.0% |
| `myopic` | 175.97 | 0.39 | 100.0% |
| **PPO — `sc0/g4/a1/h2` transferred, no retraining** | **176.09** | 0.39 | **99.9%** |
| `constant` | 340.03 | 1.36 | 51.8% |
| `zero` | 2,704.02 | 1.72 | 6.5% |
| `random` | 5,905.42 | 13.96 | 3.0% |

`lost_sales` and `simple` coincide at the optimum — 175.97 for base-stock and
myopic in both — though the problems differ everywhere off-policy (`zero` costs
2,704 here against 41,913 at `simple`). At zero lead time an order-up-to policy
restores stock before demand arrives, so backlog never persists and the two cost
recursions agree on-policy (#E1). An arm hitting that number here has shown
nothing about lost-sales structure: this scenario is RQ-3's **control**, not its
target.

#### `lt_lost_sales` — L=2, K=0, lost

Bar = **`dp_lostsales`, an exact optimum** (#E17). Sorted by mean cost,
ascending; `% of optimal` is `optimum / arm`.

| arm | mean cost | SE | `% of optimal` | closes of base-stock's gap |
|---|---:|---:|---:|---:|
| **`dp_lostsales` (exact)** | **415.79** | 0.74 | **100.0%** | — |
| **PPO tuned, `vec_ip`** (#E18) | **416.01** | 0.74 | **99.95%** | **99.4%** |
| **PPO tuned, `vec`** (#E18) | 416.12 | 0.75 | 99.92% | 99.1% |
| PPO tuned, flat head, `vec_ip` (#E21) | 416.42 | 0.75 | 99.85% | 98.3% |
| PPO tuned, flat head, `vec` (#E21) | 416.68 | 0.75 | 99.79% | 97.5% |
| `capped_basestock_opt(S=35,cap=12)` | 418.78 | 0.74 | 99.29% | 91.7% |
| PPO carried recipe, 6M (#E15) | 419.12 | 0.75 | 99.21% | 90.8% |
| PPO carried recipe, 4M (#E15) | 419.44 | 0.75 | 99.13% | 89.9% |
| PPO L0 floor, best | 443.46 | — | 93.76% | 23.2% |
| `basestock_opt(S=35)` | 451.82 | 0.73 | 92.03% | 0% |
| `myopic` | 462.76 | 0.67 | 89.9% | −30.4% |
| `constant` | 491.57 | 1.30 | 84.6% | −110.5% |
| `zero` | 2,704.02 | 1.72 | 15.4% | — |
| `random` | 14,405.71 | 25.81 | 2.9% | — |

**This is the cell where base-stock is provably not optimal, and now the
statement has a number: it leaves 36.02, or 8.66%, on the table.** That gap is
RQ-3's entire prize. The carried recipe took 90.8% of it (#E15); per-mode
tuning at 4M took **99.4%**, landing 0.22 above a verified optimum (#E18).

Two rows deserve reading together. **`capped_basestock` is Xin (2021)'s
two-parameter family** `q = min(cap, (S−IP)⁺)`, and adding one number to
base-stock recovers **91.7%** of the gap on its own — the optimal policy at this
cell *is* capped base-stock in shape (#E17). Tuned PPO beats it by −2.77
(t=−17.0), so the network is not merely rediscovering the rule. And the
**flat-head rows** are the #E21 control: the head is worth ~0.4–0.6 here, which
is small next to the 36 that the cap and the pipeline are worth.

Until 2026-09-03 this board said "no exact DP exists". That was true of the
*shipped* `dp` — its recursion is backlog-only — but it had been
over-generalized to the problem. At L=2 under `O-R-D` the exact recursion is
two-dimensional in `(u, p1)`; see the DP-notes table below and #E17.

#### `slt` — L ∈ {1,2,3} stochastic, K=0, backlog

Bar = `benchmark_dp`, which here is a **`feasible` arm, not an optimum**: once
orders cross, IP is not a sufficient statistic and the DP's own state is
misspecified. Sorted by mean cost, ascending.

| arm | mean cost | SE | `% of bar` |
|---|---:|---:|---:|
| **PPO** (`vec`, tuned) | **723.83** | 1.90 | **111.7%** |
| **PPO** (`vec_ip`, tuned) | 730.48 | 1.93 | 110.7% |
| `weighted_basestock w=(1,1,0.70) S=40` | **730.93** | 2.16 | 110.7% |
| `weighted_basestock w=net(1.02,1.00,0.86) S=41` | 732.61 | 2.18 | 110.4% |
| `basestock_opt(S=41)` | 737.96 | 2.21 | 109.6% |
| `benchmark_dp[table=lt]` | 798.13 | 2.70 | 101.3% |
| `benchmark_dp` | 808.78 | 1.96 | 100.0% |
| `myopic` | 819.36 | 1.96 | 98.7% |
| `constant` | 5,485.31 | 28.39 | 14.7% |
| `random` | 23,127.33 | 37.89 | 3.5% |
| `zero` | 41,912.59 | 30.64 | 1.9% |

A fitted base-stock **beats** the DP here by 70.82. Never quote "% of DP" at
`slt` without saying which table, and never pass `--reference` to `mdp_gates`
for this scenario.

**`weighted_basestock` is the best known RULE for this cell** (#E13): order up
to 40 on `inv + pipe₀ + pipe₁ + 0.70·pipe₂`. It beats `basestock_opt` by
**−7.03 ± 0.46** (t=−15.4), and `basestock_opt` is exactly its `w=(1,1,1)`
member — the search returns `S*=41` and 737.96 to the cent — so the contrast is
like-for-like rather than two implementations being compared. A single
coefficient beats the entire correction the crossover literature offers, which
keeps unweighted IP and adjusts only `S`.

**`723.83` is the best known policy for this cell** (#E11) — there is no
verified bar, so that is a leaderboard statement, not a distance from an
optimum. All 12 tuned artifacts beat the base-stock bar. `vec` beats `vec_ip`
by **−6.65 (t=−15.1)**, the sign flip against `lt`'s +0.94, and #E12 prices
what the pipeline information is worth: **11.36**.

#### `lt_variance_k0` / `lt_variance_k20` — 45 cells each

Two grids, 45 cells each: the lead-time **law** `[p, 1−2p, p]` on support
{1,2,3} crossed with shortage cost `b ∈ {1,…,9}` (critical fractile `b/(b+1)`,
0.50 → 0.90). `E[L] = 2` at *every* cell while `Var(L) = 2p` spans its **full
admissible range**:

| p | 0 | 1/6 | 1/4 | 1/3 | 1/2 |
|---|---|---|---|---|---|
| law | `[0, 1, 0]` | `[.17, .67, .17]` | `[.25, .5, .25]` | `[⅓, ⅓, ⅓]` | `[.5, 0, .5]` |
| `Var(L)` | 0 | 1/3 | 1/2 | 2/3 | 1 |
| slug | `p00` | `p17` | `p25` | `p33` | `p50` |

`p ≤ 1/2` is forced — beyond it `1 − 2p` is a negative probability — so both
ends are the family's own extremes, not arbitrary stopping points. Worth knowing
that `DiscreteLeadtime` catches this by an explicit non-negativity assert and
*not* by normalisation, since `[2/3, −1/3, 2/3]` already sums to 1.

**Length is held fixed on purpose.** Moving `E[L]` would move the order-up-to
level and the whole cost scale, so a change in the recovered weights could not
be attributed to variability. The crossover mechanism is a property of the
variance, not the mean — holding the mean is what makes the sweep a measurement.

**`K` is not an axis — it is what splits the two grids.** At `K=0` base-stock
is the optimal structure; at `K>0` it is `(s,S)`. One generalist spanning both
would have to switch policy *structure* mid-family, and a cross-`K` row
comparison would be meaningless — the same rule that makes every scenario its
own leaderboard. So `K=0` and `K=20` are two grids, two generalists, two
leaderboards — and `K=20` is the fixed cost `simple_k` and `grid16` already
use, so the regime is comparable with the rest of the campaign. `K` still appears in each cell id, so a row always carries the
regime it was scored under.

**`b` is uniform in training and enumerated at evaluation.** `as_sampler()`
draws cells with equal weight, so a generalist sees `b` uniform on 1…9; the
`§9.6` per-cell leaderboard still reports all nine separately.

`p=0` is degenerate at L=2 and reproduces `lt`'s dynamics; `p=1/3` is uniform
and *is* `slt`. Both are already measured (#E13: far-slot weight 1.00 and
≈0.70), so the sweep is bracketed **with a measured interior point** rather than
only at its ends. `p=1/2` is the maximum-variance endpoint and is **bimodal** —
all mass on {1,3}, none on the mean itself — so a jump there is not evidence
about variance alone and must not be read as such. Every cell keeps support
{1,2,3} hence `pipeline_len = 4`, so the observation vector never changes shape
and one policy spans the family without padding.

**No exact DP bar exists here** — once orders cross, inventory position is not a
sufficient statistic and the shipped DP is mis-specified (#E1). Cells are scored
against `weighted_basestock` and each other, never `% of DP`. `cells_depadded()`
**refuses** these grids: collapsing a cell to a deterministic twin at its mean is
right for `grid16`'s padding and would erase exactly the variance this grid
sweeps (gated by `test_cells_depadded_refuses_a_genuinely_stochastic_grid`).

Expressible only from solver **v0.10.2** — its base instance selects a
non-default candidate, which earlier versions rejected (issue #75, filed from
this campaign).

##### Results

**`lt_variance_k0` is run and crowned** (#E16, #E19). **`lt_variance_k20` is
out of scope by decision, not by oversight** — cut 2026-09-14 to an `optional`
open extension (§FRAME-CHANGELOG), because `K>0` adds a *trigger* on top of the
*statistic* this family was built to test, which is a second question and not an
extension of the first. It is untouched and has **no measured anchor**: no
stochastic-lead-time cell with a fixed cost has been trained here, so nothing
below transfers to `K>0`, and the distilled weight vector is measured only where
ordering is frictionless. The grid stays declared in the IR and built by
`inv_single_grids.py`, so running it costs a training budget and no
formalization.

Cells are separate leaderboards; the rung table below is the family-level
summary, and the per-cell rows live in each arm's
`ppo_eval_lt_variance_k0__{slug}__{ckpt}.tsv`.

| rung | `vec_ctx_slt` | `vec_ip_ctx_slt` | gap at `Var(L)=0` | % of exact DP there |
|---|---:|---:|---:|---:|
| L0 — floor | 1313.69 | 1782.04 | — | — |
| L1, §8.6-derived | 856.11 | 775.21 | +75.01 | 47.0% / 51.9% |
| L1, carried levers | 506.07 | 485.82 | +23.30 | 92.1% / 97.7% |
| **tuned** — crowned | **475.76** | 478.54 | **−0.14** | **98.9% / 99.1%** |

45-cell means on the protocol block at **2048 seeds/cell** — the whole grid is
scored on that block, so rows here are exact against each other and only
*indicative* against the 8192-seed specialist boards above.

**The `Var(L)=0` column is a control, not a result** — a deterministic lead
time makes inventory position sufficient, so the two modes carry identical
information and their gap there must be zero. It reaches **−0.14** only after
tuning, and everything above zero is learnability that has to be read out before
the rest means anything.

What the family then says, what one policy across 45 cells costs against
per-cell specialists, and the three-number rule distilled from the crowned
policy: **`INTERPRET.md` → RQ-4.7**, with the figure.

**Crowned:** `k0_vec_a` trial 19 — lr 5.43e-4→5.43e-5, `n_steps` 256,
`net_arch` [256,256], `gae_lambda` 0.834, `ent_coef` 3.21e-3, on the carried
levers (ordinal head, `norm_obs` off, `normalize_advantage` off). An
unregistered cell: `vec_ctx_slt` is a design axis and a deviation may never
spell one.

**No exact DP off the `p=0` slice** — once orders cross, inventory position is
not a sufficient statistic and the shipped DP is mis-specified (#E1). Never
quote "% of DP" for a cell with `Var(L) > 0`.

#### `grid16` — 16 cells, 16 separate leaderboards

Bar = each cell's **exact DP** — all 16 are backlog + deterministic LT, so the
`dp` benchmark's `exact` role holds at every one. **There is no legitimate
aggregate over these cells** and none is quoted as a score: the DP spans
75.05 to 1289.94, a 17× range, so a mean is dominated by `b19_K20_lt2` and
silent about `b1_K0_lt0`. The mean-over-cells appears below only as the
*selector* scale the tuner optimised (#E23).

| cell | exact DP | PPO tuned `vec_ctx` | `% of optimal` | PPO L1 carried | `% of optimal` |
|---|---:|---:|---:|---:|---:|
| `b1_K0_lt0` | 75.05 | **90.3** | 83.1% | 169.6 | 44.3% |
| `b4_K0_lt0` | 138.33 | 165.1 | 83.8% | 209.7 | 66.0% |
| `b1_K0_lt2` | 152.26 | 160.8 | 94.7% | 164.5 | 92.6% |
| `b9_K0_lt0` ★ | 175.97 | 203.4 | 86.5% | 229.3 | 76.7% |
| `b19_K0_lt0` | 211.79 | 238.5 | 88.8% | 247.3 | 85.6% |
| `b4_K0_lt2` | 340.64 | 351.5 | 96.9% | 358.4 | 95.0% |
| `b1_K20_lt0` | 425.83 | 441.1 | 96.5% | 493.6 | 86.3% |
| `b1_K20_lt2` | 450.90 | 497.9 | 90.6% | 562.7 | 80.1% |
| `b9_K0_lt2` ★ | 549.88 | 560.6 | 98.1% | 575.4 | 95.6% |
| `b4_K20_lt0` | 560.25 | 581.3 | 96.4% | 576.7 | 97.1% |
| `b9_K20_lt0` ★ | 618.71 | 637.4 | 97.1% | 628.9 | 98.4% |
| `b19_K20_lt0` | 664.62 | 685.1 | 97.0% | 702.3 | 94.6% |
| `b4_K20_lt2` | 696.50 | 730.8 | 95.3% | 764.5 | 91.1% |
| `b19_K0_lt2` | 903.52 | 927.5 | 97.4% | 967.5 | 93.4% |
| `b9_K20_lt2` | 924.41 | 953.8 | 96.9% | 987.5 | 93.6% |
| `b19_K20_lt2` | 1289.94 | 1330.5 | 97.0% | 1393.3 | 92.6% |
| **median** | — | — | **96.5%** | — | **92.6%** |

★ = a specialist exists at this cell: `simple` 176.11 (99.92%), `simple_k`
618.96 (99.96%), `lt` 550.63 (99.86%). **The generalist trails its own
specialists by 1.8, 2.9 and 13.4 points** (`lt`, `simple_k`, `simple`
respectively) — the price of one policy across regimes, and it is paid almost
entirely at the *cheapest* of the three.

The tuned arm is `ctx_b`, best of four independent `mdp_tuning` studies (179
trials, 48 h × 4 cores) that agree to **0.2** on this block. Tuning's largest
effect is at the **cheapest** cells, not the ones that dominate the mean:
`b1_K0_lt0` went 169.6 → 90.3, i.e. 44.3% → 83.1% — a 38.8-point gain at the
cell contributing least to the mean the tuner was actually reading.

**The context-blind control** (`vec`, same recipe, #E23) reaches median 78.5%
and loses at 15 of 16 cells. Observing `b/(b+h)`, `K` and `LT` is worth ~15
points of median % of optimal.

#### `cost_leadtime` — 54 cells, RQ-7's interpolation test

**No policy was trained here.** The arm is `grid16`'s tuned generalist (#E23)
scored on a finer grid; 12 cells overlap `grid16` and act as the memorization
control. Bar = each cell's exact DP; all 54 are backlog + deterministic LT.
2048 seeds per cell. The 12 shared cells reproduce `grid16`'s bars **bit-exactly
on the same seeds**, which is what makes the two grids one problem.

| stratum | cells | generalist median `% of optimal` | worst cell | context-blind |
|---|---:|---:|---:|---:|
| seen (trained) | 12 | 95.8% | 83.0% | 73.8% |
| **`b` interpolation** | 24 | **96.5%** | 81.9% | 82.4% |
| **`LT` interpolation** | 6 | **55.1%** | 41.3% | 43.3% |
| both | 12 | 54.4% | 43.1% | 46.9% |

**The two axes answer oppositely.** Unseen cost regimes are handled with *no*
degradation — 96.5% held out against 95.8% on the cells actually trained — so
the policy learned the map, not the samples. The unseen lead time `LT=1`
collapses to 55.1%, and the both-axes stratum is 54.4%, within 0.7 of it: the
damage is entirely the lead time and `b` adds nothing to it. By lead time alone
across the whole grid: `LT=0` 91.4%, `LT=2` 96.5%, **`LT=1` 54.7%**.

Why the asymmetry: `b` and `K` reach the policy as **observed values**
(`b/(b+h)`, `K/20`), so interpolating them is function approximation over a
surface that is monotone in `b` with zero violations. **Lead time is not a
value** — it is which pipeline slots are occupied, and `LT=1` occupancy appears
in no training cell. Padding keeps the observation the same width, but the
pattern is new, so this is extrapolation in a structural variable rather than
failed interpolation.

Pooling the 42 held-out cells gives 84.5% against the seen 95.8% — an 11-point
drop that reads as mild degradation and hides a total failure on 18 of them.
The strata exist to prevent exactly that.

### `2-structural` — the declared stances

Six stances are declared in `research_questions.tier2` (spec §14.0). RQ-5 was
withdrawn 2026-08-27 to the `3-methodology` section's Q-B — action encoding is a rendering question, not a
policy-structure one — and its id is **retired, not reused**.

**RQ-1 — (s, S) threshold recovery** *(confirm, primary)* — **ANSWERED**. The
claim was that the trained policy is order-up-to at K=0 and (s, S) at K>0,
recovering the thresholds the exact DP computes rather than merely matching its
cost. It does:

| target | recovered | DP | agreement |
|---|---|---|---|
| `simple` | `s` = 13, `S` = 14 — constant | 13, 14 | exact, every period |
| `simple_k` | `s` = 8 constant; `S` 25 → 23 → 15 | 8; 24 → 23 → 14 | `s` exact every period; `S` +1 throughout |

`S` is one number at `simple` and a **function of the period** at `simple_k` —
that difference *is* the end-of-horizon effect, learned.

The fitted `(s, S)` table is also a **shippable artifact**: scored on the same
protocol block it beats the net it was read from on both targets, and at
`simple_k` it is statistically indistinguishable from the exact DP.

Figures in canonical coordinates, the §14.2 reference/fitted/net scoring trio
and the full readback: **`INTERPRET.md` → RQ-1**.

**RQ-2 — inventory position as sufficient statistic** *(bypass)* —
**ANSWERED, both halves.** The claim: at `lt`, PPO on the raw state (`vec`)
matches PPO on IP (`vec_ip`) because the net rebuilds the statistic; at `slt`,
where crossing breaks sufficiency, `vec_ip` stops being enough.

| cell | `vec` | `vec_ip` | paired Δ | reading |
|---|---:|---:|---:|---|
| `lt` (L=2 fixed) | 551.57 | 550.63 | **+0.94** (0.17% of bar) | the transform need not be handed over |
| `slt` (L∈{1,2,3}) | 723.83 | 730.48 | **−6.65** (t=−15.1) | it must be |

**The sign flips across exactly the condition theory says should flip it** —
one measurement, two cells, opposite signs (#E10, #E11). `vec_ip` is dim 2 and
cannot tell `[10,0,0,0]` from `[0,0,0,10]`, so at `slt` this is information, not
capacity. A `bypass` stance owes the outcome comparison alone (§14.0), so there
is no `INTERPRET.md` section.

**Method note worth carrying:** at *both* cells the untuned contrast, run at a
recipe carried from another observation dimension, pointed the wrong way —
+97.35 at `lt` and +64.42 at `slt`, the latter reversing a sign. A recipe
carried across observation dimensions is a confound, and it twice produced a
confident wrong answer.

**RQ-3 — deviation from base-stock under lost sales** *(discover)* —
**ANSWERED on both halves** (#E17 → #E18 → #E21 → #E22). Tuned per mode at 4M,
PPO reaches **99.95%** of a verified-exact optimum this campaign had to build
itself, and beats Xin (2021)'s capped base-stock by −2.77 (t=−17.0). The
readback found mean |net − DP| = **0.24** and the deviation at **low** IP — the
direction pre-registered from the DP, and the reverse of the declared claim.
See `INTERPRET.md` §RQ-3.

**RQ-4 — structure under stochastic lead time** *(discover)* — **ANSWERED**,
and since extended across a 45-cell family (#E16, #E19).

At `slt` the learned policy on the full pipeline beats **every** rule measured
on inventory position alone — base-stock by +14.12 (t=+24.3), the DP table run
as a heuristic by +84.94, myopic by +95.53 — and its order depends on pipeline
*composition* at fixed inventory position, which is behaviour no position-only
rule can reproduce. Projecting the net onto the best position-only rule fitted
to its own behaviour prices that structure at **+11.36** (t=+21.9) at `slt`
against **−0.33** at the deterministic `lt` control: the two cells share a mean
lead time of 2.0 and differ only in its variance.

**The structure compresses to one number.** A linear information bottleneck
says a single learned weighted position `inv + 1.018·pipe₀ + 0.996·pipe₁ +
0.862·pipe₂` buys 85% of what full information is worth, and two numbers are
enough — the second being a near/far tilt, not a second prefix sum.

**Across the 45-cell family the same answer holds and acquires a control.**
The pipeline is worth **nothing** where the lead time is deterministic — a cell
where both observation modes carry identical information, so the measured gap
there is learnability and must be netted out — and **~10–15** where variance is
maximal. One policy spanning all 45 cells costs **0.3–1.2%** against specialists
tuned for a single cell, and three numbers reproduce it across the whole family.

Figures, the bottleneck ladder, the per-law gap series, the distilled rule and
what is *not* claimed: **`INTERPRET.md` → RQ-4**.

**RQ-6 — per-cell cost across `grid16`** *(confirm)* — **ANSWERED** (#E23),
after a **narrowing** recorded in `ESCALATION.md`'s §FRAME-CHANGELOG. One
`vec_ctx` generalist over the 16-cell sampler reaches **96.5% median of each
cell's exact DP**; the context-blind control reaches 78.5% and loses at 15 of
16 cells, so observing the cost regime is worth ~15 points. Tuning (4 studies,
179 trials) closed 56% of the carried recipe's gap and fixed the cheapest
cells, where L1 sat at 44%.

Two clauses were **struck** from the declared claim rather than left
unevidenced: *threshold* recovery, which the probe confirms on the 8 LT=0 cells
(`S_hat − S_dp` = +0, +0, +2, +3, +2) but cannot read at LT≥1 because it holds
the pipeline at zero — a state the policy never occupies there; and
*interpolation to interior fractiles*, which `grid16` cannot test because it
trains and evaluates on the same four `b` values. So this cell reports an
outcome comparison against a verified reference, not a structural readback.

**RQ-7 — generalization to held-out cells** *(confirm)* — **ANSWERED** (#E24),
and the answer splits by axis. `grid16`'s tuned generalist, retrained on
nothing, was scored on the 54-cell `cost_leadtime` grid; the per-stratum board
is in `1-comparative` above. The declared claim holds on cost and fails on
lead time:

- **Cost regimes interpolate at no cost.** Held-out `b ∈ {2,3,5,6,7,8}` reaches
  **96.5%** of each cell's exact DP against **95.8%** on the 12 cells actually
  trained. Scoring *above* the trained cells is the evidence that the net
  learned the map rather than the sampled points — `b` and `K` arrive as
  **observed values** (`b/(b+h)`, `K/20`) over a surface monotone in `b` with
  zero violations, so interpolating them is ordinary function approximation.
- **Lead time does not interpolate at all.** Unseen `LT=1` collapses to
  **55.1%**, no better than a control that cannot observe the cost regime
  (43.3%) and far below both trained lead times (`LT=0` 91.4%, `LT=2` 96.5%).
  The both-axes stratum is 54.4%, within 0.7 of lead time alone: the damage is
  entirely structural and `b` contributes none of it.

The mechanism is that lead time is **not a value the policy reads** — it is
which pipeline slots are occupied. `LT=1` occupancy appears in no training
cell, so serving it is *extrapolation in a structural variable*, not failed
interpolation, and no amount of sampling around it would have helped: the
trained cells already bracket it on both sides. Padding is what makes this
invisible in advance — the observation stays the same width, the forward pass
succeeds, and the arm looks like it should work.

**The consequence for the declared claim**: a generalist may be promised over
an axis the policy *reads*, and must be trained on every value of an axis that
changes *what* it reads. The general form of that rule is the transferable
part and is the reason RQ-7's grid design (stratified, one axis per stratum) is
written up in `PLAYBOOK.md` — the finding itself is inventory's and stays here.

### `3-methodology` — cross-campaign questions (NOT this domain's RQs)

Questions the operator has hit repeatedly across inventory domains, and which
this campaign is unusually well placed to answer — `inv_single` is the only
domain here with an *exactly*-solvable bar on both sides of the fixed-cost
line, so "which setting is better" can be measured against a known optimum
rather than against another arm.

**These are NOT tier-2 research questions and must not be added to
`research_questions.tier2`.** That block is about *policy structure* (confirm /
discover / bypass a structural claim) and is gated by `research.deliverables`.
These are questions about the *rendering and the recipe*. They are recorded
here so findings can be attributed to them, and so nobody later mistakes an
incidental ablation for a declared deliverable.

#### Q-A. When do `norm_obs`, `norm_reward`, `normalize_advantage` help?

The observation: outside inventory, all three go on and are never thought
about again. Inside inventory the best mix moves domain to domain — and the
suspicion is that **fixed ordering cost is what makes it move**.

There is a mechanism worth stating, because it predicts *which* of the three
should matter and where:

- `norm_obs` — the state carries a threshold (`s`, `S`, a base-stock level).
  `VecNormalize` rescales by *running* moments, so the lattice a threshold
  sits on keeps moving while the policy is still learning where the threshold
  is. Predicts harm on any threshold-optimal problem, worst early in training.
  **#E3 confirmed this at K=0 and K=20**, and the signature was not a uniform
  few percent — it was occasional total training failure (seed spread 19.06 →
  1.33).
- `norm_reward` — with `K > 0` the per-period reward has a *jump*: you pay `K`
  or you do not. The reward distribution is bimodal and its running std is
  dominated by ordering-cost spikes rather than by the holding/shortage signal
  the policy must actually resolve. Predicts harm specifically at `K > 0`, and
  little effect at `K = 0`. **Untested.**
- `normalize_advantage` — same jump, one layer down: with `(s,S)` optimal the
  advantage distribution is bimodal (order / do not order), and per-minibatch
  standardization rescales the two modes toward each other, flattening exactly
  the contrast the policy needs. Predicts harm at `K > 0`. **Untested.**

**What the plan already covers**, at L1 hyperparameters, both targets
(`simple` K=0 / `simple_k` K=20), both action encodings, 4 seeds:

| cell | `norm_obs` | `norm_reward` | `normalize_advantage` |
|---|---|---|---|
| `g0` (base) | on | on | on |
| `g4` | **off** | on | on |
| (L0 arms) | off | off | on | *confounded with L0 hyperparameters — not evidence for this question* |

**ANSWERED at the L1 centre (#E5, 2026-08-27), and the mechanism above is
falsified.** The full 2×2×2 ran on both targets at 4 seeds. Ranking: **obs OFF,
reward ON, advantage OFF** — each knob wins in 8/8 holdings. But three things
matter more than the ranking:

| knob | worth | reading |
|---|---|---|
| `norm_reward` | up to **+28164** | decides whether the run works at all |
| `norm_obs` | ~**6** | real, consistent, small |
| `normalize_advantage` | ~**4** at `simple`, **0.05** at `simple_k` | a rounding effect |

- **The `K>0` prediction is dead.** It predicted an effect at `simple_k` and
  none at `simple`; observed +28164 vs +27653, and `simple` has no fixed cost.
  It is a **scale** problem, not a fixed-cost one. Do not carry the fixed-cost
  story to another domain — it was wrong here.
- **Unpredicted interaction:** `norm_reward` off + `normalize_advantage` **on**
  is catastrophic (28k); with advantage **off** it is merely bad. The two are
  partial substitutes; reward-level is the better place to normalize.
- **It closed 4 of `simple`'s ~7 residual and none of `simple_k`'s.** The
  recipe is not what `simple_k` is short by.

#### Q-A's remaining question: three domains, three answers

Two other campaigns have run this. **`adi_flex` ran the same 2×2×2 at L1**
(its F37 / #E7, prompted by the same operator instinct — *"frankly I do not
think norm_obs is necessary in this problem"*), and `clark_scarf` flipped the
knobs at a tuned centre.

| knob | `inv_single` (#E5) | `adi_flex` (F37) | `clark_scarf` | agree? |
|---|---|---|---|---|
| `norm_reward` | **ON**, by 28164 | ON | **ON** — off 3420.28 vs 1012.40 | **unanimous** |
| `norm_obs` | **OFF**, by ~6 | **OFF**, by +25.33 | ON — +3.10/+3.31 re-tuned | 2–1 for OFF |
| `normalize_advantage` | **OFF**, by ~4 | **ON**, by +108.05 | **ON** — +45.97/+78.97 | **2–1 for ON** |
| best corner | `010` | `011` | `111` | — |

**`inv_single` is the outlier on `normalize_advantage`, and it holds the
weakest evidence**: ~4 cost, against `adi_flex`'s +108 (its largest single
effect) and `clark_scarf`'s +46/+79. A 4-cost mean difference contradicted by
two 10–25× larger ones is not a rule.

What rescues the local adoption is the **variance**, which points the same way
here and the opposite way in `adi_flex`:

| domain | effect of `normalize_advantage=OFF` on seed spread |
|---|---|
| `inv_single` | **shrinks** it, 8/8 holdings — sd ratios 0.00× to 0.40× |
| `adi_flex` | **inflates** it ~6–10× (its stated reason to keep it on) |

So the two domains disagree on mean *and* on variance, consistently within each.
That is a real domain difference, not noise — and note the centre explanation
does **not** cover it: `adi_flex` and `inv_single` both measured at **L1**.
Only `clark_scarf`'s `norm_obs` result has the tuned-centre caveat.

Candidate explanations, testable rather than paradoxical:

1. **The centres differ.** `clark_scarf` flipped each knob at a *tuned* centre
   and says so explicitly — the flip prices the lever *and* the centre-fit
   together (PLAYBOOK LV2), which is why it ran `hp3_nonorm_*` to separate them
   and found the correction **halved** the estimate without reversing it. This
   2×2×2 sits at the **L1 derived centre** with nothing re-tuned. A knob worth
   ~4 cost at one centre can plausibly flip sign at another.
2. **The disagreements are both inside the small-effect band.** Every conflict
   is on a knob worth single-digit-to-~80 cost against bars of ~180 / ~1012 —
   0.3% to 8%. The one knob worth thousands agrees. It may be that only
   `norm_reward` generalizes and the other two are domain- and centre-specific,
   in which case the transferable rule is *"normalize the reward, then tune the
   other two locally"* rather than a fixed triple.

`clark_scarf` is not uniform internally either: its `h8` adopted
`normalize_advantage OFF` on `sc3`. **Do not port `010` to a new scenario as a
default** — port `norm_reward=on`, and re-run the 2×2 for the other two.

Closing Q-A needs two more registry cells — `norm_reward` off, and
`normalize_advantage` off — run at both targets. At 4 seeds that is 16 runs,
~25 min at 8-way, and it reuses every bar and every paired comparator already
built. A full 2×2×2 would be 5 new cells (40 runs) and is only worth it if the
one-at-a-time ablation shows interaction.

#### Q-B. How should a *discrete* ordering action be encoded?

**Typed as a `design axis`, stance `discover`** (2026-08-27). The axis is
`gym.action_modes`; it is `design-axes` rather than `escalations`, so **every
option drawn on it owes a number permanently** and none can be pruned for
losing (guide §3.1). That is a property of the *axis*, not a claim that the
question is unanswered.

**The original phrasing — "what is a good universal action *encoding*" — mixed
two separable things.** An action *space* is what the gym exposes; a *head* is
how the network parameterizes a distribution over it. They sit on different
registry axes here (`gym.action_modes` and the `a` axis: `a0` = `MlpPolicy`,
`a1` = `ordinal`), and #E21 contested two heads at a **fixed** space, which
would be meaningless if one determined the other. They are constrained — an
ordinal head needs an ordered discrete space, a Gaussian needs a `Box` — but
not in bijection.

##### The space question, and why it is closed

**Is the demand/dynamics fully integer-valued?** That decides discrete vs
`Box`, and it is a fact about the domain, not a modelling preference. **Here:
yes** — Poisson demand, and the order flows into integer stock. `order` (the
`Box[0, cap]` mode) is drawn and its number is owed permanently: paired median
**+234** / **+376** against discrete, every seed, at both L0 and L1 (#E4),
which accounted for the *whole* of `simple`'s residual gap.

So the live question is the one below, and the continuous distribution families
(truncated normal, gamma/lognormal, zero-inflated continuous) are **out of
scope for this domain** rather than untested candidates. Recorded once, because
it is the reasoning and not the result that transfers: a gamma or lognormal has
support `(0, ∞)` and so can place **no** mass at exactly zero — zero-inflation
is a prerequisite there, not an improvement — while a clipped normal gets its
zero atom free from the boundary but must then represent the `(s,S)` "lump" as
a tail of a distribution centred below zero.

##### The encoding question, on a discrete space

`inv_single`'s order is zero most periods and a lump when it fires, so the
action distribution has to be **bimodal**.

| option | parameterization | status |
|---|---|---|
| (a) flat categorical | `cap+1` free logits, softmax | measured — `a0`; no notion of adjacency, so learning that 14 is right teaches nothing about 13 or 15 (#E2/#E6) |
| (b) **zero-inflated discretized-Gaussian** (shipped as `ordinal`) | `logit(0)=log w`; `logit(k)=log(1−w) − (k−μ)²/2τ² − logZ` — three numbers `(w, μ, τ)` | **ADOPTED** — `a1`, #E9, re-confirmed at a tuned centre #E21 |
| (c) cumulative-link ordinal | latent variable + `n−1` learned cutpoints, `logit P(Y≤k) = θ_k − η` | **not drawn** |
| (d) continuous kernel + rounding | sample a density, round to the grid | **not drawn** |

**What was measured: (b) beats (a).** +0.560 (t=4.9) at `vec` and +0.407
(t=3.7) at `vec_ip`, head-to-head at a tuned centre over 213 trials (#E21). The
value is **robustness across hyperparameter draws** — the flat head's median is
1.70 worse at `vec` — not peak cost, where the gap is half a unit on a 416 bar.
Every crowned artifact runs `order_discrete` + this head.

**Naming, because "ordinal" understates it.** Row (b) is *not* an ordinal model
in the statistical sense: there are no cutpoints. It is a **point mass at zero
mixed with a discretized Gaussian** over the positive quantities. "Ordinal"
names the intended property — adjacency, so the location gradient
`∂log π/∂μ ∝ (a−μ)/τ²` pools every sample into one estimate of where the level
is — not the parameterization. A true cumulative-link model, row (c), has
**never been tested here**, and a reader who took "ordinal beats categorical"
to mean otherwise would be misled.

Row (b) is also a continuous shape *rendered onto* a discrete space: it builds
the categorical's logits from `(w, μ, τ)` directly, so `log_prob`, entropy,
sampling and mode are inherited unchanged from `Categorical` — getting row (d)'s
parameter sharing without row (d)'s defect, where a rounded sample's log-prob
does not match the density that was differentiated.

#### Scope note

Both are **inventory-specific**, which is why they sit here and not in
`PLAYBOOK.md`: Q-A's answer is a statement about reward scale in cost-minimising
inventory problems, and Q-B's is about encoding a non-negative order quantity
whose optimum is zero most periods. Neither generalises to the pipeline as a
default without re-deriving it in another domain — and Q-A's own three-domain
table is the evidence that it does not.

Q-A is **answered** (#E5). Q-B is **answered for this domain**: integer
dynamics settle the space, and the zero-inflated discretized-Gaussian head beats
a flat categorical at a tuned centre. Two discrete encodings remain undrawn — a
true cumulative-link ordinal, and a continuous kernel with rounding. Its axis
stays `discover`, so any newly drawn option owes a number permanently and none
can be pruned for losing; that is a property of the axis, not an open verdict.
Decide any new option at a batch boundary, not mid-batch.

## Technical appendix

Run from `inv_single/`. Always pin threads — torch oversubscribes on a shared
box. Together these reproduce `results/` from an empty folder; `results/` is
gitignored. The **gate** commands are not here — they are in `CLAUDE.md`,
because they are run by whoever is changing the folder rather than reading it.

The three evaluation layers are **disjoint** (spec §9.7): the block that ranks
never touches the block that quotes. The train script's tail eval is a
~20-episode smoke test and is not a result; the terminal checkpoint is not the
deliverable (spec §8.6).

```bash
# 1. train — runs to budget, saves ~20 checkpoints. No live selection callback.
#    Every L1+ run cites a §CONFIG-REGISTRY tuple and its comparator.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python inv_single_ppo_train.py --config sc0/g4/a1/h2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python inv_single_ppo_train.py -s lt --seed 42 --tag A1

# 2. screen — rank the checkpoints on the screen block (first_seed 1_000_000).
#    These scores are never quoted; the top 3 go forward.
python inv_single_select.py results/simple/PPO_<timestamp>_<args>

# 3. confirm — the protocol block (first_seed 0). This TSV is the number.
python inv_single_ppo_eval.py --model-path <winner.zip> -s simple --first-seed 0 --n-seeds 8192

# benchmarks — the DP bar and the rule arms, same protocol block
python inv_single_benchmark_dp_eval.py -s simple_k --n-seeds 8192
python inv_single_benchmark_dp_eval.py -s simple_k --no-cache      # force a re-solve
python inv_single_benchmark_heuristic.py -s simple_k --policy basestock_opt

# readback + figure (spec §14)
python inv_single_policy_probe.py --model-path <winner.zip> -s simple_k --score-fitted
python inv_single_plot_policy.py  --model-path <winner.zip> -s simple_k

# RQ-4 readback — is inventory position a sufficient statistic? (#E12)
# rolls the net on-policy first, so only VISITED compositions are queried;
# the roll is seeded, so the reported band and gradient reproduce
python inv_single_policy_probe.py --model-path <winner.zip> -s slt -o vec \
    --composition --comp-period 10 --score-fitted-ip --n-seeds 8192
python inv_single_policy_probe.py --model-path <lt_winner.zip> -s lt -o vec \
    --composition --comp-period 10 --score-fitted-ip --n-seeds 8192   # the control

# the two committed figures INTERPRET cites
python inv_single_plot_composition.py   --model-path <winner.zip> -s slt
python inv_single_plot_variance_gap.py
```

### Key training arguments

| argument | default | description |
|---|---|---|
| `-s` / `--scenario_name` | `simple` | scenario name from `SCENARIOS` |
| `-o` / `--observation_mode` | `vec` | observation feature set (see below) |
| `-a` / `--action_mode` | `discrete` | action encoding (`discrete` \| `continuous` \| `hurdle`) |
| `--config` | *(none)* | §CONFIG-REGISTRY tuple, e.g. `sc0/g4/a1/h2`; fixes the design axes write-once |
| `--total-timesteps` | 2 000 000 | training budget |
| `--seed` | 1 | global random seed |
| `--tag` | *(none)* | free-text label appended to the run name |
| `--checkpoint-every-frac` | 0.05 | checkpoint cadence as a fraction of budget (~20 checkpoints) |
| `--learning_rate` / `--lr-final` | 3e-4 / 3e-5 | learning-rate schedule |
| `--clip-init` / `--clip-final` | 0.2 / 0.05 | PPO clip-range schedule |
| `--normalize-advantage` / `--no-normalize-advantage` | on | PPO advantage normalization (SB3 default) |
| `--no-norm-obs` / `--no-norm-reward` | *(both on by default)* | disable the VecNormalize halves |

The L1 hyperparameters live in `_L1_DERIVED` with their basis in `_L1_BASIS`
(`inv_single_ppo_train.py`), re-derived 2026-08-27 at the restart. The registry
self-check refuses a parser default that drifts from the derivation — promote a
value into both or neither.

### Observation modes

Selected with `-o` / `--observation_mode`. **The default is `vec` in the train,
select, eval, probe and plot scripts alike** — they disagreed twice before
2026-08-27, which silently evaluated an arm on features it was not trained on.
An eval must use the mode its arm trained with.

**Every mode leads with `time_to_go = T − period`**, counting *down* to the
horizon. This is the domain's canonical time encoding (F5); it matches
`adi_flex` and spec §7's recommendation, which this campaign's issue
[#71](https://github.com/tong-wang/auto-mdp-solver/issues/71) produced. Runs
before 2026-08-28 used the period index and are superseded, not comparable.

| `-o` | features | width at `lt` (L=2) |
|---|---|---:|
| `vec` **(default)** | time_to_go, inventory, pipeline[0..L] | 5 |
| `vec_ip` | time_to_go, inventory position | 2 |
| `vec_ctx` | `vec` + b/(b+h), K — the `grid16` generalist's cost context | 7 |
| `vec_ctx_slt` | `vec_ctx` + the lead-time **law** (`leadtime.probs`, `dim` 3) — the `lt_variance_*` generalist's regime context | 10 at `slt` |
| `vec_ip_ctx_slt` | `vec_ip` + the same law and cost context — the restricted-information control | 7 at `slt` |

`vec_ip` carries **no** pipeline: the point of the mode is that IP is the
sufficient statistic under deterministic lead time, so the pipeline is summed
away — which is precisely what RQ-2 tests. Widths are checked against the IR's
declared features by
`inv_single_test.py::test_the_rendered_vector_is_the_declared_vector`.

The two `_slt` modes carry the lead-time law **in force**: since v0.10.7
(upstream #77, filed from this campaign) the IR feature reads `leadtime.probs`,
which folds from the selected candidate at load — the categorical law under
`slt` and its grids, the one-point law `[1.0]` under a deterministic candidate —
never the `leadtime_probs` constant, which keeps a value under every candidate.
The modes are narrowed by the feature's declared width (`dim` 3, the
categorical support {1,2,3}): the gym refuses a scenario whose law has any
other width, which covers the `lt` instance (1 point) and `grid16`'s padded
short-lead-time cells (2 points) alike. Before v0.10.7 the same narrowing was
a family assertion in the gym that the IR could not express.

### The simulator

Each period executes three events in order — **R** (receive), **O** (order),
**D** (demand). The default is O→R→D; `rdo` and `rod` are the R-D-O and R-O-D
variants (R-O-D requires lead time ≥ 1). `pipeline[k]` holds the quantity
arriving at the *k*-th future R event, and its length is fixed at
`leadtime.max() + 1` throughout an episode.

Demand is **not** stored in state — it is regenerated each step from a
deterministic seed sequence over (scenario salt, episode seed, period), so two
episodes with the same seed see identical demand regardless of the actions
taken. That is what makes every board above paired.

`grid16` cells are **padded** (spec §5.6): a cell's stored generator carries the
family-maximum pipeline, so an L=0 cell reports `leadtime.max() = 2` while its
effective lead time is `leadtime.mean() = 0`. Anything per-cell must read
`leadtime.mean()`. Gated by
`inv_single_test.py::test_grid16_padded_cell_is_the_unpadded_cell_plus_zeros`.

### DP notes — where the bar is exact

`inv_single_benchmark_dp.py` computes the optimal finite-horizon policy by
backward induction and caches to `results/<scenario>/benchmark/dp/`.

**Two DP implementations, and which one is the bar depends on the instance.**
`inv_single_benchmark_dp.py` solves the backlog recursion;
`inv_single_benchmark_dp_lostsales.py` solves the lost-sales one.

| scenario type | implementation | state | exactness |
|---|---|---|---|
| L=0 | `benchmark_dp` | inventory | **exact** |
| deterministic L ≥ 1, backlog | `benchmark_dp` | inventory position | **exact** — verified at L=2 by `inv_single_dp_exactness.py` |
| stochastic L | `benchmark_dp` | inventory position | approximation — orders cross, IP is not sufficient |
| lost sales, deterministic L | **`benchmark_dp_lostsales`** | **`(u, p1)`** | **exact** (#E17) — verified against the un-reduced 3-D DP (6e-14) and by simulating its own policy (+0.19 SE) |
| lost sales, *any* instance | `benchmark_dp` | inventory position | **not a valid bar** — the cost recursion is backlog-only |
| lost sales, stochastic L / latent demand | — | — | **no bar** — the `(u, p1)` reduction needs a deterministic lead time |

Why lost sales needs its own implementation, in one line: under `O-R-D` the
receipt precedes demand, so `inv` and `p0` merge into `u = inv + p0` in **both**
recursions — but the next state is `u − d + p1` under backlog (linear, so `u`
and `p1` collapse further into inventory position) and `(u − d)⁺ + p1` under
lost sales (**not** linear, so they do not). That single truncation is why
base-stock is not optimal here.

The resulting `(u, p1)` state is **exactly the standard lost-sales state**, not
smaller: the classical formulation is (on-hand *after* receipt, plus the `L-1`
still-outstanding orders) = dimension `L`, which at L=2 is two. This DP is cheap
because this domain's L is 2, not because the state was reduced below what the
literature uses. The curse of dimensionality is real and bites as L grows
(Zipkin 2008 solves numerically to L ~ 4).

The gym's action cap is **not** the IR's declared bound: the IR declares
`order ∈ [0, order_cap_mult · demand.mean]` (= [0, 200]) while the gym builds
`2 · demand.max() · max(1, leadtime.max())` (`Discrete(46)` at `simple`,
`Discrete(91)` at `lt`). #E2 measured the cost as zero — 0 of 15360
optimal-policy actions per scenario are cut off — but the DP is not clipped to
the gym space at all, so it can order past the cap where a policy cannot.

### Results folder structure

```
results/
  <scenario>/
    PPO_<timestamp>_<level>_<design-axes>_<changed-args>/
      <scenario>_ppo_args.txt
      ppo_inv_single.zip
      vecnormalize.pkl
      checkpoints/
      train.log                # captured console output
      train_log_*.log          # per-episode/per-step gym logs, only with --gym-log 1
      monitor.csv
    benchmark/
      dp/{V.csv, pi.csv, meta.json, events.out.tfevents.*}
```

A run-dir name is a **path**: `build_run_name` renders a list as `64-64-64-64`,
never `str([64, 64, 64, 64])`. Eight run dirs from 2026-08-27 carry the older
spelling; they are archive and are not renamed.

### Comparing RL and DP in TensorBoard

Both PPO and the DP benchmark write `rollout/ep_rew_mean` on the same
pseudo-timestep scale (`episodes × horizon`), so pointing TensorBoard at the
scenario folder puts them on one plot:

```bash
tensorboard --logdir inv_single/results/simple/
```

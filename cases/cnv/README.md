# cnv — censored newsvendor with Bayesian demand learning from stock-out timing

The multi-period newsvendor of Jain, Rudi & Wang, "Demand Estimation and
Ordering Under Censoring: Stock-Out Timing Is (Almost) All You Need",
*Operations Research* 63(1), 2015, pp. 134–150: a perishable product stocked
each period against Poisson demand whose rate is unknown and learned from
censored sales, where the *time* at which stock ran out carries most of what
the censoring hid. Generated from `cnv_schema.json` by auto-mdp-solver.

**TL;DR**

- **RL gets within 1% of the optimum wherever one is known, and matches or beats the paper's myopic
  heuristics**: 0.07% under the exact DP on the four-period cell, 0.7–0.9% under the per-cell optimum
  across the 510-cell grid, and 0.06% under the full-information optimum over 100 periods (792.94
  against 793.41). With stock-out timing observed over 100 periods it sits level with the heuristic
  (+0.28 ± 0.34), and the rule read back from the event-observation network beats that type's heuristic
  across the grid (+0.10 ± 0.01); the event networks themselves stay 0.8–1.6% under it at 100 periods.
- **RL bypasses the Bayesian machinery — raw observations and the posterior make little difference**:
  fed the raw sales history or the posterior, the timing policies play identically on every seed of the
  four-period cell; across the grid the raw networks trail by 0.2; over 100 periods they lead by 0.4 to
  4.5.
- **RL confirms the over-ordering policy the paper proves**: the networks stock above the myopic level
  at the first decision — on average across the grid by one unit under timing observation, two under
  event observation and none under full observation, the shape of the exact DP's premium (+0.55, +2.15,
  0) — and timing's premium sits below event's on 350 of the 510 cells (the DP: 345).
- **One generalist network per observation type covers the paper's whole 510-cell grid**: above the
  no-learning floor on every cell (510 of 510 for the posterior-fed networks) and, under timing and full
  observation, within 1% of the per-cell optimum on the grid mean (0.7% for the posterior-fed networks,
  0.9% for the raw ones).

## The problem

At the start of each of N periods a newsvendor chooses how many units to
stock, at cost c per unit and revenue r per unit sold; the units are delivered
at once and are worthless at the period's end. Customers arrive during the
period as a Poisson process whose rate λ she does not know; she holds a
Gamma(α, β) prior on it and the same λ drives every period of the horizon.
Demand is the number of arrivals, sales are the smaller of demand and stock,
and unmet demand is lost — and, crucially, unseen. The objective is the
undiscounted sum of profit, r·sales − c·stock, over the horizon.

What makes it hard is the censoring. Once the shelf is empty the newsvendor
does not see how many more would have bought, so every stock-out is a
truncated observation, and a decision that stocks little both earns little
and learns little. The paper's point is that the *timing* of the stock-out
recovers most of the lost information: for Poisson demand the whole history
collapses into two numbers, cumulative sales and cumulative sales duration,
and the posterior stays a Gamma. Three observation types are compared —
**event** (sales plus whether a stock-out occurred), **timing** (sales plus
when the stock ran out) and **full** (every arrival, censoring removed) —
and the stocking level optimal under censoring exceeds the single-period
newsvendor quantity: the newsvendor over-orders to buy information, less so
under timing than under event observations.

What is known analytically: the exact dynamic program over the belief state
at N = 4 (the paper's §5 study) for the timing and full types, the myopic
policies that stock the single-period newsvendor quantity under the running
posterior (the paper's §5.3, within a point of optimal on average), and the
perfect-knowledge ceiling that knows λ. The event type has no fixed-size
sufficient statistic, so its optimum needs the whole history. Open, and what
the leaderboards below measure: whether a trained policy recovers the
over-ordering structure from the belief, and whether a policy fed only raw
per-period observations with a frame stack matches one handed the sufficient
statistic.

**Source.** The paper above; the numerical study's Poisson case with the
conjugate Gamma prior, its 510-cell design (17 unit costs × 6 prior scales ×
5 prior means at N = 4) and its N = 100 myopic simulations. The arrival
process is simulated by the conditional-uniform method (a Poisson count, then
uniform epochs) rather than by summed exponential gaps — identical in law.
Demand is capped at 402 arrivals per period, the 1e-8 marginal quantile of
the widest cell. The authoritative statement of the problem is
`cnv_schema.json`; `cnv.restatement.md` renders it in prose.

## Layout

Documents — where to read about the case:

| doc | what it holds |
|---|---|
| `cnv.restatement.md` | the Phase-A restatement: the problem as the IR states it, with two declared sample trajectories |
| `cnv.scenarios.md` | the Phase-A scenario-set briefing the instances and grids were chosen from |
| `cnv.runplan.json`, `cnv.signoff.json` | the confirmed run plan (target `paper510`, generalist) and the IR freeze record |
| `ESCALATION.md` | the campaign log, closed 2026-10-13 — design tree (F1–F12), changelogs, findings #E1–#E117 with verdicts |
| `INTERPRET.md` | the policy readback (spec §14): base, `paper510` (the run plan's target) and the `fig6a` / `fig6b` information ledger |
| `PLAYBOOK.md` | the case-close digest (guide §10): three transferable levers — the vine for a payoff spread over many noisy periods, a network that does Bayesian learning (per-frame log-likelihood, pooled), the oracle cut that separates encoding from optimiser |
| `figures/` | the committed static readback figure (`cnv_policy_paper510.svg`; regenerating command in `INTERPRET.md`) |
| `CLAUDE.md` | operating brief for an agent changing this folder |
| `Demand_Estimation_and_Ordering_Under_Cen.pdf` | the source paper |

Code — where things are implemented:

| module | role |
|---|---|
| `cnv_schema.json` | **the IR — authoritative** for the problem definition; `cnv.signoff.json` records the freeze |
| `cnv_uncertainty.py` | the stochastic primitives: Poisson demand with the hidden Gamma-distributed rate (the world latent), uniform arrival epochs, the family-generic bridges |
| `cnv_scenarios.py` | the world layer: `CnvScenario`, the per-seed `CnvScenarioSource`, the five instances, `SCENARIOS` |
| `cnv_grids.py` | the design layer: `paper510` and `fig3`, `GRIDS`, the grid training sampler |
| `cnv_mdp.py` | `CnvState` (clock + the running totals S, T, cumulative demand) and the transition |
| `cnv_gym.py` | the Gymnasium wrapper and `render_obs`: the IR's modes (`timing_belief` / `full_belief`, their `_ctx` grid forms, `event_raw` / `timing_raw` / `full_raw` with a frame stack) plus the campaign's declared extensions — the `_log` recodings, `event_belief_log`, the `oracle_*` diagnostics and the `_myo` assistance modes; `order_discrete` (and the `premium_discrete` assistance action) |
| `cnv_bayes.py` | the IR's expression builtins (Poisson and negative-binomial quantiles) and the dependency-free incomplete gamma / beta functions the benchmarks use |
| `cnv_event_posterior.py` | the exact event-type posterior on a grid (the `event_belief_log` mode's statistic) |
| `cnv_ir_adapter.py` | the differential adapter (portable-domain contract) |
| `cnv_benchmark_common.py` | the shared spec-§9 eval loop: seed block, CRN, TSV record, per-seed sidecar |
| `cnv_benchmark_myopic.py` / `_eval.py` | the paper's myopic policies: `timing`, `full` (conjugate) and `event` (grid posterior) — role feasible |
| `cnv_benchmark_dp.py` / `_eval.py` | the exact DP at N = 4 for `timing` (sales lattice × T grid) and `full` (= myopic, Proposition 1(b)) — role exact |
| `cnv_benchmark_random.py` / `_eval.py` | the floors: uniform and prior-mean random orders |
| `cnv_ppo_train.py` / `_ppo_eval.py` | training (specialist `-s`, generalist `-g`; every lever below is a flag) and evaluation (one row per scenario or per grid cell) |
| `cnv_select.py` | the post-hoc checkpoint screen: every checkpoint on the screen block, the top three to confirm on the protocol block |
| `cnv_dgauss_head.py` | the discretized-Gaussian action head (log location) over `Discrete(403)` (#E45) |
| `cnv_raw_extractor.py` | the raw frame-stack extractors: the per-frame map pooled by the count (frame-MEAN, #E75–#E78) and its predecessors |
| `cnv_vine_ppo.py` | the vine: side comparisons on forked copies of the world, run to the episode's end (#E110) |
| `cnv_group_baseline.py` / `cnv_branch_ppo.py` | the group baseline and branch-and-grid levers (#E87, #E88) |
| `cnv_popart.py` | the PopArt value head (#E57, a diagnostic) |
| `cnv_belief_estimator.py` | the supervised event-belief estimator (#E109 — a distilled rule, never crowned) |
| `cnv_policy_probe.py` | the §14 readback probe: belief-surface sweep and on-policy agreement against each type's myopic level and the DP, self-checked on the myopic rules |
| `cnv_fitted_rule.py` | the fitted rule order = y^M + premium, scored on the protocol block and paired against its net and the bar (spec §14.2) |
| `cnv_plot_policy.py` | the §14.3 overlay figure in premium coordinates, with the paper's exact first-period premium per cell |
| `cnv_info_ledger.py` | the information ledger for censored policies: what a policy pays to deviate from the myopic order at its own belief vs the information its beliefs gain (#E83, #E105–#E109, #E118) |
| `cnv_grid_gate.py` | the per-cell paired gate on a grid (above the floor, within the bar's band, beats the bar) behind the `paper510` board's columns |
| `cnv_probe_stat.py` / `cnv_probe_loglik.py` | readouts of a raw extractor's pooled features against the exact posterior and the summed log-likelihood (#E72, #E75) |
| `cnv_order_by_lambda.py` | the paired order error bucketed by the episode's realised rate (the band readings) |
| `cnv_vine_gate.py` | the vine's gates on one real rollout: main rollout untouched, every fork credited, common-noise copies agree, side pass moves (#E110) |
| `cnv_prune_results.py` | the `results/` retention rule (stated in its docstring): dry run to a plan, `--apply` exactly the reviewed plan |
| `cnv_policy.py` | the deployable wrapper over the shipped artifacts (spec §12), smoke-tested against the raw `cnv_mdp` loop |
| `requirements.txt` | the folder's own dependency beyond the workspace install: scipy, for the timing DP (spec §1.2) |
| `cnv_test.py` | the domain's own tests: laws, the differential over the covering set, negative controls, grid ↔ IR agreement, rendered-vector check, the probe's self-checks |

## Results

### Protocol

Every number below is quoted at 8192 seeds from `--first-seed 0` (the protocol
block) on a single scenario, and on the `paper510` grid at 2048 seeds per cell
for RL arms (the first quarter of the bars' 8192 block; contrasts pair on the
shared prefix — `ESCALATION.md` #E27), CRN across arms, reference bar = the exact DP under the same
observation type (`dp_timing` for the timing arms), record eval =
**deterministic argmax** (the policy's stocking level is a decision, not a
draw; a stochastic eval appears only as a labelled diagnostic). Checkpoints are
ranked on a disjoint screen block (2048 seeds from 1,000,000) and the top three
confirmed on the protocol block; the screen score is never quoted.

Symbols: **profit** = Σ over the N periods of r·sales − c·stock, per episode;
**Δ** = paired per-seed difference against the bar, ± its paired SE;
**y^M** = the myopic (single-period newsvendor) stocking level under the
current posterior; **premium** = order − y^M; **z** = Δ / its SE; **L0 / L1** = spec §8.6 solve levels
(library defaults / the derived configuration), **L2 / L3** = one / two escalation layers on top of L1
(hp = hyperparameters, gym = observation and normalisation settings, arch = network form); a run id `PPO_<timestamp>_<level>_...` names its
directory under `results/base/`.

### `1-comparative` — how does RL compare with the existing solutions?

The scenarios, and the role each plays in the study:

| scenario | config | what it is for |
|---|---|---|
| `base` | N=4, c=1, prior Gamma(10, 1) | the paper's Fig. 5(a) cell; the cheap specialist target and the gate's smoke instance |
| `fig3_c10` / `fig3_c15` | N=4, c=1.0 / 1.5, prior Gamma(10/16, 1/16) | the Fig. 3(a)/4(a) family at ratio 0.5 and 0.25: diffuse prior, where timing's edge over event is largest |
| `fig3` (grid) | the 17-cost sweep at the Fig. 3 prior | the paper's Figures 3(a) and 4(a) |
| `paper510` (grid) | 17 costs × 6 prior scales × 5 prior means, N=4 | the §5.1 study: the loss-recovery distribution; the primary generalist target |
| `fig6a` | N=100, c=1.0 (ratio 0.5), the Fig. 3 prior | the paper's long-horizon myopic simulation; the campaign's main line runs here |
| `fig6b` | N=100, c=1.5 (ratio 0.25), the Fig. 3 prior | the same at a higher cost, where information is worth more; read with fig6a's recipes, trained here |

The N = 4 boards and the N = 100 boards are separate leaderboards, never
compared: different horizons, different bars, different profit scales; `fig6a` and `fig6b` are separate
boards too (different costs, different bars). Two scenarios carry no board:
`fig3` is 17 of `paper510`'s cells and is read from its rows; `fig3_c10` / `fig3_c15` served the exact
DP checks of the assistance rung (#E84, #E85; outside the IR's arms) and were never an RL target.

#### base — the paper's Fig. 5(a) cell (N=4, r=2, c=1, prior Gamma(10, 1))

Sorted by profit; every arm on the protocol block above. The RL rows are each
mode's study winner (bold, one artifact each) and the untuned L1 centre the
`timing_belief` study started from; the per-mode L0 / L1 ladders (#E2, #E3,
#E10–#E13) and the convergence test (#E5) are in `ESCALATION.md`. The base
pass is closed (#E20): every mode's winner sits within 0.06 of its optimum or
ties its best heuristic. Rows for the full-information type are compared
against dp_full (their own optimum), the timing rows against myopic_timing /
dp_timing, the event row against myopic_event; the types share the floor and
the ceiling but not the bar.

| arm | role | profit | Δ vs myopic_timing | z | scope |
|---|---|---|---|---|---|
| perfect knowledge (eval column) | relaxed | 29.783 | +2.442 | — | knows λ; unattainable |
| dp_full | exact | 27.501 | +0.160 | — | full observation; = myopic_full (Prop. 1(b)) |
| myopic_full | feasible | 27.501 | +0.160 | — | full observation |
| **PPO L2(hp) `full_belief`, study winner** | feasible | **27.464** | vs dp_full −0.037 ± 0.014 | — | 99.9% of dp_full; 82 trials (#E14) |
| **PPO L2(hp) `full_raw`, study winner** | feasible | **27.443** | vs dp_full −0.058 ± 0.015 | — | 99.8% of dp_full; `mdp_tuning`, 82 trials (#E15) |
| dp_timing | exact | 27.363 | +0.022 | — | timing observation; T grid 1/40 (played profit, grid-insensitive) |
| **PPO L3(hp+gym) `timing_belief`, study winner** | feasible | **27.342** | +0.001 ± 0.016 | 0.1 | 99.9% of dp_timing; `mdp_tuning` over all knobs, 92 trials at 4M (#E8) |
| **PPO L2(hp) `timing_raw`, study winner** | feasible | **27.342** | +0.001 ± 0.015 | 0.1 | identical per-seed profit to timing_belief's winner (paired Δ 0.000 ± 0.011); 87 trials (#E17) |
| **myopic_timing** | feasible | **27.341** | — | — | the must-beat bar for the timing arms |
| **PPO L3(hp+gym) `event_raw`, study winner** | feasible | **27.124** | vs myopic_event +0.017 ± 0.021 | — | ties the paper's myopic event policy; 82 trials (#E18) |
| myopic_event | feasible | 27.107 | −0.234 | — | event observation (grid posterior) |
| PPO L1 `timing_belief` at 4M (study trial 0) | feasible | 27.074 | −0.267 ± 0.021 | −12.5 | the derivation at the settled budget — the untuned centre of the #E8 study (#E9) |
| random prior_mean | floor | 25.722 | −1.619 ± 0.051 | −8.2 | never learns; PPO L0 `timing_belief` converged exactly here on both seeds (#E2) |
| random uniform | floor | −724.25 | — | — | the did-it-learn-anything floor |

Loss recovery of timing over event on the myopic policies:
(27.341 − 27.107) / (27.501 − 27.107) = 59.4%.

**Validated against the paper's own raw output** (github.com/tong-wang/NVLearning,
`output/NVLearning_*.txt`: the exact DP profit of every model on the same 510
cells, and a 101-period simulation for §5.3). Our bars reproduce it to Monte
Carlo error — at this cell their exact `Pi_F` is 27.8039 and our estimator gives
27.8038 on 65,536 seeds from a different offset. The protocol block 0..8191 is
an unlucky draw, so every N=4 bar reads ~1% low (1.1% at `base`) and every
N=100 bar ~1.5% high (1.3–1.6%);
the offset is **identical across arms** (all cells share the block by design) and
cancels in every gap the paper plots — full − timing 5.78 vs our 5.63 at `fig6a`,
6.64 vs 6.63 at `fig6b`. See `ESCALATION.md` #E43.

**Shipped, one artifact per mode: each study winner** (see the bold rows). For `timing_belief` it is the #E8 winner (trial 90's terminal
artifact, 4M steps). It is inside the DP's band and tied with the myopic
timing policy; the gate for this cell is the DP band with prior_mean as the
must-beat floor, because the optimum exceeds myopic_timing by only 0.022 here,
so a must-beat against myopic is unsatisfiable by construction. The L1 centre
at the same budget is 0.27 lower; the study moved the observation and advantage
normalisers off, the clip to 0.1 and the width to 128. Its readback is in
`INTERPRET.md` (#E16). The other four modes' winners are the bold rows above.

How it was reached: `ESCALATION.md` §MAP and #E1–#E9.

#### paper510 — the paper's §5.1 grid (510 cells: 17 costs × 6 prior scales × 5 prior means, N=4), one generalist per mode

**Cells are separate leaderboards** (spec §9.6); the numbers below are means
over the 510 cells (equal weight per cell) and summarise the per-cell record,
they are never a score. Bars at 8192 seeds per cell; RL arms at 2048 per cell
(#E27), the crown of each run = the best of its top-3 screened checkpoints
(`cnv_select.py`, screen block). The gate is per cell, paired on the shared
seed prefix (`cnv_grid_gate.py` → `<crown>.gate.tsv` beside each sidecar):
above the floor = z ≥ 2 vs random prior_mean; within the band = Δ ≥ −2 SE vs
the type's myopic bar, and beats the bar = z ≥ 2 above it (myopic_full = dp_full for the full type;
myopic_timing for the timing type, with the played dp_timing — a 1/40 T-grid policy, not a per-cell
upper bound — judged beside it in the gate files; myopic_event for the event type).

Loss recovery of timing over event on the myopic policies, per cell: median
79.6% (IQR 72–85%) over the 491 cells whose full−event gap exceeds 0.05; the
paper reports ≈75%. It rises with the prior's diffuseness and the prior mean
and vanishes in the low-mean × diffuse × expensive corner (`ESCALATION.md`
#E21).

Sorted by the mean over cells on the 2048-seed prefix every row shares (the RL arms' protocol, #E27);
the bars' 8192-seed means are given in brackets. The RL rows are each mode's study winner (bold), the
readback's fitted rules (the shipped artifacts, bold) and the L1 round's best run; the L0 round (every
generalist one fixed order, below the floor) and the other L1 runs are in `ESCALATION.md` #E22–#E26.

| arm | mean over cells | above floor | within band | worst cell vs bar | ledger |
|---|---|---|---|---|---|
| perfect knowledge (eval column; relaxed ceiling) | 105.83 [105.84] | — | — | — | #E21 |
| dp_full = myopic_full (the full bar) | 99.66 [99.83] | — | — | — | #E21 |
| **fitted rule, full: y^M, no premium — shipped** | **99.66** (= myopic_full on every seed) | 510 / 510 | 510 / 510 (identical to the bar) | 0.00 | #E118 |
| **fitted rule, timing: y^M + 1 at the first decision — shipped** | **99.28** | 510 / 510 | 418 / 510, and beats the bar on 36 | −0.17 at `c=1.4, r=2, mean=10` | #E118 |
| myopic_timing / dp_timing (the timing bars) | 99.28 / 99.27 [99.41 / 99.41] | — | — | per cell neither dominates: the 1/40 T-grid DP wins 33 cells, loses 168 | #E21, #E22 |
| **L2(hp) `full_belief_ctx`, study winner (trial 62 of 89)** | **98.96** | **510 / 510** | 0 / 510 | −5.78 at `c=0.5, r=0.0625, mean=50` | #E30 |
| **L2(hp) `full_raw`, study winner (trial 66 of 80)** | **98.75** | 509 / 510 | 0 / 510 | −6.44 at `c=0.2, r=0.0625, mean=50` | #E31 |
| **L2(hp) `timing_belief_ctx`, study winner (trial 62 of 80)** | **98.59** | **510 / 510** | **10 / 510, and beats the bar on 2** | −8.40 at `c=0.2, r=0.0625, mean=50` | #E28 |
| **L2(hp) `timing_raw`, study winner (trial 79 of 80)** | **98.42** | 508 / 510 | 5 / 510, and beats the bar on 1 | −13.59 at `c=0.2, r=0.0625, mean=30` | #E29 |
| **fitted rule, event: y^M + 2 at the first decision — shipped** | **97.54** | 509 / 510 | 395 / 510, and **beats the bar on 208** | −0.47 at `c=1, r=2, mean=10` | #E118 |
| myopic_event (the event bar; no exact event DP built here, though the paper solves one — #E43) | 97.44 [97.70] | — | — | — | #E21 |
| **L2(hp) `event_raw`, union winner (shard trial 23 of 80)** | **96.75** | 504 / 510 | **81 / 510, and beats the bar on 15** | −6.22 at `c=0.2, r=0.0625, mean=50` | #E32 |
| L1 `timing_raw` seed 2 (crown 4M), the L1 round's best run | 95.52 | 478 / 510 | 0 / 510 (dp_timing band 0) | −34.4 at `c=0.2, r=0.0625, mean=50` | #E23 |
| random prior_mean (floor) | 81.20 [81.74] | — | — | — | #E21 |

**Shipped: the readback's fitted rules — order = y^M + a first-decision premium (timing +1, event +2,
full none), `cnv_policy.PremiumRule` — with the three generalists they were read from kept loadable as
`cnv_policy.CnvPolicy`.** A choice among trained arms is a claim, and this one rests on the paired
readback (`INTERPRET.md`): every generalist sits under its type's myopic rule on the grid mean (timing
−0.69, full −0.71, event −0.69 on the shared 2048-seed prefix; above it on 2, 0 and 47 of 510 cells),
and the rule distilled from each net beats that net (+0.69 / +0.71 / +0.79) by keeping its structure
and its grid-average premium while dropping its per-cell scatter. The rules reach the myopic rule's level and,
under event observation, pass it — timing +0.004 ± 0.007 over cells, full exactly myopic_full (the
optimum), event **+0.100 ± 0.011** (above myopic_event on 303 of 510 cells) — so the event rule is the
one artifact on this grid that beats the paper's heuristic, and RL's contribution elsewhere is the
structure it recovered. Of the generalists, the belief arms are the ones loaded: higher on the board
(+0.17 / +0.21) and their inputs are two running sums the operator keeps anyway.

What the L1 round says (#E22–#E26): every L0 generalist is a single order quantity applied
to all 510 cells (profit exactly linear in the unit cost, flat in the prior
mean above 30) and lands below the constant prior-mean floor; every L1 learns
the context (the derived recipe's observation / reward normalisers are what
lets the grid axes steer the order) and clears the floor on ~82–94% of the
cells, but **no L1 arm reaches its bar on any cell** (event_raw touches the
band of its lower bar on 13–19 cells, the exception that proves it). The loss sits in one
corner — small prior mean × diffuse prior × high unit cost, where the
generalist over-orders — except timing_raw seed 2, which instead under-orders
on the large, cheap, diffuse cells. The seed spread inside a mode (~3) exceeds
every mode gap on this board, so the L1 round does not order the observation
types. **The tuned round (#E28–#E32)**, one `mdp_tuning` study per mode, two workers,
4M steps per trial, stopped at 80 complete trials (the human's rule,
2026-09-23). The first winner in, `full_belief_ctx`, clears the floor on
**every** cell and lands 0.7% under the per-cell optimum, and it moves the
loss corner: the small-mean × diffuse × expensive cells that broke L1 are now
its best, and the residual sits at the large-mean diffuse cells. Its trial
score was 1.0 optimistic against the protocol re-score — the aggregate a study
optimises is never the record.

#### `fig6a` — the long horizon (N=100) at unit cost 1.0

**No exact bar exists at N=100**: the timing DP would need a ~0.1 TB lattice, and
the full DP is redundant (Prop. 1(b), `dp_full = myopic_full`, verified on all
510 `paper510` cells). The myopic policies are the bars; the gate is "beat the
myopic policy of your own observation type". 8192 protocol seeds.

**RL, one crown per observation mode** (protocol 8192, deterministic, best
screened checkpoint; `ESCALATION.md` entries in the cells). What moved the
record from the first ladder (#E33: 75–91% of the bars at 4M) to the rows
below: a discretized-Gaussian action head in place of the categorical softmax
(#E45), the belief vectors recoded as (log posterior mean, CV) (#E59, #E65),
the raw frame stack read by one per-frame map pooled by the count (#E75–#E78),
one `mdp_tuning` study per mode on top (#E66, #E67, #E79–#E81), and on the raw
line the vine — the main rollout left untouched, with comparisons added on the
side: at 10% of states the world is forked into copies that take sibling
actions from the policy and run to the episode's end on common noise, scored
against each other in a side PPO pass (#E110–#E112) — tuned for itself in one
10M study per raw mode (#E116).

Sorted by profit; % = of the mode's own myopic bar.

| arm | role | profit | % of its bar | the artifact |
|---|---|---|---|---|
| perfect knowledge | relaxed | 804.818 | — | knows λ; unattainable |
| **myopic_full** (= dp_full) | exact | **793.413** | — | the full bar, and the full arms' optimum |
| **PPO `full_raw`** | feasible | **792.94** | 99.9% | the vine on the #E116 study's t48 recipe, frame-MEAN (E78 form), 10M, ckpt 9.5M; −0.48 ± 0.05 paired vs myopic_full (#E116) |
| **PPO `full_belief`** | feasible | **792.07** | 99.8% | `full_belief_log`, study t37's recipe, 20M, ckpt 14M; −1.34 ± 0.07 (#E67) |
| **PPO `timing_raw`** | feasible | **788.06** | 100.0% | the vine on the #E116 study's t44 recipe, frame-MEAN (E78 form), 10M, ckpt 9.5M; +0.28 ± 0.34 — at the bar, not above it (#E116) |
| **myopic_timing** | feasible | **787.776** | — | the timing bar |
| **PPO `timing_belief`** | feasible | **785.22** | 99.7% | `timing_belief_log`, study t47's recipe, 20M, ckpt 20M; −2.55 ± 0.28 (#E66) |
| **myopic_event** | feasible | **781.206** | — | the event bar |
| **PPO `event_raw`** | feasible | **775.25** | 99.2% | the vine on the #E116 study's t53 recipe, frame-MEAN (E78 form), 10M, ckpt 8.5M; −5.95 ± 0.86 (#E116) |
| random prior_mean | floor | 88.874 | — | never learns |
| random uniform | floor | −18,128.8 | — | the did-it-learn-anything floor |

**Shipped: none** — the cell extends the run plan (`paper510`); its crowns are the evidence for the
two stances at N = 100 (`INTERPRET.md`), each loadable as `cnv_policy.CnvPolicy`.

**No arm inside the IR clears a myopic bar by more than its SE here**: timing_raw reaches its bar,
full_raw comes within 0.48 of the full-information optimum, event_raw sits 0.8% under its bar. The
raw crowns lead the recoded belief crowns on both types (+0.9 full, +2.8 timing), because the vine
lifts every raw mode and none of the belief modes (#E111). Off the board, and why: `event_belief_log`
reads 768.97 at L1 (#E107) but is outside the IR's observation menu; a belief estimator fitted by
supervision on the simulator's hidden rate and frozen under the policy reads 775.94 at L1 on one seed
of two (#E109), level with the vine's 775.25 — a distilled rule, never crowned: it learns the exact
posterior in its hand-designed (log mean, CV) form from the hidden λ, and only the order is learned
from reward; a branch-and-grid training lever on `timing_belief_log` reads 786.0–786.4 at its best
checkpoint (#E89, #E91, #E95), not crowned, and its tuned study on `timing_raw` peaked at 783.13
(#E100), under the vine; the R3 assistance rung — the myopic order handed over in the observation and
as the action's anchor — sits +2.5 over myopic_timing (#E88), the only arm above a bar by more than
its SE, outside the IR (a step back), the size of the information prize measured without training
(+2.33, #E85).

Loss recovery of timing over event: **53.8%** (the paper reports ≈54%). myopic_full lands within 1.4%
of perfect knowledge, so 100 periods are nearly enough to learn the rate outright; and the
no-learning floor collapses — 88.9 against a bar of 787.8 — so the floor is a formality and the myopic
bar is the only real test.

#### `fig6b` — the long horizon (N=100) at unit cost 1.5

No study was run on this cell: its RL rows are fig6a's recipes trained here (the belief modes by hp
transfer, #E83; the raw modes under the vine, #E112 with the plain-PPO winners' recipes, #E117 with
the #E116 records). Same protocol, columns and sort as `fig6a`.

| arm | role | profit | % of its bar | the artifact |
|---|---|---|---|---|
| perfect knowledge | relaxed | 348.499 | — | knows λ; unattainable |
| **myopic_full** (= dp_full) | exact | **341.915** | — | the full bar, and the full arms' optimum |
| **PPO `full_raw`** | feasible | **340.91** | 99.7% | the #E116 study's t48 recipe with the vine, 10M, seed 1 ckpt 5M; −1.00 ± 0.08 paired vs myopic_full (#E117) |
| **PPO `full_belief`** | feasible | **340.57** | 99.6% | `full_belief_log`, t37's recipe, 20M, seed 2 ckpt 19M; −1.34 ± 0.11 (#E83) |
| **PPO `timing_raw`** | feasible | **335.56** | 100.1% | the vine on t74's plain-PPO recipe, 10M, seed 2 ckpt 9.5M; +0.26 ± 0.37 — at the bar (#E112); the #E116 t44 recipe reads 335.27 (#E117) |
| **myopic_timing** | feasible | **335.293** | — | the timing bar |
| **PPO `timing_belief`** | feasible | **331.13** | 98.8% | `timing_belief_log`, t47's recipe, 20M, seed 1 ckpt 10M; −4.16 ± 0.24 (#E83) |
| **myopic_event** | feasible | **323.987** | — | the event bar |
| **PPO `event_raw`** | feasible | **318.76** | 98.4% | the #E116 study's t53 recipe with the vine, 10M, seed 2 final network; −5.23 ± 0.75 (#E117) |
| random prior_mean | floor | −411.126 | — | never learns |
| random uniform | floor | −28,170.3 | — | the did-it-learn-anything floor |

**Shipped: none** — as `fig6a`: beyond the run plan, its crowns are evidence for the stances, each
loadable as `cnv_policy.CnvPolicy`.

timing_raw reaches its bar here too; full_raw comes within 1.00 of the optimum; event_raw sits 1.6%
under its bar. The raw crowns lead the belief crowns on both types (+0.4 full, +4.5 timing); the fig6a
tuning carries over on full_raw only (#E117). Off the board: R3 sits +2.7 over myopic_timing (#E88),
the information prize measured without training is +1.95 (#E85). Loss recovery of timing over event:
**63.1%**; myopic_full lands within 1.9% of perfect knowledge, and the no-learning floor is negative
(−411.1). `ESCALATION.md` #E33–#E117 carry every arm behind both boards.

### `2-structural` — over-ordering (confirm), the sufficient statistic (bypass)

From the crowned arms (`INTERPRET.md` carries the evidence and the figure, `ESCALATION.md` #E16, #E19,
#E20 and #E118 the numbers):

- **Confirm, over-ordering — the structure is learned, the per-cell size is not.** On `paper510` every
  generalist is best read as myopic + an additive premium: rebuilt that way from its own orders it
  scores higher than itself (+0.69 to +0.92 per type), and its grid-average premium is the theory's
  shape — +1 at the first decision under timing, +2 under event, none under full (the
  exact DP's mean first-period premium: +0.55 timing, +2.15 event, 0 full). Timing's premium sits below
  event's on most cells (DP 345 of 510; the raw nets 350). At N = 100 the vine-trained `timing_raw`
  crowns over-order at the prior (+2 / +4) and earn most of it back as information (2.81 of the 3.36
  that over- and under-ordering cost at fig6a, 3.56 of 3.69 at fig6b; `INTERPRET.md`'s information ledger); the belief-fed timing crowns do not. On `base` all three types stock one
  unit above the myopic level in period 0, the timing one exactly the DP's answer.
- **Bypass, the sufficient statistic — holds at N = 4, reverses at N = 100.** On `paper510` the raw
  generalists trail the belief ones by 0.17 (timing) and 0.21 (full) on the grid mean; on `base` the
  timing pair plays identically on every seed. At N = 100 the raw crowns lead the belief crowns on both
  types and both cells (+0.9 / +2.8 at fig6a, +0.4 / +4.5 at fig6b), and the ledger says why on timing:
  the raw arm buys the information, the belief arm does not.

## Technical appendix

Run from `cnv/` with the workspace interpreter (`../.venv/bin/python`). Pin threads for anything that
trains. Together these reproduce `results/` from an empty folder; `results/` is gitignored. Every line
below was run on 2026-10-13 — outputs redirected away from the record, the training, tuning and grid lines
at a reduced budget (their full runs take hours; the budgets in the lines are the campaign's).

```bash
# environment: the harness's [domain] extra, plus this folder's requirements.txt
pip install "auto-mdp-solver[domain]==0.11.10"
pip install -r requirements.txt

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# benchmarks — bars and floors on the protocol block (8192 seeds from 0), one row per scenario or per grid cell
python cnv_benchmark_myopic_eval.py -s base --policy timing --n-seeds 8192
python cnv_benchmark_dp_eval.py -s base --policy timing --n-seeds 8192
python cnv_benchmark_random_eval.py -s fig6a --policy prior_mean --n-seeds 8192
python cnv_benchmark_dp_eval.py -g paper510 --policy timing --n-seeds 8192 --shard 1/4   # the slow grid arms: shards 1/4..4/4, then
python cnv_benchmark_common.py --merge dp_timing paper510

# train — specialist (-s) or generalist (-g); --level L1 is the spec-§8.6 derivation; every lever is a flag (--help)
python cnv_ppo_train.py -s base -o timing_belief --level L1 --total-timesteps 4000000 --seed 1
python cnv_ppo_train.py -g paper510 -o timing_belief_ctx --level L1 --total-timesteps 4000000 --seed 1

# tune — one mdp_tuning study per mode; each study's full command is in its ESCALATION.md entry
#        (base #E8, #E14–#E18; paper510 #E28–#E32; fig6a #E66, #E67, #E79–#E81, #E116; their commands name the domain `cnv`, a path from the parent directory — from cnv/ it is `.`)
python -m mdp_tuning . -s base --algo ppo --knobs all --n-trials 200 --total-timesteps 4000000 --eval-seeds 2048 --eval-arg first_seed=2000000

# select and evaluate — every checkpoint screened on 2048 seeds from 1,000,000, the top three confirmed at protocol
python cnv_select.py results/base/PPO_<run> --n-seeds 2048 --top-k 3
python cnv_ppo_eval.py --model-path results/base/PPO_<run>/checkpoints/ppo_cnv_<steps>_steps.zip -s base -o timing_belief --n-seeds 8192

# readback (spec §14) — the probe's self-check, a crown's probe, the fitted rule, the figure, the information ledger
python cnv_policy_probe.py --model-path myopic_timing -g paper510 -o timing_belief_ctx --n-seeds 32 --dp-auto --outdir results/paper510/probe/selfcheck/myopic_timing_timing_belief_ctx
python cnv_policy_probe.py --model-path results/tuning/<study>/trial_<n>/paper510/PPO_<run>/paper510_ppo.zip -g paper510 -o timing_belief_ctx --n-seeds 256 --dp-auto
python cnv_fitted_rule.py --probe results/tuning/<study>/trial_<n>/paper510/PPO_<run>/probe/onpolicy_paper510.tsv -g paper510 -o timing_belief_ctx \
    --form cell_period --n-seeds 2048 --name tb_ctx_cell_period --net-seeds results/tuning/<study>/winner_protocol_paper510.seeds.tsv \
    --bar-seeds results/paper510/benchmark/benchmark_dp_timing_eval_paper510.seeds.tsv
git clone https://github.com/tong-wang/NVLearning results/paper510/reference/NVLearning-repo \
    && cp -r results/paper510/reference/NVLearning-repo/output results/paper510/reference/NVLearning   # the paper's per-cell exact orders
python cnv_plot_policy.py -g paper510 --arm timing_belief_ctx=<probe dir>/onpolicy_paper510.tsv --arm event_raw=<probe dir>/onpolicy_paper510.tsv \
    --paper-dir results/paper510/reference/NVLearning
python cnv_info_ledger.py --model-path results/tuning/<study>/trial_<n>/fig6a/PPO_<run>/checkpoints/ppo_cnv_<steps>_steps.zip -s fig6a -o timing_raw --n-seeds 2048
python cnv_grid_gate.py results/paper510/benchmark/benchmark_fitted_tb_ctx_period_eval_paper510.seeds.tsv \
    --bar results/paper510/benchmark/benchmark_myopic_timing_eval_paper510.seeds.tsv \
    --floor results/paper510/benchmark/benchmark_random_prior_mean_eval_paper510.seeds.tsv \
    --ref results/paper510/benchmark/benchmark_dp_timing_eval_paper510.seeds.tsv   # the per-cell gate

# deploy — the shipped generalists replayed on the raw cnv_mdp loop against their protocol record
python cnv_policy.py
```

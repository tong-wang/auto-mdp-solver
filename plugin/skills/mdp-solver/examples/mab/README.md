# mab — standard stochastic multi-armed bandit

## The problem

K arms, T rounds, one pull per round. At the start of each episode nature
draws every arm's mean payout from a known prior and hides them; the agent
pulls one arm per round and observes **only that arm's payout**. The
objective is the undiscounted total over the finite horizon — equivalently,
minimize regret against the clairvoyant best arm.

What makes it hard is that information is bought with the same currency it
earns. A pull spent learning about an arm is a pull not spent on the arm
currently believed best, and because only the pulled arm reveals anything,
the agent's belief about every other arm goes stale by omission. The horizon
is what prices that trade: with 1,000 rounds left an exploratory pull is
cheap, with 10 left it is nearly pure loss. So the optimal policy is
**horizon-aware** — explore early, commit late — which is why `time_to_go` is
a feature of every observation mode here.

**This is the regret-minimization formulation** (Lai & Robbins 1985; Auer et
al. 2002; Sutton & Barto's 10-armed testbed): fixed finite T, undiscounted,
`objective.discount_factor = 1.0`, and PPO trains at `gamma = 1.0` because
spec §8.6 forces `gamma = beta`. **The discounted / Gittins bandit is a
different problem, not a knob on this one** — it is infinite-horizon with a
geometric discount, its optimal policy is a *stationary* index taking no
horizon input, and UCB1 and Thompson would stop targeting the scored
objective. It belongs as its own composition; not planned as of 2026-07-29.

What is known analytically: no finite-horizon Bayes-optimal reference exists
for this setting, so the campaign brackets rather than closes the gap. The
ceiling is a **clairvoyant oracle** that reads the hidden means (unattainable
by any deployable policy); the field's standard policies — Thompson sampling
(exact conjugate) and UCB1 — are the references the artifact is read against;
myopic-Bayes greedy and uniform random are the floors it must clear. There is
no `exact` arm, which is why every table below reads as a bracket from one
side rather than as a gap to a known target.

**Source.** A textbook problem, not a paper: the standard stochastic MAB with
a per-episode world latent. The authoritative statement of the problem is
`mab_schema.json`; `mab.restatement.md` renders it in prose. The IR is in
**catalog form** — one `payout` slot whose `bernoulli` and `gaussian`
candidates are the two branches — and its `pull` stage is keyed on the chosen
arm, so the whole payout table is fixed by the episode seed and the decision
only selects which entry is revealed. Fingerprints of the shipped schema:
`mdp` **`5bb25e684405`**, structural **`36688c7707f1`**; every move since the
Phase-A freeze is recorded in `ESCALATION.md` §IR-CHANGELOG (**F1**–**F6**).

> ### Scope: this case reports the GAUSSIAN branch only
>
> Bernoulli is fully implemented and differentially gated — it carries its own
> IR instance and the differential covers both branches — but it is
> deliberately **not evaluated**: one payout family carries what this case is
> for. That is a scope decision (operator, 2026-08-11), not an omission, and
> it **bounds** every claim below to Gaussian payouts. Three are most likely to
> be quoted without their qualifier: the #E26 rule's constant `2.5` is fitted
> on Gaussian and its transfer is **untested**; **"bayes ≫ stats"** may be a
> fact about bandits or only about Gaussian ones, and nothing here separates
> those; and the #E34 operating envelope. The four baselines run on Bernoulli
> in minutes for anyone who wants them.

## Layout

Documents — where to read about the case:

| doc | what it holds |
|---|---|
| `mab.restatement.md` | the Phase-A restatement: the problem as the IR states it |
| `ESCALATION.md` | the campaign — design tree, changelogs, numbered findings (#E…) with verdicts |
| `INTERPRET.md` | the policy readback (spec §14): what the trained nets actually do |
| `PLAYBOOK.md` | the case-close digest (guide §10) — 12 lever entries, frame moves, modeling rules |
| `CLAUDE.md` | operating brief for an agent changing this folder |

Code — where things are implemented:

| module | role |
|---|---|
| `mab_schema.json` | **the IR — authoritative** for the problem definition |
| `mab.signoff.json` / `mab.runplan.json` | pipeline state (CONTRACTS.md): the frozen-model sign-off and the confirmed run plan |
| `mab_bayes.py` | conjugate posterior mean/sd builtins, shared by the IR and the gym |
| `mab_uncertainty.py` | `SamplingContext` + the Bernoulli / Gaussian payout generators |
| `mab_scenarios.py` | `MabScenario`, `MabScenarioSource` (world-latent composer), `SCENARIOS` |
| `mab_grids.py` | `MabScenarioGrid` + `GRIDS` — the spec-§5.6 generalist targets |
| `mab_mdp.py` | pure simulator: `MabState`, `init_state`, `advance` (no reward) |
| `mab_gym.py` | `MabEnv` — obs modes `stats` / `bayes` / `bayes_h`, action `arm`, reward `payout` |
| `mab_equinet.py` | `IndexPolicy` — the equivariant per-arm scorer, the deployed policy class |
| `mab_ppo_train.py` / `mab_ppo_eval.py` | SB3 PPO training / spec-§9 eval |
| `mab_benchmark_{random,greedy,ucb1,thompson}[_eval].py` | the four baseline arms + their spec-§9 evals |
| `mab_benchmark_rule[_eval].py` | the #E26 two-constant rule as a benchmark policy — no network, no torch |
| `mab_benchmark_common.py` | shared benchmark seed loop / TSV writer (emits the `oracle_mean` column) |
| `mab_policy.py` | the deployable wrapper over the shipped artifact (spec §12) |
| `mab_ir_adapter.py` | differential adapter (portable-domain contract) |
| `mab_test.py` | the domain's own tests (spec §1.2): engine laws, differential, censoring + belief claims, negative controls |
| `mab_policy_probe.py` | spec-§14 index readback of the crown: surface, `c(ttg)` fit, scored trio (#E24) |
| `mab_interpret.py` | spec-§14 metric battery, replay GIFs, tail analysis (#E24) |
| `mab_stats_probe.py` / `mab_stats_plot.py` | spec-§14 readback of the stats-obs scorer in belief coordinates, and its figures (#E30) |
| `mab_generalist_readback.py` | spec-§14 readback of the generalist + the arm-coverage phase change (#E36, #E37) |
| `mab_grid_probe.py` | the K × T × σ rule-vs-Thompson census incl. the long-horizon ladder (#E32, #E34) |
| `mab_grid_select.py` | re-select a generalist's checkpoint on the grid rather than one cell (#E36) |
| `mab_selection_probe.py` | checkpoint-selection validity: oracle vs callback choice (#E33) |
| `mab_coverage_probe.py` | arm-coverage / starvation accounting (#E31) |
| `mab_a5_probe.py` | distillation ceilings, equivariance error, regret profile; hosts `VecSim` |
| `mab_anneal_probe.py` / `mab_beta_probe.py` | deployment temper β(t) and the anneal sweeps (#E17); the identified-temperature head (#E19) |
| `mab_kl_anchor.py` | the KL-to-Thompson anchor lever (#E31 rung 5) |
| `mab_plot_policy.py` | policy-surface figures |
| `figures/` | the committed static figures the campaign docs inline (spec §14.3): 4 policy/index SVGs and the 3 rung-0 replay GIFs `INTERPRET.md` opens with, downscaled to 700px (~2.5 MB; full size regenerates) |
| `mab_gate_mutation_check.py` | negative control: injects the bugs the simulator gates exist to catch |
| `mab_runs_index.py` | maps escalation actions (`A6b`) ↔ run dirs; regenerates `ESCALATION.md` §RUNS and the TensorBoard label farm |

## Results

### Protocol

Every number below: **8192 episode seeds from 0**, **CRN across arms** (paired
per-seed differences), reference bar **thompson**, record eval
**`--stochastic`** — policy entropy *is* this domain's exploration mechanism,
so an argmax eval under-explores, and once re-pruned an encoding whose value
lived entirely in its sampling (#E30). Deterministic mode is a labelled figure,
never a record; this deviates from the spec §9 template.

Symbols: **`pmᵢ`/`psdᵢ`** = arm *i*'s conjugate posterior mean and sd ·
**`ttg`** = rounds remaining · **`T`** horizon · **`K`** arms · **`n = T/K`** ·
**`c`** = the coefficient in `argmaxᵢ(pmᵢ + c·psdᵢ)` · **`oracle`** =
clairvoyant best-arm return, 1530.7471 · **`regret`** = `oracle − reward_mean` ·
**`% oracle`** = `reward_mean / oracle` · **`±SE`** = eval standard error over
the block. Spec-§9.9 roles are sense-free: `relaxed ≽ opt` (unattainable, never
a gate), `feasible ≼ opt` (any real policy, RL included). There is **no `exact`
row** — the optimum is unknown here, so the board brackets from one side.

### `1-comparative` — how does RL compare with the existing solutions?

**Every scenario is its own leaderboard; scores may never be subtracted across
them** (Bernoulli/Gaussian differ ~1000× in payout scale, and across horizons
the oracle itself scales with T). Only the Gaussian record cell has a board.
`RQ1`/`RQ2` below name the two tier-2 stances declared in `mab_schema.json`
at Phase A; both are stated and answered in `2-structural`.

| scenario | config | what it is for |
|---|---|---|
| `gauss_K10_T1000` | K=10, T=1000, σ=1, μᵢ ~ N(0,1) | **the record cell** — the board below, and the tuning cell every HP was derived at |
| `bern_K10_T1000` | K=10, T=1000, payout 0/1, pᵢ ~ U(0,1) | the second branch: gated, **not evaluated** (scope decision) |
| 32 further `gauss_K{K}_T{T}` cells | K ∈ {5,10,20}, T ∈ {10…40000} | where cell-local claims are re-tested off the tuning cell (#E32, #E34) |
| `gauss_K10_Tlog` (a `GRIDS` entry) | 40 log-spaced cells, T ∈ [500,10000] | the generality target one net is trained across (#E36) |

#### `gauss_K10_T1000` — the record cell

Sorted by `reward_mean` descending; every column in payout units except the
two marked percentages. `oracle_mean` = 1530.7471.

| policy | reward_mean | ±SE | regret | % oracle | note |
|---|---|---|---|---|---|
| oracle (clairvoyant best arm) — `relaxed ≽ opt` | 1530.75 | — | 0.00 | 100% | unattainable by construction; never a gate |
| **rule `argmax(pmᵢ + 2.5·(ttg/T)^0.15·psdᵢ)`** | **1486.14** | **6.47** | **44.61** | **97.1%** | **+22.95 vs thompson, z=36.8 — two constants, no torch (#E26)** |
| free rule `argmax(pmᵢ + √(2 ln(ttg/nᵢ))·psdᵢ)` | 1464.87 | 6.52 | 65.88 | 95.7% | best **parameter-free** form; −21.27 vs tuned (#E27) |
| thompson (exact conjugate) | 1462.38 | 6.52 | 68.37 | 95.5% | the reference bar |
| 12-knot readback of the A8-a net | 1459.56 | 6.33 | 71.19 | 95.3% | 98% action agreement with the net it fits (#E24) |
| PPO obs=bayes index @ tuned HP (**A8-a**, 3 seeds) | 1453.57 | 4.80 | 77.18 | 95.0% | best **trained** config, 99.4% of thompson (#E23) |
| PPO obs=bayes index + ttg-weighted entropy (**A7-t2**, 3 seeds) | 1445.32 | — | 85.43 | 94.4% | +75.40 vs its control, t(2df)=4.44 (#E21) |
| A6b crown + deployment temper β(t) | 1445.18 | 6.35 | 85.57 | 94.4% | post-hoc calibration (#E17) |
| ucb1 (sigma-scaled) | 1440.79 | 6.52 | 89.95 | 94.1% | reference, not a must-beat bar |
| **PPO obs=stats index @ tuned HP (A11-a, 3 seeds)** | **1408.55** | **5.00** | **122.20** | **92.0%** | the reopened encoding: −45.02 vs bayes at the same centre (#E29); a **learned Thompson** (#E30) |
| PPO obs=bayes index policy (**A6b**, deployed artifact) | 1387.57 | 6.30 | 143.18 | 90.6% | PASS: z=41.01 vs greedy (#E14) |
| PPO obs=bayes L1 MLP + frame-avg 16 | 1353.35 | 6.51 | 177.40 | 88.4% | PASS (z=36.64 vs greedy) |
| PPO obs=bayes L1 MLP (raw single-pass) | 1314.46 | 6.27 | 216.29 | 85.9% | PASS (z=33.02 vs greedy) |
| PPO obs=bayes L1 MLP + belief shaping | 1293.03 | 6.36 | 237.72 | 84.5% | PASS vs bar, but −21.4 vs raw (z=−2.40) |
| PPO obs=stats MLP @ tuned HP (A11-b) | 1060.81 | 7.50 | 469.94 | 69.3% | HP alone rescues +133; the index arch adds +348 (#E29) |
| greedy (myopic Bayes) | 1015.69 | 6.52 | 515.05 | 66.4% | baseline the artifact must clear |
| PPO obs=stats L1 (best: norm_obs=True) | 927.63 | 6.88 | 603.11 | 60.6% | FAIL (z=−9.29 vs greedy) — superseded by A11 (#E29) |
| PPO obs=stats L0 | 732.17 | 7.57 | 798.58 | 47.8% | FAIL (z=−28.38 vs greedy) |
| PPO obs=bayes L0 | −1.67 | 10.38 | 1532.42 | −0.1% | FAIL (z=0.08 vs random) |
| random | −2.51 | 3.50 | 1533.26 | −0.2% | the floor |

No arm is absent: all six declared benchmarks appear, plus every trained arm
scored at protocol. The Bernoulli branch has **no board at all** — see the
scope note above.

**Shipped: the A6b artifact**, which `mab_policy.py` currently wraps
(`PPO_obsbayes_L1_policyindex_normobsFalse_seed1_20260729_185003/`). It is
**not** the top row, and that is a live decision rather than an oversight: the
two better objects — the A8-a seeds, and the two-constant rule of #E26, which
needs no artifact at all — are candidates in an open **★-crown / packaging
call** that belongs to the operator (`ESCALATION.md` §Frontier). The rule is
the best object on this branch and the first policy here to beat Thompson, but
shipping a fitted rule over a trained net is a claim about what the case is
*for*, which is the operator's to make.

**Reading the board.** `±SE` is the spread of one policy's mean over the seed
block; it is **not** the training-seed sd. Comparing two *trained* rows needs
that instead (16.70), so single-seed contrasts below ~50 points are not
established (#E21); rule-vs-rule contrasts are deterministic on CRN and use the
eval SE. `INTERPRET.md` quotes `±` as seed-sd across 3 training seeds, which
answers a different question. And `reward_mean` is not comparable across seed
blocks or horizons — `regret` nearly is: a 20–39% regret effect reads as ~1% in
reward and was nearly discarded as noise (#E28, #E35).

**The rule's two constants were not fitted on the block they are reported on.**
The quantile sweep ranks candidates on a separate 2048-seed block; only the top
three are then scored once on the 8192-seed record block — the same
screen-and-confirm discipline the campaign asks of its trained policies,
applied to a two-parameter fit.

How it was reached: `ESCALATION.md` §MAP and #E7 → #E14 → #E23 → #E26.

### `2-structural` — the declared stances, and what else the readbacks found

`mab_schema.json` declares **two** tier-2 stances, both fixed at Phase A
alongside the objective. They are the questions this campaign set out to
answer, and only they owe §14 artifacts. Evidence and the full readbacks are in
`INTERPRET.md`; this is the summary, not a second copy.

**RQ1 · `confirm` — "the trained net's action rule is an index policy of the
classical form, recoverable from the artifact." CONFIRMED.** The readback
recovers `argmax(pmᵢ + c·psdᵢ)` with c ≈ 1.5 — a fixed ~93rd posterior
percentile — at **98% action agreement** with the net it fits, and the 12-knot
fit scores 1459.56, within 6 points of the net. The probe was validated on
known answers first: it recovered UCB1's analytic bonus to 0.12% median error.
The artifact is an **equivariant index policy** (`mab_equinet.IndexPolicy`):
one scoring MLP φ(meanᵢ, sdᵢ, ttg) shared across arms, so relabeling the arms
relabels the policy exactly — equivariance error TV 2.3e-8 against the plain
MLP's 0.298, at +73.11 and a third of the parameters. Every *enlargement* of
that class failed (mean/max pooling, attention, a learned temperature head):
findability, not capacity.

**RQ2 · `discover` — "the recovered coefficient depends on time-to-go, and the
fitted rule scored as a benchmark beats thompson." CONFIRMED.** The level, not
the shape, was the net's error: sweeping `c` in behaviour space with no
training gives `c = 2.5·(ttg/T)^0.15`, scoring **1486.14, beating Thompson by
+22.95 (z=36.8)** — the first policy here to do so. No parameter-free form
comes within 21 points, and every one *over*-explores: with a known prior and
horizon the optimal identification rate is ~0.93, not the ~0.95 the classical
rules buy. The rule ships as a first-class spec-§9 benchmark
(`mab_benchmark_rule_eval.py`), not as a probe artifact.

#### Further structural findings — **not** declared questions

The two below are **by-products**: they came out of chasing RQ1 and RQ2 and
were never pre-registered, so they carry no `research_questions` stance and owe
no §14 deliverable. They are reported here because they are structural in kind
and because the first of them **bounds RQ2's answer** — but the distinction is
kept visible, since a question the campaign committed to in advance and a
finding it stumbled into are different evidence.

**The coefficient family is horizon-inconsistent, and that bounds RQ2**
(#E32, #E34). Off its tuning cell the rule still beats Thompson in **all 21
cells** of the K × n grid, capturing ~⅓ of Thompson's closable gap — but the
*level* follows **`c* = 1.204 + 0.286·ln n`**, so the fitted 2.5 is cell-local
and only the shape transfers. Asymptotically the family is **inconsistent**:
regret **linear in T** (R² 0.99) where Thompson's is **logarithmic** (R² 0.99),
because a fixed quantile never grows and an abandoned arm is never revisited.
They cross at **T ≈ 14,000 total rounds**, near-independent of K (14,149 at
K=10; 14,513 at K=20) — quote the *horizon*, not pulls-per-arm, since n\*
varies 2× across K while T\* barely moves. The repair must **grow** with
elapsed time: `log:0.7` turns every crossover back into a win (+15.6 / +17.0 /
+12.7) and drives starvation to ~0, at 8–10 points of relative gain in the
short-horizon cells, while a *constant* floor does nothing (−58.1 → −58.2).
**Every repair converges toward Thompson** — the two are the same form with
`q ~ N(0,1)` drawn instead of scheduled — so the rule's advantage *is* its
finite-horizon specialization. That is where the operator stopped the policy
search. One honest limit: closing ⅓ of *Thompson's* gap is not being ⅓ of the
way to *optimal*, and with no finite-horizon Bayes-optimal reference here the
rule's absolute quality stays unmeasured.

**A net trained across a horizon grid did not recover the schedule**
(#E36, #E37). It works as a *policy* — trained on `gauss_K10_Tlog` it **ties
Thompson at T=500 (1.05×)**, which no specialist here has done, costing +10% at
T=1000 and +76% at T=10000 budget-matched — but the readback says it learned a
**fixed quantile, c ≈ 0.85 constant**: exponent 0.003 against RQ2's 0.15 on
`ttg/T`, and 0.003 against 0.286 on `T`, *despite being handed both raw and
never their ratio*. It sits at ~⅓ of `c*(n)`, and the shortfall widens with the
horizon because `c*` grows and `c` does not. The finding above already showed
that class inconsistent, so the rest follows: regret near-linear in T
(R² 0.9999), and at T=10000 only ~6 of 10 arms are ever pulled, the runner-up
getting a median of 2 pulls out of 10,000. **It carries a stated confound**: a
scalar `gae_lambda` cannot hold credit coverage constant across a 20× T span,
so the worst cell is also the thinnest-coverage cell, and this round does
**not** separate generality from credit starvation (upstream issue #6). The
engineering finding behind that confound is below.

### `3-engineering` — which encoding, architecture and HP train best?

Digested; each line's full argument is the cited entry, and the reusable form
is in `PLAYBOOK.md`.

- **Set `gae_lambda` by coverage, not by steps (#E35)** — the confound behind
  the generalist finding above. Coverage = `(1/(1−gae_lambda))/T`. Carrying A8-a's tuned
  `0.98784` (82 steps) to longer horizons costs 20–39% of regret at
  T = 2,000–10,000, and at T = 20,000 — 0.41% coverage — **two of three seeds
  never learn at all**. On this domain the break sits between 0.41% and 0.82%.
  **Do not carry the number to another domain**: `game2048` measured the
  opposite direction, its summit at λ=0.90. Both campaigns confirm the
  *mechanism* and jointly refute any numeric band (upstream issue #4). It is
  invisible in the logs you would normally watch — selection curves are in
  reward, where a 20% regret change is ~1%. With coverage held constant PPO
  sits at a flat 2–3× Thompson from T = 2,000 to 20,000 and does **not**
  diverge, so the long-horizon deficit is a constant factor, not a widening gap.
- **Three simulators, and the rule for using them.** The canonical
  `mab_mdp` + `mab_gym` *is* the domain and produces every number that gates,
  ships or enters a leaderboard. `mab_a5_probe.VecSim` is a **bit-exact**
  replica of the record cell (~10×) for work that must line up seed-for-seed.
  `mab_grid_probe.GridSim` is a **distributional** reimplementation (~1000×)
  for sweeps whose cost is (policies × cells × seeds × T); it abandons the
  domain's seed scheme, so **quote paired differences from it, never levels** —
  on its own block thompson reads 1475.13 against the committed 1463.19. Each
  fast path declares its equivalence and carries a gate in `mab_test.py`;
  `mab_gate_mutation_check.py` is the negative control proving neither
  `GridSim` gate is redundant.
- **Selection and early stopping (#E33).** The 256-seed callback picks the
  oracle checkpoint in 5 of 6 runs — noise costs +1.44 against the +42.90 that
  selecting at all is worth. Plateau early-stopping is **off**: it fires before
  the true peak. That was a deliberate deviation when the campaign ran, and
  v0.7.0's spec §8.6 has since made it the rule, partly on this evidence.
- **Training a generalist: build the selection env from the grid, not from
  `-s`.** These runs shipped checkpoints selected on a single cell because the
  callback used the unused `-s` default; nothing failed, because the selection
  log was a valid evaluation of the wrong scenario. Fixed in
  `mab_ppo_train.py`; `mab_grid_select.py` re-selects for any run predating the
  fix (cost ~0.8%).

## Technical appendix

Run from the case folder, using the project's own virtualenv. Thread pinning matters on
shared boxes — torch oversubscribes. Together these reproduce `results/` from
an empty folder; `results/` is gitignored. **The gate commands are not here —
they are in `CLAUDE.md`**, because they are run by whoever is changing the
folder rather than reading it.

```bash
# baselines — writes results/{scenario}/benchmark/benchmark_{method}_eval_{scenario}.tsv
python mab_benchmark_random_eval.py   -s gauss_K10_T1000 --n-seeds 8192
python mab_benchmark_greedy_eval.py   -s gauss_K10_T1000 --n-seeds 8192
python mab_benchmark_ucb1_eval.py     -s gauss_K10_T1000 --n-seeds 8192
python mab_benchmark_thompson_eval.py -s gauss_K10_T1000 --n-seeds 8192
# the fitted-rule row — no artifact, no torch
python mab_benchmark_rule_eval.py     -s gauss_K10_T1000 --n-seeds 8192

# train — script defaults, which are NOT the config behind the leaderboard's trained rows.
# NOTE: this script uses a live in-training selection callback. v0.7.0's spec §8.6/§9.7
# replaced that with a post-hoc three-layer screen; the spec cites this campaign's
# #E33 and #E36 V4 for the change. Every number above was produced under the older
# design and the code is kept as it ran — a re-run should use the new path.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python mab_ppo_train.py -s gauss_K10_T1000 -o bayes

# A8-a — the config of record for every "PPO obs=bayes index @ tuned HP" row,
# A3's harvested trial 6 re-ranked at protocol (#E28). Three seeds.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python mab_ppo_train.py \
  -s gauss_K10_T1000 -o bayes --policy index \
  --learning_rate 1.58e-05 --lr-final 1.58e-06 \
  --n-steps 2048 --batch-size 32 --n-epochs 20 \
  --gae-lambda 0.98784 --ent-coef 3.44e-05 --vf-coef 0.381 \
  --no-norm-obs --seed 1

# evaluate — the record eval is stochastic
python mab_ppo_eval.py -s gauss_K10_T1000 --stochastic \
  --model-path results/gauss_K10_T1000/<run>/gauss_K10_T1000_ppo.zip

# readback + figures
python mab_policy_probe.py --part all          # spec-§14 index readback of the crown
python mab_interpret.py                        # metric battery, replays, tail
python mab_generalist_readback.py --part all   # the #E36 generalist
python mab_plot_policy.py

# deploy
python mab_policy.py -o bayes \
  --model-path results/gauss_K10_T1000/<run>/gauss_K10_T1000_ppo.zip

# action -> run-dir index, and the TensorBoard label farm
python mab_runs_index.py
tensorboard --logdir results/by_action
```

Run names encode the *config*, not the experiment that motivated them, so to
go from an escalation action (`A6b`) or a ledger entry (`#E14`) to its
artifacts, read the generated table in `ESCALATION.md` §RUNS.

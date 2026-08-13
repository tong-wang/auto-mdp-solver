# mab — standard stochastic multi-armed bandit

K=10 arms, T=1000 rounds. Nature draws the hidden arm means fresh each
episode (world latent); the agent pulls one arm per round, observes only
that arm's payout, and maximizes total payout. Two independent branches
with separate leaderboards (payout scales differ, no head-to-head):

| branch | scenario | a pull pays | means drawn each episode |
|---|---|---|---|
| Bernoulli | `bern_K10_T1000` | 0/1, success prob p_i | p_i ~ Uniform(0,1) i.i.d. |
| Gaussian | `gauss_K10_T1000` | N(mu_i, 1), sigma=1 known | mu_i ~ N(0,1) i.i.d. |

> ### Scope of every result below: the GAUSSIAN branch
>
> This case reports the **Gaussian branch only**, by design. Bernoulli is
> fully implemented and differentially gated — it is a candidate of the same
> `payout` slot, it carries its own IR instance, and the differential covers
> both branches — but it is deliberately **not evaluated**: one payout family is enough to carry what this case is for, and
> a second would double the evaluation surface without adding a shape the
> pipeline has not already been shown.
>
> That is a scope boundary, not an omission, and it does not weaken any claim
> below — but it does **bound** them. Read every result as a statement about
> Gaussian payouts. Three in particular, because they are the ones most likely
> to be quoted without their qualifier:
>
> - the **#E26 rule** `argmax(pm + 2.5·(ttg/T)^0.15·psd)` and its beating
>   Thompson — the constant `2.5` is fitted on Gaussian, and its transfer to
>   Bernoulli is **untested** (#E27 identified that transfer as the sharpest
>   available test of the constant, and it was not run);
> - **"bayes ≫ stats"** (+458.73), the campaign's largest single lever — it
>   may be a fact about bandits or only about *Gaussian* bandits, and nothing
>   here separates those;
> - the **#E34 operating envelope** and its T ≈ 14,000 crossover.
>
> Anyone wanting the Bernoulli numbers can produce them — the branch runs
> today, and the four baselines take minutes
> (`mab_benchmark_{random,greedy,ucb1,thompson}_eval.py -s bern_K10_T1000`).
> They are simply not part of what this case claims.

## Which bandit this is: finite-horizon and **undiscounted**

This is the **regret-minimization** formulation (Lai & Robbins 1985; Auer et
al. 2002; Sutton & Barto's 10-armed testbed): a fixed finite horizon T=1000 and
an undiscounted objective, max E[sum_{t=1..T} r_t] — equivalently, minimize
regret T·mu* − E[sum r_t]. The IR carries `objective.discount_factor = 1.0`,
PPO trains with `gamma = 1.0` (spec §8.6 forces gamma = beta), and `time_to_go`
is a feature of both observation modes, because the optimal policy under a
finite horizon is horizon-aware: explore early, commit late.

**If you want the discounted bandit, that is a different problem — not this
one.** The Bayesian / Gittins formulation (Gittins & Jones 1974) is
*infinite-horizon* with a geometric discount, max E[sum_{t=0..inf} beta^t r_t],
beta < 1. There the discount is not a tuning knob: it is what makes the sum
converge and what makes the Gittins index theorem hold, and the optimal policy
is a *stationary* index rule that takes no horizon input at all. Setting
beta < 1 on this branch would not simply be a harder version of it — UCB1 and
Thompson both target undiscounted regret and would stop optimizing the scored
objective, the regret column would stop meaning what it says, and `time_to_go`
would become vestigial.

So it belongs as its own composition rather than as a knob here. It would also
arrive with something no branch above has: an **exactly optimal** reference
policy — the Gittins index, computable by DP for the Beta/Bernoulli case —
instead of a clairvoyant-best-arm oracle. Not planned as of 2026-07-29.

(A third, unrelated use of discounting: discounted / sliding-window UCB
(Garivier & Moulines 2008) down-weights stale data in *non-stationary*
bandits. That changes the algorithm, not the objective, and is orthogonal to
both formulations above.)

Generated from the IR `mab_schema.json` (Phase A frozen 2026-07-24; seed scheme
v2); restatement of record: `mab.restatement.md`. Differential vs the IR
interpreter is bit-exact on both compositions, with zero invariant violations.

The IR is in the **catalog form**: the problem has one source of randomness —
the pulled arm's payout — and Bernoulli / Gaussian are two *candidates* of its
single `payout` slot, sharing one stream identity across both seed branches.
Each candidate draws the ten arm means as one `iid` latent (`{of, size}`,
solver ≥ v0.5.7), desugared at load into the hidden `payout_arm_means` constant
that the pulled arm indexes.

The `pull` stage is **keyed on the chosen arm** (`key_exprs: ["int(arm)"]`,
solver ≥ v0.5.9), which refines the period slot into K independent per-arm
streams of which a round reads exactly one. The whole payout table is therefore
fixed by the episode seed and the decision only selects which entry is
revealed. Before this (2026-07-29) a single primitive variate per period was
shared by all ten arms, so the round's luck was settled before the choice:
Gaussian payouts differed across arms by exactly their means, and Bernoulli
payouts were comonotone in p. The law the agent saw was still correct — one arm
is revealed per round — but it is not the standard bandit model. `mab_test.py`
carries the regression test.

Fingerprints of the shipped schema: `mdp` **`e385f662a857`**, structural
**`36688c7707f1`**. Both moved after the Phase-A freeze and each move is
recorded in §IR-CHANGELOG — **F3** (K and T become axis-tagged scenario
constants), **F4** (state-variable bounds widened for the long-horizon
instances) and **F5** (`gaussian` becomes the slot default, with `bernoulli`
gaining an explicit instance so it stays gated). Earlier values, in order: `546617c32435` / `072c7f55c967` pre-F3;
`cb3cf419ebf9` / `8cc218500b55` before the arm key; `ad610581f715` when the
branches were two guarded sources with twenty scalar latent constants.

## Layout

| file | role |
|---|---|
| `mab_schema.json` | MDP-IR (frozen `mdp` block + gym/rl design axes) |
| `CLAUDE.md` | the domain's operating brief (spec §1, from the skill's template) — pointer-first: where to look and what will bite |
| `PLAYBOOK.md` | case-close digest (guide §10): lever entries (context/symptom/diagnosis/prescription/failed), frame moves, modeling rules |
| `mab_bayes.py` | conjugate posterior mean/sd builtins (shared by IR + gym) |
| `mab_uncertainty.py` | SamplingContext + Bernoulli/Gaussian payout generators |
| `mab_scenarios.py` | `MabScenario`, generic `MabScenarioSource` (world-latent composer), `SCENARIOS` |
| `mab_mdp.py` | pure simulator: `MabState`, `init_state`, `advance` (no reward) |
| `mab_gym.py` | `MabEnv` — obs modes `stats` / `bayes`, action `arm`, reward `payout` |
| `mab_ir_adapter.py` | differential adapter (portable-domain contract) |
| `mab_test.py` | the domain's own tests (spec §1.2): engine laws, differential over the covering set, censoring + belief claims, negative controls |
| `mab_benchmark_{random,greedy,ucb1,thompson}[_eval].py` | baselines + spec-§9 evals |
| `mab_benchmark_rule[_eval].py` | the #E26 two-constant rule as a benchmark policy — the ★ deliverable, no network, scored by the same §9 loop |
| `mab_benchmark_common.py` | shared benchmark seed loop / TSV writer |
| `mab_ppo_train.py` / `mab_ppo_eval.py` | SB3 PPO training / spec-§9 eval (`--stochastic`) |
| `mab_equinet.py` | `IndexPolicy` — equivariant per-arm scorer (the deployed policy class) |
| `mab_grids.py` | `MabScenarioGrid` + `GRIDS` — §5.6 generalist targets (`gauss_K10_Tlog`, #E36) |
| `mab_a5_probe.py` | diagnostic suite: distillation ceilings, equivariance error, regret profile |
| `mab_policy.py` | deployable policy wrapper (spec §12) |

Probes — each is the reproduction path for a finding that cites it, and none
is optional scaffolding (the §14 readbacks especially: a number that only
exists in a deleted script is not a result):

| probe | reproduces |
|---|---|
| `mab_interpret.py` | the §14 metric battery, replay GIFs, tail analysis (#E24) |
| `mab_policy_probe.py` | §14 index readback of the crown: surface, `c(ttg)` fit, scored trio (#E24) |
| `mab_stats_probe.py` | §14 readback of the stats-obs scorer in belief coordinates (#E30) |
| `mab_stats_plot.py` | #E30 figures (`figs/fig_stats_*.svg`) from the stats-probe artifacts |
| `mab_generalist_readback.py` | §14 readback of the #E36 generalist — is the index ratio-only? plus the arm-coverage phase change (#E37) |
| `mab_grid_probe.py` | the K × T × σ rule-vs-Thompson census, incl. the long-horizon ladder (#E32, #E34) |
| `mab_grid_select.py` | re-select a generalist's checkpoint on the grid rather than one cell (#E36) |
| `mab_selection_probe.py` | checkpoint-selection validity: oracle vs callback choice (#E33) |
| `mab_coverage_probe.py` | arm-coverage / starvation accounting (#E31) |
| `mab_anneal_probe.py` | deployment temper β(t) and the anneal sweeps (#E17) |
| `mab_beta_probe.py` | the identified-temperature head (#E19) |
| `mab_kl_anchor.py` | the KL-to-Thompson anchor lever (#E31 rung 5) |
| `mab_plot_policy.py` | policy-surface figures |
| `mab_gate_mutation_check.py` | negative control: injects the bugs the simulator gates exist to catch, proving neither gate is redundant |
| `mab_runs_index.py` | maps escalation actions (`A6b`) ↔ run dirs; regenerates ESCALATION.md §RUNS + `by_action/` TB labels, across **all** scenarios (`-s all`, default) |

Observation modes (both lossless belief parametrizations, 21 features):
`stats` = per-arm pull counts + payout totals + time-to-go;
`bayes` = per-arm conjugate posterior mean + sd + time-to-go.

## The rule's operating envelope — where it wins, and where it stops (#E32, #E34)

`argmax_i(pm_i + 2.5·(ttg/T)^0.15·psd_i)` beats Thompson across **all 21 cells**
of the K × n grid it was tested on (K ∈ {5,10,20}, n = T/K ∈ {2…100}), capturing
**~a third of Thompson's closable gap**, (rule − thom)/(oracle − thom). It is
not universally better, and the boundary is now measured:

| n = T/K | K=5 | K=10 | K=20 |
|---|---|---|---|
| 20 | 30.7% | 32.0% | **33.4%** |
| 100 | 35.2% | **33.7%** | 31.5% |
| 200 | **36.2%** | 32.7% | 26.9% |
| 500 | 35.9% | 25.1% | 12.8% |
| 1000 | 33.2% | 10.7% | **−11.1%** |
| 2000 | 24.7% | **−10.7%** | **−58.1%** |

**Crossover at T ≈ 14,000 total rounds**, near-independent of K (measured
14,149 at K=10, 14,513 at K=20). Below it, use the rule; above it, use
Thompson. Quote the *horizon*, not pulls-per-arm — n\* varies 2× across K
(1415 vs 726) while T\* barely moves, because both regret curves scale roughly
linearly in K and K cancels from the crossing condition.

**Why**: the rule's regret is **linear** in T (R² 0.99), Thompson's is
**logarithmic** (R² 0.99). A fixed quantile never grows, so an abandoned arm is
never revisited — P(best arm untouched) stays constant along each column and
those episodes cost ∝ T. This is the textbook inconsistent-vs-consistent
crossing; what is ours is the *constants*, not the shape.

**If you need the long-horizon regime**, the one-term fix is `log:0.7` —
`c = max(2.5·(ttg/T)^0.15, 0.7·√(2 ln t))` — which turns every crossover back
into a win (+15.6 / +17.0 / +12.7 at the three crossed cells) and drives
starvation to ~0, at a cost of 8–10 points of relative gain in the
short-horizon cells. A *constant* floor does not work (−58.1 → −58.2): the
repair must **grow** with t.

Two honest limits on the claim. Every fix converges toward Thompson — the two
policies are the same form `argmax_i(pm_i + q_i·psd_i)`, ours fixing
`q_i = c(ttg)` and Thompson drawing `q_i ~ N(0,1)` — so the rule's advantage
*is* its finite-horizon specialization. And closing a third of *Thompson's*
gap is not being a third of the way to *optimal*: this campaign has no
finite-horizon Bayes-optimal reference, so the rule's absolute quality is
unmeasured.

## Training PPO at long horizon — set `gae_lambda` by *coverage* (#E35)

Carrying a tuned `gae_lambda` to a longer horizon is the one configuration
mistake that can stop training outright rather than merely degrade it. What
matters is **coverage** — the credit horizon as a fraction of the episode:

    coverage = (1 / (1 - gae_lambda)) / T          # the lens, not a target

A8-a's tuned `0.98784` is 82 absolute steps: 8.3% of a 1,000-round episode and
**0.41%** of a 20,000-round one. Regret at 2048 seeds, three training seeds per
cell, against a same-path Thompson reference:

| T | `fix` = 0.98784 (coverage) | `scl` = 1 − 12/T (8.33%) | Thompson |
|---|---|---|---|
| 2,000 | 187.6 (4.11%) | **150.2** | 78.5 |
| 5,000 | 379.3 (1.64%) | **230.8** | 95.5 |
| 10,000 | 469.8 (0.82%) | **363.0** | 110.0 |
| 20,000 | **4003.2** (0.41%) | **286.6** | 128.3 |

The first three cells cost 20–39% of regret. The fourth is a different failure:
at 0.41% coverage **two of three seeds never learn**, sitting flat at 21–25k
reward from 20M steps through 400M while every `scl` seed clears 29k by 60M.
On this domain the break sits between 0.41% and 0.82%.

> **Do not carry that number to another domain.** These cells put mab's optima
> at ~8–17% coverage, and a second campaign in this pipeline (a tile-puzzle
> domain) pre-registered its own λ sweep and measured the *opposite* direction:
> 0.98+ losing at both scales, its summit at λ=0.90, optima at ~2–4% coverage.
> Both campaigns confirm the **mechanism** — credit horizon versus where
> consequences realize — and jointly refute any numeric band. An earlier draft
> of this section proposed "aim ~8%, never below ~1%"; upstream rejected that
> on the second campaign's evidence, because it would have pushed that domain
> the wrong way, and restated spec §8.6's λ row as mechanism with no numbers
> (issue #4). Coverage is the lens and the re-derivation trigger; the number is
> this domain's.

**It is invisible in the logs you would normally watch.** Selection curves are
in *reward*, where a 20% regret change is ~1% — so a mis-configured run looks
like a slightly worse one. Read regret (#E28).

Two consequences worth carrying: with coverage held constant PPO sits at a
**flat 2–3× Thompson from T = 2,000 to 20,000 and does not diverge**, so the
long-horizon deficit is a constant factor, not a widening gap; and `gae_lambda`
is **not a transferable tuned constant** on this domain — it is per-instance,
because the instance family varies T. Spec §8.6 states the row in absolute
steps, which is what licenses the mistake; see
upstream issue #4 (https://github.com/tong-wang/auto-mdp-solver/issues/4).

## One policy across horizons — the generalist (#E36)

Everything else trained here is a **specialist** at one cell. `mab_grids.py`
defines a §5.6 `ScenarioGrid` (`gauss_K10_Tlog`: 40 log-spaced cells,
T ∈ [500, 10000]) so one net can be trained across a range, with `-g/--grid_name`
selecting it, obs mode `bayes_h` (which adds the episode's own `T`), and policy
`index_h`. Per-cell regret, 3 seeds, budget-matched to the specialists:

| cell | coverage | generalist | vs Thompson | vs specialist |
|---|---|---|---|---|
| T=500 | 33% | **61.96** | **1.05×** | (none trained) |
| T=1000 | 16.7% | 85.14 | 1.25× | +10% (A8-a 77.18) |
| T=10000 | 1.67% | 637.74 | 5.80× | +76% (#E35 363.0) |

It works — and at T=500 it **ties Thompson**, which no specialist here has done.
But generality costs, and the cost concentrates at long horizon.

**The confound is in the table.** A scalar `gae_lambda` cannot hold coverage
constant on a grid spanning 20× in T, so the worst cell is also the
thinnest-coverage cell. How much of the +76% is generality and how much is
credit starvation is **not separated** by this round. If you train a generalist
here, derive λ from the grid's *shortest* cell and declare the coverage you end
up with at the longest — see upstream issue #6 (https://github.com/tong-wang/auto-mdp-solver/issues/6).

**What it actually learned (#E37 readback)**: a **fixed quantile** —
`argmax_i(pm_i + c·psd_i)` with **c ≈ 0.85, constant**. Not a function of
`ttg/T` (exponent 0.003 against the #E26 rule's 0.15) and not of `T`
(0.003 against #E32's 0.286), despite being handed both raw and never their
ratio. It is ~⅓ of the fitted optimum `c*(n) = 1.204 + 0.286·ln(T/K)`, and the
shortfall widens with the horizon because `c*` grows and `c` does not. #E34
already proved that class is inconsistent, so the rest follows: regret near
linear in T, and at T=10000 only ~6 of 10 arms ever pulled, the runner-up
getting a median of 2 pulls out of 10,000. Reproduce with
`mab_generalist_readback.py --part all`.

**Trap when training on a grid**: the selection callback must build its env from
the *grid*, not from `-s`. These runs shipped checkpoints selected on a single
cell because it used the unused `-s` default; nothing failed, because the
selection log was a valid evaluation of the wrong scenario. Fixed in
`mab_ppo_train.py`; `mab_grid_select.py` re-selects on the grid for any run
predating the fix (the cost turned out to be ~0.8%).

## Simulators — three of them, and when each may be used

The domain deliberately carries **more than one implementation of the MDP
core**. That is licensed by the pipeline's doctrine — the *schema* is the
source of truth, not any one implementation, which is the same premise
`mdp_ir.differential` rests on when it diffs an independent IR interpreter
against this code. But plurality is only safe when the equivalence is
**proven and continuously re-provable**, so each fast path must declare what
equivalence it claims and carry a gate in `mab_test.py` that asserts it.

| simulator | claims | speed | gate |
|---|---|---|---|
| `mab_mdp` + `mab_gym` (**canonical**) | *is* the domain | ~1x | `mdp_ir.differential` vs the IR interpreter, base + all 34 instances (35 compositions, both payout branches) |
| `mab_a5_probe.VecSim` | **bit-exact replica** of `gauss_K10_T1000` | ~10x | `test_vec_sim_is_bit_exact_against_the_gym` (payout + obs error exactly 0.0) |
| `mab_grid_probe.GridSim` | **distributional reimplementation**, any (K, T, σ, prior) | ~1000x | `test_grid_sim_posterior_matches_the_domain_belief` + `..._is_calibrated_off_the_base_cell` |

**Use the canonical path by default.** Every number that gates, ships or
enters a leaderboard comes from `mab_ppo_eval.py` / the benchmark evals, which
run the gym.

**Use `VecSim`** when you need the *same* episodes as the domain, faster:
replaying a shipped artifact, or any sweep that must line up seed-for-seed
with recorded numbers (#E24, #E31). It reproduces the seed tree exactly, so
its outputs are directly comparable to committed results. It hard-codes
K=10 / T=1000 / σ=1 and the N(0,1) prior — `test_vec_sim_constants_still_match_the_base_instance`
is the tripwire that fires if an IR edit invalidates that.

**Use `GridSim`** only when the experiment is infeasible through the others —
in practice, sweeps whose cost is (policies × cells × seeds × T) rather than a
single evaluation: **#E32's** 21-cell grid with a per-cell (c, p) re-tune.
(#E26's quantile sweep and #E31's floor sweep ran on `VecSim`, not this — they
sweep policies over ONE cell, which the bit-exact replica already makes cheap.
GridSim earns its licence only when the CELL is what varies.) It abandons the domain's seed scheme for
a vectorized per-step draw, so **it can never be diffed episode-by-episode
against the domain, and its absolute levels are not comparable to committed
numbers** — only its *paired* differences are. #E32 shows the failure mode
concretely: on its own block thompson reads 1475.13 against the committed
1463.19, a +12 offset that is the block and not a bug, while the paired
contrast reproduces the record (−23.62 vs +22.95). **Quote paired differences
from `GridSim`, never levels.**

**The ★ deliverable satisfies this too, now.** The two-constant rule was
originally produced *and scored* inside `mab_policy_probe.py --part csweep`, a
readback instrument needing a fitted-c artifact and the campaign's trained runs
on disk — so the leaderboard's top number was reproducible only from a machine
that already had those run directories. That contradicted the rule above, and
did so for the one number most likely to be quoted. It is now
`mab_benchmark_rule[_eval].py`: a policy with no artifact, no network and no
torch, scored by the same spec-§9 loop as every other benchmark and writing the
same TSV. Verified identical — **1486.1418 ± 6.4691, regret 44.6053** against
the committed 1486.14 / 6.47 / 44.61.

### Why two gates, and why neither is redundant

`GridSim` cannot be gated end-to-end, so it is gated on its deterministic
core, in two complementary places:

- **against the domain's own belief** — `mab_bayes.bayes_post_mean/sd`, which
  is what the gym's `bayes` observation is built from and what the IR names in
  `expr_builtins`. Exact (1e-12), but only at the base cell, since that is the
  only cell the domain implements.
- **by calibration** — off the base cell there is no reference at all, so µ
  drawn from the prior must fall within `psd` of `pm` at the right rate
  (standardized residual variance 1). Independent of *how* the posterior is
  computed, which is what makes it able to catch a σ mis-scaling.

Verified by fault injection (`mab_gate_mutation_check.py`, a negative
control): dropping σ from the precision is invisible to the first gate — it
vanishes at σ=1 — and caught by the second; an off-by-one in the pull count is
caught by both; dropping the prior precision is caught by the first only.
Every injected bug is caught by at least one gate, and neither gate alone
suffices.

**Adding a fourth simulator**: declare the equivalence it claims, gate that
claim in `mab_test.py` (not a manual `--part`), gate *components* against the
canonical functions rather than only aggregate outputs — aggregates absorb
compensating errors — and derive scenario constants from the scenario object
instead of hard-coding them, so an IR change breaks it loudly. An ungated
implementation may not produce a number that enters `ESCALATION.md`.

## Results — gauss_K10_T1000 (8192 shared eval seeds 0..8191)

Stream: the **per-arm payout** realization (F1), oracle 1530.7471. All PPO
rows are **stochastic** evals (`--stochastic`); rules and readbacks are
deterministic on the same CRN seeds. Escalation ids in the last column point
at the entry that produced the number.

| policy | reward_mean | ±SE¹ | regret | % oracle | note |
|---|---|---|---|---|---|
| oracle (clairvoyant best arm) | 1530.75 | — | 0.00 | 100% | — |
| **rule `argmax(pmᵢ + 2.5·(ttg/T)^0.15·psdᵢ)`** | **1486.14** | **6.47** | **44.61** | **97.1%** | **+22.95 vs thompson, z=36.8 — two constants, no torch (#E26); `mab_benchmark_rule_eval.py`** |
| free rule `argmax(pmᵢ + √(2 ln(ttg/nᵢ))·psdᵢ)` | 1464.87 | 6.52 | 65.88 | 95.7% | best **parameter-free** form; −21.27 vs tuned (#E27) |
| thompson (exact conjugate) | 1462.38 | 6.52 | 68.37 | 95.5% | reference |
| 12-knot readback of the A8-a net | 1459.56 | 6.33 | 71.19 | 95.3% | 98% action agreement with the net it fits (#E24) |
| PPO obs=bayes index @ tuned HP (**A8-a**, 3 seeds) | 1453.57 | 4.80 | 77.18 | 95.0% | best **trained** config, 99.4% of thompson; ent→0 (#E23) |
| PPO obs=bayes index + ttg-weighted entropy (**A7-t2**, 3 seeds) | 1445.32 | — | 85.43 | 94.4% | +75.40 vs its control, t(2df)=4.44 (#E21) |
| A6b crown + deployment temper β(t) | 1445.18 | 6.35 | 85.57 | 94.4% | post-hoc calibration, sets B_cal (#E17) |
| ucb1 (sigma-scaled) | 1440.79 | 6.52 | 89.95 | 94.1% | reference |
| **PPO obs=stats index @ tuned HP (A11-a, 3 seeds)** | **1408.55** | **5.00** | **122.20** | **92.0%** | the reopened encoding: −45.02 vs bayes at the same centre — 91% of the old gap was the centre (#E29); a **learned Thompson**: exploration lives in the sampling, not the index (#E30, `INTERPRET.md` Part II) |
| PPO obs=bayes index policy (**A6b**, deployed artifact) | 1387.57 | 6.30 | 143.18 | 90.6% | PASS: z=41.01 vs greedy, 192.77 vs random (#E14) |
| PPO obs=bayes L1 MLP + frame-avg 16 | 1353.35 | 6.51 | 177.40 | 88.4% | PASS (z=36.64 vs greedy) |
| PPO obs=bayes L1 MLP (raw single-pass) | 1314.46 | 6.27 | 216.29 | 85.9% | PASS (z=33.02 vs greedy) |
| PPO obs=bayes L1 MLP + belief shaping | 1293.03 | 6.36 | 237.72 | 84.5% | PASS vs bar, but −21.4 vs raw (z=−2.40) |
| PPO obs=stats MLP @ tuned HP (A11-b) | 1060.81 | 7.50 | 469.94 | 69.3% | PASS vs greedy (z=4.5) — HP alone rescues +133; the index arch adds +348 (#E29) |
| greedy (myopic Bayes) | 1015.69 | 6.52 | 515.05 | 66.4% | baseline |
| PPO obs=stats L1 (best: norm_obs=True) | 927.63 | 6.88 | 603.11 | 60.6% | FAIL (z=−9.29 vs greedy) — superseded by A11 (#E29): the verdict was about the centre (ent 0.01, raw MLP, untuned), not the encoding |
| PPO obs=stats L0 | 732.17 | 7.57 | 798.58 | 47.8% | FAIL (z=−28.38 vs greedy) |
| PPO obs=bayes L0 | −1.67 | 10.38 | 1532.42 | −0.1% | FAIL (z=0.08 vs random) |
| random | −2.51 | 3.50 | 1533.26 | −0.2% | baseline |

¹ `±SE` here is the **eval standard error** over the 8192-seed block (the
spread of the mean, one policy). It is not the training-seed sd — `INTERPRET.md`
quotes `±` as seed-sd across 3 training seeds, which is the larger number and
answers a different question (would another training run land here?).

**How the ★ rule's constants were chosen, since the headline is a fitted rule.**
The two constants were *not* fitted on the block they are reported on. The
quantile sweep ranks candidates on a separate 2048-seed block; only the top
three are then scored once on the 8192-seed record block, and 1486.14 is that
single scored number (`mab_policy_probe.py --part csweep`). That is the same
screen-and-confirm discipline the campaign asks of its trained policies — a
selection block disjoint from the reporting block — applied to a two-parameter
fit, which is where overfitting would otherwise be easiest and least visible.

**Comparing two trained rows needs the training-seed sd (16.70), not the eval
SE**: single-seed contrasts below ~50 points are not established (`#E21`).
Rule-vs-rule contrasts are deterministic on CRN and use the eval SE.

**The best object on this branch is a formula, not a network.** Reading the
trained crown back (spec §14) showed it *is* an index — `argmax(m + c·s)` with
c ≈ 1.5, a fixed ~93rd posterior percentile — so the open question became the
quantile *schedule*, answerable by sweeping c in behavior space with no
training. The level was the error, not the shape: c ≈ 2.5 (~99th percentile)
scores 1486.14 and is the first policy here to beat Thompson. No
parameter-free form comes within 21 points of it, and every one of them
*over*-explores: with a known prior and horizon the optimal identification
rate is ~0.93, not the ~0.95 the classical rules buy. See `INTERPRET.md` and
`ESCALATION.md` #E24–#E27.

The trained artifact behind that readback is an **equivariant index policy**
(`mab_equinet.IndexPolicy`): one scoring MLP φ(meanᵢ, sdᵢ, time_to_go) is
shared across arms and its per-arm outputs *are* the action logits, so
relabeling the arms relabels the policy exactly — the class UCB1 itself
belongs to. Measured equivariance error TV 2.3e-8 vs the plain MLP's 0.298.
It beat the raw MLP by +73.11 at a third of the parameters, and every
*enlargement* of that class then failed (mean pooling, max pooling, attention,
a learned temperature head): findability, not capacity.

Two things it retires. Frame averaging (π̄(s) = mean over 16 arm relabelings
σ of σ⁻¹(π(σ(s)))) was worth +38.9 on the *non*-equivariant MLP, but is an
exact no-op here — `mab_policy.py` keeps it for MLP artifacts and
auto-disables it for equivariant ones. Belief-potential shaping
(Φ = −c·Σᵢ sdᵢ, trains-only) *regressed* the MLP by 21.4 (z=−2.40); see
`ESCALATION.md` #E15.

**Open** — the ★-crown / packaging decision (which of the three candidates
above ships: the A6b artifact `mab_policy.py` currently wraps, the A8-a net,
or the two-constant rule) is with the operator, and the Bernoulli branch still
has no bar of its own (A4). Both are tracked in `ESCALATION.md` §Frontier.

The re-realization was a pure realization swap: `oracle_mean` is **identical**
(1530.7471 — the arm-mean latents live on the meta branch, which the arm key
does not touch) and every benchmark moved by well under one SE (thompson +0.74,
ucb1 +1.04, greedy −4.78, random +0.43). So the per-arm key changed *which*
draws happen, not the distribution they come from.

**The O1 fix is visible directly.** With selection now sampling the policy, the
two criteria agree — and, decisively, agree on the *winner*:

| | selection (256 CRN) | reporting (8192) | agree? |
|---|---|---|---|
| obs=bayes L1 | 1265.47 | 1314.46 | ✓ +3.7% |
| obs=stats L1 | 793.10 | 855.73 | ✓ +7.3% |
| *before the fix* | bayes 951.06 **<** stats 977.40 | bayes 1284.33 **>** stats 899.59 | ✗ inverted |

**The O2 per-mode split was refuted by isolation (`ESCALATION.md` #E10).**
On the same stream with only the flag changed, `stats` L1 scores 927.63 with
`norm_obs=True` vs 855.73 with it off (+71.9, z≈7.2) — normalization helps
this mode despite its accumulator drift, so `norm_obs` defaults to True for
both obs modes again. The table row above is the norm_obs=True run.

Bernoulli branch (`bern_K10_T1000`): still never trained, and its benchmarks
have not been run on the current stream either.

### Standing design rationale (not stream-dependent)

- **Evaluate stochastically.** The PPO policy's entropy *is* its exploration
  mechanism, so argmax evaluation turns it into an under-explorer. This is a
  property of the policy class in a bandit, not a measurement, so it survives
  the re-realization: all gated numbers use per-step sampling (`--stochastic`)
  and `mab_policy.py` defaults to `deterministic=False` for the same reason.
  Note this deviates from the spec §9 eval template, which uses
  `deterministic=True`.

The campaign's measured findings, their addresses, and the open items now live
in `ESCALATION.md`.

## Usage (run from `cases/mab/` unless noted; gates run from the repo root)

```bash
# validate IR + gates
# all five are what `case-gates` CI re-runs on a PR touching this folder
python -m mdp_ir cases/mab/mab_schema.json
python -m mdp_conformance cases/mab
python -m mdp_ir.laws cases/mab [--instance gaussian]   # engine laws + invariants
python -m mdp_ir.differential cases/mab/mab_schema.json --episodes 40 --all-instances
pytest cases/mab/mab_test.py                            # needs the [dev] extra

# baselines (writes results/{scenario}/benchmark/benchmark_{method}_eval_{scenario}.tsv)
python mab_benchmark_random_eval.py   -s gauss_K10_T1000 --n-seeds 8192
python mab_benchmark_greedy_eval.py   -s gauss_K10_T1000 --n-seeds 8192
python mab_benchmark_ucb1_eval.py     -s gauss_K10_T1000 --n-seeds 8192
python mab_benchmark_thompson_eval.py -s gauss_K10_T1000 --n-seeds 8192
# the ★ row — the distilled rule, no artifact and no torch needed
python mab_benchmark_rule_eval.py     -s gauss_K10_T1000 --n-seeds 8192

# train + eval (OMP pinning matters on shared boxes)
# defaults — NOT the config behind the leaderboard's trained rows
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python mab_ppo_train.py -s gauss_K10_T1000 -o bayes

# A8-a, the config of record for every "PPO obs=bayes index @ tuned HP" row
# (A3's harvested trial 6, re-ranked at protocol — see #E28). Three seeds.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python mab_ppo_train.py \
  -s gauss_K10_T1000 -o bayes --policy index \
  --learning_rate 1.58e-05 --lr-final 1.58e-06 \
  --n-steps 2048 --batch-size 32 --n-epochs 20 \
  --gae-lambda 0.98784 --ent-coef 3.44e-05 --vf-coef 0.381 \
  --no-norm-obs --seed 1
python mab_ppo_eval.py -s gauss_K10_T1000 --stochastic \
  --model-path results/gauss_K10_T1000/<run>/gauss_K10_T1000_ppo.zip

# gate
python -m mdp_gates \
  --candidate results/gauss_K10_T1000/<run>/ppo_eval_stoch_gauss_K10_T1000.tsv \
  --baseline  results/gauss_K10_T1000/benchmark/benchmark_random_eval_gauss_K10_T1000.tsv \
  --baseline  results/gauss_K10_T1000/benchmark/benchmark_greedy_eval_gauss_K10_T1000.tsv \
  --reference results/gauss_K10_T1000/benchmark/benchmark_thompson_eval_gauss_K10_T1000.tsv \
  --metric reward_mean --n-seeds 8192

# deploy
python mab_policy.py --model-path results/gauss_K10_T1000/<run>/gauss_K10_T1000_ppo.zip -o bayes
```

Trained artifacts under `results/` are gitignored; the commands above
reproduce them. Currently wrapped by `mab_policy.py`:
`results/gauss_K10_T1000/PPO_obsbayes_L1_policyindex_normobsFalse_seed1_20260729_185003/`
(train with `--policy index`). Two better objects exist and neither is
packaged yet, pending the ★-crown decision: the **A8-a** seeds
(`…_policyindex_lr1.58e-05_…_seed{1,2,3}_20260801_230542/`, the same class at
the A3-tuned HP) and the **two-constant rule** of #E26, which needs no
artifact at all — it is `argmax(pm + 2.5·(ttg/T)^0.15·psd)` over
`mab_bayes.py`'s posteriors, and is scored today inside
`mab_policy_probe.py --part csweep`. The superseded
MLP crown and its frame-averaged gate record
(`ppo_eval_stoch_frameavg16_gauss_K10_T1000.tsv`) sit under
`PPO_obsbayes_L1_seed1_20260729_115417/`.
The `VOID_PPO_*_20260728_173709/` dirs beside it were trained against the
pre-arm-key payout stream and are kept only as provenance.

Run names encode the *config*, not the experiment that motivated them, so to
go from an escalation action (`A6b`) or a ledger entry (`#E14`) to its
artifacts, read the generated table in `ESCALATION.md` §RUNS — or, for
TensorBoard, point `--logdir` at the label farm:

```bash
python mab_runs_index.py                                 # the action ↔ dir table
tensorboard --logdir results/by_action    # every round: A6b, e35-T20000-scl, e36-generalist, …
```

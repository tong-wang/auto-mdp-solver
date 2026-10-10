# owmr — one-warehouse, N-retailer periodic-review distribution

Central control of one warehouse supplying five retailers under periodic
review — the model of Doğru, de Kok & van Houtum (2009), "A numerical study on
the effect of the balance assumption in one-warehouse multi-retailer inventory
systems", *Flexible Services and Manufacturing Journal* 21(3–4):114–147 —
generated from `owmr_schema.json` by auto-mdp-solver.

**TL;DR**

- **Where the heuristic has room, RL beats it; where it has none, RL gets close.** On `hicv` (demand cv = 2; the lower bound sits 10% below the existing heuristic) PPO with a categorical retention head and the coordinate vine beats the heuristic by 3.3% and passes the gate. On `base` (cv = 0.5; the bound is 0.1% below the heuristic, so the heuristic is essentially optimal) the best RL arm stops 0.6% above it, and nothing ships there.
- **The discovery on `hicv` is retention at the warehouse, done by under-allocating.** The learned policy ships each retailer only up to about three quarters of its newsvendor target and keeps the rest as a buffer the next period's allocation can redirect, paying for it with a slightly higher echelon order-up-to level. Written as that two-scalar change to the heuristic, the rule scores as well as the network.

*Status (2026-10-10): `base` closed, `hicv` complete and packaged (`owmr_policy.py`: the vine network and its two-scalar readback), `hipen` and `hipen_hicv` not opened; the playbook is written; contributed upstream as auto-mdp-solver PR #95 (merge pending).*

## The problem

One warehouse supplies N identical retailers under periodic review. Each
period the warehouse decides how much to **order** from an ample outside
supplier (delivered after a lead time) and how much to **ship** to each
retailer from what it has on hand (delivered one period later); demand then
hits every retailer, i.i.d. gamma, and unmet demand is backordered. Holding
is charged at the warehouse and at the retailers, backorders are penalised,
and the objective is the undiscounted total cost over a finite horizon. The
model is Doğru, de Kok & van Houtum (2009, *Flexible Services and Manufacturing Journal* 21(3–4):114–147, doi:10.1007/s10696-010-9064-1); the parameters of each scenario
are in the results table and, authoritatively, in `owmr_schema.json`.

What makes it hard is the **allocation**: stock held at the warehouse can
still be sent wherever demand turns out high, but only one lead time later;
stock at a retailer serves demand now but cannot be moved. The literature
sidesteps that coupling with the *balance assumption* — allocations may be
negative, so only system stock matters and the problem decomposes into
newsvendors. Under it the optimal policy is an echelon order-up-to level plus
newsvendor targets, in closed form; its cost is a **lower bound** on the true
optimum, and the same policy with non-negativity restored (the *LB heuristic*)
is a feasible **upper bound**. The true optimum is uncharacterised and the
dynamic program is intractable beyond two retailers, so every leaderboard here
is a bracket. The bracket is narrow when demand is steady and wide when it is
volatile — which is exactly where RL has room to find structure the heuristic
lacks.

## Layout

Documents — where to read about the case:

| doc | what it holds |
|---|---|
| `owmr.restatement.md` | the Phase-A restatement: the problem as the IR states it, with the declared sample trajectory |
| `owmr.scenarios.md` | the scenario set and why each fixed value was chosen |
| `ESCALATION.md` | the campaign — design tree, changelogs, numbered findings (#E…) with verdicts; seeded at Stage 2 with the gate battery (#E1) |
| `INTERPRET.md` | the policy readback (spec §14): the `hicv` winner read as a two-scalar rule on the LB heuristic, the swap decomposition, the figure, and the instrument's negative control on `base` |
| `PLAYBOOK.md` | the digest (guide §10): the two transferable entries — the coordinate vine (LV1) and the split-and-keep action interface (LV2) |
| `CLAUDE.md` | operating brief for an agent changing this folder |

Code — where things are implemented:

| module | role |
|---|---|
| `owmr_schema.json` | **the IR — authoritative** for the problem definition |
| `owmr.signoff.json` / `owmr.runplan.json` | the Phase-A freeze token, and the Stage-0 run plan (target `base`, specialist, protocol sizes, budgets, escalation budget) |
| `owmr_uncertainty.py` | the stochastic primitives: the vector-valued `GammaDemand` (one draw per retailer per period) and the `FamilyDemand` bridge |
| `owmr_scenarios.py` | the world layer: `OwmrScenario`, the four cells in `SCENARIOS` |
| `owmr_mdp.py` | `OwmrState` + the transition functions (`advance1` = the two receipts, `advance2` = order, allocation with proportional scale-down, demand) and the paper's derived coordinates |
| `owmr_gym.py` | the Gymnasium wrapper: observation modes `raw` / `raw_d` / `position` / `raw_seq`; action modes from `order_frac` (L1) through `order_softmax`, `order_ship`, `order_relu`, `order_nested` to `order_catkeep` / `order_catkeep_fine` and `order_dirichlet` (which modes are structure-free is in the class docstring) |
| `owmr_ir_adapter.py` | the differential adapter (interpreter vs `owmr_mdp`, bit-exact) |
| `owmr_test.py` | the domain's own tests: laws, differential over the covering set, negative controls, the demand second-implementation gate, the rendered-vector and allocation claims |
| `owmr_ppo_train.py` / `_ppo_eval.py` / `owmr_select.py` | PPO training (L0/L1 per spec §8.6, `_L1_DERIVED` checked at launch), protocol-block evaluation with harvest provenance, and the post-hoc checkpoint screen on the selection block |
| `owmr_eval_common.py` | the shared CRN rollout, the §9.3 record and the `.seeds.tsv` sidecar every arm is scored through |
| `owmr_benchmark_lb.py` / `_eval.py` | the balance-assumption lower bound (Doğru et al. §3.3), **relaxed**: y0* and the retailer targets solved analytically (gamma convolutions, SciPy incomplete-gamma), the relaxed optimum on the finite-horizon criterion (100 periods from the mean-cover start) recorded as the reference arm (#E26), the paper's long-run average solved alongside for the paper check only; also the myopic allocation the heuristic uses |
| `owmr_benchmark_lb_heuristic.py` / `_eval.py` | the paper's LB heuristic (§3.4), **feasible**: order up to y0*, myopic allocation with non-negativity; simulated on the protocol block |
| `owmr_benchmark_random.py` / `_eval.py` | the random floor, **feasible**: uniform in the default action box |
| `owmr_policy.py` | the deployable wrapper (spec §12): `OwmrPolicy` = the shipped network (model + observation statistics + the gym's decode) and `FittedRulePolicy` = its two-scalar readback, both behind `decide(state)` / `act(obs)`; `python owmr_policy.py` replays both on the raw MDP loop against the eval records |
| `owmr_policy_probe.py` | the §14 readback probe: action surfaces, order-up-to / target flatness, retention curve, time sensitivity; validated on a planted rule in `owmr_test.py`; run on the `hicv` winner and its control (#E33 addendum 1) |
| `owmr_catkeep_policy.py` / `owmr_dirichlet_policy.py` | the two allocation heads: a Gaussian × categorical policy over the retention menu (`CatKeepPolicy`), and a Gaussian × Dirichlet policy over the shares (`DirichletPolicy`) |
| `owmr_vine_ppo.py` / `owmr_vine_gate.py` | the coordinate vine (`VinePPO`: paired common-world siblings per action coordinate, a side PPO pass on its own optimizer, the menu-grid variant) and its executable gate G1–G10 |
| `owmr_split_policy.py` / `owmr_seq_policy.py` / `owmr_ask_features.py` | the architecture arms that lost: split order / allocation trunks, the two-phase phase-masked policy, the ask-feature extractor (#E7–#E10) |
| `owmr_swap_probe.py` / `owmr_branch_probe.py` / `owmr_snr_probe.py` / `owmr_fit_probe.py` / `owmr_lb_transient_probe.py` | diagnostics, never arms: the order / allocation swap decomposition, the same-world branch probe that sized the vine, the gradient-SNR probe, the representation-fit probe, the heuristic's transient profile |
| `owmr_keep_rule_probe.py` | the fitted rule of the discover stance: the LB heuristic with scaled targets and a moved order-up-to level, swept and scored paired (#E33 addenda 2–4, #E36) |
| `configs/` | the recipes behind the rows as argument files, one per CONFIG-REGISTRY id (`h5`, `h5_terminal`, `h8`, `vine`, `vine_shared`, `L0`), read by the train script as `@configs/<id>.args` |
| `owmr_plot_policy.py` | the figure contract (spec §14.3): one spec, two renders — the committed SVG under `figures/` that `INTERPRET.md` inlines and an interactive HTML under the cell's `results/…/figures/` (gitignored), from the two probe JSONs |

## Results

### Protocol

Every number below is quoted at: **8192 seeds from 0**, common random numbers
across arms (paired per-seed deltas), reference bar = the **LB heuristic**
(feasible), the **lower bound** reported beside it but never gated (it is not
attainable, spec §9.9), record eval = **deterministic argmax** (a continuous
allocation policy has no exploration role at eval time; the stochastic eval
is a labelled figure, never a record). Checkpoints are screened on a disjoint
selection block (2048 seeds from 1,000,000), top-3 confirmed on the protocol
block; screen numbers are never quoted.

Symbols: **cost** = total holding + backorder cost over the 100-period episode
(the objective; per-period = cost / 100); **LB** = the balance-assumption
bound on this problem's criterion — the relaxed system's optimum over the 100
periods from the mean-cover start, in this domain's accounting (#E26; the
paper's long-run average per period, plus the constant h0 Σ l_i μ_i = 2.5 for
transit, is a different criterion and is not the record); **L0** =
SB3 defaults with γ = 1, no normalization, 1 env, 2M steps; **L1** = the §8.6
derivation (see `owmr_ppo_train.py` `_L1_DERIVED`), 5M steps; arm names are
run-dir names, checkpoint step in the row.

### `1-comparative` — how does RL compare with the existing solutions?

The scenarios, and the role each plays in the study:

| scenario | config | bound (`lb`, finite-horizon criterion, #E26) | bar (`lb_heuristic`, 8192 seeds) | bracket (bar − bound, as % of the bound) | what it is for |
|---|---|---|---|---|---|
| `base` | p = 4, cv = 0.5 | 603.18 | 603.81 ± 0.44 | 0.63 (0.10%) | the bound is known tight (paper gap ≈ 0.05%): validates the readback instrument and shows RL reaching a near-optimal bar |
| `hicv` | p = 4, cv = 2 | 2,499.99 | 2,769.76 ± 4.01 | 269.76 (10.8%) | the bound is known loose (paper average gap ≈ 5%, up to 17%): the discover cell |
| `hipen` | p = 19, cv = 0.5 | 937.36 | 938.44 ± 0.87 | 1.08 (0.11%) | high service target (95%), bound tight |
| `hipen_hicv` | p = 19, cv = 2 | 4,849.32 | 5,415.30 ± 11.45 | 565.98 (11.7%) | high service target, bound loose: the second discover cell |

All four share N = 5, lead times (3, 1), h0 = h_rt = 0.5, μ = 1, T = 100.
**They are separate leaderboards**: cost scales differ across cells, so each
is scored against its own bracket and never against another cell.

#### `base` — p = 4, cv = 0.5, the tight-bound cell

Sorted by cost (minimize). Δ = paired per-seed difference against the bar (`lb_heuristic`), with Δ as a percentage of the bar in parentheses; z = Δ / paired SE. The rows echo the two playbook entries — the action head (categorical vs Dirichlet, `PLAYBOOK.md` LV2) and the coordinate vine on each (LV1) — beside the references and the L0 / L1 backbone; every other arm (each lever, dose, seed and study) is in `ESCALATION.md`'s MAP and ledger.

| arm | role | cost / episode | Δ vs bar | z | wh share | stockout retailer-periods |
|---|---|---|---|---|---|---|
| `lb` — balance-assumption bound, finite-horizon criterion (#E26) | relaxed | 603.18 | −0.63 (−0.10%) | — | — | — |
| `lb_heuristic` — the paper's LB heuristic | feasible, **bar** | 603.81 ± 0.44 | 0 (0%) | — | 0.056 | 97.4 |
| PPO, **categorical head + the coordinate vine** — `order_catkeep` + `CatKeepPolicy` at h5, `VinePPO` (h6), seed 2, extension checkpoint +4.75M of 5M + 5M (#E31 addendum 2; 607.90 at 5M, #E30) | feasible | 607.63 ± 0.45 | +3.82 ± 0.05 (+0.63%) | +76 | 0.033 | 96.3 |
| PPO, **categorical head, no vine** — `order_catkeep` + `CatKeepPolicy` at h5 (= #E24 study trial 34, training seed 31), checkpoint 4.75M (#E24) | feasible | 613.90 ± 0.47 | +10.09 ± 0.10 (+1.67%) | +100 | 0.046 | 98.2 |
| PPO, **Dirichlet head + the coordinate vine** (per-share Gamma form) — `order_dirichlet` + `DirichletPolicy` at h8, seed 1 checkpoint 10M (#E32) | feasible | 615.45 ± 0.48 | +11.64 ± 0.09 (+1.93%) | +129 | 0.000 | 101.0 |
| PPO, **Dirichlet head, no vine** — `order_dirichlet` + `DirichletPolicy` at h8, seed 1, checkpoint 10M (#E32, the vine row's seed-matched control) | feasible | 619.63 ± 0.48 | +15.82 ± 0.12 (+2.62%) | +132 | 0.044 | 100.5 |
| PPO L1, checkpoint 3.25M of `PPO_20260922_122559_L1_obsraw_actorder_frac_E2L1` | feasible | 659.78 ± 0.50 | +55.97 ± 0.19 (+9.27%) | +302 | 0.000 | 96.0 |
| PPO L0, terminal of `PPO_20260922_122558_L0_…_E2L0` | feasible | 720.59 ± 0.63 | +116.78 ± 0.41 (+19.34%) | +283 | 0.000 | 74.7 |
| `random` | feasible, floor | 443,090 ± 348 | — | — | 0.000 | 1.7 |

**Closed 2026-10-09 — nothing ships on `base`; the heuristic is the deliverable there.** It sits within 0.63 (0.10%) of the bound (#E26), and the best RL arm stops +0.63% above it after every lever was tried: interface, head, budget, seven tuning studies (#E3, #E5, #E16, #E24, #E25, #E27, #E28), the vine and its variants (#E2–#E36). The vine is worth 1 point of the gap on this cell and the categorical head 7.5–7.8 (1.2%) over the Dirichlet in the seed-matched #E32 pairs; the residual is retention, and the fitted-rule instrument finds no rule in the heuristic's class that beats it here (#E36). The full ladder is `ESCALATION.md` §MAP; the mechanisms are `PLAYBOOK.md` LV1–LV2.

#### `hicv` — p = 4, cv = 2, the loose-bound (discover) cell

Same protocol and columns (Δ with its percentage of the bar); on this cell the checkpoint screen runs at **8192** seeds on the selection block (the heuristic's SE here is ten times `base`'s). Trained on an HPC cluster (#E33); gates run locally under the pinned harness.

| arm | role | cost / episode | Δ vs bar | z | wh share | stockout retailer-periods |
|---|---|---|---|---|---|---|
| `lb` — balance-assumption bound, finite-horizon criterion (#E26) | relaxed | 2,499.99 | −269.76 (−9.74%) | — | — | — |
| **Fitted rule (#E33 addendum 4): the LB heuristic with the order-up-to level raised to 37 (y0* + 2.4) and each retailer shipped up to 0.75 · z*, the rest kept** (`owmr_keep_rule_probe.py --alpha 0.75 --y0 37`; −1.4 ± 0.6 paired vs the RL winner) | feasible, fitted rule — not an RL arm | 2,676.40 ± 3.45 | −93.35 ± 0.90 (−3.37%) | −104 | 0.198 | 100.0 |
| **PPO, categorical head + the coordinate vine** — `order_catkeep` + `CatKeepPolicy` at h5, `VinePPO` (h7, s = 1), seed 2, checkpoint 10M (#E33) | feasible, **GATE PASS** | **2,677.78 ± 3.76** | **−91.98 ± 0.81** **(−3.32%)** | −114 | 0.134 | 103.4 |
| PPO, **Dirichlet head + the coordinate vine** (per-share Gamma form) — `order_dirichlet` + `DirichletPolicy` at h8, seed 2, checkpoint 10M (#E33) | feasible, gate pass | 2,711.89 ± 4.15 | −57.87 ± 1.01 (−2.09%) | −58 | 0.139 | 115.2 |
| PPO, **categorical head, no vine** — `order_catkeep` + `CatKeepPolicy` at h5, seed 2, checkpoint 10M (#E33, the vine's matched control) | feasible, gate pass | 2,744.42 ± 3.95 | −25.34 ± 1.22 (−0.91%) | −21 | 0.133 | 101.8 |
| `lb_heuristic` — the paper's LB heuristic | feasible, **bar** | 2,769.76 ± 4.01 | 0 (0%) | — | 0.017 | 103.9 |
| PPO L1 (`order_frac`, the derivation), seed 2, checkpoint 4.75M (#E33) | feasible | 2,874.79 ± 4.20 | +105.04 ± 1.44 (+3.79%) | +73 | 0.003 | 92.6 |
| PPO, **Dirichlet head, no vine** — `order_dirichlet` + `DirichletPolicy` at h8, seed 2, checkpoint 5M (#E33, the vine's matched control; seed 1 collapsed to 3,765.53) | feasible | 3,082.96 ± 5.31 | +313.20 ± 2.62 (+11.31%) | +120 | 0.093 | 99.0 |
| PPO L0, terminal (#E33) | feasible | 5,947.87 ± 14.54 | +3,178.12 ± 12.44 (+114.74%) | +256 | 0.397 | 206.1 |
| `random` | feasible, floor | 443,165 ± 350 | — | — | 0.000 | 4.2 |

**Result: a two-scalar rule.** PPO with the categorical head and the vine beats the heuristic by 3.3% and passes the gate (#E33); read back, the policy is the LB heuristic with its order-up-to level raised by 2.4 and each retailer shipped only up to 0.75 of its newsvendor target, the rest kept at the warehouse — and that rule alone scores 2,676.4, at or below the network (#E33 addenda 3–4). The vine carries the result (67 of the 92 against its matched control on this seed; 53 on the other), the head 34 against the Dirichlet vine, the recipe nothing (#E35) and the seed up to 25. **Ships:** `owmr_policy.py` wraps the seed-2 vine checkpoint at 10M as `OwmrPolicy` — chosen over the trial-22 and trial-0 networks (within 2 of it, one seed each) because it is the gate-passing arm the readback was done on and the best single checkpoint on the protocol block — and the two-scalar rule as `FittedRulePolicy` beside it: the rule scores −1.4 ± 0.6 paired against the network, needs no model file and no seed, and is the deliverable a planner would adopt; the network is its witness. `INTERPRET.md` holds the readback, `ESCALATION.md` #E33 and #E35 the campaign.

### `2-structural` — the declared stance

**Discover** (primary): in the loose-bound cells, RL finds an ordering /
allocation structure the LB heuristic lacks — retaining stock at the
warehouse, allocating non-myopically — and a rule fitted to the trained
policy scores between the heuristic and the bound. The tight cells validate
the readback instrument first.

**Verdict on `hicv`: confirmed.** The structure, in one sentence: **the
learned policy under-allocates on purpose — it ships each retailer only up to
about three quarters of its newsvendor target, keeps the remainder at the
warehouse as a buffer that the next period's allocation can direct to whoever
turns out to need it, and raises the echelon order-up-to level by about 2.4
units to pay for the stock it now holds back** — the LB heuristic's own form
with two scalars moved (targets 0.75 · z*, level y0* + 2.4), scoring 2,676.4
against the heuristic's 2,769.8 and the network's 2,677.8 (#E33 addenda 3–4).
What makes it non-myopic is the under-allocation, not a held-back fraction: a
rule that keeps a fixed share of on-hand loses at every share and threshold,
because the value of the buffer comes from matching it to the retailers'
realised needs one period later, which the heuristic's "ship everything to
the myopic targets" cannot do when demand variability is high (cv = 2). The
heuristic's own level is then too low once stock is retained, which is the
"order part" of the gain. On `base` (cv = 0.5) the heuristic is within 0.1%
of the bound and the same instrument recovers it as the family's optimum —
no such structure exists there, and the RL residual is not a missed rule
(#E36). Details, curves and the swap decomposition: `INTERPRET.md`.

## Technical appendix — a first run

**What you need.** Python 3.12 with the `auto-mdp-solver` harness and its
domain extra — `pip install "auto-mdp-solver[domain]"`, or
`pip install -e "./harness[domain]"` from the solver repo's root — which brings
Stable-Baselines3; plus `pip install scipy plotly`: SciPy solves the
heuristic's newsvendor targets, plotly draws the interactive figure (the
committed SVG does not need it). Run everything from `owmr/`, with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` set for training.

**Where things land.** Everything a command writes goes under `results/`
(gitignored): each cell has a `benchmark/` folder for the references and one
directory per training run, named by the run's tag, holding its checkpoints
and, after the funnel, its confirm records. Training recipes are the files
under `configs/`, one per registry id, passed to the train script as
`@configs/<id>.args`.

**One cell, start to finish.** The `hicv` cell, which produced the result.
`<run>` is the directory step 3 prints as `outdir=…`; `<ckpt>` is a
checkpoint name step 4 prints.

```bash
# 1. the layers run (smoke tests)
python owmr_scenarios.py
python owmr_mdp.py
python owmr_gym.py

# 2. the references on the protocol block: the bound, the heuristic (the bar), the floor
python owmr_benchmark_lb_eval.py -s hicv --n-seeds 8192
python owmr_benchmark_lb_heuristic_eval.py -s hicv --n-seeds 8192
python owmr_benchmark_random_eval.py -s hicv --n-seeds 8192

# 3. train the shipped arm (the categorical head's recipe plus the vine, 10M steps)
#    and its matched control (the same recipe without the vine), which step 9 needs
python owmr_ppo_train.py -s hicv @configs/h5.args @configs/vine.args \
    --total-timesteps 10000000 --tag E33catkeepvine --seed 2
python owmr_ppo_train.py -s hicv @configs/h5.args \
    --total-timesteps 10000000 --tag E33catkeepctl --seed 2

# 4. screen a run's checkpoints on the selection block; prints the confirm commands
#    (steps 4–6 once per run: the vine arm and the control)
python owmr_select.py results/hicv/<run> -a order_catkeep --n-seeds 8192 --top-k 3

# 5. confirm a top-3 checkpoint on the protocol block, paired with the bar
python owmr_ppo_eval.py -s hicv -a order_catkeep --first-seed 0 --n-seeds 8192 \
    --model-path results/hicv/<run>/checkpoints/<ckpt>.zip \
    --vecnorm-path results/hicv/<run>/checkpoints/<ckpt vecnormalize>.pkl \
    --outfile results/hicv/<run>/confirm_<ckpt>.tsv \
    --paired-with results/hicv/benchmark/benchmark_lb_heuristic_eval_hicv.seeds.tsv

# 6. the gate
python -m mdp_gates --n-seeds 8192 --sense minimize --metric cost_total_mean \
    --candidate results/hicv/<run>/confirm_<ckpt>.tsv \
    --baseline results/hicv/benchmark/benchmark_random_eval_hicv.tsv \
    --baseline results/hicv/benchmark/benchmark_lb_heuristic_eval_hicv.tsv \
    --reference results/hicv/benchmark/benchmark_lb_eval_hicv.tsv \
    --ir owmr_schema.json

# 7. read the policy back: once on the vine arm's winning checkpoint, once on the
#    control's (each writes a probe JSON beside its checkpoints)
python owmr_policy_probe.py -s hicv -a order_catkeep \
    --model-path results/hicv/<vine run>/checkpoints/<ckpt>.zip \
    --vecnorm-path results/hicv/<vine run>/checkpoints/<ckpt vecnormalize>.pkl
python owmr_policy_probe.py -s hicv -a order_catkeep \
    --model-path results/hicv/<control run>/checkpoints/<ckpt>.zip \
    --vecnorm-path results/hicv/<control run>/checkpoints/<ckpt vecnormalize>.pkl

# 8. the fitted rule, scored on the protocol block
python owmr_keep_rule_probe.py -s hicv --alpha 0.75 --y0 37 --n-seeds 8192

# 9. the figure, from step 7's two JSONs (it finds them by the runs' tags)
python owmr_plot_policy.py

# 10. the deployable, replayed against the records
python owmr_policy.py --n-episodes 8
```

Every other row is produced the same way: a cell, a recipe file and a tag
for the train line, then steps 4–6 with that cell's benchmark files. The
train lines behind every row follow, collapsed; the gate commands that check
the folder itself are in `CLAUDE.md`.

<details>
<summary>The train line behind every row (then steps 4–6; base screens at 2048)</summary>

```bash
# base — the references
python owmr_benchmark_lb_eval.py -s base --n-seeds 8192
python owmr_benchmark_lb_heuristic_eval.py -s base --n-seeds 8192
python owmr_benchmark_random_eval.py -s base --n-seeds 8192

# base — L0 and L1
python owmr_ppo_train.py -s base @configs/L0.args --tag L0
python owmr_ppo_train.py -s base --tag L1

# base — categorical head, no vine (the #E24 study's trial 34: h5 at seed 31)
python owmr_ppo_train.py -s base @configs/h5.args --tag E24t34 --seed 31

# base — categorical head + the vine on the main optimizer (h6), 5M,
# then its +5M extension at the terminal learning rate: the record arm (607.63)
python owmr_ppo_train.py -s base @configs/h5.args @configs/vine_shared.args \
    --tag E30vine --seed 2
python owmr_ppo_train.py -s base @configs/h5_terminal.args @configs/vine_shared.args \
    --init-from results/base/<the E30vine run> --total-timesteps 5000000 \
    --tag E31xvine --seed 2

# base — Dirichlet head without and with the per-share vine (h8), 10M
python owmr_ppo_train.py -s base @configs/h8.args \
    --total-timesteps 10000000 --tag E32dirichletctl --seed 1
python owmr_ppo_train.py -s base @configs/h8.args @configs/vine.args \
    --total-timesteps 10000000 --tag E32dirichletvine --seed 1

# base — the tuning study behind the h5 recipe (two workers, sampler seeds 31 and 32;
# it reproduces trial 34 only if the sampler replays)
python -m mdp_tuning owmr -s base --knobs all --beta 1.0 --episode-len 100 \
    --min-rollout-episodes 10 --total-timesteps 5000000 --eval-seeds 512 \
    --eval-arg first_seed=3000000 --minimize --metric cost_total_mean --n-trials 40 \
    --study-name owmr_base_ppo_catkeep_fixnr --fix norm_reward --seed 31 \
    --train-arg action_mode=order_catkeep --train-arg policy=catkeep --train-arg tag=E24

# base — the readback instrument's negative control (#E36): the sweep, then its table
for y in 26.36 27.36 28.363030605487545 29.36 30.36 31.36; do
    for a in 0.7 0.8 0.9 0.95 1.0; do
        python owmr_keep_rule_probe.py -s base --alpha $a --y0 $y --n-seeds 8192
    done
done
python owmr_keep_rule_probe.py -s base --summary

# hicv — the backbone and the carried bundle (#E33), seed 2 shown; seed 1 likewise
python owmr_ppo_train.py -s hicv @configs/L0.args --tag E33l0 --seed 1
python owmr_ppo_train.py -s hicv --tag E33l1 --seed 2
python owmr_ppo_train.py -s hicv @configs/h5.args \
    --total-timesteps 10000000 --tag E33catkeepctl --seed 2
python owmr_ppo_train.py -s hicv @configs/h5.args @configs/vine.args \
    --total-timesteps 10000000 --tag E33catkeepvine --seed 2
python owmr_ppo_train.py -s hicv @configs/h8.args \
    --total-timesteps 10000000 --tag E33dirichletctl --seed 2
python owmr_ppo_train.py -s hicv @configs/h8.args @configs/vine.args \
    --total-timesteps 10000000 --tag E33dirichletvine --seed 2

# hicv — the control's +10M extension at the terminal learning rate (#E33 addendum 5)
python owmr_ppo_train.py -s hicv @configs/h5_terminal.args \
    --init-from results/hicv/<the E33catkeepctl run> --total-timesteps 10000000 \
    --tag E33xctl --seed 2

# hicv — the swap decomposition on the winner (#E33 addendum 1)
for arm in rl_rl h_h h_order h_alloc h_split; do
    python owmr_swap_probe.py -s hicv --rl-action-mode order_catkeep \
        --arm $arm --n-seeds 8192 \
        --model-path results/hicv/<winner>/checkpoints/ppo_owmr_10000000_steps.zip \
        --vecnorm-path \
            results/hicv/<winner>/checkpoints/ppo_owmr_vecnormalize_10000000_steps.pkl \
        --outfile results/hicv/<winner>/swap/swap_$arm.tsv
done

# hicv — the fitted rule's sweeps (#E33 addenda 2–4):
# the step form, then the scaled targets
for t in 0 5 10 15 20; do
    for k in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5; do
        python owmr_keep_rule_probe.py -s hicv --k $k --theta $t --n-seeds 8192
    done
done
for y in 30 32 35 36 37 38 39 40 43 46; do
    for a in 0.7 0.75 0.8 0.85; do
        python owmr_keep_rule_probe.py -s hicv --alpha $a --y0 $y --n-seeds 8192
    done
done
python owmr_keep_rule_probe.py -s hicv --summary
```

</details>

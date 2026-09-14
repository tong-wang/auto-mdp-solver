# adi_flex — Inventory with Advance Demand Information and Flexible Delivery

An RL study of Wang & Toktay (2008), *Inventory Management with Advance Demand
Information and Flexible Delivery*, **Management Science 54(4), 716–732**.
Generated from the IR `adi_flex_schema.json` by **auto-mdp-solver**.

**TL;DR**

- **RL is competitive with the paper's analytic policies, and the split tracks
  how much room the reference leaves.** It lands 0.95% above the exact optimum
  (`homog_L0_T2`), 0.36% behind the paper's protection-level heuristic where that
  heuristic is only 1.5% above the unattainable lower bound (`het_exp4`), and
  2.7% ahead of it where the heuristic sits 4.3% above the bound (`het3_exp2`).
- **The paper's order policy is recovered, down to its trigger.** The learned
  reorder point equals the analytic one at every level of outstanding advance
  demand; the order-up-to level is flat in it and about 5 units lower.
- **The paper's sufficient statistic is not needed as an input.** A policy
  trained on the raw state ties one handed the modified inventory position on
  every board, so the network rebuilds the transform the paper had to prove.
- **The allocation is a constant protection level, and nothing beyond one is
  learned.** The protection-level shape is built into the shipped policy's
  action, so it holds by construction; the one rule beyond a constant that is
  worth having, zero protection in the last period, is hand-written and beats
  every learned policy's protection.

## 1. The problem

Customers order **in advance**, with due dates up to `T_dl` periods out, and
stock may be shipped **early**. Each period:

1. place a replenishment order — fixed cost `K`, supply lead time `L` —
   **before** the period's demand is observed;
2. receive the pipeline and observe the new demand, split by due date;
3. serve the forced fills (overdue backlog, then due-now — nothing arriving
   later may cross them);
4. **allocate**: decide how much of each not-yet-due class to pre-fill early
   from what is left.

Backlog costs `p` per unit per period, stock `h`; minimise expected total cost
over a finite horizon. The two decisions sit at **two different information
sets** — the order before the draw, the allocation after — which is the
structural feature the domain exists to study.

The paper's protection level `σ` parameterises its §4.2 *heuristics* for the
allocation. It is a policy family, not the decision itself.

Two quantities from the paper recur throughout this file:

- **`u`, the modified inventory position** — on-hand plus the supply pipeline
  minus everything already promised: `u = inv + Σpipe − Σadv`. Wang & Toktay's
  central structural result is that the state collapses onto it (eq. 7).
- **`V̂`** — the advance-demand still outstanding beyond the lead time, the one
  component `u` does not absorb when the horizon is long enough to need it
  (eq. 11). The order policy is `(s(V̂), S)`: order when `u` falls below a
  trigger `s` that depends on `V̂`, and order up to a level `S` that does not.

Section references are to **the paper** where they say "paper §" and to the
**auto-mdp-solver spec** where they say "spec §".

**Two branches, reported separately and never head-to-head:**

| branch | paper | the allocation | reference available |
|---|---|---|---|
| **homogeneous** | paper §3 | not live — maximal early fill is provably optimal, so RL decides ordering only | an **exact** DP |
| **heterogeneous** | paper §4 | live — RL decides `(order, allocation)` jointly | the paper §4.1 AP **relaxation** (a lower bound, not attainable) plus the PL heuristics |

## 2. Layout

### Documents

| file | what it holds |
|---|---|
| `README.md` | this overview: the problem, the results by research question, and how to run things |
| [`ESCALATION.md`](ESCALATION.md) | the campaign record — **57 formalization reversals (F1–F57)**, 25 numbered findings (`#E1`–`#E25`), the design tree, the config registry, and the frame/IR changelogs |
| [`INTERPRET.md`](INTERPRET.md) | the spec §14 policy readback of the crowned artifacts `sc4/g6/a1/h4` / `sc4/g7/a1/h5` (`#E26`), with four figures; the 2026-08-31 readback of the categorical crowns is kept inside it as labelled comparisons |
| [`PLAYBOOK.md`](PLAYBOOK.md) | operator lessons that generalise beyond this domain |
| `adi_flex.restatement.md` | the frozen Phase-A problem statement |
| `adi_flex_schema.json` | **the IR — authoritative** for the problem definition |
| `CLAUDE.md` | pointer file and technical instructions for agents working here |
| `figures/` | committed SVGs cited by `INTERPRET.md` (`figures/a0/`: the same four for the retired categorical crowns) |
| `results/` | the run archive (gitignored) — every `select.tsv` and eval TSV the log cites |

### Code

| file | role |
|---|---|
| `adi_flex_uncertainty.py` | the demand generator — the whole vector in **one** draw, components in index order (F8) |
| `adi_flex_scenarios.py` | `AdiFlexScenario`, the 30 declared instances and 3 samplers, `SCENARIOS` |
| `adi_flex_mdp.py` | the two-step core (F7): `advance1` (order → demand → forced fills), `advance2` (allocation → costs), `valid_allocation` |
| `adi_flex_gym.py` | `AdiFlexEnv` — six action modes, five observation modes, reward on the period's last step |
| `adi_flex_policy.py` | the deployable policy — `act(obs)` decodes whichever action space the model was trained in (`seq_mask` → int; `order_protection` → `[order, σ…]`), mode read off the model |
| `adi_flex_ir_adapter.py` | the differential adapter (portable-domain contract) |
| **benchmarks** | |
| `adi_flex_benchmark_common.py` | the shared `(u, v̂)` backward induction and `ObsView` |
| `adi_flex_benchmark_dp.py` / `_dp_eval.py` | role **`exact`** — the homogeneous branch, where the recursion is tight |
| `adi_flex_benchmark_ap.py` / `_ap_eval.py` | role **`relaxed`** — the heterogeneous branch, where it is the AP lower bound |
| `adi_flex_benchmark_rule.py` | role **`feasible`** — `myopic` and the paper's `pl0` / `plsigma` / `plmax` |
| **RL** | |
| `adi_flex_ppo_train.py` / `_ppo_eval.py` / `_ppo_select.py` | train (`--config sc…/g…/a…/h…`, `--order-head {categorical,ordinal}`), protocol eval, and the spec §9.7 three-layer checkpoint screen |
| `adi_flex_ordinal_head.py` | the ordinal order head (`a1`): a trigger probability plus a discretized-Gaussian location/width over the same MultiDiscrete space — the campaign's largest lever (`#E22`–`#E25`) |
| `adi_flex_configs.py` | the §CONFIG-REGISTRY data half; run it for the archive audits |
| `adi_flex_fixed_order.py` | the RQ2 isolation instrument — order from AP, protection learned |
| **gates and readback** | |
| `adi_flex_test.py` | the spec-§1.2 gates: laws, differential, model/rendering boundary, negative controls |
| `adi_flex_action_mode_probe.py` | paired acceptance for the action-mode decodes no harness gate covers |
| `adi_flex_ordinal_head_probe.py` | the head's acceptance gate G1–G5: exactness, point masses, trigger/magnitude gradient separation, masking, σ passthrough |
| `adi_flex_order_gap_probe.py` | the `#E21` instrument — per-period / per-seed decomposition of the order gap against `y*` |
| `adi_flex_policy_probe.py` | the spec §14 probe — `--identified` first, then `--decompose` |
| `adi_flex_readback_score.py` | every scored number in `INTERPRET.md`, reproducible |
| `adi_flex_plot_policy.py` | regenerates `figures/` |

## 3. Results

### Protocol

Every number below is quoted at **8192 seeds from `first_seed 0`, CRN across
arms**, reference bar = each board's own best heuristic, record eval =
**deterministic argmax** — the deployment mode is a masked argmax over a
discrete action, and no exploration mechanism lives in this policy's entropy.
RL rows are **6-seed means** with the across-seed SE (seeds 21–26; `het_exp4`'s
`vec` arm 11–16) unless a row says otherwise.
The other mode appears only as a labelled figure, never as a record. Lower cost
is better; because the seeds are shared, differences between rows are paired.

Symbols used in the tables below:

- **`u`, the modified inventory position** and **`V̂`**, the advance demand
  outstanding beyond the lead time — both defined in §1.
- **`sc…/g…/a…/h…`** — a configuration id, addressing the scenario, the
  environment as presented, the algorithm and the hyperparameters in
  `ESCALATION.md` §CONFIG-REGISTRY.
- **`RQ1`–`RQ4`** — the four tier-2 research questions declared in the IR's
  `research_questions` block, numbered in declaration order: RQ1 the
  protection-level allocation rule (`confirm`), RQ2 a rule beyond a constant
  protection level (`discover`), RQ3 the state-dependent `(s(V̂), S)` order
  policy (`confirm`), RQ4 the modified inventory position as sufficient
  statistic (`bypass`; declared `confirm` until 2026-09-04, F58). The tier-1
  question is `1-comparative` itself.
- **action encodings** — `seq_mask`, the original two-phase quantity action, and
  `order_protection`, this campaign's one-shot encoding (see *The shipped
  policies*).
- the comparison columns are a **percentage of** the named arm, except
  "vs AP bound" which is a **percentage above** it.

Each table lists one row per *arm* — a policy with a declared **role**:
`exact` (provably optimal), `relaxed` (a bound, not attainable by any real
policy), or `feasible` (a policy you could actually run). **★ marks the
shipped RL artifact** on that board, and the `sc…/g…/a…/h…` beside it is its
configuration id.

**The shipped policy on every board is the `vec` observation arm**, which sees
the raw state. Its `vec_mip` sibling is handed `u` precomputed — the transform
the paper had to prove — so it is reported as a **reference row**, never as the
artifact. Shipping the arm that was given a hand-derived feature would answer a
different question than the one the study asks. The two tie closely, and that
tie *is* the `2-structural` sufficient-statistic finding: the network rebuilds `u` when
it is not given it.

### `1-comparative` — how does RL compare with the existing solutions?

The question every scenario asks. **Three boards carry it**, and no number
crosses between them — they are different problems with different references.
Other instances in the registry are probes for specific questions rather than
study boards, and are reported where they are used, not here.

**The answer is split.** RL beats the analytic policy on one board and loses on
the other two, and the split is not about RL: it tracks how much room the
reference leaves. Where the reference is an *exact* DP there is by definition
nothing to win, and RL lands 0.95% above it (1.4% before the ordinal head). Where the reference is a
*relaxation*, RL takes whatever slack the relaxation's own policy carries —
little on `het_exp4`, a great deal on `het3_exp2`.

| scenario | `N` | `T_dl` | `L` | demand mix `λ` | allocation | reference | role in the study |
|---|---|---|---|---|---|---|---|
| `homog_L0_T2` | 30 | 2 | 0 | (0, 0, 6) | not live | exact DP | the correctness fixture — the branch where optimality is known |
| `het_exp4` | 12 | 2 | 0 | (2, 1, 3) | **live** | AP bound + PL | the headline branch — joint order and allocation |
| `het3_exp2` | 12 | 3 | 1 | (2, 1, 1, 2) | **live** | AP bound + PL | the general case: a two-component protection cascade and `L > 0` |

Costs are `K = 100`, `h = 1`, `p = 9` on every board.

#### `homog_L0_T2` — the exact branch

| policy | role | mean cost | SE | vs optimum |
|---|---|---|---|---|
| DP | `exact` | **685.33** | 0.34 | — (DP value 685.2363) |
| PPO — `order_protection`/`vec_mip` + ordinal head (`sc0/g7/a1/h3`), *reference* | `feasible` | 690.87 | 0.73 | 100.82% |
| **★ PPO** — `order_protection`/`vec` + ordinal head (`sc0/g6/a1/h6`) | `feasible` | **691.78** | 0.15 | **100.95%** |
| PPO — `seq_mask`/`vec`, categorical head (`sc0/g2/a0/h2`), *the crown it replaced* | `feasible` | 694.89 | 0.67 | 101.4% |
| myopic | `feasible` | 2162.82 | 0.92 | 315.6% |

Best single artifact 688.74 = 100.51% (`vec_mip`, seed 23). The crowned
artifacts carry the protection-cap width from before F54 (`MultiDiscrete([236,
2])` against today's `[236, 1]`; the component is inert here), so the eval
script and the wrapper size the action mask from the model (F57).

#### `het_exp4` — the headline branch

| policy | role | mean cost | SE | vs AP bound | vs the bar |
|---|---|---|---|---|---|
| AP relaxation | `relaxed` | **336.19** | — | — (lower bound, not attainable) | 98.5% |
| PL(σ), σ=4 — **the bar** | `feasible` | **341.26** | 0.33 | +1.51% | — |
| PPO — tuned `order_protection`/`vec_mip` + ordinal head, *reference* | `feasible` | 342.12 | 0.27 | +1.76% | 100.25% |
| **★ PPO** — tuned `order_protection`/`vec` + ordinal head | `feasible` | **342.48** | 0.26 | **+1.87%** | **100.36%** |
| PL(Σ), Σ=8 | `feasible` | 344.60 | 0.33 | +2.50% | 101.0% |
| PL(0) | `feasible` | 352.40 | 0.37 | +4.82% | 103.3% |
| myopic | `feasible` | 736.24 | 0.74 | +119.0% | 215.7% |

**RL loses here**, at 100.36% of the bar (100.7% before the ordinal head). The
`vec_mip` reference gets closer — 100.25%, 0.86 above the bar — but neither arm
reaches it; what remains is order-magnitude noise below the training signal's
resolution (`#E21`).
The PL ordering reproduces the paper's §4.3 result (PL(σ) best, PL(0) worst).

#### `het3_exp2` — the general case, and the campaign's best result

| policy | role | mean cost | SE | vs AP bound | vs the bar |
|---|---|---|---|---|---|
| AP relaxation | `relaxed` | **332.82** | — | — (lower bound) | 95.9% |
| **★ PPO** — tuned `order_protection`/`vec` + ordinal head (`sc4/g6/a1/h4`) | `feasible` | **337.74** | 0.19 | **+1.48%** | **97.3%** |
| PPO — tuned `order_protection`/`vec_mip` + ordinal head (`sc4/g7/a1/h5`), *reference* | `feasible` | 337.75 | 0.16 | +1.48% | 97.3% |
| PPO — tuned `order_protection`/`vec`, categorical head (`sc4/g6/a0/h4`), *the crown it replaced* — one artifact (placed by its 6-seed control, 338.87) | `feasible` | 337.40 | — | +1.38% | 97.2% |
| PPO — the **derived rung** (spec §8.6 defaults, no tuning) | `feasible` | 345.90 | 0.56 | +3.93% | 99.6% |
| PL(σ) + terminal σ=0 (off-bar variant) | `feasible` | 346.00 | — | +3.96% | 99.7% |
| **PL(σ), ladder (4, 1) — the bar** | `feasible` | **347.17** | — | +4.31% | — |
| PL(Σ) | `feasible` | 350.88 | — | +5.43% | 101.1% |
| PL(0) | `feasible` | 364.78 | — | +9.60% | 105.1% |

There is no `myopic` row: that heuristic is derived for `L = 0` and refuses to
run on this board.

Sorted by mean cost, two decimals; the retired categorical-crown row is placed
by its 6-seed control on the same seeds (338.87), not by its single-artifact
number. Scope: the ★ and reference rows are **6-seed means** at the protocol (seeds
21–26; best artifact 337.166, `vec` seed 21). The categorical-head row is a
**single artifact** re-scored after its search (spec §9.7 ships a tuning winner
without a retrain, so it carries no across-seed SE); its own configuration's
6-seed control on the same seeds is 338.87 ±0.20, so the head's paired gain is
−1.13 (`vec`) / −1.85 (`vec_mip`) even though the single-artifact number reads
below the new mean. The derived rung is a 3-seed mean. The σ=0 variant is **not a bar** — the
benchmarks stay as the paper published them.

#### The shipped policies

All three ship **MaskablePPO** — the algorithm is fixed by the IR, because the
original action encoding needs a per-step action mask. What differs by board is
the encoding, the observation, and how far up the tuning ladder the artifact
came from:

| board | action encoding | order head | observation | rung | config |
|---|---|---|---|---|---|
| `homog_L0_T2` | `order_protection` | ordinal | `vec` | `L4(hp+gym+arch)`, hp tuned for the head (`h6`) | `sc0/g6/a1/h6` |
| `het_exp4` | `order_protection` | ordinal | `vec` | `L4(hp+gym+arch)`, hp from `#E16` | no id — the cell loses to its bar |
| `het3_exp2` | `order_protection` | ordinal | `vec` | `L4(hp+gym+arch)`, hp from `#E18` (`h4`) | `sc4/g6/a1/h4` |

**`order_protection` is the campaign's own encoding**, and it is what changed
the heterogeneous branch. The order and the protection levels `σ₁…σ_{T_dl−1}`
are chosen together at the **pre-demand** information set, and the allocation
then follows mechanically from a priority cascade. It replaces a masked
two-phase quantity action, collapses a period from four agent steps to one, and
needs no mask at all, because every action in its box is feasible by
construction. It beat the quantity encoding at every rung on both heterogeneous
boards; `homog_L0_T2` was later re-run on it with the ordinal head, where the
two encodings are behaviourally identical because the allocation is not live.

**The ordinal order head** (`--order-head ordinal`, id `a1`) is the campaign's
second own component. The order's one-logit-per-quantity head is replaced by a
trigger probability and a discretized-Gaussian location and width, so the
magnitude gradient pools across quantities instead of splitting over adjacent
logits. Measured against the categorical head at identical hyperparameters and
seeds on all three boards — six paired contrasts, all negative — and adopted on
2026-09-04 (`#E22`–`#E25`). It is a policy-architecture knob: same action
space, same cascade, same algorithm, no change to the problem statement.

`het3_exp2`'s hyperparameters came from a 379-trial search over the spec's
`breadth` knob tier, and the head was then added at those hyperparameters
(`het_exp4`'s from a separate study on its own board; `homog_L0_T2`'s `h6` from
a study run for the head). `het_exp4`'s
artifact carries **no configuration id**: an id marks a promotion, and a cell
that loses to its bar was not promoted.

**Why the boards disagree:** the AP relaxation's *policy* sits +1.51% above its
own bound on `het_exp4` and +4.31% on `het3_exp2`. RL takes what the relaxation
leaves — little on the first board, a great deal on the second. On the two
homogeneous boards the reference is an exact DP, so there is nothing to take.

**Details** — the ladder from `L0` to the crowned rung, the tuning studies, the
screening protocol, and the config registry that addresses every run — are in
[`ESCALATION.md`](ESCALATION.md): `#E9`–`#E12` (het_exp4), `#E15`–`#E18`
(het3_exp2), `#E21`–`#E25` (the ordinal head and its adoption), and
§CONFIG-REGISTRY for the ids.

### `2-structural` — does the learned policy have the structure the paper proves?

All four declared stances are closed. Evidence, scope and two retracted
readings: **[`INTERPRET.md`](INTERPRET.md)**.

**Scope.** Read back on the crowned artifacts `sc4/g6/a1/h4` (ships) and
`sc4/g7/a1/h5` on 2026-09-04 (`#E26`), 8192-seed CRN scoring; the 2026-08-31
readback of the categorical crowns (`#E19`, `#E20`) is kept in `INTERPRET.md` as
labelled comparisons, and where the two disagree the crown's number is the
verdict (F59).

| id | stance | the claim as declared | finding |
|---|---|---|---|
| **RQ1** (primary) | `confirm` | on the heterogeneous branch the learned allocation is PL-shaped: allocated ≈ clip(surplus − σ, 0, outstanding) for a recoverable protection level σ | **Not answerable on this arm** — `order_protection` makes `σ` the action, so it is true by construction; the shipped encoding *is* the structure |
| **RQ2** (primary) | `discover` | the learned allocation varies where a fixed-σ PL policy cannot (with state and time-to-go), and the extracted rule scores as a feasible policy | **No, on the shipped policy.** Its protection component, in isolation, is worth −0.06 against the paper's ladder and **+0.42 behind the best constant**; the categorical crown's 0.63 ± 0.05 did not survive the change of order head. The one rule beyond a constant that is worth having — zero protection in the last period, −1.18 alone, −1.84 with the best constant — is hand-written and learned by no head |
| **RQ3** (secondary) | `confirm` | on the homogeneous branch the learned policy recovers Prop 2's `(s(V̂), S)`: `S` independent of the advance-demand profile, `s` decreasing in it | **Confirmed, and the trigger is the paper's.** `S` ≈ 29.5 is flat in `V̂`; `s` falls at slope −1 and **equals AP's at every `V̂`** (the categorical crown sat one unit high). Measured on `het3_exp2` against the AP policy, so closed as supported, not as declared |
| **RQ4** (secondary) | `bypass` | the state collapses onto the modified inventory position `u` (eq. 7): a `vec` arm that must rebuild `u` does as well as a `vec_mip` arm handed it | **Bypassed — the tie is the success.** The arms tie on every board (a dead heat on `homog_L0_T2`, +0.01 on `het3_exp2` under the head), so the paper's transform is not needed as an input; the readback shows why — `vec` rebuilds `u` exactly on the supply side, while the demand profile adds 4–6 points beyond it. Declared `confirm` until 2026-09-04, under which the same tie read only "consistent" (F58) |

**In one paragraph.** The paper's structures survive, and the ordinal head
brought the learned policy closer to them: `(s(V̂), S)` ordering with the paper's
own `s`, a protection-level allocation at the paper's near-class level, `u` as
the statistic both run on. The learned policy differs from the analytic one in
**one constant** — `S` about 5 lower — and **one boundary condition on each
side**: it stops ordering when a delivery can no longer serve the horizon, which
AP never does, and it never zeroes its protection in the last period, which the
paper's ladder never does either. The ordering is **93% of the −10.0 edge**; two
integers applied to AP recover **90%** of it. Both analytic policies are
stationary approximations of a finite-horizon problem, drawn from a relaxation
that over-values inventory.

## 4. Technical appendix

The commands that run the code, each verified to run as written and together
sufficient to reproduce `results/` from an empty folder. **The gate commands are
not here** — validation, conformance, laws, differential and pytest are in
`CLAUDE.md`, because they are run by whoever is changing the folder rather than
reading it.

### Benchmarks

From `adi_flex/`, and **without** `PYTHONSAFEPATH` — these scripts import their
siblings by name and need the working directory on `sys.path`.

```bash
# homogeneous: the exact DP
python adi_flex_benchmark_dp.py -s homog_L0_T2
python adi_flex_benchmark_dp_eval.py -s homog_L0_T2 \
       --dp-solutions results/homog_L0_T2/dp/homog_L0_T2_policy.npz

# heterogeneous: the AP bound, then the PL heuristics against it
python adi_flex_benchmark_ap.py      -s het3_exp2
python adi_flex_benchmark_ap_eval.py -s het3_exp2
for pol in pl0 plsigma plmax myopic; do
  python adi_flex_benchmark_rule.py -s het3_exp2 --policy $pol \
         --dp-solutions results/het3_exp2/ap/het3_exp2_policy.npz
done
```

### Training, evaluation, selection

`--total-timesteps` counts **agent** steps, and a period is `1 + n_alloc` of
them under `seq_mask` but **one** under `order_protection`. Budgets are set in
episodes, not steps, so the two encodings stay comparable.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# the crowned cells, by config id (the id fixes action/observation mode, the
# ordinal head and every hp; the level is derived and named in the run dir)
python adi_flex_ppo_train.py -s het3_exp2   --config sc4/g6/a1/h4 --total-timesteps 666672  --seed 21
python adi_flex_ppo_train.py -s homog_L0_T2 --config sc0/g6/a1/h6 --total-timesteps 1500000 --seed 21
# the derived rung, for a ladder floor
python adi_flex_ppo_train.py -s het3_exp2 -a order_protection -o vec \
       --level L1 --total-timesteps 666672
python adi_flex_ppo_eval.py  -s het3_exp2 -a order_protection -o vec \
       --model-path results/het3_exp2/PPO_<run>/het3_exp2_ppo_final.zip \
       --n-seeds 8192 --first-seed 0
python adi_flex_ppo_select.py -s het3_exp2 --runs results/het3_exp2/PPO_<run>/ \
       --screen-seeds 2048 --screen-first 100000 \
       --protocol-seeds 8192 --protocol-first 0 --topk 3
```

### Readback and audits

```bash
python adi_flex_configs.py                                   # archive audits
python adi_flex_action_mode_probe.py                         # decode acceptance
python adi_flex_ordinal_head_probe.py                        # a1 acceptance gate G1–G5
python adi_flex_policy_probe.py --validate                   # the instrument against a known constant
python adi_flex_policy_probe.py --identified --decompose     # the crown (default --config sc4/g6/a1/h4); a0 ids reproduce #E19/#E20
python adi_flex_readback_score.py --all --n-seeds 8192       # every scored number in INTERPRET.md
python adi_flex_readback_score.py --n-seeds 8192 --fig4 figures/fig4_order_sS.svg
python adi_flex_plot_policy.py --head a1                     # figures 1–3 (--head a0 --outdir figures/a0 for the retired crowns)
# the deployable wrapper, replaying the crowned artifact; the mode is read off the model
python adi_flex_policy.py -s het3_exp2 -o vec --episodes 5 \
       --model-path results/het3_exp2/PPO_<run>/het3_exp2_ppo_final.zip
```

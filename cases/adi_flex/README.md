# `adi_flex` — inventory with advance demand information and flexible delivery

A joint **ordering + allocation** policy for the heterogeneous-customer model of

> Tong Wang and Beril L. Toktay (2008), "Inventory Management with Advance
> Demand Information and Flexible Delivery", *Management Science* 54(4),
> 716–732 — §4.

Customers order ahead and differ in how long they will wait: an order placed
now may be due **now**, **next period**, or **two periods out**, and you see it
the moment it is placed. You may ship any time up to the due date, so shipping
early converts your holding cost into the customer's.

The difficulty is **demand crossover**: an order arriving next period may be due
before one you are already holding, so stock spent today shipping a not-yet-due
order may be stock you needed tomorrow for someone urgent. The manager
therefore chooses not only *how much to buy* but *how much to withhold*.

The paper cannot solve that joint problem. It brackets it instead — a relaxation
(`AP`) bounds the optimum from below, and three protection-level heuristics
bound it from above, the best (`PL(σ)`) averaging a 2.08% gap. **This domain
asks whether a learned policy can do better**, which is the improvement the
paper itself points at: its σ is a constant, and §4.2 notes the rule "could be
further refined by incorporating the effect of the fixed-ordering cost K, ...,
inventory level `x_i`, or even the whole system state."

## Layout

| file | role |
|---|---|
| `adi_flex_schema.json` | frozen MDP-IR (`seed_scheme: v2`; `mdp` fingerprint `602b284da491`) |
| `adi_flex.restatement.md` | Phase-A round-trip artifact: plain-English model + annotated trajectory |
| `adi_flex_uncertainty.py` | the three Poisson demand streams (one per due date) |
| `adi_flex_scenarios.py` | `AdiFlexScenario` + the eight-instance ladder |
| `adi_flex_mdp.py` | `AdiFlexState` and the five-region fulfilment dynamics |
| `adi_flex_gym.py` | Gymnasium wrapper; `MultiDiscrete([61, 21])` joint action |
| `adi_flex_ir_adapter.py` | differential adapter (portable-domain contract) |
| `adi_flex_benchmark_ap.py` | the `AP` relaxation, solved exactly by DP |
| `adi_flex_benchmark_pl.py` | `PL(0)` / `PL(σ)` / `PL(Σ)` protection-level heuristics |
| `adi_flex_benchmark_myopic.py` | synthesised single-period order-up-to |
| `adi_flex_benchmark_random.py` | uniform over the action space |
| `adi_flex_benchmark_common.py` | shared §9 seed loop and TSV writer |
| `adi_flex_ppo_train.py` | SB3 PPO training |
| `adi_flex_ppo_eval.py` | §9 eval, incl. the `--force-hold-back` ablation |
| `adi_flex_policy.py` | deployable wrapper: observation in, (order, hold-back) out |

## The model

| | |
|---|---|
| Horizon | 12 periods, finite, undiscounted (α = 1) |
| State | `inventory` (negative = overdue backlog), `due_now`, `due_next`, `period` |
| Decisions | `order_quantity` ∈ {0…60}, `hold_back` ∈ {0…20}, both integer, both committed before demand arrives |
| Uncertainty | three independent Poisson streams, one per due date |
| Objective | minimise `100·1{ordered} + 1·on-hand + 9·overdue` |

**Structural boundaries** (moving either requires re-entering Phase A):
supply lead time `L = 0`, so there is no pipeline; demand window `T = 2`, so the
advance profile is exactly `(due_now, due_next)`. `T > 2` additionally changes
the *decision dimension* — §4.2 needs one order plus `T−1` protection levels —
so it would need an `mdp_ir` extension, not just a new instance.

## Scenarios

The paper's Experiments 0–7 (§4.3), holding total arrival rate at 6 and sliding
mass between due dates. `L=0, K=100, h=1, p=9, N=12`.

| instance | (λ₀,λ₁,λ₂) | what it is | crossover |
|---|---|---|---|
| `exp0` | (6,0,0) | no ADI — the traditional model (homogeneous, T=0) | none |
| `exp1` | (5,1,0) | nothing ever due two periods out | none |
| `exp2`–`exp5` | (4,1,1)→(1,1,4) | the crossover ladder | **yes** |
| `exp6` | (0,1,5) | no urgent arrivals to protect against | none |
| `exp7` | (0,0,6) | full ADI — exactly §3's homogeneous model at T=2 | none |

## Reproducing

```bash
PY=../../.venv/bin/python        # or $MDP_SOLVER_PYTHON

# --- correctness gates -----------------------------------------------------
$PY -m mdp_ir            adi_flex_schema.json          # IR validates
$PY -m mdp_conformance   .                             # 14/16 (2 SKIP: no samplers/grids)
for i in 0 1 2 3 4 5 6 7; do                           # bit-exact vs interpreter
  $PY -m mdp_ir.differential adi_flex_schema.json --episodes 40 --instance exp$i
done

# --- benchmarks (exp4 is the Stage-0 target) -------------------------------
$PY adi_flex_benchmark_ap.py --check-tables             # reproduces paper Tables 2 & 3
$PY adi_flex_benchmark_ap.py        -s exp4
$PY adi_flex_benchmark_ap_eval.py   -s exp4
for pol in pl0 plsigma plmax; do
  $PY adi_flex_benchmark_pl_eval.py -s exp4 --policy $pol --n-seeds 8192
done
$PY adi_flex_benchmark_myopic_eval.py -s exp4 --n-seeds 8192
$PY adi_flex_benchmark_random_eval.py -s exp4 --n-seeds 8192

# --- train + evaluate ------------------------------------------------------
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY adi_flex_ppo_train.py \
    -s exp4 --total_timesteps 500000 --n_envs 8 --n_steps 512 --seed 42
$PY adi_flex_ppo_eval.py --model-path results/exp4/<RUN>/exp4_ppo.zip \
    -s exp4 --n-seeds 8192            # ~345.3 (single seed; ~1 unit run-to-run)
```

`results/` is gitignored; the commands above regenerate it.

## Results

Instance `exp4` = (2,1,3), the split where the paper reports `PL(σ)`'s
optimality gap peaks. All rows are 8192 episodes on identical seeds (0…8191),
same simulator, same TSV protocol. Cost — **lower is better**.

The PPO rows are a **single training seed** (42); the benchmarks are analytic
or seed-averaged. Cost — **lower is better**.

| policy | cost | % over AP | SE |
|---|---|---|---|
| `AP` — relaxation lower bound (*not attainable*) | **336.19** | — | exact |
| `PL(σ)` — the paper's best heuristic | **341.33** | 1.53% | 0.33 |
| `PL(Σ)` | 344.65 | 2.52% | 0.33 |
| **PPO, `obs=vec`** ← shipped | **345.33** | 2.72% | 0.37 |
| PPO, `obs=vec_mip` (failed repair axis) | 346.11 | 2.95% | 0.34 |
| `PL(0)` | 352.43 | 4.83% | 0.37 |
| myopic | 1203.25 | 257.9% | 0.21 |
| random | 3059.15 | 810.0% | 5.08 |

Pairwise against the shipped `vec` model (positive z = PPO cheaper; z from
marginal SEs, so conservative — the protocol is paired on seeds 0…8191):

| comparison | z | verdict |
|---|---|---|
| PPO vs `PL(σ)` | **−8.10** | PPO significantly **worse** |
| PPO vs `PL(Σ)` | −1.37 | statistically tied |
| PPO vs `PL(0)` | **+13.64** | PPO significantly better |

**Eval gate: PASS** — the candidate beats both mandatory baselines by far more
than 2 SE (random z ≈ 533, myopic z ≈ 2040). But the gate's baselines are not
the interesting bar: **the learned policy did not beat the paper's best
heuristic.** It reached 102.7% of the AP bound where `PL(σ)` reaches 101.5%,
and lands essentially on `PL(Σ)`.

## Validation

The benchmarks are checked against the paper rather than only against
themselves:

- **Table 3** — the `AP` dynamic program reproduces the period-1
  `(s(v̂), S(v̂))` policy for all eight instances exactly, and independently
  recovers both structural propositions (`S` constant in `v̂`, `s` decreasing).
- **Table 2** — all ten protection levels for `PL(σ)` and `PL(Σ)` match.
- **No-crossover instances** — the paper proves the heuristics attain the bound
  where crossover is absent. Simulated `PL(0)` matches the DP value to within
  ±0.1% at `exp0`, `exp1`, `exp6`, `exp7`, which jointly validates the DP, the
  policy extraction, the simulator, and the `v̂` bookkeeping.
- **Differential gate** — the generated dynamics reproduce the IR interpreter
  bit-exactly across all eight instances (4,320 periods, 15 fields including
  the fulfilment `region`, so the branch *selection* agrees and not merely the
  resulting state).

## Findings

**1. PPO rediscovered the paper's policy structure without being told it.**
Replaying the shipped model episode by episode:

```
orders    = [0, 31, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0]
hold_back = [5, 10, 2, 5, 5, 11, 1, 4, 4, 4, 4, 4]
```

Two large batches rather than steady replenishment — the economic order
quantity here is `sqrt(2Kd/h) ≈ 34.6` and the DP's order-up-to level is 30, and
the two batches (31, 41) bracket both — and, off the order periods, a protection
level sitting at ≈4, which is exactly `PL(σ)`'s newsvendor value (Table 2). Both
structures were learned from reward alone (pattern consistent across seeds).

**2. The learned allocation is no better than the paper's constant.** This
domain existed to test the opening in §4.2, that σ "could be further refined by
incorporating ... inventory level `x_i`, or even the whole system state". The
`--force-hold-back` ablation overrides the policy's allocation while leaving its
ordering intact (4096 matched seeds, SE ≈ 0.52):

| allocation | cost |
|---|---|
| forced σ = 4 (the paper's constant) | 345.11 |
| **learned** | 345.16 |
| forced σ = 8 (`PL(Σ)`'s value) | 348.54 |
| forced σ = 0 (no protection) | 359.05 |

Learned and constant-σ are **statistically indistinguishable** (0.05 apart on a
paired comparison). Forcing σ=0 costs 14 units, so the policy did learn that
protecting matters — it simply found no state-dependent refinement worth
anything. On this instance, the answer to the paper's open question is *no*.

**3. Handing the policy the paper's sufficient statistic made it (weakly)
worse.** `vec_mip` exposes `(period, u, due_next)` where `u = inventory −
due_now − due_next`, the statistic Propositions 1–2 prove the optimal *ordering*
policy depends on. Under v2 it came out **0.8 units worse** than the raw `vec`
state (z ≈ 1.6) — nominally worse but within single-seed training noise, so a
much weaker signal than the pre-v2 run's 3.2-unit gap suggested. The *direction*
matches the mechanism below and it is still recorded as a failed design axis,
but the magnitude is not robust to the training seed.

The reason is instructive rather than incidental: `u` is sufficient for the
**ordering** sub-problem under the AP relaxation, but this domain also decides
**allocation**, and that decision needs `due_now` — how much is due right now —
which `u` has already summed away. Compressing to the paper's statistic is
lossless for the problem the paper solves and lossy for the joint problem. A
sufficient statistic is only sufficient for the question it was derived for.

**4. Where the remaining gap is.** PPO sits 2.72% above the AP bound against
`PL(σ)`'s 1.53%. Since the allocation is already at parity (finding 2), the
entire deficit is in **ordering** — unsurprising, given `PL(σ)` orders using an
exactly-solved dynamic program while PPO must learn the same non-stationary
`(s,S)` structure from returns. The honest summary is that the learned policy
matched the easier half of the problem and lost the half the paper had already
solved optimally.

## Known limitations

- Only `exp4` was trained and evaluated (the Stage-0 target). The other seven
  instances have gated dynamics but no trained policy.
- **The PPO rows are a single training seed (42).** Re-training the same config
  shifts the result by ≈1 cost unit (observed directly across the v1→v2
  re-keying), which is comparable to the gaps *between* `PPO`, `PL(Σ)`, and
  `vec_mip`. So the coarse ranking — PPO ≈ `PL(Σ)`, below `PL(σ)`, far above
  `PL(0)` — is robust, but fine distinctions among the middle rows are not
  resolved without multi-seed averaging.
- **Hyperparameter tuning was explored but yielded no reproducible gain.** An
  Optuna study (~690 trials, 500k steps/trial) found a best config ~1 unit under
  the default, but that config, retrained cleanly, regressed to 344.8 —
  indistinguishable from the default. The apparent gain was winner's curse (the
  best-of-690 order statistic under single-seed noise), so the shipped result is
  the default configuration and tuning is not reported as a result.
- The two homogeneous corners use N=12 (§4's horizon), so they do not reproduce
  Figure 4's N=30 costs.
- The delivery-flexibility comparison (the paper's Δ, ≈14% average saving) is
  out of scope: no exact-delivery dynamics exist in this IR.

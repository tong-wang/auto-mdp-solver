# dynamic_pricing — finite-horizon dynamic pricing of a fixed stock

An RL study of the Gallego & van Ryzin (1994) pricing problem, *Management
Science* **40**(8): sell a fixed initial stock over a finite horizon by
setting a price each period under price-sensitive Poisson demand, against the
exact dynamic-programming optimum of the discretized problem and the paper's
own fluid heuristics. Generated from the IR `dynamic_pricing_schema.json` by
**auto-mdp-solver**.

**TL;DR**

- **RL lands within one percent of the exact optimum on the record instance
  (`simple`).** The tuned PPO policy scored 32.96 against the DP's 33.15 on
  the same 8192 seeds (99.4%); the untuned one scored 32.82 (99.0%).
- **RL beats the paper's asymptotically optimal fixed-price heuristic.** The
  fixed price scores 31.92 (96.3% of the DP); the myopic price sells out early
  at 24.97, and a random price scores 12.56.
- **The RL rows are a July 2026 record, not a reproduction.** No trained
  artifact survives and the seed tree has changed since; the four benchmark
  arms were re-measured on 2026-09-14 and agree with the record within two
  standard errors, the RL arms have not been.
- **On `ample_stock` the stock constraint never binds.** The DP, the fixed
  price and the myopic price coincide at 7.39, so there is nothing for a
  learned policy to add on that instance and no RL arm was trained there.

## The problem

A seller holds `n0` units of one product and a selling season of `T`
periods. At the start of each period they set a price `p`; customers then
arrive as a Poisson stream whose intensity falls exponentially in the price,
`lambda(p) = a * exp(-alpha * p)`, and each arrival buys one unit while stock
lasts. Unsold demand is lost, stock is never replenished, and unsold units at
the horizon are worth a salvage value `q` (zero here). The objective is the
expected total revenue of the season. The state is `(period, inventory)`, the
decision is one continuous price.

What makes it more than a static pricing problem is the coupling through the
stock: a low price sells quickly but exhausts the inventory before the season
ends, a high price preserves stock that may never sell. The optimal price is
therefore state-dependent — it rises as stock runs low relative to the time
remaining and falls as the horizon approaches with stock still on hand — and
the paper's analysis is in continuous time, where the seller controls the
intensity directly. This domain discretizes that control problem into `T = 50`
periods of length `dt = 0.02` (a unit-length horizon), so a period sees on
average `lambda(p) * dt` arrivals.

What is known analytically is the whole answer for the discretized problem:
backward induction over `(period, inventory)` with the price optimized on a
fine grid is exact for the MDP the agent faces. The paper also supplies a
deterministic fluid relaxation whose value is at least the optimum and never
attainable, and two static policies drawn from it — the revenue-rate
maximizing myopic price `p* = 1/alpha`, and the fixed price
`p_FP = max(p*, p0)` where `p0` sells exactly `n0` units in expectation over
the season, which the paper proves asymptotically optimal as stock and demand
scale up. The leaderboards below are measured against the DP, with the fixed
price as the must-beat heuristic.

**Source.** Guillermo Gallego and Garrett van Ryzin (1994), "Optimal Dynamic
Pricing of Inventories with Stochastic Demand over Finite Horizons",
*Management Science* **40**(8), 999–1020. Changed from the paper: the
continuous-time intensity control is discretized into 50 equal periods, and
the price (not the intensity) is the canonical decision, with the paper's
native intensity control kept as a second action encoding. The authoritative
statement of the problem is `dynamic_pricing_schema.json`;
`dynamic_pricing.restatement.md` renders it in prose.

## Layout

Documents — where to read about the case:

| doc | what it holds |
|---|---|
| `dynamic_pricing.restatement.md` | the Phase-A restatement: the problem as the IR states it, with a rendered trajectory — backfilled 2026-09-14 from the frozen IR, no interview record survives |
| `ESCALATION.md` | the campaign — design tree, changelogs, config registry, findings `#E1`–`#E8` with verdicts; seeded 2026-09-14 from the record |
| `INTERPRET.md` | not owed: the IR declares no tier-2 stance, so spec §14 owes only its generic pieces, and none were produced |
| `PLAYBOOK.md` | the case-close digest (guide §10): levers LV1–LV2, frame moves FM1–FM2, the envelope verdict — distilled 2026-09-14 at re-contribution; the campaign is dormant, not closed |
| `CLAUDE.md` | operating brief for an agent changing this folder |

Code — where things are implemented:

| module | role |
|---|---|
| `dynamic_pricing_schema.json` | **the IR — authoritative** for the problem definition |
| `dynamic_pricing_uncertainty.py` | the price-conditioned Poisson arrivals generator and its seed key (`SamplingContext`, `PoissonArrivals`) |
| `dynamic_pricing_scenarios.py` | the world layer: `DynamicPricingScenario` and the `SCENARIOS` registry (`simple`, `ample_stock`) |
| `dynamic_pricing_mdp.py` | state and transition — pure, no reward; the episode ends early when stock hits zero |
| `dynamic_pricing_gym.py` | the Gymnasium wrapper: action modes `price` (clip) and `intensity` (reparametrized), observation modes `vec` and `vec_d`, reward mode `revenue` |
| `dynamic_pricing_ir_adapter.py` | the differential adapter (portable-domain contract) |
| `dynamic_pricing_test.py` | the domain's own tests: engine laws, the differential over the covering set, and the price-path independence claim |
| `dynamic_pricing_ppo_train.py` / `dynamic_pricing_ppo_eval.py` | SB3 PPO training with periodic checkpoints; seed-block evaluation with VecNormalize injection |
| `dynamic_pricing_ppo_tune.py` | thin wrapper over the `mdp_tuning` harness, pre-filling `--metric revenue_mean` |
| `dynamic_pricing_benchmark_dp.py` | role `exact`: backward induction over `(period, inventory)` on a price grid; writes the policy table |
| `dynamic_pricing_benchmark_dp_eval.py` | evaluates the DP table as a policy, same columns and seed block as the RL eval |
| `dynamic_pricing_benchmark_fluid.py` | role `feasible`: the paper's static policies `fixed` and `myopic`, plus a `random` floor, selected by `--policy` |
| `dynamic_pricing_policy.py` | the deployable wrapper over a trained artifact (`DynamicPricingPolicy.act`) — nothing to load until an artifact is re-trained |

## Results

### Protocol

Every number below is quoted at **8192 episode seeds, seeds 0…8191, the same
block on every arm of a board**, so a Δ between two arms of one board is a
comparison on common random numbers. Reference bar = the DP's evaluated row
on that block (not the solver's computed value, which is quoted beside it as
a calibration). Record eval = **deterministic argmax**: the action is one
continuous price and exploration plays no role at inference, so the argmax is
the deployment mode. The other mode appears only as a labelled figure, never
as a record.

**Two protocol generations, and they are different frames.** The July 2026
board was produced in the author's research workspace under the four-word
seed key that preceded the solver's v2 seed tree; every arm on it shares that
key, so the board is one frame. The 2026-09-14 re-measurement runs the same
seed block through the current v2 key — a different draw of the same
episodes — for the four benchmark arms only, since no RL artifact survives.
Numbers may be compared within a generation and only *reported* across one.

Symbols used in the tables below:

- **`a`, `alpha`, `n0`, `T`, `dt`, `q`** — the demand scale and price
  sensitivity of `lambda(p) = a * exp(-alpha * p)`, the initial stock, the
  number of periods, the period length and the salvage value.
- **`p*`, `p_FP`** — the myopic price `1/alpha` and the paper's fixed price
  `max(p*, p0)` (defined in **The problem**).
- **role** — `exact` solves the MDP; `feasible` is a real policy under the
  real information set. The fluid relaxation's *value* (34.66 on `simple`) is
  a `relaxed` quantity, at least the optimum and unattainable; it is not
  emitted as an eval column, so it has no arm on the board.
- **`L1`, `L2(hp)`** — the two configuration rungs trained: the train
  script's defaults, which `_L1_DERIVED` records as the spec §8.6 derivation,
  and the winner of a 25-trial Optuna study. Their config ids in
  `ESCALATION.md` §CONFIG-REGISTRY are `sc0/g0/a0/h0` and `sc0/g0/a0/h1`.
  No `L0` faithful-defaults run exists.
- **`price`, `intensity`** — the two action encodings: the agent sets the
  price directly on `[0, 8]`, or sets a target intensity on `[0, a/e]` that
  the wrapper inverts to a price.
- **`±`** — the unpaired standard error of a difference, `sqrt(SE1² + SE2²)`
  from the two rows' variances; the July board records means only, so its
  Δ column carries no `±`.

### `1-comparative` — how does RL compare with the existing solutions?

The scenarios, and the role each plays in the study:

| scenario | config | what it is for |
|---|---|---|
| `simple` | `a` 100, `alpha` 1, `n0` 25, `T` 50, `dt` 0.02, `q` 0 — demand high relative to stock (the paper's Fig. 1, bottom) | the record board: the stock constraint binds, so the optimal price is state-dependent and there is something to learn |
| `ample_stock` | as `simple` with `a` 20 — demand low relative to stock | the degenerate control: run-out is unlikely, the DP collapses to the constant myopic price, and the board exists to show that |

**The two scenarios are separate leaderboards whose scores may never be
compared.** Demand on `ample_stock` is a fifth of `simple`'s, so its revenues
are on a different scale and its bar is a different number.

#### `simple` — the record board (July 2026, pre-v2 seed key)

| arm | role | revenue | Δ vs bar | % of bar | scope |
|---|---|---|---|---|---|
| `dp` | **exact** | **33.15** | — | — | **the bar**: the evaluated DP table; the solver's computed `V[0][n0]` is 33.10 |
| PPO `price`, `L2(hp)` | feasible | 32.96 | −0.19 | 99.4% | Optuna trial 16, 200k steps, re-scored on the protocol block |
| PPO `price`, `L1` | feasible | 32.82 | −0.33 | 99.0% | the script's defaults, 500k steps, one seed |
| `fixed` (`p_FP` = 1.386) | feasible | 31.92 | −1.23 | 96.3% | the must-beat baseline: the paper's asymptotically optimal static price |
| PPO `intensity`, `L2(hp)` | feasible | 25.95 | −7.20 | 78.3% | best of two runs: 1M steps with `ent_coef` 0.01; the 500k-step defaults run scored 17.34 |
| `myopic` (`p*` = 1.0) | feasible | 24.97 | −8.18 | 75.3% | sells out early — the price is too low for `n0` 25 |
| `random` (uniform on `[0, 8]`) | feasible | 12.56 | −20.59 | 37.9% | the sanity floor |

Sorted by revenue. Every learned `price` arm clears the fixed-price baseline;
the `intensity` arms do not, and the reason is recorded as a tier-3 finding
in `ESCALATION.md` `#E3`: the policy's Gaussian initializes at raw action 0,
which in the intensity box is the shut-off price, so half the initial action
mass earns nothing. Neither `intensity` row is a pruned arm — both encodings
are declared in the IR and both are covered and reported.

Two caveats the numbers hide. The tuned row is one artifact selected as the
maximum over 25 trials, so it measures that artifact and not the tuning
procedure: retraining its configuration for 500k steps scored 32.31 (`#E5`).
And every RL row is one training seed; no retrain spread was measured.

**Re-measured 2026-09-14 under the v2 seed tree** (same block, different
draw; benchmark arms only):

| arm | role | revenue | Δ vs bar | % of bar | scope |
|---|---|---|---|---|---|
| `dp` | **exact** | **33.02** | — | — | **the bar**; SE 0.05; computed `V[0][n0]` 33.10 |
| `fixed` | feasible | 31.85 | −1.17 ± 0.07 | 96.5% | |
| `myopic` | feasible | 24.95 | −8.06 ± 0.05 | 75.6% | |
| `random` | feasible | 12.44 | −20.58 ± 0.08 | 37.7% | |

Sorted by revenue. Each row sits within two standard errors of its July
counterpart (`#E8`), so the benchmarks reproduce; the RL rows await a
re-trained artifact (`ESCALATION.md` §Frontier, A3).

**Shipped: nothing, today.** The record's shipped artifact was the trial-16
model itself — chosen over the 500k retrain of its own configuration because
the retrain scored lower (`#E5`) — but no artifact from the July campaign is
in this folder, so `dynamic_pricing_policy.py` has nothing to load. When the
RL arms are re-trained, the choice is made again on that board.

How it was reached: `ESCALATION.md` §MAP and `#E2`–`#E5`.

#### `ample_stock` — the degenerate control

| arm | role | revenue | Δ vs bar | % of bar | scope |
|---|---|---|---|---|---|
| `dp` | **exact** | **7.39** | — | — | **the bar**; the DP policy is the constant myopic price |
| `fixed` | feasible | 7.39 | 0.00 | 100.0% | `p_FP` = `p*` = 1.0 here |
| `myopic` | feasible | 7.39 | 0.00 | 100.0% | identical policy, identical trajectories |
| `random` | feasible | 2.49 | −4.90 | 33.7% | |

Sorted by revenue; July 2026 record. Re-measured 2026-09-14 under the v2 key:
`dp` = `fixed` = `myopic` 7.31 (SE 0.03; computed `V[0][n0]` 7.36), `random`
2.45 — the three coincide to the last digit because they are one policy. No
RL arm was trained: the bar and the heuristic coincide, so there is nothing
to learn (`#E6`).

### `2-structural` — no stance declared

The IR declares no `research_questions` block, so this campaign makes no
tier-2 claim and spec §14 owes only its generic pieces, none of which were
produced. The one structural observation on record — that the paper's native
intensity control is the harder encoding for PPO — is a tier-3 finding about
the action interface (`ESCALATION.md` `#E3`), not a claim about the learned
policy's structure. A stance would be declared through `mdp-formalize`,
which re-opens the sign-off.

## Technical appendix

Run from `cases/dynamic_pricing/`, with the repo's resolved interpreter as
`python` (repo `CLAUDE.md`, Rules). Pin torch to one thread for training. Together these
reproduce every number above from an empty folder. The gate
commands are not here — they are in `CLAUDE.md`, because they are run by
whoever is changing the folder rather than reading it. Every command below
was run as written on 2026-09-14 (the training line at a smoke budget).

```bash
# the bar and the heuristics, on the protocol block
python dynamic_pricing_benchmark_dp.py -s simple            # writes the DP table, prints V[0][n0]
python dynamic_pricing_benchmark_dp_eval.py --solutions <dp table> -s simple --n-seeds 8192   # the path the line above printed
for pol in fixed myopic random; do
  python dynamic_pricing_benchmark_fluid.py -s simple --policy $pol --n-seeds 8192
done

# the solve leg: the script's defaults ARE the L1 derivation (sc0/g0/a0/h0)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python dynamic_pricing_ppo_train.py -s simple -a price --total-timesteps 500000              # L1
python dynamic_pricing_ppo_train.py -s simple -a price --total-timesteps 200000 \
    --learning_rate 1.6e-4 --ent-coef 2.6e-3 --gamma 0.9995 --n-steps 512 --batch-size 32 --n-epochs 20   # h1, the trial-16 configuration
python dynamic_pricing_ppo_eval.py -s simple -a price --n-seeds 8192 --first-seed 0 \
    --model-path <run dir>/simple_ppo.zip                                                   # protocol eval, beside the model

# the hyper-parameter study that produced h1 (25 TPE trials x 200k steps)
python dynamic_pricing_ppo_tune.py -s simple --n-trials 25 --total-timesteps 200000
python dynamic_pricing_ppo_tune.py --show-space                                             # the resolved tunable set

# the deployable wrapper, smoke-tested against the raw _mdp loop
python dynamic_pricing_policy.py -s simple -a price --model-path <run dir>/simple_ppo.zip --episodes 5
```

Substitute `-s ample_stock` for the second board; every command takes the
same flags, and `-a intensity` selects the paper's native encoding. Outputs
follow the spec §8.4 tree; each script prints where it wrote.

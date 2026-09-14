# fnv — fresh-newsvendor sequential ordering under MMFE

An RL study of the multi-ordering newsvendor of Wang, Atasu & Kurtuluş
(2012), *Manufacturing & Service Operations Management* **14**(3): `N`
sequential orders at rising cost while a martingale demand forecast (MMFE)
is revealed, against the paper's exact base-stock optimum. Generated from the
IR `fnv_schema.json` by **auto-mdp-solver**.

**TL;DR**

- **RL lands within half a percent of the exact optimum on both branches and
  never below the must-beat baseline.** The best confirmed checkpoint sits
  0.21% under the bar on the additive branch (`FNV-aMMFE`) and 0.51% under it
  on the multiplicative one (`FNV-mMMFE`); every one of the nine retrains per
  branch beats the myopic safety stocks, which sit 1.09% and 2.22% under.
- **The paper's base-stock structure is recovered at the final order.**
  Post-order inventory rises one-for-one with the information signal (slope
  1.109 ± 0.122 over 36 fits, r² 0.966) and the recovered safety stock lands
  0.026 from the exact one; on the multiplicative branch the same holds in
  logs.
- **The small profit gap hides a structural miss that the score cannot see.**
  On the additive branch the policy uses its middle ordering opportunity in
  0–24% of episodes against the optimum's 50%, and the profit landscape is
  flat enough not to show it; on the multiplicative branch it uses it in
  17–43% against 47%, so the miss is branch-specific, not a property of the
  learner.
- **Distilling the policy into the paper's offsets ties the network and does
  not beat it.** The rule can be completed on only 209 of 540 additive cells,
  because a policy that abandons an order cannot be read back over it; on the
  508 multiplicative cells it covers, the network wins the typical cell.

## The problem

A retailer places `N` sequential orders for a single selling season, each at
a higher unit cost than the last, while a demand forecast is progressively
revealed. Demand realizes once, at the horizon, and is sold against the
cumulative inventory at a fixed price. Each period the decision is one
continuous order quantity; the state is the period, the inventory already
committed, and the cumulative forecast signal. The objective is expected
profit: revenue on the sold units less the ordering cost paid along the way.

The forecast follows the martingale model of forecast evolution (MMFE) of
Heath & Jackson (1994): the signal accumulates independent normal increments,
one per period, and the terminal demand is the mean plus the accumulated
signal (the **additive** branch, `D = mu + I`) or its exponential (the
**multiplicative** branch, `D = exp(mu + I)`). What makes the problem hard is
the trade-off between committing early at a low cost against a noisy forecast
and committing late at a high cost against a sharp one. The obvious policy, a
newsvendor order at each period against the uncertainty still remaining, is
provably wrong: it ignores the option to order again and over-orders early.

What is known analytically is the whole answer. Wang, Atasu & Kurtuluş
(2012) prove that the optimal policy is a state-dependent base-stock: order
up to `S_n = mu + I + b_n` (or `exp(mu + I + b_n)` on the multiplicative
branch), where the safety-stock offsets `b_1..b_N` do not depend on the
forecast evolution and solve a one-dimensional recursion off-line
(Proposition 2). Corollary 1 gives the myopic offsets as an upper bound on
the optimal ones. There is nothing left open on the leaderboard; the exact
bar is cheap. What the campaign asks is whether a policy trained only on
profit rediscovers that structure, and where it does not.

**Source.** Tong Wang, Atalay Atasu, Mümin Kurtuluş (2012), "A Multiordering
Newsvendor Model with Dynamic Forecast Evolution", *Manufacturing & Service
Operations Management* **14**(3), 472–484,
[doi:10.1287/msom.1120.0387](https://doi.org/10.1287/msom.1120.0387). © 2012
INFORMS, which holds copyright — the PDF is not redistributed here; read it
via the DOI or your library. What this case
reimplements are mathematical facts, each attributed at the call site: the
Proposition 2 recursion and the Corollary 1 inequality. Changed from the
paper: the cost ladder is arithmetic (`c_n = c1 + (n-1)*lamb`) and the
forecast variance schedule is Brownian — both design choices the paper leaves
free, declared as such in the IR. The authoritative statement of the problem
is `fnv_schema.json`; `fnv.restatement.md` renders it in prose.

## Layout

Documents — where to read about the case:

| doc | what it holds |
|---|---|
| `fnv.restatement.md` | the Phase-A restatement: the problem as the IR states it, with the annotated trajectories |
| `ESCALATION.md` | the campaign — design tree, changelogs, config registry, findings `#E1`–`#E12` with verdicts |
| `INTERPRET.md` | the policy readback (spec §14): what the trained nets actually do, both branches |
| `PLAYBOOK.md` | the case-close digest (guide §10): lessons LV1–LV8, frame moves, modeling rules, envelope verdict |
| `CLAUDE.md` | operating brief for an agent changing this folder |
| `figures/` | the committed policy-structure figures, one per branch, cited by `INTERPRET.md` |
| `results/` | the run archive (gitignored): every checkpoint, screen and eval TSV the log cites |

Code — where things are implemented:

| module | role |
|---|---|
| `fnv_schema.json` | **the IR — authoritative** for the problem definition |
| `fnv_uncertainty.py` | the demand-signal generator and its seed keys (`SamplingContext`, `NormalDemandSignal`) |
| `fnv_scenarios.py` | the world layer: `FnvScenario` and the `SCENARIOS` registry |
| `fnv_grids.py` | the design layer: the two 540-cell grids and the `GRIDS` registry |
| `fnv_mdp.py` | state and transition — pure, no reward |
| `fnv_gym.py` | the Gymnasium wrapper, one observation mode (`vec`) and one action mode (`box`) |
| `fnv_ir_adapter.py` | the differential adapter (portable-domain contract) |
| `fnv_test.py` | the domain's own tests: engine laws and the differential over the covering set |
| `fnv_ppo_train.py` / `fnv_ppo_eval.py` | SB3 PPO training with periodic checkpoints; protocol-tier grid evaluation |
| `fnv_select.py` | the checkpoint screen: ranks a run's checkpoints on a seed block disjoint from the protocol's |
| `fnv_benchmark_prop2.py` | role `exact`: the paper's Proposition 2 recursion for the offsets |
| `fnv_benchmark_dp.py` | role `exact`: the same offsets from an independent value-function DP |
| `fnv_benchmark_myopic.py` | role `feasible`: the myopic safety stocks, the must-beat baseline |
| `fnv_benchmark_fitted.py` | role `feasible`: distils a trained net into per-cell offsets (spec §14.2) |
| `fnv_benchmark_dp_eval.py` | evaluates any offsets file as a policy, same columns and seed block as the RL eval |
| `fnv_policy.py` | the deployable wrapper over a shipped checkpoint (`FnvPolicy.act`) |
| `fnv_policy_probe.py` | the §14 anchored readback probe: action surfaces, structure fit, agreement, fitted-rule scoring |
| `fnv_plot_policy.py` | the §14.3 overlay figure, static and interactive |

## Results

### Protocol

Every number below is quoted at **540 grid cells × 2048 CRN seeds, seed block
0…2047**, identical for every arm, so a Δ between two arms is a paired
per-cell comparison; grid means weight each cell equally. Reference bar =
`prop2`, the exact optimum. Record eval = **deterministic argmax**: the action
is one continuous order quantity and exploration plays no role at inference,
so the argmax is the deployment mode. The other mode appears only as a
labelled figure, never as a record. The screen that ranks checkpoints runs on
a disjoint block (2048 seeds from 1,000,000) and is never quoted.

Symbols used in the tables below:

- **`mu`, `I`, `b_n`** — the demand mean, the cumulative forecast signal, and
  the per-period safety-stock offset of the base-stock policy `S_n = mu + I +
  b_n` (defined in **The problem**). `b̂_n` is an offset recovered from a
  trained policy.
- **`stdev`, `T`, `lamb`** — the three grid axes: total demand standard
  deviation, the time location of the last order (`0 < T < 1`), and the
  per-period cost increment.
- **role** — `exact` solves the MDP; `feasible` is a real policy under the
  real information set. There is no `relaxed` arm, because there is an exact
  bar. A `feasible` arm above an `exact` one is a bug report, not a result,
  and the role gate in `CLAUDE.md` enforces that.
- **`L0`, `L1`, `L1′`** — the three configuration rungs trained per branch,
  three seeds each: the library defaults, the spec §8.6 derivation as first
  read, and its corrected re-derivation. Their config ids in `ESCALATION.md`
  §CONFIG-REGISTRY are `{sc}/L0`, `sc{0,1}/g0/a0/h0`, `sc{0,1}/g0/a0/h0b`.
- **`±`** on a paired Δ is the across-cell standard error (std of the 540
  cell-level Δs over √540). Because the grid is enumerated rather than
  sampled, that is a dispersion measure, not a sampling error; the honest SE
  for a grid-mean claim uses seeds as the replication unit and runs about
  1.8× larger where it was measured. No ladder conclusion changes at that
  factor; one readback claim did (see `2-structural`).

### `1-comparative` — how does RL compare with the existing solutions?

The scenarios, and the role each plays in the study. The two boards are
**grids** in `GRIDS`, not entries in `SCENARIOS`: training draws one cell per
episode through the grid's sampler, evaluation enumerates the cells one row
each, and a grid is never passed where a scenario source belongs.

| scenario | config | what it is for |
|---|---|---|
| `simple` (`SCENARIOS`) | additive, one cell: `stdev` 0.1, `T` 0.9, `lamb` 0.1 | a sanity check and fast-training fixture; no leaderboard |
| `FNV-aMMFE` (`GRIDS`) | additive `D = mu + I`; 540 cells, `stdev` ∈ {0.05…0.3} × 9 values of `T` × 10 of `lamb` | the primary board: does a generalist policy recover the linear base-stock structure? |
| `FNV-mMMFE` (`GRIDS`) | multiplicative `D = exp(mu + I)`; 540 cells, `stdev` ∈ {0.1…0.6} × the same `T` × `lamb` | the replication board: does every additive verdict survive a change of demand scale? |

**The two grids are separate leaderboards whose scores may never be
compared.** Demand on the multiplicative branch is `exp(mu + I)`, so its
profits are on a different scale and its bar is a different number.

#### `FNV-aMMFE` — the additive branch

| arm | role | profit | Δ vs bar | % of bar | scope |
|---|---|---|---|---|---|
| `prop2` | **exact** | **0.888334** | — | — | **the bar**: the paper's Proposition 2 recursion |
| `dp` | **exact** | 0.888334 | −0.0000001 | 0.00% | the same optimum from the independent value-function solver |
| PPO, best of 9 | feasible | 0.886428 | −0.001907 ± 0.000077 | −0.21% | best confirmed checkpoint: an `L0` seed (seed 2, step 1.4M) |
| PPO, `L1′` mean | feasible | 0.885487 | −0.002847 | −0.32% | 3-seed mean of the ship-by-default rung; retrain spread 0.000138 |
| `myopic` | feasible | 0.878686 | −0.009649 ± 0.000517 | −1.09% | the must-beat baseline |

Sorted by profit. `dp` and `prop2` are two independent derivations of the
*same* optimal policy, so they are one bar, not two arms: they agree to a
paired mean of 1e-07 (max 2.4e-05 over 540 cells) against a per-cell eval SE
of 0.0043. Both are listed because their agreement **is** the spec §1.2
second-implementation gate; if they ever diverge, the bar itself is in
question. All nine trained models clear the baseline, spread 0.8825–0.8864;
the best beats it by +0.0077 ± 0.0005 paired.

Two caveats the single number hides. The gap to the optimum is small because
the profit landscape is flat, not because the policy is nearly right: it
misses one of its three ordering opportunities outright (`2-structural`). And
`myopic` is a deliberately weak arm — Corollary 1 makes it over-order early
by construction — so beating it is a floor check, not a result.

#### `FNV-mMMFE` — the multiplicative branch

| arm | role | profit | Δ vs bar | % of bar | scope |
|---|---|---|---|---|---|
| `prop2` | **exact** | **2.291618** | — | — | **the bar** |
| `dp` | **exact** | 2.291616 | −0.000002 | 0.00% | the same optimum, independent solver |
| PPO, best of 9 | feasible | 2.279888 | −0.011730 ± 0.000793 | −0.51% | best confirmed checkpoint: the `L1′` seed 3 model at step 1.7M |
| PPO, `L1′` mean | feasible | 2.277242 | −0.014376 | −0.63% | 3-seed mean of the ship-by-default rung; retrain spread 0.002299 |
| `myopic` | feasible | 2.240843 | −0.050776 ± 0.002640 | −2.22% | the must-beat baseline |

Sorted by profit. All nine models clear the baseline. Unlike the additive
branch, the policy does use the middle ordering opportunity here, in 17–43%
of episodes against the optimum's 47%.

**Shipped: the `L1′` rung on both branches** (`sc0/g0/a0/h0b` and
`sc1/g0/a0/h0b`), not the best single checkpoint. On the multiplicative board
`L1′` beats the library defaults outright (+0.003651 ± 0.000365 paired, about
10 SE) and the best checkpoint is one of its seeds. On the additive board its
3-seed mean *ties* the defaults (t = −0.26) with a 6× tighter retrain spread,
and the best single number belongs to an `L0` seed: best-of-three
systematically flatters the higher-variance arm, so the crown rests on
reproducibility, never on the best draw. The derivation as first read (`L1`)
lost to the defaults on both branches — under-training, not instability — and
was superseded by the re-derivation, not out-competed by an escalation.

How it was reached: `ESCALATION.md` §MAP and `#E1`–`#E3` (the additive
ladder), `#E6` (the replication).

### `2-structural` — is the paper's base-stock structure learned?

One stance is declared in the IR's `research_questions.tier2`, `confirm`,
primary: *the learned policy implements Proposition 2's form — it orders up to
a level that moves one-for-one with the information state (one-for-one in
logs on the multiplicative branch), with a per-period offset independent of
the forecast evolution.* Evidence, method and the two retracted readings are
in `INTERPRET.md`; this is the summary.

**Verdict: confirmed at the final order, on both branches; not at the middle
order on the additive branch.** Across 36 fits (9 models × 4 cells), the slope
of post-order inventory against the signal at the last order is
**1.109 ± 0.122** with r² 0.966, and the recovered `b̂₃` lands a mean
**0.026** from the exact `b₃`; the six properly trained models sit at
1.03–1.08. On the multiplicative branch the same fit in logs recovers the
log-linear form, with the slope approaching 1 as a cell carries more
information. Structural fidelity tracks the configuration error: the refuted
`L1` rung is also worst on structure (slope 1.20–1.26 against `L1′`'s
1.05–1.07). Fits use only periods where the policy actually ordered, one fit
per cell — the order quantity is censored at zero, so a period with no order
identifies no level.

**The middle order is the exception, and the score does not show it.** The
additive policies use their second ordering opportunity in 0–24% of episodes
against the optimum's 50%, collapsing a three-order policy into two while
ordering the same total quantity, at a profit cost the flat landscape hides.
The multiplicative policies use it in 17–43% against 47%, which fired the
tripwire that had parked the finding and closed it as branch-specific: under
`exp(mu + I)` a stale early commitment is punished multiplicatively, so the
middle order earns its place and the learner finds it (`#E4`, `#E8`).

**The fitted rule, scored as a policy, ties the net it was read from.** On the
additive branch the rule can be completed on **209 of 540** cells — the
policy never orders at period 2 on the other 331, so `b̂₂` is unidentifiable
there — and within them it scores −0.000126 ± 0.000049 against the net. On
the multiplicative branch it covers 508 of 540 and reads +0.002426 on the grid
mean, but that is not a win: the across-seed SE is 0.001365 (1.78 SE), the
median cell Δ is −0.0011, only 38% of cells favour the rule, and about 20
cells at `stdev` 0.6, `T` 0.9 supply 111% of the positive total (`#E9`,
`#E11`). Both branches satisfy spec §14.2's verdict branch 1 (|fitted − net|
within 1% of the bar), so the structural claim stands; the earlier claim that
the distilled constants *improve* on the network is withdrawn. The coverage
split is the sharper diagnostic: a policy that abandons an ordering
opportunity cannot be distilled into a rule over it, however well it scores.

**Envelope.** RL is competitive but not preferred on this structure class: the
exact optimum is a one-dimensional recursion solved off-line in milliseconds
per cell, so a learned policy has no efficiency argument and lands 0.2–0.6%
below it. What the campaign delivers is the readback — the policy
demonstrably rediscovers the paper's linear and log-linear base-stock form
with unit slope — and the record of where it does not (`PLAYBOOK.md`).

## Technical appendix

Run from `cases/fnv/`. Pin torch to one thread for
training. Together these reproduce `results/` from an empty folder;
`results/` is gitignored. The gate commands are not here — they are in
`CLAUDE.md`, because they are run by whoever is changing the folder rather
than reading it.

```bash
# the optimal-policy bar: two independent solvers, one evaluator
python fnv_benchmark_prop2.py -s FNV-aMMFE --check-invariants --cross-check   # the paper's eqs (6)-(7)
python fnv_benchmark_dp.py    -s FNV-aMMFE --self-test                        # value-function DP
python fnv_benchmark_myopic.py -s FNV-aMMFE                                   # the must-beat baseline
for m in prop2 dp myopic; do
  python fnv_benchmark_dp_eval.py -s FNV-aMMFE --n-seeds 2048 --seed-base 0 \
      --dp-solutions results/FNV-aMMFE/benchmark/$m/FNV-aMMFE.txt
done

# the solve leg, in order — the terminal checkpoint is never the deliverable.
# The script's defaults ARE the spec §8.6 derivation (L1, h0); L0 spells out
# the library defaults, and L1′ (h0b) changes two of the derived values
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python fnv_ppo_train.py -s FNV-aMMFE --seed 2 --total-timesteps 2000000 \
    --learning_rate 3e-4 --lr-final 3e-4 --clip-final 0.2 --n-steps 2048 --batch-size 64 \
    --ent-coef 0 --target-kl none --n-envs 1 --no-norm-reward --no-norm-obs   # L0
python fnv_ppo_train.py -s FNV-aMMFE --seed 2 --total-timesteps 2000000 \
    --learning_rate 3e-4 --lr-final 3e-5 --batch-size 64                       # L1′ (h0b), ship-by-default
python fnv_select.py results/FNV-aMMFE/PPO_<run>                              # screen, 2048 seeds from 1,000,000
python fnv_ppo_eval.py -s FNV-aMMFE --n-seeds 2048 --seed-base 0 \
    --model-path results/FNV-aMMFE/PPO_<run>/checkpoints/fnv_ppo_<step>_steps.zip   # protocol eval

# readback (spec §14): anchored on the DP reference
python fnv_plot_policy.py  -s FNV-aMMFE --model-path <ckpt>                   # slopes + figure (needs plotly, kaleido)
python fnv_policy_probe.py -s FNV-aMMFE --model-path <ckpt>                   # surfaces, sensitivity, fitted-rule score
python fnv_benchmark_fitted.py -s FNV-aMMFE --model-path <ckpt>               # distil the net into offsets …
python fnv_benchmark_dp_eval.py -s FNV-aMMFE --n-seeds 2048 --seed-base 0 \
    --dp-solutions results/FNV-aMMFE/benchmark/fitted/FNV-aMMFE.txt          # … and score them as a policy

# the deployable wrapper, smoke-tested on a shipped checkpoint
python fnv_policy.py -s FNV-aMMFE --model-path <ckpt> --n-episodes 5
```

Substitute `FNV-mMMFE` for the multiplicative branch; every command takes the
same flags. `--per-seed-out` on either evaluator dumps per-seed profits, which
is what a paired Δ between two arms needs. Outputs follow the spec §8.4 tree:
`results/{scenario}/{run_name}/` holds a training run (terminal model,
`checkpoints/`, `screen.tsv`, `ppo_eval_*.tsv`, `probe/`, `interpret/`,
TensorBoard events), `results/{scenario}/benchmark/{method}/` the offsets and
`results/{scenario}/benchmark/benchmark_{method}_eval_*.tsv` their scores;
the interactive figures land in `results/{scenario}/figures/`, the committed
static ones in `figures/`.

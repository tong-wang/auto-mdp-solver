# clark_scarf — serial multi-echelon inventory (Clark & Scarf 1960)

Periodic-review ordering down a serial chain of stocking points, formalizing
Andrew J. Clark and Herbert Scarf, *Optimal Policies for a Multi-Echelon
Inventory Problem*, Management Science 6(4), 1960 — the paper that introduced
the echelon decomposition. Generated from `clark_scarf_schema.json` by
auto-mdp-solver.

**TL;DR**

- **Reinforcement learning gets within 0.70% of a verified optimum.** The best
  learned policy costs 989.23 against an exact 982.36 — a bar this case can
  state because the Clark–Scarf decomposition is solvable and was checked
  against a brute-force joint dynamic program to 0.000000. That is one
  training run, and the campaign has deliberately not yet adopted it.
- **Handing the agent the echelon coordinates buys nothing — for a policy
  class strong enough to do without them.** Given raw installation stock alone
  the best policy costs 989.62, against 989.23 with the transform supplied: a
  difference of 0.39 that the run-to-run spread covers several times over.
- **It rediscovers Clark & Scarf's rule at two of three echelons.** Read as a
  function, the top echelon stops ordering at 78 against the paper's 79 — but
  the retailer never stops at all, shipping a residual trickle even when it
  already holds four times its optimal level.

## The problem

A single product moves down a serial chain of stocking points. Level 1 is a
retailer facing Poisson demand; level N buys from an outside supplier with
unlimited stock. Every period each link dispatches simultaneously — N numbers,
one per link — and each dispatch is clipped to what its own source
installation is holding. Shipments take `leadtime` periods to arrive and
cannot be recalled or redirected once sent. Demand the retailer cannot meet is
backlogged, not lost.

The cost has two opposed parts. Holding is charged per period on echelon
stock, so inventory parked high in the chain is cheap and inventory pushed to
the retailer is dear; shortage is charged only at the retailer. Pushing stock
down early protects against shortage and pays holding for it; holding stock
back is cheap until a demand spike arrives that the lead time makes it too
late to answer. The objective is expected discounted cost over a finite
horizon.

What makes it hard is that the links are coupled in both directions and the
coupling is delayed. A shipment decided now at level 3 constrains what level 2
can dispatch two periods from now, and the retailer's shortage risk that far
out depends on decisions not yet made. The state is the full vector of
installation stocks plus everything in transit, so the natural dynamic program
runs over a state space exponential in chain length — the curse of
dimensionality that the paper exists to defeat.

Clark & Scarf's result is that this particular structure *decomposes*: written
in **echelon** coordinates — where an echelon's stock is everything at that
level or below, including goods in transit toward lower levels — the N-link
problem separates into N single-link problems solved in sequence, and the
optimal policy is order-up-to in those coordinates. That is what makes an
exact bar available here, and what the campaign's primary question asks about:
the transform is a known-optimal change of coordinates, so a learner that has
to find it is being asked to rediscover the paper's central insight. The
heuristics the field uses are echelon base-stock rules fitted to that form;
the structure-blind alternative is to run each installation on its own local
rule.

**Source.** Clark and Scarf (1960), read from the 2004 reprint, Management
Science 50(12S). The model is taken as stated; per-link shipping costs and
per-link lead times are carried in the formalization though this cell sets
them to zero and to a common value. The authoritative statement of the problem
is `clark_scarf_schema.json`; `clark_scarf.restatement.md` renders it in prose.

## Layout

Documents — where to read about the case:

| doc | what it holds |
|---|---|
| `clark_scarf.restatement.md` | the Phase-A restatement: the problem as the IR states it |
| `clark_scarf.scenarios.md` | the scenario set, its axes, and why each constant was chosen |
| `ESCALATION.md` | the campaign — design tree, changelogs, numbered findings (#E1–#E10) with verdicts |
| `INTERPRET.md` | the policy readback (spec §14): what the trained nets actually do, and the verdict on the `confirm` stance |
| `clark_scarf_readback_raw.svg` / `_echelon.svg` | the readback figures — echelon position after ordering against before, one panel per echelon |
| `PLAYBOOK.md` | the case-close digest (guide §10): five lever entries, one frame move, one modeling rule, and what did not transfer |
| `CLAUDE.md` | operating brief for an agent changing this folder, and the solver version its numbers were produced under |

Code — where things are implemented:

| module | role |
|---|---|
| `clark_scarf_schema.json` | **the IR — authoritative** for the problem definition |
| `clark_scarf_uncertainty.py` | the stochastic primitives: Poisson demand |
| `clark_scarf_scenarios.py` | the world layer — scenarios, sources, the `SCENARIOS` registry |
| `clark_scarf_mdp.py` | state and transition; `echelon_stock()` and `echelon_position()` are deliberately separate |
| `clark_scarf_gym.py` | the Gymnasium wrapper, its observation modes and action modes |
| `clark_scarf_eval_common.py` | the shared scoring harness: seed blocks, common random numbers, the vectorized rollout |
| `clark_scarf_ppo_train.py` / `clark_scarf_ppo_eval.py` | training, and scoring at the record protocol |
| `clark_scarf_select.py` | the post-hoc checkpoint screen (spec §9.7 middle layer) |
| `clark_scarf_configs.py` | the config-registry data half — ids, origins, cross-axis constraints |
| `clark_scarf_ordinal_head.py` | the hurdle-discretized-Gaussian action head |
| `clark_scarf_dgauss_head.py` | the two atomless discretized-Gaussian heads |
| `clark_scarf_beta_policy.py` | the Beta and Gamma heads for the continuous action modes |
| `clark_scarf_policy.py` | the deployable wrapper over the shipped artifact (spec §12) |
| `clark_scarf_policy_probe.py` | the spec §14 readback probe |
| `clark_scarf_plot_policy.py` | the policy figure |
| `clark_scarf_benchmark_dp.py` / `_dp_eval.py` | the exact Clark–Scarf decomposition |
| `clark_scarf_benchmark_echelon.py` / `_echelon_eval.py` | the echelon base-stock reference |
| `clark_scarf_benchmark_local.py` / `_local_eval.py` | the structure-blind local baseline |
| `clark_scarf_benchmark_random.py` / `_random_eval.py` | the random baseline |
| `clark_scarf_dp_exactness.py` | the brute-force joint DP that verifies the decomposition on a tiny instance |
| `clark_scarf_ir_adapter.py` | the differential adapter (portable-domain contract) |
| `clark_scarf_test.py` | the domain's own tests |

## Results

### Protocol

Every number below is quoted at **8192 seeds starting from seed 0**, common
random numbers **shared across arms** so any two rows of one board see the
same demand paths, with **`dp` as the reference bar**. The record eval is the
**deterministic argmax**: the decision is a shipment quantity with no
exploration role at deployment, and the benchmark it is scored against is
itself deterministic. A stochastic eval is a separate labelled figure and
never a record.

Definitions used below, each before its first use:

- **installation stock** — what one stocking point physically holds.
  **Echelon stock** at level *k* — everything at level *k* or below,
  *including goods in transit toward lower levels*. The two observation modes
  differ by exactly this transform and by nothing else; they carry the same
  information, which is what makes the comparison a question about
  representation rather than about knowledge.
- **cost** — expected discounted cost over the 50-period horizon, in the
  model's own cost units. Lower is better throughout, and every board is
  sorted ascending by it.
- **% of exact** — the row's cost divided by the `dp` row's cost. It is
  defined only on this board, because `dp` is exact only here.
- **seed floor** — the run-to-run spread of independently seeded training runs
  at one fixed configuration. It bounds what a difference between two single
  runs can mean, and on this cell it is about 6 cost units.

One caveat travels with every percentage on this page: the `dp` row is not
unconditionally exact. It truncates the Poisson probability mass at a high
quantile while the simulator does not, dropping less than 1e-9 of the mass.

### `1-comparative` — how does RL compare with the existing solutions?

The scenarios, and the role each plays in the study:

| scenario | config | what it is for |
|---|---|---|
| `n3_l2_p09` | 3 levels, lead time 2, shortage penalty 9.0, holding (2.0, 1.0, 0.5) down the chain, discount 0.95, horizon 50, Poisson(10) demand | the only cell in scope. Chosen because the decomposition is exactly solvable on it, so every learned number is legible as a percentage of a true optimum rather than against another policy |

**Each `(levels, lead time, shortage penalty)` combination is a separate
leaderboard, and scores never cross between them.** Levels and lead time change
the width of the observation and of the action, so no single policy spans two
of them; the shortage penalty changes the cost scale. This board reports the
one cell above. The same model also admits a continuous-demand rendering, on
which `dp` is a feasible reference rather than an exact one — a third frame,
and equally non-comparable.

#### `n3_l2_p09` — three levels, lead time 2

| arm | cost | SE | Δ vs `dp` | % of exact | significance | scope |
|---|---|---|---|---|---|---|
| `dp` | **982.36** | 1.189 | — | 100.00% | the bar | exact — the Clark–Scarf decomposition, checked against a brute-force joint DP to 0.000000 |
| `ppo` | 989.23 | 1.300 | +6.87 | 100.70% | z ≥ 3.9 | the best confirmed policy; **not adopted** — see below |
| `echelon_bs` | 1021.13 | 0.865 | +38.77 | 103.95% | z ≥ 26 | feasible — theory-informed reference, not a gate |
| `local` | 1120.92 | 0.762 | +138.56 | 114.10% | z ≥ 98 | feasible — structure-blind baseline |
| `random` | 37069.12 | 48.948 | +36086.76 | 3773.46% | z ≥ 737 | the floor that makes "a policy was learned" checkable |

Every row is one policy scored on the block above, so the `cost` and `SE`
columns hold the same estimand throughout: an expected cost and the standard
error of that expectation over 8192 seeds. A *cell mean over several training
seeds* is a different measurement and never appears as a row here — it is
discussed under the board, where it can be labelled.

Every entry in the `significance` column is that row against `dp`, the bar —
one basis for the whole column. Arms are scored on common random numbers, so
the paired standard error of a difference is smaller than the unpaired
combination used here; each z is therefore a **lower bound** on separation
rather than an estimate, which is why they are written `≥`. The comparison the
campaign cares about most does not appear in the column because it is not
against the bar: the best learned arm beats `echelon_bs`, the reference the
field actually uses, by 31.90 cost units. No arm is absent — all four
benchmarks are defined at every setting this cell uses.

**Shipped: nothing yet — and that is the honest state, not an oversight.**
The 989.23 on the board is one training run. A single run sits inside a
run-to-run spread of about 6 cost units, so the ranking that produced it is
not yet distinguishable from noise; it is also confounded, because the policy
class that produced it was tuned at a 25% larger step budget than the classes
it beat. Adopting it would be adopting a coin-flip.

What the campaign *has* adopted is a **configuration**, not an artifact:
`sc0/g3/a1/h0` — echelon observations, the hurdle policy class, hyperparameters
at their derived origin, 4M steps. Its evidence is stronger than anything on
the board and is a different measurement, so it is quoted here rather than as a
row: **a mean of 1001.37 over six independently seeded runs, ± 2.44 across
those seeds** (each run itself scored on the block above). That is the weaker
score and the stronger claim — six runs agreeing beats one run winning.

Designating an artifact means picking one run and standing behind it, and the
replication in `ESCALATION.md`'s plan is what makes that possible. Until then
this case ships a recipe, and `clark_scarf_policy.py` will wrap whichever
artifact that replication crowns.

How it was reached: `ESCALATION.md` — the design tree for the shape of the
search, and #E3, #E5, #E7 and #E10 for the findings that decided it.

### `2-structural` — the declared stances

Three stances are declared in the IR. Evidence and the full readback belong in
`INTERPRET.md`; this is the summary.

**1. `bypass` — the echelon coordinate transform (primary).** *Claim as
declared: a policy given only raw installation stock matches one given the
echelon transform, so handing the transform over should be unnecessary rather
than merely unhelpful.*

**Verdict: upheld for the stronger policy classes, refuted for the weakest —
so the claim holds conditionally, not generally.** Best raw 989.62 against
best echelon 989.23: a difference of 0.39, which the seed floor of about 6
covers many times over. A second policy class agrees, and differs in the
opposite direction (991.87 raw against 992.95 echelon), which is what a real
null looks like rather than a small true effect. A third class does not: it
costs 998.49 on raw and 993.73 on echelon, a 4.76 gain from being handed the
transform. The three differ only in how the shipment quantity is
parameterized — a flat distribution over quantities, a smooth one, and a
smooth one carrying a separate switch for shipping nothing; the third is the
one that gains. `ESCALATION.md` carries the head-by-head board. The useful form of the result is therefore the interaction — a
policy able to learn the transform internally is indifferent to receiving it,
and one unable to learn it is helped. Whether the coordinates matter is a
property of the policy class, not of the problem.

**2. `confirm` — echelon base-stock, the paper's optimal policy form
(secondary).** *Claim as declared: the raw-trained policy IS an echelon
base-stock rule — it recovers the aggregation and applies Clark & Scarf's own
critical numbers.*

**Verdict: confirmed at the two upper echelons, refuted at the retailer.**
Read as a function — every echelon position enumerated, the policy queried
deterministically, the upstream source unable to run out — the top echelon
stops ordering at 78 against the optimum's 79, and the middle one at 52 against
59. Those are order-up-to thresholds, which is what the claim asserts.

The retailer is not. Its order decays toward a floor of about 5 units and never
reaches zero: swept to four times its optimal level it is still shipping, and a
base-stock rule has a shutoff by definition. That deviation is invisible to any
readback taken from the policy's own trajectories, for the same reason it
survived training — a good policy rarely occupies deeply overstocked retailer
states, so nothing ever charges much for the floor.

The echelon aggregation is approximate rather than exact: redistributing a
fixed echelon total across installations moves the shipment by 9.15, 12.11 and
1.43 units. The policy implements the right rule in the right coordinate while
still keying partly on how that coordinate is split.

Evidence, both figures, and a control showing that handing the policy the
coordinates buys coordinate-faithfulness but *not* cleaner thresholds:
`INTERPRET.md`.

**3. `bypass` — the same transform on the continuous rendering (primary
there).** **Verdict: out of scope on this board.** That stance is asked on the
continuous-demand cells, where `dp` is a feasible reference rather than an
exact bar. This board renders the integer-lattice frame only, and a score
cannot cross between the two.

## Technical appendix

Run from `clark_scarf/`. Pin threads on every training run — torch
oversubscribes on a shared machine and the wall-clock cost is large. Together
these reproduce `results/` from an empty folder; `results/` is gitignored and
no number above depends on a file inside it surviving.

The `--config` flag takes a four-part address, one part per axis —
scenario / gym / architecture / hyper-parameters — naming a row of the config
registry rather than restating its contents. `python clark_scarf_configs.py`
prints the registry and resolves one address; `ESCALATION.md` says why each
exists.

```bash
# the config registry — origins, ids, and one resolved address
python clark_scarf_configs.py

# train
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python clark_scarf_ppo_train.py \
    -s n3_l2_p09 -o raw -a ship_discrete --policy-dist dgauss \
    --config sc0/g2/a2/h0

# screen the checkpoints on a block disjoint from the reporting one,
# then score the winner at the record protocol
python clark_scarf_select.py results/n3_l2_p09/{run}
python clark_scarf_ppo_eval.py -s n3_l2_p09 -o raw -a ship_discrete \
    --model-path {run}/checkpoints/{winner}.zip \
    --vecnorm-path {run}/checkpoints/{winner-vecnormalize}.pkl --n-seeds 8192

# the benchmarks, at the same protocol
python clark_scarf_benchmark_dp_eval.py -s n3_l2_p09 --n-seeds 8192
python clark_scarf_benchmark_echelon_eval.py -s n3_l2_p09 --n-seeds 8192
python clark_scarf_benchmark_local_eval.py -s n3_l2_p09 --n-seeds 8192
python clark_scarf_benchmark_random_eval.py -s n3_l2_p09 --n-seeds 8192

# the exactness fixture behind the bar's "verified optimal"
python clark_scarf_dp_exactness.py

# the deployable policy (spec §12) — replays against the raw MDP loop, no gym
python clark_scarf_policy.py --model-path {run}/n3_l2_p09_ppo_final.zip \
    --vecnorm-path {run}/vecnormalize_final.pkl -s n3_l2_p09 --episodes 512

# readback + figure
python clark_scarf_policy_probe.py --model-path {run}/n3_l2_p09_ppo_final.zip
python clark_scarf_plot_policy.py --model-path {run}/n3_l2_p09_ppo_final.zip
```

The gate commands are not here — they are in `CLAUDE.md`, because they are run
by whoever is changing the folder rather than reading it.

# `adi_flex` — plain-English restatement

Phase-A round-trip artifact for `adi_flex_schema.json`.
Source: **Wang & Toktay (2008), "Inventory Management with Advance Demand
Information and Flexible Delivery", *Management Science* 54(4), 716–732** —
specifically §4, the heterogeneous-customer model.

IR `mdp` fingerprint: `602b284da491` (`seed_scheme: v2`) — the token the
step-9 gate froze, re-recorded 2026-08-18 after it was found one move behind
the schema beside it.

---

## 1. The problem

You run a single stock point for **12 periods**. Customers order ahead, and
they differ in how long they are willing to wait: an order placed this period
may be due **this period**, **next period**, or **two periods out**. You can
see these orders the moment they are placed — that is the *advance demand
information*.

You may ship an order **any time up to its due date**, not only on it — that
is the *flexible delivery*. Shipping early is how you turn your own holding
cost into the customer's, so it is usually attractive.

The complication is that waiting orders and future orders compete for the same
stock. An order arriving *next* period might be due *next* period, while an
order you are already holding is not due for another two. So stock spent today
shipping a not-yet-due order may be stock you needed tomorrow for someone
urgent. The paper calls this **demand crossover**, and it is the whole
difficulty of §4: the firm must choose not only *how much to buy* but *how much
to withhold*.

Each period, in this exact order:

1. You see the state and **decide two numbers** — how much to order, and how
   many units to hold back.
2. The order arrives immediately (the supply lead time is zero here).
3. This period's new orders arrive: three independent Poisson draws, one per
   due date.
4. Stock is issued **earliest due date first**. Everything due now or next
   period is served as far as stock allows. Only then, and only after setting
   aside your hold-back, is anything shipped against orders due two periods
   out.

Demand that is **overdue and unmet is backlogged** and penalised every period
until it is served. Demand that is not yet due simply waits at no penalty.

## 2. Objective and mode stance

**Minimise total cost over the 12 periods**, where each period costs

```
  fixed_order_cost  if you ordered anything at all   (K = 100, charged once, regardless of size)
+ holding_cost      per unit of stock left on hand   (h = 1)
+ backorder_cost    per unit of overdue unmet demand (p = 9, recurring each period)
```

This is Eq. (5) with `c(z) = K·1{z>0}` — a fixed ordering cost and **no
variable cost**, exactly as §3 states. The discount factor is α = 1 in every
experiment the paper reports, so "discounted total cost" is a plain sum.

**Mode stance: there are no scenario modes.** Every attribute that varies is
numerical, so the scenario set is a grid of instances, not a set of categorical
branches — and all eight instances belong on **one leaderboard**. The paper's
only genuinely categorical attribute is whether delivery is flexible at all
(its `ADI` vs `ADI-F` comparison, the Δ metric), and that was explicitly
excluded from this pass.

## 3. Randomness classification

This is the classification the pipeline treats as authoritative, so it is
stated explicitly rather than left implied:

| source | realised | classification |
|---|---|---|
| `demand_now` — orders due immediately | **every period** | `uncertainty_sources`, source 0 |
| `demand_next` — orders due next period | **every period** | `uncertainty_sources`, source 1 |
| `demand_later` — orders due in two periods | **every period** | `uncertainty_sources`, source 2 |

All three are per-transition draws, so all three are uncertainty sources. Each
is Poisson with its own rate, and the three are independent.

**There is no episode-level randomness and no `ScenarioSampler`.** The demand
rates `(λ₀, λ₁, λ₂)` are *not* drawn — they are fixed constants that define
which instance you are in, chosen deliberately, not sampled. Likewise the
initial state is deterministic: `(inventory, due_now, due_next) = (0, 0, 0)`,
per §3.3. Nothing in the model is latent; the policy sees the entire state.

## 4. The scenario set

Everything is pinned at the paper's base setting — `L = 0`, `K = 100`, `h = 1`,
`p = 9`, `T = 2`, `N = 12`, `α = 1`, total demand rate 6 — and a **single axis
varies**: how demand splits across the three due dates. This is the paper's own
Experiments 0–7 ladder (§4.3), which runs from "no advance information at all"
to "the homogeneous model of §3".

| instance | (λ₀, λ₁, λ₂) | what it is | crossover? |
|---|---|---|---|
| `exp0` | (6, 0, 0) | no ADI — every order due on arrival; **the homogeneous model at T=0** | none |
| `exp1` | (5, 1, 0) | nothing is ever due two periods out | none |
| `exp2` | (4, 1, 1) | crossover live | yes |
| `exp3` | (3, 1, 2) | crossover live | yes |
| `exp4` | (2, 1, 3) | crossover live | yes |
| `exp5` | (1, 1, 4) | crossover live | yes |
| `exp6` | (0, 1, 5) | no urgent arrivals, so nothing to protect against | none |
| `exp7` | (0, 0, 6) | full ADI — **exactly the homogeneous model of §3 at T=2** | none |

Every one of these eight is backed by printed numbers in the paper — Figure 7's
costs, Table 2's protection levels, Table 3's `(s(v̂), S(v̂))` policies — which
is an oracle independent of the differential gate.

**Train/eval strategy: specialist per instance.** One policy trained and
evaluated on each mix, against per-instance benchmarks on identical seeds. The
four no-crossover instances double as correctness checks: the paper proves the
heuristics are already optimal there, so a learned policy that fails to match
them has a bug.

## 5. The frozen model

| | |
|---|---|
| **Horizon** | 12 periods, 0-based, ends `terminated` |
| **State** | `inventory` (net; negative = backlog), `due_now`, `due_next`, `period` |
| **Decisions** | `order_quantity` ∈ {0…60}, `hold_back` ∈ {0…20}, both integer, both committed at period start |
| **Uncertainty** | three independent Poisson streams, one per due date |
| **Objective** | minimise `order_fixed + holding + shortage` |
| **Observation** | `vec` (default): period, inventory, due_now, due_next · `vec_mip`: period, modified inventory position, due_next |
| **Action** | `joint` (default): one 2-vector driving both decisions |

## 6. Assumptions, with resolved source

| assumption | source |
|---|---|
| Order quantity and hold-back are **whole units** | `human_confirmed` — §3 p.719 states all quantities are integers |
| Bounds **0–60** and **0–20** | `human_confirmed` — covers every policy in Tables 2 and 3 with headroom |
| **No memory** needed; feed-forward policy | `human_confirmed` — at L=0 nothing is hidden, so the observation is Markov |
| Both decisions **committed before demand arrives** | `human_confirmed` — same information the paper's own PL heuristics use, whose σ is a constant |
| `region` recorded in **info**, not state | `human_confirmed` — pure diagnostic; no later transition reads it |
| **L = 0 is structural**, not an axis | derived from the chosen scenario set — there is no supply pipeline, so reaching the paper's L=1…4 needs a new Phase A |
| **T = 2 is structural** | derived — fixes the advance profile at exactly (due_now, due_next) |
| **N = 12** throughout (§4's setting; §3's figures use N = 30) | derived — the homogeneous corners will not match Figure 4's printed costs |
| Ordering cost is **fixed-only**, no variable cost | derived — §3, "only a fixed-order cost and no variable cost" |
| Costs charged at period end on `x(i+1)` | derived — `L(x_{i+1})` in Eq. (5) |
| α = 1, so cost is an undiscounted sum | derived — every reported experiment uses α = 1 |
| Orders due **after** the horizon are never penalised | derived — the DP recursion terminates with `C(N+1) = 0` |
| A Poisson rate of 0 **still consumes a draw** | derived — generated code must not short-circuit it, or streams desynchronise |
| Delivery-flexibility comparison (Δ) **out of scope** | decided — no exact-delivery dynamics branch exists in this IR |

---

## 7. Annotated sample trajectory

Produced by the restricted interpreter executing the IR's `dynamics`
expressions directly — **no generated code exists yet**:

```step7b
python -m mdp_ir.interpreter adi_flex/adi_flex_schema.json \
    --instance exp3 --decision order_quantity=5 --decision hold_back=3 \
    --episode-seed 235 --seed-salt 1
```

A deliberately weak policy: order 5 per period against a mean demand of 6, and
always hold back 3. Under-ordering is what forces the system through all five
fulfilment regimes of Eqs. (18)–(19). (`inventory`, `due_now`, `due_next` are
end-of-period; costs are assessed on the final inventory. The block is the
CLI's output pasted verbatim — the columns are the IR's own names, not
shortened ones, so `docs.restatement_current` can diff it against a live
re-run of the command above.)

```
adi_flex v0.4  episode_seed=235 instance=exp3
 t  order_quantity  hold_back  demand_now  demand_next  demand_later  region  action_period  inventory  due_now  due_next  order_fixed  holding  shortage   total   reward
 0            5.00       3.00           1            0             0       5              0       4.00        0         0       100.00     4.00      0.00  104.00  -104.00
 1            5.00       3.00           3            2             1       5              1       3.00        0         0       100.00     3.00      0.00  103.00  -103.00
 2            5.00       3.00           4            0             6       4              2       3.00        0      5.00       100.00     3.00      0.00  103.00  -103.00
 3            5.00       3.00           3            0             4       3              3       0.00        0         4       100.00     0.00      0.00  100.00  -100.00
 4            5.00       3.00           0            0             3       3              4       1.00        0         3       100.00     1.00      0.00  101.00  -101.00
 5            5.00       3.00           3            3             2       2              5          0     3.00         2       100.00     0.00      0.00  100.00  -100.00
 6            5.00       3.00           3            1             4       1              6      -1.00        3         4       100.00     0.00      9.00  109.00  -109.00
 7            5.00       3.00           2            1             0       1              7      -1.00        5         0       100.00     0.00      9.00  109.00  -109.00
 8            5.00       3.00           4            2             3       1              8      -5.00        2         3       100.00     0.00     45.00  145.00  -145.00
 9            5.00       3.00           6            2             1       1              9      -8.00        5         1       100.00     0.00     72.00  172.00  -172.00
10            5.00       3.00           1            1             1       1             10      -9.00        2         1       100.00     0.00     81.00  181.00  -181.00
11            5.00       3.00           4            0             0       1             11     -10.00        1         0       100.00     0.00     90.00  190.00  -190.00
episode total = 1517.00   reward total = -1517.00   periods = 12
```

**t = 0 — region 5, everything served.** Start `(0,0,0)`; 5 units available,
only 1 demanded, due now. It clears and 4 units are left on hand → holding 4.
Note the fixed cost of 100 is paid every period here because this policy always
orders — a real policy would batch.

**t = 1 — region 5 again.** Start `(4,0,0)`; 9 available. Demand `(3,2,1)`.
After serving the 3 due now, six remain; the hold-back of 3 is set aside and
the other 3 exactly cover the 2-due-next + 1-due-later, so both future classes
ship early. `due_next` ends at 0 and inventory ends at exactly the hold-back, 3.

**t = 2 — region 4, the protection actually binds.** Start `(3,0,0)`; 8 units
available. Demand `(4,0,6)`: a big far-future batch of 6. After covering the 4
due now, 4 remain; the hold-back of 3 is reserved first, so only **1** ships
early against the 6 — the remaining 5 roll to `due_next`, and inventory ends at
exactly the hold-back, **3**. This is the one regime where the allocation
decision changes anything.

**t = 3 — region 3, surplus below the hold-back.** Start `(3,0,5)`; 8 available
against 3 due now and 5 due next (aged from last period). After both are served
nothing is left over — the surplus (0) is below the hold-back of 3, so *nothing*
ships early and the 4 new far-future units roll to `due_next`. Inventory ends 0.

**t = 5 — region 2, stock runs out mid-queue.** Start `(1,0,3)`; 6 available
against 3 due now and 6 due next (3 carried + 3 new). The 3 due now are covered,
the remaining 3 go against the 6, leaving `due_now = 3` for next period and
inventory at 0. Nothing is overdue yet, so there is still no penalty.

**t = 6 onward — region 1, overdue and backlogged.** Start `(0,3,2)`; 5
available against 6 now due (3 carried + 3 new). One unit short, so inventory
goes to **−1** and the penalty starts: 9 × 1 = **9**. From here the policy never
catches up and the backlog compounds to −10 by the horizon.

**Conservation.** Over the episode 60 units were ordered and 71 demanded; at
the end 10 are backlogged and 1 is due now — 11 outstanding. So 60 were served,
exactly matching the 60 ordered.

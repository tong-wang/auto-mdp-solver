# adi_flex — plain-English restatement (Phase-A round-trip artifact)

**Source:** Wang, T. & Toktay, B.L. (2008), *Inventory Management with Advance
Demand Information and Flexible Delivery*, Management Science 54(4), 716–732.
**IR:** `adi_flex/adi_flex_schema.json` — mdp fingerprint `65cc43ca9f0d`,
model fingerprint `ddfcc21dec00`.

Fingerprint history: originally frozen at `f9de5170ba0b`; re-fingerprinted
2026-07-10 by a pure re-encoding — `horizon.T` now names the scenario constant
`N` directly via the schema's per-instance horizon extension, replacing the
T=30-cap + gym-early-termination device — giving `95e09e3630d3` under the
harness of the day. That token then moved to `2012a17bf377` with **no IR edit
at all**: v0.9.0 redefined the structural fingerprint as explicitly the
*rendering* hash and split the model layer out of it, so a pin bump alone
re-signed it. **F1** (2026-08-17, the v0.9.5 adoption) moved it once more, to
`552b4c3fb7f0`, by declaring `mdp.model` and naming the two rendering widths.
**F2** (same day) moved it to `f7c66b5eeecb` by taking the *remaining* literals
out of the rendering: every upper bound and every width now names a scenario
constant. `model_fingerprint` did **not** move across F2 — the theory layer was
untouched and only the rendering changed, which is the separation working.
**F3** (same day) moved it to `15ca390d1acc`, and `model_fingerprint` with it to `c4177a00d2c2`.
F3 removed enumeration wherever it had been standing in for a model: the three
demand **sources** collapsed into one process indexed by due-offset τ; the three
rate **constants** `lambda0/1/2` became the vector `lambda_seg` whose length is
T_dl + 1; the discount factor moved from a scenario constant (which the IR
referenced nowhere — it was decorative, and the rendering hardcoded α
independently) onto the objective's `discount_factor`, its schema home; and the
model layer stopped naming rendered slots at all.

**F5** (same day) declared the two state quantities the model layer had been
missing. The paper's state is `(x_i, W_i, V_i)`; only `x_i` was declared, so
`pipe` and `adv` were rendered with widths that named no theory quantity — which
is exactly what `model.boundary` had been reporting all along as *"unpaired caps
… named for no declared quantity, so coverage is unchecked"*. The same round
found that `state_variables.length` is **declarative metadata**: the width that
actually sizes the vector is the `initial_state` expression, and that was still
`zeros(5)` / `zeros(2)`. F2 had named the constant at the site that *documents*
the width and missed the site that *sets* it. Model fingerprint
`3431a2a05046` → `1d3c1eb91b62`, mdp → `a89da11f27d3`.

**F6** (same day) then took the width from the paper. `W_i = (w_i, …,
w_{i+L−1})` — the dimension **is** L, and eq. (1) is what makes L sufficient:
the pipeline shifts *first* and the new order is inserted last, where the
rendering had inserted at slot L before shifting and so needed one slot more.
The extra slot was always zero (checked at L=2 and L=4), and at L=0 the entire
vector was dead — five permanently-zero observation features on every instance
this campaign has run. `pipe_len` is deleted. The **model fingerprint did not
move**: the theory never said 5, only the rendering did. mdp → `350f6f0445ed`.

**F7** (same day, the decision correction) moved both: mdp → `ee47d0b8f675`
(via two intermediate tokens as the round landed), model → **`1ccea5442318`**
— and this time the model layer *should* move, because the theory it stated
was wrong: the second decision is §4's allocation, not §4.2's protection
level, which is the heuristics' parameter. The domain became two-step
(advance1/advance2), the gym two-phase under one masked Discrete, and the
homogeneous branch was verified **byte-identical** through the change (forced
maximal fill ≡ σ=0), so the exact DP and the AP bound both stand.

**F8** (same day) collapsed the demand draw: the D event had rendered T_dl + 1
enumerated `(tau = k; dk ~ demand.sample)` pairs, keyed on `tau` — a dynamics
local — because no family expressed *independent but not identically
distributed* components. This campaign proposed one (auto-mdp-solver#49,
accepted, shipped v0.9.9 as `independent`); adoption then hit a second defect,
a stage-level wrapper failing validation because the structural key `of` was
checked as an expression (#52, fixed in v0.9.10). At that pin it is **one
statement**, `d ~ demand.sample` over `{of: poisson, size: 'T_dl + 1', rate:
'lambda_seg'}`. Gone with it: the enumeration, the stage's `key_exprs`, and
the `tau` info field that existed only to smuggle the loop index into the seed
key. **`path_independence` returns to PASS** (`exogenous: ['d']`) — laws 6/9 →
7/9, recovering the law the τ key had cost. mdp → `b56a0fbf7a8d`; the model
layer never moved, because the theory always said one process indexed by
due-offset. This **re-seeds** — the third and last time — and cost nothing
only because the solution stage was reset and deliberately held for this pin.

**F9–F21** (2026-08-18, the line-by-line review) moved it five more times,
none of them theory: `f41cbfc3000a` (F9, the fill cascade expressed without a
fold), `2e7ce70c2838` (F11, the state collapsed to one class), `fade883ab044`
(F14, the order cap re-derived), `fd9cd0353099` (F16's first half),
`6d57eb261f89` (F16, the cap's *derivation* moved into the IR once
auto-mdp-solver#53 shipped, deleting 16 restatements of it and the `order_max`
constant with them), and **`b9a30bc7caee`** (F20, the `fills` info field).
`model ddfcc21dec00` has not moved since **F11**, which is the only model move
of this stretch: F10 unfroze the demand lead time and F11 corrected the
fulfillment — **only the due-now class is forced**, everything ahead is the
allocation — so `n_alloc` went from `T_dl - 1` to `T_dl`. Every number in this
document was re-measured after F20.

F1, F2, F5 and F6 were all trajectory-identical (MATCH ×24 at each, §7b
byte-identical): the AP bound still solves to 336.1867 and the `plsigma`
benchmark reproduces −340.452881 to the last digit. What F6 *does* move is the
observation: `vec` at L=0 goes 9 features → 4, so **both shipped models no
longer load** (`spaces must have the same shape: (9,) != (4,)`) and must be
retrained — which the F3 re-seed had already made necessary for their numbers.
F7 then changes heterogeneous trajectories *semantically* (the same nominal
action means a different thing), while the homogeneous branch and every
analytic value hold still.

**F3 is the one that moves numbers, and deliberately so**: the seed key gained τ, so every episode
draws a different demand path. Same distribution, different realization. The
differential still MATCHes ×24 — which is the point, since it re-checks
interpreter against simulator under the *new* key — but every **simulated**
number recorded before F3 is superseded rather than restated. Analytic values
are unaffected: the DP optimum and the AP bound are backward inductions over the
demand *distribution* and never touch a seed.

## The problem

Each period you manage a single uncapacitated stock point. Customers order in
advance: demand observed in period *i* carries a due date up to two periods
out, and **delivery is flexible** — any observed demand may be filled at any
time from observation through its due date (early shipment allowed, partial
fulfillment allowed). The sequence within a period is: review state → **place
a replenishment order** *z* (arrives after the supply lead time L, fixed cost
K if positive, no variable cost) → receive whatever the pipeline delivers →
demand for the period is observed → **fulfillment**: on-hand stock first
clears overdue backlog and demand due now — the *only* forced move, since
nothing can be delivered later than its due date — and then **the allocation
decision**: how much of the remaining surplus to spend pre-filling each class
still ahead of its due date, versus carrying it for next period's urgent
arrivals. Every class ahead is a decision, so the allocation is `T_dl` wide
(F11; it was `T_dl − 1` while the due-next class was mistakenly treated as
forced too). That choice is the paper's §4 allocation decision; its §4.2
protection level σ parameterizes the PL *heuristics* for it (allocate =
clip(surplus − σ, 0, outstanding)) and is not itself the control — F7's
correction, after the IR had carried σ as the decision since Phase A.
Unfilled due-now demand is backlogged at p per unit per period; unfilled
not-yet-due demand simply carries forward (no penalty); positive stock
(including withheld stock) costs h per unit per period. Objective:
**minimize expected total cost** (α = 1) over the horizon.

## Objective & mode stance

- Objective: minimize E[Σ (K·1{z>0} + h·inv⁺ + p·inv⁻)], evaluated on
  end-of-period net inventory; RL reward = −cost.
- **Mode stance (human-decided):** the two `customer_base` branches are
  **independent branches with separate leaderboards** — never compared
  head-to-head:
  - `homogeneous` (paper §3): all customers share one demand lead time T ∈
    {0,1,2}; FCFS/earliest-due-date fulfillment is provably optimal, so the
    allocation is not a live decision (maximal early fill forced) and **the
    only decision is the order quantity z**. Exact DP (state-dependent
    (s(V̂), S(V̂))) is the optimality reference.
  - `heterogeneous` (paper §4): demand is a vector over lead-time segments
    (λ₀, λ₁, λ₂), demand crossover exists, and **RL decides (z, allocation)
    jointly — at two different information sets**: z before the period's
    demand is observed, the allocation after (two-phase env, one MaskablePPO
    policy). The paper's AP relaxation bounds the problem and its
    PL(0)/PL(σ)/PL(Σ) heuristics are feasible policies over the allocation.

## The unified model (one simulator, both branches)

The homogeneous branch is encoded as one-hot demand-rate instances of the
heterogeneous T=2 model: λ = 6 placed on slot T, `alloc_enabled=false`
(maximal early fill forced ≡ fill-as-early-as-possible FCFS — the policy the
pre-F7 encoding wrote as σ=0). Verified: `het_exp7` (0,0,6) under a
maximal-fill policy reproduces `homog_L0_T2` trajectories **bit-exactly**
(now `test_the_homogeneous_branch_is_the_maximal_fill_policy`). The paper's
five-case evolution (eq. 18–19) is implemented as a branch-free min/max fill
cascade, algebraically equal to eqs. (2)–(4) under the maximal fill.

- **State** (fully observed; = the paper's (x, W, V)): net inventory `inv`
  (negative = overdue backlog only), supply pipeline `pipe` (**width L**, slot k
  arrives in k periods — so it is *empty* at L=0 and the state is (x, V), which
  is every instance this campaign has run), advance-demand profile `adv` =
  (due now, due next; width `T_dl` = 2), and `period`. All three of `inv`,
  `pipe` and `adv` are declared model quantities as of F5; the pipeline width
  became the paper's own L at F6, replacing a `pipe_len` = 5 register — see
  §Model envelope.
- **Decisions (human-confirmed integer; redefined at F7, rewidened at F11):**
  `order` z ≥ 0, a scalar, chosen at the review point BEFORE the period's
  demand; and `allocate`, a **vector of dimension `n_alloc` = `T_dl`** — one
  component per due-date class still ahead, chosen at fulfillment AFTER the
  demand is observed. Component k = units of that class shipped early this
  period; feasible in {0..min(remaining surplus, class outstanding)}, a
  state-dependent set the action mask enforces (the declared cap `alloc_max`
  only sizes the space and never binds a feasible choice — unlike its
  predecessor `protect_max`, which DID bind: with surplus above 20 the σ-agent
  could not choose to withhold everything). The dimension is the fulfillment
  rule's own: only the due-now class is forced, so **every** class ahead is a
  decision — two components at `T_dl` = 2, three at `T_dl` = 3. §4.2 counts one
  fewer protection level than that ("when T > 2, more than one protection level
  is needed") because a protection level is a *parameter of a heuristic for*
  the allocation, not the allocation itself. The components are taken
  sequentially against the residual (per-step masks cannot express the coupled
  sum constraint), an action encoding in the gym, not extra MDP periods — so a
  period is `1 + n_alloc` agent steps, **3** at `T_dl` = 2 and 4 at `T_dl` = 3.
  The order's action-space ceiling is a *derivation*, not a constant: since F16
  the IR carries `ceil(N·Σλ + 4·√(N·Σλ) + 1)` at `order.bounds` and resolves it
  per instance (107 at N = 12, 235 at N = 30). It bounds the horizon's total
  demand, which is the most an order could ever be useful for, so the claim
  holds for instances nobody has run — the `order_max` constant and its 15
  per-instance restatements are deleted.
- **Horizon:** N = 30 (homogeneous, Fig 3) / N = 12 (heterogeneous, §4.3);
  0-based periods; `horizon.T` names the scenario constant `N`, resolved per
  instance (schema per-instance-horizon extension, 2026-07-10).
- **Initial state (human-confirmed):** empty system (0, 0, 0) per Figure 4.

## Model envelope — what the theory admits, versus what this study renders

Declared machine-readably in the schema's `mdp.model` (12 quantities,
10 `out_of_scope` exclusions, 9 quantified dynamics rules, and a `notation` map
recording Wang & Toktay's symbols **without adopting them**), and gated by
`mdp_conformance`'s `model.boundary`. The separation is the point: three layers
hide in one `mdp` block — what the theory admits, which points this study
evaluates, and what the code renders — and until v0.9.0 only the middle one was
expressible, so the rendering silently passed itself off as the model.

What the model layer caught here, each recorded as a cited `narrowed`:

| rendering | model actually admits | narrowed by |
|---|---|---|
| `order` ≤ `ceil(N·Σλ + 4·√(N·Σλ) + 1)` | any non-negative integer — **supply is uncapacitated** | implementation: action-space cap, the horizon's total demand and so the most an order could ever be useful for (F14). The IR carries the **derivation** and resolves it per instance (F16, once auto-mdp-solver#53 shipped); it is 4σ rather than the observation sites' 6σ because a discrete action space is explored and slack is paid in sample efficiency |
| `allocate`: `n_alloc` components, each ≤ `alloc_max`, live range masked to min(surplus, outstanding) | a vector of dimension `T_dl`, component k at most its class's outstanding demand, jointly at most the surplus | `alloc_max` sizes the declared space only (never binds — the mask carries the real, state-dependent bound); narrowed *again* per instance to the forced maximal fill on the homogeneous branch, by `design:alloc_enabled=false` |
| `inv` — **no bound declared** | unbounded below — backlog is never lost | **not narrowed at all any more** (F15). The observation Box is the model's own domain: `(−∞, ∞)` where the theory says nothing, `[0, ∞)` where it says non-negative. A finite envelope is part of a trained artifact's compatibility key — SB3 compares `low`/`high` on load — so a *derived* one invalidates every saved model whenever the derivation is retuned, for magnitude checking nothing consumes |
| `adv` — **no bound declared** | bounded only below, at 0 | same (F15): declared `[0, ∞)`, which still fails `contains()` on the sign violations a mis-indexed slot produces |
| `pipe` slot ≤ the order cap | a slot holds one order | implementation: inherited from the order cap, and carrying the same expression |
| `pipe` width `L` | a pipeline exactly as long as the supply lead time — `W_i = (w_i, …, w_{i+L−1})` | **not narrowed at all any more**: the rendering is now the model's own dimension. Was a literal 5, then the cap `pipe_len` = 5, then L at F6 |
| `adv` width `T_dl` | a profile as wide as the demand lead time | **was a literal 2**; now names the paper's T |
| `period` ≤ `N` | a counter running to the horizon | resolved per instance, so no union-over-instances literal |
| `allocate` **width** `n_alloc` | one decision per class not yet due — `T_dl` of them | **was a literal 1**, then `T_dl − 1`; now names `n_alloc`, which equals `T_dl` since F11 corrected which classes are forced. The decision it replaced, `protect` of width `n_protect`, was removed at F7: a protection level is the heuristics' parameter, not the control |

**Every upper bound and every width in the rendering names a scenario
constant.** The only numeric literals left at a bound site are the lower bounds
of `0`, and those are *model facts* the theory states ("bounded only below, at
0"), not design choices. This matters beyond tidiness: §5.1 warns that a
union-over-instances literal "is correct for no instance and drags the
structural fingerprint on every widen" — `period`'s old `[0, 30]` was exactly
that, correct for the homogeneous instances and wrong for every heterogeneous
one (N = 12).

The two width rows are the defect §5.0 exists for: a literal at a width site,
for a quantity the theory declares, is how a *sweep maximum* (L ∈ 0..4) becomes
a *capacity limit* nobody chose. Both now name their constant.

**Closed at F6 — `pipe_len` was a cap, and a cap is not a model.** The width is
now the design value resolved *per instance*, as `clark_scarf` did at F7/F9
where the lead-time cap was **deleted** rather than paired. Two things had kept
it: the belief that the honest width was `L + 1` (an expression, which `length`
cannot take), and the cost of changing the shipped observation contract.

The first was wrong. The paper's `W_i = (w_i, …, w_{i+L−1})` is **L** wide, and
eq. (1) shows why that suffices — the pipeline shifts *first* and the order is
inserted last, where the rendering inserted at slot L *before* shifting and so
needed one slot more. `L` is a bare constant name, so `length` takes it as-is.
The `L + 1` never came from the theory; it came from the order of two events in
the rendering.

The second is real and was paid: `vec` at L = 0 goes 9 features → 4, so both
shipped models fail to load (`(9,) != (4,)`) and need retraining — which the F3
re-seed had already made necessary for their numbers, so F6 was the cheapest it
would ever be. `test_the_pipeline_width_is_the_supply_lead_time` replaces the
interim cap guard, pinning the width to L at both the declaration and the
simulator; `test_a_sampler_observation_covers_every_member_leadtime` covers what
the cap was really protecting, since a Gym Box is fixed at construction while a
sampler's members now differ in width (`homog_grid` spans L ∈ 0..4). **That
padding is a gym concern and lives in `adi_flex_gym.pipe_obs_slots`** — the cap
did not vanish so much as move to the layer where it is honestly a rendering
device, out of the state where it was masquerading as model structure.

The model layer may not argue from a benchmark or from tractability, and may
not name rendered slots — both are gated, and both were written to.

## Randomness classification

- **Per-transition → `uncertainty_sources`:** exactly one kind, and exactly
  **one source**: `demand`, a single process yielding the paper's whole vector
  `D_i` in ONE draw — component τ ∈ 0..`T_dl` Poisson at `lambda_seg[τ]`,
  independent across components but **not identically distributed**. The seed
  key is the plain period key `[stream_id, period, episode_seed, seed_salt]`
  and the components come in index order from that one rng, so realized demand
  is decision-path independent. Poisson(0) is degenerate-zero for one-hot
  (homogeneous) instances and still consumes its draw, so trajectories stay
  comparable across instances. The key has moved three times and this is the
  last: per-offset streams, then a τ-keyed stage, then no index at all — a
  single vector draw needs none.

  It was three enumerated sources (`demand0`/`demand1`/`demand2`, stream_ids
  1–3) until **F3**. That shape stated the *rendering* as if it were the model:
  the paper has one demand process whose customers differ by due date, and a
  heterogeneous customer base is a **mix over offsets**, not three demands. The
  enumeration then propagated upward — it forced the model layer to declare
  `demand0/1/2` as separate theory quantities, which is precisely what
  `ModelStatement` forbids ("speaks in index variables and quantified rules,
  never in rendered names or enumerated slots"). `mab` is the pattern: k arms,
  **one** `payout` slot keyed by `int(arm)`.

  The collapse left the *draw* enumerated even so — one statement per offset,
  the stage keyed on τ — because no family expressed independent-but-not-
  identically-distributed components, and it cost a law: `path_independence`
  went PASS → SKIP, since keying on a dynamics local stops the checker proving
  a property that holds. It also forced a τ **info field**, declared purely so
  the rate selection could see the index (`key_exprs` may name a dynamics
  local; a source's `settings` are validated against value names only).

  **All of that is gone at F8.** The campaign proposed the missing family
  (auto-mdp-solver#49 → `independent`, v0.9.9), hit a validator defect
  adopting it (#52, fixed v0.9.10), and now declares one draw:
  `{of: poisson, size: 'T_dl + 1', rate: 'lambda_seg'}`, components in index
  order from one rng at the plain period key. The τ field, the `key_exprs`,
  and the enumeration all went with it, and **`path_independence` is back to
  PASS** (`exogenous: ['d']`).
- **Once-per-episode:** nothing intrinsic (deterministic initial state,
  deterministic lead time L). The only episode-level draw is **problem-
  instance selection** for the generalist strategy → `AdiFlexScenarioSampler`
  (scenario stream 0) in Phase B, *not* a generator.
- Everything else (L, K, h, p, N, the rate vector `lambda_seg`, and the
  demand-lead-time window `T_dl` with the allocation width `n_alloc` derived
  from it) is an axis-tagged scenario constant; `alloc_enabled` is the one
  categorical scenario-mode flag. **`alloc_max` is the only untagged constant**
  — an action-space size, not a design value, and deliberately not an axis.
  Three things that used to sit beside it are gone. The observation envelopes
  `inv_bound` / `inv_min` / `adv_bound` went at F12 as literals frozen from a
  study (one was violated by reachable states at 7×) and their *replacement*
  went at F15: the Box is the model's declared domain, so there is no envelope
  constant to maintain and no derivation to retune. `order_max` went at F16,
  when auto-mdp-solver#53 made a bounds entry able to hold the expression that
  defines it — the constant and its 15 per-instance overrides collapsed to one
  derivation. `pipe_len` went at F6, the pipeline's width being the axis-tagged
  `L` itself. The discount factor is *not* a scenario constant: β belongs to
  the objective (`discount_factor`).
- **10 constants, 29 instances**, and every upper bound and width in the
  rendering either names a constant or states a derivation over them.

## Scenario set (human-confirmed)

- **Homogeneous branch — 15 instances** `homog_L{l}_T{t}`, (L,T) ∈
  {0..4}×{0..2} (Figure 4's grid), shared λ=6, K=100, h=1, p=9, N=30, α=1.
  Base: `homog_L0_T2` (Figure 3's setting).
- **Heterogeneous branch — 8 instances** `het_exp0..7`, demand mixes
  (6,0,0), (5,1,0), (4,1,1), (3,1,2), (2,1,3), (1,1,4), (0,1,5), (0,0,6) at
  L=0, T_dl=2, K=100, h=1, p=9, N=12, α=1 (Table 2's Experiments 1–6 are
  `het_exp1..6`). Base: `het_exp4` = (2,1,3), where the paper's heuristic gaps
  peak. The two ends are **validation endpoints, never training targets**:
  `het_exp0` (no ADI → classic (s,S)) and `het_exp7` (≡ homogeneous).
- **Third board — 6 instances** `het3_exp1..6` at **`T_dl` = 3, L = 1**, mixes
  (3,1,1,1), (2,1,1,2), (1,1,1,3), (0,1,1,4), (0,0,1,5), (0,0,0,6), otherwise
  the heterogeneous cost set (K=100, h=1, p=9, N=12). Added at **F10**, when
  the rendering stopped restating `T_dl` = 2 and started honouring the declared
  width — these are the instances that make `n_alloc`, the profile width and
  the MIP tail vary rather than merely being named. They have **no reference
  bracket**: the exact DP and the AP relaxation are both L=0/T=2 derivations
  and the PL σ-ladder is §4.2's T=2 construction, so nothing yet bounds this
  board.
- **Fourth board — 1 instance** `homog_L0_Tdl1` at **`T_dl` = 1, L = 0**,
  λ = (0, 6) so all demand is due one period out, otherwise the homogeneous
  cost set (N = 30). Added at **F26**. `T <= L + 1` makes it the paper's
  **Case 1**, where Prop. 1 collapses the state to the scalar MIP: `vec_mip`
  renders `(u, time_to_go)` with an EMPTY `V~` tail against `vec`'s
  `(inv, adv0, time_to_go)`. Two free state numbers against one, which is the
  sharpest form of RQ4 this domain admits. Exact DP **828.9100**, floor
  `myopic` 2240.26.
- **Fifth board — 1 scenario** `homog_L0_Tdl0` at **`T_dl` = 0, L = 0**,
  λ = (6,) so there is no advance demand at all. Added at **F27**; the paper's
  T = 0, which reduces this domain to a plain periodic-review (s,S) system.
  `adv` is empty and `n_alloc` is 0, so `mip = inv` exactly and `vec` and
  `vec_mip` render the SAME observation — it is RQ4's **null control**. Exact
  DP **1026.7621**, floor `myopic` 2072.09. It is a scenario but **not an IR
  instance**: `Decision.dim` must resolve >= 1 upstream and the allocation has
  zero components here, so it is runnable through the gym and NOT
  differentially verified (F27).
- **Optimal cost is monotone in the demand lead time across the three
  L = 0 homogeneous boards** — 1026.76 (T=0) > 828.91 (T=1) > 685.24 (T=2) —
  a consistency check on the whole model that no single board could give.
- **30 IR instances in total** (15 + 8 + 6 + 1), which is what the differential
  sweeps; 31 scenarios counting the T_dl = 0 board.
- **Axis-registered only:** the paper's 540-cell sweep (Exps 2–5 × L ∈ {0..4}
  × K ∈ {50,100,200} × h ∈ {1,3,5} × p ∈ {9,19,29}); any cell is a one-line
  instance addition.
- **Samplers (Phase B):** `homog_grid` uniform over the 15; `het_mix` uniform
  over `het_exp1..6`.

## Train/eval strategy (human-decided)

**Both strategies per branch from the start:** a specialist trained on the
base instance and evaluated there, and a generalist trained on the branch's
sampler (instance constants in the observation) and evaluated per instance.
Phase-B Stage 0 re-confirms which target(s) each build pass actually runs.

Baselines, with the §9.9 role each sits at relative to the optimum — declared
in the IR's root `benchmarks` block, one entry per `adi_flex_benchmark_*.py`:

| arm | role | branch | what it is |
|---|---|---|---|
| `dp` | `exact` | homogeneous | the recursion where the relaxation is tight, so its value *is* the optimum |
| `ap` | `relaxed` | heterogeneous | the same recursion read as the §4.1 AP relaxation — a **lower bound**, not attainable |
| `rule` | `feasible` | both | `myopic` as the floor arm, plus the paper's §4.2 protection-level heuristics PL(0)/PL(σ)/PL(Σ) |

A uniform-random policy is deliberately **not** among them: it draws over the
action space, so its value tracks the order cap rather than the problem, and it
moved twice in one day for reasons that had nothing to do with policy quality.
It stays runnable as a diagnostic (`--policy random`) and is reported nowhere.

`relaxed` + `feasible` **brackets** the heterogeneous optimum, which is what
lets that branch quote an optimality gap without an exact solver — on the
`het_exp*` board only; the `het3_*` board has no such bracket yet. The roles are
enforced, not decorative: `mdp_gates --ir` refuses `--baseline` on `dp`/`ap`
(exit 2) and flags any `feasible` arm that beats one of them as a defect in the
eval, the bound or the simulator — both verified against this folder.

## Key assumptions and their sources

| Assumption | Source |
|---|---|
| Branches are independent (separate leaderboards) | human_decided |
| Het branch decision = joint (z, allocation), at split information sets (z pre-demand, allocation post-demand) | human_decided (F7 redefinition 2026-08-17; was joint (z, σ)) |
| Integer decisions; the order's action cap is a **derivation** from the dynamics, never a measurement of solved runs — `ceil(N·Σλ + 4·√(N·Σλ) + 1)`, 107 at N = 12; allocation masked to min(surplus, outstanding), space sized by alloc_max = 20 | human_confirmed (caps re-derived at F14, moved into the IR at F16) |
| α = 1, axis-tagged | human_confirmed |
| N = 30 homog / 12 het, axis-tagged | human_confirmed |
| Empty initial state (0,0,0) | human_confirmed |
| Scenario set & bases as above | human_confirmed |
| Both train/eval strategies per branch | human_decided |
| the demand vector `d` placed in info, not state (was three scalars d0/d1/d2 until F7 consolidated the interface) | human_confirmed |
| `fills`, the realized per-class allocation, reported beside `allocate`, the request — they differ whenever the shared surplus binds, and systematically on the homogeneous branch where the request is inert | F20 |
| requires_memory = false (fully observed) | human_confirmed |
| Backorder penalty on overdue demand only; unmet future demand unpenalized | derived from paper eq. 18 case 1 |
| Costs at END_OF_PERIOD; no variable order cost | derived from paper §3 |
| The allocation is a vector of dimension `n_alloc` = `T_dl` — one component per class still ahead of its due date, since only the due-now class is forced; at `T_dl` = 2 that is two components. σ is the PL heuristics' parameter, not the decision | paper §4 (the allocation decision) + §4.2 (incl. its T=3 construction); F7, rewidened at F11 |
| Observation Box = the model's declared domain, not a derived envelope; the action cap is the opposite case and is derived and tight | F15 |
| `vec_mip` observes the paper's sufficient statistic — the MIP of eq. (7) plus `V~` = `adv[L+1:]` of eq. (11), empty when `T_dl ≤ L+1` — not the MIP plus the whole profile | paper §3.1–3.2; F18 |

## Annotated sample trajectory

```step7b
python -m mdp_ir.interpreter adi_flex/adi_flex_schema.json --instance het_exp4 --decision order=6 --decision allocate=2 --episode-seed 8
```

— heterogeneous base
instance (λ = 2,1,3; L=0; K=100, h=1, p=9; N=12), constant policy z=6, and
a=2 **broadcast to both allocation components** (spec Phase-A step 7b; one
value fills the resolved width since auto-mdp-solver#39 shipped in v0.9.5).

Re-rendered six times, each for a stated reason: **F3** (the τ key re-seeded
the domain), **F6** (`pipe` reads `[]` — at L = 0 the paper's supply pipeline
is an empty vector, where a fixed 5-slot register used to sit), **F7** (the
second decision changed meaning: the nominal 2 was σ=2, "fill everything above
a reserve of 2", and is now a=2, "ship at most 2 early"), **F8** (the vector
draw re-seeded it a final time), and **F21** — which is the one worth reading
twice. F21 re-rendered it only to pick up F20's new `fills` column and got a
**different trajectory**: `allocate` renders `[2.00,2.00]` where the recorded
one showed a scalar `2.00`, and the episode total is `1375.00` against a
recorded `1362.00`. The broadcast width is `n_alloc`, and **F11 changed it from
`T_dl − 1` to `T_dl`** when it corrected which classes are forced. So this
document had been showing a one-component allocation for eleven findings, and
nothing caught it — the differential compares interpreter against simulator,
and both had moved together. Re-render whenever the IR moves; a changed COLUMN
SET is a prompt to re-read the prose, not just to paste a new table.

The seed stays 8, chosen the way seed 5 was chosen before it: it exercises
every mechanism this domain exists to model — early delivery, the decision
*withholding* stock it could have shipped, the shared surplus binding one
component against another, and a run of backlogged periods. The
`surplus`/`outstanding` columns are the allocation's feasible-set coordinates
**for the component being asked** (the nearest class), and what the gym's
action mask is computed from.

```
adi_flex v0.4  episode_seed=8 instance=het_exp4
 t  order     allocate         d  received        fills  early_fill  surplus  outstanding    inv  pipe          adv  order_fixed  holding  backorder   total   reward
 0   6.00  [2.00,2.00]   [2,0,4]      6.00     [0,2.00]        2.00     4.00            0   2.00    []     [0,2.00]       100.00     2.00       0.00  102.00  -102.00
 1   6.00  [2.00,2.00]   [2,1,2]      6.00  [2.00,2.00]        4.00     6.00         3.00   2.00    []  [1.00,0.00]       100.00     2.00       0.00  102.00  -102.00
 2   6.00  [2.00,2.00]   [0,0,3]      6.00  [0.00,2.00]        2.00     7.00         0.00   5.00    []  [0.00,1.00]       100.00     5.00       0.00  105.00  -105.00
 3   6.00  [2.00,2.00]   [4,2,2]      6.00  [2.00,2.00]        4.00     7.00         3.00   3.00    []  [1.00,0.00]       100.00     3.00       0.00  103.00  -103.00
 4   6.00  [2.00,2.00]   [1,0,5]      6.00  [0.00,2.00]        2.00     7.00         0.00   5.00    []  [0.00,3.00]       100.00     5.00       0.00  105.00  -105.00
 5   6.00  [2.00,2.00]   [3,0,4]      6.00  [2.00,2.00]        4.00     8.00         3.00   4.00    []  [1.00,2.00]       100.00     4.00       0.00  104.00  -104.00
 6   6.00  [2.00,2.00]   [5,1,5]      6.00  [2.00,2.00]        4.00     4.00         3.00   0.00    []  [1.00,3.00]       100.00     0.00       0.00  100.00  -100.00
 7   6.00  [2.00,2.00]   [0,1,7]      6.00  [2.00,2.00]        4.00     5.00         4.00   1.00    []  [2.00,5.00]       100.00     1.00       0.00  101.00  -101.00
 8   6.00  [2.00,2.00]   [2,3,5]      6.00  [2.00,1.00]        3.00     3.00         8.00   0.00    []  [6.00,4.00]       100.00     0.00       0.00  100.00  -100.00
 9   6.00  [2.00,2.00]   [2,4,3]      6.00        [0,0]           0        0         8.00  -2.00    []     [8.00,3]       100.00     0.00      18.00  118.00  -118.00
10   6.00  [2.00,2.00]  [4,1,10]      6.00        [0,0]           0        0            4  -8.00    []       [4,10]       100.00     0.00      72.00  172.00  -172.00
11   6.00  [2.00,2.00]   [1,0,5]      6.00        [0,0]           0        0           10  -7.00    []       [10,5]       100.00     0.00      63.00  163.00  -163.00
episode total = 1375.00   reward total = -1375.00   periods = 12
```

Annotations (hand-checked against eq. 18–19, cascade by cascade):

- **t=0:** start empty; order 6 arrives immediately (L=0), so inv=6. Demand
  (2,0,4): the forced fill serves the 2 due now — the *only* forced move —
  leaving `surplus=4`. Nothing is due next period, so the nearest class shows
  `outstanding=0` and its component fills 0 however much is asked; the far
  class holds 4 and takes the requested 2. `fills=[0,2]`, so the allocation
  ships **2 of the 4 it could have** — the decision visibly declining the
  maximal fill, which is the thing no σ-as-decision trajectory could display.
  End stock 2, `adv=[0,2]`, unpenalized because not yet due.
- **t=1:** both components now bite. `surplus=6` against a nearest class of 3
  and a far class of 2; the request `[2,2]` is feasible in full, `fills=[2,2]`,
  `early_fill=4`. This row is what the pre-F11 rendering could not express at
  all: with one component, one of these two classes was not a decision.
- **t=8:** the shared budget binds. `surplus=3` with the nearest class owing 8
  and the far class 5 — the first component takes its 2, leaving 1, so the
  second is clipped from 2 to 1: `fills=[2,1]`. The components are *coupled*
  through the surplus, which is why they are asked sequentially against the
  residual and why a per-step mask, not a per-component bound, carries the
  feasible set. An unmasked policy would be clipped identically inside
  `advance2`, the transition being total.
- **t=9 onward:** the system tips. `surplus=0` from here, so `early_fill=0`
  regardless of the decision, and t=9–11 are backlogged: at t=10, `adv=[4,10]`
  against inv −8 charges `backorder=72`. The constant z=6 cannot keep up with
  a mean-6 arrival stream once the profile has built — exactly the trade-off
  an (s,S)-style policy exists to manage.
- **`allocate` vs `fills`** is the F20 separation: the request as driven, and
  what it actually delivered. They agree at t=1–7 and part at t=0 (a class with
  nothing outstanding) and t=8 (the surplus binding). On the homogeneous branch
  they part *systematically* — `allocate` is inert there and the maximal fill is
  forced — which is why the record carries both.
- **`d`** is the period's realized demand vector — one column, one draw since
  F8. The `tau` column is **gone**: it was an info field declared only so the
  stage's `key_exprs` could name a loop index, and the vector draw needs
  neither.
- `order_fixed=100` every period because the constant policy always orders; an
  (s,S)-style policy would order intermittently — that trade-off is exactly what
  the agent must learn, and it is a finding about this policy, not the model.

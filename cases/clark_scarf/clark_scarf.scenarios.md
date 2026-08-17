# clark_scarf — scenario set

Clark & Scarf (1960) is a pure theory paper: it proves the decomposition and runs
no numerical study. There is **no experiment design to extract** — this set is
*proposed*, and every value in it is the human's call.

Model: serial N-echelon chain (§2–§3). Level 1 is the retailer facing demand;
level N buys from an outside supplier with unlimited stock. Demand at level 1
only, backlogged. Costs charged on **echelon** stock (Assumption 3). Objective:
minimize expected discounted total cost, β = 0.95.

---

## Two independent structural axes

`n_echelons` and `leadtime` are **not** interchangeable, and an earlier draft of
this file wrongly claimed chain length subsumed lead time. It does not:

- an extra **installation** is a *discretionary buffer* — units sit there and the
  policy re-decides each period whether to move them on;
- an extra period of **lead time** is *committed flow* — once dispatched, a unit
  arrives on schedule and no decision can touch it.

A 2-level chain with a 2-period lead and a 3-level chain with a free middle
impose the same delay but offer completely different control. Both axes are swept.

## The model at chain length N and lead time L

**State.** `stock[k]` = physical on-hand at installation *k+1*; `arriving[k]` =
the in-flight batch landing at the next arrival event; `pending[k]` = the batch
landing one event later (non-zero only at L = 2). Slots at index ≥ `n_echelons`
are inert padding — never read by any cost or constraint, and never shown to a
policy.

**Action — N numbers**, one per link, as §3 counts them:

```
supplier ──q_N──▶ [N] ── … ──▶ [2] ──q_1──▶ [1] ──▶ demand
```

Each shipment is clipped to what its source installation holds; the top link
draws from the outside supplier and is unclipped.

**Within a period:** `A` (arrivals land) → `S` (all links dispatch
simultaneously) → `D` (demand hits the retailer, backlogging if short).

**Cost.** Echelon holding on end-of-period **echelon stock**, plus the retailer's
shortage penalty:

| term | amount |
|---|---|
| holding | `Σ_k H_k · stock[k]` (retailer term on `stock[0]⁺`) `+ Σ_k H_{k+1} · in-flight[k]` |
| shortage | `p_short · max(0, −stock[0])` |

`H_k = Σ_{j≥k} h_j` is the cumulative echelon rate, carried directly as the
`h_install` constant; charging position *k* that rate is algebraically identical
to charging each echelon `h_k` on its echelon stock `x_k`.

**The in-flight terms are offset one level**, per Assumption 3: a level's costs
cover stock at that level plus stock at a *lower* level **or in transit to a
lower level**. *Lower* is the operative word — stock in transit *to* level *j*
is heading to *j*, not below it, so it sits in echelon *j+1* and is charged its
**source's** rate until it lands. A unit in flight to the retailer therefore
costs the same as one sitting at the depot; shipping downstream is what *starts*
the higher charge, and it starts on arrival, not on dispatch.

`h_install` being 0 above the chain then does double duty: padding is never
charged, and stock in transit to the *top* level is charged nothing either —
correct, since it is on order rather than in the system. Neither needs a special
case.

## Two observation modes — the campaign's headline experiment

Echelon stock is an artifact of the paper's analysis, not something a warehouse
manager sees, so the gym exposes both views:

| mode | what the agent sees |
|---|---|
| `raw` | physical stock at each installation, plus what is in flight toward it |
| `echelon` | the running sums `x_j` — echelon inventory position, the paper's coordinates |

The map between them is an invertible running sum, so the two carry **identical
information**. This tests **representation, not information**: not whether the
agent can uncover something hidden, but whether the echelon coordinate system must
be handed over or whether the network builds it itself. An MLP can express a
cumulative sum (a lower-triangular matrix of ones), so a `raw` shortfall would be
an optimization finding, not an impossibility.

At L = 1 the aggregate is a running sum over installations. At **L = 2 it is a
running sum over installations *and* their pipelines** — a markedly less trivial
feature to construct, so the comparison gets more informative exactly where the
problem gets harder.

Action encoding is held **fixed** at `ship` (a quantity per link, feasibility a
per-component box, raw action 0 = "ship nothing" — a live low-value action, which
is where SB3's Gaussian initializes). The echelon-coordinate alternative is held
in reserve as a Stage-4 escalation.

## Constants held fixed in every cell

| constant | value | why |
|---|---|---|
| `horizon_T` | 50 periods | with β = 0.95 the effective horizon is ~20 periods, so 30 leaves the stationary middle crowded out by ramp-up and wind-down; 50 gives a genuine steady stretch for the base-stock readback to recover `x̄` from |
| `demand_mean` | 10.0 (Poisson) | set at 10 rather than 5 so integrality is fine-grained relative to the decision scale (sd ≈ 3.2); at 5 a single unit is a large relative move and would contaminate the base-stock readback |
| `c_purchase`, `c_ship` | 0.0 | buying and shipping are free — the experiment isolates the holding-vs-shortage trade-off, and linear purchase cost largely telescopes under backlogging (`inv_single` sets `c = 0.0` for the same reason) |
| `h_retail` | 1.0 | echelon holding increment at level 1 |
| `h_echelon` | 0.5 | echelon holding increment at every level above |
| `K_setup` | 0.0 | no fixed order charge ⇒ the optimum is a pure **base-stock** rule, one critical number per echelon, not `(S,s)` |
| `ship_max` | 200.0 | caps the action scale at 20 × mean demand; not a physical limit |
| initial state | one period of cover at **every stage** | `stock[k] = arriving[k] = demand_mean`, and `pending[k] = demand_mean` at L = 2. Stated in raw coordinates; in echelon terms `x_k = k · demand_mean`. Not the optimum, so it hands the agent nothing — it only starts the chain in flow, rather than after an N-period forced stockout |
| shipments | integer | Poisson demand keeps every stock integral, which is what keeps the DP reference exact rather than gridded |

With purchase and shipping free, the only surviving forces are the **holding-cost
gradient up the chain** and the **shortage penalty at the bottom**: a unit resting
at the retailer costs `h_retail + h_echelon·(N−1)` per period, one at the top only
`h_echelon`. Holding upstream stays cheap, holding downstream stays expensive, and
free shipping merely removes the friction that obscured that gradient. This makes
the Stage-5 prediction crisper — recovered base-stock levels should track the
**newsvendor critical fractile** of lead-time demand rather than mixing in purchase
economics.

There is no salvage value and holding is still charged, so stock left at the
horizon end is pure waste: the optimal critical numbers remain **time-varying**, as
the paper's `x̄_n` subscript indicates. The readback compares against `x̄_n` period
by period, not against one number per echelon.

## Swept axes and the full cell set

| axis | values |
|---|---|
| `n_echelons` | 2, 3, 4 |
| `leadtime` | 1, 2 |
| `p_short` | 4, 9, 19 → critical ratios 0.80, 0.90, 0.95 |

3 × 2 × 3 = **18 cells**. Each is its own trained policy — chain length and lead
time both change the observation and action width, so no policy spans them.
**Stage 0 picks one cell**; the rest are later passes reusing everything.

Proposed canonical cell: **`n3_l2_p09`** — mid critical ratio, a chain long enough
that the decomposition's *recursion* runs at least once past the base case, and
`L = 2` so in-flight stock is present (the paper's own worked lead time, §2).

Two falsifiable predictions give the readback more than a single-point comparison:

- base-stock levels must rise monotonically along `p_short`;
- the RL-to-optimal gap should widen along `n_echelons` and along `leadtime`,
  since coordination deepens while the exact benchmark does not degrade.

## Instances registered in the IR

The schema registers a structural covering set — every `(n_echelons, leadtime)`
combination, so the differential exercises each — plus cost variation and the test
fixture:

`n2_l1_p09` · `n2_l2_p09` · `n3_l1_p09` · `n3_l2_p09` (the base constants) ·
`n3_l2_p04` · `n3_l2_p19` · `n4_l1_p09` · `n4_l2_p09` · `verify_tiny`

## Test-only fixture

| scenario | settings | purpose |
|---|---|---|
| `verify_tiny` | N = 2, L = 1, μ = 2, horizon 4, `ship_max` 40 | small enough to brute-force the **full joint** DP, so the decomposition's claim of exact optimality is *checked* rather than assumed |

This is the second-implementation gate behind the DP row: the shipped benchmark is
the Clark–Scarf decomposition, and this instance is the independent evidence that
it really is the optimum.

## Scenario modes

**None.** Every attribute the paper varies — `K`, `c`, `c₁`, `h`, `h̃`, `p`, the
lead times, the horizon, the demand rate, the chain length — is numerical, so all
of it classifies as swept axes. The one categorical candidate, the demand family,
was settled to a single branch (Poisson); the IR keeps a candidate pool at that
slot, so a second family is a one-line addition later against the same frozen
model. One leaderboard; the head-to-head-versus-separate-branch question does not
arise.

## Randomness classification

- **Per-period** — retailer demand, independent across periods. The only
  stochastic primitive.
- **Drawn once per episode by nature** — **none**. Costs and the demand
  distribution are fixed across the horizon; nothing is redrawn per episode.
- **Experimenter's training range** — the axis table above; design layer, not
  part of the model.

## Known approximation, documented

Poisson has unbounded support. The exact DP truncates the pmf at a high quantile
and renormalizes; the simulator does not. The DP is therefore optimal for a
distribution differing from the simulated one by mass below `1e-9`. Recorded so
the DP row is never read as unconditionally exact.

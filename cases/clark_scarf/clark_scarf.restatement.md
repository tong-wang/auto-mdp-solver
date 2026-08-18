# clark_scarf — plain-English restatement

Round-trip artifact for the Phase-A gate. Source: Andrew J. Clark and Herbert
Scarf, *Optimal Policies for a Multi-Echelon Inventory Problem*, Management
Science 6(4), 1960; read from the 2004 reprint, Management Science 50(12S),
pp. 1782–1790. The model below is §2–§3 (the **serial** chain), generalized to
N levels as §3's scheme `[N] → … → [2] → [1]` describes.

IR: `clark_scarf_schema.json` · scenario set: `clark_scarf.scenarios.md`

---

## The problem in prose

A single product moves down a chain of stocking points. Level 1 is a retailer
facing customer demand; level 2 supplies level 1, level 3 supplies level 2, and
the top level *N* buys from an outside supplier with unlimited stock. Nobody but
the retailer sees demand (Assumption 1).

Each period, every link dispatches simultaneously: you choose how much to move
into each level. A level can never send more than it is holding — that is the
coupling that makes the problem hard, since the top of the chain must anticipate
what the bottom will need before knowing the demand. The top link is the one
exception: the outside supplier never runs out.

A shipment takes `leadtime` periods to arrive and cannot be recalled or
redirected once dispatched. Customer demand that cannot be met is **backlogged**
(Assumption 4) — the customer waits and is served as soon as stock arrives.

Holding stock costs money everywhere, and it costs *more* the further down the
chain it sits. Running out at the retailer costs a shortage penalty per unit per
period. Buying and shipping are free, so those two forces — the holding gradient
and the shortage penalty — are the whole trade-off.

### What is being decided

Each period, `N` numbers: one shipment quantity per link.

```
supplier ──q_N──▶ [N] ── … ──▶ [2] ──q_1──▶ [1] ──▶ demand
```

This is `N`, not `2N`. §3 counts it the same way for the two-installation case —
*"At the beginning of the period two decisions are made"* — and Eq. (14)
minimizes over exactly two variables. The `2N` that appears in this paper is the
`(S_n, s_n)` pair per echelon under a setup cost (§2, p. 1784): a description of
the **solution**, not the width of the decision.

### Objective

Minimize the **expected discounted total cost** over a finite horizon of 50
periods, discounting at **β = 0.95** — the paper's `αⁿ` weighting. Undiscounted
total cost is carried as a report-only column; it describes, never decides.

## Model envelope (added 2026-08-17, F3)

What the **theory** admits, per structural quantity — recorded verbatim-level
from the model, before any experiment-design or implementation concern. The
distinction this section enforces: *a swept set is a choice of study points
inside the domain; a cap is an implementation width; neither is the model.*

| quantity | model domain | experiment sweep | implementation cap |
|---|---|---|---|
| lead times `leadtime_k` | per-link, each any integer >= 1 (the paper admits per-installation lags; 0 is void under the event order) | {1, 2, 3}, **identical on every link** — the identical-lags restriction is a cited rendering narrowing, not model structure | **none** — `pipe` is `length: ["n_echelons", "leadtime"]`, so the width names the constant and any `L` renders (F7 removed the former `LEADTIME_CAP = 4`) |
| shipping/ordering costs `c_k` | per-link linear unit costs, real >= 0 (the paper's model carries them; the decomposition survives them) | **0 on every link — operator convenience at the interview**, a design value, not model structure | none — fully rendered (`c_ship` constant + `shipping` cost component), nonzero is a value change |
| `n_echelons` | any integer >= 2 (the paper's scheme `[N] -> ... -> [2] -> [1]` is stated for general N) | {2, 3, 4} | **none** — `stock` is `length: "n_echelons"`, `pipe`'s first axis is `n_echelons`, and the `ship` decision is `dim: "n_echelons"`. The former `N_LEVELS_MAX = 4` padding was a rendering artifact of `Decision.dim` being a literal `int`; upstream #33/#36 lifted that and F9 deleted the constant |
| demand | independent across periods, any distribution on [0, inf) — **the paper permits per-period distributions** (§2 parenthetical, p. 1783), so stationarity is design, and the Poisson **family** is a *selection* (candidate `poisson`) | stationary Poisson, rate 10.0 (2.0 / 4.0 in fixtures) | none |
| shipments `ship_k` | real, >= 0, unbounded above | integer lattice (design, for DP exactness) | `ship_max` action-scale cap |
| horizon | any finite T >= 1 | 50 (4 / 12 in test fixtures) | none |

The machine-readable form of this table is the schema's **`mdp.model`**
(v0.9.0, upstream #29): quantities with a theoretical `domain` and a source
citation, `out_of_scope`, quantified `dynamics`, and the `notation` map — with
`narrowed: {to, by}` on each element whose rendering is stricter than its
domain. Gated by `mdp_conformance`'s `model.boundary`, which now derives
**every** width this domain has — `horizon_T`, `leadtime`, `n_echelons`,
`ship_max` — since F9 left no cap for it to miss. The local tests that covered
the gap (a cap rendered as a *count of variables*, `pipe1..pipeN`, rather than
at a width site) are kept as the regression that keeps it derivable: the check
can only see a width while the constant is named.

*Provenance note.* The original freeze carried the sweep where the domain
belongs — the pipeline was two named slots, capacity L <= 2, and the sign-off
never surfaced it because this section did not exist. Corrected over four
re-freezes: **F3** generalized the register, **F7** named the lead time at the
width site (deleting `LEADTIME_CAP`), and **F9** named the chain length at all
three width sites (deleting `N_LEVELS_MAX`). Each is recorded in ESCALATION.md
§IR-CHANGELOG, and the missing Phase-A step went upstream as issue #29,
accepted and shipped as v0.9.0's `mdp.model`.

## Mode stance

**There are no scenario modes.** Every attribute the paper varies — the setup
cost `K`, unit cost `c`, shipping cost `c₁`, holding rates, shortage penalty,
lead times, horizon, demand rate, chain length — is numerical, so all of it
classifies as swept axes. The one categorical candidate, the demand family, was
settled to a single branch (Poisson). There is therefore **one leaderboard**, and
the head-to-head-versus-separate-branches question does not arise.

## Randomness classification

| bin | contents |
|---|---|
| realized per transition | **retailer demand** — Poisson, mean 10, independent across periods. The only stochastic primitive. |
| drawn once per episode by nature (world latent) | **none** — the paper holds costs and the demand distribution fixed across the horizon; nothing is redrawn per episode |
| the experimenter's training range | the swept axes below — design layer, not part of the model |

## The scenario set

Three axes, all numerical: `n_echelons` ∈ {2,3,4}, `leadtime` ∈ {1,2},
`p_short` ∈ {4,9,19} (critical ratios 0.80/0.90/0.95) — **18 cells**.

`n_echelons` and `leadtime` are **independent**, not interchangeable: an extra
installation is a *discretionary buffer* whose units may be held and re-decided
each period, whereas an extra period of lead time is *committed flow* that no
decision can touch. Same delay, different control.

Each cell is its own trained policy — both axes change the observation and action
width, so no single policy spans them. **Stage 0 picks one cell** to build
against; proposed canonical target `n3_l2_p09`.

Train/eval strategy: **specialist** per cell.

## Assumptions filled in without being told

| assumption | source |
|---|---|
| Serial chain per §2–§3; costs on echelon stock (Assumption 3); linear shipping, no setup on internal links (Assumption 2); backlogging (Assumption 4) | from the paper |
| Stock in transit **to** level *j* belongs to echelon *j+1*, not *j* — Assumption 3's "at a lower level **or in transit to a lower level**". It is charged its source's rate until it lands; in transit to the top level is in no echelon at all (on order, not yet in the system) | from the paper — **corrected post-freeze**, see `assumptions_log` |
| Objective discounts at β = 0.95 | `human_confirmed` |
| Demand is Poisson with fixed mean **10** — chosen over 5 so integrality is fine-grained relative to the decision scale (sd ≈ 3.2); at 5 a single unit is a large relative move and would contaminate the base-stock readback | `human_confirmed` |
| Purchase and shipping are **free** (`c = c₁ = 0`), isolating the holding-vs-shortage trade-off; linear purchase cost largely telescopes under backlogging | `human_confirmed` |
| Chain length is a swept axis, not fixed at 2 | `human_confirmed` |
| Lead time is a **separate** swept axis over {1,2} — a design choice inside the model domain L >= 1 (see Model envelope) | `human_confirmed` |
| Every episode starts with one period of cover at every stage — `stock[k] = arriving[k] = demand_mean`, `pending[k] = demand_mean` at L=2. Stated in raw coordinates; in echelon terms `x_k = k · demand_mean`. Not the optimum, so it hands the agent nothing | `human_confirmed` |
| Action width is N, one shipment per link | `human_confirmed` |
| Shipments are integer-valued, keeping every stock integral so the DP reference stays exact rather than gridded | `derived` |
| Shipment upper bound `ship_max = 200` = 20 × mean demand — an action-scale cap, not a physical limit | `derived` — **the one item still open at sign-off** |
| State carried in raw installation coordinates, echelon stock derived as a running sum. The two are related by an invertible linear map, so this is an implementation choice with no bearing on the model; raw was chosen because the availability constraint is then a per-component box | `derived` |
| Every state and action vector is exactly `n_echelons` wide — `stock` is `length: "n_echelons"`, `pipe` is `["n_echelons", "leadtime"]`, `ship` is `dim: "n_echelons"`. **Superseded at F9**: this line previously read "the stock vector always carries 4 slots; levels ≥ `n_echelons` are inert padding". The padding was a rendering artifact of `Decision.dim` being a literal `int`; upstream #33/#36 lifted that and the constant was deleted | `derived` |
| No world latent; `requires_memory = false` — the current stock and pipeline vectors are a sufficient statistic under either observation mode | `human_confirmed` |

## The headline experiment

Two observation modes, `raw` (physical stock at each installation plus what is in
flight toward it) and `echelon` (the running sums — the paper's coordinates).
They are related by an invertible running sum, so they carry **identical
information**. The question is therefore about **representation, not
information**: must the echelon coordinate system be handed over, or does the
network build it itself?

Clark & Scarf's optimum is `y_k = min(x̄_k(n), x_{k+1})` — order-up-to *in echelon
coordinates*, clipped by availability above. Stage 5 trains on `raw` and probes
whether the learned shipments are a function of the echelon aggregates and match
that rule. If they are, the structure was rediscovered. That is falsifiable, and
it is the campaign's point.

An MLP can express a cumulative sum (a lower-triangular matrix of ones), so a
`raw` shortfall would be an **optimization** finding, not an impossibility.

### The research questions, declared

| tier | stance | question | probe owed? |
|---|---|---|---|
| **1-comparative** | — *(standing: every campaign asks it)* | how does RL compare with the existing solutions — the exact DP **and** the heuristics? | no |
| **2-structural** | **bypass** *(primary)* | is the echelon transform *required* to solve the problem well? Evidence is the outcome alone: the ordered `means` split `echelon ⊂ raw` and its cross-link delta | no |
| **2-structural** | **confirm** *(secondary)* | is the learned policy nonetheless echelon-structured — does it match `y_k = min(x̄_k(n), x_{k+1})`? | **yes** — `clark_scarf_policy_probe.py`, spec §14 |

The stances are not interchangeable, and the pair is deliberate: `raw ≈
echelon` is a **success** under bypass — the coordinates were not needed — and
**inconclusive** under confirm, since not needing them is consistent with
having found them and does not demonstrate it. Confirm is carried because
echelon stock is *known* optimal, so a good enough policy has to arrive there.
A campaign whose tier-2 stance were bypass alone would owe no `INTERPRET.md`.

*(This section was the interim home while the IR had nowhere to put a research
question. Upstream **#24** — filed from this campaign — shipped the root-level
`research_questions` block in **v0.8.8**, so the tier-2 stance is now
declarable, and `mdp_conformance`'s `research.deliverables` ties the §14
interpretation artifact to it. **Both stances are now declared** in the IR
root — bypass primary, confirm secondary — and the check PASSes, reporting
that the probe and `INTERPRET.md` the confirm stance owes are both present.
Root-level, so declaring them moved neither fingerprint.)*

## Invariants transcribed into the IR

Claims taken from the problem statement, not from the model — these are what can
catch a mis-formalization the differential cannot see, since the differential
only proves the interpreter and the generated domain agree.

| name | claim in plain words |
|---|---|
| `retailer_balance` | retailer stock changes only by what lands from above and what customers take |
| `conservation` | stock in the system changes only by what is bought from the outside supplier and what demand consumes — internal shipments move stock between levels without creating or destroying it |
| `upper_levels_non_negative` | only the retailer can go negative; no installation ever ships more than it holds |

Two further invariants were **retired with the widths they policed**, not
weakened: `no_flow_through_padding` (inert levels never move stock — F9 left
no inert levels) and `pipe_slots_beyond_leadtime_empty` (no slot beyond the
lead time is ever occupied — F7 made every pipeline exactly `leadtime` long,
so the region it guarded does not exist). Each was a guard against a
fixed-width rendering, and each became vacuous the moment the width named its
constant.

All three hold across the rendered trajectories; the interpreter reported no
violation. They are advisory here and **fatal at the Stage-1 gate**.

---

## Annotated sample trajectory

`n3_l2_p09` — 3 levels, lead time 2, shortage 9, `h_install = [2.0, 1.0, 0.5, 0.0]`.
Every link asked to ship 10 each period (a fixed, deliberately unoptimized policy).

> **Regenerated 2026-08-17 (F9).** The table below previously showed
> `holding = 109.00` at t=0 and an episode total of 778.00. Those were
> **pre-F1** numbers: F1 corrected in-transit stock from being charged one
> echelon too low, and this artifact was never re-rendered afterwards. The
> discrepancy decomposes exactly — stock component 39.00, in-transit 70.00 at
> the wrong (source) rate versus 30.00 at the correct one, giving 109.00 and
> 69.00. **The round-trip artifact that exists to let a human check the model
> was showing a model corrected nine re-signings earlier.** Re-render it on
> every re-freeze; a stale trajectory is worse than none, because it reads as
> confirmation.

The vectors are `n_echelons` wide and `pipe` is a matrix (F9), so this is also
the shape check: three links, two slots each, nothing padded.

```step7b
python -m mdp_ir.interpreter clark_scarf/clark_scarf_schema.json \
  --instance n3_l2_p09 --decision ship=10 --episode-seed 3 --max-periods 8
```

*(A single value broadcasts to the decision's resolved width, so `ship=10`
means "10 on every link"; `ship=10,0,5` sets components. Until **v0.9.5** the
CLI parsed only a scalar and could not drive a vector decision at all, so this
section carried an API snippet instead — spec Phase-A step 7b names this CLI
as the way to render the artifact, which made it unavailable on exactly the
domains that most need a human to look at one. Upstream #39, filed from this
campaign.)*

```
clark_scarf v0.4  episode_seed=3 instance=n3_l2_p09
t                 ship  demand              arrived              shipped                stock                                         pipe  holding  shortage  shipping  total  reward
0  [10.00,10.00,10.00]       8  [10.00,10.00,10.00]  [10.00,10.00,10.00]  [12.00,10.00,10.00]  [[10.00,10.00],[10.00,10.00],[10.00,10.00]]    69.00      0.00      0.00  69.00  -69.00
1  [10.00,10.00,10.00]      14  [10.00,10.00,10.00]  [10.00,10.00,10.00]   [8.00,10.00,10.00]  [[10.00,10.00],[10.00,10.00],[10.00,10.00]]    61.00      0.00      0.00  61.00  -61.00
2  [10.00,10.00,10.00]      13  [10.00,10.00,10.00]  [10.00,10.00,10.00]   [5.00,10.00,10.00]  [[10.00,10.00],[10.00,10.00],[10.00,10.00]]    55.00      0.00      0.00  55.00  -55.00
3  [10.00,10.00,10.00]       5  [10.00,10.00,10.00]  [10.00,10.00,10.00]  [10.00,10.00,10.00]  [[10.00,10.00],[10.00,10.00],[10.00,10.00]]    65.00      0.00      0.00  65.00  -65.00
4  [10.00,10.00,10.00]      18  [10.00,10.00,10.00]  [10.00,10.00,10.00]   [2.00,10.00,10.00]  [[10.00,10.00],[10.00,10.00],[10.00,10.00]]    49.00      0.00      0.00  49.00  -49.00
5  [10.00,10.00,10.00]       9  [10.00,10.00,10.00]  [10.00,10.00,10.00]   [3.00,10.00,10.00]  [[10.00,10.00],[10.00,10.00],[10.00,10.00]]    51.00      0.00      0.00  51.00  -51.00
6  [10.00,10.00,10.00]      11  [10.00,10.00,10.00]  [10.00,10.00,10.00]   [2.00,10.00,10.00]  [[10.00,10.00],[10.00,10.00],[10.00,10.00]]    49.00      0.00      0.00  49.00  -49.00
7  [10.00,10.00,10.00]       5  [10.00,10.00,10.00]  [10.00,10.00,10.00]   [7.00,10.00,10.00]  [[10.00,10.00],[10.00,10.00],[10.00,10.00]]    59.00      0.00      0.00  59.00  -59.00
episode total = 458.00   reward total = -458.00   periods = 8
eval metrics: undiscounted_cost = 458.00
```

*(Verbatim CLI output — pasted rather than trimmed, so the block is a
literal diff against the documented command. The hand-edited table this
replaced is how the pre-F1 numbers survived: a summary can drift from its
source without looking wrong.)*

**Reading period 0.** The episode starts with one period of cover everywhere:
`stock = [10,10,10]` and every pipeline slot holding 10.

1. **A (arrivals land).** Each link's head slot lands: `stock` becomes
   `[20,20,20]`, `arrived = [10,10,10]` records it, and every pipeline shifts
   one step with a zero appended.
2. **S (all links dispatch).** `ship[0] = min(10, stock[1]=20) = 10` — the
   retailer's supplier has enough. `ship[1] = min(10, stock[2]=20) = 10`.
   `ship[2] = 10` **unclipped**, because with N=3 level 3 is the top and buys
   from the outside supplier. Sources are debited: `stock → [20,10,10]`.
   Because L=2, dispatched units enter each pipeline's LAST slot, two arrivals
   away.
3. **D (demand).** 8 units are taken from the retailer: `stock[0] = 20 − 8 = 12`.

**The cost.** Holding is charged on end-of-period **echelon stock**. Per
Assumption 3 a level's cost covers stock at that level plus stock at a *lower*
level **or in transit to a lower level** — so stock in transit *to* a level
belongs to the echelon **above** it, and is charged its source's rate until it
lands:

```
on hand      2.0·12  +  1.0·10  +  0.5·10   =  39
                L1        L2        L3

in flight    1.0·20  +  0.5·20              =  30
             ->L1      ->L2
                                               ---
                                                69
```

Stock in transit to the **top** level appears in neither column, and that is
the point rather than an omission: those units are on order from the outside
supplier, so they are in no echelon's stock at all. Since F9 the rule says so
directly — the in-flight sum runs over `range(n_echelons - 1)`, stopping one
short of the top — where it used to rely on `h_install` happening to be 0
above the chain. Same numbers, stated instead of inherited. (The former fourth
row here was inert padding, charged nothing because its rate was 0; F9 removed
the padding, so there is nothing left to be invisible.)

Note this is echelon **stock**, what holding is charged on — distinct from
echelon **position** `u = x_1 + w_1 + ...`, which is what `f_n(u)` optimizes
over. Conflating the two double-charges pipeline stock.

Shortage is 0 here — the retailer never ran out under this policy. Later periods
in the full 30-period episode do stock out, which is where the trade-off bites.

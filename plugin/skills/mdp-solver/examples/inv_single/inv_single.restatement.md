# inv_single — plain-English restatement

The Phase-A restatement (SKILL.md step 7a/7b): **the artifact a human reads to
check that the formalization matches the world.** The `mdp` block freezes; this
document does not — it is **re-rendered whenever the model moves**, and the
`step7b` block below is re-run and diffed by
`mdp_conformance docs.restatement_current`.

Authoritative source: `inv_single_schema.json`. Where this prose and the IR
disagree, the IR is right and this document is stale.

## The problem in words

A single stocking point holds one item. Time runs in **30 discrete periods**.
In each period, in this order:

1. **O (order)** — you choose a replenishment quantity. It does not arrive now.
2. **R (receive)** — whatever was ordered `L` periods ago arrives and joins stock.
3. **D (demand)** — customer demand is realized and served from stock.

Whatever demand you cannot serve is either **backlogged** (it stays owed and is
served later, and you pay a shortage penalty every period it remains owed) or
**lost** (the customer goes away), depending on the instance. Whatever stock is
left at the end of a period costs you to hold.

You are trying to **minimize total cost over the 30 periods, undiscounted**
(β = 1.0). The tension is the classic one: order too little and you pay
shortage, order too much and you pay holding — and because orders take `L`
periods to arrive, you are always deciding against demand you have not seen yet.

## What the theory says, and what is only a modelling choice

The `mdp.model` block separates these deliberately, because conflating them is
how a sweep maximum silently becomes a capacity limit.

**Quantities the theory declares:**

| quantity | domain | random? |
|---|---|---|
| `demand` | real, `[0, ∞)`; independent across periods given the period's distribution, which may itself carry a per-episode latent | **yes** |
| `leadtime` | integer `≥ 0` | **yes** — the theory admits a random lead time; a fixed one is a *scenario* choice |
| `order` | real, `[0, ∞)` — the control | no |
| `inventory` | real; unbounded below under backlog, non-negative under lost sales | no |
| `T` | integer `≥ 1` — finite horizon, no salvage value | no |
| `h`, `b`, `K`, `c` | real, `[0, ∞)` — holding, shortage, fixed-order, unit-order costs | no |

**Explicitly out of scope** (the theory does *not* speak to these):

- capacity limits on the order quantity — the theory admits **any** non-negative
  order, so the IR's upper bound is a **design cap, never a physical one**
- supplier reliability and partial delivery: an order once placed always arrives in full
- perishability, obsolescence, any age-dependent stock value
- salvage value or terminal cost at the horizon end
- pricing, demand shaping, substitution

**What the rendering narrows** (design and implementation, not theory) — each is
cited in the IR's `narrowed` fields:

- the order is capped at `order_cap_mult × demand.mean` (= 20 × mean) — an
  action-scale cap, symbolic so each demand candidate resizes to its own
  scale and the coefficient has one named home
- orders are placed on an **integer lattice** (the default action mode is
  discrete), because the order flows into integer stock
- the pipeline carries a fixed `pipeline_len` slots

## State, decision, objective

**State** — only what the next transition needs:

| variable | meaning |
|---|---|
| `period` | current period `t`; the episode ends when `period == T` |
| `inventory` | net on-hand after all of the period's events |
| `pipeline` | `pipeline[k]` = quantity arriving at the `k`-th future receive event |

**Decision** — `order`, one non-negative quantity per period.

**Objective** — minimize the sum of four per-step components:
`holding` + `shortage` + `order_fixed` + `order_variable`. The gym's single
reward mode `neg_cost` negates it, because RL maximizes.

## Instances — each one is a separate leaderboard

13 scenarios vary lead time (0, 2, or stochastic uniform{1,2,3} — every
stochastic-lead-time instance shares the one base support, E[L] = 2, so
`lt` → `slt` varies lead-time *variability* at a matched mean), stockout mode
(backlog vs lost sales), demand family (Poisson, discrete, latent-bearing, a
mixture), event order (O-R-D, R-D-O, R-O-D) and fixed cost (`simple_k` has
`K = 20`). **They are different problems, not settings of one** — never
compare a score across them or quote an aggregate over them.

Beside the instances sits one declared **grid** (`grid16`, the root `grids`
block): shortage cost b ∈ {1, 4, 9, 19} × fixed cost K ∈ {0, 20} ×
deterministic lead time ∈ {0, 2}, crossed on `lt` so every cell keeps the
family-maximum pipeline (the §5.6 obs-dim padding). A grid is a *generality
target*, not 16 more leaderboard instances of the same kind: one generalist
policy trains on its sampler and is evaluated per cell against each cell's
exact DP, all cells on one seed block. The `vec_ctx` observation mode exists
for it — the `vec` state plus the critical fractile b/(b+h) and K, the
context a cross-regime policy must see to condition on the cell.

The campaign's **six** declared research questions live in
`research_questions.tier2` (authoritative) and are tabulated in
`ESCALATION.md` §Campaign question. A seventh id was declared and withdrawn on
2026-08-27 — action encoding, which is a question about the rendering rather
than about the policy structure this domain poses, and which now lives in
`README.md`'s `3-methodology` section as Q-B (answered: #E4). The id is retired,
not reused, and the withdrawal moved no fingerprint.

## Sample trajectory

Held at a constant order of 14 units on `lt` (deterministic `L = 2`, backlog,
Poisson(10) demand), so the mechanics are readable: the first order cannot
arrive until `t = 2`, so the opening backlog is unavoidable and the run then
climbs out of it into positive stock. Watch `pipeline` shift left each period
and `received` lag `order` by exactly two.

**The command that renders it**, run from this folder's parent:

```step7b
python -m mdp_ir.interpreter inv_single/inv_single_schema.json --instance lt --decision order=14 --episode-seed 0 --max-periods 10
```

Its verbatim output:

```
inv_single v0.4  episode_seed=0 instance=lt
t  order  demand  received  leadtime  lost_sales  action_period  inventory         pipeline  holding  shortage  order_fixed  order_variable   total   reward
0  14.00       8         0         2           0              0         -8      [0,14.00,0]     0.00     72.00         0.00            0.00   72.00   -72.00
1  14.00      11         0         2           0              1        -19  [14.00,14.00,0]     0.00    171.00         0.00            0.00  171.00  -171.00
2  14.00      10     14.00         2           0              2     -15.00  [14.00,14.00,0]     0.00    135.00         0.00            0.00  135.00  -135.00
3  14.00      12     14.00         2           0              3     -13.00  [14.00,14.00,0]     0.00    117.00         0.00            0.00  117.00  -117.00
4  14.00       7     14.00         2           0              4      -6.00  [14.00,14.00,0]     0.00     54.00         0.00            0.00   54.00   -54.00
5  14.00      14     14.00         2           0              5      -6.00  [14.00,14.00,0]     0.00     54.00         0.00            0.00   54.00   -54.00
6  14.00      13     14.00         2           0              6      -5.00  [14.00,14.00,0]     0.00     45.00         0.00            0.00   45.00   -45.00
7  14.00       7     14.00         2           0              7       2.00  [14.00,14.00,0]     2.00      0.00         0.00            0.00    2.00    -2.00
8  14.00      14     14.00         2           0              8       2.00  [14.00,14.00,0]     2.00      0.00         0.00            0.00    2.00    -2.00
9  14.00      14     14.00         2           0              9       2.00  [14.00,14.00,0]     2.00      0.00         0.00            0.00    2.00    -2.00
episode total = 654.00   reward total = -654.00   periods = 10
```

Reading it: at `t = 0` the order of 14 enters `pipeline[1]` and nothing is
received, so demand of 8 goes straight to backlog — `inventory = -8`, and
`shortage = 8 × b = 72.00` at `b = 9`. Receipts begin at `t = 2`. By `t = 7`
inventory turns positive and the cost switches from the shortage column to the
holding column at `h = 1`. `order_fixed` stays `0.00` throughout because `lt`
has `K = 0`.

## Fingerprints

| hash | value | what it covers |
|---|---|---|
| `mdp_fingerprint` | `84a22814f666` | the freeze token — model + design + rendering |
| `model_fingerprint` | `17606699e437` | the theory layer alone |
| `structural_fingerprint` | `40f9149f9a0e` | the rendering |

Recorded 2026-09-07 (F6: `vec_d` / `vec_d_ip` retired and the `demand`
info-field rationale corrected). Previously `mdp b17fce75f34d` /
`structural 5228e6de79fb` from 2026-08-27's restart batch (F2) and literal
audit (F3). **The theory hash `17606699e437` has never moved**: the model did
not change then and did not change now — F6 removed two observation modes and
rewrote one justification, neither of which is theory.
`desc` fields are pruned from both hashes; `rationale` and `source` are
**not** — promoting a `source` moves them.

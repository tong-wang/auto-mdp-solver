# owmr — scenario set (Phase A, step 4 — decided)

Source: Doğru, de Kok & van Houtum (2009), *Flex Serv Manuf J* 21:114–147, §4.1
(identical-retailer test bed). Status: DECIDED 2026-09-22 (sign-off on owmr.restatement.md) — lead times (3, 1),
T = 100, mean-cover start, specialist per cell. Kept as the record of why.

Human's direction (2026-09-22): few scenarios, no grids; one simple base cell
crossed with high/low penalty and high/low demand variability; N large (5) to
show dimensionality. The benchmarks are not a DP: the balance-assumption lower
bound is closed-form and the LB heuristic is a simulation, both linear in N.

## 1. Constants (all numerical; no categorical mode in the paper's model)

| constant | meaning | value here | swept? |
|---|---|---|---|
| `N` | retailers | 5, identical | no (tier 1: action width) |
| `l0` | supplier → warehouse lead time | **to choose: 1 / 3 / 5** | no |
| `l_rt[i]` | warehouse → retailer lead time | 1 | no |
| `h0` | added holding value at the warehouse, per unit-period | 0.5 | no |
| `h_rt[i]` | added holding value at retailer i (on top of h0) | 0.5 | no |
| `p_rt[i]` | backorder penalty at retailer i, per unit-period | **4 (low) / 19 (high)** | yes |
| `mu_rt[i]` | mean one-period demand | 1 | no |
| `cv_rt[i]` | coefficient of variation | **0.5 (low) / 2 (high)** | yes |

Demand: `gamma(shape = 1/cv², scale = mu·cv²)` per retailer, independent
streams. Lead times fixed. Costs at end of period: `h0` on warehouse on-hand,
`h0 + h_rt` on retailer on-hand, `p_rt` on retailer backlog; transit uncharged.

Why these fixed values: `h0 = h_rt = 0.5` is the paper's "equal added value at
every stock point" case, the neutral holding split; `mu = 1` is the paper's
normalization; `l_rt = 1` is the short-retailer-lead-time setting in which the
paper's gaps are largest, so the cv contrast has room to show.

## 2. Named cells — four, a 2 × 2 over penalty × variability

| cell | p_rt | cv_rt | paper's regime (identical retailers, Tables 3–4) |
|---|---|---|---|
| `base` | 4 | 0.5 | tight: ave gap ≈ 0.04–0.07 %, max 0.36 % — the instrument-validation cell, LB ≈ optimum |
| `hicv` | 4 | 2 | loose: ave gap ≈ 5–7 %, max 15–17 % at l0 ≥ 3 |
| `hipen` | 19 | 0.5 | tight; 95 % target no-stockout probability |
| `hipen_hicv` | 19 | 2 | loose, with the high service target |

All four share one observation shape and one action width (N = 5, same
lead times), so they are one leaderboard family; scores across cells are
still not compared to each other (different cost scales) — each cell is
scored against its own bracket.

## 3. The lead-time pair — the one open value

| (l0, l_rt) | paper's gap statistics at cv = 2, identical retailers | consequence for RL |
|---|---|---|
| (1, 1) | ave 0.73 %, max 3.64 % (Table 3, l0 = 1 column) | shortest pipelines (state dim 1 + 1 + 5 + 5); the cv contrast is small |
| (3, 1) | ave 1.33–4.9 %, max 15–17 % | moderate pipelines (1 + 3 + 5 + 5); clear contrast |
| (5, 1) | ave 1.15–4.9 %, max 15–17 %; cv 3 reaches 38.6 % | longest pipelines (1 + 5 + 5 + 5); largest transient |

Recommendation: (3, 1).

## 4. Horizon and initial state (next question)

Objective = total cost over T periods (β = 1), quoted also as total / T
beside the paper's per-period bound. Options: T = 100 or 200; initial state
= mean cover (every pipeline slot at one period of mean demand, warehouse
on-hand μ0 = 5, each retailer on-hand 1) or empty. Recommendation: T = 100,
mean cover.

## 5. Training target (question after that)

Specialist per cell (four policies, four brackets) — recommended. No grids
are declared.

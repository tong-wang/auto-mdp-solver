# cnv — candidate scenario set (Phase A step 4, 2026-09-17)

Source: Jain, Rudi & Wang (2015), §5 "Numerical Illustration". Every number
below is the paper's. Settled so far: Poisson arrivals with hidden rate
λ ~ Gamma(α, β); arrivals simulated as exponential gaps; running totals in the
state; one head-to-head board over event / timing / full.

## Scenario constants (the `mdp` layer)

| constant | meaning | paper symbol | axis? |
|---|---|---|---|
| `horizon_N` | ordering periods | N | no (structural: 4 for the DP study, 100 for the myopic study) |
| `price` | revenue per unit sold | r = 2 | no |
| `unit_cost` | cost per unit stocked | c | **axis**: 0.2, 0.3, …, 1.8 (17 values; newsvendor ratio 0.1 … 0.9) |
| `prior_rate` | Gamma rate parameter of the prior on λ | β | **axis**: 1/β ∈ {0.5, 1, 2, 4, 8, 16} |
| `prior_mean` | prior expected arrival rate | α/β | **axis**: {10, 20, 30, 40, 50} |
| `arrivals_buffer` | exponential gaps drawn per period (demand cap) | — (paper: uncapped) | no; default 402 = the 1e-8 upper quantile of the worst grid cell's marginal demand (the legacy code used 26–86, which capped demand in the diffuse-prior cells) |

Derived: prior shape α = `prior_rate × prior_mean`; newsvendor ratio (r − c)/r.

## Fixed instances (`SCENARIOS`)

| instance | N | c | 1/β | α/β | paper role |
|---|---|---|---|---|---|
| `base` | 4 | 1.0 | 1 | 10 | Fig. 5(a) checkpoint study cell (α=10, β=1); the legacy "simple" |
| `fig3_c10` | 4 | 1.0 | 16 | 10 | Fig. 3(a)/4(a) family at ratio 0.5 (α=10/16, β=1/16) |
| `fig3_c15` | 4 | 1.5 | 16 | 10 | same family at ratio 0.25 |
| `fig6a` | 100 | 1.0 | 16 | 10 | Fig. 6(a): long-horizon myopic simulation |
| `fig6b` | 100 | 1.5 | 16 | 10 | Fig. 6(b) |

`fig6a`/`fig6b` have a different horizon and are separate leaderboards; they
are never compared with the N=4 boards.

## Grids (`GRIDS`, design layer)

| grid | axes | cells | paper role |
|---|---|---|---|
| `paper510` | unit_cost × prior_rate × prior_mean (17 × 6 × 5) | 510 | §5.1 full study: mean/median loss recovery η over all cells |
| `fig3` | unit_cost only, at 1/β = 16, α/β = 10 | 17 | Fig. 3(a), 4(a): the cost sweep |

Training samples a grid uniformly (generalist); evaluation enumerates it, one
row per cell on the same seeds.

## Candidate compositions (choose one)

| composition | instances | grids | what it can answer |
|---|---|---|---|
| **A — figures only** | base, fig3_c10, fig3_c15 (+ fig6a/b) | fig3 | Fig. 3/4 curves; the N=100 myopic gaps |
| **B — full paper study** | as A | fig3 + paper510 | A, plus the 510-cell η distribution (Fig. 4(b)) |
| **C — N=4 only** | base, fig3_c10, fig3_c15 | fig3 + paper510 | B without the long-horizon pair |

The N=100 instances cost nothing to declare; whether they get trained is a
run-plan question at Stage 3.

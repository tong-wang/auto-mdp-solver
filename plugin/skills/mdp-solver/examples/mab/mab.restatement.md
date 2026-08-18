# mab — plain-English restatement (Phase A round-trip artifact)

IR: `mab_schema.json` · `mdp` fingerprint `5bb25e684405` ·
structural fingerprint `36688c7707f1` · seed scheme v2

*(Fingerprints current as of the shipped schema. They moved after the Phase-A
freeze, each move recorded in §IR-CHANGELOG: **F3** promoted K and T to
axis-tagged scenario constants, **F4** widened state-variable bounds to cover
the long-horizon instances, **F5** made `gaussian` the slot default (the
branch this campaign reports) while adding an explicit `bernoulli` instance,
and **F6** declared the `mdp.model` theory layer and regrouped the file into
model / design / rendering — which moved the `mdp` freeze token once
(`e385f662a857` → `5bb25e684405`) and left the structural hash untouched.
`546617c32435` / `072c7f55c967` are the pre-F3 values and appear in entries
written before that move.)*

(Frozen 2026-07-24 at `ad610581f715`. Re-confirmed three times since, each
structural, so each moved the fingerprint. 2026-07-28: the claims below were
declared as `mdp.invariants`; then the IR moved to the **catalog form** — one
`payout` slot with `bernoulli` | `gaussian` candidates, each drawing its arm
means as a single `iid` latent — saying there is one source of randomness
rather than two, which is what the world always was. 2026-07-29: the `pull`
stage took `key_exprs: ["int(arm)"]`, giving each arm its own per-period
stream. That last one is the only re-confirmation that **corrected the model**
rather than restating it — see "Randomness classification" below. The problem
as written in plain English is unchanged throughout.)

## The problem

You face a row of **10 slot machines (arms)**. At the start of each episode,
nature secretly fixes each arm's average payout — you never see these values.
Each round (**1000 rounds** total) you **pull exactly one arm** and observe
**only that arm's payout**; the other arms reveal nothing. Your goal is to
**maximize the total payout over the 1000 rounds** — which means balancing
*exploring* arms you know little about against *exploiting* the arm that
currently looks best. Because the simulator knows the hidden means, evaluation
also reports **regret**: how far you fell short of always pulling the truly
best arm.

## Mode stance (human-decided)

Two **independent branches**, each its own scenario instance and its own
leaderboard — no head-to-head comparison (payout scales differ):

| branch | scenario instance | a pull pays | hidden means drawn each episode |
|---|---|---|---|
| Bernoulli | base (`mab`) | 0 or 1, success prob $p_i$ | $p_i \sim \text{Uniform}(0,1)$ i.i.d. |
| Gaussian | `gaussian` | $N(\mu_i, 1)$, known $\sigma = 1$ | $\mu_i \sim N(0,1)$ i.i.d. |

## Randomness classification

There is exactly **one source of randomness**: the payout of the arm pulled
this round. Bernoulli and Gaussian are *alternative recipes* for that one
source — they never coexist — so the IR declares them as two candidates of a
single `payout` slot sharing one stream identity, not as two sources.

- **Per-round pull payout** — the `payout` slot, `stream_id` 0, realization =
  period, **keyed on the chosen arm** (`key_exprs: ["int(arm)"]`, solver ≥
  v0.5.9). The selected candidate resolves to the one source; there is no guard
  and no branch flag in the dynamics. The arm key *refines* the period slot
  into K independent per-arm streams, of which a round reads exactly one — so
  the full payout table $\{r_{i,t}\}$ is fixed by the episode seed and the
  decision only selects which entry is revealed. Without it a single primitive
  variate per period was shared by all ten arms, which fixed the round's luck
  before the choice was made: Gaussian payouts differed across arms by exactly
  their means (identical noise) and Bernoulli payouts were comonotone in $p$.
  The realized law the agent saw was still correct — exactly one arm is
  revealed per round — but it is not the standard bandit model, and it would
  break for any variant revealing more than one arm per round or scoring
  realized-path rather than mean-based regret. Keying on a decision is sound
  because the arm selects *which* pre-determined stream is read, never what it
  contains, so path-independence holds (and is in fact strengthened).
- **Arm means** — the slot's **hidden world latent**, drawn once per seed on
  the meta branch at the *same* id 0 (one stream identity per source, both
  branches). Owned by the generator that parameterizes it
  (`LatentBernoulliPayout` / `LatentGaussianPayout` in `mab_uncertainty.py`,
  spec §5.2); `mab_scenarios.py` only composes. Because the arms are i.i.d.
  the vector is one `iid` draw — `{of: uniform|normal, size: 10, ...}` — which
  desugars to the synthesized constant `payout_arm_means`, indexed by the
  pulled arm (`p: "payout_arm_means[int(arm)]"`). The policy never observes it;
  it surfaces only in the diagnostic info fields `mean_pulled` / `opt_mean`
  for eval-time regret accounting.
- No training-only distributions: the per-seed redraw *is* the problem's own
  story (a fresh bandit instance each seed).

## Decision, horizon, objective

- **Decision**: `arm` — discrete index 0–9, one pull per round, always
  feasible (source: human_confirmed).
- **Horizon**: T = 1000 rounds, 0-based; horizon end is `terminated`.
- **Objective**: maximize the sum of realized pull payouts
  (single component `pull`; regret is an eval-time report, not the reward).

## Claims that must hold (declared as `mdp.invariants`)

Transcribed from the restatement above, not read off the model — the
differential only proves the interpreter and the domain *agree*, so a shared
mis-formalization needs an independent claim to catch it. Checked on every row
of every trajectory, on both branches, by `mdp_ir.laws` and the differential:

| claim | expression | what it pins |
|---|---|---|
| `one_pull_per_round` | `sum(pulls) == t + 1` | "you pull exactly one arm each round" — no skipped or doubled round |
| `pulled_arm_advances` | `pulls[int(arm)] == prev.pulls[int(arm)] + 1` | the pull landed on the arm chosen (with the above: on *only* that arm) |
| `payout_ledger` | `close(sum(payouts), sum(prev.payouts) + reward)` | the ledger grows by exactly the realized payout — nothing credited to an arm not played |
| `regret_is_nonnegative` | `opt_mean >= mean_pulled` | "how far you fell short of the truly best arm" is well-posed |

Claims an expression cannot state live in `mab_test.py` instead: that a
Bernoulli pull pays only 0 or 1 (a claim about one *candidate*, and an
expression cannot name the selection), that the payout depends on the round and
the arm but not the pull history, that **the arms are independent streams
within a round** (a claim about counterfactual arms, which no single trajectory
can express — Gaussian pairs must not differ by a constant, Bernoulli pairs
must not be comonotone), that no observation or reward mode reads the hidden
means, and that the `bayes` features are the true conjugate posteriors (checked
by quadrature against the exact likelihood).

## What the policy observes (human-decided)

Sufficient-statistics observation (`stats`, default): per-arm **pull counts**,
per-arm **payout totals**, and **time-to-go** (21 numbers). This is the exact
sufficient statistic for both conjugate priors, so the belief-state MDP is
fully observed — `requires_memory` overridden to **false** (worked
human_override; the derivation suggested true because the samplers are
hidden). A second mode `bayes` (per-arm conjugate **posterior mean** +
**posterior sd** of the true arm mean, + time-to-go; Beta(1,1) posterior on
the Bernoulli branch, Normal posterior on the Gaussian branch, computed by
the domain-owned builtins in `mab/mab_bayes.py`) is registered as a Phase-B
design axis. Both modes carry the same information (the belief state); they
differ only in parametrization.

## Scenario set & training target (human-decided)

One scenario per branch, no size grid:

| scenario | K | T | prior |
|---|---|---|---|
| `bern_K10_T1000` (base instance) | 10 | 1000 | $p_i \sim U(0,1)$ |
| `gauss_K10_T1000` (`gaussian` instance) | 10 | 1000 | $\mu_i \sim N(0,1)$, $\sigma=1$ |

One policy trained per branch on its scenario's per-episode redraw;
evaluated on the same scenario with a shared eval-seed block; baselines
(random / greedy / UCB1 / Thompson) on the same seeds; separate leaderboards.

## Key assumptions (with source)

- Bernoulli + Gaussian as independent branches — **human-decided** (menu).
- Canonical priors — **human-decided** (menu).
- K=10, T=1000 (Sutton & Barto testbed size) — **human-decided** (menu).
- Sufficient-statistics observation, no memory — **human-decided** (menu);
  `rl.requires_memory` = false, **human_override**.
- Decision type discrete, bounds [0, 9] — **human_confirmed**.
- Arm means as a single `iid` latent draw of `size` 10 — **derived** (the arms
  are i.i.d., so one recipe covers them; components are drawn in arm order from
  the sampler's one rng, bit-reproducible by construction). Superseded the
  original encoding as 10 hand-listed scalar draws (`p0..p9` / `m0..m9`), which
  predated the `iid` family.
- Bernoulli/Gaussian as two candidates of one `payout` slot rather than two
  sources — **derived** (they are alternative recipes for the same randomness
  and never coexist, so they share one stream identity; spec §5.2).
- Per-arm payout streams via `key_exprs` on the `pull` stage — **derived**
  (the arms are independent in the standard model, so their noise must be too;
  the shared-variate encoding was a formalization artefact, not a modelling
  choice). Required the upstream relaxation shipped as solver v0.5.9.
- `payouts` element bounds ±4000 — **derived** (≥6σ margin for T=1000
  Gaussian draws).
- Obs normalization enabled — **derived** (heterogeneous stationary scales).
- `bayes` obs mode = exact conjugate posterior (mean, sd) per arm —
  **human-decided**; implemented as `mdp.expr_builtins`
  (`bayes_post_mean` / `bayes_post_sd` in `mab/mab_bayes.py`).

## Annotated sample trajectories (interpreter output, episode_seed=3, fixed decision arm=3)

Bernoulli branch (`--instance bernoulli`) — this episode nature drew arm 3's
success probability $p_3 = 0.97$, which happens to also be the best arm
(`opt_mean` 0.97); every pull pays 0/1 and increments `pulls[3]` / adds to
`payouts[3]`:

```step7b
python -m mdp_ir.interpreter mab/mab_schema.json --instance bernoulli --decision arm=3 --episode-seed 3
```

```
mab v0.4  episode_seed=3 instance=bernoulli
 t   arm  reward  mean_pulled  opt_mean                   pulls                 payouts  pull  total
 0  3.00    0.00         0.97      0.97   [0,0,0,1,0,0,0,0,0,0]   [0,0,0,0,0,0,0,0,0,0]  0.00   0.00
 1  3.00    1.00         0.97      0.97   [0,0,0,2,0,0,0,0,0,0]   [0,0,0,1,0,0,0,0,0,0]  1.00   1.00
 2  3.00    1.00         0.97      0.97   [0,0,0,3,0,0,0,0,0,0]   [0,0,0,2,0,0,0,0,0,0]  1.00   1.00
 3  3.00    1.00         0.97      0.97   [0,0,0,4,0,0,0,0,0,0]   [0,0,0,3,0,0,0,0,0,0]  1.00   1.00
 4  3.00    1.00         0.97      0.97   [0,0,0,5,0,0,0,0,0,0]   [0,0,0,4,0,0,0,0,0,0]  1.00   1.00
 5  3.00    1.00         0.97      0.97   [0,0,0,6,0,0,0,0,0,0]   [0,0,0,5,0,0,0,0,0,0]  1.00   1.00
 6  3.00    1.00         0.97      0.97   [0,0,0,7,0,0,0,0,0,0]   [0,0,0,6,0,0,0,0,0,0]  1.00   1.00
 7  3.00    1.00         0.97      0.97   [0,0,0,8,0,0,0,0,0,0]   [0,0,0,7,0,0,0,0,0,0]  1.00   1.00
 8  3.00    1.00         0.97      0.97   [0,0,0,9,0,0,0,0,0,0]   [0,0,0,8,0,0,0,0,0,0]  1.00   1.00
 9  3.00    1.00         0.97      0.97  [0,0,0,10,0,0,0,0,0,0]   [0,0,0,9,0,0,0,0,0,0]  1.00   1.00
10  3.00    1.00         0.97      0.97  [0,0,0,11,0,0,0,0,0,0]  [0,0,0,10,0,0,0,0,0,0]  1.00   1.00
11  3.00    1.00         0.97      0.97  [0,0,0,12,0,0,0,0,0,0]  [0,0,0,11,0,0,0,0,0,0]  1.00   1.00
12  3.00    1.00         0.97      0.97  [0,0,0,13,0,0,0,0,0,0]  [0,0,0,12,0,0,0,0,0,0]  1.00   1.00
13  3.00    1.00         0.97      0.97  [0,0,0,14,0,0,0,0,0,0]  [0,0,0,13,0,0,0,0,0,0]  1.00   1.00
14  3.00    1.00         0.97      0.97  [0,0,0,15,0,0,0,0,0,0]  [0,0,0,14,0,0,0,0,0,0]  1.00   1.00
15  3.00    1.00         0.97      0.97  [0,0,0,16,0,0,0,0,0,0]  [0,0,0,15,0,0,0,0,0,0]  1.00   1.00
16  3.00    1.00         0.97      0.97  [0,0,0,17,0,0,0,0,0,0]  [0,0,0,16,0,0,0,0,0,0]  1.00   1.00
17  3.00    1.00         0.97      0.97  [0,0,0,18,0,0,0,0,0,0]  [0,0,0,17,0,0,0,0,0,0]  1.00   1.00
18  3.00    1.00         0.97      0.97  [0,0,0,19,0,0,0,0,0,0]  [0,0,0,18,0,0,0,0,0,0]  1.00   1.00
19  3.00    0.00         0.97      0.97  [0,0,0,20,0,0,0,0,0,0]  [0,0,0,18,0,0,0,0,0,0]  0.00   0.00
... (980 more periods)
episode total = 964.00   reward total = 964.00   periods = 1000
```

Gaussian branch (`--instance gaussian`) — same seed, same stream identity (the
two candidates share the `payout` slot's id, so selecting `gaussian` swaps the
recipe, not the stream): nature drew $\mu_3 = -1.21$ while the best arm has mean
0.98 (so always pulling arm 3 accrues regret ≈ 2.19/round); payouts are
$N(-1.21, 1)$ draws:

```step7b
python -m mdp_ir.interpreter mab/mab_schema.json --instance gaussian --decision arm=3 --episode-seed 3
```

```
mab v0.4  episode_seed=3 instance=gaussian
 t   arm  reward  mean_pulled  opt_mean                   pulls                     payouts   pull  total
 0  3.00    0.21        -1.21      0.98   [0,0,0,1,0,0,0,0,0,0]    [0,0,0,0.21,0,0,0,0,0,0]   0.21   0.21
 1  3.00   -0.46        -1.21      0.98   [0,0,0,2,0,0,0,0,0,0]   [0,0,0,-0.25,0,0,0,0,0,0]  -0.46  -0.46
 2  3.00    0.66        -1.21      0.98   [0,0,0,3,0,0,0,0,0,0]    [0,0,0,0.41,0,0,0,0,0,0]   0.66   0.66
 3  3.00   -1.52        -1.21      0.98   [0,0,0,4,0,0,0,0,0,0]   [0,0,0,-1.11,0,0,0,0,0,0]  -1.52  -1.52
 4  3.00   -1.02        -1.21      0.98   [0,0,0,5,0,0,0,0,0,0]   [0,0,0,-2.13,0,0,0,0,0,0]  -1.02  -1.02
 5  3.00   -2.59        -1.21      0.98   [0,0,0,6,0,0,0,0,0,0]   [0,0,0,-4.72,0,0,0,0,0,0]  -2.59  -2.59
 6  3.00   -0.32        -1.21      0.98   [0,0,0,7,0,0,0,0,0,0]   [0,0,0,-5.04,0,0,0,0,0,0]  -0.32  -0.32
 7  3.00   -1.48        -1.21      0.98   [0,0,0,8,0,0,0,0,0,0]   [0,0,0,-6.52,0,0,0,0,0,0]  -1.48  -1.48
 8  3.00   -1.25        -1.21      0.98   [0,0,0,9,0,0,0,0,0,0]   [0,0,0,-7.77,0,0,0,0,0,0]  -1.25  -1.25
 9  3.00   -0.73        -1.21      0.98  [0,0,0,10,0,0,0,0,0,0]   [0,0,0,-8.50,0,0,0,0,0,0]  -0.73  -0.73
10  3.00   -1.11        -1.21      0.98  [0,0,0,11,0,0,0,0,0,0]   [0,0,0,-9.61,0,0,0,0,0,0]  -1.11  -1.11
11  3.00   -2.76        -1.21      0.98  [0,0,0,12,0,0,0,0,0,0]  [0,0,0,-12.37,0,0,0,0,0,0]  -2.76  -2.76
12  3.00   -2.84        -1.21      0.98  [0,0,0,13,0,0,0,0,0,0]  [0,0,0,-15.20,0,0,0,0,0,0]  -2.84  -2.84
13  3.00   -1.15        -1.21      0.98  [0,0,0,14,0,0,0,0,0,0]  [0,0,0,-16.36,0,0,0,0,0,0]  -1.15  -1.15
14  3.00   -2.60        -1.21      0.98  [0,0,0,15,0,0,0,0,0,0]  [0,0,0,-18.96,0,0,0,0,0,0]  -2.60  -2.60
15  3.00   -2.11        -1.21      0.98  [0,0,0,16,0,0,0,0,0,0]  [0,0,0,-21.06,0,0,0,0,0,0]  -2.11  -2.11
16  3.00   -2.33        -1.21      0.98  [0,0,0,17,0,0,0,0,0,0]  [0,0,0,-23.40,0,0,0,0,0,0]  -2.33  -2.33
17  3.00   -1.72        -1.21      0.98  [0,0,0,18,0,0,0,0,0,0]  [0,0,0,-25.11,0,0,0,0,0,0]  -1.72  -1.72
18  3.00   -2.15        -1.21      0.98  [0,0,0,19,0,0,0,0,0,0]  [0,0,0,-27.27,0,0,0,0,0,0]  -2.15  -2.15
19  3.00   -2.53        -1.21      0.98  [0,0,0,20,0,0,0,0,0,0]  [0,0,0,-29.79,0,0,0,0,0,0]  -2.53  -2.53
... (980 more periods)
episode total = -1219.62   reward total = -1219.62   periods = 1000
```

Each block is preceded by the `step7b` fence that produced it, run from
`examples/`, the parent of `mab/`, and pasted verbatim — which is what lets
`docs.restatement_current` re-run the command and diff it.

> **Corrected 2026-08-18.** Both blocks previously carried three rows and an
> ellipsis, and the Bernoulli one was attributed to the bare
> `mab_schema.json` invocation. The `payout` slot's catalog `default` is
> `gaussian`, so that command has rendered the *Gaussian* branch since this
> domain was promoted — the labelled block and the command below it had
> disagreed the whole time. The numbers were right; the invocation named to
> reproduce them was not, which is exactly the drift a fingerprint check
> cannot see (the `mdp` token never moved) and the reason step 7b now declares
> the command next to its output.

# mab — case-close playbook

Digest per `ESCALATION_LOG_GUIDE.md` §10, written at campaign close
(2026-08-11, revised to the v0.6.0 schema 2026-08-13; re-checked against the
guide at the v0.10.0 pin on 2026-09-02 — §10's entry shape, lever layers and
`FM`/`MR` sections are unchanged, so no revision was owed and none was made).
The guide's §-numbers are cited from case logs and never renumber. Entries carry what is
needed to *reuse* the experience and **cite `ESCALATION.md` for everything
else** — the log is beside this file; nothing is copied out of it.

**Structure class** — sequential choice among K exchangeable stochastic
options under a per-episode latent, exactly one option observed per period,
finite undiscounted horizon, with a computable per-episode clairvoyant oracle.

**Protocol for every number below**, unless an entry says otherwise: Gaussian
branch `gauss_K10_T1000` (K=10, T=1000), 8192 shared episode seeds (common
random numbers), **stochastic** evaluation, `reward_mean` in payout units.
Oracle 1530.75 · thompson 1462.38 · ucb1 1440.79 · greedy 1015.69 · random
−2.51. Training-seed sd ≈ 16.7, ~2.7× the eval SE, so single-seed contrasts
below ~50 points are not established (#E21).

---

## Lever entries

### LV1 — hand the agent the posterior, not the raw sufficient statistic
```
context:      first design axis, before any architecture or HP work. Both
              encodings are lossless and identically shaped (21 features).
symptom:      with counts+totals the agent must learn the conjugate update
              itself and plateaus far below the reference; with posterior
              (mean, sd) it trains to a usable policy at the same budget.
diagnosis:    encoding, not capacity — confirmed training-free by distilling
              the reference policy into the class (it fits; see LV2).
prescription: gym-obs — per-option posterior (mean, sd) + time-to-go.
              +458.73 at the original L1 centre (1314.46 vs 927.63).
              SCOPE, and it is the important half: re-measured at the tuned
              centre the gap is only 45.02 (1453.57 vs 1408.55) — **91% of
              what looked like an encoding effect was the HP centre.** What
              survives is qualitative and did hold: the encoding decides
              WHERE exploration lives — a deterministic index vs a randomized
              sampler (#E30) — not how much of it there is.
failed:       raw MLP at the same tuned HP reached only 1060.81 — the
              architecture, not the HP, is what rescued the stats encoding.
evidence:     #E1/#E2 (original), #E29 (reversal), #E30 (readback)
```

### LV2 — equivariance is an optimization prior, not extra capacity
```
context:      after LV1, with a 147.92-point excess regret decomposed into
              two named potentials (symmetry vs exploration) and both bounded
              training-free before any RL spend (see FM1).
symptom:      the observation is a SET of exchangeable per-arm tuples; a flat
              MLP underperforms and its argmax flips under permutation —
              total variation 0.30, 34.5% of argmaxes flip.
diagnosis:    the symmetric solution is representable but not findable.
              Probe: a flat MLP distils to 97.5% of thompson, the index class
              to 99.5% at 4.5k params — so capacity is refuted outright.
prescription: arch — one shared per-arm scorer phi, argmax over arms.
              1314.46 -> 1387.57 (+73.11). SCOPE: `norm_obs` must be OFF —
              VecNormalize normalizes per DIMENSION, so arm slots acquire
              different running statistics and the equivariance breaks.
failed:       every enlargement of the shared scorer lost: cross-arm context
              1310.16, max-pooling 1328.16, attention 1336.65 (all vs
              1387.57). Frame-averaging is a real but eval-only mitigation
              (+38.9) — it patches the symptom, not the optimizer.
evidence:     #E11 (probe), #E14 (crown), #E20 (the three enlargements)
```

### LV3 — drive the entropy bonus to ~0 when exploration IS the task
```
context:      after LV2, once the policy class can carry directed
              exploration. Before this the campaign had spent four rounds
              trying to place exploration in a randomness knob (see `failed`).
symptom:      the policy stays stochastic late; the gap between stochastic
              and argmax play sits at ~450 points after convergence.
diagnosis:    the entropy bonus buys UNDIRECTED randomness in a task whose
              exploration must be state-directed, and the policy cannot stop
              paying while the subsidy exists.
prescription: HP — ent_coef 0.01 -> 3.44e-5, effectively off. 1387.57 ->
              1453.57 ± 4.80 (3 seeds), the largest single HP effect here,
              and the stochastic-argmax gap collapses 450 -> 37: exploration
              MOVED INTO the index rather than vanishing. SCOPE: requires a
              class that can hold a directed index (LV2); removing the
              subsidy with nowhere for exploration to go is untested here.
failed:       belief-potential reward shaping regressed to 1293.03; learned
              temperature heads were refuted four times, once with the anneal
              learned SIGN-INVERTED. Placement, not parametrization, was the
              defect (#E22).
evidence:     #E22 (design), #E23 (verdict)
```

### LV4 — state the credit horizon as COVERAGE of the episode, not as steps
```
context:      fires when an instance family varies episode length. Protocol
              delta: 2048 seeds, cells gauss_K10_T{2000,5000,10000,20000},
              3 training seeds per arm, episodes held constant at 20,000
              (NOT steps — holding steps would strawman the long cells).
symptom:      an HP tuned at one horizon transfers badly to longer ones;
              regret grows disproportionately and seed spread widens.
diagnosis:    gae_lambda fixes an ABSOLUTE credit horizon of 1/(1-lambda)
              steps, but the payoff of exploration materializes over a
              FRACTION of the episode — so one lambda means different things
              at different T. Probe: a ladder with lambda fixed vs lambda
              holding coverage = (1/(1-lambda))/T constant.
prescription: HP — treat coverage = (1/(1-lambda))/T as the DIAGNOSTIC
              LENS, and re-derive lambda whenever mean episode length moves.
              Here that meant raising lambda with T: 20-39% regret cut in the
              mild cells, and at 0.41% coverage 2 of 3 seeds NEVER LEARNED —
              flat at 21-25k reward from 20M steps through 400M, regret 4003
              against 287 for the scaled arm. At constant coverage the deficit
              is FLAT in T (1.9/2.4/3.3/2.2× thompson), not diverging.
              **SCOPE — do not carry a NUMBER out of this entry.** This
              campaign's optima sit at ~8-17% coverage; **game2048**, a
              second campaign in this pipeline, pre-registered its own lambda
              sweep and measured the OPPOSITE direction — 0.98+ losing
              at both scales, its summit at lambda=0.90, optima at ~2-4%
              coverage. Both campaigns confirm the MECHANISM and jointly
              refute any band: an earlier version of this entry proposed
              "~8%, floor ~1%", which upstream rejected on that evidence
              because it would have pushed the other campaign the wrong way.
              Read the two as a bracketing pair. What transfers is the lens
              and the re-derivation trigger, not the number.
failed:       the inherited value (0.98784, tuned at T=1000) as-is — that IS
              the failing arm. A partial draft of this round also
              extrapolated "the gap grows with T" from three cells; the
              fourth refuted it.
evidence:     #E35; the refutation of the numeric band is upstream issue #4's
              disposition, which restated spec §8.6's lambda row as mechanism
              with no numbers.
```

### LV5 — read effects in the decision units, not the logged units
```
context:      applies to any round whose lever is judged from training logs.
symptom:      a real lever looks like a ~1% effect in the selection logs and
              is nearly discarded as noise.
diagnosis:    the logged quantity (reward) carries a large constant offset
              that the decision quantity (regret) does not, so it compresses
              real effects toward zero.
prescription: protocol — report regret and PAIRED CRN differences; never
              quote reward LEVELS across seed blocks. A 20-39% regret effect
              reads as ~1% in reward. Reward is not comparable across blocks
              at all (a +12 block offset was briefly read as a bug); regret
              nearly is. SCOPE: needs a computable per-episode oracle or
              baseline — which is exactly what makes a domain suitable.
failed:       —
evidence:     #E28, #E32 (the block offset), #E35 (the near-miss)
```
*This one bit three times in a single campaign — a tuner's ranking, an HP
effect, and a cross-instance results table — which is the argument for making
it a default rather than a caution.*

### LV6 — periodic checkpoint selection pays; plateau early-stopping does not
```
context:      raised as an objection to the whole reporting protocol: is the
              eval callback picking a good checkpoint or adding noise?
symptom:      unclear selection validity; separately, runs stopping before
              their real peak.
diagnosis:    the checkpoint curve is flat near its top, so a noisy selector
              still lands near the right answer — but a plateau detector
              fires on that same flatness, before the peak.
prescription: protocol — keep periodic selection, RETIRE plateau early
              stopping. Probe: score EVERY checkpoint on a block disjoint
              from both the selection and reporting blocks, paired. The
              256-seed callback picks the ORACLE checkpoint in 5 of 6 runs
              and leaves +1.44 on the table, against the +42.90 that
              selecting at all is worth over the terminal checkpoint —
              selection noise is ~3% of selection value.
              SCOPE: noisy-eval domains whose checkpoint curve is flat near
              its top — which is what makes a noisy selector land near the
              right answer AND makes a plateau detector fire early. A domain
              with a sharp optimum inverts both halves of this.
failed:       plateau early-stopping (previously on by default): it forfeits
              most of that +42.90.
evidence:     #E33
```

### LV7 — harvest a tuner's CENTRE, not its ranking
```
context:      at the close of a 73-trial Optuna study (clean exit, 73/73).
symptom:      the study's ranking does not survive re-scoring at the reporting
              protocol — its top trial's advantage evaporates.
diagnosis:    the tuner scores a different checkpoint (the newest artifact it
              finds, not the run's selected one) on a different, easier seed
              block, so its objective and the reporting protocol are not the
              same function.
prescription: protocol — take the HP CENTRE from the study, then RE-RANK the
              top trials at full protocol before adopting. The study ranked
              trial 36 above trial 6 by 15.52 (1458.67 vs 1443.15, tuner
              units); at protocol that margin is **+20.58 on the tuner's own
              checkpoint (seed-z 0.87) and −0.30 on the canonical one** — i.e.
              **the study's ordering is uninformative here, not inverted.**
              Both sit under this playbook's own ~50-point bar, so the honest
              claim is that the ranking carries no signal, and the campaign
              adopted trial 6 for reasons the study did not supply.
              SCOPE: any tuner differing from the protocol in checkpoint, seed
              block, or eval depth — most of them, silently.
failed:       adopting the study's own ranking, which is the default move.
              (Upstream since fixed the mechanism: `newest_model` is replaced
              by `resolve_model` behind a `--score-checkpoint
              {canonical,final}` contract defaulting to canonical — so a
              periodic checkpoint can no longer outrank the run's own
              artifact. The re-ranking discipline stands; the specific trap
              does not.)
evidence:     #E28
```

### LV8 — distil the net into a rule, then SCORE the rule as a policy
```
context:      Stage-5 readback of the crowned net, after the policy search
              had stopped producing gains.
symptom:      a trained policy whose behaviour resembles a known heuristic.
diagnosis:    the actor IS phi(features) per arm, so read it directly: fit
              the exchange rate c = (dphi/dsd)/(dphi/dmean) over an (m, s)
              grid and publish the linear R^2 and monotonicity so the fit's
              validity is visible (R^2 0.90-0.96 here, monotone in m at 100%).
prescription: interpretation — extract argmax(m + c(ttg)·s) and SCORE the
              fitted rule at full protocol beside the net. 98% action
              agreement; the 12-knot table scores 1459.56 ± 6.33 against the
              net's own 1462.39 — the softmax adds nothing the bonus does not.
              Then re-tuning the single constant gives
              `argmax(pm + 2.5·(ttg/T)^0.15·psd)` = **1486.14**, +22.95 over
              thompson at z=36.8. **The deliverable became a two-constant
              formula with no network** — and it beat the reference the net
              itself lost to. Flat c=2.5 gives 1482.82 at ONE parameter.
              SCOPE — three bounds on that headline, all measured elsewhere in
              this campaign and none of them optional: (i) **Gaussian payouts
              only**, the Bernoulli branch is out of scope so the constant's
              transfer is untested; (ii) **cell-local** — fitted at K=10,
              T=1000, and #E32 shows the level follows c* = 1.204 + 0.286·ln n
              rather than being 2.5, while LV9 gives the horizon at which the
              whole form loses; (iii) **the bar is thompson, not optimal** —
              #E34 V5 records that this campaign has no finite-horizon
              Bayes-optimal reference, so "beats the reference" is not
              "near-optimal".
failed:       parameter-free forms — the best of ~6 families reached 1464.87,
              21.27 short. The constant carries real information.
evidence:     #E24 (readback), #E26 (the tuned rule), #E27 (parameter-free)
```

### LV9 — a fixed-quantile index is inconsistent: measure its envelope before shipping
```
context:      after LV8 produced a rule that beats the reference at the tuned
              cell, and after #E32 confirmed it beats thompson in all 21
              cells of a K × n grid. Protocol delta: grid census via
              `mab_grid_probe.py`, horizons swept to 20× the tuned one.
symptom:      a tuned index rule beating an asymptotically-optimal reference
              — which should be impossible in the limit.
diagnosis:    a bonus that never GROWS with accumulated information never
              revisits an abandoned option, so a constant fraction of
              episodes each cost ∝ T: regret linear in T (R² 0.99) against
              thompson's log (R² 0.99).
prescription: — (envelope statement, not a lever). The crossover is at
              T* ≈ 14,000 TOTAL ROUNDS and is K-invariant (14,149 at K=10,
              14,513 at K=20) because both curves scale ~linearly in K, so
              quote the HORIZON, not pulls-per-arm (n* moves 2× across K
              while T* barely moves). Below it use the rule; above it use
              thompson. The level is also cell-local: c* = 1.204 + 0.286·ln n
              (R² 0.743), not the fitted 2.5.
              SCOPE: any tuned index/quantile heuristic whose bonus does not
              grow with information. Measured on gaussian payouts at
              K ∈ {5,10,20}; the crossover's K-invariance is measured at two
              values of K and extrapolated at the third.
failed:       a CONSTANT floor as the repair (−58.1 → −58.2, no effect). The
              repair must grow with t: `log:0.7` turns every crossover back
              into a win, at 8-10 points of short-horizon gain. And every
              repair that works converges toward thompson — which is the
              principled stop condition, and where the search was stopped.
evidence:     #E32 (grid, c* law), #E34 (envelope)
```

### LV10 — one policy over an episode-length grid, and what a scalar HP cannot do
```
context:      the only target here that is a §5.6 GRID rather than a §5.4
              instance. Protocol delta: 40 log-spaced cells, T ∈ [500,10000],
              3 seeds, 190M steps — budget-matched to the specialists it is
              compared against (200M).
symptom:      one policy trained across a horizon family underperforms
              per-horizon specialists, increasingly at the long end.
diagnosis:    two candidates — (a) generality genuinely costs, (b) a SCALAR
              gae_lambda cannot hold LV4's coverage constant across a 20×
              span (33% at T=500, 1.67% at T=10000). Probe: read c back over
              a CROSSED (ratio, horizon) design so the two time arguments are
              not confounded, with two ANALYTIC controls of known opposite
              answer — one ratio-only, one horizon-dependent — to prove the
              probe can DETECT the dependence it tests for.
prescription: gym-obs + arch — feed the episode's own T as an observation and
              let the per-arm scorer take (pm, psd, ttg, T) raw, never their
              ratio (that ratio is the hypothesis under test). At 0.5× the
              tuned horizon it TIES thompson (61.96 vs 58.85 regret, 1.05×) —
              better than any specialist in this campaign. But +10% vs the
              specialist at T=1000 and **+76% at T=10000**. The readback then
              showed it learned a CONSTANT c ≈ 0.85 — no dependence on ttg/T
              (exponent 0.003 vs the rule's 0.15) nor on T (0.003 vs #E32's
              0.286) — i.e. it landed in LV9's inconsistent class, and at
              T=10000 touches only ~6 of 10 arms, the runner-up getting a
              median of 2 pulls out of 10,000.
              **SCOPE — the two candidates are NOT separated**: coverage
              varies 20× across the grid and c does not vary at all, so (b)
              does not explain the shape either. Recorded open.
failed:       hard limits found, not worked around: a grid cannot vary the
              NUMBER of arms (the scorer is count-agnostic, the family
              assertion is not), and no scalar lambda serves a >10× span.
evidence:     #E36 (training), #E37 (readback)
```

### LV11 — a never-tried-best-option tail is usually an OPTIMIZER failure
```
context:      raised from watching interpretation replays, once the crown's
              mean was within 1% of thompson and the residual had no
              explanation.
symptom:      in a fraction of episodes the best arm is NEVER pulled, and
              that tail accounts for the entire remaining gap to thompson.
diagnosis:    H-obj (the objective rationally prefers this) vs H-opt (the
              objective wants the coverage, the optimizer cannot reach it).
              The revisit gradient dies ~20 rounds into the episode, and the
              achieved coverage is set by wherever entropy decay leaves the
              policy when selection snapshots it.
prescription: algo — a forward-KL anchor to thompson's action law,
              dose-ordered in β. Removes the tail 15-22× at no measurable
              cost in mean, which establishes H-opt CONSTRUCTIVELY rather
              than by elimination. SCOPE: a caution rides along — on a
              TRAINED policy the coverage proved worth less than the
              untrained analysis had predicted.
failed:       —
evidence:     #E31
```

### LV12 — licensing a second simulator
```
context:      a grid census (LV9) needed a vectorized simulator over K/T/σ
              that the domain layer does not provide.
symptom:      a probe outgrows the gym, and re-deriving the dynamics risks a
              second source of truth diverging silently.
diagnosis:    aggregate agreement is not evidence — aggregates absorb
              compensating errors, and a single gate is blind wherever its
              reference is degenerate.
prescription: process — permit it under a licence: (i) declare the
              equivalence claimed; (ii) gate that claim in the TEST SUITE,
              not a manual command; (iii) gate COMPONENTS against the
              canonical functions, not just outputs; (iv) derive constants
              from the scenario object so an IR change breaks it loudly;
              (v) quote only PAIRED differences from it, never levels.
              Fault injection then proved the two gates non-redundant:
              dropping σ from the precision is invisible to the exact gate
              (it vanishes at σ=1) and caught only by calibration; an
              off-by-one is caught by both; a dropped prior only by the exact
              one. No single gate sufficed.
              SCOPE: any campaign where a probe outgrows the domain's own
              simulator. The two-gate split is specific to having a canonical
              reference at ONE cell and none elsewhere — a domain with a
              reference everywhere needs only the exact gate.
failed:       an earlier version of this simulator was gated by a manual
              `--part verify` outside the suite, and had silently gone stale
              against the IR (hard-coded K=10, T=1000, σ=1).
evidence:     #E32 (census), README "Simulators — three of them",
              `mab_gate_mutation_check.py`
```

---

## Frame moves

**FM1 — bound-before-build.** Decompose the gap, then BOUND each bucket with
one training-free probe bundle before spending RL budget, with the branch
order pre-decided on the probe's verdict. *Paid off at #E11*: distillation
ceiling + equivariance error + regret profile, pre-registered as "both fire →
arch before exploration; neither → promote tuning". It refuted the capacity
hypothesis for the cost of no training runs, and set the next three rounds.

**FM2 — gap-decomposition-into-buckets.** Convert an unexplained excess into
NAMED competing potentials before designing anything. *Paid off at #E9*: the
147.92 excess regret became "representation/symmetry" vs "exploration", which
is what made FM1's probe designable at all — an unnamed gap admits no probe.

**FM3 — type the root split: coverage (S) vs selection (P).** Ask whether the
children are two halves of ONE problem (S: each owns its protocol, bar and
leaderboard, neither prunable, completing one does not discharge the other) or
competing designs on one leaderboard (P: crown one, prune the rest). *Paid off
at the 2026-07-29 re-typing*: it converted "bernoulli, later" from a parked
option into a permanent obligation rather than a vague "later" — so when the
campaign did close on T1 alone, it had to do so **as an explicit scope
decision, with every claim reported bounded to Gaussian**, instead of quietly
never mentioning the other half. Typing the split is what made the boundary
sayable; an untyped one would simply have gone unstated.

**FM4 — a prune is conditional on the centre it was measured at.** When the
centre moves, previously pruned siblings must be re-tested before the prune is
quoted. *Paid off at #E29*: the stats encoding, pruned at 927.63, reversed at
the tuned centre — and 91% of what had been attributed to the encoding turned
out to be the centre (LV1).

**FM5 — "the criteria agree on the winner" is the acceptance test.** Not "the
numbers are close". *Paid off at #E7*: the pre-fix failure was a RANK
INVERSION between selection and reporting criteria, which closeness-of-numbers
cannot detect.

**FM6 — diagnoses are timeline events, not map state.** Reasoning that
reorders the frontier is event-shaped: put it in the ledger with its reads and
pre-decided branches, and have the frontier CITE it. *Paid off at #E9*, when
the MAP's "potentials" block migrated into a dated entry and stopped being
silently rewritten.

**FM7 — pre-register the reads, including the refutation condition.** *Paid
off repeatedly*: every round that stated its hypotheses in advance produced a
usable verdict even when refuted — two of this campaign's hypotheses were
refuted by their own final cell (#E35 V2, #E36 V3 via #E37). Rounds that
extrapolated from partial results produced claims later cells overturned; those
were recorded in place rather than edited away, so the log shows the reversal.

---

## Modeling rules (lever layer `IR-formalization`)

Trigger-first phrasing is mandatory: the promotion target is the Phase-A
checklist consulted at the classifying-randomness step.

**MR1 — when a decision selects WHICH exogenous stream is read.** (Bandit
shape: K parallel streams, one observed per period; side observations,
duelling and batch pulls all inherit it.) The selector must enter the seed key
(`key_exprs` on the period/event stage); settings-only dependence shares one
variate across all counterfactuals, so the round's luck is fixed before the
choice. *Probe*: realize two selector values at one (period, episode_seed) —
if their difference is deterministic, the streams are coupled. *Why it hides*: **every gate stayed green — nothing prompted a second
look.** The differential matched, the tests passed, and the law the agent sees
is still *correct*; only its factorization across arms is wrong, and no gate
was asking that question. **F1**, cost: voided #E1–#E4 and the entire baseline
table. (The frequently-quoted "matched bit-exactly and 28 tests passed for
three weeks" belongs to **F3**, a different reversal with a different hiding
mechanism — do not attach it here.)

**MR2 — when a latent is hidden but its posterior has a finite sufficient
statistic.** Ask "does the observation history determine the posterior through
a fixed-size summary the state already holds?", not "is the parameter hidden?".
The conjugate families are the tell (Beta–Bernoulli, Normal–Normal known
variance, Gamma–Poisson). If yes the belief-state MDP is fully observed and
memory is not required. *Probe*: name the sufficient statistic; if you can
write it down and it is in the state vector, set `requires_memory=false`. **F2**

**MR3 — when a numerical attribute sets a DIMENSION.** Vector length,
action-index ceiling, horizon, draw size: formalize it as an axis-tagged
scenario constant **even when the case poses exactly one value**, and let the
dimension sites name it. *Trigger to test in the Phase-A interview*: "if
someone later asks whether this result holds at a different size or horizon,
can the IR express that instance today?" If no, and the attribute is numeric,
tag it. *Cost when missed*: the whole class of "does this generalize?"
questions becomes a structural IR change mid-campaign, with both fingerprints
moving and every gate to re-run. Contrast a genuinely structural constant —
one whose change alters the SHAPE of the dynamics, not just a dimension. **F3**

**MR4 — when a size axis exists, check which sites can FOLLOW it.** Not all
can. Here `StateVariable.length` and `Decision.bounds` resolve a constant
name, but `StateVariable.bounds`/`element_bounds` are literal-only — so they
must be sized for the union over every registered instance and re-widened by
hand for each new one. Consequences: the declared bound is correct for no
single instance, and the edit moves the structural fingerprint, converting
"register a bigger cell" from a free operation into a logged structural
change. *Probe*: for each site the axis touches, try writing the constant's
NAME; where the schema rejects it, expect manual maintenance and say so in the
log. **F4**

---

## What did not transfer

Expectations imported from earlier campaigns (rule-8 imported levers, labelled
with their provenance until re-validated here) that failed on this domain.

- **Belief-potential reward shaping — imported from another project**, where
  belief-obs + potential shaping was the instance-A gate-pass pairing. Here it
  REGRESSED the policy to 1293.03, −21.4 against the raw MLP. Shaping a
  quantity the agent already observes adds gradient without adding
  information. The exploration deficit it was meant to fix was real; the
  placement was wrong, and the fix that worked was LV3 (#E15, #E22).
- **Attention over entities — imported from another project's full tier** (multi-head
  + LayerNorm + PMA readout). Here even the *minimal* attention rung lost
  (1336.65 vs 1387.57), so the full tier was never run. That project's claim
  narrows rather than transfers: that boundary was a different selection
  shape, this one is rank-1 vs rank-2 among exchangeable arms (#E20).
- **What DID transfer, recorded so the prior stays calibrated**: equivariant
  nets (a +0.061 arch win there) transferred and took the architecture crown
  (LV2); and the imported *caution* that HP were decisive there — hence a
  null at an inherited centre refutes "rung X at that centre", not rung X —
  was vindicated hard when the stats prune reversed at the tuned centre (FM4,
  LV1).

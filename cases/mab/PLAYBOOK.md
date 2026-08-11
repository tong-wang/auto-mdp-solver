# mab — case-close distillation

Produced per `ESCALATION_LOG_GUIDE.md` §10 at campaign close (2026-08-11).
Three passes: **ledger → lever entries**, **frame-changelog → frame-move
entries**, **ir-changelog → modeling-rule entries**.

This is the local, full-fidelity version: real domain, absolute numbers.
Contributing upstream re-runs a sanitize pass (domain → structure class,
absolute numbers → % over baseline).

**Structure class**: *sequential choice among K exchangeable stochastic options
under a per-episode latent, one observed per period, finite undiscounted
horizon.* Protocol for every number below: 8192 shared episode seeds (CRN),
stochastic evaluation, oracle 1530.75, reference (Thompson) 1462.38 — except
the long-horizon cells, at 2048 seeds with same-path references.

---

## 1. Lever entries (ledger → playbook)

### LV1 — belief features instead of raw sufficient statistics
```
symptom:    the agent must learn a conjugate update from raw counts/totals;
            a belief-parametrized observation of identical shape trains far better
hypothesis: the ENCODING is the bottleneck, not capacity
probe:      distil the reference policy into the policy class — if it fits, the
            class can represent the answer and the deficit is elsewhere
lever:      gym-obs — feed per-option posterior (mean, sd) instead of (count, total)
verdict:    +458.73 (~31%) at the ORIGINAL centre. But re-measured at a re-tuned
            centre the gap collapses to 45.02 — **91% of the apparent encoding gap
            was the HP centre**. What survives is qualitative: the encoding decides
            WHERE exploration lives (a deterministic index vs a randomized
            sampler), not how much there is.
scope:      conjugate-posterior domains. NOT a licence to skip re-measuring a
            pruned encoding after the centre moves (see FM4).
evidence:   #E1/#E2 (original), #E29/#E30 (reversal + readback)
```

### LV2 — equivariance is an optimization prior, not extra capacity
```
symptom:    the observation is a SET of exchangeable per-entity feature tuples;
            a flat MLP underperforms and its argmax flips under permutation
            (TV 0.30, 34.5% argmax flips)
hypothesis: the symmetric solution is REPRESENTABLE but not FINDABLE
probe:      three training-free measurements — distillation ceiling (capacity),
            equivariance error (symmetry), regret profile (where loss accrues);
            plus frame-averaging as an eval-only mitigation (+38.9 alone)
lever:      arch — one shared per-entity scorer phi, argmax over entities
verdict:    capacity REFUTED (flat MLP distils to 97.5% of the reference; the
            index class to 99.5% at 4.5k params). The equivariant head then gained
            +73 over the raw MLP — and every ENLARGEMENT of it LOST:
            cross-arm context -77, max-pool -59, attention -51.
scope:      any domain whose observation is a set of exchangeable entities.
            Enlarging an already-equivariant scorer is the failure mode to expect.
            **Interacts with obs normalization**: per-dimension normalizers
            (e.g. VecNormalize) give entity slots different running statistics and
            silently BREAK permutation equivariance — must be off.
evidence:   #E11 (probe), #E14 (crown), #E20 (the three enlargements)
```

### LV3 — drive the entropy bonus to ~0 when exploration IS the task
```
symptom:    the policy stays stochastic late in the episode; a large gap between
            stochastic and argmax play (450 points) persists after convergence
hypothesis: the entropy bonus is buying UNDIRECTED randomness in a task that
            needs DIRECTED exploration — it pays for the wrong thing and the
            policy cannot stop paying
probe:      sweep ent_coef down three orders of magnitude; watch the
            stochastic-minus-argmax gap, not the mean
lever:      HP — ent_coef ~3.4e-5, i.e. effectively off
verdict:    new best trained config (+66 over the previous crown). The argmax gap
            collapsed 450 -> 37: exploration MOVED INTO the index rather than
            disappearing. Two independent subsidy-removals (this and a
            time-weighted entropy price) converge to the same place.
scope:      requires a policy class that CAN carry directed exploration (LV2's
            index head). Removing the subsidy without somewhere for exploration to
            go is untested and probably harmful.
evidence:   #E22 (design), #E23 (verdict)
```

### LV4 — state the credit horizon as COVERAGE, not as steps
```
symptom:    an HP tuned at one episode length transfers badly to longer ones;
            regret grows disproportionately and training-seed spread widens
hypothesis: gae_lambda fixes an ABSOLUTE credit horizon (1/(1-lambda) steps),
            while the payoff of exploration materializes over a FRACTION of the
            episode — so the same lambda means different things at different T
probe:      a horizon ladder with two arms: lambda held fixed vs lambda holding
            coverage = (1/(1-lambda))/T constant
lever:      HP — set coverage ~8% of mean episode length; floor ~1%
verdict:    20-39% regret cut in the mild cells. Below ~1% coverage it stops being
            a rate and becomes a THRESHOLD: at 0.41%, **2 of 3 seeds never learn at
            all** (flat for 20x the budget that suffices at 8%), giving 31x the
            reference's regret. At constant coverage the deficit is FLAT in T
            (1.9/2.4/3.3/2.2x), not diverging.
scope:      any domain whose instance family varies episode length. The threshold
            (~1%) is measured on a domain where all reward is deferred; densely
            rewarded domains should tolerate lower coverage.
evidence:   #E35 (24-run controlled ladder, K=10, T in {2,5,10,20}k, 3 seeds)
```

### LV5 — read effects in the decision units, not the logged units
```
symptom:    a lever looks like a ~1% effect in the training logs and is nearly
            discarded as noise
hypothesis: the LOGGED quantity (reward) carries a large constant offset that the
            DECISION quantity (regret) does not, so it compresses real effects
probe:      recompute the identical contrast in regret against the same oracle
lever:      protocol — report regret and PAIRED CRN differences; never quote
            reward levels across seed blocks
verdict:    a 20-39% regret effect reads as ~1% in reward. Separately: reward_mean
            is NOT comparable across seed blocks (a +12 block offset was mistaken
            for a bug) while regret nearly is.
scope:      any domain with a computable per-episode oracle or baseline. This bit
            the campaign THREE times — a tuner ranking, an HP effect, and a
            cross-cell results table — which is the argument for making it a
            default rather than a caution.
evidence:   #E28, #E32 (block offset), #E35 (the near-miss)
```

### LV6 — periodic checkpoint selection pays; plateau early-stopping does not
```
symptom:    unclear whether the eval callback picks the right checkpoint or just
            adds noise; separately, runs stop before their real peak
probe:      score EVERY checkpoint of every run on a seed block disjoint from both
            the selection block and the reporting block; compare the oracle
            checkpoint against the callback's pick, paired
lever:      protocol — keep periodic selection; RETIRE plateau early-stopping
verdict:    a 256-seed callback picks the ORACLE checkpoint in 5 of 6 runs and
            leaves +1.44 on the table, against the +42.90 that selecting at all is
            worth over the terminal checkpoint. Selection noise is ~3% of
            selection value. Plateau detection fires before the true peak.
scope:      noisy-eval domains where the checkpoint curve is flat near its top.
evidence:   #E33
```

### LV7 — harvest a tuner's CENTRE, not its ranking
```
symptom:    the tuner's best trial loses to a lower-ranked trial when both are
            re-evaluated at the reporting protocol
hypothesis: the tuner scores a different checkpoint (its own last, not the
            selected one) on a different, easier seed block
probe:      re-evaluate the top-N trials at full protocol depth before adopting
lever:      protocol — take the HP centre from the study, re-rank at protocol
verdict:    the study's top trial was refuted; the harvested trial became the
            crown and remains the config of record.
scope:      any tuner whose objective differs from the reporting protocol in
            checkpoint, seed block, or eval depth — i.e. most of them by default.
evidence:   #E28 (73/73-trial study, clean exit)
```

### LV8 — distil the net into a rule, then SCORE the rule
```
symptom:    a trained policy that behaves like a known heuristic
probe:      treat the actor as phi(features) per entity; fit the exchange rate
            c = (dphi/dsd)/(dphi/dmean) over a (mean, sd) grid; report linear R^2
            and monotonicity so the fit's validity is visible
lever:      interpretation — extract argmax(mean + c(t)*sd) and score the FITTED
            RULE as a policy at full protocol, alongside the net
verdict:    98% action agreement; on the deep-dive seed the fitted 12-knot table
            scored 1459.56 +/- 6.33 against the net's own 1462.39 — the softmax
            contributes nothing the bonus does not. Re-tuning the single constant
            then produced a rule that BEAT both the net and the reference
            (1486.14, +22.95 over the reference, z=36.8). **The deliverable became
            a 2-constant formula with no neural network.**
scope:      index-shaped policies over exchangeable entities. The readback is also
            the cheapest route to a deployable artifact.
evidence:   #E24 (readback), #E26 (the tuned rule)
```

### LV9 — a fixed-quantile index is inconsistent: find the envelope before shipping
```
symptom:    a tuned index rule beats an asymptotically-optimal reference at the
            horizon it was tuned on
hypothesis: a bonus that does not GROW with accumulated information never revisits
            an abandoned option, so a fixed fraction of episodes carry a cost
            proportional to the horizon -> linear regret vs the reference's log
probe:      sweep the horizon ~20x; fit regret ~ a + b*T against a + b*log T
lever:      — (this is an envelope statement, not a lever)
verdict:    crossover at T* ~ 14,000 total rounds, K-INVARIANT (14,149 at K=10,
            14,513 at K=20) because both curves scale ~linearly in K. The repair
            must GROW with t (a log-form floor works; a constant floor does not),
            and **every repair that works converges to the reference** — which is
            the principled stop condition for the policy search.
scope:      any tuned index/quantile heuristic. Quote the HORIZON, not
            observations-per-option: the latter varies 2x across K while T* barely
            moves.
evidence:   #E34, mechanism confirmed independently at #E37
```

### LV10 — one policy over a horizon grid: what a scalar HP cannot do
```
symptom:    a single policy trained over a family of episode lengths underperforms
            per-length specialists, increasingly at the long end
hypothesis: (a) generality genuinely costs, and/or (b) a SCALAR credit-horizon HP
            cannot hold LV4's coverage constant across the family
probe:      readback c over a CROSSED (ratio, horizon) design so the two arguments
            are not confounded, with two analytic controls of known opposite
            answer — one ratio-only, one horizon-dependent — to prove the probe can
            DETECT the dependence it is testing for
lever:      gym-obs + arch — feed the episode's own length as an observation and
            let the per-entity scorer take (features, time-to-go, horizon) raw
verdict:    it works, and at the SHORT end it ties the reference (1.05x) — better
            than any specialist in the campaign. But the tax grows: +10% vs the
            specialist at 1x, +76% at 20x, budget-matched. The readback then showed
            the net learned a CONSTANT exploration coefficient (~1/3 of optimal, no
            dependence on either time argument), i.e. LV9's inconsistent class.
scope:      **the two hypotheses are NOT separated by this evidence.** Coverage
            varies 20x across the grid while the learned constant does not, so
            (b) does not explain the shape either. Recorded as open.
            Hard limits found: one grid cannot vary the number of entities
            (the scorer is entity-count-agnostic but the family assertion is not),
            and no scalar lambda serves a >10x horizon span.
evidence:   #E36 (training), #E37 (readback)
```

### LV11 — a never-tried-best-option tail is usually an OPTIMIZER failure
```
symptom:    in a fraction of episodes the best option is never tried at all, and
            that tail accounts for the entire remaining gap to the reference
hypothesis: H-obj (the objective rationally prefers this) vs H-opt (the objective
            wants coverage but the optimizer cannot reach it)
probe:      ask whether the OBJECTIVE prices the coverage — then try to buy the
            coverage directly and see if the mean survives
lever:      algo — a forward-KL anchor to a reference action law, dose-ordered in
            its coefficient
verdict:    H-opt, shown constructively: the anchor removes the tail 15-22x at no
            measurable cost in mean. The gradient that would fund revisiting dies
            ~20 rounds into the episode, and the achieved coverage is set by
            wherever entropy decay leaves the policy when selection snapshots it.
scope:      a caution rides along — on a TRAINED policy, coverage turned out to be
            worth less than the untrained analysis predicted.
evidence:   #E31
```

### LV12 — governing a second simulator
```
symptom:    a probe needs a vectorized or generalized simulator the domain layer
            does not provide, and re-deriving it risks a second source of truth
lever:      process — permit it under a licence: (i) declare the equivalence it
            claims; (ii) gate that claim in the TEST SUITE, not a manual command;
            (iii) gate COMPONENTS against the canonical functions, never only
            aggregate outputs, which absorb compensating errors; (iv) derive
            scenario constants from the scenario object so an IR change breaks it
            loudly; (v) quote only PAIRED differences from it, never levels
verdict:    fault injection proved a two-gate design non-redundant — dropping a
            scale parameter was invisible to the exact gate (it vanishes at the
            base value) and caught only by the calibration gate; an off-by-one was
            caught by both; a dropped prior only by the exact one. No single gate
            sufficed.
scope:      any campaign where a probe outgrows the domain's own simulator.
evidence:   README "Simulators — three of them"; mab_gate_mutation_check.py
```

---

## 2. Frame-move entries (changelog → playbook)

**FM1 — bound-before-build.** Before spending RL budget on a decomposed gap, run
one training-free probe bundle that BOUNDS each bucket, and pre-decide the branch
order on its verdict. Here: distillation ceiling (capacity), equivariance error
(symmetry), regret profile (where loss accrues) — pre-registered as "both fire →
arch before exploration; neither → promote tuning". It refuted capacity for the
price of no training runs.

**FM2 — gap-decomposition-into-buckets.** Convert an unexplained excess (147.92
regret) into NAMED competing potentials — representation/symmetry vs exploration —
and arbitrate with one instrument rather than a sequence of RL runs. The named
buckets are what make a probe designable.

**FM3 — type the root split: coverage (S) vs selection (P).** Ask whether the
children are two halves of ONE problem (S: each owns its protocol, bar and
leaderboard; neither is prunable; completing one does not discharge the other) or
competing designs on one leaderboard (P: crown one, prune the rest). Getting this
wrong hides a permanent obligation as a parked option. This campaign closed with
its S2 debt unpaid and had to say so on every claim.

**FM4 — a prune is conditional on the centre it was measured at.** When the centre
moves (a tuning round, an architecture change), previously pruned siblings must be
re-tested before the prune is quoted. Here the pruned encoding REVERSED at the new
centre, and 91% of what had been attributed to the encoding turned out to be the
centre.

**FM5 — "the criteria agree on the winner" is the acceptance test.** Not "the
numbers are close". When a selection criterion and a reporting criterion disagree,
the failure mode is a RANK INVERSION, which closeness-of-numbers cannot detect.

**FM6 — diagnoses are timeline events, not map state.** Reasoning that reorders
the frontier is event-shaped: put it in the ledger with its reads and its
pre-decided branches, and have the frontier CITE it. Keeping it in the map loses
the dating and the "what would have changed my mind".

**FM7 — pre-register the reads, including the refutation condition.** Every round
here that stated its hypotheses in advance produced a usable verdict even when
refuted (two of this campaign's hypotheses were refuted by their own last cell).
Rounds that extrapolated from partial results produced claims that later cells
overturned — recorded in place rather than edited away.

---

## 3. Modeling-rule entries (ir-changelog → playbook, layer `IR-formalization`)

Trigger-first phrasing is mandatory: the promotion target is a Phase-A checklist
consulted at the classifying-randomness step.

**MR1 — when a decision selects WHICH exogenous stream is read.** (Bandit shape: K
parallel streams, one observed per period; side observations, duelling and batch
pulls all inherit it.) The selector MUST enter the seed key (`key_exprs` on the
period/event stage). Settings-only dependence shares one variate across all
counterfactuals, so the round's luck is fixed before the choice is made — the law
the agent sees is still correct, which is why this survives casual review.
*Probe*: realize two selector values at one (period, episode_seed); if their
difference is deterministic, the streams are coupled. (F1 — cost: voided four
ledger entries and the entire baseline table.)

**MR2 — when a latent is hidden but its posterior has a finite sufficient
statistic.** Ask "does the observation history determine the posterior through a
fixed-size summary the state already holds?", NOT "is the parameter hidden?". The
conjugate families are the tell (Beta–Bernoulli, Normal–Normal known variance,
Gamma–Poisson). If yes, the belief-state MDP is fully observed and memory is not
required. *Probe*: name the sufficient statistic; if you can write it down and it
is in the state vector, set `requires_memory=false`. (F2)

**MR3 — when a numerical attribute sets a DIMENSION.** Vector length, action-index
ceiling, horizon, draw size: formalize it as an axis-tagged scenario constant
**even when the case poses exactly one value**, and let the dimension sites name
it. *Trigger to test in the Phase-A interview*: "if someone later asks whether
this result holds at a different size or horizon, can the IR express that instance
today?" If no, and the attribute is numeric, tag it. Cost of not doing it: the
whole class of "does this generalize?" questions becomes a structural IR change
mid-campaign, with both fingerprints moving and every gate to re-run. Contrast a
genuinely structural constant: one whose change alters the SHAPE of the dynamics,
not just a dimension. (F3)

**MR4 — when a size axis exists, check which sites can FOLLOW it.** Not all can.
Here `length` and decision `bounds` resolve a constant name, but state-variable
`bounds`/`element_bounds` are literal-only — so they must be sized for the union
over every registered instance and re-widened by hand for each new one. Two
consequences: the declared bound is correct for no single instance, and the edit
moves the structural fingerprint, converting "register a bigger cell" from a free
operation into a logged structural change. *Probe*: for each site the axis
touches, try writing the constant's NAME; where the schema rejects it, expect
manual maintenance and say so in the log. (F4)

---

## 4. What did NOT transfer

Recorded because a playbook of only-successes teaches the wrong prior.

- **Belief-potential reward shaping** regressed the policy (-21.4). Shaping a
  quantity the agent already observes adds gradient without adding information.
- **Learned temperature heads — refuted four times.** A per-state temperature
  learned the schedule with the sign INVERTED. The defect was PLACEMENT, not
  parametrization: exploration belongs in the index, not in a randomness knob
  bolted beside it.
- **Enlarging an equivariant scorer** — cross-arm context, max-pooling, attention
  — lost every time (LV2).
- **Parameter-free forms of the distilled rule**: the best of ~6 families came
  21.27 short of the two-constant fit. The constant is carrying real information,
  not overfitting.

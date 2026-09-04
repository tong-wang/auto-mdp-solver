# PLAYBOOK — adi_flex

Operator-level lessons, in the guide-§10 digest form. Started 2026-08-31 (the
campaign is not closed; entries are added as they are paid for, not at the end).
Converted to the full §10 shape on 2026-09-04 at the head's adoption (`#E25`):
four lever entries, the frame moves, the modeling rules, and what did not
transfer. Everything not copied here is cited by its `#E` / `F` number.

**Structure class** — finite-horizon periodic-review inventory with advance
demand information and flexible (early) delivery: one discrete **order
quantity** per period under a fixed ordering cost (so the optimum is (s,S) —
a mass at zero plus a magnitude), and, where the allocation is live, per-class
**protection levels** chosen at the same pre-demand information set. References:
an exact DP on the homogeneous branch; an AP relaxation lower bound and the
paper's PL(σ) heuristic bar on the heterogeneous boards.

**Protocol for every number below**, unless an entry says otherwise: 8192
shared episode seeds from `first_seed 0` (common random numbers across arms),
deterministic-argmax record eval, cost in the paper's units (lower is better).
RL cells are **6-seed means** with the across-seed SE (seeds 21–26; `het_exp4`'s
`vec` arm 11–16); paired differences are on the shared seeds. References:
`homog_L0_T2` DP 685.2363; `het_exp4` PL(σ) 341.26, AP 336.19; `het3_exp2`
PL(σ) 347.1749, AP 332.8232.

---

## Lever entries

### LV1 — a discrete *quantity* decision gets an ordinal head, zero-inflated when "do nothing" is a real mass

```
context:      het_exp4, after the action encoding (order_protection, F51/#E11)
              and two rounds of hp tuning (#E16, 820 trials) had left RL +1.64
              above the PL(σ) bar; observation (vec/vec_mip) a dead heat, the
              order encodings that were tried worth nothing (#E11), the
              order-up-to encoding withdrawn (F30). The order head was a plain
              categorical over quantities 0..107.
symptom:      the residual is ORDER MAGNITUDE, and it is diffuse: |rl − y*| =
              1.13 units with a systematic +0.57 over-order, spread over
              periods 5–7; the trigger is learned (misses 0.66%), the
              protection side is worth +0.10, and further tuning reorders on
              confirm instead of improving (#E21, #E22 reading 4). A ~1-unit
              signal spread across ~5 adjacent logits is below PPO's
              resolution — the symptom of a head with no notion of adjacency.
diagnosis:    a categorical head over an ORDERED quantity splits one gradient
              signal into independent per-category logits; the location cannot
              sharpen. And near the (s,S) boundary the target is BIMODAL — "0
              or ≈ S−u" — so a plain unimodal head cannot express the trigger
              either: the zero needs its own mass. Confirmed training-free by
              the head's gate (exact log-probs vs a reference, a point mass
              expressible for every quantity, trigger and magnitude gradients
              separable, masking and passthrough intact), then by a 2×2 that
              showed the effect additive with norm_obs, so no confound.
prescription: arch. Replace the per-quantity logits with three numbers —
              P(q = 0) = sigmoid(t) (the trigger, zero-inflation), a
              discretized-Gaussian location μ and width τ over q ≥ 1 — and
              EXPAND them back to logits, so the action space, the mask, the
              entropy and the algorithm are untouched (MaskablePPO, `a0`'s
              constraint, still hold). A policy-architecture knob like
              net_arch; minted as an `a` id (a1), level L4(hp+gym+arch).
              Effect, paired against the categorical head at identical hp and
              seeds: het_exp4 −8.3/−9.4 at L1 hp and −0.85 ±0.15 best-vs-best
              at tuned hp; homog_L0_T2 −3.40 ±0.54 / −3.56 ±0.96 at the crowned
              hp; het3_exp2 −1.13 ±0.36 / −1.85 ±0.37 at the crowned hp — six
              contrasts, all negative, t −3.1..−6.3. Crowned on every board:
              homog 691.78 ±0.15 = 100.95% of the exact DP (from 101.4%), het3
              337.74 ±0.19 = 97.28% of the bar, het_exp4 342.48 ±0.26 (the bar
              still stands there). Mechanism as prescribed: the order policy
              becomes a 95.7% replica of y*, |d| 1.13 → 0.30 with the bias
              gone, seed sd collapses (0.16 at the derived rung), the vec/
              vec_mip gap vanishes, and untuned-vs-tuned shrinks ~10 → ~2 —
              the head does most of what 820 trials of tuning did.
              SCOPE: (i) for a decision that is a QUANTITY whose neighbours are
              substitutes — order size, batch, allocation amount — not for a
              nominal choice; (ii) zero-inflate only when the problem has a
              genuine "do nothing" mass (a fixed cost ⇒ an (s,S) trigger);
              otherwise the plain ordinal body; (iii) leave a head categorical
              where a smoothness prior points the wrong way — here the σ heads,
              +0.10 at stake and a terminal-boundary effect (#E14); (iv) the
              gain shrinks as the categorical baseline is better tuned (−8..−9
              at L1 → −0.8..−1.0 at tuned) but never reached zero on three
              boards; (v) re-tune the hp centre FOR the head: it wants a 4096
              rollout and λ ≈ 0.995 where the categorical wanted 512–2048 and
              0.81–0.95 (h6 vs h2), worth −1.33 on vec and nothing on vec_mip;
              (vi) gate it training-free before any run — an exactness error in
              the expanded logits would be invisible in a learning curve.
failed:       on the same symptom — more hp/budget (stage 2, 820 trials: 342.90,
              +1.64 and a ranking that inverts on confirm); alternative order
              encodings (target_ip/target_mip: three of four contrasts
              positive, #E11); the MIP observation (a dead heat); a heuristic
              terminal-σ patch stacked on the best seed (341.30, a statistical
              tie with the bar, not a policy).
evidence:     #E21 (the dissection), #E22 (the head on het_exp4, the gate, the
              mechanism), #E23 (homog_L0_T2, tuning for the head), #E24
              (het3_exp2 stacks), #E25 (adoption, a1/h6), F57 (a shipped
              artifact pins its action widths).
```

### LV2 — a decision bounded by an availability is encoded as a policy PARAMETER, not a quantity — the constraint is dissolved, not enforced

```
context:      het_exp4, the first heterogeneous board, where the allocation
              is live: each period the agent orders (a quantity into a cap)
              and then allocates on-hand stock across demand classes (a
              quantity ≤ what is available). Both were quantities; the
              allocation's feasible set is state-dependent, so it needed a
              per-step mask, and the mask left 3.01 of 108 actions legal on
              average (F51). RL sat at 103.8–104.6% of the PL(σ) bar with
              seed sd 5–17 at the derived rung (#E9, #E11).
symptom:      a sequential two-phase quantity action: 1 + n_alloc agent steps
              per period, a mask the algorithm must be able to read
              (MaskablePPO forced), collapsed cells (2 of 6 seq_mask cells
              on het3 fell to a constant policy: 834.86, 934.80 — #E17), and
              a learned allocation that the reference heuristic already
              solves in closed form. Independently, both trained arms held
              the ORDER almost fixed where the exact DP swings it (sd 0.6/1.3
              vs 5.7 — F30).
diagnosis:    a quantity is the wrong coordinate for a flow bounded by an
              availability. Re-parametrize the decision as the PARAMETER of
              the rule the quantity would follow — a base-stock LEVEL instead
              of an order, a PROTECTION LEVEL (how much to withhold from the
              near class for the far one) instead of an allocation, a ship-
              up-to TARGET instead of a shipment — and every point of the
              action box is feasible by construction: the constraint is
              dissolved (IR feasibility_strategy = "none"), no mask, one
              agent step per period, and the agent's output is already in
              the units the heuristic is written in. The paper's own §4.2
              heuristics ARE protection-level rules, so the encoding hands
              the agent the heuristic's coordinate and lets it learn the
              constants.
prescription: gym-action. `order_protection`: (order quantity, σ_1..σ_{T_dl−1})
              chosen together at the pre-demand information set; a priority
              cascade turns the σ's into the allocation. Effect @ protocol:
              the protection encoding of the ALLOCATION is worth −4.65 (vec)
              / −0.47 (vec_mip) at the derived rung against the masked
              quantity, a complete 2×2 with the order encoding (#E11); best
              cell 352.47 ±0.27 = 103.3% of the bar from 104.6%; seed spread
              collapses ~20× (0.88 vs 17.17); untuned PPO nearly matches the
              tuned version (L0 → best rung +1.76/+1.33 against +19..+76 under
              the other encodings), i.e. the encoding does the work tuning was
              doing. Replicates on het3_exp2: best at EVERY rung, zero
              collapsed cells against seq_mask's two (#E17). Every crowned
              artifact on every board is in this encoding (#E18, #E25).
              SCOPE: (i) the ORDER half did NOT benefit from the same move —
              an order-up-to target fixes the magnitude while the trigger
              stays state-dependent, and with a fixed ordering cost (K=100)
              the optimum is (s,S): three of four target-order contrasts were
              positive (#E11), and F30 had withdrawn the order-up-to mode
              before a run on exactly this argument. Dissolve the constraint
              on the decision that IS a rule parameter; keep a quantity (or,
              per LV1, an ordinal quantity) where the optimum needs a trigger.
              (ii) What the encoding buys is LEARNABILITY, not a better
              allocation rule: with the order held at the AP policy and only
              the protection learned, PPO converges to 341.2645 against
              PL(σ)'s 341.2600 — the entire remaining gap on this board is in
              the ORDERING (#E13). (iii) The price is interpretability per
              action, see P1/F55: a parameter acts through the constraint it
              shapes, so where the constraint is slack the action is
              unidentified (σ binding in 25.7%/8.7% of events, pinned in
              6.7%/3.1%) — audit identification before reading structure off
              such an arm. (iv) Where the allocation is NOT live (the
              homogeneous branch) the encoding is behaviourally the quantity
              encoding — no gain, no loss (#E23).
failed:       on the same symptom — a per-step feasibility mask over the
              quantity (the seq_mask baseline itself); an order-up-to target
              for the ORDER (withdrawn F30; run as target_ip/target_mip in
              #E11, worthless); the constant-target degenerate member that
              F30 used as evidence (2893 — one bad member is not a class).
evidence:     F30, F51 (the rule), #E10 (built and gated before any run),
              #E11 (the 2×2 on het_exp4), #E13 (the isolation), #E17
              (het3_exp2), F55 (the identification hazard), #E25 (every crown
              is in this encoding).
```

### LV3 — an action framed as a policy parameter is only *partly identified*: audit identification before reading structure

```
context:      the readback (#E19) of het_exp4's crowned artifacts under
              order_protection (LV2): the protection levels σ_1, σ_2 are the
              action, and a parameter acts THROUGH the constraint it shapes.
              Where the constraint does not bind the parameter has no effect on
              the trajectory, the gradient is flat, and the network is free to
              emit anything. A quantity encoding has no such region: ship 7 and
              7 units move.
symptom:      the first readback, written on the unfiltered actions, reported a
              "decreasing, saturating step function of the inventory position"
              and a "scarcity regime" with cumulative protection five times the
              bar's. Both were artifacts: the scarcity regime is where σ does
              nothing, and the output there is free-parameter noise drifting off
              centre under the entropy bonus (biased UPWARD by ~4 units).
diagnosis:    three nested identification tests, and the loose one is the trap:
              RANGE-IDENTIFIED (σ=0 or σ=max would change the outcome) —
              58.5% / 41.8% of events; BINDING (the CHOSEN σ changes the
              allocation vs the null action) — 25.7% / 8.7%; PINNED (binding
              and not saturated, i.e. the VALUE was determined) — 6.7% / 3.1%.
              Where the action bites the policy is nearly constant, and the
              whole gain over the paper's heuristic is one integer (σ_2 1 → 2,
              worth −0.48; the state dependence −0.0005) — see LV4 for the half
              this fit missed.
prescription: interpretation. (1) Audit identification BEFORE reading
              structure, and report the strict pair (binding, pinned) beside
              every structural statistic — a mean over unidentified events is
              a measurement of nothing and does not average out. (2) Plot the
              EFFECTIVE quantity, min(resource, action), never the nominal
              level: σ = 17 against 3 units withholds 3, so every value above
              3 is the same decision, and nominal values near the cap are the
              signature. Compare against the reference REPLAYED on identical
              states — its nominal line saturates too. (3) Suspect the region
              where the parameter looks most interesting: the slack region is
              the state-space extreme, exactly where a readback looks for a
              "regime". (4) Score the constant; a state-dependent rule fitted
              to unidentified noise scores the same while looking like a
              discovery. (5) This does not retract LV2 — the encoding still
              wins on cost, episode length and needing no mask; it costs
              interpretability per action, a price worth paying once known.
              TRIPWIRE: any domain whose action is a level, target, threshold,
              protection level or index owes this audit at the readback.
failed:       the readback as first committed (retracted — INTERPRET.md §9);
              plotting nominal σ against time (F55's first two figures, redrawn
              on identified events only).
evidence:     F55 (the three tests and the retraction), #E19, #E20, INTERPRET.md
              §2–§4 and §9.
```

### LV4 — the policy must be in its own interpretation: score the policy's component, not only the rule fitted to it

```
context:      spec §14.2: a readback fits a rule to the learned policy and
              scores the rule. The right deliverable — a fitted rule ships
              where a network cannot — but not, on its own, evidence about the
              policy.
symptom:      the campaign fitted a σ(u) step rule, scored it at 346.6985
              against the best constant's 346.6980, and concluded the learned
              protection had no state-dependent structure worth anything. A
              later factorial put the policy's OWN protection component under
              the same fixed order for the first time: 346.0685 — beating the
              best constant by 0.63 ± 0.05 (13 SE). The structure was there;
              the fit had missed it.
diagnosis:    two extracted rules agreeing with each other say nothing about
              the thing they were extracted from, and the error runs in the
              flattering direction: "my fit explains no more than a constant"
              and "the policy is no better than a constant" produce the same
              table. The structure turned out to be a horizon boundary
              condition (the last periods), which a step-in-u rule cannot
              express.
prescription: interpretation. A discover-style claim needs FOUR numbers under
              one harness and one seed block: the reference (PL(σ) 347.1749);
              the best member of the null class, SEARCHED (best constant (4,2)
              = 346.6980, from a 56-ladder grid — a gap against the class's
              ceiling, not a convenient member); the policy's own component in
              isolation (346.0685); and the fitted rule (346.6985). Policy vs
              null class answers the stance; fitted rule vs policy measures
              how much of it you explained — here 0 of 0.63, an honest open
              gap. Report the two failure modes separately: "the structure is
              absent" is a finding; "the structure is there and I did not find
              it" is work outstanding.
              SCOPE (2026-09-04): the 0.63 was the categorical crown's. On the
              ordinal-head crown the same isolation scores +0.42 BEHIND the best
              constant (#E26) — a component verdict is one artifact's, and a
              crown change owes the measurement again (F59).
failed:       the σ(u) step fit as the sole evidence (F56); reading the two
              rules' agreement as a null result; carrying the verdict across a
              crown change on a scope caveat (F59).
evidence:     F56, #E20 (the 2×2 factorial), #E13 (the isolation instrument
              that made "the policy's own component" scorable), INTERPRET.md
              §5–§7.
```

---

## Frame moves

```
FM1  bracket before build — bound the problem with a relaxation and a
     heuristic before any training. AP relaxation (a lower bound no policy
     attains) below, the paper's PL(σ) heuristic above, on both heterogeneous
     boards (#E1, #E15). The bracket's width IS the campaign's headroom: 1.51%
     on het_exp4, 4.31% on het3_exp2 — and RL's fate on each board tracked it
     (0.36% short of the bar where the bracket is tight, 2.7% under it where
     it is wide). Paid off at the first het3 run: the board was chosen because
     the bracket said the heuristic had slack.
FM2  decompose the gap with an ISOLATION INSTRUMENT before escalating — hold
     one decision at the reference's own policy and learn the other. With the
     order fixed at AP and only the protection learned, PPO lands +0.004 on
     the bar (#E13): the entire remaining gap was in the ORDERING. That priced
     every further allocation lever at zero and every order lever at the
     whole gap, before #E21 dissected the ordering per period and per seed
     and #E22 built the lever the dissection named. Diagnosis → lever, not
     lever → hope (LV1).
FM3  one tree, three kinds of split — cases (own frame, scores incomparable),
     design-axes (shared frame, crown forks), escalations (shared frame, crown
     passes one). The head could be placed in one line: an arch-layer
     escalation under the crowned cells, no design axis moved, no IR change,
     and its id an `a` (F42–F50, #E25). Placement decided what had to be
     re-measured (nothing above it) and what could be crowned (everything
     under it).
FM4  a factorial when two encoding decisions are entangled — the 2×2 of order
     encoding × allocation encoding (#E11) answered them separately: the
     allocation half is worth −4.65, the order half nothing, where a single
     "order_protection vs seq_mask" contrast would have credited both.
FM5  confirm the top-k of a tuning study, never the winner alone — the trial
     layer (one seed, 2048 episodes) inverted on confirm three times (#E22
     reading 4, #E23 reading 2, trial 137: h3 ranks 30/129 at the trial layer
     and is the best cell at the protocol). Standing practice since #E22.
FM6  the control is the in-frame retrain, not the registered number — the
     §9.7 single-artifact crown on het3 (337.4015) sat 1.5 below its own
     config's 6-seed mean (338.87); a lever measured against it would have
     read as a loss. Pair every contrast on the same seeds and hp (#E24).
FM7  the archive is enumerated, not assumed — two worktrees, one registry:
     the audit must walk `git worktree list`, a clause like "no trained arm on
     this instance" expires the day a sibling worktree trains one (F52, F57).
```

---

## Modeling rules  (lever layer `IR-formalization`; trigger-first)

```
MR1  if the source names a decision AND a parameter of a rule for it, the IR
     declares the DECISION (the allocation), never the parameter (the
     protection level) — declaring the parameter makes every expressible
     policy rule-shaped and answers the research question by fiat (F7, F30).
MR2  if an encoding computes the research question's own transform inside the
     action decode, it is not an arm of that question — it is the answer
     assumed (F30; RQ4's `vec_mip` is an OBSERVATION arm for that reason).
MR3  if a width is one more than a design axis, ask which event order it
     encodes; a width the IR declares and no instance exercises is untested
     by every gate (F6, F10) — and once an arm is trained under it, the width
     is pinned by that artifact in every worktree (F57).
MR4  if a feature is "history", test it by SUBSTITUTION: does the optimal
     policy change when it is replaced by the state it summarizes? "Already
     derivable" is a measurement, not an argument (F24, F25).
MR5  if a decision is a flow bounded by an availability, dissolve the
     constraint by encoding the rule's parameter (LV2) — and then budget the
     identification audit it will cost at the readback (LV3; F51, F55).
MR6  if a knob was never opened by the tuning tier, it is not a configuration
     choice and gets no id; an id is a promotion — crowned, shipped, or made a
     parent — never a record that something ran (F44, F45).
MR7  if a workaround is added, write its owner and its expiry into the
     frontier before the proposal that describes it is filed; "until upstream
     ships the fix" is not an expiry anyone checks (F54).
MR8  if a tier-2 stance's instrument is an outcome comparison between arms,
     the stance is `bypass`; `confirm` needs a readback that recovers the
     structure — declare the stance the instrument can decide, and re-read
     every stance against its instrument at Phase A (F58).
MR9  if the decision is a discrete QUANTITY, start with an ordinal head, and
     zero-inflate it when the problem has a fixed cost or any other genuine
     "do nothing" mass (LV1) — a per-quantity categorical head is the choice
     that needs justifying, not the default.
```

---

## What did not transfer

- **The paper's sufficient statistic as an observation win.** Handing the
  network the modified inventory position `u` was expected to help; it never
  did — a dead heat on every board at every rung, and +0.01 under the head
  (#E8, #E17, #E24). The network rebuilds `u` when not given it, which is why
  RQ4 became a `bypass` (F58, MR2).
- **The order-up-to encoding from `clark_scarf`.** An inventory-target action
  had been slightly better there; here it was withdrawn before a run (F30) and
  then measured worthless when it did run (#E11: three of four contrasts
  positive), because a fixed ordering cost makes the optimum (s,S) and a
  target cannot express the trigger. The move that DID transfer was the
  parameter encoding of the *allocation* (LV2).
- **Observation normalization as a prior.** `norm_obs=on`, the §8.3 default,
  cost 7–83 points across the boards' arms and was the whole of a gap first blamed on
  the encoding (F36, F38); off is this domain's base everywhere.
- **"Tune harder."** 820 trials and a stage-2 retrain moved het_exp4 by 1.6
  (#E16); the head moved it 8–9 at the derived rung and the trial-layer
  ranking inverted on confirm each time it was checked (FM5). Tuning FOR the
  head then helped `vec` (−1.33) and not `vec_mip` (#E23) — an hp centre is a
  covariate of the lever, not a substitute for it.
- **A readback verdict across a crown change.** "The crowned policy has
  structure beyond a constant protection level" was true of the categorical
  crown and false of the ordinal-head crown that replaced it on the same board,
  hp and seeds (#E26). Verdicts are indexed by artifact (F59).
- **The single artifact as the comparison point.** The §9.7 crown was a
  favourable draw 1.5 below its own config's mean (#E24, FM6); the campaign's
  numbers are 6-seed means from that point on.

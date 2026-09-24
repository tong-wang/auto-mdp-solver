# PLAYBOOK — `clark_scarf`

Guide-§10 distillation. **Serial multi-echelon inventory with a verified-exact
DP on the integer lattice** — so every policy here was read against a known
optimum rather than against the previous rung, and "100.7% of optimal" means it.

Protocol every number below is quoted at: **8192 CRN seeds from offset 0**,
shared across arms, deterministic argmax, reference bar `dp` = **982.362986**,
on the one cell `n3_l2_p09` (3 levels, lead time 2, shortage 9.0, β = 0.95,
T = 50, Poisson(10)). Screens rank on a disjoint block and are never quoted.
The within-cell training-seed spread at an adequate budget is **~6 cost units**;
that floor governs every margin here.

**Already upstream; not repeated here.** Post-hoc checkpoint screening is `mab`
LV6, graduated into spec §8.6/§9.7. Harvesting a tuner's centre is `mab` LV7.
Pricing a representation lever at L2 rather than L1 is this domain's own earlier
board, carried forward.

---

## LV1 — a head that wins untuned can lose tuned, and the ordering can fully invert

context:      four action heads on one MultiDiscrete space — flat categorical,
              a hurdle-discretized-Gaussian with an exclusive zero atom, and two
              atomless discretized-Gaussian bodies. Ranked first at L1 (the §8.6
              derivation, 4M steps), then each tuned at its own centre in a
              separate study.
symptom:      the L1 ranking and the tuned ranking are reverses of each other.
              `a1` (atom) is best untuned at 1001.74 and worst tuned at 998.49;
              `a2` (no atom, affine location) is worst untuned at 1040.84 — 39
              units behind — and best tuned at 989.62.
diagnosis:    tuning does not shift heads by a common amount. It paid `a2`
              −51.22 / −82.33 on the two observation arms, `a0` −11.46 / −9.58,
              `a1` −3.25 / −7.64. The atom is an optimization device, not a
              representation: it buys a workable policy from a bad centre, which
              is exactly why it leads untuned and gains least from tuning. A
              head's untuned quality and its tunability are different
              properties, and only the second survives a tuning round.
prescription: **arch.** Never rank heads at a single centre. Either tune each at
              its own centre before ranking, or report the L1 ordering as a
              statement about that centre and nothing else. Scope: any campaign
              comparing action parameterizations where one carries an
              initialization prior the others lack.
failed:       ranking at L1 and treating it as the head verdict (#E4, #E7 — the
              first withdrawn by budget, the second by tuning).
evidence:     #E7 (the L1 decomposition), #E10 (the tuned three-way and the
              reversal).

---

## LV2 — tune the arms at one budget, or the head axis is unreadable

context:      the same six tuning studies as LV1, launched across two sessions.
symptom:      the winning head's studies ran at 5M steps; the two rival heads'
              at 4M. The winner leads by 8.87 on one arm.
diagnosis:    a 25% budget difference in the direction of the result, on a board
              where 2M→4M was measured at 13–24 units per cell. The margin is
              not attributable to the head. It is *bounded* — the last fifth of
              a 4M run buys −2.1/−2.8, so a further 1M plausibly buys single
              digits, not nine — but a bound taken from another head's decay
              curve is an argument, not a measurement.
prescription: **protocol.** Fix the per-trial budget across every arm of a
              comparison before the first study launches, and record it on the
              arm rather than in the launcher. A budget difference is invisible
              in the study name, the trial values and the leaderboard alike.
failed:       reading the head axis off the confounded board; withdrawn in the
              same entry that produced it.
evidence:     #E10.

---

## LV3 — a coordinate transform pays only the policy class that cannot learn it

context:      the campaign's declared tier-2 question — does a policy given raw
              installation stock match one given the echelon transform? Asked at
              a tuned centre on three heads.
symptom:      the answer differs by head. Null for the two stronger heads
              (+0.39, z 0.22; −1.08, z 0.66) and **+4.76 (z 2.89)** for the
              weakest.
diagnosis:    the two nulls point in opposite directions from each other, which
              is what a genuine null looks like rather than a small real effect.
              The head that gains is the one whose location parameter couples
              state regimes, i.e. the one least able to synthesize the transform
              internally.
prescription: **gym-obs.** Report a representation result as an interaction with
              the policy class, not as a property of the domain. A
              representation lever measured on one architecture is a statement
              about that pair. Scope: any campaign whose research question is
              "is transform X necessary" — run it on at least two architectures
              of different strength before answering.
failed:       the untuned single-head reading (#E3) called the arms a tie at
              L1 on the head where the tuned board later finds the largest
              effect.
evidence:     #E3 (the untuned tie), #E10 (the tuned interaction), #E11 (the
              readback reaching the same conclusion from a second instrument).

---

## LV4 — read the policy by enumeration; a trajectory readback shows the constraint

context:      the tier-2 `confirm` stance — is the raw-trained policy an echelon
              base-stock rule with Clark & Scarf's critical numbers? Stage-5
              readback on the tuned winners.
symptom:      the first readback, taken on the policies' own trajectories,
              showed a tight order-up-to level at the retailer (spread 8 units,
              84.4% agreement with the optimum) and nothing recognisable above
              it. The conclusion drawn — "found the coordinates, not the rule" —
              was the reverse of the truth.
diagnosis:    two causes, both invisible in the output. **(i)** The availability
              clip binds on ~2/3 of the lower echelons' executed decisions: link
              *k* draws from installation *k+1* and is capped by what it holds,
              so a scatter of executed shipments is mostly a picture of the
              upstream constraint. The retailer's "tight rule" was the
              constraint, and the level where the policy was actually free —
              the top echelon, clip never binding — had the widest spread.
              **(ii)** The probe was reading the nets on unnormalized
              observations (LV5).
prescription: **interpretation.** Read a policy by **enumerating the input**:
              one synthetic state per grid point, queried deterministically, the
              upstream source raised past the largest shippable quantity so the
              clip cannot bind, swept past every critical number so the plot
              shows where ordering *stops*. One input, one output — a function,
              not a sample. Hold the rest of the chain at the reference
              optimum and vary only the echelon of interest; state the
              convention, because a cumulative coordinate is reachable by many
              splits and the policy is sensitive to which.
              Corrected result: the top two echelons shut off at 78 and 52
              against critical numbers 79 and 59 — the paper's rule — while the
              **retailer never shuts off at all**, trickling ~5 units at four
              times its optimal position. That defect is invisible to any
              trajectory read, for the same reason it survived training: a good
              policy never occupies those states, so nothing charges for it.
failed:       the executed-shipment scatter (retracted); a first enumeration
              convention that put the whole echelon position on the target level
              and emptied everything below it, which for the top echelon emptied
              the entire downstream and produced non-monotone curves.
evidence:     #E11.

---

## LV5 — every consumer must resolve a run's contract FROM the run, never from its own default

context:      four modules load a trained artifact: eval, the §9.7 screen, the
              §12 deployable policy, and the §14 probe. Each must reconstruct
              two things the model file does not carry — the normalizer, and the
              observation/action modes.
symptom:      three separate silent mis-scorings, one per missed contract.
              **(a)** The probe looked for `vecnormalize.pkl` while every run
              since §9.7's machinery was removed writes
              `vecnormalize_final.pkl`; it found neither, scored raw, and
              produced a complete, plausible readback of a policy that was never
              trained. **(b)** The screen had the same lookup and no refusal, so
              it could rank every checkpoint raw and still print a confident
              top-k. **(c)** The §12 CLI defaulted `--action-mode` to
              `ship_fraction`; replaying a `ship_discrete` net without `-a`
              decoded the action as a fraction and reported **35267** against a
              true **998.59** — the magnitude of the random baseline, so it
              reads as a bad policy rather than a wiring error.
diagnosis:    one mechanism. Anything the model file does not carry must come
              off the run's own args, and a default that is *plausible* is worse
              than one that is absent: every failure here produced a well-formed
              number with no error, and two of the three were caught only by a
              magnitude that happened to look wrong.
prescription: **process.** For each consumer: resolve from the run's args file,
              walking up from the model path (a checkpoint's parent is
              `checkpoints/`, not the run dir); consult **every** historical
              filename, not just the current one; and **refuse** — never warn,
              never default — when the run records a contract the environment
              cannot satisfy. Verify the §12 replay against eval at *matched*
              seeds, which is the only check that catches (c): here
              998.5860 vs 998.586015.
failed:       a guard test asserting the string `REFUS` appeared in each
              source. It passed on a file whose normalizer refusal had been
              deleted, because that file refuses something else elsewhere — the
              detect-by-name failure this campaign had itself filed upstream as
              issue #89, reproduced in its own test file. Replaced with
              behavioural tests, each mutation-tested by reverting its fix.
evidence:     #E1 (the first instance and the fix for eval + §12 loading),
              #E11 (the probe, the screen, and the §12 CLI).

---

## FM1 — a live selection callback can violate §9.7 while every gate stays green

context:      the board opened by auditing the previous one rather than by
              training.
diagnosis:    a `CRNSelectionCallback` ran *during* training, selecting
              checkpoints on a live eval — the §9.7 violation — and the
              harness's detector missed it because it matched on class name
              (`endswith("EvalCallback")`), then PASSed with a message asserting
              the opposite. A negative it never tested.
prescription: **protocol.** When a check reports the absence of something, ask
              what it examined. Filed upstream the day it was found, not at case
              close; accepted as v0.10.14 with a behavioural detector beside the
              name one. The general form — *a name check that cannot fail is not
              a check* — recurred inside this campaign's own test file (LV5).
evidence:     #E1, #E2, upstream issue #89.

---

## MR1 — echelon STOCK and echelon POSITION are different quantities; a decision moves only one

trigger:      any multi-echelon formalization where holding is charged on an
              aggregate and the decision is a transfer between levels.
rule:         charge holding on echelon **stock** (a level plus everything below
              it, including goods in transit *toward* lower levels) and state the
              policy on echelon **position** (stock plus everything on order into
              it). Keep them as separate functions and never let one call the
              other.
why:          a shipment moves goods *within* an echelon, so it cannot change
              echelon stock at all — measured here as exactly **0.000000** over
              1500 decisions at every level. It moves position. Conflating them
              double-charges pipeline stock (2.0 vs 1.0 per period at this cell)
              and biases the shipping incentive against moving stock downstream —
              precisely the trade-off the campaign measures. It also silently
              picks the wrong axis for the §14 readback: a before/after plot in
              stock coordinates is three 45° lines by construction.
evidence:     #E11; `clark_scarf_mdp.py::echelon_stock` / `echelon_position`.

---

## What did not transfer

- **The hurdle head's zero atom, imported from `adi_flex` LV1 / `inv_single`.**
  It is earned there by a fixed ordering cost, which creates a genuine
  "do nothing" mass. This domain has **no fixed cost**, and the atom's own scope
  clause says a domain without that mass wants the body without the gate. Here
  it led untuned and finished last once every head was tuned (LV1). Import the
  *body* — adjacency pooling — and leave the gate behind unless a fixed cost
  pays for it.
- **"The categorical is the head nothing beats here."** True at L1 and false at
  a tuned centre; superseded within the same campaign (#E7 → #E10).
- **A tuner's trial-layer ranking as a stand-in for the protocol layer.** It
  ranked reliably here — the rank-1 trial won the confirm layer in all six
  studies — but its *values* sit 5–7 units above the confirmed ones uniformly.
  Usable as a selector, never as a number.

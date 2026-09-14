# PLAYBOOK.md — `dynamic_pricing`

The case-close digest (ESCALATION_LOG_GUIDE §10) for the Gallego & van Ryzin
(1994) finite-horizon pricing problem: a **single-entity continuous control
with an absorbing sell-out and an exact DP bar**, where the only lever the
campaign found was the action interface. Distilled 2026-09-14 at
re-contribution; the campaign itself is dormant, not closed — its RL arms
have no artifact on the current seed tree (`ESCALATION.md` A3).

Every number below is quoted at the header protocol of `ESCALATION.md` §MAP:
8192 episode seeds from 0, the same block on every arm of a board,
deterministic argmax, reference bar = the evaluated DP row (33.15 on `simple`
in the July 2026 frame). Ledger entries are cited, not copied.

## Levers

### LV1 — a continuous action box whose floor is a dead zone stalls PPO

```
context:      L1 on `simple`, the two action encodings the IR declares:
              `price` on [0, 8] (identity + clip) and the paper's native
              `intensity` on [0, a/e] (inverted to a price). The IR's own
              candidate text predicted `intensity` would be the easier
              encoding, since revenue is linear in sales under it.
symptom:      `intensity` at the same defaults and budget scores 17.34 against
              `price`'s 32.82 — below the myopic price — and the learning
              curve crawls out of zero-revenue territory for hundreds of
              thousands of steps.
diagnosis:    SB3's Gaussian policy initializes at raw action 0. In the
              intensity box that is the null (shut-off) price, so half the
              initial action mass earns nothing and the gradient has little
              to work with; in the price box raw 0 is a low-but-selling price
              and the signal is immediate. An interface effect, not a
              representational one — no probe was run beyond the two arms.
prescription: gym-action — prefer the encoding whose raw 0 is a live,
              low-value action; keep the paper-native encoding declared as a
              non-default mode so the design axis stays covered.
              Effect: +15.48 at L1, +7.01 after `intensity`'s own hp
              escalation. Scope: any Box action with a dead zone at the box
              floor; the lesson has since reached `mdp-build` Stage 2 as the
              "action-box design lesson" and fires there before a run.
failed:       `ent_coef` 0.01 with double the budget (1M steps) moved
              `intensity` to 25.95 — still below the fixed-price heuristic.
evidence:     #E2, #E3
```

### LV2 — a tuning winner's recipe does not owe a better number at a larger budget

```
context:      L2(hp): the 25-trial TPE winner (trial 16 — rollout 512, batch
              32, 20 epochs, lr 1.6e-4, γ 0.9995) re-scored at 32.96, +0.14
              over L1, one seed per arm.
symptom:      the same configuration retrained at 2.5× the budget (500k)
              scored 32.31, 0.65 below the 200k artifact it was meant to
              improve on.
diagnosis:    two things the single seed cannot separate: the winner is a
              selection maximum over 25 trials and a fresh draw regresses
              toward the procedure's mean; and the heavy 20-epoch recipe was
              tuned at 200k, so its rollout/epoch balance is a property of
              that budget. No retrain spread was measured.
prescription: process — ship the tuning artifact itself once its protocol
              re-score clears the baselines; treat a budget extension as its
              own arm with its own address, never as a free upgrade of the
              crowned one; and quote the winner's-curse size beside any
              L2(hp) delta that rests on one seed. Effect: −0.65 for the
              extension, which is also the one-sample curse estimate.
failed:       —
evidence:     #E4, #E5
```

## Frame moves

### FM1 — a record frame and a replay frame, never subtracted

When the seed tree moved under the folder (v2, 2026-07-22) after the numbers
were produced, the honest board is two: the July record, all arms on one
key, and a re-measurement of what *can* be re-run on the new key. The four
benchmark arms reproduced within two SE of a two-block comparison (#E8), the
RL rows could not, and the README says which is which above the tables. The
move that paid off was refusing to mix them — a re-measured DP beside a
recorded PPO row would have manufactured a 0.13 shift in the gap. Evidence:
#E7, #E8.

### FM2 — calibrate the bar by its own two numbers

An exact DP carries a computed value and an evaluated row, and they differ by
eval noise: 33.1047 computed against 33.15 (July) and 33.02 ± 0.05 (v2). Quote
the evaluated row as the bar, since it shares the protocol with every other
arm, and keep the computed value in the off-tree register as the calibration
the evaluated row must straddle. Evidence: #E1, #E8.

## Modeling rules

None — the §IR-CHANGELOG holds no reversal. The two `mdp`-block moves the IR
took upstream (the v2 seed scheme, the declared invariants) were conformance
edits without a recorded fingerprint, not re-formalizations.

## What did not transfer

- **The IR's own prediction.** The candidate text on the `intensity` mode
  said the paper's native control was "typically easier for PPO". It was the
  harder one by 15 points (LV1). A prediction written into a `desc` is a
  hypothesis, and the campaign is where it gets tested.

## Envelope verdict

RL is **competitive and not preferred** on this structure class: the tuned
policy reaches 99.4% of the exact optimum and beats the paper's
asymptotically optimal fixed price by 1.04 (#E4), but the bar is a backward
induction over 50 × 25 states on a price grid that solves in seconds, so a
learned policy has no efficiency argument here. What the case delivers is the
interface lesson (LV1), a benchmark scaffold that reproduces (#E8), and the
degenerate-control board (`ample_stock`, #E6) showing where there is nothing
to learn at all.

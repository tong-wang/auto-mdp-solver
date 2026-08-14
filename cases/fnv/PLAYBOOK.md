# fnv — playbook digest

**Structure class:** a small number of sequential procurement commitments at
rising cost, against a single terminal demand whose forecast evolves as a
martingale (MMFE); the decision is one continuous order per period and all
uncertainty resolves at the horizon.

**Protocol every number below is quoted at:** 540-cell design grid × 2048 CRN
seeds per cell, seed block 0…2047 shared by every arm, paired per cell;
grid-mean profit weights cells equally. Reference = the paper's exact optimum
(Wang, Atasu & Kurtuluş 2012, Prop 2), computed two independent ways that agree
to 2.4e-05. Two branches, separate boards: a-MMFE bar 0.888334, m-MMFE bar
2.291618. Campaign 2026-08-13/14, closed after 18 training runs.

---

## LV1 — a "derived" config can lose to library defaults; L0 is what catches it

context: first solve rung on both branches, before any escalation. L0 was run
  only as the spec's ruler, not as a candidate.
symptom: the derived L1 config scored **below** faithful defaults on both
  branches — a-MMFE −0.002673 ± 0.000106 (~25 SE, no seed overlap), m-MMFE
  −0.001771 ± 0.000301.
diagnosis: under-training, not instability. The telemetry separates the two
  cheaply: **`approx_kl` averaged 0.0015 against L1's own 0.02 valve** (an
  order of magnitude of unused headroom, and the valve never truncated an
  epoch), while the batch choice took **78k gradient updates vs L0's 312k** at
  equal env-step budget — together ~40× less total policy movement. Two of
  three seeds still peaked at the terminal checkpoint.
prescription: lever layer **HP**. Re-derive *within the spec's own stated
  ranges*: LR to the row's other branch (1e-4→1e-5 ⇒ 3e-4→3e-5) and batch to
  the band floor (rollout/8 ⇒ rollout/32). Measured: **+0.001889 ± 0.000087**
  on a-MMFE (recovers 71% of the loss) and **+0.005421 ± 0.000262** on m-MMFE.
  Against L0 the corrected config then **ties** on a-MMFE with a 6× tighter
  retrain spread and **wins outright on m-MMFE (+0.003651 ± 0.000365)**.
  Scope: short-horizon domains where a rollout already holds many episodes.
failed: nothing else was tried — the diagnosis was cheap and pre-committed the
  branch (if L1′ had not cleared L1, `ent_coef` was the next suspect).
evidence: #E1, #E2, #E3, #E6.

## LV2 — read the LR rule against the rollout's episode count, not the reward noise

context: the LR row keyed on "exogenous noise dominates reward variance", which
  is *true* here — profit is set by one terminal demand draw.
symptom: the rule fired correctly by its own terms and still produced an
  under-trained config (LV1).
diagnosis: the rule reads the **reward** distribution but sets a knob governing
  **gradient** noise. With a 3-step episode the rollout holds ~683 episodes, so
  the gradient is heavily averaged before any update lands; the two noise
  quantities come apart exactly when T̄ is small.
prescription: lever layer **HP**. Treat sustained `approx_kl` an order of
  magnitude *below* `target_kl` as the LR-too-low signal — the mirror of the
  spec's existing "repeated truncation ⇒ LR too high". Cheap, and it localizes
  the fault before any retraining.
failed: —
evidence: #E2, confirmed by the same pattern on the second branch (#E6).

## LV3 — a policy 0.2% from optimal can be qualitatively wrong

context: after the ladder, both branches sat within 0.2–0.5% of the exact bar
  and cleared the must-beat baseline comfortably.
symptom: on a-MMFE the policies used their **middle** ordering opportunity in
  0.000–0.235 of episodes against the optimum's 0.504, collapsing three orders
  into two — while ordering the same **total** quantity (1.0041 vs 1.0043).
diagnosis: the profit landscape is flat in that direction. Ordering early is
  cheaper (c₁ = 1.00 vs c₂ = 1.04) and at T = 0.5 only half the demand variance
  resolves before the last order, so the option value of waiting is small. The
  policy substitutes cheap early stock for late information and loses ~0.1%.
prescription: lever layer **interpretation**. Score the readback separately
  from the policy; no leaderboard number distinguishes this policy from one
  that uses all three orders. Sweeping the information timing T does **not**
  expose it either — P(order@2) rises 0.000 → 0.223 as T goes 0.1 → 0.9 while
  the profit gap stays flat at ~0.1%, because the policy compensates exactly
  where the stake rises.
failed: the prediction that a costlier middle order would surface the defect in
  the score — refuted by the T sweep.
evidence: #E4, #E5.

## LV4 — check a structural verdict on a second branch before generalizing it

context: LV3 looked like a statement about the learner.
symptom: the identical setup on the multiplicative branch used the middle order
  in **0.168–0.433** of episodes against an optimum of 0.474 — two arms within
  0.04 of the optimal rate.
diagnosis: branch-specific, not learner-specific. Under `exp(mu + I)` the
  demand *scale* moves with the information, so a stale early commitment is
  punished multiplicatively; the middle order earns its place and the policy
  finds it. What LV3 measured was the flat **additive** landscape.
prescription: lever layer **process**. Park a structural finding behind an
  explicit tripwire rather than concluding from one branch. The tripwire here
  ("any arm reaching P(order@2) > 0.35") fired on the replication and un-parked
  the node without anyone re-reading the old entry.
failed: —
evidence: #E4 (parked), #E8 (fired).

## LV5 — the fitted rule can beat the net, and its coverage is a diagnostic

context: §14.2 readback, after the structure was confirmed on both branches.
symptom: distilling each policy into per-cell constants `b̂_1..b̂_N` and scoring
  them on the same CRN block: m-MMFE **+0.002426 ± 0.000746 (3.3 SE)** for the
  rule over the net it came from; a-MMFE a tie (−0.000126 ± 0.000049,
  −0.014% of bar).
diagnosis: fitting constants deletes the raggedness the network carries. Three
  numbers per cell replace a neural policy and score at least as well.
prescription: lever layer **interpretation**. Always score the fitted rule as a
  policy — and treat **coverage of the fit as a first-class signal**. The
  full-horizon assertion dropped **331 of 540 a-MMFE cells** (the policy never
  orders at period 2 there, so `b̂_2` is unidentifiable) against only 32 on
  m-MMFE. A policy that abandons an ordering opportunity **cannot be distilled
  into a rule over it**, however well it scores — the coverage number states
  the structural gap in LV3 more sharply than the profit gap does.
failed: —
evidence: #E9.

## LV6 — identify a censored rule only where it is uncensored

context: the readback itself; three wrong answers preceded the right one.
symptom: the same probe returned slope 1.24 / 0.94 / 0.87, then 0.00, then
  0.96–1.16 with r² 0.82 — three incompatible verdicts on one policy.
diagnosis: `q = max(0, S − x)` is censored at zero, and `S` — not `q` — is the
  policy. A period with no order certifies only `S ≤ x` and identifies no
  slope. Sweeping inventory freely instead measures the network **off its own
  state manifold**: because `I_1 ≡ 0` the first action is deterministic, so the
  reachable set at later periods collapses to a single trajectory prefix
  (x = 0.9501 exactly, not a distribution).
prescription: lever layer **interpretation**. Regress the realized
  `(I_n, x_{n-1}+q_n)` pairs from the policy's own trajectories, **acting
  periods only, one fit per scenario cell** — pooling cells smears the
  per-cell intercept `mu + b_n` and destroys the line. Result: slope
  **1.109 ± 0.122** over 36 fits with **r² 0.966** (a-MMFE) and r² 0.976–0.999
  (m-MMFE, in logs).
failed: free inventory sweep at fixed I (off-manifold); reachable-states-only
  (censored, reported a false zero); affine plane `A + B·I + C·x` (fights the
  censoring kink, R² 0.82).
evidence: #E5, #E7.

## LV7 — fit the branch in the coordinate its structure is linear in

context: replicating the readback onto the multiplicative branch.
symptom: none observed — caught before running, but it would have produced a
  curve fitted as a line and a meaningless slope.
diagnosis: a-MMFE predicts `S = mu + I + b`; m-MMFE predicts
  `S = exp(mu + I + b)`, i.e. **log**-linear. The same regression on the raw
  level is simply the wrong model on the second branch.
prescription: lever layer **interpretation**. Map to the structural coordinate
  (identity / `log`) before fitting, and let the axis label and the reference
  line follow the mode. Both branches then predict the same thing: slope 1
  against I, intercept `mu + b_n`.
failed: —
evidence: #E7.

---

## Frame moves

**FM1 — bound before build.** The exact optimum was implemented, gated, and
validated *before* any policy was trained, so every ladder rung was read
against a real bar rather than against the previous rung. It also meant a
0.2%-from-optimal result could be recognized as "the landscape is flat" rather
than "the agent is good" (LV3).

**FM2 — two independent solvers as the bar's own gate.** The optimum was
computed twice by different routes (backward value-function DP on a reduced
grid; the paper's 1-D threshold recursion) and cross-checked to 2.4e-05 in
scored profit. When a downstream arm sits 0.2% from the bar, the bar has to be
beyond question first — and keeping the two implementations *independent* is
what makes the agreement mean anything. Do not refactor them onto shared
numerics.

**FM3 — replicate on the sibling branch before generalizing.** Running the
whole ladder a second time on m-MMFE confirmed one finding (LV1), strengthened
another (L1′ ties → wins), and **reversed** a third (LV4). The marginal cost
was one unattended pass.

---

## Modeling rules

**MR1 — when a decision bound is derived from one branch, check it against
every instance the IR carries.** The `order` bound was the literal `[0, 3]`,
reasoned from the additive branch (mu = 1, stdev ≤ 0.3). The IR also carried a
multiplicative instance whose own ceiling is `exp(mu + 5·stdev) = 7.39`, so the
differential drew decisions from [0, 3] while the domain reached 7.39 —
everything above 3 unverified on that instance, and simultaneously too loose on
base (real ceiling 1.5). Fix: name a scenario constant re-baked per instance,
which is what the schema's constant-naming bound form exists for.
evidence: F1.

**MR2 — a quantity the interpreter must reproduce bit-exactly belongs in the IR
as resolved data, not as a formula to re-derive.** FNV bakes `signal_stdevs`
(and, after F1, `order_max`) as exact float reprs; an instance that changes the
inputs re-bakes them. The cost is that instances must remember to; the benefit
is a differential that can be bit-exact at all.

---

## What did not transfer

- **"The derived config beats defaults."** Imported as an assumption from the
  spec's framing of L1 as the mandatory backbone. False on both branches here
  as first derived (LV1). L0 is not ceremony on short-horizon domains.
- **"A costlier decision surfaces a structural defect in the score."** The T
  sweep refuted it: the policy compensates where the stake rises, so the profit
  gap stayed flat at ~0.1% while the structural gap moved a lot (LV3).
- **"Probe the policy on a declared state grid."** The generic §14.1 sweep
  measures a trained net off its own manifold whenever the reachable set is
  degenerate — here a single trajectory prefix, because `I_1 ≡ 0` makes the
  first action deterministic (LV6).

## Envelope verdict

RL is **competitive but not preferred** on this structure class. The exact
optimum is cheap (a 1-D recursion solved off-line, milliseconds per cell), so a
learned policy has no efficiency argument, and it lands 0.2–0.5% below the bar
while missing an ordering opportunity outright on the additive branch. What the
campaign *does* deliver is the readback: the policy demonstrably rediscovers
the paper's linear/log-linear base-stock structure with unit slope, and the
distilled constants score at least as well as the network (LV5). On a class
where a classical solver already exists, that — not the leaderboard number — is
the result worth having.

# PLAYBOOK — `inv_single`

Guide-§10 distillation. **Single-item periodic-review inventory, finite
horizon, with an exact DP on both sides of the fixed-cost line and on both
sides of the lost-sales line** — so every policy here was read against a known
optimum rather than against the previous rung.

Protocol every number below is quoted at: **2048 CRN seeds per cell** for a
grid, **8192** for a single scenario, with checkpoint screening and tuner
objectives on seed blocks disjoint from the reported one. Magnitudes are given
as **% of that cell's exact optimum** — raw cost units mean nothing outside
this folder and stay in the `#E` citations.

**Already upstream; not repeated here.** The hurdle-discretized-Gaussian
quantity head ("ordinal" informally) is `adi_flex` LV1 + MR9 — one name over
two separable mechanisms: the adjacency-pooling discretized-Gaussian body,
plus a hurdle zero gate that exclusively owns P(0), earned here by the fixed
cost. This campaign confirms the head a second time and, for the first time,
at a *tuned* centre, where its value shows up as reduced seed variance rather
than a better peak (#E21). Checkpoint screening is
`mab` LV6, already graduated into spec §8.6/§9.7; harvesting a tuner's centre
is `mab` LV7; distilling a net into a rule and scoring the rule is `mab` LV8.
Pricing a representation lever at L2 rather than L1 is `clark_scarf` LV2.

Two upstream entries bound the one new thing below, and it is written to sit
on top of them rather than restate them:

- **`fnv` LV8** owns the *statistic*: an equal-weight mean over enumerated
  cells is a scale-weighted average, and must be reported with a median, a
  sign test, and an SE taken across seeds. The grid here spans **17×** in
  optimal cost from cheapest to dearest cell, squarely in LV8's dangerous band.
- **`mab` LV10** owns *one policy over a one-axis grid*: feed the grid
  parameter as an observation, and expect the generalist to lose to
  specialists at the far end of the span.

Neither covers how the training, scoring and testing targets differ, or how to
build a held-out set that can be read. That is FM1, and it is the only entry
here: everything else this campaign found is either upstream already or is a
statement about inventory, which belongs in `README.md`.

---

## FM1 — sample to train, enumerate to score, hold out a finer grid to test

**context.** A generalist over a parameterized instance family (an IR `grids`
block), where the family has **more than one axis** and the axes are not
interchangeable. Applies to any such family, not to inventory.

**the protocol, in three targets that must not be the same object.**

1. **Train** on the grid's sampler — uniform draws over cells, one artifact for
   the whole family. The checkpoint is chosen **once**, never per cell: a
   per-cell pick silently converts the arm into an ensemble and stops testing
   the claim that one policy serves the family.
2. **Score** by enumerating every cell, one record per cell (spec §9.6).
   The enumerated mean is a legal **selector** — checkpoint screening and any
   tuner objective need a scalar and this is the only one available — and is
   **never a report**. The two uses look identical on screen, print the same
   number, and the selector has optimized it; that is the trap `fnv` LV8
   describes from the reporting side. Prescription: wherever the mean appears,
   print what it is being used for next to it.
3. **Test** on a **second, finer grid** whose cells are a superset of the
   training grid's. Overlap cells are the memorization control; the rest are
   the test. Three constraints make the comparison mean anything:
   - **identical observation width.** Pad every instance to the family maximum
     (spec §5.6) in *both* grids. If the width moves, the artifact cannot be
     fed the eval grid at all.
   - **anchor the range.** Keep the extreme value of each axis in the
     *training* grid and outside the eval grid, so held-out points lie between
     trained ones. Without this you measure extrapolation and report it as
     interpolation.
   - **gate the overlap** with a test asserting the shared cells are the same
     problem field by field. Two traps cost real time here: generator objects
     may define no `__eq__`, so identical laws compare unequal and every shared
     cell reports as differing (compare `repr`); and bars built at different
     seed counts differ by sampling alone, so compare only on matched seeds.

**stratify the held-out set by axis** — one stratum per axis held out alone,
plus one where all are. This is the step that carries the method, because axes
of one family can answer *oppositely* and a pooled number cannot show it.
Here the 42 held-out cells pooled to **84.5%** of optimal against **95.8%** on
trained cells — an 11-point dip that reads as ordinary generalization decay,
and was in fact **96.5%** on one axis and **55.1%** on the other: a pass and a
total failure averaged into "slightly worse". A pooled generalization number
over more than one axis is not a weak result, it is an unreadable one. *Why*
those two axes diverged is this domain's RQ-7 answer and is in `README.md`'s
`cost_leadtime` board, not here; the transferable part is that you cannot find
out without the strata, and the strata cost nothing but a `GROUP BY`.

**failed.** Pooling (above). Reporting the enumerated mean as a score. Choosing
a checkpoint per cell.

**evidence.** #E23 (generalist trained and tuned; context-vs-blind control),
#E24 (the stratified held-out test); `inv_single_grids.py`, and
`inv_single_test.py::test_cost_leadtime_overlaps_grid16_bit_exactly` as the
overlap gate.

---

## Operational note — per-cell records collide silently

Grid evaluation writes one record per cell keyed `(grid, cell, model_stem)`
beside the model. **A second grid eval of the same artifact overwrites the
first**, whatever output flag is passed — a redirect names only the aggregate
row — and every surviving file still looks complete. Two evaluation rounds were
lost to this. Snapshot per-cell records under a run-specific name as they land.
When aggregating them over a glob, pin the **trial**, not the study: a tuning
study leaves per-cell files in every trial directory, and a glob matching all
of them mixes dozens of different policies into one arm. That happened here and
produced a confident, fully-reversed conclusion — the arm read as *worse* than
its comparator when it was better — which then supported four hypotheses before
the join was checked (#E23).

---

## What did not transfer

- **A mechanism imported from an earlier campaign was falsified.** The
  prediction that a fixed ordering cost is what makes reward normalization
  matter was wrong: the effect was as large without the fixed cost. It is a
  **scale** problem, not a structural one — do not carry the structural reading
  to another inventory domain (#E5).
- **Only one of the three normalization knobs generalizes.** Across three
  campaigns in this repo, `norm_reward` ON is unanimous and worth orders of
  magnitude more than the other two; `norm_obs` and `normalize_advantage`
  disagree domain to domain, and this campaign holds the weakest evidence on
  the latter. The transferable rule is *"normalize the reward, then tune the
  other two locally"*, not a fixed triple (#E5).
- **Q-A and Q-B stay in `README.md`.** Reward scale in cost-minimising
  inventory and the encoding of an order quantity that is zero most periods are
  **domain** answers; the general form of the second is already `adi_flex` LV1.

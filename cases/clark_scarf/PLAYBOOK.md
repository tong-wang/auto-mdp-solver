# PLAYBOOK.md — `clark_scarf`

**Structure class.** Serial multi-echelon inventory: a chain of stocking
points, one decision per link per period, committed pipeline flow, backlogged
demand at the tail, finite horizon, discounted cost minimized — and, unusually,
an *exactly solvable* reference (Clark & Scarf 1960).

**Protocol every number below is quoted at.** 8192 deterministic CRN seeds,
deterministic eval, discounted total cost (minimize). Reference: the
verified-optimal decomposition DP at **982.36**; best learned policy 989.07 =
**100.69%** of it. Measured training-seed sd (the resolution floor) **1.57** —
any delta inside it is not a claim.

**What is deliberately absent.** The process, spec and schema lessons from this
campaign went upstream as issues #18–#29 (`ESCALATION.md` §UPSTREAM), which is
where guide §10 routes mid-campaign walls. Restating them here would duplicate
a stronger record. What follows is domain-level, and is written to be read by a
campaign that is **not** this one.

---

## Lever entries

### LV1 — parametrize the action in the coordinates where the optimal policy is CONSTANT

```
context:      after the §8.6 L1 centre was established, with the observation
              axis still open. Fires wherever a reference or predicted optimal
              policy is a threshold / order-up-to / "adjust to level" rule —
              inventory base-stock, (s,S), reservoir and thermostat control.
              Protocol as header.
symptom:      the same problem, same centre, same budget, two action encodings
              11-17 units apart — five times the next largest design lever, and
              far outside the 1.57 floor.
diagnosis:    an order-up-to optimum is a CONSTANT in target coordinates
              (y* = S) and a state-dependent function in quantity coordinates
              (q* = S - u). Encoding the action as the increment forces the
              network to learn the identity map on state before it can express
              the optimum; encoding it as the target hands that for free. The
              argument was pre-registered from the reference policy's form, not
              found by search — the cheap probe is simply writing the known
              optimum in both coordinate systems and seeing which one is
              constant.
prescription: gym-action. Encode the decision as the TARGET LEVEL, not the
              increment, whenever the reference policy is a threshold rule.
              Measured -17.23 and -14.29 (two observation modes) @ header
              protocol; won in all four cells of the 2x2, i.e. the effect is
              configuration-independent, not an interaction.
failed:       masking the infeasible region of the SAME axis instead of
              re-encoding it — rejected twice, on two structurally different
              masks (+3.61 / +2.64 under the quantity encoding, where the mask
              is a one-sided bound; +8.03 / +6.01 under the target encoding,
              where it is two-sided). Constraining an awkward parametrization
              is not a substitute for changing it, and here it cost.
evidence:     #E6 (encoding), #E5 (the masking attempts)
```

### LV2 — price a representation lever at L2, never at L1

```
context:      the campaign's research axis was a pure change of coordinates
              (two observation modes carrying identical information, guarded by
              an invertibility test). Fires for any feature / encoding /
              representation comparison. Protocol as header.
symptom:      at the L1 derived centre the two arms sat 0.55 apart (eval-paired
              +/- 0.14, two training seeds per arm) — inside the 1.57 floor, so
              not a claim, but directionally consistent and easy to narrate as
              a small real effect. After tuning each arm SEPARATELY the same
              comparison read 0.0014 (eval-paired +/- 0.0123, ONE training seed
              per arm).
diagnosis:    at ONE shared centre each arm's score reflects the representation
              AND how well that centre happens to fit it; the two cannot be
              separated, so a small delta is unreadable in either direction.
              Per-arm tuning removes the centre-fit confound from the POINT
              ESTIMATE — here it moved ~400x toward zero. It does NOT buy
              uncertainty: both intervals above are eval-paired, and the
              relevant uncertainty for a representation claim is
              TRAINING-SEED, which per-arm tuning leaves untouched.
              Read the eval interval as evidence and you will conclude the
              tuned comparison is ~11x sharper when its seed design is in fact
              thinner (this campaign made exactly that error and corrected it).
prescription: protocol, in two parts and the second is the one that gets
              skipped. (1) Tune each arm to its own L2(hp) centre before
              quoting the delta. (2) Run MULTIPLE TRAINING SEEDS per tuned arm
              and quote the seed-inclusive interval; an eval-paired SE on one
              seed per arm answers a different question than the one asked.
              Cheap corroboration if a full seed sweep is out of budget: score
              several distinct tuned configs per arm — four here (two per arm,
              different learning rates, rollout lengths, architectures) landed
              within 0.027 of one another, which is convergence evidence even
              though it is not seed evidence.
scope:        one domain, one axis, arms information-identical by construction.
              The DIRECTION does not transfer — a per-arm tuned gap could as
              easily widen. What transfers: a shared-centre delta is
              confounded, and a per-arm-tuned delta on one seed each is
              precise about the wrong thing.
cost:         two studies, ~260 trials each, 2M steps per trial, 20h wall clock
              on 8 workers. Both studies found their best by trial ~80; the
              remaining two thirds bought nothing. Budget the tuning, then stop
              it early on a plateau.
evidence:     #E4 (L1), #E7 (L2 and the tuning trajectory)
```

---

## Frame moves

**FM1 — verify the reference before you trust it, on a fixture small enough to
brute-force.** An "exact" benchmark transcribed from a paper's theorems is
code, and code is wrong in ways that look right: this campaign's first
decomposition DP was **30% suboptimal and entirely convincing** — plausible
policy shape, plausible costs, no symptom. A reference that is wrong *low*
makes every policy look good; wrong *high*, and a genuine result reads as
failure. Neither shows up in the learned policy's own numbers.

The move: before quoting any comparison, build the smallest instance whose
FULL joint state can be enumerated (here N=2, L=1, T=4, mean demand 2 — chosen
so the pipeline dimension collapses), solve it by brute force, and require the
reference to match to **0.000000**. It paid off immediately, and it is what
later licenses calling the reference `exact` rather than merely `feasible` —
the role that makes a leaderboard readable, and the one whose ordering
invariant a gate can enforce.

Cheap, one-off, and the fixture stays in the covering set as a regression.

*(Evidence: #E1.)*

**FM2 — a document that quotes computed numbers is a snapshot; re-render it
when the model moves, and paste the tool's output rather than a summary of
it.** Campaigns produce artifacts that are rendered *once* from the model and
thereafter read as evidence — a worked sample trajectory, a cost breakdown, a
leaderboard row. Nothing re-renders them when the model changes and nothing
diffs them against the source, so they rot silently. A stale artifact is worse
than a missing one, because it reads as confirmation.

The case that earns this: a defect had been charging in-transit stock one
echelon too low. Fixing it moved every cost. The DP and the trained policy were
re-run, so those numbers stayed true — but the Phase-A **round-trip
trajectory**, whose entire purpose is to let a human check the model, was not,
and went on displaying `holding = 109.00` where the corrected model gives
`69.00`. It sat that way for nine re-signings, and surfaced only because an
unrelated schema change forced it to be regenerated. Reading it would never
have caught it: every number in it was internally consistent, just consistent
with a model that no longer existed.

The move, in two parts:

1. **Re-render on every re-freeze.** Anything derived from the model is
   invalidated by a model change, documents included. Put it in the re-signing
   checklist beside the gates or it will not happen.
2. **Paste verbatim tool output, never a hand-trimmed table.** The trimming is
   what let this drift: a summary can diverge from its source without ever
   looking wrong, whereas raw output is a literal diff against the command that
   produced it.

*Scope.* One verified instance, not a pattern established by repetition — but
the mechanism is domain-independent and the failure is silent, which is the
combination worth a standing habit rather than a watch. A later audit found the
leaderboard's `random` row also not reproducing; no cause was established for
that one and none is claimed here.

The stronger form, which this campaign did **not** build: a gate that
re-renders each artifact and diffs it. That needs a convention for recording
each artifact's inputs, which does not exist yet.

*(Evidence: §IR-CHANGELOG F1 for the original defect; the restatement's
regeneration note for the discovery. Upstream took the same lesson into the
pipeline's Phase-A step 7b.)*

---

## Onward

Both lever entries are stated at trigger level rather than in this domain's
vocabulary, because their triggers — "the reference policy is a threshold
rule", "you are comparing two representations" — recur across the inventory
family and beyond. Neither has a second independent campaign behind it yet;
under guide §10 that is what "graduation to a rule" would require.

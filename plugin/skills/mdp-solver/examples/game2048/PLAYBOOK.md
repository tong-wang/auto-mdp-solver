# PLAYBOOK.md — `game2048`

**Structure class.** A spatial-board game MDP: deterministic player effect
followed by an exogenous state-dependent spawn, discrete masked actions, an
unbounded value ladder, no exact DP.

**Protocol every number below is quoted at.** 8192 CRN episode seeds,
deterministic argmax over masked logits, merge score (maximize). The resolution
floor is the **training-seed** spread, not the eval SE: sd 442 at 4×4 against an
eval SE of ~±75; sd 39.6 at 3×3 (n=8).

**What is deliberately absent.** This campaign's process and gate lessons went
upstream as issues **#16, #60–#63**; its IR-formalization fixes are in
`ESCALATION.md` §IR-CHANGELOG. Neither is restated here.

---

## Lever entries

*None.* This campaign's transferable findings are either textbook
(identity encodings, post-decision states), general experimental method that
belongs in the guide rather than a case record, or — for the one genuinely
non-obvious lever it measured, the GAE-λ descent — established on a chassis
this case does not publish and **not used by either shipped crown**, both of
which run at the λ = 0.95 default. An entry asserting a lever the case cannot
demonstrate would be worse than none.

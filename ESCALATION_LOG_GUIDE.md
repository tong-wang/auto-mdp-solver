# Escalation Log Guide — documenting experiments and escalations

*Trial guideline, 2026-07-24. Companion to `AGENT_PLAN.md` §4 (solve = backbone +
escalation layers) and §14 (gym gates). Status: to be validated on a live
experimental project, then revised from feedback and eventually promoted into the
plugin skill as the trace convention feeding the escalation playbook. Distilled from
what worked — and what stayed invisible — in `topk_id` (EQUINET.md, BAYES11.md).
2026-07-29: §IR-CHANGELOG added (formalization reversals), seeded on `mab` P9.
2026-07-29 (same day, second revision): first live trial ran — `game2048` in
rl_test. §3 reframed around the priority-ordered design tree with a frontier
(operator direction), id spaces fixed (§2), interview-reversal tripwire added
(§5), imported-verdict rule added (§7), template updated (§11).
2026-07-29 (third revision): diagnosis entries added to §6 — typed checkpoint
records in the ledger timeline (reads / observed / missing / plan, arbiters
with pre-decided branches); the frontier cites its governing diagnosis.
Proposed from the second live trial, `mab`.
2026-08-12 (fourth revision): §10 rewritten from the `mab` case-close review —
distillation lands as an in-folder `PLAYBOOK.md` of digest entries
(context / symptom / diagnosis / prescription / failed attempts — experiences,
not rules; real names and numbers, ledger cited not copied); the three onward
promotions named (contribution / curation / graduation-to-rule); mid-campaign
upstream proposals split out to the mdp-propose skill.
2026-08-14 (fifth revision): §6 — multi-arm verdicts are tables, not prose,
and §11 gains a multi-arm entry shape. Proposed from the `game2048` campaign
(issue #16), which measured the cost of the prose form: numbers had to be
re-extracted by hand when a later diagnosis entry consolidated dozens of
runs, and two ledger corrections traced to transcription from prose.*

## 1. Why this format

An escalation campaign produces two kinds of knowledge with **different data
shapes**, and only one of them records itself:

- **The ledger** — hypothesis → run → verdict events. Event-shaped, append-only.
  Numbers force it into existence; every lab notebook has it.
- **The map** — the current *frame*: how the gap decomposes into named causes, which
  design axes span the experiment space, what the best validated bundle is, what is
  open vs parked. State-shaped, revised in place. It develops **in discussion** and
  evaporates unless mandated — in `topk_id` it surfaced only as buried consolidation
  checkpoints (EQUINET §8.0, §8.6), found by knowing where to look.

The mandate is therefore asymmetric: the ledger barely needs one; **the map is the
point**. Two audiences read it: the human running the campaign (this is the mindmap
you wish you had), and — later — the mode-2 orchestrator brain, whose diagnosis-seam
working memory this file *is* (AGENT_PLAN §3: cross-stage state must be
filesystem-discoverable). One artifact, both front-doors.

Everything else in the research folder stays free-form. Deep-dive notes in
thinking-voice (EQUINET-style) are encouraged — they just hang off the log via
links, they don't replace it.

## 2. The log — one file, four sections

One `ESCALATION.md` per project/case, git-tracked:

```
ESCALATION.md
├── §MAP              current frame — revised in place, dated
├── §FRAME-CHANGELOG  append-only, one line per re-framing
├── §IR-CHANGELOG     append-only, one entry per formalization reversal
└── §LEDGER           append-only, hypothesis → runs → verdict
```

Cross-references use **four id spaces, one prefix each** — `F{n}` is reserved
by §IR-CHANGELOG (the first trial's frontier briefly squatted on F and
collided with it):

| prefix | space | lifecycle |
|---|---|---|
| `P{n}` | **P = priority**: orders competing siblings at a `designs` split (§3.1) | standing; revised by tree surgery |
| `S{n}` | **S = schedule**: orders coverage at a `cases` or `means` split (§3.1) | standing; children postponable, never prunable |
| `A{n}` | frontier agenda item (§3.5) | consumed → becomes a ledger entry |
| `#E{n}` | ledger entry | append-only |
| `F{n}` | IR-changelog reversal | append-only, sparse |

## 3. §MAP — the current frame

The map's spine is **one priority-ordered design tree**: nodes are design
choices, and children are designed *in the context of* their parent's choice.
(Revised after the first live trial, `game2048`: the earlier decomposition +
grids pair kept generating cross-product cells that were structurally void or
meaningless, while the real structure — each level conditioned on the one
above — was a tree all along.)

**3.1 The design tree.**

- **Levels order by conditioning strength, not expected gain**: a choice sits
  above another when changing it would invalidate the work below it
  (game2048: target scale ≻ action interface ≻ obs transform ≻ obs encoding ≻
  arch ≻ HP — the *biggest* lever, obs encoding, sits mid-tree; HP is at the
  bottom because nothing conditions on it).
- **The root is the frozen IR**, carrying both fingerprints (`mdp` and
  `structural`). The tree is then self-evidently a tree over *one* problem, and
  the §MAP ↔ §IR-CHANGELOG link is mechanical: a moved `mdp` fingerprint
  re-roots the frame, a moved `structural` one need not.
- **The first two layers are fixed.** Below the root comes the **`scenario`
  layer** — siblings are study bases, one named registry object each
  (`{Domain}Scenario`, a sampler, or a `{Domain}ScenarioGrid`), so specialist
  vs generalist stops being an invented axis and becomes *which kind of object
  the sibling names*. Below that comes the **`solver` layer** — siblings
  grouped by `role`, each named by its spec-§9 `{method}` (`method=dp`, never
  "bar": §9 makes `--baseline`/`--reference` a per-comparison choice, not part
  of an artifact's identity). Benchmarks live here, on the tree, because a
  benchmark *is* a solution to the problem; grouping by role is what makes the
  bracket legible, since the two bounding groups sit adjacent. Only opened
  children are drawn, so a single-scenario campaign draws one — no phantom
  coverage debt. The standard eval protocol and the reference bar are
  **per-`cases`-node annotations**, not campaign globals. A campaign whose
  problem demands a different skeleton may deviate, *provided the deviation is
  recorded in §FRAME-CHANGELOG* — a default with a visible cost, not a rule
  enforced by nothing.
- **Default level order below the solver layer** (override per domain by
  conditioning analysis): action interface ≻ observation ≻
  architecture / extractor ≻ training signal (reward shaping, HP) — matching
  the spec-§8.6 escalation layers `L2(gym)` / `L2(arch)` / `L2(hp)`. The
  order encodes **conditioning, not importance**: HP is last because nothing
  conditions on it, not because it moves results least — it is a *covariate*
  of every comparison above it, not a parent (topk_id: "plain HP is bigger
  than either research issue"), which is what rule 10 exists for.
- **Type every split by two questions, into three kinds.** *Does the crown
  fork here or pass through one child? Do the siblings share a frame — the
  same protocol, bar and seed block — so their scores may be subtracted?*

  | | **own frame** (scores incomparable) | **shared frame** (scores comparable) |
  |---|---|---|
  | **crown forks** | **`cases`** — target scales (3×3 → 4×4), scenario families (Gaussian vs Bernoulli), specialist vs generalist | **`means`** — an exact solver, heuristics, the RL artifact; `masked` vs `free`; `echelon` vs `raw` |
  | **crown passes one** | *empty, necessarily* | **`designs`** — competing encodings, architectures, HP |

  The empty cell is the point, and it generalizes into the rule the older
  `S`/`P` typing left implicit:

  > **A score may be subtracted only within a frame.** Across frames a number
  > may be *reported* — labeled as such — but it never selects: no crown, no
  > prune, no `✗`.

  A crown therefore cannot cross a scenario boundary — a consequence now,
  rather than a rule to remember. Specialist vs generalist settles the same
  way: they are `cases`, so tabling them together is a legitimate report and
  never a basis for selection, unless the general branch is re-scored on the
  restricted branch's own frame (same scenario, same seed block, paired).

  **`cases` and `means` children are postponable, never prunable** (`✗`/`∅`
  are illegal on their edges; a child is ✓ covered, ▶ active, or ⏸ postponed
  with a *scheduled return*) — that, not disjointness, is what makes coverage
  coverage. `designs` siblings are rows on one shared leaderboard: crown one,
  prune the rest. **The litmus is redundancy at crowning**: a `designs`
  sibling becomes redundant the moment one is crowned; a `cases` or `means`
  sibling never does (game2048: crowning the masked interface did not make the
  free interface redundant — its value is less machinery, and that survives
  the win). A `cases` split can sometimes be **collapsed by a generalist** —
  one policy covering all cells — where the axis permits (game2048: `prob_4`
  could, `grid_size` cannot: the obs shape changes); `designs` has no
  analogue. The typing is the Phase-A **mode stance** carried onto the tree:
  modes declared "independent branches, reported separately" arrive as
  `cases`, "competing designs to compare head-to-head" as `designs`.

  *(This replaces the `S`/`P` split typing. Two things the old text got wrong:
  it called a coverage split a partition of the problem while its own second
  shape — the extension chain — was nested rather than disjoint; and it left
  sibling sets that are neither, such as an exact DP beside heuristics beside
  the RL artifact, with no legal type. The extension chain is now a `means`
  split with `order: by generality`, which is what licenses its cross-link
  delta as the price of generality. Two shipped campaigns show what the old
  typing cost, one per failure mode. **Mis-typing**: `game2048`'s root splits
  `3x3_20 (P1 ★) / 4x4_20 (P2)` — board size, a coverage axis by this guide's
  own example — as a selection split, so the coverage cell the guide names is
  drawn formally prunable. **Mis-ranking**: `cases/fnv`'s map co-ranked two
  children of one split `P1 ★` — `L0 faithful defaults` beside `L1' corrected
  derivation`, and again under its second target — which a selection split
  cannot mean, since spec §8.6 makes `L0` reporting-only and never
  crown-eligible. That was the §8.6 ladder drawn as a selection split, and it is
  what §3.2's floor-control rule now prevents. **`fnv`'s map was redrawn to this
  taxonomy on 2026-08-17** — the ladder is a chain there now — so read the
  example from this text rather than from the live file.)*
- **Every split edge carries a split id, `{locus}.{axis}`.** The `axis` is the
  same code-read name the child node uses, which is what makes the
  mis-attachment check below performable rather than aspirational — a node's
  axis needs something to disagree *with*. The `locus` is the spec-§8.6 layer
  the split opens (`gym` / `arch` / `hp`), and it **composes** — `arch+hp` for
  a split spanning two, mirroring §8.6's own `L3(hp+arch)` — because a campaign
  that varies `net_arch` with the extractor pinned has one split across two
  layers. The `scenario` and `solver` layers sit above the solve levels and
  take no locus. Beyond the check, the locus is what puts the *cost* of
  re-opening a branch on the tree: a `gym` split means a rebuild, an `hp` split
  a tuning study, and these differ by orders of magnitude. `tier` deliberately
  does not encode that — tier says what a split is *for*, not what it costs.
- **Attributes ride on the edge, not the split** — one split may carry
  children that differ in them:

  | attribute | values | note |
  |---|---|---|
  | `coverage` | `required` (default — blocks case close) / `optional` | the old `must`/`stretch` obligation, now on every crown-forks edge. An `optional` child may stay open at close as an *open extension*, but is never `✗` refuted: crowning a restricted sibling cannot make a more general contract redundant |
  | `order` | none / `by generality` | what makes a chain a chain (game2048: masked ⊂ free; topk_id: bayes (0,0) ⊂ (0,1) ⊂ (1,1)). Climb from the restricted end, levers transferring up as imported priors (§3.2); the cross-link delta is the **price of generality** |
  | `role` | `relaxed` / `exact` / `feasible` | benchmark roles, sense-free — see below |
  | `question` | `1-comparative` / `2-structural` / `3-engineering` | which question the edge answers — see below |
- **Benchmark `role` is defined sense-free.** Let `≽` mean *at least as good
  as*, read off `objective.sense`. Then `relaxed ≽ opt` (unattainable — an LP
  or information relaxation), `exact = opt`, and `feasible ≼ opt` (any real
  policy, **including the RL artifact**). Never "upper"/"lower" bound: the same
  construction is an upper bound in a maximize domain and a lower bound in a
  minimize one, so the word names the domain, not the benchmark. Two things
  follow: a `feasible` sibling scoring strictly better than an `exact` or
  `relaxed` one is impossible and indicts the eval, the bound or the simulator;
  and a campaign holding both `relaxed` and `feasible` **brackets** the
  optimum, so it has a certified gap without an exact solver.
- **Tier the question each edge answers.** `1-comparative` — how does RL
  compare with the existing solutions, exact *and* heuristic? `2-structural` —
  does RL *discover* the structural insight those solutions embody?
  `3-engineering` — which encoding, architecture or HP trains best? Tier 1 is
  **standing**: every campaign asks it, so it is never a per-case declaration.
  A tier may repeat across edges, so "the tier-2 question" means the set of
  edges carrying it. This is what lets a tree say that its largest measured
  effect is not its claim — layer order encodes conditioning, and tier encodes
  what the campaign is *for*, without reordering anything.
- **Siblings carry ranks** — local and standing: **P = priority**, P1 > P2 > …
  orders competitors at a `designs` split; **S = schedule**, S1 > S2 > …
  schedules coverage at a `cases` or `means` split. The letters rank the
  siblings; the **kind on the edge** says what the split is. Both revise via
  REPRIORITIZED lines, but they license different things: a low-P child may
  never run at all (a crowned winner prunes it); a low-S child runs *later*,
  never *never*. S-order is set by transfer (cover the child that teaches the
  most about the others first), cost, claim importance, and prerequisites — and
  pairs with a per-child *return condition* saying when the next child
  activates.
- **Marks**: `✓` validated · `✗` closed · `⏸` pruned/parked (always with a
  reason + tripwire) · `∅` structurally void (*cannot* exist — say why) · `▶`
  in flight · `★` the crowned path. **Marks ride on the edge**, where the
  selection they judge happened.
- **A node carries two lines, fixed.** The shape is

  ```
  {axis}={option}
  {score}{ (Δ)} · {#E ids}
  ```

  with the **axis name read from the code** — the gym kwarg, the registry key,
  the `{method}` of `{domain}_benchmark_{method}.py` — never coined. A node
  whose axis disagrees with the axis in its incoming edge's split id is then
  visibly mis-attached, which is the cheapest way to catch a child parented
  *under* a sibling rather than beside it — the check that caught a
  continuous-action variant sitting under `action_mode=target_discrete` instead
  of beside it, five revisions in. A parent's score is the best in its subtree. Mechanisms,
  pre-registrations and diagnostics do **not** go in nodes.

  The principle that keeps this stable when a campaign wants one more line:
  **the tree is an index into the ledger.** A node label must be *stable*, not
  *complete* — an index that reproduces its target is not an index. That also
  disposes of config bundles (`--extractor small --channels 64 128
  --features_dim 256`) with no naming machinery: the bundle lives in the entry,
  one hop away.
- **A node's score and its Δ must come from the same configuration.** If the
  delta was measured elsewhere, either the node moves to where it was measured
  or the Δ leaves the node; a Δ against the node's own *parent* belongs on the
  connecting edge, which names both endpoints. This is the node-level form of
  what rules 3 and 9 govern for protocols, and it is invisible without the rule:
  a leaf reading `992.54 (Δ −183.58)` whose Δ was measured at the
  zero-knowledge start asserts that the derivation was worth 183 units *at the
  crowned configuration*, which was never measured. Checkable by eye once the
  `#E` id is on the node.
- **Layers are named by axis and cited by ledger id — never numbered.**
  "`gym.action_mode` (#E6)", not "T1" or "level 3". A numbered scheme collides
  with spec §8.6's `L0/L1/L2`, whose contents are the *training-signal layer* —
  so "level 2" and "L2" would mean different things three lines apart — and it
  adds a second index the reader must join back to the ledger.
- **Bounds attach to nodes**, and are recorded in the readings table beneath
  the diagram rather than inside the node label. A branch with a validated
  delta is worth ten without (topk_id's localization control bounded
  node-pooling at ≤0.105 OC; game2048's ladder: +29% config, +46% 2-D
  structure, +64% value one-hot — each rung isolated by its own control).
- **Exactly one table beneath the diagram**, keyed by node:
  `node | kind · attributes · tier | reading | entry`. Typing and finding on
  one row — two tables restate each other (a `crowned` column repeats the
  diagram's `★`, an `entries` column repeats the reading's citation).
- **The tree is dynamic.** REPARENTED / SPLIT / REPRIORITIZED / SHATTERED are
  expected moves, each costing one changelog line (§4). A campaign that
  passes its bar re-roots the decomposition on *headroom above current best*;
  the bar itself moves to the off-tree register as a calibration item.

**3.2 Expansion disciplines** — what keeps the tree from losing the grid's one
virtue, mechanical generation of the cell nobody thought of:

- **Prune visibly.** A branch declined by judgment appears anyway: one line,
  reason, tripwire (`⏸`). Never silently omitted, and never confused with `∅`
  impossible.
- **Contrastive controls are siblings; floor controls are chain parents.** A
  crowned branch's children must include the controls that decompose its win —
  when the control *competes* on the same leaderboard (game2048: onehot's +64%
  stayed confounded with the CNN until an MLP-on-onehot sibling existed). That
  is a `designs` sibling, and a control is precisely a tier-3 sibling of a
  tier-2 claim, which marks it as a control with no new mark. It is the wrong
  shape for a **floor** control: spec §8.6's `L0` is reporting-only and "never
  a gate", so it is never eligible for the crown, and drawn as a sibling it
  asserts a selection that never happened — a reader cannot tell "we compared
  these and one lost" from "we started here and escalated". §8.6's levels are a
  ladder, and a ladder draws as a **chain**: `L0 → L1 → L2`, with the
  escalation delta on the edge, where both of its endpoints are named.
- **Imported levers are labeled.** A lever pre-validated in a prior campaign
  enters as a high-prior branch with its provenance attached; it is either
  re-validated in-campaign or its claims ship with the label (game2048:
  `onehot` re-validated at the standard protocol; `log2` labeled "operator
  recollection, no surviving A/B" with a parked re-validation control).

**3.3 Slice views (grids, on demand).** Render a 2-axis table only when a
genuine cross-branch interaction is under test; cells hold run IDs + headline
metric + mark (`#38 −0.3625 ✗`). Generated from the tree when the question
arises, never maintained as a second source of truth (topk_id's C 2×2 was
exactly this — a slice, not the map).

**3.4 Current best bundle = the crowned path.** The `★` path root→leaf, with
pairing constraints on jointly-validated edges (topk_id: γ=1 × nopass —
"neither validated alone"; game2048: the shaping penalty exists only under
the free interface, so adopting one commits to the other). Adopt/reject
paths, not components.

**3.5 The frontier — the execution plan.** A priority-ordered queue of `A{n}`
**agenda items**: each is an action *at* a tree node (act on the incumbent /
expand a new child) or an **off-tree obligation** — work that consumes budget
**without being a solution**: bar *calibration* (verifying a DP against a
brute-force joint DP), protocol construction, floor measurement. Keep an
off-tree register beside the queue. The benchmarks themselves are **not**
off-tree: they are solutions, they live on the `solver` layer (§3.1), and
filing them in the register is what forces a campaign to leave the tree to read
a score against its bar.  Semantics:

- P is local and standing; A is global and dated. Default order = follow the
  P's down the crowned path.
- Legitimate divergences, each explicit: off-tree items carry no P; P's never
  order actions under *different* parents (the queue resolves those, crowned
  path first); overrides are allowed at the cost of a REPRIORITIZED line.
- `⏸`/`∅` nodes emit no agenda items until their tripwire fires (topk_id
  issue (3): parked with the "never route MAP μ,σ into the forward path"
  guardrail).
- A completed A becomes a ledger entry; the P's persist. The frontier
  subsumes "open questions, ranked" — parked items live as `⏸` nodes.
- **The frontier cites the diagnosis that last reordered it** (§6): "frontier
  as of #E{n}" — every queue order traces to a dated reasoning record.
- **Split kinds bind the queue**: an agenda item that crowns a `designs` child
  licenses pruning its siblings; completing a `cases` or `means` child never
  reduces its siblings' obligation — postponed children are coverage *debt*, and a
  campaign does not close while one is outstanding (a deliberate scope cut
  is a re-framing of the root, changelog line owed, not a prune).

## 4. §FRAME-CHANGELOG — the evolution that matters

Append **one line per re-framing** — not per routine update. Date + change +
forcing evidence (run IDs). Vocabulary: *introduced / split / merged / narrowed /
refuted / parked / un-parked*, plus the tree-surgery verbs *reparented /
reprioritized / retyped / shattered* (§3.1). Example lines (topk_id, reconstructed):

```
2026-07-17  two-issue → three-bucket: plain HP is bigger than either research issue (C 2×2)
2026-07-18  "k-features pay for k≥2" NARROWED to k=2-only (D: kf −0.073 vs attn1)
2026-07-18  "C record is mostly HP" REFUTED (honest 8192: arch win +0.061 real)
```

This ten-line spine is the "how did our understanding develop" artifact — and the
raw feed for the playbook's frame-move entries. Full history of the map is git's
job (§9); the changelog is the *interesting* history.

## 5. §IR-CHANGELOG — formalization reversals

Phase A produces its own kind of lesson, and it is precisely the kind no gate
records: an IR can validate, differentially match bit-exactly, and pass every
conformance check while faithfully simulating the **wrong model** (mab P9: the
twin was self-consistent; the arms just shared one primitive draw per period).
These lessons exist only as experience, and experience evaporates. Reversals
are sparse — a handful per case — so they need none of the map/ledger
machinery, just the changelog shape: append-only, one entry per reversal.

**When an entry is owed** — three tripwires, the first two machine-visible:

1. **Phase-A veto** — a `human_override` in `assumptions_log` whose *reason*
   generalizes beyond this case (skip routine parameter overrides).
2. **Post-freeze fingerprint move** — any commit where the structural
   fingerprint changes after the Phase-A gate owes an entry citing the ledger
   id that forced it (binding rule 7).
3. **Interview reversal** — the draft model changed shape *during* Phase A
   because a human answer falsified a plausible formalization, before any
   freeze existed for the machine tripwires to guard. This one is on the
   operator. (game2048 F1: the inherited `sum` reward measured survival, not
   play — caught in the objective round; the most teachable of that
   campaign's four entries, and neither machine tripwire would ever have
   fired. All Confirmables ended `human_confirmed`, so tripwire 1 was silent
   too.)

Entry schema:

```
### F{n}  {date} — {one line: what was re-formalized}
decision:  <the IR construct at stake>
initial:   <what was formalized first AND why it looked right — the
            plausible-wrong derivation is the teachable part>
signal:    <human-veto | gate:<name> | differential | downstream-impl |
            training-symptom | upstream-change> + citation (ledger id, commit)
symptom:   <the observable that falsified it>
fix:       <the corrected IR shape; upstream change if one was needed, cited>
rule:      <condition-first generalization — indexed by problem-shape trigger,
            usable verbatim as a Phase-A checklist line>
```

`initial` is not "what was wrong" — it is the derivation that made wrong look
right, because the next formalizer will re-derive exactly it. `rule` must lead
with its trigger condition ("if a decision selects *which* exogenous stream is
read …"), never with the domain: a rule that cannot be tested against a raw
problem statement during the Phase-A interview is not yet a rule. Where a
cheap probe exists (a two-draw coupling check, a reward-information audit),
put it in the rule — detection moved left is the entire value: a Phase-A veto
costs one interview round; the same defect caught downstream costs voided
runs (mab: four PPO runs, four benchmark evals, an upstream schema change).

Worked example (retailer, reconstructed):

```
### F1 — per-step reward leaked the latent
decision:  trained-on reward mode
initial:   reward referenced the latent-optimal gap — the objective already
           references the latent, so reusing it per-step looked consistent
signal:    human-veto
symptom:   the per-step reward fed to the policy reveals hidden state the
           agent is supposed to infer from its observations
fix:       latent confined to the eval metric / terminal reward / an
           action-independent normalizer (retailer `gap`); every trained-on
           reward is computable from the agent's own information set
rule:      if the objective references a latent, audit every per-step reward
           mode the policy trains on: latent terms must be absent or
           action-independent — the objective and eval metrics may keep them
```

## 6. §LEDGER — the experiment record

Append-only entries; corrections are new entries citing the old (never edit a
verdict in place). Per entry:

- **id** — short, stable (`#38`, or `E12`); this is what map cells and changelog
  lines cite.
- **address** — the map bucket it attacks + the grid cell it occupies.
- **hypothesis** — what this run decides, stated *before* launch.
- **runs** — commands / artifact paths, enough to reproduce.
- **verdict** — numbers, at the standard eval protocol (see §7), with the
  comparison that gives them meaning. **Neutral and negative verdicts are
  first-class** ("roughly free, not yet proven positive" is a verdict; so is
  "8M ≈ 4M — not undertrained").
  **Three or more arms, or any readout the reader must scan across, goes in
  a table** — arm | metric | delta | z-or-band — with prose kept for
  mechanism, scope conditions and interpretation. One or two numbers stay
  inline; a table for a single contrast is ceremony. The reason is not
  tidiness: cross-entry contrasts at a shared protocol are the campaign's
  core operation, and a table aligns them by construction while prose
  leaves them to be re-extracted by hand — a real cost when a later
  diagnosis entry consolidates dozens of runs, and a source of
  transcription errors when it is paid. This does **not** contradict §3.3's
  "generate slice views, never maintain them": that rule governs *standing
  state* which can drift out of sync with the tree. A verdict is
  event-shaped and append-only, so a table inside one is part of the
  immutable record, not a second source of truth. The same applies to
  `README.md` leaderboards and to multi-arm effects in a `PLAYBOOK.md`
  `prescription:` (§10).
- **status** — `✓ / ✗ / ~ (neutral) / ▶`.

**Diagnosis entries.** Not every ledger entry tests a hypothesis. At a
checkpoint — several agenda items landed, or the next spend depends on
attributing the remaining gap to competing causes — append a **diagnosis**
entry: same `#E` id space, typed in the heading (`DIAGNOSIS:`). It is the
mandated record of the map discussion that otherwise evaporates (§1): the
map carries the *conclusions* as standing state; the diagnosis carries the
*reasoning*, event-shaped and interleaved with the runs it reads and the runs
it spawns. Schema:

- **reads** — the verdicts consolidated (#E ids; imported evidence with its
  provenance label, rule 8).
- **observed** — what is now established, scope conditions attached.
- **missing** — the named gaps; these become tree nodes (the candidate
  causes/potentials the next budget arbitrates between).
- **plan** — the A items spawned, each with its arbiter and **pre-decided
  verdict branches, stated before the probe runs** — the ledger's
  hypothesis-before-launch discipline, applied to planning: without it a
  diagnosis is retrofitted storytelling.

Map surgery from a diagnosis costs the usual changelog lines, and the
frontier cites the diagnosis that last reordered it (§3.5). No verdict/status
fields — a diagnosis is superseded by the next diagnosis, never marked
right or wrong in place.

## 7. Binding rules

1. **No run without an address.** Every launched run cites a map bucket + grid
   cell. Can't place it → either the map is stale (re-frame first) or the run is
   unmotivated (don't launch). In agent form this is the budget-allocation
   invariant: allocating compute *is* coloring the map.
2. **Bidirectional citation.** Map revisions cite the ledger IDs that forced them;
   frame changes get a changelog line; ledger entries carry their address. An
   *address* is a link, not prose — "under the crowned interface" is
   unfollowable and goes stale unnoticed:

   | direction | mechanism |
   |---|---|
   | tree → ledger | the readings row cites the entry: `[#E4](#E4)` |
   | ledger → tree | the entry carries a back-link **naming its nodes**: `↑ [design tree](#MAP) — explains {node}, {node}` |
   | targets | `<a id="MAP"></a>` before the map, `<a id="E{n}"></a>` before each entry |

   Explicit anchors rather than heading-derived ones, which are built from the
   whole title and die on any retitle — the mutable-identity trap one level
   down. Naming the nodes is not decoration: mermaid nodes cannot be anchored
   individually, so the entry must say *which* node it explains, and one entry
   explaining two nodes on different layers (a factorial re-measuring an axis
   under a newly crowned parent) is then visible instead of hidden in prose.
   Node clicks are not a substitute for the readings table: as observed in one
   markdown preview, explicit in-document anchors resolve and mermaid
   `click … href` reaches external URLs, but a *fragment* href from inside the
   mermaid SVG does not navigate.
3. **Eval honesty.** Verdicts and records only at the campaign's declared standard
   eval protocol (fixed episode/seed count, faithful reward mode). Numbers from
   smaller evals are marked *provisional* and never crown a record (topk_id:
   −0.2829@1024 was selection-bias inflation; honest 8192 said −0.3482).
4. **Bundles stay bundles.** Interacting levers validated jointly are recorded and
   adopted jointly (§3.3).
5. **Scope conditions travel with claims.** "Attention pays" is not a claim;
   "attention pays for k≥2, robust across C/D" is — and it names the structural
   driver (boundary complexity, not instance size), so later evidence can narrow
   it cleanly.
6. **Gate obligations noted at the lever.** A ledger entry that touches the gym
   layer notes its gate per AGENT_PLAN §14 (obs within the filtered view / action
   soundness + coverage / shaping trains-only, selection on the faithful mode).
   HP/arch entries: no gate.
7. **No fingerprint move without an IR-changelog entry.** The structural
   fingerprint changing after the Phase-A gate means the model itself was
   re-formalized; the F-entry (§5) records why the first formalization looked
   right and what falsified it. Machine-checkable, CI-enforceable later.
8. **Imported verdicts carry provenance.** Evidence from outside the
   campaign's protocol (a prior project, a recollection) may crown or prune a
   branch only with its provenance attached; dropping the label requires an
   in-campaign re-validation at the standard protocol. Rule 3's eval honesty,
   extended across campaign boundaries.
9. **No number from an ungated implementation.** A quantity produced by a
   second implementation of the MDP core (a vectorized probe, a batched
   replay) enters the ledger only if that implementation passes its
   spec-§1.2 gates in `{domain}_test.py` — declared equivalence, proven, in
   the tracked suite. And a *distributionally*-gated implementation licenses
   contrasts, not levels: paired differences on its own block may be cited;
   absolute levels come only from the canonical eval. Rule 3's eval honesty,
   extended to the instrument that produced the number.
10. **Prune on decisive margins only — HP is a covariate.** Comparisons above
   the HP level are taken at the shared derived center (the spec-§8.6 L1
   config): a valid *screen*, and a margin far beyond plausible HP
   sensitivity prunes (game2048's +46%/+64% representation rungs). A close
   margin is **HP-confounded**: status `~`, never `✗` — tune each surviving
   sibling (per-cell `L2(hp)`) before crowning or pruning (topk_id's C 2×2:
   no judgment was possible before per-combination tuning). The scope
   condition "at the L1 center, untuned" travels with every screened verdict
   (rule 5).

## 8. What stays out

- **Dispatch state** — what is running where (WATCH_STATUS-style). Ephemeral,
  different lifecycle; keep it in its own file or the dispatcher. Don't let it
  colonize the map (`▶` marks are pointers, not process management).
- **Raw outputs** — `results/` folders, TB logs. The ledger cites **run
  names, never paths** — the timestamped run name is the join key, stable
  under any reorganization, and `grep <run_name>` in this file is the
  reverse index (which entries used this run; a run no entry cites is
  deletable). Two disciplines keep the key stable: **run dirs are immutable
  once created** — status (voided, superseded, crowned) belongs in the
  ledger, never in the folder name (renaming runs is how a campaign killed
  4 of its first 9 citations in a day) — and a run's birth entry rides its
  own args log via the train script's `--tag` flag (spec §9.1), so the
  forward join needs no grep at all.
- **Spec/playbook content** — conventions live in the spec; the log records what
  *happened*, cites the rest.

## 9. History and rendering

- **Revise the map in place.** Git is the full history — commit map edits with the
  forcing evidence in the message; `git log -p ESCALATION.md` replays every frame
  state. Snapshot-hoarding inline is the failure mode this section exists to
  prevent; at most, park a superseded diagram of a *major* re-framing in a
  collapsed `<details>` block.
- **Ephemeral views are generated, never maintained.** A live dashboard for a big
  fan-out (map + running-status coloring) is rendered *from* this file on demand;
  it is not a second source of truth.

## 10. Case close — distillation into `PLAYBOOK.md`

When the campaign ends (positive or negative), run one distillation pass into
**`PLAYBOOK.md` in the case folder**. Distilled experiences are still
experiences, not rules — but distilling means *digesting*, not dumping: an
entry carries only what is necessary to reuse the experience — context,
symptom, diagnosis, prescription, failed attempts — with real names and real
numbers (the file sits beside the case), and **cites** the ledger for
everything else instead of copying it. An entry without its story is
unusable; an entry buried in its story is unread.

File layout — a header, three passes, and the negative space:

- **header** — the structure class in one sentence, plus the eval protocol
  every number below is quoted at (seeds, CRN, oracle/reference), stated once;
- **ledger → lever entries** (`LV{n}`), one block each:

  ```
  context:      <only what the entry cannot be read without: where in the
                campaign it fired, what was already ruled out; protocol
                deltas from the header>
  symptom:      <observable signature>
  diagnosis:    <mechanism — plus the cheap probe that confirmed it, if any>
  prescription: <lever layer (HP | gym-obs | gym-action | gym-reward | arch |
                algo | protocol | process | interpretation) + the change;
                measured effect @ the header protocol; scope conditions>
  failed:       <attempts on the same symptom that lost, one line each>
  evidence:     <#E ledger citations>
  ```

- **frame-changelog → frame moves** (`FM{n}`) — reusable ways of structuring
  the search (bound-before-build, gap-decomposition-into-buckets,
  cell-taxonomy placement), each with the moment it paid off;
- **ir-changelog → modeling rules** (`MR{n}`) — the `rule:` lines, under
  lever layer `IR-formalization`, trigger-first phrasing mandatory;
- **what did not transfer** — expectations imported from earlier campaigns
  that failed here. Negative campaigns distill too: an envelope verdict
  ("feasibility-dominated combinatorial + classical solver exists → RL not
  competitive") is playbook content, not just a dead end.

The log format above is designed to make this pass near-mechanical:
addresses, citations, and scope conditions are already in place.

From here the entries have three onward promotions, in increasing rarity:

1. **Contribution** — the playbook travels *inside the case folder* on a
   case PR (mdp-contribute). There is no playbook-only path: a case too
   sensitive to contribute (even re-skinned) keeps its playbook at home.
2. **Curation** — the maintainer indexes shipped entries in the plugin's
   `PLAYBOOK.md` (links into promoted examples' playbooks); solve-time
   consultation is symptom-matched and analogized, never obeyed.
3. **Graduation to a rule** — reserved for entries confirmed across ≥ 2
   independent campaigns that should fire *without* symptom-matching.
   Targets: the Phase-A checklist consulted at the classifying-randomness
   step (modeling rules — why trigger-first phrasing is mandatory), or a
   spec-§8.6 default (lever entries).

Separately from distillation: walls hit **mid-campaign** — a spec/schema/
process gap the campaign had to work around — are filed when fresh via
**mdp-propose** (`UPSTREAM_PROPOSAL_<slug>.md` → one issue each), never
batched to case close; a proposal is pinned against a spec version and goes
stale.

## 11. Template

```markdown
# {project} — escalation log

<a id="MAP"></a>
## MAP  (as of {date})

### Design tree
​```mermaid
graph TD
    ROOT["IR {domain} v{n}<br/>mdp {fingerprint} · structural {fingerprint}"]
    ROOT ==>|"cases · scenario · S1 · required ▶"| SC1["scenario={registry key}<br/>{score} · #E{n}"]
    ROOT -->|"cases · scenario · S2 · required ⏸"| SC2["scenario={next}<br/>coverage debt; return: {when}"]
    SC1 ==>|"means · solver · S1 · role=exact · tier=1 ★"| M1["method=dp<br/>{score} · #E{n}"]
    SC1 ==>|"means · solver · S2 · role=feasible · tier=1 ★"| M2["method=ppo<br/>{score} · #E{n}"]
    M2 ==>|"L0→L1 escalation (Δ {x})"| L1["level=L1<br/>{score} (Δ {x}) · #E{n}"]
    L1 ==>|"designs · gym.{axis} · P1 · tier=3 ★"| C1["{axis}={option}<br/>{score} (Δ {x}) · #E{n}"]
    L1 -->|"designs · gym.{axis} · P2 ⏸"| C2["{axis}={option}<br/>{score} · #E{n}"]
    L1 -.->|"∅"| C3["{axis}={option}<br/>{why it cannot exist}"]
    C1 ==>|"designs · arch+hp.{axis} · P1 · tier=3 ▶"| C4["{axis}={option}<br/>{score} · #E{n}"]
​```

### Layers and node readings   (one table — kind, attributes, tier, reading, entry)
| node | kind · attributes · tier | reading | entry |
|---|---|---|---|
| `{axis}={option}` | designs · tier=3 | {what it established, with scope} | [#E{n}](#E{n}) |

### Frontier
1. **A1 — {action} @ {tree path}**  {▶|queued} — {evidence; cost}
- parked: {⏸ node} — tripwire: {…}

### Off-tree register   (budget-consuming, *not* solution-touching)
- {bar calibration | protocol construction | floor measurement | next campaign roots}

### Current best bundle
{the ★ path + jointly-validated edges + headline @ standard eval}

### Slice view  (only while a cross-branch interaction is under test)
|              | {axis-B v1}   | {axis-B v2}   |
|--------------|---------------|---------------|
| {axis-A v1}  | #1 {num} ✓    | #4 {num} ✗    |

## FRAME-CHANGELOG
{date}  {INTRODUCED|SPLIT|NARROWED|REFUTED|PARKED|UN-PARKED} {claim} ({evidence ids})

## IR-CHANGELOG
### F{n}  {date} — {re-formalization}
decision: {IR construct}
initial: {what + why it looked right}
signal: {type} ({citation})
symptom: {observable}
fix: {corrected shape}
rule: {trigger-first generalization}

## LEDGER
<a id="E{n}"></a>
### #{id} {date} — {one-line hypothesis}
address: {tree path} / A{n}
↑ [design tree](#MAP) — explains {node}, {node}
runs: {commands or paths}
verdict: {numbers @ protocol; comparison}   status: {✓|✗|~|▶}

### #{id} {date} — {one-line hypothesis, multi-arm}
address: {tree path} / A{n}
runs: {commands or paths}
verdict: {what the arms share — protocol, seeds, common start}
         | arm        | {metric} | delta  | z    |
         |------------|----------|--------|------|
         | {control}  |    {x}   |    —   |   —  |
         | {arm 1}    |    {x}   |  {+/-} | {z}  |
         | {arm 2}    |    {x}   |  {+/-} | {z}  |
         {one line of mechanism — the table carries the numbers}
         status: {✓|✗|~|▶}

### #{id} {date} — DIAGNOSIS: {checkpoint one-liner}
reads:    {#E ids + imported evidence w/ provenance}
observed: {established, scope attached}
missing:  {named gaps → tree nodes}
plan:     {A items + arbiters + pre-decided branches}
```

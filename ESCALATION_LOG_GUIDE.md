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
runs, and two ledger corrections traced to transcription from prose.
2026-08-21 (sixth revision): §12 added — local extensions, and the one under
evaluation (`§CONFIG-REGISTRY`, upstream issue #60, proposed from `game2048`).
Guidance only: written down so two campaigns trying it produce comparable
artifacts rather than two dialects, with its open points recorded beside it.
2026-08-24 (seventh revision): two proposals from `adi_flex`, the second
campaign to run §12's axis test. **Issue #65** — the shared-frame split kinds
are renamed to upstream's own words, `means` → `design-axes` and `designs` →
`escalations` (the old `designs` was the near-opposite of spec §8.4's "design
axes"); a `design-axes` split must now cite the IR declaration it draws, and
design axes are write-once (§3.1). **Issue #60** — `§CONFIG-REGISTRY` graduates
from guidance to a rule and moves to §13, with its four open points settled or
stated; §2's spine is five sections, §7 rule 1 asks for a base as well as an
address, and §11 gains the section. Section numbers below §12 are unchanged on
purpose: existing logs cite them.*

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

## 2. The log — one file, five sections

One `ESCALATION.md` per project/case, git-tracked:

```
ESCALATION.md
├── §MAP              current frame — revised in place, dated
├── §FRAME-CHANGELOG  append-only, one line per re-framing
├── §IR-CHANGELOG     append-only, one entry per formalization reversal
├── §CONFIG-REGISTRY  living — the bases runs cite; ids append-only (§13)
└── §LEDGER           append-only, hypothesis → runs → verdict
```

Five sections are the **spine**, not a ceiling: a campaign that needs more adds
a *local extension* and labels it as one, so a reader can tell campaign-local
structure from format (§12).

The four narrative sections answer *what question does this run answer*;
§CONFIG-REGISTRY answers *what was this run configured from*. It is the one
**living** section besides §MAP, and unlike §MAP it is not a rendering of
anything on disk — it is authored design state, the input to runs (§13).

Cross-references use **five id spaces, one prefix each** — `F{n}` is reserved
by §IR-CHANGELOG (the first trial's frontier briefly squatted on F and
collided with it):

| prefix | space | lifecycle |
|---|---|---|
| `P{n}` | **P = priority**: orders competing siblings at an `escalations` split (§3.1) | standing; revised by tree surgery |
| `S{n}` | **S = schedule**: orders coverage at a `cases` or `design-axes` split (§3.1) | standing; children postponable, never prunable |
| `A{n}` | frontier agenda item (§3.5) | consumed → becomes a ledger entry |
| `#E{n}` | ledger entry | append-only |
| `F{n}` | IR-changelog reversal | append-only, sparse |
| `sc`/`g`/`a`/`h`+`{n}` | config-registry axis id (§13) | append-only, dense from the L1 origin at 0 |

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
  | **crown forks** | **`cases`** — target scales (3×3 → 4×4), scenario families (Gaussian vs Bernoulli), specialist vs generalist | **`design-axes`** — an exact solver, heuristics, the RL artifact; `masked` vs `free`; `echelon` vs `raw` |
  | **crown passes one** | *empty, necessarily* | **`escalations`** — competing encodings, architectures, HP |

  The two shared-frame kinds take their names from vocabulary upstream already
  owns: spec §8.4's **experiment design axes** ("define what problem is being
  solved") and §8.6's **L2+ — escalations**. That is also the table's
  structural reading: the root region (`cases` × `design-axes`) is the
  **question space**, fixed by the IR at Phase A; the leaf region
  (`escalations`) is the **search**; the L1 origin is the boundary between
  them.

  *(Named `means`/`designs` from v0.8.3 to v0.9.24. The old `designs` was the
  near-opposite of the spec's own "design axes" — one names a prunable
  candidate answer, the other the fixed question — and a real tree contained
  the sentence "a `designs` split on a design axis", which reads as either a
  tautology or a contradiction and is neither. Append-only ledger text keeps
  the old spelling wherever it was written; grep for both.)*

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

  **Design axes are address-forming, and therefore write-once.** A
  `design-axes` value names a **cell**, exactly as the scenario above it does:
  fixed where it is drawn, never moved below it.

  > **No escalation may overwrite a design axis.** An arm below a fixed
  > design-axis value that sets that axis to something else claims one cell's
  > address while producing another cell's number. It is a *different cell* —
  > refused rather than recorded, and reached by drawing it where it belongs.

  The same shape as the frame rule one layer down: that one forbids crowning
  across a scenario boundary, this one forbids a run drifting across a
  design-axis boundary its own address cites. Where §13's config registry is
  adopted, a declared deviation on a design axis is refused for the same
  reason — the deviation grammar spells escalations, never cells.

  **`cases` and `design-axes` children are postponable, never prunable** (`✗`/`∅`
  are illegal on their edges; a child is ✓ covered, ▶ active, or ⏸ postponed
  with a *scheduled return*) — that, not disjointness, is what makes coverage
  coverage. `escalations` siblings are rows on one shared leaderboard: crown one,
  prune the rest. **The litmus is redundancy at crowning**: an `escalations`
  sibling becomes redundant the moment one is crowned; a `cases` or `design-axes`
  sibling never does (game2048: crowning the masked interface did not make the
  free interface redundant — its value is less machinery, and that survives
  the win). A `cases` split can sometimes be **collapsed by a generalist** —
  one policy covering all cells — where the axis permits (game2048: `prob_4`
  could, `grid_size` cannot: the obs shape changes); `escalations` has no
  analogue. The typing is the Phase-A **mode stance** carried onto the tree:
  modes declared "independent branches, reported separately" arrive as
  `cases`, "competing designs to compare head-to-head" as `escalations`.

  **Where a `design-axes` split comes from is derivable, not judgment.** The
  paragraph above states the consequence (never prunable) and the litmus
  (redundancy at crowning), and neither says where such a split *originates*,
  which leaves the typing unlintable. It need not be: a `design-axes` edge is
  the **drawing of an IR declaration**, and must cite one.

  | `design-axes` split | the IR declaration behind it |
  |---|---|
  | `dp` / `ap` / `rule` / `ppo` | the `benchmarks` block + the artifact's `rl.algo` (spec §9.9 roles) |
  | `echelon` vs `raw` | `gym.observation_modes` + the research question's declared instrument |
  | `masked` ⊂ `free` | `gym.action_modes` |
  | algo families | `rl.algos` — spec §8.6 gives each family its own L0 line and L1 table |

  That is the derivable statement of "never prunable": the IR said the
  comparison is the deliverable, so crowning one side cannot make the other
  redundant. An `escalations` edge is campaign-invented by contrast, and must
  **not** be the declared instrument of a research question — which is the
  check, in both directions. **Kind and axis stay orthogonal**: an
  `escalations` split *on* a design axis is legal, and means the campaign
  escalated by minting new points on that axis (probe encodings, which enter
  the IR by a logged F-entry and are prunable). What is illegal is the third
  thing — an escalation that *moves* a design axis whose value its own address
  fixes, which the write-once rule above refuses.

  **The cost of getting this wrong is measured.** One campaign drew `vec` vs
  `vec_mip` — a research question's declared instrument — as an `escalations`
  split, under which "crown one, prune the rest" licensed pruning the claim
  arm. Round 1 ended with `vec_mip` losing by 74.6 points at seed sd 887, a
  textbook prune. Coverage-required — the IR declaration — is what forced
  round 2, which found the gap was a VecNormalize artifact and that the arms
  tie at 0.02 ± 1.12: the campaign's headline. Under the mistyping, that
  finding does not exist.

  *(This replaces the `S`/`P` split typing. Two things the old text got wrong:
  it called a coverage split a partition of the problem while its own second
  shape — the extension chain — was nested rather than disjoint; and it left
  sibling sets that are neither, such as an exact DP beside heuristics beside
  the RL artifact, with no legal type. The extension chain is now a `design-axes`
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
  orders competitors at an `escalations` split; **S = schedule**, S1 > S2 > …
  schedules coverage at a `cases` or `design-axes` split. The letters rank the
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
  is an `escalations` sibling, and a control is precisely a tier-3 sibling of a
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
- **Split kinds bind the queue**: an agenda item that crowns an `escalations` child
  licenses pruning its siblings; completing a `cases` or `design-axes` child never
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

1. **No run without an address, and no run without a base.** Every launched
   run cites a map bucket + grid cell. Can't place it → either the map is stale
   (re-frame first) or the run is unmotivated (don't launch). In agent form this
   is the budget-allocation invariant: allocating compute *is* coloring the map.
   The address says which *question* the run answers, and two arms can share one
   exactly while differing on four knobs — so a campaign carrying a
   §CONFIG-REGISTRY also cites a config tuple plus its declared deviations
   (§13). A design axis in that tuple is part of the address, not part of the
   configuration: it is write-once (§3.1), so a deviation may never spell one.
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
- **Renderings of what is on disk** — run tables, step counts, headline metrics
  spliced back into the log. Those are generated views (§9), and the issue-#11
  disposition rejected a maintained one. §CONFIG-REGISTRY is not an exception
  to this: it runs the other way, holding *authored design state* that runs are
  configured **from**, with the same lifecycle as §MAP and nothing generating
  it. The tell is direction — a rendering can be rebuilt from the archive, a
  base cannot be recovered from it at all.

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
    SC1 ==>|"design-axes · solver · S1 · role=exact · tier=1 ★"| M1["method=dp<br/>{score} · #E{n}"]
    SC1 ==>|"design-axes · solver · S2 · role=feasible · tier=1 ★"| M2["method=ppo<br/>{score} · #E{n}"]
    M2 ==>|"L0→L1 escalation (Δ {x})"| L1["level=L1<br/>{score} (Δ {x}) · #E{n}"]
    L1 ==>|"escalations · gym.{axis} · P1 · tier=3 ★"| C1["{axis}={option}<br/>{score} (Δ {x}) · #E{n}"]
    L1 -->|"escalations · gym.{axis} · P2 ⏸"| C2["{axis}={option}<br/>{score} · #E{n}"]
    L1 -.->|"∅"| C3["{axis}={option}<br/>{why it cannot exist}"]
    C1 ==>|"escalations · arch+hp.{axis} · P1 · tier=3 ▶"| C4["{axis}={option}<br/>{score} · #E{n}"]
​```

### Layers and node readings   (one table — kind, attributes, tier, reading, entry)
| node | kind · attributes · tier | reading | entry |
|---|---|---|---|
| `{axis}={option}` | escalations · tier=3 | {what it established, with scope} | [#E{n}](#E{n}) |

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

## CONFIG-REGISTRY   (living — ids append-only; §13)
### scenario
| id | key in SCENARIOS | note |
|---|---|---|
| <a id="sc0"></a>`sc0` | `{registry key}` | {what this base is} |

### gym · `g`   |   arch · `a`   |   hp · `h`   (one table each, same shape)
| id | parent | delta | why it exists / what promoted it | cell tuned in |
|---|---|---|---|---|
| <a id="g0"></a>`g0` | — (L1 origin) | {the §8.6 derivation's gym output} | the derivation; reserved | — |
| <a id="g1"></a>`g1` | `g0` | `{knob}={value}` | {#E{n} — what promoted it} | — |

### Constraints   (R1c — declared on the constraining id, refused at launch)
| id | requires |
|---|---|
| `a2` | `g.norm_reward = false` — {why the force exists} |

### Current bases
| axis | current | since |
|---|---|---|
| `sc` / `g` / `a` / `h` | `{id}` | {#E{n}} |

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
## 12. Local extensions

§2's five sections are the spine. A campaign that needs more structure adds a
**local extension** and marks it as one — `examples/mab` carries two, a
`### Deviation register` and a `## RUNS` table, each labelled *(local extension
— not a guide §11 section)*. That label is the whole convention: it tells the
next reader which parts of the log are format and which are one campaign's
apparatus, and it keeps an extension from being copied as though it were
mandated.

`§CONFIG-REGISTRY` entered the guide here (v0.9.21) as an extension under
evaluation, written down rather than left in its inventing campaign because its
ids land in archives that must be reconcilable. It graduated to a rule on
2026-08-24 and is now §13.

## 13. §CONFIG-REGISTRY — the bases runs cite

**Status: rule** (2026-08-24, upstream issue #60). Promoted under §10 rule 3 on
two campaigns: `game2048`, which invented it (2026-08-21), and `adi_flex`, a
structurally unlike one — tuning-heavy inventory control, 583 trials across two
studies, five observation encodings, an exact DP reference — which adopted the
shape verbatim and ran the axis test to the end of its archive without bending
it. The graduation bar asked for exactly that second campaign, plus the data
half (`{domain}_configs.py`) built and run; both are met, with one residue
stated at the end.

**The problem.** The log answers *what question does this run answer* — §7 rule
1's address — and until this section, nothing answered *what was this run
configured from*. Two arms can share an address exactly and differ on four
knobs. The map holds the **crowned** bundle (§3.4), so an arm's base is not
there at launch time, when it is by definition not crowned; and the run name is
a diff against the derivation (spec §8.4), so a knob sitting at the derived
value leaves no trace in the join key. An inherited base is therefore invisible
in both directions. `game2048` reports nine incidents of one shape — a knob
nobody chose — including ~15 arms sharing a `gae_lambda` derived at a different
board scale, and a declared base that had never actually been run.

**The shape.** A **living** section (edited in place, like §MAP; unlike the
three append-only ones), holding one table per axis, with per-id anchors so a
ledger entry links straight to a definition (`[a8](#a8)`), matching §7 rule 2's
anchor discipline.

### 13.1 Three independently versioned axes, plus the scenario

They are the spec's own L2 sub-layers (`L2(gym)` / `L2(arch)` / `L2(hp)`), not
a new taxonomy, and they move on different clocks — a tuning sweep moves `h`
and leaves `a` alone, so one bundled id would churn on every run and hide which
axis moved. *Independently versioned*, deliberately, rather than *orthogonal*:
§13.3 records real cases where one axis forces a value on another.

**The axis a knob belongs to is decided by the layer that implements it**, not
by what it feels like. Spec §1.1's layering is already the arbiter.

| axis | scope | the test — *which layer implements it* |
|---|---|---|
| `sc` | the study base | an alias for one key in the domain's `SCENARIOS` registry (spec §5.4); it **references**, never defines |
| `g` | the env as presented to the algorithm: `observation_mode`, `action_mode`, `reward_mode`, and the §8.3 vec-env wrapper stack | something between the MDP and the algorithm implements it — `{domain}_gym.py` or a vec-env wrapper |
| `a` | policy family, feature extractor, critic form, value routing. **Not** loss weights | the policy/extractor code implements it — a custom class, not a constructor argument |
| `h` | optimizer, schedules, rollout geometry, loss weights, epochs | it is an argument to the algorithm constructor, consumed by the learner |

Six boundary calls, each of which has been got wrong somewhere:

- **The vec-env wrapper stack is `g`** — `norm_obs`, `norm_reward`, frame-stack
  depth in particular. They read as hyperparameters because they are booleans
  configured next to the learner, and they are the easiest knobs here to
  misfile. They change *what the agent sees and what it is paid*, spec §8.3
  makes the saved normalizer part of the artifact contract (eval reloads it),
  and §8.6 derives them from the IR's observation structure and objective —
  from the model, never from optimization behaviour.
- **A transform of the environment's reward is `g`; a transform of the critic's
  target or output space is `a`.** The first changes what is optimized, the
  second how a value is represented. A symlog value head is `a` on both
  readings — a policy-class mixin, and the objective is untouched.
- **`gamma` stays `h`** even though §8.3 hands it to the normalizer. The wrapper
  *consumes* the discount; it does not define it.
- **Loss weights are `h`**, even when they weight a head only an `a` introduces.
  The head is `a`; the coefficient multiplying its loss is `h`.
- **`net_arch` width is `h`; a custom extractor is `a`** — inherited from spec
  §8.6's `net_arch` row, not a new line.
- **Rollout geometry is `h`**, including `n_envs`. Being *locked* as a
  structural forced move does not make a knob its own axis.

**One knob, one home — and it is checkable.** Because the partition is by
implementing layer, no argument may appear in two axes' deltas; the module
asserts exactly that at import, and the rule stops being a documented
convention. **It fired on its first execution and was right.** The naive origin
— `h0` = the train script's `_L1_DERIVED` wholesale — double-homed `norm_obs`
and `norm_reward`, because **the §8.6 derivation is not single-axis**: it emits
the gym origin and the hp origin together, exactly as §13.4's ladder says
("defines the origin of each axis", plural). Partition the derivation by the
script's own gym-knob set. A rule violated by its first enforcer inside thirty
seconds is a rule that needed the enforcer.

### 13.2 Ids

`{axis}{integer}`, dense and append-only from the L1 origin at 0 — `g0`, `g1`,
… — with a generation letter for a re-derivation (`g0b`). The integer is an
**identity, not a description**: nothing about the config may be encoded in it,
or the id becomes a rename waiting to happen. Each row records its parent and
its delta from that parent, so a full expansion is recoverable by walking to
the origin.

- **Mint the origin rows first, and read them rather than write them.** The
  §8.6 derivation's output is `g0`/`a0`/`h0`, reserved, and nothing else may
  occupy it. The origin is *read* from the train script — derived value where
  §8.6 speaks, parser default where it does not — so it cannot drift from the
  derivation it names. `_L1_DERIVED` alone is not enough: it holds only what
  §8.6 derived, which is right for a run name and wrong for an origin, and a
  knob no row derives (`vf_coef`) would be silently absent, letting two runs
  differing only there cite one tuple. The general rule: **a config must be
  complete; a run name must be a diff.**
- **Budget and seed are not config.** `total_timesteps` is recoverable from the
  run log and gets extended constantly; a seed is a replicate, not a design
  choice. (A comparison must still *state* its budget — a reporting rule.)
- **Ids are append-only.** Never edit `a4`; add `a5 = a4 + {delta}`, recording
  parent, delta, and what promoted it. Only the CURRENT-BASE line changes in
  place, so a historical citation of `a8` still means what it meant.
- **An id is a promotion, not a record that something ran.** Rows are minted
  for **adopted** bases — crowned, shipped, or parented — never for probes or
  failed arms, however fully replicated. §LEDGER is what was run;
  §CONFIG-REGISTRY is what was kept. The deviation count then becomes
  diagnostic: one adopting archive partitions into 21 runs at `{sc}/L0`, 15
  probe runs in unregistered cells (ledger-addressed only), and 90 tuple-cited
  runs of which 60 sit at zero deviations — and the only runs beyond two
  deviations are the one failed arm.
- **A delta may only contain knobs someone chose.** A shipped row's delta is
  exactly what the study searched. A value the tuning driver *rendered* rather
  than a person selecting it is not a choice and does not belong in a delta —
  the failure spec §8.6 now prevents at source (upstream #64, v0.9.25), where a
  trial's rendering of an out-of-tier knob was mistaken for that knob's value
  and briefly minted as two configurations.

### 13.3 Constraints across axes

Where one axis forces a value on another, **the constraint is declared on the
constraining id, and a citation that violates it is refused**. The axes version
independently; they are not causally independent, and pretending otherwise is
how a run resolves to something no cited id describes. Selecting a symlog value
head forces `norm_reward` off, so an `a` id silently sets a `g` knob. Three
ways to record that, and only the third survives "a knob not in a cited config
is not in the run":

1. move the knob to the forcing axis — misfiles it, and the next `a` id that
   does not force it inherits the wrong home;
2. leave it implicit — a silently doubled loss coefficient in different clothes;
3. **declare it**: the `a` row carries `requires: g.norm_reward = false`, and an
   arm citing that `a` with a `g` whose `norm_reward` is true is refused at
   launch rather than resolved silently.

**The constraint binds the tuning sampler, not only the launch.** A forced knob
that enters an `mdp_tuning` tier is drawn freely, overridden silently, and
recorded at the drawn value — so the study's parameters disagree with the runs'
resolved args and the trials on the forced side are duplicates. One campaign
carried `γ = β` this way by passing `--fix gamma` by hand on every launch;
declaring it on the `sc` id is what makes that mechanical. Wherever the axes
meet a sampler, this is a constraint on the space, not a note in a table.

This is §7 rule 4 ("bundles stay bundles") and §3.4's pairing constraints one
level down: the map governs adoption units, the registry governs config
components.

### 13.4 How this meshes with the L0/L1/L2 ladder

The registry is a taxonomy **of the L2 layer**, and the ladder attaches to it at
exactly one point — the origin of each axis.

| level | what it is | how a run cites it |
|---|---|---|
| **L0** | faithful defaults: the library's defaults plus only what the *problem* forces. Not configured from the registry at all | `{sc}/L0` — the scenario and nothing else. No axis ids, because there is no derived config to name |
| **L1** | the derivation's output. It **defines** the origin of each axis: `g0`, `a0`, `h0` | `{sc}/{design axes}/L1`, the canonical spelling of the all-origin tuple **within that cell** |
| **L2+** | every escalation: a non-origin id on a **non-design** knob, or a declared deviation | `sc0/g2/a0/h0`, `sc0/g0/a0/h0_lam0.9` |

**A design axis names a cell, not a level.** The `g` axis carries the design
axes (`observation_mode`, `action_mode`, `reward_mode`) as well as the wrapper
stack, and §8.6's ladder is per-cell — *"Δ(L1−L0) measures the configuration
layer per case"*. So the level is the set of layers moved off the complete
origin **among non-design knobs**; a non-origin id that differs only in a
design axis is a different *cell*, whose own ladder starts again at L1.

This was measured. The first faithful implementation read the clause as *any*
non-origin id, and produced **33 disagreements in 105 runs** against the
knob-derived level — every `vec_mip` arm reading `L2(gym)`, relabelling half of
a declared research instrument as an escalation, all in the direction that
inflates levels. Worked example from that tree: `sc0/g1/a0/h0` — the `vec_mip`
arm, wrapper and hp at the derivation — is **L1 in its own cell**, spellable
`sc0/vec_mip/L1`; `sc0/g2/a0/h0`, with `norm_obs` moved, is `L2(gym)`; and an
explicit `-o vec` against a citation that fixes `vec_mip` is refused at launch
by §3.1's write-once rule rather than recorded as a deviation.

The quieter half of the same correction: a knob **no** §8.6 row derives
(`vf_coef`) still makes a level when it moves, per §8.6's own invariant — *more
than one training configuration was tried*. Derivation coverage decides what
the origin must contain; it does not decide what counts as an escalation.

Three properties follow:

1. **L1 keeps its status as a derivation rather than a choice.** `g0`/`a0`/`h0`
   are not entries anyone picked; they are what the derivation emitted. The
   registry's append-only ids begin *after* it, which is why every row above an
   origin is L2 permanently, and why the obvious objection — giving L1 a config
   id makes it look chosen — does not bite.
2. **The level stops being authored and becomes derivable.** Origin ids on
   every axis with no deviations is L1; anything else is L2, and which axes
   moved says which sub-layer. An authored tag is a claim nothing checks: one
   campaign's `L1_onehot_CNN` names an L2(gym) arm in ~90 run dirs, and another
   found 66 of 126 training runs carrying a `solve_level` their own knobs
   contradict — 24 of them "L2(hp)" rounds that had opened the gym layer.
3. **A wrong derivation has a home, and it is emphatically not a deviation.**
   A rule L1-derived at one scale and applied at another is a
   **re-derivation**: a new L1 generation, minting fresh origins on the
   affected axes and restarting the ladder above them. Its cost is the point —
   L2 results measured against the old origin are not comparable to those
   measured against the new one, and Δ(L1−L0) is re-measured. Generations are
   lettered (`L1b` with `g0b`/`a0b`/`h0b`), never primed: these tokens are
   grepped, so they stay shell-safe.

### 13.5 Two artifacts, and what stops them drifting

A base change *is* an escalation, so it must leave a mark in the log — §1's own
argument, one level down. But a launch check has to **resolve** an id, and
parsing markdown tables to do it would be the fragile thing this section exists
to avoid. So a new base is registered in both places, with the overlap
**checked rather than trusted**:

| artifact | holds | authority |
|---|---|---|
| `{domain}_configs.py` | the id definitions as data — parent, delta, cross-axis constraints | **authoritative for what a config is**; it is what a run resolves against |
| §CONFIG-REGISTRY | the same ids as a reader-facing table, plus the reasons: why an id exists, what promoted it, what it supersedes | **authoritative for why**; the rendering a human reads beside the verdicts that cite it |

**The module registers axis components, never whole configs.** There is no
"config" object: a run cites a *tuple* (`sc3 g1 a3 h1`) plus its deviations, and
the cross product is implicit. Registering bundles rebuilds the failure the axes
exist to prevent — two arms were called different *architectures* for two days
when they were one `a` differing in `h` — because the axis that moved becomes
recoverable only by diffing two bundle definitions. It is also what makes the
cross-axis query the operation the registry exists for: *every arm that used
`a3`* is a grep, not an expansion.

This is deliberate redundancy made safe by a check, a shape the spec already
uses: §8.4 records `algo_class` in the args log even though it is inferable in
principle, precisely so the artifact can be checked against the declaration. The
contrast with the deferred `run_status.json` is the instructive one — that was a
*mutable, multi-writer, unchecked* sidecar. This pair is single-writer per side,
changes only on a deliberate act, and disagreement fails a launch.

**What the module checks at import**, as run in the adopting campaign: one knob
one home (§13.1); ids dense per generation; every `sc` alias live in
`SCENARIOS`; every design-axis value a `g` delta sets validated against the IR;
and, in the campaign's audit, every id present in both module and table with the
same parent.

### 13.6 The run name, and where ids are recorded

**Ids go in the args log, not in run-directory names.** Run dirs are immutable
(§8), so a spelling change orphans an archive — and this is not hypothetical:
one adopting archive had already survived a token-grammar change
(`--no-vecnorm` becoming a split flag pair) in which the *config* survived and
the directory name did not. Recording `config_ids` and `config_deviations` in
the args log put 709 pre-existing directories at zero orphaned. The run name
keeps doing its own job — diffing the derivation, per spec §8.4 — and the two
readings agree because both are anchored to the same origin.

### 13.7 The residue, stated

**Registry enforcement has no upstream gate.** Upstream #62 (v0.9.23) gates the
**§8.6 derivation** at launch — `assert_l1_current`, checking that a run's
resolved args match what the script derived — which is a different check from
"the run matches the tuple it cited". A campaign adopting this section carries
its own import-time and launch-time checks, domain-locally. That is the honest
gap in the graduation: the rule is confirmed across two campaigns and the
checks exist, but they live in `{domain}_configs.py` rather than in the harness.

Second, smaller: the adopting campaign's module resolves citations for every
training launch and its rows were verified knob-for-knob against a completed
study's winners, but it has not yet been the launcher of a *new* tuning study.
The axes collide most where a study returns a config differing from its parent
on more than one of them (§8.6's `core` tier reaches the arch layer, `breadth`
and `all` reach gym), and the rule there is §13.4's: **read the level off the
knobs that actually moved**, never off the fact that a study produced it.

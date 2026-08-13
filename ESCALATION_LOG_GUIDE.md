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
upstream proposals split out to the mdp-propose skill.*

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
| `P{n}` | selection-split priority at a tree node (§3.1) | standing; revised by tree surgery |
| `S{n}` | coverage-split schedule at a tree node (§3.1) | standing; children postponable, never prunable |
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
  bottom because nothing conditions on it). The root may split on **targets**
  (3×3 → 4×4); the standard eval protocol and the reference bar are then
  **per-target-node annotations**, not campaign globals.
- **Default level order** (override per domain by conditioning analysis):
  S-splits (targets / scenario families) ≻ action interface ≻ observation ≻
  architecture / extractor ≻ training signal (reward shaping, HP) — matching
  the spec-§8.6 escalation layers `L2(gym)` / `L2(arch)` / `L2(hp)`. The
  order encodes **conditioning, not importance**: HP is last because nothing
  conditions on it, not because it moves results least — it is a *covariate*
  of every comparison above it, not a parent (topk_id: "plain HP is bigger
  than either research issue"), which is what rule 10 exists for.
- **Type every split: coverage (AND) vs selection (OR).** A **coverage
  split** — `S{n}` edges — partitions the *problem*: target scales
  (3×3 → 4×4), scenario families (Gaussian vs Bernoulli arms). Every child
  must eventually carry a reasonable solution, so S-children are
  **postponable, never prunable** (`✗`/`∅` are illegal on S-edges; an
  S-child is ✓ covered, ▶ active, or ⏸ postponed with a *scheduled return*),
  and — for partitions — each owns its **own protocol, bar and leaderboard**
  (extension-chain links instead share the target's protocol: the cross-link
  comparison measures the *price of generality*, and losing it never prunes
  a link). A **selection
  split** — `P{n}` edges — compares *solutions*: modeling and design
  alternatives (counts vs bayes, grid vs onehot). The point is to crown one
  child and prune the rest; siblings are rows on **one shared leaderboard**.
  The crown forks at S-splits (one ★ path per covered child) and passes
  through exactly one child at P-splits. An S-split can sometimes be
  **collapsed by a generalist** — one policy covering all cells — where the
  axis permits (game2048: `prob_4` could, `grid_size` cannot: the obs shape
  changes); a P-split has no analogue. This typing is the Phase-A **mode
  stance** carried onto the tree: scenario modes declared "independent
  branches, reported separately" arrive as S-splits, "competing designs to
  compare head-to-head" as P-splits. **The litmus is redundancy at
  crowning**: P-siblings become redundant the moment one is crowned; an
  S-sibling never does (game2048: crowning the masked interface did not make
  the free interface redundant — its value is less machinery, and that
  survives the win).
- **S-splits carry two annotations — shape and obligation — not a third edge
  type.** Shape: a **partition** (3×3 vs 4×4, Gaussian vs Bernoulli) has
  disjoint cells, each covered on its own; an **extension chain** (game2048:
  masked ⊂ free — the free interface drops the mask oracle; topk_id: bayes
  (0,0) ⊂ (0,1) ⊂ (1,1) — progressively less dependency on the Bayes
  machinery) has *nested contracts ordered by generality*: solving a more
  general link subsumes the restricted ones, so the chain is climbed from
  the restricted end, levers transferring up as imported priors (§3.2).
  Obligation: `must` (default — blocks case close) or `stretch`
  (nice-to-have generality: may stay open at close, recorded as an *open
  extension* — but never ✗ refuted, since crowning a restricted sibling
  cannot make a more general contract redundant).
- **Siblings carry ranks** — local and standing: P1 > P2 > … orders
  competitors at a selection split; S1 > S2 > … *schedules coverage* at a
  coverage split. Both are priorities and both revise via REPRIORITIZED
  lines, but they license different things: a low-P child may never run at
  all (a crowned winner prunes it); a low-S child runs *later*, never
  *never*. S-order is set by transfer (cover the child that teaches the most
  about the others first), cost, claim importance, and prerequisites — and
  pairs with a per-child *return condition* saying when the next S-child
  activates.
- **Marks**: `✓` validated · `✗` closed · `⏸` pruned/parked (always with a
  reason + tripwire) · `∅` structurally void (*cannot* exist — say why) · `▶`
  in flight · `★` the crowned path.
- **Bounds attach to nodes.** A branch with a validated delta is worth ten
  without (topk_id's localization control bounded node-pooling at ≤0.105 OC;
  game2048's ladder: +29% config, +46% 2-D structure, +64% value one-hot —
  each rung isolated by its own control).
- **The tree is dynamic.** REPARENTED / SPLIT / REPRIORITIZED / SHATTERED are
  expected moves, each costing one changelog line (§4). A campaign that
  passes its bar re-roots the decomposition on *headroom above current best*;
  the bar itself moves to the off-tree register as a calibration item.

**3.2 Expansion disciplines** — what keeps the tree from losing the grid's one
virtue, mechanical generation of the cell nobody thought of:

- **Prune visibly.** A branch declined by judgment appears anyway: one line,
  reason, tripwire (`⏸`). Never silently omitted, and never confused with `∅`
  impossible.
- **Controls are siblings.** A crowned branch's children must include the
  controls that decompose its win (game2048: onehot's +64% stayed confounded
  with the CNN until an MLP-on-onehot sibling existed).
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
expand a new child) or an **off-tree obligation** (bar calibration, protocol
bookkeeping — budget-consuming but not solution-touching; keep an off-tree
register beside the queue). Semantics:

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
- **Split types bind the queue**: an agenda item that crowns a P-child
  licenses pruning its siblings; completing an S-child never reduces its
  siblings' obligation — postponed S-children are coverage *debt*, and a
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
   frame changes get a changelog line; ledger entries carry their address.
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

## MAP  (as of {date})

### Design tree
​```mermaid
graph TD
    ROOT["root: {campaign}"]
    ROOT ==>|"S1 ▶"| T1["target: {name}<br/>protocol: {eval}; bar: {reference}"]
    ROOT -->|"S2 postponed"| T2["target: {next} — coverage debt; return: {when}"]
    T1 ==>|"P1 ★"| C1["{choice} {mark} {bound if any}"]
    T1 -->|"P2 ⏸"| C2["{choice} — {reason}; tripwire: {…}"]
    T1 -.->|"∅"| C3["{impossible cell} — {why}"]
​```

### Frontier
1. **A1 — {action} @ {tree path}**  {▶|queued} — {evidence; cost}
- parked: {⏸ node} — tripwire: {…}

### Off-tree register
- {bar calibration | protocol bookkeeping | next campaign roots}

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
### #{id} {date} — {one-line hypothesis}
address: {tree path} / A{n}
runs: {commands or paths}
verdict: {numbers @ protocol; comparison}   status: {✓|✗|~|▶}

### #{id} {date} — DIAGNOSIS: {checkpoint one-liner}
reads:    {#E ids + imported evidence w/ provenance}
observed: {established, scope attached}
missing:  {named gaps → tree nodes}
plan:     {A items + arbiters + pre-decided branches}
```

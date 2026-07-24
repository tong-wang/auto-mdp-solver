# Escalation Log Guide — documenting experiments and escalations

*Trial guideline, 2026-07-24. Companion to `AGENT_PLAN.md` §4 (solve = backbone +
escalation layers) and §14 (gym gates). Status: to be validated on a live
experimental project, then revised from feedback and eventually promoted into the
plugin skill as the trace convention feeding the escalation playbook. Distilled from
what worked — and what stayed invisible — in `topk_id` (EQUINET.md, BAYES11.md).*

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

## 2. The log — one file, three sections

One `ESCALATION.md` per project/case, git-tracked:

```
ESCALATION.md
├── §MAP              current frame — revised in place, dated
├── §FRAME-CHANGELOG  append-only, one line per re-framing
└── §LEDGER           append-only, hypothesis → runs → verdict
```

## 3. §MAP — the current frame

Four blocks. The map is **not one diagram** — it is a small set of typed views:

**3.1 Decomposition (mermaid tree).** The gap to the current bar, cut into named
buckets. Attach a **bound** wherever a control experiment has produced one — a
bucket with a number is worth ten without (topk_id's localization control bounded
node-pooling at ≤0.105 OC of a ~2.5× total gap, redirecting weeks of effort).
Status-mark every node: `✓` validated `✗` refuted `⏸` parked `▶` in flight.

**3.2 Design grids (markdown tables).** The orthogonal axes (obs mode × head × HP ×
scenario…), one table per meaningful slice; **cells hold run IDs + headline metric +
mark** (`#38 −0.3625 ✗`). You already draw these as "results tables" — recognize
them as map views and keep them current. Tables diff cleanly and generalize by
slicing; do not force grids into the tree.

**3.3 Current best bundle.** The validated-positive package, stated **as a bundle
with its pairing constraints**. Levers validated together stay together (topk_id:
γ=1 × nopass — "neither validated alone"). Adopt/reject bundles, not components.

**3.4 Open questions, ranked; parked items with guardrails.** What the next budget
should attack and in what order; what is deliberately not being pursued and the
tripwire that would un-park it (topk_id issue (3): parked, with the "never route
MAP μ,σ into the forward path" guardrail).

## 4. §FRAME-CHANGELOG — the evolution that matters

Append **one line per re-framing** — not per routine update. Date + change +
forcing evidence (run IDs). Vocabulary: *introduced / split / merged / narrowed /
refuted / parked / un-parked*. Example lines (topk_id, reconstructed):

```
2026-07-17  two-issue → three-bucket: plain HP is bigger than either research issue (C 2×2)
2026-07-18  "k-features pay for k≥2" NARROWED to k=2-only (D: kf −0.073 vs attn1)
2026-07-18  "C record is mostly HP" REFUTED (honest 8192: arch win +0.061 real)
```

This ten-line spine is the "how did our understanding develop" artifact — and the
raw feed for the playbook's frame-move entries. Full history of the map is git's
job (§8); the changelog is the *interesting* history.

## 5. §LEDGER — the experiment record

Append-only entries; corrections are new entries citing the old (never edit a
verdict in place). Per entry:

- **id** — short, stable (`#38`, or `E12`); this is what map cells and changelog
  lines cite.
- **address** — the map bucket it attacks + the grid cell it occupies.
- **hypothesis** — what this run decides, stated *before* launch.
- **runs** — commands / artifact paths, enough to reproduce.
- **verdict** — numbers, at the standard eval protocol (see §6), with the
  comparison that gives them meaning. **Neutral and negative verdicts are
  first-class** ("roughly free, not yet proven positive" is a verdict; so is
  "8M ≈ 4M — not undertrained").
- **status** — `✓ / ✗ / ~ (neutral) / ▶`.

## 6. Binding rules

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

## 7. What stays out

- **Dispatch state** — what is running where (WATCH_STATUS-style). Ephemeral,
  different lifecycle; keep it in its own file or the dispatcher. Don't let it
  colonize the map (`▶` marks are pointers, not process management).
- **Raw outputs** — `results/` folders, TB logs. The ledger cites paths.
- **Spec/playbook content** — conventions live in the spec; the log records what
  *happened*, cites the rest.

## 8. History and rendering

- **Revise the map in place.** Git is the full history — commit map edits with the
  forcing evidence in the message; `git log -p ESCALATION.md` replays every frame
  state. Snapshot-hoarding inline is the failure mode this section exists to
  prevent; at most, park a superseded diagram of a *major* re-framing in a
  collapsed `<details>` block.
- **Ephemeral views are generated, never maintained.** A live dashboard for a big
  fan-out (map + running-status coloring) is rendered *from* this file on demand;
  it is not a second source of truth.

## 9. Case close — distillation

When the campaign ends (positive or negative), run one distillation pass:

- **ledger → symptom-indexed lever entries** (symptom / hypothesis / cheap probe /
  lever / verdict + scope / evidence) for the escalation playbook;
- **changelog → frame-move entries** (reusable ways of structuring the search:
  bound-before-build, gap-decomposition-into-buckets, cell-taxonomy placement);
- negative campaigns distill too — an envelope verdict ("feasibility-dominated
  combinatorial + classical solver exists → RL not competitive") is playbook
  content, not just a dead end.

The format above is designed to make this pass near-mechanical: addresses,
citations, and scope conditions are already in place.

## 10. Template

```markdown
# {project} — escalation log

## MAP  (as of {date})

### Decomposition
​```mermaid
graph TD
    GAP["gap to {bar}: {size}"] --> B1["{bucket} {mark}<br/>{bound if any}"]
    GAP --> B2["{bucket} {mark}"]
​```

### Grids
|              | {axis-B v1}   | {axis-B v2}   |
|--------------|---------------|---------------|
| {axis-A v1}  | #1 {num} ✓    | #4 {num} ✗    |

### Current best bundle
{levers + pairing constraints + headline number @ standard eval}

### Open / parked
1. {question — next attack}
- ⏸ {parked item} — guardrail: {tripwire}

## FRAME-CHANGELOG
{date}  {INTRODUCED|SPLIT|NARROWED|REFUTED|PARKED|UN-PARKED} {claim} ({evidence ids})

## LEDGER
### #{id} {date} — {one-line hypothesis}
address: {bucket} / {grid cell}
runs: {commands or paths}
verdict: {numbers @ protocol; comparison}   status: {✓|✗|~|▶}
```

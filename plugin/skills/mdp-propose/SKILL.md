---
name: mdp-propose
description: >
  File an upstream extension proposal against the auto-mdp-solver spec, IR
  schema or pipeline process, as one GitHub issue, when a campaign hits a
  wall the current version cannot express. Use on "propose this upstream",
  "file a proposal", "this needs a spec/schema change", or when
  UPSTREAM_PROPOSAL_*.md drafts exist in scratch/. Nothing is sent without
  the user's approval of the final body.
---

# Propose an upstream spec/schema/process extension

Target repo: **`tong-wang/auto-mdp-solver`**. A proposal is a request for
change with evidence, not a patch: upstream implements accepted proposals
itself, with the motivating case as the regression test (precedent:
`key_exprs` on period-realization stages, proposed from a bandit campaign,
landed as v0.5.9 with the case as its regression).

**File it when the wall is hit, not at case close.** A proposal is pinned
against the spec version it argues with and goes stale as upstream moves;
batching to campaign end is how a folder ends up with ten unfiled drafts.

Hard rule: **assemble → show → confirm → send** — the user sees each final
issue body and approves it before any `gh issue create` runs.

## 1. Draft locally

One `UPSTREAM_PROPOSAL_<slug>.md` per independent change, in `scratch/` and
**never tracked** — a draft is working material, the same shape as a round
plan. Once the issue exists the issue *is* the proposal, and a tracked local
copy is a second source of truth that cannot follow the issue's edits, its
disposition, or its rejection. Required sections:

- **Target** — the upstream file or spec section (e.g. `mdp_ir/schema.py
  StateVariable`, spec §8.6), and a version pin in the header:
  `Against v<X.Y.Z> / main @ <sha>.` Nothing about the draft's own status —
  the body is filed verbatim (§2), and a line saying "not yet proposed" is
  false the moment it is sent.
- **The gap** — what the current version does, quoted or cited precisely
  enough that upstream can find it without the campaign's context. **First
  check the upstream specs and understand why they are designed that way**:
  search the spec, the sample, and the plan docs for existing text on the
  same question — several sections carry titled rationales (e.g. §8.4 "Why
  `tuning/` is not under `{scenario_name}/`"), and a proposal that walks
  past one arrives pre-refuted. If such text exists, quote it and answer
  it; if the answer is "the rationale is right and my case is different",
  say exactly how.
- **Proposed change** — semantics, not a diff; include the smallest example
  that exercises it.
- **Cost of the status quo** — what the campaign had to do instead, and what
  the workaround cost (hand-edits per instance, a moved structural
  fingerprint, a second source of truth…). This is the evidence a reviewer
  weighs; without it a proposal is an opinion.
- **Impact sketch** — the three blast radii upstream checks on every breaking
  change: structural fingerprint, seed-key construction, migration for
  version-pinned downstreams. Say "none affected" explicitly when true.
- **Regression case** — where the motivating case demonstrates the gap, so an
  accepted proposal ships with its test.

The motivating campaign is **evidence, not contamination**: keep the real
domain and numbers when the case is publishable. Scrub only business-private
material (same provenance rule as mdp-contribute) — and local paths,
credentials, hostnames always. The **cross-project citation pass**
(mdp-contribute §2a) applies here too: a proposal's evidence sections quote
campaign records, which carry rule-8 provenance labels naming sibling
projects — run the same three-way disposition before the issue is filed.

## 2. Send

- `gh auth status` first — issues need no fork or push rights.
- **One issue per proposal** — proposals are accepted, rejected, or deferred
  independently; a batch issue can only be closed once.
- Title `proposal: <slug> — <one-line gap>`, label `upstream-proposal` (omit
  if the repo lacks it), body = the draft.
- On confirm, send; then **record the issue URL in `ESCALATION.md`** — on the
  ledger entry that hit the wall, where the disposition lands when it arrives —
  and delete the draft. The issue is the proposal; the log is the campaign's
  memory that it filed one. A folder that "shows what is filed" is showing a
  copy that will not move when the issue does.

## 3. What happens upstream (the proposer's contract)

The maintainer records one disposition on the issue:

- **accept** — implemented upstream, the motivating case becomes the
  regression test, version bump; the issue closes on the release tag.
- **reject** — with the reasoning kept on the issue.
- **defer** — with the tripwire that would reopen it.

Silence is not a disposition; ping the issue if a release passes without one.

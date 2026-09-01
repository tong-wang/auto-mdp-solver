# DOMAIN_CLAUDE_TEMPLATE.md — the per-domain `CLAUDE.md`

Template for the `{domain}/CLAUDE.md` the pipeline emits at Stage 1, so every
generated domain carries its own operating brief from the first commit. It is
the one file whose *placement* decides whether an agent sees it at all: the
host loads a folder's `CLAUDE.md` exactly when work touches that folder — the
repo-root file stays short and cross-domain, the domain's brief rides with
the domain (and travels with it on contribution or promotion).

Its counterpart is `DOMAIN_README_TEMPLATE.md`, and the two divide by
direction (spec §1.3): **this file points up** — at the specs that govern the
folder and the version of them it was built against; the README points across,
at everything the campaign produced. A campaign document is inventoried in
exactly one of the two.

Filling rules:

- `{braced}` slots are filled by the pipeline (Stage 1) and verified at
  Stage 6; fixed text is kept **verbatim** — it encodes rules that were each
  paid for in a live campaign, and keeping it identical across domains is
  what lets a reader skip what they already know.
- **Five sections, in this order** (spec §1.3): what this file is (the
  preamble under the title), the specs that govern the folder, hard rules,
  gate commands, file hygiene. **And nothing else** — see below.
- **Pointer-first is a rule, not a style.** The operative form is spec §1.3's:
  *a pointer file may name a destination; it may never describe, summarise or
  score what is inside it.* The older negative phrasing — "never a second copy
  of the README" — let three things through on live campaigns, because none of
  them looks like a copy: a question→document routing table, a file inventory,
  and a full command reference. A `CLAUDE.md` that restates the README goes
  stale and then actively misleads.
- **Hard rules are phrased trigger → destination, never trigger → answer.**
  This file's value is that it is *pushed* — loaded whenever work touches the
  folder, while `README.md`, `ESCALATION.md` and `PLAYBOOK.md` must be pulled.
  So it carries the moment a rule fires and the address of the answer, and
  stops there.
- **There is no traps section, by design** (spec §1.3). A per-domain list of
  "what bites here" grows into a restatement of the campaign log, and every
  entry one has held already has an owner: a rule a gate enforces is owned by
  the gate, whose failure message is the reminder; an ungated spec rule is
  identical in every domain and belongs upstream in one copy; a cross-domain
  convention belongs in the repo-root `CLAUDE.md`; the record-eval stance
  qualifies every leaderboard number and belongs above them in `README.md`;
  and a trap the campaign paid for is a `#E` ledger entry, digested at close
  into `PLAYBOOK.md`. What survives here is the trigger, as a hard rule.

---

````markdown
# CLAUDE.md — `{domain}/` ({one-line domain name})

Domain-local operating brief for an agent about to change this folder.
**This file points up: at the specs that govern the folder, and the version
of them its numbers were produced under.** Everything the campaign produced —
its documents, its code, its numbers — is inventoried in `README.md` and is
linked from here, never copied.

{2–4 lines: the problem — state/decision/objective in words — and which
instances/branches are SEPARATE LEADERBOARDS whose scores must never be
compared.}

## The specs that govern this folder

`{domain}/` was produced by the **auto-mdp-solver** skill from
`{domain}_schema.json`. It is not a hand-written project, and it must not
drift into one.

**Before changing anything here, read the skill docs — they are authoritative
over this file, over the code, and over local habit:**

| doc | governs |
|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG/seed tree, script + eval conventions (§1–§14) |
| `SKILL.md` | the pipeline stages, their gates, and what each stage must emit |
| `ESCALATION_LOG_GUIDE.md` | the campaign log (§MAP, §FRAME-CHANGELOG, §IR-CHANGELOG, §CONFIG-REGISTRY, §LEDGER) and the §10 playbook digest |

Solver provenance — **two versions, two meanings** (keep both; they answer
different questions, and a case that is later contributed upstream needs the
distinction — see `cases/README.md`, "After a case is merged"):

- **Built and gated at: {tag + commit}** — the checkout every number in this
  folder was produced under. This line never moves. Moving it would claim the
  results were re-measured.
- **Conformance maintained through: {tag}** — how far declarations, drawing
  conventions and gate compatibility have been carried forward. Updated when
  the case is brought to a newer spec *without* re-running anything; log each
  such move in `ESCALATION.md` §FRAME-CHANGELOG.

While the campaign is live these two are the same tag. They diverge only once
someone conforms the case to a spec that landed after its results did.

The repo-root `CLAUDE.md` carries the cross-domain rules. `README.md` carries
the case: what the problem is, what is in the folder, what the arms scored,
and the commands that reproduce them.

## Hard rules

Each fires at a moment and names where the answer lives. None of them states
the answer — that is what makes them safe to keep current.

- **Before inventing a mechanism, check the spec.** The pipeline usually has
  one already, and its version composes with the gates.
- **Structural changes go through the IR first**, then the code, then the
  gates — never code-first. `{domain}_schema.json` is the source of truth;
  `{domain}_scenarios.py` and friends materialize it.
  Observation/architecture levers are NOT IR changes — they live in the gym /
  train script with their own executable gate; only the *problem* goes
  through the schema.
- **Re-run the gates after any IR change and after any solver bump** — the
  IR depends on host behavior, not just on this folder. A moved fingerprint
  owes a §IR-CHANGELOG entry (guide §5, tripwire 2).
- **Before quoting a number, say which artifact produced it.** Selection and
  terminal networks are different objects (`{scenario}_ppo.zip` vs
  `_ppo_final.zip`, spec §8.4), and the protocol every number is quoted at —
  including which eval mode is the record here — is stated once in
  `README.md`'s results section.
- **Editing `_L1_DERIVED` is a re-derivation, not a default edit** (spec
  §8.4). It re-bases every Δ(L2−L1) above it, so it takes a logged basis and
  a ledger entry — see spec §8.6 for when re-deriving is the right move.
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in `ESCALATION.md`, and, if the spec should change, as an
  upstream issue filed via the mdp-propose skill. The issue is the record and
  the log carries its URL; the draft behind it is scratch (see File hygiene).

## Gate commands

Run from `{domain}/` unless noted. These are the checks; the commands that
*run* the code — train, eval, benchmark, probe, plot — are in `README.md`'s
technical appendix. Always pin threads for training — torch oversubscribes.

```bash
# conformance / laws / differential run from the PARENT of {domain}/
python -m mdp_ir {domain}_schema.json
python -m mdp_conformance {domain}
python -m mdp_ir.laws {domain}
python -m mdp_ir.differential {domain}/{domain}_schema.json --episodes 40 --all-instances
pytest {domain}_test.py
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `{domain}/` only if it is (a) spec-§1 layout, (b) an
implementation a finding cites and someone must re-run to reproduce it
(probes, extra gates), or (c) a campaign document (`README`, `ESCALATION`,
`INTERPRET`, `PLAYBOOK`). **Everything else goes to `scratch/` (gitignored):
launchers, monitors, one-off checks, throwaway analysis. All output goes to
`results/` (gitignored).**

**Round plans and upstream-proposal drafts are never tracked.** A `*_PLAN.md`
lives in `scratch/` while it is being drafted AND while it is being executed;
findings go into `ESCALATION.md` and `README.md` **as they land**, and the
plan is deleted
once written up. If a plan is the only place a result exists, that is a bug
in `ESCALATION.md`. **The probe a plan drives is the opposite — it stays,
permanently**: an escalation entry cites numbers that only exist if the code
behind them can be re-run. A probe is written in `{domain}/` from the start
(it imports its siblings) and **committed in the same commit as the
escalation entry that cites it** — `git log ESCALATION.md` then shows each
finding beside the code that produced it. Design rationale that must outlive
the round has two tracked homes, neither of them the plan: the escalation
entry and the probe's module docstring.

An `UPSTREAM_PROPOSAL_*.md` is the same shape and for the same reason: it is
drafted in `scratch/`, and once the mdp-propose skill files the issue, the
issue is the proposal — the draft goes, and `ESCALATION.md` carries the issue
URL and, later, the maintainer's disposition. A proposal is pinned to a spec
version and goes stale; a tracked copy goes stale *silently*, which is how a
folder ends up asserting a claim upstream has already rejected.

Check with `git status --short {domain}/`: untracked files should be rare
and deliberate.
````

# DOMAIN_CLAUDE_TEMPLATE.md — the per-domain `CLAUDE.md`

Template for the `{domain}/CLAUDE.md` the pipeline emits at Stage 1, so every
generated domain carries its own operating brief from the first commit. It is
the one file whose *placement* decides whether an agent sees it at all: the
host loads a folder's `CLAUDE.md` exactly when work touches that folder — the
repo-root file stays short and cross-domain, the domain's brief rides with
the domain (and travels with it on contribution or promotion).

Filling rules:

- `{braced}` slots are filled by the pipeline (Stage 1) and verified at
  Stage 6; fixed text is kept **verbatim** — it encodes rules that were each
  paid for in a live campaign, and keeping it identical across domains is
  what lets a reader skip what they already know.
- **Pointer-first is a rule, not a style**: the file says where to look and
  what will bite, never what the answer is. Anything that changes as the
  campaign progresses (leaderboards, findings, open items) lives in
  `README.md`/`ESCALATION.md` and is linked, not copied. A `CLAUDE.md` that
  restates the README will go stale and then actively mislead.
- The Traps section is *seeded*, then grows: every trap the campaign pays
  for is added the same day, citing the finding (#E…) that paid for it.

---

````markdown
# CLAUDE.md — `{domain}/` ({one-line domain name})

Domain-local operating brief. **Pointer-first: this file says where to look
and what will bite you, never what the answer is.** Anything that changes as
the campaign progresses lives in the docs below and is linked, not copied.

{2–4 lines: the problem — state/decision/objective in words — and which
instances/branches are SEPARATE LEADERBOARDS whose scores must never be
compared.}

## This domain is GENERATED — the skill is authoritative

`{domain}/` was produced by the **auto-mdp-solver** skill from
`{domain}_schema.json`. It is not a hand-written project, and it must not
drift into one.

**Before changing anything here, read the skill docs — they are authoritative
over this file, over the code, and over local habit:**

| doc | governs |
|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG/seed tree, script + eval conventions (§1–§14) |
| `SKILL.md` | the pipeline stages, their gates, and what each stage must emit |
| `ESCALATION_LOG_GUIDE.md` | the campaign log (§MAP, §FRAME-CHANGELOG, §IR-CHANGELOG, §LEDGER) and the §10 playbook digest |

Consequences, each already paid for somewhere:

- **Check the spec before inventing a mechanism.** The pipeline usually has
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
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in the log, and as an upstream proposal
  (`UPSTREAM_PROPOSAL_*.md`, filed via the mdp-propose skill) if the spec
  should change.

The repo-root `CLAUDE.md` carries the cross-domain rules.

## Where the answers are

| doc | what it is |
|---|---|
| `README.md` | layout, usage commands, leaderboards |
| `ESCALATION.md` | the campaign: §MAP, changelogs, numbered findings (#E…) with verdicts |
| `INTERPRET.md` | policy readback (spec §14) — what the trained nets actually do |
| `PLAYBOOK.md` | the guide-§10 case-close digest (exists once the campaign closes) |
| `{domain}_schema.json` | **the IR — authoritative** for the problem definition |
| `{domain}.restatement.md` | the frozen Phase-A restatement |
| `UPSTREAM_PROPOSAL_*.md` | drafts against the solver spec, not local decisions |

{A doc the spec requires but the campaign hasn't written is marked "owed",
never silently dropped from this table.}

Round plans are deliberately **not** in this table: they live in `scratch/`
and are deleted once their findings are in `ESCALATION.md` (see File
hygiene).

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

## Commands

Run from `{domain}/` unless noted. Always pin threads for training — torch
oversubscribes.

```bash
# gates (conformance/laws/differential run from the PARENT of {domain}/)
python -m mdp_ir {domain}_schema.json
python -m mdp_conformance {domain}
python -m mdp_ir.laws {domain}
python -m mdp_ir.differential {domain}/{domain}_schema.json --episodes 40 --all-instances
pytest {domain}_test.py

# train / eval  (the record eval is {stochastic | deterministic} — see Traps)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python {domain}_ppo_train.py -s {scenario} {…}
python {domain}_ppo_eval.py -s {scenario} --model-path {…}
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `{domain}/` only if it is (a) spec-§1 layout, (b) an
implementation a finding cites and someone must re-run to reproduce it
(probes, extra gates), or (c) a campaign document (`README`, `ESCALATION`,
`INTERPRET`, `PLAYBOOK`, `UPSTREAM_PROPOSAL_*`). **Everything else goes to
`scratch/` (gitignored): launchers, monitors, one-off checks, throwaway
analysis. All output goes to `results/` (gitignored).**

**Round plans are never tracked.** A `*_PLAN.md` lives in `scratch/` while
it is being drafted AND while it is being executed; findings go into
`ESCALATION.md` and `README.md` **as they land**, and the plan is deleted
once written up. If a plan is the only place a result exists, that is a bug
in `ESCALATION.md`. **The probe a plan drives is the opposite — it stays,
permanently**: an escalation entry cites numbers that only exist if the code
behind them can be re-run. A probe is written in `{domain}/` from the start
(it imports its siblings) and **committed in the same commit as the
escalation entry that cites it** — `git log ESCALATION.md` then shows each
finding beside the code that produced it. Design rationale that must outlive
the round has two tracked homes, neither of them the plan: the escalation
entry and the probe's module docstring.

Check with `git status --short {domain}/`: untracked files should be rare
and deliberate.

## Traps

Seeded by the pipeline; **every trap the campaign pays for is added here the
same day**, citing the finding (#E…) that paid for it.

- **The record eval is {`--stochastic` | deterministic argmax} here** —
  {one line on why: e.g. policy entropy IS the exploration mechanism /
  masked argmax is the deployment mode}. The other mode is a separate,
  labelled figure and never a record.
- **Selection and terminal artifacts are different networks**
  (`{scenario}_ppo.zip` vs `_ppo_final.zip`, spec §8.4) — say which one a
  number came from.
- **Compute sites/venues are cited by alias, never hostname** — venue
  config lives at the repo root; a venue change is a confound to record,
  not a detail.
- **Never use `param`, `params`, or `param_*`** anywhere (conformance
  fails).
````

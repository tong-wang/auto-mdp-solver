---
name: mdp-contribute
description: >
  Contribute a finished MDP case, its playbook riding in the folder,
  upstream to github.com/tong-wang/auto-mdp-solver — optionally re-skinned
  to hide business context; a case too sensitive even re-skinned is not
  contributed. Use on "contribute this case", "share this upstream", "submit
  to auto-mdp-solver". Spec/schema extensions go through mdp-propose.
  Nothing is sent without the user's approval of the final content.
---

# Contribute a case, playbook included

Target repo: **`tong-wang/auto-mdp-solver`**. The contribution is the case
**with its distilled playbook in the folder**. (A separate flow — proposing an
extension to the spec/schema itself — is the **mdp-propose** skill, filed
mid-campaign the moment the wall is hit, not here at case close.)

| tier | what travels | lands as |
|---|---|---|
| 1 — full case | domain + IR + docs + campaign record incl. `PLAYBOOK.md` (no `results/`) | PR adding `cases/<name>/` |
| 1b — re-skinned case | same, after an isomorphic rename | PR adding `cases/<name>/` |

**Sensitive means not contributed.** A case ships whole and real — real
domain name (a textbook name is the best retrieval key there is), real
numbers, `#E` ledger citations that resolve because `ESCALATION.md` ships
beside them. When the *business context* is the only sensitive part, tier 1b
hides it by isomorphic rename, verified by the gates. There is no partial
path below that — no sanitized excerpts, no playbook-only contribution: what
a rename cannot hide stays home. What is always scrubbed, both tiers:
absolute local paths, credentials/keys, internal hostnames, names/emails the
user didn't choose to publish.

Hard rule: **assemble → show → confirm → send.** The user sees the exact file
list / PR body before anything leaves the machine, and each `gh` command that
publishes (push, `pr create`) runs only after an explicit yes. If the user
declines at any point, stop cleanly and leave the assembled material on disk
for them.

## 0. Scope check

- Identify the case: a domain folder with `{name}_schema.json`, restatement,
  domain modules, benchmarks, README. If there is no finished case, say what is
  missing and stop (point at the mdp-solver skill).
- Resolve the interpreter exactly as the mdp-solver skill does
  (`$MDP_SOLVER_PYTHON` → workspace `.venv/bin/python` → create). Gates below run
  through it.
- Check `gh auth status`. Not authenticated → tell the user to run
  `gh auth login` and pause.
- Ask which tier (1 / 1b) unless the request already says. If the case is
  sensitive beyond what a re-skin hides, say so plainly and stop — nothing is
  contributed.

## 1. Always first: distill locally

If the case folder has an escalation log (`ESCALATION.md` per
`ESCALATION_LOG_GUIDE.md` — it ships in the sibling **`mdp-solver` skill
directory**, not this one, and is the only copy to read; do not search the
filesystem for it, since a development checkout of the solver may also be on
disk) and no `PLAYBOOK.md` yet, run the guide's §10 case-close
distillation now, into the folder: digest entries — only the necessary
context / symptom / diagnosis / prescription / failed attempts, real names
and numbers, ledger citations for everything else. This happens whether or
not the case ends up contributed — it is the user's own playbook; tier 1
sends the file as-is.

## 2. Tier 1 — full case

**2a. Assemble.** Copy the case into a staging dir as `cases/<name>/` with:

- `{name}_schema.json` (frozen IR) + `{name}.restatement.md`
- domain modules per the spec chain: `_uncertainty`, `_scenarios`, `_mdp`,
  `_gym`, `{name}_ir_adapter.py` (portable-domain contract)
- benchmarks + their `_eval` scripts, `_ppo_train` / `_ppo_eval`
  (+ `_ppo_tune` if used), `{name}_policy.py`
- `README.md` — the lead, the TL;DR block and the four sections spec §1.3 fixes:
  the problem, the layout tables, results by research-question tier (**every
  leaderboard** under `1-comparative`, the declared stances under
  `2-structural`), and the technical appendix, whose commands reproduce every
  number above them
- `CLAUDE.md` — the domain's operating brief, five sections per spec §1.3
  (emitted at Stage 1 from the skill's template); on staging, re-point its two
  provenance lines at the upstream repo and drop machine-local paths
- the campaign record: `ESCALATION.md`, `PLAYBOOK.md` (§1), `INTERPRET.md`
  and committed figures if the campaign kept them — the record is half the
  value, and for a case later promoted to the examples set it ships with the
  plugin

Exclude: `results/`, `__pycache__`, logs, anything gitignored. Then run the
always-scrub (paths, credentials, hostnames, identities) over the staged
files and show anything found.

**Cross-project citations — a separate pass, run here, before anything is
committed.** The always-scrub catches *accidental* leakage; citations of
other projects are deliberate, guide-mandated content (rule 8 requires an
imported lever to carry its provenance label), so they need a disposition,
not a grep-and-delete. Enumerate every other project/domain the staged files
name — search prose as well as identifiers, and search the campaign-record
files (`ESCALATION.md`, `PLAYBOOK.md`, `INTERPRET.md`) and code comments
first, since rule-8 labels concentrate there. Classify each:

1. **Already published** — in `cases/`, the skill's `examples/`, or
   otherwise public: cite freely by name. Provenance to a case a reader can
   open beats uniformity; anonymizing it destroys a working reference.
2. **Private** — list every citation with file and line, and ask the user
   per project: publish the name (a valid answer is "it's about to be
   contributed"), or anonymize.
3. **Anonymize** — ask *how*; never invent a descriptive alias. A shape
   description identifies a project as precisely as its name; whether it
   does is the user's call, not the assistant's.

Timing is the point: the disposition happens on the staged tree, **before
the contribution commit exists**. A redaction commit on top of an earlier
commit withholds nothing — the name stays recoverable from the branch via
`git log -S`, and the PR's own refs keep pre-squash commits reachable even
after a squash-merge. If a leak is discovered after commits exist, the
remedy is to rewrite the branch or open a fresh single-commit PR — never a
follow-up commit.

**2b. Provenance (must ask, verbatim topics).** (i) Does the user have the
right to publish this problem *and its campaign record*? A case accepted into
`cases/` may later be promoted whole into the plugin's examples set, so the
decision covers `ESCALATION.md`/`PLAYBOOK.md` too. Paper-derived cases:
citation present in the README. Business-derived: the user confirms ownership
or approval — if unsure, offer tier 1b; if a re-skin still doesn't clear it,
stop without contributing. (ii) Is the leaderboard honest — produced by the
README's own commands at the stated eval protocol? Record both answers in the
PR body.

**2c. Gates — must be green before any PR** (CI re-runs these on the PR):

```bash
$PY -m mdp_ir cases/<name>/<name>_schema.json
$PY -m mdp_conformance cases/<name>
$PY -m mdp_ir.differential cases/<name>/<name>_schema.json --all-instances --episodes 40
```

Red gate → fix or stop; never open a PR with red gates.

**2d. Send.** Determine push rights:
`gh repo view tong-wang/auto-mdp-solver --json viewerPermission`.

- No push rights (normal contributor): `gh repo fork tong-wang/auto-mdp-solver
  --clone` (temp dir), branch `case/<name>`, copy the staged folder in, commit,
  push to the fork, then `gh pr create --repo tong-wang/auto-mdp-solver`.
- Maintainer: branch in the working clone instead.

PR body: what the case stresses that the pipeline hasn't handled before (the
`cases/` admission question), leaderboard summary (is any pipeline deliverable
— trained policy or its §14 readback — competitive? negative verdicts are fine
and said plainly), provenance statements from 2b, gates-green statement. Show
the user the final file list and PR body; send on confirm; report the PR URL.

## 3. Tier 1b — re-skinned case

As tier 1, with an **isomorphic rename** before 2c: new domain name and story;
rename identifiers, docstrings, README narrative; abstract units/constants'
*labels* — while structure, distributions, dynamics, and numbers' roles stay
intact. The IR makes this mechanical: rename via the schema, regenerate names in
code, keep everything else byte-equivalent in behavior. **Proof of integrity =
the gates**: re-run 2c on the re-skinned folder; green means the rename broke
nothing. The rename extends to the campaign record — entry text and the
labels on numbers, never the numbers' roles; if the record cannot be renamed
faithfully, leave the record out and say so in the PR. Ask the user to review
the re-skinned README specifically for residual business context before 2d.

## Failure notes

- `gh repo fork` on an existing fork just reuses it — fine.
- If `pr create` fails for a first-time fork (detached upstream), retry with
  `--head <user>:case/<name>`.
- Never `git push` to `tong-wang/auto-mdp-solver` directly without maintainer
  rights confirmed; never force-push anything.

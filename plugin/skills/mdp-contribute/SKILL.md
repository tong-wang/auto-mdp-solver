---
name: mdp-contribute
description: >
  Contribute a finished MDP case or distilled escalation-playbook entries
  upstream to the public auto-mdp-solver repo (github.com/tong-wang/auto-mdp-solver).
  Use when the user wants to share/submit/contribute a case they built with the
  mdp-solver skill ("contribute this case", "share this upstream", "submit to
  auto-mdp-solver"), optionally re-skinned to hide the business context, or to
  contribute lessons ("contribute the playbook entries") from an escalation
  campaign. Everything is assembled and shown locally first; nothing is sent
  without the user's explicit approval of the final content.
---

# Contribute a case or playbook entries upstream

Target repo: **`tong-wang/auto-mdp-solver`**. Two kinds of contribution, mapping
onto the repo's two knowledge assets:

| tier | what travels | lands as |
|---|---|---|
| 1 — full case | domain + IR + docs (no `results/`) | PR adding `cases/<name>/` |
| 1b — re-skinned case | same, after an isomorphic rename | PR adding `cases/<name>/` |
| 2 — playbook entries | sanitized symptom→lever lessons | structured GitHub issue |

Hard rule for every tier: **assemble → show → confirm → send.** The user sees the
exact file list / body before anything leaves the machine, and each `gh` command
that publishes (push, `pr create`, `issue create`) runs only after an explicit
yes. If the user declines at any point, stop cleanly and leave the assembled
material on disk for them.

## 0. Scope check

- Identify the case: a domain folder with `{name}_schema.json`, restatement,
  domain modules, benchmarks, README. If there is no finished case, say what is
  missing and stop (point at the mdp-solver skill).
- Resolve the interpreter exactly as the mdp-solver skill does
  (`$MDP_SOLVER_PYTHON` → workspace `.venv/bin/python` → create). Gates below run
  through it.
- Check `gh auth status`. Not authenticated → tell the user to run
  `gh auth login` and pause.
- Ask which tier (1 / 1b / 2) unless the request already says.

## 1. Always first: distill locally

If the case folder has an escalation log (`ESCALATION.md` per
ESCALATION_LOG_GUIDE), run its case-close distillation now, into the folder:
ledger → symptom-indexed lever entries; frame-changelog → frame-move entries.
This happens regardless of tier — it is for the user's own playbook; tiers only
decide what leaves.

## 2. Tier 1 — full case

**2a. Assemble.** Copy the case into a staging dir as `cases/<name>/` with:

- `{name}_schema.json` (frozen IR) + `{name}.restatement.md`
- domain modules per the spec chain: `_uncertainty`, `_scenarios`, `_mdp`,
  `_gym`, `{name}_ir_adapter.py` (portable-domain contract)
- benchmarks + their `_eval` scripts, `_ppo_train` / `_ppo_eval`
  (+ `_ppo_tune` if used), `{name}_policy.py`
- `README.md` — problem statement, layout table, **leaderboard**, and the exact
  commands that reproduce every number in it
- `ESCALATION.md` if the campaign kept one (recommended — it is half the value)

Exclude: `results/`, `__pycache__`, logs, anything gitignored. Then scrub the
staged files for private material — absolute local paths, credentials/keys,
internal hostnames, names/emails the user didn't choose to publish — and show
anything found.

**2b. Provenance (must ask, verbatim topics).** (i) Does the user have the right
to publish this problem? Paper-derived cases: citation present in the README.
Business-derived: the user confirms ownership or approval — if unsure, offer
tier 1b or 2 instead. (ii) Is the leaderboard honest — produced by the README's
own commands at the stated eval protocol? Record both answers in the PR body.

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
`cases/` admission question), leaderboard summary (is RL competitive? negative
verdicts are fine and said plainly), provenance statements from 2b, gates-green
statement. Show the user the final file list and PR body; send on confirm; report
the PR URL.

## 3. Tier 1b — re-skinned case

As tier 1, with an **isomorphic rename** before 2c: new domain name and story;
rename identifiers, docstrings, README narrative; abstract units/constants'
*labels* — while structure, distributions, dynamics, and numbers' roles stay
intact. The IR makes this mechanical: rename via the schema, regenerate names in
code, keep everything else byte-equivalent in behavior. **Proof of integrity =
the gates**: re-run 2c on the re-skinned folder; green means the rename broke
nothing. Ask the user to review the re-skinned README specifically for residual
business context before 2d.

## 4. Tier 2 — playbook entries only

**4a. Draft entries** in the playbook schema, one block each:

```
symptom:    <observable signature>
hypothesis: <mechanism>
probe:      <cheap confirmation, if any>
lever:      <layer: HP | gym-obs | gym-action | gym-reward | arch — and the change>
verdict:    <effect, normalized (% over baseline); scope conditions; interactions>
evidence:   <domain class + eval protocol — no business context>
```

**4b. Sanitize, default-deny.** Structure-level schema fields travel as-is;
free-text gets a scrub pass (domain renamed to its structure class — e.g.
"entity-ranking under noisy pairwise probes" — absolute numbers → % over
baseline; no constants that identify the business). Show the user a
before/after diff of every entry; they approve or edit each.

**4c. Send** as one issue:
`gh issue create --repo tong-wang/auto-mdp-solver --title "playbook: <n> entries — <structure class>" --label playbook-entry`
(omit the label if the repo lacks it), body = the approved entries + eval
protocol note. Show body, confirm, send, report the URL. The shipped playbook is
maintainer-curated: entries land via review, with attribution.

## Failure notes

- `gh repo fork` on an existing fork just reuses it — fine.
- If `pr create` fails for a first-time fork (detached upstream), retry with
  `--head <user>:case/<name>`.
- Never `git push` to `tong-wang/auto-mdp-solver` directly without maintainer
  rights confirmed; never force-push anything.

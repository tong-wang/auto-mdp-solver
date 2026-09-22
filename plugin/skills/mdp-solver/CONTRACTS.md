# The op contracts

The pipeline is six step skills plus this conductor, cut at durable-artifact
seams. Each op **re-validates its upstream gate at its own entry** — never
trust that someone "just signed off"; the entry gate re-derives it. The gate
is executable:

```bash
<python> -m mdp_stage {domain}              # stage table — where the folder stands
<python> -m mdp_stage {domain} --for solve  # one op's full entry gate, exit 0 required
```

| op | consumes | entry gate | durable exit artifact |
|---|---|---|---|
| `mdp-formalize` | verbal / paper | — (source op; scope check inside) | frozen IR + restatement + `{name}.signoff.json` |
| `mdp-build` | frozen IR | `--for build` | domain code + gym + tests + campaign docs |
| `mdp-solve` | domain passing build's gates | `--for solve` (runs conformance + laws + differential) | `{name}.runplan.json` + baselines + L0/L1 leaderboard row |
| `mdp-escalate` | an L1 row with a gap | `--for escalate` + the campaign log | L2+ winner + `ESCALATION.md` progress |
| `mdp-interpret` | winning artifact + declared stance | `--for interpret` | probe stats, fitted-rule score, overlay figure |
| `mdp-package` | winning artifact + eval TSVs | `--for package` (blocks on the §14 readback when a stance owes it) | `{domain}_policy.py` + finished README |
| `mdp-solver` | verbal / paper | env `ENV OK` | conducts the above; the final report |

## What each op reads

The governing docs all live in this directory (`${CLAUDE_SKILL_DIR}/../mdp-solver/`
from a step skill — `${CLAUDE_SKILL_DIR}` is the skill's own directory,
substituted by Claude Code at load; on Codex it is the path printed beside
the skill in the skills list. Read them from that path, never from a
filesystem search — a development checkout on disk would silently
substitute unreleased content).
They are large, and no op needs all of them. Each step skill's opening
paragraph names its slice; this table is the same map in one place. Sections
are cited by number because the spec's and the guide's §-numbers are
append-only.

| op | reads | does not read |
|---|---|---|
| `mdp-formalize` | `INTERVIEW.md`, `MDP_IR_SAMPLE.md`, spec §4.3, §5.1, §5.5, §9.3, §14.0 | the rest of the spec, the guide, the templates |
| `mdp-build` | spec §1–§7, `MDP_IR_SAMPLE.md`, both `DOMAIN_*_TEMPLATE.md`, `examples/MANIFEST.md` + one exemplar | spec §8–§14, the guide |
| `mdp-solve` | spec §5.6, §8, §9, §13; `INTERVIEW.md` (Stage 0) | the IR sample, the guide, `PLAYBOOK.md` |
| `mdp-escalate` | spec §8.4, §8.6, §13; `ESCALATION_LOG_GUIDE.md`; `PLAYBOOK.md` + matched example entries | spec §1–§7, the IR sample |
| `mdp-interpret` | spec §14 (+ §1.2, §8.4 where cited) | everything else |
| `mdp-package` | spec §12, §1.3, §9; both `DOMAIN_*_TEMPLATE.md` | everything else |
| `mdp-solver` | this file, `ENVIRONMENT.md` | the spec — each step reads its own slice |

`ENVIRONMENT.md` (interpreter, `ENV OK`, retry budgets, background runs)
binds every op and is short; read it once per session. Every op that acts on
an existing domain folder also reads `{domain}/CLAUDE.md` at entry, right
after its gate: it is the folder's operating brief (spec §1.3), Claude Code
pushes it when work touches the folder, other hosts do not, and its hard
rules bind the op either way. A step that finds
itself needing a section outside its slice may read it — the map is a
default, not a fence — but a section it cannot name is one it does not need.

## Cross-op state — derivable is re-derived

Nothing an op needs from an earlier op lives in conversation memory.
Everything derivable — IR validity, `unconfirmed()`, the fingerprint,
conformance / laws / differential verdicts, artifact presence — is re-derived
by `mdp_stage` at entry and **never recorded**: a recorded verdict goes stale
silently the moment the folder moves on. What is *not* derivable lives in two
single-writer JSON files beside the schema, both committed:

- **`{name}.signoff.json`** — written by `mdp-formalize` at its gate, and by
  nothing else:

  ```json
  {"mdp_fingerprint": "<mdp_fingerprint() at sign-off>",
   "signed_off": "<ISO date>",
   "restatement": "{name}.restatement.md"}
  ```

  The always-on freeze check compares it to the *current* fingerprint at
  every later op's entry, so a post-freeze edit of the `mdp` block is caught
  mechanically. gym/rl edits leave the fingerprint still — Phase B's mutable
  layers stay mutable.

- **`{name}.runplan.json`** — written by `mdp-solve` at run-plan
  confirmation, and by nothing else:

  ```json
  {"target": "<SCENARIOS or GRIDS entry name>",
   "target_kind": "scenario | grid",
   "strategy": "specialist | generalist",
   "escalation_budget": "<what L2+ may spend, in rounds or wall-clock>",
   "confirmed": "<ISO date>"}
  ```

  Downstream ops locate artifacts through `target`
  (`results/{target}/...`); the budget is `mdp-escalate`'s stopping rule.
  Returning for another target rewrites this file — that *is* re-entering at
  Stage 0.

`mdp_stage` only ever reads these files; a gate that repaired its own
preconditions would be no gate. Neither file is consulted by generated code —
they are pipeline state, not domain state, so the portable-domain contract is
untouched.

## Pre-split domains

A domain formalized before this convention has no sign-off file, so every
gate blocks. Backfill honestly, don't bypass: the fingerprint is derivable
(`<python> -m mdp_ir {name}/{name}_schema.json` prints it), and the sign-off
date is in the campaign record (restatement header, README, or the log). If
no human sign-off is on record anywhere, that is a real gap — take the IR
back through `mdp-formalize` steps 7–9 rather than inventing a date.

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
| `mdp-package` | winning artifact + eval TSVs | `--for package` | `{domain}_policy.py` + finished README |
| `mdp-solver` | verbal / paper | env `ENV OK` | conducts the above; the final report |

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

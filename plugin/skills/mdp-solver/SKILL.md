---
name: mdp-solver
description: >
  Build a trained, deployable RL policy from a verbal description of a
  dynamic decision problem: formalize it into an MDP-IR, generate the domain,
  baselines, PPO training, a policy-structure readback and a
  {domain}_policy.py wrapper, with executable gates between stages. Use when
  the user describes a sequential decision problem ("build a domain for...",
  "train a policy for...", "formalize this problem"), names an existing
  {name}/{name}_schema.json, or asks for "the MDP pipeline" / "Phase A" /
  "Phase B".
---

# MDP solver pipeline — the conductor

This skill is the fully automatic front door: it conducts the six step
skills of the split pipeline end to end, with an executable gate between
every pair. The steps are siblings in this plugin — `mdp-formalize`,
`mdp-build`, `mdp-solve`, `mdp-escalate`, `mdp-interpret`, `mdp-package` —
each independently invocable for re-entry; `CONTRACTS.md` (this directory)
is the op map: what each consumes, its entry gate, its durable exit.

The governing docs sit in this directory; `CONTRACTS.md` maps the ops and
says what each one reads — the conductor itself needs only `CONTRACTS.md`
and `ENVIRONMENT.md`; each step skill names its own slice of the spec.
**Read them from this skill's own directory, never from a filesystem
search**: a development checkout of the solver may also be on disk, and
reading that instead silently substitutes unreleased content for the version
you are installed at. The pipeline runs in whatever workspace holds the
domain; use that workspace's venv.

**Environment (once, before anything).** Resolve the one interpreter and
pass the `ENV OK` gate per `ENVIRONMENT.md` — required before any Phase-B
op, and its retry budgets, >30-minute ask, and background-run rules bind
every step. Use that exact interpreter path for every command, never a bare
`python`.

## Conducting

Dispatch the steps in order: for each, open the sibling step skill's
`SKILL.md` (`${CLAUDE_SKILL_DIR}/../mdp-<step>/SKILL.md` — this skill's own
directory is substituted by Claude Code at load and printed beside the skill
in Codex's list) and follow it to its exit; where the host offers a skill
invocation by name, that is the same file:

```
mdp-formalize → mdp-build → mdp-solve → [mdp-escalate]* → mdp-interpret → mdp-package
```

Between ops, run the next op's entry gate yourself and require exit 0:

```bash
<python> -m mdp_stage {domain} --for <next-op>
```

Never advance on the previous op's say-so — the gate re-derives what it
claims (the freeze check, the artifact trail, and at the solve seam the full
conformance + laws + differential re-run). A blocked gate names what is
missing; route back to the op that owns it rather than repairing state by
hand.

Branching is decided by evidence, not preference:

- **`mdp-escalate` opens only on `mdp-solve`'s exit diagnosis** — a gap
  against the baselines with a healthy build — and only within the run
  plan's escalation budget. Competitive at L1 skips it entirely.
- **`mdp-interpret` is owed by the IR's declared stances** — the entry
  gate's `interpret.owed` line quotes spec §14.0: confirm/discover owe the
  probe and INTERPRET.md, bypass (or no declared stance) owes only the
  generic pieces. Run what is owed; do not silently skip, do not gold-plate.
- **Re-entry for another target** goes to `mdp-solve` Stage 0 (a new run
  plan), then onward — never back through `mdp-build` unless a gate sends
  you there.

The run plan is the one seam between attended and unattended operation: if
the user supplied a run-plan file up front, `mdp-solve` consumes it instead
of asking (see its Stage 0). Everything else already runs on gates. The
human moments the conductor itself owns are the >30-minute compute ask
(`ENVIRONMENT.md`) and the final report; each step's own rules (the
Phase-A interview, the run-plan round, sign-offs) belong to the steps.

## Final report

Lead with the leaderboard (DP / PPO / heuristics / random, same seeds),
the % of optimal reached, the paths to the shipped model and policy file,
and anything the user must decide (failed axes, envelope compromises,
assumptions added to the IR's `assumptions_log`). When Stage 5 ran anchored,
the leaderboard carries the fitted rule as its own row, and the report states
the structure readback in one line (recovered rule, agreement, fitted vs net
— absolute gap beside the percentage). **Respect the Phase-A mode
stance:** put scenario modes on one head-to-head leaderboard only when they
were declared comparable; when they are independent branches (e.g. additive
vs multiplicative MMFE), report each as its own leaderboard — a cross-branch
"winner" is meaningless.

---
name: mdp-package
description: >
  Stage 6 of the MDP pipeline: wrap the winning artifact as a deployable
  {domain}_policy.py (spec §12) and finish the campaign README in its §1.3
  shape. Use when eval TSVs exist for the run plan's target and the winning
  artifact is chosen. Full pipeline: mdp-solver.
---

# mdp-package — Stage 6 (deployable policy + README)

One op of the split MDP pipeline; `mdp-solver` is the conductor. The governing
docs are at `${CLAUDE_SKILL_DIR}/../mdp-solver/` — read them from that path,
never from a filesystem search (a dev checkout on disk would silently
substitute unreleased content). Read `CONTRACTS.md` there first: it maps the
ops and says what each reads. **This op needs:** spec `MDP_PROJECT_SPEC.md`
§12, §1.3 and §9 (consult them while writing; do not code from memory of them)
and the two `DOMAIN_*_TEMPLATE.md` files — nothing else of the spec, no guide,
no IR sample. Everything else in that directory is consulted only when a step
below names it. Venv and run discipline per `ENVIRONMENT.md`.

**Entry gate — run first, exit 0 required:**

```bash
<python> -m mdp_stage {domain} --for package
```

The ordering-sanity half of the entry stays a command you run, not a file
check: re-run the Stage-4 `mdp_gates` command for the artifact being shipped
before writing a README that quotes it.

- `{domain}_policy.py` per spec §12 (model + vecnorm stats + action
  transform behind `act(obs)`; documented observation contract; `__main__`
  smoke test against the raw `_mdp` loop — run it).
- Complete `{domain}/README.md` in the spec-§1.3 shape it was seeded with at
  Stage 1 — the problem, the two layout tables, results by research-question
  tier, technical appendix. **Results is where the campaign's answers land:**
  the protocol every number is quoted at (including which eval mode is the
  record) and every symbol the tables use, stated once above them; then the
  scenario table before the boards that name it; then one §9 leaderboard per
  scenario carrying every baseline and the shipped model; then **which
  artifact ships and why that one** — a choice among trained arms is a claim,
  not a ranking. The `2-structural` block reports the verdict on each declared
  stance (§14.0), the Stage-5 readback among them: the recovered rule, its
  agreement with the reference, and the fitted rule's paired score.
- **Run the seven-question checklist (spec §1.3) over the finished README** —
  symbols defined before use, `§` references attributed, every scenario
  tabled with matching columns, one sort key and one unit per table, each
  board naming its shipped artifact and any absent arm, every percentage
  recomputed from the numbers beside it, every TL;DR bullet with a home below
  it and none of them tier 3. No gate sees any of these; the pass is the
  instrument. Fix what it finds before packaging.
- **Write the `**TL;DR**` block last**, replacing the Stage-1 placeholder,
  from the finished boards and after the checklist pass: two to five bullets,
  the claim in bold then the one number that carries it, tier 1 and tier 2
  only, no symbol the reader has not met. It is the one part of the README
  written for someone who will read nothing else, which is why it is written
  by the author who has just read everything — after, not before, the boards
  it summarises are final.
- Final pass over `{domain}/CLAUDE.md` (emitted at Stage 1 from
  `DOMAIN_CLAUDE_TEMPLATE.md`): the gate commands run as written, the hard
  rules are current and still phrased trigger → destination, and the two
  provenance lines are right — "built and gated at" naming the checkout the
  numbers came from, "conformance maintained through" naming how far the
  folder has been carried. **And check what the file has accreted:** anything
  describing, summarising or scoring what lives in another document is a copy
  and comes out — the README owns the case, this file owns the specs above
  it. Then add the domain to the root `CLAUDE.md`'s domain list — one line;
  the folder's own file carries the rest.
- Trained artifacts (`results/`) are gitignored; the README's commands must
  reproduce them.

Exit: the domain folder is the deliverable — reproducible from its README's
commands alone (`results/` is gitignored). Report per the conductor's final
report shape, or hand back to it.

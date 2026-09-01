# DOMAIN_README_TEMPLATE.md — the per-domain `README.md`

Template for the `{domain}/README.md` the pipeline emits at Stage 1 and
completes at Stage 6, so the case has a human-facing document from the first
commit rather than one accreted at the end. Its counterpart is
`DOMAIN_CLAUDE_TEMPLATE.md`, and the two divide by direction (spec §1.3):
**this file points across** — it inventories what the campaign produced and
reports what the campaign found; `CLAUDE.md` points up, at the specs that
govern the folder. A campaign document is inventoried in exactly one of the
two.

Filling rules:

- `{braced}` slots are filled by the pipeline; fixed text is kept **verbatim**.
- **The four sections stay in this order**, because each one must be readable
  by someone who has read only the sections above it. That is what puts the
  problem before the layout, the scenarios before the leaderboards that name
  them, and the commands last. A case with nothing to say in a section keeps
  the heading and says so in a line; it does not reorder.
- **The README grows during the campaign, not after it.** A finding lands here
  the day it lands in `ESCALATION.md` — a result whose only home is a round
  plan is a bug in the log (see `CLAUDE.md`'s file hygiene).
- **Every command is verified to run as written** before Stage 6 closes, and
  together they reproduce `results/` from an empty folder.
- Stage 6 runs the six-question checklist (spec §1.3) over the finished file.
  It exists because the shape alone has not been sufficient: the defects it
  catches — an undefined symbol the headline table turns on, two boards sorted
  by different keys, a column mixing units, a percentage that does not
  recompute — all survive a correctly-structured document, and no gate can see
  any of them.

---

````markdown
# {domain} — {one-line domain name}

## The problem

{2–5 paragraphs, before any file list. What decision is made, on what
cadence, under what uncertainty; what it costs to get wrong; what makes it
hard — the property that defeats the obvious policy (a curse of
dimensionality, a non-convexity, a partial observation, a coupling across
periods). A reader who stops here should know whether this case is relevant
to them.}

{One paragraph: what is known analytically. The reference or oracle if one
exists, the heuristics the field uses, and what is open — this is what the
leaderboards below are measured against.}

**Source.** {the paper, textbook problem or verbal brief this formalizes, and
what was changed if anything}. The authoritative statement of the problem is
`{domain}_schema.json`; `{domain}.restatement.md` renders it in prose.

## Layout

Documents — where to read about the case:

| doc | what it holds |
|---|---|
| `{domain}.restatement.md` | the Phase-A restatement: the problem as the IR states it |
| `ESCALATION.md` | the campaign — design tree, changelogs, numbered findings (#E…) with verdicts |
| `INTERPRET.md` | the policy readback (spec §14): what the trained nets actually do |
| `PLAYBOOK.md` | the case-close digest (guide §10) — {owed until the campaign closes} |
| `CLAUDE.md` | operating brief for an agent changing this folder |

{A document the spec requires and this campaign has not written stays in the
table marked **owed**, never silently dropped.}

Code — where things are implemented:

| module | role |
|---|---|
| `{domain}_schema.json` | **the IR — authoritative** for the problem definition |
| `{domain}_uncertainty.py` | {the stochastic primitives} |
| `{domain}_scenarios.py` | {the world layer: scenarios, sources, mixtures, `SCENARIOS`} |
| `{domain}_grids.py` | {the design layer: `GRIDS`} |
| `{domain}_mdp.py` | {state + transition} |
| `{domain}_gym.py` | {the Gymnasium wrapper and its observation/action modes} |
| `{domain}_ppo_train.py` / `_ppo_eval.py` | {training and grid evaluation} |
| `{domain}_benchmark_{method}.py` / `_eval.py` | {each non-RL arm} |
| `{domain}_policy.py` | {the deployable wrapper over the shipped artifact} |
| `{domain}_policy_probe.py` | {the §14 readback probe} |
| `{domain}_test.py` | {the domain's own tests} |

{Every tracked file in the folder appears in one of these two tables, this
README and `.gitignore` aside. A probe, an
extra gate or a config registry is code and belongs in the second — omitting
it is how a layout table starts describing a folder that no longer exists.}

## Results

### Protocol

Every number below is quoted at: {n} seeds from {first_seed}, {CRN across
arms | independent draws}, reference bar = {arm}, record eval =
**{`--stochastic` | deterministic argmax}** ({one line on why — e.g. policy
entropy IS the exploration mechanism / masked argmax is the deployment
mode}). The other mode appears only as a labelled figure, never as a record.
{Any board quoted at a different protocol says so in its own line.}

{Define here every symbol the tables below use — each value function, the
config-id grammar, each action encoding, the comparison column's unit. A
symbol first used inside a table is a symbol the reader meets undefined.}

### `1-comparative` — how does RL compare with the existing solutions?

The scenarios, and the role each plays in the study:

| scenario | config | what it is for |
|---|---|---|
| `{scenario}` | {the constants that distinguish it} | {the question this instance exists to answer} |

{**Which scenarios are separate leaderboards whose scores may never be
compared**, and why — different frames, different bars, different problem
scales. State it here, once, above the boards.}

#### {scenario} — {what it is}

| arm | {metric} | Δ vs {bar} | significance | scope |
|---|---|---|---|---|
| {…} | | | | |

{One leaderboard per scenario, per spec §9: one row per arm — the trained
policy, every baseline, the reference/oracle if one exists. Sorted by one
stated key, every column in one unit. An arm absent from this board says why
it is absent (e.g. the heuristic is undefined at `L > 0`).}

**Shipped: {artifact}** — {why this one}. {When several arms were trained,
the choice is a claim and is argued here, not implied by the ranking: a
lower-scoring arm may ship deliberately because its sibling was handed a
transform this result is meant to be evidence for.}

How it was reached: `ESCALATION.md` {§MAP and the findings that decided it}.

### `2-structural` — {the declared stance(s)}

{One block per stance in `research_questions.tier2` (spec §14.0): the claim as
declared, the verdict in brief, and the number that carries it. Evidence and
the full readback are in `INTERPRET.md` — this is the summary, not a second
copy. A `bypass` stance is reported the same way: the outcome comparison is
the evidence, and "the structure was not needed" is a result.}

## Technical appendix

Run from `{domain}/`. {Thread pinning / environment notes.} Together these
reproduce `results/` from an empty folder; `results/` is gitignored.

```bash
# train
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python {domain}_ppo_train.py -s {scenario} {…}

# evaluate — RL and each benchmark, same protocol
python {domain}_ppo_eval.py -s {scenario} --model-path {…}
python {domain}_benchmark_{method}_eval.py -s {scenario} {…}

# readback + figure
python {domain}_policy_probe.py {…}
python {domain}_plot_policy.py {…}
```

{The gate commands are not here — they are in `CLAUDE.md`, because they are
run by whoever is changing the folder rather than reading it.}
````

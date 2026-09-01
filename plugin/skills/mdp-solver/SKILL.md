---
name: mdp-solver
description: >
  Build a trained, deployable RL policy from a verbal description of a dynamic
  decision-making problem: formalize it into an MDP-IR, generate a
  spec-conformant domain (_uncertainty/_scenarios/_mdp/_gym), baselines,
  PPO training, tuning, a policy-structure readback, and a
  {domain}_policy.py wrapper — with executable gates between stages. Use when the user describes a sequential decision
  problem to model ("build a domain for...", "train a policy for...",
  "formalize this problem"), names an existing IR ({name}/{name}_schema.json in a
  domain folder), or asks to run "the MDP pipeline" / "Phase A" / "Phase B".
---

# MDP solver pipeline

This skill *is* the two-phase pipeline and is self-contained — run it end to
end from here. `MDP_PROJECT_SPEC.md`, co-located with this file, is the
canonical convention reference — consult it while writing each file; do not
code from memory of it. `MDP_IR_SAMPLE.md` (same directory) is the annotated
IR reference. The pipeline runs in whatever workspace holds the domain; use
that workspace's venv.

**Environment (once, before anything).** Every gate and every generated script
runs through **one** interpreter. Resolve it first, then use that exact path
for all commands in this skill — never a bare `python`.

Resolve in this order:

1. `$MDP_SOLVER_PYTHON`, if set.
2. `.venv/bin/python` under the **workspace root** (the directory holding the
   domain folders / `cases/`), if it exists.
3. Otherwise create it there — `uv venv --python 3.12` if `uv` is available
   (seconds; fetches a conforming Python if the box has none), else
   `python3 -m venv .venv`.

   The package needs **Python ≥ 3.12**. Check `python3 --version` *before*
   the fallback: a distro `python3` is often older (3.10 is common) and may
   also lack `ensurepip`, in which case `python3 -m venv` fails outright.
   If no conforming interpreter exists, stop and say so — do not build a venv
   on an unsupported version.

Install into it (skip if the probe below already prints `ENV OK`):

```bash
uv pip install --python .venv/bin/python "auto-mdp-solver[domain]"
# no uv:              .venv/bin/python -m pip install "auto-mdp-solver[domain]"
# from a repo checkout: ... install -e "<repo-root>/harness[domain]"
```

The **`[domain]` extra is required**: the bare package is deliberately
torch-free (it carries only the IR/conformance/gate/tuning harness), so
without it Stage 4 cannot train. Torch is a large download — say so before
starting one.

**GATE — run verbatim; `ENV OK` is required before Phase B:**

```bash
.venv/bin/python - <<'EOF'
import sys, importlib.util as u
core   = ["mdp_ir","mdp_conformance","mdp_gates","mdp_tuning","numpy","pydantic","gymnasium","optuna"]
domain = ["stable_baselines3","sb3_contrib","torch","pandas","tensorboard"]
miss = []
for m in core + domain:
    try:
        if u.find_spec(m) is None: miss.append(m)
    except (ModuleNotFoundError, ValueError): miss.append(m)
bad_py = sys.version_info < (3, 12)
if bad_py: print("BAD PYTHON:", sys.version.split()[0], "- need >= 3.12")
print("ENV OK" if not miss and not bad_py else "MISSING: " + " ".join(miss))
EOF
```

The venv lives in the workspace, not beside this skill: a plugin install is
replaced on update, and the generated domains plus each case README must stay
reproducible from the workspace alone.

**Retry budget: 3 repair attempts per gate, 1 per failing training design
axis.** When a budget is exhausted, stop and surface the failure — do not
loop. Ask before launching anything expected to take > 30 minutes of
compute; run training/tuning in the background and keep working.

## Scope check (before anything)

v1 envelope: finite horizon, low-dimensional state/action, single agent.
Latent state is allowed (gym hides it; may need frame-stack/RecurrentPPO).
The IR interpreter supports per-period distribution families: `categorical`,
`poisson`, `normal`, `lognormal`, `uniform`, `bernoulli` (plus the scenario-
sampler recipe families `choice_without_replacement`,
`normalized_uniform_weights`, and `iid` — a vector of `size` independent
draws of any base family, the per-episode latent-vector idiom); scalar
`[lo,hi]` decision bounds. If the problem doesn't fit, say exactly what doesn't fit
and stop — extending `mdp_ir/` is a separate task the user must approve.

## Phase A — formalize (human in the loop)

Phase A is a guided interview, not a form dump. At every moment the human
must know **where we are, what is settled, and what is still open**, and be
able to stop and ask instead of answering. Five interaction rules govern
this — they override the mechanical step list below whenever they conflict.

**1. Brief in the message, choose in the menu.** Every question round is two
parts of *one* turn: a **briefing message** in plain markdown, then the
`AskUserQuestion` call. The menu is a ballot, not a document — it is a narrow
column and long text there is unreadable. All exposition lives in the
briefing; the menu carries only the choices' names.

The briefing message (markdown, headings/bullets/tables — not a wall of
prose) carries, in this order:

- the **status board** (rule 2);
- **what is ambiguous here**, in two or three sentences: what the model needs
  to know and what the choice changes downstream;
- **the candidates**, one short subsection each, using the *same name* the
  menu option will use as its label: what it means in the problem's own
  terms, then its pros and its cons — the trade-off, not a lecture;
- **your recommendation and why**, unless the candidates are genuinely
  equivalent;
- one line noting they can answer "Other" to ask a question instead of
  picking (rule 4).

The `AskUserQuestion` call is then deliberately tiny. Hard limits:

| field | limit |
|---|---|
| `question` | ≤ 2 short sentences — restate the choice, point at the briefing above |
| `header` | ≤ 12 chars |
| option `label` | ≤ 5 words, **verbatim** the name used in the briefing so menu ↔ briefing map 1:1 |
| option `description` | one line, ≤ ~15 words — the consequence in a nutshell, not a paragraph |
| option `preview` | only for genuinely *visual* side-by-side content (a scenario grid, a composition table, a code sketch). Never prose — prose belongs in the briefing |

A description that wraps to three lines in a narrow column, or a `question`
carrying the status board, is the failure mode this rule exists to prevent.

**2. Post a status board at the top of every briefing.** A short *structured*
snapshot (not prose the human has to mine):

- **Step:** which Phase-A step we are in (objective & mode stance → drafting
  → classifying randomness → designing the scenario set → resolving
  confirmables → confirming restatement).
- **Settled:** the facts now pinned down — decision, horizon, objective,
  each resolved assumption — one terse line each.
- **Open:** the still-unresolved points, *named*, listed in the order you
  will ask them. This is the agenda; the human should see the whole
  remaining queue, not discover it one surprise at a time.

Re-post the board as items move Open → Settled. Keep it tight; it orients,
it is not the restatement.

**3. Write for the problem owner, not for an RL engineer.** The human knows
their problem; assume they do not know the solution techniques. In the
briefing, describe each option in the problem's own vocabulary first
("unmet demand is simply lost" / "unmet demand waits and is served next
period"), and only then, if it helps, name the modelling consequence —
defining the term on first use ("this adds a *state variable*: a number the
policy sees each period"). In labels and descriptions, no unexplained
jargon at all: no `Confirmable`, `ScenarioSampler`, MMFE, frame-stack,
observation space, or IR field paths unless the human introduced the term.
If a choice cannot be stated without a technical term, define the term in
the briefing and use the plain phrasing in the menu.

**4. Ask one question at a time, and make pausing a first-class move.**
Default to a single question per round so the human can push back between
them. Batch only when questions are mutually independent *and* trivial
confirmations unlikely to spark discussion — and even then keep it to a few.
Every briefing ends with a standing note that answering "Other" (always
present) pauses the interview to ask a question or challenge the framing
instead of picking an option — the human cannot be expected to know this.
When they do, stop the sequence, resolve it, then re-post the board and
continue. A one-at-a-time cadence is what makes this possible — do not trap
the human in a batch they must answer before they can speak.

**5. Anything large ends the turn.** The restatement, a full scenario set, a
long table — print it in a plain message with **no tool call after it**, and
save it to a co-located file; ask the confirmation in the *next* round,
naming that file. Never ask the human to confirm something they have not
been shown.

Assistant text is normally rendered, but if the human reports they cannot see
what they are being asked to confirm, treat that briefing as lost: re-send it
as a turn-ending message under this rule, then re-ask in the next turn.

**Shape of one good round.** Briefing message:

> **Step 3 of 6 — how unmet demand behaves**
> **Settled:** decision = order quantity each week · horizon = 52 weeks ·
> objective = minimize total cost
> **Open:** unmet demand → shortage cost → lead time → scenario grid
>
> When a week's demand exceeds what you have on hand, the model has to say
> what happens to the excess. This changes what the policy has to keep track
> of, and it changes what "cost" means, so it is worth getting right.
>
> **Lost sales** — the customer goes elsewhere; the excess demand disappears.
> *Pro:* simpler, and matches walk-in retail. *Con:* if your customers
> actually wait, it understates the pain of running out.
>
> **Backlog** — the customer waits and is served first next week.
> *Pro:* right for contracted/B2B supply. *Con:* the model must carry the
> outstanding amount from week to week, which makes the problem a little
> harder to learn.
>
> I'd suggest **Lost sales** unless your buyers reliably wait — you described
> walk-in customers earlier.
>
> If you'd rather ask something than answer, pick "Other".

Then the call: header `Unmet demand`; question "When demand exceeds stock,
does it vanish or wait? (see the two options above)"; labels `Lost sales` /
`Backlog`; descriptions "Excess demand disappears; pay a shortage penalty" /
"Excess demand waits, served first next week". Nothing longer.

**Respect dependencies across rounds.** If an answer could eliminate or
reshape a later question's options, ask the gating question first and build
the later options from the answer actually given — never from an assumed one.
(E.g. "backlogged or lost?" gates the shortage-cost options; "discrete or
continuous decision?" gates the bounds/masking question.)

1. **Establish the objective and the mode stance — first, before drafting.**
   Pin down what the project optimizes (the reward/objective) and, crucially,
   how the problem's *scenario modes* (the categorical attributes, step 2) are
   meant to be used: are they **competing designs to compare head-to-head**
   (the point is to find which wins), or **independent branches reported
   separately** where a head-to-head comparison is meaningless (e.g. additive
   vs multiplicative MMFE — different demand models, not rival designs)? This
   stance drives the scenario set, the train/eval strategy, and the final
   leaderboard; get it explicitly. Ask about **intrinsic discounting** here
   too: does the objective discount future rewards (time value of money, a
   continuation/survival probability)? That is `objective.discount_factor`
   (β, default 1.0) — part of the *objective*, never a solver knob: eval and
   every baseline score `Σ β^t r_t`, training γ defaults to β (spec §8.6),
   and rewards are never pre-discounted inside the env.
   **When the objective was chosen among competing candidates**, follow up
   in the next round: "the candidates you didn't pick — keep any as
   report-only columns? They'll appear in every evaluation table but never
   decide anything." A multi-select over the declined candidates; each kept
   one becomes a root-level `eval_metrics` entry (`name`, `expr`, `reduce`,
   with `source` recording the answer, e.g. "objective candidate, declined
   at Phase A"). Bystander metrics describe, never decide (spec §9.3) — and
   they sit outside the mdp block, so they never move the fingerprint.
   **Ask the tier-2 research question in this same round** (spec §14.0): is the
   campaign trying to **confirm** a structure already known for this problem
   ("does RL recover the `(s,S)` policy?"), to **discover** structure nobody has
   named, or to **bypass** it ("does RL do as well without the transform, or
   without the machinery that produces it?")? It is the same kind of commitment
   as the mode stance and it decides a Stage-5 deliverable: confirm and discover
   owe a `{domain}_policy_probe.py` and an `INTERPRET.md`, bypass owes neither.
   Record the answer as a root-level `research_questions.tier2` entry (`stance`,
   `structure`, `claim`, `instrument`); more than one stance on the same
   structure is legitimate. Undeclared, the same measurement reads three ways —
   `raw ≈ echelon` is a success under bypass and merely inconclusive under
   confirm — so a finding and a gap become indistinguishable in the record.
2. Interview the user; draft `{name}/{name}_schema.json` — the IR lives in the
   domain's own folder; create `{name}/` now if this is a new domain (v0.4 root:
   `{domain, mdp, gym, rl, assumptions_log}`; `MDP_IR_SAMPLE.md` is the
   annotated reference). **Classify every scenario attribute at the `mdp`
   layer** (spec §5.1): numerical-valued (cost params, horizon, distributional params) →
   `mdp.scenario.constants`, `axis`-tagged — the candidates you sweep into a
   *grid*; categorical-valued (bool flags, string enums) → **scenario-mode**
   attributes, whose allowed enum values you enumerate here. This split is a
   fact about the model — authoritative and frozen — not a design choice.
3. **Classify every randomness source** (spec §4.3, §5.5) — three bins, and
   only the first two are asked about here:
   - realized per transition, or evolving *within* the episode (a regime
     that can flip mid-episode is intrinsic, however mixture-like it looks)
     → an `uncertainty_slots` entry: structural skeleton (name, interface,
     `stream_id`, stages) + a `candidates` pool holding the concrete
     family the problem poses (`default` names it). The forward run authors
     exactly one candidate; more append later as one-line selections, never
     a second schema (`MDP_IR_SAMPLE.md` §1, §7);
   - realized once per episode **as part of the problem's story** — nature
     draws it (a hidden market size, a demand-regime pick, a random problem
     instance) → a **world latent** → a `draw` spec on the candidate
     setting it realizes (desugared to a `ScenarioSampler` at the slot's
     stream id; mixtures stay `scenario.mixtures`), *not* a generator.
     Ask the deployment test in the problem's own words:
     "in the real system, would this be drawn afresh each episode by the
     environment?" Then the follow-up that shapes observations and
     baselines: does the decision-maker *see* the realized value (observed)
     or must the policy cope without it (hidden)? Record `hidden` on the
     draw spec accordingly. A draw hiding *inside* a per-period source (its
     distribution's parameters fixed at episode start) is still a world
     latent — extract it into a draw spec.
   - a distribution that exists only to train one policy across many
     complete problem variants is **neither** — that is the training-target
     question, asked in step 4, never during randomness classification.

   State this classification to the user explicitly — it is the most common
   formalization error.
4. **Design the scenario set (human-decided) — never silently pick one.** The
   attribute *classification* (step 2) is fixed; the concrete *values and
   compositions* are the human's call. If the source (a paper, a brief)
   already carries an experiment design, **extract and translate it faithfully**
   into concrete `instances`, world-latent `draw` specs on the relevant
   candidates, and/or a `{Domain}ScenarioGrid`, then ask the human to
   confirm the translation. If it does not, **propose a reasonable grid + mode
   composition, with your reasoning,** and ask. Either way the concrete set
   being confirmed must be visible before the human answers: lay the
   candidates out in the briefing (rule 1), and — since a grid is genuinely
   visual — give each candidate composition a `preview` showing the actual
   axes × values × modes and instance count; if too large for that, print it
   in a turn-ending message saved to `{name}/{name}.scenarios.md` and ask in
   the next round (rule 5). Two coupled decisions, both
   following the step-1 stance: (a) which numerical axes vary and over what
   values/ranges, crossed with which scenario modes; (b) the
   **training-target question** this implies, asked in plain words: "do you
   want one policy tuned to this exact setting (a *specialist*), or one
   policy that works across a range of settings (a *generalist*) — and over
   what range?" A generalist's range is a `{Domain}ScenarioGrid` over the
   step-2 axes (spec §5.5–§5.6): training samples it via
   `grid.as_sampler()`, evaluation enumerates it — one leaderboard row per
   cell, all cells on the same eval seeds. Do not confuse this grid with
   step 3's world latents: the grid is the experimenter's choice and lives
   in the design layer; changing its weights changes only the training
   recipe. Also settle whether scenario modes land on one leaderboard or as
   separate branches. Emit whatever the strategy needs (fixed instances, a
   world sampler, a grid, or a combination) into the IR.
5. Validate: `python -m mdp_ir {name}/{name}_schema.json` must print OK.
6. Resolve every `Confirmable` (decision type/bounds, `requires_memory`,
   borderline placements) via AskUserQuestion — one at a time, under the
   interaction rules above; set `source` to `human_confirmed` /
   `human_override` in the JSON. Do this *before* the restatement so the
   restatement reflects the final, resolved model.
7. Round-trip artifact — build it now that the model is settled:
   a. Write the plain-English restatement to
      `{name}/{name}.restatement.md` (co-located with the IR so it
      is reviewable later): the problem in prose, the objective and mode
      stance, the randomness classification, the scenario set (grid axes,
      scenario modes, train/eval strategy), the decision/objective/horizon,
      and every assumption filled in without being told (each tagged with its
      resolved `source`).
   b. Render a sample trajectory and append it, annotated, to that file —
      **declaring the command that produced it** in a `step7b`-tagged fence
      immediately above the block holding its output:

      ````markdown
      ```step7b
      python -m mdp_ir.interpreter {name}/{name}_schema.json \
        --decision <name>=<value> --episode-seed 3
      ```

      ```
      <the CLI's output, pasted verbatim>
      ```
      ````

      Two rules make that block checkable and both are free. Write the path
      relative to the domain folder's **parent** (`{name}/{name}_schema.json`)
      so the command travels with the folder. And paste the CLI's output
      **verbatim** — never trimmed rows, never shortened column names: the
      block is a literal diff target, and a hand-tidied one is a block nobody
      can re-run. Prose may sit between the two fences; a command pairs with
      the next fenced block after it, and a file may declare several. For a
      decision whose `dim` is above 1, one value broadcasts to the resolved
      width (`--decision ship=10`) and a comma-separated list sets components
      (`ship=10,0,5`); add `--instance` to render the width that instance
      selects.

      **Re-render this whenever the model changes** — it is the artifact a
      human reads to check the world, so a stale one is worse than none, and
      no other gate can reach it: the differential proves the interpreter and
      the generated domain *agree*, so when the model moves both sides move
      together, the document drifts from both, and the differential still says
      MATCH. Declaring the command is what turns "re-render it" from an
      instruction into `docs.restatement_current`, the Stage-1 conformance
      check that re-runs it and diffs the result. Recording the `mdp`
      fingerprint in the file header is worth doing but is not that check: a
      token catches neglect, not the likelier **partial diligence** where the
      one-line field is dutifully updated and the expensive regeneration is
      skipped.
   c. Transcribe the user's **invariants** into `mdp.invariants` — the things
      they said must always be true ("stock only changes by what arrives and
      what sells", "every arrival either buys or is lost"). Write them as
      boolean exprs over the end-of-period namespace, using `prev.<name>` for
      the previous period and `close(a, b)` for float balances; put the claim
      in the user's own words in `desc`. This is the only artifact that can
      catch a mis-formalization the Stage-1 differential cannot see: the
      differential proves the interpreter and the generated domain *agree*, so
      a wrong sign that both sides share passes it. A claim taken from the
      problem statement is independent of the model and does not.
      Show the claims in the sign-off summary and report any violation the
      interpreter reports — advisory here, fatal at the Stage-1 gate.
8. **Final presentation & sign-off — the last thing before the gate.** In a
   single message: re-post the status board with everything now in *Settled*;
   print the full restatement and the annotated trajectory; and add a compact
   **frozen-model summary** — objective + mode stance, decision type/bounds,
   randomness classification, the scenario set (grid axes + scenario modes +
   train/eval strategy), horizon, and each key assumption with its `source`.
   Then ask for one explicit confirmation that this is correct and the `mdp`
   block may freeze. Never ask the user to confirm anything not shown in that
   same message — and per rule 5, the presentation message must **end its
   turn** (no tool call after it); ask for the sign-off in the next round,
   naming the restatement file. If they change something, apply it,
   re-validate, and re-present — the sign-off is always on the current
   displayed result.
9. **GATE (hard):** validator OK, `unconfirmed()` empty, the restatement file
   exists **and its declared `step7b` block re-renders** (run the command,
   diff it against the block — this is the last moment the trajectory and the
   about-to-freeze model are guaranteed to agree; from Stage 1 on
   `docs.restatement_current` re-runs it for you), and the user has given the
   final sign-off on the *displayed* result. Record `mdp_fingerprint()` in the
   conversation. The `mdp` block is now frozen; only `gym`/`rl` may change in
   Phase B.

## Phase B — build (gated stages)

Pick few-shot exemplars by problem shape. Three ship with this plugin in the
`examples/` directory beside this file (see `examples/MANIFEST.md`):
`inv_single` vector state + two-step advance + episode-support demand +
exact-DP baseline; `mab` exploration/exploitation over an equivariant state +
censored feedback + a fitted rule scored as a benchmark; `game2048` a
variable, long horizon + board state + CNN extractor + action masking. If the
current workspace contains other spec-conformant domains, prefer whichever
matches the problem shape (multi-entity, discrete + action masking,
deterministic dynamics + scenario-sampled instances, …); otherwise
generalize from the shipped two plus the spec's patterns.

### Stage 0 — run plan (confirm before building)

Phase A froze the *whole* scenario set; Phase B does **not** run all of it in
one pass. Before writing code, confirm a **run plan** with the human (via
AskUserQuestion, under the Phase-A interaction rules) — and treat it as a live
choice, not a Phase-A relic to rubber-stamp:

- **Target.** Choose what this pass actually builds baselines / trains /
  evaluates against — one `SCENARIOS` entry (a fixed scenario or a world
  sampler), or one `GRIDS` entry (a generalist target; and, if Phase A
  produced separate mode *branches*, which branch). Default to one target;
  never silently kick off every registered target. The Stage 1–2 code
  supports every registered scenario and grid regardless — this choice
  scopes only the expensive Stage 3–4 runs.
- **Train/eval strategy.** Re-confirm *or switch* the strategy the Phase-A
  scenario set implied: a **generalist** — trained on `grid.as_sampler()`,
  evaluated by enumerating the grid (one leaderboard row per cell, one
  shared eval-seed block, spec §5.6, §9.6) — vs. a **specialist** trained
  and evaluated on the single target. The strategy is not static — the
  human may change it here; honor it.

Carry the confirmed run plan as the contract for Stages 3–6: the selected
scenario name(s) drive the baseline, training, and eval commands, and the
leaderboard reports only what this pass actually ran. Returning for another
target later re-enters at Stage 0, not Stage 1.

### Stage 1 — model layers

Write in dependency order: `{domain}_exceptions.py` (optional) →
`{domain}_uncertainty.py` (omit if dynamics deterministic) →
`{domain}_scenarios.py` → `{domain}_mdp.py`.

- Seed keys follow the **§6.3 v2 seed tree**: declare `SEED_SCHEME = "v2"`
  and set `seed_scheme: "v2"` in the IR; build every key through the
  domain's `intrinsic_key()` / `meta_key()` helpers (never a raw
  `SeedSequence`). The interpreter implements both schemes and the
  differential gate is bit-exact under v2 (`examples/inv_single` is the
  reference); `"v1"` exists only for pre-redesign domains with recorded
  results.
- A generator whose distribution depends on the current decision takes it
  as an extra `sample(ctx, <decision>)` argument; the seed key must still
  contain only `(period, episode_seed, seed_salt)` slots so realized
  randomness is decision-path independent.
- Write a differential adapter **in the domain folder**:
  `{name}/{ir_name}_ir_adapter.py` exposing
  `make_adapter(ir, instance=None, seed_salt=0, domain_dir=None)` with
  `domain_dir` defaulting to the adapter's own directory
  (`mdp_ir.differential.load_adapter_factory` discovers it from the IR
  file's directory; `examples/inv_single/inv_single_ir_adapter.py` is the
  reference). Build the scenario **from the IR's constants**; emit
  row keys named exactly like interpreter rows; pass `seed_salt` through.
- Give `{domain}_mdp.py` a `__main__` smoke episode and run it.

- Write `{domain}_test.py` in the domain folder (spec §1.2): the engine laws,
  the differential parametrized over the covering set derived from the schema,
  at least one negative control, and any claim an IR expression cannot state.

- Emit the folder's two campaign docs from the templates beside this skill,
  in the shapes spec §1.3 fixes. They divide by direction: `CLAUDE.md` points
  **up**, at the specs governing the folder; `README.md` points **across**, at
  everything the campaign produces. A campaign document is inventoried in
  exactly one of them.
  - `{domain}/CLAUDE.md` from `DOMAIN_CLAUDE_TEMPLATE.md` — five sections:
    fill the problem statement, the leaderboard-commensurability line, the
    two provenance lines and the gate-command slots; keep the fixed text
    verbatim. The template's filling rules are binding — a pointer may name a
    destination, never describe or score what is inside it, and a hard rule is
    phrased trigger → destination.
  - `{domain}/README.md` from `DOMAIN_README_TEMPLATE.md` — four sections:
    fill **The problem** and the two layout tables now (documents the campaign
    owes are listed "owed", never omitted); the results and appendix sections
    are seeded as headings and grow as findings land, not at Stage 6.

- Emit `{domain}/.gitignore` beside it — three lines: `results/`, `scratch/`,
  `__pycache__/`. The folder must carry its own ignore rules (portable-domain
  contract): a contributed or promoted folder cannot assume the destination
  repo's root `.gitignore` covers them.

**GATE:** all three must exit 0 —
`python -m mdp_conformance {domain}` (generated-code shape),
`python -m mdp_ir.laws {domain}` (IR execution semantics), and
`python -m mdp_ir.differential {name}/{name}_schema.json --all-instances --episodes 40`.
The differential must be MATCH (bit-exact) with **no invariant violation**.
Then run `pytest {domain}` for the domain's own tests, and `pytest` at the repo
root to prove no regression to the shipped examples.

**Why bit-exact, and not just "tests pass":** formalization has no oracle —
dynamics can be modeled plausibly but wrongly, and a generated domain that
is self-consistent will pass every unit test while encoding the wrong
problem. The IR interpreter is the only independent implementation of those
dynamics, so replaying identical `(instance, episode_seed)` through both and
demanding an exact trajectory match is what actually catches a
mis-formalization. This is the reason the differential gate exists; treat a
MATCH failure as a modeling error to diagnose, never as a tolerance to relax.

### Stage 2 — gym wrapper

`{domain}_gym.py` from the IR's `gym` block: every observation mode (never
exposing latent state), every action mode with its declared strategy
(mask → `valid_actions` in the MDP layer + `action_masks()` on the env;
clip / reparametrize per spec §7.2), reward modes from objective
components in `info`, `terminated` at horizon + any early-termination expr.

- **The rendered vector is the declared vector** (spec §7): the modes the
  wrapper accepts are the modes the IR names, and each renders exactly that
  mode's `features`, in order. When the wrapper wants a component the IR does
  not declare, edit the IR first — `gym` is a mutable block, so a feature line
  costs no re-confirmation and moves no fingerprint. **The time feature is the
  one this loses**, in the domains that have one: where the horizon is the
  problem's own boundary (`termination.horizon_end: "terminated"`) the gym
  prepends one as a formatting decision, so it is written in the code and
  nowhere else unless you declare it. Where the horizon is only a cap
  (`"truncated"`) the problem is stationary, the policy must not see the clock,
  and there is nothing to declare — spec §7. No check catches either half; read the vector against
  the list by hand at the gate below — by *provenance*, not by width, since
  a mode may re-encode what it declares (a one-hot expansion of a declared
  board is compliant; an undeclared scalar prepended to it is not, and the
  two look alike if you only count). Spec §7 also states which encoding to
  prefer (`time_to_go = T - t`) and the horizon-proportional exception.

- **Action-box design lesson:** SB3 PPO's Gaussian initializes at raw
  action 0. If raw 0 maps to a dead zone (e.g. a shut-off intensity),
  learning stalls. Prefer encodings where raw 0 is a live, low-value
  action; keep the paper-native encoding as a non-default mode.

**GATE:** the module `__main__` runs random-action episodes in every
obs/action mode with `observation_space.contains(obs)` asserted each step,
and `python -m mdp_conformance {domain}` still passes.

### Stage 3 — baselines (mandatory)

Build and run **benchmarks** (any non-RL solution — from the source
paper/document or synthesized on the fly) **for the Stage-0 target
scenario(s) only**, not the whole frozen grid. Each is
`{domain}_benchmark_{method}.py` with an eval
`{domain}_benchmark_{method}_eval.py`, where `{method}` is a short method tag
(`lp`, `dp`, `myopic`, `greedy`, `fluid`, or a domain-custom heuristic) — not
a hard-coded `_lp`/`_dp`. Minimum: **random** and a **myopic/greedy
heuristic**. Add exact **DP** (`{domain}_benchmark_dp.py` +
`{domain}_benchmark_dp_eval.py`) when the state is fully observed and small
enough to enumerate — deriving the recurrence is your job; verify its value
against a closed-form or limiting case when one exists. All evals share the
spec-§9 seed loop, TSV columns
(`<axes>... <metric>_mean <metric>_var semivar_d semivar_u`), and one seed
count (default 8192). A benchmark that precomputes a solution table writes it
to `results/{scenario}/benchmark/{method}/{scenario}.txt`; eval TSVs go to
`results/{scenario}/benchmark/benchmark_{name}_eval_{scenario}.tsv`. Resolve
both from `Path(__file__).resolve().parent`, never from the CWD (spec §8.4) —
and redirect each eval's console output into that same `benchmark/` directory:

```bash
cd {domain}
out=results/{scenario}/benchmark; mkdir -p $out
for m in random myopic dp; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup ../.venv/bin/python \
    {domain}_benchmark_${m}_eval.py -s {scenario} --n-seeds 8192 \
    > $out/benchmark_${m}.log 2>&1 &
done
```

**GATE:** baselines run to completion and their ordering is sane
(DP ≥ heuristics ≥ random); they now bound the reward scale. Nothing was
written outside `results/` — `git status --short {domain}/` shows no stray
`*.log`/`*.tsv` in the domain folder, and no `results/` tree appeared anywhere
but the domain directory.

### Stage 4 — train + eval (+ tune)

Training runs are **leveled** (spec §8.6): **L0** = faithful defaults
(`PPO("MlpPolicy", env, gamma=β, seed=s).learn(2M)`, no VecNormalize — the
control run and the ruler for agent contribution; reporting only, never a
gate), **L1** = the spec-§8.6 derivation table applied — the mandatory run,
with every derived knob logged with its one-line rationale — and **L2+** =
escalations (hp / gym / arch), only after the L1 gate shows a gap. A tuned
result is always L2: level ≥ L2 ⟺ more than one training config was tried.
Run L0 by default (it is usually the cheapest run on the board); put both on
the leaderboard, clearly labeled.

Before designing any L2+ escalation, consult the shipped playbook:
`PLAYBOOK.md` beside this skill is the curated index, linking into
`examples/*/PLAYBOOK.md` — each a campaign's distilled experiences. Match on
*symptom*, read the matched entry **with its context**, and analogize —
entries are experiences, not rules; their scope conditions decide whether one
applies here. An entry's `#E` citations resolve in that example's
`ESCALATION.md`; drill down only when the entry's story needs chasing.

- Train on the **Stage-0 target and strategy** (the selected fixed scenario or
  world sampler for a specialist, or `grid.as_sampler()` for a generalist) —
  eval enumerates the grid's cells only when the strategy is generalist.
- `{domain}_ppo_train.py` per spec §8 (`_build_arg_parser`/`parse_args`/
  `main`; the **tier-1 CLI surface** of §8.2's table — every name there is one
  tooling and humans join on across domains, and `scripts.cli_contract` reports
  what is missing; tier-2 knobs are whatever `mdp_tuning --show-space` names for
  the family, expose all of them or the study searches a smaller space than it
  reports; VecNormalize per the IR's `obs_normalization` decision **with
  `gamma=args.gamma` passed**; script defaults = the L1-derived values, so the
  tuner's warm-start trial 0 is the L1 center; run-name encodes obs/act/rew +
  non-default hyperparameters).
- Launch with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, in the background;
  watch `ep_rew_mean` against the baseline bounds while it runs. The train
  script tees its own stdout/stderr to `{run_dir}/train.log` (spec §8.4), so
  send the shell redirect to the scratchpad and poll the run's own log:

  ```bash
  cd {domain}
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup ../.venv/bin/python \
    {domain}_ppo_train.py -s {scenario} -o {obs} > $SCRATCH/train_launch.log 2>&1 &
  # the script prints `outdir=` on its first lines; watch results/{scenario}/*/train.log
  ```

  Never redirect a run's output into the domain folder — that is what leaves
  orphaned `results_train_*.log` files behind.
- `{domain}_ppo_eval.py` per spec §9.5 (VecNormalize injection), same seed
  protocol as the baselines — the same `(--first-seed, --n-seeds)` block on
  every arm, which is what makes the per-seed differences paired. It knows its
  run directory from `--model-path`, so redirect it straight there:
  `> $(dirname <model-path>)/eval.log 2>&1`.

**GATE:** `python -m mdp_gates --candidate <ppo_eval.tsv>
--baseline <random.tsv> --baseline <myopic.tsv> [--reference <dp.tsv>]
--n-seeds <N>` must exit 0 (beats every baseline by ≥ 2 SE). It also refuses
a candidate whose run stopped short of its declared budget (§13's harvest
precondition) — re-run it, or `--short-ok "<reason>"` if the stop was
deliberate. Report the gap to the DP reference. If a design axis (an action/obs mode) fails after
its one repair attempt, ship the best passing axis and record the failure.

**GATE (layout):** the produced tree matches spec §8.4 exactly — run dirs
directly under `results/{scenario}/` with no interposed `RL/`, each holding
`train.log`, `{scenario}_{algo}_args.txt`, the model, and the eval TSV; no
`results/` anywhere but the domain directory; no `*.log`/`*.tsv` loose in the
domain folder. Check it, don't assume it:

```bash
cd {domain}
find results -maxdepth 2 -type d | sort          # expect {scenario}/{run_name}, {scenario}/benchmark
ls *.log *.tsv 2>/dev/null && echo "STRAY FILES — fix the path anchoring"
ls ../results 2>/dev/null && echo "CWD-ANCHORED WRITE — fix the path anchoring"
```

A miss here is almost always a script resolving `Path(args.outdir)` instead of
`Path(__file__).resolve().parent / args.outdir`, or a shell redirect aimed at
the CWD.

- Tuning — the escalation opened only when the L1 gate shows a
  gap (or on request): `{domain}_ppo_tune.py` thin wrapper over `mdp_tuning`
  pre-filling `--metric <metric>_mean` (and `--beta`/`--episode-len` from
  the IR). The driver warm-starts trial 0 from the script defaults (the L1
  center) and tunes the `core` knob tier by default — `--knobs breadth`
  needs ≥ ~40 trials, and is also where the two normalization priors
  (`norm_obs`, `normalize_advantage`) get checked, since §8.6's rows for
  them are settled by a run and by nothing earlier. **Tag the round by what
  moved, not by the fact that a study produced it:** the tiers cross layer
  lines (`core` reaches the extractor knobs, `breadth` reaches `norm_obs`),
  so a winner can be `L3(hp+arch)` or `L3(hp+gym)` — `--show-space` says
  which layers a study can reach before it runs, `--fix` closes one.
  Diagnosis before escalating (spec §8.6): L1 ≤ random
  → build bug; L1 < L0 → derivation misfired; competitive → stop at L1.
  **Selection-bias rule:** the study winner was selected on its tuning
  seeds — always re-evaluate the winning artifact with the full protocol
  before comparing or shipping, and expect the score to drop. A config
  tuned at a small budget does not necessarily improve when retrained
  longer; prefer shipping the tuned artifact itself. If you do retrain a
  winner, read its config off the trial's own args log: a trial command is
  the **derivation plus the delta the study searched** (spec §8.6), so
  "the script's defaults plus the winning cfg" is a different configuration
  wherever a default has moved off the derivation.

### Stage 5 — interpret

What did the policy learn? Read the winning artifact back into the domain's
policy-structure vocabulary per spec §14. **Anchored** whenever a reference
policy or a predicted structural class exists — any domain with a DP
baseline qualifies. With neither, only the generic pieces apply (the
action-surface figure, the feature-sensitivity sweeps); record what they
show and move on.

- `{domain}_policy_probe.py` per spec §14.1: the action surface over a
  state grid, the structural-form statistic, recovered thresholds + action
  agreement vs the reference, feature-sensitivity sweeps. Validate the
  probe first on an instance whose optimal structure is known before
  trusting it where none is. If a probe carries its own implementation of
  the MDP core (a vectorized sim for sweeps), it is licensed but gated:
  declare the equivalence it claims and prove it in `{domain}_test.py`
  (spec §1.2's second-implementation gates) — probe outputs land in the
  run's `probe/` dir, readback conclusions in `interpret/` (spec §8.4).
- Fit the predicted structural rule and **score it under the Stage-4 eval
  protocol** (same seeds, paired): report reference / fitted rule / net,
  with the verdict branches pre-decided per spec §14.2. The fitted rule
  beating its own net is a real outcome — when it happens, the fitted rule
  is a shippable artifact; ship it and say so.
- `{domain}_plot_policy.py` per spec §14.3: the overlay figure in canonical
  coordinates; committed static render in `{domain}/figures/`, interactive
  HTML beside the runs in `results/{scenario}/figures/` (gitignored);
  committed markdown cites the regenerating command.

No hard gate: a failed structural recovery fires spec §14.2's diagnosis
branches but does not block packaging — the finding (including "it is not
doing the classical thing") is a verdict on a declared stance and goes in the
README's `2-structural` results (spec §1.3).

### Stage 6 — package

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
- **Run the six-question checklist (spec §1.3) over the finished README** —
  symbols defined before use, `§` references attributed, every scenario
  tabled with matching columns, one sort key and one unit per table, each
  board naming its shipped artifact and any absent arm, every percentage
  recomputed from the numbers beside it. No gate sees any of these; the pass
  is the instrument. Fix what it finds before packaging.
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

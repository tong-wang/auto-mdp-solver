---
name: mdp-formalize
description: >
  Phase A of the MDP pipeline: a guided interview (or faithful paper
  translation) that turns a verbal dynamic decision problem into a validated,
  human-signed-off MDP-IR — the frozen contract every later op builds from.
  Use when a problem needs formalizing and no frozen, signed-off IR exists
  yet, or when a model change re-opens the mdp block. Exit: frozen IR +
  restatement + {name}.signoff.json. For the full automatic pipeline, use
  mdp-solver instead.
---

# mdp-formalize — Phase A (human in the loop)

One op of the split MDP pipeline (`mdp-solver` is the conductor; its
`CONTRACTS.md` maps the ops). The governing docs live beside the conductor in
this plugin's `mdp-solver` skill directory —
`${CLAUDE_SKILL_DIR}/../mdp-solver/` holds `MDP_PROJECT_SPEC.md` (canonical
conventions), `MDP_IR_SAMPLE.md` (the annotated IR reference),
`ESCALATION_LOG_GUIDE.md`, `ENVIRONMENT.md`, `INTERVIEW.md`, `CONTRACTS.md`,
`PLAYBOOK.md`, the two `DOMAIN_*_TEMPLATE.md` files and `examples/`. **Read
them from that path. Do not search the filesystem for them**: a development
checkout of the solver may also be on disk, and reading that instead silently
substitutes unreleased content for the version you are installed at. The
pipeline runs in whatever workspace holds the domain; use that workspace's
venv, resolved per `ENVIRONMENT.md` — whose retry budgets and background-run
rules govern everything below.

Entry needs no gate — this is the source op. But if `{name}/{name}_schema.json`
already exists, run `<python> -m mdp_stage {name}` first and say where the
folder stands: re-opening a signed-off model voids its sign-off (the
fingerprint moves), and the human should choose that knowingly.

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

## The interview

Phase A is a guided interview, not a form dump. The five interaction rules in
`INTERVIEW.md` (same directory as the spec, see above) govern every question
round below — brief in the message + choose in the menu, status board every
briefing, problem-owner vocabulary, one question at a time with pausing
first-class, anything large ends the turn — and they override the mechanical
step list whenever they conflict. Read them before the first round.

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
   final sign-off on the *displayed* result. Record the freeze durably: write
   `{name}/{name}.signoff.json` per `CONTRACTS.md` — the current
   `mdp_fingerprint()`, the sign-off date, the restatement filename. This op
   is that file's only writer, and every later op's entry gate compares it
   against the fingerprint as it stands then. The `mdp` block is now frozen;
   only `gym`/`rl` may change in Phase B.

Exit: `<python> -m mdp_stage {name} --for build` now passes — hand off to
`mdp-build` (or back to the conductor).

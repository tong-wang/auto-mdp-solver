---
name: mdp-solver
description: >
  Build a trained, deployable RL policy from a verbal description of a dynamic
  decision-making problem: formalize it into an MDP-IR, generate a
  spec-conformant domain (_uncertainty/_scenarios/_mdp/_gym), baselines,
  PPO training, tuning, and a {domain}_policy.py wrapper — with executable
  gates between stages. Use when the user describes a sequential decision
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

**Toolchain check (once, before anything).** Every gate requires the
`auto-mdp-solver` Python package (import names `mdp_ir`, `mdp_conformance`,
`mdp_gates`, `mdp_tuning`). Verify with `python -c "import mdp_ir"`; if it
fails, install it into the workspace venv (create `.venv` first if the
workspace has none): `pip install auto-mdp-solver` — or, when working from a
checkout of the auto-mdp-solver repo itself, `pip install -e <repo-root>`.
`python -m mdp_ir` / `mdp_conformance` / `mdp_gates` / `mdp_tuning` must all
resolve before Phase B begins.

**Retry budget: 3 repair attempts per gate, 1 per failing training design
axis.** When a budget is exhausted, stop and surface the failure — do not
loop. Ask before launching anything expected to take > 30 minutes of
compute; run training/tuning in the background and keep working.

## Scope check (before anything)

v1 envelope: finite horizon, low-dimensional state/action, single agent.
Latent state is allowed (gym hides it; may need frame-stack/RecurrentPPO).
The IR interpreter supports distribution families: `episode_categorical`,
`categorical`, `poisson`, `normal`, `lognormal`, `uniform`, `bernoulli`;
scalar `[lo,hi]` decision bounds. If the problem doesn't fit, say exactly what doesn't fit
and stop — extending `mdp_ir/` is a separate task the user must approve.

## Phase A — formalize (human in the loop)

Phase A is a guided interview, not a form dump. At every moment the human
must know **where we are, what is settled, and what is still open**, and be
able to stop and ask instead of answering. Four interaction rules govern
this — they override the mechanical step list below whenever they conflict.

**1. Post a status board with every question round.** A short *structured*
snapshot (not prose the human has to mine), carried inside the question text
per rule 4 — never only as free text before the tool call:

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

**2. Ask one question at a time, each self-contained.** Default to a single
question per AskUserQuestion round so the human can pause and push back
between them. Batch only when questions are mutually independent *and*
trivial confirmations unlikely to spark discussion — and even then keep it
to a few. For each question:

- Put the context *inside the question text*: two or three sentences on what
  is ambiguous here, why it matters, and what the choice changes downstream.
  Never make the human hunt an earlier wall of text for the relevant
  paragraph.
- Write each option `description` as its **consequence**, not a bare label:
  the state variable, cost term, or dynamics it implies. E.g. "Lost sales —
  unmet demand disappears; per-unit shortage penalty, no backlog state" vs
  "Backlog — unmet demand carries to next period as a new state variable and
  is served first." Brief labels with no implication are the failure mode.

**3. Make pausing a first-class move.** End *every* question's text with a
standing note that answering "Other" (always present in the menu) pauses the
interview to ask a question or challenge the framing instead of picking an
option — the human cannot be expected to know this. When they do, stop the
sequence, resolve it, then re-post the board and continue. A one-at-a-time
cadence is what makes this possible — do not trap the human in a batch they
must answer before they can speak.

**4. The question is the only reliable display.** Assistant text emitted
between tool calls — including "printing" a proposal right before an
AskUserQuestion call — may never be rendered to the human, and thinking never
is. Never say "the above" or "as shown earlier" in a question. Every round
must be self-carrying:

- the compact status board and the decision context go **inside the
  `question` text**;
- each candidate's concrete content (a scenario grid, a composition table, a
  code sketch) goes in that option's **`preview`**, so the choices are
  visually comparable in the menu itself;
- anything too large for a question (the restatement, a full scenario set)
  is printed in a plain message that **ends the turn** — no tool call after
  it — and also saved to a co-located file; the confirmation is then asked
  in the *next* round, naming that file.

If the human reports they cannot see what they are being asked to confirm,
this rule was violated: re-present the content under it before re-asking.

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
   leaderboard; get it explicitly.
2. Interview the user; draft `{name}/{name}_schema.json` — the IR lives in the
   domain's own folder; create `{name}/` now if this is a new domain (v0.4 root:
   `{domain, mdp, gym, rl, assumptions_log}`; `MDP_IR_SAMPLE.md` is the
   annotated reference). **Classify every scenario attribute at the `mdp`
   layer** (spec §5.1): numerical-valued (cost params, horizon, distributional params) →
   `mdp.scenario.constants`, `axis`-tagged — the candidates you sweep into a
   *grid*; categorical-valued (bool flags, string enums) → **scenario-mode**
   attributes, whose allowed enum values you enumerate here. This split is a
   fact about the model — authoritative and frozen — not a design choice.
3. **Classify every randomness source** (spec §4.3): realized per transition
   → `uncertainty_sources`; realized once per episode with no per-transition
   sibling → problem-instance selection → `ScenarioSampler`, *not* a
   generator. State this classification to the user explicitly — it is the
   most common formalization error.
4. **Design the scenario set (human-decided) — never silently pick one.** The
   attribute *classification* (step 2) is fixed; the concrete *values and
   compositions* are the human's call. If the source (a paper, a brief)
   already carries an experiment design, **extract and translate it faithfully**
   into concrete `instances` and/or a `ScenarioSampler`, then ask the human to
   confirm the translation. If it does not, **propose a reasonable grid + mode
   composition, with your reasoning,** and ask. Either way the concrete set
   being confirmed must be visible *in the confirmation round itself* (rule 4):
   each candidate composition as an option whose `preview` shows the actual
   axes × values × modes and instance count — or, if too large, printed in a
   turn-ending message and saved to `{name}/{name}.scenarios.md`
   before asking. Two coupled decisions, both
   following the step-1 stance: (a) which numerical axes vary and over what
   values/ranges, crossed with which scenario modes; (b) the **train/eval
   strategy** this implies — one generalist trained on a `ScenarioSampler`
   over the grid then evaluated per instance, vs. a specialist trained and
   evaluated per instance; and whether scenario modes land on one leaderboard
   or as separate branches. Emit whatever the strategy needs (a fixed list, a
   sampler, or both) into the IR.
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
   b. Render a sample trajectory with
      `python -m mdp_ir.interpreter {name}/{name}_schema.json
      --decision <name>=<value> --episode-seed 3` and append one annotated
      trajectory to that file.
8. **Final presentation & sign-off — the last thing before the gate.** In a
   single message: re-post the status board with everything now in *Settled*;
   print the full restatement and the annotated trajectory; and add a compact
   **frozen-model summary** — objective + mode stance, decision type/bounds,
   randomness classification, the scenario set (grid axes + scenario modes +
   train/eval strategy), horizon, and each key assumption with its `source`.
   Then ask for one explicit confirmation that this is correct and the `mdp`
   block may freeze. Never ask the user to confirm anything not shown in that
   same message — and per rule 4, the presentation message must **end its
   turn** (no tool call after it); ask for the sign-off in the next round,
   naming the restatement file. If they change something, apply it,
   re-validate, and re-present — the sign-off is always on the current
   displayed result.
9. **GATE (hard):** validator OK, `unconfirmed()` empty, the restatement file
   exists, and the user has given the final sign-off on the *displayed*
   result. Record `mdp_fingerprint()` in the conversation. The `mdp` block is
   now frozen; only `gym`/`rl` may change in Phase B.

## Phase B — build (gated stages)

Pick few-shot exemplars by problem shape. Two ship with this plugin under
`examples/` at the plugin root (two directories above this file's real
location; see `examples/MANIFEST.md` there): `inv_single` two-step advance +
episode-support demand + exact-DP baseline; `dynamic_pricing` continuous
price control + decision-conditioned generator + exact-DP baseline. If the
current workspace contains other spec-conformant domains, prefer whichever
matches the problem shape (multi-entity, discrete + action masking,
deterministic dynamics + scenario-sampled instances, …); otherwise
generalize from the shipped two plus the spec's patterns.

### Stage 0 — run plan (confirm before building)

Phase A froze the *whole* scenario set; Phase B does **not** run all of it in
one pass. Before writing code, confirm a **run plan** with the human (via
AskUserQuestion, under the Phase-A interaction rules) — and treat it as a live
choice, not a Phase-A relic to rubber-stamp:

- **Target scenario(s).** Choose which `SCENARIOS` entry this pass actually
  builds baselines / trains / evaluates against — a single fixed scenario, or
  one `ScenarioSampler` (and, if Phase A produced separate mode *branches*,
  which branch). Default to one target; never silently kick off the entire
  grid. The Stage 1–2 code supports every registered scenario regardless — this
  choice scopes only the expensive Stage 3–4 runs.
- **Train/eval strategy.** Re-confirm *or switch* the strategy the Phase-A
  scenario set implied: a generalist trained on a `ScenarioSampler` then
  evaluated per instance, vs. a specialist trained and evaluated on the single
  target. The strategy is not static — the human may change it here; honor it.

Carry the confirmed run plan as the contract for Stages 3–5: the selected
scenario name(s) drive the baseline, training, and eval commands, and the
leaderboard reports only what this pass actually ran. Returning for another
target later re-enters at Stage 0, not Stage 1.

### Stage 1 — model layers

Write in dependency order: `{domain}_exceptions.py` (optional) →
`{domain}_uncertainty.py` (omit if dynamics deterministic) →
`{domain}_scenarios.py` → `{domain}_mdp.py`.

- Seed keys are **stream-before-period**: `[entity_id?, sub_stream?,
  stream_id, period?, episode_seed, seed_salt]` — matching the interpreter
  (`mdp_ir/interpreter.py:_numeric_seed_key`), *not* the prose snippet in
  spec §6.3. Bit-exactness with the interpreter depends on this.
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

**GATE:** both must exit 0 —
`python -m mdp_conformance {domain}` and
`python -m mdp_ir.differential {name}/{name}_schema.json --episodes 40`
(repeat with `--instance <x>` for every instance). The differential must be
MATCH (bit-exact). Also re-run `python -m mdp_ir.interpreter_test` to prove
no regression to the shipped examples.

### Stage 2 — gym wrapper

`{domain}_gym.py` from the IR's `gym` block: every observation mode (never
exposing latent state), every action mode with its declared strategy
(mask → `valid_actions` in the MDP layer + `action_masks()` on the env;
clip / reparametrize per spec §7.2), reward modes from objective
components in `info`, `terminated` at horizon + any early-termination expr.

- **Action-box design lesson:** SB3 PPO's Gaussian initializes at raw
  action 0. If raw 0 maps to a dead zone (e.g. a shut-off intensity),
  learning stalls. Prefer encodings where raw 0 is a live, low-value
  action; keep the paper-native encoding as a non-default mode.

**GATE:** the module `__main__` runs random-action episodes in every
obs/action mode with `observation_space.contains(obs)` asserted each step,
and `python -m mdp_conformance {domain}` still passes 11/11.

### Stage 3 — baselines (mandatory)

Build and run baselines **for the Stage-0 target scenario(s) only**, not the
whole frozen grid. Minimum: **random** and a **myopic/greedy heuristic**
(`{domain}_lp.py`).
Add exact **DP** (`{domain}_dp.py` + `{domain}_dp_eval.py`) when the state
is fully observed and small enough to enumerate — deriving the recurrence
is your job; verify its value against a closed-form or limiting case when
one exists. All evals share the spec-§9 seed loop, TSV columns
(`<axes>... <metric>_mean <metric>_var semivar_d semivar_u`), and one seed
count (default 8192). DP solutions go to
`results/{scenario}/dp/{scenario}.txt`.

**GATE:** baselines run to completion and their ordering is sane
(DP ≥ heuristics ≥ random); they now bound the reward scale.

### Stage 4 — train + eval (+ tune)

- Train on the **Stage-0 target and strategy** (the selected fixed scenario for
  a specialist, or the `ScenarioSampler` for a generalist) — not the whole grid.
- `{domain}_ppo_train.py` per spec §8 (`_build_arg_parser`/`parse_args`/
  `main`; expose `--learning_rate` under that dest so `mdp_tuning` can
  reach it; VecNormalize per the IR's `obs_normalization` decision;
  run-name encodes obs/act/rew + non-default hyperparameters).
- Launch with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, in the background;
  watch `ep_rew_mean` against the baseline bounds while it runs.
- `{domain}_ppo_eval.py` per spec §9.5 (VecNormalize injection), same seed
  protocol as the baselines.

**GATE:** `python -m mdp_gates --candidate <ppo_eval.tsv>
--baseline <random.tsv> --baseline <myopic.tsv> [--reference <dp.tsv>]
--n-seeds <N>` must exit 0 (beats every baseline by ≥ 2 SE). Report the
gap to the DP reference. If a design axis (an action/obs mode) fails after
its one repair attempt, ship the best passing axis and record the failure.

- Tuning (on request, or if the DP gap is large): `{domain}_ppo_tune.py`
  thin wrapper over `mdp_tuning` pre-filling `--metric <metric>_mean`.
  **Selection-bias rule:** the study winner was selected on its tuning
  seeds — always re-evaluate the winning artifact with the full protocol
  before comparing or shipping, and expect the score to drop. A config
  tuned at a small budget does not necessarily improve when retrained
  longer; prefer shipping the tuned artifact itself.

### Stage 5 — package

- `{domain}_policy.py` per spec §12 (model + vecnorm stats + action
  transform behind `act(obs)`; documented observation contract; `__main__`
  smoke test against the raw `_mdp` loop — run it).
- Domain `README.md`: layout table, usage commands, results table with all
  baselines and the shipped model, and any empirical findings (which modes
  won/lost and why).
- Add the domain to `CLAUDE.md`'s Core Domains list.
- Trained artifacts (`results/`) are gitignored; the README's commands must
  reproduce them.

## Final report

Lead with the leaderboard (DP / PPO / heuristics / random, same seeds),
the % of optimal reached, the paths to the shipped model and policy file,
and anything the user must decide (failed axes, envelope compromises,
assumptions added to the IR's `assumptions_log`). **Respect the Phase-A mode
stance:** put scenario modes on one head-to-head leaderboard only when they
were declared comparable; when they are independent branches (e.g. additive
vs multiplicative MMFE), report each as its own leaderboard — a cross-branch
"winner" is meaningless.

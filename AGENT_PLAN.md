# Agentization Plan — from skill to standalone agent

*Design record, 2026-07-23; revised 2026-07-24 (solve = backbone + escalation;
gym-gate design, §14). Captures the decisions, architecture, and roadmap for
turning the `mdp-solver` pipeline into a deployable agent. This is a plan, not yet
built; nothing here is committed to code.*

## 1. Goal and premise

Turn the verbal-problem → trained-policy pipeline (today a single Claude Code
**skill**, `plugin/skills/mdp-solver/`) into a form that can also run **headless, in
batch, and in parallel** — a standalone **agent**.

The 2026-07-21 restructuring already did the hard structural half: it split the repo
into `harness/` (deterministic, LLM-free CLI tools, published to PyPI) and `plugin/`
(the driving playbook), and made domains location-independent (portable-domain
contract). What remains is a thin driver plus one design fork (the human loop).

**Key framing:** the "agent" today *is* Claude Code interpreting `SKILL.md`. What we
build replaces the **harness** (Claude Code) with the **Claude Agent SDK**, and moves
the human's role from live steering to a pre-declared contract. Claude the reasoner,
bash, the venv, and the generated python are unchanged.

## 2. Guiding principle — two front-doors on one core

The skill and the agent are **not** either/or. `harness/` + the playbook are the
reusable core; there are two front-doors onto it:

- **skill** — the interactive front-door (one human, cold start, co-formalize);
- **agent** — the programmatic/headless/parallel front-door (warm-start fleets,
  embedded steps, scheduled re-runs, distribution to non-Claude-Code users).

The agent buys the most exactly where the skill is weakest (parallel / headless /
embedded / distributable) and the least where the skill is strong (the human-guided
interview). It does not retire the skill.

## 3. Phasing decision — split the skill *first*, then convert to agent

**Do the granular skill split before writing any SDK code.** Rationale:

- The hard part of the agent is not the SDK wrapper — it is nailing the **contracts and
  entry gates** between stages. Splitting the skill forces those contracts to exist, in
  the cheapest medium, with a human in the loop to catch seam bugs.
- It forces implicit cross-stage state (today carried in the conversation) to become
  **filesystem-discoverable and re-validatable** — the single hardest prerequisite for
  headless operation.
- It delivers real **re-entry ergonomics** immediately (jump straight to `solve` or
  `package`), valuable even if the agent is never built.
- The SDK loads `SKILL.md` files directly (`setting_sources` + `skills`), so the split
  skills *become* the agent's brain verbatim — no re-expression of the playbook.

## 4. The skill split — granular operations

Split the monolithic `mdp-solver` skill at **durable-artifact seams** (the gate
boundaries), not per-stage. Four operations plus a conductor:

| op | consumes | produces | re-checks at entry |
|---|---|---|---|
| `formalize` (Phase A) | verbal / paper | frozen IR + restatement | — (source) |
| `build` (Stages 1–2) | frozen IR | domain code + gym | IR validates, `unconfirmed()` empty, fingerprint |
| `solve` (Stages 3–4) | IR + domain + run-plan (incl. escalation budget) | leaderboard row + trained artifact | conformance 11/11, differential MATCH |
| `interpret` *(deferred)* | winning artifact + IR (+ known policy) | policy-behavior findings → feed `package` | winning artifact exists |
| `package` (Stage 5) | IR + winning artifact | `policy.py` + README | eval TSVs exist, ordering sane |
| `auto` (= `mdp-solver`) | verbal / paper | everything | conducts the above with gates between |

Rules:

- **Don't go finer.** Splitting `build` internals (uncertainty/scenarios/mdp/gym) is
  over-granular — they share one differential gate and aren't independently entrant.
- **Each op re-validates its upstream gate at its own entry** — never trust a human who
  "just signed off," because the agent form has no such human. This is the same
  discipline the headless agent needs.
- **Single-source the playbook.** All ops reference one shared Phase-B playbook + the
  spec; no inlined copies, or they drift.
- **Naming:** prefix the family (`mdp-formalize`, `mdp-build`, `mdp-solve`,
  `mdp-package`) to avoid mis-triggering; keep `mdp-solver` as the `auto` umbrella (it
  has an established trigger description). Avoid the name `model` — ambiguous, and it
  collides with the `/model` built-in.
- **`formalize` stays interactive-only forever** — the human sign-off is load-bearing
  (formalization has no oracle). The other three are the dual-form (skill + agent)
  candidates. The "must stay a skill" vs "can become an agent" boundary becomes a file
  boundary instead of a paragraph in a 490-line document.
- **`formalize` has two *modes*, not two skills: `elicit` (verbal / underdetermined) and
  `replicate` (paper / source-specified).** They share the IR schema, Confirmable
  resolution, restatement, and sign-off, and converge on the identical frozen-IR contract
  — same entry→exit, no intermediate seam — so this is a mode fork, not a split. Real
  inputs are a spectrum: papers still leave gaps (RNG, bounds, observed/hidden,
  requires-memory) that need elicitation, so `replicate` is "mostly-determined + residual
  gap-filling," not zero-human. The skill already forks this way internally (SKILL.md
  step 4: source carries the design → translate faithfully; else → propose). What
  differs by mode: `replicate` does extract→translate→**fidelity-check** (vs
  interview→decide), the human is a fidelity-*confirmer* (vs decision-*maker*), and — key
  — `replicate` has an **oracle `elicit` lacks: the paper's reported results**, so it adds
  a *reproduce-the-results* validation gate. This refines the bullet above: it is the
  `elicit` mode that is irreducibly interactive; `replicate`, having an oracle, has a
  **higher automation ceiling** and is the mode that could later become agent-able.
  Optional: expose two thin trigger front-doors (`/mdp-formalize`, `/mdp-replicate`) over
  the one shared engine for discoverability — distinct briefings, same core; never fork
  the engine.
- **`solve` = mechanical backbone + conditional escalation layers (2026-07-24).** The
  backbone — train (vanilla PPO) → eval vs baselines — is the mandatory must-run and the
  thing that fans out. After it comes a **diagnosis point**: competitive vs baselines →
  done (`adi_flex`: vanilla PPO cleared the bar; even tuning added little). Gap → the
  brain judges which of **three orthogonal escalation layers** to open, singly or in
  combination:
  1. **HP layer** — hyperparameter tuning (`mdp_tuning`/Optuna);
  2. **gym layer** — obs/action/reward reshaping;
  3. **arch layer** — network architecture via `policy_kwargs` (feature extractor,
     attention, CNN kernels).
  All three are *optional* — escalation is a policy the brain applies to an outcome, not
  a pipeline stage (an earlier draft had a mandatory `improve` op; `adi_flex` refutes
  that). Precedent for the full-blown case: `topk_id` — many experiments running in
  parallel across the grid of all three axes. Two asymmetries keep "orthogonal" from
  meaning "interchangeable":
  - **Verification cost.** HP and arch are gate-free — pure solve knobs, no MDP
    implications, escalate freely. The **gym layer has its own gates** — resolved
    design in §14: obs changes are free *by construction* inside the filtered view
    (visibility tags); action changes need the feasibility-soundness check (+ coverage
    declaration); reward shaping is training-only with all selection on the faithful
    mode. Correction vs the first cut: gym edits never re-run the differential — that
    gates `_mdp` vs the IR and is blind to gym. The genuinely gated events are the
    upstream leaks: an info-completeness amendment (touches `_mdp` → differential
    re-run) or a new-information need (IR change → back to `formalize`). In headless
    form this partition is load-bearing — no human will catch a reward tweak that
    silently redefined the problem.
  - **Upstream reach.** Gym is the only layer that can leak past Phase B: an obs feature
    needing information the MDP state doesn't carry sends you back into `build` (or the
    IR). HP/arch never can. Hence a cheap-and-safe-first ordering for the automatic
    case, even though a supervised session (`topk_id`) can run all three in parallel.
  - **The brain's value is pruning, not searching.** The design space is high-dim and
    the feedback loop (train→eval) very slow — nobody affords the grid. What the brain
    contributes is a *prior* over which axis pays given the domain's structure, plus
    *symptom → lever* diagnosis from cheap evidence: plateaus **near** best baseline
    with a healthy curve → HP (squeezing, not missing); obs provably not a sufficient
    statistic → gym/obs; obs rich but *structured* (spatial grid, set/permutation,
    sequence) fed to a flat MLP → arch matched to the structure (the `topk_id`
    signature); at or below **random** → suspect the backbone (build bug), don't
    escalate at all.
  - **Escalation playbook — an accreting asset**, same pattern as `interpret`'s to-dos:
    a diagnosis table + per-layer levers + per-layer gate obligations, single-sourced in
    the skill, seeded from `topk_id`/`adi_flex` and grown per case. Each case's
    escalation trace ("symptom X, tried Y, result Z") is an entry; failures are content
    too (sudoku: budget exhausted without clearing the bar → negative-case entry).
  - **Stopping rule** = competitive-vs-baselines (the same criterion that admits a
    domain into `examples/`) or escalation-budget exhaustion; the budget lives in the
    run-plan.
- **`interpret` — deferred, underspecified (to-dos to accumulate from examples).** After
  `solve`, make sense of the winning policy: recover its *structure*, not just its score.
  Sits between `solve` and `package` (findings feed the package README's empirical-findings
  section): pipeline becomes `formalize → build → solve → interpret → package`. Precedent:
  `adi_flex` — compared the tuned model's actual actions against the paper's PL(σ) policy
  and found them behaviourally near-identical. Has the same **anchored vs open** duality as
  formalize's replicate/elicit: *anchored comparison* against a known structural policy
  (PL(σ), (s,S), base-stock, index, threshold — the benchmark is an oracle-like reference)
  vs *open discovery* where none exists. Candidate to-dos (seeds, not a spec): (1)
  behavioural comparison vs a known policy; (2) policy-structure visualization (action vs
  state — thresholds/monotonicity/kinks); (3) which state features drive decisions; (4)
  deviation analysis (where/why it departs from the benchmark — often the key finding). Like
  formalize, judgment-heavy → likely human-in-loop; the observable half (plots, action
  comparisons) is mechanical. **Note:** if this solidifies, the public pipeline string in
  `README.md` and `CLAUDE.md` (currently `formalize → build → solve → package`) becomes five
  verbs — not updated yet, since `interpret` is deferred.
- **Visualization (cross-cutting, deferred with `interpret`).** Two folds, both carrying
  the anchored/open axis: (1) **policy-structure viz** — render the learned policy's
  actions over the state space to expose thresholds/monotonicity/kinks; this is how
  `interpret`'s insight is actually delivered (a plot is usually the insight). *Anchored* =
  overlay against a known form (PL(σ), base-stock, *(s,S)*); *open* = discover structure
  with no reference. (2) **results-figure reproduction** — regenerate a paper's results
  figure with the RL policy dropped in (optionally alongside the paper's own policy). This
  is the *visual counterpart of `replicate`'s reproduce-the-results oracle gate* — same
  oracle (the paper's figure), and it doubles as a publishable deliverable. Homes: fold 1
  → `interpret`; fold 2 spans `replicate` (figure template) → `solve` (RL numbers) →
  `package` (artifact). **Fold 2's figure is dual-purpose — it also *specifies the scenario
  set*:** a paper's results table/figure encodes the experiment design (axes × values ×
  modes) that Phase-A step 4 extracts into the IR grid/instances. So in `replicate` the
  scenario set can be specified *by reference* — the interview offers the paper's own
  artifacts as the choices ("reproduce Table 3" vs "Fig 4", as in the `adi_flex` interview)
  — and the output then reproduces that same artifact populated with RL numbers. Input spec
  and output template are one object: the paper's table/figure **bookends** the pipeline
  (scenario spec in, results figure out), making the scenario set and the results
  visualization two views of one experiment design. One shared viz engine underneath,
  pointed at decision structure vs performance results, against open discovery vs the
  paper's figure. Objective already
  covers the *aim* (insight as a co-equal deliverable); viz is the *medium*, so it need not
  be a separate objective pillar unless we want it explicit.
- **`auto` is the agent orchestrator in disguise:** it sequences the ops with the
  interactive Stage-0 run-plan question in the middle. The batch agent is the same
  conductor with the run-plan *supplied* instead of *asked*.

## 5. The contract between phases

The split point is a **hard contract that already exists**: the frozen `mdp` block.
Phase A's exit == Phase B's entry == a validated, fingerprinted IR
(`{name}_schema.json`) + restatement. `python -m mdp_ir` validates it;
`mdp_fingerprint()` pins it. `plugin/skills/mdp-solver/examples/` are already frozen
Phase-B fixtures (IR → expected leaderboard) and become the agent's regression suite.

The agent's runtime contract is `(frozen IR, run-plan)`. The **only** behavioral fork
between the skill form and the agent form is Stage 0: the run-plan is *asked*
interactively in the skill, *supplied* as input in the agent (and the ">30-min compute"
pause becomes a policy flag). The run-plan carries scope (which targets/cells) **and
the escalation budget** — how much diagnosis-driven HP/gym/arch search `solve` may
spend beyond the backbone (§4).

## 6. Agent architecture — mode-2 central orchestrator

The fan-out boundary is **inside** Phase B, between Stage 2 and Stage 3
(already documented in Stage 0): **build once, fan out solve.** Stages 1–2 are per-IR
and shared; Stages 3–4 are per-target/grid-cell.

**Decision: mode 2 — one central orchestrator brain over LLM-free workers.** Not mode 1
(a full agent per fan-out cell).

- Reasoning is concentrated at the **seams** (build, gates, collect, **escalation
  diagnosis**, repair), not spread through the compute. The per-cell train/eval between
  seams is mechanical. The escalation diagnosis point (§4) is the richest of these
  seams — it is where an LLM in the Phase-B loop earns its keep.
- The fan-out unit **generalizes from scenario cell to (design-variant × cell)** when
  escalation opens: `topk_id`-style experiment grids over HP/gym/arch and the scenario
  fan-out are the same dispatcher fanning over runs; workers stay LLM-free either way.
- The central brain has **global / cross-cell view** — it can tell "every cell fails the
  differential the same way → build bug, fix once" from "cell 7 alone underperforms →
  tune cell 7." A per-cell brain is blind to that.
- The workers are the **plain generated python** (`_ppo_train.py`, `_ppo_eval.py`) —
  **no SDK, no API key, no egress.** This is what makes cheap cloud instances and
  air-gapped HPC compute nodes viable.
- One sparse-call session, one auth/egress point → trivially within rate limits.
- Mode 1 is a **degenerate special case** of mode 2 (orchestrator over one co-located
  cell); you can collapse mode 2 → mode 1 but not cleanly expand mode 1 → mode 2.

**Separate two decisions that are easy to conflate:**

1. **Brain topology** (per-cell vs central) — *choose now: mode 2.* This is the
   architecture.
2. **Dispatch backend** (local subprocess / `ssh`+`sbatch` / container) — *runtime
   config, not build-time.* The orchestrator talks to workers through one narrow
   interface; Alicloud-vs-Atlas is a backend swap, not a rebuild.

Mode 1 (per-cell agents) is only justified for **continuous, rich, run-specific live
control** — which is a bigger, pricier problem than anything in this pipeline and is
usually better served by an automated method (PBT/ASHA) than an LLM.

## 7. Mid-run supervision (smart early-stopping)

"Smart monitoring" does **not** overturn mode 2. Workers are **LLM-free, not
logic-free**; the decision splits into three tiers landing in three places:

- **Tier 1 — rule-expressible stops → callbacks in the worker (no LLM).** Plateau,
  divergence/NaN, "no eval improvement in K blocks," "below random baseline at step X."
  Use SB3 callbacks (`StopTrainingOnNoModelImprovement`, `StopTrainingOnRewardThreshold`,
  custom `BaseCallback`) and Optuna pruners (ASHA/median, already available via
  `mdp_tuning`) for population-level pruning. Kills most wasted compute, zero Claude.
  Under escalation (§4) this is what makes the slow feedback loop affordable: Tier 1
  prunes *within* running experiments; the brain's scarce judgment allocates budget
  *across* the three axes per diagnosis round.
- **Tier 2 — judgment / cross-cell stops → the central orchestrator.** Monitoring is
  *observe → decide → act*; observe (read the curve off shared FS/TB) and act
  (`scancel`/kill) are cheap data + control channels the central brain holds for all
  cells. Decisions are **coarse** (per eval interval, not per step), so one brain
  supervises tens of runs comfortably; shard supervisors at very large N.
- **Tier 3 — continuous live per-run control → the only real case for mode 1**, and even
  then an automated tuner usually wins. Not needed for early-stopping.

**Architectural cost:** widen the dispatcher interface from `submit → await result` to
`submit → stream progress → cancel`. Still backend-pluggable; workers stay LLM-free
(they already log progress; the orchestrator tails it).

## 8. Dispatch backends (deployment targets)

The orchestrator is a **dispatcher with a pluggable backend** behind a
`submit → (stream) → await/cancel` seam:

- **Local subprocess / GNU parallel** — Alicloud ECS or a workstation. Fan-out bounded
  by vCPUs.
- **`ssh` + Slurm (`sbatch`/`squeue`/`scancel`)** — Atlas HPC. The fan-out is a **Slurm
  array** over grid cells; each array task is bare `python …_train.py && …_eval.py` with
  no brain. The brain runs only at build / collect / repair.
- **Container runtime** — optional, for isolation when running a fleet.

**HPC (Atlas) specifics:**

- **Drop Docker** — often disallowed and heavy. Use a **shared venv/conda on the cluster
  filesystem** (the portable-domain contract makes a single shared env sufficient).
  Apptainer/Singularity `.sif` only if reproducibility later demands it.
- **Egress is a non-issue by design: the brain stays off the cloud/HPC entirely.**
  Chosen topology is (b) — **brain off-cluster** (on a machine that already has internet)
  driving via `ssh`+`sbatch`. It runs where it has egress; the only things inside
  Alicloud/Atlas are the **LLM-free workers**, which need no API egress. This also avoids
  the login-node reaper. (Alternative (a), brain on the login node under `tmux`, is not
  the plan of record.)

## 9. Deployment form and host prep

- **The brain is a thin Python program** using `claude-agent-sdk`. It bundles its own
  `claude` binary — **no Node, no separate Claude Code CLI.** Needs Python ≥3.10 and
  auth. It loads the split skill off disk (`setting_sources` + `skills`) — the playbook
  is not re-expressed. (Gotcha: SKILL.md `allowed-tools` frontmatter is ignored by the
  SDK; set `allowed_tools` in query options.)
- **Tools run in-process on the same host, unsandboxed by default.** The Bash tool runs
  `python …_train.py` in the venv on that machine. Containerize for isolation, not
  packaging.
- **Host prep ≈ the existing `ENV OK` box.** Python ≥3.12 + `.venv` with
  `auto-mdp-solver[domain]` (torch/SB3) — unchanged — plus `pip install
  claude-agent-sdk` and auth. **No GPU** (the v1 envelope is finite-horizon,
  low-dimensional → CPU-bound). The one undocumented item (managed-sandbox GPU) is moot.
- **Fan-out ceiling is CPU cores, not the API** — the solve brain is sparse in model
  calls; wall-clock is dominated by training on your own hardware.

## 10. Auth, billing, spend control

- **Subscription (Pro/Max) can drive the SDK without an API key** — leave
  `ANTHROPIC_API_KEY` unset; the bundled CLI does a browser OAuth login to your own
  account. The SDK is Anthropic's own product, so it is exempt from the third-party
  claude.ai-login ban.
- **On a subscription, spend is capped by construction** (flat fee — no metered bill to
  overshoot). The exhaustible resource is the **weekly usage quota** (throttles/blocks
  until reset, does not charge), shared across interactive + SDK use, **per-account not
  per-session** (wide fan-out draws one pool). Mode-2's sparse brain + own-hardware
  training keeps quota use light — this is what makes subscription-only viable.
- **API rail (if you outgrow the quota):** set a monthly **spend limit**
  (Console → Settings → Limits) — org-level, hard-pauses at the cap. That is the
  explicit runaway guard. Given sparse token use, metered cost is likely small.
- **Skip Managed Agents** — GPU undocumented, and unneeded (self-host CPU).

## 11. Publishing and who bears cost

The cost question turns on **whose credentials the brain runs under.**

- **Ship the code (recommended).** Publish the launcher as a console-script entry point
  alongside the existing PyPI package + plugin. Each user brings **their own credentials
  and their own compute** → **you bear nothing.** Marginal cost per user ≈ 0; it scales
  as a package download. This is continuous with how the plugin already works (installing
  it runs under the user's account, not the author's).
- **Host as a service (avoid).** Your key + your infrastructure → you bear tokens *and*
  the far-larger **training compute** for everyone. Architecturally wrong for a
  train-on-your-own-machine tool.

## 12. Roadmap

1. **Split the skill** into `formalize` / `build` / `solve` / `package` + the `auto`
   conductor, single-sourced, each independently entrant with entry-gate re-validation.
   Validate the contracts interactively. *(No SDK yet; pure skill-layer work.)*
2. **Convert `build`/`solve`/`package` to a mode-2 orchestrator agent** on the Agent SDK,
   with the **local subprocess backend** first. Run-plan supplied, not asked.
3. **Add the Slurm/`ssh` backend** for Atlas; add **mid-run supervision** (progress
   stream + `scancel`) and the Tier-1 worker callbacks / Optuna pruning.
4. **Publish as code** — launcher entry point, docs for BYO-credentials + BYO-compute.

## 13. Open items to verify (external mechanics — do not assume)

- ~~Anthropic API egress from Alicloud / Atlas compute nodes~~ — **resolved:** the brain
  stays off-cluster (§8), so nothing that needs `api.anthropic.com` runs there; the
  remote workers are LLM-free.
- ~~Gate design for gym-layer escalation~~ — **resolved 2026-07-24**, design in §14;
  implementation deferred to the coordinated schema+spec batch listed there.
- **Atlas** scheduler assumed Slurm — confirm the scheduler and `ssh` submission path.
- **Subscription/SDK current caps** and whether the once-planned separate SDK credit pool
  is live (it was **paused** as of 2026-06); figures drift — check the console.
- **ToS for unattended and *redistributed* subscription auth** — is an end user
  authenticating via subscription through a redistributed SDK launcher permitted, or must
  downstream users BYO-API-key? Confirm with Anthropic support before advertising
  subscription auth to downstream users.
- **Managed Agents GPU / wall-clock limits** — moot if self-hosting CPU boxes.

## 14. Gym-gate design — visibility, objectives, the problem/solution boundary

*Added 2026-07-24. Resolves the §13 gym-gate open item: what a gym-layer escalation
(§4) may change, and how validity is enforced without a human in the loop. Decisions
only — implementation is one deferred, coordinated schema+spec batch (end of section).*

### 14.1 The ruling — gym is solution-side, inside problem-side admissibility rules

The IR already splits correctly (MDP_IR_SAMPLE §0): `mdp` = the *problem* (frozen,
fingerprinted, human-confirmed), `gym` = the *interface menus* (mutable Phase-B design
axis, empirical oracle — "editing gym/rl does not change the fingerprint"). RL's
env-vs-agent vocabulary cuts along a different axis and only *coincides* with
problem-vs-solution for prefabricated benchmarks (Atari): here the Gym env is
manufactured — `_mdp` (problem) × interface lens (solution) — and reward shaping /
obs engineering / action design are solution work in RL practice too, just never
formally bounded because a human researcher holds the boundary in their head.

What escalation needs is not a new boundary but **richer admissibility constraints on
the solution-side menu, declared problem-side**. Today's constraint is one bit:
`observability: latent` / `hidden: true` bars a var from every obs mode (validated) —
a **blocklist with a permissive default** (everything non-latent is presumed
observable; censored demand is inexpressible; no timing; reward visibility absent).
Decision: flip to an **allowlist with explicit timing** — default-deny, the safe
polarity once a headless agent edits the gym block.

### 14.2 Obs gate — visibility tags, enforced by construction

- Per state/info element, a visibility tag in the **`mdp` block**, `Confirmable`-wrapped
  (who-knows-what-when has no oracle → Phase-A interview + sign-off; `ir.unconfirmed()`
  picks it up for free): `observed` (pre-decision at t) | `observed_expost` (enters the
  information set at t+1) | `latent` (never). Timing granularity: three-way enum with
  lag-1 semantics; exotic delays (episode-end revelation, lag-L) deferred until a case
  forces them.
- Formally the tags define the decision-maker's **filtration**; the gate in one
  sentence: *every gym obs feature must be measurable w.r.t. it* — a function of
  tagged-observable histories + own past actions + time index + scenario constants.
- **Enforce by construction, not review:** the gym is handed only the **filtered view**
  (the observable projection of state/info/reward histories per tags). Leaks become
  unrepresentable; obs engineering inside the view is free escalation territory — the
  semantic gate becomes a type-system gate.
- Prerequisite: **info-completeness** — `_mdp` publishes *every* potentially-observable
  realized quantity into `info` (transparency already implies it; the tags make it an
  obligation). Needing a quantity `_mdp` doesn't publish = amendment → differential
  re-run; needing information not in the problem = IR change → back to `formalize`.
- **Reward gets a visibility tag too** (`per_step` | `terminal` | `never`): "prev-reward
  in obs" is a standard trick, valid iff reward is tagged observable (topk_id: per-step
  reward would reveal the latent target — a leak through the reward channel).
- Cross-op dividends: the filtered view *is* the deployable `{domain}_policy.py` API;
  baselines must respect the filtration to count as competitors (tags let the
  leaderboard formally split admissible baselines from oracle/clairvoyant *bounds*);
  `interpret`'s feature attribution is only meaningful over filtration-legal features.

### 14.3 Action gate — soundness required, coverage declared

- **Soundness (required):** the gym action map lands inside the IR-declared feasible
  set — never emits an infeasible MDP action. Mechanically checkable.
- **Coverage (recorded, not required):** reparametrizations (discrete↔box,
  factorization, masking, order-up-to vs order-quantity) preserve the reachable set;
  deliberate action pruning restricts the attainable optimum — legitimate escalation,
  but declared on the leaderboard row ("RL lost" ≠ "RL-over-a-subset lost").
- **Correction to §4's first cut:** gym edits never re-run the differential — it proves
  interpreter ≡ `_mdp` and is blind to the gym layer.

### 14.4 Reward framework — objective families, scenario-declared target

The true objective already lives problem-side (`mdp.objective`, frozen; `info` mirrors
its components, validated) and the whole selection chain — eval TSVs, `resolve_metric`,
`mdp_gates` — already runs on info-derived economics, never the training reward. Two
things were missing: the **stated link** and the **tag**. Evidence across repos:
topk_id has the full distinction (`neg_oc` "THE eval metric" vs `shaped_round`
"TRAINING-ONLY", enforced by eval argparse `choices=["neg_oc"]`) but as domain-local
folklore; sudoku respects it by instinct (eval on `info["solved"]`, ignoring shaped
modes); 2048 and retailer never declared theirs (below).

- **The link (invariant to state in spec):** leaderboard/eval/tuning metric ≡
  `mdp.objective` evaluated from `info`, independent of `gym.reward_mode`. All
  selection decisions — eval, Optuna metric, best-checkpoint, leaderboard — on the
  faithful mode. Shaping needs no invariance proof (sudoku's non-potential `fill` is
  legal); eval-on-faithful is the arbiter, so shaping can only fail to help.
- **Objective families:** multiple legitimate objectives = a *family of problems
  sharing dynamics*, not one problem with many rewards (2048: `sum`/`max`/`logmax` are
  members, not shapings — the objective was never pinned; a formalize gap). The
  interview disambiguates three kinds of multiplicity: *ambiguity* → pick one
  (`Confirmable`); *plurality* → family members, each with its own leaderboard,
  baselines, and positivity verdict (per (domain, target), cf.
  examples-must-be-positive); *preference trade-off* → true multi-objective RL, **out
  of the v1 envelope** (a fixed scalarization collapses it to ambiguity).
- **Scenario-declared target:** the target objective is a **scenario field**. Rationale:
  the scenario already parameterizes the objective's numbers (cost coefficients); the
  would-be (objective × design × cell) fan-out collapses into the existing scenario
  axis (variants as `instances`, grids may put objective on an axis, run-plan selects
  targets by selecting scenarios); gym / baselines / packaged policy become
  self-describing (`resolve_metric("auto")` gets a principled answer); cross-objective
  eval degenerates into cross-scenario eval. Guardrails: (1) **family-level, never
  sampled** — no latent objectives (kin to the grid-never-in-SCENARIOS rule);
  (2) **measurement, not world** — invisible to `init_state`/`advance`, consumed only
  by gym reward assembly + eval/selection tooling (`sense: minimize` precedent:
  declared problem-side, applied gym-side); (3) **optional, singleton default** —
  required iff the family is plural, so inv_single-class domains never see it.
- **Compute-when-determined:** `_mdp` computes **all** variants' components into
  `info`, *unconditionally* — the target field selects among computed columns
  downstream, never gates computation upstream. Otherwise cross-objective eval, the
  multi-column TSV free lunch, and trajectory identity under target swap all silently
  die. "Unconditional" ≠ "per-step": objectives have a **timing class** —
  *accumulative* (episode value = Σ per-step components) or *terminal* (a functional
  determined at episode end, landing in the terminal transition's `info`: topk_id OC,
  sudoku `solved`, retailer gap). Schema gap: only `per_step_components` exists
  (schema.py:526) — per-step-ness is baked into the name. Conformance check (gate, not
  discipline): a scenario pair differing only in the target field must produce
  bit-identical state/info trajectories. Escape hatch if a component ever gets
  expensive: an explicit compute flag (kin to `logger_filename`), never the target
  field.
- **Role-relative shaping:** faithful-vs-shaping is **relative to the target binding**,
  not intrinsic to a mode. Each variant has exactly one faithful mode (invariant:
  episode-sum of faithful reward ≡ the variant's episode value — timing-agnostic; a
  sparse terminal reward satisfies it trivially); under a chosen target, *every other*
  mode — including other variants' faithful modes — is eligible training-only shaping
  (2048 practice: train on `sum`, want max tile; retailer: train dense `revenue`, care
  about `gap`). Terminal-class objectives are the canonical customers of shaping —
  their faithful modes are sparse by nature (`shaped_round` exists because `neg_oc` is
  terminal).
- **Normalized objectives are distinct family members:** retailer's `gap`
  (`100·total/opt − 100`) reweights episodes by 1/opt → a genuinely different optimal
  policy than `revenue` whenever `opt` varies per episode; `regret` ≡ `revenue` iff the
  benchmark is decision-independent. A real trained domain whose "which problem did we
  solve?" currently lives in a default argument (`reward_mode: str = "gap"`) — the
  motivating exhibit for explicit declaration. (Retailer also computes gap from
  gym-side accumulators, predating spec §6.4's rule that `total_revenue`/`revenue_opt`
  are `info` fields — under the new discipline its terminal modes become `expr` over
  terminal info like any other.)

### 14.5 Implementation batch — landing sites (deferred)

Additive migration, seed-scheme-v2 pattern: every pre-existing IR validates unchanged;
schema + spec + sample + examples move together; coordinated downstream check per
CLAUDE.md before release. Do **not** implement piecemeal — half-migrated visibility
semantics are worse than the honest 1-bit status quo.

| item | landing site |
|---|---|
| three-way visibility tags + timing; reward visibility; `Confirmable`-wrapped | `mdp_ir/schema.py` + MDP_IR_SAMPLE §5.2 |
| allowlist semantics replacing the latent blocklist; obs ⊆ filtration validator | schema validators + spec §5.2 |
| filtered-view gym construction | spec §7 |
| info-completeness obligation; terminal components in terminal `info` | spec §6.4 |
| action soundness check + coverage declaration on leaderboard | spec §7 + leaderboard conventions |
| `mdp.objective` → variants list; timing class; `per_step_components` rename/extension | schema + sample |
| scenario objective field + guardrail validators (never sampled; invisible to dynamics; singleton default) | schema validators + spec §5 |
| faithful-mode linkage (`objective_ref`); eval/tuning hard-wired to faithful mode (topk_id argparse pattern, generalized) | schema + spec §8 / generated eval scripts |
| target-swap trajectory-invariance test | `mdp_conformance` |
| interview additions: visibility elicitation; objective disambiguation (ambiguity / plurality / trade-off) | SKILL.md Phase A |

**Sequencing:** couples to roadmap step 1 (the skill split) — the tags and objective
interview touch `formalize`; the filtered view and faithful-mode eval touch `build`
codegen; and the gates are exactly what makes gym-layer escalation (§4) safe headless.
The escalation framework itself stays out of the spec — the spec describes domains,
not the process that searches over them.

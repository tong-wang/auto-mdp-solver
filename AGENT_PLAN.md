# Agentization Plan — from skill to standalone agent

*Design record, 2026-07-23. Captures the decisions, architecture, and roadmap for
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
| `solve` (Stages 3–4) | IR + domain + run-plan | leaderboard row + trained artifact | conformance 11/11, differential MATCH |
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
pause becomes a policy flag).

## 6. Agent architecture — mode-2 central orchestrator

The fan-out boundary is **inside** Phase B, between Stage 2 and Stage 3
(already documented in Stage 0): **build once, fan out solve.** Stages 1–2 are per-IR
and shared; Stages 3–4 are per-target/grid-cell.

**Decision: mode 2 — one central orchestrator brain over LLM-free workers.** Not mode 1
(a full agent per fan-out cell).

- Reasoning is concentrated at the **seams** (build, gates, collect, repair), not spread
  through the compute. The per-cell train/eval between seams is mechanical.
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
- **Atlas** scheduler assumed Slurm — confirm the scheduler and `ssh` submission path.
- **Subscription/SDK current caps** and whether the once-planned separate SDK credit pool
  is live (it was **paused** as of 2026-06); figures drift — check the console.
- **ToS for unattended and *redistributed* subscription auth** — is an end user
  authenticating via subscription through a redistributed SDK launcher permitted, or must
  downstream users BYO-API-key? Confirm with Anthropic support before advertising
  subscription auth to downstream users.
- **Managed Agents GPU / wall-clock limits** — moot if self-hosting CPU boxes.

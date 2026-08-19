# Solve Levels — L0/L1/L2+, the L1 derivation table, and the tuning contract

*Design record, 2026-07-27. Distilled from the vanilla-PPO-configuration
discussion (triggered by the MAB case in `rl_test/mab` underperforming on the
fixed recipe: PPO 671 vs UCB1 1444). Companion to `AGENT_PLAN.md` ("solve =
backbone + escalation") — this doc pins the level naming, the L1 derivation
table, the derivation↔tuning consistency contract, the defects found during
the audit, and the routing of each conclusion to its final home (§5). Until
the routed edits land, this file is the source of truth; after that, the
spec/harness are authoritative and this file is history.*

## 1. The solve ladder: L0 / L1 / L2+

| level | name | definition |
|---|---|---|
| **L0** | faithful-defaults PPO | library defaults + problem-forced settings — the right problem, zero solver judgment |
| **L1** | configured PPO | the derivation table (§2) applied — everything decidable from the IR *before* the first training run |
| **L2+** | escalations | anything conditioned on observed training results; level = escalation round, tagged with the layer(s) opened: `L2(hp)`, `L2(gym)`, `L2(arch)`, `L3(hp+arch)`, … |

- **L0 — faithful-defaults PPO.** Boundary test: *anything that changes
  **which** problem is being solved is problem-side — L0 inherits it; anything
  that changes **how well** the solver climbs is solver-side — L0 takes the
  library default.* Problem-side (inherited): **γ = β**; the algorithm class
  the IR's validators force (maskable/recurrent — plain PPO is invalid
  there); the env as built; the shared eval protocol; and one declared house
  constant, the 2M-step budget (SB3 has no default budget). Solver-side
  (library defaults; pin the SB3 version in args.txt): **no VecNormalize**,
  constant LR 3e-4, constant clip 0.2, `ent_coef` 0.0, batch 64, `n_steps`
  2048, `target_kl` None, MLP (64,64), 1 env, terminal checkpoint. In one
  line: `PPO("MlpPolicy", env, gamma=β, seed=s).learn(2_000_000)` — anything
  that cannot fit in that line is L1+. L0 may be awful on some domains (raw
  obs scales, no schedules): a saturated Δ(L1−L0) reads "defaults fail here;
  configuration was necessary", a strong L0 reads "RL-with-zero-judgment
  suffices". Only correctness fixes (§4.1) apply at every level. The pre-plan
  **house recipe** (schedules + VecNormalize + target_kl + γ 0.999 + 2M) is
  *not* a ladder level — keep it, clearly labeled, as an optional leaderboard
  reference row for continuity with results banked before this plan.
- **L1 — configured PPO.** Single training config, rule-derived from the IR;
  the **mandatory backbone** in the AGENT_PLAN sense — the must-run that fans
  out. Includes the within-run selection protocol (§2, "model selection").
- **L2+ — escalations.** The three orthogonal layers (hp-tuning, gym-shaping,
  arch-design) per AGENT_PLAN, with their existing asymmetries (hp/arch
  gate-free; gym carries its own gates; cheap-and-safe-first when headless).
  Budget extension ("still climbing at ceiling") is the cheapest L2 move.

**Invariant: level ≥ L2 ⟺ more than one training configuration was tried.**
Replicate seeds are copies, not search. Within-run model selection (plateau
stop, best-checkpoint, tail weight-averaging) happens inside L1's single run
and is not tuning. A tuned result is always an escalation.

**Diagnosis rules** (level language; supersedes none — restates AGENT_PLAN):

- L1 ≤ random → suspect the build (bug), don't escalate.
- L1 < L0 → the derivation misfired — lint on the brain and on the table.
- L1 competitive vs baselines → done at L1 (stopping criterion unchanged:
  competitive-vs-baselines, or escalation-budget exhaustion).
- Gap → open L2 per playbook priors.

**Conventions.**

- L0 runs by default (usually the cheapest run on the board; it is the ruler
  for agent contribution). Leaderboard row "PPO L0 (faithful defaults)". Reporting
  only — never a gate; `examples/` MANIFEST gates never depend on it.
- Seeds: 1 training seed exploratory; **3 seeds on both L0 and L1 for
  evidence-grade claims** (deltas must clear retrain spread; eval SE measures
  a frozen policy, not what a retrain would give — dynamic_pricing's
  17.34-vs-25.95 intensity spread is the in-repo demonstration).
- Ledger: every run's address carries its level tag
  (no-run-without-an-address; grammar addition routed to
  `ESCALATION_LOG_GUIDE.md`).
- **Eval tiers.** A ladder run *exists* only once scored on the reporting
  protocol — the ledger entry closes with the eval TSV, for L0 exactly as for
  L1/L2; one protocol, one seed set, shared with all baselines. Three tiers:
  **smoke** (train-script tail, 50 episodes, sanity only, never quoted);
  **selection** (EvalCallback during L1 training, ~256–512 CRN seeds,
  disjoint from reporting; L0 has none — terminal checkpoint by definition);
  **reporting** (`{domain}_ppo_eval.py`, **2048 seeds default, 8192
  evidence-grade**). Sizing rule: from a ~500-episode pilot SD σ,
  N ≈ (2σ/ε)² for the smallest delta ε to resolve — and make comparative
  claims on **per-seed paired differences** (CRN cancels instance luck; the
  paired SE is typically several × smaller than the naive SE). Cost scales
  with N × T̄ (MAB at 8192 = 8.2M env steps > the 2M training budget) —
  smallest power of two that meets precision, not 8192 by reflex. At
  evidence-grade, each of the 3 replicates gets the full reporting eval;
  retrain spread and eval SE are reported as separate uncertainties.
- **Attribution chain:** Δ(L1−L0) = value of the configuration layer — the
  derivation table *plus* the house training machinery (normalization,
  schedules, selection), now honestly counted;
  Δ(L2ᵢ−prev) = value of each escalation round. Per-case deltas, with which
  derivation rows fired, accrete into the escalation playbook.

## 2. The L1 derivation table

Each row: read the signal from the IR / problem context → set the knob → log
the decision with a one-line rationale (the IR's `obs_normalization`
decision+rationale pattern, generalized). Rows are **forced moves** — hard
rules, not judgment. Judgment lives in L2.

| knob | signal | rule |
|---|---|---|
| **β** (problem discount) | objective in the problem text (time value of money, continuation probability) | new IR field `objective.discount_factor`, β ∈ (0,1], default 1.0. Part of the *objective*, never tuned. Eval and **all baselines** score J = E[Σ βᵗ rₜ]. Never pre-discount env rewards (breaks stationarity + the differential contract); β is implemented solver-side as γ and eval-side in the metric. |
| **γ** | β, horizon | **γ = β.** Undiscounted indefinite horizon: γ = 1 − 1/T̄. γ < β only as an explicit, logged L2(hp) bias-variance move; **γ > β never** (more bias *and* more variance). |
| `gae_lambda` | delay of action consequences | 0.95 default; raise toward 0.98+ when consequences materialize ≫ 1/(1−λ) steps out (exploration value, long pipelines). |
| `n_envs`, `n_steps` | T̄ | rollout = n_steps × n_envs ≥ max(2048 transitions, **10 episodes**). n_envs = 4 structural default, never tuned; satisfy the episode floor by raising n_envs before inflating n_steps. Prereq: §4.1. |
| `batch_size` | rollout | rollout/32 … rollout/8, power of two. |
| LR schedule | noise regime (uncertainty block: does exogenous noise dominate reward variance?) | noise-dominated (typical ops domain): **1e-4 → 1e-5**. Near-deterministic, dense reward: 3e-4 → 3e-5. One degree of freedom: `lr_final = lr_init/10`. |
| clip schedule | — | 0.2 → 0.05; one DOF: `clip_final = clip_init/4`. |
| `ent_coef` | stochasticity/exploration need | 0.005 ops default; 0.01+ where premature determinism is a known hazard (bandit-like exploration, sparse success). |
| `n_epochs`, `target_kl` | — | 10 / 0.02 fixed. target_kl is the safety valve — never tuned (it masks LR pathology; watch approx_kl truncations as the LR-too-high signal). |
| `norm_obs` | spec §8.3 (unchanged) | heterogeneous stationary scales → on; drifting/accumulator obs → off, prefer sufficient-statistic obs. |
| `norm_reward` | — | on; **pass `gamma=γ` to VecNormalize** (fixes the silent internal 0.99, §4.2). |
| `net_arch` | obs dim; obs structure | (64,64) for obs dim ≤ ~32 (evidence: dynamic_pricing's 25-trial winning region); scale first hidden layer to ~2–4× obs dim above. Structured obs (set/permutation, grid, sequence) is never a width problem → predicted-escalation note. **Boundary: `net_arch` list = hp layer; custom feature extractor = arch layer.** |
| budget | T̄ | ceiling = **20k–50k episodes × T̄** steps AND ≥ ~300 updates (schedules need room to anneal). Plateau early-stop on the *eval-callback curve* (not the rollout curve). Ceiling hit while still climbing → first L2 move = extend. |
| model selection | — | EvalCallback on a fixed CRN selection seed set (**disjoint** from the reporting protocol's seeds — winner's curse), best-model saving, `sync_envs_normalization` before each eval. Post-train: weight-average the last ~5 checkpoints (tail of the LR anneal), validate vs best single checkpoint on the selection set, ship the winner. **The terminal checkpoint is never the deliverable.** |
| predicted escalations | structure signals | recorded at L1 time as priors for the brain, never preempted: exchangeable entities → L2(arch) shared encoder; obs provably not a sufficient statistic → L2(gym); etc. |

**Vec-env contract** (prerequisite rows): all env access through the VecEnv
API — **never `.envs`**; zero-arg factories; rank-suffixed per-env log
filenames. `DummyVecEnv` default; `SubprocVecEnv` only on *measured* env-step
cost ≳ 1 ms (a one-argument swap under the contract). Eval stays 1-env
(spec §9.5) at every level.

Display/monitoring: `stats_window_size ≥ 500` for high instance-variance
domains (the default 100-episode rolling mean swings even under a static
policy; zero cost).

## 3. Derivation ↔ tuning contract ("center and radius")

**The derivation table sets the center; the tuning space provides the radius —
and must contain the derived point.**

1. **Containment:** every derived value lies inside its search range — γ's
   space upper bound becomes β *inclusive* (today 1.0 is structurally
   unreachable: `1 − loguniform(1e-4, 0.05)`).
2. **Forced-move locks:** knobs fixed by hard rule leave `tunable` (γ=β,
   n_envs, β itself). Rediscovering a theorem costs trials.
3. **One-DOF schedules:** tune `lr_init` only, `lr_final = init/10` (kills the
   inversion bug §4.5); clip likewise (resolves §4.3).
4. **Domain-aware bounds at runtime:** n_steps lower bound
   `≥ 10·T̄/n_envs` intersected at trial time (same pattern as the existing
   flag intersection); batch cap becomes `min(batch, n_steps × n_envs)` (§4.6).
5. **Tiers** (priority = membership, not weights — Optuna has no per-param
   priority; at 25 trials × 10 dims TPE ≈ random search). Membership follows
   one question: how much does the L1 derivation already know?
   **core** = {learning_rate, net_arch *width*, n_steps, ent_coef, gae_lambda}
   — the knobs whose derived value is a real guess; always tuned.
   **breadth** = {net_arch *depth*, n_epochs, batch_size, vf_coef, gamma} —
   what the derivation does not produce at all, or (gamma) is bounded by the
   problem; trials ≥ ~40 or second stage.
   **frozen** = {clip, max_grad_norm} + locks — held at the derived value
   unless explicitly opened. Driver flag: `--knobs core|breadth|all` /
   `--fix KEY`.
   *Revised 2026-08-19 (#56 follow-up):* `gae_lambda` moved core ← breadth on
   the evidence of two campaigns bracketing its optimum an order of magnitude
   apart (mab #E35, game2048 #E34/#E38) against a spec rule that calls it
   per-instance and non-transferable; `gamma` moved breadth ← frozen, since the
   sampler cannot exceed β and opening it buys only a logged bias-variance
   move; `net_arch` split into an ordered width axis (core) and depth axis
   (breadth), which is what makes a 4×3 shape grid searchable at core budget.
6. **Warm start:** enqueue trial 0 = L1's config (`study.enqueue_trial`) —
   the derived center; the study directly measures "what does tuning add over
   L1". (L0 is generally not expressible through the train script's flags —
   no obs-norm toggle, schedules always on — and its comparison lives at the
   ladder level, at full budget, not inside the study.)
7. **Trial budget:** ~10% of the L1 ceiling, fixed, no early-stop within
   trials (comparability); the winner re-runs at full ceiling with the L1
   selection protocol (dynamic_pricing's 200k-trials/full-final was this rule,
   informally).
8. **Report:** append fANOVA `get_param_importances` to the tuning report;
   per-case importances accrete into the playbook — tier membership becomes a
   measured prior per problem class.
9. **Lint:** warn when a `PPO_KNOBS` entry matches no exposed dest — would
   have caught §4.3 years early.

## 4. Defects found during the audit (fix regardless of the rest)

1. **Episode-seed provenance** — template + all frozen example gyms (+ MAB
   downstream): on `reset(seed=None)` the episode seed is drawn from **global**
   `np.random` (`np.random.randint`). Under SubprocVecEnv+fork, workers
   inherit identical global state → identical episode-seed streams from
   episode 2 on → perfectly correlated "parallel" envs; under DummyVecEnv it
   is reset-order-fragile. Fix (one line):
   `self._episode_seed = int(self.np_random.integers(0, 2_147_483_647))` —
   gymnasium's `super().reset(seed=seed)` already maintains per-env
   `np_random` (SB3 seeds it `training_seed + rank`). Explicit-seed resets
   unchanged (eval/benchmark instance-pairing contract). Everything below
   `episode_seed` is already counter-keyed (v2 seed keys) and untouched.
   **Prerequisite for n_envs > 1.**
2. **VecNormalize discounting mismatch:** its return-normalization gamma
   defaults to 0.99; no script passes it → normalizes against a different
   discount than training. Template: `VecNormalize(..., gamma=args.gamma)`.
3. **Clip never tuned:** space dest `clip_range` matches no script dest
   (`clip_init`/`clip_final`) → silently dropped from every study to date.
   Resolved by §3.3 + §3.9.
4. **γ = 1.0 unreachable** in the tuning space. Resolved by §3.1.
5. **LR schedule inversion:** tuned `learning_rate` (init) below the frozen
   `lr_final` default (3e-5) yields a *rising* schedule. Resolved by §3.3.
6. **Batch cap assumes single env:** `min(batch, n_steps)`. Resolved by §3.4.

## 5. Routing

| item | destination | kind |
|---|---|---|
| level definitions, diagnosis rules, L0 conventions, attribution chain | `SKILL.md` Phase B (+ pointer added in `AGENT_PLAN.md` §solve) | plugin doc |
| L1 derivation table + vec-env contract + selection protocol | `MDP_PROJECT_SPEC.md`, new subsection under §8 (+ §8.3 VecNormalize-gamma edit; §9 eval scores with β; RNG rules gain the episode-seed idiom) | plugin doc (spec) |
| ledger addresses gain a level tag | `ESCALATION_LOG_GUIDE.md` (on trial) | root doc |
| β field `objective.discount_factor` | `harness/mdp_ir/schema.py` (+ `MDP_IR_SAMPLE.md`, Phase A interview question in `SKILL.md`) | **schema — additive** (default 1.0); coordinate downstream repos before release |
| episode-seed fix | spec RNG rules + the frozen example gyms (spec-required edit — allowed under the freeze rule; re-run manifest gates) + train-script template | plugin doc + frozen examples |
| `--n-envs`, required `--net_arch` dest, EvalCallback machinery, `stats_window_size` | spec §8 script conventions + example train scripts | plugin doc + examples |
| tiers, locks, one-DOF schedules, warm-start, runtime bounds, fANOVA report, unmatched-knob lint | `harness/mdp_tuning/` (`spaces.py`, `driver.py`) | harness code |

Sequencing: (a) this record lands; (b) `mdp_tuning` changes + tests;
(c) spec + template + example edits as **one plugin release** (single
`plugin.json` bump + tag, manifest gates re-run); (d) schema β field with the
downstream check; (e) the MAB pilot (below) exercises the ladder end to end
and seeds the playbook with the first derivation-delta entry.

## 6. Pilot: MAB (downstream, `rl_test/mab`)

- **House-recipe reference banked (not L0):** 671 (stats obs) / 659 (bayes
  obs) vs UCB1 1444, Thompson 1465, greedy ~1024, oracle 1535 (8192-seed
  protocol) — run with schedules + VecNormalize + γ 0.999 + 2M.
- **True L0 still to run** (cheap): `PPO("MlpPolicy", MabEnv, gamma=1.0,
  seed=1).learn(2M)`, no VecNormalize. Expect the stats obs mode's raw scales
  (~10³) to hurt badly without normalization — itself informative.
- **L1 per table:** γ = 1.0 (β = 1, T = 1000); n_envs 4 × n_steps 2560
  (rollout 10,240 ≈ 10 episodes); batch 512; LR 1e-4→1e-5; λ 0.98
  (exploration payoff is delayed); ent_coef 0.01; net (64,64) (obs dim 21);
  `VecNormalize(gamma=1.0)`; ceiling 20M–50M steps with plateau stop +
  selection protocol. Requires the §4.1 fix first.
- **Predicted L2(arch):** shared per-arm encoder — 10 exchangeable arms into a
  flat MLP, and PPO scoring *below greedy-on-posterior-mean* (a trivial
  function of the obs) is the flat-MLP symmetry signature.

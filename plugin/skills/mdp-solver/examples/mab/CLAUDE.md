# CLAUDE.md — `mab/` (standard stochastic multi-armed bandit)

Domain-local operating brief. **Pointer-first: this file says where to look
and what will bite you, never what the answer is.** Anything that changes as
the campaign progresses lives in the docs below and is linked, not copied.

K arms, T rounds. Nature draws each arm's mean payout fresh per episode (a
world latent the agent never sees); the agent pulls one arm per round, observes
only that arm's payout, and maximizes the undiscounted total over a finite
horizon. **Bernoulli and Gaussian are SEPARATE LEADERBOARDS** — payout scales
differ, so their scores must never be compared; likewise each `gauss_K{K}_T{T}`
instance is its own leaderboard, since `reward_mean` is not comparable across
horizons at all (regret nearly is — see Traps).

## This domain is GENERATED — the skill is authoritative

`mab/` was produced by the **auto-mdp-solver** skill from
`mab_schema.json`. It is not a hand-written project, and it must not
drift into one.

**Before changing anything here, read the skill docs — they are authoritative
over this file, over the code, and over local habit:**

| doc | governs |
|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG/seed tree, script + eval conventions (§1–§14) |
| `SKILL.md` | the pipeline stages, their gates, and what each stage must emit |
| `ESCALATION_LOG_GUIDE.md` | the campaign log (§MAP, §FRAME-CHANGELOG, §IR-CHANGELOG, §LEDGER) and the §10 playbook digest |

Consequences, each already paid for somewhere:

- **Check the spec before inventing a mechanism.** The pipeline usually has
  one already, and its version composes with the gates. (Paid here: varying K
  and T looked like it needed a hand-written side registry; §5.4 instances vs
  §5.6 grids already defined it — #E32.)
- **Structural changes go through the IR first**, then the code, then the
  gates — never code-first. `mab_schema.json` is the source of truth;
  `mab_scenarios.py` and friends materialize it.
  Observation/architecture levers are NOT IR changes — they live in the gym /
  train script with their own executable gate; only the *problem* goes
  through the schema.
- **Re-run the gates after any IR change and after any solver bump** — the
  IR depends on host behavior, not just on this folder. A moved fingerprint
  owes a §IR-CHANGELOG entry (guide §5, tripwire 2). (Paid here as **F4**: a
  structural move went unlogged, then got misattributed to the instances that
  shipped in the same commit rather than the bounds edit that caused it.)
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in the log, and as an upstream proposal
  (`UPSTREAM_PROPOSAL_*.md`, filed via the mdp-propose skill) if the spec
  should change. (Live example: `--patience 0`.)

The repo-root `CLAUDE.md` carries the cross-domain rules.

## Where the answers are

| doc | what it is |
|---|---|
| `README.md` | layout, usage commands, leaderboards |
| `ESCALATION.md` | the campaign: §MAP, changelogs, numbered findings (#E…) with verdicts |
| `INTERPRET.md` | policy readback (spec §14) — what the trained nets actually do |
| `PLAYBOOK.md` | the guide-§10 case-close digest (exists once the campaign closes) |
| `mab_schema.json` | **the IR — authoritative** for the problem definition |
| `mab.restatement.md` | the frozen Phase-A restatement |
| `UPSTREAM_PROPOSAL_*.md` | drafts against the solver spec, not local decisions |

**No `UPSTREAM_PROPOSAL_*.md` are open here.** All ten this campaign filed are
dispositioned; a filed draft is deleted, because the issue carries the same
body with its outcome attached and cannot drift from what was argued.
`ESCALATION.md` §UPSTREAM records where each went (issues #3–#12).

Round plans are deliberately **not** in this table: they live in `scratch/`
and are deleted once their findings are in `ESCALATION.md` (see File
hygiene).

Solver provenance — **two versions, two meanings**
(github.com/tong-wang/auto-mdp-solver):

- **Built and gated at: v0.7.0** (`2e50c60`). Every number in this folder was
  produced under that tag. This line never moves; moving it would claim the
  results were re-measured.
- **Conformance maintained through: v0.9.6.** Declarations, drawing
  conventions and gate compatibility have been carried forward **without
  re-running anything** — the `benchmarks` block (including the column-sourced
  oracle), the `research_questions` stances, the §8.4 provenance emit, the §MAP
  retype to the `cases`/`means`/`designs` taxonomy, and the `mdp.model` theory
  layer plus the grouped `mdp` file layout (**F6**). Each is logged in
  `ESCALATION.md` §FRAME-CHANGELOG, and F6 additionally in §IR-CHANGELOG
  because declaring the model layer moves the freeze token. **No verdict, mark
  or number moved.**

The two lines differing is the intended state, not drift (`cases/README.md`,
"After a case is merged"). Verify before relying on either.

## Commands

Run from `mab/` unless noted. Always pin threads for training — torch
oversubscribes.

```bash
# gates (conformance/laws/differential run from the PARENT of mab/)
python -m mdp_ir mab_schema.json
python -m mdp_conformance mab
python -m mdp_ir.laws mab
python -m mdp_ir.differential mab/mab_schema.json --episodes 40 --all-instances
pytest mab_test.py

# train / eval  (the record eval is STOCHASTIC — see Traps)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python mab_ppo_train.py -s gauss_K10_T1000 -o bayes
python mab_ppo_eval.py -s gauss_K10_T1000 --stochastic \
  --model-path results/gauss_K10_T1000/<run>/gauss_K10_T1000_ppo.zip
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `mab/` only if it is (a) spec-§1 layout, (b) an
implementation a finding cites and someone must re-run to reproduce it
(probes, extra gates), or (c) a campaign document (`README`, `ESCALATION`,
`INTERPRET`, `PLAYBOOK`, `UPSTREAM_PROPOSAL_*`). **Everything else goes to
`scratch/` (gitignored): launchers, monitors, one-off checks, throwaway
analysis. All output goes to `results/` (gitignored).**

**Round plans are never tracked.** A `*_PLAN.md` lives in `scratch/` while
it is being drafted AND while it is being executed; findings go into
`ESCALATION.md` and `README.md` **as they land**, and the plan is deleted
once written up. If a plan is the only place a result exists, that is a bug
in `ESCALATION.md`. **The probe a plan drives is the opposite — it stays,
permanently**: an escalation entry cites numbers that only exist if the code
behind them can be re-run. A probe is written in `mab/` from the start
(it imports its siblings) and **committed in the same commit as the
escalation entry that cites it** — `git log ESCALATION.md` then shows each
finding beside the code that produced it. Design rationale that must outlive
the round has two tracked homes, neither of them the plan: the escalation
entry and the probe's module docstring.

Check with `git status --short mab/`: untracked files should be rare
and deliberate.

## Traps

Seeded by the pipeline; **every trap the campaign pays for is added here the
same day**, citing the finding (#E…) that paid for it.

- **The IR declares vectors; it never enumerates them** (spec §5.0,
  `mdp_conformance schema.no_enumeration`). A pipeline is one state variable
  with `length: "n_arms"` and a quantified update, never `x_now`/`x_next` or
  `x1`/`x2`; a constant that names a width must actually be *read* by the
  rendering, or it is decoration and changing it changes nothing. This is not
  style: the schema is the first artifact the user reads back as the statement
  of their problem, and an enumerated one reads as a transcript of one case
  rather than a formalization. Before adding a sibling, ask whether it is
  element `k` of something.
- **The `mdp` block is in the GROUPED layout** (`model` / `design` /
  `rendering`, spec §5.0, adopted as **F6**). It is a *file layout only* — in
  memory the block is flat, both layouts load, and every hash is computed after
  flattening — but **anything reading the raw JSON by path must flatten first**,
  because `["mdp"]["scenario"]` exists only in the flat layout. Use
  `mdp_ir.schema.ungroup_mdp`, **not** `load_ir`, for any claim about the
  *declared* document: `load_ir` resolves the catalog and silently drops every
  instance that re-selects a slot, which here means `bernoulli` — one of the two
  leaderboards (34 declared instances, 33 after `load_ir`). `mab_test.py`'s
  `_raw_flat()` is the only raw reader and carries that reasoning;
  `mab_ppo_train.py` goes through `load_ir` and needed no change. `--regroup`
  warns about neither, and regrouping before the reader was fixed would have
  taken this file from 69 tests to 0 at collection (issue #42).
- **`mdp.model` is the theory, and design/rendering choices do not go in it**
  (§5.0). The model says `n_arms >= 2` and `horizon_T >= 1` with **no upper
  limit**; K=10/T=1000 are *design* values, and `arm_max` plus the
  state-variable `bounds` are *rendering* widths. Keep it that way — writing a
  swept maximum into the model layer is how a sweep ceiling becomes a capacity
  limit nobody chose, which is the failure §5.0 exists to prevent and the one
  **F4** already paid for once from the other direction. `arm_means` carries no
  `stochastic` flag on purpose (see its `source`); benchmark and tractability
  vocabulary stays out of the layer entirely — `model.boundary` WARNs on it.
- **The record eval is `--stochastic` here** — policy entropy *is* this
  domain's exploration mechanism, and an argmax eval once re-pruned an entire
  encoding whose whole value lived in its sampling (#E30). The other mode is
  a separate, labelled figure and never a record.
- **Selection and terminal artifacts are different networks**
  (`{scenario}_ppo.zip` vs `_ppo_final.zip`, spec §8.4) — say which one a
  number came from (#E28).
- **Compute sites/venues are cited by alias, never hostname** — venue config
  lives at the repo root; a venue change is a confound to record, not a
  detail. Here the torch version differs by venue, so every cross-venue
  comparison in the log carries that as a stated confound.
- **Never use `param`, `params`, or `param_*`** anywhere (conformance
  fails).
- **`reward_mean` is not comparable across seed blocks or horizons;
  `regret_mean` nearly is** — paired CRN differences are the tight statistic.
  A 20–39% regret effect reads as ~1% in reward and was nearly discarded as
  noise (#E28, #E35).
- **Single-seed contrasts below ~50 points are not established** — training
  seed sd is ~2.7× the eval SE, and most of this campaign's early z-values
  used the wrong denominator until #E21 corrected them.
- **K and T are axis-tagged scenario constants** (`n_arms`, `arm_max`,
  `horizon_T`) since #E32. Decision/action bounds resolve constant *names*,
  not expressions, so **`arm_max` must equal `n_arms − 1` in every instance**
  — `mab_test.py` asserts it. State-variable `bounds` cannot name a constant
  at all, so they are sized for the union of instances and must be widened by
  hand for each larger cell (**F4**).
- **Plateau early-stopping is OFF** (`--patience` defaults to 0) — it fires
  before the true peak and forfeits most of the +42.90 that selecting a
  checkpoint at all is worth (#E33). This was a deliberate deviation when the
  campaign ran; **v0.7.0's §8.6 now mandates it** ("a training run runs to its
  budget — no early stopping"), partly on this campaign's evidence, so the
  deviation became the rule.
- **The selection machinery here PREDATES v0.7.0's design and is not
  conformant to it.** This campaign used a live in-training
  `SelectionEvalCallback`; §8.6/§9.7 now specify a post-hoc three-layer
  screen — `CheckpointCallback` only, then rank ~20 checkpoints on a ~2048-seed
  block, confirm the top-k on the protocol block — with "no `EvalCallback`, no
  live selection env". The spec cites **#E33 and #E36 V4** for that change:
  the callback that selected a generalist's checkpoint on one wrong cell is
  exactly this one. Every number here was produced under the old design, which
  is why the code still carries it; a re-run should use the new path.
  `mab_selection_probe.py` is the post-hoc screen this campaign ran by hand.
- **`norm_obs` must stay OFF for any equivariant policy** — VecNormalize
  normalizes each observation *dimension* independently, so arm slots acquire
  different running statistics and the permutation equivariance that is the
  architecture's whole point silently breaks (#E14).
- **`gae_lambda` is per-instance, not a transferable constant** — re-derive
  it whenever mean episode length moves, using coverage (`1/(1-λ)` over T) as
  the lens. On this domain the optima sit at ~8–17% and training failed
  outright below ~0.8% (#E35) — but that band is mab's, not a rule: a second
  campaign measured optima at ~2–4% with a *lower* λ winning, so the mechanism
  transfers and the number does not (upstream issue #4).

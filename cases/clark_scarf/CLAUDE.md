# CLAUDE.md — `clark_scarf/` (ordinal board)

Domain-local operating brief. **Pointer-first: this file says where to look and
what will bite you, never what the answer is.**

Opened 2026-09-14 with an **empty result history by construction** — the code,
the IR and the gates are inherited; no number is. An earlier campaign on this
same domain ran downstream and is not cited from here. Moving a number across
is a deliberate act that re-earns it at this pin, never a quotation.

## This domain is GENERATED — the skill is authoritative

Produced by the **auto-mdp-solver** skill from `clark_scarf_schema.json`. It is
not a hand-written project and must not drift into one.

| doc | governs |
|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG/seed tree, script + eval conventions |
| `SKILL.md` | the pipeline stages, their gates, what each stage must emit |
| `ESCALATION_LOG_GUIDE.md` | the campaign log (§MAP, changelogs, §CONFIG-REGISTRY, §LEDGER) |

- **Check the spec before inventing a mechanism.** The pipeline usually has one.
- **Structural changes go through the IR first**, then the code, then the gates
  — never code-first. Observation/architecture levers are NOT IR changes; they
  live in the gym / train script with their own executable gate.
- **Re-run the gates after any IR change and after any solver bump.** A moved
  fingerprint owes a §IR-CHANGELOG entry.
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in the log, and as an upstream proposal if the spec should change.

## Where the answers are

| doc | what it is |
|---|---|
| `ESCALATION.md` | the campaign: §MAP, changelogs, §CONFIG-REGISTRY, numbered findings |
| `README.md` | layout, commands, the four benchmark bars |
| `clark_scarf_schema.json` | **the IR — authoritative** for the problem definition |
| `clark_scarf.restatement.md` | the frozen Phase-A restatement (model-layer; travels with the IR) |
| `clark_scarf.scenarios.md` | the scenario set, its axes, why each value was chosen |
| `clark_scarf_configs.py` | the §CONFIG-REGISTRY data half. Run it to print origins and resolve an address |
| `clark_scarf_policy.py` | the deployable policy (spec §12); its docstring is the observation contract |

Pinned solver: **`auto-mdp-solver==0.11.0`**, installed from PyPI. Verify with
`python -c "import importlib.metadata as m;
print(m.version('auto-mdp-solver'))"`, and check the `mdp_*` packages resolve
to site-packages rather than to a development checkout — a dev-tree import
gates a pinned domain against unpinned code. Read the skill docs from the
plugin cache matching that version; newer versions may sit beside it and are
not what this folder runs.

**Gated here 2026-09-24, at 0.11.0** — this stamp replaces the v0.10.12 one,
which was a measurement of different code:

| gate | result |
|---|---|
| `mdp_ir` validate | structural fingerprint `d2462f976e61`, 24 instances |
| conformance | **27/30 passed**, zero FAIL |
| laws | 8/9 (`mixture_equivalence` SKIPs, no mixtures declared) |
| differential | MATCH ×24 under `--all-instances` |
| `pytest` | 134 passed |

**Re-verified twice on 2026-09-24, because a version stamp is a claim about
code rather than about a number.** (i) Against **0.11.3**, the plugin release
installed at the time: identical, and checkably so — the five `mdp_*` packages
are byte-identical to 0.11.0 (`diff -r` empty for `mdp_ir`, `mdp_conformance`,
`mdp_gates`, `mdp_tuning`, `mdp_stage`). (ii) Against **upstream HEAD
(v0.11.6)**, which is what CI actually runs — it installs the harness from the
repo, not a released version — and where **ten harness `.py` files have
changed** since 0.11.0, `mdp_conformance/checks.py` and `mdp_ir/schema.py`
among them, one commit of which alters how `desc`/`note` participate in the
freeze token. Result there: `structural_fingerprint` **unmoved** at
`d2462f976e61`, conformance 26/30 zero FAIL, laws 8/9, differential MATCH x24.
The engine moved and the verdict did not.

The stamp above was first taken with `research.deliverables` FAILing — the
declared `confirm` stance owed the spec §14 readback and `INTERPRET.md` did not
exist. It was written the same day and the gate now passes. The remaining WARN
is `schema.no_enumeration` on the re-baked `h_install` vectors; the two SKIPs
are `scenario.samplers` and `grids.registry`, neither of which this domain
declares.

Fingerprints unmoved at this pin: `model 8692bea53c9d` / `mdp 9720fe8bbd74` /
`structural d2462f976e61`. The `dp` bar's **982.362986** was last replayed at
v0.10.12 and is **not** re-measured in the stamp above — say so rather than
implying a fresh replay.

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs here only if it is (a) spec-§1 layout, (b) an implementation
a finding cites and someone must re-run to reproduce it, or (c) a campaign
document. **Everything else goes to `scratch/` (gitignored): launchers,
monitors, one-off checks, throwaway analysis. All output goes to `results/`
(gitignored).** Round plans are never tracked; a probe is, and is committed in
the same commit as the entry that cites it.

## Traps

Every trap paid for here is added the same day, citing the finding. These are
inherited because they are about the MODEL and the CODE, not about any result.

- **Echelon STOCK is not echelon POSITION.** Holding is charged on echelon
  *stock*, where stock in transit **to** a level belongs to the echelon
  **above** it (Assumption 3). Echelon *position* `u = x_1 + w_1 + …` is a
  different quantity — what `f_n(u)` optimizes over and what a base-stock rule
  is stated on. Conflating them double-charges pipeline stock and biases the
  shipping incentive against moving stock downstream, which is the exact
  trade-off this domain measures. `echelon_stock()` and `echelon_position()`
  are deliberately separate functions; do not merge them.
- **`raw` and `echelon` must stay a pure change of coordinates.** They are
  informationally identical *only* because `echelon` sums stock and in-flight
  stock **separately**. A single running sum over their total is lossy at
  `leadtime = 2` and would silently turn the headline experiment into a test of
  information rather than representation.
  `test_raw_and_echelon_are_a_pure_change_of_coordinates` is the guard — never
  weaken it.
- **An action mode's decode may not contain reference-theory constructs.** A
  decode that computes echelon position from simulator state and ships
  `clip(y − u)` supplies the paper's aggregation and its policy class whatever
  the agent observes, so an arm using it can never support a structure-discovery
  claim. Audit any new action mode's decode before crowning anything on it.
- **The IR's `features` list is a CONTRACT, and no gate checks it** (spec §7).
  Every component the observation renders must derive from a quantity the mode's
  `features` names, in render order. `test_declared_features_are_exactly_what_
  the_gym_renders` is the local guard, because upstream ships none. Add a
  feature to the gym and you add a line to the IR in the same commit; `gym` is
  outside every fingerprint, so it costs nothing to keep true.
- **A head's hyperparameters must live INSIDE the model.** A class attribute or
  a module global does not travel: `PPO.load` rebuilds the head with whatever
  default is in force and scores a policy that was never trained. Pass them via
  `policy_kwargs` and carry them in `_get_constructor_parameters`. Guarded for
  the Beta, Gamma and ordinal heads.
- **The ordinal head's init is a PRIOR, not a neutral start.** `P(ship = 0) =
  0.5` against a flat categorical's 0.0244, with only the positive branch
  near-flat. Defensible on a domain whose optimum ships nothing most periods,
  but it is a real prior and it is pinned by test rather than inherited silently.
- **The observation box is a VALIDITY envelope, not a scale.** Sized to the
  reachable range so `contains()` holds under random actions. Normalization is
  VecNormalize's job. Do not "tighten" it — clipping into a smaller box destroys
  the invertibility above.
- **A sweep is not a domain, and a cap is not a model.** There is no lead-time
  cap: every `pipe_k` declares `length: "leadtime"`, so the width NAMES the
  constant. **A width that can name a constant must name it** — a literal there
  is a ceiling nobody declared.
- **A vector decision must be resolved PER INSTANCE, not once.** A width or
  bound taken from the *base* IR is a plausible number that is simply wrong for
  the instance being run. If you hand-build a decision dict, pass the instance.
- **`ClarkScarfScenario` must NOT be callable.** A self-returning `__call__`
  makes "did this resolve to a concrete scenario?" unanswerable and fails
  `scenario.samplers`.
- **The record eval is DETERMINISTIC here** — the policy is a shipment quantity
  with no exploration role at eval time, and the benchmark it is scored against
  is deterministic. A stochastic eval is a separate, labelled figure, never a record.
- **The DP row is not unconditionally exact** — it truncates the Poisson pmf at
  a high quantile while the simulator does not (mass < 1e-9). Say so wherever
  "% of optimal" is quoted.
- **A number from a run that stopped early is not a worse result — it is not a
  result.** `mdp_gates.completion` blocks the verdict before any comparison; a
  deliberate stop is declared with `--short-ok "<reason>"`. A killed run and a
  silently dead one leave identical files, which is why the declaration is the
  one input no artifact can carry.
- **A masked run would FAIL `run.provenance`.** The args log records the
  resolved `algo_class` and the IR declares `ppo` alone; `--mask` constructs
  `MaskablePPO`. Declare `rl.algos` in the IR before running a masked arm.
- **A different action mode is a different CELL, not an escalation.** The level
  is read off the knobs that moved among **non-design** knobs;
  `clark_scarf_configs.level()` is the implementation.
- **`--show-space` does not show the space a launch with the same flags will
  search.** It returns before `--train-arg` pins are parsed. Verify pins from
  trial 0's args log, not from the banner.
- **Every §8.6 derivation row needs a DEST, or it silently does not happen.**
  Check `python -m mdp_tuning clark_scarf --show-space` against §8.6's table
  before trusting any "L1 = the derivation" claim.
- **A pin bump can move a frozen fingerprint with no IR edit.** Re-gate before
  trusting any result after a bump, and never assume a moved token means someone
  edited the model.
- **Compute sites/venues are cited by alias, never hostname.**
- **Never use `param`, `params`, or `param_*`** anywhere (conformance fails).

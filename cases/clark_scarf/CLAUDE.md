# CLAUDE.md — `clark_scarf/` (serial multi-echelon inventory, Clark & Scarf 1960)

Domain-local operating brief. **Pointer-first: this file says where to look
and what will bite you, never what the answer is.** Anything that changes as
the campaign progresses lives in the docs below and is linked, not copied.

A single product moves down a serial chain of stocking points: level 1 is a
retailer facing Poisson demand, level N buys from an unlimited outside
supplier. Each period every link dispatches simultaneously (N numbers, one per
link), clipped to what its source installation holds; shipments take `leadtime`
periods to land and cannot be recalled. Unmet demand backlogs. Minimize
expected discounted cost (β = 0.95) over 50 periods — echelon holding up the
chain versus the retailer's shortage penalty.

**Every `(n_echelons, leadtime, p_short)` cell is a SEPARATE LEADERBOARD.**
Chain length and lead time change the observation and action width, so no
policy spans them and no score may be compared across cells. `p_short` changes
the cost scale, so those are not comparable either. The two *observation
modes* (`raw`, `echelon`) within one cell **are** comparable — that comparison
is the whole point of the campaign.

## This domain is GENERATED — the skill is authoritative

`clark_scarf/` was produced by the **auto-mdp-solver** skill from
`clark_scarf_schema.json`. It is not a hand-written project, and it must not
drift into one.

**Before changing anything here, read the skill docs — they are authoritative
over this file, over the code, and over local habit:**

| doc | governs |
|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG/seed tree, script + eval conventions (§1–§14) |
| `SKILL.md` | the pipeline stages, their gates, and what each stage must emit |
| `ESCALATION_LOG_GUIDE.md` | the campaign log (§MAP, §FRAME-CHANGELOG, §IR-CHANGELOG, §CONFIG-REGISTRY, §LEDGER) and the §10 playbook digest |

Consequences, each already paid for somewhere:

- **Check the spec before inventing a mechanism.** The pipeline usually has
  one already, and its version composes with the gates.
- **Structural changes go through the IR first**, then the code, then the
  gates — never code-first. `clark_scarf_schema.json` is the source of truth;
  `clark_scarf_scenarios.py` and friends materialize it.
  Observation/architecture levers are NOT IR changes — they live in the gym /
  train script with their own executable gate; only the *problem* goes
  through the schema.
- **Re-run the gates after any IR change and after any solver bump** — the
  IR depends on host behavior, not just on this folder. A moved fingerprint
  owes a §IR-CHANGELOG entry (guide §5, tripwire 2).
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in the log, and as an upstream proposal
  (`UPSTREAM_PROPOSAL_*.md`, filed via the mdp-propose skill) if the spec
  should change.

The repo-root `CLAUDE.md` carries the cross-domain rules.

## Where the answers are

| doc | what it is |
|---|---|
| `README.md` | layout, usage commands, leaderboards (benchmarks final; PPO pending Stage 4) |
| `ESCALATION.md` | the campaign: §MAP, changelogs, numbered findings (#E…) with verdicts |
| `ESCALATION.md` §UPSTREAM | the proposal round — eleven filed, eleven accepted, with issue links and dispositions. The drafts are deleted by design: the issue cannot drift from what was argued, a local copy can |
| `INTERPRET.md` | policy readback (spec §14) — the tier-2 *confirm* deliverable. The raw-trained crown recovers Clark & Scarf's base-stock rule with the paper's own critical numbers (#E8) |
| `clark_scarf_policy.py` | the **deployable** policy (spec §12): model + vecnorm + action transform behind `act(obs)`; its docstring is the observation contract |
| `PLAYBOOK.md` | the guide-§10 case-close digest: two lever entries (action encoding; pricing a representation lever at L2) and one frame move (verify the reference on a brute-forceable fixture), stated at trigger level. Process/spec lessons are NOT here — they went upstream as #18–#29, see ESCALATION §UPSTREAM |
| `clark_scarf_schema.json` | **the IR — authoritative** for the problem definition |
| `clark_scarf.restatement.md` | the frozen Phase-A restatement |
| `clark_scarf.scenarios.md` | the scenario set, its axes, and why each value was chosen |

Round plans are deliberately **not** in this table: they live in `scratch/`
and are deleted once their findings are in `ESCALATION.md` (see File
hygiene).

Built and last gated against **auto-mdp-solver v0.9.5**. Every gate figure
below is at that version; re-run them after a version bump before trusting a
new result — this case is the reason several v0.9.x fixes exist, and each was
found by a bump changing behaviour under it.

Frozen Phase-A fingerprints: `mdp = da62301e56b4`, `structural = 7138a8f1ce4e`
(**F9**, 2026-08-17: the last hardcoding removed — one `pipe` matrix
(`length: ["n_echelons", "leadtime"]`), one `ship` vector
(`dim: "n_echelons"`), `stock` unpadded, dynamics quantified, and
`N_LEVELS_MAX` deleted from code and IR. Both records replay to the digit.
F8 immediately before it declared `quantities[].stochastic` and grouped the
file `model`/`design`/`rendering`, moving `mdp` alone.)

Re-signed six times on 2026-08-17, one operator review round. The
model/design/implementation boundary is machine-readable in the schema's
`mdp.model`, gated by `model.boundary`, and now *visible in the file* via the
grouped layout. Every campaign number replays to the digit at every
re-signing — DP `982.362986`, crowned PPO `989.065774`. History: first freeze
`53d5d1f9cd5e`; F1 → `2171a651d328`; F2 moved `structural` on a pin bump;
**F3** generalized the pipeline from two named slots to a shift register
(`83588654745e`/`26ab0de96d58`); **F4** restored shipping/ordering costs `c_k`
and per-link lead times `L_k` to the model (`3fb17ad1eae9`/`31635a1de6fd`);
**F5** corrected a paper misquote (stationarity) in a candidate desc; **F6**
declared the model layer in-schema at the v0.9.0 pin, `structural` unmoved at
both; **F7** removed the lead-time cap by transposing the register to
per-installation vectors whose width *names* `leadtime`
(`20f732827637`/`2ff6e8344181`) — the first `structural` move from a rendering
change since F3; **F8** → the pair above. See §IR-CHANGELOG for each.

## Commands

Run from `clark_scarf/` unless noted. Always pin threads for training — torch
oversubscribes.

```bash
# gates (conformance/laws/differential run from the PARENT of clark_scarf/)
python -m mdp_ir clark_scarf/clark_scarf_schema.json
python -m mdp_conformance clark_scarf
python -m mdp_ir.laws clark_scarf
python -m mdp_ir.differential clark_scarf/clark_scarf_schema.json --episodes 40 --all-instances
pytest clark_scarf

# layer smoke tests (from clark_scarf/)
python clark_scarf_mdp.py          # fixed-policy episodes across chain lengths
python clark_scarf_gym.py          # random episodes, every obs mode + the coordinate-change check

# train / eval  (the record eval is DETERMINISTIC — see Traps)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python clark_scarf_ppo_train.py -s n3_l2_p09 -o raw
python clark_scarf_ppo_eval.py -s n3_l2_p09 --model-path results/n3_l2_p09/{run}/n3_l2_p09_ppo.zip

# the deployable policy (spec §12) — replays against the raw MDP loop, no gym.
# At 512 episodes it must reproduce ppo_eval to the digit; a drift means this
# file and the gym have diverged on normalization or the action transform.
python clark_scarf_policy.py --model-path results/n3_l2_p09/{run}/n3_l2_p09_ppo.zip \
    --vecnorm-path results/n3_l2_p09/{run}/vecnormalize.pkl -s n3_l2_p09 --episodes 512
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `clark_scarf/` only if it is (a) spec-§1 layout, (b) an
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
behind them can be re-run. A probe is written in `clark_scarf/` from the start
(it imports its siblings) and **committed in the same commit as the
escalation entry that cites it** — `git log ESCALATION.md` then shows each
finding beside the code that produced it. Design rationale that must outlive
the round has two tracked homes, neither of them the plan: the escalation
entry and the probe's module docstring.

Check with `git status --short clark_scarf/`: untracked files should be rare
and deliberate.

## Traps

Seeded by the pipeline; **every trap the campaign pays for is added here the
same day**, citing the finding (#E…) that paid for it.

- **Unseeded `reset()` must draw a fresh episode seed** — gymnasium auto-reset
  passes no seed, so omitting that branch silently trains on `n_envs` fixed
  exogenous paths while evals (which seed explicitly) keep looking healthy.
  Guarded by `test_unseeded_resets_draw_fresh_episodes`; proposed upstream as a
  conformance check (issue #18).
- **`raw` and `echelon` must stay a pure change of coordinates.** They are
  informationally identical *only* because `echelon` sums stock and in-flight
  stock **separately**. A single running sum over their total is lossy at
  `leadtime = 2` and would silently turn the headline experiment into a test
  of information rather than representation.
  `clark_scarf_test.py::test_raw_and_echelon_are_a_pure_change_of_coordinates`
  is the guard — never weaken it.
- **Echelon STOCK is not echelon POSITION.** Holding is charged on echelon
  *stock*, where stock in transit **to** a level belongs to the echelon
  **above** it (Assumption 3: "at a lower level **or in transit to a lower
  level**"). Echelon *position* `u = x_1 + w_1 + …` is a different quantity —
  what `f_n(u)` optimizes over and what a base-stock rule is stated on. The
  first build conflated them and double-charged pipeline stock (2.0 vs 1.0 per
  period at `n3_l2_p09`), biasing the shipping incentive against moving stock
  downstream — the exact trade-off this campaign measures. `echelon_stock()`
  and `echelon_position()` in `clark_scarf_mdp.py` are deliberately separate
  functions; do not merge them.
- **The observation box is a VALIDITY envelope, not a scale.** It is sized to
  the reachable range (`mean + ship_max * horizon`), which is far wider than
  sensible play, so `contains()` holds under random actions. Normalization is
  VecNormalize's job. Do not "tighten" it — clipping into a smaller box
  destroys the invertibility above.
- **A sweep is not a domain, and a cap is not a model.** `leadtime` admits
  any integer >= 1 (model); the campaign sweeps {1,2,3} (design). There is
  **no lead-time cap** — every `pipe_k` declares `length: "leadtime"`, so the
  width names the constant and each instance renders exactly the slots it
  selects. It was not always so: F3 unrolled a fixed `LEADTIME_CAP = 4`, which
  made `L = 5` unrepresentable, and F7 removed it. **A width that can name a
  constant must name it** — a literal there is a ceiling nobody declared. The
  three layers are recorded separately: machine-readable in the schema's
  `mdp.model` (+ `narrowed` citations), gated upstream by `mdp_conformance`'s
  `model.boundary`, and in prose in the restatement §Model envelope. The model
  layer may not mention rendered names, cap literals, or solver tractability.
  The first freeze violated this and had to be re-signed (F3).
- **`model.boundary` PASSes with every width derived** — its inventory reads
  `horizon_T (tier-2 horizon)`, `leadtime (tier-2 obs-dim)`,
  `n_echelons (tier-1)`, `ship_max (tier-1)`. There is no unpaired cap because
  there is no cap (F9). It briefly reported `n_levels_max` as unpaired — named
  for no declared quantity, so its coverage went unchecked — and that residual
  was closed by deleting the constant rather than by pairing it, which is the
  better of the two fixes: nothing left to check.
- **`N_LEVELS_MAX` is gone — do not reintroduce it (F9).** It existed only
  because `Decision.dim` could not name a constant; upstream #33/#36 lifted
  that, and the `ship` decision is now `dim: "n_echelons"`. A constant named
  at `dim` classifies **tier-1**, so `grids.axes` warns about sweeping it —
  correct, since chain length changes the action space and no policy spans
  two widths. That warning is expected here, not a defect.
- **A pin bump can move a frozen fingerprint with no IR edit.** v0.9.2 did:
  `ModelQuantity.stochastic` escapes `_prune_absent`, so every IR declaring
  `mdp.model` hashed differently (upstream #34). F8 made it moot here by
  declaring the field on every quantity. **Re-gate before trusting any result
  after a bump**, and never assume a moved token means someone edited the
  model.
- **A vector decision must be resolved PER INSTANCE, not once.** Five harness
  sites built decision values from `dim`/`bounds`; all now thread the instance
  through (upstream #33/#36/#38, v0.9.4). The failure mode this leaves behind
  is subtle rather than loud: a width or bound taken from the *base* IR is a
  plausible number that is simply wrong for the instance being run. Upstream
  found the same bug live in `mab` — `arm` narrows from `(0, 9)` to `(0, 4)`
  on its 5-arm instances, and the laws were being fed an index those instances
  do not have. If you ever hand-build a decision dict here, pass the instance.
- **A mixture NAME is not a scenario instance.** `decision_bounds` raises
  `KeyError` on one, and the laws *are* called with mixture names — so any
  code threading an instance into a decision builder needs
  `inst if inst in scenario.instances else None`. This domain declares no
  mixtures, so it never hit it; `inv_single` would have.
- **There is no padding any more (F9).** Every state and action vector is
  exactly `n_echelons` long, because all three widths NAME the constant. The
  old trap here warned that inert slots must stay economically invisible; the
  slots are gone, so the claim to hold is the one that replaced it — a width
  must never become a literal. `h_install` and `c_ship` are still declared
  four long, but that is a per-instance **data** tail nothing indexes past
  `n_echelons`, not a rendering width.
- **`ClarkScarfScenario` must NOT be callable.** A self-returning `__call__`
  makes "did this resolve to a concrete scenario?" unanswerable and fails
  `mdp_conformance`'s `scenario.samplers` check. Latent resolution belongs on
  `ClarkScarfScenarioSource`.
- **The record eval is DETERMINISTIC here** — the policy is a continuous
  shipment quantity with no exploration role at eval time, and the benchmark
  it is scored against (the Clark–Scarf DP) is deterministic. A stochastic
  eval is a separate, labelled figure and never a record.
- **The DP row is not unconditionally exact** — it truncates the Poisson pmf
  at a high quantile while the simulator does not (mass < 1e-9). Say so
  wherever "% of optimal" is quoted.
- **Selection and terminal artifacts are different networks**
  (`{scenario}_ppo.zip` vs `_ppo_final.zip`, spec §8.4) — say which one a
  number came from.
- **Compute sites/venues are cited by alias, never hostname** — venue
  config lives at the repo root; a venue change is a confound to record,
  not a detail.
- **Never use `param`, `params`, or `param_*`** anywhere (conformance
  fails).

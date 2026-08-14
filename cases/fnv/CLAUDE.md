# CLAUDE.md — `fnv/` (fresh-newsvendor sequential ordering under MMFE)

Domain-local operating brief. **Pointer-first: this file says where to look
and what will bite you, never what the answer is.** Anything that changes as
the campaign progresses lives in the docs below and is linked, not copied.

Place `N` sequential orders at rising unit cost while a martingale demand
signal is progressively revealed (Heath & Jackson 1994 MMFE); demand realizes
once at the horizon and is sold against cumulative inventory. State is
`(period, inventory, information)`; the decision is one continuous order
quantity; the objective is expected profit.

**`FNV-aMMFE` (additive, `D = mu + I`) and `FNV-mMMFE` (multiplicative,
`D = exp(mu + I)`) are SEPARATE LEADERBOARDS** — different demand scales,
different bars. Their scores must never be compared. Within a grid, each cell
is its own row.

## This domain is GENERATED — the skill is authoritative

`fnv/` was produced by the **auto-mdp-solver** skill from `fnv_schema.json`.
It is not a hand-written project, and it must not drift into one.

**Before changing anything here, read the skill docs — they are authoritative
over this file, over the code, and over local habit:**

| doc | governs |
|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG/seed tree, script + eval conventions (§1–§14) |
| `SKILL.md` | the pipeline stages, their gates, and what each stage must emit |
| `ESCALATION_LOG_GUIDE.md` | the campaign log (§MAP, §FRAME-CHANGELOG, §IR-CHANGELOG, §LEDGER) and the §10 playbook digest |

Consequences, each already paid for somewhere:

- **Check the spec before inventing a mechanism.** The pipeline usually has
  one already, and its version composes with the gates.
- **Structural changes go through the IR first**, then the code, then the
  gates — never code-first. `fnv_schema.json` is the source of truth;
  `fnv_scenarios.py` and friends materialize it.
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
| `README.md` | layout, usage commands, leaderboards |
| `ESCALATION.md` | the campaign: §MAP, §IR-CHANGELOG (F1), §LEDGER #E1–#E9. **Closed 2026-08-14** |
| `INTERPRET.md` | policy readback, both branches — the structure IS recovered (§14) |
| `PLAYBOOK.md` | the case-close digest: LV1–LV7, FM1–FM3, MR1–MR2, envelope verdict |
| `fnv_schema.json` | **the IR — authoritative** for the problem definition |
| `fnv.restatement.md` | the Phase-A restatement (reverse-engineered — the IR was ported, see Traps) |
| `UPSTREAM_PROPOSAL_*.md` | drafts against the solver spec, not local decisions |

Round plans are deliberately **not** in this table: they live in `scratch/`
and are deleted once their findings are in `ESCALATION.md` (see File
hygiene).

Solver: this case lives **inside** the auto-mdp-solver repo, so the harness it
builds against is the repo's own `harness/` at whatever revision you have
checked out. The campaign below was run against **v0.8.0**. The IR half landed
in v0.8.0 (demoted from `examples/` for shipping no train/eval pair); the solve
leg and campaign record were contributed after.

## Commands

Run from `fnv/` unless noted. Always pin threads for training — torch
oversubscribes.

```bash
# gates (conformance/laws/differential run from the PARENT of fnv/)
python -m mdp_ir fnv/fnv_schema.json
python -m mdp_conformance fnv
python -m mdp_ir.laws fnv
python -m mdp_ir.differential fnv/fnv_schema.json --episodes 40 --all-instances
pytest fnv_test.py

# the solve leg, in order (§8.6 / §9.7 — the terminal checkpoint is NEVER the deliverable)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python fnv_ppo_train.py -s FNV-aMMFE
python fnv_select.py results/FNV-aMMFE/PPO_<run>          # screen  ~2048 CRN seeds
python fnv_ppo_eval.py --model-path <top-k ckpt> -s FNV-aMMFE   # protocol ~8192

# the optimal-policy benchmark: two independent solvers, one eval
python fnv_benchmark_dp.py -s FNV-aMMFE --self-test              # value-function DP
python fnv_benchmark_prop2.py -s FNV-aMMFE --check-invariants --cross-check  # the paper's eq (6)-(7)
python fnv_benchmark_dp_eval.py --dp-solutions results/FNV-aMMFE/benchmark/prop2/FNV-aMMFE.txt -s FNV-aMMFE

# readback (§14) — FNV has a DP reference, so the ANCHORED form is required
python fnv_plot_policy.py --model-path <ckpt> -s FNV-aMMFE   # slopes + figure
python fnv_policy_probe.py --model-path <ckpt> -s FNV-aMMFE  # surfaces, sensitivity

# §14.2 — distil the net into offsets, then SCORE them as a policy
python fnv_benchmark_fitted.py --model-path <ckpt> -s FNV-aMMFE
python fnv_benchmark_dp_eval.py --dp-solutions results/FNV-aMMFE/benchmark/fitted/FNV-aMMFE.txt -s FNV-aMMFE --n-seeds 2048
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `fnv/` only if it is (a) spec-§1 layout, (b) an
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
behind them can be re-run. A probe is written in `fnv/` from the start
(it imports its siblings) and **committed in the same commit as the
escalation entry that cites it** — `git log ESCALATION.md` then shows each
finding beside the code that produced it. Design rationale that must outlive
the round has two tracked homes, neither of them the plan: the escalation
entry and the probe's module docstring.

Check with `git status --short fnv/`: untracked files should be rare
and deliberate.

## Traps

Seeded by the pipeline; **every trap the campaign pays for is added here the
same day**, citing the finding (#E…) that paid for it.

- **The record eval is the deterministic argmax here** — the action is a
  single continuous order quantity and exploration plays no role at
  inference, so `deterministic=True` is the deployment mode. A stochastic
  eval would be a separate, labelled figure and never a record.
- **`fnv_policy.py` reproduces the action clip** rather than importing a helper
  from `fnv_gym.py` — a historical artifact of the period when the IR half was
  kept byte-identical to an upstream copy. Harmless, and asserted against the
  gym in that file's `__main__` smoke test.


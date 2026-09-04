# CLAUDE.md — `adi_flex/` (inventory with advance demand information and flexible delivery)

Domain-local operating brief for an agent about to change this folder.
**This file points up: at the specs that govern the folder, and the version of
them its numbers were produced under.** Everything the campaign produced — its
documents, its code, its numbers — is inventoried in `README.md` and linked from
here, never copied.

Customers order in advance with due dates up to `T_dl` periods out and stock may
ship early, so each period sets a replenishment order *before* demand is
observed and an early-fill allocation *after* it — two decisions at two
information sets. Minimise fixed-order + holding + backlog cost over a finite
horizon.

**Three separate leaderboards, whose scores may never be compared:** `homog_*`
(one demand lead time, allocation not live, exact DP reference), `het_*`
(a demand mix at `T_dl = 2`), and `het3_*` (`T_dl = 3, L = 1`). Different
problems, different references, different horizons.

## The specs that govern this folder

| doc | governs | where |
|---|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG and seed tree, script and eval conventions | the `mdp-solver` skill's own directory (`plugin/skills/mdp-solver/` in this repository) |
| `SKILL.md` | the pipeline stages, their gates, what each must emit | same |
| `ESCALATION_LOG_GUIDE.md` | the campaign log's structure and its §10 digest | same |

They live in the `mdp-solver` skill's own directory — wherever the plugin is
installed, they sit beside its `SKILL.md`; read them from there and do not
search the filesystem for them by name. **Minimum v0.9.11** — older harnesses
cannot load this IR.

**Two versions, and they answer different questions:**

- **Built and gated at: v0.9.11** for the leaderboards through `#E18`; the
  ordinal-head cells (`#E22`–`#E25`) were trained under v0.9.33–v0.9.35 and
  re-gated under v0.10.4/v0.10.5. This line moves only when numbers are re-measured.
- **Conformance maintained through: v0.10.5** — the release that ships the case;
  its harness is byte-identical to v0.10.4, where the gates ran (2026-09-04: conformance 24/30 no
  FAIL, laws 7/9, differential 31 MATCH, 63 tests, head gate G1–G5) — how far
  declarations, drawing conventions and gate compatibility have been carried
  forward. Every move is logged in `ESCALATION.md` §FRAME-CHANGELOG.

Upgrading: read the release notes, move the harness to the new tag, re-run
every gate below, then log the move.

## Hard rules

- **Before changing the IR, the gym or a benchmark** → read the F-notes in
  `ESCALATION.md` §IR-CHANGELOG. They are what this folder learned the hard way,
  each with the evidence that paid for it.
- **Structural change** → IR first, then code, then gates. Never code-first.
  Observation and architecture levers are not IR changes.
- **Any IR change or solver bump** → re-run every gate below. A moved `mdp`
  fingerprint → an §IR-CHANGELOG entry.
- **Deviating from the spec** → record it in the code comment, in the log, and
  as an `UPSTREAM_PROPOSAL_*.md` filed through the `mdp-propose` skill.
- **Launching a run** → cite a config (`--config sc0/g6/a1/h6`); explicit flags
  override and are recorded as deviations. A run is
  `sc + g + a + h + deviations + seed`; if it cannot be written that way, the
  registry is wrong → `adi_flex_configs.py`.
- **Promoting a config** → register it in both halves, `adi_flex_configs.py` and
  §CONFIG-REGISTRY, and record the crowned artifact's action space beside the
  id (F57).
- **Changing the policy's distribution parameterization** (an `a`-axis knob such
  as `--order-head`) → an arch-layer escalation: it mints an `a` id, derives
  `L4(hp+gym+arch)`, and is neither an action mode nor an IR change. Its gate is
  `adi_flex_ordinal_head_probe.py`.
- **Quoting a number** → say which artifact it came from (spec §8.4) and which
  leaderboard it belongs to → `README.md` §3.
- **Writing to the log** → §FRAME-CHANGELOG, §IR-CHANGELOG and §LEDGER are
  append-only, new entries at the end; §MAP and §CONFIG-REGISTRY are living.
- **Never** use `param`, `params`, or `param_*`; conformance fails.

## Gate commands

Run from `cases/` (the parent of this folder), with `PYTHONSAFEPATH=1` so a
stray module on the working directory cannot shadow the harness.

```bash
PYTHONSAFEPATH=1 python -m mdp_ir adi_flex/adi_flex_schema.json
PYTHONSAFEPATH=1 python -m mdp_conformance adi_flex
PYTHONSAFEPATH=1 python -m mdp_ir.laws adi_flex
PYTHONSAFEPATH=1 python -m mdp_ir.differential adi_flex/adi_flex_schema.json \
                     --episodes 40 --all-instances
cd adi_flex && pytest -q adi_flex_test.py
cd adi_flex && python adi_flex_ordinal_head_probe.py     # a1 acceptance gate G1–G5
cd adi_flex && python adi_flex_action_mode_probe.py      # action-mode decode acceptance — needs the
#   solved tables homog_L0_T2/dp, homog_L0_Tdl1/dp, het_exp4/ap and het_exp7/ap
#   (README §4's benchmark commands; heterogeneous instances solve under `ap`)
```

The §9.9 **role gate** is a check too, so it lives here rather than in the
README. `--ir` is what activates the role reasoning, and the eval TSVs must keep
their `benchmark_{method}_eval` filename marker or the roles go inert.
**The sense is `maximize`** even though this domain minimizes cost: every eval
TSV reports `reward_mean` = the negated cost (see `adi_flex_benchmark_common`),
so higher is better and the harness default is right. `--sense minimize` is for
a cost column and here inverts every verdict — it declares the artifact beats
the AP lower bound. Pass `--metric reward_mean` so the choice is explicit.

```bash
cd adi_flex && python -m mdp_gates --ir adi_flex_schema.json --n-seeds 8192 \
  --sense maximize --metric reward_mean \
  --candidate  results/<scenario>/PPO_<run>/ppo_eval_<scenario>.tsv \
  --baseline   results/<scenario>/benchmark/benchmark_rule_eval_<scenario>_plsigma.tsv \
  --reference  results/<scenario>/benchmark/benchmark_ap_eval_<scenario>.tsv
```

`PYTHONSAFEPATH` must **not** be set for the domain scripts in `README.md`'s
appendix: they import their siblings by name and need the working directory on
`sys.path`.

## File hygiene — the folder is the deliverable, not the workbench

A file belongs in `adi_flex/` only if it is spec-§1 layout, an implementation a
finding cites and someone must re-run to reproduce it, or a campaign document.
**Everything else goes to `scratch/`** (gitignored); all output goes to
`results/` (gitignored).

- **Round plans are never tracked.** A `*_PLAN.md` lives in `scratch/` while it
  is drafted and while it runs, and is deleted once its findings are logged. If
  a plan is the only place a result exists, that is a bug in `ESCALATION.md`.
- **A probe a finding cites stays permanently**, in `adi_flex/`, committed with
  the entry that cites it. **No document may cite `scratch/`** — it is
  gitignored, so the citation points at nothing for anyone else.
- Check with `git status --short adi_flex/`. Untracked files should be rare and
  deliberate.

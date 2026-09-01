---
name: mdp-solve
description: >
  Stages 0+3–4 of the MDP pipeline, the mechanical backbone: confirm a run
  plan, build and run baselines, train PPO at L0 (faithful defaults) and L1
  (IR-derived), eval against the baselines under the shared seed protocol,
  and diagnose the outcome. Use when a generated domain passes build's gates
  (conformance + laws + differential) and a leaderboard row is wanted — or to
  re-enter for another target. Entry: python -m mdp_stage {domain} --for
  solve. Escalation beyond L1 is mdp-escalate; the full automatic pipeline is
  mdp-solver.
---

# mdp-solve — run plan, baselines, L0/L1 train + eval

One op of the split MDP pipeline (`mdp-solver` is the conductor; its
`CONTRACTS.md` maps the ops). The governing docs live beside the conductor in
this plugin's `mdp-solver` skill directory —
`${CLAUDE_SKILL_DIR}/../mdp-solver/` holds `MDP_PROJECT_SPEC.md` (the
canonical convention reference — consult it while writing each script; do not
code from memory of it), `MDP_IR_SAMPLE.md`, `ESCALATION_LOG_GUIDE.md`,
`ENVIRONMENT.md`, `INTERVIEW.md`, `CONTRACTS.md`, `PLAYBOOK.md` and
`examples/`. **Read them from that path. Do not search the filesystem for
them**: a development checkout of the solver may also be on disk, and reading
that instead silently substitutes unreleased content for the version you are
installed at. The pipeline runs in whatever workspace holds the domain; use
that workspace's venv, resolved per `ENVIRONMENT.md` — whose retry budgets,
>30-minute ask, and background-run rules govern every launch below.

**Entry gate — run first, exit 0 required:**

```bash
<python> -m mdp_stage {domain} --for solve
```

This *runs* build's exit gates — conformance, IR laws, and the bit-exact
differential — rather than trusting that they once passed; it is the honest
form of "the domain is ready to spend compute on". A blocked entry goes back
to `mdp-build` (or `mdp-formalize` if the freeze check failed), not around
the gate.

### Stage 0 — run plan (confirm before running)

Phase A froze the *whole* scenario set; one solve pass does **not** run all
of it. Before launching anything, confirm a **run plan** with the human (via
AskUserQuestion, under the interaction rules in `INTERVIEW.md`) — and treat
it as a live choice, not a Phase-A relic to rubber-stamp. When the user
supplied a run-plan file up front (the unattended path), take it as the
confirmed answer and skip the round — that supplied-vs-asked fork is the only
behavioural difference:

- **Target.** Choose what this pass actually builds baselines / trains /
  evaluates against — one `SCENARIOS` entry (a fixed scenario or a world
  sampler), or one `GRIDS` entry (a generalist target; and, if Phase A
  produced separate mode *branches*, which branch). Default to one target;
  never silently kick off every registered target. The Stage 1–2 code
  supports every registered scenario and grid regardless — this choice
  scopes only the expensive Stage 3–4 runs.
- **Train/eval strategy.** Re-confirm *or switch* the strategy the Phase-A
  scenario set implied: a **generalist** — trained on `grid.as_sampler()`,
  evaluated by enumerating the grid (one leaderboard row per cell, one
  shared eval-seed block, spec §5.6, §9.6) — vs. a **specialist** trained
  and evaluated on the single target. The strategy is not static — the
  human may change it here; honor it.
- **Escalation budget.** How much L2+ search this pass may spend if the L1
  gate shows a gap — rounds or wall-clock; "none" is a legitimate answer.
  It is `mdp-escalate`'s stopping rule, so it is decided here, before any
  gap exists to tempt anyone.

Write the confirmed plan to `{name}/{name}.runplan.json` per `CONTRACTS.md`
(target, target_kind, strategy, escalation budget, date) — this op is that
file's only writer, and every downstream op locates its artifacts through it.
It is the contract for Stages 3–6: the selected scenario name(s) drive the
baseline, training, and eval commands, and the leaderboard reports only what
this pass actually ran. Returning for another target later re-enters here at
Stage 0, rewriting the file — not at Stage 1.

### Stage 3 — baselines (mandatory)

Build and run **benchmarks** (any non-RL solution — from the source
paper/document or synthesized on the fly) **for the Stage-0 target
scenario(s) only**, not the whole frozen grid. Each is
`{domain}_benchmark_{method}.py` with an eval
`{domain}_benchmark_{method}_eval.py`, where `{method}` is a short method tag
(`lp`, `dp`, `myopic`, `greedy`, `fluid`, or a domain-custom heuristic) — not
a hard-coded `_lp`/`_dp`. Minimum: **random** and a **myopic/greedy
heuristic**. Add exact **DP** (`{domain}_benchmark_dp.py` +
`{domain}_benchmark_dp_eval.py`) when the state is fully observed and small
enough to enumerate — deriving the recurrence is your job; verify its value
against a closed-form or limiting case when one exists. All evals share the
spec-§9 seed loop, TSV columns
(`<axes>... <metric>_mean <metric>_var semivar_d semivar_u`), and one seed
count (default 8192). A benchmark that precomputes a solution table writes it
to `results/{scenario}/benchmark/{method}/{scenario}.txt`; eval TSVs go to
`results/{scenario}/benchmark/benchmark_{name}_eval_{scenario}.tsv`. Resolve
both from `Path(__file__).resolve().parent`, never from the CWD (spec §8.4) —
and redirect each eval's console output into that same `benchmark/` directory:

```bash
cd {domain}
out=results/{scenario}/benchmark; mkdir -p $out
for m in random myopic dp; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup ../.venv/bin/python \
    {domain}_benchmark_${m}_eval.py -s {scenario} --n-seeds 8192 \
    > $out/benchmark_${m}.log 2>&1 &
done
```

**GATE:** baselines run to completion and their ordering is sane
(DP ≥ heuristics ≥ random); they now bound the reward scale. Nothing was
written outside `results/` — `git status --short {domain}/` shows no stray
`*.log`/`*.tsv` in the domain folder, and no `results/` tree appeared anywhere
but the domain directory.

### Stage 4 — train + eval (+ tune)

Training runs are **leveled** (spec §8.6): **L0** = faithful defaults
(`PPO("MlpPolicy", env, gamma=β, seed=s).learn(2M)`, no VecNormalize — the
control run and the ruler for agent contribution; reporting only, never a
gate), **L1** = the spec-§8.6 derivation table applied — the mandatory run,
with every derived knob logged with its one-line rationale — and **L2+** =
escalations (hp / gym / arch), only after the L1 gate shows a gap. A tuned
result is always L2: level ≥ L2 ⟺ more than one training config was tried.
Run L0 by default (it is usually the cheapest run on the board); put both on
the leaderboard, clearly labeled.

L2+ design is `mdp-escalate`'s job, not this op's — the backbone ends at the
L1 verdict below.

- Train on the **Stage-0 target and strategy** (the selected fixed scenario or
  world sampler for a specialist, or `grid.as_sampler()` for a generalist) —
  eval enumerates the grid's cells only when the strategy is generalist.
- `{domain}_ppo_train.py` per spec §8 (`_build_arg_parser`/`parse_args`/
  `main`; the **tier-1 CLI surface** of §8.2's table — every name there is one
  tooling and humans join on across domains, and `scripts.cli_contract` reports
  what is missing; tier-2 knobs are whatever `mdp_tuning --show-space` names for
  the family, expose all of them or the study searches a smaller space than it
  reports; VecNormalize per the IR's `obs_normalization` decision **with
  `gamma=args.gamma` passed**; script defaults = the L1-derived values, so the
  tuner's warm-start trial 0 is the L1 center; run-name encodes obs/act/rew +
  non-default hyperparameters).
- Launch with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, in the background;
  watch `ep_rew_mean` against the baseline bounds while it runs. The train
  script tees its own stdout/stderr to `{run_dir}/train.log` (spec §8.4), so
  send the shell redirect to the scratchpad and poll the run's own log:

  ```bash
  cd {domain}
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup ../.venv/bin/python \
    {domain}_ppo_train.py -s {scenario} -o {obs} > $SCRATCH/train_launch.log 2>&1 &
  # the script prints `outdir=` on its first lines; watch results/{scenario}/*/train.log
  ```

  Never redirect a run's output into the domain folder — that is what leaves
  orphaned `results_train_*.log` files behind.
- `{domain}_ppo_eval.py` per spec §9.5 (VecNormalize injection), same seed
  protocol as the baselines — the same `(--first-seed, --n-seeds)` block on
  every arm, which is what makes the per-seed differences paired. It knows its
  run directory from `--model-path`, so redirect it straight there:
  `> $(dirname <model-path>)/eval.log 2>&1`.

**GATE:** `python -m mdp_gates --candidate <ppo_eval.tsv>
--baseline <random.tsv> --baseline <myopic.tsv> [--reference <dp.tsv>]
--n-seeds <N>` must exit 0 (beats every baseline by ≥ 2 SE). It also refuses
a candidate whose run stopped short of its declared budget (§13's harvest
precondition) — re-run it, or `--short-ok "<reason>"` if the stop was
deliberate. Report the gap to the DP reference. If a design axis (an action/obs mode) fails after
its one repair attempt, ship the best passing axis and record the failure.

**GATE (layout):** the produced tree matches spec §8.4 exactly — run dirs
directly under `results/{scenario}/` with no interposed `RL/`, each holding
`train.log`, `{scenario}_{algo}_args.txt`, the model, and the eval TSV; no
`results/` anywhere but the domain directory; no `*.log`/`*.tsv` loose in the
domain folder. Check it, don't assume it:

```bash
cd {domain}
find results -maxdepth 2 -type d | sort          # expect {scenario}/{run_name}, {scenario}/benchmark
ls *.log *.tsv 2>/dev/null && echo "STRAY FILES — fix the path anchoring"
ls ../results 2>/dev/null && echo "CWD-ANCHORED WRITE — fix the path anchoring"
```

A miss here is almost always a script resolving `Path(args.outdir)` instead of
`Path(__file__).resolve().parent / args.outdir`, or a shell redirect aimed at
the CWD.

### Exit — diagnose before anything else

Diagnosis before escalating (spec §8.6): **L1 ≤ random** → suspect the
backbone, not the knobs — a build bug; back to `mdp-build`. **L1 < L0** → the
derivation misfired; recheck it against the spec-§8.6 table. **Competitive**
(the `mdp_gates` gate exits 0) → stop at L1: report the leaderboard and hand
to `mdp-interpret`. **Gap with a healthy build** → escalation is on the
table: report the gap, the budget the run plan grants, and hand to
`mdp-escalate` — opening L2+ is a decision the human hears about, never a
default. Either way the leaderboard rows this pass produced (L0 and L1,
clearly labeled) are the durable exit.

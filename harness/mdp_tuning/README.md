# mdp_tuning

Domain-generic hyperparameter tuning for spec-conformant MDP domains — the
tuning sibling of `mdp_conformance`. Any domain that follows
`MDP_PROJECT_SPEC.md`'s script conventions can be tuned with **zero
per-domain code**.

## Why this is generic

Hand-written per-domain grid searches encode per-problem judgment (which
knobs, which values). This harness removes that judgment in three ways:

1. **Algorithm-level search space** (`spaces.py`): broad, RL-Baselines3-Zoo
   style ranges over PPO's knobs, defined once for *all* domains. Judgment
   about ranges is made once per algorithm, not per problem. Knobs are grouped
   into three budget tiers by how much spec §8.6's L1 derivation already knows:
   `core` corrects the knobs whose derived value is a real guess
   (`learning_rate`, `net_arch` width, `n_steps`, `ent_coef`, `gae_lambda`),
   `breadth` opens what the derivation does not produce (`net_arch` depth,
   `n_epochs`, `batch_size`, `vf_coef`, `gamma`), and `all` adds the frozen
   pair (`clip_init`, `max_grad_norm`). Schedule finals are never tuned —
   `lr_final` and `clip_final` are derived from the tuned inits (§8.2), so a
   schedule cannot invert.
2. **CLI introspection**: the spec mandates `_build_arg_parser()` in every
   train/eval script (§8.2), so the harness imports the domain's scripts,
   reads their real CLIs, and tunes exactly the knobs the domain exposes —
   skipping the rest. Fixed `--train-arg KEY=VALUE` overrides are validated
   the same way, and shared settings (e.g. `observation_mode`) are
   auto-forwarded to the eval script. "Skipping the rest" is also how a study
   can quietly search a smaller space than it was asked for, so **naming a tier
   makes the skip fatal**: pass `--knobs` explicitly (or `--strict-knobs`) and
   an in-tier knob with no matching dest aborts the launch instead of warning
   once. `--show-space` reports the same thing without launching.
3. **Optuna TPE**: where to sample next is model-based, not hand-picked.
   Studies live in sqlite under `{domain}/results/tuning/`, so interrupted
   sweeps resume with their full history.

Each trial shells out to the domain's own `{prefix}_{algo}_train.py` →
locates the saved model → `{prefix}_{algo}_eval.py` → reads the eval TSV.
The objective is a TSV column: `--metric NAME`, or auto — the first `*_mean`
column in header order, which spec §9.3 fixes as the domain's primary
objective. Add `--minimize` when that objective is lower-is-better.

## Usage

```bash
# from the repo root, using the project venv
python -m mdp_tuning plugin/skills/mdp-solver/examples/inv_single -s simple --metric cost_total_mean \
    --n-trials 25 --total-timesteps 200000 --eval-seeds 500

# what would be tuned, no training — plus the two things that otherwise only
# surface at launch: which tiers this script can actually reach, and whether
# trial 0 would really be the L1 centre
python -m mdp_tuning plugin/skills/mdp-solver/examples/inv_single --show-space
#   tier reach : core ok  |  breadth BLOCKED(vf_coef)  |  all BLOCKED(vf_coef, ...)
#   warm start : trial 0 = L1 ROUNDED to the grid — ent_coef 0.0->1e-08
# best trials so far
python -m mdp_tuning plugin/skills/mdp-solver/examples/inv_single -s simple --summary-only
```

Domains may ship a thin `{domain}_ppo_tune.py` wrapper that pre-fills
domain-appropriate defaults.

## Requirements on the domain

- `{prefix}_{algo}_train.py` with `_build_arg_parser()` exposing at least
  `--scenario_name`, `--total_timesteps`, `--outdir` (spec §8.2/§8.4).
- `{prefix}_{algo}_eval.py` with `_build_arg_parser()` exposing
  `--model-path`, `--outfile`, and an episode-count flag (`--n-seeds` per
  spec §9.1; `--episodes` accepted for legacy scripts).
- The eval TSV has one header row and numeric metric columns (§9.3).

Those three are the hard minimum this harness cannot run without. The full
tier-1 surface it expects — and the check that reports it before a study is
ever launched — is spec §8.2 and `mdp_conformance`'s `scripts.cli_contract`.

**Warm start.** Trial 0 is enqueued as the train script's own defaults, so the
study measures what tuning adds over the L1 centre (§8.6). The space is a grid,
so a default off it (`n_steps` 2560, an `ent_coef` of exactly 0) is **snapped to
its nearest representable neighbour** — dropping it would hand the knob to the
sampler and put trial 0 an unbounded distance from L1, where a rounding step is
bounded by the grid. Every move is named in `--show-space`, in the launch
banner, and in the study's `warm_start_snapped` attribute, so a study is never
read as a Δ(L2−L1) it did not measure. Values the space refuses *on purpose* —
γ > β, an `n_steps` under the rollout floor, a channel stack — are not rounded;
they are signals, and they still skip.

**Parallel workers.** Several workers on one study (same `--study-name` and
`--storage`, different `--seed`) share the table, and TPE models its COMPLETE
rows only: every worker that draws against the same completed rows fits the
same model and proposes its mode. A wave of twelve workers started minutes
apart then trains one configuration twelve times, the sampler seeds moving
only which candidates are scored. Pass **`--constant-liar` on every worker of
a parallel launch**: a RUNNING trial then counts as an observation at the worst
value seen until its real score lands, each draw is written to the storage
before the next worker draws, and each worker sees the earlier proposals
marked bad and goes elsewhere. It is off by default because the sampler is not
persisted with the study — the default fixes whether a study's draws can be
reproduced from its seed, and studies drawn before the flag existed were drawn
without it. The lie is only ever about RUNNING rows, so one worker samples the
same either way. Two things the flag does not change: workers must still start
far enough apart for a draw to reach the storage (seconds; stagger them —
except under `--max-trials`, whose lock holds through the draw), and a draw is
guided by however many completed trials exist — the liar spreads a wave, it
does not sharpen it.

**Study-wide budget.** `--n-trials k` is optuna's own count of the trials
*this worker* runs, so W workers pre-assign k trials to each, and a worker that
draws slow configurations still owes its k after the rest have exited —
observed as five of twelve workers idle on day 3 of a 2048 study while the slow
worker's last trial had not been asked for. **`--max-trials N`** makes the
budget the study's: every worker takes the next trial whenever it is free,
until the study holds N trials that did not FAIL — first come, first served.
The count, the ask and the draw sit under one lock beside the storage, so the
study never overshoots N, a worker launched after it is full runs nothing, and
workers need no stagger (`MaxTrialsCallback` would not do: it checks at trial
end only).

```bash
# 12 workers on one GPU, one study of 36 trials — the same line on every worker
for w in $(seq 1 12); do
  python -m mdp_tuning <domain> -s <scenario> --study-name s --max-trials 36 \
    --seed $((41 + w)) --constant-liar --total-timesteps 60000000 &
done
```

The launch that creates the study **records its settings on the study** —
scenario, knob tier, `--fix`, `--train-arg` / `--eval-arg`,
`--total-timesteps`, `--eval-seeds`, metric, `--constant-liar`, N, the FAIL
cap — and its `--seed` as the study's **seed base: trial n trains on
`base + n`**, whichever worker runs it. A later worker takes the record's
settings and refuses a flag that contradicts it; its own `--seed` seeds only
its sampler. Every trial records its training seed and where it ran (host, pid,
`CUDA_VISIBLE_DEVICES` — the device is the worker's, inherited by the train
subprocess). A study launched with `--n-trials` is not adopted: its trials
trained on per-worker seeds that `base + n` cannot describe.

Two stops keep a shared budget finite. FAIL does not count toward N, so an
unscored crash is redrawn — and the study takes no new trial once it holds
**`--max-fails`** (default 5) FAIL rows. And a worker stops after
**`--max-consecutive-crashes`** (default 3) crashed trials in a row, FAILED or
penalized: first come, first served hands the most trials to the fastest
worker, and a worker on a broken device is fast — after the first completion
each of its crashes is a penalized row that counts toward N. `--timeout` is
per worker: no new trial starts after that many seconds from the worker's own
launch.

**Pools across studies.** Repeat `--study-name` and one worker serves several
budgeted studies of one storage: when free it takes the next trial of
whichever study has the largest share of its budget left (ties by join
order), which keeps paired studies at the same stage, and it exits when every
joined study holds its N. It keeps one sampler per study, seeded from its
`--seed` and the study's name, so it does not replay the same random-startup
draws in every study it serves. A per-study budget fixes idle workers inside one
study; only a pool fixes them across studies — a campaign's narrow-network
study had five workers exit with nothing left while its wide twin still had
nine trials unasked. Each trial runs with its own study's recorded settings, so
a pool worker carries no per-study flag and refuses a study with no record.
`--create-only` creates and records one study, enqueues its warm start, and
exits:

```bash
for c in 64 128; do
  python -m mdp_tuning <domain> -s <scenario> --study-name w$c --max-trials 36 \
    --seed 42 --constant-liar --train-arg channels=$c --create-only
done
for w in $(seq 1 24); do
  python -m mdp_tuning <domain> --study-name w64 --study-name w128 --seed $((41 + w)) &
done
```

The lock decides where a study's workers can run. `sqlite:///…` takes an
`flock` beside the database: workers on **one host** (sqlite itself is unsafe
across hosts on a shared filesystem). `journal:///…` takes an O_EXCL lock file
beside the journal — the protocol optuna's journal lock uses on NFS: workers on
**several hosts** sharing one filesystem, no database server needed. Any other
storage is refused under `--max-trials`: without a shared lock, two workers
that finish at once both start a trial and the study overshoots N by up to
W−1.

Crashed trials (e.g. NaN policy divergence at aggressive learning rates) are
**penalized to the study's worst completed value** rather than marked failed:
a failed trial carries no signal, so TPE would happily resample the divergent
region — observed as 8 consecutive failures next to a high-scoring trial
before this behavior was added.

The one case that rule cannot cover is a crash **before any trial has
completed**, where there is no worst value to take. Such a trial is left
FAILED and unscored, and the summary names it separately from the penalized
ones. The invariant is that a crashed trial must never become the study's
`best_trial`: the earlier direction-blind fallback of `0.0` broke it outright
in any MINIMIZE study, where 0.0 beats every attainable cost (upstream #76).
Nothing is taught by the omission, correctly — with no completed trial there
is no scale on which "avoid this region" could be expressed — and the
worst-completed rule resumes the moment one trial finishes. The penalty
covers a crash in *training*; an eval or scoring crash ends its trial FAILED at
any point in the study, which is why a study-wide budget caps FAIL rows.

## Limitations / future work

- **No mid-trial pruning yet.** Train scripts are single-shot subprocesses,
  so Hyperband/median pruning has nowhere to report from. Once train scripts
  support periodic-eval callbacks, wiring Optuna's pruners in makes cheap
  trials cheaper still.
- One algorithm space so far (`ppo`, used for PPO/MaskablePPO). Add entries
  to `spaces.SPACES` for new algorithms.
- **A worker killed mid-trial leaves its trial RUNNING**, and under
  `--max-trials` a RUNNING row counts toward N, so the study ends one trial
  short. Mark it FAIL to have it redrawn —
  `optuna.load_study(study_name=…, storage=…).tell(<number>, state=optuna.trial.TrialState.FAIL)`.
  optuna's storage heartbeat would do this automatically and is not wired in.
- The study record is fixed at creation: a budget cannot be raised on a live
  study, only continued in a new one.

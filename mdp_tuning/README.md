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
   about ranges is made once per algorithm, not per problem.
2. **CLI introspection**: the spec mandates `_build_arg_parser()` in every
   train/eval script (§8.2), so the harness imports the domain's scripts,
   reads their real CLIs, and tunes exactly the knobs the domain exposes —
   skipping the rest. Fixed `--train-arg KEY=VALUE` overrides are validated
   the same way, and shared settings (e.g. `observation_mode`) are
   auto-forwarded to the eval script.
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
python -m mdp_tuning examples/dynamic_pricing -s simple --metric revenue_mean \
    --n-trials 25 --total-timesteps 200000 --eval-seeds 500

# what would be tuned, no training
python -m mdp_tuning examples/dynamic_pricing --show-space
# best trials so far
python -m mdp_tuning examples/dynamic_pricing -s simple --summary-only
```

Domains may ship a thin `{domain}_ppo_tune.py` wrapper that pre-fills
domain-appropriate defaults (see
`examples/dynamic_pricing/dynamic_pricing_ppo_tune.py`).

## Requirements on the domain

- `{prefix}_{algo}_train.py` with `_build_arg_parser()` exposing at least
  `--scenario_name`, `--total_timesteps`, `--outdir` (spec §8.2/§8.4).
- `{prefix}_{algo}_eval.py` with `_build_arg_parser()` exposing
  `--model-path`, `--outfile`, and an episode-count flag (`--n-seeds` per
  spec §9.1; `--episodes` accepted for legacy scripts).
- The eval TSV has one header row and numeric metric columns (§9.3).

Crashed trials (e.g. NaN policy divergence at aggressive learning rates) are
**penalized to the study's worst completed value** rather than marked failed:
a failed trial carries no signal, so TPE would happily resample the divergent
region — observed as 8 consecutive failures next to a high-scoring trial
before this behavior was added.

## Limitations / future work

- **No mid-trial pruning yet.** Train scripts are single-shot subprocesses,
  so Hyperband/median pruning has nowhere to report from. Once train scripts
  support periodic-eval callbacks, wiring Optuna's pruners in makes cheap
  trials cheaper still.
- One algorithm space so far (`ppo`, used for PPO/MaskablePPO). Add entries
  to `spaces.SPACES` for new algorithms.

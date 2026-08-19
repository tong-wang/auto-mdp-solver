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
python -m mdp_tuning plugin/skills/mdp-solver/examples/dynamic_pricing -s simple --metric revenue_mean \
    --n-trials 25 --total-timesteps 200000 --eval-seeds 500

# what would be tuned, no training
python -m mdp_tuning plugin/skills/mdp-solver/examples/dynamic_pricing --show-space
# best trials so far
python -m mdp_tuning plugin/skills/mdp-solver/examples/dynamic_pricing -s simple --summary-only
```

Domains may ship a thin `{domain}_ppo_tune.py` wrapper that pre-fills
domain-appropriate defaults (see
`plugin/skills/mdp-solver/examples/dynamic_pricing/dynamic_pricing_ppo_tune.py`).

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
study measures what tuning adds over the L1 centre (§8.6). A default the space
cannot encode — most often a `net_arch` tuple matching no declared shape — is
dropped and *sampled* instead, which silently makes trial 0 something other
than L1; the launch banner now warns and names the value, so the study is not
read as a Δ(L2−L1) it did not measure.

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

# mdp_gates

Executable Stage-4 gate for the MDP solver pipeline (`skills/mdp-solver/SKILL.md`
Stage 4, spec §13): compares a candidate's spec-§9 eval TSV against baseline TSVs on
a `*_mean` metric, with standard errors from the matching `*_var` column and
the shared seed count.

```bash
python -m mdp_gates \
  --candidate dynamic_pricing/results/simple/PPO_.../ppo_eval_simple.tsv \
  --baseline  dynamic_pricing/results/simple/benchmark/benchmark_random_eval_simple.tsv \
  --baseline  dynamic_pricing/results/simple/benchmark/benchmark_myopic_eval_simple.tsv \
  --reference dynamic_pricing/results/simple/benchmark/benchmark_dp_eval_simple.tsv \
  --n-seeds 8192
```

- **PASS (exit 0)** when the candidate beats every `--baseline` by
  `z >= --z` (default 2.0) standard errors, where
  `z = (cand - base) / sqrt((var_cand + var_base) / n_seeds)`.
- `--reference` files (e.g. the DP optimum) are reported — gap and % of
  reference — but never gate.
- All TSVs must use the same seed protocol (seeds `0..n-1`, spec §9.2).
  Shared seeds make the unpaired SE conservative, so PASS is trustworthy.
- Comment lines starting with `#` are skipped; multi-row grid TSVs are
  aggregated by averaging means/variances across rows.

# secretary — classical secretary problem

This case formalizes the classical no-information secretary problem from the
user's verbal brief. It was generated from `secretary_schema.json` by
auto-mdp-solver.

**TL;DR**

- **PPO reaches the exact optimum.** The selected L1 checkpoint scores `0.371826 ± 0.005340 SE` over 8,192 episodes, with zero measured gap to backward induction.
- **The classical structure is recovered exactly.** PPO rejects 37 candidates, then accepts precisely the next record; it agrees with the reference on every valid nonterminal observation state.
- **The fitted rule, network, and DP are episode-identical.** All three produce the same outcomes and selected ranks on the complete protocol seed block.

## The problem

One hundred candidates arrive in a uniformly random order. At each arrival,
the decision-maker sees the candidate's position and rank relative to
candidates already seen, then must reject or irrevocably accept. Rejecting all
earlier candidates forces selection of candidate 100.

The objective is the undiscounted probability of selecting the unique overall
best candidate. Absolute ranks and the future order are hidden. A poor early
acceptance ends the episode, while waiting too long risks losing the best
candidate forever; this stopping tradeoff is the entire difficulty.

The finite-horizon optimum is available by backward induction. Its familiar
form rejects an initial prefix and accepts the next record, with a cutoff near
`N/e`. For `N=100`, exact finite-horizon optimization rejects 37 candidates
and then accepts the next record; its theoretical success probability is
`0.371042778713`. The exact DP and the independently computed finite-`N`
threshold rule are the references for the RL policy.

**Source.** The classical secretary problem as specified in the user's verbal
brief, with `N=100`, relative-rank-only observations, and forced final
selection. The authoritative statement is `secretary_schema.json`;
`secretary.restatement.md` renders it in prose.

The benchmark is anchored to Gilbert and Mosteller, “Recognizing the Maximum
of a Sequence,” *JASA* 61(313), 1966,
[doi:10.1080/01621459.1966.10502008](https://doi.org/10.1080/01621459.1966.10502008).
Lindley’s earlier dynamic-programming treatment is “Dynamic Programming and
Decision Theory,” *Applied Statistics* 10(1), 1961,
[doi:10.2307/2985407](https://doi.org/10.2307/2985407).

## Layout

Documents — where to read about the case:

| doc | what it holds |
|---|---|
| `secretary.restatement.md` | the Phase-A restatement: the problem as the IR states it |
| `ESCALATION.md` | campaign map, IR correction history, configuration origins, and evidence ledger |
| `INTERPRET.md` | anchored confirmation readback of the learned cutoff-37 structure |
| `PLAYBOOK.md` | case-close digest of reusable modeling, training, and interpretation lessons |
| `figures/secretary_policy_overlay.svg` | committed PPO-versus-exact structural overlay |
| `CLAUDE.md` | operating brief for an agent changing this folder |

Code — where things are implemented:

| module | role |
|---|---|
| `secretary_schema.json` | **the authoritative IR** |
| `secretary.signoff.json` | durable Phase-A fingerprint sign-off |
| `secretary_scenarios.py` | setting-only scenario objects and `SCENARIOS` |
| `secretary_mdp.py` | reset-time permutation draw, episode state, and deterministic transitions |
| `secretary_gym.py` | relative-rank Gymnasium wrapper |
| `secretary_ir_adapter.py` | interpreter-to-domain differential bridge |
| `secretary.runplan.json` | confirmed `standard` specialist pass; no automatic L2+ escalation |
| `secretary_ppo_train.py` / `secretary_ppo_eval.py` | leveled PPO training, checkpointing, and shared-protocol evaluation |
| `secretary_benchmark_dp.py` / `_eval.py` | exact backward-induction solution and protocol evaluator |
| `secretary_benchmark_threshold.py` / `_eval.py` | literature-optimal finite-`N` threshold and evaluator |
| `secretary_benchmark_random.py` / `_eval.py` | seeded random-action floor and evaluator |
| `secretary_benchmark_common.py` | shared CRN protocol and TSV writer for all benchmarks |
| `secretary_policy.py` | deployable selected-model wrapper with normalization and raw-MDP smoke test |
| `secretary_policy_probe.py` | complete action-surface and feature-sensitivity probe |
| `secretary_plot_policy.py` | static and interactive threshold-overlay renderer |
| `secretary_benchmark_fitted.py` / `_eval.py` | fitted cutoff-rule artifact and protocol evaluator |
| `secretary_test.py` | engine laws, differential, negative control, and domain claims |

## Results

### Protocol

The confirmed run plan targets `standard` with a specialist policy and grants
no automatic L2+ rounds. Benchmark and PPO protocol evaluations use
the shared episode-seed block `0..8191` (`n=8192`), enabling paired comparisons.
`success_mean` is the sole objective and gate metric; expected selected rank is
report-only. PPO record evaluation uses deterministic argmax actions, matching
deployment; stochastic sampling is not used for records.

### `1-comparative` — how does RL compare with the existing solutions?

The scenarios, and the role each plays in the study:

| scenario | config | what it is for |
|---|---|---|
| `standard` | `N=100`; fresh uniform order at reset; relative-rank observation | the sole specialist training and leaderboard target |

There is one leaderboard. Results use the shared 8,192-seed block. `± SE` is
computed from the Bernoulli success variance; `Δ vs DP` is success-probability
difference, and `z` divides that difference by the conservative unpaired SE.
The table is sorted by success mean.

| arm | role | success mean ± SE | Δ vs DP | significance | expected selected rank |
|---|---|---:|---:|---:|---:|
| exact backward-induction DP | exact reference | 0.371826 ± 0.005340 | 0 | identical | 20.0392 |
| exact finite-`N` threshold (skip 37) | exact reference | 0.371826 ± 0.005340 | 0 | paired-identical | 20.0392 |
| selected PPO L1 checkpoint (1.8M steps) | feasible learned policy | 0.371826 ± 0.005340 | 0 | paired-identical | 20.0392 |
| fitted PPO rule (skip 37) | feasible readback | 0.371826 ± 0.005340 | 0 | paired-identical | 20.0392 |
| independent fair random actions | feasible floor | 0.008789 ± 0.001031 | −0.363037 | z = −66.75 | 50.4430 |
| PPO L0 terminal control (2M steps) | reporting-only control | 0.000000 ± 0.000000 | −0.371826 | z = −69.63 | 76.9453 |

The DP, threshold, selected PPO, and fitted-rule arms agree
episode-for-episode on all 8,192 seeds. Their Monte Carlo mean is consistent
with the analytic optimum `0.371042778713`. L1 beats the random baseline by
66.75 SE and passes the campaign gate. L0 is reporting-only and never gates.

**Shipped: `standard_ppo.zip` from the 1.8M-step L1 checkpoint** — it won the
disjoint checkpoint-selection screen, then tied the exact reference on the
protocol block. The 2M terminal network remains beside it as
`standard_ppo_final.zip`; selection and terminal artifacts are not conflated.
The compact fitted cutoff-37 rule is also reproducible, but the trained
artifact ships because this campaign’s deliverable is the learned policy.

How it was reached: `ESCALATION.md` §MAP and findings #E1–#E4.

### `2-structural` — confirm the classical threshold

The declared `confirm` claim is satisfied exactly. Across all 4,950 valid
nonterminal observation states, the learned policy agrees 100% with the exact
rule: reject the first 37 candidates, then accept the next record, with zero
non-record acceptance. The fitted rule, raw network, and DP produce identical
protocol outcomes. See `INTERPRET.md` and the committed overlay in
`figures/secretary_policy_overlay.svg`.

## Technical appendix

Run from `secretary/` with the repository virtual environment. Pinning BLAS
threads is required for training and evaluation throughput. Together these
commands reproduce `results/` from an empty folder.

```bash
# build and evaluate the exact references and random floor
../.venv/bin/python secretary_benchmark_dp.py -s standard
../.venv/bin/python secretary_benchmark_dp_eval.py -s standard --n-seeds 8192 --first-seed 0
../.venv/bin/python secretary_benchmark_threshold_eval.py -s standard --n-seeds 8192 --first-seed 0
../.venv/bin/python secretary_benchmark_random_eval.py -s standard --n-seeds 8192 --first-seed 0

# train L1 and faithful-default L0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python secretary_ppo_train.py -s standard --level L1 --tag backbone
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python secretary_ppo_train.py -s standard --level L0 --learning-rate 0.0003 --lr-final 0.0003 --clip-init 0.2 --clip-final 0.2 --n-envs 1 --n-steps 2048 --batch-size 64 --gae-lambda 0.95 --ent-coef 0 --target-kl 0 --no-norm-obs --no-norm-reward --checkpoint-every-frac 0 --tag control

# evaluate the selected artifact after the disjoint checkpoint screen
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python secretary_ppo_eval.py -s standard --model-path results/standard/PPO_20260922_161352_L1_obsrelative_actaccept_rewsuccess_backbone/standard_ppo.zip --n-seeds 8192 --first-seed 0

# readback, fitted-rule score, figure, and deployment smoke test
../.venv/bin/python secretary_policy_probe.py --model-path results/standard/PPO_20260922_161352_L1_obsrelative_actaccept_rewsuccess_backbone/standard_ppo.zip
../.venv/bin/python secretary_benchmark_fitted_eval.py -s standard --fit-path results/standard/PPO_20260922_161352_L1_obsrelative_actaccept_rewsuccess_backbone/interpret/fitted_rule.json --n-seeds 8192 --first-seed 0
MPLCONFIGDIR=/tmp/secretary-mpl ../.venv/bin/python secretary_plot_policy.py --model-path results/standard/PPO_20260922_161352_L1_obsrelative_actaccept_rewsuccess_backbone/standard_ppo.zip
../.venv/bin/python secretary_policy.py --model-path results/standard/PPO_20260922_161352_L1_obsrelative_actaccept_rewsuccess_backbone/standard_ppo.zip --episodes 5
```

The gate commands are in `CLAUDE.md`.

# Secretary MDP restatement

**Status:** awaiting Phase-A sign-off
**MDP fingerprint:** `36aa528b6f34`
**Schema:** `secretary/secretary_schema.json`

## Problem

There are exactly 100 candidates. The scenario contains only setting
hyperparameters, principally `n_candidates=100`. When a new episode is
initialized or reset, nature draws a fresh uniformly random permutation of the
candidates' distinct absolute ranks 1 through 100, where rank 1 is the unique
overall best. This realized arrival order is stored in that episode's hidden
state and is never part of the scenario. The complete permutation is hidden
from the decision-maker.

Candidates arrive one at a time. At each arrival, the decision-maker sees the
candidate's position and relative rank among all candidates seen so far. An
absolute rank, the future order, and all other hidden ranks are unavailable.
The decision is binary: reject or irrevocably accept the current candidate.
Accepting any candidate is legal, including a candidate who is not a record;
such a choice cannot win. If the first 99 candidates are rejected, candidate
100 is selected even if the final action says reject, so every episode selects
exactly one candidate.

The primary objective is to maximize the undiscounted probability of selecting
the overall-best candidate. A successful selection pays 1 and every other
selection pays 0. The selected candidate's absolute rank is also reported and
averaged across evaluation episodes, but this bystander metric never affects
training, model selection, or gates.

## Mode and research stance

There are no categorical scenario modes to compare or split into independent
branches. This campaign trains one specialist policy for the fixed 100-candidate
setting and evaluates it on fresh random permutations from that same setting.

The declared structural stance is **confirm**: determine whether PPO recovers
the known classical skip-then-select policy—reject an initial prefix, then
accept the next record—with its finite-horizon cutoff near N/e. Stage 5 therefore
owes a policy probe and `INTERPRET.md`, including a fitted cutoff and paired
scoring against the exact threshold reference.

## Randomness classification

The complete arrival permutation is the one genuine random object. It is an
**episode-initialization draw**, sampled by initialization/reset and stored as
the latent `arrival_order` state variable. It is not a scenario field. A
deterministic per-period transition merely reveals the appropriate entry from
that already-drawn state; it introduces no additional randomness. There is no
training-only distribution over problem variants and no within-episode regime.

The v0.11.4 IR interpreter represents a reset-time random initializer through
its world-sampler mechanism: it draws `initial_arrival_order` and immediately
copies that value into the initial latent state. This is an interpreter encoding
detail, not domain ownership; generated domain code must perform the draw in
`init_state`/`reset`, while scenario objects retain only setting hyperparameters.

## State, decisions, and horizon

- Horizon: 100 arrivals, indexed 0 through 99 internally.
- Observable decision state: time-to-go and current relative rank.
- Hidden episode state: the realized absolute-rank permutation
  `arrival_order`, initialized afresh on reset.
- Reporting data: the current absolute rank and selected absolute rank; these
  are never included in the policy observation.
- Decision: discrete `accept` in `{0,1}`, with `0=reject`, `1=accept`.
- Decision bounds and type: human-confirmed.
- Termination: immediately after selection; rejecting at the last arrival
  forces selection and reaches the horizon termination simultaneously.
- Policy memory: no recurrent state or frame stack. Position and current
  relative rank are a sufficient decision state. This is a human override of
  the conservative schema heuristic triggered by a hidden episode-level draw.

## Invariants

The interpreter checks these claims after every decision:

1. The displayed relative rank equals the candidate's rank among all arrivals
   observed through the current position.
2. Selection occurs exactly when `accept=1` or the current candidate is the
   final candidate.
3. A selection records exactly the current candidate's hidden absolute rank;
   a rejection records no selected rank.
4. Success is 1 exactly when the selected candidate has absolute rank 1.

The sample trajectory below reports no invariant violations.

## Assumptions and provenance

- `N=100`: human-confirmed specialist design point.
- Uniform fresh permutation per episode, drawn during initialization/reset and
  owned by episode state rather than scenario: human-corrected and confirmed.
- Relative-rank-only observation: human-confirmed information structure.
- Accepting non-record candidates remains legal: human-confirmed.
- Forced final selection: human-confirmed.
- No intrinsic discount (`beta=1`): human-confirmed.
- Expected selected rank is report-only: human-confirmed after declining it as
  the primary objective.
- No recurrent memory: human override, justified by the sufficient current
  observation.
- PPO and observation normalization are mutable Phase-B defaults rather than
  frozen claims about the decision problem.

## Annotated sample trajectory

This replay holds `accept=0`, so all candidates are rejected until the final
candidate is forced. `decision_relative_rank` is what the policy saw;
`candidate_rank` and `selected_rank` are hidden diagnostic fields. The complete
100-period episode is rendered by the command, while the CLI displays the first
3 rows and its exact truncation marker. The repeated `arrival_order` column is
the hidden state retained throughout the episode; it is shown only for audit.

```step7b
.venv/bin/python -m mdp_ir.interpreter secretary/secretary_schema.json --decision accept=0 --episode-seed 3 --max-rows 3
```

```
secretary v0.4  episode_seed=3
t  accept  decision_relative_rank  candidate_rank  selected_rank  relative_rank  selected                                                                                                                                                                                                                                                                                          arrival_order  success  total  reward
0    0.00                       1               3              0              2         0  [3,91,12,40,67,58,51,44,11,2,78,60,39,35,89,4,24,47,37,73,28,18,42,27,33,32,15,14,23,59,41,84,99,49,87,20,95,45,63,16,36,76,71,22,81,74,65,96,29,43,90,6,77,61,21,83,85,48,86,17,53,68,19,64,75,38,93,1,54,10,88,46,69,80,97,34,94,25,56,55,79,70,57,8,50,13,72,82,26,31,52,30,66,7,5,9,100,92,62,98]     0.00   0.00    0.00
1    0.00                       2              91              0              2         0  [3,91,12,40,67,58,51,44,11,2,78,60,39,35,89,4,24,47,37,73,28,18,42,27,33,32,15,14,23,59,41,84,99,49,87,20,95,45,63,16,36,76,71,22,81,74,65,96,29,43,90,6,77,61,21,83,85,48,86,17,53,68,19,64,75,38,93,1,54,10,88,46,69,80,97,34,94,25,56,55,79,70,57,8,50,13,72,82,26,31,52,30,66,7,5,9,100,92,62,98]     0.00   0.00    0.00
2    0.00                       2              12              0              3         0  [3,91,12,40,67,58,51,44,11,2,78,60,39,35,89,4,24,47,37,73,28,18,42,27,33,32,15,14,23,59,41,84,99,49,87,20,95,45,63,16,36,76,71,22,81,74,65,96,29,43,90,6,77,61,21,83,85,48,86,17,53,68,19,64,75,38,93,1,54,10,88,46,69,80,97,34,94,25,56,55,79,70,57,8,50,13,72,82,26,31,52,30,66,7,5,9,100,92,62,98]     0.00   0.00    0.00
... (97 more periods)
episode total = 0.00   reward total = 0.00   periods = 100 (terminated early)
eval metrics: expected_selected_rank = 98.00
```

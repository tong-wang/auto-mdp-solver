# Secretary case playbook

Structure class: finite-horizon stopping with a sparse terminal success reward,
relative-rank observations, and a known threshold optimum. Unless stated
otherwise, evidence uses the shared 8,192-episode seed block `0..8191`,
deterministic PPO actions, and the exact backward-induction policy as reference.

## Lever entries

### LV1 — derive long credit and exploration before training sparse stopping tasks

context: The standard `N=100` specialist has a binary terminal reward and the
consequence of rejecting a record can span most of the remaining horizon.

symptom: Faithful SB3 defaults produced zero successes under deterministic
protocol evaluation after 2M transitions; its selected-rank mean was 76.9453.

diagnosis: The default configuration combines short GAE credit, one correlated
environment, no entropy coefficient, and no normalization. The one-shot L1
derivation instead used `gae_lambda=0.99`, four environments, a 2,048-transition
rollout, entropy `0.01`, noisy-reward learning-rate decay, and normalization.

prescription: For sparse finite-horizon stopping problems, derive credit length
from when rejection consequences realize, guarantee multiple episodes per
rollout, and protect exploration before considering a search. Here the single
L1 configuration reached the exact `0.371826 ± 0.005340` protocol score with no
L2 trials.

failed: The L0 faithful-default control scored `0.000000` at the same protocol.

evidence: [#E3](ESCALATION.md#E3), [#E4](ESCALATION.md#E4).

### LV2 — fit the predicted rule over the complete valid state grid

context: The campaign declared `confirm` for the classical skip-then-record
structure and had an exact DP reference.

symptom: Outcome optimality alone could not show whether PPO implemented the
known rule or an outcome-equivalent irregular surface.

diagnosis: A complete sweep over all 4,950 valid nonterminal observations found
one record-action transition, at period 37; non-record acceptance was zero.

prescription: In low-dimensional stopping problems, sweep every valid state,
fit the threshold, and score the fitted rule paired with the network and exact
reference. Here action agreement and protocol outcome agreement were both 100%.

failed: None; the raw network was already perfectly threshold-structured, so
fitting neither improved nor degraded it.

evidence: [#E3](ESCALATION.md#E3).

## Frame moves

### FM1 — bind the exact bar before spending PPO compute

The campaign implemented the literature formula and an independent backward
recursion first. Their common skip count 37 and byte-identical 8,192-seed
outcomes validated both the simulator and the later policy probe. This turned
“PPO looks near 37%” into the much stronger paired claim “PPO is episode-identical
to the optimum.” Evidence: [#E1](ESCALATION.md#E1).

## Modeling rules

### MR1 — use the scenario-sampler contract for reset-time world draws

When the IR declares a scenario sampler, place its callable source in
`SCENARIOS`, give it the declared `substream_id`, and realize one concrete
scenario per reset with `meta_key()`. Initialization may then copy the realized
world into episode state for deterministic transitions; it must not perform a
second draw. Evidence: [F2](ESCALATION.md#f2).

## What did not transfer

The generic belief that PPO’s faithful library defaults are a useful learning
baseline did not transfer to this sparse stopping problem. L0 remains valuable
as a reporting ruler, but it was not a viable policy here: zero protocol
successes versus exact-optimal L1 after the same 2M-transition budget.

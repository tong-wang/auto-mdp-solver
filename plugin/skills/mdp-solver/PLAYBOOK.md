# Escalation playbook — curated index

Distilled experiences from completed escalation campaigns, indexed for
solve-time consultation. **Entries are experiences, not rules**: each one
happened somewhere, under a protocol, with scope conditions. Match on
*symptom*, read the matched entry **with its context**, and analogize — the
scope conditions decide whether it applies here; the recurrence marks say how
often it has transferred so far. Entries confirmed across independent
campaigns may graduate into rules (a Phase-A checklist line, a spec default) —
that is a separate, rarer promotion; this file is the pool they graduate from.

Entries land here by maintainer curation, always **by link**: a promoted
example ships its in-folder `PLAYBOOK.md` (`examples/<name>/PLAYBOOK.md`,
digest format per ESCALATION_LOG_GUIDE §10 — context / symptom / diagnosis /
prescription / failed attempts); the index row carries the id, a one-line
hook, and the link. The example's `ESCALATION.md` is the drill-down when an
entry's `#E` citations need chasing — never a default read. There are no
inline entries: a case too sensitive to ship keeps its playbook at home.

Index format — one section per structure class, one row per entry:

```markdown
## <structure class, one sentence>
| id | hook | where | confirmed |
|---|---|---|---|
```

## Sequential choice among K exchangeable stochastic options under a per-episode latent, exactly one observed per period, finite undiscounted horizon

From `examples/mab/PLAYBOOK.md` (campaign closed 2026-08-11, promoted
2026-08-13). Protocol and scope conditions are stated in that file's header;
every number below is Gaussian-branch `gauss_K10_T1000` unless the entry says
otherwise. Read the entry before analogizing — several are bounded to this
structure class in ways the hook cannot carry.

| id | hook | where | confirmed |
|---|---|---|---|
| LV1 | hand the agent the posterior, not the raw sufficient statistic | `mab` | 1 |
| LV2 | equivariance is an optimization prior, not extra capacity | `mab` | 1 |
| LV3 | drive the entropy bonus to ~0 when exploration IS the task | `mab` | 1 |
| LV4 | state the credit horizon as coverage of the episode, not as steps — **mechanism only, the numeric band was refuted by a second campaign** (spec §8.6) | `mab` | 2 (opposed) |
| LV5 | read effects in the decision units, not the logged units | `mab` | 1 |
| LV6 | periodic checkpoint selection pays; plateau early-stopping does not — **graduated into spec §8.6/§9.7 as the post-hoc three-layer design** | `mab` | 1 |
| LV7 | harvest a tuner's centre, not its ranking | `mab` | 1 |
| LV8 | distil the net into a rule, then SCORE the rule as a policy | `mab` | 1 |
| LV9 | a fixed-quantile index is inconsistent: measure its envelope before shipping | `mab` | 1 |
| LV10 | one policy over an episode-length grid, and what a scalar HP cannot do | `mab` | 1 |
| LV11 | a never-tried-best-option tail is usually an optimizer failure | `mab` | 1 |
| LV12 | licensing a second simulator — **graduated into spec §1.2's second-implementation gates** | `mab` | 1 |

Modeling rules (formalize-time, checked at Phase A rather than matched on a
symptom): MR1 a decision that selects *which* exogenous stream is read must
put the selector in the seed key; MR2 a hidden latent with a finite sufficient
statistic; MR3 a numerical attribute that sets a dimension; MR4 when a size
axis exists, check which sites can follow it (**graduated: v0.7.0 lets state
bounds name a scenario constant**).

Frame moves (FM1–FM7) stay campaign-process material and live with the
escalation guide's vocabulary, not here.

# Competition Plan — problem/solution separation, contributions, leaderboards

*Added 2026-07-27. Companion to AGENT_PLAN.md; decisions for the mid-term
(user contributions of problems and solutions) and long-term ("Kaggle for
MDP") roadmap. Builds directly on AGENT_PLAN §14 (gym-gate design) — the
visibility tags, filtered-view gym, info-completeness, and faithful-mode
eval specified there are prerequisites, not parallel work.*

## 1. Goal

Three horizons, one architecture:

- **Short term** (largely done): the agent auto/semi-auto-solves a given MDP.
- **Mid term:** split the skill; users run it fully or partially, and
  contribute back — finished cases (exists: `mdp-contribute`), **problems**
  (new), and **solutions to a posted problem** (new). Objective: collect
  user experience, strengthen the agent.
- **Long term:** competition. One user posts a problem; others compete on
  solution performance, per-problem leaderboards. The agent is
  **competitor zero** — its auto-solve seeds every leaderboard, and any
  human submission that beats it is by construction a capability gap →
  raw material for escalation-playbook entries. The competition *is* the
  data-collection instrument for strengthening the agent.

The blocker all three share: a clean, enforceable separation of *problem*
from *solution*.

## 2. The ruling — frozen bundle as problem, canonical gym as referee, IR as certification

Rejected candidates, and why:

- **Gym as problem definition** — a gym env bundles world dynamics with
  obs featurization, reward shaping, and action encoding; the latter three
  are solution work (AGENT_PLAN §14.1). Also `_gym` alone declares a
  *family* (mode menus × scenario space), not an instance.
- **IR (or `_mdp`) alone as problem definition** — too open: `_mdp` is
  functional pieces (`init`/`advance`) competitors could reassemble into
  different dynamics; historically its `info` carries all secrets; and it
  too is a family, unpinned to a scenario.

Adopted — three artifacts with distinct competition roles:

| artifact | role |
|---|---|
| **frozen bundle** = domain code snapshot (`_uncertainty`/`_scenarios`/`_mdp`/`_gym`) + one scenario + pinned eval config | **the problem** — the operational ground truth; content-hashed, versioned |
| **referee runner** rolling out the bundle's canonical gym against a submitted policy | **the scorer** — unforgeable: competitors never touch it |
| **IR + differential + conformance gates** | **the certification** — semantic ground truth; a bundle is accepted only if its shipped code provably agrees with its IR |

This is the standard structure of real RL competitions (L2RPN, Learning to
Run, NetHack Challenge): canonical env + pinned config; competitors wrap it
for training but are scored on the raw env. The flexibility of `_mdp`'s
functional pieces stops mattering: solution-side, competitors may assemble
anything; the referee only ever runs the canonical bundle.

## 3. The problem bundle

A stamped release, not a living case folder (kin: a Kaggle dataset). Contents:

- **Code snapshot:** `_uncertainty`, `_scenarios`, `_mdp`, `_gym`, the IR
  (`{name}_schema.json`), `{domain}_ir_adapter.py`.
- **One scenario** — pins the family to an instance, including the target
  objective (scenario-declared target, §14.4). Where the scenario file is a
  grid, the bundle names one cell.
- **Pinned eval config** — the one cell of the gym's mode menu that scoring
  uses:
  - *Obs mode: the maximal filtered view.* Every observable quantity per the
    visibility tags, raw. Load-bearing because wrappers are one-directional —
    a competitor can project information away or re-encode it, never recover
    what the canonical env withholds.
  - *Action space: the raw decision space* (soundness required per §14.3;
    competitors' re-encodings must map back — automatic when their wrapper
    wraps the canonical env).
  - *Metric: the faithful objective computed from `info`* (§14.4 invariant),
    never the gym reward channel. Horizon/termination semantics pinned.
- **Seed protocol** (§6 below): the public dev seed list; the private-scoring
  mechanism reference (never the private seeds themselves).
- **Baseline pack:** competitor-zero entries (baselines + the agent's PPO
  policy) with scores, so no leaderboard is ever empty. Bounds
  (oracle/clairvoyant) shown but not ranked, admissible baselines ranked —
  the filtration tags make the split formal (§14.2 dividend).
- **Narrative:** the verbal problem statement — the browsable, human-legible
  face of the problem (the IR is its formal twin).
- **Fingerprint:** content hash over code + scenario + eval config. Problem
  identity and version in one value; any change is a new problem version.
  Bug found post-release → re-release under a new fingerprint with an
  explicit leaderboard-reset note; never patch in place.

**Certification gates (acceptance = all green):** IR validates; differential
passes (interpreter ≡ `_mdp` on the bundle's scenario); conformance passes;
baselines run and reproduce their claimed scores on the public seeds;
visibility tags confirmed (a wrong tag is not a modeling nit — it is an
exploitable information leak, so the information set is a first-class item
of problem review).

## 4. The submission contract

A solution = a policy object with a minimal interface, in canonical spaces:

```
Policy.reset(seed)            # referee-supplied seed; stochastic policies allowed, seeded
Policy.act(obs) -> action     # obs = canonical maximal filtered view; action = raw decision
```

- §14.2 already observed the filtered view *is* the deployable
  `{domain}_policy.py` API — the competition contract is that API, tightened
  and made referee-loadable. Pipeline users get the adapter **generated**:
  a competitor who trained in a wrapped env ships a policy that internally
  applies their obs transform and inverse action map; the referee neither
  knows nor cares. The back-and-forth mapping burden falls only on
  off-road solutions, where it belongs.
- **Wrappers are the solution-side escape hatch:** competitors may wrap the
  canonical gym in any stack of obs/reward/action wrappers for training —
  shaping, featurization, action pruning (coverage declared on the
  leaderboard row, §14.3) — with zero effect on scoring, which always runs
  the raw env and the faithful metric.
- Submission metadata: approach note, whether pipeline-built, coverage
  declaration, deps. Runtime budget per episode enforced by the referee.

## 5. The referee runner

New harness module (working name `mdp_referee`, runnable as
`python -m mdp_referee <bundle> <submission>`). Loop:

1. Load bundle, verify fingerprint; load submission policy.
2. For each seed in the eval list: seeded reset, roll out — **policy receives
   `obs` only, never `info`**. The info-leak problem is solved at the
   referee, not by sanitizing the env: locally competitors may read `info`
   freely (diagnostics, privileged critics); none of it helps at scoring
   time because the scored policy is only ever fed `obs`. One env, full
   transparency, the boundary in one small piece of referee code.
3. Referee reads `info` itself to accumulate the faithful objective
   (per-step or terminal per its timing class, §14.4).
4. Emit a score row: mean episode value over the seed list + CI, per-episode
   values (paired comparison across submissions — same seeds for everyone =
   common random numbers, variance reduction for free), runtime, fingerprint,
   coverage declaration.

Residual hole, acknowledged: a malicious submission running in-process can
reach into the env object. This is the generic untrusted-code problem every
competition has; eventual fix is process isolation between policy and env.
Early phase: review + CI on public seeds is enough (see §7).

## 6. Seed protocol — public/private split

- **Public dev seeds:** fixed list shipped in the bundle; development and the
  public leaderboard run on it.
- **Private final seeds:** held out — derived via a secret seed salt (the
  interpreter/env seed-scheme already threads a salt), so final-scoring
  episodes are unguessable without shipping any hidden file. Same private
  list for every submission (paired).
- Without the split, the leaderboard measures who overfit the eval seeds
  hardest. Public score = provisional; private score = final ranking,
  revealed at close (or on a lag, for rolling leaderboards).

## 7. Trust model and infrastructure (GitHub-first)

Start where the repo lives; graduate to hosted infrastructure only if volume
demands it.

- **Problem contribution** = PR with a bundle; case-gates CI runs the
  certification gates of §3. The trade accepted knowingly: the problem
  artifact is executable code others will run — the trust burden sits at
  problem acceptance, mitigated because pipeline-generated bundles are
  gate-checked, structurally uniform code (far easier to review than
  freeform envs). Off-pipeline bundles get stricter review.
- **Solution contribution** = PR with a submission folder; CI runs the
  referee on **public seeds only** and posts the provisional score.
- **CI security rule (hard):** untrusted PR code never runs in a privileged
  context — no `pull_request_target` with secrets exposed, private salt
  never in a workflow reachable by fork PRs. Private-seed scoring runs
  post-merge (or maintainer-triggered) in a separate workflow holding the
  salt as a secret.
- **Leaderboard** = per-problem file in the repo (TSV/MD), updated by the
  scoring workflow, keyed by bundle fingerprint. Rows: submission, public
  score ± CI, private score (when revealed), coverage declaration,
  admissible-vs-bound flag, pipeline-built flag.
- **Reproducibility:** referee env pinned (Python + dep lockfile per bundle,
  CPU-only eval); score rows carry the referee version.

## 8. Contribution modes — the full set

| mode | artifact | gate | exists today |
|---|---|---|---|
| case | finished end-to-end case | case-gates CI | `mdp-contribute` |
| playbook entry | distilled escalation lessons | review | `mdp-contribute` |
| **problem** | frozen bundle (§3) | certification gates | new |
| **solution** | submission (§4) | referee on public seeds | new |

`mdp-contribute` grows the two new modes rather than spawning a sibling
skill — the local-assembly + explicit-approval flow carries over unchanged.

## 9. Work items (landing sites)

| item | landing site |
|---|---|
| §14.5 implementation batch (visibility tags, filtered view, info-completeness, faithful-mode eval, scenario-declared target) | **prerequisite** — see AGENT_PLAN §14.5 |
| bundle format + fingerprint tool (`freeze` — content hash, eval-config pinning) | new `harness/mdp_bundle` (or `mdp_conformance` extension) |
| eval contract file (`eval.json`: seeds, episodes, horizon, metric ref, baseline scores, fingerprint) | bundle format; spec appendix |
| referee runner (`python -m mdp_referee <bundle> <submission>`) | new `harness/mdp_referee` |
| submission contract: `{domain}_policy.py` → referee-loadable interface; generated adapter for wrapped-env policies | spec §policy + skill codegen |
| certification gate composition (IR + differential + conformance + baseline-reproduction) as one command | case-gates CI + `mdp_conformance` |
| leaderboard file conventions (coverage flag, admissible/bound, pipeline-built) | spec leaderboard conventions + scoring workflow |
| public/private seed workflows; CI security split | `.github/workflows/` |
| `mdp-contribute` problem/solution modes | `plugin/skills/mdp-contribute/SKILL.md` |

## 10. Sequencing

1. **§14.5 batch first** — without visibility tags and the filtered view,
   the canonical obs leaks; without faithful-mode eval, the metric is
   gameable via reward modes. The competition is the §14 machinery's
   second customer (headless escalation was the first).
2. Bundle format + referee runner + submission contract (harness work,
   testable on the frozen examples — each example becomes a trial bundle).
3. `mdp-contribute` modes + CI workflows (mid-term contributions live).
4. Leaderboards + private-seed scoring (competition proper).

Steps 2–4 also depend on the skill split (AGENT_PLAN §3–4) only loosely —
the referee and bundle tooling are pure harness and can land independently.

## 11. Open items to verify (do not assume)

- Whether the current seed scheme lets a single secret salt derive a private
  episode list for the *gym* path exactly as for the interpreter path
  (seed-key construction is shared, but verify end-to-end).
- GitHub Actions mechanics for the private-salt workflow (environment
  protection rules, fork-PR secret isolation) — verify against current GH
  docs before building (kin to AGENT_PLAN §13's rule).
- Runtime limits / process isolation options for untrusted submissions when
  volume outgrows review (nsjail, containers, or hosted eval).
- Whether per-bundle dep lockfiles are enough for bit-level score
  reproducibility across machines (BLAS nondeterminism on CPU is usually
  fine; verify on one example bundle).

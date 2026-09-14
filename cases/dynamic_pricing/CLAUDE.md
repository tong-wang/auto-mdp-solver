# CLAUDE.md — `dynamic_pricing/` (finite-horizon dynamic pricing of a fixed stock)

Domain-local operating brief for an agent about to change this folder.
**This file points up: at the specs that govern the folder, and the version
of them its numbers were produced under.** Everything the campaign produced —
its documents, its code, its numbers — is inventoried in `README.md` and is
linked from here, never copied.

Sell a fixed initial stock over a finite selling horizon by setting a price
each period (Gallego & van Ryzin 1994). Demand arrives as Poisson with an
intensity that falls exponentially in price; sales are capped by remaining
stock, there is no reorder and no backlog, and the objective is expected
revenue. State is `(period, inventory)`; the decision is one continuous price.

**`simple` (a=100) and `ample_stock` (a=20) are SEPARATE LEADERBOARDS** — two
demand scales, two bars, incomparable revenues. Their scores must never be
compared. Both are fixed entries in `SCENARIOS`; there is no grid.

## The specs that govern this folder

`dynamic_pricing/` was produced by the **auto-mdp-solver** skill from
`dynamic_pricing_schema.json`. It is not a hand-written project, and it must
not drift into one.

**Before changing anything here, read the skill docs — they are authoritative
over this file, over the code, and over local habit:**

| doc | governs |
|---|---|
| `MDP_PROJECT_SPEC.md` | architecture, layering, naming, RNG/seed tree, script + eval conventions (§1–§14) |
| `CONTRACTS.md` + the step skills (`mdp-formalize` … `mdp-package`) | the pipeline ops, their entry gates (`python -m mdp_stage`), and what each op must emit |
| `ESCALATION_LOG_GUIDE.md` | the campaign log (§MAP, §FRAME-CHANGELOG, §IR-CHANGELOG, §CONFIG-REGISTRY, §LEDGER) and the §10 playbook digest |

All of these live **in the `mdp-solver` skill's own directory** (the step
skills are its siblings) — in this repo, `plugin/skills/mdp-solver/`. This
case lives **inside** the auto-mdp-solver repo, so the harness it builds
against is the repo's own `harness/` at whatever revision is checked out;
the two provenance lines below say which revisions its numbers and its
declarations belong to. Never trust a version number written in a document
over `git describe --tags`.

Solver provenance — **two versions, two meanings** (keep both; they answer
different questions, and a case that is later contributed upstream needs the
distinction — see `cases/README.md`, "After a case is merged"):

- **Built and gated at: no release tag** — every RL number in this folder was
  produced on 2026-07-09 in the author's research workspace (commit
  `86d7385`), under the in-tree, pre-split `mdp_ir` at IR v0.4 and the
  four-word seed key that preceded the v2 seed tree. No run artifact from
  that campaign survives in this folder. This line never moves. Moving it
  would claim the results were re-measured.
- **Conformance maintained through: v0.10.11** (`5f6c8d1`) — re-gated
  2026-09-14 from the repo root at the pinned worktree: IR OK, `mdp`
  fingerprint `a1bf3475a7da`, structural `5b5d30fa24b8`; conformance 22/30
  with no FAIL (two WARNs, both recorded in `ESCALATION.md` §MAP); laws 7/9,
  both un-passed are SKIPs; differential MATCH on both instances; 9 domain
  tests. How far declarations, drawing conventions and gate compatibility
  have been carried forward: the folder travelled upstream from 2026-07-21 to
  2026-09-14 through conformance edits only (seed tree v2, the
  `benchmark_{method}` naming, declared `benchmarks`, the §8.6 derivation as
  data, the launch check) with no number re-run, and the two documents came
  to v0.10.11's §1.3 shapes on 2026-09-14. Every move is logged in
  `ESCALATION.md` §FRAME-CHANGELOG.

Upgrading: read the tag's notes, re-run every gate below at the new
revision, then log the move. The author's working copy lives downstream (the
`dynamic_pricing/` folder of a research repo) and re-ports here; a release
that invalidates a statement in this folder names it in its commit body
rather than editing it (repo `CLAUDE.md`, Rules).

The repo-root `CLAUDE.md` carries the cross-case rules. `README.md` carries
the case: what the problem is, what is in the folder, what the arms scored,
and the commands that reproduce them.

## Hard rules

Each fires at a moment and names where the answer lives. None of them states
the answer — that is what makes them safe to keep current.

- **Before inventing a mechanism, check the spec.** The pipeline usually has
  one already, and its version composes with the gates.
- **Structural changes go through the IR first**, then the code, then the
  gates — never code-first. `dynamic_pricing_schema.json` is the source of
  truth; `dynamic_pricing_scenarios.py` and friends materialize it.
  Observation/architecture levers are NOT IR changes — they live in the gym /
  train script with their own executable gate; only the *problem* goes
  through the schema.
- **Re-run the gates after any IR change and after any solver bump** — the
  IR depends on host behavior, not just on this folder. A moved fingerprint
  owes a §IR-CHANGELOG entry (guide §5, tripwire 2).
- **Before quoting a number, say which artifact produced it.** Selection and
  terminal networks are different objects (`{scenario}_ppo.zip` vs
  `checkpoints/*.zip`, spec §8.4), and the protocol every number is quoted
  at — including which eval mode is the record here, and which rows are the
  2026-07 record versus the 2026-09-14 re-measurement — is stated once in
  `README.md`'s results section. The two sets are on different seed keys;
  never subtract across them.
- **Quoting the bar** → `README.md` Protocol says which of the two DP numbers
  is the bar: the solver's computed value or its evaluated row.
- **Choosing an action mode for a run** → the IR's `gym.action_modes` and the
  `ESCALATION.md` entry that measured the two (`#E3`, and the §MAP readings).
- **Launching a run** → cite a `§CONFIG-REGISTRY` address (`sc0/g0/a0/h0` is
  the origin; `sc0/g0/a0/h1` the crowned tuple). No
  `dynamic_pricing_configs.py` exists yet; the registry's residue says it is
  owed the moment an arm is launched here again.
- **Editing `mdp.scenario.constants`** — including removing the two flags
  `schema.no_enumeration` WARNs on — → moves the `mdp` fingerprint and voids
  any sign-off; `ESCALATION.md` §IR-CHANGELOG owes the F-entry first.
- **Confirming a `Confirmable` or writing `dynamic_pricing.signoff.json`** →
  `mdp-formalize` steps 6–9 and `CONTRACTS.md`, "Pre-split domains". The two
  `derived` fields and the missing state files are listed in `ESCALATION.md`
  §Frontier; the sign-off is a human act, never a backfilled date.
- **Editing `_L1_DERIVED` is a re-derivation, not a default edit** (spec
  §8.4). It re-bases every Δ(L2−L1) above it, so it takes a logged basis and
  a ledger entry — see spec §8.6 for when re-deriving is the right move.
- **Writing to the log** → §FRAME-CHANGELOG, §IR-CHANGELOG and §LEDGER are
  append-only, new entries at the end; §MAP and §CONFIG-REGISTRY are living.
- **A deliberate deviation from the spec is recorded, not silent** — in the
  code comment, in `ESCALATION.md`, and, if the spec should change, as an
  upstream issue filed via the mdp-propose skill. The issue is the record and
  the log carries its URL; the draft behind it is scratch (see File hygiene).
- **Never** use `param`, `params`, or `param_*`; conformance fails.

## Gate commands

Run from `cases/dynamic_pricing/` unless noted. These are the checks; the commands
that *run* the code — train, eval, benchmark, probe, plot — are in
`README.md`'s technical appendix. Always pin threads for training — torch
oversubscribes. `python` below means the one resolved interpreter (repo
`CLAUDE.md`, Rules), never a bare interpreter.

```bash
# conformance / laws / differential run from cases/, the parent of dynamic_pricing/
python -m mdp_ir dynamic_pricing/dynamic_pricing_schema.json
python -m mdp_conformance dynamic_pricing
python -m mdp_ir.laws dynamic_pricing
python -m mdp_ir.differential dynamic_pricing/dynamic_pricing_schema.json --episodes 40 --all-instances
python -m mdp_stage dynamic_pricing            # where the folder stands; every op is BLOCKED until sign-off
cd dynamic_pricing && pytest dynamic_pricing_test.py
```

The spec §9.9 **role gate** is a check too, so it lives here. `--ir` is what
activates the role reasoning: a `feasible` arm beating the `exact` DP fails
as a bug report, and `--baseline` on the DP is refused.

```bash
python -m mdp_gates --ir dynamic_pricing_schema.json --n-seeds 8192 --metric revenue_mean --sense maximize \
  --candidate <ppo eval tsv> \
  --baseline  <fixed eval tsv> --baseline <myopic eval tsv> --baseline <random eval tsv> \
  --reference <dp eval tsv>
```

## File hygiene — the folder is the deliverable, not the workbench

A new file belongs in `dynamic_pricing/` only if it is (a) spec-§1 layout,
(b) an implementation a finding cites and someone must re-run to reproduce it
(probes, extra gates), or (c) a campaign document (`README`, `ESCALATION`,
`INTERPRET`, `PLAYBOOK`). **Everything else goes to `scratch/` (gitignored):
launchers, monitors, one-off checks, throwaway analysis. All output goes to
the gitignored output tree, never into the folder itself.**

**Round plans and upstream-proposal drafts are never tracked.** A `*_PLAN.md`
lives in `scratch/` while it is being drafted AND while it is being executed;
findings go into `ESCALATION.md` and `README.md` **as they land**, and the
plan is deleted once written up. If a plan is the only place a result exists,
that is a bug in `ESCALATION.md`. **The probe a plan drives is the opposite —
it stays, permanently**: an escalation entry cites numbers that only exist if
the code behind them can be re-run. A probe is written in `dynamic_pricing/`
from the start (it imports its siblings) and **committed in the same commit
as the escalation entry that cites it** — `git log ESCALATION.md` then shows
each finding beside the code that produced it. Design rationale that must
outlive the round has two tracked homes, neither of them the plan: the
escalation entry and the probe's module docstring.

An `UPSTREAM_PROPOSAL_*.md` is the same shape and for the same reason: it is
drafted in `scratch/`, and once the mdp-propose skill files the issue, the
issue is the proposal — the draft goes, and `ESCALATION.md` carries the issue
URL and, later, the maintainer's disposition. A proposal is pinned to a spec
version and goes stale; a tracked copy goes stale *silently*, which is how a
folder ends up asserting a claim upstream has already rejected.

Check with `git status --short dynamic_pricing/`: untracked files should be
rare and deliberate.

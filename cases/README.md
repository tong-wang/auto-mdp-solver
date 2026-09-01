# cases/ — auto-solve test cases

Each subfolder is one end-to-end run of the mdp-solver skill on a new
problem: IR + restatement (Phase A), generated domain + benchmarks + trained
policy (Phase B), and a README with the leaderboard. The purpose of this
directory is to grow the solver's capability envelope — every case should
stress something the pipeline hasn't handled before, and a case that forces
an `mdp_ir/` extension is a good case.

Trained artifacts (`results/`) are gitignored; each case README's commands
must reproduce them.

## What is here

Two version columns, and they mean different things (see the last section):
**built** is the tag a case's numbers were produced and gated against and never
changes; **gated** is the tag its conformance has been carried through.

| case | source | what it stresses | built | gated |
|---|---|---|---|---|
| `clark_scarf` — serial multi-echelon inventory | Andrew J. Clark and Herbert Scarf, "Optimal Policies for a Multi-Echelon Inventory Problem", *Management Science* **6**(4), 1960 — read from the 2004 reprint, *Management Science* **50**(12S), 1782–1790. Model is §2–§3, generalized to N levels. | A **coupled multi-stage decision**: N simultaneous shipments per period, each clipped to what its source holds, so the top of the chain anticipates demand it never sees. Carries an `exact` DP verified against a brute-force joint-state solve rather than taken on the theorems' word — the first draft was 30% suboptimal and entirely convincing. | v0.9.5 | v0.9.5 |
| `fnv` — fresh-newsvendor sequential ordering under MMFE | Tong Wang, Atalay Atasu, Mümin Kurtuluş (2012), "A Multiordering Newsvendor Model with Dynamic Forecast Evolution", *Manufacturing & Service Operations Management* **14**(3), 472–484, [doi:10.1287/msom.1120.0387](https://doi.org/10.1287/msom.1120.0387). Forecast process is the MMFE of Heath & Jackson (1994). | **Progressive information revelation**: a martingale signal refines while cost rises, so the decision is when to commit rather than how much. Two demand transforms are separate leaderboards. Demoted from `examples/` 2026-08-13 for shipping no train/eval pair; the solve leg and record were contributed after. | v0.8.0 | v0.9.5 |
| `dynamic_pricing` — finite-horizon revenue management | Guillermo Gallego and Garrett van Ryzin (1994), "Optimal Dynamic Pricing of Inventories with Stochastic Demand over Finite Horizons", *Management Science* **40**(8), 999–1020. | **Continuous price control over a depleting stock**: price sets the arrival intensity, so the decision shapes the demand it then faces, and the episode can end early when inventory hits zero. Carries an exact DP and the paper's fluid policies; PPO reaches within 0.6% of DP. Demoted from `examples/` 2026-09-01 — the shape is one `inv_single` already teaches, not a defect in the folder. Its numbers predate the version stamps and it carries no `built` tag, but unlike `adi_flex` it does meet the contract below. | — | v0.9.37 |
| `adi_flex` — inventory with advance demand information and flexible delivery | Tong Wang and Beril L. Toktay (2008), "Inventory Management with Advance Demand Information and Flexible Delivery", *Management Science* **54**(4), 716–732 — §4. | A **joint ordering + allocation** decision under **demand crossover**: orders are seen when placed but due now, next period, or two out, so stock spent shipping a not-yet-due order may be stock an urgent one needs tomorrow — the manager chooses how much to buy *and* how much to withhold. The one case whose optimum is **bracketed rather than exact**: a relaxation bounds it from below and heuristics from above — the shape the `relaxed`/`feasible` roles exist for, though this case declares no `benchmarks` block and its four benchmark files carry no role. **Predates the contribution contract** — see below. | — | v0.9.5 |

**No paper PDF is in this repository, at any revision** — the root
`.gitignore`'s `*.pdf` covers every case folder, and no PDF has ever been
committed. Read the sources via their DOI or a library. What a case reimplements
from one are mathematical facts, attributed at the call site.

Gates green at v0.9.6, **zero FAILs on all three**: conformance `clark_scarf`
18/22, `fnv` **20/23**, `adi_flex` 15/22; `mdp_ir.laws` 8/9, 7/9, 7/9. Every
un-passed law is a SKIP, and which one it is says something: on all three it is
`mixture_equivalence` (none declares a mixture), plus `path_independence` on
`fnv` and `invariants` on `adi_flex`, which declares none to check.

Every denominator gained one at v0.9.6: `schema.no_enumeration` (spec §5.0) is
new, and **all three cases WARN on it** — `adi_flex` for enumerating one demand
pipeline as `due_now`/`due_next` and one rate vector as
`lambda_now`/`lambda_next`/`lambda_later`, with a `demand_window` constant that
appears exactly once in the file and therefore controls nothing; `clark_scarf`
and `fnv` for re-baking a derived numeric vector per instance (`h_install` in
ten instances, `signal_stdevs` in one). It WARNs rather than FAILs because the
re-baking shape has no declarable alternative yet (issue #45). The counts above
are otherwise unchanged: no check that passed before stopped passing.

`fnv` moved 17/22 → 20/22 on 2026-08-17 by **declaring** what it already was —
`mdp.model`, `benchmarks` and `research_questions` — with no number re-run; its
`built` tag is therefore still v0.8.0 while its `gated` tag is v0.9.5, which is
the intended state for a case whose declarations have been carried forward and
whose results have been left alone. The three checks it turns from SKIP to PASS
are the ones a bare conformance count hides: a SKIP is not a pass, and
`model.boundary` in particular cannot fire at all until the theory layer exists.

**`adi_flex` does not meet the contract below** and is kept because it gates,
not as an exemplar of a contribution. It has no `CLAUDE.md`, no `ESCALATION.md`,
no `PLAYBOOK.md` and no `adi_flex_test.py`, so it carries no campaign record and
no claim about its own numbers — which is why it has no `built` tag. It is the
one entry here to read *last*: `clark_scarf` is the current reference for what a
case looks like.

## Contributing a case

External case contributions are welcome — they are exactly how this directory
is meant to grow. The easiest path is the **`mdp-contribute` skill** (ships
with the plugin): it assembles your case to the contract below, runs the gates,
and opens the PR via `gh` (fork-and-PR; nothing is sent without your approval).
It also supports a **re-skinned** contribution — an isomorphic rename that
shares the MDP structure while abstracting away your business context. A case
too sensitive even re-skinned is simply not contributed; there is no partial
path. Spec/schema extension proposals are a separate flow — the
**`mdp-propose` skill**, one `upstream-proposal` issue per proposal, filed
the moment a campaign hits the wall rather than at case close.

Contributing manually instead: add one folder `cases/<name>/` containing the
frozen IR (`<name>_schema.json`), the restatement, the spec-conformant domain
modules (`_uncertainty`/`_scenarios`/`_mdp`/`_gym` + `<name>_ir_adapter.py`),
benchmarks with their eval scripts, `_ppo_train`/`_ppo_eval`, `<name>_policy.py`,
the domain's `CLAUDE.md` operating brief (spec §1, from the skill's template),
and a README whose commands reproduce every leaderboard number (no `results/`).
Include your campaign record — `ESCALATION.md` and its case-close
`PLAYBOOK.md` (ESCALATION_LOG_GUIDE §10: digest entries — necessary context /
symptom / diagnosis / prescription / failed attempts, real names and numbers,
ledger cited not copied) — it is half the value. State the
provenance of the problem in the PR (paper cases: cite it; business cases:
confirm you may publish it — the confirmation covers the campaign record
too, since a promoted case ships whole). CI (`case-gates`) runs IR
validation, spec conformance, and the differential on every case dir the PR
touches; all three must be green. Negative cases — where no pipeline
deliverable beats the baselines — are accepted when stated plainly; they
stress the pipeline too (they just don't get promoted to the examples set).

## Promotion to the examples set

Promotion out of `cases/` into `plugin/skills/mdp-solver/examples/` is an
internal maintainer decision, separate from contribution — and it is also a
**publication decision**: the whole folder moves, campaign record included,
and everything under `examples/` ships with the plugin (which is the point —
the in-folder `PLAYBOOK.md` becomes reachable by the installed skill).
Criteria:

- **deliverable-competitive** — a pipeline deliverable beats the classical
  baselines at the case's protocol. The deliverable may be the trained
  policy *or* its §14 readback: a fitted rule that outperforms its own net
  is a Stage-5 success, not a negative case.
- **coverage** — the case teaches a shape none of the shipped examples
  already teaches, or is the regression case for an `mdp_ir` change. The set
  is deliberately small and there is no list of shapes it is trying to fill:
  a candidate has to displace nothing and add something.
- **no declared debt** — coverage debt is paid before the freeze; frozen
  entries take no research edits afterward.
- **gates green** on the folder as it will ship, and the case README's
  commands reproduce every number.

Mechanics: move the whole folder — never copy; two folders with the same
`{name}_*` modules break pytest's `prepend` import mode — add the manifest
row with what the entry teaches, extend its shape-coverage line, record the
promotion in the table below, and bump the plugin version. See `examples/MANIFEST.md` for the
scoped freeze (code/schema/tests frozen; campaign-record docs
maintainer-curated).

| promoted case | when | now at |
|---|---|---|
| `mab` — stochastic multi-armed bandit | 2026-08-13 | `plugin/skills/mdp-solver/examples/mab/` |
| `game2048` — the 2048 sliding-tile game (scoped subset) | 2026-09-01 | `plugin/skills/mdp-solver/examples/game2048/` |

`game2048` was contributed and promoted in **one step** (PR #73) and so never
appeared in the table above — the folder went straight to `examples/`. One of
the four criteria was waived by maintainer decision rather than met — **no
declared debt** — against which two things were owed: its §14.3 figure contract
and its `discover` fitted rule. `examples/MANIFEST.md` records the waiver and
what it costs, which is where a reader of the promoted folder will be looking.
Waived is not foreclosed: the freeze bars research edits, so a spec-contract
deliverable can still arrive through the entry's author later. Contribution and promotion staying
separate decisions is still the norm; this was an exception, taken knowingly.

Demotion is the same move in reverse, and is not a failure verdict: an
example that no longer earns its place as a *few-shot exemplar* — too
specific a problem, or incomplete as a pipeline run — returns here, where it
still gates. `fnv` moved back on 2026-08-13: at the time it shipped no
train/eval pair, so it never exercised the solve leg, and its MMFE ordering
problem is narrower than the shapes an exemplar should teach.

`dynamic_pricing` moved back on 2026-09-01 for the other reason the paragraph
above names — not incompleteness, which it never had, but redundancy: its
continuous single-entity control is the shape `inv_single` teaches, and a
few-shot set is stronger at three distinct shapes than at four overlapping
ones. It arrives here complete, with its exact DP, its fluid benchmarks and a
train/eval pair, and it gates unchanged.

That first reason no longer holds — PR #15 contributed the solve leg, the
campaign record and a §14 readback that recovers the paper's structure, and
PR #17 retracted a §14.2 improvement claim after a grid mean was found to hide
a sign flip. Both leaderboards now run to their DP. Demotion is not a standing
verdict: re-promotion is a fresh decision against the criteria above, and the
open question for `fnv` is the second reason — whether the shape is broad
enough for an exemplar — not the missing scripts.

## After a case is merged

Merging transfers **maintenance**, not authorship. The argument stays the
contributor's — their record, their verdicts, their numbers. What upstream
takes on is keeping the folder *gating* as the spec moves, and a merged case is
a regression fixture for the pipeline rather than a living project: the
contributor's own campaign continues in their repo against a pinned tag and
never needs to track this copy.

**When the spec moves under a merged case**, exactly one question decides what
happens: *can the change be made without invalidating a number?*

| | what upstream does | example |
|---|---|---|
| **yes** — declarations, drawing conventions, conformance edits | maintainer updates the case, with a changelog line in its own `ESCALATION.md` §FRAME-CHANGELOG | `mab` declaring its six `benchmarks` and two tier-2 stances (2026-08-16) |
| **no** — the fix needs the campaign re-run | **caveat it** in `examples/MANIFEST.md` or the case README; disclose, never quietly re-run someone's numbers | `mab`'s selection machinery predating v0.7.0 — every number was produced under the old design, so the code stays and the caveat says so |
| the case no longer earns its shipped place | demote (above); still gates from `cases/` | `fnv`, 2026-08-13 |

The line a maintainer does not cross: conformance edits touch code, schema,
declarations and add changelog lines. **They never rewrite a contributor's
reasoning, verdicts, or numbers.** Redrawing a campaign map to a new taxonomy
sits at the edge of that line — it changes form, not substance — and for an
externally contributed case it is worth asking the author first.

A contributor who disagrees with a conformance edit files an
`upstream-proposal` (the `mdp-propose` skill) rather than needing write access.

## Updating your own merged case

Contribution is one-shot for *ownership* and continuous for *improvement* —
returning with more work is expected, not an imposition. Which path depends on
where the case currently lives and whether numbers move:

1. **Still in `cases/`** — just open a PR against the folder, the same way the
   case arrived. Numbers may move; bring the record with them (`ESCALATION.md`
   entries, an updated `PLAYBOOK.md` if the digest changed) and make sure the
   README's commands still reproduce every number. CI re-runs `case-gates` on
   the dirs you touch. **No demotion, no re-contribution.**
2. **Promoted to `examples/`, and the change moves no number** — declarations,
   conformance to a new spec, a record correction. This lands as a maintainer
   edit under the scoped freeze; send it as a PR or an issue and it is applied
   in place.
3. **Promoted to `examples/`, and the change moves numbers or re-opens the
   campaign** — new arms, another escalation round, retuned results. **Demote
   first**: the folder moves back to `cases/`, your update lands there, and
   re-promotion is a fresh maintainer decision against the same criteria. This
   is not a demerit — it is what keeps "what shipped at tag `v0.X.Y`" a stable
   answer, which is the whole reason `examples/` is frozen. Say so in the PR
   and the demotion is part of accepting it.
4. **A different question on the same domain** — a new IR, a new campaign
   framing — is a **new case with a new name**, not an update. The old one
   keeps gating the shape it was admitted for, and two folders sharing
   `{name}_*` module names would break pytest's import mode anyway.

Cost asymmetry worth knowing before you choose: a `cases/` update ships in
neither published artifact, so it is cheap. An `examples/` update is a release
event — it bumps the plugin version and moves the tag every installed copy is
compared against.

**Every case records two versions, and they mean different things**: the tag it
was *built and gated against*, which is the provenance of its numbers and never
changes, and the tag its conformance has been *maintained through*, which the
maintainer updates. A case whose two lines differ has had its declarations
moved forward and its results left alone — which is exactly the intended state.

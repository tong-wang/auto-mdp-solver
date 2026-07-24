# cases/ — auto-solve test cases

Each subfolder is one end-to-end run of the mdp-solver skill on a new
problem: IR + restatement (Phase A), generated domain + benchmarks + trained
policy (Phase B), and a README with the leaderboard. The purpose of this
directory is to grow the solver's capability envelope — every case should
stress something the pipeline hasn't handled before, and a case that forces
an `mdp_ir/` extension is a good case.

Trained artifacts (`results/`) are gitignored; each case README's commands
must reproduce them. A case whose domain proves broadly useful as a few-shot
exemplar can be promoted to the examples set (`plugin/skills/mdp-solver/examples/`, add manifest row).

## Contributing a case

External case contributions are welcome — they are exactly how this directory
is meant to grow. The easiest path is the **`mdp-contribute` skill** (ships
with the plugin): it assembles your case to the contract below, runs the gates,
and opens the PR via `gh` (fork-and-PR; nothing is sent without your approval).
It also supports a **re-skinned** contribution — an isomorphic rename that
shares the MDP structure while abstracting away your business context — and,
separately, contributing distilled escalation-playbook entries as a structured
issue.

Contributing manually instead: add one folder `cases/<name>/` containing the
frozen IR (`<name>_schema.json`), the restatement, the spec-conformant domain
modules (`_uncertainty`/`_scenarios`/`_mdp`/`_gym` + `<name>_ir_adapter.py`),
benchmarks with their eval scripts, `_ppo_train`/`_ppo_eval`, `<name>_policy.py`,
and a README whose commands reproduce every leaderboard number (no `results/`).
Include your `ESCALATION.md` if you kept one — it is half the value. State the
provenance of the problem in the PR (paper cases: cite it; business cases:
confirm you may publish it). CI (`case-gates`) runs IR validation, spec
conformance, and the differential on every case dir the PR touches; all three
must be green. Negative cases — where RL does not beat the baselines — are
accepted when stated plainly; they stress the pipeline too (they just don't
get promoted to the examples set).

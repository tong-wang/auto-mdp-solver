# auto-mdp-solver

**A generic solver for the vast family of Markov Decision Processes (MDPs).**

Markov Decision Processes have long resisted general-purpose tooling: each has
tended to demand its own *custom model* and its own *custom solution procedure*.
auto-mdp-solver sets out to collapse both by composing two engines of
generality — **Claude to formalize** a verbally-described problem into a
standardized representation and construct its simulation environment (a verified
*digital twin*), and **reinforcement learning (PPO) to solve** it competitively,
with no bespoke algorithm to derive.

What keeps the automation trustworthy is an executable gate between every
stage — above all a **bit-exact differential check** that replays identical
randomness through the generated simulator and an independent IR interpreter, so
the twin is a *verified* model of the problem, not a plausible-looking guess.
Every policy ships benchmarked against baselines and, where tractable, an exact
dynamic-programming reference, so you always know how close to optimal you are.

Why *learn* the policy rather than compute it? The textbook route — dynamic
programming — is exact but collapses under the **curse of dimensionality**: its
cost explodes with the size of the state and action spaces, ruling it out for
problems of any realistic scale. High dimensionality is exactly where RL
shines — it learns strong policies by simulating the twin rather than
enumerating it, reaching problems classical DP cannot touch. For now the solver
is validated *against* exact DP on classical problems small enough to solve
optimally — the deliberate proving ground before the real target: the far
larger problems DP has never reached, where no optimal reference exists at all.

And the aim is not a black box. Dynamic programming was prized for two things —
an optimal policy *and* the structural insight that came with it (base-stock
levels, *(s, S)* thresholds, monotone rules). We want both: a competitive policy
*and* an interpretation of what it learned — recovering the policy's structure
and, where a classical policy form exists, testing whether it rediscovers one.
A solution to understand, not only to deploy.

The whole semi-automatic pipeline — **formalize → build → solve → interpret →
package** — is encapsulated as Claude Code skills (and, increasingly,
deployable agents), so a competitive policy can be reached from a
plain-English problem description.

**Status: first public release.** The plugin and the PyPI package
(`auto-mdp-solver`) are cut from the same tag and carry the same version.

A Claude Code plugin + Python toolchain that builds a **trained, deployable
RL policy from a verbal description of a dynamic decision-making problem**:

1. **Phase A — formalize (human in the loop).** A guided interview turns the
   problem into a validated MDP intermediate representation (IR): decisions,
   dynamics, randomness sources, scenario set, objective — every assumption
   surfaced and signed off, then frozen with a fingerprint.
2. **Phase B — build (gated stages).** From the frozen IR: a spec-conformant
   simulator domain (`_uncertainty` / `_scenarios` / `_mdp` / `_gym` layers),
   a bit-exact differential check against the IR interpreter, benchmarks
   (random / myopic / exact DP where tractable), PPO training and Optuna
   tuning, a policy-structure readback (the learned rule recovered, fitted,
   and scored paired against the reference), and a `{domain}_policy.py`
   deployment wrapper — with an executable gate between every stage.

## Repository layout

The repo carries **two independently-published artifacts** in their own
subtrees, plus root-level material for public browsers. The two packaging
systems are disjoint: the PyPI wheel contains only the `harness/` packages;
the plugin ships only `plugin/`'s recognized dirs (so `examples/` must live
**inside** `plugin/skills/mdp-solver/` to ship). Neither looks at the
other's metadata. `cases/` and the root-level material ship in neither —
they exist for the public GitHub repo.

### `harness/` — the PyPI package `auto-mdp-solver`

Has its own `pyproject.toml`; deliberately torch-free at base install (the
`[domain]` extra adds the training stack).

| path | role |
|---|---|
| `harness/mdp_ir/` | IR schema (pydantic), interpreter, differential runner |
| `harness/mdp_ir/layering.py` | catalog ⊕ selection resolution + symbolic bounds (one `{domain}_schema.json`; each uncertainty slot declares a `candidates` pool + `default`, instances select candidates and override constants; `load_ir(path, instance=, select=)` resolves, so nothing downstream sees the catalog) |
| `harness/mdp_ir/families.py` | distribution-family registry: derived `mean`/`max`/`min`/`is_discrete` (with latent composition) that symbolic bounds resolve against — never hand-authored per domain |
| `harness/mdp_ir/runtime.py` | generic generator runtime: the one family→numpy sampling dispatch (shared with the interpreter) + `FamilyGenerator`, a spec-§4.2 generator over any registered family; domains bridge it once per slot, so a new catalog candidate needs zero new domain Python |
| `harness/mdp_conformance/` | spec-conformance harness (`python -m mdp_conformance <domain-dir>`) |
| `harness/mdp_gates/` | eval-gate comparison (`python -m mdp_gates`) |
| `harness/mdp_tuning/` | Optuna tuning driver (`python -m mdp_tuning <domain-dir> ...`) |
| `harness/mdp_stage/` | pipeline entry gates (`python -m mdp_stage <domain-dir> [--for <op>]`) — the split ops' executable entry re-validation, incl. the always-on freeze check against `{name}.signoff.json` |

### `plugin/` — the Claude Code plugin

`.claude-plugin/plugin.json` + `skills/`; the plugin is the versioned unit
(every release is a `v<version>` tag matching `plugin.json`).

| path | role |
|---|---|
| `plugin/skills/mdp-solver/SKILL.md` | the pipeline **conductor** — dispatches the six step skills with `mdp_stage` gates between |
| `plugin/skills/mdp-{formalize,build,solve,escalate,interpret,package}/` | the six **step skills** — one op each, cut at durable-artifact seams, independently entrant; they read the shared corpus from `../mdp-solver/` |
| `plugin/skills/mdp-solver/{ENVIRONMENT,INTERVIEW,CONTRACTS}.md` | the split's shared corpus: interpreter/budgets/run discipline; the five interview rules; the op contracts + the two single-writer state files (`{name}.signoff.json`, `{name}.runplan.json`) |
| `plugin/skills/mdp-solver/MDP_PROJECT_SPEC.md` | **canonical** per-domain architecture/naming/RNG/script conventions |
| `plugin/skills/mdp-solver/MDP_IR_SAMPLE.md` | annotated MDP-IR reference |
| `plugin/skills/mdp-solver/DOMAIN_CLAUDE_TEMPLATE.md` | template for the per-domain `{domain}/CLAUDE.md` emitted at Stage 1 — the folder's *upward* pointers: governing specs, provenance, hard rules, gate commands, file hygiene (spec §1.3) |
| `plugin/skills/mdp-solver/DOMAIN_README_TEMPLATE.md` | template for the per-domain `{domain}/README.md` emitted at Stage 1 — the folder's *across* view: the problem, layout, results by research-question tier, technical appendix (spec §1.3) |
| `plugin/skills/mdp-solver/ESCALATION_LOG_GUIDE.md` | **canonical** campaign-log format (§MAP, §FRAME-CHANGELOG, §IR-CHANGELOG, §CONFIG-REGISTRY, §LEDGER) and the §10 playbook digest — a governing doc for every domain folder, so it ships with the plugin; its section numbers are cited from case logs and are append-only |
| `plugin/skills/mdp-solver/examples/` | frozen exemplar domains — ship with the plugin as few-shot exemplars **and** are the regression suite (see `examples/MANIFEST.md` there) |
| `plugin/skills/mdp-solver/PLAYBOOK.md` | shipped escalation-playbook **index** — maintainer-curated; links into promoted examples' in-folder `PLAYBOOK.md`s (digest entries per ESCALATION_LOG_GUIDE §10) |
| `plugin/skills/mdp-contribute/SKILL.md` | the contribution skill (case PRs with the playbook riding in the folder / re-skinned cases; a case too sensitive even re-skinned is not contributed; `.github/workflows/case-gates.yml` is its CI counterpart) |
| `plugin/skills/mdp-propose/SKILL.md` | the proposal skill — mid-campaign spec/schema/process extension proposals, one `upstream-proposal` issue each; maintainer disposition accept (case becomes the regression) / reject / defer, recorded on the issue |

### Root

| path | role |
|---|---|
| `.claude-plugin/marketplace.json` | marketplace manifest (points at `./plugin`) |
| `cases/` | auto-solve test cases: one folder per case, built end-to-end by the skill |

## Install

As a Claude Code plugin (ships the skill + examples; the skill installs the
Python toolchain into your workspace venv on first run):

```
/plugin marketplace add tong-wang/auto-mdp-solver
/plugin install auto-mdp-solver@auto-mdp-solver
```

Python toolchain only:

```bash
pip install "auto-mdp-solver[domain]"   # harness + the training stack a generated domain needs
# the same from the repo at a release tag, no PyPI needed:
pip install "auto-mdp-solver[domain] @ git+https://github.com/tong-wang/auto-mdp-solver@v0.10.16#subdirectory=harness"
# or, from a checkout:
pip install -e ./harness
```

## Verify (regression suite)

From the repo root, with pytest installed — `[dev]` is pytest alone (torch-free
and enough for the whole suite), `[domain]` includes it alongside the training
stack:

```bash
pytest        # harness engine tests + every example domain's own {domain}_test.py
```

The same checks are also available as CLI gates — what the skill runs between
stages:

```bash
E=plugin/skills/mdp-solver/examples
python -m mdp_conformance $E/inv_single $E/mab $E/game2048   # generated-code shape
python -m mdp_ir.laws     $E/inv_single $E/mab $E/game2048   # IR execution semantics
python -m mdp_ir.differential $E/inv_single/inv_single_schema.json --all-instances --episodes 40
```

All gates must pass: engine laws green, conformance green (SKIPs only for
inapplicable checks), differential bit-exact MATCH with no declared-invariant
violation.

## Example domains

Under `plugin/skills/mdp-solver/examples/`:

- `inv_single` — single-item periodic-review inventory control: two-step
  advance, episode-support demand, exact-DP benchmarks on both sides of the
  fixed-cost and lost-sales lines, a generalist over a cost × lead-time grid
  tested on a held-out grid, and a §14 readback.
- `mab` — the standard multi-armed bandit: exploration against exploitation
  over an equivariant state, with a per-episode latent inferred from censored
  feedback, and a fitted structural rule scored as a first-class benchmark.
- `game2048` — the 2048 sliding-tile game: a variable, long horizon over a
  board state read by a CNN extractor, with masked discrete actions and no DP
  to bracket the optimum.

## License

MIT — see [LICENSE](LICENSE).

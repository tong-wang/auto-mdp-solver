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

**Status: pre-release.** The PyPI name (`auto-mdp-solver`) currently holds a
placeholder; the first functional release is being prepared from this repo.

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

## Components

Two independently-published artifacts live in their own subtrees:

| path | role |
|---|---|
| `harness/` | **the PyPI package** `auto-mdp-solver`: `mdp_ir` (IR schema + interpreter + differential runner), `mdp_conformance` (architecture/RNG/purity checks), `mdp_gates` (statistical eval gates, ≥ 2 SE), `mdp_tuning` (Optuna driver) |
| `plugin/` | **the Claude Code plugin**: `skills/mdp-solver/` (SKILL.md + canonical spec `MDP_PROJECT_SPEC.md` + annotated IR reference `MDP_IR_SAMPLE.md`) and, shipped alongside it, `examples/` — frozen exemplar domains that double as the regression suite (`examples/MANIFEST.md`) |
| `cases/` | end-to-end pipeline test cases (browse on GitHub) |
| `README.md`, `docs/` | public landing + guides |

The two packaging systems are disjoint: `pip install` sees only `harness/`; the
plugin ships only `plugin/`.

## Install

As a Claude Code plugin (ships the skill + examples; the skill installs the
Python toolchain into your workspace venv on first run):

```
/plugin marketplace add tong-wang/auto-mdp-solver
/plugin install auto-mdp-solver@auto-mdp-solver
```

Python toolchain only:

```bash
pip install auto-mdp-solver   # placeholder today; real release upcoming
# or, from source:
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
python -m mdp_conformance $E/inv_single $E/dynamic_pricing $E/fnv   # generated-code shape
python -m mdp_ir.laws     $E/inv_single $E/dynamic_pricing $E/fnv   # IR execution semantics
python -m mdp_ir.differential $E/inv_single/inv_single_schema.json --all-instances --episodes 40
```

All gates must pass: engine laws green, conformance green (SKIPs only for
inapplicable checks), differential bit-exact MATCH with no declared-invariant
violation.

## Example domains

Under `plugin/skills/mdp-solver/examples/`:

- `inv_single` — single-echelon inventory control with stochastic
  lead times (two-step advance, episode-support demand, exact-DP benchmark).
- `dynamic_pricing` — finite-horizon revenue management (Gallego &
  van Ryzin 1994): continuous price control, decision-conditioned demand
  generator, exact-DP benchmark; PPO reaches within 0.6% of DP.

## License

MIT — see [LICENSE](LICENSE).

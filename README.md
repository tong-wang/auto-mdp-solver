# auto-mdp-solver

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
   tuning, and a `{domain}_policy.py` deployment wrapper — with an executable
   gate between every stage.

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

From the repo root, with the harness installed (`pip install -e ./harness`):

```bash
E=plugin/skills/mdp-solver/examples
python -m mdp_ir.interpreter_test
python -m mdp_conformance $E/inv_single $E/dynamic_pricing
python -m mdp_ir.differential $E/inv_single/inv_single_schema.json --episodes 40
python -m mdp_ir.differential $E/dynamic_pricing/vanryzin_pricing_schema.json --episodes 40
```

All gates must pass: interpreter invariants green, conformance green (SKIPs only
for inapplicable checks), differential bit-exact MATCH.

## Example domains

Under `plugin/skills/mdp-solver/examples/`:

- `inv_single` — single-echelon inventory control with stochastic
  lead times (two-step advance, episode-support demand, exact-DP benchmark).
- `dynamic_pricing` — finite-horizon revenue management (Gallego &
  van Ryzin 1994): continuous price control, decision-conditioned demand
  generator, exact-DP benchmark; PPO reaches within 0.6% of DP.

## License

MIT — see [LICENSE](LICENSE).

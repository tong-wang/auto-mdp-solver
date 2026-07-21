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
   a bit-exact differential check against the IR interpreter, baselines
   (random / myopic / exact DP where tractable), PPO training and Optuna
   tuning, and a `{domain}_policy.py` deployment wrapper — with an executable
   gate between every stage.

## Components

| path | role |
|---|---|
| `skills/mdp-solver/` | the Claude Code skill (SKILL.md + canonical spec `MDP_PROJECT_SPEC.md` + annotated IR reference `MDP_IR_SAMPLE.md`) |
| `mdp_ir/` | IR schema (pydantic), reference interpreter, differential runner |
| `mdp_conformance/` | architecture/RNG/purity conformance harness (11 checks) |
| `mdp_gates/` | statistical eval gates (candidate vs baselines, ≥ 2 SE) |
| `mdp_tuning/` | Optuna tuning driver for the generated training scripts |
| `examples/` | frozen example domains — the regression suite (`examples/MANIFEST.md`) |
| `cases/` | end-to-end pipeline test cases |

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
pip install -e .
```

## Verify (regression suite)

From the repo root, with the package installed:

```bash
python -m mdp_ir.interpreter_test
python -m mdp_conformance examples/inv_single examples/dynamic_pricing
python -m mdp_ir.differential examples/inv_single/inv_single_schema.json --episodes 40
python -m mdp_ir.differential examples/dynamic_pricing/vanryzin_pricing_schema.json --episodes 40
```

All gates must pass: interpreter invariants green, conformance 11/11,
differential bit-exact MATCH.

## Example domains

- `examples/inv_single` — single-echelon inventory control with stochastic
  lead times (two-step advance, episode-support demand, exact-DP baseline).
- `examples/dynamic_pricing` — finite-horizon revenue management (Gallego &
  van Ryzin 1994): continuous price control, decision-conditioned demand
  generator, exact-DP baseline; PPO reaches within 0.6% of DP.

## License

MIT — see [LICENSE](LICENSE).

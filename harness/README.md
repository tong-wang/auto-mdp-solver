# auto-mdp-solver — the harness

The Python toolchain behind the [auto-mdp-solver](https://github.com/tong-wang/auto-mdp-solver)
Claude Code plugin: a pipeline that turns a verbal dynamic-decision problem
into a trained, deployable RL policy. This package is the executable half —
the IR schema and interpreter, the gates, and the tuning driver. The plugin
(skills, spec, examples) ships separately and installs this package into the
workspace venv on first run.

| package | what it is | entry point |
|---|---|---|
| `mdp_ir` | MDP-IR schema (pydantic), reference interpreter, catalog ⊕ selection layering, execution laws, bit-exact differential runner | `python -m mdp_ir <schema>`, `python -m mdp_ir.laws <dir>`, `python -m mdp_ir.differential <schema>` |
| `mdp_conformance` | spec-conformance checks on a generated domain | `python -m mdp_conformance <dir>` |
| `mdp_gates` | eval-gate comparison of a candidate against baselines under the shared seed protocol | `python -m mdp_gates` |
| `mdp_tuning` | Optuna tuning driver over a domain's training script | `python -m mdp_tuning <dir> ...` |
| `mdp_stage` | the pipeline's entry gates — where a domain folder stands, and whether an op may start | `python -m mdp_stage <dir> [--for <op>]` |

## Install

```bash
pip install "auto-mdp-solver[domain]"   # everything a generated domain needs (SB3 + torch + pandas + pytest)
pip install auto-mdp-solver             # harness only, torch-free: IR, conformance, gates, tuning
pip install "auto-mdp-solver[dev]"      # harness + pytest
# the same, straight from the repo at a release tag (no PyPI needed):
pip install "auto-mdp-solver[domain] @ git+https://github.com/tong-wang/auto-mdp-solver@v0.10.16#subdirectory=harness"
```

Requires Python ≥ 3.12. The version number tracks the plugin release it was
cut from (`v0.10.16` ↔ plugin `0.10.16`), so the skills and the harness a
workspace holds are comparable by one number.

Documentation, the per-domain spec, the frozen example domains and the test
cases live in the repository. MIT license.

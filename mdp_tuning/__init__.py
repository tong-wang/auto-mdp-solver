"""Domain-generic hyperparameter tuning for spec-conformant MDP domains.

Like mdp_conformance, this package is parametrized by a domain directory and
driven entirely by the conventions in MDP_PROJECT_SPEC.md: any domain with
{prefix}_{algo}_train.py / {prefix}_{algo}_eval.py scripts exposing
_build_arg_parser() (spec §8.2, §9.1) can be tuned with zero per-domain code.

    python -m mdp_tuning sudoku -s 4x4_8 --metric solve_rate --n-trials 25

The search space is defined once per *algorithm* (spaces.py, RL-Zoo-style
broad ranges), not per problem, and is intersected at runtime with the knobs
the domain's train script actually exposes.
"""

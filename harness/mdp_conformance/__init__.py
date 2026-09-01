"""MDP conformance harness.

Domain-parametrized checker for the conventions in ``MDP_PROJECT_SPEC.md``.
Point it at any ``{domain}/`` directory:

    python -m mdp_conformance plugin/skills/mdp-solver/examples/inv_single plugin/skills/mdp-solver/examples/mab plugin/skills/mdp-solver/examples/game2048

See ``loader.py`` (discovery), ``checks.py`` (the invariants), and
``runner.py`` (reporting).
"""

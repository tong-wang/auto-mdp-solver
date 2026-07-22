"""MDP conformance harness.

Domain-parametrized checker for the conventions in ``MDP_PROJECT_SPEC.md``.
Point it at any ``{domain}/`` directory:

    python -m mdp_conformance examples/inv_single examples/dynamic_pricing

See ``loader.py`` (discovery), ``checks.py`` (the invariants), and
``runner.py`` (reporting).
"""

"""MDP conformance harness.

Domain-parametrized checker for the conventions in ``MDP_PROJECT_SPEC.md``.
Point it at any ``{domain}/`` directory:

    python -m mdp_conformance inv_single cnv fnv owmr 2048

See ``loader.py`` (discovery), ``checks.py`` (the invariants), and
``runner.py`` (reporting).
"""

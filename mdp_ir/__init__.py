"""MDP intermediate representation (IR).

The Phase-A artifact of the MDP agent (see ``MDP_AGENT_PLAN.md`` §3): a
machine-checkable pydantic model of a dynamic decision problem that the user
confirms before any code is generated. ``MDP_IR_SAMPLE.md`` is the annotated
reference instance.

Validate an IR JSON file:

    python -m mdp_ir inv_single/inv_single_schema.json

Execute an IR directly (round-trip trajectories / codegen oracle):

    python -m mdp_ir.interpreter inv_single/inv_single_schema.json --decision order=40

See ``schema.py`` for the models and invariants, ``interpreter.py`` for the
restricted executor.
"""

from mdp_ir.schema import Confirmable, MdpIR, load_ir

__all__ = ["Confirmable", "IrInterpreter", "MdpIR", "Trajectory", "load_ir", "simulate"]

_INTERPRETER_EXPORTS = ("IrInterpreter", "Trajectory", "simulate")


def __getattr__(name: str):  # lazy: keeps `python -m mdp_ir.interpreter` runpy-clean
    if name in _INTERPRETER_EXPORTS:
        from mdp_ir import interpreter

        return getattr(interpreter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

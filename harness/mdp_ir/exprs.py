"""The one core-builtin table every IR expression evaluates against.

There are two evaluators in this package — the interpreter (dynamics, guards,
invariants, objective) and the layering resolver (symbolic bounds, and any
load-time constant resolution) — and the schema validator decides, for both,
which names an expression may use (``schema._BUILTINS``).

They used to disagree. The validator declared ``exp`` / ``sqrt`` / ``len`` /
``range`` legal in *any* IR expression, the interpreter implemented them, and
the layering resolver carried seven names, so a bound written ``exp(mu +
5*stdev)`` validated, shipped, and died at load with ``name 'exp' is not
defined``. Latent rather than live only because every author who needed one
baked a number instead — which is the circularity upstream #45 is about.

So the table lives here, in a leaf module both evaluators import: what
validates is what resolves, structurally, rather than by two files
remembering to agree. ``schema._BUILTINS`` is the same set plus the
keywords/literals that are syntax rather than callables (``if``, ``and``,
``True``, …) and ``T``, which is bound as a value.

Domain-owned functions do NOT belong here — an IR declares them in
``mdp.expr_builtins`` and the interpreter injects them per IR
(``_declared_funcs``). They are deliberately absent from the layering side:
bounds resolve *during* load, before the IR those builtins are declared on
exists.
"""

from __future__ import annotations

import math

CORE_FUNCS: dict[str, object] = {
    "min": min, "max": max, "sum": sum, "abs": abs, "len": len,
    "round": round, "int": int, "float": float,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "floor": math.floor, "ceil": math.ceil,
    "zeros": lambda n: [0] * int(n),
    "phi": lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))),
    "topk": lambda values, k: sorted(values, reverse=True)[: int(k)],
    "range": range,
    # float-tolerant equality — the balance-law idiom in `mdp.invariants`,
    # so a conservation claim reads as one expression instead of an
    # abs()-difference-under-epsilon dance
    "close": lambda a, b, tol=1e-9: abs(float(a) - float(b)) <= tol,
}


def eval_expr(expr: str, ns: dict, funcs: dict | None = None, where: str = "expr"):
    """Evaluate ``expr`` over ``ns`` with ``funcs`` (default: the core table).

    The namespace is merged into **globals**, never passed as locals. Python
    gives a comprehension its own scope and resolves free names in it through
    globals, so with ``ns`` as locals a genexpr nested in a listcomp cannot
    see the surrounding names —

        [sum(h[j] for j in range(k, n)) for k in range(n)]
        -> NameError: name 'h' is not defined

    — which is upstream #32's defect (a whitelisted comprehension that cannot
    bind what it reads), one evaluator over. Callers own their own error
    wrapping; this raises whatever Python raises.
    """
    table = CORE_FUNCS if funcs is None else funcs
    return eval(  # noqa: S307 — namespace is the caller's restricted one
        compile(expr, f"<{where}>", "eval"), {"__builtins__": table, **ns}
    )

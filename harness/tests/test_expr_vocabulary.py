"""The two evaluators share one builtin table, and it matches the validator.

Upstream #45, stage 1. Two independent defects lived here:

- the layering resolver carried seven names while `schema._BUILTINS` declared
  eighteen legal in *any* IR expression, so a bound written `exp(mu+5*stdev)`
  validated, shipped, and died at load with `name 'exp' is not defined`;
- it passed its namespace as **locals**, so a genexpr nested in a listcomp
  could not see the constants around it even once `range` existed — upstream
  #32's defect, one evaluator over.

Both are silent-by-construction: an author who hits either bakes a number
instead, which is exactly what #45 is filed about. Hence the regressions.
"""

import math

import pytest

from mdp_ir import exprs
from mdp_ir.interpreter import _FUNCS
from mdp_ir.layering import LayeringError, _eval_expr

# syntax and value names, not callables — they are legal in an expression but
# never live in the function table
_NON_CALLABLE = {"if", "else", "not", "and", "or", "in", "for",
                 "T", "True", "False", "None", "true", "false"}


def test_one_table_backs_both_evaluators():
    """Not "the same contents" — the same object, so they cannot drift."""
    from mdp_ir.layering import _SAFE_BUILTINS

    assert _FUNCS is exprs.CORE_FUNCS
    assert _SAFE_BUILTINS is exprs.CORE_FUNCS


def test_table_matches_what_the_schema_validates():
    from mdp_ir.schema import _BUILTINS

    assert (_BUILTINS - _NON_CALLABLE) - set(exprs.CORE_FUNCS) == set()
    assert set(exprs.CORE_FUNCS) - _BUILTINS == set()


@pytest.mark.parametrize("expr,want", [
    ("exp(1.0 + 5*0.2)", math.exp(2.0)),      # fnv.order_max, multiplicative
    ("sqrt(0.45) * 0.2", math.sqrt(0.45) * 0.2),   # fnv.signal_stdevs
    ("len([1, 2, 3])", 3),
    ("floor(2.7)", 2),
    ("ceil(2.1)", 3),
    ("log(1.0)", 0.0),
    ("zeros(3)", [0, 0, 0]),
    ("topk([3, 1, 2], 2)", [3, 2]),
    ("close(0.1 + 0.2, 0.3)", True),
    ("sum(range(4))", 6),
])
def test_validator_legal_names_resolve_in_a_bound(expr, want):
    """Each of these raised `name ... is not defined` at load before stage 1."""
    assert _eval_expr(expr, {}, "bound") == want


def test_a_name_in_neither_set_still_fails():
    """Widening the vocabulary must not become "evaluate anything"."""
    with pytest.raises(LayeringError):
        _eval_expr("cumsum_reverse([1, 2])", {}, "bound")
    with pytest.raises(LayeringError):
        _eval_expr("__import__('os').listdir('.')", {}, "bound")


def test_nested_comprehension_sees_the_surrounding_namespace():
    """The #32 defect, one evaluator over.

    A genexpr inside a listcomp gets its own scope and resolves free names
    through globals; with the namespace passed as locals this raised
    `NameError: name 'h_echelon' is not defined`.
    """
    ns = {"h_echelon": [1.0, 0.5, 0.5], "n_echelons": 3}
    got = _eval_expr(
        "[sum(h_echelon[j] for j in range(k, n_echelons)) for k in range(n_echelons)]",
        ns, "bound",
    )
    # clark_scarf's shipped h_install, minus the width padding
    assert got == [2.0, 1.0, 0.5]


def test_the_two_evaluators_agree_on_one_expression():
    """The property the shared table exists to guarantee."""
    ns = {"mu": 1.0, "stdev": 0.2}
    expr = "exp(mu + 5*stdev)"
    assert _eval_expr(expr, ns, "bound") == exprs.eval_expr(expr, ns, _FUNCS)

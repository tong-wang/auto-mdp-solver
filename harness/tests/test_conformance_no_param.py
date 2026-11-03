"""`static.no_param` — no module in the folder binds `param`, `params` or `param_*` (spec §2).

The rule is stated for the whole codebase, and until #106 the check read only
the six layered modules. A train script, a custom head or the test file could
bind the name, and the gate still said PASS over files it had never opened.
That is how a shipped example carried `def expand_order_logits(params, n)` in
its ordinal head. The check now reads every `*.py` in the folder, as
`static.imports` does, and flags names the folder *binds*. Reading a library's
own attribute (`optimizer.param_groups`) is not a choice the domain made, so
it is not flagged. Synthetic folders only: the check must not learn the shape
of any shipped domain.
"""

from __future__ import annotations

import pytest

from mdp_conformance.checks import check_no_param
from mdp_conformance.loader import DomainHandle


def domain(tmp_path, modules: dict[str, str]) -> DomainHandle:
    for name, source in modules.items():
        (tmp_path / name).write_text(source)
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


CLEAN = {"d_mdp.py": "def step(state, action):\n    return state\n"}


@pytest.mark.parametrize("source", [
    "def expand_order_logits(params, n):\n    return params\n",        # an argument
    "def f(*, param_scale=1.0):\n    return param_scale\n",           # a keyword-only argument
    "def f(*params):\n    return params\n",                          # *args
    "params = [1, 2]\n",                                              # an assignment target
    "a, params = 1, 2\n",                                             # a tuple target
    "for param in range(3):\n    pass\n",                            # a loop target
    "with open('x') as params:\n    pass\n",                         # a with target
    "xs = [param for param in range(3)]\n",                          # a comprehension target
    "class Head:\n    def __init__(self):\n        self.params = 0\n",   # an attribute store
    "def param_grid():\n    return 0\n",                             # a def name
    "class Params:\n    pass\nclass param_set:\n    pass\n",         # a class name (the second)
    "import numpy as params\n",                                       # an import alias
    "from torch.optim import param_groups\n",                         # an imported name
    "try:\n    pass\nexcept ValueError as param:\n    pass\n",        # an except target
    "match 1:\n    case [*params]:\n        pass\n",                  # a match capture
])
def test_a_bound_forbidden_name_fails_in_any_module(tmp_path, source):
    h = domain(tmp_path, {**CLEAN, "d_ordinal_head.py": source})
    r = check_no_param(h)
    assert r.status == "FAIL", source
    assert "d_ordinal_head.py:" in r.detail


def test_the_scripts_and_the_test_file_are_the_codebase_too(tmp_path):
    # The two binding sites from the issue, in the two files the old
    # predicate never parsed.
    h = domain(tmp_path, {
        **CLEAN,
        "d_ordinal_head.py": "import torch as th\n\n"
                             "def expand_order_logits(params, n):\n    return params[:, 0]\n",
        "d_test.py": "import torch as th\n\ndef test_head():\n"
                     "    params = th.tensor([[0.0, 0.0, 0.0]])\n",
    })
    r = check_no_param(h)
    assert r.status == "FAIL"
    assert "d_ordinal_head.py:3 'params'" in r.detail
    assert "d_test.py:4 'params'" in r.detail


def test_reading_a_librarys_attribute_is_not_a_binding(tmp_path):
    # torch's own API: a domain cannot rename it, and the rule governs the
    # names the codebase chooses.
    h = domain(tmp_path, {
        **CLEAN,
        "d_vine_ppo.py": "def set_lr(optimizer, lr):\n"
                         "    for g in optimizer.param_groups:\n        g['lr'] = lr\n"
                         "    n = len(optimizer.param_groups)\n"
                         "    model.policy.optimizer.param_groups[0]['lr'] = lr\n"
                         "    return n\n",
        "d_ppo_train.py": "net = make(params=3)\nx = spec.param_count\n",  # a keyword, an attribute read
    })
    r = check_no_param(h)
    assert r.status == "PASS", r.detail


def test_pass_names_how_many_modules_it_read(tmp_path):
    h = domain(tmp_path, {**CLEAN, "d_gym.py": "x = 1\n", "d_ppo_train.py": "y = 2\n"})
    r = check_no_param(h)
    assert r.status == "PASS"
    assert "3 module(s)" in r.detail


def test_near_misses_pass(tmp_path):
    h = domain(tmp_path, {**CLEAN, "d_gym.py": "raw = 1\nparameters = 2\nparamount = 3\nmy_params = 4\n"})
    assert check_no_param(h).status == "PASS"


def test_an_unparseable_module_is_not_passed_over(tmp_path):
    h = domain(tmp_path, {**CLEAN, "d_broken.py": "def f(:\n"})
    r = check_no_param(h)
    assert r.status == "FAIL"
    assert "d_broken.py: unparseable" in r.detail

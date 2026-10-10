"""`static.imports` — a domain imports only what the documented install provides (spec §1.2).

A domain folder travels: promoted into the shipped examples, cloned into a
research repo, gated by a CI runner that installs `[dev]` plus the folder's own
`requirements.txt`. A package that was only ever present in its author's venv
makes a module that imports nowhere else, and nothing said so until a reader
ran the command. The check reads every module's imports and accepts the
standard library, the harness, `[domain]`, the folder's siblings and what the
folder declares; an optional import needs no declaration.

The cases below are the shapes a real folder takes: a module-level import, a
lazy one inside a function (still owed — the function fails without it), the
fallback pattern guarded directly or one call away, a test's importorskip, and
a requirements file that restates what `[domain]` already installs (which would
pull torch into CI's torch-free runner). Synthetic folders only: the check must
not learn the shape of any shipped domain.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from mdp_conformance.checks import (
    DOMAIN_IMPORTS, HARNESS_IMPORTS, check_imports, dist_import_name,
)
from mdp_conformance.loader import DomainHandle

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def domain(tmp_path, modules: dict[str, str], requirements: str | None = None) -> DomainHandle:
    for name, source in modules.items():
        (tmp_path / name).write_text(source)
    if requirements is not None:
        (tmp_path / "requirements.txt").write_text(requirements)
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


# --- what needs no declaration ----------------------------------------------

def test_the_standard_library_the_harness_domain_and_siblings_pass(tmp_path):
    h = domain(tmp_path, {
        "d_mdp.py": "from __future__ import annotations\nimport math, json\nfrom dataclasses import dataclass\n",
        "d_gym.py": "import numpy as np\nimport gymnasium as gym\nimport d_mdp\nfrom d_mdp import x\n",
        "d_ppo_train.py": "import torch\nfrom stable_baselines3 import PPO\nfrom sb3_contrib import MaskablePPO\n",
        "d_plot_policy.py": "import plotly.graph_objects as go\nimport matplotlib.pyplot as plt\nimport pandas\n",
        "d_test.py": "import pytest\nfrom mdp_ir.testing import x\nfrom mdp_conformance import y\n",
    })
    r = check_imports(h)
    assert r.status == "PASS", r.detail


def test_a_relative_import_is_a_sibling(tmp_path):
    assert check_imports(domain(tmp_path, {"d_a.py": "from . import d_b\nfrom .d_b import f\n"})).status == "PASS"


# --- what is owed ------------------------------------------------------------

def test_an_undeclared_module_level_import_warns_with_its_location(tmp_path):
    r = check_imports(domain(tmp_path, {"d_bound.py": "import numpy as np\nfrom scipy import special\n"}))
    assert r.status == "WARN"
    assert "d_bound.py:2 scipy" in r.detail


def test_a_lazy_import_inside_a_function_is_still_owed(tmp_path):
    src = "def pmf(x):\n    from scipy.stats import gamma\n    return gamma.pdf(x, 2.0)\n\n\nvalue = pmf(1.0)\n"
    r = check_imports(domain(tmp_path, {"d_dp.py": src}))
    assert r.status == "WARN" and "d_dp.py:2 scipy" in r.detail


def test_declaring_it_in_the_folder_requirements_passes_and_is_reported(tmp_path):
    r = check_imports(domain(tmp_path, {"d_bound.py": "from scipy import special, optimize\n"},
                             requirements="# the bound's special functions\nscipy>=1.11,<2  # d_bound\n"))
    assert r.status == "PASS", r.detail
    assert "requirements.txt adds scipy" in r.detail


@pytest.mark.parametrize("requirement, module", [
    ("scikit-learn>=1.4", "sklearn"),
    ("PyYAML==6.0.1", "yaml"),
    ("python-dateutil", "dateutil"),
    ("Some_Package.Name ; python_version >= '3.12'", "some_package_name"),
    ("pulp @ https://example.org/pulp.tar.gz", "pulp"),
])
def test_a_distribution_is_matched_to_its_import_name(tmp_path, requirement, module):
    assert dist_import_name(requirement) == module
    r = check_imports(domain(tmp_path, {"d_x.py": f"import {module}\n"}, requirements=requirement + "\n"))
    assert r.status == "PASS", r.detail


# --- optional imports ----------------------------------------------------------

def test_an_import_guarded_by_import_error_is_optional(tmp_path):
    src = ("try:\n    from scipy.special import erfinv\nexcept ImportError:\n    erfinv = None\n"
           "try:\n    import numba\nexcept (ModuleNotFoundError, OSError):\n    numba = None\n")
    r = check_imports(domain(tmp_path, {"d_probe.py": src}))
    assert r.status == "PASS", r.detail
    assert "2 optional" in r.detail


def test_the_fallback_one_call_away_is_optional(tmp_path):
    src = ("def _fast(p):\n    from scipy.special import erfinv\n    return erfinv(p)\n\n\n"
           "def ppf(p):\n    try:\n        return _fast(p)\n    except ImportError:\n        return bisect(p)\n")
    assert check_imports(domain(tmp_path, {"d_probe.py": src})).status == "PASS"


def test_one_unguarded_call_makes_the_function_import_owed(tmp_path):
    src = ("def _fast(p):\n    from scipy.special import erfinv\n    return erfinv(p)\n\n\n"
           "def ppf(p):\n    try:\n        return _fast(p)\n    except ImportError:\n        return bisect(p)\n\n\n"
           "def direct(p):\n    return _fast(p)\n")
    assert check_imports(domain(tmp_path, {"d_probe.py": src})).status == "WARN"


def test_a_test_modules_importorskip_and_type_checking_imports_need_nothing(tmp_path):
    src = ("from typing import TYPE_CHECKING\nimport pytest\nif TYPE_CHECKING:\n    import scipy\n\n\n"
           "def test_bound():\n    pytest.importorskip('scipy')\n")
    assert check_imports(domain(tmp_path, {"d_test.py": src})).status == "PASS"


def test_a_guard_that_does_not_catch_import_error_is_no_guard(tmp_path):
    src = "try:\n    import scipy\nexcept ValueError:\n    scipy = None\n"
    assert check_imports(domain(tmp_path, {"d_x.py": src})).status == "WARN"


# --- the requirements file itself ------------------------------------------------

def test_restating_domain_warns_since_ci_would_install_it_torch_free(tmp_path):
    r = check_imports(domain(tmp_path, {"d_x.py": "import torch\n"}, requirements="torch==2.6.0\nscipy\n"))
    assert r.status == "WARN"
    assert "restates what [domain] installs: torch" in r.detail


def test_an_option_line_is_not_a_requirement(tmp_path):
    r = check_imports(domain(tmp_path, {"d_x.py": "import scipy\n"}, requirements="-r ../requirements.txt\nscipy\n"))
    assert r.status == "WARN" and "line(s) 1" in r.detail


def test_an_unparseable_module_is_reported_not_raised(tmp_path):
    r = check_imports(domain(tmp_path, {"d_x.py": "def (:\n"}))
    assert r.status == "WARN" and "unparseable" in r.detail


# --- the allowed set stays in step with what the package installs ------------------

def test_every_core_and_domain_requirement_is_importable_without_declaring_it():
    toml = tomllib.loads(PYPROJECT.read_text())
    project = toml["project"]
    requirements = project["dependencies"] + project["optional-dependencies"]["domain"]
    names = {dist_import_name(r) for r in requirements} - {"auto_mdp_solver"}
    assert names <= DOMAIN_IMPORTS, sorted(names - DOMAIN_IMPORTS)
    assert "scipy" not in DOMAIN_IMPORTS
    assert set(toml["tool"]["setuptools"]["packages"]) == HARNESS_IMPORTS

"""Domain discovery and loading for the MDP conformance harness.

A domain is discovered by the spec's file-naming convention (§1): a directory
containing ``{prefix}_gym.py``, ``{prefix}_mdp.py`` and ``{prefix}_scenarios.py``.
The prefix is read from the files, not the folder name, because it may differ
(e.g. the ``2048/`` directory uses the ``game2048_`` prefix).

``load_domain`` imports the domain's modules (adding its directory to
``sys.path`` so the domain's bare-name sibling imports resolve) and returns a
``DomainHandle`` that normalizes access to the pieces the checks need:
``SCENARIOS``, the ``{Domain}State`` type, ``init_state`` / ``advance*``, and the
``{Domain}Env`` class. Checks are written against this handle, never against a
concrete domain, so one harness covers every domain.
"""

from __future__ import annotations

import ast
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable


ROLE_SUFFIXES = {
    "exceptions":  "_exceptions.py",
    "uncertainty": "_uncertainty.py",
    "scenarios":   "_scenarios.py",
    "grids":       "_grids.py",
    "gym":         "_gym.py",
}


@dataclass
class CheckResult:
    """Outcome of one conformance check.

    status is one of PASS / FAIL / WARN / SKIP / ERROR. Only FAIL and ERROR are
    treated as gate failures (non-zero exit); WARN and SKIP are informational.
    """

    check: str
    status: str
    detail: str = ""


@dataclass
class DomainHandle:
    """Normalized view of one domain under test."""

    name: str                       # file prefix, e.g. "inv_single", "game2048"
    directory: Path
    files: dict[str, Path]          # role -> path (roles in ROLE_SUFFIXES + "mdp")
    modules: dict[str, ModuleType]  # role -> imported module
    mdp_module_name: str

    SCENARIOS: dict
    env_cls: type
    state_cls: type | None
    init_state: Callable
    # exactly one of these is populated (single-step vs two-step domains)
    advance: Callable | None = None
    advance_pair: tuple[Callable, Callable] | None = None

    # parsed source trees, role -> ast.Module (for static checks)
    trees: dict[str, ast.Module] = field(default_factory=dict)

    def source(self, role: str) -> str:
        return self.files[role].read_text()


def discover_prefix(directory: Path) -> str:
    """Return the domain file prefix from a directory, or raise ValueError."""
    gyms = sorted(directory.glob("*_gym.py"))
    for gym in gyms:
        prefix = gym.name[: -len("_gym.py")]
        if (directory / f"{prefix}_mdp.py").exists() and (
            directory / f"{prefix}_scenarios.py"
        ).exists():
            return prefix
    raise ValueError(
        f"no domain found in {directory}: need matching "
        f"{{prefix}}_gym.py / _mdp.py / _scenarios.py"
    )


def _mdp_module_name_from_gym(gym_tree: ast.Module, prefix: str) -> str:
    """Read which ``*_mdp`` module the gym imports (owmr has two candidates)."""
    for node in ast.walk(gym_tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("_mdp"):
            return node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("_mdp"):
                    return alias.name
    return f"{prefix}_mdp"


def _find_env_class(gym_module: ModuleType) -> type:
    import gymnasium as gym

    candidates = [
        obj
        for obj in vars(gym_module).values()
        if isinstance(obj, type)
        and issubclass(obj, gym.Env)
        and obj.__module__ == gym_module.__name__
    ]
    if not candidates:
        raise ValueError(f"no gym.Env subclass defined in {gym_module.__name__}")
    candidates.sort(key=lambda c: not c.__name__.endswith("Env"))
    return candidates[0]


def _find_state_class(mdp_module: ModuleType) -> type | None:
    candidates = [
        obj
        for obj in vars(mdp_module).values()
        if isinstance(obj, type)
        and obj.__name__.endswith("State")
        and obj.__module__ == mdp_module.__name__
    ]
    return candidates[0] if candidates else None


def load_domain(directory: str | Path) -> DomainHandle:
    """Discover, import and normalize the domain in ``directory``."""
    directory = Path(directory).resolve()
    prefix = discover_prefix(directory)

    files: dict[str, Path] = {"mdp": directory / f"{prefix}_mdp.py"}
    for role, suffix in ROLE_SUFFIXES.items():
        path = directory / f"{prefix}{suffix}"
        if path.exists():
            files[role] = path

    trees = {role: ast.parse(path.read_text()) for role, path in files.items()}
    mdp_module_name = _mdp_module_name_from_gym(trees["gym"], prefix)

    # domain files import their siblings by bare module name; make them resolvable
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

    modules: dict[str, ModuleType] = {}
    gym_module = importlib.import_module(f"{prefix}_gym")
    modules["gym"] = gym_module
    scenarios_module = importlib.import_module(f"{prefix}_scenarios")
    modules["scenarios"] = scenarios_module
    mdp_module = importlib.import_module(mdp_module_name)
    modules["mdp"] = mdp_module
    for role in ("uncertainty", "exceptions", "grids"):
        if role in files:
            modules[role] = importlib.import_module(f"{prefix}{ROLE_SUFFIXES[role][:-3]}")

    advance = getattr(mdp_module, "advance", None)
    advance_pair = None
    if advance is None and hasattr(mdp_module, "advance1") and hasattr(mdp_module, "advance2"):
        advance_pair = (mdp_module.advance1, mdp_module.advance2)

    return DomainHandle(
        name=prefix,
        directory=directory,
        files=files,
        modules=modules,
        mdp_module_name=mdp_module_name,
        SCENARIOS=getattr(scenarios_module, "SCENARIOS", {}),
        env_cls=_find_env_class(gym_module),
        state_cls=_find_state_class(mdp_module),
        init_state=getattr(mdp_module, "init_state"),
        advance=advance,
        advance_pair=advance_pair,
        trees=trees,
    )

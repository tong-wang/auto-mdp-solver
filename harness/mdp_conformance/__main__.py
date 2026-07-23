"""CLI entry point: ``python -m mdp_conformance <domain-dir> [<domain-dir> ...]``.

With no arguments, discovers every direct subdirectory of the current working
directory that looks like a domain (contains a ``*_mdp.py``) and checks them
all — the package is installed, so the cwd, not the package location, is the
workspace.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .runner import run


def _discover_domains(root: Path) -> list[Path]:
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and any(d.glob("*_mdp.py"))
    )


def main() -> int:
    args = sys.argv[1:]
    if args:
        directories = [Path(a) for a in args]
    else:
        directories = _discover_domains(Path.cwd())
        if not directories:
            print("no domains found", file=sys.stderr)
            return 2
    return run(directories)


if __name__ == "__main__":
    raise SystemExit(main())

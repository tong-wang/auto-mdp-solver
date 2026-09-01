"""dynamic_pricing tuning — thin wrapper around the domain-generic mdp_tuning harness.

Equivalent to running, from the repo root:

    python -m mdp_tuning dynamic_pricing --metric revenue_mean [your args...]

All arguments are forwarded to mdp_tuning (see `--help`); the only default
pre-filled here is `--metric revenue_mean` (the eval TSV's objective column).
Pass --metric yourself to override.

Typical usage (from inside the dynamic_pricing/ directory):

    python dynamic_pricing_ppo_tune.py -s simple --n-trials 25 --total-timesteps 200000
    python dynamic_pricing_ppo_tune.py --show-space
    python dynamic_pricing_ppo_tune.py -s simple --summary-only
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_DOMAIN_DIR = Path(__file__).resolve().parent

_DEFAULTS = ["--metric", "revenue_mean"]


def main() -> None:
    sys.path.insert(0, str(_DOMAIN_DIR.parent))
    # later duplicates win in argparse, so user-supplied flags override defaults
    sys.argv = ["mdp_tuning", str(_DOMAIN_DIR), *_DEFAULTS, *sys.argv[1:]]
    runpy.run_module("mdp_tuning", run_name="__main__")


if __name__ == "__main__":
    main()

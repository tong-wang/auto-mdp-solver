"""Thin wrapper over the repo-level `mdp_tuning` harness for AdiFlex.

Pre-fills the domain defaults so a study is a one-liner:

    python adi_flex_ppo_tune.py -s exp4 --timeout 86400 --n-trials 2000

Everything else is forwarded to `python -m mdp_tuning` unchanged.

Domain defaults filled in here:
  --metric reward_mean   the objective column in the eval TSV. AdiFlex is a
                         MINIMIZE-cost problem, but the eval TSVs report
                         reward = -(cost) precisely so that "higher is better"
                         holds throughout the gate and the tuner. Do NOT add
                         --minimize; that would tune for the worst policy.
  --train-arg n_envs=8   episodes are 12 steps, so PPO's per-call torch
                         overhead — not the simulator — sets throughput.
                         Batching the forward pass over 8 envs measured ~7x
                         (220 -> 1500 fps). Not a tunable knob; a fixed fix.

SELECTION BIAS (spec section 13): the winning trial was *selected* on its
tuning eval seeds, so its study score is optimistic. Always re-evaluate the
winning artifact with the full protocol (--n-seeds 8192) before comparing it
with the benchmarks or shipping it, and expect the number to get worse.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DOMAIN_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    "--metric": "reward_mean",
    "--total-timesteps": "500000",
    "--eval-seeds": "2048",
    "-s": "exp4",
}
FIXED_TRAIN_ARGS = ["--train-arg", "n_envs=8"]


def main() -> int:
    argv = sys.argv[1:]
    cmd = [sys.executable, "-m", "mdp_tuning", str(DOMAIN_DIR)]

    for flag, value in DEFAULTS.items():
        # a long/short alias supplied by the caller wins over the default
        aliases = {"-s": ("-s", "--scenario_name")}.get(flag, (flag,))
        if not any(a in argv for a in aliases):
            cmd += [flag, value]

    if "--train-arg" not in argv:
        cmd += FIXED_TRAIN_ARGS
    cmd += argv

    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

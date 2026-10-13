"""Post-hoc checkpoint screen (spec §8.6 / §9.7): rank a run's ~20 checkpoints
on the SCREEN seed block (first seed 1_000_000, disjoint from the protocol
block), print the top-k, and name the confirmation commands. Screen scores are
never quoted; the protocol eval of the top-k crowns the winner.

    python cnv_select.py results/base/PPO_<run> [--n-seeds 2048] [--top-k 3]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from cnv_benchmark_common import eval_seed_block, rollout_batched
from cnv_grids import GRIDS
from cnv_ppo_eval import RAW_MODES, _vecnorm_for, load_policy
from cnv_gym import CnvEnv
from cnv_scenarios import SCENARIOS

SCREEN_FIRST_SEED = 1_000_000


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Screen a run's checkpoints on the screen seed block.")
    p.add_argument("run_dir", type=str)
    p.add_argument("--n-seeds", type=int, default=2048)
    p.add_argument("--first-seed", type=int, default=SCREEN_FIRST_SEED)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=False)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def read_args_log(run_dir: Path) -> dict[str, str]:
    log = next(run_dir.glob("*_ppo_args.txt"))
    out = {}
    for line in log.read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            out[k] = v
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    meta = read_args_log(run_dir)
    scenario_name, obs_mode, act_mode = meta["scenario_name"], meta["observation_mode"], meta["action_mode"]
    family = GRIDS[scenario_name] if scenario_name in GRIDS else SCENARIOS[scenario_name]
    frame_stack = int(meta.get("frame_stack") or (family.horizon if obs_mode in RAW_MODES else 1))
    source = family.as_sampler() if scenario_name in GRIDS else family
    seeds = eval_seed_block(args.first_seed, args.n_seeds)

    from gymnasium.wrappers import FrameStackObservation

    def probe_env_fn():
        env = CnvEnv(scenario=family, observation_mode=obs_mode, action_mode=act_mode)
        return FrameStackObservation(env, stack_size=frame_stack, padding_type="zero") if frame_stack > 1 else env

    ckpts = sorted(run_dir.glob("checkpoints/ppo_cnv_*_steps.zip"),
                   key=lambda p: int(re.search(r"_(\d+)_steps", p.name).group(1)))
    final = run_dir / f"{scenario_name}_ppo.zip"
    candidates = ckpts + ([final] if final.exists() else [])
    if not candidates:
        raise SystemExit(f"no checkpoints under {run_dir}")
    print(f"[select] {run_dir.name}: {len(candidates)} artifacts, screen block "
          f"{args.first_seed}..{args.first_seed + args.n_seeds - 1} (never quoted)")
    scores = []
    for path in candidates:
        policy, _ = load_policy(path, _vecnorm_for(path, None), not args.stochastic, probe_env_fn)
        per_seed = rollout_batched(source, seeds, policy.batch, observation_mode=obs_mode, action_mode=act_mode,
                                   frame_stack=frame_stack, batch=64, progress_every=0)
        mu = float(per_seed["profit"].mean())
        se = float(per_seed["profit"].std(ddof=1) / np.sqrt(len(seeds)))
        scores.append((mu, se, path))
        print(f"  {path.name:40s} {mu:9.3f} ± {se:.3f}")
    scores.sort(key=lambda t: -t[0])
    print(f"\n[select] top-{args.top_k} by screen mean — confirm each on the protocol block:")
    for mu, se, path in scores[: args.top_k]:
        print(f"  python cnv_ppo_eval.py --model-path {path} -s {scenario_name} -o {obs_mode} --n-seeds 8192"
              + ("" if scenario_name not in GRIDS else f"   # or -g {scenario_name}"))


if __name__ == "__main__":
    main()

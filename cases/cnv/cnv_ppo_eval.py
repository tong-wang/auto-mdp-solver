"""Protocol-tier evaluation of a trained PPO model (spec §9).

Scores one artifact on one target — a SCENARIOS entry, or a GRIDS entry cell
by cell — over the CRN seed block ``first_seed .. first_seed + n_seeds - 1``,
through the same ``cnv_benchmark_common.rollout`` every benchmark uses, so
the per-seed differences against them are paired. Writes the §9.3 record
beside the model (``ppo_eval_{target}.tsv`` + the ``.seeds.tsv`` sidecar) with
the §13 provenance lines.

    python cnv_ppo_eval.py --model-path results/base/PPO_<run>/base_ppo.zip -s base
    python cnv_ppo_eval.py --model-path results/base/PPO_<run>/checkpoints/ppo_cnv_500000_steps.zip -s base
    python cnv_ppo_eval.py --model-path results/paper510/PPO_<run>/paper510_ppo.zip -g paper510
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cnv_benchmark_common import (HEADER, METRICS, eval_seed_block, eval_targets, format_row,
                                  rollout, rollout_batched, summarize)
from cnv_grids import GRIDS
from cnv_gym import ALL_ACTION_MODES, ALL_OBSERVATION_MODES, CnvEnv
from cnv_scenarios import SCENARIOS

HERE = Path(__file__).resolve().parent
RAW_MODES = ("event_raw", "timing_raw", "full_raw")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model on a cnv scenario or grid.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None,
                   help="defaults to the checkpoint's own stats, else <model_dir>/vecnormalize.pkl")
    p.add_argument("-s", "--scenario_name", type=str, default="base", choices=list(SCENARIOS) + list(GRIDS))
    p.add_argument("-g", "--grid_name", type=str, default=None, choices=list(GRIDS.keys()),
                   help="evaluate per cell over this GRIDS entry; overrides -s")
    p.add_argument("-o", "--observation_mode", type=str, default="timing_belief", choices=list(ALL_OBSERVATION_MODES))
    p.add_argument("-a", "--action_mode", type=str, default=None, choices=list(ALL_ACTION_MODES),
                   help="default: the run's own (its *_ppo_args.txt), else order_discrete — an action mode is a "
                        "property of the artifact, so a caller that omits it must not silently evaluate another")
    p.add_argument("-r", "--reward_mode", type=str, default="profit", choices=["profit"])
    p.add_argument("--frame-stack", type=int, default=None,
                   help="must match training; default: horizon for *_raw modes, 1 otherwise")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0,
                   help="protocol block = 0 (the only block a number is quoted from); screen = 1_000_000")
    p.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=False,
                   help="sample actions instead of the argmax")
    p.add_argument("--batch-envs", type=int, default=64,
                   help="episodes rolled out in lockstep to amortise the forward pass; 1 = the sequential loop. "
                        "Does not affect results: each slot is seeded explicitly")
    p.add_argument("--arm", type=str, default=None, help="label for the record; defaults to the run dir name")
    p.add_argument("--outfile", type=str, default=None)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def _vecnorm_for(model_path: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    # a checkpoint saved with save_vecnormalize=True carries its own stats
    name = model_path.name
    if name.startswith("ppo_cnv_") and name.endswith("_steps.zip"):
        cand = model_path.with_name(name.replace("ppo_cnv_", "ppo_cnv_vecnormalize_").replace(".zip", ".pkl"))
        if cand.exists():
            return cand
    cand = model_path.parent / "vecnormalize.pkl"
    return cand if cand.exists() else None


def load_policy(model_path: Path, vecnorm_path: Path | None, deterministic: bool, probe_env_fn):
    try:
        model = PPO.load(str(model_path), device="cpu")
    except Exception as e:   # SB3 version skew (e.g. a 2.8 model's pickled schedules under 2.6, #E68 on the HPC):
        # the schedules are training-only; eval needs the policy alone, so override them and reload
        print(f"[eval] PPO.load failed ({type(e).__name__}); retrying with the training-only schedules overridden")
        model = PPO.load(str(model_path), device="cpu",
                         custom_objects={"lr_schedule": lambda _: 0.0, "learning_rate": 0.0, "clip_range": lambda _: 0.0})
    normalizer = None
    if vecnorm_path is not None:
        normalizer = VecNormalize.load(str(vecnorm_path), DummyVecEnv([probe_env_fn]))
        normalizer.training = False
        normalizer.norm_reward = False

    def act(env: CnvEnv, obs) -> int:
        x = np.asarray(obs, dtype=np.float32)[None, ...]   # keep the (stack, features) shape the normaliser was fitted on
        if normalizer is not None:
            x = normalizer.normalize_obs(x)
        action, _ = model.predict(x, deterministic=deterministic)
        # the env's ACTION (step decodes it); under order_discrete this is the order itself
        return max(0, int(np.asarray(action).reshape(-1)[0]))

    def batch_act(obs_batch) -> np.ndarray:
        x = np.asarray(obs_batch, dtype=np.float32)
        if normalizer is not None:
            x = normalizer.normalize_obs(x)
        actions, _ = model.predict(x, deterministic=deterministic)
        return np.maximum(0, np.asarray(actions).reshape(len(x), -1)[:, 0].astype(int))
    act.batch = batch_act
    return act, model


def _run_action_mode(model_path: Path) -> str | None:
    """The action mode the run that produced this artifact trained under (its args file)."""
    for d in (model_path.parent, model_path.parent.parent):
        for log in d.glob("*_ppo_args.txt"):
            for line in log.read_text().splitlines():
                if line.startswith("action_mode: "):
                    return line.split(": ", 1)[1].strip()
    return None


def main() -> None:
    args = parse_args()
    trained = _run_action_mode(Path(args.model_path).resolve())
    if args.action_mode is None:
        args.action_mode = trained or "order_discrete"
    elif trained is not None and trained != args.action_mode:
        raise SystemExit(f"--action_mode {args.action_mode} but the run trained under {trained}")
    model_path = Path(args.model_path).resolve()
    model_dir = model_path.parent
    vecnorm_path = _vecnorm_for(model_path, args.vecnorm_path)
    target, rows = eval_targets(args)
    family = GRIDS[args.grid_name] if args.grid_name else SCENARIOS[args.scenario_name]
    frame_stack = args.frame_stack if args.frame_stack is not None else (
        int(family.horizon) if args.observation_mode in RAW_MODES else 1)
    arm = args.arm or model_dir.name
    outfile = Path(args.outfile) if args.outfile else model_dir / f"ppo_eval_{target}.tsv"
    seeds = eval_seed_block(args.first_seed, args.n_seeds)

    from gymnasium.wrappers import FrameStackObservation

    def probe_env_fn():
        env = CnvEnv(scenario=family, observation_mode=args.observation_mode, action_mode=args.action_mode)
        return FrameStackObservation(env, stack_size=frame_stack, padding_type="zero") if frame_stack > 1 else env

    policy, model = load_policy(model_path, vecnorm_path, not args.stochastic, probe_env_fn)

    # §9.3 provenance: the three completion lines, from the artifact itself
    try:
        from mdp_gates.completion import provenance_lines
        completion = provenance_lines(model_path)
    except Exception as e:  # pragma: no cover — the lines are evidence, never a blocker
        completion = [f"# completion lines unavailable: {type(e).__name__}: {e}"]
    stamps = [f"# arm: {arm}", f"# model: {model_path}", f"# vecnorm: {vecnorm_path}",
              f"# target: {target}", f"# seed block: first_seed={args.first_seed} n_seeds={args.n_seeds}",
              f"# observation_mode={args.observation_mode} action_mode={args.action_mode} "
              f"frame_stack={frame_stack} deterministic={not args.stochastic}"] + list(completion)
    block = "protocol" if args.first_seed == 0 else "screen" if args.first_seed == 1_000_000 else "non-standard"
    print("\n".join(stamps)); print(f"[eval] seed block: {block}")
    if args.first_seed != 0:
        print("[eval] WARNING: not the protocol block — ranking only, never quote this number (§9.7).")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    seeds_file = outfile.with_name(outfile.stem + ".seeds.tsv")
    header = "\t".join(HEADER)
    print(header)
    with open(outfile, "w") as f, open(seeds_file, "w") as fs:
        f.write("\n".join(stamps) + "\n" + header + "\n"); f.flush()
        fs.write("target\tseed\t" + "\t".join(METRICS) + "\n")
        means = []
        for row_id, source in rows:
            if args.batch_envs > 1:
                per_seed = rollout_batched(source, seeds, policy.batch, observation_mode=args.observation_mode,
                                           action_mode=args.action_mode, frame_stack=frame_stack,
                                           batch=args.batch_envs, progress_every=0 if args.grid_name else 2048)
            else:
                per_seed = rollout(source, seeds, policy, observation_mode=args.observation_mode,
                                   action_mode=args.action_mode, frame_stack=frame_stack,
                                   progress_every=0 if args.grid_name else 2000)
            stats = summarize(per_seed)
            row = format_row(row_id, source, len(seeds), stats)
            print(row, flush=True)
            f.write(row + "\n"); f.flush()
            for k, seed in enumerate(seeds):
                fs.write(f"{row_id}\t{seed}\t" + "\t".join(f"{per_seed[m][k]:.6f}" for m in METRICS) + "\n")
            fs.flush()
            means.append(stats["profit_mean"])
    if args.grid_name:
        print(f"[eval] cells {len(means)}: mean over cells {np.mean(means):.3f}, worst cell {min(means):.3f}"
              " (cells are separate leaderboards; the aggregate is a summary, never a score)")
    print(f"[eval] record   -> {outfile}")
    print(f"[eval] per-seed -> {seeds_file}")


if __name__ == "__main__":
    main()

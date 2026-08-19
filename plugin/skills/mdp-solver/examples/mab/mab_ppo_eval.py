"""Spec-§9 evaluation of a trained PPO model on the mab domain.

Runs the shared seed protocol (episode seeds 0..n_seeds-1, same as the
benchmark evals) with the saved VecNormalize stats injected (spec §9.5),
and writes ppo_eval_stoch_{scenario}.tsv next to the model with the same
columns as the benchmark TSVs, ready for `python -m mdp_gates`.

The policy is evaluated **stochastically by default** (`--no-stochastic`
restores argmax): the policy's entropy *is* this domain's exploration
mechanism, so argmax collapses it into an under-explorer. Every number
reported since ESCALATION #E7 is the stochastic one.

Usage:
    python mab_ppo_eval.py -s gauss_K10_T1000 --model-path results/gauss_K10_T1000/<run>/gauss_K10_T1000_ppo.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mab_benchmark_common import reward_stats
from mab_gym import MabEnv
from mab_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model on mab.")
    p.add_argument("-s", "--scenario_name",    type=str, default="gauss_K10_T1000",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", type=str, default=None,
                   choices=["stats", "bayes"],
                   help="default: read from the run's {scenario}_ppo_args.txt")
    p.add_argument("-a", "--action_mode",      type=str, default="arm",
                   choices=["arm"])
    p.add_argument("-r", "--reward_mode",      type=str, default="payout",
                   choices=["payout"])
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None,
                   help="VecNormalize stats (default: vecnormalize.pkl next to "
                        "the model). They are part of the policy's input "
                        "contract (spec §9.5), not an optional decoration.")
    p.add_argument("--n-seeds",    type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0,
                   help="First episode seed (spec §9.2). The protocol block is "
                        "0; §9.7's checkpoint screen runs at 1000000 so the "
                        "layer that ranks never touches the block that quotes.")
    p.add_argument("--outfile",    type=str, default=None,
                   help="write the TSV here instead of the derived path next "
                        "to the model (spec §9.1). mdp_tuning requires this "
                        "dest: it collects each trial's TSV under the study "
                        "directory rather than in the trial's run dir.")
    p.add_argument("--stochastic", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="sample the policy instead of argmax (default: on). "
                        "In this domain the policy's entropy implements the "
                        "exploration, so argmax eval collapses it into an "
                        "under-explorer (~684 vs ~1135 mean reward for the "
                        "same model); stochastic eval reflects the policy's "
                        "real behavior and is what every reported number "
                        "since ESCALATION #E7 uses. Writes "
                        "ppo_eval_stoch_{scenario}.tsv; --no-stochastic "
                        "restores argmax and the unprefixed name.")
    return p


def read_run_obs_mode(run_dir: Path) -> str:
    # spec §8.4 name first, then the pre-rename "args.txt" of older runs
    candidates = sorted(run_dir.glob("*_ppo_args.txt")) + [run_dir / "args.txt"]
    for args_file in candidates:
        if not args_file.exists():
            continue
        for line in args_file.read_text().splitlines():
            if line.startswith("observation_mode:"):
                return line.split(":", 1)[1].strip()
    return "stats"


def main() -> None:
    args = _build_arg_parser().parse_args()
    model_path = Path(args.model_path).resolve()
    run_dir = model_path.parent
    obs_mode = args.observation_mode or read_run_obs_mode(run_dir)

    model = PPO.load(str(model_path), device="cpu")
    raw_env = MabEnv(scenario=SCENARIOS[args.scenario_name],
                     observation_mode=obs_mode,
                     action_mode=args.action_mode,
                     reward_mode=args.reward_mode)
    env = DummyVecEnv([lambda: raw_env])
    vecnorm_path = (Path(args.vecnorm_path) if args.vecnorm_path
                    else run_dir / "vecnormalize.pkl")
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training    = False
        env.norm_reward = False
        print(f"loaded VecNormalize stats: {vecnorm_path}")
    else:
        print("WARNING: no vecnormalize.pkl next to the model — evaluating raw")

    rewards = np.zeros(args.n_seeds)
    oracles = np.zeros(args.n_seeds)
    for i, seed in enumerate(range(args.first_seed,
                                   args.first_seed + args.n_seeds)):
        if i % 1000 == 0:
            print(f"  seed {i}/{args.n_seeds}", flush=True)
        raw_env.reset(seed=seed)          # fix the episode seed protocol
        obs = env.normalize_obs(raw_env._get_obs()[np.newaxis]) \
            if isinstance(env, VecNormalize) else raw_env._get_obs()[np.newaxis]
        done = False
        total = 0.0
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            o, r, done, _, info = raw_env.step(int(action[0]))
            obs = env.normalize_obs(o[np.newaxis]) \
                if isinstance(env, VecNormalize) else o[np.newaxis]
            total += float(r)
        rewards[i] = total
        oracles[i] = raw_env.horizon * info["opt_mean"]

    stats = reward_stats(rewards)
    stats["oracle_mean"] = float(oracles.mean())
    stats["regret_mean"] = stats["oracle_mean"] - stats["reward_mean"]

    family = "gaussian" if raw_env._is_gauss else "bernoulli"
    variant = "stoch_" if args.stochastic else ""
    out = (Path(args.outfile) if args.outfile
           else run_dir / f"ppo_eval_{variant}{args.scenario_name}.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("scenario\tK\tT\tfamily\treward_mean\treward_var\t"
              "semivar_d\tsemivar_u\toracle_mean\tregret_mean")
    row = (f"{args.scenario_name}\t{raw_env.n_arms}\t{raw_env.horizon}\t{family}\t"
           f"{stats['reward_mean']:.6f}\t{stats['reward_var']:.6f}\t"
           f"{stats['semivar_d']:.6f}\t{stats['semivar_u']:.6f}\t"
           f"{stats['oracle_mean']:.6f}\t{stats['regret_mean']:.6f}")
    # The leading '#' provenance line is a mab convention that mdp_gates skips
    # explicitly (mdp_gates/compare.py) but mdp_tuning's csv.DictReader cannot:
    # it takes line 1 as the header, finds no numeric columns, and the trial
    # dies in resolve_metric. --outfile is the machine-consumption path (the
    # tuner names each trial's TSV), so it gets the bare spec-§9.3 table; the
    # derived path stays the human/ledger artifact and keeps its provenance.
    provenance = (f"# ppo obs={obs_mode} "
                  f"policy={'stochastic' if args.stochastic else 'deterministic'} "
                  f"| n_seeds={args.n_seeds} from {args.first_seed}\n")
    out.write_text(f"{'' if args.outfile else provenance}{header}\n{row}\n")

    print(f"\n{'─' * 56}")
    print(f"  model       : {model_path.name}  (obs={obs_mode}, "
          f"{'stochastic' if args.stochastic else 'deterministic'})")
    print(f"  scenario    : {args.scenario_name}  "
          f"(n_seeds={args.n_seeds} from {args.first_seed})")
    print(f"  reward_mean : {stats['reward_mean']:10.4f}  "
          f"± {np.sqrt(stats['reward_var'] / args.n_seeds):.4f} (SE)")
    print(f"  oracle_mean : {stats['oracle_mean']:10.4f}")
    print(f"  regret_mean : {stats['regret_mean']:10.4f}")
    print(f"  TSV         : {out}")
    print(f"{'─' * 56}\n")


if __name__ == "__main__":
    main()

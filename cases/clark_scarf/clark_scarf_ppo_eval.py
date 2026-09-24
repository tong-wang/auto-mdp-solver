"""Evaluate a trained PPO policy on clark_scarf (spec §9.5).

Same seed protocol, metric set, and record format as every benchmark, so the RL
arm is directly and *pairwise* comparable with the DP, myopic, and random rows.

VecNormalize injection: the saved ``vecnormalize.pkl`` statistics MUST be reused
at eval time (spec §9.5). Evaluating a normalized policy against raw
observations silently scores a different function than the one that was trained.

The record eval is **deterministic** (see ``CLAUDE.md`` Traps): the policy emits
a continuous shipment quantity with no exploration role at eval time, and the
benchmark it is scored against is deterministic. ``--stochastic`` produces a
separate, labelled figure and is never a record.

Usage:
    python clark_scarf_ppo_eval.py -s n3_l2_p09 \
        --model-path results/n3_l2_p09/<run>/n3_l2_p09_ppo.zip \
        --paired-with results/n3_l2_p09/benchmark/benchmark_dp_eval_n3_l2_p09_perseed.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from clark_scarf_eval_common import (
    eval_seed_block,
    paired_report,
    rollout_seeds,
    write_record,
)
from clark_scarf_gym import ClarkScarfEnv
from clark_scarf_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    p.add_argument("-s", "--scenario_name", default="n3_l2_p09",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", default=None,
                   choices=list(ClarkScarfEnv.OBS_MODES),
                   help="defaults to the mode recorded in the run's args file")
    p.add_argument("-a", "--action_mode", default=None,
                   choices=list(ClarkScarfEnv.ACTION_MODES),
                   help="defaults to the mode recorded in the run's args file")
    p.add_argument("--model-path", required=True)
    p.add_argument("--vecnorm-path", default=None,
                   help="defaults to vecnormalize.pkl beside the model")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0,
                   help="First episode seed of the eval block (spec §9.2). The "
                        "record protocol is the block starting at 0; a nonzero "
                        "offset is a held-out block and must be labelled as one.")
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--arm", default=None,
                   help="leaderboard label; defaults to ppo_<obs>_<level>")
    p.add_argument("--outfile", default=None)
    p.add_argument("--stochastic", action="store_true",
                   help="sample instead of argmax; a labelled figure, never a record")
    p.add_argument("--paired-with", default=None,
                   help="another arm's *_perseed.npy; prints the paired delta")
    return p


def _resolve_run_dir(model_path: Path, scenario: str) -> Path:
    """The directory holding the run's §8.4 args log.

    A model path is either `<run>/{scenario}_ppo_final.zip` or, under §9.7,
    `<run>/checkpoints/{scenario}_ppo_{n}_steps.zip` — and for the second
    `model_path.parent` is `checkpoints/`, where no args log lives. Reading the
    args from there returns `{}` and EVERY field silently falls back to a
    parser default: an echelon checkpoint scores as `raw`, a run's level reads
    `L?`, and the missing-normalizer refusal cannot fire because it tests
    `norm_obs` from a mapping that is empty. Walk up instead.
    """
    for cand in (model_path.parent, model_path.parent.parent):
        if (cand / f"{scenario}_ppo_args.txt").exists():
            return cand
    return model_path.parent


def _read_run_args(run_dir: Path, scenario: str) -> dict:
    f = run_dir / f"{scenario}_ppo_args.txt"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> None:
    args = _build_arg_parser().parse_args()
    model_path = Path(args.model_path).resolve()
    run_dir = _resolve_run_dir(model_path, args.scenario_name)
    run_args = _read_run_args(run_dir, args.scenario_name)
    if not run_args:
        print(f"[eval] WARNING: no {args.scenario_name}_ppo_args.txt found near "
              f"{model_path.name} — observation/action mode and the normalizer "
              f"guard fall back to defaults. Pass -o/-a explicitly.", flush=True)

    obs_mode = args.observation_mode or run_args.get("observation_mode", "raw")
    # the action encoding must match training, or the policy's output is decoded
    # as a different quantity than the one it learned to emit
    act_mode = args.action_mode or run_args.get("action_mode", "ship_discrete")
    level = run_args.get("level", "L?")
    # a masked run must be reloaded as MaskablePPO and evaluated WITH masks —
    # scoring it unmasked lets it emit infeasible actions the env then clips,
    # which is a different policy than the one that was trained
    masked = run_args.get("mask", "False").strip() == "True"
    arm = args.arm or f"ppo_{obs_mode}_{act_mode}_{level}" + ("_masked" if masked else "")
    scenario = SCENARIOS[args.scenario_name]

    # A custom action head's hyperparameters are NOT saved inside the model:
    # `BetaDistribution.MIN_CONCENTRATION` is a class attribute and the Gamma
    # head's scale hint is a module global, both set by the TRAIN script. Load
    # them back from the run's own args log before reconstructing, or the policy
    # is rebuilt with different parameters than it was trained with — measured
    # once, and it is not subtle: a Gamma arm whose selection score was ~1004
    # scored ~13000 because the scale hint reverted to its 1.0 default and the
    # policy shipped a tenth of what it had learned to ship.
    dist = run_args.get("policy_dist", "gaussian")
    if dist in ("beta", "gamma"):
        import clark_scarf_beta_policy as _bp
        if dist == "beta":
            _bp.BetaDistribution.MIN_CONCENTRATION = float(
                run_args.get("beta_min_conc", 1.0))
            print(f"[eval] beta head restored: min_concentration="
                  f"{_bp.BetaDistribution.MIN_CONCENTRATION}", flush=True)
        else:
            _bp._GAMMA_SCALE_HINT[0] = float(scenario.demand.mean())
            print(f"[eval] gamma head restored: scale_hint="
                  f"{_bp._GAMMA_SCALE_HINT[0]:.1f}", flush=True)

    model = (MaskablePPO if masked else PPO).load(str(model_path), device="cpu")
    if masked:
        print("[eval] MaskablePPO — feasibility masks applied at every step")

    # §9.5: the saved statistics MUST be reused. Two names are possible and
    # which one exists depends on the selection protocol the run used:
    # `vecnormalize.pkl` is written by the (deprecated, §9.7-violating) live
    # selection callback beside its chosen artifact, `vecnormalize_final.pkl`
    # at the end of every run. A screened checkpoint carries its own, passed
    # explicitly by clark_scarf_select.py's confirm line.
    if args.vecnorm_path:
        vn_path = Path(args.vecnorm_path)
    else:
        vn_path = next((p for p in (run_dir / "vecnormalize.pkl",
                                    run_dir / "vecnormalize_final.pkl")
                        if p.exists()), run_dir / "vecnormalize.pkl")
    vn = None
    if vn_path.exists():
        vn = VecNormalize.load(str(vn_path), DummyVecEnv(
            [lambda: ClarkScarfEnv(scenario, observation_mode=obs_mode,
                                   action_mode=act_mode)]))
        vn.training = False
        print(f"[eval] VecNormalize loaded from {vn_path} "
              f"(norm_obs={vn.norm_obs})")
    else:
        # REFUSE rather than score raw. A policy trained under normalization and
        # evaluated without it is not a worse number, it is a different policy —
        # and the failure is silent: the TSV is well-formed, the SEs are right,
        # and nothing anywhere says the observations were wrong. This domain has
        # paid for that shape once already (the Gamma head's scale hint: a
        # selection score of ~1004 that evaluated at ~13000).
        trained_normed = str(run_args.get("norm_obs", "")).strip() == "True"
        if trained_normed:
            raise SystemExit(
                f"[eval] REFUSING: the run's args record norm_obs=True and no "
                f"VecNormalize statistics were found at {vn_path} (nor at "
                f"vecnormalize_final.pkl). Scoring this policy on raw "
                f"observations would silently produce a well-formed, wrong "
                f"number. Pass --vecnorm-path explicitly if the file lives "
                f"elsewhere (a screened checkpoint carries its own)."
            )
        print(f"[eval] no VecNormalize at {vn_path}, and the run did not "
              f"normalize — scoring raw observations, as trained")

    def act_fn(obs, envs):
        o = vn.normalize_obs(obs) if (vn is not None and vn.norm_obs) else obs
        if masked:
            m = np.stack([e.action_masks() for e in envs])
            a, _ = model.predict(o, deterministic=not args.stochastic,
                                 action_masks=m)
        else:
            a, _ = model.predict(o, deterministic=not args.stochastic)
        return a

    print(f"[eval] arm={arm} scenario={args.scenario_name} obs={obs_mode} "
          f"act={act_mode} "
          f"seeds={args.n_seeds}@{args.first_seed} "
          f"deterministic={not args.stochastic}", flush=True)
    per_seed = rollout_seeds(
        scenario=scenario, observation_mode=obs_mode,
        seeds=eval_seed_block(args.n_seeds, args.first_seed), act_fn=act_fn,
        batch=args.batch_envs, action_mode=act_mode)

    outfile = (Path(args.outfile) if args.outfile
               else run_dir / f"ppo_eval_{args.scenario_name}.tsv")
    write_record(outfile, args.scenario_name, arm, per_seed)
    if args.paired_with:
        paired_report(per_seed["cost_total"], Path(args.paired_with), "dp")


if __name__ == "__main__":
    main()

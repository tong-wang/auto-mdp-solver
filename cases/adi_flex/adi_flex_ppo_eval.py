"""Evaluate a trained MaskablePPO model on the ADI-flex domain (F7 two-phase env).

Plays the model deterministically through the gym wrapper with the spec §9
seed protocol (seeds 0..n_seeds-1), reusing the saved VecNormalize obs stats
(§9.5 injection). Writes the shared eval TSV (reward = -cost) so results are
directly comparable with adi_flex_benchmark_dp_eval.py / adi_flex_benchmark_rule.py.

Example usage:
    python adi_flex_ppo_eval.py \
        --model-path results/homog_L0_T2/PPO_.../homog_L0_T2_ppo.zip -s homog_L0_T2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from adi_flex_gym import ACTION_MODES, MIP_MODES, AdiFlexEnv
from adi_flex_scenarios import SCENARIOS, AdiFlexScenario, AdiFlexScenarioSampler
from adi_flex_benchmark_common import EVAL_HEADER, eval_row


# The SAME defect #E3 fixed in the train script, and it was never applied here:
# torch's `Simplex` constraint uses a FIXED 1e-6 tolerance, and a float32
# softmax over Discrete(order_max + 1) = 236 categories breaks it once the max
# probability passes ~0.9986 — which is where a converged (s,S) policy lands.
# Training survived and SCORING crashed, so the failure scaled with policy
# QUALITY: it took 5 of 60 trials in the `vec` tuning study and 0 of 60 in
# `vec_mip`, biasing the comparison against the arm that converges harder.
torch.distributions.Distribution.set_default_validate_args(False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model on the ADI-flex domain.")
    p.add_argument("--model-path",   type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None,
                   help="VecNormalize stats (defaults to <model_dir>/vecnormalize.pkl)")
    p.add_argument("-s", "--scenario_name",    type=str, default="homog_L0_T2",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", type=str, default="vec",
                   choices=["vec", *MIP_MODES])
    p.add_argument("-a", "--action_mode",      type=str, default="seq_mask",
                   choices=list(ACTION_MODES))
    p.add_argument("-r", "--reward_mode",      type=str, default="neg_cost",
                   choices=["neg_cost"])
    # RQ2's isolation instrument: a model trained with the order taken from the
    # AP policy has a sigma-ONLY action space, so scoring it in a plain env
    # fails outright (`shape '[1, 9]' is invalid for input of size 117`). The
    # flag has to exist here and not only on the train script, because a tuning
    # trial drives BOTH — this script is the eval half of every trial.
    p.add_argument("--fixed-order-ap", type=str, default=None, metavar="PATH",
                   help="AP y* table (.npz): the run's order came from it, so "
                        "rebuild the same wrapper to score it")
    p.add_argument("--n-seeds", type=int, default=8192,
                   help="Number of episode seeds")
    # spec sec 9.7: the post-hoc screen runs on a SELECTION block that must be
    # disjoint from the protocol block, or the checkpoint is chosen on the same
    # seeds it is then reported against. Default 0 keeps the protocol block.
    p.add_argument("--first-seed", type=int, default=0,
                   help="first episode seed (seeds first..first+n-1)")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV output (defaults to <model_dir>/ppo_eval_<scenario_name>.tsv)")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_env(scenario: AdiFlexScenario, observation_mode: str,
              action_mode: str, reward_mode: str = "neg_cost",
              fixed_order_ap: str | None = None):
    """The env a run was trained on — ONE definition, shared by eval and select.

    Every downstream script used to construct its own, and each new environment
    variant then had to be threaded through all of them by hand. That failed
    three times: the screen scored `order_protection` models in a `seq_mask`
    env (`spaces must have the same shape: (4,) != (10,)`), it defaulted the
    action mode to seq_mask through a parameter nobody passed, and it built a
    plain env for a run trained on the fixed-order wrapper (`shape '[1, 9]' is
    invalid for input of size 117`). Each crashed on EVERY run while still
    writing an empty select.tsv, so the ladder read as unscreened rather than
    broken. The resolution belongs in one function that a run's own args log
    drives, not in each caller's assumptions.
    """
    if fixed_order_ap:
        from adi_flex_fixed_order import FixedOrderProtectEnv
        return FixedOrderProtectEnv(
            scenario=scenario, ap_table=fixed_order_ap,
            observation_mode=observation_mode, reward_mode=reward_mode,
        )
    return AdiFlexEnv(
        scenario=scenario, observation_mode=observation_mode,
        action_mode=action_mode, reward_mode=reward_mode,
    )


def evaluate_scenario(
    model: MaskablePPO,
    vecnorm_path: Path | None,
    scenario: AdiFlexScenario,
    n_seeds: int,
    first_seed: int,
    observation_mode: str,
    action_mode: str,
    reward_mode: str,
    fixed_order_ap: str | None = None,
) -> dict:
    """Run episodes with seeds first_seed..first_seed+n_seeds-1, return stats."""
    env = build_env(scenario, observation_mode, action_mode, reward_mode,
                    fixed_order_ap)
    venv = DummyVecEnv([lambda: env])
    # An L0 run has NO VecNormalize — the level forbids it — so there are no
    # obs stats to reuse and none to apply. Anything else is a §9.5 violation:
    # a model trained under normalization must be evaluated under its own saved
    # stats, and a model trained without it must be evaluated without.
    normalized = vecnorm_path is not None and vecnorm_path.exists()
    if normalized:
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training    = False
        venv.norm_reward = False

    rewards = np.zeros(n_seeds)

    # Under a protection mode every point of the action box is feasible, so
    # the mask is all-true and its WIDTH is the only thing that matters — and
    # the width the model expects is the model's, not the live env's. The two
    # differ for artifacts trained before F54 unpadded the protection cap on
    # instances where the allocation is not live (the `sigma` component is 2
    # wide in the model and 1 in today's env on `homog_L0_T2`, #E25). There the
    # component is inert — the cascade never runs — so the model's width is
    # taken and the mismatch is reported once. Where the allocation IS live a
    # width mismatch is a real incompatibility and is refused.
    model_space = model.action_space
    fixed_mask = None
    if hasattr(model_space, "nvec"):
        env_nvec = np.asarray(env.action_space.nvec)
        model_nvec = np.asarray(model_space.nvec)
        if not np.array_equal(env_nvec, model_nvec):
            if scenario.alloc_enabled:
                raise RuntimeError(
                    f"model action space {model_space} != env {env.action_space} "
                    f"on {getattr(scenario, 'scenario_name', scenario)}, where the allocation is live — the "
                    f"artifact and this gym disagree on a decision's width")
            print(f"  note: model action space {model_space} vs env "
                  f"{env.action_space} — pre-F54 sigma width on a board with no "
                  f"live allocation; the component is inert, mask built at the "
                  f"model's width", flush=True)
        fixed_mask = np.ones(int(model_nvec.sum()), dtype=bool)[np.newaxis]

    for i, ep_seed in enumerate(range(first_seed, first_seed + n_seeds)):
        if i % 10000 == 0:
            print(f"  seed {i}/{n_seeds}", flush=True)
        # inject seed via env_method so VecNormalize normalization stays active
        raw_obs, _ = venv.env_method("reset", seed=ep_seed)[0]
        obs        = (venv.normalize_obs(raw_obs[np.newaxis, :]) if normalized
                      else raw_obs[np.newaxis, :])

        done  = False
        total = 0.0
        while not done:
            # per-step mask from the live env (order phase: full range;
            # allocation phase: the feasible set) — spec sec 7.1 contract
            mask = (fixed_mask if fixed_mask is not None
                    else venv.env_method("action_masks")[0][np.newaxis])
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, step_rewards, done_arr, _ = venv.step(action)
            total += float(step_rewards[0])
            done   = bool(done_arr[0])

        rewards[i] = total

    venv.close()

    mu = rewards.mean()
    return {
        "reward_mean": float(mu),
        "reward_var":  float(rewards.var(ddof=1)),
        "semivar_d":   float(np.sum(np.maximum(mu - rewards, 0.0) ** 2) / (n_seeds - 1)),
        "semivar_u":   float(np.sum(np.maximum(rewards - mu, 0.0) ** 2) / (n_seeds - 1)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model_path   = Path(args.model_path)
    vecnorm_path = Path(args.vecnorm_path) if args.vecnorm_path else model_path.parent / "vecnormalize.pkl"
    outfile = (
        Path(args.outfile) if args.outfile
        else model_path.parent / f"ppo_eval_{args.scenario_name}.tsv"
    )

    scenario_or_sampler = SCENARIOS[args.scenario_name]
    if isinstance(scenario_or_sampler, AdiFlexScenarioSampler):
        grid = list(scenario_or_sampler.members)
    else:
        grid = [scenario_or_sampler]

    model = MaskablePPO.load(str(model_path), device="cpu")
    print(f"model         {model_path}")
    print(f"vecnormalize  {vecnorm_path if vecnorm_path.exists() else '(none)'}")
    print(f"output        {outfile}")
    print(f"n_seeds       {args.n_seeds}")

    header = EVAL_HEADER
    print(header)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()
        for sc in grid:
            stats = evaluate_scenario(
                model=model,
                vecnorm_path=vecnorm_path,
                scenario=sc,
                n_seeds=args.n_seeds,
                first_seed=args.first_seed,
                fixed_order_ap=args.fixed_order_ap,
                observation_mode=args.observation_mode,
                action_mode=args.action_mode,
                reward_mode=args.reward_mode,
            )
            row = (
                eval_row(sc, stats)
            )
            print(row, flush=True)
            f.write(row + "\n")
            f.flush()

    print(f"\nresults saved -> {outfile}")


if __name__ == "__main__":
    main()

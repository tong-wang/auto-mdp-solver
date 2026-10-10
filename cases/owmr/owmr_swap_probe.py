"""Order / allocation swap probe for owmr (ESCALATION #E6).

Decomposes the gap between a trained RL policy and the LB heuristic by decision:
each period both policies see the same post-receipt state, and the probe
executes the ORDER of one with the ALLOCATION of the other. Four arms on one
CRN block, so every difference is paired per seed:

    rl_rl       RL order  + RL allocation         (validation: must reproduce the RL confirm)
    h_h         heuristic order + heuristic alloc (validation: must reproduce the heuristic eval)
    h_order     heuristic order + RL allocation   (what RL's allocation costs alone)
    h_alloc     RL order  + heuristic allocation  (what RL's ordering costs alone)

Gap G = rl_rl − h_h splits into an allocation part (h_order − h_h), an order
part (h_alloc − h_h) and their interaction (the rest). Hybrids put each policy
on states the other partly generated, so they are read as contrasts, not as
standalone policies.

The RL side is decoded exactly as the `order_softmax` gym mode does (anchored
softmax shares of warehouse on-hand, order clipped to [0, order_max]); the
heuristic side is `owmr_benchmark_lb_heuristic.LbHeuristicPolicy.decide`. The
env runs in `order_ship` mode so the probe hands the MDP quantities directly.
RL checkpoints trained on `order_softmax`, `order_relu` (#E14) or `order_catkeep`
(#E21, #E30) are supported (`--rl-action-mode`); the RL side is decoded exactly as
that gym mode does. A fifth arm (#E30 step 0) isolates the SPLIT:

    h_split     RL order + RL retention + heuristic SPLIT of the shipped quantity
                (the heuristic's myopic allocation of on-hand, renormalised to RL's
                shipped amount; equal shares if the heuristic ships nothing)

Usage:
    python owmr_swap_probe.py --model-path <ckpt.zip> --vecnorm-path <pkl> -s base --arm h_order --n-seeds 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from owmr_benchmark_lb_heuristic import build as build_heuristic
from owmr_eval_common import eval_seed_block, rollout_seeds, write_record
from owmr_catkeep_policy import CatKeepPolicy  # noqa: F401  (registers the class for PPO.load)
from owmr_gym import LOGIT_BOX, anchored_softmax_shares, catkeep_shares, relu_shares
from owmr_scenarios import SCENARIOS

ARMS = ("rl_rl", "h_h", "h_order", "h_alloc", "h_split")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Order/allocation swap probe (ESCALATION #E6).")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None)
    p.add_argument("-s", "--scenario_name", type=str, default="base", choices=list(SCENARIOS))
    p.add_argument("-o", "--observation_mode", type=str, default="raw")
    p.add_argument("--arm", type=str, required=True, choices=ARMS)
    p.add_argument("--rl-action-mode", type=str, default="order_softmax",
                   choices=["order_softmax", "order_relu", "order_catkeep"],
                   help="the gym mode the RL checkpoint was trained on (its decode is reproduced here)")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--outfile", type=str, default=None)
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def make_act_fn(arm: str, model: PPO, heuristic, rl_action_mode: str = "order_softmax"):
    def act(obs, envs):
        a_rl, _ = model.predict(obs, deterministic=True)
        rows = []
        for i, e in enumerate(envs):
            st, sc = e._state, e._scenario_ep
            onhand = max(0.0, float(st.wh_stock))
            rl_order = float(np.clip(a_rl[i][0], 0.0, sc.order_max))
            x = np.clip(a_rl[i][1:], -LOGIT_BOX, LOGIT_BOX)
            if rl_action_mode == "order_catkeep":        # [order, k_index, x_1..x_N]
                shares, kept = catkeep_shares(np.concatenate([[a_rl[i][1]], x[1:]]))
            elif rl_action_mode == "order_relu":
                shares, kept = relu_shares(x)
            else:
                shares = anchored_softmax_shares(x)
                kept = 1.0 - float(shares.sum())
            rl_ship = shares * onhand
            h_order, h_ship = heuristic.decide(st)
            order = h_order if arm in ("h_h", "h_order") else rl_order
            if arm == "h_split":
                hs = np.asarray(h_ship, dtype=np.float64)
                split = hs / hs.sum() if hs.sum() > 0 else np.full(len(hs), 1.0 / len(hs))
                ship = (1.0 - kept) * onhand * split
            else:
                ship = h_ship if arm in ("h_h", "h_alloc") else rl_ship
            rows.append(np.concatenate([[order], ship]))
        return np.asarray(rows, dtype=np.float64)   # the gym decodes in float64; keep the hybrids on the same arithmetic
    return act


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    model_path = Path(args.model_path).resolve()
    vecnorm = Path(args.vecnorm_path) if args.vecnorm_path else model_path.parent / "vecnormalize.pkl"
    model = PPO.load(str(model_path), device="cpu")
    heuristic = build_heuristic(args.scenario_name, here)
    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)
    run_dir = model_path.parent.parent if model_path.parent.name == "checkpoints" else model_path.parent
    outfile = Path(args.outfile) if args.outfile else run_dir / "probe" / f"swap_{args.arm}_{model_path.stem}.tsv"
    print(f"[swap] arm={args.arm}  model={model_path.name}  seeds {args.first_seed}..{args.first_seed + args.n_seeds - 1}")
    per_seed = rollout_seeds(scenario=SCENARIOS[args.scenario_name], observation_mode=args.observation_mode,
                             action_mode="order_ship", seeds=seeds, act_fn=make_act_fn(args.arm, model, heuristic, args.rl_action_mode),
                             batch=args.batch_envs, vecnorm_path=vecnorm)
    prov = [f"# DIAGNOSTIC (ESCALATION #E6/#E15), not a record: swap arm {args.arm}   rl decode: {args.rl_action_mode}",
            f"# model: {model_path}   vecnorm: {vecnorm}   heuristic y0*: {heuristic.y0:.4f}",
            f"# scenario: {args.scenario_name}   seeds: {args.first_seed}..{args.first_seed + args.n_seeds - 1}"]
    write_record(outfile, args.scenario_name, f"swap_{args.arm}", per_seed, seeds=seeds, provenance=prov)


if __name__ == "__main__":
    main()

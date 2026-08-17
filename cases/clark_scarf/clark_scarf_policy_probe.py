"""Policy readback (spec §14.1) — what did the network actually learn?

The campaign's headline question: trained on **raw installation stock**, does the
policy behave as a function of the **echelon aggregates**, and does it match
Clark & Scarf's optimal form

    y_i = median(u_i, xbar_i(t), x_{i+1})

i.e. order the echelon position up to a critical number, clipped by what the
level above holds?

Three things are measured, on states drawn from the policy's OWN trajectories
(off-policy grids would probe regions no policy visits):

1. **Is it a base-stock rule at all?** Recover the implied target
   ``y_i = u_i + q_i``. Under a base-stock rule ``y_i`` is *constant* wherever
   the availability clip is not binding, whatever the raw components look like.
   The spread of ``y_i`` over unclipped states is therefore the structural-form
   statistic: small spread = base-stock, large spread = something else.

2. **Are the recovered critical numbers the paper's?** Compare the fitted
   ``ybar_i`` against the DP's table period by period — the horizon is finite,
   so the optimum is genuinely time-varying and a single number would be the
   wrong thing to compare.

3. **Did it find the ECHELON coordinates, or is it keying on raw stock?** The
   discriminating test: hold the echelon position ``u_i`` fixed and vary how the
   same total is *split* across installations and pipelines. A policy that has
   found the echelon aggregation ships the same amount; one keying on raw
   components does not. This separates "rediscovered the structure" from
   "learned something that happens to score well".

Second-implementation note (spec §1.2): this module contains no re-implementation
of the MDP core — every state comes from ``ClarkScarfEnv`` and every reference
action from ``ClarkScarfDP``, so there is no equivalence to prove.

Usage:
    python clark_scarf_policy_probe.py -s n3_l2_p09 --model-path <run>/n3_l2_p09_ppo.zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from clark_scarf_benchmark_dp import ClarkScarfDP
from clark_scarf_gym import ClarkScarfEnv
from clark_scarf_eval_common import eval_seed_block, rollout_seeds
from clark_scarf_mdp import echelon_position, echelon_stock, ship_capacity
from clark_scarf_scenarios import SCENARIOS

CLIP_EPS = 1e-6


def _load_policy(model_path: Path, scenario, obs_mode: str, act_mode: str):
    """Load the trained policy exactly as it was trained.

    A masked run MUST be reloaded as MaskablePPO and predicted WITH masks —
    scoring it unmasked lets it emit infeasible actions the env then clips,
    which probes a different policy than the one that learned.
    """
    args_f = model_path.parent / f"{scenario.scenario_name}_ppo_args.txt"
    masked = False
    if args_f.exists():
        for line in args_f.read_text().splitlines():
            if line.startswith("mask: "):
                masked = line.split(": ", 1)[1].strip() == "True"
    model = (MaskablePPO if masked else PPO).load(str(model_path), device="cpu")
    vn_path = model_path.parent / "vecnormalize.pkl"
    vn = None
    if vn_path.exists():
        vn = VecNormalize.load(str(vn_path), DummyVecEnv(
            [lambda: ClarkScarfEnv(scenario, observation_mode=obs_mode,
                                   action_mode=act_mode)]))
        vn.training = False

    def act(obs_batch: np.ndarray, masks: np.ndarray | None = None) -> np.ndarray:
        o = vn.normalize_obs(obs_batch) if (vn is not None and vn.norm_obs) else obs_batch
        if masked:
            return model.predict(o, deterministic=True, action_masks=masks)[0]
        return model.predict(o, deterministic=True)[0]
    return act


def collect(scenario, act, obs_mode: str, act_mode: str, n_ep: int = 400) -> dict:
    """Roll the policy and record, per decision, the echelon view and its action."""
    rows = {k: [] for k in ("t", "u", "x", "q", "cap", "stock", "transit")}
    for ep in range(n_ep):
        env = ClarkScarfEnv(scenario, observation_mode=obs_mode, action_mode=act_mode)
        obs, _ = env.reset(seed=50_000 + ep)
        while True:
            st = env._state
            a = act(obs[None, :], env.action_masks()[None, :])[0]
            q = env._to_quantities(np.asarray(a, dtype=np.float64).reshape(-1))
            caps = ship_capacity(scenario, st)
            N = scenario.n_echelons
            rows["t"].append(st.period)
            rows["u"].append(echelon_position(scenario, st))
            rows["x"].append(echelon_stock(scenario, st))
            rows["q"].append([min(float(q[k]), caps[k]) for k in range(N)])
            rows["cap"].append(caps[:N])
            rows["stock"].append(list(st.stock[:N]))
            rows["transit"].append([st.pipeline[k][0] for k in range(N)])
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}


def structural_form(scenario, d: dict) -> dict:
    """Is it a base-stock rule, and with what critical numbers?

    ``y_i = u_i + q_i`` is the implied order-up-to level. Where the availability
    clip does not bind, a base-stock policy holds ``y_i`` constant.
    """
    N = scenario.n_echelons
    y = d["u"] + d["q"]
    out = {}
    for i in range(N):
        # unbinding: the clip did not truncate the shipment, and the horizon-end
        # wind-down has not started (there the optimum is genuinely not constant)
        free = (d["q"][:, i] < d["cap"][:, i] - CLIP_EPS) & (d["t"] < scenario.horizon - 6)
        yi = y[free, i]
        out[f"echelon_{i + 1}"] = {
            "n_unclipped": int(free.sum()),
            "ybar_fitted": float(np.median(yi)) if len(yi) else float("nan"),
            "iqr": float(np.subtract(*np.percentile(yi, [75, 25]))) if len(yi) else float("nan"),
            "std": float(yi.std()) if len(yi) else float("nan"),
        }
    return out


def agreement_with_dp(scenario, d: dict, dp: ClarkScarfDP) -> dict:
    """Fraction of visited states where the policy's shipment matches the DP's."""
    N = scenario.n_echelons
    ref = np.zeros_like(d["q"])
    for r in range(len(d["t"])):
        t = int(min(d["t"][r], scenario.horizon - 1))
        for k in range(N):
            target = float(dp.ybar[t, k])
            hi = d["x"][r, k + 1] if k + 1 < N else float(scenario.ship_max) + d["u"][r, k]
            yk = min(max(target, d["u"][r, k]), hi)
            ref[r, k] = max(0.0, round(yk - d["u"][r, k]))
    diff = d["q"] - ref
    return {
        f"echelon_{k + 1}": {
            "exact_match": float((np.abs(diff[:, k]) < 0.5).mean()),
            "within_2_units": float((np.abs(diff[:, k]) <= 2.0).mean()),
            "mean_signed_error": float(diff[:, k].mean()),
        }
        for k in range(N)
    }


def echelon_invariance(scenario, act, obs_mode: str, act_mode: str,
                       n_probe: int = 300) -> dict:
    """THE discriminating test: same echelon position, different raw split.

    Take a visited state, then **redistribute** stock between an installation and
    the pipeline feeding it, holding every echelon position fixed. A policy that
    found the echelon aggregation ships the same amount; a policy keying on raw
    components does not.

    Reported as the spread of shipments across splits, in units, per echelon.
    Near zero = the aggregation was recovered.
    """
    rng = np.random.default_rng(0)
    env = ClarkScarfEnv(scenario, observation_mode=obs_mode, action_mode=act_mode)
    N = scenario.n_echelons
    spreads = [[] for _ in range(N)]

    for p in range(n_probe):
        obs, _ = env.reset(seed=90_000 + p)
        for _ in range(int(rng.integers(3, scenario.horizon - 5))):
            obs, _, term, _, _ = env.step(act(obs[None, :], env.action_masks()[None, :])[0])
            if term:
                break
        base = env._state
        if scenario.leadtime != 2:
            continue  # with no pipeline there is nothing to redistribute
        qs = []
        for shift in (-4.0, -2.0, 0.0, 2.0, 4.0):
            st = base.copy()
            ok = True
            for k in range(N):
                # move `shift` units between installation k+1 and the pipeline
                # feeding it: u_j is unchanged for every j, x_j changes only
                # where the paper says it should
                if st.stock[k] - shift < 0 or st.pipeline[k][0] + shift < 0:
                    ok = False
                    break
                st.stock[k] -= shift
                st.pipeline[k][0] += shift
            if not ok:
                continue
            env._state = st
            o = env._get_obs(st)
            a = act(o[None, :], env.action_masks()[None, :])[0]
            qs.append(env._to_quantities(np.asarray(a, dtype=np.float64).reshape(-1))[:N])
        if len(qs) >= 3:
            arr = np.asarray(qs)
            for k in range(N):
                spreads[k].append(float(arr[:, k].max() - arr[:, k].min()))

    return {
        f"echelon_{k + 1}": {
            "n": len(spreads[k]),
            "mean_spread_units": float(np.mean(spreads[k])) if spreads[k] else float("nan"),
            "p90_spread_units": float(np.percentile(spreads[k], 90)) if spreads[k] else float("nan"),
        }
        for k in range(N)
    }


def offset_sweep(scenario, model_path: Path, obs_mode: str, act_mode: str,
                 n_seeds: int = 8192) -> dict:
    """Score the trained net with a CONSTANT OFFSET added to each echelon's target.

    Spec §14.2's fitted-rule evaluation, in the form this domain's diagnosis
    calls for. The sensitivity probe showed the whole PPO-to-DP gap is ~85%
    attributable to ONE scalar — the top echelon's order-up-to level sitting ~5
    units low — on a loss that punishes under-stocking 2.2x harder than over-.
    If that attribution is right, adding a constant to that single action
    component must recover most of the gap; if it does not, the attribution is
    wrong and the escalation ladder built on it is void.

    Only meaningful for `target_discrete`, where the action component IS the
    order-up-to level and a constant offset is a constant shift of the fitted
    rule. Scored under the Stage-4 protocol (same CRN block, deterministic,
    VecNormalize injected) so it is directly paired with the net.
    """
    assert act_mode == "target_discrete", (
        "an offset is only interpretable when the action IS the target"
    )
    act = _load_policy(model_path, scenario, obs_mode, act_mode)
    env0 = ClarkScarfEnv(scenario, observation_mode=obs_mode, action_mode=act_mode)
    bins = env0._target_bins()
    N = scenario.n_echelons
    out = {}

    for level in range(N):
        for delta in (0, 2, 5, 8):
            if delta == 0 and level > 0:
                continue                      # the zero row is shared
            def shifted(obs, envs, _l=level, _d=delta):
                m = np.stack([e.action_masks() for e in envs])
                a = np.asarray(act(obs, m), dtype=np.int64).copy()
                a[:, _l] = np.clip(a[:, _l] + _d, 0, bins[_l] - 1)
                return a
            per = rollout_seeds(scenario=scenario, observation_mode=obs_mode,
                                action_mode=act_mode, seeds=eval_seed_block(n_seeds),
                                act_fn=shifted, batch=64, progress_every=0)
            out[(level, delta)] = per["cost_total"]
            print(f"  echelon {level+1}  offset {delta:+d}  "
                  f"cost = {per['cost_total'].mean():.2f}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Read a trained policy back into "
                                            "Clark-Scarf policy structure.")
    p.add_argument("-s", "--scenario_name", default="n3_l2_p09",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--model-path", required=True)
    p.add_argument("-o", "--observation_mode", default=None)
    p.add_argument("-a", "--action_mode", default=None)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--offset-sweep", action="store_true",
                   help="spec §14.2 fitted-rule check: add a constant to each "
                        "echelon's target and re-score under the Stage-4 protocol")
    p.add_argument("--n-seeds", type=int, default=8192)
    args = p.parse_args()

    model_path = Path(args.model_path).resolve()
    run_dir = model_path.parent
    ra = {}
    f = run_dir / f"{args.scenario_name}_ppo_args.txt"
    if f.exists():
        ra = dict(line.split(": ", 1) for line in f.read_text().splitlines() if ": " in line)
    obs_mode = args.observation_mode or ra.get("observation_mode", "raw").strip()
    act_mode = args.action_mode or ra.get("action_mode", "ship_discrete").strip()

    scenario = SCENARIOS[args.scenario_name]

    if args.offset_sweep:
        print(f"=== fitted-rule offset sweep: {obs_mode} / {act_mode} ===")
        res = offset_sweep(scenario, model_path, obs_mode, act_mode, args.n_seeds)
        base = res[(0, 0)]
        print(f"\nnet (offset 0) = {base.mean():.2f}")
        print("paired deltas vs the net (negative = the fitted rule is better):")
        for (lvl, d), v in sorted(res.items()):
            if d == 0:
                continue
            diff = v - base
            se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
            print(f"  echelon {lvl+1}  offset {d:+d}: {diff.mean():+8.2f} "
                  f"+/- {se:.2f}   z = {diff.mean()/se:+7.2f}")
        return

    dp = ClarkScarfDP(scenario)
    act = _load_policy(model_path, scenario, obs_mode, act_mode)
    d = collect(scenario, act, obs_mode, act_mode, args.episodes)

    report = {
        "scenario": args.scenario_name,
        "observation_mode": obs_mode,
        "action_mode": act_mode,
        "n_decisions": int(len(d["t"])),
        "dp_ybar_midhorizon": dp.ybar[scenario.horizon // 2].tolist(),
        "structural_form": structural_form(scenario, d),
        "agreement_with_dp": agreement_with_dp(scenario, d, dp),
        "echelon_invariance": echelon_invariance(scenario, act, obs_mode, act_mode),
    }

    out = run_dir / "probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "readback.json").write_text(json.dumps(report, indent=2))

    print(f"=== policy readback: {obs_mode} / {act_mode} ===")
    print(f"DP critical numbers (mid-horizon): {report['dp_ybar_midhorizon']}")
    print("\n-- is it a base-stock rule? (implied y_i where the clip is slack)")
    for k, v in report["structural_form"].items():
        print(f"  {k}: ybar_fitted={v['ybar_fitted']:.1f}  IQR={v['iqr']:.1f}  "
              f"n={v['n_unclipped']}")
    print("\n-- agreement with the optimal policy")
    for k, v in report["agreement_with_dp"].items():
        print(f"  {k}: exact={v['exact_match']:.1%}  within2={v['within_2_units']:.1%}  "
              f"bias={v['mean_signed_error']:+.2f}")
    print("\n-- echelon invariance (same u, redistributed raw split)")
    for k, v in report["echelon_invariance"].items():
        print(f"  {k}: mean spread={v['mean_spread_units']:.2f} units  "
              f"p90={v['p90_spread_units']:.2f}  n={v['n']}")
    print(f"\n[probe] -> {out / 'readback.json'}")


if __name__ == "__main__":
    main()

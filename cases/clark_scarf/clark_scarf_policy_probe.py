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
import copy
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from clark_scarf_benchmark_dp import ClarkScarfDP
from clark_scarf_gym import ClarkScarfEnv
from clark_scarf_eval_common import eval_seed_block, rollout_seeds
from clark_scarf_mdp import (echelon_position, echelon_stock, init_state,
                             ship_capacity)
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

    # The normalizer this policy trained under. Two names are in play: runs that
    # carried a live selection callback wrote `vecnormalize.pkl` beside the
    # chosen artifact; every run since §9.7's machinery was removed writes
    # `vecnormalize_final.pkl` at the end of training. Looking for only the
    # first silently yields vn=None, and the probe then reads back a policy fed
    # observations it never saw -- the #E1 silent-corruption class, which cost
    # this campaign a full readback round (#E11).
    vn = None
    for cand in ("vecnormalize.pkl", "vecnormalize_final.pkl"):
        vn_path = model_path.parent / cand
        if vn_path.exists():
            vn = VecNormalize.load(str(vn_path), DummyVecEnv(
                [lambda: ClarkScarfEnv(scenario, observation_mode=obs_mode,
                                       action_mode=act_mode)]))
            vn.training = False
            print(f"[probe] VecNormalize loaded from {vn_path.name} "
                  f"(norm_obs={vn.norm_obs})")
            break

    # REFUSE rather than read back raw. A policy trained under normalization and
    # queried without it is a DIFFERENT policy, and every statistic below would
    # be a well-formed description of something that was never trained.
    if vn is None and args_f.exists():
        for line in args_f.read_text().splitlines():
            if line.startswith("norm_obs: ") and line.split(": ", 1)[1].strip() == "True":
                raise SystemExit(
                    f"[probe] REFUSING: the run's args record norm_obs=True and no "
                    f"normalizer was found beside {model_path.name} (looked for "
                    f"vecnormalize.pkl, vecnormalize_final.pkl). Reading this "
                    f"policy back on raw observations would describe a policy "
                    f"that was never trained.")

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


def sweep_policy(scenario, act, obs_mode: str, act_mode: str, period: int,
                 span: float = 1.5) -> dict:
    """ENUMERATE the input, read the output: the policy as a function, not a sample.

    This is a pure function plot and has nothing to do with training or eval
    realizations. For each echelon k and each echelon position u on a grid, one
    synthetic state is built, the policy is queried **deterministically**, and
    the ordered-up-to level y = u + q is recorded. One input, one output.

    Two properties the trajectory-based reads cannot have:

    * **The sweep is centred on the kink.** Each echelon is swept over its own
      window, `span` times the distance from its floor to its critical number,
      so the shutoff sits at the right third of the axis (span = 1.5): enough
      room past it to show the policy has actually stopped, without the dead
      space a wide sweep leaves. It still runs past every critical number, which
      a trajectory sample never reaches because a good policy does not visit
      deeply overstocked states.
    * **The source cannot bind.** `stock[k+1]` is set past the largest
      shippable quantity, so what is drawn is the order the policy WANTS, never
      a truncation of it. The top echelon draws on the outside supplier and is
      unconstrained already.

    **The state convention, stated because it is a choice.** Echelon position
    `u_k` sums inventory position over levels 0..k, so one `u_k` is reachable by
    many splits and this policy is measurably sensitive to which. The slice here
    holds **the rest of the chain at the DP's own optimum** — every level other
    than k carries the increment the optimal policy would hold, `ybar_j -
    ybar_{j-1}` — and varies only installation k's on-hand stock. So each curve
    reads "with the rest of the chain stocked optimally, how does this echelon
    respond to its own position?".

    An earlier convention put the whole echelon position on level k and emptied
    everything below it. That produced non-monotone curves — for the top echelon
    it emptied the entire downstream, a state no sane chain occupies — and is
    the wrong question: it varies the echelon of interest AND the rest of the
    chain at once. Recorded so it is not re-invented.
    """
    from clark_scarf_gym import observation as gym_observation

    env = ClarkScarfEnv(scenario, observation_mode=obs_mode, action_mode=act_mode)
    N, L = scenario.n_echelons, scenario.leadtime
    max_ship = float(env.n_bins - 1) if act_mode == "ship_discrete" else float(scenario.ship_max)
    env.reset(seed=0)          # stand the env up so its decoders are live
    base, _ = init_state(scenario, episode_seed=0)
    ybar = ClarkScarfDP(scenario).ybar[period][:N].astype(float)
    # the optimal per-level increment: what level j holds when every echelon
    # sits at its own critical number
    inc = [float(ybar[0])] + [float(ybar[j] - ybar[j - 1]) for j in range(1, N)]

    # Each echelon gets its OWN window, centred on its critical number, so the
    # kink lands mid-axis instead of crowding the left with dead space to its
    # right. Under this convention u_k starts at ybar_{k-1} (the chain below is
    # at the optimum), and the kink sits at u_k = ybar_k, i.e. own = inc[k] —
    # so sweeping own over [0, span*inc[k]] puts the kink at the middle when
    # span = 2. Lengths differ per echelon, hence lists rather than one array.
    out = {"u": [], "y": [], "q": []}

    for k in range(N):
        width = max(4.0, span * inc[k])
        grid = np.arange(0.0, float(np.ceil(width)) + 1.0)
        uu = np.zeros(len(grid)); yy = np.zeros(len(grid)); qq = np.zeros(len(grid))
        for i, own in enumerate(grid):
            st = copy.deepcopy(base)
            st.period = period
            st.terminated = False
            st.pipeline = [[0.0] * L for _ in range(len(st.pipeline))]
            st.stock = [0.0] * len(st.stock)
            for j in range(N):                          # rest of the chain at the optimum
                st.stock[j] = inc[j]
            st.stock[k] = float(own)                    # the one level we vary
            if k < N - 1:
                st.stock[k + 1] = max(inc[k + 1], max_ship + 1.0)   # source cannot bind
            u = echelon_position(scenario, st)[k]
            uu[i] = float(u)
            o = gym_observation(scenario, st, obs_mode)
            # the synthetic state IS the query; the env is only a decoder here,
            # so point it at that state before masks/quantities are read off it
            env._state = st
            a = act(o[None, :], env.action_masks()[None, :])[0]
            q = env._to_quantities(np.asarray(a, dtype=np.float64).reshape(-1))
            qq[i] = float(q[k])
            yy[i] = float(u) + float(q[k])
        out["u"].append(uu); out["y"].append(yy); out["q"].append(qq)
    return out


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

    NOT CURRENTLY RUNNABLE, and deliberately so. It required `target_discrete`,
    where the action component IS the order-up-to level so that a constant
    offset is a constant shift of the fitted rule. That mode has been removed:
    its decode computed `echelon_position` from simulator state and handed the
    agent the structure under test, which is what voided this campaign's
    original answer (ESCALATION #E11).

    Re-implementing it on an echelon-free encoding is owed and is not hard —
    the implied order-up-to level is `u_k + q_k` from the EXECUTED shipment
    rather than read off the action, which works for any mode and is what #E11
    said the reopened campaign should have measured in the first place. Left
    failing loudly rather than silently adapted, because a probe that quietly
    changes what it measures is worse than one that stops.
    """
    raise NotImplementedError(
        "offset sweep needs re-implementation on an echelon-free encoding: "
        "derive the implied order-up-to level as u_k + q_k from the executed "
        "shipment instead of reading it off the action (ESCALATION #E11)"
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

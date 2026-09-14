"""Recover inventory policy structure from a trained net (campaign question 2).

Sweeps a trained policy over a synthetic state grid and asks what *rule* it
implements: is the post-order position `y = x + q(x)` flat in `x` below a
threshold (order-up-to / base-stock), and where is the reorder point? The
recovered `(s, S)` is then scored against the DP table where one exists, and —
because a fitted rule is only interesting if it *performs* — the fitted policy
can be replayed through the real simulator on the campaign's CRN seed block.

Observations are built by calling the gym's own `_get_obs()` on a synthetic
state, so the probe cannot drift from what the policy was trained to read.

Reported per period t:
- `S_hat`  — modal post-order position over states where the net orders
- `flat`   — max |y - S_hat| over the ordering region (0 = exactly order-up-to)
- `s_hat`  — largest x at which the net still orders (the reorder point; equals
             `S_hat - 1` for a pure base-stock rule with no fixed cost)
- vs DP    — `S_dp`, `s_dp`, and action agreement over the shared grid
- `d_sens` — spread of q across demand feature values, for obs modes that carry
             `info.demand`. This is a *direct* measurement of whether the net
             uses a feature the IR predicts is uninformative (campaign A4).

`--composition` runs a second, independent probe (campaign question 4): is
inventory position a sufficient statistic? It rolls the net on-policy to learn
which (period, IP) states it actually visits, then at fixed IP varies only the
pipeline's *composition* over visited splits. Two readings — the spread in `q`
(the existence test) and the gradient against how late the outstanding stock
arrives (the structure) — plus `--score-fitted-ip`, which projects the net onto
the best IP-only rule fitted to its own behaviour and scores the residual. That
last one is the reliable test: at `lt` the gradient is nonzero but the
projection costs nothing, so the surface reading alone would mislead.

Usage:
    python inv_single_policy_probe.py -s simple -o vec \
        --model-path results/simple/PPO_.../best_model.zip
    python inv_single_policy_probe.py -s simple_k -o vec \
        --model-path ... --score-fitted --n-seeds 8192
    python inv_single_policy_probe.py -s slt -o vec --model-path ... \
        --composition --comp-period 10 --score-fitted-ip --n-seeds 8192
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from inv_single_eval_common import eval_seed_block, rollout_seeds, write_record
from inv_single_gym import InvSingleEnv
from inv_single_scenarios import SCENARIOS

OBS_MODES = ["vec", "vec_ip", "vec_ctx", "vec_ctx_slt", "vec_ip_ctx_slt"]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recover policy structure from a trained inv_single net.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None)
    p.add_argument("-s", "--scenario_name", type=str, default="simple",
                   choices=list(SCENARIOS.keys()))
    # spec 9.6: a GENERALIST is probed by enumerating its cells, one row per
    # cell, exactly as `inv_single_ppo_eval.py -g` scores it. The DP comparison
    # uses each cell's DE-PADDED twin (`cells_depadded`): a padded cell's
    # dynamics equal the unpadded cell's, but FiniteHorizonDP reads
    # `leadtime.max()` for the protection interval, so the padded law would give
    # it the family maximum instead of the cell's own lead time. #E1 scored the
    # per-cell bars the same way, so this is the comparison those bars support.
    p.add_argument("-g", "--grid_name", type=str, default=None,
                   help="probe every cell of this GRIDS entry; overrides -s")
    p.add_argument("-o", "--observation_mode", type=str, default="vec", choices=OBS_MODES)
    p.add_argument("-a", "--action_mode", type=str, default="discrete",
                   choices=["continuous", "discrete", "hurdle"])
    p.add_argument("--x-min", type=int, default=-10)
    p.add_argument("--x-max", type=int, default=30)
    p.add_argument("--periods", type=int, nargs="+", default=None,
                   help="Periods to probe; default = every period.")
    p.add_argument("--demand-value", type=int, default=None,
                   help="Demand feature value for obs modes that carry it "
                        "(default: the demand mean, rounded).")
    p.add_argument("--demand-sens", type=int, nargs="+", default=None,
                   help="Demand values to measure sensitivity over "
                        "(default: mean-relative spread 0, mean, 2*mean).")
    p.add_argument("--no-dp", action="store_true",
                   help="Skip the DP comparison (for rungs with no valid DP bar).")
    p.add_argument("--score-fitted", action="store_true",
                   help="Replay the fitted (s,S) rule through the simulator on "
                        "the CRN seed block and write a §9.3 record.")
    p.add_argument("--composition", action="store_true",
                   help="campaign question 4: at fixed inventory position, does the "
                        "order depend on the pipeline's composition? Rolls the net "
                        "on-policy first, so only visited states are queried.")
    p.add_argument("--comp-ip-range", type=int, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="inventory-position band to sweep (default: the visited IQR)")
    p.add_argument("--comp-period", type=int, default=10,
                   help="period to hold fixed while composition varies")
    p.add_argument("--comp-max-comps", type=int, default=0,
                   help="cap on compositions queried per inventory position, "
                        "0 = no cap (the default: every visited composition is "
                        "used). A cap SUBSAMPLES AT RANDOM under --comp-seed; it "
                        "must never take a prefix, because the composition list "
                        "is sorted by (inventory, pipeline) and its head is the "
                        "low-inventory / heavy-pipeline states — exactly the ones "
                        "with the largest lateness effect")
    p.add_argument("--comp-seed", type=int, default=0,
                   help="seed for the on-policy roll that harvests visited states")
    p.add_argument("--comp-steps", type=int, default=900,
                   help="vector steps to roll when harvesting visited states")
    p.add_argument("--score-fitted-ip", action="store_true",
                   help="score the best IP-only rule fitted to the net, paired against it")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--outfile", type=str, default=None)
    return p


# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------

def _make_obs_builder(scenario, observation_mode: str, action_mode: str):
    """A closure mapping (period, x, demand) -> the agent's observation.

    Delegates to the gym's `_get_obs`, so the probe reads exactly what the
    policy was trained on. `x` is net inventory for the pipeline modes and
    inventory position for the IP modes — in both cases the pipeline is held at
    zero, which is the pre-order state of a policy that has just received
    everything outstanding (and is *the* state at L = 0, where the pipeline slot
    is structurally 0).
    """
    import dataclasses

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = InvSingleEnv(
            scenario=scenario,
            observation_mode=observation_mode,
            action_mode=action_mode,
        )
    env.reset(seed=0)
    base = env._state

    def build(period: int, x: int, demand: int) -> np.ndarray:
        env._state = dataclasses.replace(
            base,
            period=int(period),
            inventory=int(x),
            pipeline=[0] * len(base.pipeline),
            demand=int(demand),
            terminated=False,
        )
        return env._get_obs()

    return build, env


def sweep(
    model: PPO,
    normalizer: VecNormalize | None,
    build_obs,
    periods: np.ndarray,
    x_grid: np.ndarray,
    demand: int,
    probe_env: InvSingleEnv,
) -> np.ndarray:
    """Deterministic order quantity q[t, x] over the grid.

    Actions are decoded by the env's own `decode_action`, so a scaled action
    mode is read in order units exactly as `step` would read it.
    """
    q = np.zeros((len(periods), len(x_grid)))
    for ti, t in enumerate(periods):
        obs = np.stack([build_obs(t, x, demand) for x in x_grid])
        if normalizer is not None:
            obs = normalizer.normalize_obs(obs)
        actions, _ = model.predict(obs, deterministic=True)
        actions = np.asarray(actions).reshape(len(x_grid), -1)
        q[ti] = [probe_env.decode_action(a) for a in actions]
    return np.maximum(q, 0.0)


def fit_rule(x_grid: np.ndarray, q_row: np.ndarray, tol: float = 0.5) -> dict:
    """Fit an order-up-to rule to one period's action curve.

    `S_hat` is the modal post-order position over the ordering region (robust
    to a few boundary states); `flat` is the worst deviation from it, which is
    the actual test of order-up-to form.
    """
    orders = q_row > tol
    if not orders.any():
        return {"S_hat": float("nan"), "flat": float("nan"),
                "s_hat": float("nan"), "n_order": 0}
    y = x_grid + q_row
    y_ord = y[orders]
    vals, counts = np.unique(np.round(y_ord).astype(int), return_counts=True)
    S_hat = float(vals[int(np.argmax(counts))])
    return {
        "S_hat": S_hat,
        "flat": float(np.max(np.abs(y_ord - S_hat))),
        "s_hat": float(x_grid[orders].max()),
        "n_order": int(orders.sum()),
    }


# ---------------------------------------------------------------------------
# Composition probe (campaign question 4): is inventory position sufficient?
# ---------------------------------------------------------------------------

def _roll_visited(model, vecnorm_path, scenario, observation_mode, action_mode,
                  n_envs: int, n_steps: int, seed: int = 0):
    """Roll the net on-policy; record the (period, IP) states it actually visits.

    This probe MUST be on-distribution. Under a stochastic lead time most
    synthetic (inventory, pipeline) splits at a given inventory position are
    never reached, and querying the net there measures extrapolation rather
    than policy — the error #E10 made and corrected.

    Returns (states, actions): both keyed by (period, IP), holding the visited
    (inventory, pipeline) compositions and the order the net emitted.
    """
    from collections import defaultdict

    def mk():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return InvSingleEnv(scenario, observation_mode=observation_mode,
                                action_mode=action_mode)

    venv = DummyVecEnv([mk for _ in range(n_envs)])
    if vecnorm_path is not None and Path(vecnorm_path).exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        raw_envs = venv.venv.envs
    else:
        raw_envs = venv.envs

    states, actions = defaultdict(set), defaultdict(list)
    # the visited set is the probe's sample frame, so it is seeded: an unseeded
    # roll moves the reported IP band and the gradient with it
    venv.seed(seed)
    obs = venv.reset()
    for _ in range(n_steps):
        keys, comps = [], []
        for e in raw_envs:
            st = e._state
            keys.append((int(st.period), int(round(st.inventory + sum(st.pipeline)))))
            comps.append((int(st.inventory), tuple(int(v) for v in st.pipeline)))
        act, _ = model.predict(obs, deterministic=True)
        flat = np.asarray(act).reshape(len(raw_envs), -1)
        for k, c, a in zip(keys, comps, flat):
            states[k].add(c)
            actions[k].append(float(raw_envs[0].decode_action(a)))
        obs, _, _, _ = venv.step(act)
    return states, actions


def composition_probe(model, normalizer, scenario, observation_mode, action_mode,
                      states, ip_lo: int, ip_hi: int, period: int, max_comps: int,
                      seed: int = 0):
    """At a FIXED inventory position, how much does the order move with the
    pipeline's composition — and in which direction?

    Two readings. The spread is the existence test: if IP were a sufficient
    statistic the net's action would be constant across compositions sharing an
    IP, so a nonzero spread is exactly the failure of sufficiency. The lateness
    gradient is the structure: `lateness = sum(k * pipe[k]) / sum(pipe)` is how
    far out the outstanding stock is, and a positive gradient says the net
    orders MORE when the same IP is backloaded — the crossover-aware response.
    """
    import dataclasses

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = InvSingleEnv(scenario, observation_mode=observation_mode,
                           action_mode=action_mode)
    env.reset(seed=0)
    base = env._state

    def order_at(inv, pipe):
        env._state = dataclasses.replace(
            base, period=int(period), inventory=int(inv),
            pipeline=list(pipe), terminated=False,
        )
        obs = env._get_obs().reshape(1, -1)
        if normalizer is not None:
            obs = normalizer.normalize_obs(obs)
        a, _ = model.predict(obs, deterministic=True)
        return float(env.decode_action(np.asarray(a).reshape(-1)))

    rng = np.random.default_rng(seed)
    rows = []
    for ip in range(ip_lo, ip_hi + 1):
        comps = sorted({c for (t, i), cs in states.items() if i == ip for c in cs})
        if max_comps and len(comps) > max_comps:
            # random, never a prefix: see --comp-max-comps
            comps = [comps[i] for i in sorted(rng.permutation(len(comps))[:max_comps])]
        pts = []
        for inv, pipe in comps:
            tot = sum(pipe)
            if tot == 0:
                continue  # no composition to vary: IP is trivially sufficient here
            late = sum(k * v for k, v in enumerate(pipe)) / tot
            pts.append((late, order_at(inv, pipe)))
        if len(pts) < 8:
            continue
        pts.sort()
        qs = np.array([o for _, o in pts])
        third = max(1, len(pts) // 3)
        lo, hi = qs[:third].mean(), qs[-third:].mean()
        rows.append({"ip": ip, "n": len(pts), "spread": float(qs.max() - qs.min()),
                     "early": float(lo), "late": float(hi), "grad": float(hi - lo)})
    return rows


def fit_ip_only(actions) -> tuple[dict, dict]:
    """The best IP-only rule *fitted to the net itself*.

    The strongest available control: not a rule from theory but the net's own
    policy projected onto inventory position, taking the modal action in every
    (period, IP) cell it visited. Whatever it loses by is information the net
    used and no function of IP can express.
    """
    from collections import Counter, defaultdict

    per_cell = {k: Counter(v).most_common(1)[0][0] for k, v in actions.items()}
    pooled = defaultdict(list)
    for (t, ip), v in actions.items():
        pooled[ip] += v
    per_ip = {ip: Counter(v).most_common(1)[0][0] for ip, v in pooled.items()}
    return per_cell, per_ip


def _ip_rule_lookup(per_cell, per_ip):
    """Read the fitted rule, extrapolating to inventory positions never visited.

    Falling back to "order nothing" is not neutral: an unvisited IP is one the
    net's own policy avoids, and answering 0 there sends the episode further
    away, so a single miss can run away. At `lt` that put 41,671 cost on 7 of
    8192 seeds and inverted the mean while the median moved -1. Extrapolating
    from the NEAREST visited position instead keeps the rule a function of IP —
    which is the whole point of the control — without inventing a cliff.
    """
    by_period = {}
    for (t, ip), q in per_cell.items():
        by_period.setdefault(t, {})[ip] = q
    sorted_ips = {t: sorted(d) for t, d in by_period.items()}
    pooled_ips = sorted(per_ip)

    def lookup(period: int, ip: int) -> float:
        d = by_period.get(period)
        if d is not None:
            if ip in d:
                return d[ip]
            near = min(sorted_ips[period], key=lambda k: abs(k - ip))
            return d[near]
        if not pooled_ips:
            return 0.0
        return per_ip[min(pooled_ips, key=lambda k: abs(k - ip))]

    return lookup


def _probe_grid(args, model, normalizer) -> None:
    """One threshold row per cell — RQ-6's declared instrument (spec 9.6).

    The POLICY is read on the padded cell (that is the observation it was
    trained on, `pipeline_len = lt_max + 1`); the DP is built on the cell's
    DE-PADDED twin, because `FiniteHorizonDP` takes its protection interval
    from `leadtime.max()` and a padded LT=0 cell carries the family maximum
    there. Same split #E1 used for the per-cell bars."""
    from inv_single_benchmark_dp import FiniteHorizonDP
    from inv_single_grids import GRIDS, cells_depadded

    grid = GRIDS[args.grid_name]
    padded = dict(grid.cells)
    depad = {cid: sc for cid, _slug, sc in cells_depadded(args.grid_name)}
    periods = np.array(args.periods if args.periods is not None else [10])
    print(f"\n[probe] grid     : {args.grid_name}  ({len(padded)} cells)  "
          f"obs={args.observation_mode}  t={list(periods)}")
    print(f"[probe] x grid   : {args.x_min}…{args.x_max}   (pipeline held at 0)")
    print(f"\n{'cell':<30} {'S_hat':>7} {'S_dp':>7} {'dS':>6} "
          f"{'s_hat':>6} {'s_dp':>6} {'ds':>5} {'agree':>7}")
    dS_all, ds_all, ag_all = [], [], []
    for cid, sc_pad in grid.cells:
        build_obs, probe_env = _make_obs_builder(
            sc_pad, args.observation_mode, args.action_mode)
        x_grid = np.arange(args.x_min, args.x_max + 1)
        d_mean = int(round(sc_pad.demand.mean()))
        q = sweep(model, normalizer, build_obs, periods, x_grid, d_mean, probe_env)
        dp = FiniteHorizonDP(depad[cid])
        fit = fit_rule(x_grid, q[0])
        q_dp = np.array([dp.act(int(periods[0]), int(x)) for x in x_grid], dtype=float)
        dpf = fit_rule(x_grid, q_dp)
        ag = float(np.mean(np.round(q[0]) == np.round(q_dp)))
        dS, ds = fit["S_hat"] - dpf["S_hat"], fit["s_hat"] - dpf["s_hat"]
        dS_all.append(dS); ds_all.append(ds); ag_all.append(ag)
        print(f"{cid:<30} {fit['S_hat']:7.1f} {dpf['S_hat']:7.1f} {dS:+6.1f} "
              f"{fit['s_hat']:6.1f} {dpf['s_hat']:6.1f} {ds:+5.1f} {ag:7.3f}")
    dS_all, ds_all, ag_all = map(np.array, (dS_all, ds_all, ag_all))
    print(f"\n[probe] S_hat - S_dp : mean {np.nanmean(dS_all):+.2f}  "
          f"median {np.nanmedian(dS_all):+.1f}  max |·| {np.nanmax(np.abs(dS_all)):.1f}")
    print(f"[probe] s_hat - s_dp : mean {np.nanmean(ds_all):+.2f}  "
          f"median {np.nanmedian(ds_all):+.1f}  max |·| {np.nanmax(np.abs(ds_all)):.1f}")
    print(f"[probe] action agreement: mean {ag_all.mean():.3f}  "
          f"min {ag_all.min():.3f}  cells >= 0.90: {int((ag_all>=0.90).sum())}/{len(ag_all)}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    model_path = Path(args.model_path).resolve()
    model_dir = model_path.parent
    if args.grid_name:
        # the VecNormalize stats and the model were built on GRID CELLS, whose
        # obs width is the family-maximum pipeline; the default scenario's is
        # narrower, and VecNormalize.load asserts shape equality. Attach the
        # dummy to a cell, not to `simple`.
        from inv_single_grids import GRIDS
        scenario = GRIDS[args.grid_name].cells[0][1]
    else:
        scenario = SCENARIOS[args.scenario_name]
    assert not callable(scenario), "the probe needs a fixed scenario, not a sampler"

    vecnorm = Path(args.vecnorm_path) if args.vecnorm_path else model_dir / "vecnormalize.pkl"
    model = PPO.load(str(model_path), device="cpu")

    build_obs, probe_env = _make_obs_builder(
        scenario, args.observation_mode, args.action_mode
    )
    normalizer = None
    if vecnorm.exists():
        # VecNormalize.load needs a venv to attach to; only normalize_obs is used
        dummy = DummyVecEnv([lambda: probe_env])
        normalizer = VecNormalize.load(str(vecnorm), dummy)
        normalizer.training = False

    if args.composition:
        # The grid sweep below holds the pipeline at zero, which is the whole
        # state at L=0 but a measure-zero corner under a stochastic lead time.
        # Question 4 needs the opposite: real compositions, at a fixed IP.
        states, actions = _roll_visited(
            model, vecnorm if vecnorm.exists() else None, scenario,
            args.observation_mode, args.action_mode,
            n_envs=args.batch_envs, n_steps=args.comp_steps, seed=args.comp_seed,
        )
        visited_ips = np.array([ip for (_, ip) in states])
        if args.comp_ip_range is not None:
            ip_lo, ip_hi = args.comp_ip_range
        else:
            ip_lo, ip_hi = (int(np.percentile(visited_ips, 25)),
                            int(np.percentile(visited_ips, 75)))
        print(f"\n[probe] model    : {model_path}")
        print(f"[probe] scenario : {args.scenario_name}  obs={args.observation_mode}")
        print(f"[probe] visited  : {sum(len(v) for v in actions.values())} transitions, "
              f"{len(states)} (period, IP) cells")
        print(f"[probe] IP band  : {ip_lo}…{ip_hi}   period held at {args.comp_period}")

        rows = composition_probe(
            model, normalizer, scenario, args.observation_mode, args.action_mode,
            states, ip_lo, ip_hi, args.comp_period, args.comp_max_comps,
            seed=args.comp_seed,
        )
        print(f"\n{'IP':>4} {'n':>4} {'spread':>7} {'early':>7} {'late':>7} {'grad':>7}")
        for r in rows:
            print(f"{r['ip']:4d} {r['n']:4d} {r['spread']:7.2f} "
                  f"{r['early']:7.2f} {r['late']:7.2f} {r['grad']:+7.2f}")
        if rows:
            sp = np.array([r["spread"] for r in rows])
            gr = np.array([r["grad"] for r in rows])
            print(f"\n[probe] spread in q at FIXED inventory position: "
                  f"median {np.median(sp):.2f}, max {sp.max():.2f}")
            print("        (0 everywhere = IP is a sufficient statistic)")
            print(f"[probe] lateness gradient (late - early): mean {gr.mean():+.2f}, "
                  f"positive in {int((gr > 0).sum())}/{len(gr)} bands")
            print("        (positive = orders MORE when the same IP is backloaded)")

        if args.score_fitted_ip:
            lookup = _ip_rule_lookup(*fit_ip_only(actions))

            def act_fn(obs, raw_envs):
                acts = np.empty((len(raw_envs), 1), dtype=np.float32)
                for i, e in enumerate(raw_envs):
                    st = e._state
                    ip = int(round(st.inventory + sum(st.pipeline)))
                    acts[i, 0] = lookup(int(st.period), ip)
                return acts

            # replayed through the encoding the net used: the fitted values ARE
            # the net's own emitted actions, so this decodes them identically
            per_seed = rollout_seeds(
                scenario=scenario, observation_mode=args.observation_mode,
                action_mode=args.action_mode, seeds=eval_seed_block(args.n_seeds),
                act_fn=act_fn, batch=args.batch_envs, vecnorm_path=None,
            )
            outfile = Path(args.outfile if args.outfile else
                           model_dir / f"fitted_iponly_eval_{args.scenario_name}.tsv")
            print(f"\n[probe] scoring the fitted IP-ONLY rule on {args.n_seeds} CRN seeds")
            write_record(outfile, args.scenario_name,
                         f"fitted_iponly[{model_dir.name}]", per_seed)
        return

    if args.grid_name:
        _probe_grid(args, model, normalizer)
        return

    periods = np.array(
        args.periods if args.periods is not None else range(scenario.horizon)
    )
    x_grid = np.arange(args.x_min, args.x_max + 1)
    d_mean = int(round(scenario.demand.mean()))
    demand = args.demand_value if args.demand_value is not None else d_mean

    q = sweep(model, normalizer, build_obs, periods, x_grid, demand, probe_env)

    dp = None
    if not args.no_dp:
        from inv_single_benchmark_dp import FiniteHorizonDP
        dp = FiniteHorizonDP(scenario)

    print(f"\n[probe] model    : {model_path}")
    print(f"[probe] scenario : {args.scenario_name}  obs={args.observation_mode}"
          f"  demand_feature={demand}")
    print(f"[probe] x grid   : {args.x_min}…{args.x_max}   (pipeline held at 0)")

    header = f"\n{'t':>3} {'S_hat':>7} {'flat':>6} {'s_hat':>6} {'n_ord':>6}"
    if dp is not None:
        header += f" {'S_dp':>6} {'s_dp':>6} {'dS':>6} {'agree':>7}"
    print(header)

    rows = []
    for ti, t in enumerate(periods):
        fit = fit_rule(x_grid, q[ti])
        line = (f"{int(t):3d} {fit['S_hat']:7.1f} {fit['flat']:6.2f} "
                f"{fit['s_hat']:6.1f} {fit['n_order']:6d}")
        if dp is not None:
            q_dp = np.array([dp.act(int(t), int(x)) for x in x_grid], dtype=float)
            dp_fit = fit_rule(x_grid, q_dp)
            agree = float(np.mean(np.round(q[ti]) == np.round(q_dp)))
            line += (f" {dp_fit['S_hat']:6.1f} {dp_fit['s_hat']:6.1f} "
                     f"{fit['S_hat'] - dp_fit['S_hat']:+6.1f} {agree:7.3f}")
            fit |= {"S_dp": dp_fit["S_hat"], "s_dp": dp_fit["s_hat"], "agree": agree}
        rows.append((int(t), fit))
        print(line)

    S_hats = np.array([r[1]["S_hat"] for r in rows], dtype=float)
    flats = np.array([r[1]["flat"] for r in rows], dtype=float)
    print(f"\n[probe] S_hat over periods : mean {np.nanmean(S_hats):.2f}  "
          f"min {np.nanmin(S_hats):.1f}  max {np.nanmax(S_hats):.1f}")
    print(f"[probe] order-up-to flatness: max |y - S_hat| = {np.nanmax(flats):.2f} "
          f"(0 = exactly order-up-to)")
    if dp is not None:
        agrees = np.array([r[1]["agree"] for r in rows])
        dS = S_hats - np.array([r[1]["S_dp"] for r in rows], dtype=float)
        print(f"[probe] vs DP: mean action agreement {agrees.mean():.3f}   "
              f"S_hat - S_dp: mean {np.nanmean(dS):+.2f}, max |·| {np.nanmax(np.abs(dS)):.1f}")

    # Demand-feature sensitivity: for an obs mode carrying info.demand, how much
    # does the net's order move when only that feature moves? The IR predicts
    # zero (iid demand, no latent), so a large spread is a finding.
    if "d" in args.observation_mode.split("_"):
        d_vals = args.demand_sens or [0, d_mean, 2 * d_mean]
        qs = np.stack([
            sweep(model, normalizer, build_obs, periods, x_grid, d, probe_env)
            for d in d_vals
        ])
        spread = float(np.max(qs.max(axis=0) - qs.min(axis=0)))
        mean_spread = float(np.mean(qs.max(axis=0) - qs.min(axis=0)))
        print(f"\n[probe] demand-feature sensitivity over d in {d_vals}:")
        print(f"        max spread in q = {spread:.3f}   mean spread = {mean_spread:.3f}")
        print("        (the IR predicts 0: iid demand, no latent ⇒ uninformative)")

    if args.score_fitted:
        # the replay looks the rule up per period, so a partial --periods sweep
        # would silently order NOTHING in the unprobed periods and report a
        # catastrophic cost that says nothing about the policy
        missing = sorted(set(range(scenario.horizon)) - {int(t) for t in periods})
        assert not missing, (
            f"--score-fitted needs a rule for every period; --periods omitted "
            f"{len(missing)} of {scenario.horizon} (e.g. {missing[:5]}). Drop "
            "--periods to sweep the full horizon."
        )
        # replay the fitted per-period rule (order up to S_hat[t] when x <= s_hat[t])
        S_by_t = {int(t): rows[i][1]["S_hat"] for i, t in enumerate(periods)}
        s_by_t = {int(t): rows[i][1]["s_hat"] for i, t in enumerate(periods)}
        lt_max = scenario.leadtime.max()

        def act_fn(obs, raw_envs):
            acts = np.empty((len(raw_envs), 1), dtype=np.float32)
            for i, e in enumerate(raw_envs):
                st = e._state
                x = st.inventory if lt_max == 0 else st.inventory + sum(st.pipeline)
                S = S_by_t.get(st.period, float("nan"))
                s = s_by_t.get(st.period, float("nan"))
                acts[i, 0] = (
                    0.0 if not np.isfinite(S) or x > s else max(0.0, S - x)
                )
            return acts

        per_seed = rollout_seeds(
            scenario=scenario,
            observation_mode=args.observation_mode,
            # the fitted rule emits order quantities directly, so it is scored
            # through the identity action encoding whatever the net was trained on
            action_mode="continuous",
            seeds=eval_seed_block(args.n_seeds),
            act_fn=act_fn,
            batch=args.batch_envs,
            vecnorm_path=None,
        )
        outfile = Path(
            args.outfile if args.outfile
            else model_dir / f"fitted_rule_eval_{args.scenario_name}.tsv"
        )
        print(f"\n[probe] scoring the FITTED rule on {args.n_seeds} CRN seeds")
        write_record(outfile, args.scenario_name, f"fitted_sS[{model_dir.name}]", per_seed)


if __name__ == "__main__":
    main()

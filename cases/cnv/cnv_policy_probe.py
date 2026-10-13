"""Policy-interpretation probe for cnv (spec §14.1) — the instrument behind the
declared tier-2 stances (``research_questions`` in ``cnv_schema.json``):

- **confirm over-ordering** (Proposition 1(b)): sweep a belief-fed policy's
  stocking level over the posterior (shape, rate) x critical ratio and compare
  it with the myopic newsvendor level y^M and, where a cached table exists,
  the exact DP level y^T. Reports the fraction of belief states at which the
  net stocks at or above y^M, the mean excess, and the action agreement with
  the DP.
- **on-policy agreement**: roll out episodes and compare the net's order with
  the myopic level OF ITS OWN observation type (timing / full: the conjugate
  posterior; event: the grid posterior, ``MyopicEvent``) and the DP level at
  the states it actually visits (the surface above covers states a policy may
  never reach). Every observation mode is probed here, raw and belief alike;
  the policy is loaded and stepped exactly as ``cnv_ppo_eval.py`` scores it
  (VecNormalize, the raw modes' frame stack, deterministic argmax). One row
  per visited state goes to ``onpolicy_{target}.tsv`` — the readback's data.
- **bypass the sufficient statistic** (Proposition 2): the raw-vs-belief
  contrast is an outcome comparison read off two spec-§9 evals on one seed
  block; ``--pair A.seeds.tsv B.seeds.tsv`` prints the paired delta and its
  SE, which is the number INTERPRET.md quotes.

Instrument outputs (raw measurements, regenerable) land beside the model in
``{model_dir}/probe/`` (spec §8.4); the conclusions drawn from them are
INTERPRET.md's. ``--model-path myopic_timing`` / ``myopic_full`` /
``myopic_event`` runs the instruments on the myopic rule itself — the
self-check that the probe reads its reference correctly (agreement must be
100%). ``--dp-auto`` loads each row's cached DP table
(``results/{target}/benchmark/dp/{cell}_{type}.npz``) when one exists.

    python cnv_policy_probe.py --model-path results/base/PPO_.../base_ppo.zip -s base --dp-auto
    python cnv_policy_probe.py --model-path myopic_event -g paper510 -o event_raw --n-seeds 64
    python cnv_policy_probe.py --pair results/base/benchmark/benchmark_myopic_timing_eval_base.seeds.tsv \\
                                      results/base/PPO_.../ppo_eval_base.seeds.tsv
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from cnv_bayes import gamma_poisson_quantile
from cnv_benchmark_common import eval_seed_block, eval_targets
from cnv_benchmark_myopic import MyopicEvent, myopic_full, myopic_timing
from cnv_grids import GRIDS
from cnv_gym import ALL_OBSERVATION_MODES, CnvEnv
from cnv_scenarios import SCENARIOS

HERE = Path(__file__).resolve().parent

RAW_MODES = ("event_raw", "timing_raw", "full_raw")
MYOPIC_RULES = ("myopic_timing", "myopic_full", "myopic_event")


def obs_type(mode: str) -> str:
    """The observation TYPE a mode renders: timing / full / event (the first token of every mode name)."""
    kind = mode.split("_")[0]
    assert kind in ("timing", "full", "event"), f"no myopic reference for observation mode {mode!r}"
    return kind


def is_belief_mode(mode: str) -> bool:
    """Modes that render the conjugate posterior — the surface sweep sets their state directly."""
    return mode.startswith(("timing_belief", "full_belief"))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe a cnv policy's stocking rule against y^M and the DP.")
    p.add_argument("--model-path", type=str, default=None,
                   help="SB3 zip, or 'myopic_timing' / 'myopic_full' / 'myopic_event' for the self-check")
    p.add_argument("--vecnorm-path", type=str, default=None)
    p.add_argument("-s", "--scenario_name", type=str, default="base")
    p.add_argument("-g", "--grid_name", type=str, default=None)
    p.add_argument("-o", "--observation_mode", type=str, default="timing_belief",
                   choices=list(ALL_OBSERVATION_MODES),
                   help="belief modes get the surface sweep and the on-policy part; every other mode the on-policy part")
    p.add_argument("-a", "--action_mode", type=str, default="order_discrete")
    p.add_argument("--frame-stack", type=int, default=None,
                   help="default: the horizon for the raw modes, 1 otherwise (as cnv_ppo_eval.py)")
    p.add_argument("--n-seeds", type=int, default=512, help="on-policy agreement episodes")
    p.add_argument("--first-seed", type=int, default=1_000_000, help="a screen block, never the protocol block")
    p.add_argument("--shape-max-mult", type=float, default=4.0,
                   help="sweep posterior shape up to this multiple of the prior shape + horizon x prior mean")
    p.add_argument("--no-surface", action="store_true", help="skip the belief-surface sweep")
    p.add_argument("--dp-table", type=str, default=None, help="cached DP table (.npz) for the y^T column")
    p.add_argument("--dp-auto", action="store_true",
                   help="per row, load results/{target}/benchmark/dp/{cell}_{type}.npz when it exists")
    p.add_argument("--pair", nargs=2, metavar=("A_SEEDS_TSV", "B_SEEDS_TSV"),
                   help="paired delta B - A on shared seeds from two eval sidecars")
    p.add_argument("--outdir", type=str, default=None, help="default: {model_dir}/probe")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# -- the policy under probe ---------------------------------------------------


def make_stepper(env: CnvEnv, frame_stack: int):
    if frame_stack > 1:
        from gymnasium.wrappers import FrameStackObservation
        return FrameStackObservation(env, stack_size=frame_stack, padding_type="zero")
    return env


def load_policy(args, family, frame_stack: int):
    """``(act, kind)``: ``act(env, obs) -> order`` for the artifact named by --model-path, loaded exactly as
    ``cnv_ppo_eval.py`` loads it; ``kind`` is 'net' or the myopic rule's name (the self-check)."""
    if args.model_path in MYOPIC_RULES:
        return None, args.model_path
    from cnv_ppo_eval import _vecnorm_for
    from cnv_ppo_eval import load_policy as eval_load_policy
    model_path = Path(args.model_path).resolve()

    def probe_env_fn():
        return make_stepper(CnvEnv(scenario=family, observation_mode=args.observation_mode,
                                   action_mode=args.action_mode), frame_stack)
    act, _ = eval_load_policy(model_path, _vecnorm_for(model_path, args.vecnorm_path), True, probe_env_fn)
    return act, "net"


class Reference:
    """The myopic level of one observation type at the env's current state, and the posterior mean it reads.
    The event type keeps a per-episode grid posterior; its prior grid is built once per row and copied."""

    def __init__(self, kind: str):
        self.kind, self._template, self._ev = kind, None, None

    def new_episode(self, env: CnvEnv) -> None:
        if self.kind != "event":
            return
        if self._template is None:
            self._template = MyopicEvent(env)
        self._ev = copy.copy(self._template)
        self._ev.log_w = self._template.log_w.copy()
        self._ev._seen_period = -1

    def __call__(self, env: CnvEnv) -> tuple[int, float]:
        sc, st = env._scenario_ep, env._state
        if self.kind == "timing":
            return myopic_timing(env), (sc.prior_shape + st.cum_sales) / (sc.prior_rate + st.cum_duration)
        if self.kind == "full":
            return myopic_full(env), (sc.prior_shape + st.cum_demand) / (sc.prior_rate + st.period)
        y = self._ev(env)                         # absorbs the last period's observation exactly once
        w = np.exp(self._ev.log_w); w /= w.sum()
        return y, float(w @ self._ev.lam)


# -- instrument 1: the belief surface -----------------------------------------


def belief_surface(act, env: CnvEnv, sc, mode: str, shape_max_mult: float,
                   dp_table=None) -> list[dict]:
    """Stocking level over (posterior shape, posterior rate, time_to_go) on a
    fixed scenario. The env's state is set directly — the surface is a
    function of the belief, and the belief modes render nothing else."""
    import cnv_mdp as mdp
    kind = obs_type(mode)
    rows = []
    N = sc.horizon
    shape_hi = shape_max_mult * (sc.prior_shape + N * sc.prior_mean)
    shapes = np.unique(np.round(np.geomspace(max(sc.prior_shape, 0.1), shape_hi, 24), 3))
    for ttg in range(N, 0, -1):
        n = N - ttg
        if kind == "full":
            rates = [sc.prior_rate + n]
        else:
            rates = np.unique(np.round(np.linspace(sc.prior_rate, sc.prior_rate + n, 6), 4))
        for a in shapes:
            for b in rates:
                if kind == "full":
                    st = mdp.CnvState(n, False, 0, float(n), int(round(a - sc.prior_shape)), 0, sc.seed_salt)
                else:
                    st = mdp.CnvState(n, False, int(round(a - sc.prior_shape)), float(b - sc.prior_rate), 0, 0, sc.seed_salt)
                env._state, env._scenario_ep = st, sc
                env._info = {"sales": 0, "stockout_event": 0, "stockout_time": 1.0, "demand": 0, "order": 0}
                y_net = int(act(env, env._get_obs()))
                a_eff = sc.prior_shape + (st.cum_sales if kind == "timing" else st.cum_demand)
                b_eff = sc.prior_rate + (st.cum_duration if kind == "timing" else st.period)
                y_m = gamma_poisson_quantile(a_eff, b_eff, sc.critical_ratio)
                # the DP table only holds states its own recursion reaches
                # (S <= n * y_max, T <= n); outside that the belief is unreachable
                reachable = (dp_table is not None and dp_table.policy == kind
                             and st.cum_sales <= n * int(dp_table.meta.get("y_max", 10**9))
                             and st.cum_duration <= n + 1e-9)
                y_dp = int(dp_table.act(env, None)) if reachable else None
                rows.append({"time_to_go": ttg, "post_shape": a_eff, "post_rate": b_eff,
                             "y_net": y_net, "y_myopic": y_m, "y_dp": y_dp})
    return rows


# -- instrument 2: on-policy agreement ----------------------------------------


def on_policy(act, policy_kind: str, source, seeds: np.ndarray, mode: str, action_mode: str,
              frame_stack: int, dp_table=None) -> list[dict]:
    """One row per visited state: the net's order, its type's myopic level, the DP level, the posterior mean."""
    kind = obs_type(mode)
    env = CnvEnv(scenario=source, observation_mode=mode, action_mode=action_mode)
    stepper = make_stepper(env, frame_stack)
    ref = Reference(kind)
    out = []
    for seed in seeds:
        obs, _ = stepper.reset(seed=int(seed))
        ref.new_episode(env)
        # the event self-check plays an INDEPENDENT grid posterior, built per episode as the benchmark builds it,
        # so 100% agreement validates the reference's copied prior grid, not the reference against itself
        indep = MyopicEvent(env) if policy_kind == "myopic_event" else None
        done = False
        while not done:
            y_m, post_mean = ref(env)
            if policy_kind == "net":
                y_net = int(act(env, obs))
            elif policy_kind == "myopic_timing":
                y_net = myopic_timing(env)
            elif policy_kind == "myopic_full":
                y_net = myopic_full(env)
            else:
                y_net = indep(env)
            y_dp = int(dp_table.act(env, None)) if dp_table is not None else None
            out.append({"seed": int(seed), "period": env._state.period, "y_net": y_net, "y_myopic": y_m,
                        "y_dp": y_dp, "post_mean": post_mean})
            obs, _, done, _, _ = stepper.step(y_net)
    env.close()
    return out


def summarize(rows: list[dict]) -> dict:
    y = np.array([r["y_net"] for r in rows]); m = np.array([r["y_myopic"] for r in rows])
    per = np.array([r["period"] for r in rows])
    dp = [r["y_dp"] for r in rows]
    res = {"states": len(rows), "agree_myopic": float(np.mean(y == m)),
           "frac_at_or_above_myopic": float(np.mean(y >= m)), "mean_excess_over_myopic": float(np.mean(y - m)),
           "excess_by_period": {int(p): float(np.mean((y - m)[per == p])) for p in np.unique(per)},
           "agree_dp": None}
    if dp and dp[0] is not None:
        res["agree_dp"] = float(np.mean(y == np.array(dp)))
    return res


# -- instrument 3: paired contrast --------------------------------------------


def paired_delta(a_path: Path, b_path: Path) -> dict:
    def read(path):
        rows = [ln.split("\t") for ln in Path(path).read_text().splitlines() if ln and not ln.startswith("#")]
        hdr, body = rows[0], rows[1:]
        i_t, i_s, i_p = hdr.index("target"), hdr.index("seed"), hdr.index("profit")
        return {(r[i_t], int(r[i_s])): float(r[i_p]) for r in body}
    a, b = read(a_path), read(b_path)
    keys = sorted(set(a) & set(b))
    d = np.array([b[k] - a[k] for k in keys])
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"n": len(d), "delta_mean": float(d.mean()), "delta_se": se,
            "a_mean": float(np.mean([a[k] for k in keys])), "b_mean": float(np.mean([b[k] for k in keys]))}


def _dp_for(args, target: str, row_id: str, kind: str, fixed):
    if fixed is not None:
        return fixed
    if not args.dp_auto or kind == "event":
        return None
    from cnv_benchmark_dp import DpTable, table_path
    path = table_path(HERE, "results", target, row_id, kind)
    return DpTable.load(path) if path.exists() else None


def main() -> None:
    args = parse_args()
    if args.action_mode != "order_discrete":
        # the instruments below read the policy's output as an ORDER and step the env with it — right
        # only under the identity encoding; premium_discrete (#E84) must hand step the premium instead
        raise SystemExit(f"cnv_policy_probe supports order_discrete only, got {args.action_mode}")
    if args.pair:
        res = paired_delta(Path(args.pair[0]), Path(args.pair[1]))
        print(f"paired B - A over {res['n']} shared seeds: {res['delta_mean']:+.4f} ± {res['delta_se']:.4f} (SE)"
              f"   [A {res['a_mean']:.4f}  B {res['b_mean']:.4f}]")
        return
    assert args.model_path, "--model-path is required unless --pair is given"
    mode, kind = args.observation_mode, obs_type(args.observation_mode)
    if args.model_path in MYOPIC_RULES and args.model_path.split("_")[1] != kind:
        raise SystemExit(f"--model-path {args.model_path} is the self-check for the {args.model_path.split('_')[1]} type; "
                         f"-o {mode} is the {kind} type")

    target, rows = eval_targets(args)
    family = GRIDS[args.grid_name] if args.grid_name else SCENARIOS[args.scenario_name]
    frame_stack = args.frame_stack if args.frame_stack is not None else (
        int(family.horizon) if mode in RAW_MODES else 1)
    if args.outdir:
        outdir = Path(args.outdir)
    elif Path(args.model_path).exists():
        run_dir = Path(args.model_path).resolve().parent
        if run_dir.name == "checkpoints":          # a checkpoint's probe lands in ITS RUN's probe/
            run_dir = run_dir.parent
        outdir = run_dir / "probe"
    else:
        outdir = HERE / "results" / target / "probe" / args.model_path
    outdir.mkdir(parents=True, exist_ok=True)
    fixed_dp = None
    if args.dp_table:
        from cnv_benchmark_dp import DpTable
        fixed_dp = DpTable.load(Path(args.dp_table))

    act, policy_kind = load_policy(args, family, frame_stack)
    do_surface = is_belief_mode(mode) and not args.no_surface and policy_kind != "myopic_event"
    seeds = eval_seed_block(args.first_seed, args.n_seeds)
    surface_path, onpolicy_path = outdir / f"surface_{target}.tsv", outdir / f"onpolicy_{target}.tsv"
    fs = open(surface_path, "w") if do_surface else None
    if fs:
        fs.write("target\ttime_to_go\tpost_shape\tpost_rate\ty_net\ty_myopic\ty_dp\n")
    with open(onpolicy_path, "w") as fo:
        fo.write(f"# model: {args.model_path}\n# observation_mode={mode} type={kind} frame_stack={frame_stack} "
                 f"seed block: first_seed={args.first_seed} n_seeds={args.n_seeds}\n")
        fo.write("target\tseed\tperiod\ty_net\ty_myopic\ty_dp\tpost_mean\n")
        for row_id, source in rows:
            dp_table = _dp_for(args, target, row_id, kind, fixed_dp)
            if fs:
                env = CnvEnv(scenario=source, observation_mode=mode, action_mode=args.action_mode)
                surf_act = act if policy_kind == "net" else (lambda e, o: myopic_timing(e) if kind == "timing" else myopic_full(e))
                for r in belief_surface(surf_act, env, source(0), mode, args.shape_max_mult, dp_table):
                    fs.write(f"{row_id}\t{r['time_to_go']}\t{r['post_shape']:.4f}\t{r['post_rate']:.4f}\t"
                             f"{r['y_net']}\t{r['y_myopic']}\t{'' if r['y_dp'] is None else r['y_dp']}\n")
                env.close()
            states = on_policy(act, policy_kind, source, seeds, mode, args.action_mode, frame_stack, dp_table)
            for r in states:
                fo.write(f"{row_id}\t{r['seed']}\t{r['period']}\t{r['y_net']}\t{r['y_myopic']}\t"
                         f"{'' if r['y_dp'] is None else r['y_dp']}\t{r['post_mean']:.6f}\n")
            s = summarize(states)
            by_p = " ".join(f"t{p}:{v:+.2f}" for p, v in s["excess_by_period"].items())
            print(f"{row_id}: on-policy over {s['states']} states — agree y^M {s['agree_myopic']:.3f}, "
                  f"at/above y^M {s['frac_at_or_above_myopic']:.3f}, mean excess {s['mean_excess_over_myopic']:+.3f} "
                  f"[{by_p}]" + (f", agree y^DP {s['agree_dp']:.3f}" if s['agree_dp'] is not None else ""), flush=True)
    if fs:
        fs.close()
        print(f"[probe] surface -> {surface_path}")
    print(f"[probe] on-policy -> {onpolicy_path}")


if __name__ == "__main__":
    main()

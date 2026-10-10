"""Policy-interpretation probe for owmr (spec section 14.1) — the Stage-5
instrument, written at Stage 1 and validated on a planted rule.

Reads a trained artifact exactly as the evaluator does (spec 9.5: SB3 model
on CPU, the saved VecNormalize observation statistics applied by hand,
deterministic actions, the gym's own ``decode_action``) and sweeps it over
declared state grids at a fixed period, reporting per sweep:

* **the raw action surface** (state -> decoded order / shipments), dumped so
  figures and fits never re-run the sweep;
* **structural-form statistics for the balance-assumption policy class**
  (Dogru, de Kok & van Houtum 2010, section 3.3): under that class the
  warehouse orders up to an echelon level y0 and the allocation raises every
  retailer toward a newsvendor target z_i. So the probe reports
  - *order-up-to flatness*: the spread of the post-order echelon position
    IP_0 + order over the region where the policy orders at all (0 = an exact
    order-up-to rule), and the recovered level;
  - *allocation-target flatness*: the spread of a retailer's post-allocation
    position over the region where it is shipped to (0 = an exact target
    rule), and the recovered target;
  - *the retention curve*: the fraction of warehouse on-hand the policy keeps
    back as on-hand grows — the quantity the discover claim is about, since
    the LB heuristic retains nothing that a retailer target can absorb;
* **feature-sensitivity sweeps**: hold the state fixed, sweep one observation
  feature (time to go), report how far the decision moves.

The policy is abstracted as ``policy(obs) -> raw action`` so the instrument
can be validated where the answer is known (spec 14.1: an instrument that has
never recovered a known answer is not measuring yet):
``owmr_test.py::test_probe_recovers_a_planted_order_up_to_rule`` plants an
exact order-up-to + target-allocation rule and requires zero flatness and the
planted levels back. Outputs land in ``<model_dir>/probe/`` (spec 8.4).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

import owmr_mdp as mdp
from owmr_gym import OwmrEnv, observation
from owmr_scenarios import SCENARIOS, OwmrScenario

Policy = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Synthetic states
# ---------------------------------------------------------------------------


def make_state(
    scenario: OwmrScenario,
    period: int,
    wh_stock: float,
    rt_stock: list[float] | None = None,
    episode_seed: int = 0,
) -> mdp.OwmrState:
    """A post-receipt state (the observation point) with pipelines at mean
    cover, the warehouse pipeline's last slot empty as it is after RW, and the
    given on-hand levels."""
    n = scenario.n_retailers
    mu = list(scenario.mu_rt)
    state, _ = mdp.init_state(scenario, episode_seed)
    state.period = period
    state.wh_stock = float(wh_stock)
    state.wh_pipe = [float(scenario.mu_sys)] * (scenario.l0 - 1) + [0.0]
    state.rt_stock = [float(x) for x in (rt_stock if rt_stock is not None else mu)]
    state.rt_pipe = [[float(mu[i])] * (scenario.l_rt - 1) + [0.0] for i in range(n)]
    return state


class _Decoder:
    """The gym's own transform, applied to a synthetic state: never a second
    copy of the arithmetic (spec 12)."""

    def __init__(self, scenario: OwmrScenario, observation_mode: str, action_mode: str):
        self.env = OwmrEnv(scenario, observation_mode=observation_mode, action_mode=action_mode)
        self.env.reset(seed=0)
        self.scenario = scenario
        self.observation_mode = observation_mode

    def decide(self, policy: Policy, state: mdp.OwmrState) -> tuple[float, list[float]]:
        self.env._state = state
        obs = observation(self.scenario, state, self.observation_mode, [0.0] * self.scenario.n_retailers)
        order, ship = self.env.decode_action(policy(obs))
        # what the S event would actually dispatch from this on-hand
        requested = sum(ship)
        scale = min(1.0, state.wh_stock / requested) if requested > 0 else 1.0
        return float(order), [q * scale for q in ship]


# ---------------------------------------------------------------------------
# Sweeps and their statistics
# ---------------------------------------------------------------------------


@dataclass
class Sweep:
    name: str
    axis: str
    grid: list[float]
    order: list[float] = field(default_factory=list)
    shipped: list[list[float]] = field(default_factory=list)
    derived: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def _spread(values: list[float]) -> float:
    return float(max(values) - min(values)) if values else float("nan")


def order_sweep(dec: _Decoder, policy: Policy, period: int, grid: list[float]) -> Sweep:
    """Warehouse on-hand x -> order q(x); post-order echelon position y0(x)."""
    sc = dec.scenario
    sw = Sweep("order_vs_wh_stock", "wh_stock", list(grid))
    y0 = []
    for x in grid:
        st = make_state(sc, period, x)
        q, ship = dec.decide(policy, st)
        sw.order.append(q)
        sw.shipped.append(ship)
        y0.append(mdp.echelon_position(st) + q)
    acting = [y for y, q in zip(y0, sw.order) if q > 1e-9]
    sw.derived["post_order_position"] = y0
    sw.stats = {
        "order_up_to_flatness": _spread(acting),
        "recovered_y0": float(np.mean(acting)) if acting else float("nan"),
        "acting_fraction": len(acting) / len(grid),
    }
    return sw


def allocation_sweep(
    dec: _Decoder, policy: Policy, period: int, grid: list[float], retailer: int = 0,
    wh_stock: float | None = None,
) -> Sweep:
    """Retailer `retailer`'s net stock z -> shipment to it; post-allocation
    position z + shipped, with the others at mean cover and the warehouse
    holding `wh_stock` (default: four periods of system demand, so the
    allocation is unconstrained by on-hand)."""
    sc = dec.scenario
    W = 4.0 * sc.mu_sys if wh_stock is None else wh_stock
    sw = Sweep(f"allocation_vs_rt{retailer}_stock", f"rt_stock[{retailer}]", list(grid))
    post = []
    for z in grid:
        rt = list(sc.mu_rt)
        rt[retailer] = z
        st = make_state(sc, period, W, rt_stock=rt)
        q, ship = dec.decide(policy, st)
        sw.order.append(q)
        sw.shipped.append(ship)
        post.append(z + sum(st.rt_pipe[retailer]) + ship[retailer])
    acting = [p for p, s in zip(post, sw.shipped) if s[retailer] > 1e-9]
    sw.derived["post_allocation_position"] = post
    sw.stats = {
        "target_flatness": _spread(acting),
        "recovered_target": float(np.mean(acting)) if acting else float("nan"),
        "acting_fraction": len(acting) / len(grid),
    }
    return sw


def retention_sweep(dec: _Decoder, policy: Policy, period: int, grid: list[float]) -> Sweep:
    """Warehouse on-hand W -> retained fraction 1 - sum(shipped)/W, retailers
    at mean cover. The discover claim's quantity."""
    sc = dec.scenario
    sw = Sweep("retention_vs_wh_stock", "wh_stock", list(grid))
    kept = []
    for W in grid:
        st = make_state(sc, period, W)
        q, ship = dec.decide(policy, st)
        sw.order.append(q)
        sw.shipped.append(ship)
        kept.append(1.0 - sum(ship) / W if W > 0 else float("nan"))
    sw.derived["retained_fraction"] = kept
    finite = [k for k in kept if np.isfinite(k)]
    sw.stats = {
        "mean_retained_fraction": float(np.mean(finite)) if finite else float("nan"),
        "max_retained_fraction": float(np.max(finite)) if finite else float("nan"),
    }
    return sw


def time_sensitivity(dec: _Decoder, policy: Policy, wh_stock: float | None = None) -> Sweep:
    """Hold the state fixed, sweep the period (so time_to_go), report how far
    the order moves — does the net condition on the clock?"""
    sc = dec.scenario
    W = sc.mu_sys if wh_stock is None else wh_stock
    periods = list(range(0, sc.horizon, max(1, sc.horizon // 20)))
    sw = Sweep("order_vs_time_to_go", "time_to_go", [float(sc.horizon - t) for t in periods])
    for t in periods:
        q, ship = dec.decide(policy, make_state(sc, t, W))
        sw.order.append(q)
        sw.shipped.append(ship)
    sw.stats = {"order_range_over_horizon": _spread(sw.order)}
    return sw


def run_probe(
    scenario: OwmrScenario,
    policy: Policy,
    observation_mode: str,
    action_mode: str,
    period: int | None = None,
    n_grid: int = 41,
) -> dict:
    """All sweeps at one fixed period (default: mid-horizon, where a finite-
    horizon policy is closest to stationary)."""
    period = scenario.horizon // 2 if period is None else period
    dec = _Decoder(scenario, observation_mode, action_mode)
    wh_grid = list(np.linspace(0.0, 6.0 * scenario.mu_sys, n_grid))
    rt_grid = list(np.linspace(-4.0 * scenario.mu_rt[0], 6.0 * scenario.mu_rt[0], n_grid))
    sweeps = [
        order_sweep(dec, policy, period, wh_grid),
        allocation_sweep(dec, policy, period, rt_grid),
        retention_sweep(dec, policy, period, wh_grid[1:]),
        time_sensitivity(dec, policy),
    ]
    return {
        "scenario": scenario.scenario_name,
        "observation_mode": observation_mode,
        "action_mode": action_mode,
        "period": period,
        "sweeps": [asdict(s) for s in sweeps],
        "summary": {s.name: s.stats for s in sweeps},
    }


# ---------------------------------------------------------------------------
# Loading a trained artifact the way the evaluator does (spec 9.5)
# ---------------------------------------------------------------------------


def load_sb3_policy(model_path: Path, vecnorm_path: Path | None) -> Policy:
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    if vecnorm_path is None:
        vecnorm_path = model_path.parent / "vecnormalize.pkl"
    obs_rms = None
    clip_obs = 10.0
    epsilon = 1e-8
    if vecnorm_path.exists():
        import pickle

        with open(vecnorm_path, "rb") as fh:
            vn = pickle.load(fh)
        # An artifact trained with norm_obs off (h4/h5/h8, #E16 on) carries no obs_rms at
        # all, and touching a missing attribute on an unpickled VecEnvWrapper recurses in
        # SB3's __getattr__ (its venv is None); read the flag from __dict__ (#E33 readback).
        if vn.__dict__.get("norm_obs", True):
            obs_rms, clip_obs, epsilon = vn.obs_rms, float(vn.clip_obs), float(vn.epsilon)

    def policy(obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if obs_rms is not None:
            x = np.clip((x - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -clip_obs, clip_obs)
        action, _ = model.predict(x.astype(np.float32), deterministic=True)
        return np.asarray(action).reshape(-1)

    return policy


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="owmr policy-interpretation probe (spec 14.1)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None)
    p.add_argument("-s", "--scenario_name", type=str, default="base", choices=list(SCENARIOS))
    p.add_argument("-o", "--observation_mode", type=str, default="raw")
    p.add_argument("-a", "--action_mode", type=str, default="order_frac")
    p.add_argument("--period", type=int, default=None)
    p.add_argument("--n-grid", type=int, default=41)
    p.add_argument("--outdir", type=str, default=None,
                   help="default: <model_dir>/probe/ (spec 8.4)")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    policy = load_sb3_policy(model_path, Path(args.vecnorm_path) if args.vecnorm_path else None)
    scenario = SCENARIOS[args.scenario_name]
    print(scenario)
    result = run_probe(scenario, policy, args.observation_mode, args.action_mode,
                       period=args.period, n_grid=args.n_grid)
    outdir = Path(args.outdir) if args.outdir else model_path.parent / "probe"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"probe_{args.scenario_name}_{args.observation_mode}_{args.action_mode}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

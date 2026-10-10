"""Gradient signal-to-noise probe for owmr (ESCALATION #E12) — a DIAGNOSTIC, never an arm.

#E11 showed the #E4 network can represent an allocation ~36 better than the one
PPO converges to, so the plateau is PPO's search. This probe asks whether the
search has a usable signal: at states the #E4 policy visits, how clean is a
single sample of PPO's policy-gradient estimate for the ALLOCATION logits,
compared with the ORDER, and what does the allocation leave on the table?

At each sampled post-receipt state s (the policy's own stochastic trajectories):

- the LOCAL gradient of the H-period return w.r.t. the action mean, per
  decision (order: component 0; allocation: components 1..N), perturbing only
  that decision by the policy's own Gaussian noise and continuing with the
  stochastic policy:
    "true" gradient  — antithetic pairs a ± eps under common random numbers
                       (same continuation seed for both), averaged over K;
    single-sample    — PPO's score-function form (G_k - mean G) * eps_k / sigma^2
                       on independent continuations (no CRN, as PPO sees it).
  SNR = |true gradient| / sd(single-sample estimate) (vector norms over the
  decision's components); samples_for_sign = 1 / SNR^2.
- the PRIZE: E[G(heuristic allocation) - G(policy's mean allocation)] at s, same
  order, CRN, continued with the policy — how much better an allocation
  exists at that state for this policy's continuation.
- ALIGNMENT: the cosine between the allocation's descent direction (minus the
  "true" cost gradient) and the direction from the policy's mean logits to the
  heuristic's logits. Positive = PPO's local signal points toward the better
  allocation (the search is slow); ~0 or negative = it points elsewhere (a
  local optimum the small steps cannot leave).

Continuations run H periods (default 20 = the GAE credit horizon at lambda 0.95)
and are truncated there. Demand is keyed on (episode_seed, period), so a
branch redraws the future by setting the state's episode_seed; two branches
with the same seed share it (CRN). Seeds: trajectories from 6e6, continuation
seeds from 7e6 — disjoint from every other block.

Usage:
    python owmr_snr_probe.py --model-path <ckpt.zip> --vecnorm-path <pkl> -s base
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import owmr_mdp as mdp
from owmr_benchmark_lb_heuristic import build as build_heuristic
from owmr_gym import LOGIT_BOX, OwmrEnv, anchored_softmax_shares, observation

TRAJ_SEED0 = 6_000_000
CONT_SEED0 = 7_000_000


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradient SNR probe (ESCALATION #E12).")
    p.add_argument("--model-path", required=True)
    p.add_argument("--vecnorm-path", required=True)
    p.add_argument("-s", "--scenario_name", default="base")
    p.add_argument("--n-states", type=int, default=100)
    p.add_argument("--k", type=int, default=64, help="samples per state and estimator")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--out", default=None)
    return p.parse_args()


class Policy:
    """The trained policy's Gaussian over the order_softmax action, on raw observations."""

    def __init__(self, model_path: str, vecnorm_path: str, sc) -> None:
        self.model = PPO.load(model_path, device="cpu")
        venv = DummyVecEnv([lambda: OwmrEnv(sc, "raw", "order_softmax")])
        self.vn = VecNormalize.load(vecnorm_path, venv)
        self.vn.training = False
        self.sc = sc

    @th.no_grad()
    def dist(self, states) -> tuple[np.ndarray, np.ndarray]:
        obs = np.stack([observation(self.sc, s, "raw", list(s.demand)) for s in states])
        obs = self.vn.normalize_obs(obs.astype(np.float32))
        d = self.model.policy.get_distribution(th.as_tensor(obs)).distribution
        return d.loc.numpy().astype(np.float64), d.scale.numpy().astype(np.float64)


def step(sc, state, action) -> tuple[object, float]:
    """One period from a post-receipt state under an order_softmax action."""
    a = np.asarray(action, dtype=np.float64)
    order = float(np.clip(a[0], 0.0, sc.order_max))
    W = max(0.0, float(state.wh_stock))
    ship = list(anchored_softmax_shares(np.clip(a[1:], -LOGIT_BOX, LOGIT_BOX)) * W)
    nxt, info = mdp.advance2(sc, state, order, ship)
    cost = float(info["cost"]["total"])
    if not nxt.terminated:
        nxt = mdp.advance1(sc, nxt)
    return nxt, cost


def continue_batch(sc, pol: Policy, states, first_actions, seeds, horizon: int) -> np.ndarray:
    """Return the H-period cost of each (state, first action, continuation seed),
    the first period's action given, the rest drawn from the stochastic policy
    with an action-noise stream seeded by the continuation seed."""
    st = []
    for s, sd in zip(states, seeds):
        c = copy.deepcopy(s)
        c.episode_seed = int(sd)             # redraw the future demand (CRN within a seed)
        st.append(c)
    rngs = [np.random.default_rng(int(sd)) for sd in seeds]
    total = np.zeros(len(st))
    acts = first_actions
    for h in range(horizon):
        alive = [i for i, c in enumerate(st) if not c.terminated]
        if not alive:
            break
        if h > 0:
            mu, sg = pol.dist([st[i] for i in alive])
            acts = np.zeros((len(st), mu.shape[1]))
            for j, i in enumerate(alive):
                acts[i] = mu[j] + sg[j] * rngs[i].standard_normal(mu.shape[1])
        for i in alive:
            st[i], cst = step(sc, st[i], acts[i])
            total[i] += cst
    return total


def myopic_logits(bound, state) -> np.ndarray:
    W = max(0.0, float(state.wh_stock))
    if W <= 0:
        return np.full(len(state.rt_stock), -LOGIT_BOX)
    s = bound.allocate_myopic(W, np.asarray(mdp.inventory_position(state), dtype=float))
    kept = max(W - s.sum(), W * np.exp(-7.0))
    return np.clip(np.log(np.maximum(s, W * np.exp(-17.0))) - np.log(kept), -LOGIT_BOX, LOGIT_BOX)


def sample_states(sc, pol: Policy, n_states: int, horizon: int) -> list:
    """Post-receipt states from the policy's own stochastic trajectories, at
    periods leaving room for an H-period continuation."""
    rng = np.random.default_rng(0)
    out = []
    ep = 0
    while len(out) < n_states:
        state, _ = mdp.init_state(sc, TRAJ_SEED0 + ep)
        state = mdp.advance1(sc, state)
        pick = set(rng.choice(np.arange(10, sc.horizon - horizon), size=5, replace=False).tolist())
        arng = np.random.default_rng(TRAJ_SEED0 + ep)
        while not state.terminated:
            if state.period in pick:
                out.append(copy.deepcopy(state))
            mu, sg = pol.dist([state])
            state, _ = step(sc, state, mu[0] + sg[0] * arng.standard_normal(mu.shape[1]))
        ep += 1
    return out[:n_states]


def main() -> None:
    args = _parse()
    th.set_num_threads(1)
    heur = build_heuristic(args.scenario_name)
    sc, bound = heur.scenario, heur.bound
    pol = Policy(args.model_path, args.vecnorm_path, sc)
    t0 = time.time()
    states = sample_states(sc, pol, args.n_states, args.horizon)
    print(f"{len(states)} states sampled ({time.time() - t0:.0f}s)", flush=True)

    K, H = args.k, args.horizon
    groups = {"order": [0], "allocation": list(range(1, sc.n_retailers + 1))}
    rows = []
    for si, s in enumerate(states):
        mu, sg = pol.dist([s])
        mu, sg = mu[0], sg[0]
        row = {"period": int(s.period), "W": float(s.wh_stock)}
        cont = CONT_SEED0 + si * 10_000
        for name, dims in groups.items():
            rng = np.random.default_rng(cont)
            eps = np.zeros((K, len(mu)))
            eps[:, dims] = rng.standard_normal((K, len(dims))) * sg[dims]
            seeds = cont + np.arange(K)
            # antithetic pairs under CRN -> low-variance "true" gradient
            gp = continue_batch(sc, pol, [s] * K, mu + eps, seeds, H)
            gm = continue_batch(sc, pol, [s] * K, mu - eps, seeds, H)
            true_g = ((gp - gm)[:, None] * eps[:, dims] / (2 * sg[dims] ** 2)).mean(0)
            # PPO's single-sample estimator: independent continuations, no CRN
            seeds_ind = cont + 5_000 + np.arange(K)
            gi = continue_batch(sc, pol, [s] * K, mu + eps, seeds_ind, H)
            single = (gi - gi.mean())[:, None] * eps[:, dims] / sg[dims] ** 2
            noise = float(np.sqrt(single.var(0, ddof=1).sum()))
            sig = float(np.linalg.norm(true_g))
            if name == "allocation":
                to_h = myopic_logits(bound, s) - mu[dims]
                row["alignment"] = float(-true_g @ to_h / (sig * np.linalg.norm(to_h) + 1e-12))
            row[name] = {"grad_norm": sig, "single_sd": noise,
                         "snr": sig / noise if noise > 0 else float("nan"),
                         "return_sd": float(gi.std(ddof=1))}
        # the prize: heuristic allocation vs the policy's mean allocation, same order, CRN
        seeds = cont + 9_000 + np.arange(K)
        a_pol = mu.copy()
        a_h = mu.copy(); a_h[1:] = myopic_logits(bound, s)
        d = continue_batch(sc, pol, [s] * K, np.tile(a_h, (K, 1)), seeds, H) \
            - continue_batch(sc, pol, [s] * K, np.tile(a_pol, (K, 1)), seeds, H)
        row["prize"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(K))}
        rows.append(row)
        if (si + 1) % 10 == 0:
            print(f"  {si + 1}/{len(states)} states ({time.time() - t0:.0f}s)", flush=True)

    def summ(name):
        snr = np.array([r[name]["snr"] for r in rows])
        return {"snr_median": float(np.median(snr)), "snr_p10": float(np.percentile(snr, 10)),
                "snr_p90": float(np.percentile(snr, 90)),
                "samples_for_sign_median": float(np.median(1.0 / snr ** 2)),
                "grad_norm_median": float(np.median([r[name]["grad_norm"] for r in rows])),
                "single_sd_median": float(np.median([r[name]["single_sd"] for r in rows])),
                "return_sd_median": float(np.median([r[name]["return_sd"] for r in rows]))}
    prize = np.array([r["prize"]["mean"] for r in rows])
    report = {"model": args.model_path, "n_states": len(rows), "k": K, "horizon": H,
              "order": summ("order"), "allocation": summ("allocation"),
              "alignment": {"median": float(np.median([r["alignment"] for r in rows])),
                            "share_positive": float(np.mean([r["alignment"] > 0 for r in rows]))},
              "prize": {"mean": float(prize.mean()), "se": float(prize.std(ddof=1) / np.sqrt(len(prize))),
                        "share_heuristic_better": float((prize < 0).mean()),
                        "median": float(np.median(prize))},
              "states": rows}
    out = Path(args.out) if args.out else Path(args.model_path).resolve().parent.parent / "probe" / "snr_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for name in ("order", "allocation"):
        r = report[name]
        print(f"{name:10s} SNR median {r['snr_median']:.3f} (p10 {r['snr_p10']:.3f}, p90 {r['snr_p90']:.3f})  "
              f"samples for sign ~{r['samples_for_sign_median']:.0f}  |grad| {r['grad_norm_median']:.3f}  "
              f"single-sample sd {r['single_sd_median']:.2f}  return sd {r['return_sd_median']:.2f}")
    p = report["prize"]
    print(f"prize (heuristic alloc − policy alloc, {H}-period cost, CRN): mean {p['mean']:+.3f} ± {p['se']:.3f}  "
          f"median {p['median']:+.3f}  heuristic better in {p['share_heuristic_better']:.0%} of states")
    al = report["alignment"]
    print(f"alignment (allocation descent vs direction to the heuristic's logits): median cosine "
          f"{al['median']:+.3f}, positive in {al['share_positive']:.0%} of states")
    print(f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

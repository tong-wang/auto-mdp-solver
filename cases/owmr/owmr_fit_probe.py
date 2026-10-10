"""Representation fit probe for owmr (ESCALATION #E11) — a DIAGNOSTIC, never an arm.

Question: can the allocation OUTPUT a policy emits represent a good allocation
at all, separately from whether RL finds it? Every bypass arm since #E4 emits
anchored-softmax LOGITS of warehouse on-hand and sits at ~640 against the
heuristic's 603.8. The heuristic's allocation is near-linear in each
retailer's raw stock with a cut at zero, max(0, w_i(lambda) - IP_i); in logit
coordinates the same allocation is log(max(0, .)) - log(kept), which sends
every retailer that should get nothing to -inf (pinned at the -10 box).

The probe trains the SAME flat network (tanh MLP on the standardized `raw`
observation; nothing shared across retailers) by supervised regression onto
the heuristic's allocation, once per output encoding, with the SAME loss —
squared error on the shipped QUANTITIES after each encoding's own decode:

    logits    ship = W * anchored_softmax(clamp(out, -10, 10))     (order_softmax)
    quantity  ship = relu(out), scaled down proportionally if sum > W   (order_ship + the MDP's S event)

Distribution shift is removed by DAgger: after the first fit on heuristic
trajectories, the probe rolls out the clone (heuristic order + cloned
allocation), relabels every visited state with the heuristic's allocation and
refits on the aggregate — so a closed-loop gap measures the representation,
not states the heuristic never visits.

Each clone is scored in closed loop with the HEURISTIC'S ORDER (a fixture: the
order is not what is being tested) on a CRN block, paired per seed against the
heuristic itself. The fitted networks never become a policy and no trained arm
reads them.

Seed blocks: data from 5,000,000 up, evaluation from 4,000,000 — disjoint from
the protocol (0), selection (1e6), smoke (2e6) and trial (3e6) blocks.

Usage:
    python owmr_fit_probe.py -s base --width 64 --out results/base/fit_probe
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as th
from torch import nn

from owmr_benchmark_lb_heuristic import build as build_heuristic
import owmr_mdp as mdp
from owmr_gym import LOGIT_BOX, OwmrEnv

ENCODINGS = ("logits", "quantity")
DATA_SEED0 = 5_000_000
EVAL_SEED0 = 4_000_000


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Allocation-output fit probe (ESCALATION #E11).")
    p.add_argument("-s", "--scenario_name", default="base")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--episodes", type=int, default=256, help="episodes per data round")
    p.add_argument("--dagger-rounds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--eval-seeds", type=int, default=2048)
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# network and decodes (torch; W is the post-receipt on-hand, obs[:, 1])
# ---------------------------------------------------------------------------

def mlp(n_in: int, n_out: int, width: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def decode(encoding: str, out: th.Tensor, W: th.Tensor) -> th.Tensor:
    """Network output -> shipped quantities, exactly as the gym + MDP would ship them."""
    if encoding == "logits":
        x = th.clamp(out, -LOGIT_BOX, LOGIT_BOX)
        # anchored softmax with the warehouse logit fixed at 0
        m = th.clamp(x.max(dim=1, keepdim=True).values, min=0.0)
        e = th.exp(x - m)
        shares = e / (th.exp(-m) + e.sum(dim=1, keepdim=True))
        return shares * W.unsqueeze(1)
    q = th.relu(out)
    tot = q.sum(dim=1, keepdim=True)
    scale = th.where(tot > W.unsqueeze(1), W.unsqueeze(1) / tot.clamp(min=1e-12), th.ones_like(tot))
    return q * scale


class Clone:
    def __init__(self, encoding: str, n_obs: int, n: int, width: int, depth: int) -> None:
        self.encoding = encoding
        self.net = mlp(n_obs, n, width, depth)
        self.mu = np.zeros(n_obs, dtype=np.float32)
        self.sd = np.ones(n_obs, dtype=np.float32)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int) -> float:
        self.mu = X.mean(0).astype(np.float32)
        self.sd = (X.std(0) + 1e-6).astype(np.float32)
        Xn = th.as_tensor((X - self.mu) / self.sd, dtype=th.float32)
        W = th.as_tensor(X[:, 1], dtype=th.float32)
        Yt = th.as_tensor(Y, dtype=th.float32)
        opt = th.optim.Adam(self.net.parameters(), lr=1e-3)
        n = len(X)
        g = th.Generator().manual_seed(0)
        for ep in range(epochs):
            if ep == int(epochs * 0.75):
                for pg in opt.param_groups:
                    pg["lr"] = 2e-4
            perm = th.randperm(n, generator=g)
            for i in range(0, n, 1024):
                b = perm[i:i + 1024]
                loss = ((decode(self.encoding, self.net(Xn[b]), W[b]) - Yt[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self.rmse(X, Y)

    @th.no_grad()
    def ship(self, X: np.ndarray) -> np.ndarray:
        Xn = th.as_tensor((X - self.mu) / self.sd, dtype=th.float32)
        W = th.as_tensor(X[:, 1], dtype=th.float32)
        return decode(self.encoding, self.net(Xn), W).numpy().astype(np.float64)

    def rmse(self, X: np.ndarray, Y: np.ndarray) -> float:
        return float(np.sqrt(((self.ship(X) - Y) ** 2).mean()))


# ---------------------------------------------------------------------------
# rollouts: the env runs in `order_ship`, so every arm hands the MDP quantities
# ---------------------------------------------------------------------------

def rollout(sc, pol, seeds, clone: Clone | None, collect: bool):
    """Heuristic order every period; allocation = the clone's (or the heuristic's
    when clone is None). Returns per-seed cost components and, if `collect`,
    the visited (obs, heuristic label) pairs."""
    n = sc.n_retailers
    envs = [OwmrEnv(sc, "raw", "order_ship") for _ in seeds]
    obs = np.stack([e.reset(seed=int(s))[0] for e, s in zip(envs, seeds)]).astype(np.float64)
    cost = np.zeros((len(seeds), 3))
    Xs, Ys = [], []
    phantom = np.zeros(2)          # [decisions where heuristic ships 0 to i, of those the arm ships > 0.05]
    for _ in range(sc.horizon):
        labels, orders = [], []
        for e in envs:
            order, ship = pol.decide(e._state)
            orders.append(order)
            labels.append(ship)
        labels = np.asarray(labels)
        ship = labels if clone is None else clone.ship(obs.astype(np.float32))
        if collect:
            Xs.append(obs.copy())
            Ys.append(labels)
        zero = labels <= 1e-9
        phantom += [zero.sum(), (zero & (ship > 0.05)).sum()]
        nxt = []
        for k, e in enumerate(envs):
            o, _, _, _, info = e.step(np.concatenate([[orders[k]], ship[k]]).astype(np.float32))
            c = info["cost"]
            cost[k] += [c["wh_holding"], c["rt_holding"], c["rt_shortage"]]
            nxt.append(o)
        obs = np.stack(nxt).astype(np.float64)
    data = (np.concatenate(Xs), np.concatenate(Ys)) if collect else None
    return cost, data, phantom


def main() -> None:
    args = _parse()
    th.manual_seed(args.torch_seed)
    th.set_num_threads(1)
    pol = build_heuristic(args.scenario_name)
    sc = pol.scenario
    n = sc.n_retailers
    out = Path(args.out or f"results/{args.scenario_name}/fit_probe")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"w{args.width}d{args.depth}"
    eval_seeds = np.arange(EVAL_SEED0, EVAL_SEED0 + args.eval_seeds)

    t0 = time.time()
    h_cost, _, h_ph = rollout(sc, pol, eval_seeds, None, collect=False)
    h_tot = h_cost.sum(1)
    print(f"heuristic on eval block: {h_tot.mean():.2f} ± {h_tot.std(ddof=1) / np.sqrt(len(h_tot)):.2f}  "
          f"({time.time() - t0:.0f}s)", flush=True)

    base_seeds = np.arange(DATA_SEED0, DATA_SEED0 + args.episodes)
    _, (X, Y), _ = rollout(sc, pol, base_seeds, None, collect=True)
    report = {"scenario": args.scenario_name, "net": tag, "heuristic": float(h_tot.mean()),
              "eval_seeds": [int(eval_seeds[0]), int(eval_seeds[-1])], "encodings": {}}
    for enc in ENCODINGS:
        Xa, Ya = X.copy(), Y.copy()
        clone = Clone(enc, X.shape[1], n, args.width, args.depth)
        rounds = []
        for r in range(args.dagger_rounds + 1):
            rm = clone.fit(Xa, Ya, args.epochs)
            c, _, ph = rollout(sc, pol, eval_seeds, clone, collect=False)
            d = c.sum(1) - h_tot
            row = {"round": r, "n_train": int(len(Xa)), "train_rmse": rm,
                   "cost": float(c.sum(1).mean()), "delta": float(d.mean()),
                   "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                   "wh": float(c[:, 0].mean()), "rt": float(c[:, 1].mean()), "short": float(c[:, 2].mean()),
                   "phantom_ship_rate": float(ph[1] / max(ph[0], 1))}
            rounds.append(row)
            print(f"[{enc:8s} {tag} round {r}] n={row['n_train']:7d} rmse {rm:.4f}  cost {row['cost']:.2f}  "
                  f"Δ vs heuristic {row['delta']:+.2f} ± {row['delta_se']:.2f}  (wh {row['wh']:.1f} rt {row['rt']:.1f} "
                  f"short {row['short']:.1f})  phantom {row['phantom_ship_rate']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            if r < args.dagger_rounds:     # DAgger: states the clone visits, heuristic labels
                seeds = np.arange(DATA_SEED0 + (r + 1) * 100_000, DATA_SEED0 + (r + 1) * 100_000 + args.episodes)
                _, (Xd, Yd), _ = rollout(sc, pol, seeds, clone, collect=True)
                Xa, Ya = np.concatenate([Xa, Xd]), np.concatenate([Ya, Yd])
        report["encodings"][enc] = rounds
    report["heuristic_phantom_ship_rate"] = float(h_ph[1] / max(h_ph[0], 1))
    path = out / f"fit_probe_{tag}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {path}")


if __name__ == "__main__":
    main()

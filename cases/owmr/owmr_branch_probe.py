"""Branch-and-grid value probe for owmr — rung 0 of the branch-and-grid lever (ESCALATION #E29).
A DIAGNOSTIC, never an arm.

The lever (cnv #E88): at one state, K members of a group take K different
actions on the SAME demand world and are credited by their return minus the
group mean — an exact paired comparison in place of PPO's single noisy sample.
#E12 measured why owmr might need it: one changed allocation decision is worth
~0.5 over the next 20 periods against a return sd near 20, so the single-sample
signal is ~1% of its noise. This probe measures, at the retention decision of
the best explicit arm (#E24, `order_catkeep` + `CatKeepPolicy`), what the paired
comparison buys BEFORE anything is built.

At each sampled post-receipt state s (the policy's own stochastic trajectories;
`--decision retention` uses SURPLUS states only — surplus = the LB heuristic's
myopic allocation keeps stock there; `order` and `split` use all periods):

- the GRID, six options of ONE decision with the other two held at the
  policy's deterministic action (mean order, argmax option, mean split):
    retention — the six menu options k in KEEP_MENU;
    order     — the policy's mean order + {-2, -1, 0, +1, +2} units, and the
                heuristic's order at s (option 5); clipped to [0, order_max];
    split     — shares of the shipped quantity: 0 the policy's mean split,
                1 the heuristic's myopic split, 2 equal shares, 3 / 5 the
                policy's split with 20% / 40% of the shipped moved from the
                retailer with the highest inventory position to the lowest
                (toward balance), 4 the 20% move the other way;
- PAIRED (the vine): option k on continuation seed j -> cost G_pair[k, j]; all
  six options share seed j (demand AND the continuation's action-noise stream);
- INDEPENDENT (PPO as it sees it): option k on its own seed -> G_ind[k, j];
- TRUTH at s: A[k] = mean_j G_pair[k, j] - mean_kj G_pair, k* = argmin A — the
  K = 64 CRN average, what the lever's exact comparison converges to;
- ONE-BRANCH RESOLUTION: P_j(argmin_k G_pair[:, j] == k*) — how often a single
  branch (six continuations, one per option, common world) picks the best
  option — against P_j(argmin_k G_ind[:, j] == k*), six independent
  continuations, the same sample budget without CRN;
- NOISE: sd over j of the paired advantage G_pair[k, j] - mean_k G_pair[:, j]
  against the independent one's; their ratio is the variance-reduction factor;
  samples to resolve the best-vs-second gap = (sd / gap)^2 for each form;
- the PRIZE at s: A[policy's argmax option] - A[k*], what the policy leaves on
  the table by its current deterministic choice, and A at the heuristic's
  kept fraction rounded to the menu.

Continuations run H periods (two horizons recorded in one pass: 10 and 20, the
GAE credit horizon) under the stochastic policy, truncated there (no bootstrap,
as #E12). Demand is keyed on (episode_seed, period), so a branch redraws the
future by setting the state's episode_seed; two branches with the same seed
share it (CRN). Seeds: trajectories from 6e6, continuation seeds from 7e6 —
disjoint from every eval block.

--validate N rolls out protocol seeds 0..N-1 deterministically with this
module's stepping and compares the episode cost with the arm's confirm sidecar
(bit-level agreement is the gate on the stepping).

Usage:
    python owmr_branch_probe.py --model-path <ckpt.zip> --vecnorm-path <pkl> -s base
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
from owmr_catkeep_policy import KEEP_MENU, CatKeepPolicy  # noqa: F401  (registers the class for PPO.load)
from owmr_gym import LOGIT_BOX, OwmrEnv, catkeep_shares, observation

TRAJ_SEED0 = 6_000_000
CONT_SEED0 = 7_000_000
HORIZONS = (10, 20)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Branch-and-grid value probe at the retention decision (rung 0).")
    p.add_argument("--model-path", required=True)
    p.add_argument("--vecnorm-path", required=True)
    p.add_argument("-s", "--scenario_name", default="base")
    p.add_argument("--decision", choices=("retention", "order", "split"), default="retention")
    p.add_argument("--n-states", type=int, default=100)
    p.add_argument("--k", type=int, default=64, help="continuation seeds per state and option")
    p.add_argument("--validate", type=int, default=0, help="compare N protocol seeds with the confirm sidecar")
    p.add_argument("--sidecar", default=None, help="confirm sidecar (.seeds.tsv) for --validate")
    p.add_argument("--out", default=None)
    return p.parse_args()


class Policy:
    """The trained CatKeep policy on raw observations: Gaussian(order, x) x Categorical(k)."""

    def __init__(self, model_path: str, vecnorm_path: str, sc) -> None:
        self.model = PPO.load(model_path, device="cpu")
        venv = DummyVecEnv([lambda: OwmrEnv(sc, "raw", "order_catkeep")])
        self.vn = VecNormalize.load(vecnorm_path, venv)
        self.vn.training = False
        self.sc = sc

    @th.no_grad()
    def dist(self, states) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(Gaussian loc [n, N+1], Gaussian scale [n, N+1], categorical probs [n, 6])."""
        obs = np.stack([observation(self.sc, s, "raw", list(s.demand)) for s in states])
        obs = self.vn.normalize_obs(obs.astype(np.float32))
        d = self.model.policy.get_distribution(th.as_tensor(obs))
        return (d.gauss.loc.numpy().astype(np.float64), d.gauss.scale.numpy().astype(np.float64),
                d.cat.probs.numpy().astype(np.float64))


def step(sc, state, cont, k: int) -> tuple[object, float]:
    """One period from a post-receipt state: cont = [order, x_1..x_N], k = menu index."""
    c = np.asarray(cont, dtype=np.float64)
    order = float(np.clip(c[0], 0.0, sc.order_max))
    W = max(0.0, float(state.wh_stock))
    shares, _ = catkeep_shares(np.concatenate([[float(k)], np.clip(c[1:], -LOGIT_BOX, LOGIT_BOX)]))
    nxt, info = mdp.advance2(sc, state, order, list(shares * W))
    cost = float(info["cost"]["total"])
    if not nxt.terminated:
        nxt = mdp.advance1(sc, nxt)
    return nxt, cost


def continue_batch(sc, pol: Policy, state, first_cont, first_k: int, seeds, horizons=HORIZONS) -> np.ndarray:
    """Cost over the first h periods for each h in `horizons`, per continuation
    seed: the first period's action given, the rest drawn from the stochastic
    policy with an action-noise stream seeded by the continuation seed.
    Returns [len(seeds), len(horizons)]."""
    st = []
    for sd in seeds:
        c = copy.deepcopy(state)
        c.episode_seed = int(sd)             # redraw the future demand (CRN within a seed)
        st.append(c)
    rngs = [np.random.default_rng(int(sd)) for sd in seeds]
    n = len(st)
    total = np.zeros(n)
    out = np.zeros((n, len(horizons)))
    for h in range(max(horizons)):
        alive = [i for i, c in enumerate(st) if not c.terminated]
        if not alive:
            break
        if h == 0:
            for i in alive:
                st[i], cst = step(sc, st[i], first_cont, first_k)
                total[i] += cst
        else:
            mu, sg, pr = pol.dist([st[i] for i in alive])
            for j, i in enumerate(alive):
                cont = mu[j] + sg[j] * rngs[i].standard_normal(mu.shape[1])
                k = int(rngs[i].choice(len(KEEP_MENU), p=pr[j] / pr[j].sum()))
                st[i], cst = step(sc, st[i], cont, k)
                total[i] += cst
        for hi, hz in enumerate(horizons):
            if h + 1 == hz:
                out[:, hi] = total
    return out


def heuristic_kept(heur, state) -> float:
    """The LB heuristic's kept fraction of on-hand at this state (0 when short)."""
    W = max(0.0, float(state.wh_stock))
    if W <= 1e-9:
        return 0.0
    _, ship = heur.decide(state)
    return max(0.0, W - float(np.sum(ship))) / W


def sample_states(sc, pol: Policy, heur, n_states: int, horizon: int, surplus_only: bool = True) -> tuple[list, dict]:
    """Post-receipt states from the policy's own stochastic trajectories (sampled
    order, split and option), at periods leaving room for H periods; SURPLUS
    states only when `surplus_only` (the retention decision's states)."""
    rng = np.random.default_rng(0)
    out, seen, surplus = [], 0, 0
    ep = 0
    while len(out) < n_states:
        state, _ = mdp.init_state(sc, TRAJ_SEED0 + ep)
        state = mdp.advance1(sc, state)
        pick = set(rng.choice(np.arange(10, sc.horizon - horizon), size=10, replace=False).tolist())
        arng = np.random.default_rng(TRAJ_SEED0 + ep)
        while not state.terminated:
            if state.period in pick:
                seen += 1
                hk = heuristic_kept(heur, state)
                if hk > 1e-6:
                    surplus += 1
                if (hk > 1e-6 or not surplus_only) and len(out) < n_states:
                    out.append((copy.deepcopy(state), hk))
            mu, sg, pr = pol.dist([state])
            cont = mu[0] + sg[0] * arng.standard_normal(mu.shape[1])
            k = int(arng.choice(len(KEEP_MENU), p=pr[0] / pr[0].sum()))
            state, _ = step(sc, state, cont, k)
        ep += 1
    return out, {"states_seen": seen, "surplus_seen": surplus, "surplus_share": surplus / max(1, seen),
                 "episodes": ep}


def validate(sc, pol: Policy, n: int, sidecar: Path) -> float:
    """Deterministic rollout of protocol seeds 0..n-1 with this module's stepping
    against the arm's confirm sidecar; returns the max |cost difference|."""
    ref = {}
    with open(sidecar) as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            ref[int(f[cols.index("seed")])] = float(f[cols.index("cost_total")])
    worst = 0.0
    for seed in range(n):
        state, _ = mdp.init_state(sc, seed)
        state = mdp.advance1(sc, state)
        total = 0.0
        while not state.terminated:
            mu, _, pr = pol.dist([state])
            state, cst = step(sc, state, mu[0], int(np.argmax(pr[0])))
            total += cst
        worst = max(worst, abs(total - ref[seed]))
    return worst


ORDER_OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _move(shares: np.ndarray, state, frac: float) -> np.ndarray:
    """Move `frac` of the shipped quantity from the retailer with the highest
    inventory position to the one with the lowest (negative frac: the other way)."""
    ip = np.asarray(mdp.inventory_position(state), dtype=float)
    hi, lo = int(np.argmax(ip)), int(np.argmin(ip))
    out = shares.copy()
    amt = abs(frac)
    src, dst = (hi, lo) if frac > 0 else (lo, hi)
    amt = min(amt, out[src])
    out[src] -= amt
    out[dst] += amt
    return out


def grid_for(decision: str, sc, heur, state, mu: np.ndarray, pr: np.ndarray) -> tuple[list, list, int, int]:
    """Six (cont, k) first actions for `decision` at `state`, with labels and the
    indices of the policy's own choice and the heuristic's. The other two
    decisions sit at the policy's deterministic action."""
    k_pol = int(np.argmax(pr))
    h_order, h_ship = heur.decide(state)
    W = max(0.0, float(state.wh_stock))
    if decision == "retention":
        return [(mu, k) for k in range(len(KEEP_MENU))], [f"keep{m:.1f}" for m in KEEP_MENU], k_pol, \
            int(np.argmin(np.abs(np.asarray(KEEP_MENU) - heuristic_kept(heur, state))))
    if decision == "order":
        opts, labels = [], []
        for off in ORDER_OFFSETS:
            c = mu.copy(); c[0] = mu[0] + off
            opts.append((c, k_pol)); labels.append(f"order{off:+.0f}")
        c = mu.copy(); c[0] = float(h_order)
        opts.append((c, k_pol)); labels.append("order_heur")
        return opts, labels, int(ORDER_OFFSETS.index(0.0)), len(opts) - 1
    # split: shares of the shipped quantity; weights == shares decode exactly (relu, renormalised)
    pol_shares, _ = catkeep_shares(np.concatenate([[float(k_pol)], np.clip(mu[1:], -LOGIT_BOX, LOGIT_BOX)]))
    kept = KEEP_MENU[k_pol]
    pol_split = pol_shares / (1.0 - kept) if kept < 1.0 and pol_shares.sum() > 0 else np.full(len(pol_shares), 1.0 / len(pol_shares))
    hs = np.asarray(h_ship, dtype=float)
    h_split = hs / hs.sum() if hs.sum() > 0 else np.full(len(hs), 1.0 / len(hs))
    splits = [pol_split, h_split, np.full(len(hs), 1.0 / len(hs)),
              _move(pol_split, state, 0.2), _move(pol_split, state, -0.2), _move(pol_split, state, 0.4)]
    labels = ["split_pol", "split_heur", "split_equal", "bal+20%", "bal-20%", "bal+40%"]
    opts = []
    for sp in splits:
        c = mu.copy(); c[1:] = sp
        opts.append((c, k_pol))
    return opts, labels, 0, 1


def main() -> None:
    args = _parse()
    th.set_num_threads(1)
    heur = build_heuristic(args.scenario_name)
    sc = heur.scenario
    pol = Policy(args.model_path, args.vecnorm_path, sc)
    t0 = time.time()
    val = None
    if args.validate:
        sidecar = Path(args.sidecar) if args.sidecar else None
        if sidecar is None:
            mp = Path(args.model_path).resolve()
            cands = sorted(mp.parent.parent.glob(f"confirm_{mp.stem}.seeds.tsv"))
            assert cands, "no confirm sidecar found beside the checkpoint; pass --sidecar"
            sidecar = cands[0]
        val = validate(sc, pol, args.validate, sidecar)
        print(f"validate: max |cost diff| vs {sidecar.name} over {args.validate} seeds = {val:.2e}", flush=True)

    H = max(HORIZONS)
    surplus_only = args.decision == "retention"
    states, census = sample_states(sc, pol, heur, args.n_states, H, surplus_only)
    print(f"{len(states)} {'surplus ' if surplus_only else ''}states sampled from {census['episodes']} episodes "
          f"(surplus share of visited periods {census['surplus_share']:.1%}; {time.time() - t0:.0f}s)", flush=True)

    K = args.k
    rows = []
    labels = None
    for si, (s, hk) in enumerate(states):
        mu, sg, pr = pol.dist([s])
        mu, pr = mu[0], pr[0]
        opts, labels, i_pol, i_heur = grid_for(args.decision, sc, heur, s, mu, pr)
        M = len(opts)
        cont = CONT_SEED0 + si * 10_000
        seeds = cont + np.arange(K)
        g_pair = np.stack([continue_batch(sc, pol, s, c, k, seeds) for c, k in opts])             # [M, K, nH]
        g_ind = np.stack([continue_batch(sc, pol, s, c, k, cont + 5_000 + i * K + np.arange(K))
                          for i, (c, k) in enumerate(opts)])                                         # [M, K, nH]
        row = {"period": int(s.period), "W": float(s.wh_stock), "heuristic_kept": float(hk),
               "heuristic_option": int(i_heur), "policy_order": float(mu[0]),
               "heuristic_order": float(heur.decide(s)[0]),
               "policy_probs": [float(x) for x in pr], "policy_argmax": int(i_pol), "by_horizon": {}}
        for hi, hz in enumerate(HORIZONS):
            gp, gi = g_pair[:, :, hi], g_ind[:, :, hi]
            A = gp.mean(1) - gp.mean()                       # truth at s (K-seed CRN average)
            k_star = int(np.argmin(A))
            order = np.argsort(A)
            gap = float(A[order[1]] - A[order[0]])
            d_pair = gp - gp.mean(0, keepdims=True)          # paired advantage, one branch
            d_ind = gi - gi.mean(0, keepdims=True)           # six independent continuations
            sd_pair = float(np.sqrt(d_pair.var(1, ddof=1).mean()))
            sd_ind = float(np.sqrt(d_ind.var(1, ddof=1).mean()))
            # the best-vs-second contrast's noise in each form
            c_pair = float((gp[order[0]] - gp[order[1]]).std(ddof=1))
            c_ind = float((gi[order[0]] - gi[order[1]]).std(ddof=1))
            row["by_horizon"][str(hz)] = {
                "A": [float(x) for x in A], "A_se": [float(x) for x in gp.std(1, ddof=1) / np.sqrt(K)],
                "k_star": k_star, "gap_best_second": gap,
                "resolve_pair": float((gp.argmin(0) == k_star).mean()),
                "resolve_ind": float((gi.argmin(0) == k_star).mean()),
                "sd_pair": sd_pair, "sd_ind": sd_ind, "return_sd": float(gi.std(ddof=1)),
                "vr_factor": sd_ind / sd_pair if sd_pair > 0 else float("nan"),
                "samples_pair": (c_pair / gap) ** 2 if gap > 0 else float("nan"),
                "samples_ind": (c_ind / gap) ** 2 if gap > 0 else float("nan"),
                "prize_policy": float(A[i_pol] - A[k_star]),
                "prize_heuristic": float(A[row["heuristic_option"]] - A[k_star]),
                "prize_keep0": float(A[0] - A[k_star]),
            }
        rows.append(row)
        if (si + 1) % 10 == 0:
            print(f"  {si + 1}/{len(states)} states ({time.time() - t0:.0f}s)", flush=True)

    def med(key, hz):
        v = np.array([r["by_horizon"][str(hz)][key] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)), "mean": float(v.mean())}

    summary = {}
    for hz in HORIZONS:
        A_all = np.array([r["by_horizon"][str(hz)]["A"] for r in rows])
        k_star = np.array([r["by_horizon"][str(hz)]["k_star"] for r in rows])
        argmax = np.array([r["policy_argmax"] for r in rows])
        summary[str(hz)] = {
            "A_mean_by_option": [float(x) for x in A_all.mean(0)],
            "k_star_hist": [int((k_star == k).sum()) for k in range(len(labels))],
            "share_keep_positive_best": float((k_star != argmax).mean()),
            "share_policy_argmax_is_best": float((argmax == k_star).mean()),
            "resolve_pair": med("resolve_pair", hz), "resolve_ind": med("resolve_ind", hz),
            "sd_pair": med("sd_pair", hz), "sd_ind": med("sd_ind", hz), "return_sd": med("return_sd", hz),
            "vr_factor": med("vr_factor", hz),
            "samples_pair": med("samples_pair", hz), "samples_ind": med("samples_ind", hz),
            "gap_best_second": med("gap_best_second", hz),
            "prize_policy": med("prize_policy", hz), "prize_heuristic": med("prize_heuristic", hz),
            "prize_keep0": med("prize_keep0", hz),
        }
    report = {"model": args.model_path, "vecnorm": args.vecnorm_path, "decision": args.decision,
              "n_states": len(rows), "k": K, "options": labels,
              "horizons": list(HORIZONS), "menu": list(KEEP_MENU), "validate_max_abs_diff": val,
              "census": census, "summary": summary, "states": rows}
    suffix = "" if args.decision == "retention" else f"_{args.decision}"
    out = Path(args.out) if args.out else Path(args.model_path).resolve().parent.parent / "probe" / f"branch_probe{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{args.decision}: {len(rows)} states  (surplus share of visited periods {census['surplus_share']:.1%})")
    for hz in HORIZONS:
        S = summary[str(hz)]
        print(f"\n== horizon {hz} ==")
        print("  mean advantage by option " + " ".join(f"{m}:{a:+.3f}" for m, a in zip(labels, S["A_mean_by_option"])))
        print(f"  k* histogram over the options {S['k_star_hist']}  non-policy option best in {S['share_keep_positive_best']:.0%}  "
              f"policy argmax = k* in {S['share_policy_argmax_is_best']:.0%}")
        print(f"  one-branch resolution: paired {S['resolve_pair']['median']:.2f} (mean {S['resolve_pair']['mean']:.2f})  "
              f"independent {S['resolve_ind']['median']:.2f} (mean {S['resolve_ind']['mean']:.2f})  [chance 1/6 = 0.17]")
        print(f"  advantage sd: paired {S['sd_pair']['median']:.2f}  independent {S['sd_ind']['median']:.2f}  "
              f"return sd {S['return_sd']['median']:.2f}  variance-reduction factor {S['vr_factor']['median']:.1f} "
              f"(p10 {S['vr_factor']['p10']:.1f}, p90 {S['vr_factor']['p90']:.1f})")
        print(f"  best-vs-second gap {S['gap_best_second']['median']:.3f}; samples to resolve it: paired "
              f"{S['samples_pair']['median']:.0f}  independent {S['samples_ind']['median']:.0f}")
        print(f"  prize (A[choice] - A[k*]): policy's option {S['prize_policy']['mean']:+.3f}  "
              f"heuristic's option {S['prize_heuristic']['mean']:+.3f}  option 0 {S['prize_keep0']['mean']:+.3f}")
    print(f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

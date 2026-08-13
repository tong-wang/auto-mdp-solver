"""Interpretation round of the crowned policy (INTERPRET_PLAN.md, crown =
A8-a, #E23). Rungs 0/1/3 live here; the spec-§14 index readback is
`mab_policy_probe.py` and its figures `mab_plot_policy.py`.

Parts (artifacts accumulate under results/{scenario}/interpret/):

  metrics  rung 1 — the exploration/exploitation metric battery over the
           8192-seed protocol block, one instrumented VecSim sweep per
           policy, CRN across policies (same latents, so per-seed paired
           deltas are valid). Policies: the three A8-a seeds, the old
           1387 crown (same architecture, ent_coef 0.01 — the "before
           ent->0" column), thompson, ucb1, greedy.
  replay   rung 0 — side-by-side GIF: crown vs thompson on the SAME seed
           (shared latents), chosen from the metrics part's paired deltas.
  tail     rung 3 — anatomy of the worst-100 paired-loss seeds vs thompson.

Metric definitions (the corner/line/snake analogs for bandit people):
  explore        a_t != argmax posterior-mean  (the campaign's standing def,
                 #E17); phases early [0,100) / mid [100,900) / late [900,1000)
  regret split   true instantaneous regret summed separately over explore
                 and exploit pulls
  paid gap       posterior gap pm[greedy] - pm[chosen], per explore pull
  final ID       argmax pm at t=T equals the true best arm
  lock time      first t after which argmax pm stays correct to the end
  commitment     pull share of the true best arm; distinct arms in the last
                 100 pulls
  alloc exponent per-episode LS slope of log(n_i+1) on log(1/Delta_i) over
                 the 9 suboptimal arms — UCB theory (n_i ~ 1/Delta_i^2)
                 predicts ~2; greedy has no such law
  agreement      on the crown's own trajectory states: action match vs
                 UCB1's index, TV distance vs thompson's action law (MC)

Methodology rules imported from game2048/INTERPRET.md: structure is
measured DURING life (phase curves, never terminal snapshots); replay seeds
come from the protocol eval's paired extremes, not curation.

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_interpret.py --part metrics --n-seeds 128
    OMP_NUM_THREADS=1 python mab_interpret.py --part metrics
    OMP_NUM_THREADS=1 python mab_interpret.py --part replay
    OMP_NUM_THREADS=1 python mab_interpret.py --part tail
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mab_a5_probe import K, T, PROTO_SEEDS, SCENARIO, VecSim, VecThompson
from mab_anneal_probe import CrownPolicy, _sample

RESULTS = Path("results") / SCENARIO
OUTDIR  = RESULTS / "interpret"

#: the policy roster; crown seeds are the A8-a committed runs (#E23),
#: oldcrown is the #E14 artifact (the 12M run #E17 profiled)
_HP = "lr1.58e-05_lrf1.58e-06_nsteps2048_bs32_ep20_gae0.988_ent3.44e-05_vf0.381"
RUNS = {
    "crown_s1": RESULTS / f"PPO_obsbayes_L1_policyindex_{_HP}_normobsFalse_seed1_20260801_230542",
    "crown_s2": RESULTS / f"PPO_obsbayes_L1_policyindex_{_HP}_normobsFalse_seed2_20260801_230542",
    "crown_s3": RESULTS / f"PPO_obsbayes_L1_policyindex_{_HP}_normobsFalse_seed3_20260801_230542",
    "oldcrown": RESULTS / "PPO_obsbayes_L1_policyindex_normobsFalse_seed1_20260729_185003",
}

EARLY, MID, LATE = slice(0, 100), slice(100, 900), slice(900, 1000)
SIGMA = 1.0            # gauss_K10_T1000 payout sd; asserted in mab_scenarios


# ---------------------------------------------------------------------------
# vectorized reference policies (must mirror the mab_benchmark_* scalar code)
# ---------------------------------------------------------------------------

class VecUcb1:
    """mab_benchmark_ucb1.Ucb1Policy, vectorized. Empirical mean (payouts /
    pulls, NOT the bayes posterior), each arm once first (lowest index), then
    argmax of mean + sigma * sqrt(2 ln(t+1) / n)."""

    def act(self, sim: VecSim) -> np.ndarray:
        unpulled = sim.pulls == 0
        has0 = unpulled.any(axis=1)
        n = np.maximum(sim.pulls, 1.0)
        idx = sim.payouts / n + SIGMA * np.sqrt(2.0 * np.log(sim.t + 1) / n)
        a = idx.argmax(axis=1)
        a[has0] = unpulled.argmax(axis=1)[has0]      # lowest zero-pull index
        return a

    @staticmethod
    def index(pulls: np.ndarray, payouts: np.ndarray, t: int) -> np.ndarray:
        """The UCB index on external belief states (agreement metric)."""
        n = np.maximum(pulls, 1.0)
        idx = payouts / n + SIGMA * np.sqrt(2.0 * np.log(t + 1) / n)
        unpulled = pulls == 0
        # unpulled arms are pulled first: give them +inf index
        return np.where(unpulled, np.inf, idx)


class VecGreedy:
    """mab_benchmark_greedy.GreedyPolicy, vectorized: argmax bayes posterior
    mean, ties -> lowest index (np.argmax semantics, same as scalar)."""

    def act(self, sim: VecSim) -> np.ndarray:
        return (sim.payouts / (1.0 + sim.pulls)).argmax(axis=1)


# ---------------------------------------------------------------------------
# rung 1 — the metric battery
# ---------------------------------------------------------------------------

def _entropy_rows(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)


def run_battery(name: str, n_seeds: int, agreement: bool) -> dict:
    """One instrumented sweep. `agreement` adds the vs-UCB / vs-thompson
    mechanism metrics (only meaningful for the learned policies)."""
    seeds = PROTO_SEEDS[:n_seeds]
    n = len(seeds)
    idx = np.arange(n)

    if name in RUNS:
        pol = CrownPolicy(RUNS[name])
        stochastic = True
    elif name == "thompson":
        pol, stochastic = VecThompson(rng_seed=7), False
    elif name == "ucb1":
        pol, stochastic = VecUcb1(), False
    elif name == "greedy":
        pol, stochastic = VecGreedy(), False
    else:
        raise ValueError(name)
    rng = np.random.default_rng(101)
    th_rng = np.random.default_rng(211)

    sim = VecSim(seeds)
    best = sim.best
    gaps = sim.opt[:, None] - sim.means                     # (n, K), >= 0

    explore_cnt = np.zeros((n, 3))
    regret_exp = np.zeros(n); regret_gre = np.zeros(n)
    paid_gap_sum = np.zeros(n); explore_tot = np.zeros(n)
    pull_best = np.zeros((n, 3))
    pulls_late = np.zeros((n, K))
    correct = np.zeros((n, T), dtype=bool)
    explore_curve = np.zeros(T); pull_best_curve = np.zeros(T)
    H_curve = np.full(T, np.nan)
    agree_ucb = np.zeros((n, 3)); tv_sum = np.zeros(3); tv_cnt = np.zeros(3)

    for t in range(T):
        phase = 0 if t < 100 else (1 if t < 900 else 2)
        obs = sim.obs()
        pm, psd = obs[:, :K], obs[:, K:2 * K]
        if name in RUNS:
            p = pol.probs(obs)
            a = _sample(p, rng)
            H_curve[t] = _entropy_rows(p).mean()
        elif name == "thompson":
            a = pol.act(sim)
        else:
            a = pol.act(sim)
        gre = pm.argmax(axis=1)
        exp_mask = a != gre

        explore_cnt[idx, phase] += exp_mask
        explore_curve[t] = exp_mask.mean()
        inst = sim.opt - sim.means[idx, a]
        regret_exp += np.where(exp_mask, inst, 0.0)
        regret_gre += np.where(~exp_mask, inst, 0.0)
        paid_gap_sum += np.where(exp_mask, pm[idx, gre] - pm[idx, a], 0.0)
        explore_tot += exp_mask
        hit = a == best
        pull_best[idx, phase] += hit
        pull_best_curve[t] = hit.mean()
        if t >= 900:
            pulls_late[idx, a] += 1.0

        if agreement:
            ucb_a = VecUcb1.index(sim.pulls, sim.payouts, sim.t).argmax(axis=1)
            agree_ucb[idx, phase] += a == ucb_a
            if t % 25 == 0:            # thompson law is MC — subsample
                th_p = VecThompson.probs_mc(pm, psd, 128, th_rng)
                tv_sum[phase] += 0.5 * np.abs(p - th_p).sum(axis=1).mean()
                tv_cnt[phase] += 1

        sim.step(a)
        pm_next = sim.payouts / (1.0 + sim.pulls)
        correct[:, t] = pm_next.argmax(axis=1) == best

    # ---- per-episode reductions ------------------------------------------
    rev = correct[:, ::-1]
    trail = rev.cumprod(axis=1).sum(axis=1)                 # trailing-True len
    lock_t = T - trail                                       # T if never locks
    final_id = correct[:, -1]

    sub = np.ones((n, K), dtype=bool); sub[idx, best] = False
    g = gaps[sub].reshape(n, K - 1)
    npull = sim.pulls[sub].reshape(n, K - 1)
    x = np.log(1.0 / np.maximum(g, 1e-9))                   # log(1/Delta)
    y = np.log1p(npull)
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    denom = (xc * xc).sum(axis=1)
    alloc_beta = np.where(denom > 1e-12, (xc * yc).sum(axis=1) / denom, np.nan)

    phase_len = np.array([100.0, 800.0, 100.0])
    out = {
        "policy": name, "n_seeds": n,
        "reward_mean": float(sim.total.mean()),
        "reward_se": float(sim.total.std(ddof=1) / np.sqrt(n)),
        "explore_early": float((explore_cnt[:, 0] / phase_len[0]).mean()),
        "explore_mid":   float((explore_cnt[:, 1] / phase_len[1]).mean()),
        "explore_late":  float((explore_cnt[:, 2] / phase_len[2]).mean()),
        "entropy_late": (float(np.nanmean(H_curve[LATE]))
                         if name in RUNS else None),
        "regret_explore_mean": float(regret_exp.mean()),
        "regret_exploit_mean": float(regret_gre.mean()),
        "paid_gap_per_explore": float(
            (paid_gap_sum.sum() / max(explore_tot.sum(), 1.0))),
        "final_id_rate": float(final_id.mean()),
        "lock_t_median": float(np.median(lock_t)),
        "locked_frac": float((lock_t < T).mean()),
        "share_best": float((sim.pulls[idx, best] / T).mean()),
        "distinct_late": float((pulls_late > 0).sum(axis=1).mean()),
        "alloc_beta_mean": float(np.nanmean(alloc_beta)),
        "alloc_beta_median": float(np.nanmedian(alloc_beta)),
        "pull_best_early": float((pull_best[:, 0] / phase_len[0]).mean()),
        "pull_best_late":  float((pull_best[:, 2] / phase_len[2]).mean()),
    }
    if agreement:
        for i, ph in enumerate(("early", "mid", "late")):
            out[f"agree_ucb_{ph}"] = float(
                (agree_ucb[:, i] / phase_len[i]).mean())
            out[f"tv_thompson_{ph}"] = float(tv_sum[i] / max(tv_cnt[i], 1))
    out["curves"] = {
        "explore_rate": explore_curve.round(4).tolist(),
        "pull_best_rate": pull_best_curve.round(4).tolist(),
        "id_rate": correct.mean(axis=0).round(4).tolist(),
        "entropy": (np.round(H_curve, 4).tolist() if name in RUNS else None),
    }
    per_seed = {
        "reward": sim.total, "regret_explore": regret_exp,
        "regret_exploit": regret_gre, "lock_t": lock_t.astype(float),
        "final_id": final_id.astype(float),
        "share_best": sim.pulls[idx, best] / T, "alloc_beta": alloc_beta,
        "explore_late": explore_cnt[:, 2] / phase_len[2],
    }
    np.savez(OUTDIR / f"perseed_{name}.npz", **per_seed)
    return out


def part_metrics(n_seeds: int) -> dict:
    policies = ["crown_s3", "crown_s1", "crown_s2", "oldcrown",
                "thompson", "ucb1", "greedy"]
    summary = {}
    for name in policies:
        t0 = time.time()
        res = run_battery(name, n_seeds,
                          agreement=(name in RUNS))
        res["wall_seconds"] = round(time.time() - t0, 1)
        (OUTDIR / f"metrics_{name}.json").write_text(json.dumps(res, indent=2))
        summary[name] = {k: v for k, v in res.items() if k != "curves"}
        print(json.dumps(summary[name], indent=2), flush=True)
    # CRN-paired deltas vs thompson, per seed
    th = np.load(OUTDIR / "perseed_thompson.npz")
    for name in policies:
        if name == "thompson":
            continue
        d = np.load(OUTDIR / f"perseed_{name}.npz")["reward"] - th["reward"]
        summary[name]["paired_vs_thompson_mean"] = float(d.mean())
        summary[name]["paired_vs_thompson_se"] = float(
            d.std(ddof=1) / np.sqrt(len(d)))
    (OUTDIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    return {"policies": len(policies), "n_seeds": n_seeds}


# ---------------------------------------------------------------------------
# rung 0 — side-by-side replay GIFs
# ---------------------------------------------------------------------------

def _replay_episode(seed: int, rng_seed: int):
    """Play ONE seed with crown (stochastic) and thompson on shared latents;
    returns per-t snapshots for animation + the final totals."""
    pol = CrownPolicy(RUNS["crown_s3"])
    rng = np.random.default_rng(rng_seed)
    th = VecThompson(rng_seed=rng_seed + 1)
    snaps = []
    sims = {"crown": VecSim(np.array([seed])),
            "thompson": VecSim(np.array([seed]))}
    # cumulative PSEUDO-regret sum(opt - mu[chosen]) — monotone by
    # construction and the same means-based definition the battery, the
    # tail anatomy, and the ledger use. (Realized-payout regret is a
    # zero-mean random walk once locked on the best arm; it obscures the
    # race the panel exists to show.)
    preg = {"crown": 0.0, "thompson": 0.0}
    for t in range(T):
        frame = {"t": t}
        for name, sim in sims.items():
            obs = sim.obs()
            if name == "crown":
                a = int(_sample(pol.probs(obs), rng)[0])
            else:
                a = int(th.act(sim)[0])
            preg[name] += float(sim.opt[0] - sim.means[0, a])
            frame[name] = {
                "pm": obs[0, :K].copy(), "psd": obs[0, K:2 * K].copy(),
                "a": a, "explore": bool(a != int(obs[0, :K].argmax())),
                "regret": preg[name],
            }
            sim.step(np.array([a]))
        snaps.append(frame)
    means = sims["crown"].means[0]
    return snaps, means, {n: float(s.total[0]) for n, s in sims.items()}


def _render_gif(seed: int, tag: str, stride: int = 5, rng_seed: int = 900):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    snaps, means, totals = _replay_episode(seed, rng_seed)
    best = int(means.argmax())
    frames = snaps[::stride] + [snaps[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5),
                             gridspec_kw={"height_ratios": [3, 1.6]})
    (ax_c, ax_t), (ax_r, ax_e) = axes
    fig.suptitle(f"seed {seed} — crown {totals['crown']:.0f} vs "
                 f"thompson {totals['thompson']:.0f}  (true best arm {best})")

    def draw(i):
        f = frames[i]
        for ax, name in ((ax_c, "crown"), (ax_t, "thompson")):
            ax.clear()
            d = f[name]
            colors = ["#bbbbbb"] * K
            colors[best] = "#88bb88"
            colors[d["a"]] = "#cc4444" if d["explore"] else "#2255cc"
            ax.bar(range(K), d["pm"], yerr=d["psd"], color=colors, capsize=2)
            ax.plot(range(K), means, "k*", ms=7, label="true means")
            ax.set_ylim(-2.6, 2.9); ax.set_title(f"{name}  t={f['t']}")
            ax.axhline(0, color="k", lw=0.4)
        ts = [fr["t"] for fr in frames[:i + 1]]
        ax_r.clear()
        ax_r.plot(ts, [fr["crown"]["regret"] for fr in frames[:i + 1]],
                  label="crown", color="#2255cc")
        ax_r.plot(ts, [fr["thompson"]["regret"] for fr in frames[:i + 1]],
                  label="thompson", color="#996600")
        ax_r.set_title("cumulative pseudo-regret  \u03a3(\u03bc* \u2212 \u03bc chosen)"); ax_r.legend(loc="upper left")
        ax_r.set_xlim(0, T)
        ax_e.clear()
        w = 20
        ex_c = [np.mean([fr["crown"]["explore"] for fr in frames[max(0, j - w):j + 1]])
                for j in range(i + 1)]
        ex_t = [np.mean([fr["thompson"]["explore"] for fr in frames[max(0, j - w):j + 1]])
                for j in range(i + 1)]
        ax_e.plot(ts, ex_c, color="#2255cc"); ax_e.plot(ts, ex_t, color="#996600")
        ax_e.set_ylim(0, 1); ax_e.set_title("explore rate (rolling)")
        ax_e.set_xlim(0, T)

    ani = animation.FuncAnimation(fig, draw, frames=len(frames))
    out = OUTDIR / f"replay_{tag}_seed{seed}.gif"
    ani.save(out, writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    # pseudo-regret delta (crown better => positive), the plotted quantity
    dpr = snaps[-1]["thompson"]["regret"] - snaps[-1]["crown"]["regret"]
    return out, totals, dpr


def part_replay(n_seeds: int) -> dict:
    """Pick extreme seeds from the battery's paired deltas, replay each until
    the replayed outcome reproduces the recorded sign (stochastic policy —
    the recorded trajectory itself is not replayable), render GIFs."""
    crown = np.load(OUTDIR / "perseed_crown_s3.npz")["reward"]
    th = np.load(OUTDIR / "perseed_thompson.npz")["reward"]
    delta = crown - th
    order_win = np.argsort(-delta); order_loss = np.argsort(delta)
    med = int(np.argsort(np.abs(delta - np.median(delta)))[0])
    out = {}
    # A stochastic policy does not replay its recorded episode, so the gate
    # demands the replay reproduce a MATERIAL effect in pseudo-regret (the
    # plotted quantity), not merely the recorded sign — a "wrong-lock" GIF
    # that shows a tie illustrates nothing. Thresholds sit at the tail
    # anatomy's scale (dpr > 0 means the crown is ahead).
    for tag, order, accept in (("win", order_win, lambda d: d > 50.0),
                               ("loss", order_loss, lambda d: d < -100.0)):
        for cand in order[:15]:
            seed = int(PROTO_SEEDS[cand])
            path, totals, dpr = _render_gif(seed, tag)
            if accept(dpr):
                out[tag] = {"seed": seed, "recorded_delta": float(delta[cand]),
                            "replayed_pseudoregret_delta": float(dpr),
                            "gif": str(path)}
                break
            path.unlink()                  # effect not reproduced: next seed
    path, totals, dpr = _render_gif(int(PROTO_SEEDS[med]), "median")
    out["median"] = {"seed": int(PROTO_SEEDS[med]),
                     "recorded_delta": float(delta[med]),
                     "replayed_pseudoregret_delta": float(dpr),
                     "gif": str(path)}
    (OUTDIR / "replay_cases.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# rung 3 — anatomy of the paired-loss tail
# ---------------------------------------------------------------------------

def part_tail(n_seeds: int) -> dict:
    from mab_a5_probe import realize_means
    crown = np.load(OUTDIR / "perseed_crown_s3.npz")
    th = np.load(OUTDIR / "perseed_thompson.npz")
    delta = crown["reward"] - th["reward"]
    n = len(delta)
    worst = np.argsort(delta)[:100]
    rest = np.argsort(delta)[100:]
    means = realize_means(PROTO_SEEDS[:n])
    srt = np.sort(means, axis=1)
    gap12 = srt[:, -1] - srt[:, -2]              # top-two latent gap

    def stats(idx):
        return {
            "paired_delta_mean": float(delta[idx].mean()),
            "final_id_rate": float(crown["final_id"][idx].mean()),
            "thompson_final_id_rate": float(th["final_id"][idx].mean()),
            "lock_t_median": float(np.median(crown["lock_t"][idx])),
            "share_best_mean": float(crown["share_best"][idx].mean()),
            "regret_explore_mean": float(crown["regret_explore"][idx].mean()),
            "regret_exploit_mean": float(crown["regret_exploit"][idx].mean()),
            "explore_late_mean": float(crown["explore_late"][idx].mean()),
            "gap12_median": float(np.median(gap12[idx])),
        }

    out = {"worst100": stats(worst), "rest": stats(rest),
           "corr_delta_gap12": float(np.corrcoef(delta, gap12)[0, 1]),
           "worst100_where_thompson_also_misid": float(
               (1 - th["final_id"][worst]).mean())}
    (OUTDIR / "tail.json").write_text(json.dumps(out, indent=2))
    return out


PARTS = {"metrics": part_metrics, "replay": part_replay, "tail": part_tail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=list(PARTS))
    ap.add_argument("--n-seeds", type=int, default=len(PROTO_SEEDS))
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    res = PARTS[args.part](args.n_seeds)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

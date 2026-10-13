"""A SUPERVISED belief estimator for the raw EVENT history, and its frozen use inside a policy (#E109).

The human's design (2026-10-06): isolate the two parts the raw line learns at once. Stage 1 trains
an estimator of the hidden rate from the raw frame stack ALONE, by supervised learning on the
simulator's realised rate — a Gamma head (log shape, log rate) scored by the Gamma negative
log-likelihood of lambda, a proper scoring rule whose minimiser is the Gamma (moment) projection of
the exact posterior p(lambda | history) (the projection costs +0.22 against the exact rule on the
myopic part, #E107 gate G2). Stage 2 FREEZES it and trains the policy on its output rendered exactly
as `event_belief_log` renders the exact posterior — [time_to_go, critical_ratio, log mean, CV] —
so the policy pair reads against #E107's belief pair (the exact statistic, 768.97 / 762.52) and
against event_raw at the same recipe (715.7 / 732.8). The policy never sees the rate; the estimator
reads nothing hidden at inference. The frozen extractor's parameters travel in the saved policy.

Form: the E78 frame-MEAN map (phi 2 -> 64 -> 32 GELU on the observation columns of each real
history frame, pooled by the count; count and present state beside it, in horizon units), then a
64-64 GELU body to the two Gamma parameters — the same channel bank the raw policies use, so the
only thing that changes between the raw policy and this arm is WHO trains the channel bank and WHEN.

    python cnv_belief_estimator.py train  --out results/fig6a/estimator/E109 [--n-episodes 2000]
    python cnv_belief_estimator.py gate   --estimator results/fig6a/estimator/E109/estimator.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


N_STATE = 4   # leading frame columns: time-to-go, critical ratio, prior rate, prior mean


class BeliefEstimator(nn.Module):
    def __init__(self, T: int, F: int, horizon: float, channels: int = 32, hidden: int = 64, body: int = 64) -> None:
        super().__init__()
        self.T, self.F, self.horizon = int(T), int(F), float(horizon)
        self.phi = nn.Sequential(nn.Linear(F - N_STATE, hidden), nn.GELU(), nn.Linear(hidden, channels))
        self.body = nn.Sequential(nn.Linear(channels + 1 + N_STATE, body), nn.GELU(), nn.Linear(body, body), nn.GELU(), nn.Linear(body, 2))
        self.register_buffer("state_scale", th.tensor([self.horizon, 1.0, 1.0, self.horizon]))

    def features(self, obs: th.Tensor) -> th.Tensor:
        ttg = obs[..., 0]
        real = ((ttg >= 1.0) & (ttg < self.horizon)).float()            # history frames only, reset frame out
        z = self.phi(obs[..., N_STATE:]) * real[..., None]
        n = real.sum(1, keepdim=True).clamp(min=1.0)
        state = obs[:, -1, :N_STATE] / self.state_scale
        return th.cat([z.sum(1) / n, n / self.horizon, state], dim=1)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """(log shape, log rate) of the Gamma belief."""
        return self.body(self.features(obs))

    def belief(self, obs: th.Tensor) -> th.Tensor:
        """[log posterior mean, posterior CV] — the event_belief_log rendering of the Gamma belief."""
        p = self.forward(obs)
        return th.stack([p[:, 0] - p[:, 1], th.exp(-0.5 * p[:, 0])], dim=1)


def gamma_nll(p: th.Tensor, lam: th.Tensor) -> th.Tensor:
    k, log_beta = th.exp(p[:, 0]), p[:, 1]
    return -(k * log_beta - th.lgamma(k) + (k - 1.0) * th.log(lam) - th.exp(log_beta) * lam)


# ------------------------------------------------------------------------------------------------
# data: histories under a MIXTURE of behaviours on the event information set (the estimator must
# read any history the policy may produce, so the orders are varied deliberately)
# ------------------------------------------------------------------------------------------------
def _behaviour(kind: str, rng: np.random.Generator, y_myo: int, cap: int) -> int:
    if kind == "myopic_plus":        # the bar's order plus a premium 0..6 (the DP's shape lives here)
        return min(y_myo + int(rng.integers(0, 7)), cap)
    if kind == "myopic_jitter":      # the bar's order -1..+2 (zero and low orders occur)
        return int(np.clip(y_myo + int(rng.integers(-1, 3)), 0, cap))
    return int(np.exp(rng.uniform(0.0, np.log(cap + 1.0)))) - 1   # log-uniform on [0, cap]


KINDS = ("myopic_plus", "myopic_jitter", "loguniform")


def generate(scenario, n_episodes: int, seed0: int, rng_seed: int, with_exact: bool = False):
    """Returns the frame stacks (float16, exact for these integer-valued columns), the realised rate per row,
    the behaviour kind per episode, and (with_exact) the exact grid posterior's (log mean, CV, y_myo) per row."""
    from gymnasium.wrappers import FrameStackObservation
    from cnv_benchmark_myopic import MyopicEvent
    from cnv_event_posterior import EventGridPosterior
    from cnv_gym import CnvEnv
    N = int(scenario.horizon)
    env = FrameStackObservation(CnvEnv(scenario=scenario, observation_mode="event_raw"), stack_size=N, padding_type="zero")
    base = env.unwrapped; cap = int(scenario.arrivals_buffer); rng = np.random.default_rng(rng_seed)
    O, LAM, KIND, EX = [], [], [], []
    for i in range(n_episodes):
        kind = KINDS[i % len(KINDS)]
        o, _ = env.reset(seed=seed0 + i); lam = float(base._scenario_ep.demand.rate)
        ref = MyopicEvent(base)
        ex = EventGridPosterior(scenario.prior_shape, scenario.prior_rate, cap) if with_exact else None
        for t in range(N):
            y_myo = int(ref(base))
            if ex is not None:
                m, cv = ex.stats(); EX.append((np.log(m), cv, ex.myopic_order(float(scenario.critical_ratio))))
            O.append(o.astype(np.float16)); LAM.append(lam); KIND.append(kind)
            y = _behaviour(kind, rng, y_myo, cap)
            o, r, done, _, info = env.step(y)
            if ex is not None:
                ex.update(int(info["sales"]), int(info["stockout_event"]), y)
            if done:
                break
    out = (np.stack(O), np.array(LAM), np.array(KIND))
    return out + ((np.array(EX),) if with_exact else ())


def train(args) -> None:
    from cnv_scenarios import SCENARIOS
    sc = SCENARIOS[args.scenario]; th.manual_seed(0); th.set_num_threads(1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    O, LAM, KIND = generate(sc, args.n_episodes, args.seed0, rng_seed=1)
    Ov, LAMv, KINDv = generate(sc, max(args.n_episodes // 10, 30), args.seed0 + args.n_episodes, rng_seed=2)
    print(f"data: {len(O)} train rows, {len(Ov)} val rows ({time.time() - t0:.0f}s); kinds {dict(zip(*np.unique(KIND, return_counts=True)))}")
    T, F = O.shape[1:]
    net = BeliefEstimator(T, F, float(sc.horizon))
    opt = th.optim.Adam(net.parameters(), lr=args.lr)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_final)   # a flat 1e-3 left the validation NLL bouncing 0.53-0.75 (E109a)
    X = th.as_tensor(O); Y = th.as_tensor(LAM, dtype=th.float32); Xv = th.as_tensor(Ov).float(); Yv = th.as_tensor(LAMv, dtype=th.float32)
    n, bs = len(X), args.batch; best = (np.inf, None)
    for ep in range(args.epochs):
        perm = th.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; loss = gamma_nll(net(X[idx].float()), Y[idx]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tot += float(loss) * len(idx)
        with th.no_grad():
            vl = float(th.cat([gamma_nll(net(Xv[i:i + 4096]), Yv[i:i + 4096]) for i in range(0, len(Xv), 4096)]).mean())
        print(f"epoch {ep + 1:3d}  lr {opt.param_groups[0]['lr']:.2e}  train nll {tot / n:.4f}  val nll {vl:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        sched.step()
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    # fixed standardisation of the rendered belief, from the training rows (the belief mode's VecNormalize converges to this)
    with th.no_grad():
        B = th.cat([net.belief(X[i:i + 4096].float()) for i in range(0, n, 4096)])
        ttg = X[:, -1, 0].float(); ratio = X[:, -1, 1].float()
        cols = th.stack([ttg, ratio, B[:, 0], B[:, 1]], dim=1)
        mean, sd = cols.mean(0), cols.std(0)
        sd = th.where(sd < 1e-6, th.ones_like(sd), sd)   # a constant column (the ratio) renders as 0, as VecNormalize does
    th.save({"state_dict": net.state_dict(), "T": T, "F": F, "horizon": float(sc.horizon), "obs_mean": mean, "obs_sd": sd,
             "val_nll": best[0], "n_train": n, "scenario": args.scenario}, out / "estimator.pt")
    (out / "train_meta.json").write_text(json.dumps({"n_train": n, "n_val": len(Xv), "epochs": args.epochs, "lr": args.lr, "batch": bs,
                                                      "lr_final": args.lr_final, "val_nll_best": best[0], "seed0": args.seed0, "scenario": args.scenario}, indent=1))
    print(f"saved {out / 'estimator.pt'}  val nll {best[0]:.4f}")


def gate(args) -> None:
    """Held-out episodes under each behaviour kind, against the EXACT grid posterior (cnv_event_posterior): the
    estimator's log mean and CV (R²), its Gamma-projected myopic order against the exact order (agreement), and its
    NLL on lambda against the exact posterior's own Gamma projection (the attainable floor)."""
    from cnv_scenarios import SCENARIOS
    ck = th.load(args.estimator, weights_only=False); sc = SCENARIOS[ck["scenario"]]
    net = BeliefEstimator(ck["T"], ck["F"], ck["horizon"]); net.load_state_dict(ck["state_dict"]); net.eval(); th.set_num_threads(1)
    from cnv_bayes import gamma_poisson_quantile
    O, LAM, KIND, EX = generate(sc, args.n_episodes, args.seed0, rng_seed=3, with_exact=True)
    with th.no_grad():
        X = th.as_tensor(O).float()
        P = th.cat([net(X[i:i + 4096]) for i in range(0, len(X), 4096)])
        B = th.stack([P[:, 0] - P[:, 1], th.exp(-0.5 * P[:, 0])], dim=1).numpy()
        nll = gamma_nll(P, th.as_tensor(LAM, dtype=th.float32)).numpy()
    lpm, cv, ymyo = EX[:, 0], EX[:, 1], EX[:, 2]
    # the exact posterior's Gamma projection: shape 1/cv², rate shape/mean — its NLL on lambda is the floor
    k_ex = 1.0 / np.maximum(cv, 1e-6) ** 2; beta_ex = k_ex / np.exp(lpm)
    nll_ex = -(k_ex * np.log(beta_ex) - np.array([float(th.lgamma(th.tensor(v))) for v in k_ex]) + (k_ex - 1) * np.log(LAM) - beta_ex * LAM)
    q = float(sc.critical_ratio)
    y_hat = np.array([gamma_poisson_quantile(float(1.0 / max(c, 1e-6) ** 2), float(1.0 / max(c, 1e-6) ** 2 / np.exp(l)), q) for l, c in B])
    def r2(y, p): return 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    print(f"gate on {args.n_episodes} held-out episodes ({len(O)} rows), estimator val nll {ck['val_nll']:.4f}")
    print(f"  {'rows':>14s} {'n':>6s} {'logmean R²':>10s} {'CV R²':>7s} {'y_myo agree':>11s} {'MAE':>5s} {'NLL est':>8s} {'NLL exact-proj':>14s}")
    sel = {"all": np.ones(len(O), bool), **{k: KIND == k for k in KINDS}}
    lam_bands = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 25), (25, np.inf)]
    sel.update({f"λ[{lo:g},{hi:g})": (LAM >= lo) & (LAM < hi) for lo, hi in lam_bands})
    T_idx = np.array([int(round(sc.horizon - o[-1, 0])) for o in O])
    sel.update({"t<5": T_idx < 5, "t≥60": T_idx >= 60})
    for name, m in sel.items():
        if m.sum() < 50: continue
        print(f"  {name:>14s} {m.sum():6d} {r2(lpm[m], B[m, 0]):10.3f} {r2(cv[m], B[m, 1]):7.3f} {np.mean(y_hat[m] == ymyo[m]):11.1%} {np.abs(y_hat[m] - ymyo[m]).mean():5.2f} {nll[m].mean():8.4f} {nll_ex[m].mean():14.4f}")


# ------------------------------------------------------------------------------------------------
# stage 2: the frozen estimator as the policy's features extractor
# ------------------------------------------------------------------------------------------------
try:
    import gymnasium as gym
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class FrozenBeliefExtractor(BaseFeaturesExtractor):
        """Renders [time_to_go, critical_ratio, log mean, CV] from the raw frame stack through the trained
        estimator, standardised by the fixed statistics saved with it (the belief mode's VecNormalize at
        convergence), clipped at ±10. Built from the checkpoint's shape; the weights load from the path when it
        exists and are overwritten by the policy's own state dict on `PPO.load` (they are saved with the
        policy), so a saved policy evaluates anywhere. `cnv_ppo_train.py --belief-estimator` freezes it."""

        def __init__(self, observation_space: gym.Space, estimator_path: str = "", clip: float = 10.0) -> None:
            super().__init__(observation_space, features_dim=4)
            T, F = observation_space.shape
            horizon = float(np.asarray(observation_space.high)[..., 0].max())
            self.net = BeliefEstimator(T, F, horizon)
            self.register_buffer("obs_mean", th.zeros(4)); self.register_buffer("obs_sd", th.ones(4))
            self.clip = float(clip)
            self.estimator_path = str(estimator_path)
            if estimator_path and Path(estimator_path).exists():
                self.load_estimator(estimator_path)
            self.net.eval()

        def load_estimator(self, path: str) -> None:
            """(Re)load the trained weights and the fixed statistics. Called in __init__ AND again by the train
            script after the policy is built: SB3's `ActorCriticPolicy._build` applies its orthogonal init to the
            features extractor after construction, which silently replaced the loaded weights (caught in the
            #E109 smoke: the saved extractor differed from the estimator)."""
            ck = th.load(path, weights_only=False)
            assert (ck["T"], ck["F"]) == (self.net.T, self.net.F), f"estimator shape {(ck['T'], ck['F'])} vs observation {(self.net.T, self.net.F)}"
            self.net.load_state_dict(ck["state_dict"]); self.obs_mean.copy_(ck["obs_mean"]); self.obs_sd.copy_(ck["obs_sd"])

        def forward(self, obs: th.Tensor) -> th.Tensor:
            with th.no_grad():
                b = self.net.belief(obs)
                x = th.cat([obs[:, -1, :1], obs[:, -1, 1:2], b], dim=1)
                return th.clamp((x - self.obs_mean) / self.obs_sd, -self.clip, self.clip)
except ImportError:   # the estimator itself needs neither gym nor SB3
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--out", required=True); t.add_argument("--scenario", default="fig6a")
    t.add_argument("--n-episodes", type=int, default=2000); t.add_argument("--seed0", type=int, default=5_000_000)
    t.add_argument("--epochs", type=int, default=60); t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--lr-final", dest="lr_final", type=float, default=1e-5); t.add_argument("--batch", type=int, default=1024)
    g = sub.add_parser("gate"); g.add_argument("--estimator", required=True); g.add_argument("--n-episodes", type=int, default=300); g.add_argument("--seed0", type=int, default=6_000_000)
    a = ap.parse_args()
    (train if a.cmd == "train" else gate)(a)

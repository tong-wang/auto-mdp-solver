"""Beta probe (ESCALATION.md A7-t3, post-mortem): what did the optimizer DO
with the learned temperature head? No RL training — eval-only, on the t3
artifacts.

Context. t3 (`TempIndexPolicy`) factors the policy into "which arm" (the
equivariant scores z) times "how sure" (one invariant scale beta(s)), with
`tau_out` zero-initialized so beta == 1 everywhere at init: t3 STARTS as
exactly the crowned index policy and can only diverge through what training
does with the new degree of freedom. It ended at 1299.28 selection against
its own 1351.33 control, peaking at 3M and never recovering over 17M more
steps. A strictly larger class, initialized on the incumbent's manifold,
got worse.

Hypothesis (the "entropy hijack", 2026-08-01). beta is a single dof that
moves entropy directly and costs NOTHING in ranking. Before t3, the flat
entropy bonus could only raise entropy by distorting z, which degrades the
arm ordering and is pushed back on by the advantage term — that tension is
what bounded it. With beta present the bonus has a dedicated free lever:
shrink beta, entropy rises, the ordering is untouched. So a constant-price
entropy term should capture beta and pin it LOW, everywhere, early in
training. The parametrization fix would then have handed the pathology a
better tool than it had before.

Note this is decidable because beta is not merely observable — it is
representationally CAPABLE of the Thompson-like behavior we want. It reads
`pooled(post_mean, post_sd)` and ttg (mab_equinet `_TempIndexExtractor`),
so "sharpen as the posteriors tighten" is a function it can express. If it
did not learn that, the defect is in the objective, not the parametrization.

PRE-REGISTERED READS (fixed before the first run, ESCALATION discipline).
Let b_early = median beta over t in [0,100), b_late = median over [900,1000),
and dH = H(softmax(beta*z)) - H(softmax(z)), the entropy beta ADDS over the
same scorer read at the index manifold.

  HIJACK CONFIRMED   b_late < 1 and b_late/b_early <= 1.2 (pinned low, flat)
                     with dH > 0 late — beta is spending its dof buying
                     entropy the flat bonus pays for. Prescribes t3'-a
                     (detach beta from the entropy term).
  HIJACK REFUTED     b_late >= 1 and b_late/b_early >= 1.5 — the head DID
                     learn to anneal; t3's deficit is elsewhere and the
                     t3'-a/-c rungs are not indicated.
  PARTIAL            b_late < 1 but b_late/b_early >= 1.5 — the shape was
                     learned, the level was held down. Same prescription,
                     weaker claim.
  INCONCLUSIVE       anything else (report the numbers, claim nothing).

Comparability: explore-rate uses A7-rung-c's definition (a != argmax
post_mean) so the curves sit alongside #E17's crown numbers (22% -> 30.5%,
late entropy 0.78 nats). Trajectories are the DEPLOYED distribution — actions
sampled from the policy, not argmax.

The vectorized simulator and its faithful payout streams are imported from
mab_a5_probe (bit-exactness vs MabEnv asserted by its `verify` part). These
are diagnostic curves on the PROBE_SEEDS block, never protocol records.

Usage (from mab/):
    OMP_NUM_THREADS=1 python mab_beta_probe.py
    OMP_NUM_THREADS=1 python mab_beta_probe.py --n-seeds 128     # smoke
    OMP_NUM_THREADS=1 python mab_beta_probe.py --checkpoints     # beta vs step
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mab_a5_probe import K, T, PROBE_SEEDS, SCENARIO, VecSim

T3     = (Path("results") / SCENARIO
          / "PPO_obsbayes_L1_policyindex_temp_normobsFalse_seed1_20260801_001833")
OUTDIR = Path("results") / SCENARIO / "beta_probe"

EARLY = slice(0, 100)
LATE  = slice(900, 1000)


class TempPolicy:
    """The t3 artifact, batched, with beta and z exposed separately.

    Mirrors mab_anneal_probe.CrownPolicy on the VecNormalize contract (the
    equivariant policies train on raw obs; applying tracked-but-unused
    obs_rms stats would corrupt them), and additionally reaches into the
    extractor to read the temperature head.
    """

    def __init__(self, model_path: Path, run_dir: Path):
        import pickle
        import torch
        from stable_baselines3 import PPO
        self.torch = torch
        self.model = PPO.load(str(model_path), device="cpu")
        self.model.policy.set_training_mode(False)
        self.ex = self.model.policy.mlp_extractor
        # A8's identified head normalizes z before beta scales it; detect from
        # the extractor class so the same probe reads both parametrizations
        from mab_equinet import _IdTempIndexExtractor
        self.identified = isinstance(self.ex, _IdTempIndexExtractor)
        with open(run_dir / "vecnormalize.pkl", "rb") as fh:
            vn_state = pickle.load(fh).__dict__
        self._rms = vn_state["obs_rms"] if vn_state.get("norm_obs", True) else None
        self._clip = vn_state.get("clip_obs", 10.0)

    def _prep(self, obs: np.ndarray):
        if self._rms is not None:
            obs = np.clip((obs - self._rms.mean) / np.sqrt(self._rms.var + 1e-8),
                          -self._clip, self._clip)
        return self.torch.as_tensor(obs.astype(np.float32))

    def read(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> (probs, beta, z). z is the scorer's raw per-arm score, i.e. the
        logits t3 would emit on the index manifold it was initialized on."""
        import torch.nn.functional as F
        from mab_equinet import _BETA0, _IndexExtractor
        t = self._prep(obs)
        with self.torch.no_grad():
            z = _IndexExtractor.forward_actor(self.ex, t)      # base class: no beta
            if self.identified:                                # A8: beta scales zhat
                z = (z - z.mean(dim=1, keepdim=True)) / self.torch.sqrt(
                    z.var(dim=1, unbiased=False, keepdim=True) + self.ex._EPS)
            x, y, ttg = self.ex._split(t)
            pooled = self.ex.tau_enc(self.torch.stack([x, y], dim=2)).mean(dim=1)
            beta = F.softplus(
                self.ex.tau_out(self.torch.cat([pooled, ttg], dim=1)) + _BETA0)
            logits = self.ex.forward_actor(t)
            # the decomposition must be the policy's own, not a re-derivation
            assert self.torch.allclose(beta * z, logits, atol=1e-5), \
                "beta*z != forward_actor — extractor contract changed"
            probs = self.model.policy.get_distribution(t).distribution.probs
            assert self.torch.allclose(probs, F.softmax(logits, dim=1), atol=1e-5), \
                "get_distribution != softmax(logits) — action_net not Identity?"
            return (probs.numpy(), beta.numpy().ravel(), z.numpy())


def _sample(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (p.cumsum(axis=1) > rng.random((len(p), 1))).argmax(axis=1)


def _entropy_rows(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def profile(model_path: Path, n_seeds: int, rng_seed: int = 17) -> dict:
    seeds = PROBE_SEEDS[:n_seeds]
    idx = np.arange(len(seeds))
    pol, rng = TempPolicy(model_path, T3), np.random.default_rng(rng_seed)
    sim = VecSim(seeds)

    beta_med = np.empty(T); beta_p10 = np.empty(T); beta_p90 = np.empty(T)
    H_act = np.empty(T); H_b1 = np.empty(T)
    explore = np.empty(T); psd_mean = np.empty(T)
    for t in range(T):
        obs = sim.obs()
        p, beta, z = pol.read(obs)
        a = _sample(p, rng)
        beta_med[t], beta_p10[t], beta_p90[t] = np.percentile(beta, [50, 10, 90])
        H_act[t] = _entropy_rows(p).mean()
        # the same scorer read at beta == 1: t3's own index-manifold control
        H_b1[t] = _entropy_rows(_softmax(z)).mean()
        explore[t] = float((a != obs[:, :K].argmax(axis=1)).mean())
        psd_mean[t] = float(obs[:, K:2 * K].mean())
        sim.step(a)

    b_early = float(np.median(beta_med[EARLY]))
    b_late = float(np.median(beta_med[LATE]))
    ratio = b_late / b_early if b_early else float("nan")
    dH_late = float((H_act[LATE] - H_b1[LATE]).mean())
    # does beta track uncertainty at all (Thompson's structure), or only noise?
    corr_psd = float(np.corrcoef(beta_med, psd_mean)[0, 1])
    corr_t = float(np.corrcoef(beta_med, np.arange(T))[0, 1])

    if b_late < 1.0 and ratio <= 1.2 and dH_late > 0:
        verdict = "HIJACK CONFIRMED"
    elif b_late >= 1.0 and ratio >= 1.5:
        verdict = "HIJACK REFUTED"
    elif b_late < 1.0 and ratio >= 1.5:
        verdict = "PARTIAL"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "model": model_path.name,
        "n_seeds": len(seeds),
        "verdict": verdict,
        "beta_early_median": round(b_early, 4),
        "beta_late_median": round(b_late, 4),
        "beta_late_over_early": round(ratio, 4),
        "beta_min_median": round(float(beta_med.min()), 4),
        "beta_max_median": round(float(beta_med.max()), 4),
        "entropy_actual_early": round(float(H_act[EARLY].mean()), 4),
        "entropy_actual_late": round(float(H_act[LATE].mean()), 4),
        "entropy_at_beta1_late": round(float(H_b1[LATE].mean()), 4),
        "dH_late_from_beta": round(dH_late, 4),
        "explore_rate_early": round(float(explore[EARLY].mean()), 4),
        "explore_rate_late": round(float(explore[LATE].mean()), 4),
        "corr_beta_post_sd": round(corr_psd, 4),
        "corr_beta_t": round(corr_t, 4),
        "curves": {
            "beta_median": beta_med.round(4).tolist(),
            "beta_p10": beta_p10.round(4).tolist(),
            "beta_p90": beta_p90.round(4).tolist(),
            "entropy_actual": H_act.round(4).tolist(),
            "entropy_at_beta1": H_b1.round(4).tolist(),
            "explore_rate": explore.round(4).tolist(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=len(PROBE_SEEDS))
    ap.add_argument("--checkpoints", action="store_true",
                    help="also profile every checkpoints/*.zip — beta vs "
                         "training step, which dates the hijack")
    ap.add_argument("--run-dir", type=str, default=None,
                    help="probe this run instead of the A7-t3 default "
                         "(A8-c's identified head is auto-detected); output "
                         "goes to beta_probe/profile_{dirname}.json")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    global T3
    out_name = "profile.json"
    if args.run_dir:
        T3 = Path(args.run_dir)
        out_name = f"profile_{T3.name}.json"

    models = [T3 / f"{SCENARIO}_ppo.zip",          # best-selection checkpoint
              T3 / f"{SCENARIO}_ppo_final.zip"]    # end of budget (20M)
    if args.checkpoints:
        models += sorted((T3 / "checkpoints").glob("*.zip"))

    out = []
    for m in models:
        if not m.is_file():
            print(f"SKIP missing {m}", flush=True)
            continue
        t0 = time.time()
        res = profile(m, args.n_seeds)
        res["wall_seconds"] = round(time.time() - t0, 1)
        out.append(res)
        print(json.dumps({k: v for k, v in res.items() if k != "curves"},
                         indent=2), flush=True)
    (OUTDIR / out_name).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUTDIR / out_name}")


if __name__ == "__main__":
    main()

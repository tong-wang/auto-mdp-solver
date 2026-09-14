"""Shared evaluation core for the inv_single campaign (spec §9).

Driver-level module: sits above the gym, imported by `inv_single_ppo_eval.py`,
`inv_single_benchmark_*_eval.py`, and the training script's selection callback,
so the RL arm and every benchmark are scored by *identical* code on an
identical seed block.

Why it exists: the campaign's verdicts are **paired** differences under common
random numbers (see ESCALATION.md "Standard eval protocol"). Pairing is only
valid if every arm sees the same episode seeds in the same objective, so the
seed block, the accumulators, and the TSV record are defined once here rather
than per script.

Key components:
- `eval_seed_block` / `SELECT_SEED_OFFSET` — the reporting block `0…n-1` and a
  disjoint block for in-training model selection (spec §8.6: the selection
  seeds must never overlap the reporting seeds).
- `rollout_seeds` — batched CRN rollout. Episodes here have a fixed length
  (`horizon_end=terminated`, no early termination), so a whole batch of seeds
  starts and finishes in lockstep; the batch exists purely to amortize the
  policy's forward pass.
- `summarize` / `write_record` — the §9.3 record, objective column first.
- `paired_report` — per-seed paired Δ against a benchmark's per-seed dump.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# Episode seeds used for in-training model selection. Disjoint from the
# reporting block 0…n_seeds-1 by construction (spec §8.6, §9.7): selecting a
# checkpoint on the seeds that later report it is selection bias.
SELECT_SEED_OFFSET = 1_000_000

# Per-episode metrics accumulated by `rollout_seeds`. cost_total is the
# objective and leads the record (§9.3); the rest are diagnostics.
METRICS = (
    "cost_total",
    "cost_holding",
    "cost_shortage",
    "cost_ordering",
    "lost_sales",
    "n_orders",
)


def eval_seed_block(n_seeds: int, offset: int = 0) -> np.ndarray:
    """Episode seeds `offset … offset+n_seeds-1` (spec §9.2)."""
    return np.arange(offset, offset + n_seeds, dtype=np.int64)


# ---------------------------------------------------------------------------
# Batched CRN rollout
# ---------------------------------------------------------------------------

# An action callback: (obs_batch, raw_envs) -> action array of shape (K, ...).
# `obs_batch` is normalized when a VecNormalize was supplied; a benchmark
# policy that reads simulator state instead uses `raw_envs[i]._state`.
ActFn = Callable[[np.ndarray, Sequence], np.ndarray]


def rollout_seeds(
    *,
    scenario,
    observation_mode: str,
    action_mode: str,
    seeds: np.ndarray,
    act_fn: ActFn,
    batch: int = 64,
    vecnorm_path: Path | None = None,
    progress_every: int = 10_000,
) -> dict[str, np.ndarray]:
    """Roll out one episode per seed; return per-seed metric arrays.

    Every episode is driven through `InvSingleEnv` — the same wrapper RL
    trains on — so benchmark and RL numbers differ only in the policy.
    """
    import warnings

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from inv_single_gym import InvSingleEnv

    n = len(seeds)
    batch = max(1, min(batch, n))

    with warnings.catch_warnings():
        # the action-space/demand-discreteness advisory is not actionable here:
        # the eval must use whatever mode the arm was trained with
        warnings.simplefilter("ignore")
        raw_envs = [
            InvSingleEnv(
                scenario=scenario,
                observation_mode=observation_mode,
                action_mode=action_mode,
            )
            for _ in range(batch)
        ]

    venv = DummyVecEnv([lambda e=e: e for e in raw_envs])
    normalizer = None
    if vecnorm_path is not None and Path(vecnorm_path).exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
        normalizer = venv

    out = {m: np.zeros(n, dtype=np.float64) for m in METRICS}

    for start in range(0, n, batch):
        block = seeds[start : start + batch]
        if progress_every and start % progress_every == 0:
            print(f"  seed {start}/{n}", flush=True)

        # Seed each slot explicitly (spec §9.5): env_method bypasses
        # VecNormalize.reset, which would otherwise drop the seed.
        raw_obs = []
        for i in range(batch):
            # tail block: surplus slots replay the block's first seed and are
            # discarded below, so they cannot contaminate a reported metric
            s = int(block[i]) if i < len(block) else int(block[0])
            obs_i, _ = venv.env_method("reset", seed=s, indices=[i])[0]
            raw_obs.append(obs_i)
        obs = np.asarray(raw_obs, dtype=np.float32)
        if normalizer is not None:
            obs = normalizer.normalize_obs(obs)

        acc = {m: np.zeros(batch, dtype=np.float64) for m in METRICS}
        done = np.zeros(batch, dtype=bool)
        while not done.all():
            actions = act_fn(obs, raw_envs)
            obs, _, dones, infos = venv.step(actions)
            for i in range(batch):
                if done[i]:
                    continue
                info = infos[i] or {}
                cost = info.get("cost", {})
                acc["cost_total"][i] += float(cost.get("total", 0.0))
                acc["cost_holding"][i] += float(cost.get("holding", 0.0))
                acc["cost_shortage"][i] += float(cost.get("shortage", 0.0))
                acc["cost_ordering"][i] += float(
                    cost.get("order_fixed", 0.0) + cost.get("order_variable", 0.0)
                )
                acc["lost_sales"][i] += float(info.get("lost_sales", 0.0))
                acc["n_orders"][i] += float(float(info.get("order", 0.0)) > 0.0)
            done |= np.asarray(dones, dtype=bool)

        for m in METRICS:
            out[m][start : start + len(block)] = acc[m][: len(block)]

    venv.close()
    return out


# ---------------------------------------------------------------------------
# Record (spec §9.3)
# ---------------------------------------------------------------------------

def summarize(per_seed: dict[str, np.ndarray]) -> dict[str, float]:
    """Means for every metric, plus objective variance / SE / semi-variances."""
    x = per_seed["cost_total"]
    n = len(x)
    mu = float(x.mean())
    var = float(x.var(ddof=1)) if n > 1 else 0.0
    stats = {f"{m}_mean": float(per_seed[m].mean()) for m in METRICS}
    stats["cost_total_var"] = var
    stats["cost_total_se"] = float(np.sqrt(var / n)) if n > 1 else 0.0
    denom = max(1, n - 1)
    stats["semivar_d"] = float((np.maximum(mu - x, 0.0) ** 2).sum() / denom)
    stats["semivar_u"] = float((np.maximum(x - mu, 0.0) ** 2).sum() / denom)
    return stats


HEADER = (
    ["scenario", "arm", "n_seeds"]
    + [f"{m}_mean" for m in METRICS]
    + ["cost_total_var", "cost_total_se", "semivar_d", "semivar_u"]
)


def format_row(scenario: str, arm: str, n_seeds: int, stats: dict[str, float]) -> str:
    cells = [scenario, arm, str(n_seeds)]
    cells += [f"{stats[c]:.6f}" for c in HEADER[3:]]
    return "\t".join(cells)


def write_record(
    outfile: Path,
    scenario: str,
    arm: str,
    per_seed: dict[str, np.ndarray],
) -> dict[str, float]:
    """Write the §9.3 TSV row plus the per-seed objective dump.

    The `.npy` companion is what makes pairing possible later: means alone
    cannot produce a paired standard error.
    """
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    stats = summarize(per_seed)
    row = format_row(scenario, arm, len(per_seed["cost_total"]), stats)
    with open(outfile, "w") as f:
        f.write("\t".join(HEADER) + "\n")
        f.write(row + "\n")
    per_seed_path = outfile.with_name(outfile.stem + "_perseed.npy")
    np.save(per_seed_path, per_seed["cost_total"])
    print("\t".join(HEADER))
    print(row, flush=True)
    print(f"[eval] record   → {outfile}")
    print(f"[eval] per-seed → {per_seed_path}")
    return stats


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def paired_report(arm: np.ndarray, bar_path: Path, bar_name: str = "bar") -> None:
    """Paired Δ (arm − bar) on shared episode seeds, with the paired SE.

    Both arrays are indexed by episode seed, so `arm - bar` cancels instance
    luck; the paired SE is the only honest error bar for a CRN comparison.
    """
    bar = np.load(str(bar_path))
    n = min(len(arm), len(bar))
    if n < len(arm) or n < len(bar):
        print(
            f"[eval] WARNING: seed blocks differ in length "
            f"(arm {len(arm)}, {bar_name} {len(bar)}); pairing on the first {n}"
        )
    d = arm[:n] - bar[:n]
    mu = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    bar_mean = float(bar[:n].mean())
    pct = 100.0 * mu / abs(bar_mean) if bar_mean else float("nan")
    print(f"\n[paired vs {bar_name}]  n={n}")
    print(f"  {bar_name} mean cost : {bar_mean:.4f}")
    print(f"  arm mean cost       : {float(arm[:n].mean()):.4f}")
    print(f"  paired delta        : {mu:+.4f}  ± {se:.4f} (SE)   [{pct:+.2f}% of bar]")
    print(f"  t                   : {mu / se:+.2f}" if se else "  t: n/a")
    print("  (cost is minimized: delta < 0 means the arm BEATS the bar)")

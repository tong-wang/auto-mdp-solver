"""Shared evaluation core for the owmr campaign (spec 9).

Driver-level module: sits above the gym, imported by `owmr_ppo_eval.py`,
`owmr_select.py` and every `owmr_benchmark_*_eval.py`, so the RL arm and every
benchmark are scored by *identical* code on an identical seed block.

The campaign's verdicts are **paired** differences under common random numbers.
Pairing is valid only if every arm sees the same episode seeds in the same
objective, so the seed block, the accumulators and the TSV record are defined
once here.

Key components:
- `eval_seed_block` / `SELECT_SEED_OFFSET` — the protocol block `0…n-1` and the
  disjoint selection block (spec 9.7).
- `rollout_seeds` — batched CRN rollout through `OwmrEnv`; episodes have a
  fixed length, so a batch of seeds runs in lockstep and the batch exists only
  to amortize the policy's forward pass.
- `summarize` / `write_record` — the spec-9.3 record, objective column first,
  the IR's bystander metrics after it, `#` provenance lines above, and the
  per-seed `.seeds.tsv` sidecar.
- `paired_report` — per-seed paired delta against another arm's sidecar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

SELECT_SEED_OFFSET = 1_000_000

# objective first (spec 9.3), then its components, then the IR's eval_metrics
METRICS = (
    "cost_total",
    "cost_wh_holding",
    "cost_rt_holding",
    "cost_rt_shortage",
    "wh_share",
    "stockout_periods",
    "system_stock",
)
# per-period metrics reported as episode MEANS (the IR divides by horizon_T)
_MEAN_OVER_HORIZON = ("wh_share", "system_stock")


def eval_seed_block(n_seeds: int, offset: int = 0) -> np.ndarray:
    """Episode seeds `offset … offset+n_seeds-1` (spec 9.2)."""
    return np.arange(offset, offset + n_seeds, dtype=np.int64)


# An action callback: (obs_batch, raw_envs) -> action array of shape (K, N+1).
# `obs_batch` is normalized when a VecNormalize was supplied; a benchmark that
# reads simulator state instead uses `raw_envs[i]._state` (post-receipt).
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
    progress_every: int = 2048,
) -> dict[str, np.ndarray]:
    """Roll out one episode per seed; return per-seed metric arrays."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from owmr_gym import OwmrEnv

    n = len(seeds)
    batch = max(1, min(batch, n))
    raw_envs = [
        OwmrEnv(scenario, observation_mode=observation_mode, action_mode=action_mode)
        for _ in range(batch)
    ]
    venv = DummyVecEnv([lambda e=e: e for e in raw_envs])
    normalizer = None
    if vecnorm_path is not None and Path(vecnorm_path).exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
        normalizer = venv

    horizon = float(scenario.horizon)
    out = {m: np.zeros(n, dtype=np.float64) for m in METRICS}

    for start in range(0, n, batch):
        block = seeds[start:start + batch]
        if progress_every and start % progress_every == 0:
            print(f"  seed {start}/{n}", flush=True)
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
                acc["cost_wh_holding"][i] += float(cost.get("wh_holding", 0.0))
                acc["cost_rt_holding"][i] += float(cost.get("rt_holding", 0.0))
                acc["cost_rt_shortage"][i] += float(cost.get("rt_shortage", 0.0))
                met = info.get("metrics", {})
                acc["wh_share"][i] += float(met.get("wh_share", 0.0)) / horizon
                acc["stockout_periods"][i] += float(met.get("stockout_periods", 0.0))
                acc["system_stock"][i] += float(met.get("system_stock", 0.0)) / horizon
            done |= np.asarray(dones, dtype=bool)
        for m in METRICS:
            out[m][start:start + len(block)] = acc[m][:len(block)]

    venv.close()
    return out


# ---------------------------------------------------------------------------
# Record (spec 9.3)
# ---------------------------------------------------------------------------

HEADER = (
    ["scenario", "arm", "n_seeds"]
    + [f"{m}_mean" for m in METRICS]
    + ["cost_total_var", "cost_total_se", "semivar_d", "semivar_u"]
)


def summarize(per_seed: dict[str, np.ndarray]) -> dict[str, float]:
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


def format_row(scenario: str, arm: str, n_seeds: int, stats: dict[str, float]) -> str:
    cells = [scenario, arm, str(n_seeds)] + [f"{stats[c]:.6f}" for c in HEADER[3:]]
    return "\t".join(cells)


def write_record(
    outfile: Path,
    scenario: str,
    arm: str,
    per_seed: dict[str, np.ndarray],
    seeds: np.ndarray | None = None,
    provenance: Sequence[str] = (),
) -> dict[str, float]:
    """Write the spec-9.3 TSV (provenance `#` lines, header, one row) and the
    per-seed sidecar `<name>.seeds.tsv` (seed, objective, every metric) that
    makes paired comparisons possible later."""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    stats = summarize(per_seed)
    n = len(per_seed["cost_total"])
    row = format_row(scenario, arm, n, stats)
    with open(outfile, "w") as f:
        for line in provenance:
            f.write(line if line.startswith("#") else f"# {line}")
            f.write("\n")
        f.write("\t".join(HEADER) + "\n")
        f.write(row + "\n")
    sidecar = outfile.with_name(outfile.stem + ".seeds.tsv")
    if seeds is None:
        seeds = np.arange(n)
    with open(sidecar, "w") as f:
        f.write("seed\t" + "\t".join(METRICS) + "\n")
        for i in range(n):
            f.write(str(int(seeds[i])) + "\t"
                    + "\t".join(f"{per_seed[m][i]:.6f}" for m in METRICS) + "\n")
    print("\t".join(HEADER))
    print(row, flush=True)
    print(f"[eval] record   -> {outfile}")
    print(f"[eval] per-seed -> {sidecar}")
    return stats


def load_sidecar(path: Path) -> dict[str, np.ndarray]:
    """Read a `.seeds.tsv` sidecar back into per-metric arrays keyed by seed order."""
    path = Path(path)
    data = np.genfromtxt(path, delimiter="\t", names=True)
    return {name: np.asarray(data[name], dtype=np.float64) for name in data.dtype.names}


def paired_report(arm: np.ndarray, bar_path: Path, bar_name: str = "bar") -> None:
    """Paired delta (arm − bar) on shared episode seeds, with the paired SE."""
    bar = load_sidecar(bar_path)["cost_total"]
    n = min(len(arm), len(bar))
    if n < len(arm) or n < len(bar):
        print(f"[eval] WARNING: seed blocks differ in length (arm {len(arm)}, "
              f"{bar_name} {len(bar)}); pairing on the first {n}")
    d = arm[:n] - bar[:n]
    mu = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    bar_mean = float(bar[:n].mean())
    pct = 100.0 * mu / abs(bar_mean) if bar_mean else float("nan")
    print(f"\n[paired vs {bar_name}]  n={n}")
    print(f"  {bar_name} mean cost : {bar_mean:.4f}")
    print(f"  arm mean cost       : {float(arm[:n].mean()):.4f}")
    print(f"  paired delta        : {mu:+.4f}  ± {se:.4f} (SE)   [{pct:+.2f}% of bar]")
    print(f"  z                   : {mu / se:+.2f}" if se else "  z: n/a")
    print("  (cost is minimized: delta < 0 means the arm BEATS the bar)")

"""Shared evaluation harness for clark_scarf (spec §9).

Every arm — random, myopic, the Clark–Scarf DP, and the trained PPO policies —
scores through this module, so they share one seed protocol, one metric set,
and one record format and are therefore **pairwise** comparable.

Common random numbers: every arm replays the same episode-seed block
``0 … n_seeds-1`` (spec §9.2), so a paired difference cancels demand noise and
the paired SE is far tighter than the unpaired one. The per-seed objective is
dumped alongside the TSV precisely so pairing is possible after the fact.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from clark_scarf_gym import ClarkScarfEnv
from clark_scarf_scenarios import SCENARIOS, ClarkScarfScenario

METRICS = (
    "cost_total",       # the objective: discounted total cost
    "cost_undiscounted",  # report-only column (Phase A: declined objective candidate)
    "cost_holding",
    "cost_shortage",
    "cost_shipping",
    "units_shipped",
)

# (obs_batch, raw_envs) -> action array (K, n_live). A benchmark policy that
# reads simulator state rather than the agent observation uses `envs[i]._state`.
ActFn = Callable[[np.ndarray, Sequence[ClarkScarfEnv]], np.ndarray]


# §9.7's screen block. The block that RANKS must be disjoint from the block
# that REPORTS (the protocol block starts at 0), or a checkpoint is chosen on
# the seeds that later judge it. Lives here rather than in the train script
# because the screen, the train script and any future evaluator must all agree
# on it, and none of them owns it.
SELECT_SEED_OFFSET = 1_000_000


def eval_seed_block(n_seeds: int, offset: int = 0) -> np.ndarray:
    """Episode seeds ``offset … offset+n_seeds-1`` (spec §9.2)."""
    return np.arange(offset, offset + n_seeds, dtype=np.int64)


def rollout_seeds(
    *,
    scenario: ClarkScarfScenario,
    observation_mode: str,
    seeds: np.ndarray,
    action_mode: str = "ship_absolute",
    act_fn: ActFn,
    batch: int = 64,
    progress_every: int = 20_000,
) -> dict[str, np.ndarray]:
    """Roll every seed through ``ClarkScarfEnv`` — the same wrapper RL trains on.

    Batched only for speed; each episode is independent and keyed by its own
    episode seed, so batching cannot change any result.

    ``action_mode`` defaults to ``"ship_absolute"`` — the identity encoding —
    because a benchmark computes a shipment QUANTITY directly. An RL arm must
    pass the mode its policy was trained under, or the action is decoded wrongly.
    """
    # the instance's own discount (F11): the IR names it, the scenario carries
    # it, and every arm on one leaderboard is scored with the same one
    beta = scenario.beta
    out = {m: np.zeros(len(seeds), dtype=np.float64) for m in METRICS}

    for start in range(0, len(seeds), batch):
        chunk = seeds[start : start + batch]
        envs = [
            ClarkScarfEnv(scenario, observation_mode=observation_mode,
                          action_mode=action_mode)
            for _ in chunk
        ]
        obs = np.stack([e.reset(seed=int(s))[0] for e, s in zip(envs, chunk)])
        live = np.ones(len(chunk), dtype=bool)
        t = 0
        while live.any():
            acts = act_fn(obs, envs)
            new_obs = []
            for i, env in enumerate(envs):
                if not live[i]:
                    new_obs.append(obs[i])
                    continue
                o, r, term, trunc, info = env.step(acts[i])
                j = start + i
                c = info["cost"]
                out["cost_total"][j] += (beta**t) * c["total"]
                out["cost_undiscounted"][j] += c["total"]
                out["cost_holding"][j] += (beta**t) * c["holding"]
                out["cost_shortage"][j] += (beta**t) * c["shortage"]
                out["cost_shipping"][j] += (beta**t) * c.get("shipping", 0.0)
                out["units_shipped"][j] += float(sum(info["shipped"]))
                new_obs.append(o)
                if term or trunc:
                    live[i] = False
            obs = np.stack(new_obs)
            t += 1
        if progress_every and (start // max(batch, 1)) % max(
            1, progress_every // max(batch, 1)
        ) == 0:
            print(f"[eval] {start + len(chunk)}/{len(seeds)} seeds", flush=True)
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

    The ``.npy`` companion is what makes pairing possible later: means alone
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
    print(f"[eval] record   -> {outfile}")
    print(f"[eval] per-seed -> {per_seed_path}")
    return stats


def paired_report(arm: np.ndarray, bar_path: Path, bar_name: str = "bar") -> None:
    """Paired difference against another arm's per-seed dump (same CRN block)."""
    bar = np.load(bar_path)
    n = min(len(arm), len(bar))
    d = arm[:n] - bar[:n]
    mu = float(d.mean())
    se = float(np.sqrt(d.var(ddof=1) / n)) if n > 1 else 0.0
    bar_mean = float(bar[:n].mean())
    pct = 100.0 * mu / abs(bar_mean) if bar_mean else float("nan")
    print(f"\n[paired vs {bar_name}]  n={n}")
    print(f"  {bar_name} mean cost : {bar_mean:.4f}")
    print(f"  delta (arm - bar)    : {mu:.4f} +/- {se:.4f} (1 SE)   {pct:+.2f}%")
    if se > 0:
        print(f"  z = {mu / se:.2f}   (negative favours the arm; cost is minimized)")


def resolve_outfile(
    scenario: str, arm: str, outdir: str, outfile: str | None
) -> Path:
    """Anchor results at THIS FILE's directory, never the CWD (spec §8.4)."""
    root = Path(__file__).resolve().parent
    if outfile is not None:
        p = Path(outfile)
        return p if p.is_absolute() else root / p
    return root / outdir / scenario / "benchmark" / f"benchmark_{arm}_eval_{scenario}.tsv"


# ---------------------------------------------------------------------------
# Shared benchmark driver
# ---------------------------------------------------------------------------


def benchmark_main(arm: str, build_policy, description: str) -> None:
    """Argparse + rollout + record, shared by every benchmark eval script.

    ``build_policy(scenario)`` returns an object with ``actions(state)``. The
    policy reads *simulator state*, not the agent observation — a benchmark is
    not restricted to what a policy network can see, and saying so explicitly
    keeps the comparison honest.
    """
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("-s", "--scenario_name", default="n3_l2_p09",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", default="raw",
                   choices=list(ClarkScarfEnv.OBS_MODES),
                   help="Cosmetic for a benchmark: the policy reads simulator "
                        "state. Kept so the env is built exactly as in the RL arm.")
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--outdir", default="results")
    p.add_argument("--outfile", default=None)
    p.add_argument("--paired-with", default=None,
                   help="another arm's *_perseed.npy; prints the paired delta")
    args = p.parse_args()

    scenario = SCENARIOS[args.scenario_name]
    policy = build_policy(scenario)

    def act_fn(obs, envs):
        return np.stack([policy.actions(e._state) for e in envs])

    seeds = eval_seed_block(args.n_seeds)
    print(f"[{arm}] scenario={args.scenario_name} seeds={args.n_seeds} "
          f"N={scenario.n_echelons} L={scenario.leadtime}", flush=True)
    per_seed = rollout_seeds(
        scenario=scenario,
        observation_mode=args.observation_mode,
        seeds=seeds,
        act_fn=act_fn,
        batch=args.batch_envs,
        # benchmarks emit quantities directly, so they want the IDENTITY
        # encoding -- named `ship_absolute` since the three-mode rename. It was
        # left as the old `ship` here and the gym's assert has refused every
        # benchmark eval since: the recorded rows predate the rename and are
        # reproducible (a replay through this mode matches them to 9e-13), but
        # nothing could regenerate them until this line was corrected.
        action_mode="ship_absolute",
    )
    outfile = resolve_outfile(args.scenario_name, arm, args.outdir, args.outfile)
    write_record(outfile, args.scenario_name, arm, per_seed)
    if args.paired_with:
        paired_report(per_seed["cost_total"], Path(args.paired_with), "reference")

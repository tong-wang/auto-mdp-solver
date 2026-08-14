"""Protocol-tier evaluation of a trained PPO model on an FNV scenario or grid (spec §9).

This is the **protocol** layer of the three-layer evaluation (§9.7): it confirms
the top-k checkpoints the screen (fnv_select.py) ranked, and its numbers are the
ones that go on the leaderboard. A ladder run exists only once this TSV does.

Seed blocks are mutually disjoint by construction (§9.7) — a layer that ranks
must not touch the block that quotes:

    protocol block  seeds PROTOCOL_SEED_BASE .. +n_seeds-1   (this file)
    screen block    seeds SCREEN_SEED_BASE   .. +n_seeds-1   (fnv_select.py)

Common random numbers: every arm evaluated on the same block sees identical
demand-signal draws per seed, so instance luck cancels in a paired comparison.
Use --per-seed-out to dump the per-seed profits a paired Δ needs.

Example usage:
    python fnv_ppo_eval.py --model-path results/simple/PPO_.../simple_ppo.zip -s simple
    python fnv_ppo_eval.py --model-path results/FNV-aMMFE/PPO_.../checkpoints/fnv_ppo_1800000_steps.zip \
        -s FNV-aMMFE --n-seeds 8192
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from fnv_gym import FnvEnv, ScenarioSource
from fnv_grids import GRIDS
from fnv_scenarios import FnvScenario, SCENARIOS


# ---------------------------------------------------------------------------
# Seed blocks (§9.7) — disjoint by construction
# ---------------------------------------------------------------------------

PROTOCOL_SEED_BASE = 0            # evidence-grade block, quoted
SCREEN_SEED_BASE   = 1_000_000    # checkpoint-ranking block, never quoted

# Evidence grade (§9.7). FNV episodes are 3 steps, so 8192 seeds is cheap;
# size down only with a pilot-SD argument, not by reflex.
DEFAULT_N_SEEDS = 8192


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate PPO on an FNV scenario or grid.")
    p.add_argument("--model-path",   type=str, required=True,
                   help="Path to the saved model (.zip)")
    p.add_argument("--vecnorm-path", type=str, default=None,
                   help="Path to vecnormalize.pkl (defaults beside the model, "
                        "then the run dir above a checkpoints/ folder)")
    p.add_argument("-s", "--scenario_name", type=str, default="FNV-aMMFE",
                   choices=list(SCENARIOS) + list(GRIDS),
                   help="Scenario (§5.4) or grid (§5.6) to evaluate; a grid "
                        "enumerates to one row per cell")
    p.add_argument("-o", "--observation_mode", type=str, default="vec", choices=["vec"])
    p.add_argument("-a", "--action_mode",      type=str, default="box", choices=["box"])
    p.add_argument("-r", "--reward_mode",      type=str, default="profit",
                   choices=["profit", "regret"])
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS,
                   help=f"Episode seeds per cell (default {DEFAULT_N_SEEDS}, §9.7 evidence grade)")
    p.add_argument("--seed-base", type=int, default=PROTOCOL_SEED_BASE,
                   help="First seed of the CRN block; keep the protocol block "
                        "disjoint from the screen block")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Episodes evaluated concurrently (vectorized, §9.7)")
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV output (defaults to <model_dir>/ppo_eval_<scenario_name>.tsv)")
    p.add_argument("--per-seed-out", type=str, default=None,
                   help="Optional TSV of per-seed profits, for paired Δ vs another arm")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# Console capture (§8.4) — stdout/stderr is an artifact, not scratch
# ---------------------------------------------------------------------------

class _Tee:
    """Write-through proxy mirroring a stream into a second file object."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._fh.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def tee_console(outdir: Path, name: str) -> None:
    """Mirror stdout/stderr into `outdir/name` (spec §8.4).

    Defined here rather than in the train script because every driver needs it
    and the destination is only known once the run/benchmark dir is resolved.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    fh = open(outdir / name, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


# ---------------------------------------------------------------------------
# Observation normalization (§9.5) — the saved stats are part of the contract
# ---------------------------------------------------------------------------

def load_obs_rms(vecnorm_path: Path | None):
    """Return (obs_rms, clip_obs) from a saved VecNormalize, or (None, 10.0).

    Loaded directly rather than through VecNormalize.load so the evaluator's
    batch width is independent of the env count the stats were trained with.
    """
    if vecnorm_path is None or not Path(vecnorm_path).exists():
        return None, 10.0
    with open(vecnorm_path, "rb") as f:
        vecnorm = pickle.load(f)
    # An L0 run trains without obs normalization (§8.6). Its saved stats are
    # untouched defaults, so applying them would impose a transform — and a
    # clip — the policy never saw. The flag travels with the artifact.
    if not getattr(vecnorm, "norm_obs", True):
        return None, float(vecnorm.clip_obs)
    return vecnorm.obs_rms, float(vecnorm.clip_obs)


def normalize_obs(obs: np.ndarray, obs_rms, clip_obs: float) -> np.ndarray:
    if obs_rms is None:
        return obs
    return np.clip(
        (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
        -clip_obs, clip_obs,
    ).astype(np.float32)


def resolve_vecnorm_path(model_path: Path, explicit: str | None) -> Path | None:
    """Find vecnormalize.pkl: explicit, beside the model, or in the run dir.

    Checkpoints live in <run>/checkpoints/, so a checkpoint's stats sit one
    level up unless CheckpointCallback saved its own copy alongside.
    """
    if explicit:
        return Path(explicit)
    beside = model_path.parent / "vecnormalize.pkl"
    if beside.exists():
        return beside
    run_dir = model_path.parent.parent / "vecnormalize.pkl"
    return run_dir if run_dir.exists() else None


# ---------------------------------------------------------------------------
# Batched rollout — shared with fnv_select.py
# ---------------------------------------------------------------------------

def rollout_seeds(
    act_batch,
    scenario: ScenarioSource,
    seeds,
    observation_mode: str = "vec",
    action_mode: str = "box",
    reward_mode: str = "profit",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll `seeds` through `scenario`, returning (profits, regrets) per seed.

    Vectorized over episodes (§9.7): FNV episodes are exactly N steps and share
    no cross-episode state, so a batch of envs advances in lockstep and the
    policy sees one (B, obs_dim) array per step instead of B scalar calls.

    `act_batch` maps a raw (B, obs_dim) observation array to (B, 1) actions; it
    owns whatever normalization the policy's input contract requires.
    """
    seeds   = np.asarray(list(seeds), dtype=np.int64)
    profits = np.zeros(len(seeds))
    regrets = np.zeros(len(seeds))

    pool = [
        FnvEnv(scenario=scenario, observation_mode=observation_mode,
               action_mode=action_mode, reward_mode=reward_mode)
        for _ in range(min(batch_size, len(seeds)))
    ]

    for start in range(0, len(seeds), len(pool)):
        chunk = seeds[start:start + len(pool)]
        envs  = pool[:len(chunk)]

        obs = np.stack([env.reset(seed=int(s))[0] for env, s in zip(envs, chunk)])
        live = [True] * len(envs)
        infos: list[dict] = [{} for _ in envs]

        while any(live):
            actions = act_batch(obs)
            for i, env in enumerate(envs):
                if not live[i]:
                    continue
                o, _, terminated, truncated, info = env.step(actions[i])
                obs[i]  = o
                infos[i] = info
                live[i] = not (terminated or truncated)

        for i, _ in enumerate(chunk):
            profit = float(infos[i].get("profit", 0.0))
            profits[start + i] = profit
            regrets[start + i] = profit - float(infos[i].get("profit_max", 0.0))

    for env in pool:
        env.close()
    return profits, regrets


def summarize(profits: np.ndarray, regrets: np.ndarray) -> dict:
    """Per-cell statistics, including the SE mdp_gates compares on (§13)."""
    n  = len(profits)
    mu = float(profits.mean())
    return {
        "profit_mean": mu,
        "profit_se":   float(profits.std(ddof=1) / np.sqrt(n)),
        "regret_mean": float(regrets.mean()),
        "regret_se":   float(regrets.std(ddof=1) / np.sqrt(n)),
        "profit_var":  float(profits.var(ddof=1)),
        "semivar_d":   float(np.sum(np.maximum(mu - profits, 0.0) ** 2) / (n - 1)),
        "semivar_u":   float(np.sum(np.maximum(profits - mu, 0.0) ** 2) / (n - 1)),
    }


COLUMNS = ["stdev", "T", "lamb", "n_seeds", "profit_mean", "profit_se",
           "regret_mean", "regret_se", "profit_var", "semivar_d", "semivar_u"]


def resolve_cells(scenario_name: str) -> list[tuple[str, FnvScenario]]:
    """A name is either a scenario (§5.4) or a grid (§5.6); grids enumerate to
    fully-built cells, one leaderboard row each."""
    if scenario_name in SCENARIOS:
        return [(scenario_name, SCENARIOS[scenario_name])]
    if scenario_name in GRIDS:
        return list(GRIDS[scenario_name])
    raise SystemExit(
        f"unknown scenario/grid {scenario_name!r}; "
        f"scenarios={sorted(SCENARIOS)} grids={sorted(GRIDS)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model_path   = Path(args.model_path)
    vecnorm_path = resolve_vecnorm_path(model_path, args.vecnorm_path)
    # Confirming several checkpoints from one run (§9.7 top-k) would collide on
    # a single name, since they share the checkpoints/ parent — so a checkpoint
    # writes into the RUN dir, tagged with which checkpoint it was.
    if args.outfile:
        outfile = Path(args.outfile)
    elif model_path.parent.name == "checkpoints":
        outfile = (model_path.parent.parent
                   / f"ppo_eval_{args.scenario_name}_{model_path.stem}.tsv")
    else:
        outfile = model_path.parent / f"ppo_eval_{args.scenario_name}.tsv"
    # §8.4: the eval's console belongs with the run it describes
    tee_console(outfile.parent, "eval.log")

    cells = resolve_cells(args.scenario_name)
    model = PPO.load(str(model_path), device="cpu")
    obs_rms, clip_obs = load_obs_rms(vecnorm_path)

    def act_batch(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(
            normalize_obs(obs, obs_rms, clip_obs), deterministic=True)
        return action

    seeds = range(args.seed_base, args.seed_base + args.n_seeds)

    print(f"model     {model_path}")
    print(f"vecnorm   {vecnorm_path}  (obs_rms={'loaded' if obs_rms is not None else 'NONE'})")
    print(f"scenario  {args.scenario_name}  ({len(cells)} cell(s))")
    print(f"seeds     {args.seed_base}..{args.seed_base + args.n_seeds - 1}  (protocol block)")
    print(f"output    {outfile}")
    if obs_rms is None:
        print("WARNING: no vecnormalize.pkl found — the model was trained on "
              "normalized observations, so scores will be meaningless (§9.5).")

    header = "\t".join(COLUMNS)
    print(header)

    per_seed_fh = None
    if args.per_seed_out:
        Path(args.per_seed_out).parent.mkdir(parents=True, exist_ok=True)
        per_seed_fh = open(args.per_seed_out, "w")
        per_seed_fh.write("cell_id\tepisode_seed\tprofit\tregret\n")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()

        for cell_id, scenario in cells:
            profits, regrets = rollout_seeds(
                act_batch, scenario, seeds,
                observation_mode=args.observation_mode,
                action_mode=args.action_mode,
                reward_mode=args.reward_mode,
                batch_size=args.batch_size,
            )
            stats = summarize(profits, regrets)
            row = (
                f"{scenario.stdev}\t{scenario.T}\t{scenario.lamb}\t{args.n_seeds}\t"
                f"{stats['profit_mean']:.6f}\t{stats['profit_se']:.6f}\t"
                f"{stats['regret_mean']:.6f}\t{stats['regret_se']:.6f}\t"
                f"{stats['profit_var']:.6f}\t{stats['semivar_d']:.6f}\t{stats['semivar_u']:.6f}"
            )
            print(row, flush=True)
            # incremental (§9.4): a killed run keeps every finished cell
            f.write(row + "\n")
            f.flush()

            if per_seed_fh is not None:
                for s, p, rg in zip(seeds, profits, regrets):
                    per_seed_fh.write(f"{cell_id}\t{s}\t{p:.6f}\t{rg:.6f}\n")
                per_seed_fh.flush()

    if per_seed_fh is not None:
        per_seed_fh.close()
        print(f"per-seed profits → {args.per_seed_out}")
    print(f"\nresults saved → {outfile}")


if __name__ == "__main__":
    main()

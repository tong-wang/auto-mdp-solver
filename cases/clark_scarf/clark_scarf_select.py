"""Post-hoc checkpoint screen — the middle layer of the three-layer evaluation
(spec §9.7).

Training runs to its budget and saves ~20 checkpoints; **the terminal
checkpoint is never the deliverable** (§8.6). This script ranks those
checkpoints on a CRN block disjoint from the protocol block and emits the top-k
for confirmation.

    python clark_scarf_ppo_train.py -s n3_l2_p09 -o raw -a ship_discrete
    python clark_scarf_select.py results/n3_l2_p09/PPO_...        # <- here
    python clark_scarf_ppo_eval.py --model-path <winner> ...      # confirm

WHY THIS FILE EXISTS AT ALL, on this domain
-------------------------------------------
The train script shipped a live `CRNSelectionCallback` that scored the policy
every 100k steps on 256 episodes and saved its best as `{scenario}_ppo.zip`.
That is the protocol §9.7 replaced in v0.7.0, in terms that exclude it outright
— "no `EvalCallback`, no live selection env". Every number this domain produced
before 2026-09-14 was selected that way.

It survived because `mdp_conformance`'s `scripts.selection_protocol` detects a
violation **by class name** (`endswith("EvalCallback")`), and this one was
called something else. The check reported "no live selection callback" while
the callback ran. The gate is lexical; the rule is behavioural.

Three rules this file exists to keep:

1. **Screen scores are never quoted.** They rank; the protocol block confirms.
   The blocks are disjoint (`SELECT_SEED_OFFSET` vs the protocol block's 0), so
   the number that ranks a checkpoint is never the number that reports it.
2. **Each checkpoint is scored under its own normalizer.** `CheckpointCallback`
   saves the VecNormalize statistics beside every checkpoint; scoring an early
   checkpoint under the run's terminal statistics would rank it on observations
   it never saw.
3. **No live selection env.** The screen reads saved checkpoints after training
   has finished.

This domain MINIMIZES cost, so the ranking is ascending and the tie band opens
upward from the best.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from clark_scarf_eval_common import SELECT_SEED_OFFSET, eval_seed_block, rollout_seeds
from clark_scarf_gym import ClarkScarfEnv
from clark_scarf_scenarios import SCENARIOS

# §9.7: the screen layer's block. Cheaper than the protocol block because it
# ranks rather than reports — but large enough that the ranking is not draw
# noise, which the batched evaluator makes affordable.
DEFAULT_SCREEN_SEEDS = 2048

METRIC = "cost_total"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Screen a training run's checkpoints (spec §9.7 screen layer).")
    p.add_argument("run_dir", type=str,
                   help="training run directory (the one holding checkpoints/)")
    p.add_argument("-s", "--scenario_name", type=str, default=None,
                   help="inferred from the run directory's parent when omitted")
    p.add_argument("-o", "--observation_mode", type=str, default=None,
                   choices=list(ClarkScarfEnv.OBS_MODES),
                   help="defaults to the mode recorded in the run's args file")
    p.add_argument("-a", "--action_mode", type=str, default=None,
                   choices=list(ClarkScarfEnv.ACTION_MODES),
                   help="defaults to the mode recorded in the run's args file")
    p.add_argument("--n-seeds", type=int, default=DEFAULT_SCREEN_SEEDS)
    p.add_argument("--first-seed", type=int, default=SELECT_SEED_OFFSET,
                   help="first seed of the screen block; must stay disjoint "
                        "from the protocol block (0…n-1)")
    p.add_argument("--top-k", type=int, default=3,
                   help="checkpoints passed to confirmation (widened "
                        "automatically when leaders sit within one screen-SE)")
    p.add_argument("--batch-envs", type=int, default=64)
    p.add_argument("--outfile", type=str, default=None,
                   help="TSV of the ranking (defaults to <run_dir>/screen.tsv)")
    return p


def _step_of(path: Path) -> int:
    """Sort key: the step count CheckpointCallback encodes in the filename."""
    m = re.search(r"_(\d+)_steps", path.name)
    return int(m.group(1)) if m else -1


def find_checkpoints(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"no checkpoints/ under {run_dir}")
    ckpts = sorted((p for p in ckpt_dir.glob("*.zip")
                    if "vecnormalize" not in p.name), key=_step_of)
    if not ckpts:
        raise SystemExit(f"no checkpoint .zip files in {ckpt_dir}")
    return ckpts


def resolve_vecnorm(ckpt: Path, run_dir: Path) -> Path | None:
    """The normalizer this checkpoint was trained under.

    `CheckpointCallback(save_vecnormalize=True)` writes
    `{prefix}_vecnormalize_{steps}_steps.pkl` beside `{prefix}_{steps}_steps.zip`,
    keyed on the same step count. Falls back to the run's terminal statistics
    for runs saved before that was switched on — reported, because it makes the
    early checkpoints' scores optimistic in an unknown direction.
    """
    sibling = ckpt.parent / re.sub(r"_(\d+)_steps\.zip$",
                                   r"_vecnormalize_\1_steps.pkl", ckpt.name)
    if sibling.exists():
        return sibling
    for terminal in (run_dir / "vecnormalize.pkl", run_dir / "vecnormalize_final.pkl"):
        if terminal.exists():
            return terminal
    return None


def read_run_args(run_dir: Path) -> dict:
    f = next((p for p in run_dir.glob("*_ppo_args.txt")), None)
    if f is None:
        return {}
    return dict(l.split(": ", 1) for l in f.read_text(errors="replace").splitlines()
                if ": " in l)


def infer_scenario_name(run_dir: Path, explicit: str | None) -> str:
    """Runs live at results/{scenario_name}/{run_name}/ (§8.4)."""
    if explicit:
        return explicit
    name = run_dir.resolve().parent.name
    if name not in SCENARIOS:
        raise SystemExit(
            f"cannot infer the scenario from {run_dir} (got {name!r}); pass -s")
    return name


def widen_to_ties(ranked: list[dict], top_k: int) -> list[dict]:
    """Top-k, widened to everything within one screen-SE of the best.

    §8.6 makes k adjustable for exactly this reason: when the leaders sit inside
    one screen-SE the ranking cannot separate them, so confirmation — not the
    screen — should decide. Minimize sense, so the band opens upward.
    """
    cut = ranked[0]["mean"] + ranked[0]["se"]
    keep = [r for r in ranked if r["mean"] <= cut]
    return ranked[:top_k] if len(keep) <= top_k else keep


def score_checkpoint(ckpt: Path, vecnorm: Path | None, scenario, obs_mode: str,
                     act_mode: str, seeds: np.ndarray, batch: int) -> dict:
    model = PPO.load(str(ckpt), device="cpu")
    vn = None
    if vecnorm is not None:
        vn = VecNormalize.load(str(vecnorm), DummyVecEnv(
            [lambda: ClarkScarfEnv(scenario, observation_mode=obs_mode,
                                   action_mode=act_mode)]))
        vn.training = False

    def act_fn(obs: np.ndarray, raw_envs) -> np.ndarray:
        o = vn.normalize_obs(obs) if (vn is not None and vn.norm_obs) else obs
        action, _ = model.predict(o, deterministic=True)
        return action

    per = rollout_seeds(scenario=scenario, observation_mode=obs_mode,
                        action_mode=act_mode, seeds=seeds, act_fn=act_fn,
                        batch=min(batch, len(seeds)), progress_every=0)
    vals = per[METRIC]
    return {"mean": float(vals.mean()),
            "se": float(vals.std(ddof=1) / np.sqrt(len(vals)))}


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    rec = read_run_args(run_dir)
    scenario_name = infer_scenario_name(run_dir, args.scenario_name)
    scenario = SCENARIOS[scenario_name]
    obs_mode = args.observation_mode or rec.get("observation_mode", "raw").strip()
    act_mode = args.action_mode or rec.get("action_mode", "ship_discrete").strip()

    if args.first_seed == 0:
        raise SystemExit(
            "refusing to screen on the protocol block (--first-seed 0): the "
            "block that ranks must not be the block that reports (§9.7).")

    ckpts = find_checkpoints(run_dir)

    # A screen that scores raw observations for a policy trained normalized is
    # not a worse ranking, it is a ranking of different policies — and it still
    # prints a confident top-k. Same contract as `clark_scarf_ppo_eval.py` and
    # `clark_scarf_policy.py`: REFUSE rather than rank. (#E11: the probe was the
    # last consumer to gain this, after it read two nets back unnormalized.)
    if rec.get("norm_obs", "").strip() == "True":
        unnormalized = [c for c in ckpts if resolve_vecnorm(c, run_dir) is None]
        if unnormalized:
            raise SystemExit(
                f"[select] REFUSING: the run's args record norm_obs=True and "
                f"{len(unnormalized)}/{len(ckpts)} checkpoint(s) have no "
                f"normalizer (no per-checkpoint sibling, no vecnormalize.pkl or "
                f"vecnormalize_final.pkl in the run dir). Ranking them on raw "
                f"observations would rank policies that were never trained.")

    outfile = Path(args.outfile) if args.outfile else run_dir / "screen.tsv"
    seeds = eval_seed_block(args.n_seeds, offset=args.first_seed)

    print(f"run         {run_dir.name}")
    print(f"scenario    {scenario_name}   obs={obs_mode}  act={act_mode}")
    print(f"seeds       {args.first_seed}…{args.first_seed + args.n_seeds - 1}"
          f"  (screen block)")
    print(f"checkpoints {len(ckpts)}")
    print(f"metric      {METRIC} (minimize)")
    print("NOTE: screen scores RANK only — they are never quoted (§9.7).\n")

    ranked: list[dict] = []
    header = f"checkpoint\tsteps\t{METRIC}_mean\t{METRIC}_se\tvecnorm"
    print(header)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write(header + "\n")
        f.flush()
        for ckpt in ckpts:
            vecnorm = resolve_vecnorm(ckpt, run_dir)
            e = score_checkpoint(ckpt, vecnorm, scenario, obs_mode, act_mode,
                                 seeds, args.batch_envs)
            e["path"], e["steps"] = ckpt, _step_of(ckpt)
            e["vecnorm"] = ("per-ckpt" if vecnorm and "vecnormalize_" in vecnorm.name
                            else "terminal" if vecnorm else "none")
            ranked.append(e)
            row = (f"{ckpt.name}\t{e['steps']}\t{e['mean']:.6f}\t"
                   f"{e['se']:.6f}\t{e['vecnorm']}")
            print(row, flush=True)
            f.write(row + "\n")
            f.flush()

    terminal = [r for r in ranked if r["vecnorm"] == "terminal"]
    none_at_all = [r for r in ranked if r["vecnorm"] == "none"]
    if terminal:
        print(f"\nWARNING: {len(terminal)}/{len(ranked)} checkpoint(s) had no "
              f"per-checkpoint normalizer and were scored under the run's "
              f"TERMINAL statistics — observations they never saw, so their "
              f"ranking is not trustworthy.")
    if none_at_all:
        print(f"\nWARNING: {len(none_at_all)}/{len(ranked)} checkpoint(s) were "
              f"scored under NO normalizer at all. If the run trained "
              f"normalized this ranking is meaningless; the refusal above only "
              f"fires when the args record norm_obs=True, so an unrecorded run "
              f"reaches here.")

    ranked.sort(key=lambda r: r["mean"])
    winners = widen_to_ties(ranked, args.top_k)
    print(f"\nranking saved → {outfile}")
    print(f"\ntop-{len(winners)} for confirmation on the protocol block:")
    for i, r in enumerate(winners, 1):
        print(f"  {i}. {r['path'].name}  (screen {r['mean']:.6f} "
              f"± {r['se']:.6f}, step {r['steps']})")
    if len(winners) > args.top_k:
        print(f"  (widened from {args.top_k}: leaders sit within one screen-SE)")

    print("\nconfirm each on the protocol block, then quote the winner:")
    for r in winners:
        vecnorm = resolve_vecnorm(r["path"], run_dir)
        vn = f" --vecnorm-path {vecnorm}" if vecnorm else ""
        # -o and -a are as load-bearing as the path: an arm evaluated in a mode
        # it did not train with is silently mis-scored, never an error. And
        # --outfile is load-bearing for a top-k confirmation, because every
        # checkpoint shares checkpoints/ as its model dir and the record write
        # overwrites: without a distinct name only the last confirm survives,
        # silently, each file still looking complete.
        print(f"  python clark_scarf_ppo_eval.py --model-path {r['path']}"
              f"{vn} -s {scenario_name} -o {obs_mode} -a {act_mode}"
              f" --n-seeds 8192 --outfile {run_dir}/confirm_{r['path'].stem}.tsv")


if __name__ == "__main__":
    main()

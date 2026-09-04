"""Post-hoc three-layer checkpoint selection (spec §9.7) for ADI-flex.

The terminal checkpoint is never the deliverable. §8.6 forbids `EvalCallback`
and live selection, so a run saves a ladder of checkpoints and the winner is
chosen *after* training, in three layers:

1. **screen** — every checkpoint on a SELECTION block of CRN seeds that is
   **disjoint** from the protocol block. Disjointness is the whole point: a
   checkpoint chosen on the seeds it is then reported against is chosen on its
   own noise.
2. **top-k** — keep the best k (default 3; widen when the leaders sit inside
   one screen-SE, which this script reports so the call is visible).
3. **confirm** — re-evaluate the survivors on the protocol block and ship the
   best.

Each layer's rows are written out, so the selection is auditable rather than a
number that appeared. Usage:

    python adi_flex_ppo_select.py -s homog_L0_T2 --runs 'results/homog_L0_T2/PPO_*L1*'
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO

from adi_flex_ppo_eval import evaluate_scenario
from adi_flex_scenarios import SCENARIOS


def _obs_mode_of(run_dir: Path) -> str:
    """Read the observation mode back off the args log — never guessed."""
    args = next(run_dir.glob("*_ppo_args.txt"), None)
    if args is None:
        raise SystemExit(f"{run_dir}: no args log; cannot resolve observation_mode")
    for line in args.read_text().splitlines():
        if line.startswith("observation_mode:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"{run_dir}: args log has no observation_mode")


def _checkpoints(run_dir: Path) -> list[Path]:
    ck = sorted(run_dir.glob("checkpoints/*_steps.zip"),
                key=lambda p: int(p.stem.split("_")[-2]))
    terminal = run_dir / f"{run_dir.parent.name}_ppo_final.zip"
    if terminal.exists():
        ck.append(terminal)          # eligible, but never privileged
    return ck


def _arg_of(run_dir: Path, key: str, default: str | None = None) -> str | None:
    """One field from a run's OWN args log. The run records what it was; the
    scorer must not assume it (see `adi_flex_ppo_eval.build_env`)."""
    logs = list(run_dir.glob("*_ppo_args.txt"))
    if not logs:
        raise SystemExit(f"{run_dir}: no args log; cannot resolve {key}")
    for line in logs[0].read_text().splitlines():
        if line.startswith(f"{key}:"):
            v = line.split(":", 1)[1].strip()
            return None if v in ("None", "") else v
    return default


def _act_mode_of(run_dir: Path) -> str:
    """The run's OWN action mode, read from its args log — never assumed.

    The mode decides the observation width (a one-shot mode renders no
    within-period block) and the action space shape, so scoring a checkpoint
    under the wrong one does not merely mis-score it: SB3 refuses to attach
    VecNormalize at all, `spaces must have the same shape: (4,) != (10,)`.
    That is what a default of "seq_mask" here bought — a screen that crashed on
    every run of a new mode while still writing an (empty) select.tsv, so the
    ladder looked merely unscreened rather than broken.
    """
    logs = list(run_dir.glob("*_ppo_args.txt"))
    if not logs:
        raise SystemExit(f"{run_dir}: no args log; cannot resolve action_mode")
    for line in logs[0].read_text().splitlines():
        if line.startswith("action_mode:"):
            return line.split(":", 1)[1].strip()
    # pre-dates the field: every mode but one was added later, so the archive's
    # older runs are seq_mask by construction
    return "seq_mask"


def _score(model_path: Path, vecnorm: Path | None, scenario, n_seeds: int,
           first_seed: int, obs_mode: str, act_mode: str,
           fixed_ap: str | None = None) -> dict:
    model = MaskablePPO.load(str(model_path), device="cpu")
    return evaluate_scenario(
        model=model, vecnorm_path=vecnorm, scenario=scenario,
        n_seeds=n_seeds, first_seed=first_seed, observation_mode=obs_mode,
        action_mode=act_mode, reward_mode="neg_cost",
        fixed_order_ap=fixed_ap,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-s", "--scenario_name", default="homog_L0_T2",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--runs", required=True, help="glob over run directories")
    p.add_argument("--screen-seeds",   type=int, default=2048)
    p.add_argument("--screen-first",   type=int, default=100_000)
    p.add_argument("--protocol-seeds", type=int, default=8192)
    p.add_argument("--protocol-first", type=int, default=0)
    p.add_argument("--topk", type=int, default=3)
    a = p.parse_args()

    if a.screen_first < a.protocol_first + a.protocol_seeds:
        raise SystemExit(
            f"selection block [{a.screen_first}, {a.screen_first + a.screen_seeds}) "
            f"overlaps the protocol block "
            f"[{a.protocol_first}, {a.protocol_first + a.protocol_seeds}) — §9.7 "
            f"requires them disjoint")

    scenario = SCENARIOS[a.scenario_name]
    for run_dir in sorted(Path(d) for d in glob.glob(a.runs)):
        if not run_dir.is_dir():
            continue
        obs_mode = _obs_mode_of(run_dir)
        act_mode = _act_mode_of(run_dir)
        fixed_ap = _arg_of(run_dir, "fixed_order_ap")
        vecnorm  = run_dir / "vecnormalize.pkl"
        vecnorm  = vecnorm if vecnorm.exists() else None
        cks = _checkpoints(run_dir)
        if not cks:
            print(f"{run_dir.name}: no checkpoints and no terminal model — skipped")
            continue

        print(f"\n=== {run_dir.name}  obs={obs_mode}  act={act_mode}"
              + (f"  ap-fixed-order" if fixed_ap else "")
              + f"  {len(cks)} checkpoints")
        out = run_dir / "select.tsv"
        rows = []
        with open(out, "w") as f:
            f.write("layer\tcheckpoint\tn_seeds\tfirst_seed\treward_mean\treward_var\n")
            for ck in cks:
                st = _score(ck, vecnorm, scenario, a.screen_seeds, a.screen_first,
                            obs_mode, act_mode, fixed_ap)
                rows.append((ck, st["reward_mean"], st["reward_var"]))
                f.write(f"screen\t{ck.name}\t{a.screen_seeds}\t{a.screen_first}\t"
                        f"{st['reward_mean']:.6f}\t{st['reward_var']:.6f}\n")
                f.flush()
                print(f"  screen {ck.name:<44} {st['reward_mean']:12.4f}")

            rows.sort(key=lambda r: -r[1])                 # reward: higher is better
            se = math.sqrt(rows[0][2] / a.screen_seeds)
            k = a.topk
            while k < len(rows) and rows[k][1] > rows[0][1] - se:
                k += 1                                    # widen: inside one screen-SE
            if k > a.topk:
                print(f"  top-k widened {a.topk} -> {k}: leaders within one screen-SE ({se:.3f})")

            best = None
            for ck, _, _ in rows[:k]:
                st = _score(ck, vecnorm, scenario, a.protocol_seeds, a.protocol_first,
                            obs_mode, act_mode, fixed_ap)
                f.write(f"confirm\t{ck.name}\t{a.protocol_seeds}\t{a.protocol_first}\t"
                        f"{st['reward_mean']:.6f}\t{st['reward_var']:.6f}\n")
                f.flush()
                print(f"  confirm {ck.name:<43} {st['reward_mean']:12.4f}")
                if best is None or st["reward_mean"] > best[1]["reward_mean"]:
                    best = (ck, st)
            f.write(f"shipped\t{best[0].name}\t{a.protocol_seeds}\t{a.protocol_first}\t"
                    f"{best[1]['reward_mean']:.6f}\t{best[1]['reward_var']:.6f}\n")
        pse = math.sqrt(best[1]["reward_var"] / a.protocol_seeds)
        print(f"  SHIPPED {best[0].name}  {best[1]['reward_mean']:.4f} ±{pse:.4f}  -> {out}")


if __name__ == "__main__":
    main()

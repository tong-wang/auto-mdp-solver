"""Read a PPO policy back as a skip-then-accept-record rule (spec §14)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from secretary_benchmark_threshold import optimal_skip_count
from secretary_gym import SecretaryEnv
from secretary_scenarios import SCENARIOS


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe a trained secretary policy.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vecnorm-path", default=None)
    parser.add_argument("-s", "--scenario_name", default="standard", choices=SCENARIOS)
    return parser


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def reference_action(period: int, relative_rank: int, skip: int, horizon: int) -> int:
    """Classical policy action; final forced selection is represented as reject."""
    if period == horizon - 1:
        return 0
    return int(period >= skip and relative_rank == 1)


def _load_normalizer(path: Path, scenario) -> VecNormalize:
    dummy = DummyVecEnv([lambda: SecretaryEnv(scenario=scenario)])
    normalizer = VecNormalize.load(path, dummy)
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def _protocol_outcome_agreement(run_dir: Path, domain_dir: Path) -> dict[str, object]:
    net = run_dir / "ppo_eval_standard.seeds.tsv"
    ref = domain_dir / "results/standard/benchmark/benchmark_dp_eval_standard.seeds.tsv"
    if not net.exists() or not ref.exists():
        return {"available": False}
    return {
        "available": True,
        "episode_rows_identical": net.read_bytes() == ref.read_bytes(),
        "n_seeds": sum(1 for _ in net.open()) - 1,
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    run_dir = model_path.parent
    domain_dir = Path(__file__).resolve().parent
    scenario = SCENARIOS[args.scenario_name]
    vecnorm_path = Path(args.vecnorm_path).resolve() if args.vecnorm_path else run_dir / "vecnormalize.pkl"
    model = PPO.load(model_path, device="cpu")
    normalizer = _load_normalizer(vecnorm_path, scenario)

    observations: list[list[float]] = []
    coordinates: list[tuple[int, int]] = []
    for period in range(scenario.n_candidates):
        for relative_rank in range(1, period + 2):
            observations.append([scenario.n_candidates - period, relative_rank])
            coordinates.append((period, relative_rank))
    raw = np.asarray(observations, dtype=np.float32)
    normalized = normalizer.normalize_obs(raw.copy())
    actions, _ = model.predict(normalized, deterministic=True)
    with torch.no_grad():
        tensor = torch.as_tensor(normalized, device=model.device)
        probabilities = model.policy.get_distribution(tensor).distribution.probs[:, 1]
        accept_prob = probabilities.cpu().numpy()
    normalizer.close()

    reference_skip = optimal_skip_count(scenario.n_candidates)
    record_actions = {
        period: int(actions[index])
        for index, (period, rank) in enumerate(coordinates)
        if rank == 1 and period < scenario.n_candidates - 1
    }
    candidates = [
        cutoff for cutoff in range(scenario.n_candidates - 1)
        if all(record_actions[p] == int(p >= cutoff) for p in record_actions)
    ]
    fitted_skip = candidates[0] if candidates else min(
        range(scenario.n_candidates - 1),
        key=lambda cutoff: sum(
            record_actions[p] != int(p >= cutoff) for p in record_actions
        ),
    )

    probe_dir = run_dir / "probe"
    interpret_dir = run_dir / "interpret"
    probe_dir.mkdir(exist_ok=True)
    interpret_dir.mkdir(exist_ok=True)
    surface = probe_dir / "action_surface.tsv"
    with surface.open("w") as stream:
        stream.write("period\tposition\ttime_to_go\trelative_rank\taccept_probability\tnet_action\treference_action\n")
        for index, (period, rank) in enumerate(coordinates):
            stream.write(
                f"{period}\t{period + 1}\t{scenario.n_candidates - period}\t{rank}\t"
                f"{accept_prob[index]:.10f}\t{int(actions[index])}\t"
                f"{reference_action(period, rank, reference_skip, scenario.n_candidates)}\n"
            )

    judged = [i for i, (period, _) in enumerate(coordinates) if period < scenario.n_candidates - 1]
    reference = np.asarray([
        reference_action(period, rank, reference_skip, scenario.n_candidates)
        for period, rank in coordinates
    ])
    record_indices = [
        i for i, (period, rank) in enumerate(coordinates)
        if rank == 1 and period < scenario.n_candidates - 1
    ]
    nonrecord_indices = [
        i for i, (period, rank) in enumerate(coordinates)
        if rank > 1 and period < scenario.n_candidates - 1
    ]
    summary = {
        "scenario": args.scenario_name,
        "model_path": str(model_path),
        "reference_skip_count": reference_skip,
        "fitted_skip_count": fitted_skip,
        "exact_threshold_form": bool(candidates),
        "action_agreement_all_valid_nonterminal": float(np.mean(actions[judged] == reference[judged])),
        "action_agreement_record_states": float(np.mean(actions[record_indices] == reference[record_indices])),
        "nonrecord_acceptance_rate": float(np.mean(actions[nonrecord_indices])),
        "record_transition_count": int(sum(
            record_actions[p] != record_actions[p - 1]
            for p in range(1, scenario.n_candidates - 1)
        )),
        "protocol_outcome_validation": _protocol_outcome_agreement(run_dir, domain_dir),
        "feature_sensitivity": {
            "time_to_go_at_relative_rank_1": {
                "action_range": int(max(record_actions.values()) - min(record_actions.values())),
                "transition_periods": [
                    p for p in range(1, scenario.n_candidates - 1)
                    if record_actions[p] != record_actions[p - 1]
                ],
            },
            "relative_rank": {
                "nonrecord_acceptance_rate": float(np.mean(actions[nonrecord_indices])),
                "periods_with_any_nonrecord_accept": sorted({
                    period for i, (period, rank) in enumerate(coordinates)
                    if period < scenario.n_candidates - 1 and rank > 1 and int(actions[i]) == 1
                }),
            },
        },
    }
    (interpret_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (interpret_dir / "fitted_rule.json").write_text(json.dumps({
        "policy_class": "skip_then_accept_record",
        "fitted_skip_count": fitted_skip,
        "horizon": scenario.n_candidates,
        "source_model": str(model_path),
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"surface -> {surface}")


if __name__ == "__main__":
    main()

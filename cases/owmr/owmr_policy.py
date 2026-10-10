"""Deployable policy for owmr (spec §12) — the campaign's shipped artifact behind a small interface.

Two policies, one interface (``act(obs)`` / ``decide(state)`` -> ``(order, ship)``):

* ``OwmrPolicy`` — the trained network: PPO with the categorical-retention head (`CatKeepPolicy`,
  action mode ``order_catkeep``) trained with the coordinate vine on the `hicv` cell, seed 2,
  checkpoint 10M (ESCALATION #E33; 2,677.78 on the protocol block, −91.98 paired vs the LB
  heuristic, gate PASS). The wrapper bundles the three things of the model's I/O contract: the SB3
  model on CPU, the VecNormalize observation statistics (applied only if the run normalized
  observations — this recipe did not, and the saved pickle says so), and the gym's own action
  transform (`owmr_gym.catkeep_shares` and `clamp_shipments`, imported, never re-implemented).
* ``FittedRulePolicy`` — the readback of that network as two scalars on the LB heuristic
  (#E33 addenda 3–4; `INTERPRET.md`): order up to y0* + 2.4 and ship each retailer only up to
  0.75 of its newsvendor target, keeping the rest at the warehouse. 2,676.40 on the protocol block,
  −1.37 ± 0.57 paired vs the network — seed-free, no model file, the same arithmetic as the probe
  that scored it (`owmr_keep_rule_probe.scaled_target_decision`).

**Observation contract** (`owmr_gym.observation`, mode ``raw``, the only mode the shipped artifact
was trained on) — one float32 vector of 1 + 1 + l0 + N + N·l_rt = 15 features for the shipped cell
(N = 5 retailers, l0 = 3, l_rt = 1), read on the POST-RECEIPT state of a period (deliveries landed,
nothing ordered or shipped yet, demand not yet realized), in this order and in these units:

    [0]      time_to_go     periods remaining, T − t  (T = 100)
    [1]      wh_stock       warehouse on-hand, units (≥ 0)
    [2:5]    wh_pipe        warehouse pipeline, oldest slot first, units; the last slot reads 0 post-receipt
    [5:10]   rt_stock       each retailer's net stock, units (negative = backlog)
    [10:15]  rt_pipe        each retailer's pipeline (N rows × l_rt), units; reads 0 post-receipt at l_rt = 1

``decide(state)`` renders this vector itself from an ``owmr_mdp.OwmrState`` and is the recommended
entry point; ``act(obs)`` takes the vector for callers that already hold it. The decision is
``(order, ship)``: one non-negative order quantity and one non-negative shipment per retailer whose
sum never exceeds on-hand (the categorical head's decode guarantees it; the MDP's scale-down never
fires). Deterministic: the network's argmax action.

This file is not standalone (spec §12): it imports the head, the gym's transform, the MDP, the
scenarios and the heuristic from its siblings. Artifact paths default to the shipped run under
``results/`` (gitignored; the README's commands reproduce it).

Smoke test (``python owmr_policy.py``): replays both policies against the raw ``owmr_mdp`` loop — no
gym — for a few protocol seeds and checks each episode's cost against the eval records that the
README quotes. The rule agrees to float precision; the network to ~1e-4 in episode cost, because
the evaluator predicts in batches of 64 and torch's batched matmul rounds differently from a
single-row one — the wrapper's decode is the gym's, the residual is arithmetic (checked in-line).
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np

import owmr_mdp as mdp
from owmr_gym import LOGIT_BOX, catkeep_shares, clamp_shipments, keep_menu, observation
from owmr_scenarios import SCENARIOS

HERE = Path(__file__).resolve().parent
SHIPPED_SCENARIO = "hicv"
SHIPPED_RUN_GLOB = "results/hicv/PPO_*seed2*E33catkeepvine"
SHIPPED_CHECKPOINT = "ppo_owmr_10000000_steps"
SHIPPED_RULE = {"alpha": 0.75, "y0": 37.0}


def default_artifact() -> tuple[Path, Path]:
    """The shipped run's checkpoint and its VecNormalize pickle (results/ is gitignored)."""
    runs = sorted(glob.glob(str(HERE / SHIPPED_RUN_GLOB)))
    if not runs:
        raise FileNotFoundError(f"no shipped run under {SHIPPED_RUN_GLOB}; reproduce it with the README's commands")
    ck = Path(runs[0]) / "checkpoints"
    return ck / f"{SHIPPED_CHECKPOINT}.zip", ck / f"{SHIPPED_CHECKPOINT.replace('ppo_owmr_', 'ppo_owmr_vecnormalize_')}.pkl"


class OwmrPolicy:
    """The trained artifact: model + observation statistics + the gym's action transform."""

    def __init__(self, model_path: str | Path | None = None, vecnorm_path: str | Path | None = None,
                 action_mode: str = "order_catkeep", observation_mode: str = "raw",
                 scenario: str = SHIPPED_SCENARIO) -> None:
        from stable_baselines3 import PPO
        import owmr_catkeep_policy  # noqa: F401  registers CatKeepPolicy for PPO.load
        import owmr_dirichlet_policy  # noqa: F401  registers DirichletPolicy for the Dirichlet arms

        if model_path is None:
            model_path, default_vn = default_artifact()
            vecnorm_path = vecnorm_path or default_vn
        self.model_path = Path(model_path)
        self.vecnorm_path = Path(vecnorm_path) if vecnorm_path else self.model_path.parent / "vecnormalize.pkl"
        assert action_mode in ("order_catkeep", "order_catkeep_fine"), (
            "this wrapper decodes the categorical-retention head; other modes decode through owmr_gym.OwmrEnv.decode_action")
        assert observation_mode == "raw", "the shipped artifact was trained on the raw observation mode"
        self.action_mode, self.observation_mode = action_mode, observation_mode
        self.scenario = SCENARIOS[scenario]
        self.model = PPO.load(str(self.model_path), device="cpu")
        # VecNormalize statistics: mandatory when the run normalized observations (spec 12); this
        # recipe (h5) trained with norm_obs off, and the pickle records that — read the flag from
        # __dict__ (an unpickled wrapper's __getattr__ recurses on a missing attribute).
        self.obs_rms, self.clip_obs, self.epsilon = None, 10.0, 1e-8
        with open(self.vecnorm_path, "rb") as fh:
            vn = pickle.load(fh)
        if vn.__dict__.get("norm_obs", True):
            self.obs_rms, self.clip_obs, self.epsilon = vn.obs_rms, float(vn.clip_obs), float(vn.epsilon)
        self.menu = keep_menu(action_mode)
        self.order_max = float(self.scenario.order_max)

    # -- the interface ------------------------------------------------------
    def raw_action(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.obs_rms is not None:
            x = np.clip((x - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
        a, _ = self.model.predict(x.astype(np.float32), deterministic=True)
        return np.asarray(a, dtype=np.float64).reshape(-1)

    def act(self, obs: np.ndarray) -> tuple[float, list[float]]:
        """Observation vector (contract above) -> (order, ship), decoded exactly as the gym's step does."""
        a = self.raw_action(obs)
        onhand = max(0.0, float(np.asarray(obs, dtype=np.float64).reshape(-1)[1]))
        order = float(np.clip(a[0], 0.0, self.order_max))
        shares, _kept = catkeep_shares(np.concatenate([[a[1]], np.clip(a[2:], -LOGIT_BOX, LOGIT_BOX)]), self.menu)
        return order, clamp_shipments([float(q) * onhand for q in shares], self.order_max)   # the gym's own bound, one helper

    def decide(self, state: mdp.OwmrState) -> tuple[float, list[float]]:
        """Post-receipt MDP state -> (order, ship); renders the observation itself."""
        return self.act(observation(self.scenario, state, self.observation_mode, [0.0] * self.scenario.n_retailers))


class FittedRulePolicy:
    """The network read back as two scalars on the LB heuristic (INTERPRET.md): no model file."""

    def __init__(self, scenario: str = SHIPPED_SCENARIO, alpha: float = SHIPPED_RULE["alpha"],
                 y0: float | None = SHIPPED_RULE["y0"]) -> None:
        from owmr_benchmark_lb_heuristic import build as build_heuristic
        from owmr_keep_rule_probe import scaled_target_decision

        self.scenario = SCENARIOS[scenario]
        self.heuristic = build_heuristic(scenario, HERE)
        self.alpha, self.y0 = float(alpha), (None if y0 is None else float(y0))
        self._rule = scaled_target_decision

    def decide(self, state: mdp.OwmrState) -> tuple[float, list[float]]:
        order, ship = self._rule(self.heuristic, state, self.alpha, self.y0)
        return order, [float(q) for q in ship]

    def act(self, obs: np.ndarray) -> tuple[float, list[float]]:
        raise NotImplementedError("the rule reads inventory positions off the MDP state; use decide(state)")


# -- smoke test against the raw MDP loop (spec 12) -----------------------------------------------

def replay(policy, scenario, seed: int) -> float:
    """One episode on the raw owmr_mdp loop — init_state -> (advance1 -> decide -> advance2)* — total cost."""
    state, _ = mdp.init_state(scenario, seed)
    total = 0.0
    while not state.terminated:
        pre = mdp.advance1(scenario, state)
        order, ship = policy.decide(pre)
        state, info = mdp.advance2(scenario, pre, order, ship)
        total += float(info["cost"]["total"])
    return total


def _record_costs(path: Path, seeds: list[int]) -> dict[int, float]:
    out = {}
    for line in open(path):
        if line.startswith("#") or line.startswith("seed"):
            continue
        parts = line.rstrip("\n").split("\t")
        if int(parts[0]) in seeds:
            out[int(parts[0])] = float(parts[1])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="owmr deployable policy — smoke test on the raw MDP loop")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--model-path", default=None)
    p.add_argument("--vecnorm-path", default=None)
    args = p.parse_args()
    scenario = SCENARIOS[SHIPPED_SCENARIO]
    seeds = list(range(args.n_episodes))
    net = OwmrPolicy(args.model_path, args.vecnorm_path)
    rule = FittedRulePolicy()
    print(f"[policy] {net.model_path.name}  obs stats {'applied' if net.obs_rms is not None else 'none (run trained with norm_obs off)'}")
    net_ref = _record_costs(net.model_path.parent.parent / f"confirm_{SHIPPED_CHECKPOINT}.seeds.tsv", seeds) \
        if (net.model_path.parent.parent / f"confirm_{SHIPPED_CHECKPOINT}.seeds.tsv").exists() else {}
    rule_ref_path = HERE / "results" / SHIPPED_SCENARIO / "benchmark" / "keep_rule" / f"keep_rule_a{SHIPPED_RULE['alpha']:g}_y{SHIPPED_RULE['y0']:g}.seeds.tsv"
    rule_ref = _record_costs(rule_ref_path, seeds) if rule_ref_path.exists() else {}
    print(f"{'seed':>4s} {'network cost':>13s} {'eval record':>12s} {'rule cost':>10s} {'eval record':>12s}")
    worst_net = worst_rule = 0.0
    for s in seeds:
        c_net, c_rule = replay(net, scenario, s), replay(rule, scenario, s)
        r_net, r_rule = net_ref.get(s), rule_ref.get(s)
        if r_net is not None: worst_net = max(worst_net, abs(c_net - r_net))
        if r_rule is not None: worst_rule = max(worst_rule, abs(c_rule - r_rule))
        print(f"{s:4d} {c_net:13.4f} {('%.4f' % r_net) if r_net is not None else '—':>12s} {c_rule:10.4f} {('%.4f' % r_rule) if r_rule is not None else '—':>12s}")
    if net_ref:
        print(f"network vs its confirm record: max |Δ| = {worst_net:.2e}  ({'OK' if worst_net < 1e-3 else 'MISMATCH'}; "
              "single-row vs batched predict rounds at ~1e-4 of episode cost)")
    if rule_ref:
        print(f"rule vs its sweep record:      max |Δ| = {worst_rule:.2e}  ({'OK' if worst_rule < 1e-3 else 'MISMATCH'})")
    if (net_ref and worst_net >= 1e-3) or (rule_ref and worst_rule >= 1e-3):
        raise SystemExit("SMOKE FAIL: the wrapper does not reproduce the eval-time behaviour")
    print("SMOKE OK")


if __name__ == "__main__":
    main()

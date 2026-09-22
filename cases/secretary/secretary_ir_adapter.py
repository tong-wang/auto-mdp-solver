"""Bit-exact differential adapter for the ``secretary`` domain."""

from __future__ import annotations

from pathlib import Path

from mdp_ir.differential import DomainAdapter, _import_domain
from mdp_ir.schema import MdpIR


def make_adapter(
    ir: MdpIR,
    instance: str | None = None,
    seed_salt: int = 0,
    domain_dir: Path | None = None,
) -> DomainAdapter:
    """Build the IR-resolved setting and drive ``init_state``/``advance``."""
    scen, mdp = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["secretary_scenarios", "secretary_mdp"],
    )
    constants = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance:
        constants.update(ir.mdp.scenario.instances[instance])
    scenario = scen.SecretaryScenario(
        n_candidates=int(constants["n_candidates"]),
        seed_salt=int(seed_salt),
        scenario_name="standard",
        desc="IR differential source",
    )

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        state, _ = mdp.init_state(scenario, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            accept = int(acts["accept"])
            state, info = mdp.advance(scenario, state, accept)
            rows.append({
                "t": info["action_period"],
                "accept": accept,
                "decision_relative_rank": info["decision_relative_rank"],
                "candidate_rank": info["candidate_rank"],
                "selected_rank": info["selected_rank"],
                "relative_rank": state.relative_rank,
                "selected": state.selected,
                "arrival_order": list(state.arrival_order),
                **info["outcome"],
                "reward": info["outcome"]["total"],
            })
        return rows

    return run_episode

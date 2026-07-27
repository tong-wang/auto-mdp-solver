"""Differential adapter for the ``inv_single`` domain (portable-domain
contract: lives with the domain, discovered from the IR file's directory by
``mdp_ir.differential.load_adapter_factory``)."""

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
    """Builds the ``InvSingleScenario`` from the IR's scenario constants and
    drives the ``init_state`` → ``advance1`` → ``advance2`` loop."""
    unc, scen, mdp = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["inv_single_uncertainty", "inv_single_scenarios", "inv_single_mdp"],
    )

    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])

    # the demand slot's selected candidate (catalog model) maps to the
    # latent-bearing generator that owns the world latent; both sides draw it
    # on the meta branch at the slot's stream_id, so per-episode demand
    # distributions match bit-for-bit
    demand_src = next(s for s in ir.mdp.uncertainty_sources if s.name == "demand")
    leadtime_src = next(s for s in ir.mdp.uncertainty_sources if s.name == "leadtime")
    if demand_src.distribution.family == "poisson":
        demand = unc.LatentPoissonDemand(
            alpha=consts["demand_alpha"],
            beta=consts["demand_beta"],
            source_id=demand_src.stream_id,
        )
    else:
        demand = unc.LatentDiscreteDemand(
            support_size=consts["demand_support_size"],
            support_low=consts["demand_support_low"],
            support_high=consts["demand_support_high"],
            source_id=demand_src.stream_id,
        )
    source = scen.InvSingleScenarioSource(scen.InvSingleScenario(
        scenario_name=f"ir_differential_{instance or 'base'}",
        horizon=ir.mdp.horizon_T(instance),
        demand=demand,
        leadtime=unc.DiscreteLeadtime(
            values=consts["leadtime_values"],
            probabilities=consts["leadtime_probs"],
            source_id=leadtime_src.stream_id,
        ),
        holding_cost=consts["h"],
        shortage_cost=consts["b"],
        order_cost_linear=consts["c"],
        order_cost_fixed=consts["K"],
        allow_backlog=consts["allow_backlog"],
        event_sequence=tuple(ir.mdp.dynamics.event_sequence),
        seed_salt=seed_salt,
    ))

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        scenario = source(episode_seed)
        state, _ = mdp.init_state(scenario, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            state1 = mdp.advance1(scenario, state)
            state, info = mdp.advance2(scenario, state1, order=acts["order"])
            rows.append({
                "t": info["action_period"],
                "order": info["order"],
                "demand": info["demand"],
                "received": info["received"],
                "leadtime": info["leadtime"],
                "lost_sales": info["lost_sales"],
                "action_period": info["action_period"],
                "inventory": state.inventory,
                "pipeline": list(state.pipeline),
                **info["cost"],   # holding, shortage, order_fixed, order_variable, total
            })
        return rows

    return run_episode

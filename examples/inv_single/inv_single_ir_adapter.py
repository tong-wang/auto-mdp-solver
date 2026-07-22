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

    # world-latent sampler applicable to this instance (spec §5.2): the domain
    # sampler shares the IR sampler's substream, so both sides realize the
    # same per-episode demand distribution bit-for-bit
    ir_sampler = next(
        (s for s in ir.mdp.scenario.samplers
         if not s.instances or (instance or "") in s.instances),
        None,
    )
    source = scen.InvSingleEpisodeDemandSampler(
        scenario_name=f"ir_differential_{instance or 'base'}",
        horizon=ir.mdp.horizon_T(instance),
        leadtime=unc.DiscreteLeadtime(
            values=consts["leadtime_values"],
            probabilities=consts["leadtime_probs"],
        ),
        holding_cost=consts["h"],
        shortage_cost=consts["b"],
        order_cost_linear=consts["c"],
        order_cost_fixed=consts["K"],
        allow_backlog=consts["allow_backlog"],
        event_sequence=tuple(ir.mdp.dynamics.event_sequence),
        support_size=consts["demand_support_size"],
        support_low=consts["demand_support_low"],
        support_high=consts["demand_support_high"],
        substream_id=ir_sampler.substream_id,
        seed_salt=seed_salt,
    )

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

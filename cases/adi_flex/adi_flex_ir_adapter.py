"""Differential adapter for the ``adi_flex`` domain (portable-domain contract:
lives with the domain, discovered from the IR file's directory by
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
    """Builds the ``AdiFlexScenario`` from the IR's scenario constants and
    drives the ``init_state`` -> ``advance1`` -> ``advance2`` loop."""
    unc, scen, mdp = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["adi_flex_uncertainty", "adi_flex_scenarios", "adi_flex_mdp"],
    )

    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])

    # source ids must match the IR's uncertainty_sources, or the two sides draw
    # from different branch-1 sources and diverge for reasons unrelated to the
    # dynamics
    sources = {s.name: s.stream_id for s in ir.mdp.uncertainty_sources}

    scenario = scen.AdiFlexScenario(
        scenario_name=f"ir_differential_{instance or 'base'}",
        horizon=ir.mdp.horizon_T(instance),
        demand_now=unc.PoissonDemand(
            rate=consts["lambda_now"], source_id=sources["demand_now"],
        ),
        demand_next=unc.PoissonDemand(
            rate=consts["lambda_next"], source_id=sources["demand_next"],
        ),
        demand_later=unc.PoissonDemand(
            rate=consts["lambda_later"], source_id=sources["demand_later"],
        ),
        holding_cost=consts["holding_cost"],
        backorder_cost=consts["backorder_cost"],
        order_cost_fixed=consts["fixed_order_cost"],
        supply_leadtime=consts["supply_leadtime"],
        demand_window=consts["demand_window"],
        seed_salt=seed_salt,
    )

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        state, _ = mdp.init_state(scenario, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            state1 = mdp.advance1(scenario, state)
            state, info = mdp.advance2(
                scenario, state1,
                order_quantity=acts["order_quantity"],
                hold_back=acts["hold_back"],
            )
            rows.append({
                "t": info["action_period"],
                "order_quantity": info["order_quantity"],
                "hold_back": info["hold_back"],
                "demand_now": info["demand_now"],
                "demand_next": info["demand_next"],
                "demand_later": info["demand_later"],
                "region": info["region"],
                "action_period": info["action_period"],
                "inventory": state.inventory,
                "due_now": state.due_now,
                "due_next": state.due_next,
                **info["cost"],   # order_fixed, holding, shortage, total
            })
        return rows

    return run_episode

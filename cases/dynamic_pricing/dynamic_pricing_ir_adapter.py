"""Differential adapter for the ``dynamic_pricing`` domain (portable-domain
contract — lives with the domain, discovered from the IR file's directory by
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
    """Builds the ``DynamicPricingScenario`` from the IR's scenario constants
    and drives the ``init_state`` → ``advance`` loop."""
    (scen, mdp) = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["dynamic_pricing_scenarios", "dynamic_pricing_mdp"],
    )

    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])

    scenario = scen.DynamicPricingScenario(
        scenario_name=f"ir_differential_{instance or 'base'}",
        horizon=ir.mdp.horizon_T(instance),
        n0=consts["n0"],
        a=consts["a"],
        alpha=consts["alpha"],
        dt=consts["dt"],
        q=consts["q"],
        allow_backlog=consts["allow_backlog"],
        allow_reorder=consts["allow_reorder"],
        seed_salt=seed_salt,
    )

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        state, _ = mdp.init_state(scenario, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            state, info = mdp.advance(scenario, state, acts["price"])
            rows.append({
                "t": info["action_period"],
                "price": info["price"],
                "arrivals": info["arrivals"],
                "units_sold": info["units_sold"],
                "lost_demand": info["lost_demand"],
                "intensity": info["intensity"],
                "inventory": state.inventory,
                **info["revenue"],   # sales, salvage, total
            })
        return rows

    return run_episode

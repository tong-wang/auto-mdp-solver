"""Differential adapter for the ``fnv`` domain (portable-domain contract —
lives with the domain, discovered from the IR file's directory by
``mdp_ir.differential.load_adapter_factory``).

Builds an ``FnvScenario`` from the IR's scenario constants and drives the
``init_state`` → ``advance`` loop, vouching for the per-period fields the
interpreter also produces (t, order, cost, inventory, information, total_cost,
plus terminal demand/sales/revenue)."""

from __future__ import annotations

from pathlib import Path

from mdp_ir.differential import DomainAdapter, _import_domain
from mdp_ir.schema import MdpIR


def make_adapter(
    ir: MdpIR,
    instance: str | None = None,
    seed_salt: int = 1,
    domain_dir: Path | None = None,
) -> DomainAdapter:
    (scen, mdp) = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["fnv_scenarios", "fnv_mdp"],
    )

    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])

    # The IR carries `t_last` for the paper's epoch time T (renamed to avoid the
    # reserved horizon symbol) and models the additive MMFE case; the domain
    # re-derives signal_stdevs from stdev/t_last/N so the baked IR schedule and
    # the scenario's schedule agree bit-for-bit.
    scenario = scen.FnvScenario(
        scenario_name=f"ir_differential_{instance or 'base'}",
        stdev=consts["stdev"],
        T=consts["t_last"],
        lamb=consts["lamb"],
        N=consts["N"],
        r=consts["r"],
        c1=consts["c1"],
        mu=consts["mu"],
        mmfe_mode="additive",
        seed_salt=seed_salt,
    )

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        state, _ = mdp.init_state(scenario, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            state, info = mdp.advance(scenario, state, acts["order"])
            row = {
                "t": info["action_period"],
                "order": info["decision"],
                "cost": info["cost"],
                "inventory": state.inventory,
                "information": state.information,
                "total_cost": info["total_cost"],
            }
            if state.terminated:
                row["demand"] = info["demand"]
                row["sales"] = info["sales"]
                row["revenue"] = info["revenue"]
            rows.append(row)
        return rows

    return run_episode

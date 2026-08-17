"""Differential adapter for the ``clark_scarf`` domain (portable-domain
contract: lives with the domain, discovered from the IR file's directory by
``mdp_ir.differential.load_adapter_factory``).

Builds a ``ClarkScarfScenario`` from the IR's own scenario constants and drives
the ``init_state`` -> ``advance1`` -> ``advance2`` loop, emitting rows keyed
exactly as the interpreter names them. Dispatch is by the resolved IR's
**selection**: the shipped ``poisson`` candidate maps to ``PoissonDemand``; any
other candidate — a family appended later — falls back to the ``FamilyDemand``
bridge, so a new family differentially verifies with zero adapter edits.
"""

from __future__ import annotations

from pathlib import Path

from mdp_ir.differential import DomainAdapter, _import_domain
from mdp_ir.schema import MdpIR


def _sampler_for(samplers, source):
    """The slot's desugared latent sampler (substream = slot stream), if any.

    Always ``None`` for the shipped candidate: Clark & Scarf hold the demand
    distribution fixed across the horizon, so this domain has no world latents.
    """
    return next((s for s in samplers if s.substream_id == source.stream_id), None)


def make_adapter(
    ir: MdpIR,
    instance: str | None = None,
    seed_salt: int = 0,
    domain_dir: Path | None = None,
) -> DomainAdapter:
    unc, scen, mdp = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["clark_scarf_uncertainty", "clark_scarf_scenarios", "clark_scarf_mdp"],
    )

    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])

    demand_src = next(s for s in ir.mdp.uncertainty_sources if s.name == "demand")
    sel = (ir.selection or {}).get("demand", "poisson")
    if sel == "poisson":
        demand = unc.PoissonDemand(
            rate=consts["demand_mean"], source_id=demand_src.stream_id
        )
    else:  # appended candidate: generic family runtime, zero new domain code
        demand = unc.FamilyDemand.from_parts(
            demand_src, _sampler_for(ir.mdp.scenario.samplers, demand_src), consts
        )

    scenario = scen.ClarkScarfScenario(
        scenario_name=f"ir_differential_{instance or 'base'}",
        horizon=ir.mdp.horizon_T(instance),
        n_echelons=int(consts["n_echelons"]),
        leadtime=int(consts["leadtime"]),
        demand=demand,
        h_install=tuple(consts["h_install"]),
        p_short=float(consts["p_short"]),
        ship_max=float(consts["ship_max"]),
        seed_salt=seed_salt,
    )

    # one vector decision since F9 (upstream #33/#36): the interpreter draws a
    # list of length n_echelons, so the adapter reads it as one name
    n_ech = int(consts["n_echelons"])

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        sc = scenario(episode_seed) if callable(scenario) else scenario
        state, _ = mdp.init_state(sc, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            state1 = mdp.advance1(sc, state)
            raw = acts["ship"]
            ship = [float(x) for x in (raw if isinstance(raw, list) else [raw] * n_ech)]
            state, info = mdp.advance2(sc, state1, ship)
            rows.append({
                "t": info["action_period"],
                "ship": list(ship),
                "demand": info["demand"],
                "arrived": list(info["arrived"]),
                "shipped": list(info["shipped"]),
                "stock": list(state.stock),
                "pipe": [list(row) for row in state.pipeline],
                **info["cost"],   # holding, shortage, total
            })
        return rows

    return run_episode

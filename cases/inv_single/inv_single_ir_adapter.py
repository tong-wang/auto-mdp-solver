"""Differential adapter for the ``inv_single`` domain (portable-domain
contract: lives with the domain, discovered from the IR file's directory by
``mdp_ir.differential.load_adapter_factory``).

Dispatch is by the resolved IR's **selection** (catalog model): the shipped
candidates map to their hand-written latent generators; any other candidate
— a freshly appended family — falls back to the ``Family{Source}`` bridges
(``mdp_ir.runtime.FamilyGenerator``), so a new family differentially
verifies with zero adapter edits. ``instance`` may also name a mixture: the
adapter then builds the ``InvSingleMixtureSampler`` twin, each component
from its own resolution (cross-family mixtures re-select the demand slot
per episode)."""

from __future__ import annotations

from pathlib import Path

from mdp_ir.differential import DomainAdapter, _import_domain
from mdp_ir.schema import MdpIR


def _sampler_for(samplers, source):
    """The slot's desugared latent sampler (substream = slot stream), if any."""
    return next(
        (s for s in samplers if s.substream_id == source.stream_id), None
    )


def _build_scenario(
    unc, scen, *, name, selection, sources, samplers, consts,
    event_sequence, horizon, seed_salt,
):
    """One composition's scenario source: shipped candidates use their
    hand-written generators; unknown candidates use the family bridges."""
    demand_src = next(s for s in sources if s.name == "demand")
    leadtime_src = next(s for s in sources if s.name == "leadtime")
    sel = (selection or {}).get("demand", "simple")
    if sel == "simple":
        demand = unc.PoissonDemand(
            rate=consts["simple_rate"],
            source_id=demand_src.stream_id,
        )
    elif sel == "discrete":
        demand = unc.LatentDiscreteDemand(
            support_size=consts["demand_support_size"],
            support_low=consts["demand_support_low"],
            support_high=consts["demand_support_high"],
            source_id=demand_src.stream_id,
        )
    elif sel == "poisson":
        demand = unc.LatentPoissonDemand(
            alpha=consts["demand_alpha"],
            beta=consts["demand_beta"],
            source_id=demand_src.stream_id,
        )
    else:  # appended candidate: generic family runtime, zero new code
        demand = unc.FamilyDemand.from_parts(
            demand_src, _sampler_for(samplers, demand_src), consts
        )
    lt_sel = (selection or {}).get("leadtime", "deterministic")
    if lt_sel == "deterministic":
        leadtime = unc.DeterministicLeadtime(
            value=consts["leadtime_value"],
            source_id=leadtime_src.stream_id,
        )
    elif lt_sel == "slt":
        leadtime = unc.DiscreteLeadtime(
            values=consts["leadtime_values"],
            probabilities=consts["leadtime_probs"],
            source_id=leadtime_src.stream_id,
        )
    else:
        leadtime = unc.FamilyLeadtime.from_parts(
            leadtime_src, _sampler_for(samplers, leadtime_src), consts
        )
    template = scen.InvSingleScenario(
        scenario_name=name,
        horizon=horizon,
        demand=demand,
        leadtime=leadtime,
        holding_cost=consts["h"],
        shortage_cost=consts["b"],
        order_cost_linear=consts["c"],
        order_cost_fixed=consts["K"],
        stockout_mode=consts["stockout_mode"],
        event_sequence=tuple(event_sequence),
        seed_salt=seed_salt,
    )
    return scen.InvSingleScenarioSource(template) if template.has_latents else template


def make_adapter(
    ir: MdpIR,
    instance: str | None = None,
    seed_salt: int = 0,
    domain_dir: Path | None = None,
) -> DomainAdapter:
    """Builds the ``InvSingleScenario`` (or mixture) from the IR's scenario
    constants and drives the ``init_state`` → ``advance1`` → ``advance2``
    loop."""
    unc, scen, mdp = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["inv_single_uncertainty", "inv_single_scenarios", "inv_single_mdp"],
    )

    base_consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    mixture = next(
        (m for m in ir.mdp.scenario.mixtures if m.name == instance), None
    )
    if mixture is not None:
        # mixture twin: per-component scenario sources; a component whose
        # selection differs from the loaded IR's carries its own resolution
        components = []
        for w, comp in mixture.components:
            res = ((ir.mixture_resolutions or {}).get(mixture.name) or {}).get(comp)
            consts = dict(base_consts)
            if res is not None:
                consts.update(res.constants)
            if comp:
                consts.update(ir.mdp.scenario.instances[comp])
            components.append((w, _build_scenario(
                unc, scen,
                name=f"ir_mix_{comp or 'base'}",
                selection=res.selection if res is not None else ir.selection,
                sources=res.sources if res is not None else ir.mdp.uncertainty_sources,
                samplers=res.samplers if res is not None else ir.mdp.scenario.samplers,
                consts=consts,
                event_sequence=ir.mdp.event_sequence(comp or None),
                horizon=ir.mdp.horizon_T(comp or None),
                seed_salt=seed_salt,
            )))
        source = scen.InvSingleMixtureSampler(
            scenario_name=f"ir_differential_{mixture.name}",
            components=components,
            substream_id=mixture.substream_id,
            seed_salt=seed_salt,
        )
    else:
        consts = dict(base_consts)
        if instance is not None:
            consts.update(ir.mdp.scenario.instances[instance])
        source = _build_scenario(
            unc, scen,
            name=f"ir_differential_{instance or 'base'}",
            selection=ir.selection,
            sources=ir.mdp.uncertainty_sources,
            samplers=ir.mdp.scenario.samplers,
            consts=consts,
            event_sequence=ir.mdp.event_sequence(instance),
            horizon=ir.mdp.horizon_T(instance),
            seed_salt=seed_salt,
        )

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        scenario = source(episode_seed) if callable(source) else source
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

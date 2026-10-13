"""Differential adapter for the ``cnv`` domain (portable-domain contract:
lives with the domain, discovered from the IR file's directory by
``mdp_ir.differential.load_adapter_factory``).

Dispatch is by the resolved IR's **selection** (catalog model): the shipped
candidates map to their hand-written generators (``LatentPoissonDemand`` for
the demand slot's hidden-rate Poisson, ``UniformEpochs`` for the epoch
vector); any other candidate — a freshly appended family — falls back to the
``Family{Source}`` bridges (``mdp_ir.runtime.FamilyGenerator``), so a new
family differentially verifies with zero adapter edits."""

from __future__ import annotations

from pathlib import Path

from mdp_ir.differential import DomainAdapter, _import_domain
from mdp_ir.schema import MdpIR


def _sampler_for(samplers, source):
    """The slot's desugared latent sampler (substream = slot stream), if any."""
    return next(
        (s for s in samplers if s.substream_id == source.stream_id), None
    )


def make_adapter(
    ir: MdpIR,
    instance: str | None = None,
    seed_salt: int = 0,
    domain_dir: Path | None = None,
) -> DomainAdapter:
    """Builds the ``CnvScenarioSource`` from the IR's (instance-merged)
    scenario constants and drives the ``init_state`` → ``advance`` loop."""
    unc, scen, mdp = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["cnv_uncertainty", "cnv_scenarios", "cnv_mdp"],
    )

    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])
    sources = ir.mdp.uncertainty_sources
    samplers = ir.mdp.scenario.samplers
    demand_src = next(s for s in sources if s.name == "demand")
    epochs_src = next(s for s in sources if s.name == "epochs")
    selection = ir.selection or {}

    if selection.get("demand", "poisson") == "poisson":
        # the same expressions the IR's draw spec uses, in the same float order
        demand = unc.LatentPoissonDemand(
            alpha=consts["prior_rate"] * consts["prior_mean"],
            beta=consts["prior_rate"],
            source_id=demand_src.stream_id,
        )
    else:  # appended candidate: generic family runtime, zero new code
        demand = unc.FamilyDemand.from_parts(
            demand_src, _sampler_for(samplers, demand_src), consts
        )
    if selection.get("epochs", "uniform") == "uniform":
        epochs = unc.UniformEpochs(
            size=int(consts["arrivals_buffer"]), source_id=epochs_src.stream_id
        )
    else:
        epochs = unc.FamilyEpochs.from_parts(
            epochs_src, _sampler_for(samplers, epochs_src), consts
        )

    template = scen.CnvScenario(
        scenario_name=f"ir_differential_{instance or 'base'}",
        horizon=ir.mdp.horizon_T(instance),
        demand=demand,
        epochs=epochs,
        price=consts["price"],
        unit_cost=consts["unit_cost"],
        prior_rate=consts["prior_rate"],
        prior_mean=consts["prior_mean"],
        seed_salt=seed_salt,
    )
    source = scen.CnvScenarioSource(template) if template.has_latents else template

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        scenario = source(episode_seed) if callable(source) else source
        state, _ = mdp.init_state(scenario, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            state, info = mdp.advance(scenario, state, order=int(acts["order"]))
            rows.append({
                "t": info["action_period"],
                "order": info["order"],
                "demand": info["demand"],
                "sales": info["sales"],
                "stockout_event": info["stockout_event"],
                "stockout_time": info["stockout_time"],
                "action_period": info["action_period"],
                "cum_sales": state.cum_sales,
                "cum_duration": state.cum_duration,
                "cum_demand": state.cum_demand,
                **info["profit"],   # revenue, stocking, total
            })
        return rows

    return run_episode

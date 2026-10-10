"""Differential adapter for the ``owmr`` domain (portable-domain contract:
lives with the domain, discovered from the IR file's directory by
``mdp_ir.differential.load_adapter_factory``).

Builds an ``OwmrScenario`` from the IR's own scenario constants and drives
the ``init_state`` -> ``advance1`` -> ``advance2`` loop, emitting rows keyed
exactly as the interpreter names them. Dispatch is by the resolved IR's
**selection**: the shipped ``gamma`` candidate maps to ``GammaDemand``; any
other candidate — a family appended later — falls back to the ``FamilyDemand``
bridge, so a new family differentially verifies with zero adapter edits.
"""

from __future__ import annotations

from pathlib import Path

from mdp_ir.differential import DomainAdapter, _import_domain
from mdp_ir.schema import MdpIR


def _sampler_for(samplers, source):
    """The slot's desugared latent sampler (substream = slot stream), if any.
    Always ``None`` for the shipped candidate: no world latents here."""
    return next((s for s in samplers if s.substream_id == source.stream_id), None)


def make_adapter(
    ir: MdpIR,
    instance: str | None = None,
    seed_salt: int = 0,
    domain_dir: Path | None = None,
) -> DomainAdapter:
    unc, scen, mdp = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["owmr_uncertainty", "owmr_scenarios", "owmr_mdp"],
    )

    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])

    demand_src = next(s for s in ir.mdp.uncertainty_sources if s.name == "demand")
    sel = (ir.selection or {}).get("demand", "gamma")
    if sel == "gamma":
        demand = unc.GammaDemand(
            mu_rt=consts["mu_rt"], cv_rt=consts["cv_rt"], source_id=demand_src.stream_id
        )
    else:  # appended candidate: generic family runtime, zero new domain code
        demand = unc.FamilyDemand.from_parts(
            demand_src, _sampler_for(ir.mdp.scenario.samplers, demand_src), consts
        )

    scenario = scen.OwmrScenario(
        scenario_name=f"ir_differential_{instance or 'base'}",
        horizon=ir.mdp.horizon_T(instance),
        n_retailers=int(consts["n_retailers"]),
        l0=int(consts["l0"]),
        l_rt=int(consts["l_rt"]),
        demand=demand,
        h0=float(consts["h0"]),
        h_rt=tuple(consts["h_rt"]),
        p_rt=tuple(consts["p_rt"]),
        mu_rt=tuple(consts["mu_rt"]),
        cv_rt=tuple(consts["cv_rt"]),
        order_scale_mult=float(consts["order_scale_mult"]),
        seed_salt=seed_salt,
    )
    n = int(consts["n_retailers"])

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        sc = scenario(episode_seed) if callable(scenario) else scenario
        state, _ = mdp.init_state(sc, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            state1 = mdp.advance1(sc, state)
            raw = acts["ship"]
            ship = [float(x) for x in (raw if isinstance(raw, list) else [raw] * n)]
            state, info = mdp.advance2(sc, state1, float(acts["order"]), ship)
            rows.append({
                "t": info["action_period"],
                "order": info["order"],
                "ship": list(ship),
                "wh_arrival": info["wh_arrival"],
                "rt_arrival": list(info["rt_arrival"]),
                "shipped": list(info["shipped"]),
                "ship_scale": info["ship_scale"],
                "demand": list(info["demand"]),
                "action_period": info["action_period"],
                "wh_stock": state.wh_stock,
                "wh_pipe": list(state.wh_pipe),
                "rt_stock": list(state.rt_stock),
                "rt_pipe": [list(row) for row in state.rt_pipe],
                **info["cost"],   # wh_holding, rt_holding, rt_shortage, total
            })
        return rows

    return run_episode

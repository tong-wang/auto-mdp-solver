"""Differential adapter for the ``adi_flex`` domain (portable-domain
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
    """Builds the ``AdiFlexScenario`` from the IR's scenario constants and
    drives the two-step ``init_state`` → ``advance1``/``advance2`` loop (F7:
    order and allocation sit at different information sets; the adapter, like
    the gym, composes the pair — one period per decision dict)."""
    (scen, mdp) = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["adi_flex_scenarios", "adi_flex_mdp"],
    )

    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])

    scenario = scen.AdiFlexScenario(
        scenario_name=f"ir_differential_{instance or 'base'}",
        lambda_seg=tuple(consts["lambda_seg"]),
        L=consts["L"],
        N=consts["N"],
        K=consts["K"],
        h=consts["h"],
        p=consts["p"],
        # beta lives on the objective, not in scenario.constants (spec Objective)
        discount=ir.mdp.objective.discount_factor,
        alloc_enabled=consts["alloc_enabled"],
        seed_salt=seed_salt,
    )

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        state, _ = mdp.init_state(scenario, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            mid, info1 = mdp.advance1(scenario, state, order=int(acts["order"]))
            # `allocate` is a vector decision (IR dim n_alloc) — pass it through
            # rather than scalarizing it (F10: int() here was a second site
            # frozen at T_dl = 2, and would raise on a wider instance)
            state, info2 = mdp.advance2(scenario, mid, allocate=acts["allocate"])
            info = mdp.merge_info(info1, info2)   # one record, two halves
            rows.append({
                "t": info["action_period"],
                "order": info["order"],
                # the interpreter renders a dim-1 decision as a scalar and a
                # wider one as a list; mirror that so the comparison is on the
                # value, not on the container
                "allocate": (list(info["allocate"]) if len(info["allocate"]) != 1
                             else info["allocate"][0]),
                "d": list(info["d"]),
                "received": info["received"],
                # the REALIZED allocation, per class — the IR computes `fills`
                # as a comprehension, so it renders as a list on both sides.
                # Declaring it in info_fields is not enough to gate it; the
                # differential compares the rows this adapter builds (F20)
                "fills": list(info["fills"]),
                "early_fill": info["early_fill"],
                "surplus": info["surplus"],
                "outstanding": info["outstanding"],
                "inv": state.inv,
                "pipe": list(state.pipe),
                "adv": list(state.adv),
                **info["cost"],   # order_fixed, holding, backorder, total
            })
        return rows

    return run_episode

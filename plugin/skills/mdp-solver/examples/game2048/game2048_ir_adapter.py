"""Differential adapter for the ``game2048`` domain (portable-domain
contract: lives with the domain, discovered from the IR file's directory by
``mdp_ir.differential.load_adapter_factory``).

Builds a ``Game2048Scenario`` from the IR's scenario constants and drives the
``init_state`` -> ``advance`` loop, emitting one row per period keyed exactly
like the interpreter's.

The episode is opened the same way the interpreter opens it: from
``init_state`` (empty board), letting the first ``advance`` run the ``I``
event. ``open_episode`` — the gym's route — is deliberately not used here, so
period 0's ``spawned`` tally lines up with the IR's.

There is no ``FamilyGenerator`` fallback for
an unrecognised candidate: both slots declare **keyed** stages (on
``spawn_count``), and ``FamilyGenerator.from_parts`` rejects those — it keys
the plain v2 intrinsic template. A candidate appended to either slot needs a
hand-written generator class and a branch here.
"""

from __future__ import annotations

from pathlib import Path

from mdp_ir.differential import DomainAdapter, _import_domain
from mdp_ir.schema import MdpIR


def _build_scenario(unc, scen, *, name, selection, consts, horizon, seed_salt):
    """One composition's scenario, from the resolved constants."""
    sel = selection or {}

    cell_sel = sel.get("spawn_cell", "uniform_empty")
    if cell_sel == "uniform_empty":
        spawn_cell = unc.UniformEmptyCell(source_id=0)
    else:
        raise NotImplementedError(
            f"spawn_cell candidate {cell_sel!r} has no generator class. Both "
            "game2048 slots use keyed stages, so FamilyGenerator cannot back "
            "them; add a class in game2048_uncertainty.py and a branch here."
        )

    value_sel = sel.get("spawn_value", "bernoulli_4")
    if value_sel == "bernoulli_4":
        spawn_value = unc.Bernoulli4(prob_4=consts["prob_4"], source_id=1)
    else:
        raise NotImplementedError(
            f"spawn_value candidate {value_sel!r} has no generator class "
            "(see the spawn_cell note above)."
        )

    return scen.Game2048Scenario(
        scenario_name=name,
        grid_size=int(consts["grid_size"]),
        n_cells=int(consts["n_cells"]),
        spawn_cell=spawn_cell,
        spawn_value=spawn_value,
        T_cap=int(horizon),
        invalid_penalty=float(consts["invalid_penalty"]),
        seed_salt=seed_salt,
    )


def make_adapter(
    ir: MdpIR,
    instance: str | None = None,
    seed_salt: int = 0,
    domain_dir: Path | None = None,
) -> DomainAdapter:
    unc, scen, mdp = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["game2048_uncertainty", "game2048_scenarios", "game2048_mdp"],
    )

    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance is not None:
        consts.update(ir.mdp.scenario.instances[instance])

    scenario = _build_scenario(
        unc, scen,
        name=f"ir_differential_{instance or 'base'}",
        selection=ir.selection,
        consts=consts,
        horizon=ir.mdp.horizon_T(instance),
        seed_salt=seed_salt,
    )

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        state, _ = mdp.init_state(scenario, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            t = state.period
            move = int(acts["move"])
            state, info = mdp.advance(scenario, state, move)
            rows.append({
                "t": t,
                "move": acts["move"],
                "gained": info["gained"],
                "move_valid": info["move_valid"],
                "spawned": info["spawned"],
                "game_over": info["game_over"],
                "board": list(state.board),
                "spawn_count": state.spawn_count,
                **info["score"],          # merge, total
            })
        return rows

    return run_episode

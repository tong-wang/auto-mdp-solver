"""What the IR validator must reject.

Every fixture here is the synthetic ``minimal_ir()`` with exactly one thing
broken, so a failure names the rule rather than a domain. Rules that need a
whole domain to state are tested by that domain's own ``{domain}_test.py``.
"""

from __future__ import annotations

import pytest

from mdp_ir.schema import MdpIR


def valid(doc: dict) -> MdpIR:
    return MdpIR.model_validate(doc)


def test_the_fixture_itself_validates(ir_doc):
    """If this fails, every rejection below is testing the wrong thing."""
    assert valid(ir_doc).domain.name == "tiny"


# -- expressions -------------------------------------------------------------


def test_undeclared_function_in_an_expression_is_rejected(ir_doc):
    comp = ir_doc["mdp"]["objective"]["per_step_components"][0]
    comp["expr"] = f"triple_it({comp['expr']})"
    with pytest.raises(ValueError, match="triple_it"):
        valid(ir_doc)


def test_expr_builtin_may_not_shadow_a_core_builtin(ir_doc):
    ir_doc["mdp"]["expr_builtins"] = [{"name": "min", "module": "extras"}]
    with pytest.raises(ValueError, match="shadow"):
        valid(ir_doc)


def test_string_literals_are_data_not_identifiers(ir_doc):
    """A categorical constant may be compared against a literal without the
    literal's characters being read as unresolved names."""
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "mode", "value": "fast", "axis": "variant"})
    ir_doc["mdp"]["dynamics"]["transitions"][0]["updates"] = [
        "level += act if mode == 'fast' else 0"
    ]
    assert valid(ir_doc)


# -- horizon / event order ---------------------------------------------------


def test_horizon_T_may_name_a_constant(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "n_periods", "value": 5, "axis": "sizing"})
    ir_doc["mdp"]["horizon"]["T"] = "n_periods"
    assert valid(ir_doc).mdp.horizon_T() == 5


def test_horizon_T_naming_an_unknown_constant_is_rejected(ir_doc):
    ir_doc["mdp"]["horizon"]["T"] = "no_such"
    with pytest.raises(ValueError):
        valid(ir_doc)


def test_event_sequence_may_name_a_constant(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "evt", "value": ["A", "B"], "axis": "variant"})
    ir_doc["mdp"]["dynamics"]["event_sequence"] = "evt"
    ir_doc["mdp"]["scenario"]["instances"]["flipped"] = {"evt": ["B", "A"]}
    ir = valid(ir_doc)
    assert ir.mdp.event_sequence() == ["A", "B"]
    assert ir.mdp.event_sequence("flipped") == ["B", "A"]


def test_event_sequence_naming_an_unknown_constant_is_rejected(ir_doc):
    ir_doc["mdp"]["dynamics"]["event_sequence"] = "no_such_const"
    with pytest.raises(ValueError):
        valid(ir_doc)


def test_an_instance_order_missing_an_event_is_rejected(ir_doc):
    ir_doc["mdp"]["scenario"]["constants"].append(
        {"name": "evt", "value": ["A", "B"], "axis": "variant"})
    ir_doc["mdp"]["dynamics"]["event_sequence"] = "evt"
    ir_doc["mdp"]["scenario"]["instances"]["broken"] = {"evt": ["A"]}
    with pytest.raises(ValueError):
        valid(ir_doc)


def test_literal_sequence_requires_matching_declaration_order(ir_doc):
    ts = ir_doc["mdp"]["dynamics"]["transitions"]
    ts[0], ts[1] = ts[1], ts[0]
    with pytest.raises(ValueError):
        valid(ir_doc)


# -- seed scheme v2 ----------------------------------------------------------


def test_v2_rejects_episode_realization_stages(ir_doc):
    """Treatment A: under v2 an episode-realization stage must be a scenario
    sampler instead."""
    ir_doc["mdp"]["uncertainty_sources"][0]["stages"].append(
        {"name": "support", "realization": "episode", "sub_stream": 0})
    with pytest.raises(ValueError):
        valid(ir_doc)


def test_duplicate_meta_substream_ids_are_rejected(ir_doc):
    sampler = {
        "name": "s", "substream_id": 0,
        "draws": [{"name": "rate", "distribution": {
            "family": "uniform", "settings": {"low": 1.0, "high": 2.0}}}],
    }
    ir_doc["mdp"]["scenario"]["samplers"] = [sampler, {**sampler, "name": "s2"}]
    with pytest.raises(ValueError):
        valid(ir_doc)


# -- grids -------------------------------------------------------------------


def test_grid_axes_must_name_constants(ir_doc):
    ir_doc["grids"] = [{"name": "bad", "axes": {"no_such": [1]}}]
    with pytest.raises(ValueError):
        valid(ir_doc)


def test_grid_cells_are_row_major_with_axis_derived_ids(ir_doc):
    ir_doc["grids"] = [{"name": "sweep", "axes": {"h": [1, 2], "rate": [3, 4, 5]}}]
    cells = valid(ir_doc).grids[0].cells()
    assert len(cells) == 6
    assert cells[0][0] == "h=1,rate=3"
    assert cells[1][1] == {"h": 1, "rate": 4}


# -- invariants --------------------------------------------------------------


def test_invariant_may_reference_prev_and_constants(ir_doc):
    ir_doc["mdp"]["invariants"] = [
        {"name": "balance", "expr": "close(level, prev.level + act - outflow)"}
    ]
    assert len(valid(ir_doc).mdp.invariants) == 1


def test_invariant_prev_must_name_a_per_period_value(ir_doc):
    ir_doc["mdp"]["invariants"] = [
        {"name": "bad", "expr": "close(level, prev.no_such)"}
    ]
    with pytest.raises(ValueError, match="per-period"):
        valid(ir_doc)


def test_invariant_prev_may_not_name_a_constant(ir_doc):
    """A scenario constant is fixed for the episode, so `prev.h` is just `h`
    and is far more likely a mistake than an intent."""
    ir_doc["mdp"]["invariants"] = [{"name": "bad", "expr": "close(h, prev.h)"}]
    with pytest.raises(ValueError, match="per-period"):
        valid(ir_doc)


def test_bare_prev_is_rejected(ir_doc):
    ir_doc["mdp"]["invariants"] = [{"name": "bad", "expr": "close(level, prev)"}]
    with pytest.raises(ValueError):
        valid(ir_doc)


def test_invariant_with_an_unknown_name_is_rejected(ir_doc):
    ir_doc["mdp"]["invariants"] = [{"name": "bad", "expr": "close(nope, 1)"}]
    with pytest.raises(ValueError):
        valid(ir_doc)


def test_duplicate_invariant_names_are_rejected(ir_doc):
    ir_doc["mdp"]["invariants"] = [
        {"name": "same", "expr": "level >= 0"},
        {"name": "same", "expr": "level <= 100"},
    ]
    with pytest.raises(ValueError, match="duplicate invariant"):
        valid(ir_doc)


def test_invariant_may_use_the_row_period_t(ir_doc):
    ir_doc["mdp"]["invariants"] = [{"name": "clock", "expr": "t >= 0"}]
    assert valid(ir_doc)


# --- rl.algos: the algorithm as a declarable axis (upstream #26 Part A) ---

def test_algo_alone_still_validates_and_reads_as_one_class(ir_doc):
    """Every existing IR: `algo` scalar, no `algos` — must not need migration."""
    from mdp_ir.schema import Algo, MdpIR

    ir = MdpIR.model_validate(ir_doc)
    assert ir.rl.declared_algos() == [Algo.ppo]


def test_a_campaign_can_declare_the_algorithm_axis(ir_doc):
    from mdp_ir.schema import Algo, MdpIR

    ir_doc["rl"]["algos"] = ["ppo", "maskable_ppo"]
    ir_doc["mdp"]["decisions"][0]["type"] = {"value": "discrete",
                                             "suggested": "discrete"}
    ir_doc["gym"]["action_modes"][0]["type"] = "discrete"
    ir = MdpIR.model_validate(ir_doc)
    assert ir.rl.declared_algos() == [Algo.ppo, Algo.maskable_ppo]


def test_the_default_class_must_be_one_of_the_declared_ones(ir_doc):
    import pytest
    from mdp_ir.schema import MdpIR

    ir_doc["rl"]["algo"] = "ppo"
    ir_doc["rl"]["algos"] = ["recurrent_ppo"]
    with pytest.raises(Exception, match="not in rl.algos"):
        MdpIR.model_validate(ir_doc)


def test_masking_declared_as_an_axis_still_owes_a_discrete_action(ir_doc):
    """The discreteness condition follows the class wherever it is declared."""
    import pytest
    from mdp_ir.schema import MdpIR

    ir_doc["rl"]["algos"] = ["ppo", "maskable_ppo"]   # continuous default action
    with pytest.raises(Exception, match="discrete default action mode"):
        MdpIR.model_validate(ir_doc)

"""`scripts.l1_derived` — the run name diffs the derivation, not the defaults.

Spec §8.4's hyperparameter tier used to be a diff against the train script's
live parser defaults, and §8.6 separately calls those defaults the L1 centre.
Both hold only until a campaign adopts a discovered value: promote it into a
default and the knob stops differing from it, so the tag vanishes from every
later run name while the earlier dirs — immutable by the log guide — keep
theirs. The archive then holds two identical names meaning different configs.
`_L1_DERIVED` is the fixed baseline that dissolves that, and these tests cover
the two ways adopting it can be worse than not adopting it: a key the CLI has
no dest for, and a dict nothing reads.

Synthetic scripts only: the check must not learn the shape of any shipped
domain.
"""

from __future__ import annotations

from mdp_conformance.checks import check_l1_derived
from mdp_conformance.loader import DomainHandle


class PlainEnv:
    def __init__(self, scenario, logger_filename=None): ...


def script(dests: list[str], derived: str | None = None, *,
           read: bool = True) -> str:
    """A train script exposing `dests`, optionally declaring `_L1_DERIVED`."""
    args = "\n".join(f'    p.add_argument("--{d.replace("_", "-")}")'
                     for d in dests)
    out = ["import argparse\n"]
    if derived is not None:
        out.append(f"_L1_DERIVED: dict[str, object] = {derived}\n")
    out.append("def _build_arg_parser():\n"
               "    p = argparse.ArgumentParser()\n" + args + "\n    return p\n")
    out.append("def build_run_name(args):\n"
               "    defaults = vars(_build_arg_parser().parse_args([]))\n"
               + ("    defaults.update(_L1_DERIVED)\n" if derived and read else "")
               + "    return str(defaults)\n")
    return "\n".join(out)


def domain(tmp_path, source: str | None) -> DomainHandle:
    if source is not None:
        (tmp_path / "d_ppo_train.py").write_text(source)
    return DomainHandle(
        name="d", directory=tmp_path, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=PlainEnv,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


def test_a_declared_and_read_derivation_passes(tmp_path):
    r = check_l1_derived(domain(tmp_path, script(
        ["gamma", "gae_lambda"], '{"gamma": 0.99, "gae_lambda": 0.95}')))
    assert r.status == "PASS" and "2 derived value(s)" in r.detail


def test_a_script_without_one_is_pre_convention_not_broken(tmp_path):
    """Adoption is gradual, so absence WARNs — but it says what the absence
    costs, since the cost only appears later, at the moment of a promotion."""
    r = check_l1_derived(domain(tmp_path, script(["gamma"])))
    assert r.status == "WARN"
    assert "pre-convention" in r.detail and "promoted" in r.detail


def test_a_key_the_cli_cannot_reach_fails(tmp_path):
    """A dest renamed out from under the dict, or a typo: the diff can never
    fire on that knob, and nothing else would say so."""
    r = check_l1_derived(domain(tmp_path, script(
        ["gamma"], '{"gamma": 0.99, "gae_lamda": 0.95}')))
    assert r.status == "FAIL" and "gae_lamda" in r.detail


def test_a_derivation_nothing_reads_fails(tmp_path):
    """The worst shape: the script looks adopted, and its run names still diff
    the parser defaults. Silent, and invisible until a promotion."""
    r = check_l1_derived(domain(tmp_path, script(
        ["gamma"], '{"gamma": 0.99}', read=False)))
    assert r.status == "FAIL" and "never read" in r.detail


def test_an_empty_derivation_fails(tmp_path):
    r = check_l1_derived(domain(tmp_path, script(["gamma"], "{}")))
    assert r.status == "FAIL" and "states nothing" in r.detail


def test_no_train_script_yet_is_not_a_violation(tmp_path):
    assert check_l1_derived(domain(tmp_path, None)).status == "SKIP"

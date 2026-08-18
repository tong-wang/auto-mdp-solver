"""`docs.restatement_current` — the round-trip artifact still renders.

The restatement is the artifact whose purpose is to be believed by a human,
and it is the one nothing else can reach: the differential moves both sides
together when the model changes, conformance reads Python, the laws runner
reads execution semantics, and the fingerprints hash the IR. So these tests
cover the two failure shapes actually observed upstream — a block whose
numbers no longer reproduce, and a block whose *command* no longer names the
run that produced it — plus the two document conventions the check depends on
(a declared command, verbatim output) and the refusal that keeps it from
executing whatever a document happens to contain.

Synthetic IR only: the check must not learn the shape of any shipped domain.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from mdp_conformance.checks import check_restatement_current
from mdp_conformance.loader import DomainHandle


DOC = "d.restatement.md"
COMMAND = "python -m mdp_ir.interpreter d/d_schema.json --decision act=1 --episode-seed 3"


@pytest.fixture
def domain(tmp_path, ir_doc):
    """A one-folder domain: the IR, and nothing written to the doc yet."""
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "d_schema.json").write_text(json.dumps(ir_doc))
    return DomainHandle(
        name="d", directory=directory, files={}, modules={},
        mdp_module_name="d_mdp", SCENARIOS={}, env_cls=object,
        state_cls=None, init_state=lambda *a, **k: (None, {}),
    )


def render(domain, *extra: str) -> str:
    """What the declared command actually prints, right now."""
    argv = COMMAND.split()[1:] + list(extra)
    proc = subprocess.run([sys.executable, *argv], cwd=domain.directory.parent,
                          capture_output=True, text=True, check=True)
    return proc.stdout.rstrip("\n")


def write(domain, body: str) -> None:
    (domain.directory / DOC).write_text(body)


def declared(command: str, output: str, prose: str = "") -> str:
    return (f"# d\n\n```step7b\n{command}\n```\n{prose}\n```\n{output}\n```\n")


# -- the two SKIPs: nothing to check, and nothing declared to check with -----


def test_a_domain_without_a_restatement_skips(domain):
    assert check_restatement_current(domain).status == "SKIP"


def test_an_undeclared_command_skips_rather_than_failing(domain):
    """Adoption is one fenced block, so a document written before the
    convention SKIPs — the check must not fail a domain on upgrade."""
    write(domain, f"# d\n\n```\n{render(domain)}\n```\n")
    result = check_restatement_current(domain)
    assert result.status == "SKIP"
    assert "step7b" in result.detail


# -- the pass path -----------------------------------------------------------


def test_a_verbatim_block_passes(domain):
    write(domain, declared(COMMAND, render(domain)))
    result = check_restatement_current(domain)
    assert result.status == "PASS"
    assert "1 declared" in result.detail


def test_prose_may_sit_between_the_command_and_its_output(domain):
    """The annotation is the point of the artifact, so the check pairs a
    command with the next fenced block, not the next line."""
    write(domain, declared(COMMAND, render(domain),
                           prose="\nA deliberately weak policy: one unit a period.\n"))
    assert check_restatement_current(domain).status == "PASS"


def test_several_declarations_are_each_re_run(domain):
    body = (declared(COMMAND, render(domain))
            + declared(COMMAND + " --instance pricey",
                       render(domain, "--instance", "pricey")))
    write(domain, body)
    result = check_restatement_current(domain)
    assert result.status == "PASS"
    assert "2 declared" in result.detail


# -- the failure the fingerprint cannot see ---------------------------------


def test_a_stale_number_warns_and_names_the_line(domain):
    """Partial diligence: the header token is updated, the expensive
    regeneration is skipped, and one row still holds the old value."""
    lines = render(domain).splitlines()
    lines[3] = lines[3].replace("1.00", "9.00", 1)
    write(domain, declared(COMMAND, "\n".join(lines)))
    result = check_restatement_current(domain)
    assert result.status == "WARN"
    assert "output line 4" in result.detail


def test_a_hand_trimmed_block_warns(domain):
    """`...` in place of the remaining rows reads as tidy and silently
    removes the diff target."""
    lines = render(domain).splitlines()[:-2]
    write(domain, declared(COMMAND, "\n".join(lines)))
    result = check_restatement_current(domain)
    assert result.status == "WARN"
    assert "line(s), live has" in result.detail


def test_a_command_naming_the_wrong_run_warns(domain):
    """The `mab` shape: every number in the block is right, but the command
    printed beside it renders a different instance — so the `mdp` fingerprint
    never moves and only a re-run can tell."""
    write(domain, declared(COMMAND, render(domain, "--instance", "pricey")))
    assert check_restatement_current(domain).status == "WARN"


# -- the conventions the re-run depends on ----------------------------------


def test_a_path_that_does_not_travel_with_the_folder_warns(domain):
    """Written from the repo root instead of the folder's parent, the command
    works where its author stood and nowhere else — including here."""
    outside = f"python -m mdp_ir.interpreter cases/d/d_schema.json --decision act=1"
    write(domain, declared(outside, render(domain)))
    result = check_restatement_current(domain)
    assert result.status == "WARN"
    assert "travels with the folder" in result.detail


def test_a_command_with_no_output_block_warns(domain):
    write(domain, f"# d\n\n```step7b\n{COMMAND}\n```\n\nand then prose.\n")
    result = check_restatement_current(domain)
    assert result.status == "WARN"
    assert "no output block follows" in result.detail


def test_a_non_interpreter_command_is_refused_not_run(domain):
    """A document is untrusted input. The check re-runs one invocation shape
    and reports anything else rather than executing it."""
    hostile = f"python -c import_pathlib_and_write('{domain.directory}/pwned')"
    write(domain, declared(hostile, "whatever"))
    result = check_restatement_current(domain)
    assert result.status == "WARN"
    assert "only that is re-run" in result.detail
    assert not (domain.directory / "pwned").exists()

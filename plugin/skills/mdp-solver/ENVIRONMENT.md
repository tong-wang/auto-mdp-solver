# Environment, budgets, and run discipline

Shared ground for every op of the split pipeline (`CONTRACTS.md` is the op
map). Each op resolves the interpreter per this file before running anything,
and its budgets and background rules govern every stage.

**Environment (once, before anything).** Every gate and every generated script
runs through **one** interpreter. Resolve it first, then use that exact path
for all commands in every op — never a bare `python`.

Resolve in this order:

1. `$MDP_SOLVER_PYTHON`, if set.
2. `.venv/bin/python` under the **workspace root** (the directory holding the
   domain folders / `cases/`), if it exists.
3. Otherwise create it there — `uv venv --python 3.12` if `uv` is available
   (seconds; fetches a conforming Python if the box has none), else
   `python3 -m venv .venv`.

   The package needs **Python ≥ 3.12**. Check `python3 --version` *before*
   the fallback: a distro `python3` is often older (3.10 is common) and may
   also lack `ensurepip`, in which case `python3 -m venv` fails outright.
   If no conforming interpreter exists, stop and say so — do not build a venv
   on an unsupported version.

Install into it (skip if the probe below already prints `ENV OK`):

```bash
# the plugin's own version — the harness release must match it exactly
V=$(python3 -c "import json;print(json.load(open('${CLAUDE_SKILL_DIR}/../../.claude-plugin/plugin.json'))['version'])")
.venv/bin/python -m pip install "auto-mdp-solver[domain]==$V" \
  || .venv/bin/python -m pip install "auto-mdp-solver[domain] @ git+https://github.com/tong-wang/auto-mdp-solver@v$V#subdirectory=harness"
# uv: `uv pip install --python .venv/bin/python <the same spec>`
# from a repo checkout: ... install -e "<repo-root>/harness[domain]"
```

PyPI first, pinned; when PyPI has no release at that version the second
command installs the same tag straight from GitHub (`subdirectory=harness`
is the package root). Never install unpinned: the name existed on PyPI as an
empty stub before the first real release, and an unpinned install of it
succeeds and then fails the probe below.

The **`[domain]` extra is required**: the bare package is deliberately
torch-free (it carries only the IR/conformance/gate/tuning harness), so
without it Stage 4 cannot train. Torch is a large download — say so before
starting one.

**Network.** The install above is the one step that must reach PyPI (or
GitHub). If the host runs shell commands in a sandbox without network —
Codex's default `workspace-write` mode, Claude Code with its sandbox on —
the command fails on DNS and nothing prompts for help: **request escalated
permissions for that one command**, stating why, and let the human approve
it; do not silently report the failure and do not retry inside the sandbox.
The human may instead lift the wall per project (Codex:
`[sandbox_workspace_write] network_access = true`). The same rule covers
every later command that needs the network — the `gh` calls in
`mdp-contribute` and `mdp-propose`.

**GATE — run verbatim; `ENV OK` is required before Phase B:**

```bash
.venv/bin/python - <<'EOF'
import sys, importlib.util as u
core   = ["mdp_ir","mdp_conformance","mdp_gates","mdp_tuning","mdp_stage","numpy","pydantic","gymnasium","optuna"]
domain = ["stable_baselines3","sb3_contrib","torch","pandas","tensorboard"]
miss = []
for m in core + domain:
    try:
        if u.find_spec(m) is None: miss.append(m)
    except (ModuleNotFoundError, ValueError): miss.append(m)
bad_py = sys.version_info < (3, 12)
if bad_py: print("BAD PYTHON:", sys.version.split()[0], "- need >= 3.12")
print("ENV OK" if not miss and not bad_py else "MISSING: " + " ".join(miss))
EOF
```

The venv lives in the workspace, not beside this skill: a plugin install is
replaced on update, and the generated domains plus each case README must stay
reproducible from the workspace alone.

**Retry budget: 3 repair attempts per gate, 1 per failing training design
axis.** When a budget is exhausted, stop and surface the failure — do not
loop. Ask before launching anything expected to take > 30 minutes of
compute; run training/tuning as a **background job of the host** and keep
working — Claude Code's `run_in_background`, Codex's background terminal
(`/ps` lists it). Never a bare `nohup … &` from a foreground command: on
Codex every command runs in its own PID namespace and a detached child dies
with the turn. The generated train scripts tee their own logs, so the
background job's console output goes to the scratchpad and the run is
watched through `results/…/train.log`.

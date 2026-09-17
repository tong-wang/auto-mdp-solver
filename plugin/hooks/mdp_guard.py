#!/usr/bin/env python3
"""Pipeline guard — the executable gates, made binding by the host.

One script for both hosts (Claude Code, Codex): they feed the same JSON on
stdin (``hook_event_name``, ``cwd``, ``stop_hook_active``,
``last_assistant_message``) and both treat exit code 2 with a reason on
stderr as "block, and show the model that reason". Nothing here decides
anything the pipeline's own tools do not already decide — it only runs
them at the two moments the model could otherwise skip them:

* ``Stop`` — before a turn ends on a completion claim, run ``mdp_stage`` on
  every domain folder in the workspace and refuse the stop when the claim
  contradicts the table (an IR that does not validate, a "trained" folder
  that has no build, a "packaged" one with no run plan). Blocks once per
  turn: the host sets ``stop_hook_active`` on the re-entry.
* ``PostToolUse`` — after any tool call, validate every ``{name}_schema.json``
  whose mtime moved since the last check, and feed a failing validator's
  output straight back. An invalid IR then never survives to a sign-off.
* ``UserPromptSubmit`` — when the prompt reads as a pipeline request (the
  skill's own trigger vocabulary), record that intent for the workspace and
  tell the model to use the ``mdp-solver`` skill. With intent on record, the
  ``Stop`` rule above also fires when *no* domain folder exists at all — the
  case where the model never opened the skill and solved the problem its own
  way, which no per-folder check can see.

Runs on the system ``python3`` with the stdlib only; the pipeline tools run
under the workspace interpreter resolved exactly as ENVIRONMENT.md says
(``$MDP_SOLVER_PYTHON``, else ``<cwd>/.venv/bin/python``). No interpreter,
no domain folder, no claim → exit 0, silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SKIP_DIRS = {".venv", ".git", "node_modules", "results", "scratch", "__pycache__"}
TIMEOUT = 90

CLAIM = re.compile(
    r"\b(complete[d]?|done|finished|solved|succe(ss|ed)\w*|ready|validated|"
    r"passes|passed|frozen|signed[- ]off)\b", re.I)
FORMALIZE = re.compile(r"\b(formaliz\w*|restatement|sign[- ]?off|phase[- ]a|\bIR\b)", re.I)
SOLVE = re.compile(r"\b(train\w*|ppo|leaderboard|baselines?|solve[ds]?|benchmark\w*)\b", re.I)
PACKAGE = re.compile(r"policy\.py|deploy\w*|packag\w*", re.I)
# The skill's trigger vocabulary (mdp-solver's description + the README's
# quick-start phrasing). A prompt matching it is a pipeline request.
TRIGGER = re.compile(
    r"formaliz\w*|\bMDPs?\b|markov decision|train\w* an? (rl )?polic|"
    r"build an? domain|mdp pipeline|\bphase [ab]\b|_schema\.json|"
    r"solve it and compare|sequential decision|dynamic[- ]decision", re.I)
# A turn that ends by asking the human something is a question round, not a
# completion claim — a ballot line, an "Other" escape, or a trailing question.
ASKS = re.compile(r"(?m)^\s*(Other\b|\d+\.\s+\S)|\?\s*$|\?\s*\n[^\n]{0,200}$")


def interpreter(cwd: Path) -> Path | None:
    env = os.environ.get("MDP_SOLVER_PYTHON")
    if env and Path(env).exists():
        return Path(env)
    venv = cwd / ".venv" / "bin" / "python"
    return venv if venv.exists() else None


def domain_folders(cwd: Path, depth: int = 2) -> list[Path]:
    """Folders holding ``{name}/{name}_schema.json``, up to ``depth`` below cwd."""
    found: list[Path] = []

    def walk(d: Path, level: int) -> None:
        try:
            entries = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            return
        for p in entries:
            if p.name.startswith(".") or p.name in SKIP_DIRS:
                continue
            if (p / f"{p.name}_schema.json").exists():
                found.append(p)
            elif level < depth:
                walk(p, level + 1)

    walk(cwd, 1)
    return found


def run(py: Path, args: list[str], cwd: Path) -> tuple[int, str]:
    try:
        r = subprocess.run([str(py), "-m", *args], cwd=cwd, capture_output=True,
                           text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return 124, f"(timed out after {TIMEOUT}s)"
    except OSError as exc:
        return 127, str(exc)
    return r.returncode, (r.stdout + r.stderr).strip()


def parse_table(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s{2}(build|solve|escalate|interpret|package)\s+(.*)$", line)
        if m:
            rows[m.group(1)] = m.group(2).strip()
    return rows


def block(reason: str) -> int:
    sys.stderr.write(reason.rstrip() + "\n")
    return 2


def plugin_root() -> Path:
    for var in ("CLAUDE_PLUGIN_ROOT", "PLUGIN_ROOT"):
        v = os.environ.get(var)
        if v and Path(v).exists():
            return Path(v)
    return Path(__file__).resolve().parent.parent


def state_dir() -> Path:
    base = os.environ.get("CLAUDE_PLUGIN_DATA") or os.environ.get("PLUGIN_DATA") \
        or tempfile.gettempdir()
    d = Path(base) / "mdp_guard"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cwd_key(cwd: Path) -> str:
    return hashlib.sha1(str(cwd).encode()).hexdigest()[:12]


def intent_file(cwd: Path) -> Path:
    return state_dir() / f"{cwd_key(cwd)}.intent"


# ------------------------------------------------------ UserPromptSubmit

def on_prompt(payload: dict, cwd: Path) -> int:
    prompt = payload.get("user_prompt") or payload.get("prompt") or ""
    if not TRIGGER.search(prompt):
        return 0
    try:
        intent_file(cwd).write_text(prompt[:2000])
    except OSError:
        pass
    skill = plugin_root() / "skills" / "mdp-solver" / "SKILL.md"
    note = (
        "auto-mdp-solver: this request is a pipeline request (formalize / build / "
        "solve a dynamic decision problem). Do not solve it directly with ad-hoc "
        "code or prose. Use the `mdp-solver` skill — Claude Code: invoke "
        "`/mdp-solver`; Codex: `$mdp-solver`; either way its instructions are "
        f"at {skill}. The pipeline's guard will not let a turn end on a "
        "completion claim until a validated `{name}/{name}_schema.json` and its "
        "restatement exist and the stage gates agree."
    )
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit", "additionalContext": note}}))
    return 0


# ----------------------------------------------------------------- Stop

def on_stop(payload: dict, cwd: Path) -> int:
    if payload.get("stop_hook_active"):
        return 0
    msg = payload.get("last_assistant_message") or ""
    if not CLAIM.search(msg):
        return 0
    if ASKS.search(msg[-600:]):
        return 0
    py = interpreter(cwd)
    folders = domain_folders(cwd)
    if not folders:
        if FORMALIZE.search(msg) or intent_file(cwd).exists():
            skill = plugin_root() / "skills" / "mdp-solver" / "SKILL.md"
            return block(
                "mdp-guard: this turn ends on a completion claim, but no "
                "`{name}/{name}_schema.json` exists under the workspace — the "
                "pipeline was requested and never entered. The deliverable is a "
                "validated IR plus a restatement produced by the mdp-solver skill "
                f"(instructions: {skill}), then the built domain and its gates; "
                "a hand-written formalization or solver is not it. Invoke the "
                "skill now (Claude Code: `/mdp-solver`; Codex: `$mdp-solver`) and "
                "run its Phase A; do not end the turn on this claim.")
        return 0
    if py is None:
        return 0

    problems: list[str] = []
    for folder in folders:
        rc, out = run(py, ["mdp_stage", str(folder)], cwd)
        rows = parse_table(out)
        build = rows.get("build", "")
        why: list[str] = []
        if "ir.validates" in build:
            why.append("its IR does not validate (`python -m mdp_ir` fails)")
        if FORMALIZE.search(msg) and "restatement.exists" in build:
            why.append("no restatement file exists")
        if SOLVE.search(msg) and rows.get("interpret", "").startswith("BLOCKED"):
            # a "trained"/"leaderboard" claim needs solve's *exit*: a run plan
            # and RL artifacts — which is interpret's entry
            why.append(f"no solve exit on record ({rows['interpret']})")
        if PACKAGE.search(msg) and rows.get("package", "").startswith("BLOCKED"):
            why.append(f"package's entry is blocked ({rows['package']})")
        if rc not in (0,) and not rows:
            why.append(f"`mdp_stage` could not read the folder: {out[-400:]}")
        if why:
            problems.append(f"{folder.name}/: " + "; ".join(why) + "\n" + out)

    if not problems:
        return 0
    return block(
        "mdp-guard: this turn ends on a completion claim that the pipeline's own "
        "gate contradicts.\n\n" + "\n\n".join(problems) +
        "\n\nResume the pipeline at the first blocked op (the step skills' entry "
        "gates say what each needs). Do not report success the table does not show; "
        "if a gate cannot be passed, say that plainly instead.")


# ----------------------------------------------------------- PostToolUse

def state_file(cwd: Path) -> Path:
    return state_dir() / f"{cwd_key(cwd)}.json"


def on_post_tool(payload: dict, cwd: Path) -> int:
    py = interpreter(cwd)
    if py is None:
        return 0
    folders = domain_folders(cwd)
    if not folders:
        return 0
    sf = state_file(cwd)
    try:
        seen = json.loads(sf.read_text())
    except (OSError, ValueError):
        seen = {}
    failures: list[str] = []
    changed = False
    for folder in folders:
        schema = folder / f"{folder.name}_schema.json"
        try:
            mtime = schema.stat().st_mtime
        except OSError:
            continue
        key = str(schema)
        if seen.get(key) == mtime:
            continue
        rc, out = run(py, ["mdp_ir", str(schema)], cwd)
        seen[key] = mtime
        changed = True
        if rc != 0:
            failures.append(f"{schema.relative_to(cwd)}\n{out[-2500:]}")
    if changed:
        try:
            sf.write_text(json.dumps(seen))
        except OSError:
            pass
    if not failures:
        return 0
    return block(
        "mdp-guard: a schema you just changed does not validate. Fix it before "
        "anything else; nothing downstream can start from an invalid IR.\n\n" +
        "\n\n".join(failures))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    cwd = Path(payload.get("cwd") or os.getcwd())
    event = payload.get("hook_event_name", "")
    if event == "Stop":
        return on_stop(payload, cwd)
    if event == "PostToolUse":
        return on_post_tool(payload, cwd)
    if event == "UserPromptSubmit":
        return on_prompt(payload, cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())

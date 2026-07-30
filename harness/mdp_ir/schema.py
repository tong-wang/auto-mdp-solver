"""Pydantic schema for the MDP intermediate representation (IR), v0.4.

The IR is split into three layers that mirror the codebase's own layering
(spec §1.1) and the Phase-B pipeline (SKILL.md Phase B), with dependencies pointing
strictly downward (gym → mdp, rl → gym/mdp; mdp references nothing above it):

  * ``mdp``  — the *problem*: state, decisions, uncertainty, dynamics,
    economics, scenario constants. These calls have no oracle, so judgment
    fields use ``Confirmable[T]`` and the block is what the Phase-A gate
    freezes (see ``MdpIR.mdp_fingerprint``). Drives Stage 1 codegen
    (``_uncertainty`` / ``_scenarios`` / ``_mdp``).
  * ``gym``  — the *interface menus*: observation / action / reward modes and
    termination semantics. Empirically testable design axes, revisable in
    Phase B without re-confirmation. Drives Stage 2 (``_gym``).
  * ``rl``   — the *training configuration*: policy class (memory), algorithm,
    obs-normalization decision. Drives Stage 4.

Cross-layer validators live on the root and enforce exactly the downward
edges: gym features may reference mdp names, rl reads gym/mdp, never the
reverse.

Naming is validated: every identifier appearing in an expression (dynamics
updates, objective components, observation features, reward modes, action
transforms, guards, triggers) must resolve to a declared state variable, info
field, decision, scenario constant, uncertainty source/stage, or builtin.
Seed keys are *derived* from ``realization`` + ``entity_id_in_seed`` (spec
§6.3), never hand-written.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Confirmable<T> — the no-oracle-field convention (sample §6)
# ---------------------------------------------------------------------------

T = TypeVar("T")


class Source(str, Enum):
    derived = "derived"                  # agent proposal, not yet reviewed
    human_confirmed = "human_confirmed"  # human reviewed and kept the proposal
    human_override = "human_override"    # human reviewed and changed the value


class Confirmable(BaseModel, Generic[T]):
    """An agent-proposed value the human owns the final call on.

    Invariants: ``human_override`` iff ``value != suggested``; ``derived`` and
    ``human_confirmed`` both require ``value == suggested`` and differ only in
    whether the human has reviewed the proposal. The Phase-A gate requires no
    field left at ``derived`` (see ``MdpIR.unconfirmed``).
    """

    model_config = ConfigDict(extra="forbid")

    value: T
    suggested: T
    source: Source = Source.derived
    rationale: str = ""

    @model_validator(mode="after")
    def _check_source_invariant(self) -> "Confirmable[T]":
        agrees = self.value == self.suggested
        if self.source is Source.human_override and agrees:
            raise ValueError(
                f"source=human_override requires value != suggested (both are {self.value!r})"
            )
        if self.source is not Source.human_override and not agrees:
            raise ValueError(
                f"source={self.source.value} requires value == suggested "
                f"(got value={self.value!r}, suggested={self.suggested!r})"
            )
        return self


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PeriodIndexing(str, Enum):
    zero_based = "0-based"
    one_based = "1-based"


class EntityKind(str, Enum):
    single = "single"
    multi = "multi"


class StateRole(str, Enum):
    core = "core"
    time_index = "time_index"


class Observability(str, Enum):
    observable = "observable"
    latent = "latent"


class Placement(str, Enum):
    state = "state"
    info = "info"


class DecisionType(str, Enum):
    continuous = "continuous"
    discrete = "discrete"


class Feasibility(str, Enum):
    non_negative = "non_negative"
    simplex = "simplex"
    integer = "integer"


class Realization(str, Enum):
    episode = "episode"   # drawn once per episode; derived seed key omits `period`
    period = "period"     # drawn each period;      derived seed key includes `period`
    event = "event"       # decision-triggered;     period-keyed, trigger-gated
    keyed = "keyed"       # keyed by expression values (stage.key_exprs), no `period`
                          # slot: a fixed per-episode latent table indexed by the
                          # key values, so the same keys realize the same outcomes
                          # regardless of when they are drawn (batch semantics /
                          # common random numbers)
    # `key_exprs` composes with `period`/`event` too: there the key values
    # *refine* the period slot — one independent stream per key-tuple per
    # period, of which the episode reads one (e.g. a bandit's per-arm payout
    # streams, keyed on the chosen arm). On `keyed` they *replace* it.


class Sense(str, Enum):
    minimize = "minimize"
    maximize = "maximize"


class InvariantScope(str, Enum):
    period = "period"        # checked on every row
    terminal = "terminal"    # checked only on the row the episode ends on


class HorizonEnd(str, Enum):
    # Gymnasium semantics: `terminated` when T is part of the problem (no
    # bootstrapping past it — the finite-horizon default); `truncated` only
    # for external time limits imposed on a longer-lived problem.
    terminated = "terminated"
    truncated = "truncated"


class FeasibilityStrategy(str, Enum):
    none = "none"
    clip = "clip"                     # continuous: clip into bounds (spec §7.2)
    mask = "mask"                     # discrete: action masking (spec §7.1)
    reparametrize = "reparametrize"   # continuous: transform maps into the feasible set


class Algo(str, Enum):
    ppo = "ppo"
    maskable_ppo = "maskable_ppo"
    recurrent_ppo = "recurrent_ppo"


# ---------------------------------------------------------------------------
# Expression tokenization (identifier validation)
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMMENT = re.compile(r"#.*")
_STRLIT = re.compile(r"'[^']*'|\"[^\"]*\"")   # single/double-quoted string literals
# numeric literals, INCLUDING the exponent form: `1e-9` would otherwise
# tokenize as the identifier `e` and `1e6` as `e6`. The lookbehind keeps this
# from biting into a name like `x1e5`.
_NUMLIT = re.compile(r"(?<![A-Za-z_0-9.])(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
# an assignment `=` or draw `~`, excluding ==, !=, <=, >=, +=, -=, *=, /=
_ASSIGN = re.compile(r"(?<![=!<>+\-*/])=(?!=)|~")
# `prev.<name>` — the previous period's end-of-period value, legal only in
# invariant expressions (see Invariant)
_PREV = re.compile(r"\bprev\.([A-Za-z_]\w*)")

# Core builtins usable in any IR expression. Domain-specific functions do NOT
# belong here: a domain declares them in its IR under `mdp.expr_builtins`
# (see ExprBuiltin) and ships the implementation module next to the IR
# (portable-domain contract); the interpreter resolves them lazily.
_BUILTINS = frozenset({
    "if", "else", "not", "and", "or", "in", "for",
    "min", "max", "sum", "abs", "len", "round", "int", "float",
    "exp", "log", "sqrt", "floor", "ceil", "zeros",
    "phi", "topk", "range",    # std-normal CDF, top-k largest values,
                               # comprehension index ranges
    "close",                   # close(a, b[, tol]) — float-tolerant equality,
                               # the balance-law idiom in `mdp.invariants`
    "T",                       # horizon length
    "True", "False", "None", "true", "false",
})


def _identifiers(expr: str) -> set[str]:
    # Strip string literals before comments (so a `#` inside a string is not
    # mistaken for a comment start), then numeric literals, then extract
    # identifiers. A string's CONTENTS are data, never names — this lets
    # expressions compare a categorical constant against a literal, e.g.
    # `mmfe_mode == 'additive'`, without the literal's characters being
    # flagged as unresolved identifiers. Numeric literals are stripped for the
    # exponent form's sake: `1e-9` is one number, not the name `e`.
    return set(_IDENT.findall(
        _NUMLIT.sub(" ", _COMMENT.sub("", _STRLIT.sub(" ", expr)))
    ))


def _check_expr(expr: str, known: set[str], where: str) -> None:
    unknown = _identifiers(expr) - known - _BUILTINS
    if unknown:
        raise ValueError(
            f"{where}: unresolved identifier(s) {sorted(unknown)} in expr {expr!r}"
        )


# ---------------------------------------------------------------------------
# Structural blocks
# ---------------------------------------------------------------------------


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Domain(_Base):
    name: str
    class_prefix: str
    one_line: str = ""
    problem_class: str = ""


# ----- mdp block -----------------------------------------------------------


class Horizon(_Base):
    # v1 scope is finite-horizon only (SKILL.md scope check); an infinite/average-reward
    # variant would add a `kind` discriminator here.
    # T: a literal length, or the *name of a scenario constant* — the horizon
    # is a scenario dimension like any other numerical attribute and may vary
    # per instance. Resolve with `MdpBlock.horizon_T(instance)`.
    T: int | str
    period_indexing: PeriodIndexing = PeriodIndexing.zero_based

    @model_validator(mode="after")
    def _check_T(self) -> "Horizon":
        if isinstance(self.T, int) and self.T < 1:
            raise ValueError(f"horizon.T must be >= 1, got {self.T}")
        return self


class EntityStructure(_Base):
    kind: EntityKind = EntityKind.single
    entity_id_in_seed: bool = False


class StateVariable(_Base):
    name: str
    role: StateRole
    type: str
    observability: Observability = Observability.observable
    bounds: list[float] | None = None
    # a literal length, or the name of a scenario constant (overridable per
    # instance, like horizon.T) — e.g. a pipeline whose length tracks the
    # selected lead time. Metadata only: the runtime vector comes from
    # initial_state (`zeros(<len>)`), so nothing downstream must resolve it.
    length: int | str | None = None
    element_bounds: list[float] | None = None
    categories: list[str] | None = None
    desc: str = ""
    # borderline state-vs-info calls carry a Confirmable placement (sample §5.1)
    placement: Confirmable[Placement] | None = None


class InfoField(_Base):
    name: str
    type: str
    desc: str = ""
    components: list[str] | None = None      # for type == "decomposition"
    placement: Confirmable[Placement] | None = None


class Decision(_Base):
    name: str
    type: Confirmable[DecisionType]
    dim: int = Field(ge=1)
    # [lo, hi], applied to every dim (per-dim bounds are out of v1 scope).
    # An entry may be a number or the *name of a scenario constant* — like
    # horizon.T, bounds are a scenario dimension and may vary per instance.
    # Resolve with `MdpBlock.decision_bounds`.
    bounds: Confirmable[list[float | str]]
    feasibility: list[Feasibility] = Field(default_factory=list)
    desc: str = ""

    @model_validator(mode="after")
    def _check_bounds_shape(self) -> "Decision":
        for tag, b in (("value", self.bounds.value), ("suggested", self.bounds.suggested)):
            if len(b) != 2:
                raise ValueError(
                    f"decision {self.name!r}: bounds.{tag} must be [lo, hi], got {b}"
                )
            if all(isinstance(x, (int, float)) for x in b) and b[0] >= b[1]:
                raise ValueError(
                    f"decision {self.name!r}: bounds.{tag} must satisfy lo < hi, got {b}"
                )
        return self


class ScenarioConstant(_Base):
    """A named constant of the problem *instance* — the single home for every
    number that would otherwise be duplicated across distributions, objective
    exprs, dynamics, and initial state. `axis` tags it as a scenario dimension
    (other instances vary it); "" marks a structural constant."""

    name: str
    # a literal value merged into the expression namespace as-is (never
    # re-evaluated as an expression). `str` supports categorical constants —
    # a mode/link selector compared in a guard/update, e.g. `mmfe_mode`
    # tested by `mmfe_mode == 'additive'`.
    value: float | int | bool | str | list
    axis: str = ""
    desc: str = ""


class SampledConstant(_Base):
    """One world-latent draw inside a ScenarioSampler: realizes the named
    scenario constant at episode start. The constant must already exist in
    ``scenario.constants`` — its static value is the placeholder/example, so
    every expr and validator sees the same namespace either way."""

    name: str
    distribution: "Distribution"


class ScenarioSampler(_Base):
    """World-latent sampler (spec §5.2): nature's per-episode draw, realized
    once at episode start on the meta branch of the seed tree and written
    into scenario constants. Draws execute **in order from one rng** seeded
    by the sampler's meta key, so multi-draw recipes (a support then its
    weights) are bit-reproducible."""

    name: str
    substream_id: int = 0                    # child id under branch 0 (§6.3)
    draws: list[SampledConstant] = Field(min_length=1)
    # hidden latents (policy must infer) suggest memory and are barred from
    # observation exprs; observed latents (a forecast) are contextual inputs
    hidden: bool = True
    # instance names this sampler applies to; "" names the base constants;
    # empty list = every instance incl. base
    instances: list[str] = Field(default_factory=list)
    desc: str = ""


class ScenarioMixture(_Base):
    """World mixture (spec §5.3): nature picks which component instance runs,
    once per episode. Weights are a modeling commitment. Components name
    scenario instances ("" = the base constants); a component's samplers then
    apply with the episode seed delegated verbatim (standalone equivalence).
    Usable anywhere an instance name is (``--instance``, gates)."""

    name: str
    substream_id: int = 0                    # child id under branch 0 (§6.3)
    components: list[tuple[float, str]] = Field(min_length=2)
    desc: str = ""


class Scenario(_Base):
    constants: list[ScenarioConstant]
    # named alternative instances: overrides of constants by name -> _scenarios.py
    # (slot-valued keys select a candidate; the rest override constant values,
    # `str` included for categorical selectors like mmfe_mode='multiplicative')
    instances: dict[str, dict[str, float | int | bool | str | list]] = Field(default_factory=dict)
    # world layer (spec §5.2, §5.3): samplers and mixtures; design-layer grids
    # live OUTSIDE this node (MdpIR.grids), mirroring the two-layer split
    samplers: list[ScenarioSampler] = Field(default_factory=list)
    mixtures: list[ScenarioMixture] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_instances(self) -> "Scenario":
        names = [c.name for c in self.constants]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate scenario constant names in {names}")
        known = set(names)
        for inst, overrides in self.instances.items():
            bad = set(overrides) - known
            if bad:
                raise ValueError(
                    f"scenario instance {inst!r} overrides unknown constant(s) {sorted(bad)}"
                )
        # world-layer drawers: draw targets exist; instance refs exist; names
        # and meta substream ids unique (children of branch 0, spec §6.3)
        drawer_names = [s.name for s in self.samplers] + [m.name for m in self.mixtures]
        clash = set(drawer_names) & set(self.instances)
        if clash:
            raise ValueError(f"sampler/mixture name(s) collide with instances: {sorted(clash)}")
        if len(drawer_names) != len(set(drawer_names)):
            raise ValueError(f"duplicate sampler/mixture names in {drawer_names}")
        subs: dict[int, list[str]] = {}
        for s in self.samplers:
            subs.setdefault(s.substream_id, []).append(s.name)
            bad = {d.name for d in s.draws} - known
            if bad:
                raise ValueError(
                    f"sampler {s.name!r} draws unknown constant(s) {sorted(bad)}"
                )
            bad_inst = set(s.instances) - set(self.instances) - {""}
            if bad_inst:
                raise ValueError(
                    f"sampler {s.name!r} references unknown instance(s) {sorted(bad_inst)}"
                )
        for m in self.mixtures:
            subs.setdefault(m.substream_id, []).append(m.name)
            for w, comp in m.components:
                if w <= 0:
                    raise ValueError(f"mixture {m.name!r}: weight {w} must be > 0")
                if comp and comp not in self.instances:
                    raise ValueError(
                        f"mixture {m.name!r} references unknown instance {comp!r}"
                    )
        dupes = {k: v for k, v in subs.items() if len(v) > 1}
        if dupes:
            raise ValueError(f"meta substream_id shared by drawers: {dupes}")
        return self

    @property
    def sampled_constant_names(self) -> set[str]:
        return {d.name for s in self.samplers for d in s.draws}

    @property
    def hidden_sampled_names(self) -> set[str]:
        return {d.name for s in self.samplers if s.hidden for d in s.draws}


class Distribution(_Base):
    family: str
    # values: literals (number/list) or expr strings over scenario constants,
    # decisions, and state (e.g. "a * exp(-alpha * price) * dt")
    settings: dict[str, float | int | str | list] = Field(default_factory=dict)
    note: str = ""


class UncertaintyStage(_Base):
    name: str
    realization: Realization
    sub_stream: int | None = None
    trigger: str | None = None               # e.g. "O && order>0"; event stages only
    key_exprs: list[str] | None = None       # int-valued exprs prepended to the seed
                                             # key (extension 2026-07-13). Required on
                                             # keyed stages (replaces the period slot);
                                             # optional on period/event stages, where
                                             # they refine it: one independent stream
                                             # per key-tuple per period. The exprs may
                                             # read decisions — they select *which*
                                             # exogenous stream is read, never what it
                                             # contains, so path-independence holds.

    @model_validator(mode="after")
    def _check_trigger(self) -> "UncertaintyStage":
        if self.realization is Realization.event and not self.trigger:
            raise ValueError(f"stage {self.name!r}: event realization requires a trigger")
        if self.realization is not Realization.event and self.trigger:
            raise ValueError(
                f"stage {self.name!r}: trigger only applies to event realization"
            )
        if self.realization is Realization.keyed and not self.key_exprs:
            raise ValueError(f"stage {self.name!r}: keyed realization requires key_exprs")
        if self.realization is Realization.episode and self.key_exprs:
            raise ValueError(
                f"stage {self.name!r}: key_exprs on an episode-realization stage — "
                "that is exactly keyed realization; declare the stage keyed"
            )
        return self

    def seed_key(
        self, stream_id: int, entity_id_in_seed: bool, scheme: str = "v1"
    ) -> list[str]:
        """Derived symbolic seed key, reverse-tree order (most specific first).
        Never hand-written: `realization` fixes the `period` slot, the entity
        structure fixes the `entity_id` slot.

        **v1** (frozen legacy grammar) keys stream *before* period:
        ``[entity_id?, expr:..., sub_stream?, stream_id, period?,
        episode_seed, seed_salt]`` — matching the pre-redesign reference
        generators. The ``stream:S.U`` symbol expands numerically to ``[U, S]``.

        **v2** (spec §6.3 seed tree, leaf-first) keys the intrinsic branch:
        ``[entity_id?, expr:..., sub_stream?, period?, stream_id, 1,
        episode_seed, seed_salt]`` — period below the source, branch word 1
        above it; ``stream_id`` is the per-instance source id (may be 0)."""
        key: list[str] = []
        if entity_id_in_seed:
            key.append("entity_id")
        if self.key_exprs:
            key += [f"expr:{e}" for e in self.key_exprs]
        has_period = self.realization not in (Realization.episode, Realization.keyed)
        if scheme == "v2":
            if self.sub_stream is not None:
                key.append(f"sub:{self.sub_stream}")
            if has_period:
                key.append("period")
            key += [f"source:{stream_id}", "branch:1"]
        else:
            key.append(
                f"stream:{stream_id}" if self.sub_stream is None
                else f"stream:{stream_id}.{self.sub_stream}"
            )
            if has_period:
                key.append("period")
        key += ["episode_seed", "seed_salt"]
        return key


class UncertaintySource(_Base):
    name: str
    generator: str
    # v1: >= 1 (0 reserved for scenario sampling); v2: the per-instance source
    # id under branch 1 — may be 0. Enforced scheme-aware at the root.
    stream_id: int = Field(ge=0)
    is_discrete: bool | None = None
    latent: bool = False
    distribution: Distribution
    stages: list[UncertaintyStage] = Field(min_length=1)


class TransitionStep(_Base):
    event: str
    guard: str | None = None
    updates: list[str]


class Dynamics(_Base):
    # event_sequence: a literal list of event names, or the *name of a
    # scenario constant* whose value is such a list — the event order is then
    # a scenario dimension like horizon.T, overridable per instance
    # (event-order variants are instances, IR_LAYERING_PLAN §10f). Resolve
    # with `MdpBlock.event_sequence(instance)`; the interpreter executes
    # transitions in the resolved order. A constant referenced here is in
    # control position (§10d): a structural parameter whose used settings
    # each need differential coverage.
    event_sequence: list[str] | str
    observation_point: str = ""
    transitions: list[TransitionStep]

    @model_validator(mode="after")
    def _check_events(self) -> "Dynamics":
        if isinstance(self.event_sequence, str):
            return self  # resolved + validated at the MdpBlock level
        known = set(self.event_sequence) | {"END_OF_PERIOD"}
        pos = {e: i for i, e in enumerate(self.event_sequence)}
        last = -1
        for t in self.transitions:
            if t.event not in known:
                raise ValueError(
                    f"transition event {t.event!r} not in event_sequence {self.event_sequence}"
                )
            # execution follows the event order; require declaration order to
            # agree so a literal-sequence IR reads exactly as it runs
            p = pos.get(t.event, len(pos))
            if p < last:
                raise ValueError(
                    f"transitions must be declared in event_sequence order "
                    f"{self.event_sequence}; {t.event!r} appears after a later event"
                )
            last = p
        return self


class ObjectiveComponent(_Base):
    # exprs reference scenario constants by name; evaluated at END_OF_PERIOD
    # with state at its end-of-period values. No inline coefficient dicts —
    # every number lives in scenario.constants.
    name: str
    expr: str
    desc: str = ""


class Objective(_Base):
    """Per-step economic decomposition — mdp-layer facts that travel in `info`
    (spec §6.4). Reward *construction* is the gym block's `reward_modes`."""

    sense: Sense
    # β — the problem's intrinsic discount factor (time value of money,
    # continuation probability), asked for in the Phase-A interview. Part of
    # the OBJECTIVE, never tuned: eval and every baseline score
    # J = E[Σ β^t r_t]; training gamma = β is the faithful default (γ < β
    # only as a logged bias-variance escalation, γ > β never — see
    # SOLVE_LEVELS_PLAN §2). Never pre-discount env rewards: β enters at
    # eval time and as the solver's gamma, keeping per-step rewards
    # stationary and the differential contract intact. 1.0 = undiscounted.
    discount_factor: float = Field(default=1.0, gt=0.0, le=1.0)
    per_step_components: list[ObjectiveComponent] = Field(min_length=1)

    @property
    def total_expr(self) -> str:
        return " + ".join(c.name for c in self.per_step_components)


class Invariant(_Base):
    """A claim about the problem that must hold on every trajectory.

    Invariants exist because the differential gate proves the interpreter and
    the generated domain *agree*, not that either is *right*: a
    mis-formalization (wrong sign, a dropped term) makes both sides wrong
    identically and the gate still passes. An invariant is transcribed from
    the user's own description at Phase A — an independent statement about
    the problem, so it can catch the error the twin shares.

    ``expr`` is a boolean over the END_OF_PERIOD namespace (state, info,
    decomposition components, decisions, scenario constants), plus
    ``prev.<name>`` for the previous period's end-of-period value of any
    per-period name. On the first period ``prev`` exposes the initial state,
    with info fields, decisions and components at 0 (and ``prev.t`` one
    before the indexing origin).

    ``t`` is the row's INPUT period — the period the decision was taken in.
    Prefer it to the time-index state variable, which END_OF_PERIOD has
    already advanced by the time invariants (like objective components) are
    evaluated: on a horizon-N terminal row ``t == N`` while ``period == N+1``.

    Use ``close(a, b)`` rather than ``==`` for float balance laws::

        close(inventory, prev.inventory + received - demand)
        close(sum(pipeline), sum(prev.pipeline) + order - received)

    Declared invariants are STRUCTURAL: editing one moves the IR's structural
    fingerprint and re-opens the Phase-A confirmation, exactly like editing
    dynamics. Violations are collected onto the trajectory rather than raised,
    so Phase A can report them advisory while the Stage-1 gate fails on them.
    """

    name: str
    expr: str
    scope: InvariantScope = InvariantScope.period
    desc: str = ""


class MetricReduce(str, Enum):
    last = "last"    # the final period's value (terminal read)
    sum = "sum"      # accumulated over the episode
    max = "max"      # episode maximum
    min = "min"      # episode minimum


class EvalMetric(_Base):
    """A bystander metric: it describes, it never decides.

    Gates, crowns, and model selection stay on the objective; eval metrics
    are extra per-episode report columns (spec §9: emitted *after* the gate
    metric so ``mdp_gates``'s first-``*_mean`` convention is undisturbed,
    plus a column in the per-seed sidecar). Typical origin: an objective
    candidate the human declined at Phase A but wants to keep seeing —
    record that in ``source``.

    ``expr`` is evaluated each period on the END_OF_PERIOD namespace (same
    as objective components and invariants) and folded across the episode
    by ``reduce``. Because a metric never feeds the policy, it is one of the
    three legal homes for latent references on the eval path (the others:
    terminal rewards and action-independent normalizers).

    Declared at the IR root, deliberately OUTSIDE the mdp block: adding or
    editing a metric never moves ``mdp_fingerprint()``, so metrics are
    appendable after the Phase-A freeze without re-opening confirmation."""

    name: str
    expr: str
    reduce: MetricReduce = MetricReduce.last
    desc: str = ""
    # provenance: e.g. "objective candidate 'max', declined at Phase A"
    source: str = ""

    @model_validator(mode="after")
    def _check_name(self) -> "EvalMetric":
        if not _IDENT.fullmatch(self.name):
            raise ValueError(f"eval_metrics name {self.name!r} is not an identifier")
        return self


class ExprBuiltin(_Base):
    """A domain-owned expression builtin, declared by the IR that needs it.

    ``name`` becomes callable in this IR's expressions (validated alongside
    the core ``_BUILTINS``); the implementation is ``def {name}`` in
    ``{module}.py`` shipped next to the IR (portable-domain contract), which
    the interpreter resolves lazily on first call. This keeps ``mdp_ir``
    free of domain knowledge: no domain function names or module paths are
    hard-coded in the schema or interpreter."""

    name: str
    module: str
    desc: str = ""

    @model_validator(mode="after")
    def _check_names(self) -> "ExprBuiltin":
        for tag, v in (("name", self.name), ("module", self.module)):
            if not _IDENT.fullmatch(v):
                raise ValueError(f"expr_builtins {tag} {v!r} is not an identifier")
        if self.name in _BUILTINS:
            raise ValueError(
                f"expr_builtins name {self.name!r} shadows a core builtin"
            )
        return self


class MdpBlock(_Base):
    """The problem. Frozen at the Phase-A gate; Stage-1 codegen input."""

    horizon: Horizon
    entity_structure: EntityStructure
    state_variables: list[StateVariable] = Field(min_length=1)
    info_fields: list[InfoField] = Field(default_factory=list)
    decisions: list[Decision] = Field(min_length=1)
    uncertainty_sources: list[UncertaintySource] = Field(min_length=1)
    dynamics: Dynamics
    objective: Objective
    scenario: Scenario
    # conservation/consistency claims checked on every trajectory (Invariant):
    # the one artifact that can catch a mis-formalization the differential's
    # two sides share. Every instance and every catalog candidate inherits them
    invariants: list[Invariant] = Field(default_factory=list)
    # domain-owned expression builtins (ExprBuiltin): extra callables this
    # IR's expressions may use, implemented in modules next to the IR
    expr_builtins: list[ExprBuiltin] = Field(default_factory=list)
    # per-variable value: literal, or expr over scenario constants/builtins
    # (e.g. "zeros(6)", "n0"); time_index vars default to the indexing origin
    initial_state: dict[str, float | int | str | list] = Field(default_factory=dict)

    # -- namespaces -----------------------------------------------------

    @property
    def value_names(self) -> set[str]:
        """Names an expression may reference: state, info (incl. decomposition
        components), decisions, scenario constants."""
        names = {sv.name for sv in self.state_variables}
        names |= {f.name for f in self.info_fields}
        for f in self.info_fields:
            if f.components:
                names |= set(f.components)
        names |= {d.name for d in self.decisions}
        names |= {c.name for c in self.scenario.constants}
        return names

    @property
    def sampling_names(self) -> set[str]:
        """Names usable only in `x ~ source.stage` draws and triggers."""
        names = {src.name for src in self.uncertainty_sources}
        for src in self.uncertainty_sources:
            names |= {st.name for st in src.stages}
        return names

    @property
    def per_period_names(self) -> set[str]:
        """Names whose value changes period to period, so `prev.<name>` in an
        invariant means something: state, info (incl. decomposition
        components), decisions. Scenario constants are excluded — they are
        fixed for the episode, so `prev.h` would be `h`."""
        names = {sv.name for sv in self.state_variables}
        names |= {f.name for f in self.info_fields}
        for f in self.info_fields:
            if f.components:
                names |= set(f.components)
        names |= {d.name for d in self.decisions}
        return names | {"t"}

    @property
    def latent_names(self) -> set[str]:
        return {
            sv.name for sv in self.state_variables
            if sv.observability is Observability.latent
        }

    @property
    def builtin_names(self) -> set[str]:
        """Names of the domain-owned builtins this IR declares."""
        return {b.name for b in self.expr_builtins}

    def decision_bounds(
        self, name: str, instance: str | None = None
    ) -> tuple[float, float]:
        """Effective [lo, hi] of a decision: entries are literals or names of
        scenario constants, overridable per instance (like `horizon_T`)."""
        d = next(dd for dd in self.decisions if dd.name == name)
        consts = {c.name: c.value for c in self.scenario.constants}
        if instance is not None:
            consts.update(self.scenario.instances[instance])
        out = []
        for x in d.bounds.value:
            v = consts[x] if isinstance(x, str) else x
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(
                    f"decision {name!r} bound {x!r} resolved to non-numeric {v!r}"
                )
            out.append(float(v))
        return out[0], out[1]

    def horizon_T(self, instance: str | None = None) -> int:
        """Effective horizon length: `horizon.T` is a literal or the name of
        a scenario constant, overridable per instance like any constant."""
        t = self.horizon.T
        if isinstance(t, int):
            return t
        consts = {c.name: c.value for c in self.scenario.constants}
        if instance is not None:
            consts.update(self.scenario.instances[instance])
        v = consts[t]
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"horizon.T constant {t!r} resolved to non-int {v!r}")
        return v

    def event_sequence(self, instance: str | None = None) -> list[str]:
        """Effective event order: `dynamics.event_sequence` is a literal list
        or the name of a scenario constant, overridable per instance like
        `horizon.T`. The interpreter executes transitions in this order."""
        seq = self.dynamics.event_sequence
        if not isinstance(seq, str):
            return list(seq)
        consts = {c.name: c.value for c in self.scenario.constants}
        if instance is not None:
            consts.update(self.scenario.instances[instance])
        return list(consts[seq])

    # -- layer-local invariants ------------------------------------------

    @model_validator(mode="after")
    def _unique_value_names(self) -> "MdpBlock":
        seen: dict[str, str] = {}
        groups = [
            ("state", [sv.name for sv in self.state_variables]),
            ("info", [f.name for f in self.info_fields]),
            ("decision", [d.name for d in self.decisions]),
            ("constant", [c.name for c in self.scenario.constants]),
            ("expr_builtin", [b.name for b in self.expr_builtins]),
        ]
        for kind, names in groups:
            for n in names:
                if n in seen:
                    raise ValueError(
                        f"name {n!r} declared as both {seen[n]} and {kind}; "
                        "the value namespace must be collision-free"
                    )
                seen[n] = kind
        return self

    @model_validator(mode="after")
    def _check_horizon(self) -> "MdpBlock":
        t = self.horizon.T
        if isinstance(t, int):
            return self
        consts = {c.name: c.value for c in self.scenario.constants}
        if t not in consts:
            raise ValueError(
                f"horizon.T {t!r} must be a literal int or name a scenario constant"
            )
        values = {"base": consts[t]}
        for inst, overrides in self.scenario.instances.items():
            if t in overrides:
                values[f"instance {inst!r}"] = overrides[t]
        for where, v in values.items():
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(
                    f"horizon.T constant {t!r} must be an int >= 1 everywhere; "
                    f"{where} has {v!r}"
                )
        return self

    @model_validator(mode="after")
    def _check_event_sequence(self) -> "MdpBlock":
        """`dynamics.event_sequence` naming a constant (event-order variants
        as instances, §10d structural parameter): the constant must exist and
        resolve — in the base scenario and every instance — to a list of
        unique event-name strings (END_OF_PERIOD is implicit, never listed)
        covering every non-END transition event."""
        seq = self.dynamics.event_sequence
        if not isinstance(seq, str):
            return self
        consts = {c.name: c.value for c in self.scenario.constants}
        if seq not in consts:
            raise ValueError(
                f"dynamics.event_sequence {seq!r} must be a literal list or "
                "name a scenario constant"
            )
        needed = {t.event for t in self.dynamics.transitions} - {"END_OF_PERIOD"}
        values = {"base": consts[seq]}
        for inst, overrides in self.scenario.instances.items():
            if seq in overrides:
                values[f"instance {inst!r}"] = overrides[seq]
        for where, v in values.items():
            if (not isinstance(v, list) or not v
                    or any(not isinstance(e, str) for e in v)):
                raise ValueError(
                    f"event_sequence constant {seq!r} must be a non-empty "
                    f"list of event names everywhere; {where} has {v!r}"
                )
            if len(v) != len(set(v)):
                raise ValueError(
                    f"event_sequence constant {seq!r} repeats an event in {where}: {v}"
                )
            if "END_OF_PERIOD" in v:
                raise ValueError(
                    f"event_sequence constant {seq!r} must not list "
                    f"END_OF_PERIOD (implicit, always last); {where} has {v}"
                )
            missing = needed - set(v)
            if missing:
                raise ValueError(
                    f"event_sequence constant {seq!r} in {where} is missing "
                    f"transition event(s) {sorted(missing)}: {v}"
                )
        return self

    @model_validator(mode="after")
    def _decision_bounds_resolve(self) -> "MdpBlock":
        """String bound entries must name scenario constants and resolve to
        numeric lo < hi in the base scenario and in every instance."""
        consts = {c.name: c.value for c in self.scenario.constants}
        for d in self.decisions:
            names = [x for x in d.bounds.value + d.bounds.suggested if isinstance(x, str)]
            unknown = [x for x in names if x not in consts]
            if unknown:
                raise ValueError(
                    f"decision {d.name!r}: bound entries {unknown} name no scenario constant"
                )
            if not names:
                continue
            scopes = {"base": {}} | {
                f"instance {i!r}": ov for i, ov in self.scenario.instances.items()
            }
            for where, overrides in scopes.items():
                merged = consts | overrides
                lo, hi = (
                    merged[x] if isinstance(x, str) else x for x in d.bounds.value
                )
                for tag, v in (("lo", lo), ("hi", hi)):
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        raise ValueError(
                            f"decision {d.name!r}: bound {tag} resolves to "
                            f"non-numeric {v!r} in {where}"
                        )
                if lo >= hi:
                    raise ValueError(
                        f"decision {d.name!r}: bounds resolve to lo >= hi "
                        f"({lo} >= {hi}) in {where}"
                    )
        return self

    @model_validator(mode="after")
    def _unique_stream_ids(self) -> "MdpBlock":
        ids = [s.stream_id for s in self.uncertainty_sources]
        if len(ids) != len(set(ids)):
            raise ValueError(f"uncertainty stream_ids must be unique, got {ids}")
        return self

    @model_validator(mode="after")
    def _placement_matches_list(self) -> "MdpBlock":
        for sv in self.state_variables:
            if sv.placement and sv.placement.value is not Placement.state:
                raise ValueError(
                    f"state variable {sv.name!r} has placement "
                    f"{sv.placement.value.value!r}; must be 'state'"
                )
        for f in self.info_fields:
            if f.placement and f.placement.value is not Placement.info:
                raise ValueError(
                    f"info field {f.name!r} has placement "
                    f"{f.placement.value.value!r}; must be 'info'"
                )
        return self

    @model_validator(mode="after")
    def _initial_state_covers_core(self) -> "MdpBlock":
        state_names = {sv.name for sv in self.state_variables}
        bad = set(self.initial_state) - state_names
        if bad:
            raise ValueError(f"initial_state sets unknown state variable(s) {sorted(bad)}")
        missing = {
            sv.name for sv in self.state_variables
            if sv.role is StateRole.core and sv.name not in self.initial_state
        }
        if missing:
            raise ValueError(
                f"initial_state must cover every core state variable; missing {sorted(missing)}"
            )
        return self

    @model_validator(mode="after")
    def _objective_decomposition_consistent(self) -> "MdpBlock":
        """Economics have one home: objective components. A decomposition info
        field, if present, must carry exactly those components (+ 'total')."""
        comp_names = {c.name for c in self.objective.per_step_components}
        for f in self.info_fields:
            if f.type == "decomposition":
                declared = set(f.components or [])
                expected = comp_names | {"total"}
                if declared != expected:
                    raise ValueError(
                        f"decomposition info field {f.name!r} components {sorted(declared)} "
                        f"must equal objective components + 'total' {sorted(expected)}"
                    )
        return self

    @model_validator(mode="after")
    def _check_expressions(self) -> "MdpBlock":
        known = self.value_names | self.builtin_names
        # base event order (resolved if event_sequence names a constant);
        # instances only permute it (_check_event_sequence), so the base set
        # is the canonical event namespace for triggers
        events = set(self.event_sequence()) | {"END_OF_PERIOD"}

        # dynamics: locals defined by `x ~ ...` / `x = ...` extend the namespace
        local: set[str] = set()
        for t in self.dynamics.transitions:
            for u in t.updates:
                for stmt in _COMMENT.sub("", u).split(";"):
                    m = _ASSIGN.search(stmt)
                    if m:
                        local |= set(_IDENT.findall(stmt[: m.start()]))
        dyn_known = known | self.sampling_names | events | local
        for t in self.dynamics.transitions:
            if t.guard:
                _check_expr(t.guard, known, f"dynamics[{t.event}].guard")
            for u in t.updates:
                _check_expr(u, dyn_known, f"dynamics[{t.event}]")

        for src in self.uncertainty_sources:
            for k, v in src.distribution.settings.items():
                if isinstance(v, str):
                    _check_expr(v, known, f"uncertainty {src.name!r} settings[{k!r}]")
            for st in src.stages:
                if st.trigger:
                    _check_expr(st.trigger, known | events, f"stage {st.name!r}.trigger")
                for e in st.key_exprs or []:
                    # key exprs are evaluated in the transition namespace at
                    # draw time, so dynamics locals are in scope
                    _check_expr(e, dyn_known, f"stage {st.name!r}.key_exprs")

        for c in self.objective.per_step_components:
            _check_expr(c.expr, known, f"objective component {c.name!r}")

        # invariants: `prev.<name>` resolves against the per-period names, and
        # the rest of the expression against the END_OF_PERIOD namespace. The
        # prev-references are stripped before the general check so neither
        # `prev` nor the attribute is read as a bare identifier.
        seen: set[str] = set()
        per_period = self.per_period_names
        for inv in self.invariants:
            if inv.name in seen:
                raise ValueError(f"duplicate invariant name {inv.name!r}")
            seen.add(inv.name)
            bad = set(_PREV.findall(inv.expr)) - per_period
            if bad:
                raise ValueError(
                    f"invariant {inv.name!r}: prev.{{{','.join(sorted(bad))}}} "
                    f"names no per-period value (state, info, component or "
                    f"decision)"
                )
            _check_expr(
                _PREV.sub(" ", inv.expr), known | {"t"},
                f"invariant {inv.name!r}",
            )

        for name, v in self.initial_state.items():
            if isinstance(v, str):
                _check_expr(
                    v,
                    {c.name for c in self.scenario.constants} | self.builtin_names,
                    f"initial_state[{name!r}]",
                )
        return self


# ----- gym block -----------------------------------------------------------


class ObservationFeature(_Base):
    ref: str | None = None                   # a state var or "info.<field>"
    derived: str | None = None               # name of a derived feature
    expr: str | None = None                  # required when `derived` is set

    @model_validator(mode="after")
    def _check_one_form(self) -> "ObservationFeature":
        if bool(self.ref) == bool(self.derived):
            raise ValueError("feature must set exactly one of `ref` or `derived`")
        if self.derived and not self.expr:
            raise ValueError(f"derived feature {self.derived!r} requires `expr`")
        return self


class ObservationMode(_Base):
    name: str
    default: bool = False
    features: list[ObservationFeature] = Field(min_length=1)


class ActionMode(_Base):
    """A gym-layer *encoding* of the canonical decision(s). The ``_mdp`` layer
    consumes only the canonical decisions; each mode maps the agent-facing
    action onto them via ``transform`` (empty = identity).

    A problem whose period carries *several* simultaneous decisions (e.g. order
    quantity **and** allocation) sets ``encodes`` to the list of decision names;
    the agent action is then a vector whose k-th component drives
    ``encodes[k]``, and ``bounds`` becomes one ``[lo, hi]`` pair per component.
    The scalar forms remain valid and mean exactly what they did before."""

    name: str
    default: bool = False
    # one decision name, or several driven by one vector action
    encodes: str | list[str]
    type: DecisionType
    # entries may name scenario constants, like Decision.bounds (resolution is
    # checked by the root validator, which can see the mdp block). Multi-decision
    # modes carry one [lo, hi] pair per encoded decision, in the same order.
    bounds: list[float | str] | list[list[float | str]]
    transform: str = ""                      # expr: agent action -> canonical decision
    feasibility_strategy: FeasibilityStrategy = FeasibilityStrategy.clip
    desc: str = ""

    def encoded_decisions(self) -> list[str]:
        """The canonical decisions this mode drives, in action-component order."""
        return [self.encodes] if isinstance(self.encodes, str) else list(self.encodes)

    def bounds_per_decision(self) -> list[list[float | str]]:
        """One [lo, hi] per encoded decision, aligned with encoded_decisions()."""
        if isinstance(self.encodes, str):
            return [list(self.bounds)]                     # type: ignore[arg-type]
        return [list(b) for b in self.bounds]              # type: ignore[union-attr]

    @model_validator(mode="after")
    def _check_strategy(self) -> "ActionMode":
        if self.feasibility_strategy is FeasibilityStrategy.mask and (
            self.type is not DecisionType.discrete
        ):
            raise ValueError(f"action mode {self.name!r}: masking requires a discrete type")
        if self.feasibility_strategy is FeasibilityStrategy.reparametrize and not self.transform:
            raise ValueError(
                f"action mode {self.name!r}: reparametrize requires a non-empty transform"
            )
        return self

    @model_validator(mode="after")
    def _check_shape(self) -> "ActionMode":
        """`encodes` and `bounds` must agree on arity: a scalar mode carries one
        flat [lo, hi]; a multi-decision mode carries one pair per decision."""
        nested = bool(self.bounds) and isinstance(self.bounds[0], list)
        if isinstance(self.encodes, str):
            if nested:
                raise ValueError(
                    f"action mode {self.name!r}: single-decision mode takes a flat "
                    f"[lo, hi], got nested {self.bounds!r}"
                )
            pairs = [self.bounds]
        else:
            if len(self.encodes) < 2:
                raise ValueError(
                    f"action mode {self.name!r}: list `encodes` needs >= 2 decisions "
                    f"(use the plain string form for one), got {self.encodes!r}"
                )
            if len(set(self.encodes)) != len(self.encodes):
                raise ValueError(
                    f"action mode {self.name!r}: `encodes` repeats a decision: {self.encodes!r}"
                )
            if not nested or len(self.bounds) != len(self.encodes):
                raise ValueError(
                    f"action mode {self.name!r}: multi-decision mode needs one [lo, hi] "
                    f"per encoded decision ({len(self.encodes)}), got {self.bounds!r}"
                )
            if self.transform:
                raise ValueError(
                    f"action mode {self.name!r}: per-component transforms are not in v1; "
                    f"multi-decision modes must be identity (transform=\"\")"
                )
            pairs = self.bounds                            # type: ignore[assignment]
        for b in pairs:
            if len(b) != 2:
                raise ValueError(
                    f"action mode {self.name!r}: bounds entries must be [lo, hi], got {b!r}"
                )
            if all(isinstance(x, (int, float)) for x in b) and b[0] >= b[1]:
                raise ValueError(
                    f"action mode {self.name!r}: bounds must satisfy lo < hi, got {b!r}"
                )
        return self


class RewardMode(_Base):
    """How the gym assembles `reward` from the objective components carried in
    `info` (the mdp layer stays reward-agnostic, spec §6.4). Exprs reference
    component names and `total`; sign convention: reward is *maximized*, so a
    minimize-sense objective negates (e.g. "-total")."""

    name: str
    default: bool = False
    expr: str
    desc: str = ""


class Termination(_Base):
    horizon_end: HorizonEnd = HorizonEnd.terminated
    early_terminated_when: str | None = None  # absorbing-state expr, e.g. "inventory == 0"


class GymBlock(_Base):
    """Interface menus — Stage-2 codegen input. Design axes, not problem
    facts: revisable in Phase B without re-freezing the mdp block."""

    observation_modes: list[ObservationMode] = Field(min_length=1)
    action_modes: list[ActionMode] = Field(min_length=1)
    reward_modes: list[RewardMode] = Field(min_length=1)
    termination: Termination = Field(default_factory=Termination)

    @model_validator(mode="after")
    def _one_default_each(self) -> "GymBlock":
        for label, modes in (
            ("observation_modes", self.observation_modes),
            ("action_modes", self.action_modes),
            ("reward_modes", self.reward_modes),
        ):
            n = sum(m.default for m in modes)
            if n != 1:
                raise ValueError(f"exactly one {label} entry must be default=true, got {n}")
        return self

    def default_action_mode(self) -> ActionMode:
        return next(m for m in self.action_modes if m.default)


# ----- rl block ------------------------------------------------------------


class ObsNormalization(_Base):
    """The spec-§8.3 decision, made explicit: on for heterogeneous stationary
    features; off under distribution shift."""

    enabled: bool
    rationale: str = ""


class RlBlock(_Base):
    """Training configuration — Stage-4 input. Reads mdp/gym, never the
    reverse; consistency is enforced by root-level cross-layer validators."""

    requires_memory: Confirmable[bool]
    algo: Algo = Algo.ppo
    frame_stack: int = Field(default=1, ge=1)
    obs_normalization: ObsNormalization
    net_arch: list[int] | None = None        # None = SB3 default

    @model_validator(mode="after")
    def _memory_consistency(self) -> "RlBlock":
        needs = self.requires_memory.value
        has_recurrence = self.algo is Algo.recurrent_ppo
        has_stack = self.frame_stack > 1
        if needs and not (has_recurrence or has_stack):
            raise ValueError(
                "requires_memory=true but neither recurrent_ppo nor frame_stack>1 is set"
            )
        if not needs and (has_recurrence or has_stack):
            raise ValueError(
                "memory machinery (recurrent_ppo / frame_stack>1) set but requires_memory=false"
            )
        if has_recurrence and has_stack:
            raise ValueError("choose one memory mechanism: recurrent_ppo or frame_stack>1")
        return self


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class ScenarioGrid(_Base):
    """Design-layer grid (spec §5.6): the finite generality target of a
    generalist policy, expressed as axes over scenario constants crossed on a
    base instance. Lives OUTSIDE the scenario node — the world layer never
    contains a grid. Cell order is row-major over the axes in declaration
    order; cell ids derive from axis names/values."""

    name: str
    base_instance: str = ""                  # "" = the base constants
    axes: dict[str, list] = Field(min_length=1)   # constant -> values, ordered
    desc: str = ""

    def cells(self) -> list[tuple[str, dict]]:
        """(cell_id, constant-overrides) in canonical row-major order."""
        import itertools
        names = list(self.axes)
        out = []
        for combo in itertools.product(*self.axes.values()):
            overrides = dict(zip(names, combo))
            cell_id = ",".join(f"{k}={v:g}" if isinstance(v, (int, float))
                               else f"{k}={v}" for k, v in overrides.items())
            out.append((cell_id, overrides))
        return out


class ComponentResolution(_Base):
    """A mixture component's own catalog resolution (cross-family mixtures,
    IR_LAYERING_PLAN §10f): when a component's selection differs from the
    selection the IR was loaded under, the loader resolves the component
    separately and stashes its sources, samplers, and synthesized-constant
    delta here; the interpreter swaps them in for episodes that draw this
    component, so the episode runs exactly as the component would standalone
    (standalone equivalence). Set by ``load_ir``, never hand-authored."""

    selection: dict[str, str]
    sources: list[UncertaintySource]
    samplers: list[ScenarioSampler] = Field(default_factory=list)
    # constants the component's resolution synthesizes beyond the loaded
    # IR's (its own candidates' latent placeholders), name -> value
    constants: dict[str, float | int | bool | str | list] = Field(default_factory=dict)


class MdpIR(_Base):
    ir_version: str
    # seed-key grammar (spec §6.3): "v1" frozen legacy (default — every
    # pre-redesign IR keeps its draws bit-exact), "v2" the seed tree
    seed_scheme: str = "v1"
    domain: Domain
    mdp: MdpBlock
    gym: GymBlock
    rl: RlBlock
    # design layer (spec §5.6): generality targets for generalist training
    grids: list[ScenarioGrid] = Field(default_factory=list)
    # bystander report columns (EvalMetric): describe, never decide. Root
    # level by design — outside the mdp block, so mdp_fingerprint is
    # unaffected and metrics stay appendable after the Phase-A freeze
    eval_metrics: list[EvalMetric] = Field(default_factory=list)
    assumptions_log: list[str] = Field(default_factory=list)
    # the slot->candidate selection this IR was resolved under (catalog ⊕
    # selection, IR_LAYERING_PLAN §10); None for a legacy resolved single
    # file. Set by ``load_ir``, never hand-authored.
    selection: dict[str, str] | None = None
    # cross-family mixtures: {mixture name: {component: resolution}} for
    # components whose selection differs from `selection` (loader-set,
    # like `selection`; sits outside the mdp block, so mdp_fingerprint is
    # unaffected by the mechanism)
    mixture_resolutions: dict[str, dict[str, ComponentResolution]] | None = None

    @model_validator(mode="after")
    def _eval_metric_names(self) -> "MdpIR":
        """Metric names must be unique and must not shadow anything the
        END_OF_PERIOD namespace already carries (state, info, decisions,
        constants, objective components, `total`, `reward`)."""
        if not self.eval_metrics:
            return self
        seen: set[str] = set()
        taken = self.mdp.value_names | {
            c.name for c in self.mdp.objective.per_step_components
        } | {"total", "reward", "t"}
        for m in self.eval_metrics:
            if m.name in seen:
                raise ValueError(f"eval_metrics name {m.name!r} declared twice")
            if m.name in taken:
                raise ValueError(
                    f"eval_metrics name {m.name!r} shadows an existing "
                    f"namespace value (state/info/decision/constant/component)"
                )
            seen.add(m.name)
        return self

    @model_validator(mode="after")
    def _seed_scheme_rules(self) -> "MdpIR":
        if self.seed_scheme not in ("v1", "v2"):
            raise ValueError(f"seed_scheme must be 'v1' or 'v2', got {self.seed_scheme!r}")
        if self.seed_scheme == "v1":
            bad = [s.name for s in self.mdp.uncertainty_sources if s.stream_id < 1]
            if bad:
                raise ValueError(
                    f"v1 scheme: stream_id 0 is reserved for scenario sampling; "
                    f"sources {bad} must use stream_id >= 1"
                )
        else:
            # §4.3 enforced by grammar: an intrinsic key must contain the
            # period level, so episode-realization stages cannot exist —
            # world latents are scenario samplers under v2
            bad = [
                f"{src.name}.{st.name}"
                for src in self.mdp.uncertainty_sources
                for st in src.stages
                if st.realization is Realization.episode
            ]
            if bad:
                raise ValueError(
                    f"v2 scheme: episode-realization stages {bad} are treatment A — "
                    "model them as scenario.samplers (spec §4.3, §5.2)"
                )
            ids: dict[int, list[str]] = {}
            for src in self.mdp.uncertainty_sources:
                ids.setdefault(src.stream_id, []).append(src.name)
            dupes = {k: v for k, v in ids.items() if len(v) > 1}
            if dupes:
                raise ValueError(f"v2 scheme: source ids shared: {dupes}")
        return self

    @model_validator(mode="after")
    def _mixture_resolutions_consistent(self) -> "MdpIR":
        """Resolution keys must name declared mixtures and their components;
        a resolution's sources must cover exactly the loaded IR's source
        names (same slots, different candidates)."""
        if not self.mixture_resolutions:
            return self
        mixtures = {m.name: m for m in self.mdp.scenario.mixtures}
        source_names = {s.name for s in self.mdp.uncertainty_sources}
        for mname, comps in self.mixture_resolutions.items():
            if mname not in mixtures:
                raise ValueError(
                    f"mixture_resolutions names unknown mixture {mname!r}"
                )
            declared = {c for _w, c in mixtures[mname].components}
            bad = set(comps) - declared
            if bad:
                raise ValueError(
                    f"mixture_resolutions[{mname!r}] names non-component(s) "
                    f"{sorted(bad)}; components: {sorted(declared)}"
                )
            for comp, res in comps.items():
                got = {s.name for s in res.sources}
                if got != source_names:
                    raise ValueError(
                        f"mixture_resolutions[{mname!r}][{comp!r}] sources "
                        f"{sorted(got)} must cover the IR's slots {sorted(source_names)}"
                    )
        return self

    @model_validator(mode="after")
    def _grids_resolve(self) -> "MdpIR":
        consts = {c.name for c in self.mdp.scenario.constants}
        names = [g.name for g in self.grids]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate grid names in {names}")
        clash = set(names) & (set(self.mdp.scenario.instances)
                              | {s.name for s in self.mdp.scenario.samplers}
                              | {m.name for m in self.mdp.scenario.mixtures})
        if clash:
            raise ValueError(f"grid name(s) collide with world-layer names: {sorted(clash)}")
        for g in self.grids:
            bad = set(g.axes) - consts
            if bad:
                raise ValueError(f"grid {g.name!r}: axes {sorted(bad)} name no scenario constant")
            empty = [a for a, vals in g.axes.items() if not vals]
            if empty:
                raise ValueError(f"grid {g.name!r}: empty axis value list for {empty}")
            if g.base_instance and g.base_instance not in self.mdp.scenario.instances:
                raise ValueError(
                    f"grid {g.name!r}: unknown base_instance {g.base_instance!r}"
                )
        return self

    # -- cross-layer invariants (each enforces one downward edge) -----------

    @model_validator(mode="after")
    def _gym_refs_resolve(self) -> "MdpIR":
        """gym → mdp: every feature ref/expr resolves; latent state appears in
        no observation mode."""
        state_names = {sv.name for sv in self.mdp.state_variables}
        info_names = {f.name for f in self.mdp.info_fields}
        # hidden world latents (spec §5.2): a realized sampled constant is not
        # automatically observable — bar hidden ones from observation exprs
        latent = self.mdp.latent_names | self.mdp.scenario.hidden_sampled_names
        known = (self.mdp.value_names - latent) | self.mdp.builtin_names
        for mode in self.gym.observation_modes:
            for feat in mode.features:
                where = f"observation mode {mode.name!r}"
                if feat.ref:
                    if feat.ref.startswith("info."):
                        if feat.ref[5:] not in info_names:
                            raise ValueError(f"{where}: unknown info field ref {feat.ref!r}")
                    elif feat.ref not in state_names:
                        raise ValueError(f"{where}: unknown state ref {feat.ref!r}")
                    if (feat.ref.split(".")[-1]) in latent:
                        raise ValueError(
                            f"{where} references latent variable {feat.ref!r}; "
                            "latent state is unobservable"
                        )
                else:
                    clash = latent & _identifiers(feat.expr or "")
                    if clash:
                        raise ValueError(
                            f"{where} expr references latent variable(s) {sorted(clash)}"
                        )
                    _check_expr(feat.expr or "", known, where)
        return self

    @model_validator(mode="after")
    def _action_modes_resolve(self) -> "MdpIR":
        """gym → mdp: every action mode encodes a declared decision; transforms
        reference only the mode's own action name, decisions, and constants."""
        decision_names = {d.name for d in self.mdp.decisions}
        constants = {c.name for c in self.mdp.scenario.constants}
        for m in self.gym.action_modes:
            flat = [x for pair in m.bounds_per_decision() for x in pair]
            bad = [x for x in flat if isinstance(x, str) and x not in constants]
            if bad:
                raise ValueError(
                    f"action mode {m.name!r}: bound entries {bad} name no scenario constant"
                )
            unknown = [d for d in m.encoded_decisions() if d not in decision_names]
            if unknown:
                raise ValueError(
                    f"action mode {m.name!r} encodes unknown decision(s) {unknown}; "
                    f"must be among {sorted(decision_names)}"
                )
            if m.transform:
                _check_expr(
                    m.transform,
                    constants | decision_names | {m.name} | self.mdp.builtin_names,
                    f"action mode {m.name!r}.transform",
                )
            # the mdp layer consumes every declared decision each period, so an
            # action mode that drives only some of them cannot step the env
            missing = decision_names - set(m.encoded_decisions())
            if missing:
                raise ValueError(
                    f"action mode {m.name!r} leaves decision(s) {sorted(missing)} "
                    f"unencoded; every mode must drive all {len(decision_names)} decisions"
                )
        return self

    @model_validator(mode="after")
    def _reward_modes_resolve(self) -> "MdpIR":
        """gym → mdp: reward exprs reference objective components / `total`."""
        known = (
            {c.name for c in self.mdp.objective.per_step_components}
            | {"total"}
            | self.mdp.value_names
            | self.mdp.builtin_names
        )
        for m in self.gym.reward_modes:
            _check_expr(m.expr, known, f"reward mode {m.name!r}")
        return self

    @model_validator(mode="after")
    def _termination_resolves(self) -> "MdpIR":
        expr = self.gym.termination.early_terminated_when
        if expr:
            _check_expr(
                expr,
                self.mdp.value_names | self.mdp.builtin_names,
                "termination.early_terminated_when",
            )
        return self

    @model_validator(mode="after")
    def _requires_memory_derivation(self) -> "MdpIR":
        """rl → mdp: `suggested` must equal the derived heuristic (sample §5.3):
        a latent source realizing at episode level is temporally correlated ⇒
        memory suggested. The human may override `value`; `suggested` records
        the derivation."""
        derived = any(
            src.latent and any(
                st.realization is Realization.episode for st in src.stages
            )
            for src in self.mdp.uncertainty_sources
        ) or any(s.hidden for s in self.mdp.scenario.samplers) or any(
            s.hidden
            for comps in (self.mixture_resolutions or {}).values()
            for res in comps.values()
            for s in res.samplers
        )
        if self.rl.requires_memory.suggested != derived:
            raise ValueError(
                f"rl.requires_memory.suggested={self.rl.requires_memory.suggested} "
                f"disagrees with the derived heuristic ({derived}); set suggested "
                "to the derived value and override `value` if needed"
            )
        return self

    @model_validator(mode="after")
    def _algo_matches_action_type(self) -> "MdpIR":
        """rl → gym: masking needs a discrete default action mode."""
        default_type = self.gym.default_action_mode().type
        if self.rl.algo is Algo.maskable_ppo and default_type is not DecisionType.discrete:
            raise ValueError("maskable_ppo requires a discrete default action mode")
        return self

    # -- Phase-A gate helpers ------------------------------------------------

    def unconfirmed(self) -> list[str]:
        """Paths of Confirmable fields still at source=derived — the exact list
        the Phase-A round-trip walkthrough must clear before the mdp block is
        frozen (empty ⇒ every judgment call was reviewed)."""

        out: list[str] = []

        def walk(obj: object, path: str) -> None:
            if isinstance(obj, Confirmable):
                if obj.source is Source.derived:
                    out.append(path)
                return
            if isinstance(obj, BaseModel):
                for name in type(obj).model_fields:
                    walk(getattr(obj, name), f"{path}.{name}" if path else name)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    walk(item, f"{path}[{i}]")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{path}[{k!r}]")

        walk(self, "")
        return out

    def mdp_fingerprint(self) -> str:
        """Stable hash of the mdp block — the freeze token. Phase B records it;
        any later mdp edit changes the fingerprint and voids the confirmation.
        gym/rl edits do not."""
        canonical = json.dumps(self.mdp.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


# Directories of every IR loaded this process, most recent first. Domain-owned
# builtin implementations (interpreter._domain_builtin) and adapter discovery
# resolve against these, so a domain folder works wherever it lives — no
# repo-root assumption (portable-domain contract).
DOMAIN_DIRS: list[Path] = []


def load_ir(
    path: str | Path,
    *,
    instance: str | None = None,
    select: dict[str, str] | None = None,
) -> MdpIR:
    """Load and validate an IR file into an ``MdpIR``.

    Accepts the catalog form or a legacy resolved single file:

    - a **catalog** schema (``mdp.uncertainty_slots`` with per-slot
      ``candidates`` + ``default``) — resolved under a selection: slot
      defaults, overridden by the named ``instance``'s slot-valued keys,
      overridden by an explicit ``select`` mapping (IR_LAYERING_PLAN §10);
    - a single file with ``mdp.uncertainty_sources`` — loaded as-is, so
      pre-catalog domains keep working (``instance``/``select`` are the
      catalog's load-time axes and are ignored here; instance resolution at
      *run* time stays with the consumers, as always).
    """
    from mdp_ir import layering

    p = Path(path).resolve()
    data = json.loads(p.read_text())
    if p.parent not in DOMAIN_DIRS:
        DOMAIN_DIRS.insert(0, p.parent)

    if layering.is_catalog(data):
        data = layering.resolve_catalog(data, instance=instance, select=select)
    elif select:
        raise layering.LayeringError(
            f"{p.name}: --select applies to catalog schemas; this file is a "
            "legacy resolved IR with fixed uncertainty_sources"
        )
    return MdpIR.model_validate(data)

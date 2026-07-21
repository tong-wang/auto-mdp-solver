"""Pydantic schema for the MDP intermediate representation (IR), v0.4.

The IR is split into three layers that mirror the codebase's own layering
(spec §1.1) and the Phase-B pipeline (plan §4), with dependencies pointing
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
                          # key values — extension 2026-07-13 (topk_id): outcomes
                          # keyed by (evaluation atom, repetition) so the same
                          # evaluation set realizes the same outcomes regardless
                          # of scheduling (batch semantics / common random numbers)


class Sense(str, Enum):
    minimize = "minimize"
    maximize = "maximize"


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
# an assignment `=` or draw `~`, excluding ==, !=, <=, >=, +=, -=, *=, /=
_ASSIGN = re.compile(r"(?<![=!<>+\-*/])=(?!=)|~")

_BUILTINS = frozenset({
    "if", "else", "not", "and", "or", "in", "for",
    "min", "max", "sum", "abs", "len", "round", "int", "float",
    "exp", "log", "sqrt", "floor", "ceil", "zeros",
    "phi", "topk", "range",    # extension 2026-07-10 (topk_id): std-normal CDF,
                               # top-k largest values, comprehension index ranges
    "bayes_topk",              # extension 2026-07-13 (topk_id): Bayes-optimal
                               # top-k selection via probit MAP; implementation
                               # lives with the domain (topk_id/topk_id_probit_map.py),
                               # lazily resolved by the interpreter
    "post_mu", "post_sd",      # extension 2026-07-13 (topk_id): Laplace posterior
                               # moments for the counts_belief observation mode
                               # (same lazy domain resolution)
    "T",                       # horizon length
    "True", "False", "None", "true", "false",
})


def _identifiers(expr: str) -> set[str]:
    return set(_IDENT.findall(_COMMENT.sub("", expr)))


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
    # v1 scope is finite-horizon only (plan §1); an infinite/average-reward
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
    length: int | None = None
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
    # horizon.T, bounds are a scenario dimension and may vary per instance
    # (extension 2026-07-10, topk_id). Resolve with `MdpBlock.decision_bounds`.
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
    value: float | int | bool | list
    axis: str = ""
    desc: str = ""


class Scenario(_Base):
    constants: list[ScenarioConstant]
    # named alternative instances: overrides of constants by name -> _scenarios.py
    instances: dict[str, dict[str, float | int | bool | list]] = Field(default_factory=dict)

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
        return self


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
    key_exprs: list[str] | None = None       # keyed stages: int-valued exprs prepended
                                             # to the seed key (extension 2026-07-13)

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
        if self.realization is not Realization.keyed and self.key_exprs:
            raise ValueError(
                f"stage {self.name!r}: key_exprs only applies to keyed realization"
            )
        return self

    def seed_key(self, stream_id: int, entity_id_in_seed: bool) -> list[str]:
        """Derived symbolic seed key, reverse-tree order (most specific first).
        Never hand-written: `realization` fixes the `period` slot, the entity
        structure fixes the `entity_id` slot.

        Slot order follows the reference generators (inv_single/owmr), which
        key ``[sub_stream?, stream_id, period?, episode_seed, seed_salt]`` —
        stream *before* period. (Spec §6.3's prose snippet shows period first;
        the actual generator code does not — codegen must match the code.)
        The ``stream:S.U`` symbol expands numerically to ``[U, S]``."""
        key: list[str] = []
        if entity_id_in_seed:
            key.append("entity_id")
        if self.key_exprs:
            key += [f"expr:{e}" for e in self.key_exprs]
        key.append(
            f"stream:{stream_id}" if self.sub_stream is None
            else f"stream:{stream_id}.{self.sub_stream}"
        )
        if self.realization not in (Realization.episode, Realization.keyed):
            key.append("period")
        key += ["episode_seed", "seed_salt"]
        return key


class UncertaintySource(_Base):
    name: str
    generator: str
    stream_id: int = Field(ge=1)             # 0 is reserved for scenario sampling (spec §6.3)
    is_discrete: bool | None = None
    latent: bool = False
    distribution: Distribution
    stages: list[UncertaintyStage] = Field(min_length=1)


class TransitionStep(_Base):
    event: str
    guard: str | None = None
    updates: list[str]


class Dynamics(_Base):
    event_sequence: list[str]
    observation_point: str = ""
    transitions: list[TransitionStep]

    @model_validator(mode="after")
    def _check_events(self) -> "Dynamics":
        known = set(self.event_sequence) | {"END_OF_PERIOD"}
        for t in self.transitions:
            if t.event not in known:
                raise ValueError(
                    f"transition event {t.event!r} not in event_sequence {self.event_sequence}"
                )
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
    per_step_components: list[ObjectiveComponent] = Field(min_length=1)

    @property
    def total_expr(self) -> str:
        return " + ".join(c.name for c in self.per_step_components)


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
    def latent_names(self) -> set[str]:
        return {
            sv.name for sv in self.state_variables
            if sv.observability is Observability.latent
        }

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

    # -- layer-local invariants ------------------------------------------

    @model_validator(mode="after")
    def _unique_value_names(self) -> "MdpBlock":
        seen: dict[str, str] = {}
        groups = [
            ("state", [sv.name for sv in self.state_variables]),
            ("info", [f.name for f in self.info_fields]),
            ("decision", [d.name for d in self.decisions]),
            ("constant", [c.name for c in self.scenario.constants]),
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
        known = self.value_names
        events = set(self.dynamics.event_sequence) | {"END_OF_PERIOD"}

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

        for name, v in self.initial_state.items():
            if isinstance(v, str):
                _check_expr(
                    v,
                    {c.name for c in self.scenario.constants},
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
    """A gym-layer *encoding* of a canonical decision (cf. OWMR action modes).
    The ``_mdp`` layer consumes only the canonical decision; each mode maps the
    agent-facing action onto it via ``transform`` (empty = identity)."""

    name: str
    default: bool = False
    encodes: str                             # name of the canonical decision it maps to
    type: DecisionType
    # entries may name scenario constants, like Decision.bounds (resolution is
    # checked by the root validator, which can see the mdp block)
    bounds: list[float | str]
    transform: str = ""                      # expr: agent action -> canonical decision
    feasibility_strategy: FeasibilityStrategy = FeasibilityStrategy.clip
    desc: str = ""

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


class MdpIR(_Base):
    ir_version: str
    domain: Domain
    mdp: MdpBlock
    gym: GymBlock
    rl: RlBlock
    assumptions_log: list[str] = Field(default_factory=list)

    # -- cross-layer invariants (each enforces one downward edge) -----------

    @model_validator(mode="after")
    def _gym_refs_resolve(self) -> "MdpIR":
        """gym → mdp: every feature ref/expr resolves; latent state appears in
        no observation mode."""
        state_names = {sv.name for sv in self.mdp.state_variables}
        info_names = {f.name for f in self.mdp.info_fields}
        latent = self.mdp.latent_names
        known = self.mdp.value_names - latent
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
            bad = [x for x in m.bounds if isinstance(x, str) and x not in constants]
            if bad:
                raise ValueError(
                    f"action mode {m.name!r}: bound entries {bad} name no scenario constant"
                )
            if m.encodes not in decision_names:
                raise ValueError(
                    f"action mode {m.name!r} encodes unknown decision {m.encodes!r}; "
                    f"must be one of {sorted(decision_names)}"
                )
            if m.transform:
                _check_expr(
                    m.transform,
                    constants | decision_names | {m.name},
                    f"action mode {m.name!r}.transform",
                )
        return self

    @model_validator(mode="after")
    def _reward_modes_resolve(self) -> "MdpIR":
        """gym → mdp: reward exprs reference objective components / `total`."""
        known = (
            {c.name for c in self.mdp.objective.per_step_components}
            | {"total"}
            | self.mdp.value_names
        )
        for m in self.gym.reward_modes:
            _check_expr(m.expr, known, f"reward mode {m.name!r}")
        return self

    @model_validator(mode="after")
    def _termination_resolves(self) -> "MdpIR":
        expr = self.gym.termination.early_terminated_when
        if expr:
            _check_expr(expr, self.mdp.value_names, "termination.early_terminated_when")
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


def load_ir(path: str | Path) -> MdpIR:
    """Load and validate an IR JSON file into an ``MdpIR``."""
    p = Path(path).resolve()
    data = json.loads(p.read_text())
    if p.parent not in DOMAIN_DIRS:
        DOMAIN_DIRS.insert(0, p.parent)
    return MdpIR.model_validate(data)

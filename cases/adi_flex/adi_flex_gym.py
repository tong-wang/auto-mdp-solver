"""ADI-flex Gymnasium wrapper — two-phase env over the two-step MDP (F7).

The period's two decisions sit at different information sets (paper §2 event
sequence): the order is placed before the period's demand is observed, the
allocation at fulfillment after it. The env therefore takes one ORDER step and
then — when the allocation is a live decision — one ALLOCATION step per
eligible demand class, all through ONE shared ``Discrete(order_max + 1)`` space whose
per-step ``action_masks()`` selects the live range (the IR's ``seq_mask``
mode; spec §7.1). One policy (MaskablePPO) serves both phases, conditioning on
the ``phase`` observation feature.

The allocation has ``n_alloc = T_dl - 1`` components (one crossover-eligible
demand class each, nearest due date first) and the env asks for them one at a
time, sharing a running surplus. They sit at ONE information set — nothing
stochastic happens between them and no cost is charged — so those micro-steps
are an action-encoding device (the slot pattern: several agent steps inside one information set, the phase carried as an observation feature), NOT extra MDP periods.
A period is therefore ``1 + n_alloc`` agent steps: 3 at the campaign's T_dl = 2,
4 at T_dl = 3 (``n_alloc == T_dl`` since F10 — only the due-now class is
forced, so every class ahead is a decision; it was T_dl - 1 before). ``alloc_enabled=False`` (homogeneous branch) drops the
allocation phase entirely: single-phase ordering env, one step per period,
the simulator forcing the maximal early fill.

Rewards: the period's full cost lands on its LAST agent step (the allocation
step, or the order step on the homogeneous branch); intermediate steps pay 0.
NOTE for a discounted variant: SB3 discounts per env step, so at alpha < 1 the
gym must account for the two-steps-per-period structure — moot at alpha = 1.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import adi_flex_mdp as mdp
from adi_flex_scenarios import AdiFlexScenario, AdiFlexScenarioSampler

# NO MAGNITUDE IS DECLARED IN THIS FILE (F12, F14). The action-space cap is
# `scenario.order_max` = N * demand_max — a bound from the dynamics (an order
# can only serve demand that still arrives, and at most N periods of it do), so
# it cannot bind under any policy or any instance a later round adds. The model
# bounds an order below only, at 0 (`mdp.model` puts capacity limits out of
# scope), so this ceiling is a rendering choice and belongs here (F9) — but its
# *value* belongs to the scenario, not to a literal.
#
# There is no ALLOC_MAX here either. The IR's `alloc_max` bounds the *declared*
# allocate decision (what the differential samples from), but this env never
# reads it: both phases share ONE Discrete(order_max + 1) and the allocation's
# live range is the state-dependent mask (F13).

# phase codes (the `phase` observation feature)
PHASE_ORDER = 0.0
PHASE_ALLOC = 1.0


def adv_obs_slots(scenario: AdiFlexScenario | AdiFlexScenarioSampler) -> int:
    """How many advance-demand-profile slots the observation carries.

    The *state* profile is `T_dl` wide (paper §2, Figure 1), and a sampler's
    members must agree on `T_dl`, so this is simply the family's own width —
    unlike the pipeline, which may vary across members and gets padded.
    """
    return scenario.T_dl


#: the MIP-family observation modes: every one renders a modified inventory
#: position plus the profile V~ it does not summarize. They differ in WHICH
#: position (Wang & Toktay's vs Gallego & Ozer's) and in whether one raw
#: component rides alongside to restore what the compression dropped.
MIP_MODES = ("vec_mip", "vec_mip2", "vec_mip_plus_a", "vec_mip_plus_b")

# Action modes. `seq_mask` is the incumbent: order quantity, then one masked
# allocation quantity per class, at the POST-demand information set (F7).
#
# The others factor the two encodings independently — name the order encoding,
# append `_protection` when the allocation uses protection levels:
#   order_protection  order quantity  + protection levels, PRE-demand, one step
#   target_ip         order-up-to IP  + quantity allocation (pairs with `vec`)
#   target_mip        order-up-to u   + quantity allocation (pairs with mip modes)
# `target_ip_protection` / `target_mip_protection` are the two remaining cells
# of that cross-product and are deliberately NOT built: both compose from
# decodes below, and a declared mode no instance exercises is untested by
# construction (F10).
# The two encodings factor independently, so the modes are their cross-product:
#   order encoding  in {quantity, up-to on IP, up-to on the MIP}
#   alloc encoding  in {quantity (post-demand), protection levels (pre-demand)}
# Name the order encoding, append `_protection` when the allocation uses levels.
# `seq_mask` is the incumbent and names its mechanism instead — it is frozen
# across the IR, 144 run directories, the `g` ids and the tree, so it keeps its
# name rather than forcing consistency onto something immovable.
PROTECT_MODES = ("order_protection", "target_ip_protection", "target_mip_protection")
TARGET_MODES  = ("target_ip", "target_mip",
                 "target_ip_protection", "target_mip_protection")
ACTION_MODES  = ("seq_mask", "order_protection", "target_ip", "target_mip",
                 "target_ip_protection", "target_mip_protection")


def agent_steps_per_period(scenario, action_mode: str = "seq_mask") -> int:
    """Agent steps one PERIOD costs under `action_mode` — the T̄ multiplier.

    It is a property of the mode, not only of the branch: `order_protection`
    takes both decisions at the pre-demand information set, so its period is
    ONE step even where the allocation is live, while every two-phase mode
    costs `1 + n_alloc` (F11). The homogeneous branch is one step in every mode.

    This lives here, once, because the train script needs the same number for
    §8.6's launch check — `assert_l1_current` compares T̄ against the basis and
    raises the rollout-episode floor with it, so feeding it the two-phase count
    for a one-shot mode checks the derivation against a horizon the run does not
    have. Two copies of this rule is F6's defect (a width assumption written out
    at each call site, which broke both eval arms at once when the width moved).
    """
    if not getattr(scenario, "alloc_enabled", False):
        return 1
    if action_mode in PROTECT_MODES:
        return 1
    return 1 + scenario.n_alloc


def mip_tail_slots(scenario: AdiFlexScenario | AdiFlexScenarioSampler) -> int:
    """How many advance-demand slots the MIP observation carries beside `mip`.

    The paper's whole point is that the MIP is a *sufficient statistic*, and
    exactly how much of the profile survives beside it is the Case boundary
    (Wang & Toktay 2008 §3):

    - **Case 1, T <= L + 1** (§3.1, Prop. 1): the state collapses to the scalar
      u_i, which evolves linearly (u_{i+1} = u_i + z_i - d_i) and carries a
      modified (s, S) policy. Nothing of the profile survives — this returns 0
      and the layout is (mip, time_to_go, <within-period block>).
    - **Case 2, T > L + 1** (§3.2, eq. 11): u_i is no longer sufficient;
      x_{i+L+1} "can no longer be expressed by the MIP and the demands only",
      and an additional (T - L - 1)-dimensional vector

          V~_i = (v_i^{i+L+1}, ..., v_i^{i+T-1})

      must be recorded, under which a state-dependent (s(V), S(V)) policy is
      optimal. Since `adv[k] = v_i^{i+k}`, that vector is exactly `adv[L+1:]`.

    **The full profile is NOT carried.** `mip` already deducts all of `adv`
    (eq. 7 — this is Wang & Toktay's MIP, not Gallego & Özer's, which deducts
    only the protection period [i, i+L]; the paper flags the difference
    itself), so re-appending every component both restates what `mip` already
    contains and makes this mode something other than the paper's state. It
    did exactly that until F18.

    The width is taken against the DECLARED `T_dl`, never the largest offset
    with a positive rate: a one-hot instance has the same profile width as any
    other member of its family, and letting a zero rate narrow the observation
    would re-bake a design point into the rendering (the F10 trap). A sampler
    pads to the family maximum, which is the SMALLEST L among its members —
    the same construction as pipe_obs_slots, in the opposite direction.
    """
    members = getattr(scenario, "members", None)
    pool = members if members is not None else [scenario]
    return max(max(0, scenario.T_dl - m.L - 1) for m in pool)


def pipe_obs_slots(scenario: AdiFlexScenario | AdiFlexScenarioSampler) -> int:
    """How many pipeline slots the observation carries.

    The *state* pipeline is L wide (paper eq. 1), so a specialist observes
    exactly its own L slots — none at all at L=0, where the paper's state is
    (x_i, V_i). A sampler may draw members with different L, and a Gym Box has
    to be fixed at construction, so the observation is padded to the family
    maximum. That padding is an observation-construction choice and lives here,
    in the rendering layer; it is deliberately NOT a cap on the state.
    """
    members = getattr(scenario, "members", None)
    return max(m.L for m in members) if members is not None else scenario.L


class AdiFlexEnv(gym.Env):
    """ADI-flex Gymnasium wrapper.

    action space (IR mode ``seq_mask``, the only mode): ``Discrete(order_max + 1)``
    shared by both phases; ``action_masks()`` exposes the live range —
    everything at the ORDER phase, ``[0, min(surplus, outstanding)]`` at the
    ALLOCATION phase (validity computed in the mdp layer,
    ``mdp.valid_allocation``). Actions above the feasible allocation clip in
    the simulator, so unmasked play is safe — the mask is for exploration
    efficiency and for keeping the policy's support honest.

    Both layouts end in the **within-period block** — `phase`, `d`, `surplus`,
    `outstanding` — and that block is rendered ONLY when `alloc_enabled` is
    true (F23). A period with no second decision has no interior: `phase` never
    leaves PHASE_ORDER, `surplus`/`outstanding` are the allocation's own
    feasible-set coordinates, and `d` is the previous period's draw, already
    folded into `adv` and new information only at an allocation phase. On the
    homogeneous branch the block was 6 of 10 features and 5 of them were
    identically constant. Widths below are written with the block present.

    observation_mode:
    - "vec" (default):
        [inv, pipe(L), adv(T_dl), time_to_go, phase, d(T_dl+1), surplus,
         outstanding]
      inv/pipe/adv/time_to_go describe the state the current decision is made
      on: the period-boundary state at the order phase, and the INTERSTATE at
      the allocation phase (order placed, arrival received, demand joined) —
      the tail is the within-period block: `phase` (0 = order,
      1 = allocate), `d` = the most recently realized demand vector (last
      period's at the order phase — the paper's v-hat information — this
      period's at the allocation phase), and `surplus`/`outstanding` = the
      allocation's own feasible-set coordinates FOR THE COMPONENT BEING ASKED
      (0 at the order phase) — two scalars whatever T_dl is, because the
      components are asked one at a time.
      pipe carries the instance's own L slots — none at L=0; a sampler pads
      to its members' maximum (see pipe_obs_slots). adv is T_dl wide, which a
      sampler's members must agree on (see adv_obs_slots).
    - "vec_mip": [mip, V~(max(0, T_dl - L - 1)), time_to_go, phase,
      d(T_dl+1), surplus, outstanding],
      mip = inv + sum(pipe) - sum(adv) — the paper's modified inventory
      position, eq. (7): the inventory position less ALL the advance demands.
      (Not Gallego & Ozer's MIP, which deducts only the protection period
      [i, i+L]; Wang & Toktay flag the difference themselves.) Beside it,
      V~ = adv[L+1:] and nothing more — the profile the MIP does NOT summarize,
      eq. (11). It is EMPTY when T_dl <= L + 1, where Prop. 1 says the scalar
      is the entire state; see mip_tail_slots. Same within-period block.

      This is the whole point of the mode: `vec` hands over the raw state and
      `vec_mip` hands over the paper's sufficient statistic, so the pair asks
      whether the network rebuilds Wang & Toktay's transform for itself. It is
      a strictly LOSSY compression — inv and the pipeline breakdown are gone,
      only their sum survives — which is what makes the comparison a test of
      the paper's claim rather than a change of coordinates.

    - "vec_mip_plus_a": [mip, V~, inv, time_to_go, <within-period block>]
    - "vec_mip_plus_b": [mip, V~, adv[0], time_to_go, <within-period block>]
      `vec_mip` with ONE raw component handed back. Both use Wang & Toktay's
      mip, and both are information-equivalent to `vec`: given (mip, V~) the
      identity mip = inv - adv[0] - sum(V~) turns either extra component into
      the other. So `vec` / `_plus_a` / `_plus_b` all carry the same knowledge
      at the same width, and only `vec_mip` itself is lossy.

      The pair exists because the vec-vs-vec_mip gap survived every
      information-theoretic explanation (F34): `vec_mip2` restored the
      conditioning and changed nothing, the fixed cost is paid at the same
      rate by both arms, and the immediate cost is as well explained by
      (period, mip, V~, action) as by the raw state. What is left is that
      `vec` lets gradient descent fit one relationship along three coordinates
      instead of one. These two modes test exactly that, and separate it from
      "which raw component happens to matter".

    (The pre-F7 "vec_d" mode is gone: its feature — the realized demand — is
    part of every layout's within-period block now.)

    reward_mode:
    - "neg_cost": negated per-period cost total (sense = minimize), paid on
      the period's last agent step; other steps pay 0.

    Notes:
    - The core simulator is the source of truth for dynamics; this wrapper
      owns spaces, phase bookkeeping, observation formatting and reward.
    - ``scenario`` may be an AdiFlexScenarioSampler: a member scenario is then
      drawn per episode in reset() (generalist training); family-level
      attributes (alloc_enabled, N, T_dl) come off the sampler for space
      building.
    - Episodes end `terminated` (not truncated) at period >= N — the horizon
      is part of the problem.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: AdiFlexScenario | AdiFlexScenarioSampler,
        action_mode: str = "seq_mask",
        observation_mode: str = "vec",
        reward_mode: str = "neg_cost",
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()

        assert action_mode in ACTION_MODES, \
            f"action_mode must be one of {ACTION_MODES}, got '{action_mode}'."
        assert observation_mode in ("vec",) + MIP_MODES, \
            f"observation_mode must be 'vec' or one of {MIP_MODES}, got "\
            f"'{observation_mode}'."
        assert reward_mode in {"neg_cost"}, \
            f"reward_mode must be 'neg_cost', got '{reward_mode}'."

        self.scenario: AdiFlexScenario | AdiFlexScenarioSampler = scenario
        self.action_mode      = action_mode
        self.observation_mode = observation_mode
        self.reward_mode      = reward_mode

        # family-level attributes are identical on scenario and sampler
        self.alloc_enabled: bool = scenario.alloc_enabled
        self.horizon_n:     int  = scenario.N
        # widths (F10): every one of them is the scenario's, never a global
        self.T_dl:          int  = scenario.T_dl
        self.n_alloc:       int  = scenario.n_alloc
        # pipeline slots the OBSERVATION carries (state width is per-instance L)
        self.pipe_slots:    int  = pipe_obs_slots(scenario)
        self.adv_slots:     int  = adv_obs_slots(scenario)
        self.mip_tail_slots:int  = mip_tail_slots(scenario)
        # action-space cap: N * demand_max, from the scenario (F14)
        self.order_max:     int  = scenario.order_max
        # protection-level widths, from the scenario for the same reason the
        # order cap is (F14): a cap argued from the dynamics, never measured
        self.n_sigma:       int  = scenario.n_sigma
        self.sigma_max:     int  = scenario.sigma_max
        # `order_protection` takes BOTH decisions at the pre-demand information
        # set, so its period is one agent step and has no interior. Everything
        # the within-period block describes — `phase`, `surplus`, `outstanding`
        # and the previous draw — exists to describe what happens BETWEEN a
        # period's two decisions, so it is gated on THIS, not on alloc_enabled:
        # F23's rule ("why is the dead half being rendered at all") applied to
        # the mode rather than to the branch.
        self.two_phase:     bool = (
            agent_steps_per_period(scenario, action_mode) > 1)

        self._scenario_ep:  AdiFlexScenario
        self._state:        mdp.AdiFlexState
        self._mid:          mdp.AdiFlexState | None   # post-advance1, mid-period
        self._info1:        dict          # what advance1 realized this period
        self._phase:        float
        self._alloc_buf:    list[int]   # allocation components chosen so far
        self._info:         dict
        self._episode_seed: int
        self._step:         int
        self.total_reward:  float

        # `order_protection` emits (order, sigma_1..sigma_{n_sigma}) in ONE
        # action, so its space is a MultiDiscrete box — and every point in it
        # is feasible by construction, because pre-demand the components share
        # no budget (`surplus` does not exist yet). That is the IR's
        # feasibility_strategy="none": the constraint is not enforced, it is
        # dissolved. Every other mode keeps the shared Discrete whose per-step
        # mask selects the live range.
        if action_mode in PROTECT_MODES:
            # a protection mode is ONE action per period: the order (a quantity,
            # or a target plus its sentinel) followed by n_sigma reserve levels.
            # Every point of the box is feasible — pre-demand the components
            # share no budget, because `surplus` does not exist yet.
            head = (self.order_max + 2 if action_mode in TARGET_MODES
                    else self.order_max + 1)
            self.action_space = gym.spaces.MultiDiscrete(
                [head] + [self.sigma_max + 1] * self.n_sigma
            )
        elif action_mode in TARGET_MODES:
            # ONE extra category, and it is index 0: the NO-ORDER sentinel,
            # conceptually a target of -1. It is needed because an order-up-to
            # level cannot express "do not order" by itself once the reference
            # goes negative — `y <= reference` is the no-order condition, and
            # with backlog the reference is below every non-negative target, so
            # a [0, order_max] target would force an order in exactly the (s,S)
            # trigger region. The optimal policy does decline to order under
            # backlog here (K = 100 against p = 9), so this is not a corner.
            #
            # The sentinel sits OUTSIDE the legitimate target range rather than
            # consuming a value inside it: index 0 is -1, indices 1..order_max+1
            # are targets 0..order_max, so no target level is sacrificed.
            #
            # The range needs no negative targets even though the solver's
            # tables contain them: those arise only where `order_max` binds at
            # deep backlog (at u = -150 on het_exp4 the table stores y = -43,
            # which IS an order of exactly order_max), and the same order is
            # reachable as target = order_max. What must be onto is the ORDER
            # QUANTITY, not the target value — the target is a parametrization.
            #
            # At an ALLOCATION step the shared space keeps its usual meaning:
            # the index is the quantity, direct, and the sentinel is masked out.
            self.action_space = gym.spaces.Discrete(self.order_max + 2)
        else:
            self.action_space = gym.spaces.Discrete(self.order_max + 1)
        self.observation_space = self._build_observation_space()

        # Loggers are keyed on the OUTPUT FILE, not on a fixed name (F28).
        # `getLogger("step_logger")` returns one shared object per process, so
        # every env in a vec-env got the same logger: the worker holding the
        # file handler received all four workers' records, interleaved, and the
        # `logger_filename=None` guard on the other workers suppressed nothing
        # (it only skips handler setup — the .info() calls still land on the
        # shared logger). Measured: 121 MB per run, four episode_seeds at every
        # step index. Keying on the filename makes the guard real, and
        # `propagate = False` keeps records out of the root logger as well.
        key = logger_filename if logger_filename is not None else f"null.{id(self)}"
        self.logger_e = logging.getLogger(f"adi_flex.episode.{key}")
        self.logger_s = logging.getLogger(f"adi_flex.step.{key}")
        self.logger_e.propagate = False
        self.logger_s.propagate = False
        if logger_filename is not None:
            self._init_logger(logger_filename)

    def _init_logger(self, filename: str) -> None:
        fmt = logging.Formatter("%(asctime)s\t%(message)s")

        self.logger_e.handlers.clear()
        self.logger_e.setLevel(logging.DEBUG)
        fh_e = logging.FileHandler(f"{filename}_episode.log")
        fh_e.setFormatter(fmt)
        self.logger_e.addHandler(fh_e)

        # a header, so the file explains itself without the source open
        self._step_header = (
            "episode_seed\tstep\tperiod\t"
            "inv\tpipe\tadv\t"
            "order\tallocate\t"
            "d\treceived\tfills\tearly_fill\t"
            "inv'\tpipe'\tadv'\t"
            "K\tholding\tbackorder\ttotal\treward\tcum"
        )
        self.logger_s.handlers.clear()
        self.logger_s.setLevel(logging.DEBUG)
        step_path = f"{filename}_step.log"
        fresh = not Path(step_path).exists() or Path(step_path).stat().st_size == 0
        fh_s = logging.FileHandler(step_path)
        fh_s.setFormatter(fmt)
        self.logger_s.addHandler(fh_s)
        if fresh:
            self.logger_s.info(self._step_header)

    # -- spaces ---------------------------------------------------------------

    def _build_observation_space(self) -> gym.spaces.Box:
        """The Box is the MODEL's domain for each feature, and nothing else.

        No magnitude appears here. `mdp.model` states every quantity's domain —
        `pipe`, `adv`, `order` and `allocate` are "integer >= 0", `inv` alone is
        free — so the envelope is `[0, inf)` where the theory says non-negative
        and `(-inf, inf)` where it says nothing. Two consequences, both wanted:

        - **It is stable.** The theory does not move when a rate vector or a
          horizon does, so the Box is identical across every instance of the
          same shape, and a trained artifact stays loadable when the design
          changes. A finite envelope is part of an artifact's compatibility
          key — SB3's `check_for_correct_spaces` compares `low`/`high` on load
          — and a derived one would invalidate every saved model whenever the
          derivation was retuned.
        - **It still catches things.** `[0, inf)` is not vacuous: a negative
          `adv`, `surplus` or `d` fails `contains()`, and those are exactly the
          sign violations a mis-indexed observation slot produces. What is
          given up is magnitude checking, which nothing consumes — PPO's
          `preprocess_obs` is the identity for a non-image Box, and
          `VecNormalize` reads only the shape.

        The action space is the opposite case and is treated oppositely: a
        `Discrete` cap is explored, so it is derived and as tight as the
        dynamics support (`scenario.order_max`).
        """
        inf = float("inf")
        n_pipe, n_adv, n_d = self.pipe_slots, self.adv_slots, self.T_dl + 1
        # The WITHIN-PERIOD block, rendered only when the period has an interior
        # (F23, restored at F25 after F24 wrongly put `d` back).
        #
        # `d` is LAST period's draw at the order phase — history, not state.
        # F24 restored it on the argument that the exact DP indexes on `d[2]`
        # and that `d[2]` is not a function of the state. Both true; neither
        # matters. The decisive test is substitution, and it was not run until
        # the operator asked whether `d` belongs in the state at all: playing
        # the SAME y* table with `vhat` taken from the state's own `adv[1]`
        # instead of `d[2]` costs **685.263 against 685.263** over 2048 CRN
        # seeds — identical to the last digit across 61440 decisions. Taking
        # `vhat = 0` costs 791.486, so the coordinate matters and `adv[1]`
        # carries all of it. The DP reads `d[2]` only because its own
        # `u = inv - adv0 - adv1` discarded the split that `adv[1]` keeps.
        #
        # `phase`, `surplus` and `outstanding` are the allocation's own
        # feasible-set coordinates, structurally dead without an allocation.
        #                    d(T_dl+1)   phase  surplus  outstanding
        tail_low  = ([0.0] * n_d + [0.0, 0.0, 0.0]) if self.two_phase else []
        tail_high = ([inf] * n_d + [1.0, inf, inf]) if self.two_phase else []
        if self.observation_mode in MIP_MODES:
            # every mip variant inherits inv's free sign; the profile beside
            # them is V~ = adv[L+1:], identical in all and empty in Case 1.
            # A `_plus_` mode carries one more raw component after V~: `inv`
            # keeps inv's free sign, `adv[0]` is a demand and cannot go below 0
            n_tail = self.mip_tail_slots
            extra_low  = ([-inf] if self.observation_mode == "vec_mip_plus_a" else
                          [0.0]  if self.observation_mode == "vec_mip_plus_b" else [])
            extra_high = [inf] * len(extra_low)
            low  = [-inf] + [0.0] * n_tail + extra_low  + [0.0] + tail_low
            high = [ inf] + [inf] * n_tail + extra_high + [inf] + tail_high
        else:
            low  = [-inf] + [0.0] * n_pipe + [0.0] * n_adv + [0.0] + tail_low
            high = [ inf] + [inf] * n_pipe + [inf] * n_adv + [inf] + tail_high
        return gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    # -- observation ----------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # THE STATE THE DECISION IS BEING MADE ON, not the period's opening one
        # (F19). At the allocation phase that is the interstate: the order has
        # been placed into the pipeline, the arrival has landed in `inv`, and
        # this period's demand has joined `adv` — all three of which the
        # allocation weighs. Reading `self._state` here left the observation one
        # event-step stale, and at L >= 1 nothing else carried the order, so a
        # memoryless policy could not see its own action from one step earlier.
        s = self._mid if (self._phase == PHASE_ALLOC and self._mid is not None) \
            else self._state
        time_to_go = float(self._scenario_ep.N - s.period)
        # the allocation's own feasible-set coordinates, for the component the
        # env is asking about right now: the surplus still unspent and that
        # class's outstanding demand. Two scalars at any T_dl — the components
        # are asked one at a time, so the width never enters the observation.
        if self._phase == PHASE_ALLOC:
            assert self._mid is not None
            k           = len(self._alloc_buf)
            surplus     = float(max(self._mid.inv, 0) - sum(self._alloc_buf))
            outstanding = float(self._mid.adv[k])
            last_d      = self._info1["d"]     # this period's draw
        else:
            surplus, outstanding = 0.0, 0.0
            last_d = self._info["d"]           # last period's, off the closed record
        # the within-period block — only when the period HAS an interior (F25);
        # see _build_observation_space for why each component goes
        tail = ([*[float(x) for x in last_d], self._phase, surplus, outstanding]
                if self.two_phase else [])

        adv = [float(x) for x in s.adv]
        if self.observation_mode in MIP_MODES:
            L = self._scenario_ep.L
            if self.observation_mode != "vec_mip2":
                # Wang & Toktay eq. (7): the inventory position less ALL the
                # advance demands.
                mip = float(s.inv + sum(s.pipe) - sum(s.adv))
            else:
                # Gallego & Ozer's, which W&T name and distinguish themselves
                # from: the inventory position less only the advance demands
                # WITHIN THE PROTECTION PERIOD [i, i+L]. The tail is unchanged,
                # so the two modes carry the SAME information in a different
                # basis — `u = w - sum(V~)` recovers one from the other exactly.
                # A difference between them is therefore about the coordinate,
                # not about what the agent knows (F33).
                mip = float(s.inv + sum(s.pipe) - sum(s.adv[: L + 1]))
            vtil = adv[L + 1:]
            vtil += [0.0] * (self.mip_tail_slots - len(vtil))   # pad to family width
            # the de-compression probe (F34): one raw component restored beside
            # the statistic. Either choice makes the mode information-equivalent
            # to `vec` -- adv[0] = inv - u - sum(V~) recovers the one from the
            # other -- so what the pair varies is redundancy, not knowledge.
            # `adv[0]` reads 0 where the instance has no advance demand at all.
            if self.observation_mode == "vec_mip_plus_a":
                extra = [float(s.inv)]
            elif self.observation_mode == "vec_mip_plus_b":
                extra = [float(s.adv[0]) if s.adv else 0.0]
            else:
                extra = []
            feats = [mip, *vtil, *extra, time_to_go, *tail]
        else:
            pipe = [float(x) for x in s.pipe]
            pipe += [0.0] * (self.pipe_slots - len(pipe))   # pad to the family width
            feats = [float(s.inv), *pipe, *adv, time_to_go, *tail]
        return np.array(feats, dtype=np.float32)

    # -- action decodes (the maps the IR cannot declare) ----------------------
    #
    # Schema v1 refuses a per-component `transform` on a multi-decision action
    # mode, so all four modes ship transform="" and these two functions ARE the
    # decode. Nothing gates them: the differential drives the mdp layer through
    # `decisions`, not through a gym mode, so a wrong decode here is invisible
    # to conformance, laws and the differential alike. That is why each mode
    # carries a bit-exact reproduction test instead (F30's limitation, second
    # occurrence — see UPSTREAM_PROPOSAL_degenerate_discrete_bound.md's sibling).

    def _sigma_cascade(self, mid: mdp.AdiFlexState,
                       sigmas: Sequence[int]) -> tuple[int, ...]:
        """`order_protection`: protection levels -> the allocation vector.

        Available stock is poured into the interleaved priority cascade

            sigma_0, adv[0], sigma_1, adv[1], ..., sigma_{T-1}, adv[T-1]

        where a sigma box RETAINS stock (it stays in inventory) and an adv box
        SHIPS it early. `sigma_0` is identically 0 and is not part of the
        action — reserving ahead of the nearest class still ahead is weakly
        dominated (see `AdiFlexScenario.n_sigma`), so `sigmas` is indexed from
        t = 1 and `sigmas[k - 1]` is the box that precedes `adv[k]`.

        AT T_dl = 2 THIS IS EXACTLY PL(sigma). The cascade yields
        `a[1] = clip(surplus - a[0] - sigma_1, 0, adv[1])`, and §4.2's rule is
        `clip(surplus - sigma, 0, outstanding)` evaluated where `surplus` is
        already the RUNNING remainder after component 0 took its share — the
        same quantity. That equivalence is the mode's acceptance test, and it
        is checked to the float, not to a tolerance.
        """
        rem = max(mid.inv, 0)
        out: list[int] = []
        for k in range(self.n_alloc):
            if k >= 1:
                rem -= min(rem, int(sigmas[k - 1]))     # retained, not shipped
            a_k = min(rem, int(mid.adv[k]) if k < len(mid.adv) else 0)
            out.append(a_k)
            rem -= a_k
        return tuple(out)

    def _target_order(self, state: mdp.AdiFlexState, action: int) -> int:
        """`target_ip` / `target_mip`: an order-up-to level -> an order.

        The reference is THE ARM'S OWN SUFFICIENT STATISTIC, which is the whole
        reason the mode is admissible where F30's was not. F30 computed the MIP
        for every arm, handing `vec` the transform RQ4 asks whether a network
        can rebuild; splitting the mode in two hands each arm only what it
        already observes:

          target_ip   IP = inv + sum(pipe)     — `vec` observes inv and pipe
          target_mip  u  = IP - sum(adv)       — eq. (7), `vec_mip` observes it

        IP, not on-hand: the two coincide at L = 0 (every het board) but on-hand
        would re-order stock already in transit at L > 0, so this keeps the mode
        correct on the parked L-sweep instances rather than accidentally right
        on the ones we run.

        `action` is the raw index: 0 is the no-order sentinel, and i >= 1 is the
        target level i - 1. See __init__ for why the sentinel is needed and why
        no negative target is.
        """
        if action == 0:
            return 0                      # the sentinel, unconditionally
        ref = state.inv + sum(state.pipe)
        if "mip" in self.action_mode:
            ref -= sum(state.adv)
        return int(np.clip((action - 1) - ref, 0, self.order_max))

    # -- action mask (sb3_contrib MaskablePPO contract) -----------------------

    def action_masks(self) -> np.ndarray:
        """Boolean mask over Discrete(order_max + 1): the current phase's live range."""
        if self.action_mode in PROTECT_MODES:
            # every point of the MultiDiscrete box is feasible — nothing to
            # enforce (IR feasibility_strategy="none"). The all-true mask is
            # kept so the algorithm axis does not move with the action axis:
            # `a0` requires MaskablePPO, and swapping to plain PPO here would
            # confound the encoding comparison with an algorithm change.
            return np.ones(int(sum(self.action_space.nvec)), dtype=bool)
        mask = np.zeros(int(self.action_space.n), dtype=bool)
        if self._phase == PHASE_ORDER:
            # a TARGET is legal at any value — `y <= reference` simply orders
            # nothing — so the order phase is unmasked in every mode
            mask[:] = True
        else:
            assert self._mid is not None
            hi = mdp.valid_allocation(
                self._scenario_ep, self._mid,
                index=len(self._alloc_buf), taken=sum(self._alloc_buf),
            )
            mask[: min(hi, self.order_max) + 1] = True
        return mask

    # -- episode flow ----------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        self._episode_seed = seed if seed is not None else int(np.random.randint(0, 2_147_483_647))

        sc = self.scenario
        self._scenario_ep = sc(self._episode_seed) if callable(sc) else sc

        self._state, self._info = mdp.init_state(self._scenario_ep, self._episode_seed)
        # the state the period's decisions are made ON — held across the
        # period so the log can print state -> action -> result in that order
        self._state_in    = self._state
        self._mid         = None
        self._info1       = {}
        self._phase       = PHASE_ORDER
        self._alloc_buf   = []
        self._step        = 0
        self.total_reward = 0.0

        return self._get_obs(), self._info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        vec = np.asarray(action).reshape(-1)
        act = int(vec[0])

        if self.action_mode in PROTECT_MODES:
            # ONE agent step per period: both decisions were taken here, at the
            # pre-demand information set. advance1 realizes the draw, the
            # cascade turns the protection levels into the allocation vector,
            # and advance2 closes the period — no control is handed back in
            # between, because there is no second decision to take.
            self._state_in = self._state
            if self.action_mode in TARGET_MODES:
                act = self._target_order(self._state, act)
            mid, self._info1 = mdp.advance1(
                self._scenario_ep, self._state, order=act)
            if self.alloc_enabled and self.n_alloc > 0:
                allocate = self._sigma_cascade(mid, [int(x) for x in vec[1:]])
                _s, _i2 = mdp.advance2(self._scenario_ep, mid, allocate)
            else:
                # homogeneous branch: advance2 forces the maximal fill and
                # IGNORES its argument, so nothing is passed (F20)
                _s, _i2 = mdp.advance2(self._scenario_ep, mid)
            self._state, self._info = _s, mdp.merge_info(self._info1, _i2)
            self._mid   = None
            self._phase = PHASE_ORDER
        elif self._phase == PHASE_ORDER:
            self._state_in = self._state          # the period opens here
            if self.action_mode in TARGET_MODES:
                act = self._target_order(self._state, act)
            mid, self._info1 = mdp.advance1(self._scenario_ep, self._state, order=act)
            if self.alloc_enabled and self.n_alloc > 0:
                # hand control back for the allocation decision — demand is now
                # visible, the period is still open, no cost has been charged
                self._mid       = mid
                self._phase     = PHASE_ALLOC
                self._alloc_buf = []
                self._step     += 1
                info = dict(self._info)
                info["phase"] = "allocate"
                return self._get_obs(), 0.0, False, False, info
            # homogeneous branch: there is no allocation decision here.
            # advance2 IGNORES its `allocate` argument when alloc_enabled is
            # False and forces the maximal fill, so nothing is passed — a
            # literal here reads as though the value caused the fill, and 0
            # means "fill nothing" everywhere else in this file (F20).
            _s, _i2 = mdp.advance2(self._scenario_ep, mid)
            self._state, self._info = _s, mdp.merge_info(self._info1, _i2)
        else:
            assert self._mid is not None
            self._alloc_buf.append(act)
            if len(self._alloc_buf) < self.n_alloc:
                # more components of the SAME decision to collect — the classes
                # share one information set, so these micro-steps are an action
                # encoding (the slot pattern: several agent steps inside one information set, the phase carried as an observation feature), not extra MDP periods:
                # nothing stochastic happens between them and no cost is charged
                self._step += 1
                info = dict(self._info)
                info["phase"] = "allocate"
                return self._get_obs(), 0.0, False, False, info
            _s, _i2 = mdp.advance2(
                self._scenario_ep, self._mid, tuple(self._alloc_buf)
            )
            self._state, self._info = _s, mdp.merge_info(self._info1, _i2)
            self._mid       = None
            self._phase     = PHASE_ORDER
            self._alloc_buf = []

        terminated = self._state.terminated
        reward = -self._info["cost"]["total"]
        self.total_reward += reward
        self._step        += 1

        # One row per PERIOD (the alloc micro-steps return early), laid out
        # state -> action -> result, which is the order the period happens in
        # (F29). Before: the action came first and the only state printed was
        # the CLOSING one, so a reader could not see what the decision was made
        # on; `adv` and per-class `fills` were absent entirely, which made the
        # early-delivery cascade unreadable and `d` easy to misread as the
        # profile. `pipe` matters at L > 0 and was never there at all.
        i, a, b = self._info, self._state_in, self._state
        self.logger_s.info(
            f"{self._episode_seed}\t{self._step}\t{i['action_period']}\t"
            # state the decision was made on
            f"{a.inv}\t{a.pipe}\t{a.adv}\t"
            # action
            f"{i['order']}\t{i['allocate']}\t"
            # what the period realized
            f"{i['d']}\t{i['received']}\t{i.get('fills', ())}\t{i['early_fill']}\t"
            # resulting state
            f"{b.inv}\t{b.pipe}\t{b.adv}\t"
            # what it cost
            f"{i['cost']['order_fixed']:.0f}\t{i['cost']['holding']:.0f}\t"
            f"{i['cost']['backorder']:.0f}\t{i['cost']['total']:.0f}\t"
            f"{reward:.4f}\t{self.total_reward:.4f}"
        )

        if terminated:
            sc = self._scenario_ep
            rates = "\t".join(str(r) for r in sc.lambda_seg)   # one column per due-offset
            self.logger_e.info(
                f"{self._episode_seed}\t{self._step}\t{sc.scenario_name}\t"
                f"{rates}\t{sc.L}\t{self.total_reward:.4f}"
            )

        return self._get_obs(), reward, terminated, False, self._info


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from adi_flex_scenarios import SCENARIOS

    rng = np.random.default_rng(0)
    for name in ("homog_L0_T2", "het_exp4", "homog_grid", "het_mix"):
        sc = SCENARIOS[name]
        for observation_mode in ("vec",) + MIP_MODES:
            env = AdiFlexEnv(scenario=sc, observation_mode=observation_mode)
            obs, info = env.reset(seed=42)
            assert env.observation_space.contains(obs), f"reset obs {obs} out of space"
            terminated = False
            steps = 0
            while not terminated:
                mask = env.action_masks()
                assert mask.any(), "mask must never be empty (0 is always feasible)"
                action = int(rng.choice(np.flatnonzero(mask)))
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                assert env.observation_space.contains(obs), \
                    f"{name}/{observation_mode}: obs {obs} out of space"
            # a period is one order step plus one per allocation component
            expected = env.horizon_n * (1 + env.n_alloc if env.alloc_enabled else 1)
            assert steps == expected, f"{name}: {steps} steps, expected {expected}"
            print(
                f"{name:12s} obs={observation_mode:8s} "
                f"steps={steps:3d}  total_reward={env.total_reward:10.2f}"
            )

"""Gymnasium wrapper for the serial multi-echelon simulator (Clark & Scarf 1960)."""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

from clark_scarf_mdp import (
    ClarkScarfState,
    advance1,
    advance2,
    echelon_stock,
    init_state,
    ship_capacity,
)
from clark_scarf_scenarios import ClarkScarfScenario


def _cumsum(v: list[float]) -> list[float]:
    out, run = [], 0.0
    for x in v:
        run += x
        out.append(run)
    return out


def to_echelon(
    stock: list[float], transits: list[list[float]]
) -> list[list[float]]:
    """Raw installation view -> the paper's echelon coordinates.

    ``echelon_stock[j] = sum(stock[0..j]) + sum over slots of transit_s[0..j-1]``
    — the in-flight terms offset one level, because Assumption 3 puts stock in
    transit *to* a level in the echelon **above** it ("at a lower level or in
    transit to a lower level"). Each pipeline slot then gets its OWN running
    sum, mirroring the paper's separate ``w_1`` coordinate — one sum over the
    combined transit would collapse per-slot timing and be lossy.

    ``transits`` carries the observable slots 1..leadtime-1 (post-A, slot
    ``leadtime`` is empty until dispatch). Invertible per block: each
    ``transit_s = diff(ct_s)`` and ``stock = diff(es - shift(sum of ct_s))``.
    See ``from_echelon``.
    """
    cts = [_cumsum(t) for t in transits]
    es = [
        s + sum(ct[j - 1] for ct in cts) if j > 0 else s
        for j, s in enumerate(_cumsum(stock))
    ]
    return [es] + cts


_TTG_SUFFIX = "_ttg"


def split_obs_mode(observation_mode: str) -> tuple[str, bool]:
    """``"echelon_ttg"`` -> ``("echelon", True)``.

    The time encoding rides on the mode NAME rather than a separate flag so
    that the IR can declare it: spec §7 (v0.9.33) makes `features` exhaustive
    and the declared menu the implemented menu, and there is nowhere in the IR
    to say "this mode, but with a different time feature". Four names, four
    declared modes, and `test_declared_features_are_exactly_what_the_gym_
    renders` covers all of them.
    """
    if observation_mode.endswith(_TTG_SUFFIX):
        return observation_mode[: -len(_TTG_SUFFIX)], True
    return observation_mode, False


def observation(
    scenario: ClarkScarfScenario,
    state: ClarkScarfState,
    observation_mode: str,
) -> np.ndarray:
    """The agent view of a pre-dispatch state — the OBSERVATION CONTRACT.

    Module-level rather than a method because `clark_scarf_policy` must build
    the identical vector without standing up a gym (spec §12: the deployable
    policy imports the transform from the module that owns it, and does not
    re-derive it).

    Layout, for a chain of `n = n_echelons` links and lead time `L`::

        [ period / horizon ,  block_0 ... block_{L-1} ]        each block n wide

        block_0      on-hand stock at each installation
        block_{s+1}  units in flight into each installation, landing after
                     s+1 more arrival events  (s = 0 .. L-2)

    There are `L - 1` transit blocks, not `L`: at the observation point the A
    event has just shifted every pipeline and nothing has been dispatched, so
    the last slot is necessarily empty and carries no information.

    Both modes read exactly the same underlying numbers; `echelon` applies a
    component-wise running sum, which is invertible — see `from_echelon` and
    the coordinate-change test that pins it.
    """
    coords, ttg = split_obs_mode(observation_mode)
    n = scenario.n_echelons
    n_transit = scenario.leadtime - 1
    stock = list(state.stock[:n])
    transits = [[state.pipeline[k][s] for k in range(n)] for s in range(n_transit)]
    blocks = [stock] + transits

    if coords == "echelon":
        blocks = to_echelon(stock, transits)

    time_feat = (scenario.horizon - state.period) if ttg else (
        state.period / scenario.horizon)
    return np.asarray([time_feat] + [x for b in blocks for x in b], dtype=np.float32)


def from_echelon(blocks: list[list[float]]) -> list[list[float]]:
    """Inverse of ``to_echelon`` — the guarantee the headline experiment rests on.

    If this ever stops being an exact inverse, the raw/echelon comparison is
    measuring information rather than representation.
    """
    if len(blocks) == 1:
        return [list(np.diff(blocks[0], prepend=0.0))]
    es, cts = blocks[0], blocks[1:]
    ct_total = [sum(vals) for vals in zip(*cts)]
    shifted = [0.0] + list(ct_total[:-1])
    stock = np.diff([e - s for e, s in zip(es, shifted)], prepend=0.0)
    return [list(stock)] + [list(np.diff(ct, prepend=0.0)) for ct in cts]


class ClarkScarfEnv(gym.Env):
    """Gymnasium wrapper around the domain-level serial multi-echelon simulator.

    Observations are taken at the pre-dispatch state (after ``advance1``, i.e.
    after arrivals land and before any link ships), and are **sliced to the
    live levels** — a chain of N levels presents exactly N stock entries, never
    the model file's padding.

    observation_mode — the campaign's headline design axis:

    - ``"raw"``      physical stock at each installation, plus each observable
                     in-flight pipeline slot (slots 1..leadtime-1)
    - ``"echelon"``  the same vector under the paper's coordinates: the running
                     sums, applied **component-wise** to stock and to in-flight
                     stock separately

    The component-wise part matters. A single running sum over stock plus the
    combined transit would collapse distinct vectors into one and be genuinely
    LOSSY at any leadtime >= 2 — it would not be a re-representation at all.
    Clark & Scarf's own state for the two-echelon, two-period-lead case is
    ``C_n(x1, w1, x2)``: in-transit ``w1`` is carried *separately* from echelon
    stock. Summing each component on its own keeps the map invertible, so the
    two modes carry **identical information** and differ only in coordinates.

    That is the point of the experiment: it tests REPRESENTATION, not
    information. An MLP can express a cumulative sum (a lower-triangular matrix
    of ones), so if ``raw`` underperforms, that is an optimization finding
    rather than an impossibility.

    The ACTION mode is part of what the agent is given, too. A
    ``target_discrete`` mode once decoded through ``echelon_position`` and the
    order-up-to form, which handed the agent Clark & Scarf's two central
    constructs whatever it observed and voided the campaign's structure claim
    (ESCALATION #E11). It has been REMOVED rather than kept as a non-default:
    on a branch whose question is whether the agent finds that structure, an
    encoding that supplies it is not a lever, it is a leak. Every surviving
    mode is echelon-free — each reads only ``ship_capacity``, raw on-hand at
    the source.

    Both OBSERVATION modes are prefixed by the normalized period, because the
    horizon is finite and the optimal critical numbers are time-varying (the
    paper's ``x_bar_n`` subscript). It is mode-independent, so it does not
    disturb the comparison.

    Action — the IR decision ``ship``: one shipment quantity per link, an
    N-vector. Feasibility is a per-component box: each link is clipped to what
    its source installation holds, and the top link (drawing from the outside
    supplier) to the action-scale cap ``ship_max``. Three encodings of that one
    decision survive, and each is echelon-free:

    * ``ship_fraction`` (default) — the action IS the fraction of what the
      source can supply, on ``[0,1]``, rounded iff the demand distribution is
      discrete. The rounding is a property of the RENDERING, not of the
      encoding, which is why there is no second "discrete fraction" mode;
    * ``ship_discrete`` — whole units on a categorical head, the lattice
      branch's arm and the encoding every recorded run used;
    * ``ship_absolute`` — identity on ``[0, ship_max]``, the paper-native
      encoding kept as the naive control. Raw action 0 means "ship nothing"
      there, and the useful range is a sliver of the box, which is why it
      measured 22560 against an optimum of 982 — the result the other two
      encodings exist because of.

    Reward: ``-total_cost`` per period.
    """

    metadata = {"render_modes": []}

    OBS_MODES = ("raw", "echelon", "raw_ttg", "echelon_ttg")
    # Exactly the modes the IR declares, in its order. Spec §7: what the gym
    # renders, the IR declares -- and the menu holds in BOTH directions, so a
    # mode implemented here and named nowhere in `gym.action_modes` is a
    # rendering nobody froze. `ship_fraction_bins` and `ship_scaled` were in
    # that state and are removed rather than declared: this board runs
    # `ship_discrete` only, and an encoding kept for a parked frame is code
    # that no gate exercises. The earlier downstream campaign's history has
    # both if the density work resumes.
    ACTION_MODES = ("ship_discrete", "ship_fraction", "ship_absolute")

    # `ship_discrete` grid: quantities 0 .. DISCRETE_SPAN * mean demand, in
    # whole units — demand is Poisson and every stock is integral, so a whole
    # unit is the natural resolution and nothing finer is meaningful
    DISCRETE_SPAN = 4

    def __init__(
        self,
        scenario: ClarkScarfScenario,
        observation_mode: str = "raw",
        action_mode: str = "ship_fraction",
        order_max: float | None = None,
        logger_filename: str | None = None,
    ) -> None:
        super().__init__()

        assert observation_mode in self.OBS_MODES, (
            f"observation_mode must be one of {self.OBS_MODES}, "
            f"got '{observation_mode}'."
        )
        assert action_mode in self.ACTION_MODES, (
            f"action_mode must be one of {self.ACTION_MODES}, got '{action_mode}'."
        )

        self.scenario = scenario
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        # DIAGNOSTIC LEVER (L2(gym), temporary). An upper bound on what the TOP
        # link — the one drawing from the unlimited outside supplier — may be
        # asked to ship. The IR's `ship_max` is UNTOUCHED and the model stays
        # frozen: this caps what the encoding can REQUEST, not what the
        # environment allows, so `step` still clips to `ship_capacity` exactly
        # as before and the feasible set is unchanged.
        #
        # Why it exists: with the fraction taken of `ship_capacity`, the top
        # link's cap is `ship_max` = 20x mean demand, and the trained L1 policy
        # carried a learned std of 0.178 in fraction units there = **35.6 units
        # of per-period action noise, 3.6x mean demand**, against a DP that
        # ships ~10. This lever tests whether that scale is what costs the
        # continuous head its 1570-vs-993 gap.
        self.order_max = order_max

        self.n_live = scenario.n_echelons
        # observable pipeline slots at the decision point: post-A the LAST
        # slot is empty until dispatch, so exactly leadtime-1 slots can be
        # non-zero. At leadtime 1 the pipeline is invisible (a dead input).
        # Each observable slot is its own block across installations, keeping
        # the raw <-> echelon map invertible.
        self.n_transit = scenario.leadtime - 1

        self._state: ClarkScarfState
        self._info: dict
        self._episode_seed: int = 0
        self.total_reward: float = 0.0

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        self.logger_e = logging.getLogger("episode_logger")
        self.logger_s = logging.getLogger("step_logger")
        if logger_filename is not None:
            self._init_logger(logger_filename)

    # -- spaces --------------------------------------------------------------

    @property
    def n_bins(self) -> int:
        """Number of whole-unit shipment choices per link, for `ship_discrete`."""
        return int(self.DISCRETE_SPAN * float(self.scenario.demand.mean())) + 1

    def _build_action_space(self) -> gym.spaces.Space:
        if self.action_mode == "ship_discrete":
            return gym.spaces.MultiDiscrete([self.n_bins] * self.n_live)
        if self.action_mode == "ship_fraction":
            # a fraction lives in [0,1] and the space says so. NOT symmetric:
            # the [-1,1] convention exists for quantities that are signed
            # DEVIATIONS from a reference quantity, and a fraction is not one. Mapping (a+1)/2 would buy a slightly
            # better initial action distribution at the cost of every readback,
            # figure and trajectory carrying the indirection — and under the
            # `ship_capacity` cap it would centre the TOP link at half of
            # ship_max, ten times mean demand, which makes that link's known
            # scale problem worse rather than better.
            return gym.spaces.Box(0.0, 1.0, shape=(self.n_live,), dtype=np.float32)
        return gym.spaces.Box(
            low=0.0,
            high=float(self.scenario.ship_max),
            shape=(self.n_live,),
            dtype=np.float32,
        )

    def _to_quantities(self, act: np.ndarray) -> np.ndarray:
        """Agent action -> shipment quantities, before the availability clip.

        ``ship_discrete`` (default) — MultiDiscrete, one categorical head per
            link; the bin index **is** the shipment quantity in whole units.
        ``ship_fraction`` — the decision is a FRACTION of what the source can
            supply: ``q = (a+1)/2 * ship_capacity[k]`` on a symmetric [-1,1]
            box, rounded half-up iff the demand distribution is discrete (see
            ``_fraction_to_quantities``). One mode, both renderings: a fraction
            is one decision and integrality is a property of the rendering, so
            splitting it into a discrete and a continuous mode would put a
            rendering choice into the action space. Echelon-free — it reads
            ``ship_capacity``, raw on-hand at the source, never
            ``echelon_position`` (#E11/FM3), so an arm using it can carry a
            structure claim. "Ship everything available" sits at a CONSTANT
            address (``a = +1``) in every state, which is LV1 applied to the
            availability constraint, and no mask applies or is needed because
            the fraction is taken OF the cap (LV3).
        ``ship_absolute`` — identity on [0, ship_max]; the paper-native
            encoding, kept as the NAIVE CONTROL. It carries the action-scale
            hazard by construction: SB3's Gaussian starts at raw 0 with std 1,
            so early actions cover ~[0,3] of a box 20x mean demand wide, and
            raw 0 means "ship nothing" — measured at 22560 against an optimum
            of 982. That measurement is the reason the other encodings exist,
            so the mode earns its place by failing legibly.

        Why discrete is the default, and not a rescaled Gaussian:

        1. **Integrality is the problem's own.** Demand is Poisson and every
           stock is integral (the IR marks the decisions ``integer``); a
           fractional shipment would put non-integers into an integer state and
           quietly cost the DP reference its exactness.
        2. **A categorical head carries no action-scale hazard.** A Gaussian
           over a box 20x mean demand wide must crawl its mean into the useful
           range; a categorical head reweights, and every quantity is reachable
           from the first update.

        """
        if self.action_mode == "ship_discrete":
            return np.asarray(act, dtype=np.float64)     # bin index == quantity
        if self.action_mode == "ship_fraction":
            return self._fraction_to_quantities(act)
        return np.clip(act, 0.0, float(self.scenario.ship_max))   # ship_absolute

    def _effective_caps(self) -> np.ndarray:
        """The cap the fraction encodings take their fraction OF, per link.

        It is ``ship_capacity`` itself, unclamped — the IR's own per-link bound
        on a shipment, with nothing added. Below the top link that is the source
        installation's on-hand stock, so ``a = +1`` (or the top bin) means
        "ship everything available" in every state, which is the LV1 property
        these encodings exist for. On the top link it is ``ship_max``, the
        decision's declared upper bound (``bounds: [0, ship_max]``).

        An earlier version clamped this to ``min(available, DISCRETE_SPAN*mean)``
        and a later one substituted a dedicated action-scale constant for the
        top link. Both are withdrawn, and the reason is worth keeping: the
        clamp's value had been justified from the range of shipments the *DP
        policy* makes, which sets the action space from the reference solution
        — the #E11 / FM3 failure, one layer down — and it samples only states
        the DP visits, so it says nothing about the depleted states an agent
        actually trains through. Structural reasoning gives a different answer
        anyway: the top echelon must cover ``n_echelons`` levels of committed
        flow over ``leadtime`` periods, so a legitimate recovery shipment can
        approach ``(n_echelons*leadtime + 1)*demand_mean`` = 70 here, well past
        the 40 the clamp allowed. Rather than invent a third constant, the
        encoding uses the bound the schema already declares.

        KNOWN CONSEQUENCE, carried deliberately: ``ship_max`` is generous
        (20x mean demand), so on the top link the useful region occupies a
        narrow part of the action box and the encoding inherits the action-scale
        hazard that `ship_absolute` pays for. That is left as a TRAINING question, to be
        answered by an L0/L1 result rather than pre-engineered away with a
        number nobody can derive. If it bites, the documented alternatives are a
        structure-derived scale ((n_echelons*leadtime + 1)*demand_mean), or an
        absolute encoding on the unconstrained link only.
        """
        caps = np.asarray(ship_capacity(self.scenario, self._state), dtype=np.float64)
        if self.order_max is not None:
            caps[self.n_live - 1] = min(caps[self.n_live - 1], float(self.order_max))
        return caps

    def _fraction_to_quantities(self, act: np.ndarray) -> np.ndarray:
        """Fraction of the per-link cap; integral iff the RENDERING is.

        The action IS the fraction: ``q = a * ship_capacity[k]`` with
        ``a`` in ``[0,1]``. No affine remap — an action of 0.7 means "70% of
        what my source can supply" wherever it is read, which matters because
        this domain does policy readbacks (§14).

        A symmetric ``[-1,1]`` box was used first, on the reasoning that SB3's
        Gaussian starts at raw 0 and a ``[0,1]`` box would therefore start by
        shipping nothing. That imported a caveat from `ship_absolute`, where raw 0 meant
        "ship nothing" **and** the useful range was ~1.5% of the box, hence
        unreachable — the failure was the unreachability, not the zero. Here
        the box *is* the useful range. What the two choices actually trade is
        the initial action distribution (SB3 clips an unbounded Gaussian, so
        ``[0,1]`` puts ~50% of initial mass at 0 and ``[-1,1]`` ~16%), and
        ``[-1,1]`` loses that trade under the `ship_capacity` cap: it would
        centre the TOP link at half of ship_max, ten times mean demand, every
        episode at initialization.

        **Rounding is keyed on the demand distribution, not on the action
        mode.** The decision is a fraction and stays a fraction; whether the
        quantity it resolves to must land on an integer is a property of the
        *rendering*, and the IR already says so — `narrowed` on `ship` reads
        "integer feasibility under the `poisson` selection ONLY", caused by
        "selection: demand = poisson + design: integer shipment quantities".
        This is that clause implemented rather than encoded structurally. Under
        a discrete selection the shipment is rounded half-up so stocks stay on
        the lattice and the DP reference keeps its exactness; under a
        continuous one nothing rounds and distinct actions give distinct
        quantities.

        Half-UP and not ``np.rint``: rint rounds halves to EVEN, which against
        a state-varying cap hands even quantities three times the prior mass of
        odd ones — a parity bias invisible in any aggregate score.

        There is deliberately no second "discrete fraction" mode. A fraction is
        one decision; giving it two action modes would put a rendering choice
        into the action space, which is the same category error `narrowed`
        exists to prevent.
        """
        frac = np.clip(np.asarray(act, dtype=np.float64), 0.0, 1.0)
        q = frac * self._effective_caps()
        if self.scenario.demand.is_discrete:
            q = np.floor(q + 0.5)
        return q

    def _build_observation_space(self) -> gym.spaces.Box:
        """Size the box to the genuinely REACHABLE range, not a typical one.

        The top link draws from an unlimited outside supplier, so a policy that
        ships the cap every period drives stock to
        ``mean + ship_max * horizon``; and the retailer can backlog at most
        ``demand_max * horizon``. Those are true envelopes, so
        ``contains(obs)`` holds under any policy including uniform-random —
        which is what the Stage-2 gate exercises.

        The box is therefore much wider than anything sensible play visits.
        That is deliberate: this space is a VALIDITY envelope, not a
        normalization device. Scaling is VecNormalize's job (the IR sets
        ``obs_normalization.enabled``), and clipping into a tighter box would
        distort the observation and destroy the invertibility that the
        raw/echelon comparison depends on.
        """
        sc = self.scenario
        mean = float(sc.demand.mean())
        hi_lvl = mean + float(sc.ship_max) * sc.horizon
        lo_lvl = -float(sc.demand.max()) * sc.horizon
        # echelon entries are running sums over up to n_live levels
        coords, ttg = split_obs_mode(self.observation_mode)
        scale = self.n_live if coords == "echelon" else 1
        n_blocks = 1 + self.n_transit
        width = self.n_live * n_blocks
        # the time entry: `period / horizon` lives in [0, 1], `time_to_go` in
        # [0, horizon]. The envelope must cover what is rendered or
        # `contains()` fails under random actions (the Stage-2 gate)
        t_hi = float(sc.horizon) if ttg else 1.0
        low = np.concatenate([[0.0], np.full(width, lo_lvl * scale)]).astype(np.float32)
        high = np.concatenate([[t_hi], np.full(width, hi_lvl * scale)]).astype(np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # -- observation ---------------------------------------------------------

    def _get_obs(self, state: ClarkScarfState) -> np.ndarray:
        return observation(self.scenario, state, self.observation_mode)

    # -- gym API -------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            # Draw a fresh episode from the per-env stream (the inv_single
            # exemplar's pattern). Without this branch, gymnasium's auto-reset
            # (which passes no seed) replays the SAME episode_seed forever, and
            # training silently runs on n_envs fixed demand paths — see
            # ESCALATION.md #E17, which is what that cost this campaign.
            self._episode_seed = int(self.np_random.integers(0, 2_147_483_647))
        self._state, self._info = init_state(self.scenario, self._episode_seed)
        self.total_reward = 0.0
        state1 = advance1(self.scenario, self._state)
        self._state = state1
        return self._get_obs(state1), dict(self._info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # the agent emits one quantity per link, and the model is exactly that
        # wide -- no padding step since F9 collapsed the action to `ship` with
        # dim: "n_echelons"
        act = self._to_quantities(np.asarray(action, dtype=np.float64).reshape(-1))
        caps = ship_capacity(self.scenario, self._state)
        ship = [float(np.clip(act[k], 0.0, caps[k])) for k in range(self.n_live)]

        state2, info = advance2(self.scenario, self._state, ship)
        reward = -info["cost"]["total"]
        self.total_reward += reward

        terminated = state2.terminated
        truncated = False
        self._info = info

        if not terminated:
            state2 = advance1(self.scenario, state2)
        self._state = state2

        return self._get_obs(state2), reward, terminated, truncated, dict(info)

    @property
    def last_info(self) -> dict:
        return self._info

    # -- action masking (sb3_contrib MaskablePPO) ----------------------------

    def action_masks(self) -> np.ndarray:
        """Feasible shipment quantities per link, as MaskablePPO expects.

        Only meaningful for ``ship_discrete``. Returns the per-dimension masks
        concatenated in order — length ``n_live * n_bins`` — where entry
        ``(k, j)`` is True iff shipping ``j`` units on link ``k`` is feasible,
        i.e. ``j <= ship_capacity[k]``.

        **Why this matters here.** Without masking, every quantity above the
        source installation's stock is silently clipped to the same value, so a
        large block of the categorical head's options are *behaviourally
        identical* and carry no gradient distinguishing them. Measured at
        `n3_l2_p09`: the lower links typically hold ~20 units against a 41-bin
        head, so roughly **half the categorical mass is wasted** on actions that
        cannot be taken.

        Note the **top** link draws from an unlimited outside supplier, so
        `ship_capacity` never binds there and its mask is all-True; masking
        therefore only constrains the lower links.

        The mask reads ``ship_capacity`` at the current decision point, which is
        the same per-component box ``step()`` clips against — so a masked policy
        can never emit an action the env would have to alter.
        """
        caps = ship_capacity(self.scenario, self._state)
        n = self.n_bins
        mask = np.zeros((self.n_live, n), dtype=bool)
        for k in range(self.n_live):
            hi = int(min(np.floor(caps[k]), n - 1))
            mask[k, : hi + 1] = True
            mask[k, 0] = True          # shipping nothing is always feasible
        return mask.reshape(-1)

    # -- logging -------------------------------------------------------------

    def _init_logger(self, filename: str) -> None:
        fmt = logging.Formatter("%(asctime)s\t%(message)s")
        for lg, suffix in ((self.logger_e, "episode"), (self.logger_s, "step")):
            lg.handlers.clear()
            lg.setLevel(logging.DEBUG)
            fh = logging.FileHandler(f"{filename}_{suffix}.log")
            fh.setFormatter(fmt)
            lg.addHandler(fh)


# ---------------------------------------------------------------------------
# Module self-test: random-action episodes in every obs/action mode,
# asserting observation_space.contains(obs) at every step (Stage-2 gate)
# ---------------------------------------------------------------------------


def _smoke(scenario: ClarkScarfScenario, episodes: int = 3) -> None:
    for obs_mode in ClarkScarfEnv.OBS_MODES:
        for act_mode in ClarkScarfEnv.ACTION_MODES:
            env = ClarkScarfEnv(scenario, observation_mode=obs_mode, action_mode=act_mode)
            for ep in range(episodes):
                obs, _ = env.reset(seed=100 + ep)
                assert env.observation_space.contains(obs), (
                    f"{scenario.scenario_name}/{obs_mode}/{act_mode}: reset obs "
                    f"outside space: {obs}"
                )
                steps = 0
                while True:
                    a = env.action_space.sample()
                    obs, r, term, trunc, _ = env.step(a)
                    steps += 1
                    assert env.observation_space.contains(obs), (
                        f"{scenario.scenario_name}/{obs_mode}/{act_mode}: step "
                        f"{steps} obs outside space: {obs}"
                    )
                    assert np.isfinite(r), f"non-finite reward at step {steps}"
                    if term or trunc:
                        break
                assert steps == scenario.horizon, (
                    f"expected {scenario.horizon} steps, got {steps}"
                )
            print(
                f"  {scenario.scenario_name:12s} obs={obs_mode:8s} act={act_mode:5s} "
                f"obs_dim={env.observation_space.shape[0]} "
                f"act_dim={env.action_space.shape[0]}  OK"
            )


def _check_modes_are_invertible(scenario: ClarkScarfScenario) -> None:
    """The two obs modes must be a pure change of coordinates.

    Recovers the raw view from the echelon view by differencing and asserts it
    matches. If this ever fails, the headline experiment is comparing
    information rather than representation.
    """
    n = scenario.n_echelons
    e_raw = ClarkScarfEnv(scenario, observation_mode="raw")
    e_ech = ClarkScarfEnv(scenario, observation_mode="echelon")
    o_raw, _ = e_raw.reset(seed=7)
    o_ech, _ = e_ech.reset(seed=7)
    for _ in range(scenario.horizon):
        a = e_raw.action_space.sample()
        blocks_r = o_raw[1:].reshape(-1, n)
        blocks_e = o_ech[1:].reshape(-1, n)
        recovered = np.asarray(from_echelon([list(b) for b in blocks_e]))
        # tolerance scales with the running sum's magnitude: differencing a
        # float32 cumsum loses absolute precision proportional to the largest
        # partial sum (~1e-7 relative). A genuine failure of invertibility
        # would be an error of order the values themselves, far above this.
        atol = 1e-4 * max(1.0, float(np.abs(blocks_e).max()))
        assert np.allclose(recovered, blocks_r, atol=atol), (
            f"{scenario.scenario_name}: echelon view is not invertible back to "
            f"raw\n  raw={blocks_r}\n  recovered={recovered}"
        )
        o_raw, _, t1, _, _ = e_raw.step(a)
        o_ech, _, t2, _, _ = e_ech.step(a)
        if t1 or t2:
            break
    print(f"  {scenario.scenario_name:12s} raw <-> echelon invertible  OK")


if __name__ == "__main__":
    from clark_scarf_scenarios import SCENARIOS

    print("random-action episodes, observation_space.contains asserted each step:")
    for nm in ("n2_l1_p09", "n3_l2_p09", "n4_l2_p09", "verify_tiny"):
        _smoke(SCENARIOS[nm])
    print("\ncoordinate-change check (raw <-> echelon):")
    for nm in ("n2_l1_p09", "n3_l2_p09", "n4_l2_p09"):
        _check_modes_are_invertible(SCENARIOS[nm])

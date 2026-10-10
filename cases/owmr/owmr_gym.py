"""Gymnasium wrapper for the one-warehouse, N-retailer simulator (owmr)."""

from __future__ import annotations

import copy
import logging
from typing import Any

import gymnasium as gym
import numpy as np

import owmr_mdp as mdp
from owmr_mdp import OwmrState
from owmr_scenarios import OwmrScenario


def observation(
    scenario: OwmrScenario,
    state: OwmrState,
    observation_mode: str,
    last_demand: list[float],
    phase: int = 0,
) -> np.ndarray:
    """The agent's view of a post-receipt state — the OBSERVATION CONTRACT.

    Module-level rather than a method because `owmr_policy` must build the
    identical vector without standing up a gym (spec 12).

    Every mode renders exactly the feature list the IR declares for it, in
    render order (spec 7), led by the countdown ``time_to_go = T - t``:

    - ``raw``      [time_to_go, wh_stock, wh_pipe (l0), rt_stock (N), rt_pipe (N x l_rt, row-major)]
    - ``raw_d``    raw + last period's realized demand per retailer (N)
    - ``raw_seq``  raw + [phase] (0 = order step, 1 = allocation step), for the
                   two-phase ``seq_order_softmax`` only; the caller renders the
                   allocation step from the post-O interstate
    - ``position`` [time_to_go, wh_stock, IP_0, rt_position (N)] — the paper's
                   coordinates; a lossy aggregation of raw (pipeline timing is
                   dropped), so a lever, never the discover arm

    The state is the post-receipt one (after ``advance1``): deliveries have
    landed, nothing has been ordered or shipped. So ``wh_pipe``'s last slot
    and, at l_rt = 1, every ``rt_pipe`` slot read 0 here — those components
    are still rendered because the IR declares the whole vectors, and a
    constant feature costs VecNormalize nothing.
    """
    ttg = float(scenario.horizon - state.period)
    # warehouse on-hand is non-negative by the model (the S event scales
    # requests to on-hand); after a binding scale-down the float residual of
    # `wh_stock -= sum(shipped)` can sit at -1e-15, which the IR's invariant
    # tolerates (>= -1e-9) but a Box with low 0 does not. Render the value the
    # model states.
    wh = max(0.0, float(state.wh_stock))
    if observation_mode == "position":
        obs = [ttg, wh, mdp.echelon_position(state)] + mdp.inventory_position(state)
    else:
        obs = [ttg, wh] + list(state.wh_pipe) + list(state.rt_stock)
        for row in state.rt_pipe:
            obs.extend(row)
        if observation_mode == "raw_d":
            obs.extend(last_demand)
        elif observation_mode == "raw_seq":
            obs.append(float(phase))
    return np.asarray(obs, dtype=np.float32)


# Box half-width for the `order_softmax` logits. SB3's Gaussian starts at mean 0,
# std 1, i.e. at the "one share each" point; a logit of -10 gives a share of
# ~4.5e-5, so every practical allocation is reachable inside the box.
LOGIT_BOX = 10.0
# `order_softmax_free`: SB3 needs finite bounds; this one is never reachable (#E18)
FREE_BOX = 1e9


def anchored_softmax_shares(logits) -> np.ndarray:
    """N logits -> N retailer shares of warehouse on-hand, with an implicit
    warehouse logit fixed at 0: share_i = e^{x_i} / (1 + sum_j e^{x_j}).

    The N + 1 shares (the retained one being 1 / (1 + sum e^{x_j})) lie on the
    simplex, so the retailer shares sum STRICTLY below 1 for every input: the
    MDP's proportional scale-down can never fire, and retention is a smooth,
    always-reachable coordinate. Raw action 0 gives 1/(N+1) to each retailer and
    1/(N+1) retained. Stable form: subtract max(0, max x) before exponentiating.
    """
    x = np.asarray(logits, dtype=np.float64)
    m = max(0.0, float(x.max()))
    e = np.exp(x - m)
    return e / (np.exp(-m) + e.sum())


def relu_shares(weights) -> tuple[np.ndarray, float]:
    """N + 1 weights [x_0 (warehouse), x_1..x_N (retailers)] -> (N retailer
    shares of on-hand, kept share), the `order_relu` decode (ESCALATION #E14).

    r_j = relu(x_j); every share is r_j / sum r. The N + 1 shares sum to 1, so
    the retailers' sum to at most 1 and the MDP's scale-down never fires. Every
    corner is a clip: x_i <= 0 gives retailer i exactly 0; x_0 <= 0 ships
    exactly everything; all x <= 0 (sum r = 0) keeps everything by convention.
    """
    r = np.maximum(0.0, np.asarray(weights, dtype=np.float64))
    total = float(r.sum())
    if total <= 0.0:
        return np.zeros(len(r) - 1), 1.0
    return r[1:] / total, float(r[0] / total)


def nested_shares(weights) -> tuple[np.ndarray, float]:
    """N + 1 weights [x_0 (retention), x_1..x_N (retailers)] -> (N retailer shares
    of on-hand, kept fraction), the `order_nested` decode (ESCALATION #E17).

    Top level: kept k = relu(x_0) / (1 + relu(x_0)) — exactly 0 for x_0 <= 0,
    slope 1 at the boundary, undiluted by the retailer weights. Lower level:
    the shipped (1 - k) is split by relu(x_i) / sum relu(x_j); a retailer with
    x_i <= 0 gets exactly 0; if every retailer weight is <= 0 the split is
    equal. Shares + kept = 1 exactly, so the MDP's scale-down never fires.
    """
    x = np.asarray(weights, dtype=np.float64)
    r0 = max(0.0, float(x[0]))
    kept = r0 / (1.0 + r0)
    r = np.maximum(0.0, x[1:])
    total = float(r.sum())
    split = r / total if total > 0.0 else np.full(len(r), 1.0 / len(r))
    return (1.0 - kept) * split, kept


def free_softmax_shares(logits) -> tuple[np.ndarray, float]:
    """N + 1 free logits [x_0 (retention), x_1..x_N] -> (N retailer shares of
    on-hand, kept share), the `order_softmax_free` decode (ESCALATION #E18):
    one softmax over all N + 1, no anchor, no clip (stable form)."""
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - x.max())
    sh = e / e.sum()
    return sh[1:], float(sh[0])


# `order_catkeep` (ESCALATION #E21): the retention menu, kept fraction of on-hand per option
KEEP_MENU = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)
# `order_catkeep_fine` (ESCALATION #E34, the human: "can we make the grid a bit finer too?"): 12 rungs —
# 0.05 steps across the heuristic's mass (its kept fraction in surplus periods: mean 0.27, p90 ~0.5), then
# 0.5 / 0.6 / 0.8 for the tail (rounding the heuristic to the 6-rung menu cost it +0.23 per episode, #E21)
KEEP_MENU_FINE = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8)


def keep_menu(action_mode: str) -> tuple:
    """The retention menu an action mode decodes with."""
    return KEEP_MENU_FINE if action_mode == "order_catkeep_fine" else KEEP_MENU


def clamp_shipments(ship, cap: float) -> list[float]:
    """Each shipment capped at the IR's per-shipment bound (`order_max`; the `rt_pipe` envelope rests
    on it). A share of on-hand can exceed the bound when the warehouse holds more than the cap — only
    under random play: the hicv and base winners re-evaluated on all 8192 protocol seeds reproduce their
    records with this clamp in place (#E37). The ONE place the bound is enforced: the gym's decode and
    the deployable (`owmr_policy.OwmrPolicy.act`) both call it."""
    cap = float(cap)
    return [min(float(q), cap) for q in ship]


def catkeep_shares(action_tail, menu: tuple = KEEP_MENU) -> tuple[np.ndarray, float]:
    """[k_index, x_1..x_N] -> (N retailer shares of on-hand, kept fraction):
    kept = menu[round(k_index)] (KEEP_MENU by default, KEEP_MENU_FINE for
    `order_catkeep_fine`); the shipped (1 - kept) split by relu weights (exact
    zeros; equal split if all <= 0)."""
    a = np.asarray(action_tail, dtype=np.float64)
    k = menu[int(np.clip(np.rint(a[0]), 0, len(menu) - 1))]
    r = np.maximum(0.0, a[1:])
    total = float(r.sum())
    split = r / total if total > 0.0 else np.full(len(r), 1.0 / len(r))
    return (1.0 - k) * split, k


def dirichlet_shares(shares) -> tuple[np.ndarray, float]:
    """[s_0 (kept), s_1..s_N] on the simplex -> (N retailer shares of on-hand,
    kept share), the `order_dirichlet` decode (ESCALATION #E23). The policy
    samples the simplex directly, so this only guards: negatives clipped,
    renormalized; all zero -> keep everything."""
    r = np.maximum(0.0, np.asarray(shares, dtype=np.float64))
    total = float(r.sum())
    if total <= 0.0:
        return np.zeros(len(r) - 1), 1.0
    return r[1:] / total, float(r[0] / total)


class OwmrEnv(gym.Env):
    """Gymnasium wrapper around the domain-level one-warehouse, N-retailer simulator.

    Observations are taken at the post-receipt state (after ``advance1``): the
    supplier delivery and the retailer deliveries due this period have landed,
    nothing has been ordered or shipped, demand has not been realized. The
    warehouse on-hand the agent sees is exactly the quantity the allocation
    draws on.

    observation_mode (spec 7; the IR's ``gym.observation_modes``):
    - ``"raw"`` (default) — the full physical state plus the countdown
    - ``"raw_d"``         — raw plus last period's realized demand (a control;
                            demand is i.i.d., so it carries no information)
    - ``"position"``      — the paper's coordinates: warehouse on-hand, the
                            warehouse echelon inventory position IP_0, each
                            retailer's inventory position. Lossy, a lever.

    action_mode — one homogeneous continuous vector of width N + 1 (SB3 PPO
    accepts no mixed action spaces, and IR v1 multi-decision modes are
    identity per component):
    - ``"order_frac"`` (default) — [order, f_1..f_N]: the order quantity in
      [0, order_max], then a fraction of warehouse on-hand shipped to each
      retailer, each clipped to [0, 1] INDEPENDENTLY. ``ship_i = f_i *
      wh_stock``; retention is 1 - sum f only while sum f <= 1. Nothing keeps
      the sum below 1: above it the MDP's scale-down ships the whole on-hand
      and retention is exactly 0, a flat region with no gradient toward
      retaining. The L1 winner sits there in every period (ESCALATION #E2,
      median sum 3.49); a Gaussian at init already averages ~1.7. Kept as the
      L1 record's interface, not as a recommendation.
    - ``"order_softmax"`` — [order, x_1..x_N]: the order quantity as above,
      then N logits in [-LOGIT_BOX, LOGIT_BOX] mapped by the ANCHORED softmax
      (warehouse logit fixed at 0): ``ship_i = share_i * wh_stock`` with
      ``share_i = e^{x_i} / (1 + sum e^{x_j})``. The shares sum strictly below
      1, so the scale-down never fires and retention = 1 / (1 + sum e^{x_j}) is
      smooth and reachable. Raw action 0 = order nothing, one sixth of on-hand
      to each retailer, one sixth retained. Structure-free: it reads only
      on-hand, never a position or a target.
    - ``"order_relu"`` — [order, x_0, x_1..x_N], width N + 2 (ESCALATION #E14):
      the EXPLICIT allocation with exact corners. x_0 is the warehouse's own
      weight; with r = relu(x), ship_i = wh_stock * r_i / sum r and the
      warehouse keeps r_0 / sum r (all x <= 0: keep everything). Zero to a
      retailer, all shipped and all kept are clips, not limits; the shares
      never exceed on-hand, so the MDP's scale-down — a feasibility fallback,
      not a rationing rule — does not fire. See `relu_shares`.
    - ``"order_nested"`` — [order, x_0, x_1..x_N], width N + 2 (ESCALATION #E17):
      the NESTED explicit allocation. Top level: kept fraction
      relu(x_0) / (1 + relu(x_0)) of on-hand (exactly 0 for x_0 <= 0). Lower
      level: the rest split by relu(x_i) / sum relu(x_j) (exact zeros; equal
      split if all <= 0). Same allocations as order_relu in coordinates where
      retention is its own, undiluted decision. See `nested_shares`.
    - ``"order_softmax_free"`` — [order, x_0, x_1..x_N], width N + 2 (ESCALATION
      #E18): one softmax over all N + 1 logits, the retention logit x_0 free
      and NO clip — the logits' Box is +-1e9, unbounded in practice. The test of
      whether an unbounded softmax reaches zero shares. See `free_softmax_shares`.
    - ``"order_catkeep"`` — [order, k_index, x_1..x_N], width N + 2 (ESCALATION
      #E21): retention is a DISCRETE choice — k_index (rounded) picks a kept
      fraction from KEEP_MENU — and the rest is split by relu weights as in
      order_nested. Trained with owmr_catkeep_policy.CatKeepPolicy (Gaussian
      x categorical). See `catkeep_shares`.
    - ``"order_catkeep_fine"`` — order_catkeep with the 12-rung menu
      KEEP_MENU_FINE (ESCALATION #E34): k_index in [0, 11]. Same decode, same
      policy class (n_options = 12).
    - ``"order_dirichlet"`` — [order, s_0, s_1..s_N], width N + 2 (ESCALATION
      #E23): the shares THEMSELVES, on the simplex — kept = s_0, ship_i = s_i
      * on-hand. Trained with owmr_dirichlet_policy.DirichletPolicy (Gaussian
      order x Dirichlet shares). See `dirichlet_shares`.
    - ``"order_ship"`` — [order, ship_1..ship_N], every entry a quantity in
      [0, order_max]; the MDP's proportional scale-down handles requests
      exceeding on-hand. The literal encoding, with a dead region.
    - ``"seq_order_softmax"`` — ``order_softmax`` taken in TWO agent steps per
      period (ESCALATION #E10), paired with observation ``raw_seq``: an order
      step (component 0 live, reward 0), then an allocation step (components
      1..N live, reward = -cost) that sees the order in the warehouse pipeline.
      The simulator still advances one period per allocation step; the
      micro-step only gives each decision its own value baseline. An episode is
      2 x horizon agent steps. The ignored components must be masked out of
      PPO's log-probability (owmr_seq_policy.PhaseMaskedPolicy).
    - ``"target_frac"`` — [y0, f_1..f_N]: y0 is an echelon order-up-to level and
      the gym orders max(0, y0 - IP_0); fractions as in order_frac. Hands the
      agent the lower-bound policy's ordering structure — a LEVER under this
      campaign's discover stance, never the arm the claim rests on.

    Reward: ``-cost.total`` per period (sense = minimize) under the default
    reward_mode ``"neg_cost"``; ``"neg_cost_transit"`` (ESCALATION #E19) adds
    h0 on end-of-period stock in transit to the retailers — the paper's own
    accounting, a training-only shaping whose horizon total is policy-
    independent up to the end effect. ``info["cost"]`` is always canonical.
    """

    metadata = {"render_modes": []}

    OBS_MODES = ("raw", "raw_d", "position", "raw_seq")
    REWARD_MODES = ("neg_cost", "neg_cost_transit")
    ACTION_MODES = ("order_frac", "order_ship", "target_frac", "order_softmax", "seq_order_softmax",
                    "order_relu", "order_nested", "order_softmax_free", "order_catkeep", "order_catkeep_fine",
                    "order_dirichlet")

    def __init__(
        self,
        scenario,
        observation_mode: str = "raw",
        action_mode: str = "order_frac",
        logger_filename: str | None = None,
        reward_mode: str = "neg_cost",
    ) -> None:
        super().__init__()
        assert reward_mode in self.REWARD_MODES, (
            f"reward_mode must be one of {self.REWARD_MODES}, got '{reward_mode}'."
        )
        self.reward_mode = reward_mode
        assert observation_mode in self.OBS_MODES, (
            f"observation_mode must be one of {self.OBS_MODES}, got '{observation_mode}'."
        )
        assert action_mode in self.ACTION_MODES, (
            f"action_mode must be one of {self.ACTION_MODES}, got '{action_mode}'."
        )

        # the two-phase mode and its observation come as a pair (#E10)
        assert (action_mode == "seq_order_softmax") == (observation_mode == "raw_seq"), (
            "seq_order_softmax needs observation_mode raw_seq, and raw_seq serves only it")
        self.sequential = action_mode == "seq_order_softmax"
        self._phase = 0
        self._pending_order = 0.0

        # a fixed scenario, or a sampler resolved per episode in reset().
        # Space building reads only family-level attributes (spec 5.2
        # mirroring rule), which a source exposes under the same names.
        self.scenario = scenario
        self._scenario_ep: OwmrScenario | None = None
        self.observation_mode = observation_mode
        self.action_mode = action_mode

        self.n = int(scenario.n_retailers)
        self._state: OwmrState
        self._info: dict
        self._episode_seed: int = 0
        self._step: int = 0
        self.total_reward: float = 0.0

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        self.logger_e = logging.getLogger("episode_logger")
        self.logger_s = logging.getLogger("step_logger")
        if logger_filename is not None:
            self._init_logger(logger_filename)

    # -- spaces ---------------------------------------------------------------

    def _build_action_space(self) -> gym.spaces.Box:
        sc = self.scenario
        cap = float(sc.order_max)
        n = self.n
        low = [0.0] * (n + 1)
        if self.action_mode == "order_ship":
            high = [cap] * (n + 1)
        elif self.action_mode in ("order_softmax", "seq_order_softmax"):
            low = [0.0] + [-LOGIT_BOX] * n
            high = [cap] + [LOGIT_BOX] * n
        elif self.action_mode in ("order_relu", "order_nested"):   # + the warehouse's own weight
            low = [0.0] + [-LOGIT_BOX] * (n + 1)
            high = [cap] + [LOGIT_BOX] * (n + 1)
        elif self.action_mode == "order_dirichlet":               # [order, N + 1 shares on the simplex]
            low = [0.0] + [0.0] * (n + 1)
            high = [cap] + [1.0] * (n + 1)
        elif self.action_mode in ("order_catkeep", "order_catkeep_fine"):   # [order, menu index, N relu weights]
            low = [0.0, 0.0] + [-LOGIT_BOX] * n
            high = [cap, float(len(keep_menu(self.action_mode)) - 1)] + [LOGIT_BOX] * n
        elif self.action_mode == "order_softmax_free":            # retention logit free; box never reachable
            # SB3 refuses an infinite Box; +-1e9 is unbounded in practice (a policy's
            # Gaussian mean never gets near it) and matches the IR's declared envelope
            low = [0.0] + [-FREE_BOX] * (n + 1)
            high = [cap] + [FREE_BOX] * (n + 1)
        else:  # order_frac, target_frac: [quantity-or-target, fractions]
            high = [cap] + [1.0] * n
        return gym.spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _build_observation_space(self) -> gym.spaces.Box:
        """The IR's declared envelopes, resolved from the scenario: a VALIDITY
        envelope sized to the reachable range under any policy (including
        uniform-random, which the Stage-2 gate exercises), not a scale —
        scaling is VecNormalize's job."""
        sc = self.scenario
        n, l0, l_rt = self.n, int(sc.l0), int(sc.l_rt)
        cap = float(sc.order_max)
        T = float(sc.horizon)
        stock_hi = float(sc.mu_sys) + T * cap                    # IR: sum(mu_rt) + horizon_T * cap
        rt_lo = -T * float(max(sc.demand.max()))                 # IR: -horizon_T * demand.max
        if self.observation_mode == "position":
            low = [0.0, 0.0, -np.inf] + [-np.inf] * n
            high = [T, stock_hi, np.inf] + [np.inf] * n
        else:
            low = [0.0, 0.0] + [0.0] * l0 + [rt_lo] * n + [0.0] * (n * l_rt)
            high = [T, stock_hi] + [cap] * l0 + [stock_hi] * n + [cap] * (n * l_rt)
            if self.observation_mode == "raw_d":
                low += [0.0] * n
                high += [np.inf] * n
            elif self.observation_mode == "raw_seq":
                low += [0.0]
                high += [1.0]
        return gym.spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )

    # -- action decode --------------------------------------------------------

    def decode_action(self, action) -> tuple[float, list[float]]:
        """Agent action -> (order, ship requests), the gym's transform.

        Public because anything that reads a trained policy's output — the
        policy probe, the deployable wrapper — must decode exactly as ``step``
        does. Reads the CURRENT post-receipt state (``wh_stock`` for the
        fractions, ``IP_0`` for the target mode).
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        width = self.n + 2 if self.action_mode in ("order_relu", "order_nested", "order_softmax_free", "order_catkeep",
                                                   "order_catkeep_fine", "order_dirichlet") else self.n + 1
        assert a.shape[0] == width, f"expected {width} components, got {a.shape[0]}"
        cap = float(self._scenario_ep.order_max)
        head = float(np.clip(a[0], 0.0, cap))
        if self.action_mode == "target_frac":
            order = max(0.0, head - mdp.echelon_position(self._state))
        else:
            order = head
        if self.action_mode == "order_ship":
            ship = [float(np.clip(x, 0.0, cap)) for x in a[1:]]
        elif self.action_mode in ("order_softmax", "seq_order_softmax"):
            onhand = max(0.0, float(self._state.wh_stock))
            shares = anchored_softmax_shares(np.clip(a[1:], -LOGIT_BOX, LOGIT_BOX))
            ship = [float(q) * onhand for q in shares]
        elif self.action_mode in ("order_relu", "order_nested"):
            onhand = max(0.0, float(self._state.wh_stock))
            fn = relu_shares if self.action_mode == "order_relu" else nested_shares
            shares, _ = fn(np.clip(a[1:], -LOGIT_BOX, LOGIT_BOX))
            ship = [float(q) * onhand for q in shares]
        elif self.action_mode == "order_softmax_free":
            onhand = max(0.0, float(self._state.wh_stock))
            shares, _ = free_softmax_shares(a[1:])                 # no clip by design
            ship = [float(q) * onhand for q in shares]
        elif self.action_mode == "order_dirichlet":
            onhand = max(0.0, float(self._state.wh_stock))
            shares, _ = dirichlet_shares(a[1:])
            ship = [float(q) * onhand for q in shares]
        elif self.action_mode in ("order_catkeep", "order_catkeep_fine"):
            onhand = max(0.0, float(self._state.wh_stock))
            shares, _ = catkeep_shares(np.concatenate([[a[1]], np.clip(a[2:], -LOGIT_BOX, LOGIT_BOX)]), keep_menu(self.action_mode))
            ship = [float(q) * onhand for q in shares]
        else:
            onhand = max(0.0, float(self._state.wh_stock))   # float dust after a binding scale-down
            ship = [float(np.clip(f, 0.0, 1.0)) * onhand for f in a[1:]]
        return order, clamp_shipments(ship, cap)

    # -- gym API --------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        state = self._state
        if self.sequential and self._phase == 1:
            # the allocation step sees the post-O interstate: the order just
            # placed sits in the warehouse pipeline's last slot
            state = copy.deepcopy(state)
            mdp._apply_event_O(self._scenario_ep, state, self._pending_order)
        return observation(
            self._scenario_ep, state, self.observation_mode, list(self._info["demand"]),
            phase=self._phase,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            # per-env stream (spec 7): global np.random would replay identical
            # episode-seed sequences across SubprocVecEnv workers, and omitting
            # the branch would replay ONE episode under gymnasium's auto-reset
            self._episode_seed = int(self.np_random.integers(0, 2_147_483_647))
        self._scenario_ep = (
            self.scenario(self._episode_seed) if callable(self.scenario) else self.scenario
        )
        state, self._info = mdp.init_state(self._scenario_ep, self._episode_seed)
        self._state = mdp.advance1(self._scenario_ep, state)
        self._step = 0
        self.total_reward = 0.0
        self._phase = 0
        self._pending_order = 0.0
        return self._get_obs(), dict(self._info)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.sequential and self._phase == 0:
            # ORDER step: hold the order, show the interstate, no cost yet
            self._pending_order = self.decode_action(action)[0]
            self._phase = 1
            self._step += 1
            return self._get_obs(), 0.0, False, False, {}
        if self.sequential:
            a = np.asarray(action, dtype=np.float64).reshape(-1).copy()
            a[0] = self._pending_order          # the held order; component 0 is ignored here
            order, ship = self.decode_action(a)
            self._phase = 0
        else:
            order, ship = self.decode_action(action)
        new_state, self._info = mdp.advance2(self._scenario_ep, self._state, order, ship)
        # the IR's root-level eval_metrics, evaluated per period on the
        # END-OF-PERIOD state exactly as the interpreter does (wh_share and
        # system_stock are per-period values; the evaluator averages them over
        # the horizon, which is what the IR's `/ horizon_T` + `sum` compute)
        onhand_rt = sum(max(0.0, x) for x in new_state.rt_stock)
        denom = new_state.wh_stock + onhand_rt
        self._info["metrics"] = {
            "wh_share": (new_state.wh_stock / denom) if denom > 0 else 0.0,
            "stockout_periods": float(sum(1 for x in new_state.rt_stock if x < 0)),
            "system_stock": new_state.wh_stock + sum(new_state.wh_pipe) + onhand_rt
                            + sum(sum(row) for row in new_state.rt_pipe),
        }
        self._state = (
            mdp.advance1(self._scenario_ep, new_state) if not new_state.terminated else new_state
        )
        obs = self._get_obs()
        reward = -float(self._info["cost"]["total"])
        if self.reward_mode == "neg_cost_transit":      # + h0 on retailer pipelines at period end
            reward -= float(self._scenario_ep.h0) * sum(sum(row) for row in new_state.rt_pipe)
        terminated = bool(new_state.terminated)
        truncated = False
        self.total_reward += reward
        self._step += 1

        self.logger_s.info(
            f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
            f"{order:.4f}\t{[round(q, 4) for q in self._info['shipped']]}\t"
            f"{[round(d, 4) for d in self._info['demand']]}\t{new_state.wh_stock:.4f}\t"
            f"{[round(x, 4) for x in new_state.rt_stock]}\t{reward:.4f}\t{self.total_reward:.4f}"
        )
        if terminated:
            self.logger_e.info(
                f"{self._episode_seed}\t{self._step}\t{self._info['action_period']}\t"
                f"{self.total_reward:.4f}"
            )
        return obs, reward, terminated, truncated, dict(self._info)

    @property
    def last_info(self) -> dict:
        return self._info

    # -- logging --------------------------------------------------------------

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


def _smoke(scenario: OwmrScenario, episodes: int = 2) -> None:
    for obs_mode in OwmrEnv.OBS_MODES:
        for act_mode in OwmrEnv.ACTION_MODES:
            if (act_mode == "seq_order_softmax") != (obs_mode == "raw_seq"):
                continue   # the two-phase pair only comes together
            env = OwmrEnv(scenario, observation_mode=obs_mode, action_mode=act_mode)
            env.action_space.seed(0)   # a seeded smoke: the same random actions every run
            for ep in range(episodes):
                obs, _ = env.reset(seed=100 + ep)
                assert env.observation_space.contains(obs), (
                    f"{scenario.scenario_name}/{obs_mode}/{act_mode}: reset obs outside space: {obs}"
                )
                steps = 0
                while True:
                    a = env.action_space.sample()
                    obs, r, term, trunc, _ = env.step(a)
                    steps += 1
                    assert env.observation_space.contains(obs), (
                        f"{scenario.scenario_name}/{obs_mode}/{act_mode}: step {steps} "
                        f"obs outside space: {obs}"
                    )
                    assert np.isfinite(r), f"non-finite reward at step {steps}"
                    if term or trunc:
                        break
                want = scenario.horizon * (2 if env.sequential else 1)
                assert steps == want, f"expected {want} steps, got {steps}"
            print(
                f"  {scenario.scenario_name:11s} obs={obs_mode:9s} act={act_mode:11s} "
                f"obs_dim={env.observation_space.shape[0]:2d} "
                f"act_dim={env.action_space.shape[0]}  OK"
            )


def _check_unseeded_resets_draw_fresh_episodes(scenario: OwmrScenario) -> None:
    env = OwmrEnv(scenario)
    env.reset(seed=3)
    seeds = set()
    for _ in range(3):
        env.reset()
        seeds.add(env._episode_seed)
    assert len(seeds) == 3, f"unseeded resets replayed an episode: {seeds}"
    print(f"  {scenario.scenario_name:11s} unseeded resets draw fresh episodes  OK")


if __name__ == "__main__":
    from owmr_scenarios import SCENARIOS

    print("random-action episodes, observation_space.contains asserted each step:")
    for nm in SCENARIOS:
        _smoke(SCENARIOS[nm])
    print("\nepisode-seed provenance:")
    _check_unseeded_resets_draw_fresh_episodes(SCENARIOS["base"])

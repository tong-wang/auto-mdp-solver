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
    echelon_position,
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
    n = scenario.n_echelons
    n_transit = scenario.leadtime - 1
    stock = list(state.stock[:n])
    transits = [[state.pipeline[k][s] for k in range(n)] for s in range(n_transit)]
    blocks = [stock] + transits

    if observation_mode == "echelon":
        blocks = to_echelon(stock, transits)

    period_frac = state.period / scenario.horizon
    return np.asarray([period_frac] + [x for b in blocks for x in b], dtype=np.float32)


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

    Both modes are prefixed by the normalized period, because the horizon is
    finite and the optimal critical numbers are time-varying (the paper's
    ``x_bar_n`` subscript). It is mode-independent, so it does not disturb the
    comparison.

    Action — ``"ship"``: one shipment quantity per link, an N-vector.
    Feasibility is a per-component box: each link is clipped to what its source
    installation holds, and the top link (drawing from the outside supplier) to
    the action-scale cap. Raw action 0 means "ship nothing" — a live, low-value
    action, which is where SB3's Gaussian initializes.

    Reward: ``-total_cost`` per period.
    """

    metadata = {"render_modes": []}

    OBS_MODES = ("raw", "echelon")
    ACTION_MODES = ("ship_discrete", "target_discrete", "ship_rel", "ship")

    # `ship_discrete` grid: quantities 0 .. DISCRETE_SPAN * mean demand, in
    # whole units — demand is Poisson and every stock is integral, so a whole
    # unit is the natural resolution and nothing finer is meaningful
    DISCRETE_SPAN = 4
    # `ship_rel` spread: a = +1 maps to RELATIVE_SPREAD+1 times mean demand
    RELATIVE_SPREAD = 3.0

    def __init__(
        self,
        scenario: ClarkScarfScenario,
        observation_mode: str = "raw",
        action_mode: str = "ship_discrete",
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

    def _target_bins(self) -> list[int]:
        """Per-link target grids for `target_discrete`.

        Echelon k's position covers k+1 levels, so its order-up-to level scales
        with k — hence a per-link range rather than one shared grid.
        MultiDiscrete allows different bin counts per dimension, so nothing is
        wasted on the lower links.
        """
        mean = float(self.scenario.demand.mean())
        return [int((k + 1) * self.DISCRETE_SPAN * mean) + 1
                for k in range(self.n_live)]

    @property
    def n_bins(self) -> int:
        """Number of whole-unit shipment choices per link, for `ship_discrete`."""
        return int(self.DISCRETE_SPAN * float(self.scenario.demand.mean())) + 1

    def _build_action_space(self) -> gym.spaces.Space:
        if self.action_mode == "target_discrete":
            return gym.spaces.MultiDiscrete(self._target_bins())
        if self.action_mode == "ship_discrete":
            return gym.spaces.MultiDiscrete([self.n_bins] * self.n_live)
        if self.action_mode == "ship_rel":
            return gym.spaces.Box(-1.0, 1.0, shape=(self.n_live,), dtype=np.float32)
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
        ``target_discrete`` — the bin index is the echelon **order-up-to level**
            ``y_k``; the shipment is ``clip(y_k - u_k, 0, capacity)``. An
            available alternative encoding; untested on this campaign.
        ``ship_rel`` — q = mean * (1 + SPREAD * a) on a symmetric [-1,1] box.
        ``ship`` — identity on [0, ship_max]; the paper-native encoding.

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
        if self.action_mode == "target_discrete":
            # the bin index IS the echelon order-up-to level y_k; the shipment
            # is whatever it takes to get there
            u = echelon_position(self.scenario, self._state)
            return np.array([max(0.0, float(act[k]) - u[k])
                             for k in range(self.n_live)])
        if self.action_mode == "ship_discrete":
            return np.asarray(act, dtype=np.float64)     # bin index == quantity
        if self.action_mode == "ship":
            return np.clip(act, 0.0, float(self.scenario.ship_max))
        mean = float(self.scenario.demand.mean())
        return np.maximum(0.0, mean * (1.0 + self.RELATIVE_SPREAD * act))

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
        scale = self.n_live if self.observation_mode == "echelon" else 1
        n_blocks = 1 + self.n_transit
        width = self.n_live * n_blocks
        low = np.concatenate([[0.0], np.full(width, lo_lvl * scale)]).astype(np.float32)
        high = np.concatenate([[1.0], np.full(width, hi_lvl * scale)]).astype(np.float32)
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
        if self.action_mode == "target_discrete":
            # feasible targets are u_k <= y_k <= x_{k+1}: never below the
            # current position (that is just "ship nothing", already reachable
            # at y_k = u_k) and never above what the level up holds
            u = echelon_position(self.scenario, self._state)
            x = echelon_stock(self.scenario, self._state)
            out = []
            for k, nb in enumerate(self._target_bins()):
                lo = int(min(max(0, np.ceil(u[k])), nb - 1))
                hi = x[k + 1] if k + 1 < self.n_live else float(nb - 1)
                hi = int(min(max(lo, np.floor(hi)), nb - 1))
                m = np.zeros(nb, dtype=bool)
                m[lo : hi + 1] = True
                out.append(m)
            return np.concatenate(out)
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

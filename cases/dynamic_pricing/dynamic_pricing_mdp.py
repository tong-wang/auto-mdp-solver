"""Dynamic pricing core MDP.

This module contains the domain-level simulator only.
It does not depend on Gym/Gymnasium.

State/info is represented from the *simulator's* full perspective.  What is
observable to the RL agent is NOT defined here — observations are constructed
in the gym wrapper (dynamic_pricing_gym.py) based on the ``observation_mode``
it is configured with.

Key components:
1. DynamicPricingState: keep track of the running state, including:
    a. running status: period, terminated
    b. state variables: inventory (remaining stock, non-increasing)
    c. episode specific seed
2. state transition functions:
    a. init_state(scenario, episode_seed)
    b. advance(scenario, state, decision)

The scenario config (DynamicPricingScenario) and the stochastic primitives it
is built from (PoissonArrivals) live in dynamic_pricing_scenarios.py and
dynamic_pricing_uncertainty.py respectively; this module imports
DynamicPricingScenario for type annotations only.

Notations:
    - T = scenario.horizon periods, 0-indexed; period length dt
    - lambda(p) = a * exp(-alpha * p): demand intensity at price p
    - arrivals ~ Poisson(lambda(price) * dt): raw demand this period
    - units_sold = min(arrivals, inventory); the rest is lost (no backlog)
    - salvage q * inventory is paid only when the horizon completes

NOTE: the episode terminates early when inventory hits 0 (absorbing: no
reorder). Salvage at that point is q * 0 = 0 under any q, so early
termination never drops salvage value. Revenue quantities travel in the
``info`` dict (info["revenue"]) — reward is derived in the gym wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dynamic_pricing_scenarios import DynamicPricingScenario


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DynamicPricingState:
    """Full internal state of the simulator.

    This captures everything the simulator needs to generate transitions — it is
    the MDP state from the *simulator's* perspective, not the agent's.  Whether
    a field is observable to the RL agent is NOT determined here; observation
    construction is delegated to the gym wrapper (dynamic_pricing_gym.py)
    according to the ``observation_mode`` it is configured with.
    """

    period:     int   # current period t (0-indexed; horizon ends at scenario.horizon)
    terminated: bool
    inventory:  int   # remaining stock n; monotonically non-increasing

    ## internal RNG state
    episode_seed: int
    seed_salt:    int = field(repr=False)


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------

def init_state(
    scenario: DynamicPricingScenario, episode_seed: int
) -> tuple[DynamicPricingState, dict]:
    """Initialize state at the start of an episode: full stock, period 0."""
    state = DynamicPricingState(
        period=0,
        terminated=False,
        inventory=scenario.n0,
        episode_seed=episode_seed,
        seed_salt=scenario.seed_salt,
    )
    info: dict = {
        "action_period": state.period - 1,  # no action taken yet
        "price":         0.0,
        "arrivals":      0,
        "units_sold":    0,
        "lost_demand":   0,
        "intensity":     0.0,
        "revenue":       {"sales": 0.0, "salvage": 0.0, "total": 0.0},
    }
    return state, info


def advance(
    scenario: DynamicPricingScenario,
    state:    DynamicPricingState,
    decision: float,
) -> tuple[DynamicPricingState, dict]:
    """Set the price, realize Poisson arrivals, sell up to available stock.

    Returns (next_state, info).
    """
    assert not state.terminated, "Episode already finished."
    price = float(decision)
    assert price >= 0, f"price must be non-negative, got {price}."

    intensity  = scenario.demand.intensity(price)
    arrivals   = scenario.demand.sample(state, price)
    units_sold = min(arrivals, state.inventory)

    new_inventory = state.inventory - units_sold
    lost_demand   = arrivals - units_sold
    new_period    = state.period + 1
    terminated    = new_period >= scenario.horizon or new_inventory == 0

    sales   = price * units_sold
    salvage = scenario.q * new_inventory if new_period == scenario.horizon else 0.0

    next_state = DynamicPricingState(
        period=new_period,
        terminated=terminated,
        inventory=new_inventory,
        episode_seed=state.episode_seed,
        seed_salt=state.seed_salt,
    )

    info: dict = {
        "action_period": state.period,  # = the input period
        "price":         price,
        "arrivals":      arrivals,
        "units_sold":    units_sold,
        "lost_demand":   lost_demand,
        "intensity":     intensity,
        "revenue":       {"sales": sales, "salvage": salvage, "total": sales + salvage},
    }
    return next_state, info


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

def _run_episode(scenario: DynamicPricingScenario, price: float = 1.0, episode_seed: int = 3) -> None:
    print(f"\n=== {scenario.scenario_name}: {scenario.desc} ===")
    state, _ = init_state(scenario, episode_seed)
    total = 0.0
    while not state.terminated:
        state, info = advance(scenario, state, price)
        total += info["revenue"]["total"]
        if info["units_sold"] or state.terminated:
            print(
                f"  t={info['action_period']:2d}  price={info['price']:.2f}  "
                f"arrivals={info['arrivals']}  sold={info['units_sold']}  "
                f"inv={state.inventory:2d}  revenue={info['revenue']['total']:.2f}"
            )
    print(f"  periods={state.period}  total revenue={total:.2f}")


if __name__ == "__main__":
    from dynamic_pricing_scenarios import scenario_ample_stock, scenario_simple

    _run_episode(scenario_simple)
    _run_episode(scenario_ample_stock)

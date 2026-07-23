"""Named scenario registry for the ADI + flexible-delivery (AdiFlex) domain.

Defines AdiFlexScenario and all named scenario instances. Import SCENARIOS in
training and evaluation scripts to look up a scenario by name:

    from adi_flex_scenarios import SCENARIOS
    scenario = SCENARIOS["exp4"]

The eight instances are the demand-split ladder of Wang & Toktay (2008) section
4.3, holding the total arrival rate at 6 and sliding mass between due dates.
Its two ends are not merely extreme cases but *different models*: exp0 is the
traditional no-ADI problem, and exp7 is exactly the homogeneous model of the
paper's section 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adi_flex_uncertainty import DemandGenerator, PoissonDemand


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AdiFlexScenario:
    """AdiFlex scenario configuration.

    Structural attributes (supply_leadtime, demand_window) are fixed by the
    frozen IR rather than free to vary: the state carries no supply pipeline,
    and the advance-demand profile is exactly (due_now, due_next). They are
    stored and asserted so that a mis-specified instance fails loudly instead
    of silently simulating a different problem.
    """

    # planning horizon N
    horizon: int

    # demand model, one generator per due date (branch-1 source ids 0, 1, 2)
    demand_now:   DemandGenerator   # d^i_i   — due on arrival
    demand_next:  DemandGenerator   # d^{i+1}_i — due next period
    demand_later: DemandGenerator   # d^{i+2}_i — due two periods out

    # cost parameters (paper base setting: K=100, h=1, p=9)
    holding_cost:     float   # h: per unit of positive end-of-period stock
    backorder_cost:   float   # p: per unit of overdue unmet demand, per period
    order_cost_fixed: float   # K: charged once per period in which anything is ordered

    # structural — see the class docstring
    supply_leadtime: int = 0   # L: orders arrive immediately
    demand_window:   int = 2   # T: demand is due within 2 periods

    # scenario-level seed salt (mixed with episode seed in all RNG calls)
    seed_salt: int = field(default=4729, repr=False)

    scenario_name: str | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        assert self.horizon >= 1,          "horizon must be >= 1."
        assert self.holding_cost >= 0,     "holding_cost must be >= 0."
        assert self.backorder_cost >= 0,   "backorder_cost must be >= 0."
        assert self.order_cost_fixed >= 0, "order_cost_fixed must be >= 0."
        assert self.supply_leadtime == 0, (
            "supply_leadtime is structural in this domain: the state carries no "
            "pipeline, so L>0 needs a re-derived model, not a new instance."
        )
        assert self.demand_window == 2, (
            "demand_window is structural in this domain: the advance-demand "
            "profile is exactly (due_now, due_next). T>2 additionally changes "
            "the decision dimension to 1 + (T-1) protection levels."
        )
        for label in ("demand_now", "demand_next", "demand_later"):
            assert isinstance(getattr(self, label), DemandGenerator), \
                f"{label} must be a DemandGenerator."

    @property
    def total_rate(self) -> float:
        """Mean total arrivals per period, across all three due dates."""
        return (self.demand_now.mean() + self.demand_next.mean()
                + self.demand_later.mean())

    @property
    def has_crossover(self) -> bool:
        """Whether a later-arriving order can be due before one already held.

        Requires both a stream due two periods out (something to ship early
        against) and urgent arrivals to be preempted by. Where this is False
        the paper proves every protection-level heuristic is already optimal,
        which makes those instances correctness checks rather than contests.
        """
        return self.demand_later.mean() > 0 and self.demand_now.mean() > 0


# ---------------------------------------------------------------------------
# Scenarios — the demand-split ladder at the paper's base setting
# ---------------------------------------------------------------------------
# Wang & Toktay (2008) section 4.3: K=100, h=1, p=9, N=12, L=0, T=2, alpha=1,
# with (lambda_0, lambda_1, lambda_2) summing to 6 throughout.

def _ladder(name: str, rates: tuple[float, float, float], desc: str) -> AdiFlexScenario:
    """Build one rung of the ladder; only the demand split varies."""
    lam_now, lam_next, lam_later = rates
    return AdiFlexScenario(
        scenario_name=name,
        desc=desc,
        horizon=12,
        demand_now=PoissonDemand(rate=lam_now, source_id=0),
        demand_next=PoissonDemand(rate=lam_next, source_id=1),
        demand_later=PoissonDemand(rate=lam_later, source_id=2),
        holding_cost=1.0,
        backorder_cost=9.0,
        order_cost_fixed=100.0,
        supply_leadtime=0,
        demand_window=2,
    )


scenario_exp0 = _ladder("exp0", (6, 0, 0), "no ADI at all; the traditional model (homogeneous, T=0)")
scenario_exp1 = _ladder("exp1", (5, 1, 0), "nothing ever due two periods out; no crossover")
scenario_exp2 = _ladder("exp2", (4, 1, 1), "crossover live; mild advance information")
scenario_exp3 = _ladder("exp3", (3, 1, 2), "crossover live; balanced split")
scenario_exp4 = _ladder("exp4", (2, 1, 3), "crossover live; where PL(sigma)'s optimality gap peaks")
scenario_exp5 = _ladder("exp5", (1, 1, 4), "crossover live; mostly advance orders")
scenario_exp6 = _ladder("exp6", (0, 1, 5), "no urgent arrivals to protect against; no crossover")
scenario_exp7 = _ladder("exp7", (0, 0, 6), "full ADI; exactly the homogeneous model of section 3")


# Registry mapping name -> scenario for convenient lookup in training scripts.
SCENARIOS: dict[str, AdiFlexScenario] = {
    s.scenario_name: s
    for s in [
        scenario_exp0,
        scenario_exp1,
        scenario_exp2,
        scenario_exp3,
        scenario_exp4,
        scenario_exp5,
        scenario_exp6,
        scenario_exp7,
    ]
}


if __name__ == "__main__":
    print(f"{'name':6s}  {'(lam0,lam1,lam2)':18s}  {'total':>5s}  {'crossover':>9s}  desc")
    for name, s in SCENARIOS.items():
        rates = (s.demand_now.mean(), s.demand_next.mean(), s.demand_later.mean())
        print(
            f"{name:6s}  {str(rates):18s}  {s.total_rate:5.1f}  "
            f"{str(s.has_crossover):>9s}  {s.desc}"
        )

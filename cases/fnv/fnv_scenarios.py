"""Scenario configuration and named registry for the FNV domain.

Defines FnvScenario and the named scenario instances. Import SCENARIOS in
training and evaluation scripts to look up a scenario by name:

    from fnv_scenarios import SCENARIOS
    scenario = SCENARIOS["simple"]

FNV's experiment-design families (the aMMFE / mMMFE parameter sweeps a
generalist policy is trained to cover) are NOT scenarios — they are grids,
and live in fnv_grids.py (spec §5.5–§5.6). A concrete FnvScenario together with
an episode_seed pins one episode completely.

The problem (Heath & Jackson 1994 MMFE; the "fresh-newsvendor" ordering
variant): place N sequential orders while a martingale demand signal is
progressively revealed; realize demand at the horizon and sell against
cumulative inventory. Additive MMFE: D = mu + I. Multiplicative MMFE:
D = exp(mu + I). I is the cumulative demand-signal information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fnv_uncertainty import NormalDemandSignal

SEED_SCHEME = "v2"


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FnvScenario:
    """FNV scenario parameters.

    stdev / T / N are hyper-parameters from which the per-period demand-signal
    stdev schedule is derived into the ``signal`` generator (in __post_init__).
    """

    # stochastic hyper-parameters (scenario axes; swept by the design grids)
    stdev: float  # total demand std dev; per-period signal stdevs are derived from this
    T:     float  # time location of the last ordering epoch (0 < T < 1)
    lamb:  float  # ordering cost increment per period

    # structural parameters
    N:  int    # number of ordering periods (paper default 3)
    r:  float  # sales revenue per unit (paper default 2.0)
    c1: float  # ordering cost in period 1 (paper default 1.0)
    mu: float  # a-MMFE: mean demand; m-MMFE: log-scale parameter (E[D] = exp(mu + sigma²/2))

    # MMFE mode
    mmfe_mode: str  # "additive" or "multiplicative"

    # demand-signal generator (derived from stdev / T / N in __post_init__)
    signal: NormalDemandSignal = field(init=False, repr=False)

    # reproducibility (universal knob, >= 1 — v2 rule, spec §6.3)
    seed_salt:     int         = field(default=6409, repr=False)
    scenario_name: str | None  = None

    # human-readable description of the scenario
    desc: str = ""

    def __post_init__(self) -> None:
        assert self.mmfe_mode in {"additive", "multiplicative"}, \
            f"mmfe_mode must be 'additive' or 'multiplicative', got '{self.mmfe_mode}'."
        assert self.stdev >= 0,  "stdev must be non-negative."
        assert 0 < self.T < 1,   "T must be in (0, 1)."
        assert self.lamb >= 0,   "lamb must be non-negative."
        assert self.N >= 1,      "N must be >= 1."
        assert self.r > 0,       "r must be positive."
        assert self.c1 > 0,      "c1 must be positive."
        assert self.seed_salt >= 1, "seed_salt must be >= 1 (v2 rule)."
        if self.mmfe_mode == "additive":
            assert self.mu > 0,  "mu must be positive for additive MMFE."
        # derive the per-period demand-signal stdev schedule from stdev / T / N.
        # signal[n] covers the increment tau[n] -> tau[n+1]; signal[0].stdev == 0
        # because tau[0] == tau[1] == 0.
        self.signal = NormalDemandSignal(stdevs=self.signal_stdevs())

    def ordering_cost_rate(self, period: int) -> float:
        """Per-unit ordering cost at period (1-indexed): c1 + (period-1)*lamb."""
        return self.c1 + (period - 1) * self.lamb

    def tau(self) -> np.ndarray:
        """Time epochs [tau_0, ..., tau_{N+1}] of length N+2."""
        return np.concatenate(([0.0], np.linspace(0.0, self.T, self.N), [1.0]))

    def signal_stdevs(self) -> tuple[float, ...]:
        """Per-period signal stdev schedule (length N+1): sqrt(tau[n+1]-tau[n])*stdev."""
        tau = self.tau()
        return tuple(
            float(np.sqrt(tau[n + 1] - tau[n]) * self.stdev)
            for n in range(self.N + 1)
        )


# ---------------------------------------------------------------------------
# Fixed scenarios
# ---------------------------------------------------------------------------

scenario_simple = FnvScenario(
    stdev=0.1, T=0.9, lamb=0.1,
    N=3, r=2.0, c1=1.0, mu=1.0,
    mmfe_mode="additive",
    scenario_name="simple",
    desc="fixed additive-MMFE params (stdev=0.1, T=0.9, lamb=0.1); sanity-check / fast-training",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, FnvScenario] = {
    "simple": scenario_simple,
}


if __name__ == "__main__":
    for name, s in SCENARIOS.items():
        print(f"{name:10s}  {s}")
        print(f"{'':10s}  signal={s.signal}")

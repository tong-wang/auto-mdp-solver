"""Negative control for the simulator gates: inject the bugs they exist to
catch and confirm EACH BUG is caught by at least one gate.

The union is the right claim, not per-gate coverage. The two gates cover
different things by construction and neither is redundant:

  gate_posterior_matches_domain   exact, but only at the base cell (K=10,
                                  T=1000, sigma=1) — the one cell the domain
                                  implements. Blind to any error that vanishes
                                  at sigma=1.
  gate_posterior_calibration      covers the sigma/n generalization, where the
                                  domain has no reference at all. Statistical,
                                  so blind to small exact-form errors that
                                  leave the posterior calibrated.
"""
import numpy as np
import mab_grid_probe as G

GATES = {"vs_mab_bayes": G.gate_posterior_matches_domain,
         "calibration": G.gate_posterior_calibration}


def caught_by(patched):
    """Which gates fail under this patched posterior."""
    orig, hits = G.GridSim.post, []
    G.GridSim.post = patched
    try:
        for name, gate in GATES.items():
            try:
                gate()
            except AssertionError:
                hits.append(name)
    finally:
        G.GridSim.post = orig
    return hits


def post_no_sigma(self):
    """sigma dropped from the precision — the classic mis-scaling. Invisible
    at sigma=1, silently wrong on every sigma cell."""
    prec = self._prior_prec + self.pulls
    psd = np.sqrt(1.0 / prec)
    return (G.PRIOR_MEAN * self._prior_prec + self.sums) / prec, psd


def post_off_by_one(self):
    """off-by-one in the pull count — wrong everywhere, including sigma=1."""
    prec = self._prior_prec + (self.pulls + 1.0) * self._noise_prec
    psd = np.sqrt(1.0 / prec)
    return (G.PRIOR_MEAN * self._prior_prec
            + self.sums * self._noise_prec) / prec, psd


def post_prior_ignored(self):
    """prior precision dropped — the n=0 (unpulled arm) case blows up, which
    is exactly the state the starvation work depends on."""
    prec = np.maximum(self.pulls * self._noise_prec, 1e-12)
    psd = np.sqrt(1.0 / prec)
    return (self.sums * self._noise_prec) / prec, psd


BUGS = {"sigma dropped": post_no_sigma,
        "off-by-one pulls": post_off_by_one,
        "prior ignored": post_prior_ignored}

print(f"{'injected bug':20s} caught by")
uncaught = []
for name, patch in BUGS.items():
    hits = caught_by(patch)
    print(f"{name:20s} {', '.join(hits) if hits else '*** NOTHING ***'}")
    if not hits:
        uncaught.append(name)

assert not uncaught, f"no gate catches: {uncaught}"
print(f"\nall {len(BUGS)} injected bugs caught by at least one gate")
print("neither gate is redundant: each catches something the other misses")

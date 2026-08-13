"""Conjugate-posterior belief features for the mab domain.

Declared in ``mab_schema.json`` under ``mdp.expr_builtins`` and resolved
lazily by the IR interpreter / gym wrapper next to the IR file
(portable-domain contract).

Priors are the frozen Phase-A world latents:
- Bernoulli branch: p_i ~ Uniform(0,1) = Beta(1,1); posterior Beta(1+s, 1+n-s).
- Gaussian branch: mu_i ~ N(0,1), known payout sd sigma=1; posterior
  N(payouts/(1+n), 1/(1+n)) by precision addition.
"""

import math


def bayes_post_mean(pulls, payouts, is_gauss):
    """Per-arm posterior mean of the arm's true mean payout."""
    if is_gauss:
        return [payouts[i] / (1.0 + pulls[i]) for i in range(len(pulls))]
    return [(1.0 + payouts[i]) / (2.0 + pulls[i]) for i in range(len(pulls))]


def bayes_post_sd(pulls, payouts, is_gauss):
    """Per-arm posterior standard deviation of the arm's true mean payout."""
    if is_gauss:
        return [math.sqrt(1.0 / (1.0 + pulls[i])) for i in range(len(pulls))]
    out = []
    for i in range(len(pulls)):
        a = 1.0 + payouts[i]
        b = 1.0 + pulls[i] - payouts[i]
        n = a + b
        out.append(math.sqrt(a * b / (n * n * (n + 1.0))))
    return out

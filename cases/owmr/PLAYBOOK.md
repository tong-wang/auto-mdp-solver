# PLAYBOOK — `owmr` (one-warehouse, N-retailer periodic-review distribution)

Digest entries per `ESCALATION_LOG_GUIDE.md` §10 — experiences, not rules;
real names and numbers, the ledger cited for everything else. **Only
transferable tricks of the digital-twin + RL framework belong here** (the
human, 2026-10-09): mechanisms that will matter on another problem, not the
campaign's own story — that lives in `ESCALATION.md` and `README.md`. LV1 was
started early (2026-10-08, before the case closed); LV2 was added at the
close pass (2026-10-09). The frame moves, modeling rules and negative space
that §10 also asks for were written and then cut as project-specific; the
ledger's FRAME-CHANGELOG, IR-CHANGELOG and MAP hold them.

**Structure class:** multi-echelon inventory with a central allocation
decision — a continuous order, a continuous split across N retailers and a
retention choice at the warehouse, every period, under i.i.d. gamma demand;
fully observed state; the optimal policy unknown, bracketed by a relaxed
bound and a feasible heuristic (Doğru, de Kok & van Houtum 2009).

**Protocol every number below is quoted at:** `base` cell (p = 4, cv = 0.5,
N = 5, lead times (3, 1), T = 100); 8192 seeds from 0, common random numbers,
paired per seed against the LB heuristic (603.81 ± 0.44; the bound 603.18);
deterministic argmax play; checkpoints screened on 2048 disjoint seeds from
1,000,000, top-3 confirmed on the protocol block; the 5M / 10M budgets as
stated per arm. Entries that quote the `hicv` cell (p = 4, cv = 2) say so;
there the bar is 2,769.76 ± 4.01, the bound 2,499.99, and the screen runs at
8192 seeds (the heuristic's SE is ten times `base`'s).

---

## LV1 — the coordinate vine: paired common-world rollouts as PPO's gradient for a noisy, short-consequence decision

```
context:      the campaign's best explicit-allocation arm after a tuning study
              on the categorical-retention head (closed unconverged at 56
              sampled trials) sat at 613.90,
              +10.09 above the heuristic (#E24); after the interface itself
              (−35 from L1 to the relu split, #E4/#E14) every lever — tuning,
              architecture, entropy, accounting, budget, the heads — had moved
              the residual by at most 4 in the helpful direction (#E16–#E23).
              #E11 had shown a 64x64 net REPRESENTS the heuristic's allocation
              (+0.02 in quantities); #E12 that PPO's per-sample gradient
              signal at the allocation was ~1% of its noise (one changed
              decision worth ~0.5 over 20 periods against a return sd ~20).
symptom:      a small, SYSTEMATIC gap at every state — the full-swap
              decomposition on the #E24 winner: split 3.4, retention ~4.5,
              order 3.1 of the +10.1 (#E30 step 0); no tail; the categorical over the
              retention menu spread .32/.28/.23 over keep 0/.1/.2 in surplus
              states (honest uncertainty: those options differ by ~0.1 per
              decision, #E29).
diagnosis:    search under noise, not representation. The same-world probe
              (#E29, `owmr_branch_probe.py`): pairing K options on ONE demand
              world cuts the advantage sd 7–12x (60–150x in samples) at all
              three decisions; and the option RANKING is identical at a
              10- and a 20-period window — one decision's consequence is
              realised within the window, so nothing past it needs a critic.
              Per-state stakes (0.01–0.1) stay below one branch's resolution
              (paired SE 0.24–0.30 at K = 64), so the comparisons must be
              aggregated across states by the policy network, not acted on
              per state.
prescription: algo layer — `owmr_vine_ppo.VinePPO` (#E30): at a parent step,
              with probability p = 0.2, fork the env on its own world into a
              parent copy and ONE SIBLING PER ACTION COORDINATE, each sibling
              an exact policy sample differing from the parent in that
              coordinate alone (Gaussian coordinates redrawn from their
              marginal; the categorical redrawn; for a Dirichlet head, one
              Gamma variate of the G_i / ΣG_j construction redrawn, #E32);
              H = 10 periods under the stochastic policy with a common noise
              stream, NO bootstrap; sibling advantage = -(C_sibling - C_parent),
              scored on the log-prob of ITS coordinate only (the parent's cost
              is a valid baseline for the redrawn coordinate and for no other);
              a SIDE PPO pass (clipped ratio on the masked log-prob, same clip
              and KL valve, no value/entropy term) after the unchanged main
              pass; side advantages rescaled to the main buffer's advantage sd.
              The main rollout, GAE, critic, VecNormalize and the number of
              distinct worlds per update are untouched (a sibling campaign on a
              partially observed problem found that cutting the worlds per
              update is what broke its group-rollout lever: the policy had a
              hidden parameter to learn from world diversity).
              Effect @ protocol: 613.90 -> 607.90 / 608.02 at 5M, -6.38 /
              -6.78 paired vs matched controls at the same recipe and seeds
              (#E30); order and split SOLVED (full-swap parts 3.1 -> 0.4 and
              3.4 -> 0.1); the worst episode halves (+24 vs +53 over the
              heuristic). Dose: the gain is monotone in the side pass's
              learning rate — 0.2x 1.1–2.4, 0.5x 1.5–5.7, 1x 6.4–6.8 (#E31);
              the side pass needs its OWN optimizer and lr (`vine_lr_scale`),
              a loss weight is inert under Adam. Scope conditions: a fully
              observed state (no hidden parameter to learn — the sibling
              campaign's partially observed cell broke under shared worlds); a
              decision whose consequence is realised
              within the window; coordinates with a smooth convex cost around
              the mean (the Gaussian ones) — a one-step deviation is a noisy
              derivative there.
failed:       (i) dosing the side pass DOWN (#E31) — loses monotonically; the
              std floor rises (0.65 / 0.44 / 0.37) and buys nothing.
              (ii) budget at the terminal lr (+5M, #E31 add. 2) and a straight
              10M schedule (#E32): the full-dose vine lands on the same floor
              ≈ 607.9 (+4.1) from six independent routes; lower doses and
              controls converge toward it later — the dose is a SPEED knob.
              (iii) the FULL MENU as the retention grid (#E34; every option a
              sibling, weighted pi_old(k) — the exact all-action gradient):
              drives the categorical to keep-nothing with P = 1.00 and loses
              5.4 (613.3), although the policy's OWN continuation scores keep
              .1/.2 better than keep 0 by 0.05 per surplus decision (7 SE).
              Mechanism: the one-step signal is +0.05 in 30% of states and
              -0.3 to -2 in the other 70%; wherever the net lumps them the
              average favours keep 0, the sharper estimator gets there faster,
              and at P(keep 0) ≈ 1 the softmax gradient toward the other
              options vanishes (PPO's clip slows any return). A sharper
              estimator is the wrong fix for an ABSORBING coordinate.
              (iv) the vine's own side effect: the Gaussian std collapses
              1.0 -> 0.37 (the paired signal makes the log-std gradient clean
              for a convex cost); the main pass then trips the KL valve in its
              first epoch on ~40% of updates and the side pass carries
              training. Harmless to the score; it is what the one-redraw form
              needed to keep kept-stock states in play (retention recovered to
              0.14 under it, never under the grid).
              (v) the Dirichlet head: the per-share vine takes it 619.6 ->
              615.5 but it keeps NOTHING (the kept share's corner at alpha ≤ 1,
              #E32); the head's geometry, not the lever.
limits:       what a vine cannot see. A one-step deviation under the CURRENT
              continuation values keeping stock at +0.05 per surplus decision
              (~1.5 per episode); retention's full-swap value is 4.5 — the rest
              lives in the state distribution a persistently retaining policy
              creates. The floor at +4.1 (retention 3.6) is the part of the
              gap that is coordinated across periods, not local to one
              decision. The generalisation, from a 2048 campaign where the
              same lever was weighed and declined: the vine
              pays when a decision's consequence is NOISY BUT SHORT — pairing
              removes the noise before the window closes, and whatever lies
              past the window is diluted enough by later actions and draws that
              a bootstrap (or none) is harmless. Inventory problems have that
              shape. 2048 has the opposite shape: the near window is nearly
              deterministic given the board, and the consequence that matters
              is far, concentrated, and sits exactly where the critic is
              weakest. Within an inventory problem, the retention decision is
              the 2048-shaped coordinate — its value is in the far,
              coordinated effect — which is why the lever solved order and
              split and stalled on retention.
evidence:     #E29 (probe), #E30 (lever + verdict), #E31 (dose; +5M
              extensions), #E32 (straight 10M, both heads), #E34 (menu grid);
              the lever's origin and its failure under shared worlds in a
              sibling competitive-newsvendor campaign, and its declination in
              a 2048 campaign, both in their own ledgers; `owmr_vine_ppo.py`
              docstring; gate `owmr_vine_gate.py`.
```

## LV2 — the action interface for a split-and-keep allocation: make "ship exactly nothing" and "keep" reachable options, not corners

```
context:      Stage 3 on `base`: L1 (`order_frac`, one fraction of on-hand per
              retailer) at 659.78, +9.3% over the heuristic; its probe read
              NO structure — retention 0, proportional allocation, a trickle
              order (#E2). Under `order_frac` the fractions summed above 1 in
              100% of decisions (median 3.49), so the MDP's proportional
              scale-down carried every allocation and "keep" was unreachable
              (#E4). The human's ruling (2026-09-25): the scale-down is a
              FEASIBILITY FALLBACK, never a rationing rule; an arm whose
              allocation it carries is not a result; every arm reports its
              firing rate (#E13 ruled out on this ground).
symptom:      a plateau at ~640 with the whole residual in retailer holding
              (+39 of +39, #E4, decomposed in #E6): stock spread across retailers instead of held
              back; every encoding that could not emit an exact zero shipment
              or an exact kept amount sat on it.
diagnosis:    representation is not the limit — a 64x64 net fits the
              heuristic's allocation to +0.02 in shipped quantities (#E11) —
              the REACHABILITY of the corners under PPO's Gaussian search is.
              An anchored softmax reaches a zero share only along a vanishing-
              gradient path (#E12; the logit box itself was never the
              constraint — no logit reached it, #E18); a relu split
              (`order_relu`) reaches every corner exactly but drifts to
              ship-everything (its "keep" coordinate sits in a flat region
              and is diluted, #E15/#E17); a nested keep coordinate with a
              relu/(1+relu) asymptote drifts to -3 (#E17–#E20: not the
              entropy bonus, not the transit accounting).
prescription: gym-action + arch — `order_catkeep` + `CatKeepPolicy` (#E21):
              retention as a CATEGORICAL over a short menu of kept fractions
              (0, .1, .2, .3, .5, .8) — "keep nothing" an option with
              state-dependent probability, no asymptote, no flat region — and
              the shipped remainder split by relu weights (exact zeros, the
              fallback cannot fire by construction; `owmr_test.py` pins it).
              Effect @ protocol: 659.78 (L1) -> 643.20 (softmax, #E4) ->
              624.80 (relu, #E14) -> 617.21 (catkeep, #E21) at 5M, 614.21 at
              10M (#E22); retention state-dependent for the first time (17% of
              on-hand kept in surplus periods). On `hicv` the same head at
              base's recipe lands 2,744 / 2,756, already below the bar
              (#E33). Scope: a decision with a "do nothing" option whose
              value is small and state-dependent; the menu's coarseness was
              NOT what bound (#E34 on a 12-rung menu, #E33 addendum 2 on the
              rule family).
failed:       `order_ship` quantities (#E13, 623.79) — the allocation handed
              to the fallback, ruled out; `order_softmax_free` (#E18) — no
              clip, still no zeros; ask features (#E7/#E8, 686 / 640) — the
              retailer need handed in as a feature adds nothing; split order /
              allocation trunks (#E9, +18..+41) and deep residual trunks —
              hurt; two-phase order-then-allocate steps with an interstate
              value (#E10, +1..+24) — no gain from separate credit; the
              entropy bonus off (#E20) and transit charged in training (#E19)
              — the drift was neither; the Dirichlet head (`order_dirichlet`,
              #E23/#E25/#E28/#E32) — concentrates in the interior, never
              uses its corners, 7.5–7.8 behind the categorical in the
              seed-matched #E32 pairs (6–8 at h4, #E23), at every recipe,
              and on `hicv` its tuned control collapses on one seed (#E33).
evidence:     #E2, #E4, #E6, #E7–#E14, #E15, #E17–#E23, #E33; `owmr_gym.py`
              class docstring (which modes are structure-free);
              `owmr_catkeep_policy.py`.
```

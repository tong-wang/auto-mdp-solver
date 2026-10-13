# PLAYBOOK — `cnv` (censored newsvendor with Bayesian demand learning from stock-out timing)

Digest entries per `ESCALATION_LOG_GUIDE.md` §10 — experiences, not rules; real names and numbers, the
ledger cited for everything else. **Only transferable tricks of the digital-twin + RL framework belong
here** (the human, 2026-10-09, on `owmr`): mechanisms that will matter on another problem, not the
campaign's own story — that lives in `ESCALATION.md` and `README.md`. The three entries and their order
are the human's selection at the close (2026-10-13); frame moves, modeling rules and negative space are
left to the ledger, as in `owmr`.

**Structure class:** a finite-horizon stocking problem under Bayesian learning of a hidden demand rate
from censored observations — each period's order decides how much of that period's demand is observed
(sales and, under timing observation, the stock-out time; under event observation only whether it sold
out), so the optimal order buys information; the exact DP exists at N = 4 and not at N = 100, where the
myopic newsvendor policy of each observation type is the benchmark (Jain, Rudi & Wang 2015).

**Protocol every number below is quoted at:** the N = 100 pair `fig6a` (unit cost 1.0) / `fig6b` (1.5),
prior Gamma(10/16, 1/16); 8192 seeds from 0, common random numbers, paired per seed against the myopic
benchmark of the policy's own observation type (fig6a: myopic_full 793.41 — the full-information
optimum — / myopic_timing 787.78 / myopic_event 781.21; fig6b: 341.92 / 335.29 / 323.99); deterministic
argmax play; checkpoints screened on 2048 disjoint seeds from 1,000,000, the top three confirmed on the
protocol block; the budget stated per arm. Entries that quote another protocol say so.

---

## LV1 — the vine when an action's payoff is spread over many future periods, each of them noisy

```
context:      R0 of the assistance ladder — raw frames, no statistic handed over (timing_raw) — after
              its own tuning study sat at −6.55 against myopic_timing (fig6a, study winner t74, #E79);
              the information ledger on that winner showed it does buy information (G +3.4) and loses
              it in the zero and top rate bands (#E106). The obvious lever, same-world comparisons
              built INTO the main rollout (a group of K envs sharing one world: the leave-one-out
              group baseline #E87, branch-and-grid #E88), ran from #E87 to #E104.
symptom:      the policy does not size the information premium: an early over-order's value never
              shows up cleanly in its gradient, so the learned first order sits wherever the noise
              left it, and every lever on the main rollout trades one band's error for another's.
diagnosis:    the action's payoff is not a consequence; it is a stream. An over-order now changes
              the belief, and the belief changes every later order — the value arrives as ~99 small
              terms, each buried in that period's Poisson demand and the policy's own sampling. PPO's
              advantage for the early action is the sum of all of them, and the critic cannot
              isolate it. Comparing actions on the SAME world cancels the shared noise, but doing it
              inside the main rollout costs distinct worlds: K envs per world cut the worlds per
              update to 64/K, and the high-rate states (10% of the prior) vanished from 40% of the
              8-world updates — a systematic under-order at high rates that neither a frozen
              extractor (#E96) nor GAE in place of the lever's Monte-Carlo credit (#E97) removed,
              and that sharing worlds alone reproduced (#E98); buying the worlds back with a 4×
              update and tuning that shape gained +1.9 (#E99, #E100).
prescription: algo layer — `cnv_vine_ppo.VinePPO` (#E110), owmr's coordinate vine (owmr LV1) with
              the window run to the END of the episode: leave the main rollout untouched; at each
              step, with probability 0.1 per env, fork the env (state + frame stack copied, the
              episode's realised world shared) into a parent copy playing the sampled order and 3
              siblings each drawing a fresh order from the policy at the same observation; every
              copy follows the stochastic policy to the end on a COMMON action-noise stream (the
              same uniform per period, inverse-CDF through each copy's own distribution); score
              each copy's profit-to-go against the leave-one-out mean of the others; one side PPO
              pass over those samples after the main update. No critic in the branch credit.
              @ fig6a, timing_raw on t74's recipe, 5M: vine − matched control +29.9 / +11.1 (seeds),
              fig6b +8.7 / +2.9; +5.36 ± 0.28 over t74 itself (#E110). Across the observation
              ladder, 10M: every raw mode gains on both seeds and both cells, 12 / 12 pairs,
              +5.7 to +48 (#E111, #E112). Tuned for itself (one 10M study per raw mode, #E116):
              timing_raw 788.06 = +0.28 ± 0.34 against myopic_timing (at the benchmark), full_raw
              792.94 = −0.48 of the optimum, event_raw 775.25 (+7.6 over the untuned vine). The
              ledger reads the mechanism off the crowns: the vine-trained timing policy over-orders
              at the first decision (+2 at fig6a, +4 at fig6b, where information is worth more) and
              earns back as information 2.81 of the 3.36 that costs (3.56 of 3.69 at fig6b) (#E118).
              Scope: (i) it helps policies that must LEARN the belief from raw frames, not ones
              handed the statistic — full_belief_log −0.7 / −3.7 (#E111); (ii) the gain is the
              comparison, not a sharper policy — low-entropy controls lose to the vine 8 / 8 (#E114),
              and the tuned recipes drop the large entropy bonuses the plain winners needed (1e-8 to
              1.6e-4 against 0.064); (iii) the dose is flat — five siblings, or a separate side
              optimizer at 2× lr, within the seed spread (#E115), unlike owmr's monotone dose.
              The one setting that differs from owmr: the window. owmr's decision is realised within
              10–20 periods; here it must run to the horizon — an over-order's information value is
              realised over every later period.
failed:       the group baseline (#E87) and branch-and-grid (#E88–#E104) — the comparisons built from
              the main rollout's own envs, at every group size: 16 × 400 collapses at high rates,
              K = 2 runs away in entropy, 64 × 400 restores the worlds and only ties tuned plain
              PPO until a full study (+1.9, #E100); the same lever on full_raw / event_raw is
              negative at its pilots (#E102, #E103).
evidence:     #E79, #E87–#E104 (the lever line), #E106 (ledger on t74), #E110 (lever + gates),
              #E111–#E113 (across the ladder), #E114 (sharpening ruled out), #E115 (dose), #E116
              (tuned), #E117 (fig6b transfer), #E118 (the ledger on the crowns); owmr LV1 (the same
              lever on a short-consequence decision — two campaigns now).
```

## LV2 — a neural network that does Bayesian learning: a shared per-frame map is the log-likelihood, the pool is its sum

```
context:      the raw modes hand the policy the last 100 periods as frames (time to go, critical
              ratio, the prior, the period's sales and its stock-out time / flag / demand) and no
              statistic. A generic MLP over the flattened stack learned a compression that
              rediscovered the Bayesian sums (R² 0.997) but landed far under the bar (full_raw
              745 / 757 against myopic_full 793.4, #E68–#E71). For event observation no conjugate statistic exists at all.
symptom:      raw frames trail the policy handed the posterior by a wide margin, with a learned
              representation that contains the statistic yet does not let the decision use it.
diagnosis:    Bayes' rule over exchangeable observations is a SUM: given the rate, the periods are
              independent, so log p(rate | history) = log prior + Σ_t ℓ(frame_t; rate) — and
              censoring does not break this (each order depends on the past, but the likelihood
              given the orders placed is still a product over periods). So the right architecture
              is the formula itself (the human, #E72: "frame the kernel as a log-likelihood and sum
              them over independent observations"): ONE shared network φ applied to every frame
              plays the per-observation log-likelihood ℓ(frame; ·) — a vector of features standing
              for it on a grid of rates — and pooling across frames plays the sum. For timing the
              exact term is y·log λ − λ·τ, linear in the frame, so φ needs only the two sums; for
              event it is a Poisson mass or a survival at the placed order, which no finite pair
              captures, so φ must learn an embedding. Two more facts decide whether it works:
              (1) the plain SUM hands the decision the posterior in its crude coordinates (Σ sales,
              Σ duration = shape − α, rate − β), which a standardised input cannot resolve early in
              the episode (LV3) — timing_raw with the sum pool 684 / 678 ≈ the crude timing_belief
              centre 681 (#E73); (2) dividing by the count — the MEAN of the log-likelihood terms —
              with the count passed beside it puts the statistic on the posterior-mean scale and
              the amount of evidence on its own axis: the coordinates LV3 found the belief needs.
prescription: arch layer — `cnv_raw_extractor` frame-MEAN form (#E75): φ (a small MLP, observation
              columns only) on each real history frame, the reset frame excluded, mean-pooled, the
              count as a feature, the present state passed straight to the body; the statistics
              that standardise the pooled features estimated over the first 200 rollouts and then
              frozen (#E76–#E78: with running statistics every run peaked at 1–1.6M and declined, the
              suspect a policy-dependent input; no statistics at all trains stably but leaves full /
              event under the running form; frozen after one rollout lifts full_raw only; frozen
              after 200 is the most consistent, second on every mode). @ fig6a, one untuned L1 wave,
              4M: timing_raw 750.0 — on the hand-recoded posterior's 750.86 — full_raw 781.2 (above
              the 20M tuned crude study's 774.5), event_raw 748.8, three crowns at once (#E75). A
              held-out ridge readout of the pooled features gives the exact log posterior mean at
              R² 0.97 in all three modes, trained ≈ untrained (±0.02): the pool does the recoding,
              not the training. With the vine on top (LV1), these raw frames end the campaign AHEAD
              of the belief input on both types and both cells (fig6a +0.9 full / +2.8 timing,
              fig6b +0.4 / +4.5; README).
              The bet that a sum is enough held exactly: a learned weight per history position,
              initialised at ones, stayed at 0.99–1.02 on both seeds — no recency weighting, as
              exchangeability says; only the newest frame took 1.32, a present-state gain that
              buys nothing (#E72).
failed:       the generic MLP compression (#E68–#E71; it found the sums and could not use them);
              the SUM pool (#E72, #E73 — full_raw 759, event_raw 725, timing_raw 684: the crude
              coordinates); a learned temporal kernel (stays at ones); a post-pool recoding MLP,
              closed without running once the mean pool did the job (F8, #E73 / #E74).
evidence:     #E68–#E71 (MLP compression), #E72 (the log-likelihood sum, the bet, the kernel), #E73
              (sum = crude coordinates), #E75 (the mean pool, the probe), #E76–#E78 (normalisation
              after the pool); design tree F6 → F7 → F9.
```

## LV3 — hand over the oracle to separate an encoding problem from an optimiser problem

```
context:      the belief-fed full_belief policy, tuned, sat 26.7 under myopic_full at fig6a (#E52);
              its residual lived in both tails of the rate distribution, and nothing in the tuned
              knob set moved it. A gap like this has three suspects that look identical on the
              board: the inference (the policy cannot form the belief), the encoding (it is handed
              the belief in a form it cannot read) and the optimiser (PPO cannot fit even a readable
              input).
symptom:      a policy plateaus short of a benchmark known to be reachable, and the levers on the
              optimiser — critic, advantage estimator, head, normalisation — move the residual
              between bands without removing it.
diagnosis:    two cuts, in order. (1) The ORACLE removes inference: a declared diagnostic mode hands
              the policy the episode's true rate ([time to go, critical ratio, λ]) — no censoring,
              no memory, and the optimum is one constant per episode (checked equal to the
              perfect-knowledge column, paired Δ 0.000). PPO still missed it: −14.2 / −12.4 with the
              discretised-Gaussian head, −56 / −104 with the categorical one (#E53) — half of
              full_belief's shortfall reproduced with nothing to infer, so inference was not the
              story. (2) Varying ONLY the oracle's encoding then separates encoding from optimiser:
              the same network, recipe and budget, the rate given as its log instead of
              standardised by its population spread — 803.68 / 803.72 against the 804.8 optimum,
              99.86% on both seeds (#E60). The "PPO floor" had been the input: standardised, the
              low rates sat a hair apart and the high ones seven sd out on a tanh. A cheap distance
              probe confirmed it on the real belief: the prior state and the posteriors after
              seeing 0, 10 or 30 sales — orders 0, 10 and 28 — sat 0.03–0.05 apart in the
              network's input units (#E56).
prescription: protocol layer — when a learned policy plateaus short of a reachable bar, (1) add an
              oracle mode that hands over the latent the policy must infer, and train it with the
              same recipe; if the oracle also plateaus, the problem is not inference; (2) vary only
              the oracle's ENCODING (raw / standardised / log; and measure the distance between
              decisive states in input units) before touching the optimiser; the encoding that
              clears the oracle is the one to give the real statistic. Here it transferred whole:
              the posterior recoded as (log mean, 1 / √shape) — full_belief 741 / 745 → 789.5 /
              787.7 (#E59), timing_belief to 750.86 where it had been broken (#E65) — and it is the
              same coordinate system the raw frames' mean pool produces (LV2). The oracle mode stays
              off the leaderboard (it reads the hidden rate); it is an instrument, declared as one.
failed:       a day of optimiser levers on the oracle before its encoding was varied, each measured
              and none moving the floor: GAE noise and the critic (#E54), the head's location
              geometry — affine, square root (#E55) — a zero atom and a PopArt value head (#E57),
              the fully relaxed pinned-rate newsvendor (#E58), observation normalisation off (#E61),
              the tuned study's recipe (#E62). Each was a correct description of how a network fed
              an unresolvable input fails — none was the cause.
evidence:     #E52 (the residual), #E53 (the oracle cut), #E54–#E58 (the optimiser suspects), #E56
              (the input-distance probe), #E59 / #E65 (the recoding on the real belief), #E60 (the
              oracle's log encoding), #E61, #E62; design-tree nodes F3, F4.
```

# game2048

The 2048 sliding-tile game as an MDP, generated from
[`game2048_schema.json`](game2048_schema.json) and verified against it
bit-for-bit by the differential. Built to fill the case set's
**spatial-board / action-masking** gap.

## The problem

Slide the board in one of four directions; equal tiles merge and double; a new
tile spawns into a uniformly-drawn empty cell after every slide that changed
something; the episode ends when no slide is legal.

- **Decision** — `move` ∈ {0=Up, 1=Down, 2=Right, 3=Left}, `Discrete(4)`.
- **Uncertainty** — the spawn: a cell drawn uniformly over the *empty* cells
  (`spawn_cell`, stream 0) and a value that is a 4 with probability `prob_4`
  (`spawn_value`, stream 1). The cell's support is **state-dependent** — it is
  whatever the board leaves empty.
- **Horizon** — the game ends on its own at game over
  (`early_terminated_when`); `T_cap` is an external move cap, so `horizon_end`
  is `truncated`.
- **Discount** — β = 1.0. It is a game.
- **Objective** — **merge score**: every merge producing a tile of value `v`
  pays `v`. This is the game's own score.

**What makes it hard is that the objective is a state function of the final
board.** With the board potential `Φ(B) = Σ_tiles v·(log₂v − 1)`, a merge of two
`v`s raises Φ by exactly the `2v` it pays, a spawned 2 raises it by 0, and a
spawned 4 raises it by 4 while paying nothing. Hence, exactly:

> `total merge score = Φ(final board) − 4 × (number of 4s spawned)`

Two policies that die on the same board having received the same number of 4s
earned the same score, whatever route they took — there is **no efficient or
wasteful path to a given final board**. A policy therefore moves the objective
through exactly two channels: **survival** (how much material arrives) and
**concentration** (how it is left arranged — Φ is convex in tile rank, so one
4096 beats two 2048s). Every dense reward in the design space is a proxy for
one of those two. The identity is checked by an executable gate
(`game2048_reward_gate.py`, `gr8`); the full reward taxonomy is `ESCALATION.md`
#E49.

There is **no exact DP**: tile values are unbounded, so the state space cannot
be enumerated. The strongest available reference is a depth-limited search,
which is a reference and not a free upper bound — it pays thousands of board
evaluations per move where a trained policy pays one forward pass.

## What this domain exercises

| feature | where |
|---|---|
| spatial board state | `board`, an `int_vector` of length `n_cells`, reshaped to `(1,n,n)` / `(K,n,n)` in the gym |
| domain expression builtins | `mdp.expr_builtins` → [`game2048_board.py`](game2048_board.py) — the slide/merge rules the IR calls |
| a **state-dependent** categorical support | the spawn cell is `categorical` over `empty_cells(board)`, resolved from the live namespace at draw time |
| **keyed** uncertainty stages | both spawn slots key on `spawn_count`, not `period` |
| discrete actions + masking | `masked` action mode, `valid_actions` in the mdp layer (spec §7.1) |
| a feasibility-strategy menu | `masked` vs `free` in one IR — the same world, two interfaces |
| CNN policies | `SmallBoardCnn` (spec §8.5); SB3's NatureCNN cannot run on a 3×3 board |
| obs normalization deliberately **off** | tile magnitudes drift upward all episode, so running stats chase a moving target |

## Layout

### Documents

| file | what it is |
|---|---|
| `README.md` | this file: the problem, the layout, the results, the commands |
| `CLAUDE.md` | the operating brief — what governs this folder, and the gate commands |
| `game2048.restatement.md` | the Phase-A restatement the IR was frozen against |
| `ESCALATION.md` | the campaign record: design tree, changelogs, numbered findings (`#E…`) |
| `INTERPRET.md` | the policy readback (spec §14) — what the trained nets actually do |
| `PLAYBOOK.md` | the case-close distillation (ESCALATION_LOG_GUIDE §10) |
| `figures/` | committed replay animations (board + actor π(a\|s) + critic V(s) per move), inlined by `INTERPRET.md`. **Partly owed:** §14.3's `game2048_plot_policy.py` — one spec, a static *and* an interactive render of the action surface — is not built |

`ESCALATION.md` here is a **scoped subset** of a longer campaign — 32 of 85
numbered entries. Its own header says what is held back and why, and marks
every line it redacted.

### Code

| file | role |
|---|---|
| `game2048_schema.json` | the IR: one catalog, 8 instances (board size × spawn regime) |
| `game2048_board.py` | board mechanics — the IR's `expr_builtins`, shared with `_mdp` so the rules have one implementation |
| `game2048_exceptions.py` | `Game2048Error`, `InvalidActionError` |
| `game2048_uncertainty.py` | `UniformEmptyCell`, `Bernoulli4`; the v2 keyed seed key |
| `game2048_scenarios.py` | `Game2048Scenario`, the `SCENARIOS` registry |
| `game2048_mdp.py` | `Game2048State`, `init_state` / `open_episode` / `advance`, `valid_actions` |
| `game2048_gym.py` | `Game2048Env` — 8 obs modes × 2 action modes × 4 reward modes |
| `game2048_ir_adapter.py` | differential adapter (portable-domain contract) |
| `game2048_test.py` | the domain's own tests, incl. 3 negative controls |
| `game2048_benchmark_common.py` | the one spec-§9 seed loop and TSV writer — used by the baselines **and** the RL eval |
| `game2048_benchmark_{random,greedy,expectimax}.py` | baselines, with `_eval.py` each |
| `game2048_benchmark_snake{,_expectimax}.py` | the strategy-prior heuristic and the search that uses it as its leaf, with `_eval.py` each |
| `game2048_ppo_train.py` | MaskablePPO / PPO training, `--level {L0,L1}` (spec §8.6) |
| `game2048_ppo_eval.py` | scores the deployable policy on the shared seeds |
| `game2048_policy.py` | the deployable artifact: `act(board) -> move` |
| `game2048_policy_probe.py` | the spec-§14.1 probe; delegates the ladder to `game2048_interpret.py` |
| `game2048_interpret.py` | the readback stages (replay / structure / probes / counterfactual / symmetry) |
| `game2048_{extractor,attn,enclook,reward}_gate.py` | executable gates for the levers the findings cite |

## Results

### Protocol

Every number below is quoted at **8192 episode seeds from 0**, common random
numbers (CRN) across arms — one shared seed loop through one env,
`game2048_benchmark_common.evaluate`. The record eval is **deterministic
argmax over masked logits**, which is the deployment mode; `--stochastic` is
reachable and is a cross-check only, never a record. PPO rows are scored
through `Game2048Policy`, the deployable wrapper, not the raw model.

`score_mean` is the objective (merge score) and the **only deciding metric**.
Moves, max tile and the max-tile CDF (`reach_2` … `reach_1024`) are carried in
every eval TSV as bystanders — descriptive, never deciding. Each TSV also
writes a per-seed sidecar (`*.seeds.tsv`) from which any paired comparison can
be rebuilt.

Symbols: **L0** = spec-§8.6 faithful defaults, **L1** = the derived config.
Obs-mode suffixes name what the observation carries — `onehot` (tile planes
indexed by exponent), `_la` (the four post-decision afterstates appended),
`_s` (an added scalar magnitude plane). Two uncertainties are reported and
never interchanged: **eval SE** over the 8192 episodes, and **training-seed
spread** across independent retrains — at 4×4 the binding one (`ESCALATION.md`
#E29 measures sd = 442, so any 4×4 gap under ~880 is inside single-draw noise).

### `1-comparative` — how does RL compare with the existing solutions?

| scenario | config | what it is for |
|---|---|---|
| `3x3_20` | 3×3 board, `prob_4` = 0.2 | the IR's base composition and the development target — small enough to move a lever and read it |
| `4x4_20` | 4×4 board, `prob_4` = 0.2 | the canonical game; the scale the campaign's claim is about |

Scenarios are named `{n}x{n}_{prob_4 × 100}`. The other six catalog instances
(`2x2_{0,20}`, `3x3_0`, `4x4_0`, `5x5_{0,20}`) carry no leaderboard by design:
they are swept by the differential for formalization fidelity and never
trained.

**These two are SEPARATE LEADERBOARDS whose scores may never be compared** —
different bars, different problem scales. A number may be reported across
them; it never selects. Both are sorted by `score_mean` ascending; `Δ vs bar`
is a signed difference in merge points.

#### `3x3_20` — bar: expectimax d3 = 1154.25

| arm | role | score_mean | ± eval SE | Δ vs bar |
|---|---|---|---|---|
| random-valid | baseline | 174.21 | 1.06 | −980.04 |
| greedy (myopic) | baseline | 374.29 | 2.23 | −779.96 |
| PPO `vec` MLP — L0 faithful defaults | RL floor | 480.06 | 3.14 | −674.19 |
| snake heuristic (v3, `--mode proc`) | baseline | 640.35 | 4.51 | −513.90 |
| expectimax d2 | reference | 970.36 | 4.77 | −183.89 |
| **expectimax d3** | **reference bar** | **1154.25** | **5.56** | — |
| snake-leaf expectimax d2 | reference | 1193.97 | 7.26 | +39.72 |
| snake-leaf expectimax d3 | reference | 1437.62 | 8.22 | +283.37 |
| **PPO `onehot_s_la`, L1, 5→10M** | **RL, ships** | **2688.37** | **15.04** | **+1534.12** |

**Ships: `onehot_s_la` at L1, 10M steps (a 5M run resumed to 10M), λ = 0.95,
seed 42** — 233% of the d3 bar and 187% of snake-leaf d3, so at this scale a
learned arm clears every search reference on the board.

**Why arms are missing, and what that costs this claim.** The campaign trained
arms on lever families that are **not published in this case** (`ESCALATION.md`
says which). At 10M two of them finished *above* 2688.37, within a 40-point
spread — inside the ~65–85 training-seed band at this scale. So the 3×3 summit
is a **statistical tie**, and the arm shipped here is **the best arm of this
case, not the best arm measured**. `ESCALATION.md` #E37 records the tie in
full.

#### `4x4_20` — bar: snake-leaf expectimax d3 = 51652.03

| arm | role | score_mean | ± eval SE | Δ vs bar |
|---|---|---|---|---|
| random-valid | baseline | 1014.90 | 5.60 | −50637.13 |
| greedy | baseline | 3053.01 | 17.65 | −48599.02 |
| snake heuristic (v3, `--mode proc`) | baseline | 4598.68 | 35.40 | −47053.35 |
| expectimax d2 | reference | 13155.04 | 56.08 | −38496.99 |
| **PPO `onehot_la`, L1, 10→20M** | **RL, ships** | **17084.31** | **93.52** | **−34567.72** |
| expectimax d3 | reference | 20804.48 | 92.48 | −30847.55 |
| snake-leaf expectimax d2 | reference | 29048.31 | 168.79 | −22603.72 |
| **snake-leaf expectimax d3** | **reference bar** | **51652.03** | **290.64** | — |

**Ships: `onehot_la` at L1, 20M steps (a 10M run resumed to 20M), λ = 0.95,
seed 42** — it clears expectimax d2 at 130% and reaches 82% of d3. As at 3×3,
arms on lever families not published here scored higher; this is the best arm
of this case.

**The strategy gap is the open result at this scale.** Replacing the search's
leaf evaluator with a hand-shaped monotone-chain potential — a few dozen lines
of heuristic, no training — and searching the same three plies produces
51652.03, which stands **3.0× above the arm shipped here**. Depth on that
strategy leaf pays more than depth on a score leaf (+78% vs +58%,
`ESCALATION.md` #E46). The structure the learned policies do not find is worth
more than every lever this case reports. Later work not included here narrows
the gap; it does not close it.

Note the campaign holds **no `exact` and no `relaxed` node**, so the optimum is
never bracketed: snake-leaf d3 is the best policy *built*, not a proven
ceiling.

How each arm was reached: `ESCALATION.md` §MAP for the design tree, and the
entries cited in the text above.

### `2-structural` — the declared stances

`game2048_schema.json` declares two tier-2 questions (spec §14.0). Verdicts in
brief; the evidence and the full readback are in `INTERPRET.md`.

**1. `discover` — what is the pattern of the trained policy?** *Partly
answered; the deliverable is owed.* The policy anchors a **home corner** and
holds a **monotone ordering** off it, to the end — corner occupancy 0.970
against a myopic control's 0.714, and the home corner is TR in 98.6% of
max-tile states. The corner's *identity* is arbitrary across runs; the
*commitment* is load-bearing (frame-averaged evaluation costs −69%). What is
not delivered is what `discover` owes: a fitted rule, scored paired against the
policy as a first-class §9 benchmark.

**2. `confirm` — does the policy build the snake ordering the strongest
reference plays?** *No snake observed, and the instrument is why the answer is
not stronger.* The policy's chain reaches 4.83/9, equal to the snake
reference's 4.83 and above the myopic control's 4.33 — so the metric cannot
separate them. Two things say that agreement is not recovery: the two play
differently at the same chain length (home corner 0.986 vs 0.344), and the
snake-leaf search still stands 3.0× above the shipped arm at 4×4.

## Technical appendix

Run from the case folder. Together these reproduce `results/` from an empty
folder; `results/` is gitignored. The **gate** commands are not here — they are
in `CLAUDE.md`, because they are run by whoever is changing this folder rather
than reading it.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# baselines and references (write results/{scenario}/benchmark/)
python game2048_benchmark_random_eval.py           -s 3x3_20 --n-seeds 8192
python game2048_benchmark_greedy_eval.py           -s 3x3_20 --n-seeds 8192
python game2048_benchmark_expectimax_eval.py       -s 3x3_20 --n-seeds 8192            # ~25 min
python game2048_benchmark_expectimax_eval.py       -s 3x3_20 --n-seeds 8192 --depth 3  # ~5 h
python game2048_benchmark_snake_eval.py            -s 3x3_20 --n-seeds 8192 --mode proc
python game2048_benchmark_snake_expectimax_eval.py -s 3x3_20 --n-seeds 8192 --depth 3

# train — L1 is the derived config; L0 is the faithful-defaults control
python game2048_ppo_train.py -s 3x3_20 -o onehot_s_la --level L1 --total_timesteps 5000000
python game2048_ppo_train.py -s 3x3_20 -o vec         --level L0

# extend a finished run to the shipped budget (the shipped arms are resumes)
python game2048_ppo_train.py -s 3x3_20 -o onehot_s_la --level L1 \
    --total_timesteps 10000000 --resume-from results/3x3_20/<run>/3x3_20_mppo.zip

# evaluate on the same 8192 seeds as the baselines
python game2048_ppo_eval.py -s 3x3_20 -o onehot_s_la \
    --model-path results/3x3_20/<run>/best_model.zip --n-seeds 8192

# the deployable policy, replayed against the raw _mdp loop
python game2048_policy.py -s 3x3_20 -o onehot_s_la \
    --model-path results/3x3_20/<run>/best_model.zip

# the §14 readback (writes results/{scenario}/<run>/interpret/)
python game2048_policy_probe.py stances \
    --model-path results/3x3_20/<run>/best_model.zip -s 3x3_20 -o onehot_s_la
```

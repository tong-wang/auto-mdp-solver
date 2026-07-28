# dynamic_pricing

Dynamic pricing of a fixed initial stock over a finite selling horizon
(Gallego & van Ryzin 1994, *Optimal dynamic pricing of inventories with
stochastic demand over finite horizons*). Each period the agent sets a price;
demand arrives as Poisson with intensity `lambda(p) = a * exp(-alpha * p)`;
sales are capped by remaining stock (no backlog, no reorder); the objective is
expected revenue. The continuous-time intensity-control problem is discretized
into `horizon = 50` periods of length `dt = 0.02` (unit horizon).

This domain was generated from the frozen MDP-IR
`dynamic_pricing/dynamic_pricing_schema.json` following the `mdp-solver` skill's Phase B;
`python -m mdp_ir.differential dynamic_pricing/dynamic_pricing_schema.json` replays
the IR interpreter against this implementation (bit-exact on both instances).

## Layout (spec `MDP_PROJECT_SPEC.md`)

| File | Purpose |
|---|---|
| `dynamic_pricing_uncertainty.py` | `SamplingContext`, `PoissonArrivals` (price-conditioned primitive; seed key `[stream=1, period, episode_seed, seed_salt]`) |
| `dynamic_pricing_scenarios.py` | `DynamicPricingScenario`, `SCENARIOS` (`simple`: a=100, `ample_stock`: a=20) |
| `dynamic_pricing_mdp.py` | `DynamicPricingState`, `init_state`, `advance` (pure, reward-agnostic; early termination at inventory 0) |
| `dynamic_pricing_gym.py` | `DynamicPricingEnv`: action modes `price` (Box [0,8], clip) / `intensity` (Box [0, a/e], reparametrized via `p = log(a/lambda)/alpha`); obs modes `vec` = [inventory, time_to_go] / `vec_d` (+ last units_sold); reward `revenue` |
| `dynamic_pricing_benchmark_dp.py` | Exact DP benchmark (backward induction over (t, n), price grid) → solutions file + optimal value |
| `dynamic_pricing_benchmark_dp_eval.py` | Seed-loop eval of the DP benchmark (TSV) |
| `dynamic_pricing_benchmark_fluid.py` | Fluid benchmarks (from the source paper): `fixed` (GvR fixed-price `p_FP = max(1/alpha, log(a*H/n0)/alpha)`), `myopic` (`1/alpha`), plus a `random` sanity floor |
| `dynamic_pricing_ppo_train.py` | SB3 PPO training (VecNormalize obs norm on, per IR `rl` block) |
| `dynamic_pricing_ppo_eval.py` | Seed-loop eval of a trained model (VecNormalize injection, spec §9.5) |
| `dynamic_pricing_ppo_tune.py` | Thin wrapper over the repo-level `mdp_tuning` harness (`--metric revenue_mean`) |
| `dynamic_pricing_policy.py` | Deployable policy: model + vecnorm stats + action transform behind `act(obs) -> price` |

## Usage

```bash
cd dynamic_pricing

# benchmarks
python dynamic_pricing_benchmark_dp.py -s simple            # exact DP -> results/simple/benchmark/dp/simple.txt
python dynamic_pricing_benchmark_dp_eval.py --solutions results/simple/benchmark/dp/simple.txt -s simple --n-seeds 8192
python dynamic_pricing_benchmark_fluid.py -s simple --policy fixed --n-seeds 8192

# train + eval (pin BLAS threads; policy is tiny)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python dynamic_pricing_ppo_train.py -s simple -a price
python dynamic_pricing_ppo_eval.py --model-path results/simple/PPO_.../simple_ppo.zip -a price --n-seeds 8192

# hyperparameter tuning (Optuna, repo-level harness)
python dynamic_pricing_ppo_tune.py -s simple --n-trials 25 --total-timesteps 200000

# deploy
python dynamic_pricing_policy.py --model-path results/simple/PPO_.../simple_ppo.zip -a price
```

## Results (scenario `simple`: a=100, alpha=1, n0=25, T=50; 8192 seeds)

Expected optimal revenue (DP, exact for the discretized MDP): **33.10**;
deterministic fluid upper bound: 34.66.

| policy | mean revenue | note |
|---|---|---|
| DP optimal              | 33.15 | matches the computed V[0][n0] within noise |
| **PPO, `price` mode, tuned** | **32.96** | 200k steps, Optuna trial 16; 99.4% of DP |
| PPO, `price` mode, defaults | 32.82 | 500k steps; 99.0% of DP |
| fixed price (GvR p_FP=1.386) | 31.92 | asymptotically optimal static policy |
| PPO, `intensity` mode   | 25.95 | best of two runs (1M steps, ent_coef 0.01); first run (500k, defaults) got 17.34 |
| myopic (p*=1.0)         | 24.97 | sells out early: price too low for n0=25 |
| random (U[0,8])         | 12.56 | sanity floor |

On `ample_stock` (a=20) the stock constraint never binds: DP = fixed = myopic
= 7.39 (the DP policy is the constant myopic price), random = 2.49.

**Tuning study** (`results/tuning/dynamic_pricing_simple_ppo/`, 25 Optuna TPE
trials × 200k steps, scored on 2048 seeds). Best: trial 16 —
`learning_rate=1.6e-4, ent_coef=2.6e-3, gamma=0.9995, n_steps=512,
batch_size=32, n_epochs=20, net_arch=(64,64)` → 32.96 on the 8192-seed
protocol (`trial_0016/simple/PPO_*/ppo_eval_simple_8192.tsv`). The winning
region is consistent across the top trials: small net, n_steps 512, 20
epochs, gamma ≈ 0.999 (γ ≤ 0.99 reliably hurts in this undiscounted
finite-horizon problem). Caveat: retraining the trial-16 config for 500k
steps (seed 1) scored only 32.31 — the heavy 20-epoch recipe does not
improve with a larger budget — so the shipped best model is the trial-16
artifact itself.

**Empirical action-mode finding.** The IR suggested the paper's native
`intensity` control might be easier for PPO (revenue linear in sales); with
vanilla SB3 PPO the opposite holds. The policy's Gaussian is initialized at
raw action 0, which in the `intensity` box `[0, a/e]` is the null (shut-off)
price — half the initial action mass clips to zero revenue, and learning
crawls out of that desert slowly (17.3 after 500k steps; 26.0 after 1M with
extra entropy). In the `price` box `[0, 8]`, raw 0 is a low-but-selling price,
so the gradient signal is immediate. `price` is the recommended (and default)
action mode.

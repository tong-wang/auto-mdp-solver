# inv_single — Single-Echelon Inventory RL

Single-echelon inventory management simulator with PPO training via Stable Baselines 3 and a finite-horizon DP benchmark.

## Files

| File | Purpose |
|---|---|
| `inv_single_api.py` | Pure-Python simulator (no Gym dependency) |
| `inv_single_gym.py` | Gymnasium wrapper for SB3 training |
| `inv_single_scenarios.py` | Pre-defined scenario registry |
| `inv_single_train_ppo.py` | PPO training + evaluation script |
| `inv_single_benchmark_dp.py` | Finite-horizon DP benchmark solver (backward induction) |
| `inv_single_benchmark_dp_eval.py` | DP benchmark evaluation + TensorBoard logging |

## Concepts

### Event sequence

Each period executes three events in order: **R** (receive), **O** (order), **D** (demand). The default is O→R→D. Valid sequences are O-R-D, R-D-O, and R-O-D (R-O-D requires lead time ≥ 1).

### Pipeline

`pipeline[k]` holds the quantity arriving at the *k*-th future R event. Length is fixed at `leadtime.max() + 1` throughout an episode.

### Demand seeding

Demand is re-generated each step from a deterministic seed sequence combining the scenario salt, episode seed, and period index. Two episodes with the same seed will see identical demand regardless of the actions taken.

## Scenarios

| Name | Demand | Lead time | h | b | K | Backlog |
|---|---|---|---|---|---|---|
| `simple` | Poisson(10) | 0 | 1.0 | 9.0 | 0 | yes |
| `simple-k` | Poisson(10) | 0 | 1.0 | 9.0 | 20 | yes |
| `discrete_stochastic` | sampled(3, 8–12) | Discrete{2–5} | 0.2 | 2.0 | 25 | yes |
| `discrete_lost_sales` | sampled(3, 8–12) | Discrete{2–5} | 0.2 | 2.0 | 25 | no |
| `poisson` | Poisson(λ), λ~Gamma(9,0.3) | Discrete{2–5} | 0.2 | 2.0 | 25 | yes |

The `discrete_*` and `poisson` entries are latent-bearing sources (spec §5.2): the world latent lives on the demand generator itself, realized per episode through the generic `InvSingleScenarioSource` on the demand slot's own stream — so instances sharing a demand model see identical latent draws per episode seed (common random numbers):

- `sampled(3, 8–12)` — a `LatentDiscreteDemand`: each episode draws 3 integer demand values uniformly from {8,…,12} with random weights, realized as a `DiscreteDemand`.
- `Poisson(λ), λ~Gamma(9,0.3)` — a `LatentPoissonDemand`: each episode draws a hidden Poisson rate λ ~ Gamma(9, 0.3) (mean 30), realized as a `PoissonDemand`.

## Observation modes

| `--obs-mode` | Features |
|---|---|
| `vec` | period, inventory, pipeline[0..L] |
| `vec_d` | same as `vec` + last demand (default) |
| `vec_d_ip` | period, inventory position, last demand |

Inventory position = inventory + sum(pipeline).

## Results folder structure

```
results/
  <scenario>/
    PPO_<timestamp>_<changed-args>/
      <scenario>_ppo_args.txt
      ppo_inv_single.zip
      vecnormalize.pkl
      checkpoints/
      train.log                # captured console output of the run
      train_log_*.log          # per-episode/per-step gym logs, only with --gym-log 1
      monitor.csv
    benchmark/
      dp/
        V.csv
        pi.csv
        meta.json
        events.out.tfevents.*
```

## Training (PPO)

```bash
cd inv_single
python inv_single_train_ppo.py                        # defaults
python inv_single_train_ppo.py --scenario discrete_stochastic --seed 42
python inv_single_train_ppo.py --scenario simple --total-timesteps 500000 --no-norm-reward
```

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--scenario` | `simple` | Scenario name |
| `--obs-mode` | `vec_d` | Observation feature set |
| `--total-timesteps` | 2 000 000 | Training budget |
| `--seed` | 1 | Global random seed |
| `--lr-init` / `--lr-final` | 3e-4 / 3e-5 | Learning rate schedule |
| `--clip-init` / `--clip-final` | 0.2 / 0.05 | PPO clip range schedule |
| `--no-norm-reward` | *(reward normalised by default)* | Disable reward normalisation |

## Benchmark (DP)

`inv_single_benchmark_dp.py` computes the optimal finite-horizon policy by backward induction. The state is inventory (LT=0) or inventory position (LT>0). Results are cached to `results/<scenario>/benchmark/dp/` and reloaded on subsequent runs — recomputation only triggers if grid parameters change. Pass `--no-cache` to force a full re-solve.

```bash
cd inv_single
python inv_single_benchmark_dp_eval.py                          # defaults (simple, 1000 episodes)
python inv_single_benchmark_dp_eval.py --scenario simple-k --n-episodes 500
python inv_single_benchmark_dp_eval.py --scenario simple-k --no-cache
```

### DP notes

| Scenario type | State | Exactness |
|---|---|---|
| LT=0 | inventory | Exact |
| Deterministic LT=1 | inventory position | Exact |
| Deterministic LT>1 | inventory position | Approximation |
| Stochastic LT | inventory position | Approximation |

## Comparing RL and DP in TensorBoard

Both PPO and DP write `rollout/ep_rew_mean` using the same pseudo-timestep scale (`episodes × horizon`). Point TensorBoard at the scenario folder to see both on the same plot:

```bash
tensorboard --logdir inv_single/results/simple/
# RL runs appear as PPO_<timestamp>_*/, DP benchmark under benchmark/dp/
```

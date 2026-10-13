"""The paired order error bucketed by the SEED's realised lambda, not by period.

A per-period paired comparison averages over all (seed, period) and reports a policy
that is unbiased to 0.04 units. That average hides the campaign's actual failure: the
paired profit loss is U-shaped in the demand draw. This script rebuckets the SAME
comparison by lambda, which is what the prior's CV = 1/sqrt(alpha) makes vary.
"""
import argparse, sys
import numpy as np
from scipy.stats import poisson
from pathlib import Path
from cnv_benchmark_common import eval_seed_block
from cnv_bayes import gamma_poisson_quantile
from cnv_gym import CnvEnv
from cnv_ppo_eval import _vecnorm_for, load_policy, RAW_MODES
from gymnasium.wrappers import FrameStackObservation
from cnv_scenarios import SCENARIOS

p = argparse.ArgumentParser()
p.add_argument("--model-path", required=True)
p.add_argument("-s", "--scenario_name", default="fig6a")
p.add_argument("-o", "--observation_mode", default="full_belief")
p.add_argument("--n-seeds", type=int, default=512)
p.add_argument("--first-seed", type=int, default=0)
p.add_argument("--myopic-seeds", required=True, help="a benchmark *.seeds.tsv covering the same block")
p.add_argument("--bar-col", default="profit", help="which column of that file is the bar: profit (the benchmark itself) or perfect_profit (the known-rate optimum)")
a = p.parse_args()
sc = SCENARIOS[a.scenario_name]; N = int(sc.horizon)
def env_fn():   # the raw modes are frame-stacked to the horizon, as the evaluator does
    e = CnvEnv(scenario=sc, observation_mode=a.observation_mode)
    return FrameStackObservation(e, stack_size=N, padding_type="zero") if a.observation_mode in RAW_MODES else e
mp = Path(a.model_path).resolve()
policy, _ = load_policy(mp, _vecnorm_for(mp, None), True, env_fn)

seeds = eval_seed_block(a.first_seed, a.n_seeds)
lam = np.zeros(len(seeds)); rl_p = np.zeros(len(seeds)); my_p = np.zeros(len(seeds))
sum_rl = np.zeros(len(seeds)); sum_my = np.zeros(len(seeds)); sum_abs = np.zeros(len(seeds))
env = env_fn(); base = env.unwrapped
for i, sd in enumerate(seeds):
    obs, _ = env.reset(seed=int(sd)); tot = 0.0
    for t in range(N):
        st = base._state
        if a.observation_mode in ("oracle_rate", "oracle_log"):   # nothing to infer: the optimum is the known-rate quantile
            y_my = int(poisson.ppf(sc.critical_ratio, float(base._scenario_ep.demand.rate)))   # the RESOLVED episode rate
        else:
            y_my = gamma_poisson_quantile(sc.prior_shape + st.cum_demand, sc.prior_rate + st.period, sc.critical_ratio)
        y_rl = int(policy(base, obs))
        sum_rl[i] += y_rl; sum_my[i] += y_my; sum_abs[i] += abs(y_rl - y_my)
        obs, r, done, _, _ = env.step(y_rl); tot += r
        if done: break
    rl_p[i] = tot
    lam[i] = base._state.cum_demand / N   # realised mean demand/period: the seed's draw of lambda
# myopic's OWN profit on the same seeds, read from the benchmark record (exact, already scored)
import pandas as pd
_bm = pd.read_csv(a.myopic_seeds, sep="\t", comment="#").set_index("seed")[a.bar_col]
my_p = np.array([_bm.loc[int(sd)] for sd in seeds], dtype=float)

qs = np.quantile(lam, np.linspace(0, 1, 7))
buck = np.clip(np.digitize(lam, qs[1:-1]), 0, 5)
print(f"{a.scenario_name} · {a.observation_mode} · {len(seeds)} seeds from {a.first_seed}")
sem = np.std(rl_p - my_p, ddof=1) / np.sqrt(len(seeds))
print(f"  RL {rl_p.mean():.2f}   bar[{a.bar_col}] {my_p.mean():.2f}   paired {rl_p.mean()-my_p.mean():+.2f} +/- {sem:.2f}\n")
print(f"  {'demand/period':>16s} {'n':>4s} {'mean y RL':>10s} {'mean y myo':>11s} {'y ratio':>8s} {'|err|/period':>13s} {'paired profit':>14s}")
for b in range(6):
    m = buck == b
    if not m.any(): continue
    print(f"  {qs[b]:6.2f}-{qs[b+1]:<8.2f} {m.sum():4d} {sum_rl[m].mean()/N:10.2f} {sum_my[m].mean()/N:11.2f} "
          f"{sum_rl[m].sum()/max(sum_my[m].sum(),1e-9):8.3f} {sum_abs[m].mean()/N:13.2f} {(rl_p[m]-my_p[m]).mean():+14.2f}")

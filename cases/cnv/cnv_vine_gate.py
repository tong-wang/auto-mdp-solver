"""#E110 vine gates on one real rollout at t74's recipe (fig6a, timing_raw, 4 x 1024).

G3  the MAIN rollout under the vine (p 0.2) is identical to plain PPO's at the same seed (obs, actions, rewards,
    log-probs): the vine draws its forks and continuations from its own numpy streams and never steps a main env;
G4  every fork is credited before the update (pool empty, no fork pending) and the side-sample count is
    forks x siblings (parent baseline) or forks x (siblings + 1) (loo);
G5  common world + common continuation noise: copies of one fork whose FIRST orders are equal earn exactly the
    same profit to the episode's end (the parent baseline's zero contrasts; loo's equal-order pairs);
G6  the stored old log-probs equal the policy's own at collection (recomputed), the side pass runs, moves the
    parameters, and logs.

    python cnv_vine_gate.py
"""
import sys

import numpy as np
import torch as th

import cnv_ppo_train as T

BASE = ["x", "-s", "fig6a", "-o", "timing_raw", "--policy-dist", "dgauss", "--mu-param", "log", "--level", "L2hparch",
        "--learning_rate", "0.00018330068101019389", "--lr-final", "1.83301e-05", "--clip-init", "0.2", "--clip-final", "0.05",
        "--batch-size", "128", "--n-epochs", "4", "--target-kl", "0.02", "--gamma", "0.967108", "--gae-lambda", "0.978983",
        "--ent-coef", "0.06367016202074935", "--net_arch", "64", "64", "64", "--no-norm-obs", "--normalize-advantage",
        "--raw-frame-v2", "--raw-pool", "mean", "--raw-stats-off", "--raw-stats-freeze-at", "200", "--vf-coef", "0.9654464308671344",
        "--max-grad-norm", "1.0", "--n-envs", "4", "--n-steps", "1024", "--seed", "1"]


def build(extra):
    sys.argv = BASE + extra
    a = T.parse_args(); sc = T.resolve_target(a); a.frame_stack = T.derived_frame_stack(a.observation_mode, a)
    env = T.build_training_env(a, None, sc, a.frame_stack)
    m = T.build_model(a, env, None)
    m.tensorboard_log = None
    return m


def one_rollout(m):
    _, cb = m._setup_learn(m.n_steps * m.n_envs, callback=None)
    m.collect_rollouts(m.env, cb, m.rollout_buffer, m.n_steps)
    return m


ok = True
plain = one_rollout(build([]))
PLAIN = {k: getattr(plain.rollout_buffer, k).copy() for k in ("observations", "actions", "rewards", "log_probs")}   # train() flattens the buffer in place
for bl in ("parent", "loo"):
    sib = 1 if bl == "parent" else 3
    v = build(["--vine", "--vine-p", "0.2", "--vine-siblings", str(sib), "--vine-baseline", bl])
    v.vine_record = True
    one_rollout(v)
    rb, rp = v.rollout_buffer, plain.rollout_buffer
    g3 = all(np.array_equal(getattr(rb, k), PLAIN[k]) for k in ("observations", "actions", "rewards", "log_probs"))
    forks = int(v._vine_stats["forks"]); n = len(v._side["obs"])
    g4 = (not v._pool) and (not v._forks) and n == forks * (sib if bl == "parent" else sib + 1)
    d = v._side_detail
    if bl == "parent":
        pairs = [(x["R"][0], x["R"][x["k"]]) for x in d if x["firsts"][x["k"]] == x["firsts"][0]]
    else:
        pairs = [(x["R"][i], x["R"][j]) for x in d if x["k"] == 0
                 for i in range(len(x["firsts"])) for j in range(i + 1, len(x["firsts"])) if x["firsts"][i] == x["firsts"][j]]
    g5 = len(pairs) > 0 and all(a == b for a, b in pairs)
    with th.no_grad():
        obs = th.as_tensor(v.env.normalize_obs(np.stack(v._side["obs"]).astype(np.float32)))
        _, lp, _ = v.policy.evaluate_actions(obs, th.as_tensor(np.asarray(v._side["actions"])))
    dlp = float(np.abs(lp.numpy() - np.asarray(v._side["old_log_prob"])).max())
    cfg = __import__("stable_baselines3.common.logger", fromlist=["configure"]).configure
    if not hasattr(plain, "_main_after"):            # plain PPO's update on the identical buffer: the main pass alone
        plain._logger = cfg(None, []); plain.train()
        plain._main_after = {k: p.detach().clone() for k, p in plain.policy.state_dict().items()}
    v._logger = cfg(None, [])
    v.train()                                        # the same main pass (same buffer, same global RNG) + the side pass
    moved = max(float((p - plain._main_after[k]).abs().max()) for k, p in v.policy.state_dict().items() if p.is_floating_point())
    logs = {k: v.logger.name_to_value[k] for k in v.logger.name_to_value if k.startswith("vine/")}
    g6 = dlp < 1e-5 and moved > 0 and "vine/approx_kl" in logs
    print(f"[{bl}, siblings {sib}] G3 main rollout == plain PPO's: {g3} | G4 forks {forks}, side samples {n}, pool empty: {g4} | "
          f"G5 equal-first-order copies earn the same ({len(pairs)} pairs): {g5} | G6 max |old lp - lp| {dlp:.1e}, side-pass move beyond the main pass {moved:.2e}: {g6}")
    print(f"    vine logs: " + ", ".join(f"{k.split('/')[1]}={val:.4g}" for k, val in sorted(logs.items())))
    ok &= g3 and g4 and g5 and g6
print("PASS" if ok else "FAIL")

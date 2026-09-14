"""Can (inv, pipe) be compressed to d numbers without losing the policy?

The teacher is distilled on the action it actually DEPLOYS -- its greedy
argmax -- not on its action distribution. That distribution is broad (entropy
2.57 over 136 actions, top-3 holding 0.36), so matching it would spend the
student's capacity reproducing spread that `deterministic=True` throws away at
eval time anyway. The order is a quantity, so the student regresses it.

Distils a trained net (teacher) into a student whose ONLY constraint is an
information bottleneck on the stock vector:

    e_i = inv + w_i0*pipe_0 + ... + w_i,m-1*pipe_{m-1}      i = 1..d
    action ~ MLP(64,64)( [time_to_go, e_1..e_d] )

The inventory coefficient is fixed to 1 in every row, so each embedding is in
inventory units and standard inventory position is the all-ones member of the
family. Everything ABOVE the bottleneck is left free -- a plain categorical
head, wider than the teacher's ordinal head -- so if the student fails, the
failure is attributable to the bottleneck and not to the head.

d = (number of live stock coordinates) is full rank: no compression at all, and
the student must reproduce the teacher. That is the ceiling control.

The last pipeline slot is dropped: it is structurally zero at every decision
epoch (the order enters pipeline[L] at the O event and shifts left before the
next observation), so it carries no information and would only add an
unidentifiable weight.
"""
from __future__ import annotations
import argparse, warnings, numpy as np, torch as th, torch.nn as nn
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import inv_single_ordinal_head  # noqa
from inv_single_scenarios import SCENARIOS
from inv_single_gym import InvSingleEnv
from inv_single_eval_common import eval_seed_block, rollout_seeds


class CtxBottleneck(nn.Module):
    """Generalist form: ONE shared weight vector, a cell-dependent rule on top.

    The observation is [time_to_go, inventory, pipe..., context...]. Only
    (inventory, pipeline) is compressed; the context — the lead-time law and the
    cost fractile — passes through untouched to the MLP. So the policy may order
    differently in every cell while reading the SAME linear statistic of stock.

    That separation is the whole test. #E13 found one weighted inventory
    position suffices at a single cell; the open question is whether one WEIGHT
    VECTOR serves a family, or whether the weights must move with the cell. If
    this student matches its teacher, the rule generalises; if it does not, the
    weights are cell-dependent and the object to distil is the map from cell to
    weights.
    """

    def __init__(self, n_pipe: int, n_ctx: int, d: int, stats, hidden=(64, 64)):
        super().__init__()
        w = th.ones(d, n_pipe)
        if d > 1:
            w = w + 0.30 * th.randn(d, n_pipe)
        self.w = nn.Parameter(w)
        self.n_pipe, self.n_ctx = n_pipe, n_ctx
        for k, v in stats.items():
            self.register_buffer(k, th.as_tensor(v, dtype=th.float32))
        layers, prev = [], d + 1 + n_ctx
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def embed(self, obs):
        inv, pipe = obs[:, 1:2], obs[:, 2:2 + self.n_pipe]
        return inv + pipe @ self.w.T            # inv coefficient FIXED at 1

    def forward(self, obs):
        ctx = obs[:, 2 + self.n_pipe:]
        z = th.cat([(obs[:, :1] - self.t_mu) / self.t_sd,
                    (self.embed(obs) - self.e_mu) / self.e_sd,
                    ctx], dim=-1)
        return self.mlp(z).squeeze(-1) * self.y_sd + self.y_mu


class Bottleneck(nn.Module):
    """[ttg, inv, pipe...] -> d-dim stock embedding -> MLP -> order quantity.

    Standardisation is a FROZEN affine map computed once from the data (with the
    weights at their all-ones init). It is applied to the MLP's inputs and to
    the regression target, so it changes conditioning only -- never what the
    bottleneck can represent, which is the thing under test.
    """

    def __init__(self, n_pipe: int, d: int, stats, hidden=(64, 64)):
        super().__init__()
        w = th.ones(d, n_pipe)                      # init AT plain inventory position
        if d > 1:
            w = w + 0.30 * th.randn(d, n_pipe)      # break row symmetry hard
        self.w = nn.Parameter(w)
        for k, v in stats.items():
            self.register_buffer(k, th.as_tensor(v, dtype=th.float32))
        layers, prev = [], d + 1
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def embed(self, obs):
        inv, pipe = obs[:, 1:2], obs[:, 2:]
        return inv + pipe @ self.w.T                # inv coefficient FIXED at 1

    def forward(self, obs):
        z = th.cat([(obs[:, :1] - self.t_mu) / self.t_sd,
                    (self.embed(obs) - self.e_mu) / self.e_sd], dim=-1)
        return self.mlp(z).squeeze(-1) * self.y_sd + self.y_mu


def teacher_actions(model, obs_raw, normalizer):
    """The teacher's DEPLOYED action: greedy, exactly as eval runs it."""
    obs = normalizer.normalize_obs(obs_raw) if normalizer is not None else obs_raw
    act, _ = model.predict(obs, deterministic=True)
    return np.asarray(act).reshape(-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scenario_name", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("-d", "--dim", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-states", type=int, default=120_000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-seeds", type=int, default=8192)
    ap.add_argument("--freeze-ones", action="store_true",
                    help="hold the weights at all-ones: the student is then an MLP "
                         "on plain INVENTORY POSITION. Identical architecture, data "
                         "and optimiser to the learned-weight run, so the difference "
                         "between the two is the weights and nothing else.")
    a = ap.parse_args()

    th.manual_seed(a.seed); np.random.seed(a.seed)
    sc = SCENARIOS[a.scenario_name]
    mp = Path(a.model_path); vn = mp.parent / "vecnormalize.pkl"
    teacher = PPO.load(str(mp), device="cpu")

    mk = lambda: InvSingleEnv(sc, observation_mode="vec", action_mode="discrete")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        venv = DummyVecEnv([mk for _ in range(64)])
        normalizer = None
        if vn.exists():
            venv = VecNormalize.load(str(vn), venv); venv.training = False
            normalizer = venv

    n_pipe_all = len(SCENARIOS[a.scenario_name].leadtime.max() * [0]) + 1
    n_live = n_pipe_all - 1                     # last slot is structurally zero
    # harvest the teacher's own visited distribution (#E10)
    venv.seed(a.seed); obs = venv.reset(); X = []
    while len(X) * 64 < a.n_states:
        X.append(np.stack([e._get_obs() for e in
                           (venv.venv.envs if normalizer is not None else venv.envs)]))
        act, _ = teacher.predict(obs, deterministic=True)
        obs, _, _, _ = venv.step(act)
    X = np.concatenate(X).astype(np.float32)
    X_live = np.concatenate([X[:, :2], X[:, 2:2 + n_live]], axis=1)
    Y = teacher_actions(teacher, X, normalizer)
    n_actions = int(teacher.action_space.n)

    e0 = X_live[:, 1] + X_live[:, 2:].sum(1)        # the all-ones embedding
    stats = {"t_mu": X_live[:, 0].mean(), "t_sd": X_live[:, 0].std() + 1e-6,
             "e_mu": e0.mean(), "e_sd": e0.std() + 1e-6,
             "y_mu": Y.mean(), "y_sd": Y.std() + 1e-6}
    tag = "d%d%s" % (a.dim, "-ones" if a.freeze_ones else "")
    outdir = Path("results") / a.scenario_name / "distill"
    outdir.mkdir(parents=True, exist_ok=True)
    student = Bottleneck(n_live, a.dim, stats)
    if a.freeze_ones:
        with th.no_grad():
            student.w.fill_(1.0)
        student.w.requires_grad_(False)
    opt = th.optim.Adam(student.parameters(), lr=3e-3)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    Xt, Yt = th.as_tensor(X_live), th.as_tensor(Y)
    n = len(Xt); idx = np.arange(n); BS = 512
    for ep in range(a.epochs):
        np.random.shuffle(idx); tot = 0.0
        for i in range(0, n, BS):
            b = idx[i:i + BS]
            loss = nn.functional.huber_loss(student(Xt[b]), Yt[b], delta=2.0)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        sched.step()
        if ep % 20 == 19:
            print(f"    epoch {ep+1:3d}  huber {tot/n:.4f}")

    with th.no_grad():
        pred = student(Xt)
        mae = float((pred - Yt).abs().mean())
        agree = float((pred.round() == Yt).float().mean())
        W = student.w.detach().numpy()

    # the decisive test: roll the STUDENT itself, errors compound
    def act_fn(o, envs):
        raw = np.stack([e._get_obs() for e in envs]).astype(np.float32)
        raw = np.concatenate([raw[:, :2], raw[:, 2:2 + n_live]], axis=1)
        with th.no_grad():
            q = student(th.as_tensor(raw)).numpy()
        q = np.clip(np.rint(q), 0, n_actions - 1)
        return q.reshape(-1, 1).astype(np.float32)

    per_seed = rollout_seeds(scenario=sc, observation_mode="vec",
                             action_mode="discrete", seeds=eval_seed_block(a.n_seeds),
                             act_fn=act_fn, batch=256, vecnorm_path=None)
    cost = float(np.mean(per_seed["cost_total"]))
    np.save(str(outdir / f"distil_{tag}_s{a.seed}.npy"),
            per_seed["cost_total"])
    # the weights ARE the finding; never let them survive only in a log line
    np.save(str(outdir / f"W_{tag}_s{a.seed}.npy"), W)
    print(f"RESULT {a.scenario_name} {tag} seed={a.seed} "
          f"cost={cost:.2f} MAE={mae:.3f} agree={agree:.3f} "
          f"W={np.array2string(W.ravel(), precision=3, floatmode='fixed', separator=',', max_line_width=10**6)}")


if __name__ == "__main__":
    main()

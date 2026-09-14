"""Does ONE weighted inventory position serve a whole family of cells? (#E20)

The bottleneck is CONTEXT-AWARE: only (inventory, pipeline) is compressed, while
the lead-time law and cost fractile pass through to a free MLP. A student may
therefore order differently in every cell while being forced to read the same
linear statistic of stock — which is exactly the generality question, as opposed
to "does a linear rule fit this one cell" (#E13).

`--per-law` gives the weights one row per lead-time law instead of one shared
row. It strictly CONTAINS the shared model, so it can only fit the teacher
better; the test is whether the better fit is worth anything in cost. That
framing is deliberate: `w(pipe2)` is **unidentifiable** at Var(L)=0, where the
coordinate is identically zero, so no per-cell estimate of it exists to compare.
Pricing the restriction sidesteps estimating it at all.

Do NOT regularise the weights toward (1,1,1) to make the fit well-posed: the
shrinkage tracks sd(pipe_k), which tracks the hypothesis, so it manufactures a
declining alpha(p) out of a constant one (#E20).

Distils the crowned lt_variance_k0 generalist through a bottleneck that
compresses (inventory, pipeline) to d numbers with weights SHARED across every
cell, while the context (lead-time law, cost fractile, K) passes through. The
teacher may order differently per cell either way; what is constrained is
whether it reads the same linear statistic of stock everywhere.
"""
import argparse, warnings, numpy as np, torch as th, torch.nn as nn
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import inv_single_ordinal_head  # noqa
from inv_single_distill_bottleneck import CtxBottleneck


class PerLawBottleneck(CtxBottleneck):
    """Same student, but the weights may DIFFER per lead-time law.

    5 rows of 3 instead of 1 row of 3. The row is selected by the law the
    context already carries, so nothing else changes: same MLP, same training,
    same context passthrough. The only question this asks is whether ALLOWING
    the weights to move with the law buys anything in cost.

    It strictly contains the shared model, so it can only fit the teacher
    better; the test is whether the better fit is worth anything.
    """

    LAWS = th.tensor([0.0, 1/6, 1/4, 1/3, 1/2])

    def __init__(self, n_pipe, n_ctx, d, stats, hidden=(64, 64)):
        super().__init__(n_pipe, n_ctx, d, stats, hidden)
        assert d == 1, "per-law weights are defined for the scalar bottleneck"
        self.w = nn.Parameter(th.ones(len(self.LAWS), n_pipe))

    def _rows(self, obs):
        # the context starts with the law [p, 1-2p, p]; p is its first entry
        p_of_cell = obs[:, 2 + self.n_pipe]
        j = (p_of_cell[:, None] - self.LAWS.to(obs.device)[None, :]).abs().argmin(1)
        return self.w[j]                      # (B, n_pipe)

    def embed(self, obs):
        inv, pipe = obs[:, 1:2], obs[:, 2:2 + self.n_pipe]
        return inv + (pipe * self._rows(obs)).sum(1, keepdim=True)

from inv_single_grids import GRIDS
from inv_single_gym import InvSingleEnv
from inv_single_eval_common import eval_seed_block, rollout_seeds

ap = argparse.ArgumentParser()
ap.add_argument("--model-path", required=True)
ap.add_argument("-g", "--grid_name", default="lt_variance_k0")
ap.add_argument("-o", "--observation_mode", default="vec_ctx_slt")
ap.add_argument("-d", "--dim", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-states", type=int, default=200_000)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--n-seeds", type=int, default=512)
ap.add_argument("--per-law", action="store_true",
                help="one weight vector PER lead-time law (15 params) "
                     "instead of one shared vector (3)")
ap.add_argument("--freeze-ones", action="store_true",
                help="hold w at all-ones: plain inventory position, "
                     "the null this family is measured against")
a = ap.parse_args()
th.manual_seed(a.seed); np.random.seed(a.seed)

grid = GRIDS[a.grid_name]
mp = Path(a.model_path); vn = mp.parent / "vecnormalize.pkl"
teacher = PPO.load(str(mp), device="cpu")
sampler = grid.as_sampler()
mk = lambda: InvSingleEnv(sampler, observation_mode=a.observation_mode, action_mode="discrete")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    venv = DummyVecEnv([mk for _ in range(64)])
    nz = None
    if vn.exists():
        venv = VecNormalize.load(str(vn), venv); venv.training = False; nz = venv
raw = venv.venv.envs if nz is not None else venv.envs

n_pipe_all = grid.cells[0][1].leadtime.max() + 1
n_live = n_pipe_all - 1                      # last slot is structurally zero
venv.seed(a.seed); obs = venv.reset(); X = []
while len(X) * 64 < a.n_states:
    X.append(np.stack([e._get_obs() for e in raw]))
    act, _ = teacher.predict(obs, deterministic=True); obs, _, _, _ = venv.step(act)
X = np.concatenate(X).astype(np.float32)
# drop the dead pipeline slot; keep [ttg, inv, pipe_live..., ctx...]
X = np.concatenate([X[:, :2], X[:, 2:2 + n_live], X[:, 2 + n_pipe_all:]], axis=1)
n_ctx = X.shape[1] - 2 - n_live

def teacher_act(raw_obs):
    o = nz.normalize_obs(raw_obs) if nz is not None else raw_obs
    act, _ = teacher.predict(o, deterministic=True)
    return np.asarray(act).reshape(-1).astype(np.float32)

Xfull = []
venv.seed(a.seed); obs = venv.reset()
Y = []
for _ in range(len(X) // 64):
    r = np.stack([e._get_obs() for e in raw]).astype(np.float32)
    Xfull.append(r); Y.append(teacher_act(r))
    act, _ = teacher.predict(obs, deterministic=True); obs, _, _, _ = venv.step(act)
Xfull = np.concatenate(Xfull); Y = np.concatenate(Y)
Xs = np.concatenate([Xfull[:, :2], Xfull[:, 2:2 + n_live], Xfull[:, 2 + n_pipe_all:]], axis=1)

e0 = Xs[:, 1] + Xs[:, 2:2 + n_live].sum(1)
stats = {"t_mu": Xs[:, 0].mean(), "t_sd": Xs[:, 0].std() + 1e-6,
         "e_mu": e0.mean(), "e_sd": e0.std() + 1e-6,
         "y_mu": Y.mean(), "y_sd": Y.std() + 1e-6}
student = (PerLawBottleneck if a.per_law else CtxBottleneck)(
    n_live, n_ctx, a.dim, stats)
if a.freeze_ones:
    with th.no_grad(): student.w.fill_(1.0)
    student.w.requires_grad_(False)
opt = th.optim.Adam(student.parameters(), lr=3e-3)
sch = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
Xt, Yt = th.as_tensor(Xs), th.as_tensor(Y)
idx = np.arange(len(Xt))
for ep in range(a.epochs):
    np.random.shuffle(idx)
    for i in range(0, len(idx), 512):
        b = idx[i:i + 512]
        loss = nn.functional.huber_loss(student(Xt[b]), Yt[b], delta=2.0)
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
with th.no_grad():
    mae = float((student(Xt) - Yt).abs().mean())
    W = student.w.detach().numpy()

n_actions = int(teacher.action_space.n)
def act_fn(o, envs):
    r = np.stack([e._get_obs() for e in envs]).astype(np.float32)
    r = np.concatenate([r[:, :2], r[:, 2:2 + n_live], r[:, 2 + n_pipe_all:]], axis=1)
    with th.no_grad():
        q = student(th.as_tensor(r)).numpy()
    return np.clip(np.rint(q), 0, n_actions - 1).reshape(-1, 1).astype(np.float32)

means = []
for cid, sc in grid.cells:
    r = rollout_seeds(scenario=sc, observation_mode=a.observation_mode,
                      action_mode="discrete", seeds=eval_seed_block(a.n_seeds),
                      act_fn=act_fn, batch=256, vecnorm_path=None, progress_every=0)
    means.append(float(np.mean(r["cost_total"])))
tag = f"d{a.dim}" + ("-ones" if a.freeze_ones else "") + ("-perlaw" if a.per_law else "")
print(f"RESULT {tag} seed={a.seed} mean_over_cells={np.mean(means):.2f} "
      f"MAE={mae:.3f} W={np.array2string(W.ravel(), precision=3, separator=',')}")
np.save(f"scratch/genW_{tag}_s{a.seed}.npy", W)
np.save(f"scratch/genC_{tag}_s{a.seed}.npy", np.array(means))

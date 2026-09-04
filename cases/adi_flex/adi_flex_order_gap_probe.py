"""Order-gap diagnosis probe for het_exp4 — the instrument behind #E18.

Sibling of `adi_flex_fixed_order.py` (the #E13 instrument), finer-grained:
where that wrapper fixes the whole order stream, this one composes each
decision per PERIOD from either source, logs the two policies' orders at the
same states, and returns PER-SEED cost vectors so CRN-paired differences can
be read. Batched (one model forward per period over all seeds), so a full
8192-seed arm runs in ~1 min.

Replays policies through AdiFlexEnv(action_mode="order_protection") on the
protocol CRN seed block, with the order and the protection level each taken
from a chosen source:

    order source:  rl (model head 0) | ap (the y* table, as the PL arms play it)
                   | hybrid (ap on --ap-periods, rl elsewhere)
    sigma source:  rl (model head 1) | const (--sigma-const)

At T_dl = 2 the cascade with a constant sigma IS PL(sigma) exactly (the mode's
acceptance test), so (order=ap, sigma=const 4) reproduces the bar and
(order=rl, sigma=rl) reproduces the shipped RL number — both are calibration
gates before any hybrid is read.

Per-step logging (--log-steps) records, at every order-phase state visited:
period, u, vhat, the RL order, the AP order at the same state, the sigma
played, and the order-head distribution stats (entropy, P(argmax), P(order=0)).

Outputs an .npz per arm: per-seed costs + optional step logs.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# importable from anywhere: the probe lives in the domain folder and imports
# its siblings (file-hygiene rule (b) — a finding's numbers must be re-runnable)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

torch.set_num_threads(1)
# same Simplex-validation trap as the train/eval scripts (#E3)
torch.distributions.Distribution.set_default_validate_args(False)

from adi_flex_gym import AdiFlexEnv  # noqa: E402
from adi_flex_scenarios import SCENARIOS  # noqa: E402

# default: the solved AP table beside the benchmark outputs; override with
# --ap-table when results/ lives elsewhere (it is gitignored and symlinked in
# some worktrees)
AP_NPZ = str(Path(__file__).resolve().parent
             / "results/het_exp4/ap/het_exp4_policy.npz")


def load_ap(path: str):
    z = np.load(path)
    y = z["ystar"] if "ystar" in z.files else z[z.files[0]]
    u_min = int(z["u_min"]) if "u_min" in z.files else -150
    d_max = int(z["d_max"]) if "d_max" in z.files else y.shape[-1] - 1
    return y, u_min, d_max


def ap_order_of(env: AdiFlexEnv, y: np.ndarray, u_min: int, d_max: int):
    """Mirror of FixedOrderProtectEnv._ap_order / benchmark_rule's lookup."""
    s = env._state
    sc = env._scenario_ep
    u = int(s.inv + sum(s.pipe) - sum(s.adv))
    d = (env._info or {}).get("d", ())
    vhat = int(d[sc.T_dl]) if len(d) > sc.T_dl else 0
    ui = int(np.clip(u - u_min, 0, y.shape[1] - 1))
    vi = int(np.clip(vhat, 0, d_max))
    yv = int(y[s.period, ui, vi])
    return int(np.clip(yv - u, 0, sc.order_max)), u, vhat


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--order-src", choices=["rl", "ap", "hybrid"], required=True)
    p.add_argument("--sigma-src", choices=["rl", "const"], required=True)
    p.add_argument("--sigma-const", type=int, default=4)
    p.add_argument("--sigma-zero-periods", type=str, default="",
                   help="comma list of periods where sigma is forced to 0 "
                        "(overrides the sigma source there)")
    p.add_argument("--ap-periods", type=str, default="",
                   help="comma list of periods that play the AP order (hybrid)")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--ap-table", type=str, default=AP_NPZ)
    p.add_argument("--vecnorm-path", type=str, default=None)
    p.add_argument("--obs-mode", type=str, default="vec_mip")
    p.add_argument("--stochastic", action="store_true",
                   help="sample the policy instead of deterministic argmax")
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=8192)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--log-steps", action="store_true")
    p.add_argument("--out", type=str, required=True)
    a = p.parse_args()

    sc = SCENARIOS["het_exp4"]
    n, first = a.n_seeds, a.first_seed
    N = sc.N
    y, u_min, d_max = load_ap(a.ap_table)
    ap_periods = (set(int(x) for x in a.ap_periods.split(",") if x != "")
                  if a.order_src == "hybrid" else set())
    sigma_zero = set(int(x) for x in a.sigma_zero_periods.split(",") if x != "")

    needs_model = a.order_src in ("rl", "hybrid") or a.sigma_src == "rl" \
        or a.log_steps
    model = vn = None
    if needs_model:
        from sb3_contrib.ppo_mask import MaskablePPO
        assert a.model_path, "this arm needs --model-path"
        model = MaskablePPO.load(a.model_path, device="cpu")
        vp = Path(a.vecnorm_path) if a.vecnorm_path else \
            Path(a.model_path).parent.parent / "vecnormalize.pkl"
        if not vp.exists():
            vp = Path(a.model_path).parent / "vecnormalize.pkl"
        if vp.exists():
            with open(vp, "rb") as fh:
                vn = pickle.load(fh)
        torch.manual_seed(a.torch_seed)

    envs = []
    for i in range(n):
        e = AdiFlexEnv(scenario=sc, observation_mode=a.obs_mode,
                       action_mode="order_protection")
        o, _ = e.reset(seed=first + i)
        envs.append(e)
    obs = np.stack([e._get_obs() for e in envs]).astype(np.float32)

    cost = np.zeros(n)
    logs = {k: np.zeros((N, n), dtype=np.float32)
            for k in ("u", "vhat", "rl_order", "ap_order", "order", "sigma",
                      "ent_order", "p_top", "p_zero", "ent_sigma")} \
        if a.log_steps else None

    t0 = time.time()
    for t in range(N):
        rl_actions = None
        if needs_model:
            ob = vn.normalize_obs(obs) if vn is not None else obs
            rl_actions, _ = model.predict(ob, deterministic=not a.stochastic)
            rl_actions = np.asarray(rl_actions).reshape(n, -1)
            if a.log_steps:
                with torch.no_grad():
                    dist = model.policy.get_distribution(
                        torch.as_tensor(ob))
                heads = getattr(dist, "distributions", None) \
                    or dist.distribution  # list of Categorical, per head
                po = heads[0].probs.numpy()
                ps = heads[1].probs.numpy()
                logs["ent_order"][t] = -(po * np.log(po + 1e-12)).sum(1)
                logs["p_top"][t] = po.max(1)
                logs["p_zero"][t] = po[:, 0]
                logs["ent_sigma"][t] = -(ps * np.log(ps + 1e-12)).sum(1)

        for i, e in enumerate(envs):
            apo, u, vhat = ap_order_of(e, y, u_min, d_max)
            rlo = int(rl_actions[i, 0]) if rl_actions is not None else -1
            if a.order_src == "rl":
                order = rlo
            elif a.order_src == "ap":
                order = apo
            else:
                order = apo if t in ap_periods else rlo
            sigma = (int(rl_actions[i, 1]) if a.sigma_src == "rl"
                     else a.sigma_const)
            if t in sigma_zero:
                sigma = 0
            if logs is not None:
                logs["u"][t, i] = u
                logs["vhat"][t, i] = vhat
                logs["rl_order"][t, i] = rlo
                logs["ap_order"][t, i] = apo
                logs["order"][t, i] = order
                logs["sigma"][t, i] = sigma
            o, r, term, trunc, _ = e.step([order, sigma])
            cost[i] -= r
            if not (term or trunc):
                obs[i] = o
        if t % 3 == 2:
            print(f"  period {t + 1}/{N}  {time.time() - t0:.0f}s", flush=True)

    mu = cost.mean()
    se = cost.std(ddof=1) / np.sqrt(n)
    print(f"arm order={a.order_src} sigma={a.sigma_src}"
          f"{'/stoch' if a.stochastic else ''} ap_periods={sorted(ap_periods)}"
          f"  n={n} first={first}")
    print(f"  mean cost {mu:.6f}  +-SE {se:.4f}  var {cost.var(ddof=1):.2f}")
    out = {"cost": cost,
           "meta": np.array([a.order_src, a.sigma_src, str(a.sigma_const),
                             a.ap_periods, str(a.model_path),
                             str(a.stochastic), str(n), str(first)])}
    if logs is not None:
        out.update(logs)
    np.savez_compressed(a.out, **out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()

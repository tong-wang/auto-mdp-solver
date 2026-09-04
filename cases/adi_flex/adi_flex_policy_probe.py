"""Spec §14 readback: what protection levels does the crowned policy set?

`order_protection` makes sigma the ACTION, so the §14.0 `confirm` stance on
"the learned allocation is PL-shaped" is satisfied BY CONSTRUCTION and
demonstrates nothing — F7 recorded exactly that trap when it removed sigma as
the decision. This probe therefore targets the `discover` stance: a PL policy
holds sigma CONSTANT, so the question is whether the crowned net's sigma varies
with state, and if so on what.

Three readouts, per §14.1:

1. the raw action surface, over the ON-POLICY state distribution (not a
   synthetic grid: a grid spends its resolution on states the policy never
   visits, and the question is what the shipped policy does);
2. a structural-form statistic — the spread of each sigma component. A PL
   policy scores 0. Anything else is the size of the departure, as a number;
3. feature-sensitivity sweeps — hold a reference state, move ONE observation
   feature across its observed range, report how far sigma moves. This is the
   mechanism check and it is what answers "if state dependent, on what".

INSTRUMENT VALIDATION (§14.1, "validate the instrument where the answer is
known"): `--validate` runs every readout against a stub policy whose sigma is a
known constant. The probe must report spread 0 and zero sensitivity on every
feature. A probe that has not recovered a known answer is not measuring yet.

    python adi_flex_policy_probe.py --validate
    python adi_flex_policy_probe.py --config sc4/g6/a0/h4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adi_flex_ppo_eval import build_env
from adi_flex_scenarios import SCENARIOS

HERE = Path(__file__).resolve().parent

# The artifacts a readback may address, by config id (§CONFIG-REGISTRY).
#   (scenario, obs_mode, location, model)
# Two location shapes. A tuning trial dir (model None): §9.7 ships the trial's
# own terminal model, found under <location>/<scenario>/PPO_*/. A run dir glob
# (model given): the shipped checkpoint the three-layer select named in that
# run's select.tsv — the 6-seed cells of #E24/#E25 ship a specific checkpoint,
# not necessarily the terminal one.
# a0 = the categorical order head (#E18's crowns, retired by #E25);
# a1 = the ordinal order head (#E24), the crown since 2026-09-04.
CROWNED = {
    "sc4/g6/a0/h4": ("het3_exp2", "vec",
                     "results/tuning/s4_prot_vec_obsF/trial_0185", None),
    "sc4/g7/a0/h5": ("het3_exp2", "vec_mip",
                     "results/tuning/s4_prot_vecmip_obsF/trial_0321", None),
    "sc4/g6/a1/h4": ("het3_exp2", "vec",
                     "results/het3_exp2/PPO_*_obsvec_actorder_protection_*_seed21_*_ordheadordinal",
                     "het3_exp2_ppo_final.zip"),
    "sc4/g7/a1/h5": ("het3_exp2", "vec_mip",
                     "results/het3_exp2/PPO_*_obsvec_mip_actorder_protection_*_seed25_*_ordheadordinal",
                     "het3_exp2_ppo_final.zip"),
}
CROWN = "sc4/g6/a1/h4"          # what ships (#E25); the default everywhere below
# Observation layouts for `order_protection` (one-shot, so no within-period
# block). Widths are per-instance: `pipe` is L wide, `adv` is T_dl, and the
# `vec_mip` tail is max(0, T_dl - L - 1) — so these are derived, never listed,
# or a name silently slides onto the wrong axis (it did: `vec_mip` on
# het3_exp2 is 3 features and a hand-written 5-name list labelled
# `time_to_go` as `adv1`).
def feature_names(sc, obs_mode: str) -> list[str]:
    if obs_mode == "vec":
        return (["inv"] + [f"pipe{i}" for i in range(sc.L)]
                + [f"adv{i}" for i in range(sc.T_dl)] + ["time_to_go"])
    tail = max(0, sc.T_dl - sc.L - 1)
    base = "mip" if obs_mode == "vec_mip" else obs_mode
    return [base] + [f"vtail{i}" for i in range(tail)] + ["time_to_go"]


def _resolve(cfg: str) -> tuple[str, str, Path, Path]:
    scen, obs, base, model = CROWNED[cfg]
    if model is None:                       # a tuning trial: its terminal model
        d = sorted((HERE / base / scen).glob("PPO_*"))[0]
        return scen, obs, d / f"{scen}_ppo_final.zip", d / "vecnormalize.pkl"
    ds = sorted(HERE.glob(base))            # a run dir: the shipped checkpoint
    if len(ds) != 1:
        raise FileNotFoundError(f"{cfg}: {len(ds)} run dirs match {base!r}")
    return scen, obs, ds[0] / model, ds[0] / "vecnormalize.pkl"


class _ConstSigma:
    """The known answer the instrument is validated against."""

    def __init__(self, sigma: int, order: int = 8):
        self.sigma, self.order = sigma, order

    def predict(self, obs, deterministic=True, action_masks=None):
        n = len(obs) if obs.ndim > 1 else 1
        return np.tile([self.order, self.sigma, self.sigma], (n, 1)), None


def _load(model_path: Path, vecnorm_path: Path):
    from sb3_contrib.ppo_mask import MaskablePPO
    import adi_flex_ordinal_head  # noqa: F401 — an a1 artifact's policy class resolves through it
    return MaskablePPO.load(str(model_path), device="cpu")


def collect(env, policy, n_episodes: int, seed0: int = 0):
    """On-policy states and the actions taken at them."""
    S, A = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            a, _ = policy.predict(np.asarray(obs)[None, :], deterministic=True)
            a = np.asarray(a).reshape(-1)
            S.append(np.asarray(obs, dtype=float).copy())
            A.append(a.copy())
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
    return np.array(S), np.array(A)


def sensitivity(env, policy, ref_state, feat_idx, lo, hi, steps=25):
    """Sweep ONE feature; report how far each sigma moves. The env is not
    stepped — the policy is queried directly, so this measures the NETWORK's
    dependence, not a dynamics-mediated correlation."""
    out = []
    for v in np.linspace(lo, hi, steps):
        o = ref_state.copy()
        o[feat_idx] = v
        a, _ = policy.predict(o[None, :], deterministic=True)
        out.append(np.asarray(a).reshape(-1))
    return np.linspace(lo, hi, steps), np.array(out)


def _identify(cfg: str, episodes: int) -> None:
    """Which protection components actually change the allocation?

    A sigma box only bites when stock REACHES it. The cascade pours `rem`
    through `sigma_1, adv[1], sigma_2, adv[2]`; once `rem` is spent, sigma has
    no effect and the network is free to emit anything — so those events carry
    an action with no consequence. Test it exactly rather than by proxy: replay
    each recorded allocation with the component forced to 0 and to `sigma_max`
    and see whether the allocation moves.

    This must run BEFORE any structural reading. Skipping it is what produced
    §4c's retraction: the "scarcity regime" that looked like the headline
    finding is 0.0% identified on sigma_2.
    """
    scen, obs, mp, vp = _resolve(cfg)
    sc = SCENARIOS[scen]
    env = build_env(sc, obs, "order_protection")
    pol = _load(mp, vp)
    casc, rec, pend = env._sigma_cascade, [], {}

    def spy(mid, sigmas):
        rec.append((mid, list(sigmas), pend.get("u")))
        return casc(mid, sigmas)

    env._sigma_cascade = spy
    for ep in range(episodes):
        o, _ = env.reset(seed=ep)
        done = False
        while not done:
            st = env._state
            pend["u"] = int(st.inv + sum(st.pipe) - sum(st.adv))
            act, _ = pol.predict(np.asarray(o)[None, :], deterministic=True)
            o, _, t, tr, _ = env.step(np.asarray(act).reshape(-1))
            done = t or tr
    env._sigma_cascade = casc

    U = np.array([r[2] for r in rec], dtype=float)
    S = np.array([r[1] for r in rec], dtype=float)
    ID = np.zeros((len(rec), sc.n_sigma), dtype=bool)
    for i, (mid, sg, _) in enumerate(rec):
        base = casc(mid, sg)
        for k in range(sc.n_sigma):
            lo, hi = list(sg), list(sg)
            lo[k], hi[k] = 0, sc.sigma_max
            ID[i, k] = casc(mid, lo) != base or casc(mid, hi) != base

    print(f"\n=== 0. IDENTIFICATION AUDIT — {len(rec)} allocation events ===")
    print("  an action with no effect on the allocation carries no information")
    for k in range(sc.n_sigma):
        m = ID[:, k]
        print(f"  sigma_{k+1}: identified {m.mean():6.1%}   "
              f"mean {S[:, k].mean():5.2f} overall, "
              f"{S[m, k].mean():5.2f} where it acts, "
              f"{S[~m, k].mean():5.2f} where it does not")
    print(f"\n  {'u regime':<16}{'n':>6}" +
          "".join(f"{f'ident σ_{k+1}':>12}" for k in range(sc.n_sigma)))
    for lab, m in (("u < 0", U < 0), ("0 <= u < 5", (U >= 0) & (U < 5)),
                   ("u >= 5", U >= 5)):
        if m.sum() < 20:
            continue
        print(f"  {lab:<16}{m.sum():>6}" +
              "".join(f"{ID[m, k].mean():>12.1%}" for k in range(sc.n_sigma)))


def _decompose(S, A, names) -> None:
    """Open up `u = inv + sum(pipe) - sum(adv)`: is it sufficient for sigma?

    Two questions that answer differently, which is the point (§14.1: a feature
    can be score-neutral while the net demonstrably latches onto it):

      MECHANISM  does the network's sigma move with the profile at fixed u?
      VARIANCE   does the profile explain any sigma variation the policy
                 actually EXPERIENCES?

    Uses on-policy states only. The synthetic iso-u grid that this replaced
    reported far larger composition effects, because 78% of its cells put at
    least one adv component outside the visited range and the argmax surface
    is ragged out there.
    """
    from collections import Counter
    i = {n: k for k, n in enumerate(names)}
    inv, ttg = S[:, i["inv"]], S[:, i["time_to_go"]]
    pipe = S[:, [i[n] for n in names if n.startswith("pipe")]].sum(axis=1)
    advc = [i[n] for n in names if n.startswith("adv")]
    adv = S[:, advc]
    ip = inv + pipe
    u = ip - adv.sum(axis=1)
    sig = A[:, 1:].astype(float)

    def r2(y, X):
        cols = [np.asarray(c, dtype=float).ravel()
                for c in (X if isinstance(X, list) else [X])]
        M = np.column_stack(cols + [np.ones(len(y))])
        b, *_ = np.linalg.lstsq(M, y, rcond=None)
        return 1 - ((y - M @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def resid_on_u(y):                       # remove u NONPARAMETRICALLY
        out = y.copy()
        for lo in range(int(u.min()) - 1, int(u.max()) + 2, 2):
            m = (u >= lo) & (u < lo + 2)
            if m.sum() >= 10:
                out[m] = y[m] - y[m].mean()
        return out

    print("\n=== 4. opening up the MIP: is u sufficient for sigma? ===")
    prof = Counter(map(tuple, adv.astype(int)))
    top = prof.most_common(1)[0]
    print(f"  the profile is SPARSE on-policy: {top[0]} is "
          f"{top[1]/len(S):.1%} of states, {len(prof)} distinct profiles seen")
    print(f"\n  residual variance explained AFTER u is removed nonparametrically:")
    print(f"    {'component':<24}" + "".join(f"{f'sigma_{k+1}':>10}"
                                             for k in range(sig.shape[1])))
    for nm, v in ([("ip = inv + sum(pipe)", ip)]
                  + [(f"adv{k}", adv[:, k]) for k in range(adv.shape[1])]
                  + [("sum(adv)", adv.sum(axis=1)), ("time_to_go", ttg)]):
        print(f"    {nm:<24}" + "".join(
            f"{r2(resid_on_u(sig[:, k]), v):>10.3f}" for k in range(sig.shape[1])))
    print(f"\n  nested linear fits — any gain over 'u alone' IS composition,")
    print(f"  since u = ip - sum(adv) makes the last row a reparametrization:")
    print(f"    {'model':<34}" + "".join(f"{f'sigma_{k+1}':>10}"
                                         for k in range(sig.shape[1])))
    for nm, X in [("u alone", [u]), ("u + ip", [u, ip]),
                  ("ip + every adv component", [ip] + [adv[:, k] for k in
                                                       range(adv.shape[1])]),
                  ("  + time_to_go", [ip] + [adv[:, k] for k in
                                             range(adv.shape[1])] + [ttg])]:
        print(f"    {nm:<34}" + "".join(f"{r2(sig[:, k], X):>10.3f}"
                                        for k in range(sig.shape[1])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=CROWN, choices=list(CROWNED))
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--validate", action="store_true",
                   help="run every readout against a known-constant-sigma stub")
    p.add_argument("--dump", default=None, help="write the raw surface as JSON")
    p.add_argument("--identified", action="store_true",
                   help="audit which sigma components actually AFFECT the "
                        "allocation, and re-read the statistics on those events "
                        "only. Run this FIRST: an action with no consequence "
                        "carries no information, and averaging it in reads "
                        "noise as structure (§4c)")
    p.add_argument("--decompose", action="store_true",
                   help="open up the MIP: does sigma depend on the profile's "
                        "COMPOSITION beyond u? (`vec` only — the mip arms do "
                        "not observe adv0/adv1 separately)")
    a = p.parse_args()

    scen, obs_mode, model_path, vecnorm_path = _resolve(a.config)
    sc = SCENARIOS[scen]
    env = build_env(sc, obs_mode, "order_protection")
    names = feature_names(sc, obs_mode)

    if a.validate:
        policy, label = _ConstSigma(sigma=6), "STUB const sigma=6"
    else:
        policy, label = _load(model_path, vecnorm_path), a.config

    S, A = collect(env, policy, a.episodes)
    sig = A[:, 1:]                       # (n, n_sigma)
    print(f"probe: {label}   {scen} / {obs_mode} / order_protection")
    print(f"  {len(S)} on-policy states from {a.episodes} episodes"
          f"   sigma_max={sc.sigma_max}\n")

    print("=== 1. structural-form statistic: is sigma CONSTANT? (PL scores 0) ===")
    print(f"  {'component':<12}{'mean':>8}{'sd':>8}{'min':>6}{'max':>6}"
          f"{'IQR':>6}  {'modal share':>12}")
    for k in range(sig.shape[1]):
        c = sig[:, k]
        q1, q3 = np.percentile(c, [25, 75])
        mode = np.bincount(c.astype(int)).argmax()
        print(f"  sigma_{k+1:<6}{c.mean():>8.2f}{c.std():>8.2f}{c.min():>6.0f}"
              f"{c.max():>6.0f}{q3-q1:>6.0f}  {(c == mode).mean():>11.1%} at {mode}")

    print("\n=== 2. feature sensitivity: WHICH features move sigma? ===")
    ref = np.median(S, axis=0)
    print(f"  reference state (median): "
          + ", ".join(f"{n}={v:g}" for n, v in zip(names, ref)))
    print(f"  {'feature':<12}{'range swept':>16}{'d sigma_1':>11}{'d sigma_2':>11}")
    sens = {}
    for i, nm in enumerate(names):
        lo, hi = S[:, i].min(), S[:, i].max()
        if hi <= lo:
            continue
        _, out = sensitivity(env, policy, ref, i, lo, hi)
        d1, d2 = np.ptp(out[:, 1]), np.ptp(out[:, 2])
        sens[nm] = (float(d1), float(d2))
        print(f"  {nm:<12}{f'[{lo:g}, {hi:g}]':>16}{d1:>11.0f}{d2:>11.0f}")

    print("\n=== 3. sigma against time_to_go (the #E14 pattern) ===")
    t = S[:, names.index("time_to_go")]
    print(f"  {'t_to_go':>8}{'n':>7}{'sigma_1':>10}{'sigma_2':>10}")
    for v in sorted(set(t.astype(int)), reverse=True):
        m = t.astype(int) == v
        if m.sum() < 5:
            continue
        print(f"  {v:>8}{m.sum():>7}{sig[m, 0].mean():>10.2f}{sig[m, 1].mean():>10.2f}")

    if a.validate:
        ok = (sig.std(axis=0).max() == 0
              and all(max(v) == 0 for v in sens.values()))
        print(f"\nINSTRUMENT VALIDATION: {'PASS' if ok else 'FAIL'} — "
              f"a constant-sigma policy must read back with zero spread and "
              f"zero sensitivity on every feature")
        raise SystemExit(0 if ok else 1)

    if a.identified:
        _identify(a.config, a.episodes)

    if a.decompose:
        if "inv" not in names:
            print("\n=== 4. opening up the MIP: skipped — this arm is handed u directly "
                  "(vec_mip); the decomposition is the vec arm's instrument ===")
        else:
            _decompose(S, A, names)

    if a.dump:
        Path(a.dump).write_text(json.dumps(
            {"config": a.config, "features": names,
             "states": S.tolist(), "actions": A.tolist()}))
        print(f"\nsurface -> {a.dump}")


if __name__ == "__main__":
    main()

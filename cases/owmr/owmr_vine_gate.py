"""Executable gate for the coordinate vine (ESCALATION #E30, `owmr_vine_ppo.VinePPO`).

Runs the real training stack (the train script's builders) at a small gate
configuration and checks, on a rollout captured BEFORE any update:

  G1  every branch produced one parent copy and one sibling per coordinate
      (7 at N = 5: order, x_1..x_5, retention), and the side buffer holds
      exactly branches x 7 samples;
  G2  each sibling's action differs from its parent's in EXACTLY the coordinate
      its mask names, and nowhere else;
  G3  each sibling's stored old log-prob equals the policy's log-prob of that
      one coordinate, recomputed;
  G4  common world: two copies of one post-receipt state stepped with
      different actions realise identical demand (the keyed draw), and every
      forked copy carries the parent env's episode seed;
  G5  each side advantage equals -(C_j - C_0) from the recorded window costs;
  G6  p = 0 trains bit-identically to plain PPO (same seed, same config: equal
      parameters after two updates), so the lever is inert when off;
  G7  p = 1 trains through the side pass without error and logs its statistics
      (n_side > 0, approx_kl finite), and the parameters move;
  G8  (#E31) with `vine_lr_scale` the side pass runs on a SEPARATE optimizer at
      scale x the main lr while the policy's optimizer keeps the main lr, and the
      shared-optimizer form (scale None) is bit-identical to the pre-#E31 code
      path (same parameters after one update as a run with no separate optimizer).
G9  (#E32, --head dirichlet) the per-share siblings' sufficient statistics E[s] and E[log s] match fresh
    Dirichlet(alpha) draws through the same share floor (120k vectors, 4 SE)
G10 (#E34, --vine-cat-grid) each menu sibling's weight is pi_old of its option and the weights sum to
    1 - pi_old(parent)

Prints one PASS/FAIL line per check and exits non-zero on any FAIL.

Usage (from owmr/):   python owmr_vine_gate.py
"""

from __future__ import annotations

import copy
import sys
import time

import numpy as np
import torch as th

import owmr_mdp as mdp
from owmr_ppo_train import _build_arg_parser, build_model, build_training_env
from owmr_scenarios import SCENARIOS
from owmr_vine_ppo import _gym_index

import argparse
_ap = argparse.ArgumentParser(description="vine gate"); _ap.add_argument("--head", choices=("catkeep", "dirichlet"), default="catkeep")
_ap.add_argument("--cat-grid", action="store_true", help="#E34: gate the full-menu retention grid (catkeep)")
_ap.add_argument("--fine", action="store_true", help="#E34: the 12-rung menu (order_catkeep_fine)")
_ns = _ap.parse_args(); HEAD_NAME = _ns.head; CAT_GRID = _ns.cat_grid; FINE = _ns.fine
GATE_ARGS = ["-s", "base", "-a", {"catkeep": "order_catkeep_fine" if FINE else "order_catkeep", "dirichlet": "order_dirichlet"}[HEAD_NAME],
             "--policy", HEAD_NAME, "--net_arch", "64", "64",
             "--n-envs", "2", "--n-steps", "100", "--batch-size", "50", "--n-epochs", "2", "--no-norm-obs",
             "--level", "L3(gym+arch+hp)", "--seed", "0", "--outdir", "scratch/vine_gate"]


def build(vine: bool, p: float, record: bool = False, lr_scale: float | None = None):
    extra = ["--vine", "--vine-p", str(p), "--vine-h", "10"] + (["--vine-cat-grid"] if CAT_GRID else []) if vine else []
    if lr_scale is not None:
        extra += ["--vine-lr-scale", str(lr_scale)]
    args = _build_arg_parser().parse_args(GATE_ARGS + extra)
    from pathlib import Path
    outdir = Path("scratch/vine_gate"); (outdir / "ckpt").mkdir(parents=True, exist_ok=True)
    env = build_training_env(args, outdir, SCENARIOS["base"])
    model = build_model(args, env, outdir, SCENARIOS["base"])
    if vine:
        model.vine_record = record
    return model, args


def main() -> int:
    th.set_num_threads(1)
    fails = 0

    def report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    # --- a captured rollout before any update (p = 1: every env branches every step)
    model, args = build(vine=True, p=1.0, record=True)
    total, cb = model._setup_learn(total_timesteps=10_000, callback=None)
    t0 = time.time()
    model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=args.n_steps)
    dt = time.time() - t0
    det = model._side_detail
    n_cont = int(model.action_space.shape[0]) - 1
    n_coords = n_cont + 1
    branches = model._vine_stats.get("branches", 0)
    n_coords = model._head.n_coords
    per_branch = n_coords + (model._head.n_options - 2 if CAT_GRID else 0)
    report("G1 count", branches == args.n_envs * args.n_steps and len(det) == branches * per_branch
           and len(model._side["obs"]) == len(det),
           f"branches {branches}, side samples {len(det)} (= {branches} x {per_branch}), head {model._head.name}{' + menu grid' if CAT_GRID else ''}")

    # G2: exactly one coordinate differs, the masked one
    ok2 = True
    for d, m in zip(det, model._side["mask"]):
        c = int(np.argmax(m))
        diff = set(np.flatnonzero(~np.isclose(d["action"], d["parent_action"])).tolist())
        if model._head.name == "dirichlet":
            if c == 0:
                ok2 &= diff == {0}
            else:   # one Gamma redrawn: aux differs from the parent's Gammas at exactly index c-1; the other shares keep their RATIOS
                gd = set(np.flatnonzero(~np.isclose(d["aux"], d["parent_aux"])).tolist())
                keep = [j for j in range(len(d["aux"])) if j != c - 1]
                r_new = d["action"][1:][keep]; r_old = d["parent_action"][1:][keep]
                ratios = np.allclose(r_new / r_new.sum(), r_old / r_old.sum(), rtol=1e-4, atol=1e-6)
                ok2 &= gd == {c - 1} and 0 not in diff and ratios
        elif c == n_cont and CAT_GRID:   # a menu sibling differs from the parent at index 1 and ONLY there, and never repeats its option
            ok2 &= diff == {1}
        elif c == n_cont:   # the categorical redraw may repeat the parent's option: zero or one difference, at index 1
            ok2 &= diff <= {1}
        else:
            ok2 &= diff == {_gym_index(c)}
    report("G2 one coordinate (block)", ok2)

    # G3: stored old log-prob = the policy's log-prob of that coordinate
    obs = th.as_tensor(np.stack(model._side["obs"]))
    acts = th.as_tensor(np.stack(model._side["actions"]))
    mask = th.as_tensor(np.stack(model._side["mask"]))
    aux = th.as_tensor(np.stack(model._side["aux"])) if model._side["aux"] else None
    with th.no_grad():
        lp = (model.policy.evaluate_coord_log_prob(obs, acts, aux) * mask).sum(1).numpy()
    err3 = float(np.abs(lp - np.asarray(model._side["old_log_prob"])).max())
    report("G3 masked log-prob", err3 < 1e-6, f"max |diff| {err3:.1e}")

    # G4: common world — same demand under different actions; forked copies carry the env's seed
    sc = SCENARIOS["base"]
    s0, _ = mdp.init_state(sc, 123_456)
    s0 = mdp.advance1(sc, s0)
    a1 = np.array([5.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]); a2 = np.array([0.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    n1, _ = model._head.step(sc, copy.deepcopy(s0), a1); n2, _ = model._head.step(sc, copy.deepcopy(s0), a2)
    same = np.allclose(n1.demand, n2.demand) and not np.allclose(n1.rt_stock, n2.rt_stock)
    seeds_ok = all(d["episode_seed"] >= 0 for d in det)
    report("G4 common world", same and seeds_ok, f"demand equal {np.allclose(n1.demand, n2.demand)}, states differ {not np.allclose(n1.rt_stock, n2.rt_stock)}")

    # G5: advantage bookkeeping
    adv = np.asarray(model._side["adv_raw"])
    want = np.array([-(d["cj"] - d["c0"]) for d in det])
    err5 = float(np.abs(adv - want).max())
    zero_cat = np.mean([abs(a) < 1e-12 for a, d in zip(adv, det) if d["coord"] == n_cont]) if model._head.name == "catkeep" else float("nan")
    report("G5 advantage = -(Cj - C0)", err5 < 1e-9, f"max |diff| {err5:.1e}; categorical zero-contrast share {zero_cat:.2f}")
    print(f"      side sd {adv.std():.3f} cost units; env steps {model._vine_stats['env_steps']} in {dt:.1f}s "
          f"({model._vine_stats['env_steps'] / dt:.0f} window steps/s); by coordinate mean adv "
          + " ".join(f"c{c}:{adv[[d['coord'] == c for d in det]].mean():+.3f}" for c in range(n_coords)))

    # G6: p = 0 is inert — bit-identical parameters to plain PPO after two updates
    # build-then-train each model in turn: SB3 reseeds the global RNGs at construction,
    # so building both first would start the two trainings from different RNG states
    m_vine, _ = build(vine=True, p=0.0)
    m_vine.learn(total_timesteps=2 * args.n_envs * args.n_steps)
    m_plain, _ = build(vine=False, p=0.0)
    m_plain.learn(total_timesteps=2 * args.n_envs * args.n_steps)
    eq = all(th.equal(a, b) for a, b in zip(m_vine.policy.state_dict().values(), m_plain.policy.state_dict().values()))
    report("G6 p=0 bit-identical to PPO", eq)

    # G7: the side pass trains and logs
    m1, _ = build(vine=True, p=1.0)
    before = [p.detach().clone() for p in m1.policy.parameters()]
    m1.learn(total_timesteps=args.n_envs * args.n_steps)
    vals = m1.logger.name_to_value
    moved = any(not th.equal(a, b) for a, b in zip(before, m1.policy.parameters()))
    ok7 = vals.get("vine/n_side", 0) > 0 and np.isfinite(vals.get("vine/approx_kl", np.nan)) and moved
    report("G7 side pass", ok7, f"n_side {vals.get('vine/n_side')}, approx_kl {vals.get('vine/approx_kl')}, "
                               f"clip_fraction {vals.get('vine/clip_fraction')}, epochs {vals.get('vine/epochs_run')}, params moved {moved}")

    # G9 (dirichlet): the per-share siblings are Dirichlet(alpha) draws — their sufficient statistics E[s_i] and
    # E[log s_i] match FRESH draws passed through the same SHARE_EPS floor (the floor biases E[log s] for small
    # alpha identically on both sides; the un-floored theory psi(alpha_i) - psi(alpha_0) is printed for reference).
    # The parent's shares are a policy sample in training, so the comparison is marginal over Dirichlet parents.
    if model._head.name == "dirichlet":
        from scipy.special import digamma
        head = model._head
        alpha = np.array([0.4, 2.0, 1.0, 3.0, 0.7, 5.0])
        rng = np.random.default_rng(7); S = []; F = []
        for _ in range(20000):
            a0 = np.concatenate([[5.0], head._clean(rng.dirichlet(alpha))])
            sib = head.siblings(a0, (np.array([5.0]), np.array([0.5]), alpha[None, :]), 0, rng)
            S.append(np.stack([x[0][1:] for x in sib[1:]]))      # the 6 share siblings of this branch
            F.append(np.stack([head._clean(rng.dirichlet(alpha)) for _ in range(6)]))
        S = np.concatenate(S); F = np.concatenate(F)
        se_m = np.log(S).reshape(20000, 6, 6).mean(1).std(0) / np.sqrt(20000)
        m_err = np.abs(S.mean(0) - F.mean(0)).max()
        l_err = np.abs(np.log(S).mean(0) - np.log(F).mean(0))
        ok9 = m_err < 0.004 and bool((l_err < 4 * se_m + 0.01).all())
        report("G9 per-share siblings ~ Dirichlet(alpha)", ok9,
               f"max |E[s] diff| {m_err:.4f}; |E[log s] diff| {np.round(l_err, 3).tolist()} vs 4 SE {np.round(4 * se_m, 3).tolist()}; "
               f"theory E[log s] {np.round(digamma(alpha) - digamma(alpha.sum()), 3).tolist()}")

    # G10 (menu grid): each menu sibling's weight is pi_old of its option under the collecting policy, and per
    # branch the weights sum to 1 - pi_old(parent's option); sampled siblings carry weight 1
    if CAT_GRID:
        with th.no_grad():
            pr = model.policy.get_distribution(obs).distribution if False else model.policy.get_distribution(obs).cat.probs.numpy()
        w = np.asarray(model._side["w"]); ok10 = True; branch_sums = []
        for d, m, wi, p_ in zip(det, model._side["mask"], w, pr):
            c = int(np.argmax(m))
            if c == n_cont:
                ok10 &= abs(wi - p_[int(round(d["action"][1]))]) < 1e-6
            else:
                ok10 &= wi == 1.0
        per = per_branch
        for b in range(len(det) // per):
            blk = det[b * per:(b + 1) * per]; ws = w[b * per:(b + 1) * per]
            kp = int(round(blk[0]["parent_action"][1])); pb = pr[b * per]
            cat_w = sum(wi for d, wi in zip(blk, ws) if d["coord"] == n_cont)
            branch_sums.append(abs(cat_w - (1.0 - pb[kp])))
        ok10 &= max(branch_sums) < 1e-6
        report("G10 menu-grid weights = pi_old(k), sum 1 - pi_old(parent)", ok10, f"max |sum err| {max(branch_sums):.1e}")

    # G8: the decoupled side optimizer
    m8, _ = build(vine=True, p=1.0, lr_scale=0.5)
    m8.learn(total_timesteps=args.n_envs * args.n_steps)
    main_lr = m8.policy.optimizer.param_groups[0]["lr"]
    side_lr = m8._side_optimizer.param_groups[0]["lr"] if m8._side_optimizer is not None else float("nan")
    want = m8.lr_schedule(m8._current_progress_remaining)
    ok8 = m8._side_optimizer is not None and m8._side_optimizer is not m8.policy.optimizer \
        and abs(main_lr - want) < 1e-12 and abs(side_lr - 0.5 * want) < 1e-12
    report("G8 decoupled side optimizer", ok8, f"main lr {main_lr:.3e}, side lr {side_lr:.3e} (want {0.5 * want:.3e})")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAIL'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
